from __future__ import annotations
"""
Livox-SDK2 Python 封装 (Push 模式，无广播)

本模块封装了 Livox-SDK2 的 Push 模式接口，提供了一个 Pythonic 的点云数据处理管道。
它支持通过 JSON 配置文件初始化雷达，接收点云数据，并将其转换为 NumPy 数组。

功能概述:
- 支持 Livox MID-360 激光雷达的 Push 模式点云数据接收。
- 自动加载 Livox-SDK2 的动态链接库。
- 提供点云数据的聚合和批量处理功能。
- 支持多雷达设置（未测试）。

使用方法:
1. 安装 Livox-SDK2:
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd Livox-SDK2 && mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
    sudo make install           # installs liblivox_lidar_sdk.so → /usr/local/lib
                                #liblivox_lidar_sdk_shared.so → /usr/local/lib
复制 livox_lidar_quick_start/mid360_config.json至本仓库，
修改其中ip为自己的雷达ip(192.168.123.222)

"""


import ctypes as _C
import json
import os
import sys
import threading
import time
from ctypes import (
    POINTER,
    c_char_p,
    c_uint8,
    c_uint16,
    c_uint32,
    c_float,
    c_bool,
)
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# 动态加载 Livox-SDK2 共享库
# ---------------------------------------------------------------------------

def _load_lib():
    """
    加载 Livox-SDK2 的动态链接库。

    Returns:
        CDLL: 已加载的共享库对象。

    Raises:
        OSError: 如果未找到共享库。
    """
    for name in (
        "liblivox_lidar_sdk_shared.so",  # Linux
        "liblivox_lidar_sdk.so",         # 旧版 SDK
        "livox_lidar_sdk.dll",           # Windows
    ):
        try:
            return _C.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise OSError(
        "未找到 Livox-SDK2 共享库。请确保已正确安装 Livox-SDK2。"
    )

_lib = _load_lib()

# ---------------------------------------------------------------------------
# Ctypes 结构体和回调函数原型定义
# ---------------------------------------------------------------------------

class _LivoxLidarEthernetPacket(_C.Structure):
    """
    描述 Livox 雷达的以太网数据包结构。
    """
    _pack_ = 1
    _fields_ = [
        ("version", c_uint8),
        ("length", c_uint16),
        ("time_interval", c_uint16),
        ("dot_num", c_uint16),
        ("udp_cnt", c_uint16),
        ("frame_cnt", c_uint8),
        ("data_type", c_uint8),
        ("time_type", c_uint8),
        ("rsvd", c_uint8 * 12),
        ("crc32", c_uint32),
        ("timestamp", c_uint8 * 8),
        ("data", c_uint8 * 1),
    ]

class _CartesianHighPoint(_C.Structure):
    """
    描述 Livox 雷达的高精度笛卡尔点。
    """
    _pack_ = 1
    _fields_ = [
        ("x", _C.c_int32),
        ("y", _C.c_int32),
        ("z", _C.c_int32),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]

# Callback typedef
_PointCb = _C.CFUNCTYPE(None, c_uint32, c_uint8, POINTER(_LivoxLidarEthernetPacket), _C.c_void_p)

# Info change callback
class _LivoxLidarInfo(_C.Structure):
    _fields_ = [
        ("dev_type", c_uint8),
        ("sn", _C.c_char * 16),
        ("lidar_ip", _C.c_char * 16),
    ]


_InfoChangeCb = _C.CFUNCTYPE(None, c_uint32, POINTER(_LivoxLidarInfo), _C.c_void_p)

# ---------------------------------------------------------------------------
# Additional API we use for push-mode
# ---------------------------------------------------------------------------


_lib.SetLivoxLidarInfoChangeCallback.argtypes = (_InfoChangeCb, _C.c_void_p)

_lib.SetLivoxLidarWorkMode.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)
_lib.SetLivoxLidarWorkMode.restype = c_uint32

_lib.EnableLivoxLidarPointSend.argtypes = (c_uint32, _C.c_void_p, _C.c_void_p)
_lib.EnableLivoxLidarPointSend.restype = c_uint32

_lib.SetLivoxLidarPclDataType.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)

# Point-cloud observer (interface side; lets SDK join multicast)
_lib.LivoxLidarAddPointCloudObserver.argtypes = (_PointCb, _C.c_void_p)
_lib.LivoxLidarAddPointCloudObserver.restype = c_uint16

# ---------------- function prototypes -------------------------------------------


_lib.LivoxLidarSdkInit.argtypes = (c_char_p, c_char_p, _C.c_void_p)
_lib.LivoxLidarSdkInit.restype = c_bool

_lib.LivoxLidarSdkStart.argtypes = ()
_lib.LivoxLidarSdkStart.restype = c_bool

_lib.LivoxLidarSdkUninit.argtypes = ()
_lib.LivoxLidarSdkUninit.restype = None

_lib.SetLivoxLidarPointCloudCallBack.argtypes = (_PointCb, _C.c_void_p)

# ---------------- Pythonic wrapper ----------------------------------------------


class Livox2:
    """Minimal wrapper around Livox-SDK2 push-mode pipeline."""

    def __init__(self, config_path: str | Path, host_ip: str,
                 *, frame_time: float = 0.20, frame_packets: int = 120):
        """
        初始化 Livox2 实例。

        Args:
            config_path (str | Path): JSON 配置文件路径。
            host_ip (str): 主机 IP 地址。
            frame_time (float): 聚合帧的时间间隔（秒）。
            frame_packets (int): 每帧的最大数据包数。

        Raises:
            RuntimeError: 如果 SDK 初始化失败。
        """
        self._config_path = os.fspath(config_path).encode()

        if not _lib.LivoxLidarSdkInit(self._config_path, host_ip.encode(), None):
            raise RuntimeError("LivoxLidarSdkInit 初始化失败，请检查配置文件和 JSON 格式。")

        # 注册点云回调函数
        self._cb = _PointCb(self._on_packet)
        _lib.SetLivoxLidarPointCloudCallBack(self._cb, None)

        # 启动 SDK 线程
        _lib.LivoxLidarSdkStart()

        # 注册信息变更回调函数
        self._info_cb = _InfoChangeCb(self._on_info_change)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)

        self._running = True
        self._frame_time = float(frame_time)
        self._frame_packets = int(frame_packets)

    # ------------------------------------------------------------------
    def spin(self):
        """
        阻塞主线程，直到用户按下 Ctrl-C。
        """
        try:
            while self._running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        """
        安全关闭 Livox SDK。
        """
        if self._running:
            _lib.LivoxLidarSdkUninit()
            self._running = False

    # ------------------------------------------------------------------
    def handle_points(self, xyz: np.ndarray):
        """
        处理点云数据的回调函数。

        Args:
            xyz (np.ndarray): 点云数据，形状为 (N, 3)。
        """
        print(f"接收到 {len(xyz)} 个点")

    # ------------------------------------------------------------------
    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        """
        处理 Livox 雷达的数据包。

        Args:
            handle (int): 雷达句柄。
            dev_type (int): 设备类型。
            pkt_ptr: 数据包指针。
            _client: 客户端指针。
        """
        pkt = pkt_ptr.contents
        n = pkt.dot_num
        if n == 0:
            return

        if pkt.data_type == 1:  # Cartesian High
            _Arr = _CartesianHighPoint * n
            points = _C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 1000.0
        elif pkt.data_type == 2:  # Cartesian Low (int16, cm)
            class _LowPoint(_C.Structure):
                _fields_ = [
                    ("x", _C.c_int16),
                    ("y", _C.c_int16),
                    ("z", _C.c_int16),
                    ("reflectivity", c_uint8),
                    ("tag", c_uint8),
                ]

            _ArrL = _LowPoint * n
            pts = _C.cast(pkt.data, POINTER(_ArrL)).contents
            arr = np.ctypeslib.as_array(pts)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 100.0
        else:
            return

        # --------------------------------------------------------------
        # Aggregate packets belonging to the same "frame" (full 360°)
        # --------------------------------------------------------------
        # Each UDP packet contains only a tiny slice of a full scan – for the
        # MID-360 that's merely 96 points. Feeding such sparse subsets into a
        # SLAM backend like KISS-ICP is ineffective and typically produces an
        # empty map. The packet header provides a monotonically increasing
        # `frame_cnt` field which we can use to group packets that belong to
        # the same rotation. We buffer points until the counter changes, then
        # emit the *previous* frame in one batch via ``handle_points``.
        #
        # A small dictionary maps <lidar handle> → current frame accumulator so
        # that multi-lidar setups would still work (although untested).
        # --------------------------------------------------------------

        # ------------------------------------------------------------------
        # Aggregate packets for ~1 full rotation (≈50 ms @ 20 Hz)
        # ------------------------------------------------------------------
        state = self.__dict__.setdefault("_frame_state", {})  # type: ignore[str-bytes-safe]
        buf, last_t = state.get(handle, ([], time.time()))

        buf.append(xyz)

        now = time.time()
        elapsed = now - last_t

        # Heuristic flush conditions: either 0.2 s have passed (≈4 full scans
        # at 20 Hz) *or* we already gathered ≥ 120 packets (~12 k points).
        # A denser frame gives downstream algorithms like KISS-ICP much more
        # structure to work with and greatly improves map stability.
        if elapsed >= self._frame_time or len(buf) >= self._frame_packets:
            frame_xyz = np.concatenate(buf, axis=0)
            try:
                self.handle_points(frame_xyz)
            except Exception as exc:
                print("Exception in handle_points:", exc, file=sys.stderr)

            print(f"[Livox2] frame {frame_xyz.shape[0]} pts  (Δt={elapsed*1000:.1f} ms)")

            buf = []
            last_t = now

        state[handle] = (buf, last_t)

    # ------------------------------------------------------------------
    def _on_info_change(self, handle: int, info_ptr, _client):
        """
        处理 Livox 雷达的信息变更。

        Args:
            handle (int): 雷达句柄。
            info_ptr: 信息指针。
            _client: 客户端指针。
        """
        print(f"[Livox2] InfoChange handle={handle}")

        # 设置工作模式为 NORMAL (1) to begin emitting points.
        kNormal = 1
        _lib.SetLivoxLidarWorkMode(handle, kNormal, None, None)

        # 确保点云发送已启用
        _lib.EnableLivoxLidarPointSend(handle, None, None)

        # 确保数据类型为 Cartesian High (1)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)


if __name__ == "__main__":
    cfg = Path("mid360_config.json")
    if not cfg.exists():
        # 生成默认配置文件
        host_ip = os.environ.get("HOST_IP", "192.168.123.164")
        data = {
            "MID360": {
                "lidar_net_info": {
                    "cmd_data_port": 56100,
                    "push_msg_port": 56200,
                    "point_data_port": 56300,
                    "imu_data_port": 56400,
                    "log_data_port": 56500,
                },
                "host_net_info": [
                    {
                        "host_ip": host_ip,
                        "multicast_ip": "224.1.1.5",
                        "cmd_data_port": 56101,
                        "push_msg_port": 56201,
                        "point_data_port": 56301,
                        "imu_data_port": 56401,
                        "log_data_port": 56501,
                    }
                ],
            }
        }
        cfg.write_text(json.dumps(data, indent=2))
        print("[Livox2] 默认配置文件已生成:", host_ip)

    lidar = Livox2(cfg, host_ip="192.168.123.222")
    lidar.spin()