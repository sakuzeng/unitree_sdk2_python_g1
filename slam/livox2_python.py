from __future__ import annotations
"""
Livox-SDK2 Python 封装 (Push 模式，无广播)

本模块封装了 Livox-SDK2 的 Push 模式接口，提供了一个 Pythonic 的点云和 IMU 数据处理管道。
它支持通过 JSON 配置文件初始化雷达，接收点云和 IMU 数据，并将其转换为 NumPy 数组。

功能概述:
- 支持 Livox MID-360 激光雷达的 Push 模式点云和 IMU 数据接收。
- 自动加载 Livox-SDK2 的动态链接库。
- 提供点云数据的聚合和批量处理功能。
- 支持 IMU 数据通过点云回调处理（data_type=0）。
- 支持多雷达设置（未测试）。
- 显式加入组播组以接收 LiDAR 数据，基于提供的 JSON 配置文件。

使用方法:
1. 安装 Livox-SDK2:
    git clone https://github.com/Livox-SDK/Livox-SDK2.git
    cd Livox-SDK2 && mkdir build && cd build
    cmake .. -CMAKE_BUILD_TYPE=Release && make -j$(nproc)
    sudo make install
复制 livox_lidar_quick_start/mid360_config.json 至本仓库，
修改其中 ip 为自己的雷达 ip (192.168.123.120) 和 imu_data_en=1。
"""

import ctypes as _C
import json
import os
import sys
import threading
import time
import socket
import struct
from ctypes import (
    POINTER,
    c_char_p,
    c_char,
    c_uint8,
    c_uint16,
    c_uint32,
    c_float,
    c_int32,
    c_int16,
    c_bool,
    Structure,
    CFUNCTYPE,
    c_void_p,
)
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# 动态加载 Livox-SDK2 共享库
# ---------------------------------------------------------------------------

def _load_lib():
    """
    加载 Livox-SDK2 的动态链接库。
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
    raise OSError("未找到 Livox-SDK2 共享库。请确保已正确安装 Livox-SDK2。")

_lib = _load_lib()

# ---------------------------------------------------------------------------
# Ctypes 结构体定义
# ---------------------------------------------------------------------------

class _LivoxLidarInfo(Structure):
    """
    描述 Livox 雷达的信息结构体。
    """
    _fields_ = [
        ("dev_type", c_uint8),
        ("sn", c_char * 16),
        ("lidar_ip", c_char * 16),
    ]

class _LivoxLidarEthernetPacket(Structure):
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

class _CartesianHighPoint(Structure):
    """
    高精度笛卡尔点（data_type=1）。
    """
    _pack_ = 1
    _fields_ = [
        ("x", c_int32),
        ("y", c_int32),
        ("z", c_int32),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]

class _CartesianLowPoint(Structure):
    """
    低精度笛卡尔点（data_type=2）。
    """
    _pack_ = 1
    _fields_ = [
        ("x", c_int16),
        ("y", c_int16),
        ("z", c_int16),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]

class _SphericalPoint(Structure):
    """
    球坐标点（data_type=3）。
    """
    _pack_ = 1
    _fields_ = [
        ("depth", c_uint32),
        ("theta", c_uint16),
        ("phi", c_uint16),
        ("reflectivity", c_uint8),
        ("tag", c_uint8),
    ]

class _ImuPoint(Structure):
    """
    IMU 数据点（data_type=0）。
    """
    _pack_ = 1
    _fields_ = [
        ("gyro_x", c_float),
        ("gyro_y", c_float),
        ("gyro_z", c_float),
        ("acc_x", c_float),
        ("acc_y", c_float),
        ("acc_z", c_float),
    ]

# ---------------------------------------------------------------------------
# 回调函数原型
# ---------------------------------------------------------------------------

_PointCb = CFUNCTYPE(None, c_uint32, c_uint8, POINTER(_LivoxLidarEthernetPacket), c_void_p)
_InfoChangeCb = CFUNCTYPE(None, c_uint32, POINTER(_LivoxLidarInfo), c_void_p)

# SDK 函数原型
_lib.SetLivoxLidarInfoChangeCallback.argtypes = (_InfoChangeCb, c_void_p)
_lib.SetLivoxLidarWorkMode.argtypes = (c_uint32, c_uint8, c_void_p, c_void_p)
_lib.SetLivoxLidarWorkMode.restype = c_uint32
_lib.EnableLivoxLidarPointSend.argtypes = (c_uint32, c_void_p, c_void_p)
_lib.EnableLivoxLidarPointSend.restype = c_uint32
_lib.SetLivoxLidarPclDataType.argtypes = (c_uint32, c_uint8, c_void_p, c_void_p)
_lib.LivoxLidarAddPointCloudObserver.argtypes = (_PointCb, c_void_p)
_lib.LivoxLidarAddPointCloudObserver.restype = c_uint16
_lib.LivoxLidarSdkInit.argtypes = (c_char_p, c_char_p, c_void_p)
_lib.LivoxLidarSdkInit.restype = c_bool
_lib.LivoxLidarSdkStart.argtypes = ()
_lib.LivoxLidarSdkStart.restype = c_bool
_lib.LivoxLidarSdkUninit.argtypes = ()
_lib.LivoxLidarSdkUninit.restype = None
_lib.SetLivoxLidarPointCloudCallBack.argtypes = (_PointCb, c_void_p)

# ---------------------------------------------------------------------------
# Pythonic wrapper
# ---------------------------------------------------------------------------

class Livox2:
    """Livox-SDK2 Push 模式封装，支持点云和 IMU 数据。"""

    def __init__(self, config_path: str | Path, host_ip: str,
                 *, frame_time: float = 0.20, frame_packets: int = 120):
        """
        初始化 Livox2 实例。

        Args:
            config_path (str | Path): JSON 配置文件路径。
            host_ip (str): 主机 IP 地址。
            frame_time (float): 聚合帧的时间间隔（秒）。
            frame_packets (int): 每帧的最大数据包数。
        """
        self._C = _C  # 保存 ctypes 模块
        self._config_path = os.fspath(config_path).encode()
        self._host_ip = host_ip
        self._sockets = []  # 保存组播套接字

        # 加载配置文件以获取组播地址和端口
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            mid360_config = config.get('MID360', {})
            host_net_info = mid360_config.get('host_net_info', [{}])[0]
            self._multicast_ip = host_net_info.get('multicast_ip', '224.1.1.5')
            self._ports = {
                'point_data_port': host_net_info.get('point_data_port', 56301),
                'imu_data_port': host_net_info.get('imu_data_port', 56401),
            }
        except Exception as e:
            raise RuntimeError(f"无法读取或解析配置文件 {config_path}: {e}")

        # 初始化 Livox SDK
        if not _lib.LivoxLidarSdkInit(self._config_path, host_ip.encode(), None):
            raise RuntimeError("LivoxLidarSdkInit 初始化失败，请检查配置文件和 JSON 格式。")

        # 设置组播
        self._setup_multicast()

        # 注册点云回调（包括 IMU 数据）
        self._point_cb = _PointCb(self._on_packet)
        _lib.SetLivoxLidarPointCloudCallBack(self._point_cb, None)
        _lib.LivoxLidarAddPointCloudObserver(self._point_cb, None)

        # 注册信息变更回调
        self._info_cb = _InfoChangeCb(self._on_info_change)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)

        # 启动 SDK
        if not _lib.LivoxLidarSdkStart():
            raise RuntimeError("LivoxLidarSdkStart 启动失败")

        self._running = True
        self._frame_time = float(frame_time)
        self._frame_packets = int(frame_packets)
        self._frame_state = {}  # 帧聚合状态

    def _setup_multicast(self):
        """
        配置本机加入组播组以接收点云和 IMU 数据。
        """
        for port_name, port in self._ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # 绑定到组播端口
                sock.bind(('', port))
                # 加入组播组
                group = socket.inet_aton(self._multicast_ip)
                mreq = struct.pack('4sL', group, socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                # 设置接收超时（可选）
                sock.settimeout(5.0)
                self._sockets.append(sock)
                print(f"[Livox2] 成功加入组播组 {self._multicast_ip}:{port} ({port_name})")
            except socket.error as e:
                print(f"[Livox2] 组播配置失败 ({port_name}): {e}", file=sys.stderr)
                raise RuntimeError(f"无法加入组播组 {self._multicast_ip}:{port}: {e}")

    def spin(self):
        """阻塞主线程，直到用户按下 Ctrl-C。"""
        try:
            while self._running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("[Livox2] 收到 Ctrl-C，正在关闭...")
        finally:
            self.shutdown()

    def shutdown(self):
        """安全关闭 Livox SDK 和组播套接字。"""
        if self._running:
            print("[Livox2] 正在关闭 Livox SDK 和组播套接字...")
            _lib.LivoxLidarSdkUninit()
            for sock in self._sockets:
                try:
                    sock.close()
                except socket.error as e:
                    print(f"[Livox2] 关闭套接字失败: {e}", file=sys.stderr)
            self._sockets = []
            self._running = False

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据的回调函数。

        Args:
            xyz (np.ndarray): 点云坐标，形状 (N, 3)，单位：米。
            reflectivity (np.ndarray): 反射强度，形状 (N,)，范围 0-255。
            tag (np.ndarray): 标签，形状 (N,)，用于噪声过滤。
            timestamp (int): 数据包时间戳，单位：ns。
        """
        print(f"[Livox2] 接收到 {len(xyz)} 个点，时间戳: {timestamp}")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据的回调函数。

        Args:
            imu_data (np.ndarray): IMU 数据，形状 (N, 6)，包含 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]。
            timestamp (int): 数据包时间戳，单位：ns。
        """
        print(f"[Livox2] 接收到 {len(imu_data)} 个 IMU 样本，时间戳: {timestamp}")

    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        """
        处理点云和 IMU 数据包（data_type=0/1/2/3）。
        """
        pkt = pkt_ptr.contents
        n = pkt.dot_num
        if n == 0:
            return

        timestamp = int.from_bytes(pkt.timestamp, byteorder='little')

        if pkt.data_type == 0:  # IMU 数据
            _Arr = _ImuPoint * n
            points = self._C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            imu_data = np.stack((arr["gyro_x"], arr["gyro_y"], arr["gyro_z"],
                                 arr["acc_x"], arr["acc_y"], arr["acc_z"]), axis=1)
            try:
                self.handle_imu(imu_data, timestamp)
            except Exception as exc:
                print(f"[Livox2] Exception in handle_imu: {exc}", file=sys.stderr)
            return

        if pkt.data_type == 1:  # 高精度笛卡尔
            _Arr = _CartesianHighPoint * n
            points = self._C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 1000.0
            reflectivity = arr["reflectivity"].astype(np.uint8)
            tag = arr["tag"].astype(np.uint8)

        elif pkt.data_type == 2:  # 低精度笛卡尔
            _Arr = _CartesianLowPoint * n
            points = self._C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32) / 100.0
            reflectivity = arr["reflectivity"].astype(np.uint8)
            tag = arr["tag"].astype(np.uint8)

        elif pkt.data_type == 3:  # 球坐标
            _Arr = _SphericalPoint * n
            points = self._C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            depth = arr["depth"].astype(np.float32) / 1000.0  # mm to m
            theta = arr["theta"].astype(np.float32) / 100.0   # 0.01° to °
            phi = arr["phi"].astype(np.float32) / 100.0       # 0.01° to °
            xyz = np.stack([
                depth * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi)),
                depth * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi)),
                depth * np.cos(np.deg2rad(theta))
            ], axis=1)
            reflectivity = arr["reflectivity"].astype(np.uint8)
            tag = arr["tag"].astype(np.uint8)

        else:
            return

        # 帧聚合逻辑
        state = self._frame_state
        buf_xyz, buf_ref, buf_tag, last_t = state.get(handle, ([], [], [], time.time()))

        buf_xyz.append(xyz)
        buf_ref.append(reflectivity)
        buf_tag.append(tag)

        now = time.time()
        elapsed = now - last_t

        if elapsed >= self._frame_time or len(buf_xyz) >= self._frame_packets:
            frame_xyz = np.concatenate(buf_xyz, axis=0)
            frame_ref = np.concatenate(buf_ref, axis=0)
            frame_tag = np.concatenate(buf_tag, axis=0)
            try:
                self.handle_points(frame_xyz, frame_ref, frame_tag, timestamp)
            except Exception as exc:
                print(f"[Livox2] Exception in handle_points: {exc}", file=sys.stderr)

            print(f"[Livox2] frame {frame_xyz.shape[0]} pts (Δt={elapsed*1000:.1f} ms)")
            buf_xyz, buf_ref, buf_tag = [], [], []
            last_t = now

        state[handle] = (buf_xyz, buf_ref, buf_tag, last_t)

    def _on_info_change(self, handle: int, info_ptr, _client):
        """
        处理雷达信息变更。
        """
        print(f"[Livox2] InfoChange handle={handle}")
        kNormal = 1
        _lib.SetLivoxLidarWorkMode(handle, kNormal, None, None)
        _lib.EnableLivoxLidarPointSend(handle, None, None)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)

if __name__ == "__main__":
    cfg = Path("mid360_config.json")
    if not cfg.exists():
        host_ip = os.environ.get("HOST_IP", "192.168.123.164")
        data = {
            "MID360": {
                "lidar_net_info": {
                    "lidar_ip": "192.168.123.120",
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
                ]
            }
        }
        cfg.write_text(json.dumps(data, indent=2))
        print("[Livox2] 默认配置文件已生成:", host_ip)

    lidar = Livox2(cfg, host_ip="192.168.123.164")
    lidar.spin()