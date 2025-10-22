"""
Livox 激光雷达 Python 封装 (兼容 SDK v2.3+)

本模块为 Livox-SDK C API 提供了一个最小化的 Python/NumPy 桥接。
它能自动发现子网内的所有 Livox 雷达，连接并开始采样，然后将每一帧
点云数据作为 float32 类型的 (N, 3) NumPy 数组提供。

工作原理:
1.  使用 ctypes 加载 Livox SDK 的动态链接库 (.so/.dll)。
2.  定义与 C API 匹配的结构体和回调函数原型。
3.  初始化 SDK 并设置广播回调，以自动发现和连接雷达设备。
4.  为连接的设备设置数据回调，在后台线程中接收原始点云数据包。
5.  在数据回调中，将 C 结构体数据解析并转换为 NumPy 数组。
6.  调用用户可覆写的 `handle_points` 方法，将处理好的点云帧传递给上层应用。

注意:
- 本封装仅包装了实现核心功能所需的最小 API 子集。
- 经测试兼容 Livox-SDK 2.3.0 (commit 5c3f3b) on Ubuntu 20.04 / Python 3.10。
"""

from __future__ import annotations

import ctypes as _C
import sys
import threading
import time
from ctypes import c_uint8, c_uint32, c_float, c_bool, POINTER, cdll
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# 定位并加载 Livox SDK 共享库
# ---------------------------------------------------------------------------

# 定义可能的库文件名，以兼容不同版本的 SDK 和操作系统
_CANDIDATES = (
    "liblivox_lidar_sdk_shared.so",  # Livox-SDK2 共享库
    "liblivox_sdk.so",               # Livox-SDK1 默认安装名 (Linux)
    "LivoxSdk.dll",                  # Windows 平台
)


def _load_library():
    """
    按顺序查找并加载 Livox SDK 动态链接库。

    Raises:
        OSError: 如果在所有候选路径中都找不到库文件。

    Returns:
        CDLL: 已加载的库对象。
    """
    for name in _CANDIDATES:
        try:
            return cdll.LoadLibrary(name)
        except OSError:
            continue
    raise OSError(
        "无法定位 Livox 共享库 (liblivox_sdk.so / LivoxSdk.dll)。\n"
        "请先编译 Livox-SDK，或将其所在目录添加到 LD_LIBRARY_PATH (Linux) 或 PATH (Windows)。"
    )


_lib = _load_library()

# ---------------------------------------------------------------------------
# Ctypes 结构体和回调函数原型定义 (最小子集)
# ---------------------------------------------------------------------------

_kBroadcastCodeSize = 16

# --- C 结构体定义 ---

class _LivoxEthPacket(_C.Structure):
    """对应 C API 中的 LivoxEthPacket 结构体，用于描述网络数据包。"""
    _fields_ = [
        ("version", c_uint8),
        ("slot", c_uint8),
        ("lidar_id", c_uint8),  # C中为`id`，在Python中是关键字，故重命名
        ("rsvd", c_uint8),
        ("err_code", c_uint32),
        ("timestamp_type", c_uint8),
        ("data_type", c_uint8),
        ("timestamp", c_uint8 * 8),
        ("data", c_uint8 * 1),  # 柔性数组成员，后续会重新造型
    ]


class _LivoxRawPoint(_C.Structure):
    """对应 C API 中的 LivoxRawPoint 结构体，表示笛卡尔坐标系下的原始点。"""
    _fields_ = [
        ("x", _C.c_int32),
        ("y", _C.c_int32),
        ("z", _C.c_int32),
        ("reflectivity", c_uint8),
    ]


class _BroadcastDeviceInfo(_C.Structure):
    """对应 C API 中的 BroadcastDeviceInfo 结构体，用于设备发现。"""
    _fields_ = [
        ("broadcast_code", _C.c_char * _kBroadcastCodeSize),
        ("dev_type", c_uint8),
        ("reserved", _C.c_uint16),
        ("ip", _C.c_char * 16),
    ]


class _DeviceInfo(_C.Structure):
    """对应 C API 中的 DeviceInfo 结构体，描述已连接设备的信息。"""
    _fields_ = [
        ("broadcast_code", _C.c_char * _kBroadcastCodeSize),
        ("handle", c_uint8),
        ("slot", c_uint8),
        ("id", c_uint8),
        ("type", c_uint8),
        ("data_port", _C.c_uint16),
        ("cmd_port", _C.c_uint16),
        ("sensor_port", _C.c_uint16),
        ("ip", _C.c_char * 16),
        ("state", c_uint8),
        ("feature", c_uint8),
    ]


# --- C 回调函数原型 ---

_DataCallback = _C.CFUNCTYPE(None, c_uint8, POINTER(_LivoxEthPacket), c_uint32, _C.c_void_p)
_BroadcastCallback = _C.CFUNCTYPE(None, POINTER(_BroadcastDeviceInfo))
_DeviceEvent = c_uint8  # 枚举: 0=连接, 1=断开, 2=状态改变, 3=Hub事件
_DeviceStateCallback = _C.CFUNCTYPE(None, POINTER(_DeviceInfo), _DeviceEvent)

# ---------------------------------------------------------------------------
# 定义所需 C 函数的参数和返回类型
# ---------------------------------------------------------------------------

_lib.Init.restype = c_bool
_lib.Start.restype = c_bool
_lib.Uninit.restype = None

_lib.SetBroadcastCallback.argtypes = (_BroadcastCallback,)
_lib.SetDeviceStateUpdateCallback.argtypes = (_DeviceStateCallback,)

_lib.AddLidarToConnect.argtypes = (_C.c_char_p, POINTER(c_uint8))
_lib.AddLidarToConnect.restype = c_uint32  # livox_status

_lib.SetDataCallback.argtypes = (c_uint8, _DataCallback, _C.c_void_p)
_lib.LidarStartSampling.argtypes = (c_uint8, _C.c_void_p, _C.c_void_p)

# ---------------------------------------------------------------------------
# Pythonic 封装类
# ---------------------------------------------------------------------------


class Livox:
    """
    一个 Pythonic 的 Livox SDK 封装。

    此类会自动连接到子网内的所有 Livox 雷达，并通过 `handle_points`
    回调函数提供 (N, 3) 格式的 NumPy 点云帧。
    """

    def __init__(self):
        """
        初始化 Livox SDK 并设置回调函数。

        Raises:
            RuntimeError: 如果 SDK 初始化失败。
        """
        if not _lib.Init():
            raise RuntimeError("Livox SDK Init() 失败")

        # 必须保持对回调函数的引用，否则它们可能被垃圾回收。
        self._cb_broadcast = _BroadcastCallback(self._on_broadcast)
        self._cb_dev_state = _DeviceStateCallback(self._on_device_state)
        self._cb_data = _DataCallback(self._on_data)

        _lib.SetBroadcastCallback(self._cb_broadcast)
        _lib.SetDeviceStateUpdateCallback(self._cb_dev_state)

        if not _lib.Start():
            _lib.Uninit()
            raise RuntimeError("Livox SDK Start() 失败")

        # 存储设备句柄 (handle) 到广播码 (broadcast_code) 的映射
        self._handles: Dict[int, str] = {}
        self._running = True

    def spin(self):
        """
        阻塞主线程，直到按下 Ctrl-C，期间允许后台线程处理数据。
        """
        try:
            while self._running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\n[Livox] 检测到中断，正在关闭...")
        finally:
            self.shutdown()

    def shutdown(self):
        """
        安全地反初始化 Livox SDK 并停止所有操作。
        """
        if self._running:
            self._running = False
            _lib.Uninit()
            print("[Livox] SDK 已关闭。")

    def handle_points(self, xyz: np.ndarray):
        """
        点云数据处理回调函数，用户应覆写此方法。

        此方法在 SDK 的后台线程中被调用，每次调用都提供一帧新的点云数据。

        Args:
            xyz (np.ndarray): 一个 (N, 3) 的 float32 NumPy 数组，
                              表示笛卡尔坐标系下的点云，单位为米。
        """
        print(f"收到一帧点云，包含 {xyz.shape[0]} 个点")

    # ------------------------------------------------------------------
    # 内部 C 回调实现
    # ------------------------------------------------------------------

    def _on_broadcast(self, info_ptr: POINTER(_BroadcastDeviceInfo)):
        """当发现新设备时由 SDK 调用。"""
        info = info_ptr.contents
        code = bytes(info.broadcast_code).decode("ascii", "ignore").rstrip("\x00")

        handle = c_uint8()
        stat = _lib.AddLidarToConnect(code.encode("ascii"), _C.byref(handle))
        if stat != 0:
            print(f"[Livox] 添加设备到连接列表失败: {code} (状态码: {stat})")
            return

        # 为新设备设置数据回调并立即尝试启动采样
        _lib.SetDataCallback(handle.value, self._cb_data, None)
        _lib.LidarStartSampling(handle.value, None, None)

        self._handles[handle.value] = code
        print(f"[Livox] 发现设备 {code} 并尝试连接 (句柄: {handle.value})。")

    def _on_device_state(self, info_ptr: POINTER(_DeviceInfo), event: int):
        """当设备连接状态改变时由 SDK 调用。"""
        if not info_ptr:
            return
        info = info_ptr.contents
        if event == 0:  # 连接成功
            print(f"[Livox] 设备 {info.handle} 已连接。当前状态: {info.state}")
            # 再次尝试启动采样，以防上次调用时设备未就绪
            _lib.LidarStartSampling(info.handle, None, None)
        elif event == 1:  # 断开连接
            print(f"[Livox] 设备 {info.handle} 已断开连接。")

    def _on_data(self, handle: int, pkt_ptr: POINTER(_LivoxEthPacket), n_points: int, _client: any):
        """当收到点云数据包时由 SDK 调用。"""
        if n_points == 0:
            return

        pkt = pkt_ptr.contents
        xyz_m = None

        # 根据数据类型解析点云
        if pkt.data_type == 0:  # 笛卡尔坐标
            _RawArray = _LivoxRawPoint * n_points
            raw_points = _C.cast(pkt.data, POINTER(_RawArray)).contents
            raw_np = np.ctypeslib.as_array(raw_points)
            xyz_m = np.stack((raw_np["x"], raw_np["y"], raw_np["z"]), axis=1).astype(np.float32) / 1000.0
        
        elif pkt.data_type == 2:  # 扩展笛卡尔坐标
            class _ExtPoint(_C.Structure):
                _fields_ = [
                    ("x", _C.c_int32), ("y", _C.c_int32), ("z", _C.c_int32),
                    ("reflectivity", c_uint8), ("tag", c_uint8),
                ]
            _ExtArray = _ExtPoint * n_points
            points = _C.cast(pkt.data, POINTER(_ExtArray)).contents
            p_np = np.ctypeslib.as_array(points)
            xyz_m = np.stack((p_np["x"], p_np["y"], p_np["z"]), axis=1).astype(np.float32) / 1000.0
        
        else:
            # 暂时不支持其他数据类型 (如球坐标等)
            return

        # 将解析后的 NumPy 数组传递给用户处理函数
        if xyz_m is not None:
            try:
                self.handle_points(xyz_m)
            except Exception as exc:
                print(f"[Livox] 在 handle_points 中发生异常: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 命令行自测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("启动 Livox 雷达自测试程序...")
    print("将持续打印接收到的点云帧的形状。按 Ctrl-C 退出。")
    
    lidar = Livox()

    # 简单地覆写 handle_points 方法以打印帧信息
    def _print_shape(xyz: np.ndarray):
        print(f"接收到一帧: {xyz.shape}")

    lidar.handle_points = _print_shape
    
    # 阻塞主线程，直到用户中断
    lidar.spin()