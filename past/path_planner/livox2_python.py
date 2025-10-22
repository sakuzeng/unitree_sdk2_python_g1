from __future__ import annotations
"""
Livox-SDK2 Python 封装 (Push 模式，优化版)

本模块提供了高性能的 Livox-SDK2 Python 封装，集成 SLAM 自主导航功能。

功能特性:
- 支持 Livox MID-360 激光雷达的 Push 模式
- 线程安全的数据处理管道
- 智能帧聚合和数据预处理
- 集成 SLAM 处理接口
- 内存使用优化
"""

import ctypes as _C
import json
import os
import sys
import threading
import time
import socket
import struct
import logging
from collections import deque
from ctypes import (
    POINTER, c_char_p, c_char, c_uint8, c_uint16, c_uint32, c_float,
    c_int32, c_int16, c_bool, Structure, CFUNCTYPE, c_void_p,
)
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 动态库加载
# ---------------------------------------------------------------------------

def _load_lib():
    """加载 Livox-SDK2 动态链接库"""
    lib_names = [
        "liblivox_lidar_sdk_shared.so",
        "liblivox_lidar_sdk.so", 
        "/usr/local/lib/liblivox_lidar_sdk_shared.so"
    ]
    
    for name in lib_names:
        try:
            lib = _C.cdll.LoadLibrary(name)
            logger.info(f"成功加载 Livox SDK: {name}")
            return lib
        except OSError:
            continue
    
    raise OSError("未找到 Livox-SDK2 共享库。请确保已正确安装 Livox-SDK2。")

_lib = _load_lib()

# ---------------------------------------------------------------------------
# Ctypes 结构体定义
# ---------------------------------------------------------------------------

class _LivoxLidarInfo(Structure):
    _fields_ = [
        ("dev_type", c_uint8),
        ("sn", c_char * 16),
        ("lidar_ip", c_char * 16),
    ]

class _LivoxLidarEthernetPacket(Structure):
    _pack_ = 1
    _fields_ = [
        ("version", c_uint8), ("length", c_uint16), ("time_interval", c_uint16),
        ("dot_num", c_uint16), ("udp_cnt", c_uint16), ("frame_cnt", c_uint8),
        ("data_type", c_uint8), ("time_type", c_uint8), ("rsvd", c_uint8 * 12),
        ("crc32", c_uint32), ("timestamp", c_uint8 * 8), ("data", c_uint8 * 1),
    ]

class _CartesianHighPoint(Structure):
    _pack_ = 1
    _fields_ = [("x", c_int32), ("y", c_int32), ("z", c_int32), 
               ("reflectivity", c_uint8), ("tag", c_uint8)]

class _CartesianLowPoint(Structure):
    _pack_ = 1
    _fields_ = [("x", c_int16), ("y", c_int16), ("z", c_int16),
               ("reflectivity", c_uint8), ("tag", c_uint8)]

class _SphericalPoint(Structure):
    _pack_ = 1
    _fields_ = [("depth", c_uint32), ("theta", c_uint16), ("phi", c_uint16),
               ("reflectivity", c_uint8), ("tag", c_uint8)]

class _ImuPoint(Structure):
    _pack_ = 1
    _fields_ = [("gyro_x", c_float), ("gyro_y", c_float), ("gyro_z", c_float),
               ("acc_x", c_float), ("acc_y", c_float), ("acc_z", c_float)]

# ---------------------------------------------------------------------------
# 回调函数和SDK接口配置
# ---------------------------------------------------------------------------

_PointCb = CFUNCTYPE(None, c_uint32, c_uint8, POINTER(_LivoxLidarEthernetPacket), c_void_p)
_InfoChangeCb = CFUNCTYPE(None, c_uint32, POINTER(_LivoxLidarInfo), c_void_p)

# 配置SDK函数原型
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
# 主要封装类
# ---------------------------------------------------------------------------

class Livox2:
    """
    Livox-SDK2 高性能 Python 封装
    
    集成 SLAM 自主导航功能，支持线程安全的数据处理和智能帧聚合
    """
    
    def __init__(self, config_path: str | Path, host_ip: str = "192.168.123.164", *,
                 frame_time: float = 0.1, frame_packets: int = 50,
                 enable_filter: bool = True, max_range: float = 30.0,
                 voxel_size: float = 0.1):
        """
        初始化 Livox2 实例
        
        Args:
            config_path: JSON 配置文件路径
            host_ip: 主机 IP 地址
            frame_time: 聚合帧时间间隔（秒）
            frame_packets: 每帧最大数据包数
            enable_filter: 是否启用点云过滤
            max_range: 最大距离过滤（米）
            voxel_size: 体素下采样大小（米）
        """
        self._C = _C
        self._config_path = os.fspath(config_path).encode()
        self._host_ip = host_ip
        self._frame_time = float(frame_time)
        self._frame_packets = int(frame_packets)
        self._enable_filter = bool(enable_filter)
        self._max_range = float(max_range)
        self._voxel_size = float(voxel_size)
        
        # 线程安全
        self._lock = threading.RLock()
        self._running = False
        self._sockets = []
        
        # 性能统计
        self._stats = {
            'total_packets': 0, 'total_points': 0, 'dropped_packets': 0,
            'processing_time_ms': 0.0, 'frame_rate': 0.0
        }
        self._stats_history = deque(maxlen=100)
        
        # 帧缓冲区
        self._frame_buffers = {}
        
        # SLAM 处理回调（可选）
        self._slam_callback: Optional[Callable] = None
        
        # 加载配置和初始化
        self._load_config()
        self._init_sdk()
        
        logger.info(f"Livox2 初始化完成 - 主机: {host_ip}")

    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self._config_path.decode(), 'r') as f:
                config = json.load(f)
            
            mid360_config = config.get('MID360', {})
            host_info = mid360_config.get('host_net_info', [{}])[0]
            
            self._multicast_ip = host_info.get('multicast_ip', '224.1.1.5')
            self._ports = {
                'point_data_port': host_info.get('point_data_port', 56301),
                'imu_data_port': host_info.get('imu_data_port', 56401),
            }
            
            logger.info(f"配置加载成功 - 组播: {self._multicast_ip}")
            
        except Exception as e:
            raise RuntimeError(f"配置文件加载失败: {e}")

    def _init_sdk(self):
        """初始化 Livox SDK"""
        # 初始化SDK
        if not _lib.LivoxLidarSdkInit(self._config_path, self._host_ip.encode(), None):
            raise RuntimeError("LivoxLidarSdkInit 初始化失败")
        
        # 设置组播
        self._setup_multicast()
        
        # 注册回调
        self._point_cb = _PointCb(self._on_packet)
        self._info_cb = _InfoChangeCb(self._on_info_change)
        
        _lib.SetLivoxLidarPointCloudCallBack(self._point_cb, None)
        _lib.LivoxLidarAddPointCloudObserver(self._point_cb, None)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)
        
        # 启动SDK
        if not _lib.LivoxLidarSdkStart():
            raise RuntimeError("LivoxLidarSdkStart 启动失败")
        
        self._running = True
        logger.info("Livox SDK 启动成功")

    def _setup_multicast(self):
        """配置组播接收"""
        for port_name, port in self._ports.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
                sock.bind(('', port))
                
                group = socket.inet_aton(self._multicast_ip)
                mreq = struct.pack('4sL', group, socket.INADDR_ANY)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
                sock.settimeout(5.0)
                
                self._sockets.append(sock)
                logger.info(f"组播配置成功: {self._multicast_ip}:{port}")
                
            except socket.error as e:
                raise RuntimeError(f"组播配置失败: {e}")

    def _filter_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                      tag: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """点云过滤和预处理"""
        if not self._enable_filter or len(xyz) == 0:
            return xyz, reflectivity, tag
        
        # 距离过滤
        distances = np.linalg.norm(xyz, axis=1)
        valid_mask = distances <= self._max_range
        
        # 标签过滤（移除噪声）
        if tag is not None:
            valid_mask = valid_mask & (tag != 1)
        
        return xyz[valid_mask], reflectivity[valid_mask], tag[valid_mask]

    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        """数据包处理回调"""
        start_time = time.perf_counter()
        
        try:
            pkt = pkt_ptr.contents
            n = pkt.dot_num
            if n == 0:
                return

            timestamp = int.from_bytes(pkt.timestamp, byteorder='little')

            # IMU 数据处理
            if pkt.data_type == 0:
                self._process_imu_data(pkt, n, timestamp)
                return

            # 点云数据处理
            xyz, reflectivity, tag = self._extract_point_data(pkt, n)
            if xyz is None:
                return

            # 点云过滤
            xyz, reflectivity, tag = self._filter_points(xyz, reflectivity, tag)
            
            if len(xyz) == 0:
                return

            # 帧聚合
            with self._lock:
                self._aggregate_frame(handle, xyz, reflectivity, tag, timestamp)

            # 更新统计
            processing_time = time.perf_counter() - start_time
            self._update_stats(n, processing_time)

        except Exception as exc:
            logger.error(f"数据包处理异常: {exc}")
            with self._lock:
                self._stats['dropped_packets'] += 1

    def _extract_point_data(self, pkt, n) -> Tuple[Optional[np.ndarray], 
                                                 Optional[np.ndarray], 
                                                 Optional[np.ndarray]]:
        """从数据包中提取点云数据"""
        try:
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
                depth = arr["depth"].astype(np.float32) / 1000.0
                theta = np.deg2rad(arr["theta"].astype(np.float32) / 100.0)
                phi = np.deg2rad(arr["phi"].astype(np.float32) / 100.0)
                
                xyz = np.stack([
                    depth * np.sin(theta) * np.cos(phi),
                    depth * np.sin(theta) * np.sin(phi),
                    depth * np.cos(theta)
                ], axis=1)
                reflectivity = arr["reflectivity"].astype(np.uint8)
                tag = arr["tag"].astype(np.uint8)
            else:
                return None, None, None
                
            return xyz, reflectivity, tag
            
        except Exception as e:
            logger.error(f"点云数据提取失败: {e}")
            return None, None, None

    def _process_imu_data(self, pkt, n: int, timestamp: int):
        """处理IMU数据"""
        try:
            _Arr = _ImuPoint * n
            points = self._C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            
            imu_data = np.stack((
                arr["gyro_x"], arr["gyro_y"], arr["gyro_z"],
                arr["acc_x"], arr["acc_y"], arr["acc_z"]
            ), axis=1)
            
            self.handle_imu(imu_data, timestamp)
            
        except Exception as e:
            logger.error(f"IMU数据处理失败: {e}")

    def _aggregate_frame(self, handle: int, xyz: np.ndarray, reflectivity: np.ndarray, 
                        tag: np.ndarray, timestamp: int):
        """帧聚合处理"""
        if handle not in self._frame_buffers:
            self._frame_buffers[handle] = {
                'xyz_list': [], 'ref_list': [], 'tag_list': [],
                'timestamps': [], 'start_time': time.time(), 'count': 0
            }
        
        buffer = self._frame_buffers[handle]
        buffer['xyz_list'].append(xyz)
        buffer['ref_list'].append(reflectivity)
        buffer['tag_list'].append(tag)
        buffer['timestamps'].append(timestamp)
        buffer['count'] += 1
        
        # 检查刷新条件
        elapsed = time.time() - buffer['start_time']
        if elapsed >= self._frame_time or buffer['count'] >= self._frame_packets:
            # 聚合数据
            frame_xyz = np.concatenate(buffer['xyz_list'], axis=0)
            frame_ref = np.concatenate(buffer['ref_list'], axis=0)
            frame_tag = np.concatenate(buffer['tag_list'], axis=0)
            frame_timestamp = buffer['timestamps'][-1]
            
            # 处理帧
            self._process_frame(frame_xyz, frame_ref, frame_tag, frame_timestamp)
            
            # 清空缓冲区
            buffer['xyz_list'].clear()
            buffer['ref_list'].clear()
            buffer['tag_list'].clear()
            buffer['timestamps'].clear()
            buffer['start_time'] = time.time()
            buffer['count'] = 0

    def _process_frame(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                      tag: np.ndarray, timestamp: int):
        """处理聚合后的帧数据"""
        try:
            # 调用用户处理函数
            self.handle_points(xyz, reflectivity, tag, timestamp)
            
            # SLAM 处理（如果有回调）
            if self._slam_callback:
                self._slam_callback(xyz, timestamp)
                
        except Exception as e:
            logger.error(f"帧处理异常: {e}")

    def _update_stats(self, point_count: int, processing_time: float):
        """更新性能统计"""
        with self._lock:
            self._stats['total_packets'] += 1
            self._stats['total_points'] += point_count
            self._stats['processing_time_ms'] = processing_time * 1000

    def _on_info_change(self, handle: int, info_ptr, _client):
        """雷达信息变更处理"""
        logger.info(f"雷达连接: handle={handle}")
        
        # 设置工作模式和数据类型
        kNormal = 1
        _lib.SetLivoxLidarWorkMode(handle, kNormal, None, None)
        _lib.EnableLivoxLidarPointSend(handle, None, None)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)  # 高精度笛卡尔

    # ---------------------------------------------------------------------------
    # 公共接口
    # ---------------------------------------------------------------------------

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                     tag: np.ndarray, timestamp: int):
        """
        点云数据处理回调（用户重写）
        
        Args:
            xyz: 点云坐标 (N, 3)，单位：米
            reflectivity: 反射强度 (N,)，范围 0-255
            tag: 标签 (N,)，用于噪声过滤
            timestamp: 时间戳，单位：ns
        """
        logger.debug(f"接收点云: {len(xyz)} 点")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        IMU 数据处理回调（用户重写）
        
        Args:
            imu_data: IMU 数据 (N, 6)，[gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
            timestamp: 时间戳，单位：ns
        """
        logger.debug(f"接收IMU: {len(imu_data)} 样本")

    def set_slam_callback(self, callback: Callable[[np.ndarray, int], None]):
        """设置 SLAM 处理回调"""
        self._slam_callback = callback
        logger.info("SLAM 回调已设置")

    def get_stats(self) -> dict:
        """获取性能统计"""
        with self._lock:
            return self._stats.copy()

    def is_running(self) -> bool:
        """检查是否运行中"""
        return self._running

    def spin(self, timeout: Optional[float] = None):
        """阻塞运行直到停止或超时"""
        start_time = time.time()
        try:
            while self._running:
                if timeout and (time.time() - start_time) > timeout:
                    logger.info(f"运行超时 ({timeout}s)，停止")
                    break
                time.sleep(0.01)
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        finally:
            self.shutdown()

    def shutdown(self):
        """安全关闭"""
        if not self._running:
            return
        
        logger.info("正在关闭 Livox2...")
        
        with self._lock:
            self._running = False
            self._frame_buffers.clear()
            
            for sock in self._sockets:
                try:
                    sock.close()
                except Exception as e:
                    logger.warning(f"关闭套接字失败: {e}")
            self._sockets.clear()
        
        try:
            _lib.LivoxLidarSdkUninit()
            logger.info("Livox SDK 已关闭")
        except Exception as e:
            logger.error(f"SDK关闭异常: {e}")

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def create_default_config(config_path: str | Path, 
                         lidar_ip: str = "192.168.123.120",
                         host_ip: str = "192.168.123.164") -> dict:
    """创建默认配置文件"""
    config = {
        "MID360": {
            "lidar_net_info": {
                "lidar_ip": lidar_ip,
                "cmd_data_port": 56100, "push_msg_port": 56200,
                "point_data_port": 56300, "imu_data_port": 56400,
                "log_data_port": 56500,
            },
            "host_net_info": [{
                "host_ip": host_ip, "multicast_ip": "224.1.1.5",
                "cmd_data_port": 56101, "push_msg_port": 56201,
                "point_data_port": 56301, "imu_data_port": 56401,
                "log_data_port": 56501,
            }]
        }
    }
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"默认配置已保存: {config_path}")
    return config

# ---------------------------------------------------------------------------
# 示例用法
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = Path("mid360_config.json")
    if not cfg.exists():
        host_ip = os.environ.get("HOST_IP", "192.168.123.164")
        create_default_config(cfg, host_ip=host_ip)
        print(f"默认配置已生成: {host_ip}")
    
    try:
        with Livox2(cfg, host_ip="192.168.123.164") as lidar:
            logger.info("Livox2 启动成功，按 Ctrl+C 停止")
            lidar.spin()
    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)