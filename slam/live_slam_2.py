#!/usr/bin/env python3
"""
Livox MID-360 激光雷达实时 SLAM 演示程序 (改进版)

改进特性:
- 严格的 KISS-ICP 配置优化
- 增强的点云质量检查和预处理
- ICP 配准质量验证
- 自适应参数调整
- 更好的初始化和错误恢复机制
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from datetime import datetime
import json
import csv
import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any, List
import os
import math

# ---------------------------------------------------------------------------
# 数据保存配置
# ---------------------------------------------------------------------------

# 数据保存目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 支持的地图保存格式
MAP_FORMATS = {
    "ply": "PLY 格式 (推荐)",
    "pcd": "PCD 格式",
    "xyz": "纯坐标文本格式"
}

# 默认保存格式
DEFAULT_MAP_FORMAT = os.environ.get("MAP_SAVE_FORMAT", "ply").lower()
if DEFAULT_MAP_FORMAT not in MAP_FORMATS:
    print(f"[WARNING] 未知的地图格式 '{DEFAULT_MAP_FORMAT}'，使用默认格式 'ply'")
    DEFAULT_MAP_FORMAT = "ply"

# ---------------------------------------------------------------------------
# 挂载方向校正配置
# ---------------------------------------------------------------------------

# 挂载方向有效值
_VALID_MOUNTS = {"normal", "upside_down"}

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in _VALID_MOUNTS:
    raise SystemExit(f"LIVOX_MOUNT 必须是 {_VALID_MOUNTS} 中的一个")

# 倾斜轴校正配置
_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()
if _TILT_AXIS not in {"x", "y", "z"}:
    raise SystemExit("LIDAR_TILT_AXIS 必须是 'x', 'y', 'z' 中的一个")

# 读取倾斜角度 (度数) - 默认 0° 不进行校正
try:
    _TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
except ValueError:
    _TILT_DEG = 0.0

_R_MOUNT = None  # 4×4 齐次校正矩阵 (None 表示单位矩阵)

# 构建旋转矩阵
# 1) 倒挂校正 - 绕传感器 X 轴旋转 180° (翻转 Y 和 Z)
_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)

# 2) 固定倾斜校正 - 消除物理倾斜的反向旋转
if abs(_TILT_DEG) > 1e-3:  # 忽略微小角度
    _rad = math.radians(-_TILT_DEG)  # 负号表示反向校正
    c, s = math.cos(_rad), math.sin(_rad)

    if _TILT_AXIS == "x":
        _R_TILT = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
    elif _TILT_AXIS == "y":
        _R_TILT = np.array([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
    else:  # 'z'
        _R_TILT = np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
else:
    _R_TILT = np.eye(4)

# 组合校正: 先翻转后倾斜校正
_R_TOTAL = _R_TILT @ _R_FLIP

# 仅当需要校正时才设置 _R_MOUNT
if not np.allclose(_R_TOTAL, np.eye(4)):
    _R_MOUNT = _R_TOTAL

# ---------------------------------------------------------------------------
# KISS-ICP 导入逻辑
# ---------------------------------------------------------------------------

KissICP = None
_IMPORT_ERRORS = []

try:  # v1.2+ 版本路径
    from kiss_icp.pipeline import KissICP
except Exception as e:
    _IMPORT_ERRORS.append(e)

if KissICP is None:
    try:  # 旧版本路径
        from kiss_icp.pybind import KissICP
    except Exception as e:
        _IMPORT_ERRORS.append(e)

if KissICP is None:
    _msgs = " | ".join(str(e) for e in _IMPORT_ERRORS)
    raise SystemExit(
        "无法导入 KISS-ICP (尝试了 kiss_icp.pipeline 和 kiss_icp.pybind).\n"
        "包缺失或损坏。请安装/升级:\n"
        "    pip install --upgrade 'kiss-icp'\n\n详细信息: "
        + _msgs
    )

# Livox SDK 导入 - 优先使用 SDK2，回退到 SDK1
try:
    from livox2_python import Livox2 as _Livox
except Exception as e:
    print("[INFO] livox2_python 不可用 (", e, ") – 回退到 SDK1.")
    from livox_python import Livox as _Livox

# ---------------------------------------------------------------------------
# 改进的场景预设配置
# ---------------------------------------------------------------------------

PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        # Livox2 伪帧聚合参数
        "frame_time": 0.1,          # 秒，稳定的帧率
        "frame_packets": 50,        # 减少以降低噪声
        
        # KISS-ICP 核心参数 - 更严格的室内配置
        "voxel_size": 0.25,         # 更小的体素以提高精度
        "max_range": 25.0,          # 室内合理距离
        "min_range": 0.5,           # 过滤过近的点
        "max_points_per_voxel": 10, # 减少体素内点数以提高质量
        
        # ICP 配准参数 - 更严格
        "max_num_iterations": 100,   # 增加迭代次数
        "convergence_criterion": 1e-5, # 更严格的收敛标准
        "max_num_threads": 0,        # 自动线程数
        
        # 自适应阈值 - 保守设置
        "initial_threshold": 1.0,    # 降低初始阈值
        "min_motion_th": 0.05,       # 最小运动阈值
        
        # 预处理参数
        "deskew": True,             # 启用运动补偿
        
        # 可视化参数
        "downsample_limit": 2_000_000,
    },
    "outdoor": {
        "frame_time": 0.1,
        "frame_packets": 80,
        
        "voxel_size": 0.5,
        "max_range": 100.0,
        "min_range": 1.0,
        "max_points_per_voxel": 15,
        
        "max_num_iterations": 50,
        "convergence_criterion": 5e-5,
        "max_num_threads": 0,
        
        "initial_threshold": 2.0,
        "min_motion_th": 0.1,
        
        "deskew": True,
        
        "downsample_limit": 3_000_000,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"未知预设 '{PRESET}'. 请选择 {list(_PRESETS.keys())} 中的一个.")

# 当前活动预设的快捷引用
_P = _PRESETS[PRESET]

# ---------------------------------------------------------------------------
# 数据保存工具函数 (保持原有实现)
# ---------------------------------------------------------------------------

def _generate_timestamp() -> str:
    """生成当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_point_cloud(cloud: np.ndarray, file_path: Path) -> bool:
    """保存点云到文件"""
    if cloud is None or cloud.size == 0:
        print(f"[WARNING] 空点云，跳过保存到 {file_path}")
        return False
    
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        
        # 根据文件扩展名选择保存格式
        ext = file_path.suffix.lower()
        if ext in [".ply", ".pcd"]:
            success = o3d.io.write_point_cloud(str(file_path), pcd)
        elif ext == ".xyz":
            # 纯文本格式
            np.savetxt(file_path, cloud, fmt="%.6f", delimiter=" ", 
                      header="x y z", comments="# ")
            success = True
        else:
            print(f"[ERROR] 不支持的文件格式: {ext}")
            return False
        
        if success:
            print(f"[INFO] 点云已保存: {file_path} ({cloud.shape[0]} 点)")
            return True
        else:
            print(f"[ERROR] 保存点云失败: {file_path}")
            return False
            
    except Exception as e:
        print(f"[ERROR] 保存点云时出错: {e}")
        return False

def _save_trajectory(poses: List[np.ndarray], file_path: Path) -> bool:
    """保存轨迹到文件"""
    if not poses:
        print(f"[WARNING] 空轨迹，跳过保存到 {file_path}")
        return False
    
    try:
        trajectory_data = []
        
        for i, pose in enumerate(poses):
            if pose is None:
                continue
                
            # 提取平移和旋转
            translation = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            
            # 转换为四元数 (w, x, y, z)
            try:
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()  # (x, y, z, w)
                quaternion = np.roll(quaternion, 1)  # 转换为 (w, x, y, z)
            except ImportError:
                # 回退到简单的欧拉角
                quaternion = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
            
            # 格式: timestamp x y z qw qx qy qz
            trajectory_data.append([
                float(i),  # 使用索引作为时间戳
                translation[0], translation[1], translation[2],
                quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            ])
        
        # 保存为文本文件
        np.savetxt(file_path, trajectory_data, fmt="%.6f", delimiter=" ",
                  header="timestamp x y z qw qx qy qz", comments="# ")
        
        print(f"[INFO] 轨迹已保存: {file_path} ({len(trajectory_data)} 位姿)")
        return True
        
    except Exception as e:
        print(f"[ERROR] 保存轨迹时出错: {e}")
        return False

def _save_imu_data(imu_buffer: List[tuple[np.ndarray, int]], file_path: Path) -> bool:
    """保存 IMU 数据到 CSV 文件"""
    if not imu_buffer:
        print(f"[WARNING] 空 IMU 数据，跳过保存到 {file_path}")
        return False
    
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
            for data, ts in imu_buffer:
                for row in data:
                    writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
        print(f"[INFO] IMU 数据已保存: {file_path} ({sum(len(data) for data, _ in imu_buffer)} 样本)")
        return True
    except Exception as e:
        print(f"[ERROR] 保存 IMU 数据时出错: {e}")
        return False

def _save_slam_metadata(metadata: Dict[str, Any], file_path: Path) -> bool:
    """保存 SLAM 元数据到 JSON 文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] 元数据已保存: {file_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 保存元数据时出错: {e}")
        return False

# ---------------------------------------------------------------------------
# 3D 可视化器类 (保持原有实现)
# ---------------------------------------------------------------------------

class _Viewer:
    """Open3D 可视化器，同时显示地图和当前位姿"""

    def __init__(self):
        """初始化 3D 可视化窗口"""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None

        # 线程安全的数据队列
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None

        self._first = True

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        """从后台线程接收新的地图和位姿数据"""
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        """在主线程中更新可视化"""
        updated = False

        # 更新点云
        if self._latest_pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(self._latest_pts)
            self._vis.update_geometry(self._pcd)
            self._latest_pts = None
            updated = True

        # 更新位姿
        if self._latest_pose is not None:
            self._update_pose_vis(self._latest_pose)
            self._latest_pose = None
            updated = True

        # 首次显示时自动调整视角
        if self._first and updated:
            self._vis.reset_view_point(True)
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def _update_pose_vis(self, pose: np.ndarray):
        """更新位姿可视化坐标框"""
        # 移除旧的坐标框
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)

        # 根据当前地图范围确定坐标框大小
        size = 0.5
        if len(self._pcd.points) > 0:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03  # 对角线的 3%
            size = max(0.2, min(size, 2.0))  # 限制在 [0.2m, 2m]

        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)
        self._vis.update_geometry(self._cam_frame)

    def close(self):
        """关闭可视化窗口"""
        self._vis.destroy_window()

# ---------------------------------------------------------------------------
# 改进的点云质量检查工具
# ---------------------------------------------------------------------------

def _check_point_cloud_quality(xyz: np.ndarray) -> tuple[bool, str]:
    """
    检查点云质量
    
    Args:
        xyz (np.ndarray): 点云坐标 (N, 3)
        
    Returns:
        tuple[bool, str]: (是否合格, 问题描述)
    """
    if xyz.size == 0:
        return False, "空点云"
    
    # 检查最小点数
    if xyz.shape[0] < 500:
        return False, f"点数太少 ({xyz.shape[0]} < 500)"
    
    # 检查数值有效性
    if not np.isfinite(xyz).all():
        invalid_count = (~np.isfinite(xyz)).sum()
        return False, f"包含 {invalid_count} 个无效点"
    
    # 检查点云分布范围
    ranges = np.ptp(xyz, axis=0)  # 每维的范围
    if np.any(ranges < 0.1):
        return False, f"点云分布范围过小: {ranges}"
    
    # 检查点云密度（避免过于稀疏的点云）
    volume = np.prod(ranges)
    density = xyz.shape[0] / volume if volume > 0 else 0
    if density < 1.0:  # 每立方米至少1个点
        return False, f"点云密度过低: {density:.2f} 点/m³"
    
    return True, "质量合格"

def _preprocess_point_cloud(xyz: np.ndarray) -> np.ndarray:
    """
    改进的点云预处理
    
    Args:
        xyz (np.ndarray): 原始点云
        
    Returns:
        np.ndarray: 处理后的点云
    """
    if xyz.size == 0:
        return xyz
    
    # 1. 移除无效点
    valid_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid_mask]
    
    if xyz.size == 0:
        return xyz
    
    # 2. 距离过滤 - 使用配置参数
    distances = np.linalg.norm(xyz, axis=1)
    distance_mask = (distances >= _P["min_range"]) & (distances <= _P["max_range"])
    xyz = xyz[distance_mask]
    
    if xyz.size == 0:
        return xyz
    
    # 3. 移除机器人自身反射
    try:
        r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", "0.20"))
        dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", "0.15"))
    except ValueError:
        r_xy, dz = 0.20, 0.15
    
    dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
    close_mask = dist_xy < r_xy
    near_plane_mask = np.abs(xyz[:, 2]) < dz
    self_reflection_mask = close_mask & near_plane_mask
    
    xyz = xyz[~self_reflection_mask]
    
    if xyz.size == 0:
        return xyz
    
    # 4. 统计过滤（移除离群点）
    if xyz.shape[0] > 1000:  # 只对足够大的点云进行统计过滤
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # 半径过滤：移除在指定半径内邻居少于阈值的点
            pcd, _ = pcd.remove_radius_outlier(nb_points=3, radius=0.5)
            
            if len(pcd.points) > 100:
                xyz = np.asarray(pcd.points)
        except Exception as e:
            print(f"[WARNING] 统计过滤失败: {e}")
    
    return xyz

# ---------------------------------------------------------------------------
# 改进的主要 SLAM 演示类
# ---------------------------------------------------------------------------

class LiveSLAMDemo(_Livox):
    """实时激光雷达 SLAM 演示系统 (改进版)"""

    def __init__(self):
        """初始化 SLAM 演示系统"""
        # ------------------------------------------------------------------
        # 构建底层 Livox 驱动并设置预设聚合参数
        # ------------------------------------------------------------------

        _sdk_kwargs = {}

        # Livox-SDK2 包装器支持 frame_time/packets 参数
        if _Livox.__name__ == "Livox2":
            _sdk_kwargs.update(
                frame_time=_P["frame_time"], 
                frame_packets=_P["frame_packets"]
            )

        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.164", **_sdk_kwargs)
        except TypeError:
            # 旧版 SDK1 签名（无参数或更少的 kwargs）
            super().__init__()

        # 为 KISS-ICP 构建优化配置
        self._slam = self._create_optimized_slam()
        self._viewer = _Viewer()

        # 可视化的下采样阈值
        self._vis_max_points = _P["downsample_limit"]

        # 改进的帧处理控制
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._processed_frames = 0
        self._skip_frames = 0
        
        # 初始化状态管理
        self._is_initialized = False
        self._initialization_frames = 0
        self._min_init_frames = 10  # 需要更多帧来稳定初始化
        self._init_poses = []  # 用于验证初始化质量
        
        # 配准质量监控
        self._last_successful_pose = np.eye(4)
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        
        # 数据保存相关
        self._start_time = datetime.now()
        self._trajectory: List[np.ndarray] = []
        self._total_frames_processed = 0
        self._imu_buffer: List[tuple[np.ndarray, int]] = []
        self._imu_count = 0
        
        print(f"[INFO] SLAM 系统初始化完成 (预设: {PRESET})")
        print(f"[INFO] KISS-ICP 配置: voxel_size={_P['voxel_size']}, max_range={_P['max_range']}")

    def _create_optimized_slam(self) -> KissICP:
        """创建优化配置的 KISS-ICP 实例"""
        try:
            from kiss_icp.config import load_config
            
            # 首先加载默认配置
            cfg = load_config(config_file=None, max_range=_P["max_range"])
            
            # 应用优化配置
            self._apply_optimized_config(cfg)
            
            return KissICP(cfg)
            
        except Exception as e:
            print(f"[ERROR] 创建 SLAM 配置失败: {e}")
            raise SystemExit("无法创建 KISS-ICP 实例") from e

    def _apply_optimized_config(self, cfg):
        """应用优化的 KISS-ICP 配置"""
        try:
            # 数据预处理配置
            if hasattr(cfg, 'data'):
                cfg.data.max_range = _P["max_range"]
                cfg.data.min_range = _P["min_range"]
                cfg.data.deskew = _P["deskew"]
            
            # 映射配置
            if hasattr(cfg, 'mapping'):
                cfg.mapping.voxel_size = _P["voxel_size"]
                cfg.mapping.max_points_per_voxel = _P["max_points_per_voxel"]
            
            # 配准配置
            if hasattr(cfg, 'registration'):
                cfg.registration.max_num_iterations = _P["max_num_iterations"]
                cfg.registration.convergence_criterion = _P["convergence_criterion"]
                cfg.registration.max_num_threads = _P["max_num_threads"]
            
            # 自适应阈值配置
            if hasattr(cfg, 'adaptive_threshold'):
                cfg.adaptive_threshold.initial_threshold = _P["initial_threshold"]
                cfg.adaptive_threshold.min_motion_th = _P["min_motion_th"]
            
            print("[INFO] KISS-ICP 配置已优化")
            
        except Exception as e:
            print(f"[WARNING] 应用配置时出错: {e}")

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """改进的点云处理方法"""
        current_time = time.time()
        self._frame_count += 1

        # 严格的帧率控制 - 10 Hz
        if current_time - self._last_frame_time < 0.1:
            return
        
        self._last_frame_time = current_time

        # ------------------------------------------------------------------
        # 1. 坐标系校正
        # ------------------------------------------------------------------
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        if _R_MOUNT is not None:
            xyz_homo = np.column_stack([xyz, np.ones(xyz.shape[0])])
            xyz_corrected = (xyz_homo @ _R_MOUNT.T)[:, :3]
            xyz = xyz_corrected.astype(xyz.dtype, copy=False)

        # ------------------------------------------------------------------
        # 2. 改进的点云预处理
        # ------------------------------------------------------------------
        xyz = _preprocess_point_cloud(xyz)

        # ------------------------------------------------------------------
        # 3. 严格的质量检查
        # ------------------------------------------------------------------
        is_valid, quality_msg = _check_point_cloud_quality(xyz)
        if not is_valid:
            self._skip_frames += 1
            if self._skip_frames % 10 == 0:
                print(f"[WARNING] 跳过低质量帧: {quality_msg} (已跳过 {self._skip_frames} 帧)")
            return

        self._skip_frames = 0

        # ------------------------------------------------------------------
        # 4. SLAM 处理
        # ------------------------------------------------------------------
        try:
            # 生成稳定的时间戳
            num_points = xyz.shape[0]
            timestamps = np.linspace(0.0, 0.1, num_points, dtype=np.float64)

            # 获取处理前的位姿作为参考
            prev_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)

            # SLAM 帧注册
            try:
                frame_result = self._slam.register_frame(xyz, timestamps)
            except TypeError:
                frame_result = self._slam.register_frame(xyz)

            # 获取当前位姿
            current_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)

            # 配准质量验证
            if self._validate_registration(prev_pose, current_pose):
                self._last_successful_pose = current_pose.copy()
                self._consecutive_failures = 0
                self._processed_frames += 1
                
                # 保存位姿到轨迹
                self._trajectory.append(current_pose.copy())
                
            else:
                self._consecutive_failures += 1
                print(f"[WARNING] 配准质量不佳 (连续失败: {self._consecutive_failures})")
                
                # 过多连续失败时重置
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("[WARNING] 连续配准失败，尝试系统重置")
                    self._reset_slam_system()
                    return

        except Exception as e:
            print(f"[ERROR] SLAM 处理失败: {e}")
            self._consecutive_failures += 1
            return

        # ------------------------------------------------------------------
        # 5. 初始化状态管理
        # ------------------------------------------------------------------
        if not self._is_initialized:
            self._initialization_frames += 1
            self._init_poses.append(current_pose.copy())
            
            if self._initialization_frames >= self._min_init_frames:
                if self._validate_initialization():
                    self._is_initialized = True
                    print(f"[INFO] SLAM 系统已成功初始化 ({self._initialization_frames} 帧)")
                else:
                    print("[WARNING] 初始化质量不佳，继续收集更多帧")
                    self._min_init_frames += 5  # 需要更多帧

        # ------------------------------------------------------------------
        # 6. 可视化更新
        # ------------------------------------------------------------------
        try:
            # 获取局部地图
            if hasattr(self._slam, 'get_map'):
                cloud = self._slam.get_map()
            elif hasattr(self._slam, 'local_map'):
                cloud = self._slam.local_map.point_cloud()
            else:
                print("[ERROR] 无法获取地图")
                return

            if cloud is None or cloud.size == 0:
                print("[WARNING] 空地图")
                return

            # 可视化下采样
            if cloud.shape[0] > self._vis_max_points:
                step = max(1, int(cloud.shape[0] / self._vis_max_points))
                cloud = cloud[::step]

            # 推送到可视化器
            self._viewer.push(cloud, current_pose)

        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")

        # 统计信息
        if self._processed_frames % 50 == 0:
            print(f"[INFO] 已处理 {self._processed_frames} 帧，当前位置: "
                  f"({current_pose[0,3]:.2f}, {current_pose[1,3]:.2f}, {current_pose[2,3]:.2f})")

    def _validate_registration(self, prev_pose: np.ndarray, current_pose: np.ndarray) -> bool:
        """验证 ICP 配准质量"""
        try:
            # 计算位姿变化
            delta_pose = np.linalg.inv(prev_pose) @ current_pose
            
            # 平移变化
            translation_change = np.linalg.norm(delta_pose[:3, 3])
            
            # 旋转变化 (通过旋转矩阵的迹计算)
            rotation_matrix = delta_pose[:3, :3]
            rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1.0, 1.0))
            
            # 设定合理的变化阈值
            max_translation = 5.0  # 米
            max_rotation = np.radians(30)  # 30度
            
            # 检查变化是否在合理范围内
            if translation_change > max_translation:
                print(f"[WARNING] 平移变化过大: {translation_change:.3f}m")
                return False
            
            if rotation_angle > max_rotation:
                print(f"[WARNING] 旋转变化过大: {np.degrees(rotation_angle):.1f}°")
                return False
            
            return True
            
        except Exception as e:
            print(f"[WARNING] 配准验证失败: {e}")
            return False

    def _validate_initialization(self) -> bool:
        """验证初始化质量"""
        if len(self._init_poses) < self._min_init_frames:
            return False
        
        try:
            # 检查轨迹的合理性
            total_distance = 0.0
            for i in range(1, len(self._init_poses)):
                delta = self._init_poses[i][:3, 3] - self._init_poses[i-1][:3, 3]
                total_distance += np.linalg.norm(delta)
            
            # 初始化期间应该有一定的运动
            if total_distance < 0.5:  # 至少移动 0.5 米
                print(f"[WARNING] 初始化期间运动不足: {total_distance:.3f}m")
                return False
            
            # 检查位姿的一致性（不应该有突变）
            max_single_step = 0.0
            for i in range(1, len(self._init_poses)):
                delta = self._init_poses[i][:3, 3] - self._init_poses[i-1][:3, 3]
                step_distance = np.linalg.norm(delta)
                max_single_step = max(max_single_step, step_distance)
            
            if max_single_step > 2.0:  # 单步不应该超过 2 米
                print(f"[WARNING] 初始化期间存在位姿突变: {max_single_step:.3f}m")
                return False
            
            return True
            
        except Exception as e:
            print(f"[WARNING] 初始化验证失败: {e}")
            return False

    def _reset_slam_system(self):
        """重置 SLAM 系统"""
        try:
            print("[INFO] 正在重置 SLAM 系统...")
            
            # 重新创建 SLAM 实例
            self._slam = self._create_optimized_slam()
            
            # 重置状态
            self._is_initialized = False
            self._initialization_frames = 0
            self._min_init_frames = 10
            self._init_poses.clear()
            self._consecutive_failures = 0
            self._last_successful_pose = np.eye(4)
            
            print("[INFO] SLAM 系统重置完成")
            
        except Exception as e:
            print(f"[ERROR] SLAM 系统重置失败: {e}")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理 IMU 数据，缓冲并保存"""
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        if len(self._imu_buffer) >= 100:
            print(f"[INFO] 缓冲 {self._imu_count} 个 IMU 样本")
            self._imu_buffer = []

    def save_slam_data(self) -> bool:
        """保存 SLAM 数据到 data 目录"""
        print("[INFO] 正在保存 SLAM 数据...")
        
        # 生成带时间戳的文件名前缀
        timestamp = _generate_timestamp()
        session_name = f"slam_session_{timestamp}"
        
        # 创建会话目录
        session_dir = DATA_DIR / session_name
        session_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_saves = 0
        
        # 1. 保存最终地图
        try:
            if hasattr(self._slam, 'get_map'):
                final_map = self._slam.get_map()
            elif hasattr(self._slam, 'local_map'):
                final_map = self._slam.local_map.point_cloud()
            else:
                final_map = None
            
            if final_map is not None and final_map.size > 0:
                # 保存为多种格式
                for fmt in ["ply", "pcd"]:
                    total_saves += 1
                    map_file = session_dir / f"final_map.{fmt}"
                    if _save_point_cloud(final_map, map_file):
                        success_count += 1
                    
        except Exception as e:
            print(f"[ERROR] 保存地图时出错: {e}")
        
        # 2. 保存轨迹
        if self._trajectory:
            total_saves += 1
            trajectory_file = session_dir / "trajectory.txt"
            if _save_trajectory(self._trajectory, trajectory_file):
                success_count += 1
        
        # 3. 保存 IMU 数据
        if self._imu_buffer:
            total_saves += 1
            imu_file = session_dir / "imu_data.csv"
            if _save_imu_data(self._imu_buffer, imu_file):
                success_count += 1
        
        # 4. 保存元数据
        try:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds()
            
            metadata = {
                "session_info": {
                    "start_time": self._start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "preset": PRESET,
                    "mount": MOUNT,
                    "total_frames_processed": self._processed_frames,
                    "initialization_successful": self._is_initialized,
                    "total_imu_samples": self._imu_count
                },
                "slam_config": {
                    "voxel_size": _P["voxel_size"],
                    "max_range": _P["max_range"],
                    "min_range": _P["min_range"],
                    "max_points_per_voxel": _P["max_points_per_voxel"],
                    "max_num_iterations": _P["max_num_iterations"],
                    "convergence_criterion": _P["convergence_criterion"],
                    "initial_threshold": _P["initial_threshold"],
                    "min_motion_th": _P["min_motion_th"],
                    "deskew": _P["deskew"]
                },
                "correction_settings": {
                    "mount_correction": MOUNT,
                    "tilt_axis": _TILT_AXIS,
                    "tilt_degrees": _TILT_DEG,
                    "self_filter_radius": float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.20)),
                    "self_filter_z": float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.15))
                },
                "statistics": {
                    "trajectory_poses": len(self._trajectory),
                    "final_map_points": len(final_map) if 'final_map' in locals() and final_map is not None else 0,
                    "imu_samples": self._imu_count,
                    "successful_frames": self._processed_frames,
                    "total_frames_received": self._frame_count
                }
            }
            
            total_saves += 1
            metadata_file = session_dir / "session_metadata.json"
            if _save_slam_metadata(metadata, metadata_file):
                success_count += 1
                
        except Exception as e:
            print(f"[ERROR] 保存元数据时出错: {e}")
        
        # 输出保存结果
        print(f"[INFO] SLAM 数据保存完成: {success_count}/{total_saves} 文件成功保存")
        print(f"[INFO] 数据保存位置: {session_dir}")
        
        if success_count > 0:
            print("[INFO] 保存的文件:")
            for file_path in sorted(session_dir.glob("*")):
                file_size = file_path.stat().st_size / 1024 / 1024  # MB
                print(f"  - {file_path.name} ({file_size:.2f} MB)")
        
        return success_count > 0

    def shutdown(self):
        """安全关闭所有资源并保存数据"""
        print("[INFO] 正在关闭 SLAM 系统...")
        
        # 输出最终统计信息
        print(f"[INFO] 会话统计: 收到 {self._frame_count} 帧，成功处理 {self._processed_frames} 帧")
        if self._is_initialized:
            print("[INFO] SLAM 系统已成功初始化并运行")
        else:
            print("[WARNING] SLAM 系统未能成功初始化")
        
        # 先保存数据
        try:
            self.save_slam_data()
        except Exception as e:
            print(f"[ERROR] 保存数据时出错: {e}")
        
        # 关闭 Livox 连接
        try:
            super().shutdown()
        except Exception as e:
            print(f"[WARNING] 关闭 Livox 时出错: {e}")
        
        # 关闭可视化器
        try:
            self._viewer.close()
        except Exception as e:
            print(f"[WARNING] 关闭可视化器时出错: {e}")
        
        print("[INFO] SLAM 系统已安全关闭")

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    """主函数 - 启动改进的 SLAM 演示"""
    print("="*60)
    print(f"启动 Livox SLAM (改进版)")
    print(f"预设: {PRESET.upper()}")
    print(f"挂载方向: {MOUNT}")
    print(f"数据保存目录: {DATA_DIR.absolute()}")
    print("="*60)
    
    # 显示关键配置参数
    print("[INFO] 关键配置参数:")
    print(f"  - 体素大小: {_P['voxel_size']} m")
    print(f"  - 最大距离: {_P['max_range']} m")
    print(f"  - ICP 迭代次数: {_P['max_num_iterations']}")
    print(f"  - 收敛标准: {_P['convergence_criterion']}")
    print(f"  - 初始阈值: {_P['initial_threshold']}")
    
    demo = LiveSLAMDemo()

    # 支持 Ctrl-C 中断
    stop = False

    def _sigint(*_):
        nonlocal stop
        print("\n[INFO] 收到中断信号，正在优雅关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        print("\n[INFO] SLAM 已启动，等待激光雷达数据...")
        print("[INFO] 系统将自动进行初始化，请缓慢移动传感器")
        print("[INFO] 按 Ctrl-C 停止并保存数据")
        
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[INFO] 收到键盘中断")
    finally:
        demo.shutdown()

if __name__ == "__main__":
    main()