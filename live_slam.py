#!/usr/bin/env python3
"""
Livox MID-360 激光雷达实时 SLAM 演示程序

功能特性:
- 实时 LiDAR 点云处理和 SLAM 建图
- 支持室内/室外场景预设配置
- 3D 可视化地图和位姿显示
- 挂载方向自动校正
- 运动检测和数值稳定性优化
- 程序结束时自动保存地图文件到 data/ 目录

环境依赖:
- Livox-SDK2 共享库
- Python 包: numpy, open3d>=0.16.0, kiss-icp

运行方法:
    python live_slam.py

环境变量配置:
- LIVOX_PRESET: 场景预设 (indoor/outdoor, 默认 indoor)
- LIVOX_MOUNT: 挂载方向 (normal/upside_down, 默认 upside_down)
- LIDAR_TILT_DEG: 倾斜角度校正 (度数, 默认 0)
- LIDAR_TILT_AXIS: 倾斜轴向 (x/y/z, 默认 y)
- LIDAR_SELF_FILTER_RADIUS: 机器人自身过滤半径 (米, 默认 0.20)
- LIDAR_SELF_FILTER_Z: 垂直过滤范围 (米, 默认 0.15)

地图保存:
- 地图文件自动保存到 data/ 目录
- 支持 PLY、PCD 格式的点云文件
- 保存轨迹为 TXT 格式
- 文件名包含时间戳以避免覆盖
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from datetime import datetime
import json

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
# 场景预设配置
# ---------------------------------------------------------------------------

PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        # Livox2 伪帧聚合参数
        "frame_time": 0.35,      # 秒
        "frame_packets": 200,

        # 地图和可视化参数
        "voxel_size": 0.4,       # 米
        "max_range": 30.0,       # 米
        "downsample_limit": 5_000_000,  # 可视化最大点数

        # ICP 调优参数
        "min_motion": 0.03,      # 米
        "conv_criterion": 5e-5,
        "max_iters": 800,
    },
    "outdoor": {
        "frame_time": 0.20,
        "frame_packets": 120,
        "voxel_size": 1.0,
        "max_range": 120.0,
        "downsample_limit": 3_000_000,
        "min_motion": 0.10,
        "conv_criterion": 1e-4,
        "max_iters": 500,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"未知预设 '{PRESET}'. 请选择 {list(_PRESETS.keys())} 中的一个.")

# 当前活动预设的快捷引用
_P = _PRESETS[PRESET]

# ---------------------------------------------------------------------------
# 数据保存工具函数
# ---------------------------------------------------------------------------

def _generate_timestamp() -> str:
    """
    生成当前时间戳字符串
    
    Returns:
        str: 格式化的时间戳字符串
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_point_cloud(cloud: np.ndarray, file_path: Path) -> bool:
    """
    保存点云到文件
    
    Args:
        cloud (np.ndarray): 点云数据 (N, 3)
        file_path (Path): 保存路径
        
    Returns:
        bool: 保存成功返回 True
    """
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
    """
    保存轨迹到文件
    
    Args:
        poses (List[np.ndarray]): 位姿列表 (4x4 矩阵)
        file_path (Path): 保存路径
        
    Returns:
        bool: 保存成功返回 True
    """
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

def _save_slam_metadata(metadata: Dict[str, Any], file_path: Path) -> bool:
    """
    保存 SLAM 元数据到 JSON 文件
    
    Args:
        metadata (Dict[str, Any]): 元数据字典
        file_path (Path): 保存路径
        
    Returns:
        bool: 保存成功返回 True
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] 元数据已保存: {file_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 保存元数据时出错: {e}")
        return False

# ---------------------------------------------------------------------------
# 3D 可视化器类
# ---------------------------------------------------------------------------

class _Viewer:
    """
    Open3D 可视化器，同时显示地图和当前位姿
    """

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
        """
        从后台线程接收新的地图和位姿数据
        
        Args:
            xyz (np.ndarray): 点云坐标
            pose (np.ndarray): 4x4 位姿矩阵
        """
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        """
        在主线程中更新可视化
        
        Returns:
            bool: 窗口是否仍然存活
        """
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
        """
        更新位姿可视化坐标框
        
        Args:
            pose (np.ndarray): 4x4 位姿矩阵
        """
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
# 主要 SLAM 演示类
# ---------------------------------------------------------------------------

class LiveSLAMDemo(_Livox):
    """
    实时激光雷达 SLAM 演示系统
    """

    def __init__(self):
        """初始化 SLAM 演示系统"""
        # ------------------------------------------------------------------
        # 构建底层 Livox 驱动并设置预设聚合参数
        # ------------------------------------------------------------------

        _sdk_kwargs = {}

        # Livox-SDK **2** 包装器支持 frame_time/packets 参数
        if _Livox.__name__ == "Livox2":  # type: ignore[attr-defined]
            _sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])

        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.164", **_sdk_kwargs)  # type: ignore[arg-type]
        except TypeError:
            # 旧版 SDK1 签名（无参数或更少的 kwargs）
            super().__init__()

        # 使用针对 Livox FOV / 扫描模式调优的预设
        # 为 KISS-ICP 构建默认配置（API ≥ 1.2）
        try:
            from kiss_icp.config import load_config  # type: ignore
            cfg = load_config(config_file=None, max_range=_P["max_range"])
        except Exception as e:  # pragma: no cover
            print("[KISS-ICP] 无法通过 load_config 创建配置:", e)
            raise SystemExit(
                "安装的 kiss-icp 包版本过旧 – 请升级: `pip install -U kiss-icp`. "
            ) from e

        # 应用更稳定的配置参数，使用安全的属性设置
        self._apply_safe_config(cfg)
        
        self._slam = KissICP(cfg)
        self._viewer = _Viewer()

        # 可视化的下采样阈值
        self._vis_max_points = _P["downsample_limit"]

        # 添加运动检测和帧率控制
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._skip_frames = 0
        self._last_xyz = None
        self._initialization_frames = 0
        self._min_init_frames = 5  # 需要至少5帧来初始化

        # 数据保存相关
        self._start_time = datetime.now()
        self._trajectory: List[np.ndarray] = []  # 保存完整轨迹
        self._total_frames_processed = 0

    def _apply_safe_config(self, cfg):
        """
        安全地应用 KISS-ICP 配置参数
        
        Args:
            cfg: KISS-ICP 配置对象
        """
        # 映射配置 - 使用保守参数
        try:
            if hasattr(cfg, 'mapping'):
                if hasattr(cfg.mapping, 'voxel_size'):
                    cfg.mapping.voxel_size = max(_P["voxel_size"], 0.2)  # 增加最小体素大小
                if hasattr(cfg.mapping, 'max_points_per_voxel'):
                    cfg.mapping.max_points_per_voxel = 15  # 减少每个体素的点数
        except Exception as e:
            print(f"[WARNING] 设置映射参数时出错: {e}")

        # 自适应阈值配置
        try:
            if hasattr(cfg, 'adaptive_threshold'):
                if hasattr(cfg.adaptive_threshold, 'min_motion_th'):
                    cfg.adaptive_threshold.min_motion_th = max(_P["min_motion"], 0.05)  # 更保守的运动阈值
        except Exception as e:
            print(f"[WARNING] 设置自适应阈值参数时出错: {e}")

        # 配准配置
        try:
            if hasattr(cfg, 'registration'):
                if hasattr(cfg.registration, 'convergence_criterion'):
                    cfg.registration.convergence_criterion = max(_P["conv_criterion"], 1e-4)
                if hasattr(cfg.registration, 'max_num_iterations'):
                    cfg.registration.max_num_iterations = min(_P["max_iters"], 50)  # 显著减少迭代次数
        except Exception as e:
            print(f"[WARNING] 设置配准参数时出错: {e}")

    def handle_points(self, xyz: np.ndarray):
        """
        处理单个激光雷达帧
        
        Args:
            xyz (np.ndarray): 点云坐标 (N, 3)
        """
        current_time = time.time()
        self._frame_count += 1
        
        # 帧率控制 - 最大 10 Hz
        if current_time - self._last_frame_time < 0.1:
            return
            
        self._last_frame_time = current_time
        
        # ------------------------------------------------------------------
        # 1. 移除机器人自身的反射（头部/安装位置）
        # ------------------------------------------------------------------
        try:
            r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.20))  # 减小过滤半径
            dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.15))
        except ValueError:
            r_xy, dz = 0.20, 0.15  # 更保守的默认值

        if xyz.size > 0:
            # 水平距离过滤
            dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
            close = dist_xy < r_xy

            # 垂直接近度过滤
            near_plane = np.abs(xyz[:, 2]) < dz

            mask = ~(close & near_plane)

            if mask.sum() != xyz.shape[0]:
                xyz = xyz[mask]

        if xyz.size == 0:
            print("[WARNING] 空点云，跳过帧")
            return
        
        # 数据有效性检查
        if not np.isfinite(xyz).all():
            print("[WARNING] 点云中包含非有限值，正在清理...")
            valid_mask = np.isfinite(xyz).all(axis=1)
            xyz = xyz[valid_mask]
            if xyz.size == 0:
                print("[WARNING] 清理后无有效点，跳过帧")
                return
        
        # 检查点数
        if xyz.shape[0] < 100:  # 提高最小点数要求
            self._skip_frames += 1
            if self._skip_frames % 20 == 0:
                print(f"[WARNING] 点数不足 ({xyz.shape[0]})，已跳过 {self._skip_frames} 帧")
            return
            
        self._skip_frames = 0
        
        # 初始化阶段的运动检测
        if self._initialization_frames < self._min_init_frames:
            self._initialization_frames += 1
            print(f"[INFO] 初始化阶段 {self._initialization_frames}/{self._min_init_frames}")
        else:
            # 检测运动幅度
            if self._last_xyz is not None and self._last_xyz.shape[0] > 0:
                try:
                    current_center = np.median(xyz, axis=0)  # 使用中位数更稳定
                    last_center = np.median(self._last_xyz, axis=0)
                    motion = np.linalg.norm(current_center - last_center)
                    
                    if motion < 0.005:  # 运动太小
                        return
                except Exception as e:
                    print(f"[WARNING] 运动检测失败: {e}")
        
        # 保存当前帧用于下次比较
        if xyz.shape[0] < 10000:  # 只保存小点云避免内存问题
            self._last_xyz = xyz.copy()
        
        # SLAM 处理
        try:
            # 生成时间戳 - 使用更稳定的方法
            num_points = xyz.shape[0]
            period = 0.1  # 10 Hz 等效周期
            
            # 简单线性时间戳
            timestamps = np.linspace(0.0, period, num_points, dtype=np.float64)
            
            # 尝试注册帧
            try:
                self._slam.register_frame(xyz, timestamps)
            except TypeError:
                # 回退到无时间戳的 API
                self._slam.register_frame(xyz)
                
        except Exception as e:
            print(f"[ERROR] SLAM 注册失败: {e}")
            return
        
        # 成功处理的帧计数
        self._total_frames_processed += 1
        
        # 获取地图
        try:
            if hasattr(self._slam, 'get_map'):
                cloud = self._slam.get_map()
            elif hasattr(self._slam, 'local_map'):
                cloud = self._slam.local_map.point_cloud()
            else:
                print("[ERROR] 无法获取地图 - 未知的 SLAM API")
                return
                
        except Exception as e:
            print(f"[ERROR] 获取地图失败: {e}")
            return
        
        # 地图数据安全检查
        if cloud is None or cloud.size == 0:
            print("[WARNING] 空地图，跳过可视化")
            return
        
        # 应用挂载方向校正
        if _R_MOUNT is not None:
            try:
                cloud = (cloud @ _R_MOUNT[:3, :3].T).astype(cloud.dtype, copy=False)
            except Exception as e:
                print(f"[WARNING] 挂载校正失败: {e}")

        # 可视化下采样
        if cloud.shape[0] > self._vis_max_points:
            step = max(1, int(cloud.shape[0] / self._vis_max_points))
            cloud = cloud[::step]

        # 获取当前位姿
        try:
            pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            if _R_MOUNT is not None:
                pose = _R_MOUNT @ pose
            
            # 保存位姿到轨迹
            self._trajectory.append(pose.copy())
            
        except Exception as e:
            print(f"[WARNING] 获取位姿失败: {e}")
            pose = np.eye(4)

        # 推送到可视化器
        try:
            self._viewer.push(cloud, pose)
        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")

    def save_slam_data(self) -> bool:
        """
        保存 SLAM 数据到 data 目录
        
        Returns:
            bool: 保存成功返回 True
        """
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
                # 应用挂载校正
                if _R_MOUNT is not None:
                    final_map = (final_map @ _R_MOUNT[:3, :3].T).astype(final_map.dtype, copy=False)
                
                # 保存为多种格式
                for fmt in ["ply", "pcd"]:
                    total_saves += 1
                    map_file = session_dir / f"final_map.{fmt}"
                    if _save_point_cloud(final_map, map_file):
                        success_count += 1
                    
                # 可选：保存为纯文本格式
                if DEFAULT_MAP_FORMAT == "xyz":
                    total_saves += 1
                    map_file = session_dir / "final_map.xyz"
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
        
        # 3. 保存元数据
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
                    "total_frames_processed": self._total_frames_processed
                },
                "slam_config": {
                    "voxel_size": _P["voxel_size"],
                    "max_range": _P["max_range"],
                    "min_motion": _P["min_motion"],
                    "frame_time": _P["frame_time"],
                    "frame_packets": _P["frame_packets"]
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
                    "final_map_points": len(final_map) if 'final_map' in locals() and final_map is not None else 0
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
    """主函数 - 启动 SLAM 演示"""
    print(f"[INFO] 启动 Livox SLAM (预设: {PRESET}, 挂载: {MOUNT})")
    print(f"[INFO] 数据将保存到: {DATA_DIR.absolute()}")
    
    demo = LiveSLAMDemo()

    # 支持 Ctrl-C 中断
    stop = False

    def _sigint(*_):
        nonlocal stop
        print("\n[INFO] 收到中断信号，正在优雅关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        print("[INFO] SLAM 已启动，按 Ctrl-C 停止并保存数据")
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()

if __name__ == "__main__":
    main()