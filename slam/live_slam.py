#!/usr/bin/env python3
"""
Livox MID-360 激光雷达实时 SLAM 演示程序 (改进版，集成 2D 占用网格)

改进特性:
- 严格的 KISS-ICP 配置优化
- 增强的点云质量检查和预处理
- ICP 配准质量验证
- 自适应参数调整
- 更好的初始化和错误恢复机制
- 实时 2D 占用网格生成和可视化
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
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
import time
# ---------------------------------------------------------------------------
# 数据保存配置
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

MAP_FORMATS = {
    "ply": "PLY 格式 (推荐)",
    "pcd": "PCD 格式",
    "xyz": "纯坐标文本格式"
}

DEFAULT_MAP_FORMAT = os.environ.get("MAP_SAVE_FORMAT", "ply").lower()
if DEFAULT_MAP_FORMAT not in MAP_FORMATS:
    print(f"[WARNING] 未知的地图格式 '{DEFAULT_MAP_FORMAT}'，使用默认格式 'ply'")
    DEFAULT_MAP_FORMAT = "ply"

# 占用网格参数
GRID_RESOLUTION = 0.1  # 每格 0.1 米
GRID_SIZE = 1000  # 网格尺寸 1000x1000 格（±50 米）
GRID_MIN = -50.0  # X-Y 范围：[-50, 50] 米
GRID_MAX = 50.0

# ---------------------------------------------------------------------------
# 挂载方向校正配置
# ---------------------------------------------------------------------------

_VALID_MOUNTS = {"normal", "upside_down"}
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in _VALID_MOUNTS:
    raise SystemExit(f"LIVOX_MOUNT 必须是 {_VALID_MOUNTS} 中的一个")

_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()
if _TILT_AXIS not in {"x", "y", "z"}:
    raise SystemExit("LIDAR_TILT_AXIS 必须是 'x', 'y', 'z' 中的一个")

try:
    _TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
except ValueError:
    _TILT_DEG = 0.0

_R_MOUNT = None

_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)

if abs(_TILT_DEG) > 1e-3:
    _rad = math.radians(-_TILT_DEG)
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
    else:
        _R_TILT = np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
else:
    _R_TILT = np.eye(4)

_R_TOTAL = _R_TILT @ _R_FLIP
if not np.allclose(_R_TOTAL, np.eye(4)):
    _R_MOUNT = _R_TOTAL

# ---------------------------------------------------------------------------
# KISS-ICP 导入逻辑
# ---------------------------------------------------------------------------

KissICP = None
_IMPORT_ERRORS = []

try:
    from kiss_icp.pipeline import KissICP
except Exception as e:
    _IMPORT_ERRORS.append(e)

if KissICP is None:
    try:
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
        "frame_time": 0.1,
        "frame_packets": 50,
        "voxel_size": 0.25,
        "max_range": 25.0,
        "min_range": 0.5,
        "max_points_per_voxel": 10,
        "max_num_iterations": 100,
        "convergence_criterion": 1e-5,
        "max_num_threads": 0,
        "initial_threshold": 1.0,
        "min_motion_th": 0.05,
        "deskew": True,
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

_P = _PRESETS[PRESET]

# ---------------------------------------------------------------------------
# 2D 占用网格管理器
# ---------------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, resolution: float = 0.1, size: int = 1000, min_coord: float = -50.0, max_coord: float = 50.0):
        self.resolution = resolution
        self.size = size
        self.min_coord = min_coord
        self.max_coord = max_coord
        # 使用稀疏矩阵优化内存
        self.grid = np.zeros((size, size), dtype=np.float32)
        # 初始化绘图
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10), facecolor='white')
        self.ax = self.fig.add_subplot(111)
        # 设置自定义颜色映射
        cmap = colors.ListedColormap(['white', 'gray', 'black'])
        bounds = [0, 10, 50, 100]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        # 初始化图像
        self.im = self.ax.imshow(
            self.grid,
            cmap=cmap,
            norm=norm,
            extent=[min_coord, max_coord, min_coord, max_coord],
            origin='lower',
            interpolation='nearest'
        )
        
        # 设置标题和标签
        self.ax.set_title("实时 2D 占用网格", fontsize=14, pad=15)
        self.ax.set_xlabel("X (米)", fontsize=12)
        self.ax.set_ylabel("Y (米)", fontsize=12)
        
        # 添加网格线
        self.ax.grid(True, which='major', linestyle='--', alpha=0.3)
        self.ax.set_xticks(np.arange(min_coord, max_coord + resolution, 5.0))
        self.ax.set_yticks(np.arange(min_coord, max_coord + resolution, 5.0))
        
        # 添加当前位置标记
        self.position_marker = Circle((0, 0), 0.5, color='red', alpha=0.6)
        self.ax.add_patch(self.position_marker)
        
        # 添加颜色条
        cbar = self.fig.colorbar(self.im, ax=self.ax)
        cbar.set_label('占用概率', fontsize=10)
        cbar.set_ticks([0, 50, 100])
        cbar.set_label(['空闲', '未知', '占用'])
        
        self.last_update = time.time()
        self.update_interval = 0.1  # 更新间隔0.1秒
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, xyz: np.ndarray, current_pose: np.ndarray = None):
        if time.time() - self.last_update < self.update_interval:
            return
            
        # 更新点云
        points = xyz[:, :2]
        mask = (points[:, 0] >= self.min_coord) & (points[:, 0] < self.max_coord) & \
               (points[:, 1] >= self.min_coord) & (points[:, 1] < self.max_coord)
        points = points[mask]
        
        if len(points) == 0:
            return
            
        # 转换为网格索引
        indices = ((points - self.min_coord) / self.resolution).astype(np.int32)
        valid_indices = (indices[:, 0] >= 0) & (indices[:, 0] < self.size) & \
                       (indices[:, 1] >= 0) & (indices[:, 1] < self.size)
        indices = indices[valid_indices]
        
        # 更新网格
        np.add.at(self.grid, (indices[:, 1], indices[:, 0]), 5.0)
        self.grid = np.clip(self.grid, 0, 100)
        
        # 衰减机制
        decay_mask = self.grid > 0
        self.grid[decay_mask] = np.maximum(self.grid[decay_mask] - 1.0, 0)
        
        # 更新当前位置
        if current_pose is not None:
            position = current_pose[:2, 3]
            if (self.min_coord <= position[0] <= self.max_coord and 
                self.min_coord <= position[1] <= self.max_coord):
                self.position_marker.center = (position[0], position[1])
        
        # 更新显示
        self.im.set_data(self.grid)
        self.ax.draw_artist(self.im)
        self.ax.draw_artist(self.position_marker)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()
        self.last_update = time.time()

    def save(self, filename: str):
        # 保存前确保显示最新状态
        self.im.set_data(self.grid)
        plt.imsave(filename, self.grid, cmap=self.im.cmap, vmin=0, vmax=100)
        print(f"[OccupancyGrid] 已保存网格到 {filename}")

    def close(self):
        plt.close(self.fig)

# ---------------------------------------------------------------------------
# 数据保存工具函数
# ---------------------------------------------------------------------------

def _generate_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_point_cloud(cloud: np.ndarray, file_path: Path) -> bool:
    if cloud is None or cloud.size == 0:
        print(f"[WARNING] 空点云，跳过保存到 {file_path}")
        return False
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        ext = file_path.suffix.lower()
        if ext in [".ply", ".pcd"]:
            success = o3d.io.write_point_cloud(str(file_path), pcd)
        elif ext == ".xyz":
            np.savetxt(file_path, cloud, fmt="%.6f", delimiter=" ", header="x y z", comments="# ")
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
    if not poses:
        print(f"[WARNING] 空轨迹，跳过保存到 {file_path}")
        return False
    try:
        trajectory_data = []
        for i, pose in enumerate(poses):
            if pose is None:
                continue
            translation = pose[:3, 3]
            rotation_matrix = pose[:3, :3]
            try:
                from scipy.spatial.transform import Rotation
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()
                quaternion = np.roll(quaternion, 1)
            except ImportError:
                quaternion = [1.0, 0.0, 0.0, 0.0]
            trajectory_data.append([float(i)] + translation.tolist() + quaternion.tolist())
        np.savetxt(file_path, trajectory_data, fmt="%.6f", delimiter=" ",
                   header="timestamp x y z qw qx qy qz", comments="# ")
        print(f"[INFO] 轨迹已保存: {file_path} ({len(trajectory_data)} 位姿)")
        return True
    except Exception as e:
        print(f"[ERROR] 保存轨迹时出错: {e}")
        return False

def _save_imu_data(imu_buffer: List[tuple[np.ndarray, int]], file_path: Path) -> bool:
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
    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._first = True

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        updated = False
        if self._latest_pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(self._latest_pts)
            self._vis.update_geometry(self._pcd)
            self._latest_pts = None
            updated = True
        if self._latest_pose is not None:
            self._update_pose_vis(self._latest_pose)
            self._latest_pose = None
            updated = True
        if self._first and updated:
            self._vis.reset_view_point(True)
            self._first = False
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def _update_pose_vis(self, pose: np.ndarray):
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)
        size = 0.5
        if len(self._pcd.points) > 0:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03
            size = max(0.2, min(size, 2.0))
        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)
        self._vis.update_geometry(self._cam_frame)

    def close(self):
        self._vis.destroy_window()

# ---------------------------------------------------------------------------
# 点云质量检查和预处理
# ---------------------------------------------------------------------------

def _check_point_cloud_quality(xyz: np.ndarray) -> tuple[bool, str]:
    if xyz.size == 0:
        return False, "空点云"
    if xyz.shape[0] < 500:
        return False, f"点数太少 ({xyz.shape[0]} < 500)"
    if not np.isfinite(xyz).all():
        invalid_count = (~np.isfinite(xyz)).sum()
        return False, f"包含 {invalid_count} 个无效点"
    ranges = np.ptp(xyz, axis=0)
    if np.any(ranges < 0.1):
        return False, f"点云分布范围过小: {ranges}"
    volume = np.prod(ranges)
    density = xyz.shape[0] / volume if volume > 0 else 0
    if density < 1.0:
        return False, f"点云密度过低: {density:.2f} 点/m³"
    return True, "质量合格"

def _preprocess_point_cloud(xyz: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return xyz
    valid_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid_mask]
    if xyz.size == 0:
        return xyz
    distances = np.linalg.norm(xyz, axis=1)
    distance_mask = (distances >= _P["min_range"]) & (distances <= _P["max_range"])
    xyz = xyz[distance_mask]
    if xyz.size == 0:
        return xyz
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
    if xyz.shape[0] > 1000:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd, _ = pcd.remove_radius_outlier(nb_points=3, radius=0.5)
            if len(pcd.points) > 100:
                xyz = np.asarray(pcd.points)
        except Exception as e:
            print(f"[WARNING] 统计过滤失败: {e}")
    return xyz

def _advanced_preprocess_point_cloud(xyz: np.ndarray, reflectivity: np.ndarray = None) -> np.ndarray:
    """
    增强的点云预处理，包含多级滤波和优化
    
    Args:
        xyz: 原始点云坐标
        reflectivity: 反射强度（可选）
        
    Returns:
        处理后的点云
    """
    if xyz.size == 0:
        return xyz
    
    # 1. 基础有效性检查
    valid_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid_mask]
    if reflectivity is not None:
        reflectivity = reflectivity[valid_mask]
    
    if xyz.size == 0:
        return xyz
    
    # 2. 距离范围滤波（改进版）
    distances = np.linalg.norm(xyz, axis=1)
    min_range = _P["min_range"]
    max_range = _P["max_range"]
    
    # 自适应范围调整
    if len(xyz) > 10000:  # 密集点云适当缩小范围
        max_range *= 0.9
    elif len(xyz) < 1000:  # 稀疏点云扩大范围
        max_range *= 1.1
        min_range *= 0.8
    
    distance_mask = (distances >= min_range) & (distances <= max_range)
    xyz = xyz[distance_mask]
    if reflectivity is not None:
        reflectivity = reflectivity[distance_mask]
    
    if xyz.size == 0:
        return xyz
    
    # 3. 改进的自反射过滤
    r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", "0.20"))
    dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", "0.15"))
    
    # 考虑激光器高度的动态自反射过滤
    dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
    height_dependent_radius = r_xy * (1 + np.abs(xyz[:, 2]) / 2.0)  # 高度越大，过滤半径越大
    
    close_mask = dist_xy < height_dependent_radius
    near_plane_mask = np.abs(xyz[:, 2]) < dz
    self_reflection_mask = close_mask & near_plane_mask
    xyz = xyz[~self_reflection_mask]
    if reflectivity is not None:
        reflectivity = reflectivity[~self_reflection_mask]
    
    # 4. 基于反射强度的滤波（如果有反射强度数据）
    if reflectivity is not None and len(reflectivity) == len(xyz):
        # 过滤反射强度过低的点（可能是噪声）
        reflectivity_threshold = np.percentile(reflectivity, 10)  # 动态阈值
        high_quality_mask = reflectivity > reflectivity_threshold
        xyz = xyz[high_quality_mask]
    
    # 5. 多层级统计滤波
    if xyz.shape[0] > 1000:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # 第一级：粗糙的离群点移除
            pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.8)
            
            # 第二级：精细的统计滤波
            if len(pcd.points) > 500:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            if len(pcd.points) > 100:
                xyz = np.asarray(pcd.points)
        except Exception as e:
            print(f"[WARNING] 统计滤波失败: {e}")
    
    # 6. 地面点检测和处理（改进SLAM性能）
    if xyz.shape[0] > 500:
        xyz = _process_ground_points(xyz)
    
    # 7. 自适应下采样
    if xyz.shape[0] > _P["downsample_limit"]:
        # 基于点云密度的智能下采样
        target_points = _P["downsample_limit"]
        voxel_size = _estimate_optimal_voxel_size(xyz, target_points)
        
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd = pcd.voxel_down_sample(voxel_size)
            xyz = np.asarray(pcd.points)
        except Exception as e:
            # 回退到简单下采样
            step = max(1, int(xyz.shape[0] / target_points))
            xyz = xyz[::step]
    
    return xyz

def _process_ground_points(xyz: np.ndarray) -> np.ndarray:
    """
    地面点检测和处理，保留部分地面点用于配准
    """
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # RANSAC平面分割检测地面
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) > len(xyz) * 0.3:  # 如果地面点过多
            # 保留30%的地面点，用于位姿估计
            ground_points = np.asarray(pcd.select_by_index(inliers).points)
            non_ground_points = np.asarray(pcd.select_by_index(inliers, invert=True).points)
            
            # 均匀采样地面点
            if len(ground_points) > 1000:
                indices = np.random.choice(len(ground_points), 1000, replace=False)
                ground_points = ground_points[indices]
            
            xyz = np.vstack([non_ground_points, ground_points])
    except Exception as e:
        print(f"[WARNING] 地面点处理失败: {e}")
    
    return xyz

def _estimate_optimal_voxel_size(xyz: np.ndarray, target_points: int) -> float:
    """
    估计最优体素大小进行下采样
    """
    if xyz.size == 0:
        return 0.1
    
    # 计算点云包围盒
    min_bound = np.min(xyz, axis=0)
    max_bound = np.max(xyz, axis=0)
    bbox_volume = np.prod(max_bound - min_bound)
    
    # 基于目标点数估算体素大小
    target_density = target_points / bbox_volume if bbox_volume > 0 else 1000
    voxel_volume = 1.0 / target_density
    voxel_size = np.cbrt(voxel_volume)
    
    # 限制体素大小范围
    voxel_size = np.clip(voxel_size, 0.05, 0.5)
    
    return float(voxel_size)
# ---------------------------------------------------------------------------
# 改进的主要 SLAM 演示类
# ---------------------------------------------------------------------------

class LiveSLAMDemo(_Livox):
    def __init__(self):
        _sdk_kwargs = {}
        if _Livox.__name__ == "Livox2":
            _sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])
        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.164", **_sdk_kwargs)
        except TypeError:
            super().__init__()
        self._slam = self._create_optimized_slam()
        self._viewer = _Viewer()
        self._vis_max_points = _P["downsample_limit"]
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._processed_frames = 0
        self._skip_frames = 0
        self._is_initialized = False
        self._initialization_frames = 0
        self._min_init_frames = 10
        self._init_poses = []
        self._last_successful_pose = np.eye(4)
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._start_time = datetime.now()
        self._trajectory: List[np.ndarray] = []
        self._total_frames_processed = 0
        self._imu_buffer: List[tuple[np.ndarray, int]] = []
        self._imu_count = 0
        self.occupancy_grid = OccupancyGrid()
        print(f"[INFO] SLAM 系统初始化完成 (预设: {PRESET})")
        print(f"[INFO] KISS-ICP 配置: voxel_size={_P['voxel_size']}, max_range={_P['max_range']}")

    def _create_optimized_slam(self) -> KissICP:
        try:
            from kiss_icp.config import load_config
            cfg = load_config(config_file=None, max_range=_P["max_range"])
            self._apply_optimized_config(cfg)
            return KissICP(cfg)
        except Exception as e:
            print(f"[ERROR] 创建 SLAM 配置失败: {e}")
            raise SystemExit("无法创建 KISS-ICP 实例") from e

    def _apply_optimized_config(self, cfg):
        try:
            if hasattr(cfg, 'data'):
                cfg.data.max_range = _P["max_range"]
                cfg.data.min_range = _P["min_range"]
                cfg.data.deskew = _P["deskew"]
            if hasattr(cfg, 'mapping'):
                cfg.mapping.voxel_size = _P["voxel_size"]
                cfg.mapping.max_points_per_voxel = _P["max_points_per_voxel"]
            if hasattr(cfg, 'registration'):
                cfg.registration.max_num_iterations = _P["max_num_iterations"]
                cfg.registration.convergence_criterion = _P["convergence_criterion"]
                cfg.registration.max_num_threads = _P["max_num_threads"]
            if hasattr(cfg, 'adaptive_threshold'):
                cfg.adaptive_threshold.initial_threshold = _P["initial_threshold"]
                cfg.adaptive_threshold.min_motion_th = _P["min_motion_th"]
            print("[INFO] KISS-ICP 配置已优化")
        except Exception as e:
            print(f"[WARNING] 应用配置时出错: {e}")

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        current_time = time.time()
        self._frame_count += 1
        if current_time - self._last_frame_time < 0.1:
            return
        self._last_frame_time = current_time
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
        if _R_MOUNT is not None:
            xyz_homo = np.column_stack([xyz, np.ones(xyz.shape[0])])
            xyz_corrected = (xyz_homo @ _R_MOUNT.T)[:, :3]
            xyz = xyz_corrected.astype(xyz.dtype, copy=False)
        xyz = _preprocess_point_cloud(xyz)
        is_valid, quality_msg = _check_point_cloud_quality(xyz)
        if not is_valid:
            self._skip_frames += 1
            if self._skip_frames % 10 == 0:
                print(f"[WARNING] 跳过低质量帧: {quality_msg} (已跳过 {self._skip_frames} 帧)")
            return
        self._skip_frames = 0
        try:
            num_points = xyz.shape[0]
            timestamps = np.linspace(0.0, 0.1, num_points, dtype=np.float64)
            prev_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            frame_result = self._slam.register_frame(xyz, timestamps)
            current_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            if self._validate_registration(prev_pose, current_pose):
                self._last_successful_pose = current_pose.copy()
                self._consecutive_failures = 0
                self._processed_frames += 1
                self._trajectory.append(current_pose.copy())
            else:
                self._consecutive_failures += 1
                print(f"[WARNING] 配准质量不佳 (连续失败: {self._consecutive_failures})")
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("[WARNING] 连续配准失败，尝试系统重置")
                    self._reset_slam_system()
                    return
        except Exception as e:
            print(f"[ERROR] SLAM 处理失败: {e}")
            self._consecutive_failures += 1
            return
        if not self._is_initialized:
            self._initialization_frames += 1
            self._init_poses.append(current_pose.copy())
            if self._initialization_frames >= self._min_init_frames:
                if self._validate_initialization():
                    self._is_initialized = True
                    print(f"[INFO] SLAM 系统已成功初始化 ({self._initialization_frames} 帧)")
                else:
                    print("[WARNING] 初始化质量不佳，继续收集更多帧")
                    self._min_init_frames += 5
        try:
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
            if cloud.shape[0] > self._vis_max_points:
                step = max(1, int(cloud.shape[0] / self._vis_max_points))
                cloud = cloud[::step]
            self._viewer.push(cloud, current_pose)
            self.occupancy_grid.update(cloud)  # 更新占用网格
        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")
        if self._processed_frames % 50 == 0:
            print(f"[INFO] 已处理 {self._processed_frames} 帧，当前位置: "
                  f"({current_pose[0,3]:.2f}, {current_pose[1,3]:.2f}, {current_pose[2,3]:.2f})")

    def _validate_registration(self, prev_pose: np.ndarray, current_pose: np.ndarray) -> bool:
        try:
            delta_pose = np.linalg.inv(prev_pose) @ current_pose
            translation_change = np.linalg.norm(delta_pose[:3, 3])
            rotation_matrix = delta_pose[:3, :3]
            rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1.0, 1.0))
            max_translation = 5.0
            max_rotation = np.radians(30)
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
        if len(self._init_poses) < self._min_init_frames:
            return False
        try:
            total_distance = 0.0
            for i in range(1, len(self._init_poses)):
                delta = self._init_poses[i][:3, 3] - self._init_poses[i-1][:3, 3]
                total_distance += np.linalg.norm(delta)
            if total_distance < 0.5:
                print(f"[WARNING] 初始化期间运动不足: {total_distance:.3f}m")
                return False
            max_single_step = 0.0
            for i in range(1, len(self._init_poses)):
                delta = self._init_poses[i][:3, 3] - self._init_poses[i-1][:3, 3]
                step_distance = np.linalg.norm(delta)
                max_single_step = max(max_single_step, step_distance)
            if max_single_step > 2.0:
                print(f"[WARNING] 初始化期间存在位姿突变: {max_single_step:.3f}m")
                return False
            return True
        except Exception as e:
            print(f"[WARNING] 初始化验证失败: {e}")
            return False

    def _reset_slam_system(self):
        try:
            print("[INFO] 正在重置 SLAM 系统...")
            self._slam = self._create_optimized_slam()
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
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        if len(self._imu_buffer) >= 100:
            print(f"[INFO] 缓冲 {self._imu_count} 个 IMU 样本")
            self._imu_buffer = []

    def save_slam_data(self) -> bool:
        print("[INFO] 正在保存 SLAM 数据...")
        timestamp = _generate_timestamp()
        session_name = f"slam_session_{timestamp}"
        session_dir = DATA_DIR / session_name
        session_dir.mkdir(exist_ok=True)
        success_count = 0
        total_saves = 0
        try:
            if hasattr(self._slam, 'get_map'):
                final_map = self._slam.get_map()
            elif hasattr(self._slam, 'local_map'):
                final_map = self._slam.local_map.point_cloud()
            else:
                final_map = None
            if final_map is not None and final_map.size > 0:
                for fmt in ["ply", "pcd"]:
                    total_saves += 1
                    map_file = session_dir / f"final_map.{fmt}"
                    if _save_point_cloud(final_map, map_file):
                        success_count += 1
        except Exception as e:
            print(f"[ERROR] 保存地图时出错: {e}")
        if self._trajectory:
            total_saves += 1
            trajectory_file = session_dir / "trajectory.txt"
            if _save_trajectory(self._trajectory, trajectory_file):
                success_count += 1
        if self._imu_buffer:
            total_saves += 1
            imu_file = session_dir / "imu_data.csv"
            if _save_imu_data(self._imu_buffer, imu_file):
                success_count += 1
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
        self.occupancy_grid.save(str(session_dir / "occupancy_grid.png"))
        success_count += 1
        total_saves += 1
        print(f"[INFO] SLAM 数据保存完成: {success_count}/{total_saves} 文件成功保存")
        print(f"[INFO] 数据保存位置: {session_dir}")
        if success_count > 0:
            print("[INFO] 保存的文件:")
            for file_path in sorted(session_dir.glob("*")):
                file_size = file_path.stat().st_size / 1024 / 1024
                print(f"  - {file_path.name} ({file_size:.2f} MB)")
        return success_count > 0

    def shutdown(self):
        print("[INFO] 正在关闭 SLAM 系统...")
        print(f"[INFO] 会话统计: 收到 {self._frame_count} 帧，成功处理 {self._processed_frames} 帧")
        if self._is_initialized:
            print("[INFO] SLAM 系统已成功初始化并运行")
        else:
            print("[WARNING] SLAM 系统未能成功初始化")
        try:
            self.save_slam_data()
        except Exception as e:
            print(f"[ERROR] 保存数据时出错: {e}")
        try:
            super().shutdown()
        except Exception as e:
            print(f"[WARNING] 关闭 Livox 时出错: {e}")
        try:
            self._viewer.close()
            self.occupancy_grid.close()
        except Exception as e:
            print(f"[WARNING] 关闭可视化器时出错: {e}")
        print("[INFO] SLAM 系统已安全关闭")

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print(f"启动 Livox SLAM (改进版)")
    print(f"预设: {PRESET.upper()}")
    print(f"挂载方向: {MOUNT}")
    print(f"数据保存目录: {DATA_DIR.absolute()}")
    print("="*60)
    print("[INFO] 关键配置参数:")
    print(f"  - 体素大小: {_P['voxel_size']} m")
    print(f"  - 最大距离: {_P['max_range']} m")
    print(f"  - ICP 迭代次数: {_P['max_num_iterations']}")
    print(f"  - 收敛标准: {_P['convergence_criterion']}")
    print(f"  - 初始阈值: {_P['initial_threshold']}")
    demo = LiveSLAMDemo()
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