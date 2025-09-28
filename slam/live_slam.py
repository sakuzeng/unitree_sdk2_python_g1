#!/usr/bin/env python3
"""
Livox MID-360 激光雷达实时 SLAM 演示程序 (优化版)

优化特性:
- 支持静止环境初始化
- 增强点云预处理（反射强度+噪声过滤）
- 改进IMU融合（静止检测+偏置校准）
- 放宽配准阈值，增加残差检查
- 强制关键帧生成
- 调试日志和点云保存
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
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation

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

# 新增：调试点云保存目录
DEBUG_DIR = DATA_DIR / "debug"
DEBUG_DIR.mkdir(exist_ok=True)

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
# 场景预设配置（优化静止环境）
# ---------------------------------------------------------------------------

PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        "frame_time": 0.1,
        "frame_packets": 50,
        "voxel_size": 0.1,  # 减小体素，适应稀疏点云
        "max_range": 25.0,
        "min_range": 0.5,
        "max_points_per_voxel": 5,  # 减少点数，提高精度
        "max_num_iterations": 50,  # 减少迭代，加快处理
        "convergence_criterion": 1e-4,  # 放宽收敛
        "max_num_threads": 0,
        "initial_threshold": 0.5,  # 降低阈值
        "min_motion_th": 0.01,  # 降低运动阈值
        "deskew": False,  # 静止环境禁用deskew
        "downsample_limit": 1_000_000,
        "keyframe_min_distance": 0.2,
        "keyframe_min_rotation": 5.0,
        "min_points_threshold": 200,  # 降低点数要求
    },
    "outdoor": {
        "frame_time": 0.1,
        "frame_packets": 80,
        "voxel_size": 0.3,
        "max_range": 100.0,
        "min_range": 1.0,
        "max_points_per_voxel": 10,
        "max_num_iterations": 50,
        "convergence_criterion": 5e-4,
        "max_num_threads": 0,
        "initial_threshold": 1.0,
        "min_motion_th": 0.05,
        "deskew": False,
        "downsample_limit": 2_000_000,
        "keyframe_min_distance": 0.5,
        "keyframe_min_rotation": 10.0,
        "min_points_threshold": 200,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"未知预设 '{PRESET}'. 请选择 {list(_PRESETS.keys())} 中的一个.")

_P = _PRESETS[PRESET]

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
                rot = Rotation.from_matrix(rotation_matrix)
                quaternion = rot.as_quat()
                quaternion = np.roll(quaternion, 1)
            except ImportError:
                quaternion = [1.0, 0.0, 0.0, 0.0]
            trajectory_data.append([
                float(i), translation[0], translation[1], translation[2],
                quaternion[0], quaternion[1], quaternion[2], quaternion[3]
            ])
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
# 可视化器类（优化显示）
# ---------------------------------------------------------------------------

class _Viewer:
    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        self._local_pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._local_pcd)
        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_local_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._first = True
        self._show_local_only = os.environ.get("SHOW_LOCAL_ONLY", "false").lower() == "true"

    def push(self, xyz: np.ndarray, local_xyz: np.ndarray, pose: np.ndarray):
        self._latest_pts = xyz
        self._latest_local_pts = local_xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        updated = False
        if self._latest_pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(self._latest_pts)
            self._vis.update_geometry(self._pcd)
            self._latest_pts = None
            updated = True
        if self._latest_local_pts is not None:
            self._local_pcd.points = o3d.utility.Vector3dVector(self._latest_local_pts)
            self._vis.update_geometry(self._local_pcd)
            self._latest_local_pts = None
            updated = True
        if self._latest_pose is not None:
            self._update_pose_vis(self._latest_pose)
            self._latest_pose = None
            updated = True
        if self._show_local_only:
            self._pcd.paint_uniform_color([0.5, 0.5, 0.5])
            self._local_pcd.paint_uniform_color([1.0, 0.0, 0.0])
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

def _check_point_cloud_quality(xyz: np.ndarray, reflectivity: np.ndarray = None) -> tuple[bool, str]:
    if xyz.size == 0:
        return False, "空点云"
    if xyz.shape[0] < _P["min_points_threshold"]:
        return False, f"点数太少 ({xyz.shape[0]} < {_P['min_points_threshold']})"
    if not np.isfinite(xyz).all():
        invalid_count = (~np.isfinite(xyz)).sum()
        return False, f"包含 {invalid_count} 个无效点"
    ranges = np.ptp(xyz, axis=0)
    if np.any(ranges < 0.1):
        return False, f"点云分布范围过小: {ranges}"
    volume = np.prod(ranges)
    density = xyz.shape[0] / volume if volume > 0 else 0
    if density < 0.5:  # 放宽密度要求
        return False, f"点云密度过低: {density:.2f} 点/m³"
    # 新增：反射强度检查
    if reflectivity is not None:
        if np.any(reflectivity < 10):  # 过滤低强度点
            low_reflect_count = np.sum(reflectivity < 10)
            if low_reflect_count / len(reflectivity) > 0.5:
                return False, f"低反射强度点过多 ({low_reflect_count}/{len(reflectivity)})"
    return True, "质量合格"

def _preprocess_point_cloud(xyz: np.ndarray, reflectivity: np.ndarray = None) -> np.ndarray:
    if xyz.size == 0:
        return xyz
    # 移除无效点
    valid_mask = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid_mask]
    if reflectivity is not None:
        reflectivity = reflectivity[valid_mask]
    if xyz.size == 0:
        return xyz
    # 距离过滤
    distances = np.linalg.norm(xyz, axis=1)
    distance_mask = (distances >= _P["min_range"]) & (distances <= _P["max_range"])
    xyz = xyz[distance_mask]
    if reflectivity is not None:
        reflectivity = reflectivity[distance_mask]
    if xyz.size == 0:
        return xyz
    # 移除机器人自身反射
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
    if reflectivity is not None:
        reflectivity = reflectivity[~self_reflection_mask]
    if xyz.size == 0:
        return xyz
    # 新增：反射强度过滤
    if reflectivity is not None:
        intensity_mask = reflectivity >= 10
        xyz = xyz[intensity_mask]
    if xyz.shape[0] > 1000:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.3)  # 放宽半径
            if len(pcd.points) > 100:
                xyz = np.asarray(pcd.points)
        except Exception as e:
            print(f"[WARNING] 统计过滤失败: {e}")
    if xyz.shape[0] > 1000:
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=50)
            non_ground = pcd.select_by_index(inliers, invert=True)
            xyz = np.asarray(non_ground.points)
        except Exception as e:
            print(f"[WARNING] 地面分割失败: {e}")
    if xyz.shape[0] > 1000:
        try:
            clustering = DBSCAN(eps=0.3, min_samples=3).fit(xyz)
            labels = clustering.labels_
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            valid_labels = unique_labels[counts > 30]
            mask = np.isin(labels, valid_labels)
            xyz = xyz[mask]
        except Exception as e:
            print(f"[WARNING] 动态过滤失败: {e}")
    return xyz

# ---------------------------------------------------------------------------
# SLAM演示类
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
        self._min_init_frames = 5  # 减少帧数要求
        self._init_poses = []
        self._last_successful_pose = np.eye(4)
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5
        self._start_time = datetime.now()
        self._trajectory: List[np.ndarray] = []
        self._total_frames_processed = 0
        self._imu_buffer: List[tuple[np.ndarray, int]] = []
        self._imu_count = 0
        self._last_imu_time = 0
        self._imu_bias = np.zeros(6)  # [gx, gy, gz, ax, ay, az]
        self._is_stationary = True
        self._keyframe_map = o3d.geometry.PointCloud()
        self._last_keyframe_pose = np.eye(4)
        self._keyframe_poses = []
        self._frame_counter = 0  # 用于强制关键帧
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
            if hasattr(cfg, 'loop_closure'):
                cfg.loop_closure.enabled = True
                cfg.loop_closure.threshold = 0.2
            print("[INFO] KISS-ICP 配置已优化")
        except Exception as e:
            print(f"[WARNING] 应用配置时出错: {e}")

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        current_time = time.time()
        self._frame_count += 1
        self._frame_counter += 1
        if current_time - self._last_frame_time < 0.1:
            return
        self._last_frame_time = current_time

        # 坐标系校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
        if _R_MOUNT is not None:
            xyz_homo = np.column_stack([xyz, np.ones(xyz.shape[0])])
            xyz_corrected = (xyz_homo @ _R_MOUNT.T)[:, :3]
            xyz = xyz_corrected.astype(xyz.dtype, copy=False)

        # 点云预处理
        xyz = _preprocess_point_cloud(xyz, reflectivity)

        # 质量检查
        is_valid, quality_msg = _check_point_cloud_quality(xyz, reflectivity)
        if not is_valid:
            self._skip_frames += 1
            if self._skip_frames % 10 == 0:
                print(f"[WARNING] 跳过低质量帧: {quality_msg} (已跳过 {self._skip_frames} 帧)")
            # 保存失败帧用于调试
            _save_point_cloud(xyz, DEBUG_DIR / f"failed_frame_{self._frame_count}.ply")
            return

        self._skip_frames = 0

        # SLAM处理
        try:
            num_points = xyz.shape[0]
            timestamps = np.linspace(0.0, 0.1, num_points, dtype=np.float64)
            prev_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            initial_guess = self._predict_pose_from_imu(prev_pose)
            try:
                frame_result = self._slam.register_frame(xyz, timestamps, initial_guess=initial_guess)
            except TypeError:
                frame_result = self._slam.register_frame(xyz, timestamps)
            current_pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)

            # 配准质量验证
            if self._validate_registration(prev_pose, current_pose):
                self._last_successful_pose = current_pose.copy()
                self._consecutive_failures = 0
                self._processed_frames += 1
                self._trajectory.append(current_pose.copy())
                self._update_keyframe_map(xyz, current_pose)
            else:
                self._consecutive_failures += 1
                print(f"[WARNING] 配准质量不佳 (连续失败: {self._consecutive_failures})")
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("[WARNING] 连续配准失败，尝试系统重置")
                    self._reset_slam_system()
                    return

            # 保存成功帧用于调试
            if self._processed_frames % 10 == 0:
                _save_point_cloud(xyz, DEBUG_DIR / f"frame_{self._processed_frames}.ply")

        except Exception as e:
            print(f"[ERROR] SLAM 处理失败: {e}")
            self._consecutive_failures += 1
            return

        # 初始化状态管理
        if not self._is_initialized:
            self._initialization_frames += 1
            self._init_poses.append(current_pose.copy())
            if self._initialization_frames >= self._min_init_frames:
                if self._validate_initialization():
                    self._is_initialized = True
                    print(f"[INFO] SLAM 系统已成功初始化 ({self._initialization_frames} 帧)")
                else:
                    print("[WARNING] 初始化质量不佳，继续收集更多帧")
                    self._min_init_frames += 3

        # 可视化更新
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
            local_cloud = self._slam.local_map.point_cloud() if hasattr(self._slam, 'local_map') else cloud
            self._viewer.push(np.asarray(self._keyframe_map.points), local_cloud, current_pose)
        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")

        if self._processed_frames % 50 == 0:
            print(f"[INFO] 已处理 {self._processed_frames} 帧，当前位置: "
                  f"({current_pose[0,3]:.2f}, {current_pose[1,3]:.2f}, {current_pose[2,3]:.2f})")

    def _predict_pose_from_imu(self, prev_pose: np.ndarray) -> np.ndarray:
        if not self._imu_buffer or self._is_stationary:
            return np.eye(4)
        try:
            latest_imu, latest_ts = self._imu_buffer[-1]
            if len(latest_imu) == 0:
                return np.eye(4)
            mean_imu = np.mean(latest_imu, axis=0) - self._imu_bias
            dt = (latest_ts - self._last_imu_time) / 1e9 if self._last_imu_time > 0 else 0.05
            self._last_imu_time = latest_ts
            angular_velocity = mean_imu[:3] * dt
            rotation_delta = o3d.geometry.get_rotation_matrix_from_axis_angle(angular_velocity)
            acceleration = mean_imu[3:]
            velocity_delta = acceleration * dt
            translation_delta = velocity_delta * dt / 2
            delta_pose = np.eye(4)
            delta_pose[:3, :3] = rotation_delta
            delta_pose[:3, 3] = translation_delta
            return prev_pose @ delta_pose
        except Exception as e:
            print(f"[WARNING] IMU预测失败: {e}")
            return np.eye(4)

    def _calibrate_imu_bias(self):
        if not self._imu_buffer:
            return
        try:
            imu_data = np.concatenate([data for data, _ in self._imu_buffer], axis=0)
            self._imu_bias = np.mean(imu_data, axis=0)
            print(f"[INFO] IMU偏置校准: {self._imu_bias}")
            # 检测是否静止
            std_dev = np.std(imu_data, axis=0)
            if np.all(std_dev[:3] < 0.05) and np.all(std_dev[3:] < 0.1):
                self._is_stationary = True
                print("[INFO] 检测到静止状态")
            else:
                self._is_stationary = False
        except Exception as e:
            print(f"[WARNING] IMU偏置校准失败: {e}")

    def _update_keyframe_map(self, xyz: np.ndarray, current_pose: np.ndarray):
        delta_pose = np.linalg.inv(self._last_keyframe_pose) @ current_pose
        translation_change = np.linalg.norm(delta_pose[:3, 3])
        rotation_angle = np.arccos(np.clip((np.trace(delta_pose[:3, :3]) - 1) / 2, -1.0, 1.0))
        rotation_deg = np.degrees(rotation_angle)
        # 强制每10帧添加关键帧
        if (translation_change > _P["keyframe_min_distance"] or 
            rotation_deg > _P["keyframe_min_rotation"] or 
            self._frame_counter >= 10):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.transform(current_pose)
            self._keyframe_map += pcd
            self._keyframe_map = self._keyframe_map.voxel_down_sample(voxel_size=_P["voxel_size"])
            self._last_keyframe_pose = current_pose.copy()
            self._keyframe_poses.append(current_pose.copy())
            self._frame_counter = 0
            self._attempt_loop_closure(current_pose)

    def _attempt_loop_closure(self, current_pose: np.ndarray):
        if len(self._keyframe_poses) < 10:
            return
        try:
            distances = [np.linalg.norm(pose[:3, 3] - current_pose[:3, 3]) for pose in self._keyframe_poses[:-5]]
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] < 0.5:
                print(f"[INFO] 检测到潜在回环 (距离: {distances[min_dist_idx]:.3f}m)")
                historical_pose = self._keyframe_poses[min_dist_idx]
                correction = np.linalg.inv(historical_pose) @ current_pose
                self._slam.last_pose = historical_pose @ correction
        except Exception as e:
            print(f"[WARNING] 回环检测失败: {e}")

    def _validate_registration(self, prev_pose: np.ndarray, current_pose: np.ndarray) -> bool:
        try:
            delta_pose = np.linalg.inv(prev_pose) @ current_pose
            translation_change = np.linalg.norm(delta_pose[:3, 3])
            rotation_matrix = delta_pose[:3, :3]
            rotation_angle = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1.0, 1.0))
            max_translation = 3.0  # 放宽阈值
            max_rotation = np.radians(30)  # 放宽阈值
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
            # 检查位姿稳定性（静止环境）
            max_single_step = 0.0
            for i in range(1, len(self._init_poses)):
                delta = self._init_poses[i][:3, 3] - self._init_poses[i-1][:3, 3]
                step_distance = np.linalg.norm(delta)
                max_single_step = max(max_single_step, step_distance)
            if max_single_step > 0.5:  # 放宽阈值
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
            self._min_init_frames = 5
            self._init_poses.clear()
            self._consecutive_failures = 0
            self._last_successful_pose = np.eye(4)
            self._keyframe_map = o3d.geometry.PointCloud()
            self._last_keyframe_pose = np.eye(4)
            self._keyframe_poses = []
            self._frame_counter = 0
            print("[INFO] SLAM 系统重置完成")
        except Exception as e:
            print(f"[ERROR] SLAM 系统重置失败: {e}")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        if len(self._imu_buffer) >= 100:
            self._calibrate_imu_bias()
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
            final_map = np.asarray(self._keyframe_map.points)
            if final_map.size > 0:
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
                    "total_imu_samples": self._imu_count,
                    "keyframe_count": len(self._keyframe_poses)
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
                    "deskew": _P["deskew"],
                    "keyframe_min_distance": _P["keyframe_min_distance"],
                    "keyframe_min_rotation": _P["keyframe_min_rotation"]
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
                    "final_map_points": len(self._keyframe_map.points),
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
        except Exception as e:
            print(f"[WARNING] 关闭可视化器时出错: {e}")
        print("[INFO] SLAM 系统已安全关闭")

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print("="*60)
    print(f"启动 Livox SLAM (优化版)")
    print(f"预设: {PRESET.upper()}")
    print(f"挂载方向: {MOUNT}")
    print(f"数据保存目录: {DATA_DIR.absolute()}")
    print(f"调试点云保存目录: {DEBUG_DIR.absolute()}")
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
        print("[INFO] 系统支持静止环境初始化")
        print("[INFO] 按 Ctrl-C 停止并保存数据")
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] 收到键盘中断")
    finally:
        demo.shutdown()

if __name__ == "__main__":
    main()