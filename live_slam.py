"""Real-time LiDAR-SLAM demo for a Livox MID-360.

Prerequisites
-------------
1. Build & install the Livox-SDK shared library (see ``livox_python.py``).
2. ``pip install -r requirements.txt`` where the file lists
   ``numpy``, ``open3d==0.16.0`` (or newer), and ``kiss-icp``.

Run ::

    python live_slam.py

You should see a live, growing point-cloud map in an Open3D window.
"""

# Preset-controlled real-time SLAM demo for the Livox MID-360.
# Choose INDOOR vs OUTDOOR at the top – parameters below are easy to tweak.

from __future__ import annotations

import signal
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from typing import Optional, Dict, Any
import os

# ---------------------------------------------------------------------------
# Mount orientation correction
# ---------------------------------------------------------------------------
# We support two **independent** adjustments that are applied to every point
# cloud as well as to the reported pose so that downstream consumers (for
# instance *run_geoff_gui.py*) always operate in the same, right-handed world
# frame:
#
# 1. Upside-down mounting (sensor rotated 180° about its **X-axis**) – this is
#    enabled by default because the G-1 normally carries the MID-360 on the
#    top of its head pointing *downwards*.  Override with
#        LIVOX_MOUNT=normal
#    if your sensor is mounted the right way up.
#
# 2. A *fixed* tilt offset when the robot tilts its entire head.  The LiDAR
#    and RealSense are mechanically linked, so improving the camera's viewing
#    angle often means pitching/rolling the LiDAR as well which then confuses
#    the SLAM/occupancy components.  Specify the correction via the two
#    environment variables:
#
#        LIDAR_TILT_DEG=15     # magnitude in **degrees**
#        LIDAR_TILT_AXIS=y     # axis to rotate about: x, y or z (default y)
#
#    Example – head pitched back about the *y*-axis by 15°:
#        LIDAR_TILT_DEG=15  LIDAR_TILT_AXIS=y
#
#    Example – sensor rolled 10° to the left (rare):
#        LIDAR_TILT_DEG=10  LIDAR_TILT_AXIS=x
#
#    Positive angles follow the right-hand rule for the chosen axis.  The code
#    applies the *inverse* rotation automatically so entering the physical
#    tilt you observe is all that is required.
#
#    The code used to default to a non-zero angle (15° → later 25°) matching
#    an early prototype where the MID-360 was pitched forwards.  The default
#    is now **0°** so no correction is applied on a clean install.  Specify
#    ``LIDAR_TILT_DEG`` if your physical mounting differs.
# ---------------------------------------------------------------------------

# Valid values for the mandatory mounting orientation flag
_VALID_MOUNTS = {"normal", "upside_down"}

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in _VALID_MOUNTS:
    raise SystemExit(f"LIVOX_MOUNT must be one of {_VALID_MOUNTS}")

import math  # standard

# ------------------------------------------------------------------
# Optional fixed tilt – *disabled by default* (0°).  Set the environment
# variables ``LIDAR_TILT_DEG`` and optionally ``LIDAR_TILT_AXIS`` to apply a
# correction when the sensor is not perfectly level.
# ------------------------------------------------------------------
# Read axis & angle ----------------------------------------------------------

# ------------------------------------------------------------------
# Optional fixed tilt – default **0°** so no correction is applied unless
# the user explicitly specifies their sensor inclination via the environment
# variable *LIDAR_TILT_DEG*.  Geoff’s early configuration required a 25° pitch
# (handled by a previous default) but the current standard platform mounts
# the LiDAR level, therefore zero degrees is a safer starting point.
# ------------------------------------------------------------------

_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()
if _TILT_AXIS not in {"x", "y", "z"}:
    raise SystemExit("LIDAR_TILT_AXIS must be one of 'x', 'y', 'z'")

# Read the desired tilt angle (degrees) from the environment – fall back to
# **0°** when not set or invalid so we do not apply any correction by
# default.

try:
    _TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
except ValueError:
    _TILT_DEG = 0.0

_R_MOUNT = None  # 4×4 homogeneous correction matrix (or *None* = identity)

# Build individual rotation matrices ------------------------------------------------

# 1) Upside-down correction – 180° about sensor X (flip Y and Z)
_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)

# 2) Fixed pitch about the **Y**-axis.  We follow the right-hand rule where a
#    *positive* angle corresponds to the head being tilted **back** so the
#    sensor points slightly upwards.  Applying the inverse rotation therefore
#    aligns the scan with the true horizontal plane.

# Build the inverse rotation that **undoes** the physical tilt --------------

if abs(_TILT_DEG) > 1e-3:  # negligible => identity
    _rad = math.radians(-_TILT_DEG)  # negative to *undo* the observed tilt
    c, s = math.cos(_rad), math.sin(_rad)

    if _TILT_AXIS == "x":
        _R_TILT = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    elif _TILT_AXIS == "y":
        _R_TILT = np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    else:  # 'z'
        _R_TILT = np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
else:
    _R_TILT = np.eye(4)

# Combined correction: first flip (if necessary) *then* tilt.  The order is
# important: we want to interpret the user-provided axis relative to the
# *upright* frame.

_R_TOTAL = _R_TILT @ _R_FLIP

# Use identity (=> None) when no correction required so later code can take a
# fast path.

if not np.allclose(_R_TOTAL, np.eye(4)):
    _R_MOUNT = _R_TOTAL

# ---------------------------------------------------------------------------
# KISS-ICP import logic – cope with package layout changes.
# ---------------------------------------------------------------------------

KissICP = None  # type: ignore

_IMPORT_ERRORS = []
try:  # v1.2+ exposes class under "kiss_icp.pipeline"
    from kiss_icp.pipeline import KissICP  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERRORS.append(e)

if KissICP is None:
    try:  # legacy (<1.0) path
        from kiss_icp.pybind import KissICP  # type: ignore
    except Exception as e:  # pragma: no cover
        _IMPORT_ERRORS.append(e)

if KissICP is None:
    _msgs = " | ".join(str(e) for e in _IMPORT_ERRORS)
    raise SystemExit(
        "Could not import KISS-ICP (tried kiss_icp.pipeline & kiss_icp.pybind).\n"
        "Package is missing or broken.  Install/upgrade with:\n"
        "    pip install --upgrade 'kiss-icp'\n\nDetails: "
        + _msgs
    )

# Try SDK2 first (push-mode).  Fallback to legacy SDK if not present.
try:
    from livox2_python import Livox2 as _Livox
except Exception as e:
    print("[INFO] livox2_python unavailable (", e, ") – falling back to SDK1.")
    from livox_python import Livox as _Livox

# ---------------------------------------------------------------------------
# User-selectable presets (INDOOR / OUTDOOR)
# ---------------------------------------------------------------------------

# Pick the desired preset here or export the environment variable `LIVOX_PRESET`.
PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        # Livox2 pseudo-frame aggregation
        "frame_time": 0.35,      # seconds
        "frame_packets": 200,

        # Map & viz
        "voxel_size": 0.4,       # m
        "max_range": 30.0,       # m
        "downsample_limit": 5_000_000,  # keep up to N pts in viewer

        # ICP tuning
        "min_motion": 0.03,      # m
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
    raise SystemExit(f"Unknown PRESET '{PRESET}'. Choose one of {_PRESETS.keys()}.")

# Short alias to the active dictionary so later code is concise
_P = _PRESETS[PRESET]


# ---------------------------------------------------------------------------
# Visualisation utilities
# ---------------------------------------------------------------------------


class _Viewer:
    """Open3D visualiser that shows both the map *and* the current pose."""

    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None

        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None

        self._first = True

    # ------------------------------------------------------------------
    # Thread-safe queues (very small – only last item matters)
    # ------------------------------------------------------------------

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        """Called from background thread with new map + pose."""

        self._latest_pts = xyz
        self._latest_pose = pose

    # ------------------------------------------------------------------
    # Called from the *main/UI* thread
    # ------------------------------------------------------------------

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
            self._vis.reset_view_point(True)  # auto-fit once we have data
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    # ------------------------------------------------------------------
    def _update_pose_vis(self, pose: np.ndarray):
        # Remove old geometry (if any)
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)

        # Derive a reasonable size from current map extent so the frame is
        # always visible regardless of room size.
        size = 0.5
        if len(self._pcd.points) > 0:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03  # 3 % of diagonal
            size = max(0.2, min(size, 2.0))  # clamp to [0.2 m, 2 m]

        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)
        self._vis.update_geometry(self._cam_frame)

    # ------------------------------------------------------------------
    def close(self):
        self._vis.destroy_window()


# ---------------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------------


class LiveSLAMDemo(_Livox):
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

    def _apply_safe_config(self, cfg):
        """安全地应用 KISS-ICP 配置参数"""
        # 映射配置 - 使用更保守的参数
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

        # 配准配置 - 只设置确实存在的属性
        try:
            if hasattr(cfg, 'registration'):
                if hasattr(cfg.registration, 'convergence_criterion'):
                    cfg.registration.convergence_criterion = max(_P["conv_criterion"], 1e-4)
                if hasattr(cfg.registration, 'max_num_iterations'):
                    cfg.registration.max_num_iterations = min(_P["max_iters"], 50)  # 显著减少迭代次数
        except Exception as e:
            print(f"[WARNING] 设置配准参数时出错: {e}")

    # ------------------------------------------------------------------
    # Overridden callback – receives each raw frame
    # ------------------------------------------------------------------

    def handle_points(self, xyz: np.ndarray):
        """处理单个激光雷达帧，改进错误处理和数值稳定性"""
        
        current_time = time.time()
        self._frame_count += 1
        
        # 控制帧率，避免过于频繁的更新
        if current_time - self._last_frame_time < 0.1:  # 最大 10 Hz
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
        
        # 安全检查云数据
        if cloud is None or cloud.size == 0:
            print("[WARNING] 空地图，跳过可视化")
            return
        
        # 应用挂载方向校正
        if _R_MOUNT is not None:
            try:
                cloud = (cloud @ _R_MOUNT[:3, :3].T).astype(cloud.dtype, copy=False)
            except Exception as e:
                print(f"[WARNING] 挂载校正失败: {e}")

        # 下采样用于可视化
        if cloud.shape[0] > self._vis_max_points:
            step = max(1, int(cloud.shape[0] / self._vis_max_points))
            cloud = cloud[::step]

        # 获取当前位姿
        try:
            pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            if _R_MOUNT is not None:
                pose = _R_MOUNT @ pose
        except Exception as e:
            print(f"[WARNING] 获取位姿失败: {e}")
            pose = np.eye(4)

        # 推送到可视化器
        try:
            self._viewer.push(cloud, pose)
        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")

    def shutdown(self):
        """安全关闭所有资源"""
        try:
            super().shutdown()
        except Exception as e:
            print(f"[WARNING] 关闭 Livox 时出错: {e}")
        
        try:
            self._viewer.close()
        except Exception as e:
            print(f"[WARNING] 关闭可视化器时出错: {e}")


def main():  # pragma: no cover
    demo = LiveSLAMDemo()

    # Allow Ctrl-C
    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()