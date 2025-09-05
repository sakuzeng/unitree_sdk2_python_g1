#!/usr/bin/env python3
"""
Livox MID-360 实时激光雷达 SLAM 演示

本脚本使用 KISS-ICP 算法和 Open3D 库，对来自 Livox MID-360 激光雷达的
点云数据进行实时同步定位与建图 (SLAM)。

功能:
- 自动发现并连接网络中的 Livox 雷达。
- 使用 KISS-ICP 算法处理点云帧，生成轨迹和地图。
- 通过 Open3D 实时可视化不断增长的点云地图和机器人当前位姿。
- 支持通过环境变量对雷达的物理安装（倒置、倾斜）进行坐标校正。

运行方法:
    # 确保已安装所有依赖
    pip install -r requirements.txt
    
    # 运行脚本
    python3 live_slam.py

环境变量配置:
- LIVOX_PRESET: "indoor" 或 "outdoor"，用于选择不同的 SLAM 参数预设。
- LIVOX_MOUNT: "upside_down" (默认) 或 "normal"，用于校正雷达的倒置安装。
- LIDAR_TILT_DEG: 雷达的物理倾斜角度（度数），默认为 0。
- LIDAR_TILT_AXIS: 倾斜旋转轴，"x", "y" (默认), 或 "z"。
"""

from __future__ import annotations

import math
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# 挂载方向与倾斜校正
# ---------------------------------------------------------------------------
# G1 机器人上的 MID-360 雷达通常是倒置安装的，因此默认 LIVOX_MOUNT 为 'upside_down'。
# 如果雷达因物理安装存在固定倾斜，可通过 LIDAR_TILT_DEG 和 LIDAR_TILT_AXIS
# 环境变量进行校正，以确保 SLAM 在水平坐标系中运行。
# ---------------------------------------------------------------------------

# 1. 处理倒置安装
_VALID_MOUNTS = {"normal", "upside_down"}
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in _VALID_MOUNTS:
    raise SystemExit(f"LIVOX_MOUNT 必须是 {_VALID_MOUNTS} 中的一个")

# 2. 处理固定倾斜
_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()
if _TILT_AXIS not in {"x", "y", "z"}:
    raise SystemExit("LIDAR_TILT_AXIS 必须是 'x', 'y', 'z' 中的一个")

try:
    _TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
except ValueError:
    _TILT_DEG = 0.0

# 3. 计算组合校正矩阵
# 倒置校正矩阵 (绕 X 轴旋转 180°)
_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)

# 倾斜校正矩阵 (应用物理倾斜的逆旋转)
if abs(_TILT_DEG) > 1e-3:
    _rad = math.radians(-_TILT_DEG)  # 逆向旋转以抵消物理倾斜
    c, s = math.cos(_rad), math.sin(_rad)
    if _TILT_AXIS == "x":
        _R_TILT = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    elif _TILT_AXIS == "y":
        _R_TILT = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    else:  # 'z'
        _R_TILT = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
else:
    _R_TILT = np.eye(4)

# 最终的组合校正矩阵：先翻转，再倾斜
_R_TOTAL = _R_TILT @ _R_FLIP
_R_MOUNT = _R_TOTAL if not np.allclose(_R_TOTAL, np.eye(4)) else None

# ---------------------------------------------------------------------------
# 动态导入依赖库
# ---------------------------------------------------------------------------
# 优先导入 KISS-ICP v1.2+
try:
    from kiss_icp.pipeline import KissICP
    from kiss_icp.config import load_config
except ImportError:
    try:
        # 尝试旧版 API
        from kiss_icp.pybind import KissICP
        from kiss_icp.config import KISSConfig as load_config
    except ImportError as e:
        raise SystemExit(
            "无法导入 KISS-ICP。请安装或升级: `pip install --upgrade 'kiss-icp'`\n"
            f"错误详情: {e}"
        )

# 优先使用 Livox SDK2 (push 模式)
try:
    from livox2_python import Livox2 as _Livox
except ImportError as e:
    print(f"[INFO] livox2_python 不可用 ({e}) – 切换至旧版 SDK1。")
    from livox_python import Livox as _Livox

# ---------------------------------------------------------------------------
# SLAM 参数预设
# ---------------------------------------------------------------------------
PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()
_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        "frame_time": 0.35, "frame_packets": 200, "voxel_size": 0.4,
        "max_range": 30.0, "downsample_limit": 5_000_000, "min_motion": 0.03,
        "conv_criterion": 5e-5, "max_iters": 800,
    },
    "outdoor": {
        "frame_time": 0.20, "frame_packets": 120, "voxel_size": 1.0,
        "max_range": 120.0, "downsample_limit": 3_000_000, "min_motion": 0.10,
        "conv_criterion": 1e-4, "max_iters": 500,
    },
}
if PRESET not in _PRESETS:
    raise SystemExit(f"未知的预设 '{PRESET}'。请从 {_PRESETS.keys()} 中选择。")
_P = _PRESETS[PRESET]


class _Viewer:
    """
    Open3D 可视化器，用于同时显示 SLAM 地图和当前机器人位姿。
    
    此类在主线程中运行，通过线程安全的方式从后台 SLAM 线程接收数据。
    """
    def __init__(self):
        """初始化可视化窗口、点云几何体和位姿坐标系。"""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._first = True

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        """
        从后台线程接收新的地图点云和位姿。

        Args:
            xyz (np.ndarray): 更新后的地图点云。
            pose (np.ndarray): 最新的机器人位姿 (4x4 齐次矩阵)。
        """
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        """
        在主线程中执行的单次渲染更新。
        
        检查是否有新数据，更新几何体，并处理窗口事件。

        Returns:
            bool: 如果可视化窗口仍然存活，返回 True。
        """
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
        """更新或创建表示机器人当前位姿的坐标系。"""
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)
        
        size = max(0.2, min(self._pcd.get_max_bound().max() * 0.05, 2.0))
        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)

    def close(self):
        """销毁可视化窗口。"""
        self._vis.destroy_window()


class LiveSLAMDemo(_Livox):
    """
    主演示类，继承自 Livox 驱动，将数据流接入 SLAM 和可视化器。
    """
    def __init__(self):
        """初始化 Livox 驱动、KISS-ICP 和可视化器。"""
        # 1. 初始化 Livox SDK
        sdk_kwargs = {}
        if "Livox2" in str(_Livox):
            sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])
        
        try:
            # 配置文件路径根据项目规范调整
            config_file = os.path.expanduser("~/livox_cfg/MID360_config.json")
            if not os.path.exists(config_file):
                config_file = "mid360_config.json" # 回退到本地
            super().__init__(config_file, host_ip="192.168.123.164", **sdk_kwargs)
        except TypeError:
            super().__init__() # 旧版 SDK

        # 2. 初始化 KISS-ICP
        cfg = load_config(config_file=None, max_range=_P["max_range"])
        self._apply_safe_config(cfg)
        self._slam = KissICP(cfg)
        
        # 3. 初始化可视化器
        self._viewer = _Viewer()
        self._vis_max_points = _P["downsample_limit"]

    def _apply_safe_config(self, cfg):
        """安全地应用预设参数到 KISS-ICP 配置对象。"""
        # 此函数用于将 _P 字典中的参数应用到 cfg 对象，同时处理属性不存在的情况
        param_map = {
            'mapping.voxel_size': _P["voxel_size"],
            'adaptive_threshold.min_motion_th': _P["min_motion"],
            'registration.convergence_criterion': _P["conv_criterion"],
            'registration.max_num_iterations': _P["max_iters"],
        }
        for key, value in param_map.items():
            try:
                parts = key.split('.')
                obj = cfg
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            except AttributeError:
                print(f"[Warning] 无法设置参数 '{key}'，可能 KISS-ICP 版本不兼容。")

    def handle_points(self, xyz: np.ndarray):
        """
        SDK 的点云数据回调函数，在后台线程中运行。
        
        此方法是 SLAM 流程的核心，负责数据预处理、SLAM 注册和结果推送。
        """
        # 1. 预处理：移除机器人自身的反射点
        r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.25))
        dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.20))
        if xyz.size > 0:
            dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
            mask = ~((dist_xy < r_xy) & (np.abs(xyz[:, 2]) < dz))
            xyz = xyz[mask]

        if xyz.shape[0] < 100:
            return # 点太少，跳过

        # 2. SLAM 处理：注册帧并获取结果
        timestamps = np.zeros(len(xyz)) # KISS-ICP 需要时间戳数组
        self._slam.register_frame(xyz, timestamps)
        
        # 3. 获取地图和位姿
        cloud = self._slam.get_map()
        pose = self._slam.last_pose

        if cloud.size == 0:
            return

        # 4. 应用挂载校正
        if _R_MOUNT is not None:
            cloud = (cloud @ _R_MOUNT[:3, :3].T).astype(cloud.dtype, copy=False)
            pose = _R_MOUNT @ pose

        # 5. 为可视化降采样
        if cloud.shape[0] > self._vis_max_points:
            step = cloud.shape[0] // self._vis_max_points
            cloud = cloud[::step]

        # 6. 推送到可视化器
        self._viewer.push(cloud, pose)

    def shutdown(self):
        """安全地关闭 Livox SDK 和 Open3D 窗口。"""
        super().shutdown()
        self._viewer.close()


def main():
    """程序主入口，初始化并运行 SLAM 演示。"""
    demo = LiveSLAMDemo()
    stop = False

    def _sigint_handler(*_):
        nonlocal stop
        print("\n正在关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()