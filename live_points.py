"""雷达点云查看器(Livox MID-360)
该脚本旨在简化对Livox MID-360雷达点云的查看和处理。
它使用Open3D库实时可视化点云数据，并提供基本的交互功能。
运行该脚本前，请确保已正确安装Livox SDK2和相关Python依赖包(requirements.txt)。
验证livox2_python.py中的.so文件名称
效果：实时显示雷达点云数据，esc退出
基础流程：
    1. SDK接收UDP数据 → 解析点云
    2. 调用 handle_points() → 数据预处理
    3. push() 到缓冲区 → 存储帧数据
    4. tick() 渲染循环 → 显示到屏幕
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Mount orientation: 'normal' or 'upside_down'.  Most G1 units have the MID-360
# mounted upside-down.  Therefore the *default* is now 'upside_down'.  If your
# sensor is right-side-up set the env-var `LIVOX_MOUNT=normal`.
# ---------------------------------------------------------------------------
"""调整g1机器人雷达方向，默认为倒置(upside_down)
激光雷达朝上安装：
     ↑ Z (向上)
     |
     |____→ Y (向前)
    /
   / X (向右)

激光雷达朝下安装：
     ↓ Z (向下，物理上)
     |
     |____<- Y (向后，物理上)  
    /
   / X (向右)

经过坐标转换后：
     ↑ Z (向上，逻辑上)
     |
     |____→ Y (向前，逻辑上)
    /
   / X (向右)
"""

import os

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("LIVOX_MOUNT must be 'normal' or 'upside_down'")

import numpy as np  # after env check – avoids unused import earlier
import open3d as o3d

# ---------------------------------------------------------------------------
# Dynamic import of the right SDK wrapper (SDK2 preferred, SDK1 fallback)
# ---------------------------------------------------------------------------

try:
    from livox2_python import Livox2 as _Livox

    _SDK2 = True
except Exception as _e:  # pragma: no cover – SDK2 not present / not built
    print("[INFO] livox2_python unavailable (", _e, ") – falling back to SDK1.")
    from livox_python import Livox as _Livox

    _SDK2 = False


# ---------------------------------------------------------------------------
# Minimal single-thread visualiser (same as live_slam.py)
# ---------------------------------------------------------------------------


class _Viewer:
    """Open3D visualiser whose *tick* method we drive from the main thread."""

    def __init__(self):
        # * Open3D 可视化器初始化,设置分辨率和窗口标题
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox – live point-cloud", width=1280, height=720)

        # We keep a *ring-buffer* of the most recent N frames and visualise
        # the *union*.  This removes the distracting “blinking” that occurs
        # when each new sparse MID-360 scan entirely replaces the previous
        # one.
        self._frames: list[np.ndarray] = []
        self._max_frames = 15  # ≈0.75 s @ 20 Hz – tune to taste

        self._pcd = o3d.geometry.PointCloud() # * 点云几何体创建
        self._vis.add_geometry(self._pcd) # * 将点云对象添加到可视化器中

        # Static coordinate frame at the LiDAR origin so you always know where
        # “the robot” (sensor) is and which way its local XYZ axes point.
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        if MOUNT == "upside_down":
            R180 = np.diag([1.0, -1.0, -1.0, 1.0]) # * 倒置处理坐标系
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)

        self._first = True

    # Called from any SDK (background) thread
    # * SDK线程 → push(new_frame) → append到缓冲区 → 检查容量 → 必要时删除最旧帧
    def push(self, xyz: np.ndarray):
        self._frames.append(xyz)
        # drop oldest frame when buffer full
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    # Called from the *main* thread
    def tick(self) -> bool:
        if self._frames:
            merged = np.concatenate(self._frames, axis=0) # * 批量合并
            self._pcd.points = o3d.utility.Vector3dVector(merged) # * 将 NumPy 数组转为 Open3D 点云格式
            self._vis.update_geometry(self._pcd)
            if self._first:
                self._vis.reset_view_point(True)  # fit camera once
                self._first = False
            self._latest = None

        alive = self._vis.poll_events() # * 处理用户交互
        self._vis.update_renderer() # * 更新屏幕显示
        return alive

    def close(self):
        self._vis.destroy_window()


# ---------------------------------------------------------------------------
# LiDAR wrapper subclass that forwards every frame to the viewer
# ---------------------------------------------------------------------------


class LiveViewer(_Livox):
    """Thin proxy between SDK callback and the :class:`_Viewer`."""

    def __init__(self):
        # SDK2 requires a JSON config path; SDK1 does not.  Try the new API
        # first and gracefully fall back if the signature does not match.
        if _SDK2:
            super().__init__("mid360_config.json", host_ip="192.168.123.222")  # type: ignore[arg-type]
        else:
            super().__init__()  # SDK1 has no required arguments

        self._view = _Viewer()

    # ------------------------------------------------------------------
    # Callback from SDK base-class – runs in *background* thread(s)
    # ------------------------------------------------------------------
    # - SDK后台线程 → handle_points() → 数据处理 → push() → 可视化器缓冲区
    def handle_points(self, xyz: np.ndarray):  # noqa: D401 (imperative mood)
        # Apply mount orientation correction if needed
        if MOUNT == "upside_down":
            # 180° rotation around the X axis: (x, y, z) -> (x, -y, -z)
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # Down-sample extremely dense frames for smoother rendering – 150 k pts/s
        # is plenty for a preview.
        if xyz.shape[0] > 100_000: # * 超过 10 万点时触发下采样
            step = xyz.shape[0] // 100_000
            xyz = xyz[:: step]

        self._view.push(xyz)

    # ------------------------------------------------------------------

    def shutdown(self):
        super().shutdown()
        self._view.close()


# ---------------------------------------------------------------------------
# Main entry-point – standard Ctrl-C handling
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover – manual demo script
    lidar = LiveViewer()

    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and lidar._view.tick():  # type: ignore[attr-defined]
            time.sleep(0.01)
    finally:
        lidar.shutdown()


if __name__ == "__main__":
    main()
