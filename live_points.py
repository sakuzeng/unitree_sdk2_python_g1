#!/usr/bin/env python3
"""
Livox MID-360 雷达点云实时查看器

本脚本使用 Open3D 库实时可视化来自 Livox MID-360 激光雷达的点云数据，
并提供基本的交互功能。

运行前，请确保已正确安装 Livox SDK2 和相关 Python 依赖包 (requirements.txt)，
并验证 `livox2_python.py` 中导入的 `.so` 文件名称是否正确。

效果:
    实时显示雷达点云数据，按 'ESC' 键或关闭窗口退出。

基础流程:
    1. SDK 在后台线程接收 UDP 数据并解析成点云。
    2. 调用 `handle_points()` 回调函数进行数据预处理。
    3. `push()` 方法将处理后的点云帧存入环形缓冲区。
    4. 主线程通过 `tick()` 方法循环渲染，将合并后的点云显示到屏幕。
"""
from __future__ import annotations

import os
import signal
import time
from typing import Optional

# ---------------------------------------------------------------------------
# 挂载方向说明 (Mount Orientation)
#
# G1 机器人上的 MID-360 雷达通常是倒置安装的。因此，默认值为 'upside_down'。
# 如果您的传感器是正向安装的，请设置环境变量 `LIVOX_MOUNT=normal`。
#
# 坐标系转换:
# 当设置为 'upside_down' 时，脚本会对点云数据进行 180 度翻转 (绕 X 轴)，
# 将物理上的 (x, -y, -z) 坐标映射为逻辑上的 (x, y, z) 坐标，
# 从而在可视化界面中获得与机器人前进方向一致的直观视图。
# ---------------------------------------------------------------------------
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("环境变量 LIVOX_MOUNT 的值必须是 'normal' 或 'upside_down'")

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# 动态导入 SDK 封装 (优先使用 SDK2)
# ---------------------------------------------------------------------------
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except ImportError as _e:  # pragma: no cover – SDK2 not present / not built
    print(f"[INFO] livox2_python 不可用 ({_e}) – 切换至 SDK1。")
    from livox_python import Livox as _Livox
    _SDK2 = False


class _Viewer:
    """
    一个由主线程驱动的最小化 Open3D 可视化器。

    此类不直接与 SDK 交互，而是提供一个简单的接口，用于在主循环中
    接收点云数据并更新渲染。
    """

    def __init__(self):
        """
        初始化可视化器窗口、点云对象和环形缓冲区。
        """
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox – live point-cloud", width=1280, height=720)

        # 使用环形缓冲区合并最近的 N 帧点云，以减少稀疏扫描带来的闪烁感，
        # 形成一个更稳定、更密集的点云视图。
        self._frames: list[np.ndarray] = []
        self._max_frames = 15  # 在 20Hz 下约等于 0.75 秒的累积数据

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        # 在原点创建一个静态坐标系，用于表示传感器位置和方向。
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        if MOUNT == "upside_down":
            # 如果是倒置安装，将坐标系也翻转以匹配点云。
            R180 = np.diag([1.0, -1.0, -1.0, 1.0])
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)

        self._first = True

    def push(self, xyz: np.ndarray):
        """
        从任意(SDK)后台线程接收新的点云帧并将其添加到环形缓冲区。

        Args:
            xyz (np.ndarray): 新的点云帧数据 (N, 3)。
        """
        self._frames.append(xyz)
        # 当缓冲区满时，丢弃最旧的帧。
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def tick(self) -> bool:
        """
        在主线程中执行的单次渲染更新。

        合并缓冲区中的所有点云帧，更新几何体，并处理窗口事件。

        Returns:
            bool: 如果可视化窗口仍然存活，返回 True；否则返回 False。
        """
        if self._frames:
            # 合并所有帧并更新点云几何体
            merged = np.concatenate(self._frames, axis=0)
            self._pcd.points = o3d.utility.Vector3dVector(merged)
            self._vis.update_geometry(self._pcd)
            if self._first:
                # 首次更新时，自动调整视角以适应点云。
                self._vis.reset_view_point(True)
                self._first = False

        # 处理窗口事件（如关闭、键盘输入）并更新渲染器。
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self):
        """销毁可视化窗口。"""
        self._vis.destroy_window()


class LiveViewer(_Livox):
    """
    Livox SDK 的封装类，将 SDK 回调与 `_Viewer` 可视化器连接起来。
    """

    def __init__(self):
        """
        根据 SDK 版本初始化 Livox 连接。
        """
        # SDK2 需要 JSON 配置文件路径，而 SDK1 不需要。
        if _SDK2:
            # 注意: 此处 IP 地址可能需要根据实际情况修改。
            # 默认主机 IP 为 192.168.123.164，此处示例为 222。
            super().__init__("mid360_config.json", host_ip="192.168.123.222")
        else:
            super().__init__()  # SDK1 无需参数

        self._view = _Viewer()

    def handle_points(self, xyz: np.ndarray):
        """
        SDK 的点云数据回调函数，在后台线程中运行。

        此方法负责对原始点云数据进行预处理（方向校正、下采样），
        然后将其推送到可视化器。

        Args:
            xyz (np.ndarray): 从 SDK 收到的原始点云数据。
        """
        # 根据挂载方向进行坐标校正。
        if MOUNT == "upside_down":
            # 绕 X 轴旋转 180°: (x, y, z) -> (x, -y, -z)
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # 对过于密集的帧进行下采样，以保证渲染流畅。
        # 15万点/秒对于预览已足够。
        if xyz.shape[0] > 100_000:
            step = xyz.shape[0] // 100_000
            xyz = xyz[:: step]

        self._view.push(xyz)

    def shutdown(self):
        """关闭 SDK 连接并销毁可视化窗口。"""
        super().shutdown()
        self._view.close()


def main():
    """
    程序主入口。

    初始化 `LiveViewer`，设置信号处理以捕获 Ctrl-C，
    并进入主循环直到用户退出。
    """
    lidar = LiveViewer()
    stop = False

    def _sigint_handler(*_):
        """信号处理函数，用于设置停止标志。"""
        nonlocal stop
        print("\n正在关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # 主循环：当 stop 为 False 且窗口存活时，持续调用 tick()。
        while not stop and lidar._view.tick():
            time.sleep(0.01)  # 降低 CPU 使用率
    finally:
        lidar.shutdown()


if __name__ == "__main__":
    main()
