#!/usr/bin/env python3
"""
Livox MID-360 雷达点云实时查看器

本脚本使用 Open3D 库实时可视化来自 Livox MID-360 激光雷达的点云数据，
并提供基本的交互功能，同时支持 IMU 数据记录。点云颜色基于反射强度 (reflectivity) 显示。

运行前，请确保已正确安装 Livox-SDK2 和相关 Python 依赖包 (numpy, open3d)，
并验证 `livox2_python.py` 中导入的 `.so` 文件名称是否正确。

效果:
    实时显示带颜色的雷达点云数据（反射强度从深蓝色到白色），按 'ESC' 键或关闭窗口退出。
    IMU 数据保存为 CSV 文件到 data/ 目录。

基础流程:
    1. SDK 在后台线程接收 UDP 数据并解析成点云和 IMU 数据。
    2. 调用 `handle_points()` 和 `handle_imu()` 回调函数进行数据预处理。
    3. `push()` 方法将处理后的点云帧（含颜色）存入环形缓冲区。
    4. 主线程通过 `tick()` 方法循环渲染，将合并后的点云显示到屏幕。

环境变量配置:
- LIVOX_MOUNT: 挂载方向 (normal/upside_down, 默认 upside_down)
"""
from __future__ import annotations

import os
import signal
import time
from typing import Optional
from pathlib import Path
import csv
import numpy as np
import open3d as o3d
import json

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

# 数据保存目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 导入 Livox-SDK2 封装
# ---------------------------------------------------------------------------
try:
    from livox2_python import Livox2 as _Livox
except ImportError as e:
    raise SystemExit(f"[错误] 无法导入 livox2_python 模块: {e}\n请确保 livox2_python.py 存在并已正确安装 Livox-SDK2。")

class _Viewer:
    """
    一个由主线程驱动的最小化 Open3D 可视化器。

    此类不直接与 SDK 交互，而是提供一个简单的接口，用于在主循环中
    接收点云数据（含颜色）并更新渲染。
    """
    def __init__(self):
        """
        初始化可视化器窗口、点云对象和环形缓冲区。
        """
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox – 实时点云", width=1280, height=720)
        self._is_alive = True

        # 使用环形缓冲区合并最近的 N 帧点云，以减少稀疏扫描带来的闪烁感，
        # 形成一个更稳定、更密集的点云视图。
        self._frames: list[tuple[np.ndarray, np.ndarray]] = []  # 存储 (xyz, colors) 对
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

    def push(self, xyz: np.ndarray, colors: np.ndarray):
        """
        从任意(SDK)后台线程接收新的点云帧及其颜色并添加到环形缓冲区。

        Args:
            xyz (np.ndarray): 新的点云帧数据 (N, 3)。
            colors (np.ndarray): 点云颜色数据 (N, 3)，RGB 范围 [0, 1]。
        """
        self._frames.append((xyz, colors))
        # 当缓冲区满时，丢弃最旧的帧。
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def tick(self) -> bool:
        """
        在主线程中执行的单次渲染更新。

        合并缓冲区中的所有点云帧及其颜色，更新几何体，并处理窗口事件。

        Returns:
            bool: 如果可视化窗口仍然存活，返回 True；否则返回 False。
        """
        if not self._is_alive:
            return False

        if self._frames:
            # 合并所有帧的点云和颜色
            xyz_list, color_list = zip(*self._frames)
            merged_xyz = np.concatenate(xyz_list, axis=0)
            merged_colors = np.concatenate(color_list, axis=0)
            self._pcd.points = o3d.utility.Vector3dVector(merged_xyz)
            self._pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            self._vis.update_geometry(self._pcd)
            if self._first:
                # 首次更新时，自动调整视角以适应点云。
                self._vis.reset_view_point(True)
                self._first = False

        # 处理窗口事件（如关闭、键盘输入）并更新渲染器。
        alive = self._vis.poll_events()
        self._vis.update_renderer()

        # 检查窗口是否关闭
        if not alive:
            self._is_alive = False

        return alive

    def close(self):
        """销毁可视化窗口。"""
        self._vis.destroy_window()

class LiveViewer(_Livox):
    """
    Livox SDK 的封装类，将 SDK 回调与 `_Viewer` 可视化器连接起来。
    """
    def __init__(self, config_path: str = "mid360_config.json", host_ip: str = "192.168.123.164"):
        """
        初始化 Livox 连接和 Open3D 可视化器。

        Args:
            config_path (str): JSON 配置文件路径。
            host_ip (str): 主机 IP 地址。
        """
        # 检查配置文件是否存在
        cfg = Path(config_path)
        if not cfg.exists():
            host_ip = os.environ.get("HOST_IP", host_ip)
            data = {
                "MID360": {
                    "lidar_net_info": {
                        "lidar_ip": "192.168.123.120",
                        "cmd_data_port": 56100,
                        "push_msg_port": 56200,
                        "point_data_port": 56300,
                        "imu_data_port": 56400,
                        "log_data_port": 56500,
                    },
                    "host_net_info": [
                        {
                            "host_ip": host_ip,
                            "multicast_ip": "224.1.1.5",
                            "cmd_data_port": 56101,
                            "push_msg_port": 56201,
                            "point_data_port": 56301,
                            "imu_data_port": 56401,
                            "log_data_port": 56501,
                        }
                    ]
                }
            }
            cfg.write_text(json.dumps(data, indent=2))
            print(f"[LiveViewer] 默认配置文件已生成: {config_path}")

        # 初始化 IMU 数据保存
        self._imu_csv = DATA_DIR / "imu_data.csv"
        self._imu_count = 0
        self._imu_buffer: list[tuple[np.ndarray, int]] = []
        with open(self._imu_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])

        # 初始化 Livox2
        super().__init__(
            config_path,
            host_ip=host_ip,
            frame_time=0.1,  # 降低以适配 G-1
            frame_packets=60  # 减少以降低 CPU 负载
        )

        self._view = _Viewer()

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        SDK 的点云数据回调函数，在后台线程中运行。

        此方法负责对原始点云数据进行预处理（方向校正、下采样），
        并基于反射强度生成颜色，然后推送到可视化器。

        Args:
            xyz (np.ndarray): 原始点云数据 (N, 3)。
            reflectivity (np.ndarray): 反射强度 (N,)，范围 0-255。
            tag (np.ndarray): 标签 (N,)。
            timestamp (int): 时间戳 (ns)。
        """
        # 根据挂载方向进行坐标校正。
        if MOUNT == "upside_down":
            # 绕 X 轴旋转 180°: (x, y, z) -> (x, -y, -z)
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # 对过于密集的帧进行下采样，以保证渲染流畅。
        # 10万点/秒对于预览已足够。
        if xyz.shape[0] > 100_000:
            step = xyz.shape[0] // 100_000
            xyz = xyz[::step]
            reflectivity = reflectivity[::step]

        # 基于反射强度生成颜色（从深蓝色到白色）
        norm_reflectivity = reflectivity / 255.0  # 归一化到 [0, 1]
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        colors[:, 0] = norm_reflectivity  # R: 0 -> 1
        colors[:, 1] = norm_reflectivity  # G: 0 -> 1
        colors[:, 2] = 0.5 + 0.5 * norm_reflectivity  # B: 0.5 -> 1

        self._view.push(xyz, colors)

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        SDK 的 IMU 数据回调函数，在后台线程中运行。

        Args:
            imu_data (np.ndarray): IMU 数据 (N, 6)，包含 [gx, gy, gz, ax, ay, az]。
            timestamp (int): 时间戳 (ns)。
        """
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        if len(self._imu_buffer) >= 100:  # 每 100 个样本写入
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            print(f"[LiveViewer] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []

    def shutdown(self):
        """关闭 SDK 连接并销毁可视化窗口，保存剩余 IMU 数据。"""
        # 保存缓冲中的 IMU 数据
        if self._imu_buffer:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            print(f"[LiveViewer] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []

        super().shutdown()
        self._view.close()

def main():
    """
    程序主入口。

    初始化 `LiveViewer`，设置信号处理以捕获 Ctrl-C，
    并进入主循环直到用户退出。
    """
    print(f"[LiveViewer] 启动 Livox 点云查看器 (挂载: {MOUNT})")
    print(f"[LiveViewer] IMU 数据将保存到: {DATA_DIR.absolute() / 'imu_data.csv'}")
    lidar = LiveViewer()
    stop = False

    def _sigint_handler(*_):
        """信号处理函数，用于设置停止标志。"""
        nonlocal stop
        print("\n[LiveViewer] 收到 Ctrl-C，正在关闭...")
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