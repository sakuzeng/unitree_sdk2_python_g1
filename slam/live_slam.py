#!/usr/bin/env python3
"""
Livox MID-360 雷达点云实时查看器（集成 KISS-ICP）

本脚本使用 Open3D 库实时可视化来自 Livox MID-360 激光雷达的点云数据，
并通过 KISS-ICP 进行里程计估计，生成传感器位姿轨迹和全局点云地图。
点云颜色基于反射强度显示，IMU 数据保存为 CSV 文件。

运行前，请确保已安装 Livox-SDK2、numpy、open3d、kiss-icp，
并验证 `livox2_python.py` 中导入的 `.so` 文件名称正确。

效果:
    - 实时显示配准后的点云（世界坐标系，颜色从深蓝色到白色）。
    - 保存位姿轨迹到 data/trajectory.csv。
    - 保存全局点云地图到 data/map.pcd。
    - IMU 数据保存到 data/imu_data.csv。
    - 按 'ESC' 键或关闭窗口退出。

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
from kiss_icp import Odometry

# ---------------------------------------------------------------------------
# 挂载方向说明
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
    """
    def __init__(self):
        """
        初始化可视化器窗口、点云对象和环形缓冲区。
        """
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox – 实时点云 (KISS-ICP)", width=1280, height=720)
        self._is_alive = True

        # 环形缓冲区合并最近 N 帧点云
        self._frames: list[tuple[np.ndarray, np.ndarray]] = []  # 存储 (xyz, colors) 对
        self._max_frames = 15  # 约 0.75 秒累积数据

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        # 静态坐标系表示传感器位置
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        if MOUNT == "upside_down":
            R180 = np.diag([1.0, -1.0, -1.0, 1.0])
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)

        self._first = True

    def push(self, xyz: np.ndarray, colors: np.ndarray):
        """
        接收新点云帧及其颜色，添加到环形缓冲区。

        Args:
            xyz (np.ndarray): 点云数据 (N, 3)，世界坐标系。
            colors (np.ndarray): 点云颜色 (N, 3)，RGB 范围 [0, 1]。
        """
        self._frames.append((xyz, colors))
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def tick(self) -> bool:
        """
        单次渲染更新，合并缓冲区点云并处理窗口事件。

        Returns:
            bool: 窗口存活返回 True，否则 False。
        """
        if not self._is_alive:
            return False

        if self._frames:
            xyz_list, color_list = zip(*self._frames)
            merged_xyz = np.concatenate(xyz_list, axis=0)
            merged_colors = np.concatenate(color_list, axis=0)
            self._pcd.points = o3d.utility.Vector3dVector(merged_xyz)
            self._pcd.colors = o3d.utility.Vector3dVector(merged_colors)
            self._vis.update_geometry(self._pcd)
            if self._first:
                self._vis.reset_view_point(True)
                self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()

        if not alive:
            self._is_alive = False

        return alive

    def close(self):
        """销毁可视化窗口。"""
        self._vis.destroy_window()

class LiveViewer(_Livox):
    """
    Livox SDK 封装，集成 KISS-ICP 里程计和 Open3D 可视化。
    """
    def __init__(self, config_path: str = "mid360_config.json", host_ip: str = "192.168.123.164"):
        """
        初始化 Livox 连接、KISS-ICP 和 Open3D 可视化器。

        Args:
            config_path (str): JSON 配置文件路径。
            host_ip (str): 主机 IP 地址。
        """
        # 检查配置文件
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
            frame_time=0.1,  # 适配 G-1
            frame_packets=60  # 降低 CPU 负载
        )

        # 初始化 KISS-ICP 和轨迹存储
        self.odometry = Odometry()
        self.current_pose = np.eye(4)  # 初始位姿
        self.poses = []  # 存储轨迹 (时间戳, 位姿)
        self.global_pcd = o3d.geometry.PointCloud()  # 全局点云地图
        self._view = _Viewer()

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据，调用 KISS-ICP 配准，更新全局点云。

        Args:
            xyz (np.ndarray): 原始点云数据 (N, 3)。
            reflectivity (np.ndarray): 反射强度 (N,)，范围 0-255。
            tag (np.ndarray): 标签 (N,)。
            timestamp (int): 时间戳 (ns)。
        """
        # 坐标校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)

        # 下采样
        if xyz.shape[0] > 100_000:
            step = xyz.shape[0] // 100_000
            xyz = xyz[::step]
            reflectivity = reflectivity[::step]

        # KISS-ICP 配准
        self.current_pose = self.odometry.register(xyz, self.current_pose)
        self.poses.append((timestamp / 1e9, self.current_pose.copy()))

        # 变换点云到世界坐标系
        transformed_xyz = (self.current_pose[:3, :3] @ xyz.T + self.current_pose[:3, 3]).T

        # 生成颜色
        norm_reflectivity = reflectivity / 255.0
        colors = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        colors[:, 0] = norm_reflectivity
        colors[:, 1] = norm_reflectivity
        colors[:, 2] = 0.5 + 0.5 * norm_reflectivity

        # 更新全局点云
        frame_pcd = o3d.geometry.PointCloud()
        frame_pcd.points = o3d.utility.Vector3dVector(transformed_xyz)
        frame_pcd.colors = o3d.utility.Vector3dVector(colors)
        self.global_pcd += frame_pcd

        # 推送到可视化器
        self._view.push(transformed_xyz, colors)

        print(f"[LiveViewer] 当前位姿：\n{self.current_pose}")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据，保存到 CSV。

        Args:
            imu_data (np.ndarray): IMU 数据 (N, 6)，[gx, gy, gz, ax, ay, az]。
            timestamp (int): 时间戳 (ns)。
        """
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        if len(self._imu_buffer) >= 100:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            print(f"[LiveViewer] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []

    def shutdown(self):
        """
        关闭 SDK，保存轨迹和点云地图，销毁可视化窗口。
        """
        # 保存剩余 IMU 数据
        if self._imu_buffer:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            print(f"[LiveViewer] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []

        # 保存轨迹（时间戳 + 平移）
        if self.poses:
            trajectory = np.array([[t] + pose[:3, 3].tolist() for t, pose in self.poses])
            np.savetxt(DATA_DIR / "trajectory.csv", trajectory, delimiter=",", header="timestamp,x,y,z", comments="")
            print(f"[LiveViewer] 已保存轨迹到 {DATA_DIR / 'trajectory.csv'}")

        # 保存全局点云
        if np.asarray(self.global_pcd.points).size > 0:
            o3d.io.write_point_cloud(str(DATA_DIR / "map.pcd"), self.global_pcd)
            print(f"[LiveViewer] 已保存全局点云到 {DATA_DIR / 'map.pcd'}")

        super().shutdown()
        self._view.close()

def main():
    """
    主入口，初始化 LiveViewer，捕获 Ctrl-C，运行主循环。
    """
    print(f"[LiveViewer] 启动 Livox 点云查看器 (挂载: {MOUNT})")
    print(f"[LiveViewer] IMU 数据将保存到: {DATA_DIR.absolute() / 'imu_data.csv'}")
    print(f"[LiveViewer] 轨迹将保存到: {DATA_DIR.absolute() / 'trajectory.csv'}")
    print(f"[LiveViewer] 点云地图将保存到: {DATA_DIR.absolute() / 'map.pcd'}")
    lidar = LiveViewer()
    stop = False

    def _sigint_handler(*_):
        nonlocal stop
        print("\n[LiveViewer] 收到 Ctrl-C，正在关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while not stop and lidar._view.tick():
            time.sleep(0.01)
    finally:
        lidar.shutdown()

if __name__ == "__main__":
    main()