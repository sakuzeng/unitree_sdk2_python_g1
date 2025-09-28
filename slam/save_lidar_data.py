from __future__ import annotations
"""
Livox MID-360 雷达数据保存脚本
功能：捕获并保存 Livox MID-360 的点云和 IMU 数据，用于后续分析。
- 点云数据保存为 .npy 文件（包含 x, y, z, reflectivity, tag）
- IMU 数据保存为 .csv 文件（包含时间戳、角速度、加速度）
- 支持 SDK2（优先）或 SDK1（回退）
- 按 Ctrl-C 优雅退出，数据保存到 lidar_data 目录
运行要求：
- 安装 Livox SDK2（或 SDK1）及 Python 依赖：numpy
- 配置 mid360_config.json（与雷达 IP 和主机 IP 匹配）
- 设置环境变量 LIVOX_MOUNT（可选，默认为 upside_down）
使用方法：
    python save_lidar_data.py
"""

import signal
import time
from pathlib import Path
import numpy as np
import csv
import os

# 挂载方向
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("环境变量 LIVOX_MOUNT 的值必须是 'normal' 或 'upside_down'")

# 动态导入 SDK
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except ImportError as _e:
    print(f"[INFO] livox2_python 不可用 ({_e}) – 切换至 SDK1。")
    from livox_python import Livox as _Livox
    _SDK2 = False

# 数据保存器
class LidarDataSaver(_Livox):
    def __init__(self, config_path: str = "mid360_config.json", host_ip: str = "192.168.123.164"):
        # 创建保存目录
        self._save_dir = Path("lidar_data")
        self._save_dir.mkdir(exist_ok=True)
        
        # 初始化 IMU CSV 文件
        self._imu_csv = self._save_dir / "imu_data.csv"
        self._imu_count = 0
        self._imu_buffer = []  # 缓冲 IMU 数据以减少 I/O
        with open(self._imu_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        # 初始化 Livox SDK
        if _SDK2:
            super().__init__(config_path, host_ip=host_ip, frame_time=0.1, frame_packets=60)
        else:
            super().__init__()
        
        # 保存帧计数
        self._frame_count = 0

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据，保存为 .npy 文件（x, y, z, reflectivity, tag）。
        """
        # 应用挂载方向校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0])
        
        # 合并数据为 (N, 5) 数组
        data = np.column_stack((xyz, reflectivity, tag))
        
        # 保存点云
        timestamp_sec = timestamp / 1e9  # ns to s
        np.save(self._save_dir / f"point_cloud_{timestamp_sec:.3f}.npy", data)
        self._frame_count += 1
        print(f"已保存点云帧 {self._frame_count} 到 point_cloud_{timestamp_sec:.3f}.npy ({len(xyz)} 点)")

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据，缓冲并保存到 CSV。
        """
        self._imu_buffer.append((imu_data, timestamp))
        if len(self._imu_buffer) >= 100:  # 每 100 个样本写入
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
            print(f"已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []

    def shutdown(self):
        """
        关闭 SDK 和保存器，保存剩余 IMU 数据。
        """
        # 保存缓冲中的 IMU 数据
        if self._imu_buffer:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
            print(f"已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []
        
        super().shutdown()
        print(f"总计保存 {self._frame_count} 帧点云数据到 {self._save_dir}")

def main():
    # 初始化保存器
    saver = LidarDataSaver()
    stop = False

    def _sigint_handler(*_):
        nonlocal stop
        print("\n正在关闭...")
        stop = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # 主循环：保持运行直到 Ctrl-C
        while not stop:
            time.sleep(0.01)
    finally:
        saver.shutdown()

if __name__ == "__main__":
    main()