from __future__ import annotations
"""
Livox MID-360 雷达数据保存脚本
功能：捕获并保存 Livox MID-360 的点云和 IMU 数据，用于后续分析。
- 点云数据保存为 .npy 文件（包含 x, y, z, reflectivity, tag）
- IMU 数据保存为 .csv 文件（包含时间戳、角速度、加速度）
- 使用 Livox-SDK2，支持组播数据接收
- 按 Ctrl-C 优雅退出，数据保存到 lidar_data 目录
运行要求：
- 安装 Livox-SDK2 及 Python 依赖：numpy
- 配置 mid360_config.json（与雷达 IP 和主机 IP 匹配）
- 设置环境变量 LIVOX_MOUNT（可选，默认为 upside_down）
- 确保 livox2_python.py 模块可用（包含 Livox2 类）
使用方法：
    python save_lidar_data.py
"""

import signal
import time
from pathlib import Path
import numpy as np
import csv
import os
import json
import sys

# 挂载方向
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("环境变量 LIVOX_MOUNT 的值必须是 'normal' 或 'upside_down'")

# 导入 Livox-SDK2 模块
try:
    from livox2_python import Livox2 as _Livox
except ImportError as e:
    raise SystemExit(f"[错误] 无法导入 livox2_python 模块: {e}\n请确保 livox2_python.py 存在并已正确安装 Livox-SDK2。")

# 数据保存器
class LidarDataSaver(_Livox):
    def __init__(self, config_path: str = "mid360_config.json", host_ip: str = "192.168.123.164"):
        """
        初始化 LidarDataSaver，继承 Livox2 类。

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
            print(f"[LidarDataSaver] 默认配置文件已生成: {config_path}")

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
        
        # 初始化 Livox2
        super().__init__(config_path, host_ip=host_ip, frame_time=0.1, frame_packets=60)
        
        # 保存帧计数
        self._frame_count = 0

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据，同时保存为 .npy 和 .txt 文件（x, y, z, reflectivity, tag）。

        Args:
            xyz (np.ndarray): 点云坐标，形状 (N, 3)，单位：米。
            reflectivity (np.ndarray): 反射强度，形状 (N,)，范围 0-255。
            tag (np.ndarray): 标签，形状 (N,)，用于噪声过滤。
            timestamp (int): 数据包时间戳，单位：ns。
        """
        # 应用挂载方向校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0])

        # 合并数据为 (N, 5) 数组
        data = np.column_stack((xyz, reflectivity, tag))

        # 时间戳转换为秒
        timestamp_sec = timestamp / 1e9  # ns to s

        # 保存为 .npy 文件
        npy_file = self._save_dir / f"point_cloud_{timestamp_sec:.3f}.npy"
        try:
            np.save(npy_file, data)
            print(f"[LidarDataSaver] 已保存点云帧 {self._frame_count + 1} 到 {npy_file.name} ({len(xyz)} 点)")
        except Exception as e:
            print(f"[错误] 保存 .npy 点云帧 {self._frame_count + 1} 失败: {e}")

        # 保存为 .txt 文件
        txt_file = self._save_dir / f"point_cloud_{timestamp_sec:.3f}.txt"
        try:
            np.savetxt(
                txt_file,
                data,
                fmt="%.6f %.6f %.6f %.0f %.0f",
                delimiter=" ",
                header="x y z reflectivity tag",
                comments="# "
            )
            print(f"[LidarDataSaver] 已保存点云帧 {self._frame_count + 1} 到 {txt_file.name} ({len(xyz)} 点)")
        except Exception as e:
            print(f"[错误] 保存 .txt 点云帧 {self._frame_count + 1} 失败: {e}")

        self._frame_count += 1
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据，缓冲并保存到 CSV，修复加速度单位为 m/s² 并处理坐标系翻转。

        Args:
            imu_data (np.ndarray): IMU 数据，形状 (N, 6)，包含 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]。
                                原始加速度单位为 g，角速度单位为 rad/s。
            timestamp (int): 数据包时间戳，单位：ns。
        """
        # 将加速度从 g 转换为 m/s²
        imu_data[:, 3:6] *= 9.81  # acc_x, acc_y, acc_z 乘以 9.81

        # 如果雷达倒挂，翻转 gy, gz, ay, az
        if getattr(self, 'mount', None) == "upside_down":
            imu_data[:, [1, 2, 4, 5]] *= -1  # 翻转 gyro_y, gyro_z, acc_y, acc_z

        self._imu_buffer.append((imu_data, timestamp))
        if len(self._imu_buffer) >= 100:  # 每 100 个样本写入
            try:
                with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for data, ts in self._imu_buffer:
                        for row in data:
                            writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
                self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
                print(f"[LidarDataSaver] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
                self._imu_buffer = []
            except IOError as e:
                print(f"[LidarDataSaver] CSV 写入失败: {e}", file=sys.stderr)
    # def handle_imu(self, imu_data: np.ndarray, timestamp: int):
    #     """
    #     处理 IMU 数据，缓冲并保存到 CSV。

    #     Args:
    #         imu_data (np.ndarray): IMU 数据，形状 (N, 6)，包含 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]。
    #         timestamp (int): 数据包时间戳，单位：ns。
    #     """
    #     self._imu_buffer.append((imu_data, timestamp))
    #     if len(self._imu_buffer) >= 100:  # 每 100 个样本写入
    #         with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
    #             writer = csv.writer(f)
    #             for data, ts in self._imu_buffer:
    #                 for row in data:
    #                     writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
    #         self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
    #         print(f"[LidarDataSaver] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
    #         self._imu_buffer = []

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
            print(f"[LidarDataSaver] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
            self._imu_buffer = []
        
        super().shutdown()
        print(f"[LidarDataSaver] 总计保存 {self._frame_count} 帧点云数据到 {self._save_dir}")

def main():
    # 初始化保存器
    saver = LidarDataSaver()
    stop = False

    def _sigint_handler(*_):
        nonlocal stop
        print("\n[LidarDataSaver] 收到 Ctrl-C，正在关闭...")
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