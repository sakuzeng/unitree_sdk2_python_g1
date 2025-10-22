"""
Livox2 静态地图构建器

当雷达静止时，通过累积点云数据创建可供人查看的环境地图
支持2D/3D地图生成、保存和可视化
"""

import time
import numpy as np
import open3d as o3d
import cv2
import threading
import csv
import sys
from pathlib import Path
from collections import deque
from typing import Optional, Tuple
import json
from livox2_python import Livox2, create_default_config

# 雷达挂载方向配置
MOUNT = "upside_down"  # "normal" 或 "upside_down"

class StaticMapper(Livox2):
    """
    静态地图构建器，继承 Livox2 类
    累积点云数据生成环境地图
    """
    
    def __init__(self, *args, **kwargs):
        # 地图构建参数
        self.map_size = kwargs.pop('map_size', 20.0)  # 地图大小(米)
        self.map_resolution = kwargs.pop('map_resolution', 0.1)  # 地图分辨率(米/像素)
        self.accumulation_time = kwargs.pop('accumulation_time', 30.0)  # 累积时间(秒)
        self.height_filter_min = kwargs.pop('height_filter_min', -2.0)  # 最小高度(米)
        self.height_filter_max = kwargs.pop('height_filter_max', 3.0)   # 最大高度(米)
        self.mount = kwargs.pop('mount', MOUNT)  # 雷达挂载方向
        
        super().__init__(*args, **kwargs)
        
        # 地图数据结构
        self.map_lock = threading.Lock()
        self.accumulated_points = []  # 累积的点云数据
        self.occupancy_grid = None    # 2D占用栅格地图
        self.height_map = None        # 高度地图
        self.intensity_map = None     # 强度地图
        
        # IMU 数据缓存和保存
        self._imu_buffer = []
        self._imu_count = 0
        self._setup_data_saving()
        
        # 统计信息
        self.frame_count = 0
        self.total_accumulated_points = 0
        self.mapping_start_time = time.time()
        self.is_mapping = True
        
        # 初始化地图
        self._init_maps()
        
        print(f"静态地图构建器初始化完成:")
        print(f"  地图大小: {self.map_size}x{self.map_size} 米")
        print(f"  分辨率: {self.map_resolution} 米/像素")
        print(f"  累积时间: {self.accumulation_time} 秒")
        print(f"  高度范围: [{self.height_filter_min}, {self.height_filter_max}] 米")
        print(f"  雷达挂载: {self.mount}")
    
    def _setup_data_saving(self):
        """设置数据保存文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 创建保存目录
        self.data_dir = Path("mapping_data") / f"session_{timestamp}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # IMU 数据 CSV 文件
        self._imu_csv = self.data_dir / "imu_data.csv"
        with open(self._imu_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])
        
        print(f"数据保存目录: {self.data_dir}")
        print(f"IMU 数据文件: {self._imu_csv}")
    
    def _init_maps(self):
        """初始化地图数据结构"""
        # 计算地图像素尺寸
        self.map_pixels = int(self.map_size / self.map_resolution)
        
        # 初始化各种地图
        self.occupancy_grid = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.height_map = np.full((self.map_pixels, self.map_pixels), -np.inf, dtype=np.float32)
        self.intensity_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.point_count_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.int32)
        
        print(f"地图栅格大小: {self.map_pixels}x{self.map_pixels} 像素")
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                     tag: np.ndarray, timestamp: int):
        """
        SDK 的点云数据回调函数，在后台线程中运行。

        此方法负责对原始点云数据进行预处理（方向校正、下采样），
        并累积到地图中。

        Args:
            xyz (np.ndarray): 原始点云数据 (N, 3)。
            reflectivity (np.ndarray): 反射强度 (N,)，范围 0-255。
            tag (np.ndarray): 标签 (N,)。
            timestamp (int): 时间戳 (ns)。
        """
        if not self.is_mapping:
            return
        
        self.frame_count += 1
        
        # 检查累积时间
        elapsed = time.time() - self.mapping_start_time
        if elapsed > self.accumulation_time:
            print(f"\n累积时间达到 {self.accumulation_time} 秒，停止数据收集")
            self.is_mapping = False
            self._finalize_map()
            return
        
        if len(xyz) > 0:
            # 根据挂载方向进行坐标校正
            if self.mount == "upside_down":
                # 绕 X 轴旋转 180°: (x, y, z) -> (x, -y, -z)
                xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
            
            # 过滤有效点（去除噪声点）
            valid_mask = tag == 0  # 只保留有效点
            xyz_valid = xyz[valid_mask]
            reflectivity_valid = reflectivity[valid_mask]
            
            if len(xyz_valid) > 0:
                # 高度过滤
                height_mask = (xyz_valid[:, 2] >= self.height_filter_min) & \
                             (xyz_valid[:, 2] <= self.height_filter_max)
                xyz_filtered = xyz_valid[height_mask]
                reflectivity_filtered = reflectivity_valid[height_mask]
                
                if len(xyz_filtered) > 0:
                    # 对过于密集的帧进行下采样，以保证处理效率
                    # 每帧最多处理 50,000 点
                    if xyz_filtered.shape[0] > 50_000:
                        step = xyz_filtered.shape[0] // 50_000
                        xyz_filtered = xyz_filtered[::step]
                        reflectivity_filtered = reflectivity_filtered[::step]
                    
                    # 添加到累积点云
                    with self.map_lock:
                        self.accumulated_points.append({
                            'xyz': xyz_filtered.copy(),
                            'reflectivity': reflectivity_filtered.copy(),
                            'timestamp': timestamp
                        })
                        self.total_accumulated_points += len(xyz_filtered)
                    
                    # 更新地图
                    self._update_maps(xyz_filtered, reflectivity_filtered)
        
        # 显示进度
        if self.frame_count % 20 == 0:
            progress = min(elapsed / self.accumulation_time * 100, 100)
            print(f"地图构建进度: {progress:.1f}% | "
                 f"累积点数: {self.total_accumulated_points:,} | "
                 f"帧数: {self.frame_count}")
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据，缓冲并保存到 CSV，修复加速度单位为 m/s² 并处理坐标系翻转。

        Args:
            imu_data (np.ndarray): IMU 数据，形状 (N, 6)，包含 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]。
                                原始加速度单位为 g，角速度单位为 rad/s。
            timestamp (int): 数据包时间戳，单位：ns。
        """
        if not self.is_mapping:
            return
        
        if len(imu_data) > 0:
            # 创建副本以避免修改原始数据
            imu_processed = imu_data.copy()
            
            # 将加速度从 g 转换为 m/s²
            imu_processed[:, 3:6] *= 9.81  # acc_x, acc_y, acc_z 乘以 9.81
            
            # 如果雷达倒挂，翻转 gy, gz, ay, az
            if self.mount == "upside_down":
                imu_processed[:, [1, 2, 4, 5]] *= -1  # 翻转 gyro_y, gyro_z, acc_y, acc_z
            
            # 添加到缓冲区
            self._imu_buffer.append((imu_processed, timestamp))
            
            # 每 100 个样本写入一次
            if len(self._imu_buffer) >= 100:
                try:
                    with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for data, ts in self._imu_buffer:
                            for row in data:
                                # 时间戳转换为秒
                                writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
                    
                    self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
                    if self._imu_count % 1000 == 0:  # 每1000个样本打印一次
                        print(f"[StaticMapper] 已保存 {self._imu_count} 个 IMU 样本到 {self._imu_csv}")
                    
                    self._imu_buffer = []
                    
                except IOError as e:
                    print(f"[StaticMapper] IMU CSV 写入失败: {e}", file=sys.stderr)
    
    def _update_maps(self, xyz: np.ndarray, reflectivity: np.ndarray):
        """更新各种地图数据"""
        # 转换到地图坐标系
        map_coords = self._world_to_map(xyz)
        
        # 过滤在地图范围内的点
        valid_indices = (map_coords[:, 0] >= 0) & (map_coords[:, 0] < self.map_pixels) & \
                       (map_coords[:, 1] >= 0) & (map_coords[:, 1] < self.map_pixels)
        
        if not np.any(valid_indices):
            return
        
        map_coords_valid = map_coords[valid_indices].astype(int)
        xyz_valid = xyz[valid_indices]
        reflectivity_valid = reflectivity[valid_indices]
        
        with self.map_lock:
            for i in range(len(map_coords_valid)):
                x, y = map_coords_valid[i]
                z = xyz_valid[i, 2]
                intensity = reflectivity_valid[i]
                
                # 更新占用栅格（有点就标记为占用）
                self.occupancy_grid[x, y] = 1.0
                
                # 更新高度地图（取最高点）
                if z > self.height_map[x, y]:
                    self.height_map[x, y] = z
                
                # 更新强度地图（平均值）
                current_count = self.point_count_map[x, y]
                self.intensity_map[x, y] = (self.intensity_map[x, y] * current_count + intensity) / (current_count + 1)
                self.point_count_map[x, y] += 1
    
    def _world_to_map(self, xyz: np.ndarray) -> np.ndarray:
        """将世界坐标转换为地图坐标"""
        # 假设雷达在地图中心
        map_center = self.map_pixels // 2
        
        # 转换坐标 (X, Y) -> (map_x, map_y)
        map_x = (xyz[:, 0] / self.map_resolution + map_center).astype(int)
        map_y = (xyz[:, 1] / self.map_resolution + map_center).astype(int)
        
        return np.column_stack([map_x, map_y])
    
    def _finalize_map(self):
        """完成地图构建并生成最终地图"""
        print("\n正在完成地图构建...")
        
        # 保存剩余的 IMU 数据
        if self._imu_buffer:
            try:
                with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for data, ts in self._imu_buffer:
                        for row in data:
                            writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
                
                self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
                print(f"[StaticMapper] 最终保存 {self._imu_count} 个 IMU 样本")
                self._imu_buffer = []
                
            except IOError as e:
                print(f"[StaticMapper] 最终 IMU CSV 写入失败: {e}", file=sys.stderr)
        
        with self.map_lock:
            # 处理高度地图中的无效值
            self.height_map[self.height_map == -np.inf] = 0
            
            # 归一化强度地图
            if self.intensity_map.max() > 0:
                self.intensity_map = self.intensity_map / 255.0
            
            print(f"地图构建完成:")
            print(f"  总累积点数: {self.total_accumulated_points:,}")
            print(f"  占用栅格: {np.sum(self.occupancy_grid > 0)} 个非空格子")
            print(f"  高度范围: [{self.height_map.min():.2f}, {self.height_map.max():.2f}] 米")
            print(f"  IMU 样本数: {self._imu_count}")
    
    def generate_3d_map_with_intensity_colors(self) -> o3d.geometry.PointCloud:
        """生成3D点云地图，使用改进的颜色映射"""
        print("生成3D点云地图（改进颜色映射）...")
        
        # 合并所有累积的点云
        all_points = []
        all_colors = []
        
        with self.map_lock:
            for frame in self.accumulated_points:
                xyz = frame['xyz']
                reflectivity = frame['reflectivity']
                
                # 基于反射强度生成颜色（从深蓝色到白色）
                norm_reflectivity = reflectivity / 255.0  # 归一化到 [0, 1]
                colors = np.zeros((len(xyz), 3), dtype=np.float32)
                colors[:, 0] = norm_reflectivity  # R: 0 -> 1
                colors[:, 1] = norm_reflectivity  # G: 0 -> 1
                colors[:, 2] = 0.5 + 0.5 * norm_reflectivity  # B: 0.5 -> 1
                
                all_points.append(xyz)
                all_colors.append(colors)
        
        if not all_points:
            print("没有点云数据用于生成3D地图")
            return None
        
        # 合并数据
        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # 下采样以减少点数
        if len(combined_points) > 100_000:
            pcd = pcd.voxel_down_sample(voxel_size=0.05)
            print(f"下采样后点数: {len(pcd.points)}")
        
        print(f"3D地图生成完成，总点数: {len(pcd.points):,}")
        return pcd
    
    def save_maps(self, output_dir: str = "maps"):
        """保存生成的地图到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        print(f"\n正在保存地图到: {output_path}")
        
        with self.map_lock:
            # 保存占用栅格地图 (PNG)
            occupancy_img = (self.occupancy_grid * 255).astype(np.uint8)
            occupancy_file = output_path / f"occupancy_map_{timestamp}.png"
            cv2.imwrite(str(occupancy_file), occupancy_img)
            
            # 保存高度地图 (PNG - 彩色编码)
            height_normalized = cv2.normalize(self.height_map, None, 0, 255, cv2.NORM_MINMAX)
            height_colored = cv2.applyColorMap(height_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            height_file = output_path / f"height_map_{timestamp}.png"
            cv2.imwrite(str(height_file), height_colored)
            
            # 保存强度地图 (PNG)
            intensity_img = (self.intensity_map * 255).astype(np.uint8)
            intensity_file = output_path / f"intensity_map_{timestamp}.png"
            cv2.imwrite(str(intensity_file), intensity_img)
            
            # 保存原始数据 (NPZ)
            data_file = output_path / f"map_data_{timestamp}.npz"
            np.savez_compressed(data_file,
                               occupancy_grid=self.occupancy_grid,
                               height_map=self.height_map,
                               intensity_map=self.intensity_map,
                               point_count_map=self.point_count_map)
            
            # 保存元数据 (JSON)
            metadata = {
                "timestamp": timestamp,
                "map_size_meters": self.map_size,
                "map_resolution": self.map_resolution,
                "map_pixels": self.map_pixels,
                "accumulation_time": self.accumulation_time,
                "total_points": self.total_accumulated_points,
                "total_frames": self.frame_count,
                "imu_samples": self._imu_count,
                "mount_orientation": self.mount,
                "height_filter": [self.height_filter_min, self.height_filter_max],
                "occupied_cells": int(np.sum(self.occupancy_grid > 0)),
                "height_range": [float(self.height_map.min()), float(self.height_map.max())],
                "data_directory": str(self.data_dir),
                "files": {
                    "occupancy_map": str(occupancy_file.name),
                    "height_map": str(height_file.name),
                    "intensity_map": str(intensity_file.name),
                    "raw_data": str(data_file.name),
                    "imu_csv": str(self._imu_csv)
                }
            }
            
            metadata_file = output_path / f"map_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"地图文件已保存:")
            print(f"  占用栅格: {occupancy_file}")
            print(f"  高度地图: {height_file}")
            print(f"  强度地图: {intensity_file}")
            print(f"  原始数据: {data_file}")
            print(f"  IMU 数据: {self._imu_csv}")
            print(f"  元数据: {metadata_file}")
    
    def save_3d_map(self, output_dir: str = "maps"):
        """保存3D点云地图"""
        pcd = self.generate_3d_map_with_intensity_colors()
        if pcd is None:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存为PLY格式
        ply_file = output_path / f"3d_map_{timestamp}.ply"
        o3d.io.write_point_cloud(str(ply_file), pcd)
        
        # 保存为PCD格式（如果需要）
        pcd_file = output_path / f"3d_map_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        
        print(f"3D地图已保存:")
        print(f"  PLY格式: {ply_file}")
        print(f"  PCD格式: {pcd_file}")
    
    def visualize_3d_map(self):
        """可视化3D点云地图"""
        pcd = self.generate_3d_map_with_intensity_colors()
        if pcd is None:
            return
        
        print("启动3D地图可视化...")
        print("使用鼠标操作: 左键旋转, 右键平移, 滚轮缩放")
        
        # 添加坐标系
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.0, origin=[0, 0, 0]
        )
        
        # 显示
        o3d.visualization.draw_geometries(
            [pcd, coordinate_frame],
            window_name="Livox MID-360 3D Static Map",
            width=1200,
            height=800
        )
    
    # 保留原有的其他方法...
    def visualize_maps(self):
        """可视化生成的地图"""
        print("\n显示地图可视化...")
        
        with self.map_lock:
            # 创建复合地图显示
            fig_height = 800
            fig_width = 1200
            
            # 占用栅格地图
            occupancy_display = cv2.resize(
                (self.occupancy_grid * 255).astype(np.uint8),
                (400, 400)
            )
            occupancy_colored = cv2.applyColorMap(occupancy_display, cv2.COLORMAP_GRAY)
            
            # 高度地图
            height_normalized = cv2.normalize(self.height_map, None, 0, 255, cv2.NORM_MINMAX)
            height_display = cv2.resize(height_normalized.astype(np.uint8), (400, 400))
            height_colored = cv2.applyColorMap(height_display, cv2.COLORMAP_JET)
            
            # 强度地图
            intensity_display = cv2.resize(
                (self.intensity_map * 255).astype(np.uint8),
                (400, 400)
            )
            intensity_colored = cv2.applyColorMap(intensity_display, cv2.COLORMAP_HOT)
            
            # 组合显示
            top_row = np.hstack([occupancy_colored, height_colored])
            bottom_row = np.hstack([intensity_colored, np.zeros((400, 400, 3), dtype=np.uint8)])
            combined = np.vstack([top_row, bottom_row])
            
            # 添加标题和统计信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined, "Occupancy Map", (50, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Height Map", (450, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Intensity Map", (50, 430), font, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, f"Points: {self.total_accumulated_points:,}", (450, 450), font, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f"IMU Samples: {self._imu_count:,}", (450, 480), font, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f"Mount: {self.mount}", (450, 510), font, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f"Size: {self.map_size}x{self.map_size}m", (450, 540), font, 0.6, (255, 255, 255), 2)
            
            # 显示
            cv2.imshow("Livox MID-360 Static Map", combined)
            print("按任意键关闭地图显示...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    """主函数 - 演示静态地图构建"""
    print("Livox MID-360 静态地图构建器 (改进版)")
    print("="*50)
    
    # 检查配置文件
    config_path = Path("mid360_config.json")
    if not config_path.exists():
        print("配置文件不存在，创建默认配置...")
        create_default_config(config_path)
        print(f"已创建配置文件: {config_path}")
    
    print("\n地图构建说明:")
    print("1. 请确保雷达在构建期间保持静止")
    print("2. 程序将累积指定时间内的所有点云数据")
    print("3. 支持坐标系翻转（倒挂安装）")
    print("4. 自动保存 IMU 数据到 CSV 文件")
    print("5. 构建完成后自动生成2D和3D地图")
    
    # 雷达挂载方向选择
    print(f"\n当前雷达挂载方向: {MOUNT}")
    mount_choice = input("是否需要更改挂载方向? (upside_down/normal, 默认 upside_down): ").strip().lower()
    if mount_choice == "normal":
        mount = "normal"
        print("已设置为正常挂载模式")
    else:
        mount = "upside_down"
        print("已设置为倒挂模式，将自动进行坐标系校正")
    
    # 地图参数配置
    print("\n请配置地图参数:")
    try:
        map_size = float(input("地图大小 (米, 默认20): ") or "20")
        map_resolution = float(input("地图分辨率 (米/像素, 默认0.5): ") or "0.5")
        accumulation_time = float(input("数据累积时间 (秒, 默认30): ") or "30")
        
        print(f"\n地图配置:")
        print(f"  地图大小: {map_size}x{map_size} 米")
        print(f"  分辨率: {map_resolution} 米/像素")
        print(f"  累积时间: {accumulation_time} 秒")
        print(f"  挂载方向: {mount}")
        
    except ValueError:
        print("使用默认参数")
        map_size = 100.0
        map_resolution = 0.1
        accumulation_time = 30.0
    
    try:
        print(f"\n正在初始化静态地图构建器...")
        mapper = StaticMapper(
            config_path,
            host_ip="192.168.123.164",  # 请根据实际网络配置修改
            frame_time=0.05,        # 50ms 帧聚合时间（更快）
            frame_packets=100,      # 每帧100包（更多数据）
            enable_filter=True,     # 启用过滤
            max_range=map_size/2,   # 最大距离为地图半径
            voxel_size=0.05,        # 较小的体素
            map_size=map_size,
            map_resolution=map_resolution,
            accumulation_time=accumulation_time,
            height_filter_min=-2.0,
            height_filter_max=3.0,
            mount=mount
        )
        
        with mapper:
            print("✓ 地图构建器初始化成功")
            print("等待雷达连接...")
            time.sleep(2.0)
            
            print(f"\n开始地图构建，累积时间: {accumulation_time} 秒")
            print("请确保雷达保持静止！")
            print("按 Ctrl+C 可提前停止构建")
            
            try:
                # 等待地图构建完成
                while mapper.is_mapping and mapper.is_running():
                    time.sleep(1.0)
                
                if not mapper.is_running():
                    print("雷达连接中断")
                    return
                    
            except KeyboardInterrupt:
                print("\n用户中断地图构建")
                mapper.is_mapping = False
                mapper._finalize_map()
            
            # 保存地图
            print("\n保存地图文件...")
            mapper.save_maps("maps")
            mapper.save_3d_map("maps")
            
            # 显示地图
            display_choice = input("\n是否显示地图? (Y/n): ").strip().lower()
            if display_choice != 'n':
                mapper.visualize_maps()
                
                show_3d = input("是否显示3D地图? (Y/n): ").strip().lower()
                if show_3d != 'n':
                    mapper.visualize_3d_map()
            
            print("\n地图构建完成！")
            print("地图文件已保存到 'maps' 目录")
            print(f"IMU 数据已保存到 {mapper._imu_csv}")
            
    except Exception as e:
        print(f"✗ 地图构建失败: {e}")
        print("\n可能的原因:")
        print("1. 雷达网络连接问题")
        print("2. Livox-SDK2 库未找到")
        print("3. OpenCV 或 Open3D 未正确安装")
        print("4. 磁盘空间不足")
        print("5. 网络接口配置错误（请检查 host_ip 设置）")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()