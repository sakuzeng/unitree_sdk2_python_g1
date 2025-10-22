"""
Professional Livox2 点云地图构建器

专业级静态地图构建，提供高质量的点云处理和地图生成
集成多种滤波算法、密度优化和精细化后处理
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
from typing import Optional, Tuple, List
import json
from sklearn.cluster import DBSCAN
from scipy import ndimage
from livox2_python import Livox2, create_default_config

# 雷达挂载方向配置
MOUNT = "upside_down"  # "normal" 或 "upside_down"

class ProfessionalMapper(Livox2):
    """
    专业级地图构建器，提供高质量点云处理和地图生成
    """
    
    def __init__(self, *args, **kwargs):
        # 地图构建参数
        self.map_size = kwargs.pop('map_size', 30.0)
        self.map_resolution = kwargs.pop('map_resolution', 0.05)  # 更高分辨率
        self.accumulation_time = kwargs.pop('accumulation_time', 60.0)  # 更长累积时间
        self.height_filter_min = kwargs.pop('height_filter_min', -3.0)
        self.height_filter_max = kwargs.pop('height_filter_max', 4.0)
        self.mount = kwargs.pop('mount', MOUNT)
        
        # 专业级处理参数
        self.enable_advanced_filtering = kwargs.pop('enable_advanced_filtering', True)
        self.enable_outlier_removal = kwargs.pop('enable_outlier_removal', True)
        self.enable_surface_reconstruction = kwargs.pop('enable_surface_reconstruction', True)
        self.min_cluster_size = kwargs.pop('min_cluster_size', 10)
        self.voxel_leaf_size = kwargs.pop('voxel_leaf_size', 0.02)
        self.statistical_k = kwargs.pop('statistical_k', 20)
        self.statistical_std_ratio = kwargs.pop('statistical_std_ratio', 2.0)
        
        super().__init__(*args, **kwargs)
        
        # 高级地图数据结构
        self.map_lock = threading.Lock()
        self.raw_points = []  # 原始点云数据
        self.processed_points = []  # 处理后的点云数据
        
        # 多层地图
        self.occupancy_grid = None      # 占用栅格
        self.height_map = None          # 高度地图
        self.intensity_map = None       # 强度地图
        self.density_map = None         # 点密度地图
        self.normal_map = None          # 法向量地图
        self.confidence_map = None      # 置信度地图
        
        # IMU 数据处理
        self._imu_buffer = []
        self._imu_count = 0
        self._setup_data_saving()
        
        # 统计信息
        self.frame_count = 0
        self.total_accumulated_points = 0
        self.mapping_start_time = time.time()
        self.is_mapping = True
        
        # 初始化地图
        self._init_professional_maps()
        
        print(f"专业级地图构建器初始化完成:")
        print(f"  地图大小: {self.map_size}x{self.map_size} 米")
        print(f"  分辨率: {self.map_resolution} 米/像素 (高精度)")
        print(f"  累积时间: {self.accumulation_time} 秒")
        print(f"  高级滤波: {'启用' if self.enable_advanced_filtering else '禁用'}")
        print(f"  离群点移除: {'启用' if self.enable_outlier_removal else '禁用'}")
        print(f"  表面重建: {'启用' if self.enable_surface_reconstruction else '禁用'}")
    
    def _setup_data_saving(self):
        """设置专业级数据保存结构"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 创建详细的保存目录结构
        self.data_dir = Path("professional_mapping") / f"session_{timestamp}"
        self.raw_data_dir = self.data_dir / "raw_data"
        self.processed_data_dir = self.data_dir / "processed_data" 
        self.maps_dir = self.data_dir / "maps"
        self.analysis_dir = self.data_dir / "analysis"
        
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.maps_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # IMU 数据文件
        self._imu_csv = self.raw_data_dir / "imu_data.csv"
        with open(self._imu_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z'])
        
        print(f"专业数据保存目录: {self.data_dir}")
    
    def _init_professional_maps(self):
        """初始化专业级地图数据结构"""
        self.map_pixels = int(self.map_size / self.map_resolution)
        
        # 初始化多层地图
        self.occupancy_grid = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.height_map = np.full((self.map_pixels, self.map_pixels), -np.inf, dtype=np.float32)
        self.intensity_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.density_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.normal_map = np.zeros((self.map_pixels, self.map_pixels, 3), dtype=np.float32)
        self.confidence_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.float32)
        self.point_count_map = np.zeros((self.map_pixels, self.map_pixels), dtype=np.int32)
        
        print(f"高精度地图栅格: {self.map_pixels}x{self.map_pixels} 像素")
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                     tag: np.ndarray, timestamp: int):
        """专业级点云数据处理"""
        if not self.is_mapping:
            return
        
        self.frame_count += 1
        
        # 检查累积时间
        elapsed = time.time() - self.mapping_start_time
        if elapsed > self.accumulation_time:
            print(f"\n累积时间达到 {self.accumulation_time} 秒，开始专业级地图生成")
            self.is_mapping = False
            self._generate_professional_map()
            return
        
        if len(xyz) > 0:
            # 坐标系校正
            if self.mount == "upside_down":
                xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
            
            # 基础过滤
            valid_mask = tag == 0
            xyz_valid = xyz[valid_mask]
            reflectivity_valid = reflectivity[valid_mask]
            
            if len(xyz_valid) > 0:
                # 高度过滤
                height_mask = (xyz_valid[:, 2] >= self.height_filter_min) & \
                             (xyz_valid[:, 2] <= self.height_filter_max)
                xyz_filtered = xyz_valid[height_mask]
                reflectivity_filtered = reflectivity_valid[height_mask]
                
                if len(xyz_filtered) > 0:
                    # 高级预处理
                    xyz_processed, reflectivity_processed = self._advanced_preprocessing(
                        xyz_filtered, reflectivity_filtered
                    )
                    
                    if len(xyz_processed) > 0:
                        # 存储原始数据
                        with self.map_lock:
                            self.raw_points.append({
                                'xyz': xyz_filtered.copy(),
                                'reflectivity': reflectivity_filtered.copy(),
                                'timestamp': timestamp
                            })
                            
                            # 存储处理后数据
                            self.processed_points.append({
                                'xyz': xyz_processed.copy(),
                                'reflectivity': reflectivity_processed.copy(),
                                'timestamp': timestamp
                            })
                            
                            self.total_accumulated_points += len(xyz_processed)
                        
                        # 更新专业地图
                        self._update_professional_maps(xyz_processed, reflectivity_processed)
        
        # 显示详细进度
        if self.frame_count % 10 == 0:
            progress = min(elapsed / self.accumulation_time * 100, 100)
            print(f"专业地图构建: {progress:.1f}% | "
                 f"累积点数: {self.total_accumulated_points:,} | "
                 f"帧数: {self.frame_count} | "
                 f"处理质量: {'高精度' if self.enable_advanced_filtering else '标准'}")
    
    def _advanced_preprocessing(self, xyz: np.ndarray, reflectivity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """高级点云预处理"""
        if not self.enable_advanced_filtering or len(xyz) < 10:
            return xyz, reflectivity
        
        try:
            # 转换为 Open3D 点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            
            # 1. 体素化下采样（保持密度均匀）
            if len(xyz) > 1000:  # 只对密集点云进行下采样
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_leaf_size)
            
            # 2. 统计离群点移除
            if self.enable_outlier_removal and len(pcd.points) > self.statistical_k:
                pcd, inlier_indices = pcd.remove_statistical_outlier(
                    nb_neighbors=self.statistical_k,
                    std_ratio=self.statistical_std_ratio
                )
                
                # 更新反射强度
                if len(inlier_indices) > 0:
                    original_indices = np.arange(len(xyz))
                    # 由于下采样可能改变索引，需要重新处理反射强度
                    if len(pcd.points) <= len(reflectivity):
                        reflectivity = reflectivity[:len(pcd.points)]
            
            # 3. 半径离群点移除（处理噪声）
            if len(pcd.points) > 50:
                pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=0.3)
            
            # 获取处理后的点云
            xyz_processed = np.asarray(pcd.points)
            
            # 确保反射强度数组长度匹配
            if len(xyz_processed) != len(reflectivity):
                # 简单截取或填充
                if len(xyz_processed) < len(reflectivity):
                    reflectivity_processed = reflectivity[:len(xyz_processed)]
                else:
                    # 如果处理后点数增加（不太可能），使用平均值填充
                    reflectivity_processed = np.resize(reflectivity, len(xyz_processed))
            else:
                reflectivity_processed = reflectivity
            
            return xyz_processed, reflectivity_processed
            
        except Exception as e:
            print(f"高级预处理失败: {e}，使用原始数据")
            return xyz, reflectivity
    
    def _update_professional_maps(self, xyz: np.ndarray, reflectivity: np.ndarray):
        """更新专业级多层地图"""
        map_coords = self._world_to_map(xyz)
        
        valid_indices = (map_coords[:, 0] >= 0) & (map_coords[:, 0] < self.map_pixels) & \
                       (map_coords[:, 1] >= 0) & (map_coords[:, 1] < self.map_pixels)
        
        if not np.any(valid_indices):
            return
        
        map_coords_valid = map_coords[valid_indices].astype(int)
        xyz_valid = xyz[valid_indices]
        reflectivity_valid = reflectivity[valid_indices]
        
        with self.map_lock:
            # 计算局部法向量（如果有足够点数）
            normals = None
            if len(xyz_valid) > 10:
                try:
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(xyz_valid)
                    pcd_temp.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10)
                    )
                    normals = np.asarray(pcd_temp.normals)
                except:
                    normals = None
            
            for i in range(len(map_coords_valid)):
                x, y = map_coords_valid[i]
                z = xyz_valid[i, 2]
                intensity = reflectivity_valid[i]
                
                # 更新占用栅格
                self.occupancy_grid[x, y] = 1.0
                
                # 更新高度地图（取最高点）
                if z > self.height_map[x, y]:
                    self.height_map[x, y] = z
                
                # 更新强度地图（加权平均）
                current_count = self.point_count_map[x, y]
                if current_count > 0:
                    self.intensity_map[x, y] = (self.intensity_map[x, y] * current_count + intensity) / (current_count + 1)
                else:
                    self.intensity_map[x, y] = intensity
                
                # 更新密度地图
                self.density_map[x, y] += 1
                
                # 更新法向量地图
                if normals is not None and i < len(normals):
                    normal = normals[i]
                    # 加权平均法向量
                    if current_count > 0:
                        self.normal_map[x, y] = (self.normal_map[x, y] * current_count + normal) / (current_count + 1)
                    else:
                        self.normal_map[x, y] = normal
                
                # 更新置信度地图（基于点数和强度）
                confidence = min(1.0, (current_count + 1) / 10.0) * (intensity / 255.0)
                self.confidence_map[x, y] = confidence
                
                self.point_count_map[x, y] += 1
    
    def _world_to_map(self, xyz: np.ndarray) -> np.ndarray:
        """世界坐标转地图坐标"""
        map_center = self.map_pixels // 2
        map_x = (xyz[:, 0] / self.map_resolution + map_center).astype(int)
        map_y = (xyz[:, 1] / self.map_resolution + map_center).astype(int)
        return np.column_stack([map_x, map_y])
    
    def _generate_professional_map(self):
        """生成专业级地图"""
        print("\n开始专业级地图生成和后处理...")
        
        # 保存剩余 IMU 数据
        self._save_remaining_imu_data()
        
        with self.map_lock:
            # 1. 地图后处理
            self._post_process_maps()
            
            # 2. 生成高质量3D地图
            self._generate_high_quality_3d_map()
            
            # 3. 进行地图分析
            self._analyze_map_quality()
            
            print(f"专业地图生成完成:")
            print(f"  总累积点数: {self.total_accumulated_points:,}")
            print(f"  有效占用栅格: {np.sum(self.occupancy_grid > 0):,}")
            print(f"  平均点密度: {self.density_map.mean():.2f} 点/格子")
            print(f"  高度范围: [{self.height_map[self.height_map != -np.inf].min():.2f}, {self.height_map[self.height_map != -np.inf].max():.2f}] 米")
    
    def _post_process_maps(self):
        """地图后处理和优化"""
        print("执行地图后处理...")
        
        # 处理高度地图
        self.height_map[self.height_map == -np.inf] = 0
        
        # 归一化强度地图
        if self.intensity_map.max() > 0:
            self.intensity_map = self.intensity_map / 255.0
        
        # 密度地图归一化
        if self.density_map.max() > 0:
            self.density_map = np.log1p(self.density_map)  # 对数变换处理密度
            self.density_map = self.density_map / self.density_map.max()
        
        # 应用形态学操作清理地图
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 占用栅格形态学闭运算（填充小空洞）
        occupancy_cleaned = cv2.morphologyEx(
            (self.occupancy_grid * 255).astype(np.uint8), 
            cv2.MORPH_CLOSE, kernel
        )
        self.occupancy_grid = occupancy_cleaned.astype(np.float32) / 255.0
        
        # 高斯滤波平滑高度地图
        self.height_map = ndimage.gaussian_filter(self.height_map, sigma=0.5)
        
        # 双边滤波处理强度地图（保边去噪）
        intensity_8bit = (self.intensity_map * 255).astype(np.uint8)
        intensity_filtered = cv2.bilateralFilter(intensity_8bit, 9, 75, 75)
        self.intensity_map = intensity_filtered.astype(np.float32) / 255.0
    
    def _generate_high_quality_3d_map(self) -> o3d.geometry.PointCloud:
        """生成高质量3D点云地图"""
        print("生成高质量3D点云地图...")
        
        if not self.processed_points:
            print("没有处理后的点云数据")
            return None
        
        # 合并所有处理后的点云
        all_points = []
        all_colors = []
        all_normals = []
        
        for frame in self.processed_points:
            xyz = frame['xyz']
            reflectivity = frame['reflectivity']
            
            # 高质量颜色映射
            colors = self._generate_advanced_colors(xyz, reflectivity)
            
            all_points.append(xyz)
            all_colors.append(colors)
        
        combined_points = np.concatenate(all_points, axis=0)
        combined_colors = np.concatenate(all_colors, axis=0)
        
        # 创建高质量点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # 计算法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30)
        )
        
        # 定向法向量
        pcd.orient_normals_consistent_tangent_plane(k=10)
        
        # 最终下采样（如果需要）
        if len(combined_points) > 200_000:
            pcd = pcd.voxel_down_sample(voxel_size=0.03)
            print(f"最终下采样后点数: {len(pcd.points):,}")
        
        # 保存高质量3D地图
        self._save_high_quality_3d_map(pcd)
        
        print(f"高质量3D地图生成完成，总点数: {len(pcd.points):,}")
        return pcd
    
    def _generate_advanced_colors(self, xyz: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
        """生成高级颜色映射"""
        colors = np.zeros((len(xyz), 3), dtype=np.float32)
        
        # 基于高度的颜色映射
        z_vals = xyz[:, 2]
        z_min, z_max = z_vals.min(), z_vals.max()
        
        if z_max > z_min:
            height_normalized = (z_vals - z_min) / (z_max - z_min)
        else:
            height_normalized = np.zeros_like(z_vals)
        
        # 反射强度归一化
        intensity_normalized = reflectivity / 255.0
        
        # 混合颜色方案：高度 + 强度
        # 使用 HSV 颜色空间获得更好的视觉效果
        hue = height_normalized * 240 / 360  # 蓝色到红色
        saturation = 0.7 + 0.3 * intensity_normalized  # 基于强度调节饱和度
        value = 0.5 + 0.5 * intensity_normalized  # 基于强度调节亮度
        
        # HSV 到 RGB 转换
        import colorsys
        for i in range(len(xyz)):
            r, g, b = colorsys.hsv_to_rgb(hue[i], saturation[i], value[i])
            colors[i] = [r, g, b]
        
        return colors
    
    def _analyze_map_quality(self):
        """分析地图质量"""
        print("执行地图质量分析...")
        
        quality_metrics = {
            "total_points": self.total_accumulated_points,
            "occupied_cells": int(np.sum(self.occupancy_grid > 0)),
            "coverage_ratio": np.sum(self.occupancy_grid > 0) / (self.map_pixels ** 2),
            "average_density": float(self.density_map.mean()),
            "max_density": float(self.density_map.max()),
            "height_range": [
                float(self.height_map[self.height_map != -np.inf].min()),
                float(self.height_map[self.height_map != -np.inf].max())
            ],
            "average_confidence": float(self.confidence_map.mean()),
            "high_confidence_ratio": float(np.sum(self.confidence_map > 0.7) / np.sum(self.occupancy_grid > 0)),
        }
        
        # 保存质量分析报告
        quality_file = self.analysis_dir / "quality_metrics.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        # 生成质量可视化
        self._generate_quality_visualization()
        
        print(f"地图质量分析完成，报告保存至: {quality_file}")
    
    def _generate_quality_visualization(self):
        """生成地图质量可视化"""
        print("生成质量可视化图表...")
        
        # 创建6个子图的复合显示
        fig_size = (1600, 1200)
        cell_size = (400, 400)
        
        def resize_map(map_data, colormap=cv2.COLORMAP_JET):
            if map_data.ndim == 3:  # 3通道数据
                resized = cv2.resize(map_data, cell_size)
                return (resized * 255).astype(np.uint8)
            else:  # 单通道数据
                normalized = cv2.normalize(map_data, None, 0, 255, cv2.NORM_MINMAX)
                resized = cv2.resize(normalized.astype(np.uint8), cell_size)
                return cv2.applyColorMap(resized, colormap)
        
        # 生成各种地图可视化
        occupancy_vis = resize_map(self.occupancy_grid, cv2.COLORMAP_GRAY)
        height_vis = resize_map(self.height_map, cv2.COLORMAP_JET)
        intensity_vis = resize_map(self.intensity_map, cv2.COLORMAP_HOT)
        density_vis = resize_map(self.density_map, cv2.COLORMAP_PLASMA)
        confidence_vis = resize_map(self.confidence_map, cv2.COLORMAP_VIRIDIS)
        
        # 法向量可视化（转换为颜色）
        normal_magnitude = np.linalg.norm(self.normal_map, axis=2)
        normal_vis = resize_map(normal_magnitude, cv2.COLORMAP_RAINBOW)
        
        # 组合显示
        top_row = np.hstack([occupancy_vis, height_vis, intensity_vis])
        bottom_row = np.hstack([density_vis, confidence_vis, normal_vis])
        combined = np.vstack([top_row, bottom_row])
        
        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (255, 255, 255)
        font_thickness = 2
        
        cv2.putText(combined, "Occupancy", (50, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Height", (450, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Intensity", (850, 30), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Density", (50, 430), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Confidence", (450, 430), font, font_scale, font_color, font_thickness)
        cv2.putText(combined, "Normals", (850, 430), font, font_scale, font_color, font_thickness)
        
        # 添加统计信息
        stats_text = [
            f"Points: {self.total_accumulated_points:,}",
            f"Coverage: {np.sum(self.occupancy_grid > 0) / (self.map_pixels ** 2) * 100:.1f}%",
            f"Avg Density: {self.density_map.mean():.2f}",
            f"Resolution: {self.map_resolution}m/px",
            f"Size: {self.map_size}x{self.map_size}m"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(combined, text, (50, 500 + i * 30), font, 0.6, font_color, 1)
        
        # 保存质量可视化
        quality_vis_file = self.maps_dir / "quality_visualization.png"
        cv2.imwrite(str(quality_vis_file), combined)
        
        print(f"质量可视化保存至: {quality_vis_file}")
    
    def save_professional_maps(self):
        """保存专业级地图文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        print(f"\n保存专业级地图到: {self.maps_dir}")
        
        with self.map_lock:
            # 保存所有地图层
            map_files = {}
            
            # 占用栅格
            occupancy_img = (self.occupancy_grid * 255).astype(np.uint8)
            occupancy_file = self.maps_dir / f"occupancy_map_{timestamp}.png"
            cv2.imwrite(str(occupancy_file), occupancy_img)
            map_files['occupancy'] = str(occupancy_file)
            
            # 高度地图
            height_normalized = cv2.normalize(self.height_map, None, 0, 255, cv2.NORM_MINMAX)
            height_colored = cv2.applyColorMap(height_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            height_file = self.maps_dir / f"height_map_{timestamp}.png"
            cv2.imwrite(str(height_file), height_colored)
            map_files['height'] = str(height_file)
            
            # 强度地图
            intensity_img = (self.intensity_map * 255).astype(np.uint8)
            intensity_file = self.maps_dir / f"intensity_map_{timestamp}.png"
            cv2.imwrite(str(intensity_file), intensity_img)
            map_files['intensity'] = str(intensity_file)
            
            # 密度地图
            density_img = (self.density_map * 255).astype(np.uint8)
            density_file = self.maps_dir / f"density_map_{timestamp}.png"
            cv2.imwrite(str(density_file), cv2.applyColorMap(density_img, cv2.COLORMAP_PLASMA))
            map_files['density'] = str(density_file)
            
            # 置信度地图
            confidence_img = (self.confidence_map * 255).astype(np.uint8)
            confidence_file = self.maps_dir / f"confidence_map_{timestamp}.png"
            cv2.imwrite(str(confidence_file), cv2.applyColorMap(confidence_img, cv2.COLORMAP_VIRIDIS))
            map_files['confidence'] = str(confidence_file)
            
            # 原始数据
            data_file = self.processed_data_dir / f"map_data_{timestamp}.npz"
            np.savez_compressed(data_file,
                               occupancy_grid=self.occupancy_grid,
                               height_map=self.height_map,
                               intensity_map=self.intensity_map,
                               density_map=self.density_map,
                               confidence_map=self.confidence_map,
                               normal_map=self.normal_map,
                               point_count_map=self.point_count_map)
            map_files['raw_data'] = str(data_file)
            
            # 元数据
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
                "processing_settings": {
                    "advanced_filtering": self.enable_advanced_filtering,
                    "outlier_removal": self.enable_outlier_removal,
                    "surface_reconstruction": self.enable_surface_reconstruction,
                    "voxel_leaf_size": self.voxel_leaf_size,
                    "statistical_k": self.statistical_k,
                    "statistical_std_ratio": self.statistical_std_ratio
                },
                "quality_metrics": {
                    "occupied_cells": int(np.sum(self.occupancy_grid > 0)),
                    "coverage_ratio": float(np.sum(self.occupancy_grid > 0) / (self.map_pixels ** 2)),
                    "average_density": float(self.density_map.mean()),
                    "average_confidence": float(self.confidence_map.mean())
                },
                "files": map_files
            }
            
            metadata_file = self.maps_dir / f"professional_metadata_{timestamp}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"专业级地图文件已保存:")
            for name, path in map_files.items():
                print(f"  {name}: {Path(path).name}")
            print(f"  元数据: {metadata_file.name}")
    
    def _save_high_quality_3d_map(self, pcd: o3d.geometry.PointCloud):
        """保存高质量3D地图"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # PLY格式（推荐，保持颜色和法向量）
        ply_file = self.maps_dir / f"professional_3d_map_{timestamp}.ply"
        o3d.io.write_point_cloud(str(ply_file), pcd, write_ascii=False, compressed=True)
        
        # PCD格式
        pcd_file = self.maps_dir / f"professional_3d_map_{timestamp}.pcd"
        o3d.io.write_point_cloud(str(pcd_file), pcd)
        
        print(f"高质量3D地图已保存:")
        print(f"  PLY格式: {ply_file.name}")
        print(f"  PCD格式: {pcd_file.name}")
    
    def _save_remaining_imu_data(self):
        """保存剩余IMU数据"""
        if self._imu_buffer:
            try:
                with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for data, ts in self._imu_buffer:
                        for row in data:
                            writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
                
                self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
                print(f"[ProfessionalMapper] 最终保存 {self._imu_count} 个 IMU 样本")
                self._imu_buffer = []
                
            except IOError as e:
                print(f"[ProfessionalMapper] IMU 数据保存失败: {e}", file=sys.stderr)
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理IMU数据"""
        if not self.is_mapping or len(imu_data) == 0:
            return
        
        # 数据处理（同原版本）
        imu_processed = imu_data.copy()
        imu_processed[:, 3:6] *= 9.81  # g to m/s²
        
        if self.mount == "upside_down":
            imu_processed[:, [1, 2, 4, 5]] *= -1
        
        self._imu_buffer.append((imu_processed, timestamp))
        
        # 每200个样本保存一次（减少I/O频率）
        if len(self._imu_buffer) >= 200:
            try:
                with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for data, ts in self._imu_buffer:
                        for row in data:
                            writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
                
                self._imu_count += sum(len(data) for data, _ in self._imu_buffer)
                self._imu_buffer = []
                
            except IOError as e:
                print(f"[ProfessionalMapper] IMU 写入失败: {e}", file=sys.stderr)
    
    def visualize_professional_maps(self):
        """可视化专业级地图"""
        print("显示专业级地图可视化...")
        
        # 读取质量可视化图像
        quality_vis_file = self.maps_dir / "quality_visualization.png"
        if quality_vis_file.exists():
            combined = cv2.imread(str(quality_vis_file))
            cv2.imshow("Professional Livox MID-360 Maps", combined)
            print("按任意键关闭地图显示...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("质量可视化文件不存在，请先运行地图生成")

def main():
    """主函数"""
    print("Professional Livox MID-360 地图构建器")
    print("="*60)
    
    # 检查配置文件
    config_path = Path("mid360_config.json")
    if not config_path.exists():
        print("创建默认配置文件...")
        create_default_config(config_path)
        print(f"✓ 配置文件已创建: {config_path}")
    
    print("\n专业级地图构建说明:")
    print("1. 高精度点云处理和多层地图生成")
    print("2. 高级滤波算法去除噪声和离群点")  
    print("3. 表面重建和法向量估计")
    print("4. 地图质量分析和可视化")
    print("5. 支持多种专业格式输出")
    
    # 配置专业参数
    print(f"\n当前挂载方向: {MOUNT}")
    mount = input("挂载方向 (upside_down/normal, 默认 upside_down): ").strip().lower() or "upside_down"
    
    print("\n专业级参数配置:")
    try:
        map_size = float(input("地图大小 (米, 默认30): ") or "30")
        map_resolution = float(input("地图分辨率 (米/像素, 默认0.05): ") or "0.05")
        accumulation_time = float(input("数据累积时间 (秒, 默认60): ") or "60")
        
        enable_advanced = input("启用高级滤波? (Y/n): ").strip().lower() != 'n'
        enable_outlier = input("启用离群点移除? (Y/n): ").strip().lower() != 'n'
        
        print(f"\n专业配置:")
        print(f"  地图大小: {map_size}x{map_size} 米")
        print(f"  分辨率: {map_resolution} 米/像素 (高精度)")
        print(f"  累积时间: {accumulation_time} 秒")
        print(f"  高级处理: {'启用' if enable_advanced else '禁用'}")
        
    except ValueError:
        print("使用默认专业参数")
        map_size = 30.0
        map_resolution = 0.05
        accumulation_time = 60.0
        enable_advanced = True
        enable_outlier = True
    
    try:
        print(f"\n初始化专业级地图构建器...")
        mapper = ProfessionalMapper(
            config_path,
            host_ip="192.168.123.164",
            frame_time=0.02,        # 20ms 高频帧聚合
            frame_packets=150,      # 更多数据包
            enable_filter=True,
            max_range=map_size/2,
            voxel_size=0.02,        # 更小体素
            map_size=map_size,
            map_resolution=map_resolution,
            accumulation_time=accumulation_time,
            height_filter_min=-3.0,
            height_filter_max=4.0,
            mount=mount,
            enable_advanced_filtering=enable_advanced,
            enable_outlier_removal=enable_outlier,
            enable_surface_reconstruction=True,
            voxel_leaf_size=0.02,
            statistical_k=20,
            statistical_std_ratio=2.0
        )
        
        with mapper:
            print("✓ 专业地图构建器初始化成功")
            print("等待雷达连接...")
            time.sleep(2.0)
            
            print(f"\n开始专业级地图构建 ({accumulation_time} 秒)")
            print("请保持雷达静止，进行高质量数据采集")
            print("按 Ctrl+C 提前停止")
            
            try:
                while mapper.is_mapping and mapper.is_running():
                    time.sleep(0.5)
                
                if not mapper.is_running():
                    print("雷达连接中断")
                    return
                    
            except KeyboardInterrupt:
                print("\n用户中断，开始地图生成...")
                mapper.is_mapping = False
                mapper._generate_professional_map()
            
            # 保存专业地图
            print("\n保存专业级地图...")
            mapper.save_professional_maps()
            
            # 显示结果
            display = input("\n显示专业地图? (Y/n): ").strip().lower() != 'n'
            if display:
                mapper.visualize_professional_maps()
            
            print(f"\n专业级地图构建完成！")
            print(f"数据保存目录: {mapper.data_dir}")
            print(f"地图文件目录: {mapper.maps_dir}")
            print(f"质量分析目录: {mapper.analysis_dir}")
            
    except Exception as e:
        print(f"✗ 专业地图构建失败: {e}")
        print("\n可能原因:")
        print("1. 网络连接或雷达配置问题")
        print("2. 依赖库版本问题 (Open3D, scikit-learn)")
        print("3. 内存不足（专业处理需要更多内存）")
        print("4. 磁盘空间不足")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()