#!/usr/bin/env python3
"""
PCD点云文件转换为PGM栅格地图工具

功能：
- 读取PCD点云文件
- 投影到2D平面生成栅格地图
- 保存为PGM格式
- 显示栅格地图

使用方法：
python pcd_to_pgm_converter.py --input /path/to/pointcloud.pcd --output /path/to/map.pgm
"""

import argparse
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os


class PCDToPGMConverter:
    """PCD点云文件转换为PGM栅格地图的转换器"""
    
    def __init__(self, resolution=0.05, height_threshold=0.5):
        """
        初始化转换器
        
        Args:
            resolution (float): 栅格地图分辨率，单位米/像素，默认0.05m
            height_threshold (float): 高度阈值，超过此高度视为障碍物，默认0.5m
        """
        self.resolution = resolution
        self.height_threshold = height_threshold
        
    def load_pcd(self, pcd_file):
        """
        加载PCD点云文件
        
        Args:
            pcd_file (str): PCD文件路径
            
        Returns:
            o3d.geometry.PointCloud: 点云对象
        """
        try:
            print(f"正在加载PCD文件: {pcd_file}")
            pcd = o3d.io.read_point_cloud(pcd_file)
            
            if len(pcd.points) == 0:
                raise ValueError("PCD文件为空或格式错误")
                
            print(f"成功加载点云，共 {len(pcd.points)} 个点")
            return pcd
            
        except Exception as e:
            print(f"加载PCD文件失败: {e}")
            return None
            
    def preprocess_pointcloud(self, pcd):
        """
        预处理点云数据
        
        Args:
            pcd (o3d.geometry.PointCloud): 原始点云
            
        Returns:
            np.ndarray: 预处理后的点云数组 (N, 3)
        """
        points = np.asarray(pcd.points)
        
        # 移除异常值
        pcd_filtered = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        
        # 下采样以提高处理速度
        pcd_downsampled = pcd_filtered.voxel_down_sample(voxel_size=self.resolution/2)
        
        points = np.asarray(pcd_downsampled.points)
        print(f"预处理后点云数量: {len(points)}")
        
        return points
        
    def points_to_occupancy_grid(self, points):
        """
        将3D点云转换为2D栅格地图
        
        Args:
            points (np.ndarray): 点云数组 (N, 3)
            
        Returns:
            tuple: (grid, info) 栅格地图和信息字典
        """
        # 提取X, Y, Z坐标
        x_coords = points[:, 0]
        y_coords = points[:, 1] 
        z_coords = points[:, 2]
        
        # 计算地图边界
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # 计算栅格地图尺寸
        width = int((x_max - x_min) / self.resolution) + 1
        height = int((y_max - y_min) / self.resolution) + 1
        
        print(f"地图范围: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
        print(f"栅格地图尺寸: {width} x {height}")
        
        # 初始化栅格地图 (0: 未知, 128: 自由空间, 255: 障碍物)
        occupancy_grid = np.full((height, width), 0, dtype=np.uint8)
        
        # 将点云投影到2D栅格
        for i in range(len(points)):
            # 计算栅格坐标
            grid_x = int((x_coords[i] - x_min) / self.resolution)
            grid_y = int((y_coords[i] - y_min) / self.resolution)
            
            # 确保坐标在范围内
            if 0 <= grid_x < width and 0 <= grid_y < height:
                # 根据高度判断是否为障碍物
                if z_coords[i] > self.height_threshold:
                    occupancy_grid[height - 1 - grid_y, grid_x] = 255  # 障碍物
                else:
                    # 如果当前位置不是障碍物，标记为自由空间
                    if occupancy_grid[height - 1 - grid_y, grid_x] != 255:
                        occupancy_grid[height - 1 - grid_y, grid_x] = 128  # 自由空间
        
        # 地图信息
        map_info = {
            'resolution': self.resolution,
            'origin_x': x_min,
            'origin_y': y_min,
            'width': width,
            'height': height,
            'height_threshold': self.height_threshold
        }
        
        return occupancy_grid, map_info
        
    def save_pgm(self, grid, output_file, map_info):
        """
        保存栅格地图为PGM格式
        
        Args:
            grid (np.ndarray): 栅格地图
            output_file (str): 输出PGM文件路径
            map_info (dict): 地图信息
        """
        try:
            # 保存PGM文件
            image = Image.fromarray(grid, mode='L')
            image.save(output_file)
            
            # 保存YAML配置文件 (ROS导航栈格式)
            yaml_file = output_file.replace('.pgm', '.yaml')
            with open(yaml_file, 'w') as f:
                f.write(f"image: {os.path.basename(output_file)}\n")
                f.write(f"resolution: {map_info['resolution']}\n")
                f.write(f"origin: [{map_info['origin_x']}, {map_info['origin_y']}, 0.0]\n")
                f.write("negate: 0\n")
                f.write("occupied_thresh: 0.65\n")
                f.write("free_thresh: 0.196\n")
            
            print(f"栅格地图已保存: {output_file}")
            print(f"配置文件已保存: {yaml_file}")
            
        except Exception as e:
            print(f"保存PGM文件失败: {e}")
            
    def display_map(self, grid, map_info):
        """
        显示栅格地图
        
        Args:
            grid (np.ndarray): 栅格地图
            map_info (dict): 地图信息
        """
        plt.figure(figsize=(12, 8))
        
        # 创建颜色映射
        cmap = plt.cm.colors.ListedColormap(['gray', 'white', 'black'])
        bounds = [0, 85, 170, 255]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        plt.imshow(grid, cmap=cmap, norm=norm, origin='lower')
        plt.colorbar(label='占用状态 (0:未知, 128:自由, 255:障碍物)')
        
        plt.title(f'栅格地图 - 分辨率: {map_info["resolution"]}m/pixel')
        plt.xlabel(f'X坐标 (像素) - 原点: {map_info["origin_x"]:.2f}m')
        plt.ylabel(f'Y坐标 (像素) - 原点: {map_info["origin_y"]:.2f}m')
        
        # 添加网格
        plt.grid(True, alpha=0.3)
        
        # 显示统计信息
        total_pixels = grid.size
        unknown_pixels = np.sum(grid == 0)
        free_pixels = np.sum(grid == 128)
        occupied_pixels = np.sum(grid == 255)
        
        stats_text = f"总像素: {total_pixels}\n"
        stats_text += f"未知: {unknown_pixels} ({100*unknown_pixels/total_pixels:.1f}%)\n"
        stats_text += f"自由: {free_pixels} ({100*free_pixels/total_pixels:.1f}%)\n"
        stats_text += f"障碍: {occupied_pixels} ({100*occupied_pixels/total_pixels:.1f}%)"
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
    def convert(self, pcd_file, output_file, display=True):
        """
        执行完整的转换流程
        
        Args:
            pcd_file (str): 输入PCD文件路径
            output_file (str): 输出PGM文件路径
            display (bool): 是否显示地图
            
        Returns:
            bool: 转换成功返回True
        """
        try:
            # 1. 加载PCD文件
            pcd = self.load_pcd(pcd_file)
            if pcd is None:
                return False
                
            # 2. 预处理点云
            points = self.preprocess_pointcloud(pcd)
            
            # 3. 转换为栅格地图
            grid, map_info = self.points_to_occupancy_grid(points)
            
            # 4. 保存PGM文件
            self.save_pgm(grid, output_file, map_info)
            
            # 5. 显示地图
            if display:
                self.display_map(grid, map_info)
                
            print("转换完成!")
            return True
            
        except Exception as e:
            print(f"转换过程发生错误: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将PCD点云文件转换为PGM栅格地图')
    parser.add_argument('--input', '-i', required=True, help='输入PCD文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出PGM文件路径')
    parser.add_argument('--resolution', '-r', type=float, default=0.05, help='栅格分辨率 (m/pixel)')
    parser.add_argument('--height_threshold', '-t', type=float, default=0.5, help='障碍物高度阈值 (m)')
    parser.add_argument('--no_display', action='store_true', help='不显示地图')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
        
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 创建转换器
    converter = PCDToPGMConverter(
        resolution=args.resolution,
        height_threshold=args.height_threshold
    )
    
    # 执行转换
    success = converter.convert(
        pcd_file=args.input,
        output_file=args.output,
        display=not args.no_display
    )
    
    if success:
        print("PCD到PGM转换成功完成!")
    else:
        print("转换失败!")


if __name__ == "__main__":
    main()