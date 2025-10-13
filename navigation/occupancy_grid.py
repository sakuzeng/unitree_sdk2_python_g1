#!/usr/bin/env python3
"""
2D 占用网格生成和管理模块
用于SLAM系统的环境地图表示和路径规划

技术栈:
- numpy: 数值计算和数组操作
- matplotlib: 实时可视化
- threading: 线程安全的数据更新
- json: 元数据保存和加载
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle, Rectangle
import time
import threading
from typing import Tuple, Optional, List, Dict, Any
import json
from pathlib import Path

class OccupancyGrid:
    """
    2D占用网格管理器
    
    使用贝叶斯更新和射线追踪算法生成高精度占用概率地图
    支持实时可视化和路径规划接口
    """
    
    def __init__(self, 
                 resolution: float = 0.1, 
                 size: int = 1000, 
                 min_coord: float = -50.0, 
                 max_coord: float = 50.0,
                 enable_visualization: bool = True,
                 enable_ray_tracing: bool = True):
        """
        初始化占用网格
        
        Args:
            resolution: 网格分辨率（米/格）
            size: 网格尺寸（格数）
            min_coord: 最小坐标值
            max_coord: 最大坐标值
            enable_visualization: 是否启用实时可视化
            enable_ray_tracing: 是否启用射线追踪算法
        """
        self.resolution = resolution
        self.size = size
        self.min_coord = min_coord
        self.max_coord = max_coord
        self.enable_visualization = enable_visualization
        self.enable_ray_tracing = enable_ray_tracing
        
        # 占用网格数据存储
        self.grid = np.zeros((size, size), dtype=np.float32)  # 占用概率 [0-100]
        self.hit_count = np.zeros((size, size), dtype=np.int32)  # 命中计数
        self.miss_count = np.zeros((size, size), dtype=np.int32)  # 遗漏计数
        
        # 贝叶斯更新参数
        self.log_odds_occupied = np.log(0.7 / 0.3)  # 占用时的对数几率
        self.log_odds_free = np.log(0.3 / 0.7)      # 空闲时的对数几率
        self.log_odds_grid = np.zeros((size, size), dtype=np.float32)  # 对数几率网格
        
        # 可视化组件
        self.fig = None
        self.ax = None
        self.im = None
        self.position_marker = None
        self.path_line = None
        self.goal_marker = None
        
        # 更新控制
        self.last_update = time.time()
        self.update_interval = 0.1  # 更新间隔（秒）
        self.update_lock = threading.Lock()
        
        # 统计信息
        self.total_updates = 0
        self.total_points_processed = 0
        
        # 初始化可视化
        if self.enable_visualization:
            self._initialize_visualization()
        
        print(f"[OccupancyGrid] 初始化完成:")
        print(f"  - 网格尺寸: {size}x{size}")
        print(f"  - 分辨率: {resolution}m/格")
        print(f"  - 覆盖范围: [{min_coord}, {max_coord}]米")
        print(f"  - 射线追踪: {'启用' if enable_ray_tracing else '禁用'}")
        print(f"  - 可视化: {'启用' if enable_visualization else '禁用'}")
    
    def _initialize_visualization(self):
        """初始化实时可视化组件"""
        try:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 10), facecolor='white')
            self.ax = self.fig.add_subplot(111)
            
            # 设置自定义颜色映射
            colors_list = ['white', 'lightgray', 'gray', 'darkgray', 'black']
            cmap = colors.ListedColormap(colors_list)
            bounds = [0, 20, 40, 60, 80, 100]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            
            # 初始化图像显示
            self.im = self.ax.imshow(
                self.grid,
                cmap=cmap,
                norm=norm,
                extent=[self.min_coord, self.max_coord, self.min_coord, self.max_coord],
                origin='lower',
                interpolation='nearest'
            )
            
            # 设置标题和标签
            self.ax.set_title("实时占用网格地图", fontsize=16, pad=20)
            self.ax.set_xlabel("X (米)", fontsize=14)
            self.ax.set_ylabel("Y (米)", fontsize=14)
            
            # 添加网格线
            self.ax.grid(True, which='major', linestyle='--', alpha=0.3, color='blue')
            self.ax.set_xticks(np.arange(self.min_coord, self.max_coord + 1, 10.0))
            self.ax.set_yticks(np.arange(self.min_coord, self.max_coord + 1, 10.0))
            
            # 机器人位置标记
            self.position_marker = Circle((0, 0), 0.5, color='red', alpha=0.8, linewidth=2)
            self.ax.add_patch(self.position_marker)
            
            # 目标点标记（初始隐藏）
            self.goal_marker = Circle((0, 0), 0.3, color='green', alpha=0.0, linewidth=2)
            self.ax.add_patch(self.goal_marker)
            
            # 路径线（初始为空）
            self.path_line, = self.ax.plot([], [], 'b-', linewidth=3, alpha=0.7, label='规划路径')
            
            # 添加颜色条
            cbar = self.fig.colorbar(self.im, ax=self.ax, shrink=0.8)
            cbar.set_label('占用概率 (%)', fontsize=12)
            cbar.set_ticks([0, 25, 50, 75, 100])
            cbar.set_ticklabels(['空闲', '低概率', '未知', '高概率', '占用'])
            
            # 添加图例
            self.ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            plt.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"[ERROR] 初始化可视化失败: {e}")
            self.enable_visualization = False
    
    def update(self, xyz: np.ndarray, current_pose: np.ndarray = None):
        """
        更新占用网格
        
        Args:
            xyz: 点云数据 (N x 3)
            current_pose: 当前位姿矩阵 (4x4)
        """
        current_time = time.time()
        
        # 频率控制
        if current_time - self.last_update < self.update_interval:
            return
        
        with self.update_lock:
            try:
                if self.enable_ray_tracing and current_pose is not None:
                    self._update_with_ray_tracing(xyz, current_pose)
                else:
                    self._update_simple(xyz)
                
                # 更新统计信息
                self.total_updates += 1
                self.total_points_processed += len(xyz)
                
                # 转换为概率表示
                self._update_probability_grid()
                
                # 更新可视化
                if self.enable_visualization:
                    self._update_visualization(current_pose)
                
                self.last_update = current_time
                
            except Exception as e:
                print(f"[ERROR] 占用网格更新失败: {e}")
    
    def _update_with_ray_tracing(self, xyz: np.ndarray, current_pose: np.ndarray):
        """
        使用射线追踪算法更新网格（高精度方法）
        
        Args:
            xyz: 点云数据
            current_pose: 当前机器人位姿
        """
        if xyz.size == 0:
            return
        
        # 获取机器人位置
        robot_pos = current_pose[:3, 3]
        robot_grid = self._world_to_grid(robot_pos[:2])
        
        # 过滤有效点云
        points_2d = xyz[:, :2]
        valid_mask = self._is_in_bounds(points_2d)
        valid_points = points_2d[valid_mask]
        
        if len(valid_points) == 0:
            return
        
        # 对每个有效点进行射线追踪
        for point in valid_points:
            end_grid = self._world_to_grid(point)
            
            # 使用Bresenham算法获取射线路径
            ray_points = self._bresenham_line(robot_grid, end_grid)
            
            # 更新射线路径上的自由空间
            for i, (gx, gy) in enumerate(ray_points[:-1]):  # 除了最后一个点
                if self._is_valid_grid_pos((gx, gy)):
                    self.log_odds_grid[gy, gx] += self.log_odds_free
                    self.miss_count[gy, gx] += 1
            
            # 更新终点为占用
            end_gx, end_gy = end_grid
            if self._is_valid_grid_pos((end_gx, end_gy)):
                self.log_odds_grid[end_gy, end_gx] += self.log_odds_occupied
                self.hit_count[end_gy, end_gx] += 1
    
    def _update_simple(self, xyz: np.ndarray):
        """
        简单更新方法（仅标记占用点）
        
        Args:
            xyz: 点云数据
        """
        if xyz.size == 0:
            return
        
        # 转换为2D点并过滤
        points_2d = xyz[:, :2]
        valid_mask = self._is_in_bounds(points_2d)
        valid_points = points_2d[valid_mask]
        
        if len(valid_points) == 0:
            return
        
        # 转换为网格坐标
        grid_coords = np.array([self._world_to_grid(point) for point in valid_points])
        
        # 过滤有效网格坐标
        valid_grid_mask = np.array([self._is_valid_grid_pos((gx, gy)) 
                                   for gx, gy in grid_coords])
        valid_grid_coords = grid_coords[valid_grid_mask]
        
        # 更新网格
        for gx, gy in valid_grid_coords:
            self.log_odds_grid[gy, gx] += self.log_odds_occupied
            self.hit_count[gy, gx] += 1
    
    def _update_probability_grid(self):
        """将对数几率转换为概率"""
        # 限制对数几率范围，避免数值溢出
        self.log_odds_grid = np.clip(self.log_odds_grid, -5.0, 5.0)
        
        # 转换为概率 (0-100)
        probabilities = 1.0 / (1.0 + np.exp(-self.log_odds_grid))
        self.grid = (probabilities * 100).astype(np.float32)
    
    def _is_in_bounds(self, points: np.ndarray) -> np.ndarray:
        """检查点是否在网格范围内"""
        return ((points[:, 0] >= self.min_coord) & (points[:, 0] < self.max_coord) &
                (points[:, 1] >= self.min_coord) & (points[:, 1] < self.max_coord))
    
    def _world_to_grid(self, world_point: np.ndarray) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        gx = int((world_point[0] - self.min_coord) / self.resolution)
        gy = int((world_point[1] - self.min_coord) / self.resolution)
        return (gx, gy)
    
    def _grid_to_world(self, grid_point: Tuple[int, int]) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x = grid_point[0] * self.resolution + self.min_coord
        y = grid_point[1] * self.resolution + self.min_coord
        return (x, y)
    
    def _is_valid_grid_pos(self, grid_pos: Tuple[int, int]) -> bool:
        """检查网格位置是否有效"""
        gx, gy = grid_pos
        return 0 <= gx < self.size and 0 <= gy < self.size
    
    def _bresenham_line(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham直线算法实现射线追踪"""
        points = []
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return points
    
    def _update_visualization(self, current_pose: np.ndarray = None):
        """更新实时可视化显示"""
        if not self.enable_visualization or self.fig is None:
            return
        
        try:
            # 更新网格显示
            self.im.set_data(self.grid)
            
            # 更新机器人位置和朝向
            if current_pose is not None:
                position = current_pose[:3, 3]
                if (self.min_coord <= position[0] <= self.max_coord and 
                    self.min_coord <= position[1] <= self.max_coord):
                    self.position_marker.center = (position[0], position[1])
                    
                    # 添加方向指示箭头
                    theta = np.arctan2(current_pose[1, 0], current_pose[0, 0])
                    arrow_length = 1.0
                    arrow_end_x = position[0] + arrow_length * np.cos(theta)
                    arrow_end_y = position[1] + arrow_length * np.sin(theta)
                    
                    # 清除之前的箭头
                    for artist in self.ax.patches[:]:
                        if hasattr(artist, '_arrow_flag'):
                            artist.remove()
                    
                    # 添加新的方向箭头
                    arrow = self.ax.arrow(position[0], position[1], 
                                         arrow_end_x - position[0], arrow_end_y - position[1],
                                         head_width=0.3, head_length=0.2, 
                                         fc='red', ec='red', alpha=0.8)
                    arrow._arrow_flag = True  # 标记为箭头对象
            
            # 更新标题显示统计信息
            self.ax.set_title(
                f"实时占用网格地图 | 更新次数: {self.total_updates} | "
                f"处理点数: {self.total_points_processed:,}",
                fontsize=14, pad=20
            )
            
            # 刷新显示
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"[WARNING] 可视化更新失败: {e}")
    
    def set_goal(self, goal_pos: Tuple[float, float]):
        """
        设置目标点并在可视化中显示
        
        Args:
            goal_pos: 目标位置 (x, y)
        """
        if not self.enable_visualization or self.goal_marker is None:
            return
        
        if (self.min_coord <= goal_pos[0] <= self.max_coord and 
            self.min_coord <= goal_pos[1] <= self.max_coord):
            self.goal_marker.center = goal_pos
            self.goal_marker.set_alpha(0.8)
            print(f"[OccupancyGrid] 目标点设置为: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    
    def set_path(self, path: List[Tuple[float, float]]):
        """
        在可视化中显示规划路径
        
        Args:
            path: 路径点列表 [(x1,y1), (x2,y2), ...]
        """
        if not self.enable_visualization or self.path_line is None:
            return
        
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.path_line.set_data(path_x, path_y)
            self.path_line.set_alpha(0.8)
            print(f"[OccupancyGrid] 路径已更新: {len(path)} 个路径点")
        else:
            self.path_line.set_data([], [])
            self.path_line.set_alpha(0.0)
    
    def clear_path(self):
        """清除显示的路径"""
        if self.path_line is not None:
            self.path_line.set_data([], [])
            self.path_line.set_alpha(0.0)
    
    def get_occupancy_at(self, world_pos: Tuple[float, float]) -> float:
        """
        获取指定世界坐标的占用概率
        
        Args:
            world_pos: 世界坐标 (x, y)
            
        Returns:
            占用概率 [0-100]
        """
        gx, gy = self._world_to_grid(np.array(world_pos))
        if self._is_valid_grid_pos((gx, gy)):
            return float(self.grid[gy, gx])
        return 50.0  # 未知区域返回50%
    
    def is_occupied(self, world_pos: Tuple[float, float], threshold: float = 60.0) -> bool:
        """
        检查指定位置是否被占用
        
        Args:
            world_pos: 世界坐标 (x, y)
            threshold: 占用阈值 (0-100)
            
        Returns:
            是否被占用
        """
        return self.get_occupancy_at(world_pos) > threshold
    
    def is_free(self, world_pos: Tuple[float, float], threshold: float = 40.0) -> bool:
        """
        检查指定位置是否空闲
        
        Args:
            world_pos: 世界坐标 (x, y)
            threshold: 空闲阈值 (0-100)
            
        Returns:
            是否空闲
        """
        return self.get_occupancy_at(world_pos) < threshold
    
    def get_grid_data(self) -> np.ndarray:
        """
        获取网格数据副本（线程安全）
        
        Returns:
            网格数据副本
        """
        with self.update_lock:
            return self.grid.copy()
    
    def get_grid_origin(self) -> Tuple[float, float]:
        """
        获取网格原点坐标
        
        Returns:
            网格原点 (x, y)
        """
        return (self.min_coord, self.min_coord)
    
    def save(self, filename: str, save_metadata: bool = True):
        """
        保存占用网格到文件
        
        Args:
            filename: 保存文件名
            save_metadata: 是否保存元数据
        """
        try:
            # 保存网格图像
            if self.enable_visualization and self.im is not None:
                self.im.set_data(self.grid)
                plt.imsave(filename, self.grid, cmap=self.im.cmap, vmin=0, vmax=100)
            else:
                plt.imsave(filename, self.grid, cmap='gray', vmin=0, vmax=100)
            
            # 保存原始数据
            data_filename = filename.replace('.png', '_data.npy')
            np.save(data_filename, self.grid)
            
            # 保存元数据
            if save_metadata:
                metadata = {
                    'grid_config': {
                        'resolution': self.resolution,
                        'size': self.size,
                        'min_coord': self.min_coord,
                        'max_coord': self.max_coord,
                        'enable_ray_tracing': self.enable_ray_tracing
                    },
                    'statistics': {
                        'total_updates': self.total_updates,
                        'total_points_processed': self.total_points_processed,
                        'hit_count_total': int(np.sum(self.hit_count)),
                        'miss_count_total': int(np.sum(self.miss_count)),
                        'occupied_cells': int(np.sum(self.grid > 60)),
                        'free_cells': int(np.sum(self.grid < 40)),
                        'unknown_cells': int(np.sum((self.grid >= 40) & (self.grid <= 60)))
                    },
                    'save_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                metadata_filename = filename.replace('.png', '_metadata.json')
                with open(metadata_filename, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"[OccupancyGrid] 已保存网格到 {filename}")
            print(f"[OccupancyGrid] 网格统计: 占用点 {np.sum(self.grid > 60)}, "
                  f"空闲点 {np.sum(self.grid < 40)}, 未知点 {np.sum((self.grid >= 40) & (self.grid <= 60))}")
            
        except Exception as e:
            print(f"[ERROR] 保存占用网格失败: {e}")
    
    def load(self, filename: str) -> bool:
        """
        从文件加载占用网格
        
        Args:
            filename: 文件名
            
        Returns:
            是否加载成功
        """
        try:
            data_filename = filename.replace('.png', '_data.npy')
            if Path(data_filename).exists():
                self.grid = np.load(data_filename)
                
                # 加载元数据
                metadata_filename = filename.replace('.png', '_metadata.json')
                if Path(metadata_filename).exists():
                    with open(metadata_filename, 'r') as f:
                        metadata = json.load(f)
                    
                    grid_config = metadata.get('grid_config', {})
                    self.resolution = grid_config.get('resolution', self.resolution)
                    self.size = grid_config.get('size', self.size)
                    self.min_coord = grid_config.get('min_coord', self.min_coord)
                    self.max_coord = grid_config.get('max_coord', self.max_coord)
                    
                    statistics = metadata.get('statistics', {})
                    self.total_updates = statistics.get('total_updates', 0)
                    self.total_points_processed = statistics.get('total_points_processed', 0)
                
                print(f"[OccupancyGrid] 成功加载网格: {filename}")
                return True
            else:
                print(f"[ERROR] 数据文件不存在: {data_filename}")
                return False
                
        except Exception as e:
            print(f"[ERROR] 加载占用网格失败: {e}")
            return False
    
    def reset(self):
        """重置网格数据"""
        with self.update_lock:
            self.grid.fill(0)
            self.log_odds_grid.fill(0)
            self.hit_count.fill(0)
            self.miss_count.fill(0)
            self.total_updates = 0
            self.total_points_processed = 0
        
        print("[OccupancyGrid] 网格已重置")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取网格统计信息
        
        Returns:
            统计信息字典
        """
        occupied_cells = np.sum(self.grid > 60)
        free_cells = np.sum(self.grid < 40)
        unknown_cells = np.sum((self.grid >= 40) & (self.grid <= 60))
        
        return {
            'total_updates': self.total_updates,
            'total_points_processed': self.total_points_processed,
            'occupied_cells': int(occupied_cells),
            'free_cells': int(free_cells),
            'unknown_cells': int(unknown_cells),
            'grid_coverage': float((occupied_cells + free_cells) / (self.size * self.size) * 100),
            'resolution': self.resolution,
            'grid_size': self.size,
            'world_bounds': (self.min_coord, self.max_coord)
        }
    
    def close(self):
        """关闭占用网格和可视化"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        
        print("[OccupancyGrid] 已关闭")

# ---------------------------------------------------------------------------
# 测试和演示代码
# ---------------------------------------------------------------------------

def test_occupancy_grid():
    """测试占用网格功能"""
    print("[TEST] 开始测试占用网格...")
    
    # 创建测试网格
    grid = OccupancyGrid(
        resolution=0.1, 
        size=500, 
        min_coord=-25.0, 
        max_coord=25.0,
        enable_visualization=True,
        enable_ray_tracing=True
    )
    
    # 模拟点云数据
    print("[TEST] 生成模拟点云数据...")
    test_points = np.random.uniform(-20, 20, (1000, 3))
    test_pose = np.eye(4)
    test_pose[:3, 3] = [0, 0, 1]
    
    # 更新网格
    print("[TEST] 更新占用网格...")
    grid.update(test_points, test_pose)
    
    # 显示统计信息
    stats = grid.get_statistics()
    print("[TEST] 网格统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试查询功能
    print("[TEST] 测试位置查询...")
    test_positions = [(0, 0), (10, 10), (-5, 5)]
    for pos in test_positions:
        occupancy = grid.get_occupancy_at(pos)
        is_occ = grid.is_occupied(pos)
        is_free = grid.is_free(pos)
        print(f"  位置 {pos}: 占用概率={occupancy:.1f}%, 占用={is_occ}, 空闲={is_free}")
    
    # 保存测试
    print("[TEST] 保存网格...")
    grid.save("test_occupancy_grid.png")
    
    return grid

if __name__ == "__main__":
    # 运行测试
    test_grid = test_occupancy_grid()
    
    # 保持显示
    try:
        input("按回车键关闭测试...")
    except KeyboardInterrupt:
        print("\n[INFO] 测试被中断")
    finally:
        test_grid.close()