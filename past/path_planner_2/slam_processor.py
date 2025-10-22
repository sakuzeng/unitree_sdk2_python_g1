"""
SLAM处理模块 - 优化的点云处理
"""
import numpy as np
import os
from typing import Tuple, List

from config import GridConfig

class OptimizedSLAMProcessor:
    """优化的SLAM处理器"""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.cell_size = config.grid_size / config.grid_resolution
        
        # 网格数据
        self.occupancy_grid = np.full((config.grid_resolution, config.grid_resolution), 128, dtype=np.uint8)
        self.log_odds = np.zeros((config.grid_resolution, config.grid_resolution), dtype=np.float32)
        
        # 坐标系
        self.origin = np.array([0.0, 0.0])
        self.origin_set = False
        
        # 预计算概率转换
        self.log_prob_hit = np.log(config.prob_hit / (1 - config.prob_hit))
        self.log_prob_miss = np.log(config.prob_miss / (1 - config.prob_miss))
        
        # 挂载方向
        self.mount = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
        
        print(f"[SLAMProcessor] 初始化完成")
    
    def process_points(self, xyz: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """
        处理点云数据，返回更新的占用网格
        
        Args:
            xyz: 点云数据
            robot_pose: 机器人位姿
            
        Returns:
            更新的占用网格
        """
        if len(xyz) == 0:
            return self.occupancy_grid
        
        # 设置原点
        if not self.origin_set:
            self.origin = robot_pose[:3, 3][:2].copy()
            self.origin_set = True
            print(f"[SLAMProcessor] 设置坐标原点: ({self.origin[0]:.2f}, {self.origin[1]:.2f})")
        
        # 应用挂载校正
        if self.mount == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0])
        
        # 坐标变换到世界坐标系
        xyz_world = self._transform_to_world(xyz, robot_pose)
        
        # 过滤点云
        xyz_filtered = self._filter_points(xyz_world, robot_pose[:3, 3])
        
        if len(xyz_filtered) > 0:
            # 更新占用网格
            self._update_occupancy_grid(xyz_filtered, robot_pose[:3, 3])
        
        return self.occupancy_grid
    
    def _transform_to_world(self, xyz: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """坐标变换到世界坐标系"""
        xyz_homo = np.hstack([xyz, np.ones((len(xyz), 1))])
        return (robot_pose @ xyz_homo.T).T[:, :3]
    
    def _filter_points(self, xyz: np.ndarray, sensor_pos: np.ndarray) -> np.ndarray:
        """点云过滤"""
        # 距离过滤
        distances = np.linalg.norm(xyz[:, :2] - sensor_pos[:2], axis=1)
        distance_mask = (distances > 0.3) & (distances < self.config.max_range)
        
        # 高度过滤
        height_mask = (xyz[:, 2] > self.config.min_height) & (xyz[:, 2] < self.config.max_height)
        
        return xyz[distance_mask & height_mask]
    
    def _update_occupancy_grid(self, xyz: np.ndarray, sensor_pos: np.ndarray):
        """更新占用网格"""
        # 转换到网格坐标
        grid_coords = self._world_to_grid_coord(xyz)
        sensor_grid = self._world_to_grid_coord(sensor_pos.reshape(1, -1))[0]
        
        # 重置网格
        self.log_odds *= 0.98  # 轻微衰减
        
        # 标记占用点
        for point in grid_coords:
            if self._is_valid_grid_point(point):
                self.log_odds[point[1], point[0]] += self.log_prob_hit
        
        # 简化的光线追踪（采样）
        if len(grid_coords) > 0:
            sample_size = min(len(grid_coords), 25)
            sampled_indices = np.random.choice(len(grid_coords), sample_size, replace=False)
            
            for idx in sampled_indices:
                end_point = grid_coords[idx]
                if self._is_valid_grid_point(end_point):
                    line_points = self._bresenham_line(sensor_grid[0], sensor_grid[1], 
                                                      end_point[0], end_point[1])
                    for lx, ly in line_points[::2]:  # 每隔一个点
                        if (self._is_valid_grid_point((lx, ly)) and 
                            (lx != end_point[0] or ly != end_point[1])):
                            self.log_odds[ly, lx] += self.log_prob_miss
        
        # 限制范围并转换
        self.log_odds = np.clip(self.log_odds, -5.0, 5.0)
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        
        # 生成离散网格
        self.occupancy_grid = np.full_like(self.occupancy_grid, 128, dtype=np.uint8)
        self.occupancy_grid[prob >= self.config.hit_threshold] = 255
        self.occupancy_grid[prob <= self.config.free_threshold] = 0
    
    def _world_to_grid_coord(self, xyz: np.ndarray) -> np.ndarray:
        """世界坐标转网格坐标"""
        relative_coords = xyz[:, :2] - self.origin
        center = self.config.grid_resolution // 2
        grid_x = (relative_coords[:, 0] / self.cell_size + center).astype(np.int32)
        grid_y = (-relative_coords[:, 1] / self.cell_size + center).astype(np.int32)
        return np.column_stack([grid_x, grid_y])
    
    def _is_valid_grid_point(self, point: Tuple[int, int]) -> bool:
        """检查网格点是否有效"""
        return (0 <= point[0] < self.config.grid_resolution and 
                0 <= point[1] < self.config.grid_resolution)
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法"""
        points = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
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