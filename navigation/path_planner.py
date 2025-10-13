#!/usr/bin/env python3
"""
全局路径规划模块
基于占用网格实现A*算法进行路径规划
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict
import math
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PathNode:
    """路径节点"""
    x: int
    y: int
    g_cost: float = 0.0  # 从起点到当前点的代价
    h_cost: float = 0.0  # 从当前点到终点的启发式代价
    f_cost: float = 0.0  # 总代价 f = g + h
    parent: Optional['PathNode'] = None
    
    def __post_init__(self):
        self.f_cost = self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class GlobalPathPlanner:
    """全局路径规划器"""
    
    def __init__(self, grid_resolution: float = 0.1, safety_margin: int = 2):
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin  # 安全边距（网格单元数）
        self.occupancy_threshold = 50  # 占用阈值
        
    def plan_path(self, occupancy_grid: np.ndarray, start_pos: Tuple[float, float], 
                  goal_pos: Tuple[float, float], grid_origin: Tuple[float, float] = (-50.0, -50.0)) -> Optional[List[Tuple[float, float]]]:
        """
        使用A*算法规划路径
        
        Args:
            occupancy_grid: 占用网格 (值: 0=空闲, 100=占用)
            start_pos: 起始位置 (x, y) 世界坐标
            goal_pos: 目标位置 (x, y) 世界坐标
            grid_origin: 网格原点在世界坐标系中的位置
            
        Returns:
            路径点列表 [(x1,y1), (x2,y2), ...] 或 None
        """
        # 转换为网格坐标
        start_grid = self._world_to_grid(start_pos, grid_origin)
        goal_grid = self._world_to_grid(goal_pos, grid_origin)
        
        # 检查起点和终点是否有效
        if not self._is_valid_position(start_grid, occupancy_grid.shape):
            print(f"[ERROR] 起始位置无效: {start_pos} -> {start_grid}")
            return None
            
        if not self._is_valid_position(goal_grid, occupancy_grid.shape):
            print(f"[ERROR] 目标位置无效: {goal_pos} -> {goal_grid}")
            return None
        
        # 检查起点和终点是否被占用
        if self._is_occupied(start_grid, occupancy_grid):
            print(f"[ERROR] 起始位置被占用: {start_grid}")
            return None
            
        if self._is_occupied(goal_grid, occupancy_grid):
            print(f"[ERROR] 目标位置被占用: {goal_grid}")
            return None
        
        # A*算法核心
        path_grid = self._astar_search(start_grid, goal_grid, occupancy_grid)
        
        if path_grid is None:
            print("[ERROR] 未找到可行路径")
            return None
        
        # 转换回世界坐标并进行路径平滑
        path_world = [self._grid_to_world(pos, grid_origin) for pos in path_grid]
        smoothed_path = self._smooth_path(path_world, occupancy_grid, grid_origin)
        
        print(f"[INFO] 路径规划成功: {len(smoothed_path)} 个路径点")
        return smoothed_path
    
    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                      occupancy_grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """A*搜索算法实现"""
        
        start_node = PathNode(start[0], start[1], 0, self._heuristic(start, goal))
        
        open_set = [start_node]
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        
        # 8方向移动
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)
            
            if current_pos == goal:
                # 重构路径
                path = []
                while current_pos in came_from:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                path.append(start)
                return path[::-1]  # 反转路径
            
            closed_set.add(current_pos)
            
            for dx, dy in directions:
                neighbor = (current.x + dx, current.y + dy)
                
                if neighbor in closed_set:
                    continue
                
                if not self._is_valid_position(neighbor, occupancy_grid.shape):
                    continue
                
                if self._is_occupied(neighbor, occupancy_grid):
                    continue
                
                # 计算移动代价
                move_cost = math.sqrt(dx*dx + dy*dy)  # 对角线移动代价更高
                tentative_g = g_score[current_pos] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current_pos
                    g_score[neighbor] = tentative_g
                    h_cost = self._heuristic(neighbor, goal)
                    
                    neighbor_node = PathNode(neighbor[0], neighbor[1], 
                                           tentative_g, h_cost)
                    heapq.heappush(open_set, neighbor_node)
        
        return None  # 未找到路径
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发式函数 - 欧几里得距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _is_valid_position(self, pos: Tuple[int, int], grid_shape: Tuple[int, int]) -> bool:
        """检查位置是否在网格范围内"""
        return (0 <= pos[0] < grid_shape[0] and 
                0 <= pos[1] < grid_shape[1])
    
    def _is_occupied(self, pos: Tuple[int, int], occupancy_grid: np.ndarray) -> bool:
        """检查位置是否被占用（包含安全边距）"""
        x, y = pos
        
        # 检查安全边距范围内是否有障碍物
        for dx in range(-self.safety_margin, self.safety_margin + 1):
            for dy in range(-self.safety_margin, self.safety_margin + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < occupancy_grid.shape[0] and 
                    0 <= check_y < occupancy_grid.shape[1]):
                    if occupancy_grid[check_x, check_y] > self.occupancy_threshold:
                        return True
        return False
    
    def _world_to_grid(self, world_pos: Tuple[float, float], 
                       grid_origin: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        x = int((world_pos[0] - grid_origin[0]) / self.grid_resolution)
        y = int((world_pos[1] - grid_origin[1]) / self.grid_resolution)
        return (x, y)
    
    def _grid_to_world(self, grid_pos: Tuple[int, int], 
                       grid_origin: Tuple[float, float]) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x = grid_pos[0] * self.grid_resolution + grid_origin[0]
        y = grid_pos[1] * self.grid_resolution + grid_origin[1]
        return (x, y)
    
    def _smooth_path(self, path: List[Tuple[float, float]], 
                     occupancy_grid: np.ndarray, 
                     grid_origin: Tuple[float, float]) -> List[Tuple[float, float]]:
        """路径平滑 - 移除不必要的中间点"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            
            # 从终点开始，找到最远的可直接到达的点
            while j > i + 1:
                if self._is_line_free(path[i], path[j], occupancy_grid, grid_origin):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _is_line_free(self, start: Tuple[float, float], end: Tuple[float, float],
                      occupancy_grid: np.ndarray, grid_origin: Tuple[float, float]) -> bool:
        """检查两点间直线是否无障碍"""
        start_grid = self._world_to_grid(start, grid_origin)
        end_grid = self._world_to_grid(end, grid_origin)
        
        # 使用Bresenham算法获取直线上的所有网格点
        line_points = self._bresenham_line(start_grid, end_grid)
        
        for point in line_points:
            if not self._is_valid_position(point, occupancy_grid.shape):
                return False
            if self._is_occupied(point, occupancy_grid):
                return False
        
        return True
    
    def _bresenham_line(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham直线算法"""
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
    
    def visualize_path(self, occupancy_grid: np.ndarray, path: List[Tuple[float, float]], 
                       start_pos: Tuple[float, float], goal_pos: Tuple[float, float],
                       grid_origin: Tuple[float, float] = (-50.0, -50.0)):
        """可视化路径规划结果"""
        plt.figure(figsize=(12, 10))
        
        # 显示占用网格
        extent = [grid_origin[0], grid_origin[0] + occupancy_grid.shape[1] * self.grid_resolution,
                  grid_origin[1], grid_origin[1] + occupancy_grid.shape[0] * self.grid_resolution]
        
        plt.imshow(occupancy_grid, cmap='gray_r', extent=extent, origin='lower')
        
        # 显示路径
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=3, label='规划路径')
            plt.plot(path_x, path_y, 'bo', markersize=4)
        
        # 显示起点和终点
        plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='起点')
        plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='终点')
        
        plt.xlabel('X (米)')
        plt.ylabel('Y (米)')
        plt.title('全局路径规划')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()