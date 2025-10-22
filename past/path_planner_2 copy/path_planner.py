"""
路径规划模块
"""
import heapq
import math
import cv2
import numpy as np
from typing import List, Tuple

from config import PathPlannerConfig

class AStarPlanner:
    """A* 路径规划算法"""
    
    def __init__(self, config: PathPlannerConfig):
        self.config = config
    
    def plan_path(self, grid: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """使用 A* 算法规划路径"""
        if not self._is_valid_point(grid, start) or not self._is_valid_point(grid, goal):
            print(f"[PathPlanner] 起始点 {start} 或目标点 {goal} 无效")
            return []
        
        # 膨胀障碍物
        inflated_grid = self._inflate_obstacles(grid)
        
        # A* 搜索
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        # 8连通方向
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        
        nodes_explored = 0
        while open_set and nodes_explored < 8000:
            current = heapq.heappop(open_set)[1]
            nodes_explored += 1
            
            if current == goal:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return self._smooth_path(path, inflated_grid)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_point(inflated_grid, neighbor):
                    continue
                
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print("[PathPlanner] 未找到可行路径")
        return []
    
    def _is_valid_point(self, grid: np.ndarray, point: Tuple[int, int]) -> bool:
        """检查点是否有效"""
        x, y = point
        if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
            return False
        return grid[y, x] != 255
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """欧几里得距离启发式"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _inflate_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """膨胀障碍物"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                           (2*self.config.obstacle_inflation+1, 
                                            2*self.config.obstacle_inflation+1))
        obstacle_mask = (grid == 255).astype(np.uint8)
        inflated_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
        
        inflated = grid.copy()
        inflated[inflated_mask == 1] = 255
        return inflated
    
    def _smooth_path(self, path: List[Tuple[int, int]], grid: np.ndarray) -> List[Tuple[int, int]]:
        """路径平滑处理"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self._is_line_clear(grid, path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    smoothed.append(path[i])
        
        return smoothed
    
    def _is_line_clear(self, grid: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """检查直线路径是否无障碍"""
        x0, y0 = start
        x1, y1 = end
        
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx - dy
        
        x, y = x0, y0
        while True:
            if grid[y, x] == 255:
                return False
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True