"""
智能路径规划器 - 深度集成SLAM系统
支持动态避障、自适应规划和质量评估
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple, Callable, Dict, Any
from collections import deque
import heapq
import time
import threading
import logging
from dataclasses import dataclass

from config import PathPlannerConfig, GridConfig

logger = logging.getLogger(__name__)

@dataclass
class PathQualityMetrics:
    """路径质量指标"""
    path_length: float = 0.0
    smoothness: float = 0.0
    safety: float = 0.0
    clearance: float = 0.0
    execution_time: float = 0.0
    overall_score: float = 0.0

class Node:
    """增强的A*算法节点"""
    
    def __init__(self, position: Tuple[int, int], g_cost: float = 0, h_cost: float = 0, 
                parent=None, clearance: float = 0.0):
        self.position = position
        self.g_cost = g_cost  # 从起点到当前节点的实际代价
        self.h_cost = h_cost  # 从当前节点到终点的启发式代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = parent
        self.clearance = clearance  # 到最近障碍物的距离
    
    def __lt__(self, other):
        # 优先考虑f_cost，然后考虑clearance
        if abs(self.f_cost - other.f_cost) < 0.01:
            return self.clearance > other.clearance
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.position == other.position

class IntelligentPathPlanner:
    """智能路径规划器 - 深度集成SLAM"""
    
    def __init__(self, planner_config: PathPlannerConfig, grid_config: GridConfig, 
                slam_processor=None):
        self.planner_config = planner_config
        self.grid_config = grid_config
        self.slam_processor = slam_processor
        
        # 规划参数
        self.safety_margin = planner_config.safety_margin
        self.obstacle_inflation = planner_config.obstacle_inflation
        self.goal_tolerance = planner_config.goal_tolerance
        
        # 运动参数
        self.max_velocity = planner_config.max_velocity
        self.max_angular_velocity = planner_config.max_angular_velocity
        self.lookahead_distance = planner_config.lookahead_distance
        
        # 增强参数
        self.dynamic_window_size = getattr(planner_config, 'dynamic_window_size', 3.0)
        self.path_smoothing_weight = getattr(planner_config, 'path_smoothing_weight', 0.3)
        self.clearance_weight = getattr(planner_config, 'clearance_weight', 0.2)
        self.adaptive_planning = getattr(planner_config, 'adaptive_planning', True)
        
        # A*搜索方向（8连通 + 16连通可选）
        self.directions_8 = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        self.directions_16 = self.directions_8 + [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        self.use_extended_search = getattr(planner_config, 'use_extended_search', False)
        self.directions = self.directions_16 if self.use_extended_search else self.directions_8
        
        # 移动代价
        self.diagonal_cost = np.sqrt(2)
        self.straight_cost = 1.0
        self.extended_cost = np.sqrt(5)  # 对于(2,1)类移动
        
        # 路径管理
        self.current_path: List[np.ndarray] = []
        self.path_index = 0
        self.last_replan_time = 0
        self.replan_interval = getattr(planner_config, 'replan_interval', 2.0)
        
        # 动态避障
        self.obstacle_memory = deque(maxlen=100)
        self.dynamic_obstacles: List[np.ndarray] = []
        self.last_occupancy_grid: Optional[np.ndarray] = None
        
        # 质量评估
        self.path_quality = PathQualityMetrics()
        self.quality_history = deque(maxlen=50)
        
        # 统计信息
        self.planning_stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'replans': 0,
            'average_planning_time': 0.0,
            'average_path_length': 0.0,
            'average_quality_score': 0.0,
            'dynamic_avoidance_count': 0
        }
        
        # 回调函数
        self.path_update_callbacks: List[Callable] = []
        self.replanning_callbacks: List[Callable] = []
        
        # 线程安全
        self.planning_lock = threading.RLock()
        
        logger.info(f"[IntelligentPathPlanner] 智能路径规划器初始化完成")
        logger.info(f"[IntelligentPathPlanner] 安全边距: {self.safety_margin}m, 目标容差: {self.goal_tolerance}m")
        logger.info(f"[IntelligentPathPlanner] 自适应规划: {self.adaptive_planning}, 扩展搜索: {self.use_extended_search}")
    
    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                 occupancy_grid: np.ndarray, current_pose: Optional[np.ndarray] = None) -> Optional[List[np.ndarray]]:
        """
        智能路径规划 - 集成SLAM信息
        
        Args:
            start_pos: 起点位置 (x, y) 世界坐标
            goal_pos: 终点位置 (x, y) 世界坐标
            occupancy_grid: 占用网格
            current_pose: 当前机器人位姿（可选）
            
        Returns:
            路径点列表，如果规划失败返回None
        """
        with self.planning_lock:
            start_time = time.perf_counter()
            self.planning_stats['total_plans'] += 1
            
            # 输入验证
            if not self._validate_inputs(start_pos, goal_pos, occupancy_grid):
                self.planning_stats['failed_plans'] += 1
                return None
            
            # 世界坐标转网格坐标
            start_grid = self._world_to_grid(start_pos, occupancy_grid.shape)
            goal_grid = self._world_to_grid(goal_pos, occupancy_grid.shape)
            
            # 验证起点和终点
            if not self._is_valid_point(start_grid, occupancy_grid):
                logger.warning(f"[IntelligentPathPlanner] 无效起点: {start_grid}")
                self.planning_stats['failed_plans'] += 1
                return None
            
            if not self._is_valid_point(goal_grid, occupancy_grid):
                logger.warning(f"[IntelligentPathPlanner] 无效终点: {goal_grid}")
                # 尝试寻找最近的有效终点
                goal_grid = self._find_nearest_valid_point(goal_grid, occupancy_grid)
                if goal_grid is None:
                    self.planning_stats['failed_plans'] += 1
                    return None
            
            # 动态障碍物检测
            self._detect_dynamic_obstacles(occupancy_grid)
            
            # 智能障碍物膨胀
            inflated_grid = self._adaptive_obstacle_inflation(occupancy_grid, current_pose)
            
            # 多策略A*路径搜索
            grid_path = self._multi_strategy_astar(start_grid, goal_grid, inflated_grid, occupancy_grid)
            
            if grid_path is None:
                logger.warning(f"[IntelligentPathPlanner] 路径搜索失败")
                self.planning_stats['failed_plans'] += 1
                return None
            
            # 网格坐标转世界坐标
            world_path = [self._grid_to_world(point, occupancy_grid.shape) for point in grid_path]
            
            # 智能路径优化
            optimized_path = self._intelligent_path_optimization(world_path, inflated_grid, occupancy_grid)
            
            # 路径质量评估
            self.path_quality = self._evaluate_path_quality(optimized_path, occupancy_grid)
            self.quality_history.append(self.path_quality.overall_score)
            
            # 更新统计信息
            planning_time = time.perf_counter() - start_time
            self._update_planning_statistics(planning_time, len(optimized_path), True)
            
            # 存储路径用于跟踪
            self.current_path = optimized_path
            self.path_index = 0
            self.last_occupancy_grid = occupancy_grid.copy()
            
            # 触发回调
            self._trigger_path_update_callbacks(optimized_path, self.path_quality)
            
            logger.info(f"[IntelligentPathPlanner] 路径规划成功: {len(optimized_path)}点, "
                       f"用时{planning_time*1000:.1f}ms, 质量评分: {self.path_quality.overall_score:.2f}")
            
            return [np.array(point) for point in optimized_path]
    
    def _validate_inputs(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                        occupancy_grid: np.ndarray) -> bool:
        """输入验证"""
        if not isinstance(start_pos, np.ndarray) or start_pos.shape != (2,):
            logger.error("[IntelligentPathPlanner] 无效起点格式")
            return False
        
        if not isinstance(goal_pos, np.ndarray) or goal_pos.shape != (2,):
            logger.error("[IntelligentPathPlanner] 无效终点格式")
            return False
        
        if not isinstance(occupancy_grid, np.ndarray) or occupancy_grid.ndim != 2:
            logger.error("[IntelligentPathPlanner] 无效占用网格格式")
            return False
        
        # 检查距离是否合理
        distance = np.linalg.norm(goal_pos - start_pos)
        if distance > self.grid_config.grid_size * 0.8:
            logger.warning(f"[IntelligentPathPlanner] 目标距离过远: {distance:.2f}m")
            return False
        
        return True
    
    def _find_nearest_valid_point(self, point: Tuple[int, int], 
                                 occupancy_grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """寻找最近的有效点"""
        max_search_radius = 20
        
        for radius in range(1, max_search_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        candidate = (point[0] + dy, point[1] + dx)
                        if self._is_valid_point(candidate, occupancy_grid):
                            logger.info(f"[IntelligentPathPlanner] 找到替代终点: {candidate}")
                            return candidate
        
        return None
    
    def _detect_dynamic_obstacles(self, current_grid: np.ndarray):
        """检测动态障碍物"""
        if self.last_occupancy_grid is None:
            return
        
        # 计算网格差异
        diff = np.abs(current_grid.astype(np.float32) - self.last_occupancy_grid.astype(np.float32))
        dynamic_threshold = 0.3
        
        # 寻找显著变化的区域
        dynamic_mask = diff > dynamic_threshold
        if np.any(dynamic_mask):
            # 找到动态障碍物位置
            dynamic_coords = np.where(dynamic_mask)
            for i in range(len(dynamic_coords[0])):
                obstacle_pos = self._grid_to_world((dynamic_coords[0][i], dynamic_coords[1][i]), 
                                                 current_grid.shape)
                self.obstacle_memory.append((obstacle_pos, time.time()))
            
            self.planning_stats['dynamic_avoidance_count'] += 1
            logger.debug(f"[IntelligentPathPlanner] 检测到动态障碍物: {len(dynamic_coords[0])}个")
    
    def _adaptive_obstacle_inflation(self, occupancy_grid: np.ndarray, 
                                   current_pose: Optional[np.ndarray] = None) -> np.ndarray:
        """自适应障碍物膨胀"""
        # 基础膨胀
        binary_grid = (occupancy_grid > self.grid_config.hit_threshold).astype(np.uint8)
        
        # 根据运动状态调整膨胀大小
        base_inflation = self.obstacle_inflation
        
        if self.slam_processor is not None:
            try:
                slam_stats = self.slam_processor.get_comprehensive_statistics()
                # 根据SLAM质量调整膨胀
                slam_quality = slam_stats.get('slam_processor', {}).get('average_quality', 0.5)
                if slam_quality < 0.3:  # SLAM质量较差时增加安全边距
                    base_inflation *= 1.5
                elif slam_quality > 0.8:  # SLAM质量较好时可以减少膨胀
                    base_inflation *= 0.8
            except Exception as e:
                logger.debug(f"[IntelligentPathPlanner] SLAM状态获取失败: {e}")
        
        # 计算膨胀核
        inflation_pixels = max(1, int(base_inflation / self.grid_config.resolution))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (2*inflation_pixels+1, 2*inflation_pixels+1))
        
        # 执行膨胀
        inflated_binary = cv2.dilate(binary_grid, kernel, iterations=1)
        
        # 转换回概率网格
        inflated_grid = occupancy_grid.copy()
        inflated_grid[inflated_binary > 0] = 1.0
        
        # 添加动态障碍物影响
        current_time = time.time()
        for obstacle_pos, detection_time in self.obstacle_memory:
            if current_time - detection_time < 5.0:  # 5秒内的动态障碍物
                obs_grid = self._world_to_grid(obstacle_pos, occupancy_grid.shape)
                self._add_circular_obstacle(inflated_grid, obs_grid, inflation_pixels + 2)
        
        return inflated_grid
    
    def _add_circular_obstacle(self, grid: np.ndarray, center: Tuple[int, int], radius: int):
        """在网格中添加圆形障碍物"""
        height, width = grid.shape
        cy, cx = center
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    y, x = cy + dy, cx + dx
                    if 0 <= y < height and 0 <= x < width:
                        grid[y, x] = 1.0
    
    def _multi_strategy_astar(self, start: Tuple[int, int], goal: Tuple[int, int],
                             inflated_grid: np.ndarray, original_grid: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """多策略A*搜索"""
        strategies = [
            {'weight_clearance': 0.2, 'weight_smoothness': 0.1, 'heuristic_scale': 1.0},
            {'weight_clearance': 0.5, 'weight_smoothness': 0.0, 'heuristic_scale': 1.2},  # 更安全
            {'weight_clearance': 0.0, 'weight_smoothness': 0.3, 'heuristic_scale': 0.8},  # 更直接
        ]
        
        best_path = None
        best_score = float('-inf')
        
        for i, strategy in enumerate(strategies):
            path = self._astar_search_enhanced(start, goal, inflated_grid, original_grid, strategy)
            if path is not None:
                # 快速评估路径质量
                score = self._quick_path_score(path, original_grid)
                if score > best_score:
                    best_score = score
                    best_path = path
                
                # 如果第一个策略成功且质量不错，就使用它
                if i == 0 and score > 0.7:
                    break
        
        return best_path
    
    def _astar_search_enhanced(self, start: Tuple[int, int], goal: Tuple[int, int],
                              inflated_grid: np.ndarray, original_grid: np.ndarray,
                              strategy: Dict[str, float]) -> Optional[List[Tuple[int, int]]]:
        """增强版A*搜索"""
        open_list = []
        closed_set = set()
        
        start_clearance = self._calculate_clearance(start, original_grid)
        start_node = Node(start, 0, self._enhanced_heuristic(start, goal, strategy), clearance=start_clearance)
        heapq.heappush(open_list, start_node)
        
        nodes = {start: start_node}
        max_iterations = min(inflated_grid.size, 50000)  # 限制搜索规模
        iterations = 0
        
        while open_list and iterations < max_iterations:
            iterations += 1
            
            current_node = heapq.heappop(open_list)
            current_pos = current_node.position
            
            # 到达目标
            if current_pos == goal:
                logger.debug(f"[IntelligentPathPlanner] A*搜索成功，迭代: {iterations}")
                return self._reconstruct_path(current_node)
            
            closed_set.add(current_pos)
            
            # 搜索邻居
            for direction in self.directions:
                neighbor_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                
                if (neighbor_pos in closed_set or 
                    not self._is_valid_point(neighbor_pos, inflated_grid)):
                    continue
                
                # 计算移动代价
                move_cost = self._calculate_move_cost(direction, current_node, strategy, original_grid)
                tentative_g = current_node.g_cost + move_cost
                
                # 检查是否找到更好的路径
                if neighbor_pos in nodes:
                    neighbor_node = nodes[neighbor_pos]
                    if tentative_g < neighbor_node.g_cost:
                        neighbor_node.g_cost = tentative_g
                        neighbor_node.f_cost = tentative_g + neighbor_node.h_cost
                        neighbor_node.parent = current_node
                else:
                    neighbor_clearance = self._calculate_clearance(neighbor_pos, original_grid)
                    neighbor_node = Node(
                        neighbor_pos,
                        tentative_g,
                        self._enhanced_heuristic(neighbor_pos, goal, strategy),
                        current_node,
                        neighbor_clearance
                    )
                    nodes[neighbor_pos] = neighbor_node
                    heapq.heappush(open_list, neighbor_node)
        
        logger.warning(f"[IntelligentPathPlanner] A*搜索失败，迭代: {iterations}")
        return None
    
    def _calculate_move_cost(self, direction: Tuple[int, int], current_node: Node,
                            strategy: Dict[str, float], original_grid: np.ndarray) -> float:
        """计算移动代价"""
        # 基础移动代价
        abs_sum = abs(direction[0]) + abs(direction[1])
        if abs_sum == 2:  # 对角线
            base_cost = self.diagonal_cost
        elif abs_sum == 3:  # 扩展移动如(2,1)
            base_cost = self.extended_cost
        else:  # 直线
            base_cost = self.straight_cost
        
        # 安全性代价（基于clearance）
        clearance_cost = 0.0
        if strategy['weight_clearance'] > 0:
            clearance_cost = strategy['weight_clearance'] * (1.0 / (current_node.clearance + 0.1))
        
        # 平滑性代价
        smoothness_cost = 0.0
        if strategy['weight_smoothness'] > 0 and current_node.parent is not None:
            # 计算转角
            prev_dir = (current_node.position[0] - current_node.parent.position[0],
                       current_node.position[1] - current_node.parent.position[1])
            angle_change = abs(np.arctan2(direction[1], direction[0]) - np.arctan2(prev_dir[1], prev_dir[0]))
            smoothness_cost = strategy['weight_smoothness'] * angle_change
        
        return base_cost + clearance_cost + smoothness_cost
    
    def _enhanced_heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int],
                           strategy: Dict[str, float]) -> float:
        """增强启发式函数"""
        # 基础欧几里得距离
        dx = abs(pos[0] - goal[0])
        dy = abs(pos[1] - goal[1])
        base_heuristic = np.sqrt(dx*dx + dy*dy)
        
        return base_heuristic * strategy['heuristic_scale']
    
    def _calculate_clearance(self, pos: Tuple[int, int], occupancy_grid: np.ndarray) -> float:
        """计算到最近障碍物的距离"""
        max_check_radius = 10
        row, col = pos
        height, width = occupancy_grid.shape
        
        for radius in range(1, max_check_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        check_row, check_col = row + dy, col + dx
                        if (0 <= check_row < height and 0 <= check_col < width and
                            occupancy_grid[check_row, check_col] > self.grid_config.hit_threshold):
                            return float(radius)
        
        return float(max_check_radius)
    
    def _quick_path_score(self, path: List[Tuple[int, int]], occupancy_grid: np.ndarray) -> float:
        """快速路径质量评分"""
        if len(path) < 2:
            return 0.0
        
        # 路径长度评分（越短越好）
        length_score = max(0, 1.0 - len(path) / 1000.0)
        
        # 安全性评分
        safety_score = 0.0
        for pos in path[::5]:  # 采样检查
            clearance = self._calculate_clearance(pos, occupancy_grid)
            safety_score += min(1.0, clearance / 5.0)
        safety_score /= max(1, len(path) // 5)
        
        # 平滑性评分
        smoothness_score = 1.0
        if len(path) > 2:
            angles = []
            for i in range(1, len(path) - 1):
                v1 = np.array(path[i]) - np.array(path[i-1])
                v2 = np.array(path[i+1]) - np.array(path[i])
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                    angles.append(angle)
            if angles:
                smoothness_score = 1.0 - np.std(angles) / np.pi
        
        return (length_score * 0.3 + safety_score * 0.5 + smoothness_score * 0.2)
    
    def _intelligent_path_optimization(self, path: List[np.ndarray], inflated_grid: np.ndarray,
                                      original_grid: np.ndarray) -> List[np.ndarray]:
        """智能路径优化"""
        if len(path) <= 2:
            return path
        
        # 多轮优化
        optimized = path.copy()
        
        # 1. 基础平滑（直线简化）
        optimized = self._smooth_path_basic(optimized, inflated_grid)
        
        # 2. 曲线平滑
        if self.path_smoothing_weight > 0:
            optimized = self._smooth_path_curve(optimized, self.path_smoothing_weight)
        
        # 3. 安全性检查和修正
        optimized = self._ensure_path_safety(optimized, inflated_grid, original_grid)
        
        return optimized
    
    def _smooth_path_basic(self, path: List[np.ndarray], occupancy_grid: np.ndarray) -> List[np.ndarray]:
        """基础路径平滑（直线简化）"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            farthest = i + 1
            for j in range(i + 2, len(path)):
                if self._is_line_clear(path[i], path[j], occupancy_grid):
                    farthest = j
                else:
                    break
            smoothed.append(path[farthest])
            i = farthest
        
        return smoothed
    
    def _smooth_path_curve(self, path: List[np.ndarray], weight: float) -> List[np.ndarray]:
        """曲线平滑"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        
        for i in range(1, len(path) - 1):
            # 加权平均平滑
            smoothed_point = (1 - weight) * path[i] + weight * 0.5 * (path[i-1] + path[i+1])
            smoothed.append(smoothed_point)
        
        smoothed.append(path[-1])
        return smoothed
    
    def _ensure_path_safety(self, path: List[np.ndarray], inflated_grid: np.ndarray,
                           original_grid: np.ndarray) -> List[np.ndarray]:
        """确保路径安全性"""
        safe_path = []
        
        for point in path:
            grid_point = self._world_to_grid(point, inflated_grid.shape)
            
            if self._is_valid_point(grid_point, inflated_grid):
                safe_path.append(point)
            else:
                # 寻找附近的安全点
                safe_point = self._find_nearby_safe_point(point, inflated_grid, original_grid)
                if safe_point is not None:
                    safe_path.append(safe_point)
                else:
                    logger.warning(f"[IntelligentPathPlanner] 无法找到安全替代点: {point}")
                    safe_path.append(point)  # 保留原点，但标记风险
        
        return safe_path
    
    def _find_nearby_safe_point(self, world_point: np.ndarray, inflated_grid: np.ndarray,
                               original_grid: np.ndarray) -> Optional[np.ndarray]:
        """寻找附近的安全点"""
        grid_point = self._world_to_grid(world_point, inflated_grid.shape)
        search_radius = 5
        
        best_point = None
        best_clearance = 0
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                candidate = (grid_point[0] + dy, grid_point[1] + dx)
                if self._is_valid_point(candidate, inflated_grid):
                    clearance = self._calculate_clearance(candidate, original_grid)
                    if clearance > best_clearance:
                        best_clearance = clearance
                        best_point = candidate
        
        if best_point is not None:
            return self._grid_to_world(best_point, inflated_grid.shape)
        return None
    
    def _evaluate_path_quality(self, path: List[np.ndarray], occupancy_grid: np.ndarray) -> PathQualityMetrics:
        """评估路径质量"""
        if len(path) < 2:
            return PathQualityMetrics()
        
        metrics = PathQualityMetrics()
        
        # 路径长度
        total_length = 0
        for i in range(1, len(path)):
            total_length += np.linalg.norm(path[i] - path[i-1])
        metrics.path_length = total_length
        
        # 平滑性（角度变化的标准差）
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                angles.append(angle)
        
        if angles:
            metrics.smoothness = 1.0 - np.std(angles) / np.pi
        else:
            metrics.smoothness = 1.0
        
        # 安全性（平均clearance）
        clearances = []
        for point in path[::3]:  # 采样检查
            grid_point = self._world_to_grid(point, occupancy_grid.shape)
            clearance = self._calculate_clearance(grid_point, occupancy_grid)
            clearances.append(clearance)
        
        if clearances:
            avg_clearance = np.mean(clearances)
            metrics.clearance = min(1.0, avg_clearance / 5.0)
            metrics.safety = metrics.clearance
        
        # 综合评分
        metrics.overall_score = (
            0.2 * min(1.0, 50.0 / metrics.path_length) +  # 长度评分
            0.3 * metrics.smoothness +                      # 平滑性评分
            0.5 * metrics.safety                           # 安全性评分
        )
        
        return metrics
    
    def follow_path_adaptive(self, current_pos: np.ndarray, current_pose: np.ndarray,
                            occupancy_grid: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        自适应路径跟踪 - 集成动态重规划
        
        Args:
            current_pos: 当前位置 (x, y)
            current_pose: 当前完整位姿 (4x4矩阵)
            occupancy_grid: 当前占用网格
            
        Returns:
            控制命令 (linear_vel, angular_vel) 或 None（到达目标或需要重规划）
        """
        if not self.current_path:
            return None
        
        current_time = time.time()
        
        # 检查是否需要重规划
        if self._should_replan(current_pos, occupancy_grid, current_time):
            logger.info("[IntelligentPathPlanner] 触发自适应重规划")
            self.planning_stats['replans'] += 1
            self._trigger_replanning_callbacks(current_pos, "adaptive_replan")
            return None
        
        # 检查是否到达目标
        goal_distance = np.linalg.norm(current_pos - self.current_path[-1])
        if goal_distance < self.goal_tolerance:
            logger.info(f"[IntelligentPathPlanner] 到达目标，距离: {goal_distance:.2f}m")
            return None
        
        # 更新路径索引
        self._update_path_index(current_pos)
        
        # 寻找自适应前瞻点
        lookahead_point = self._find_adaptive_lookahead_point(current_pos, current_pose)
        if lookahead_point is None:
            logger.warning("[IntelligentPathPlanner] 无法找到前瞻点")
            return None
        
        # 计算控制命令
        return self._calculate_control_command(current_pos, current_pose, lookahead_point, occupancy_grid)
    
    def _should_replan(self, current_pos: np.ndarray, occupancy_grid: np.ndarray, 
                      current_time: float) -> bool:
        """判断是否需要重规划"""
        # 时间间隔检查
        if current_time - self.last_replan_time < self.replan_interval:
            return False
        
        # 路径阻塞检查
        if self._is_path_blocked(current_pos, occupancy_grid):
            self.last_replan_time = current_time
            return True
        
        # 偏离路径检查
        if self._is_significantly_off_path(current_pos):
            self.last_replan_time = current_time
            return True
        
        # 地图变化检查
        if self._has_significant_map_change(occupancy_grid):
            self.last_replan_time = current_time
            return True
        
        return False
    
    def _is_path_blocked(self, current_pos: np.ndarray, occupancy_grid: np.ndarray) -> bool:
        """检查前方路径是否被阻塞"""
        if self.path_index >= len(self.current_path) - 1:
            return False
        
        # 检查前方几个路径点
        check_points = min(5, len(self.current_path) - self.path_index)
        for i in range(self.path_index, self.path_index + check_points):
            point = self.current_path[i]
            grid_point = self._world_to_grid(point, occupancy_grid.shape)
            if not self._is_valid_point(grid_point, occupancy_grid):
                return True
        
        return False
    
    def _is_significantly_off_path(self, current_pos: np.ndarray) -> bool:
        """检查是否显著偏离路径"""
        if self.path_index >= len(self.current_path):
            return False
        
        # 计算到最近路径点的距离
        min_distance = float('inf')
        for i in range(max(0, self.path_index - 2), min(len(self.current_path), self.path_index + 3)):
            distance = np.linalg.norm(current_pos - self.current_path[i])
            min_distance = min(min_distance, distance)
        
        return min_distance > self.safety_margin * 2
    
    def _has_significant_map_change(self, current_grid: np.ndarray) -> bool:
        """检查地图是否有显著变化"""
        if self.last_occupancy_grid is None:
            return False
        
        # 简单的变化检测
        diff = np.abs(current_grid.astype(np.float32) - self.last_occupancy_grid.astype(np.float32))
        change_ratio = np.sum(diff > 0.3) / diff.size
        return change_ratio > 0.05  # 5%的区域发生变化
    
    def _update_path_index(self, current_pos: np.ndarray):
        """更新路径索引"""
        if not self.current_path:
            return
        
        # 寻找最近的路径点
        min_distance = float('inf')
        closest_index = self.path_index
        
        # 只在当前索引附近搜索
        search_start = max(0, self.path_index - 2)
        search_end = min(len(self.current_path), self.path_index + 5)
        
        for i in range(search_start, search_end):
            distance = np.linalg.norm(current_pos - self.current_path[i])
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 向前推进索引
        self.path_index = max(self.path_index, closest_index)
    
    def _find_adaptive_lookahead_point(self, current_pos: np.ndarray, 
                                      current_pose: np.ndarray) -> Optional[np.ndarray]:
        """寻找自适应前瞻点"""
        if self.path_index >= len(self.current_path):
            return self.current_path[-1] if self.current_path else None
        
        # 根据速度调整前瞻距离
        adaptive_lookahead = self.lookahead_distance
        
        # 如果有SLAM信息，根据定位质量调整
        if self.slam_processor is not None:
            try:
                slam_stats = self.slam_processor.get_comprehensive_statistics()
                quality = slam_stats.get('slam_processor', {}).get('average_quality', 0.5)
                if quality < 0.3:
                    adaptive_lookahead *= 0.7  # 质量差时减少前瞻距离
                elif quality > 0.8:
                    adaptive_lookahead *= 1.3  # 质量好时增加前瞻距离
            except:
                pass
        
        # 寻找前瞻点
        for i in range(self.path_index, len(self.current_path)):
            distance = np.linalg.norm(current_pos - self.current_path[i])
            if distance >= adaptive_lookahead:
                return self.current_path[i]
        
        return self.current_path[-1]
    
    def _calculate_control_command(self, current_pos: np.ndarray, current_pose: np.ndarray,
                                  lookahead_point: np.ndarray, occupancy_grid: np.ndarray) -> Tuple[float, float]:
        """计算控制命令"""
        # 提取当前朝向
        current_angle = np.arctan2(current_pose[1, 0], current_pose[0, 0])
        
        # 计算到前瞻点的方向
        direction_vec = lookahead_point - current_pos
        distance_to_lookahead = np.linalg.norm(direction_vec)
        
        if distance_to_lookahead < 0.01:
            return (0.0, 0.0)
        
        # 计算期望角度
        desired_angle = np.arctan2(direction_vec[1], direction_vec[0])
        angle_error = desired_angle - current_angle
        
        # 角度归一化
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi
        
        # Pure Pursuit控制律
        curvature = 2 * np.sin(angle_error) / distance_to_lookahead
        
        # 自适应速度控制
        base_velocity = self.max_velocity
        
        # 根据角度误差调整速度
        if abs(angle_error) > np.pi / 4:  # 大角度转弯时减速
            base_velocity *= 0.5
        elif abs(angle_error) > np.pi / 6:
            base_velocity *= 0.7
        
        # 根据前方障碍物调整速度
        obstacle_factor = self._check_forward_obstacles(current_pos, current_angle, occupancy_grid)
        base_velocity *= obstacle_factor
        
        # 计算最终控制命令
        linear_vel = min(base_velocity, distance_to_lookahead * 0.8)
        angular_vel = curvature * linear_vel
        
        # 限制角速度
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        return (max(0.0, linear_vel), angular_vel)
    
    def _check_forward_obstacles(self, current_pos: np.ndarray, current_angle: float,
                                occupancy_grid: np.ndarray) -> float:
        """检查前方障碍物并返回速度因子"""
        check_distance = 2.0  # 检查前方2米
        check_steps = 10
        
        min_clearance = float('inf')
        
        for i in range(1, check_steps + 1):
            check_pos = current_pos + (check_distance * i / check_steps) * np.array([np.cos(current_angle), np.sin(current_angle)])
            grid_pos = self._world_to_grid(check_pos, occupancy_grid.shape)
            
            if self._is_valid_point(grid_pos, occupancy_grid):
                clearance = self._calculate_clearance(grid_pos, occupancy_grid)
                min_clearance = min(min_clearance, clearance)
            else:
                return 0.2  # 前方有障碍物，大幅减速
        
        # 根据最小间隙返回速度因子
        if min_clearance < 2:
            return 0.3
        elif min_clearance < 4:
            return 0.6
        else:
            return 1.0
    
    # ---------------------------------------------------------------------------
    # 辅助方法
    # ---------------------------------------------------------------------------
    
    def _world_to_grid(self, world_pos: np.ndarray, grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        resolution = self.grid_config.resolution
        grid_center = np.array(grid_shape) // 2
        
        grid_x = int(world_pos[0] / resolution + grid_center[1])
        grid_y = int(-world_pos[1] / resolution + grid_center[0])
        
        return (grid_y, grid_x)
    
    def _grid_to_world(self, grid_pos: Tuple[int, int], grid_shape: Tuple[int, int]) -> np.ndarray:
        """网格坐标转世界坐标"""
        resolution = self.grid_config.resolution
        grid_center = np.array(grid_shape) // 2
        
        world_x = (grid_pos[1] - grid_center[1]) * resolution
        world_y = -(grid_pos[0] - grid_center[0]) * resolution
        
        return np.array([world_x, world_y])
    
    def _is_valid_point(self, point: Tuple[int, int], occupancy_grid: np.ndarray) -> bool:
        """检查点是否有效"""
        row, col = point
        height, width = occupancy_grid.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return occupancy_grid[row, col] < self.grid_config.hit_threshold
    
    def _is_line_clear(self, start: np.ndarray, end: np.ndarray, occupancy_grid: np.ndarray) -> bool:
        """检查两点间直线是否无障碍"""
        start_grid = self._world_to_grid(start, occupancy_grid.shape)
        end_grid = self._world_to_grid(end, occupancy_grid.shape)
        
        points = self._bresenham_line(start_grid[1], start_grid[0], end_grid[1], end_grid[0])
        
        for point in points:
            grid_point = (point[1], point[0])
            if not self._is_valid_point(grid_point, occupancy_grid):
                return False
        
        return True
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法"""
        points = []
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
    
    def _reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """重构路径"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current.position)
            current = current.parent
        
        return path[::-1]
    
    def _update_planning_statistics(self, planning_time: float, path_length: int, success: bool):
        """更新规划统计"""
        if success:
            self.planning_stats['successful_plans'] += 1
            
            # 更新平均规划时间
            total_plans = self.planning_stats['total_plans']
            current_avg = self.planning_stats['average_planning_time']
            self.planning_stats['average_planning_time'] = (
                (current_avg * (total_plans - 1) + planning_time) / total_plans
            )
            
            # 更新平均路径长度
            current_avg_length = self.planning_stats['average_path_length']
            self.planning_stats['average_path_length'] = (
                (current_avg_length * (self.planning_stats['successful_plans'] - 1) + path_length) /
                self.planning_stats['successful_plans']
            )
        
        # 更新质量评分
        if self.quality_history:
            self.planning_stats['average_quality_score'] = np.mean(self.quality_history)
    
    def _trigger_path_update_callbacks(self, path: List[np.ndarray], quality: PathQualityMetrics):
        """触发路径更新回调"""
        for callback in self.path_update_callbacks:
            try:
                callback(path, quality)
            except Exception as e:
                logger.warning(f"[IntelligentPathPlanner] 路径更新回调失败: {e}")
    
    def _trigger_replanning_callbacks(self, current_pos: np.ndarray, reason: str):
        """触发重规划回调"""
        for callback in self.replanning_callbacks:
            try:
                callback(current_pos, reason)
            except Exception as e:
                logger.warning(f"[IntelligentPathPlanner] 重规划回调失败: {e}")
    
    # ---------------------------------------------------------------------------
    # 公共接口
    # ---------------------------------------------------------------------------
    
    def register_path_update_callback(self, callback: Callable[[List[np.ndarray], PathQualityMetrics], None]):
        """注册路径更新回调"""
        self.path_update_callbacks.append(callback)
    
    def register_replanning_callback(self, callback: Callable[[np.ndarray, str], None]):
        """注册重规划回调"""
        self.replanning_callbacks.append(callback)
    
    def get_current_path(self) -> List[np.ndarray]:
        """获取当前路径"""
        return self.current_path.copy()
    
    def get_path_quality(self) -> PathQualityMetrics:
        """获取路径质量指标"""
        return self.path_quality
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        success_rate = 0
        if self.planning_stats['total_plans'] > 0:
            success_rate = (self.planning_stats['successful_plans'] / 
                           self.planning_stats['total_plans']) * 100
        
        stats = self.planning_stats.copy()
        stats['success_rate'] = success_rate
        stats['current_path_length'] = len(self.current_path)
        stats['path_index'] = self.path_index
        stats['dynamic_obstacles_detected'] = len(self.obstacle_memory)
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.planning_stats = {
            'total_plans': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'replans': 0,
            'average_planning_time': 0.0,
            'average_path_length': 0.0,
            'average_quality_score': 0.0,
            'dynamic_avoidance_count': 0
        }
        logger.info("[IntelligentPathPlanner] 统计信息已重置")
    
    def clear_current_path(self):
        """清除当前路径"""
        self.current_path.clear()
        self.path_index = 0
        logger.info("[IntelligentPathPlanner] 当前路径已清除")

# 向后兼容别名
PathPlanner = IntelligentPathPlanner
AdvancedPathPlanner = IntelligentPathPlanner