"""
智能地图管理模块 - 修复版本
解决地图显示问题和坐标系错误
"""
import numpy as np
import time
import threading
import logging
from typing import Tuple, Optional, Dict, List, Callable
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from scipy import ndimage

from config import GridConfig

logger = logging.getLogger(__name__)

@dataclass
class MapQualityMetrics:
    """地图质量指标"""
    coverage_ratio: float = 0.0
    confidence_score: float = 0.0
    consistency_score: float = 0.0
    resolution_score: float = 0.0
    completeness_score: float = 0.0

class IntelligentMapManager:
    """智能地图管理器 - 修复版本"""
    
    def __init__(self, config: GridConfig, max_map_size: float = 200.0):
        self.config = config
        self.cell_size = getattr(config, 'resolution', 0.05)  # 修复：默认5cm分辨率
        self.max_map_size = max_map_size
        self.global_resolution = int(max_map_size / self.cell_size)
        self.local_view_size = int(getattr(config, 'grid_size', 50.0) / self.cell_size)
        
        # 确保分辨率合理
        if self.local_view_size > 1000:
            self.local_view_size = 1000
            logger.warning(f"[IntelligentMap] 局部视图尺寸过大，限制为: {self.local_view_size}")
        
        # 多层地图存储 - 内存优化
        self.global_log_odds = np.zeros((self.global_resolution, self.global_resolution), dtype=np.float32)
        self.confidence_map = np.zeros((self.global_resolution, self.global_resolution), dtype=np.float32)
        self.update_count = np.zeros((self.global_resolution, self.global_resolution), dtype=np.uint16)
        self.last_update_time = np.zeros((self.global_resolution, self.global_resolution), dtype=np.float32)
        
        # 显示管理 - 修复：使用固定尺寸便于显示
        self.display_grid = np.full((480, 480), 128, dtype=np.uint8)
        self.last_display_update = 0
        self.display_update_threshold = 3  # 减少更新阈值
        
        # 坐标系管理
        self.origin_world = np.array([0.0, 0.0], dtype=np.float64)
        self.origin_set = False
        
        # 运动状态管理
        self.robot_positions = deque(maxlen=20)
        self.motion_state = "unknown"
        self.velocity_estimate = 0.0
        
        # SLAM集成状态
        self.keyframe_data = []
        self.trajectory_poses = deque(maxlen=1000)
        
        # 质量管理
        self.quality_metrics = MapQualityMetrics()
        self.visited_regions = set()
        self.region_confidence = {}
        
        # 性能优化
        self.update_lock = threading.RLock()
        self.last_quality_update = 0
        
        # 智能更新策略
        self.update_strategies = {
            'stationary': {'threshold': 6, 'weight': 1.3},
            'slow_motion': {'threshold': 3, 'weight': 1.0},
            'fast_motion': {'threshold': 1, 'weight': 0.7},
            'unknown': {'threshold': 2, 'weight': 0.9}
        }
        
        logger.info(f"[IntelligentMap] 初始化完成: {max_map_size}m, 全局分辨率: {self.global_resolution}, 显示尺寸: 480x480")
    
    def update_with_slam(self, local_grid: np.ndarray, robot_pose: np.ndarray, 
                        is_keyframe: bool = False, frame_id: int = 0,
                        slam_quality: float = 1.0) -> np.ndarray:
        """
        使用SLAM数据智能更新地图
        """
        with self.update_lock:
            current_time = time.time()
            
            # 验证输入数据
            if local_grid is None or local_grid.size == 0:
                logger.warning("[IntelligentMap] 收到空的局部网格")
                return self.display_grid.copy()
            
            if robot_pose is None or robot_pose.shape != (4, 4):
                logger.warning("[IntelligentMap] 无效的机器人位姿")
                return self.display_grid.copy()
            
            # 初始化坐标系
            if not self.origin_set:
                self.origin_world = robot_pose[:3, 3][:2].copy()
                self.origin_set = True
                logger.info(f"[IntelligentMap] 设置地图原点: ({self.origin_world[0]:.2f}, {self.origin_world[1]:.2f})")
            
            # 更新运动状态
            self._update_motion_state(robot_pose, current_time)
            
            # 记录轨迹
            self.trajectory_poses.append((robot_pose.copy(), current_time))
            
            # 关键帧处理
            if is_keyframe:
                self._process_keyframe(local_grid, robot_pose, current_time, slam_quality)
            
            # 智能地图更新
            update_weight = self._calculate_update_weight(is_keyframe, slam_quality)
            self._update_global_map(local_grid, robot_pose, update_weight, current_time)
            
            # 智能显示更新决策
            should_update_display = self._should_update_display(frame_id)
            if should_update_display:
                self.display_grid = self._generate_enhanced_view(robot_pose)
                self.last_display_update = frame_id
            
            # 后台质量更新
            if current_time - self.last_quality_update > 5.0:
                self._update_quality_metrics()
                self.last_quality_update = current_time
            
            return self.display_grid.copy()
    
    def update_with_points(self, points: np.ndarray, robot_pose: np.ndarray, 
                          frame_id: int = 0) -> np.ndarray:
        """
        直接使用点云更新地图 - 新增方法解决点云显示问题
        """
        if points is None or points.shape[0] == 0:
            return self.display_grid.copy()
        
        # 生成局部占用网格
        local_grid = self._points_to_occupancy_grid(points, robot_pose)
        
        # 使用现有的SLAM更新方法
        return self.update_with_slam(local_grid, robot_pose, 
                                    is_keyframe=False, frame_id=frame_id, 
                                    slam_quality=0.8)
    
    def _points_to_occupancy_grid(self, points: np.ndarray, robot_pose: np.ndarray, 
                             grid_size: int = 200) -> np.ndarray:
        """
        将点云转换为局部占用网格 - 优化版本
        """
        # 创建局部网格
        local_grid = np.full((grid_size, grid_size), 128, dtype=np.uint8)  # 未知
        
        if points.shape[0] == 0:
            return local_grid
        
        # 机器人位置
        robot_pos = robot_pose[:3, 3]
        
        # 定义局部区域范围
        local_range = 15.0  # 扩大到15米
        cell_size = local_range * 2 / grid_size
        
        # 筛选局部点云
        relative_points = points - robot_pos
        mask = (np.abs(relative_points[:, 0]) < local_range) & \
               (np.abs(relative_points[:, 1]) < local_range)
        local_points = relative_points[mask]
        
        if local_points.shape[0] == 0:
            return local_grid
        
        # 改进的地面分割
        # 1. 基于高度的粗分割
        ground_threshold = -0.2  # 相对于机器人的地面高度
        obstacle_min_height = 0.15   # 障碍物最小高度
        obstacle_max_height = 2.5    # 障碍物最大高度
        
        # 分类点云
        ground_mask = (local_points[:, 2] >= ground_threshold - 0.1) & \
                      (local_points[:, 2] <= ground_threshold + obstacle_min_height)
        
        obstacle_mask = (local_points[:, 2] > ground_threshold + obstacle_min_height) & \
                        (local_points[:, 2] <= obstacle_max_height)
        
        # 转换为网格坐标的函数
        def world_to_grid(pts):
            grid_x = ((pts[:, 0] + local_range) / cell_size).astype(np.int32)
            grid_y = ((pts[:, 1] + local_range) / cell_size).astype(np.int32)
            
            # 边界检查
            valid = (grid_x >= 0) & (grid_x < grid_size) & \
                    (grid_y >= 0) & (grid_y < grid_size)
            
            return grid_x[valid], grid_y[valid]
        
        # 标记地面点为自由空间
        if np.sum(ground_mask) > 0:
            free_x, free_y = world_to_grid(local_points[ground_mask])
            if len(free_x) > 0:
                local_grid[free_y, free_x] = 0  # 自由
    
        # 标记障碍物
        if np.sum(obstacle_mask) > 0:
            obs_x, obs_y = world_to_grid(local_points[obstacle_mask])
            if len(obs_x) > 0:
                # 对障碍物点进行聚类，减少噪声
                obstacle_coords = np.column_stack([obs_x, obs_y])
                unique_coords, counts = np.unique(obstacle_coords, axis=0, return_counts=True)
                
                # 只保留有足够支持点的障碍物
                confidence_threshold = 2
                confident_obstacles = unique_coords[counts >= confidence_threshold]
                
                for coord in confident_obstacles:
                    x, y = coord[0], coord[1]
                    local_grid[y, x] = 255  # 占用
                    
                    # 添加安全边距
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                                local_grid[ny, nx] == 128):
                                local_grid[ny, nx] = 200  # 软占用
    
        # 改进的射线追踪
        center = grid_size // 2
        
        # 选择有代表性的障碍物点进行射线追踪
        obstacle_points = np.where(local_grid == 255)
        if len(obstacle_points[0]) > 0:
            obstacle_coords = list(zip(obstacle_points[1], obstacle_points[0]))  # (x, y)
            
            # 限制射线数量以提高性能
            max_rays = min(20, len(obstacle_coords))
            if len(obstacle_coords) > max_rays:
                # 按距离选择有代表性的点
                distances = [np.sqrt((x - center)**2 + (y - center)**2) 
                            for x, y in obstacle_coords]
                sorted_indices = np.argsort(distances)
                step = len(sorted_indices) // max_rays
                selected_coords = [obstacle_coords[i] for i in sorted_indices[::step][:max_rays]]
            else:
                selected_coords = obstacle_coords
            
            # 执行射线追踪
            for obs_x, obs_y in selected_coords:
                ray_points = self._bresenham_line(center, center, obs_x, obs_y)
                
                # 标记射线路径为自由（除了终点）
                for i in range(len(ray_points) - 1):  # 排除终点
                    ray_x, ray_y = ray_points[i]
                    if (0 <= ray_x < grid_size and 0 <= ray_y < grid_size and
                        local_grid[ray_y, ray_x] == 128):
                        local_grid[ray_y, ray_x] = 0  # 自由
    
        # 传感器周围清理
        sensor_clear_radius = 3
        for dx in range(-sensor_clear_radius, sensor_clear_radius + 1):
            for dy in range(-sensor_clear_radius, sensor_clear_radius + 1):
                if dx*dx + dy*dy <= sensor_clear_radius*sensor_clear_radius:
                    clear_x = center + dx
                    clear_y = center + dy
                    if (0 <= clear_x < grid_size and 0 <= clear_y < grid_size):
                        local_grid[clear_y, clear_x] = 0
    
        return local_grid
    
    def _mark_ray_free(self, grid: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        """简单的Bresenham算法标记射线路径为自由"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy
        
        while True:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                if grid[y, x] == 128:  # 只更新未知区域
                    grid[y, x] = 0  # 自由
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_inc
            if e2 < dx:
                error += dx
                y += y_inc
    
    def _update_motion_state(self, robot_pose: np.ndarray, timestamp: float):
        """更新运动状态估计"""
        current_pos = robot_pose[:3, 3][:2]
        self.robot_positions.append((current_pos.copy(), timestamp))
        
        if len(self.robot_positions) < 3:
            self.motion_state = "unknown"
            return
        
        # 计算最近的速度
        recent_positions = list(self.robot_positions)[-5:]
        if len(recent_positions) >= 2:
            pos_changes = []
            time_diffs = []
            
            for i in range(1, len(recent_positions)):
                pos_diff = np.linalg.norm(recent_positions[i][0] - recent_positions[i-1][0])
                time_diff = recent_positions[i][1] - recent_positions[i-1][1]
                if time_diff > 0:
                    pos_changes.append(pos_diff)
                    time_diffs.append(time_diff)
            
            if pos_changes and time_diffs:
                velocities = np.array(pos_changes) / np.array(time_diffs)
                self.velocity_estimate = np.mean(velocities)
                
                # 分类运动状态
                if self.velocity_estimate < 0.02:
                    self.motion_state = "stationary"
                elif self.velocity_estimate < 0.15:
                    self.motion_state = "slow_motion"
                else:
                    self.motion_state = "fast_motion"
    
    def _process_keyframe(self, local_grid: np.ndarray, robot_pose: np.ndarray, 
                         timestamp: float, quality: float):
        """处理关键帧数据"""
        keyframe_info = {
            'pose': robot_pose.copy(),
            'local_grid': local_grid.copy(),
            'timestamp': timestamp,
            'quality': quality
        }
        
        self.keyframe_data.append(keyframe_info)
        
        # 限制关键帧数量
        max_keyframes = 100  # 减少内存使用
        if len(self.keyframe_data) > max_keyframes:
            # 保留最近的帧
            self.keyframe_data = self.keyframe_data[-max_keyframes:]
        
        logger.debug(f"[IntelligentMap] 处理关键帧 #{len(self.keyframe_data)}, 质量: {quality:.3f}")
    
    def _calculate_update_weight(self, is_keyframe: bool, slam_quality: float) -> float:
        """计算智能更新权重"""
        base_weight = 1.5 if is_keyframe else 1.0
        quality_factor = 0.5 + 0.5 * slam_quality
        
        # 根据运动状态调整
        motion_factor = self.update_strategies.get(self.motion_state, {}).get('weight', 1.0)
        
        return base_weight * quality_factor * motion_factor
    
    def _update_global_map(self, local_grid: np.ndarray, robot_pose: np.ndarray, 
                          weight: float, timestamp: float):
        """高效全局地图更新"""
        robot_pos_2d = robot_pose[:3, 3][:2]
        rel_pos = robot_pos_2d - self.origin_world
        
        # 坐标转换
        global_center = self.global_resolution // 2
        global_x = int(rel_pos[0] / self.cell_size + global_center)
        global_y = int(-rel_pos[1] / self.cell_size + global_center)  # 修复：Y轴翻转
        
        # 边界检查
        if not (50 <= global_x < self.global_resolution - 50 and 
                50 <= global_y < self.global_resolution - 50):
            logger.warning(f"[IntelligentMap] 机器人位置接近边界: ({global_x}, {global_y})")
            if not (0 <= global_x < self.global_resolution and 0 <= global_y < self.global_resolution):
                return
        
        # 计算更新区域
        local_half = local_grid.shape[0] // 2
        
        # 全局地图区域
        g_x_start = max(0, global_x - local_half)
        g_x_end = min(self.global_resolution, global_x + local_half)
        g_y_start = max(0, global_y - local_half) 
        g_y_end = min(self.global_resolution, global_y + local_half)

        # 局部网格区域
        l_x_start = max(0, local_half - (global_x - g_x_start))
        l_x_end = l_x_start + (g_x_end - g_x_start)
        l_y_start = max(0, local_half - (global_y - g_y_start))
        l_y_end = l_y_start + (g_y_end - g_y_start)
        
        # 执行更新
        if (g_x_end > g_x_start and g_y_end > g_y_start and 
            l_x_end > l_x_start and l_y_end > l_y_start and
            l_x_end <= local_grid.shape[1] and l_y_end <= local_grid.shape[0]):
            
            try:
                # 提取对应区域
                local_region = local_grid[l_y_start:l_y_end, l_x_start:l_x_end]
                global_slice = (slice(g_y_start, g_y_end), slice(g_x_start, g_x_end))
                
                # 智能权重计算
                time_diff = timestamp - self.last_update_time[global_slice]
                time_diff = np.maximum(time_diff, 0.1)  # 防止除零
                age_decay = np.exp(-time_diff / 60.0)  # 1分钟衰减
                observation_weight = np.minimum(self.update_count[global_slice] / 10.0, 1.0)
                adaptive_weight = weight * (0.3 + 0.7 * (1.0 - observation_weight)) * (0.2 + 0.8 * age_decay)
                
                # 转换为对数几率
                local_log_odds = self._convert_to_log_odds(local_region)
                
                # 执行加权更新
                weighted_update = local_log_odds * adaptive_weight
                self.global_log_odds[global_slice] += weighted_update
                
                # 更新置信度和统计
                confidence_increment = adaptive_weight * 0.2
                self.confidence_map[global_slice] += confidence_increment
                self.update_count[global_slice] = np.minimum(self.update_count[global_slice] + 1, 65535)
                self.last_update_time[global_slice] = timestamp
                
                # 数值稳定性
                self.global_log_odds = np.clip(self.global_log_odds, -10.0, 10.0)
                self.confidence_map = np.clip(self.confidence_map, 0.0, 5.0)
                
                # 记录访问区域
                region_id = (global_x // 20, global_y // 20)
                self.visited_regions.add(region_id)
                self.region_confidence[region_id] = self.region_confidence.get(region_id, 0) + weight
                
            except Exception as e:
                logger.error(f"[IntelligentMap] 地图更新失败: {e}")
    
    def _convert_to_log_odds(self, grid: np.ndarray) -> np.ndarray:
        """转换占用网格为对数几率"""
        log_odds = np.zeros_like(grid, dtype=np.float32)
        
        # 明确的占用/自由标记
        occupied_mask = grid == 255
        free_mask = grid == 0
        
        log_odds[occupied_mask] = 2.0    # 占用
        log_odds[free_mask] = -1.5       # 自由
        # 未知区域保持0
        
        return log_odds
    
    def _should_update_display(self, frame_id: int) -> bool:
        """智能显示更新决策"""
        frames_since_update = frame_id - self.last_display_update
        
        strategy = self.update_strategies.get(self.motion_state, {'threshold': 3})
        threshold = strategy['threshold']
        
        # 强制更新条件
        if frames_since_update >= threshold * 2:
            return True
        
        # 正常更新条件
        if frames_since_update >= threshold:
            return True
        
        return False
    
    def _generate_enhanced_view(self, robot_pose: np.ndarray) -> np.ndarray:
        """生成增强的局部视图 - 修复版本"""
        robot_pos_2d = robot_pose[:3, 3][:2]
        rel_pos = robot_pos_2d - self.origin_world
        
        # 全局坐标计算
        global_center = self.global_resolution // 2
        global_x = int(rel_pos[0] / self.cell_size + global_center)
        global_y = int(-rel_pos[1] / self.cell_size + global_center)  # Y轴翻转
        
        # 固定显示区域范围 - 20米范围
        display_range = 20.0  # 米
        display_cells = int(display_range / self.cell_size)  # 对应的格子数
        
        # 创建480x480的显示网格
        display_size = 480
        local_view = np.full((display_size, display_size), 128, dtype=np.uint8)
        
        # 提取全局地图区域
        g_x_start = max(0, global_x - display_cells // 2)
        g_x_end = min(self.global_resolution, global_x + display_cells // 2)
        g_y_start = max(0, global_y - display_cells // 2)
        g_y_end = min(self.global_resolution, global_y + display_cells // 2)
        
        if g_x_end > g_x_start and g_y_end > g_y_start:
            try:
                # 提取数据
                log_odds_region = self.global_log_odds[g_y_start:g_y_end, g_x_start:g_x_end]
                confidence_region = self.confidence_map[g_y_start:g_y_end, g_x_start:g_x_end]
                
                # 处理地图区域
                processed_region = self._enhance_map_region(log_odds_region, confidence_region)
                
                # 重采样到480x480
                if processed_region.shape[0] > 0 and processed_region.shape[1] > 0:
                    resized_region = cv2.resize(processed_region, (display_size, display_size), 
                                               interpolation=cv2.INTER_NEAREST)
                    local_view = resized_region
                
            except Exception as e:
                logger.error(f"[IntelligentMap] 视图生成失败: {e}")
        
        return local_view
    
    def _enhance_map_region(self, log_odds: np.ndarray, confidence: np.ndarray) -> np.ndarray:
        """增强地图区域质量"""
        if log_odds.size == 0:
            return np.full((10, 10), 128, dtype=np.uint8)
        
        # 1. 置信度加权
        confidence_normalized = np.clip(confidence / 2.0, 0.0, 1.0)
        weighted_log_odds = log_odds * (0.3 + 0.7 * confidence_normalized)
        
        # 2. 轻度平滑
        try:
            if self.motion_state == "stationary":
                smoothed = ndimage.gaussian_filter(weighted_log_odds, sigma=1.0)
            else:
                smoothed = ndimage.gaussian_filter(weighted_log_odds, sigma=0.5)
        except:
            smoothed = weighted_log_odds
        
        # 3. 概率转换
        prob = 1.0 / (1.0 + np.exp(-smoothed))
        
        # 4. 自适应阈值
        if self.motion_state == "stationary":
            occ_threshold = 0.7
            free_threshold = 0.3
        else:
            occ_threshold = 0.75
            free_threshold = 0.25
        
        # 5. 生成最终网格
        result = np.full_like(prob, 128, dtype=np.uint8)
        
        # 高置信度区域
        high_conf = confidence_normalized > 0.5
        result[high_conf & (prob >= occ_threshold)] = 255  # 占用
        result[high_conf & (prob <= free_threshold)] = 0   # 自由
        
        # 中等置信度区域
        med_conf = (confidence_normalized > 0.2) & (confidence_normalized <= 0.5)
        result[med_conf & (prob >= 0.8)] = 200    # 可能占用
        result[med_conf & (prob <= 0.2)] = 50     # 可能自由
        
        return result
    
    def _update_quality_metrics(self):
        """更新地图质量指标"""
        try:
            # 覆盖率
            total_cells = self.global_resolution * self.global_resolution
            observed_cells = np.sum(self.update_count > 0)
            self.quality_metrics.coverage_ratio = observed_cells / total_cells
            
            # 置信度分数
            valid_confidence = self.confidence_map[self.update_count > 0]
            self.quality_metrics.confidence_score = np.mean(valid_confidence) if len(valid_confidence) > 0 else 0.0
            
            # 分辨率分数
            if observed_cells > 0:
                avg_observations = np.mean(self.update_count[self.update_count > 0])
                self.quality_metrics.resolution_score = min(1.0, avg_observations / 5.0)
            
            # 完整性分数
            expected_regions = max(1, (self.global_resolution // 20) ** 2)
            self.quality_metrics.completeness_score = min(1.0, len(self.visited_regions) / expected_regions)
            
        except Exception as e:
            logger.warning(f"[IntelligentMap] 质量指标更新失败: {e}")
    
    def get_quality_metrics(self) -> Dict[str, float]:
        """获取地图质量指标"""
        return {
            'coverage_ratio': self.quality_metrics.coverage_ratio,
            'confidence_score': self.quality_metrics.confidence_score,
            'resolution_score': self.quality_metrics.resolution_score,
            'completeness_score': self.quality_metrics.completeness_score,
            'motion_state': self.motion_state,
            'velocity': self.velocity_estimate,
            'visited_regions': len(self.visited_regions),
            'total_updates': int(np.sum(self.update_count))
        }
    
    def save_map(self, output_dir: str = "maps", include_metadata: bool = True) -> str:
        """保存优化地图"""
        if not self.origin_set:
            logger.warning("[IntelligentMap] 地图原点未设置，无法保存")
            return ""
        
        timestamp = int(time.time())
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成高质量全局地图
        global_prob = 1.0 / (1.0 + np.exp(-self.global_log_odds))
        confidence_weighted = global_prob * np.clip(self.confidence_map / 2.0, 0.0, 1.0)
        
        # 高质量平滑
        try:
            enhanced_map = ndimage.gaussian_filter(confidence_weighted, sigma=1.0)
        except:
            enhanced_map = confidence_weighted
        
        # 转换为PGM格式
        pgm_grid = np.full_like(enhanced_map, 205, dtype=np.uint8)  # 未知
        pgm_grid[enhanced_map >= 0.65] = 0      # 占用
        pgm_grid[enhanced_map <= 0.35] = 254    # 自由
        
        # 保存主地图
        filename = f"intelligent_map_{timestamp}"
        pgm_path = output_path / f"{filename}.pgm"
        
        try:
            from PIL import Image
            Image.fromarray(pgm_grid, mode='L').save(pgm_path)
        except ImportError:
            try:
                import cv2
                cv2.imwrite(str(pgm_path), pgm_grid)
            except Exception as e:
                logger.error(f"[IntelligentMap] 保存地图失败: {e}")
                return ""
        
        # 保存配置文件
        yaml_path = output_path / f"{filename}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(f"image: {filename}.pgm\n")
            f.write(f"resolution: {self.cell_size:.6f}\n")
            f.write(f"origin: [{self.origin_world[0]:.6f}, {self.origin_world[1]:.6f}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.35\n")
        
        # 保存元数据
        if include_metadata:
            metadata = {
                'quality_metrics': self.get_quality_metrics(),
                'map_info': {
                    'keyframes': len(self.keyframe_data),
                    'trajectory_length': len(self.trajectory_poses),
                    'visited_regions': len(self.visited_regions),
                    'total_updates': int(np.sum(self.update_count))
                },
                'parameters': {
                    'max_map_size': self.max_map_size,
                    'resolution': self.cell_size,
                    'global_resolution': self.global_resolution
                }
            }
            
            metadata_path = output_path / f"{filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"[IntelligentMap] 地图已保存: {pgm_path}")
        logger.info(f"[IntelligentMap] 质量指标: {self.get_quality_metrics()}")
        
        return str(pgm_path)
    
    def reset(self):
        """重置地图管理器"""
        with self.update_lock:
            self.global_log_odds.fill(0)
            self.confidence_map.fill(0)
            self.update_count.fill(0)
            self.last_update_time.fill(0)
            
            self.display_grid.fill(128)
            self.origin_set = False
            self.robot_positions.clear()
            self.trajectory_poses.clear()
            self.keyframe_data.clear()
            self.visited_regions.clear()
            self.region_confidence.clear()
            
            self.quality_metrics = MapQualityMetrics()
            self.motion_state = "unknown"
            self.velocity_estimate = 0.0
            
            logger.info("[IntelligentMap] 已重置")

# 向后兼容别名
EnhancedGlobalMap = IntelligentMapManager
StabilizedMapManager = IntelligentMapManager