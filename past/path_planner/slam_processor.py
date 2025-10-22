"""
SLAM处理器 - 集成优化版
统一协调KISS-ICP和地图管理，提供高级SLAM功能
"""
import numpy as np
import time
import threading
import logging
from typing import Tuple, Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from collections import deque

from config import GridConfig, KissICPConfig, SystemConfig
from kiss_icp_wrapper import RobustKissICPOdometry
from map_manager import IntelligentMapManager

logger = logging.getLogger(__name__)

@dataclass
class SLAMProcessingResult:
    """SLAM处理结果"""
    pose: np.ndarray
    is_keyframe: bool
    quality_score: float
    processing_time: float
    point_count: int
    map_grid: Optional[np.ndarray] = None

class AdvancedSLAMProcessor:
    """高级SLAM处理器 - 统一协调版"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.grid_config = config.grid
        self.kiss_icp_config = config.kiss_icp
        
        # 核心组件初始化 - 使用优化后的RobustKissICPOdometry
        self.slam_odometry = RobustKissICPOdometry(self.kiss_icp_config)
        self.map_manager = IntelligentMapManager(
            self.grid_config, 
            max_map_size=getattr(config, 'max_map_size', 200.0)
        )
        
        # 局部占用网格
        self.grid_resolution = int(self.grid_config.grid_size / self.grid_config.resolution)
        self.local_occupancy_grid = np.full((self.grid_resolution, self.grid_resolution), 128, dtype=np.uint8)
        
        # 处理状态
        self.frame_count = 0
        self.processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'keyframes_generated': 0,
            'loop_closures_detected': 0,
            'average_quality': 0.0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0,
            'pose_estimation_success_rate': 0.0
        }
        
        # 质量管理
        self.quality_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=50)
        self.pose_quality_scores = deque(maxlen=100)
        
        # 坐标系管理
        self.coordinate_correction_matrix = self._setup_coordinate_correction()
        
        # 回调管理
        self.callbacks = {
            'pose_update': [],
            'map_update': [],
            'quality_update': [],
            'keyframe_created': []
        }
        
        # SLAM状态跟踪
        self.last_loop_closure_time = 0.0
        self.loop_closure_count = 0
        self.last_successful_pose = np.eye(4, dtype=np.float64)
        
        # 性能优化
        self.processing_lock = threading.RLock()
        self.async_processing_enabled = True
        
        logger.info("[AdvancedSLAM] 高级SLAM处理器初始化完成")
    
    def _setup_coordinate_correction(self) -> np.ndarray:
        """设置坐标系校正矩阵"""
        correction_type = getattr(self.config, 'coordinate_correction', 'livox_standard')
        
        if correction_type == "upside_down":
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        elif correction_type == "rotated_90":
            return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        elif correction_type == "simple_flip_y":
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        else:
            return np.eye(3, dtype=np.float64)
    
    def process_points(self, xyz: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        处理点云数据 - 主要接口
        
        Args:
            xyz: 输入点云 (N, 3)
            timestamp: 时间戳
            
        Returns:
            稳定的占用网格
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        self.processing_stats['total_frames'] += 1
        
        # 验证输入
        if not isinstance(xyz, np.ndarray) or xyz.ndim != 2 or xyz.shape[1] != 3:
            logger.warning("[AdvancedSLAM] 无效的点云输入")
            return self.local_occupancy_grid.copy()
        
        if len(xyz) == 0:
            logger.warning("[AdvancedSLAM] 空点云输入")
            return self.local_occupancy_grid.copy()
        
        # 坐标系校正
        xyz_corrected = self._apply_coordinate_correction(xyz)
        
        # KISS-ICP处理 - 使用优化后的接口
        try:
            current_pose, is_keyframe = self.slam_odometry.process_frame(xyz_corrected, timestamp)
            slam_success = True
            quality_score = 0.8  # 默认质量分数
        except Exception as e:
            logger.error(f"[AdvancedSLAM] SLAM处理失败: {e}")
            current_pose = self.last_successful_pose.copy()
            is_keyframe = False
            slam_success = False
            quality_score = 0.0
        
        if slam_success:
            self.last_successful_pose = current_pose.copy()
            self.processing_stats['successful_frames'] += 1
            
            # 更新局部占用网格
            self._update_local_occupancy_grid(xyz_corrected, current_pose)
            
            # 使用地图管理器进行全局更新
            enhanced_grid = self.map_manager.update_with_points(
                xyz_corrected, current_pose, frame_id=self.frame_count
            )
        else:
            enhanced_grid = self.local_occupancy_grid.copy()
        
        # 性能统计更新
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)
        self.quality_history.append(quality_score)
        
        self._update_processing_statistics(processing_time, quality_score, slam_success)
        
        # 触发回调
        self._trigger_callback('pose_update', current_pose, is_keyframe)
        self._trigger_callback('map_update', enhanced_grid)
        self._trigger_callback('quality_update', self.get_comprehensive_statistics())
        
        return enhanced_grid
    
    def _apply_coordinate_correction(self, xyz: np.ndarray) -> np.ndarray:
        """应用坐标系校正"""
        if np.array_equal(self.coordinate_correction_matrix, np.eye(3)):
            return xyz
        
        # 应用校正矩阵
        return xyz @ self.coordinate_correction_matrix.T
    
    def _update_local_occupancy_grid(self, xyz: np.ndarray, robot_pose: np.ndarray):
        """更新局部占用网格 - 优化版"""
        if len(xyz) == 0:
            return
        
        # 重置网格
        self.local_occupancy_grid.fill(128)  # 128 = 未知区域
        
        sensor_pos = robot_pose[:3, 3]
        grid_center = self.grid_resolution // 2
        
        # 高度过滤 - 确保只处理障碍物高度的点
        height_mask = (xyz[:, 2] >= -0.3) & (xyz[:, 2] <= 2.0)
        filtered_xyz = xyz[height_mask]
        
        if len(filtered_xyz) == 0:
            return
        
        # 点云到网格坐标转换 - 向量化操作
        relative_points = filtered_xyz - sensor_pos
        distances = np.linalg.norm(relative_points[:, :2], axis=1)
        
        # 距离过滤
        valid_distance = (distances >= 0.3) & (distances < self.grid_config.max_range)
        valid_points = relative_points[valid_distance]
        
        if len(valid_points) == 0:
            return
        
        # 转换为网格坐标
        grid_coords = np.column_stack([
            (valid_points[:, 0] / self.grid_config.resolution + grid_center).astype(int),
            (-valid_points[:, 1] / self.grid_config.resolution + grid_center).astype(int)
        ])
        
        # 有效性检查
        valid_mask = (
            (grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < self.grid_resolution) &
            (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < self.grid_resolution)
        )
        valid_coords = grid_coords[valid_mask]
        valid_heights = valid_points[valid_mask][:, 2]
        
        if len(valid_coords) == 0:
            return
        
        # 分类处理：地面点和障碍物点
        ground_threshold = -0.1
        obstacle_threshold = 0.2
        
        ground_mask = valid_heights < ground_threshold
        obstacle_mask = valid_heights > obstacle_threshold
        
        # 标记地面为自由空间
        if np.any(ground_mask):
            ground_coords = valid_coords[ground_mask]
            for coord in ground_coords:
                col, row = coord[0], coord[1]
                self.local_occupancy_grid[row, col] = 0  # 自由空间
        
        # 标记障碍物
        if np.any(obstacle_mask):
            obstacle_coords = valid_coords[obstacle_mask]
            unique_obstacle_coords = np.unique(obstacle_coords, axis=0)
            
            for coord in unique_obstacle_coords:
                col, row = coord[0], coord[1]
                self.local_occupancy_grid[row, col] = 255  # 占用
                
                # 在障碍物周围添加软占用（安全边距）
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        adj_row, adj_col = row + dr, col + dc
                        if (0 <= adj_row < self.grid_resolution and 
                            0 <= adj_col < self.grid_resolution and
                            self.local_occupancy_grid[adj_row, adj_col] == 128):
                            self.local_occupancy_grid[adj_row, adj_col] = 200  # 软占用
        
        # 射线追踪标记自由空间
        sensor_grid_x, sensor_grid_y = grid_center, grid_center
        
        # 优化射线追踪：只对部分障碍物进行
        unique_coords = np.unique(valid_coords, axis=0)
        max_rays = min(30, len(unique_coords))
        
        if max_rays > 0:
            # 选择有代表性的点进行射线追踪
            if len(unique_coords) > max_rays:
                # 选择距离分布均匀的点
                distances_to_sensor = np.linalg.norm(unique_coords - [grid_center, grid_center], axis=1)
                sorted_indices = np.argsort(distances_to_sensor)
                step = len(sorted_indices) // max_rays
                sample_indices = sorted_indices[::step][:max_rays]
                sample_coords = unique_coords[sample_indices]
            else:
                sample_coords = unique_coords
            
            for coord in sample_coords:
                ray_points = self._bresenham_line(sensor_grid_x, sensor_grid_y, coord[0], coord[1])
                
                # 标记射线路径为自由空间（除了终点）
                for i, (ray_x, ray_y) in enumerate(ray_points[:-1]):  # 排除终点
                    if (0 <= ray_x < self.grid_resolution and 
                        0 <= ray_y < self.grid_resolution and
                        self.local_occupancy_grid[ray_y, ray_x] == 128):  # 只更新未知区域
                        self.local_occupancy_grid[ray_y, ray_x] = 0  # 自由空间
        
        # 传感器周围标记自由空间
        sensor_clear_radius = 3
        for dx in range(-sensor_clear_radius, sensor_clear_radius + 1):
            for dy in range(-sensor_clear_radius, sensor_clear_radius + 1):
                if dx*dx + dy*dy <= sensor_clear_radius*sensor_clear_radius:
                    clear_x = grid_center + dx
                    clear_y = grid_center + dy
                    if (0 <= clear_x < self.grid_resolution and 
                        0 <= clear_y < self.grid_resolution):
                        self.local_occupancy_grid[clear_y, clear_x] = 0
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Bresenham直线算法 - 优化版"""
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
    
    def _update_processing_statistics(self, processing_time: float, quality_score: float, success: bool):
        """更新处理统计"""
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['average_processing_time'] = (
            self.processing_stats['total_processing_time'] / max(1, self.processing_stats['total_frames'])
        )
        
        if success:
            self.processing_stats['pose_estimation_success_rate'] = (
                self.processing_stats['successful_frames'] / max(1, self.processing_stats['total_frames'])
            )
        
        # 更新平均质量
        if self.quality_history:
            self.processing_stats['average_quality'] = np.mean(self.quality_history)
    
    def _trigger_callback(self, callback_type: str, *args, **kwargs):
        """触发回调"""
        callbacks = self.callbacks.get(callback_type, [])
        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"[AdvancedSLAM] 回调执行失败 ({callback_type}): {e}")
    
    # ---------------------------------------------------------------------------
    # 公共接口 - 与优化后的KISS-ICP集成
    # ---------------------------------------------------------------------------
    
    def register_callback(self, callback_type: str, callback: Callable):
        """注册回调"""
        if callback_type in self.callbacks:
            self.callbacks[callback_type].append(callback)
        else:
            logger.warning(f"[AdvancedSLAM] 未知的回调类型: {callback_type}")
    
    def get_current_pose(self) -> np.ndarray:
        """获取当前位姿"""
        return self.last_successful_pose.copy()
    
    def get_trajectory(self) -> List[np.ndarray]:
        """获取轨迹"""
        return self.slam_odometry.get_trajectory()
    
    def get_map_points(self) -> Optional[np.ndarray]:
        """获取地图点"""
        return self.slam_odometry.get_map_points()
    
    def get_keyframes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取关键帧"""
        return self.slam_odometry.get_keyframes()
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """获取综合统计"""
        stats = self.processing_stats.copy()
        stats.update({
            'current_pose': self.last_successful_pose.tolist(),
            'frame_count': self.frame_count,
            'quality_history_length': len(self.quality_history),
            'recent_quality': list(self.quality_history)[-10:] if self.quality_history else [],
            'map_manager_stats': self.map_manager.get_quality_metrics() if hasattr(self.map_manager, 'get_quality_metrics') else {}
        })
        return stats
    
    def save_trajectory(self, filepath: str) -> bool:
        """保存轨迹"""
        try:
            trajectory = self.get_trajectory()
            if trajectory:
                np.save(filepath, np.array(trajectory))
                logger.info(f"[AdvancedSLAM] 轨迹已保存到: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"[AdvancedSLAM] 保存轨迹失败: {e}")
            return False
    
    def save_map(self, output_dir: str = "maps", include_metadata: bool = True) -> str:
        """保存地图"""
        return self.map_manager.save_map(output_dir, include_metadata)
    
    def reset(self):
        """重置SLAM系统"""
        logger.info("[AdvancedSLAM] 重置SLAM系统")
        self.slam_odometry.reset()
        self.map_manager.reset()
        self.local_occupancy_grid.fill(128)
        self.frame_count = 0
        self.last_successful_pose = np.eye(4, dtype=np.float64)
        self.quality_history.clear()
        self.processing_times.clear()
        self.pose_quality_scores.clear()
        
        # 重置统计
        self.processing_stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'keyframes_generated': 0,
            'loop_closures_detected': 0,
            'average_quality': 0.0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0,
            'pose_estimation_success_rate': 0.0
        }
    
    def get_optimized_poses(self) -> List[np.ndarray]:
        """获取优化后的位姿"""
        return self.get_trajectory()
    
    def get_occupancy_grid_debug_info(self) -> Dict[str, Any]:
        """获取占用网格调试信息"""
        if self.local_occupancy_grid is None:
            return {}
        
        grid = self.local_occupancy_grid
        unique, counts = np.unique(grid, return_counts=True)
        grid_stats = dict(zip(unique, counts))
        
        total_cells = grid.size
        occupied_cells = grid_stats.get(255, 0)
        free_cells = grid_stats.get(0, 0)
        unknown_cells = grid_stats.get(128, 0)
        soft_occupied_cells = grid_stats.get(200, 0)
        
        return {
            'grid_shape': grid.shape,
            'total_cells': total_cells,
            'occupied_cells': occupied_cells,
            'free_cells': free_cells,
            'unknown_cells': unknown_cells,
            'soft_occupied_cells': soft_occupied_cells,
            'occupied_ratio': occupied_cells / total_cells,
            'free_ratio': free_cells / total_cells,
            'unknown_ratio': unknown_cells / total_cells,
            'grid_stats': grid_stats
        }