"""
KISS-ICP 封装模块 - 深度优化版
集成真正的KISS-ICP库，提供高性能里程计估算
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import time
import threading
from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path

# 尝试导入真正的KISS-ICP
_KISS_ICP_AVAILABLE = False
try:
    from kiss_icp import KissICP
    from kiss_icp.config import load_config
    _KISS_ICP_AVAILABLE = True
    logging.info("[KissICPWrapper] 使用官方KISS-ICP库")
except ImportError:
    logging.warning("[KissICPWrapper] 官方KISS-ICP不可用，使用备用实现")

# 备用导入
if not _KISS_ICP_AVAILABLE:
    try:
        import open3d as o3d
        from scipy.spatial.transform import Rotation
        _OPEN3D_AVAILABLE = True
    except ImportError:
        _OPEN3D_AVAILABLE = False
        logging.error("[KissICPWrapper] Open3D不可用，功能受限")

from config import KissICPConfig

logger = logging.getLogger(__name__)

@dataclass
class RegistrationResult:
    """配准结果数据类"""
    transformation: np.ndarray
    error: float
    converged: bool
    num_iterations: int
    fitness: float
    processing_time: float = 0.0
    point_count: int = 0

@dataclass
class SLAMStatistics:
    """SLAM统计信息"""
    total_frames: int = 0
    successful_registrations: int = 0
    keyframes_count: int = 0
    loop_closures: int = 0
    avg_processing_time: float = 0.0
    current_position: List[float] = field(default_factory=list)
    trajectory_length: float = 0.0
    map_points_count: int = 0
    quality_score: float = 0.0

class OptimizedPointCloudProcessor:
    """优化的点云处理器"""
    
    def __init__(self, config: KissICPConfig):
        self.config = config
        self.min_range = getattr(config, 'min_range', 0.5)
        self.max_range = getattr(config, 'max_range', 50.0)
        self.voxel_size = getattr(config, 'voxel_size', 0.1)
        self.height_threshold = getattr(config, 'height_threshold', 3.0)
        
        # 自适应参数
        self.adaptive_voxel_size = self.voxel_size
        self.dynamic_range_factor = 1.0
        
        # 性能统计
        self.processing_times = deque(maxlen=50)
        self.input_point_counts = deque(maxlen=50)
        self.output_point_counts = deque(maxlen=50)
        
        logger.debug(f"[PointProcessor] 初始化完成: range=[{self.min_range}, {self.max_range}]")
    
    def process_points(self, points: np.ndarray, sensor_position: np.ndarray = None) -> np.ndarray:
        """
        智能点云预处理 - 保守过滤策略
        
        Args:
            points: 输入点云 (N, 3)
            sensor_position: 传感器位置（可选）
            
        Returns:
            处理后的点云
        """
        start_time = time.perf_counter()
        
        if len(points) == 0:
            return points
        
        self.input_point_counts.append(len(points))
        
        # 1. 基础距离过滤 - 保守策略
        if sensor_position is not None:
            distances = np.linalg.norm(points - sensor_position, axis=1)
        else:
            distances = np.linalg.norm(points, axis=1)
        
        dynamic_max_range = self.max_range * self.dynamic_range_factor
        valid_range = (distances >= self.min_range) & (distances <= dynamic_max_range)
        filtered_points = points[valid_range]
        
        if len(filtered_points) < 50:
            self.output_point_counts.append(len(filtered_points))
            return filtered_points
        
        # 2. 保守的高度过滤
        height_filter = np.abs(filtered_points[:, 2]) < self.height_threshold
        filtered_points = filtered_points[height_filter]
        
        # 3. 智能体素降采样
        if len(filtered_points) > 8000:  # 提高阈值
            if _OPEN3D_AVAILABLE:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filtered_points)
                pcd = pcd.voxel_down_sample(self.adaptive_voxel_size)
                filtered_points = np.asarray(pcd.points)
            else:
                # 简单随机采样备用方案
                target_size = min(6000, len(filtered_points))
                indices = np.random.choice(len(filtered_points), target_size, replace=False)
                filtered_points = filtered_points[indices]
        
        # 4. 移除极端离群点（保守）
        if len(filtered_points) > 100:
            center = np.mean(filtered_points, axis=0)
            distances_to_center = np.linalg.norm(filtered_points - center, axis=1)
            outlier_threshold = np.percentile(distances_to_center, 95)  # 保留95%
            filtered_points = filtered_points[distances_to_center <= outlier_threshold]
        
        # 更新统计
        processing_time = time.perf_counter() - start_time
        self.processing_times.append(processing_time)
        self.output_point_counts.append(len(filtered_points))
        
        # 自适应参数调整
        self._adaptive_parameter_update()
        
        return filtered_points
    
    def _adaptive_parameter_update(self):
        """自适应参数更新"""
        if len(self.processing_times) < 10:
            return
        
        avg_processing_time = np.mean(self.processing_times)
        avg_input_count = np.mean(self.input_point_counts)
        avg_output_count = np.mean(self.output_point_counts)
        
        # 根据处理时间调整体素大小
        if avg_processing_time > 0.1:  # 处理太慢
            self.adaptive_voxel_size = min(self.adaptive_voxel_size * 1.1, self.voxel_size * 2)
        elif avg_processing_time < 0.05:  # 处理很快
            self.adaptive_voxel_size = max(self.adaptive_voxel_size * 0.95, self.voxel_size * 0.5)
        
        # 根据点云密度调整范围因子
        density_ratio = avg_output_count / max(1, avg_input_count)
        if density_ratio < 0.1:  # 过滤太多
            self.dynamic_range_factor = min(self.dynamic_range_factor * 1.05, 1.5)
        elif density_ratio > 0.8:  # 过滤太少
            self.dynamic_range_factor = max(self.dynamic_range_factor * 0.98, 0.7)
    
    def get_statistics(self) -> Dict[str, float]:
        """获取处理统计"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'avg_input_count': np.mean(self.input_point_counts) if self.input_point_counts else 0,
            'avg_output_count': np.mean(self.output_point_counts) if self.output_point_counts else 0,
            'adaptive_voxel_size': self.adaptive_voxel_size,
            'dynamic_range_factor': self.dynamic_range_factor
        }

class RobustKissICPOdometry:
    """增强版KISS-ICP里程计 - 集成官方库"""
    
    def __init__(self, config: KissICPConfig):
        self.config = config
        
        # 初始化点云处理器
        self.point_processor = OptimizedPointCloudProcessor(config)
        
        # 初始化KISS-ICP核心
        self.kiss_icp = None
        self._init_kiss_icp()
        
        # 状态管理
        self.current_pose = np.eye(4, dtype=np.float64)
        self.last_pose = np.eye(4, dtype=np.float64)
        self.trajectory: List[np.ndarray] = []
        self.timestamps: List[float] = []
        
        # 关键帧管理
        self.keyframes: List[Tuple[np.ndarray, np.ndarray]] = []  # (points, pose)
        self.keyframe_distance_threshold = getattr(config, 'keyframe_distance', 0.5)
        self.keyframe_angle_threshold = getattr(config, 'keyframe_angle', 0.2)
        
        # 地图管理
        self.local_map_points: Optional[np.ndarray] = None
        self.local_map_lock = threading.RLock()
        self.map_update_counter = 0
        
        # 性能和质量管理
        self.statistics = SLAMStatistics()
        self.frame_count = 0
        self.last_keyframe_pose = np.eye(4, dtype=np.float64)
        
        # 质量评估
        self.registration_qualities = deque(maxlen=100)
        self.pose_uncertainties = deque(maxlen=100)
        
        # 回环检测状态
        self.last_loop_closure_time = 0.0
        self.loop_closure_poses: List[np.ndarray] = []
        
        logger.info("[RobustKissICP] 初始化完成")
    
    def _init_kiss_icp(self):
        """初始化KISS-ICP核心"""
        if _KISS_ICP_AVAILABLE:
            try:
                # 使用官方KISS-ICP
                kiss_config = load_config(config_file=None, max_range=getattr(self.config, 'max_range', 50.0))
                
                # 应用配置
                if hasattr(kiss_config, 'data'):
                    kiss_config.data.max_range = getattr(self.config, 'max_range', 50.0)
                    kiss_config.data.min_range = getattr(self.config, 'min_range', 0.5)
                    kiss_config.data.deskew = getattr(self.config, 'deskew', True)
                
                if hasattr(kiss_config, 'mapping'):
                    kiss_config.mapping.voxel_size = getattr(self.config, 'voxel_size', 0.1)
                    kiss_config.mapping.max_points_per_voxel = getattr(self.config, 'max_points_per_voxel', 20)
                
                if hasattr(kiss_config, 'registration'):
                    kiss_config.registration.max_num_iterations = getattr(self.config, 'max_iterations', 50)
                    kiss_config.registration.convergence_criterion = getattr(self.config, 'convergence_threshold', 1e-6)
                
                self.kiss_icp = KissICP(kiss_config)
                logger.info("[RobustKissICP] 使用官方KISS-ICP核心")
                return
                
            except Exception as e:
                logger.warning(f"[RobustKissICP] 官方KISS-ICP初始化失败: {e}")
        
        # 备用实现
        logger.info("[RobustKissICP] 使用备用ICP实现")
        self.kiss_icp = None
    
    def process_frame(self, points: np.ndarray, timestamp: Optional[float] = None) -> Tuple[np.ndarray, bool]:
        """
        处理点云帧
        
        Args:
            points: 输入点云 (N, 3)
            timestamp: 时间戳
            
        Returns:
            (当前位姿, 是否为关键帧)
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        self.frame_count += 1
        self.statistics.total_frames += 1
        
        # 验证输入
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            logger.warning(f"[RobustKissICP] 无效输入: {points.shape if hasattr(points, 'shape') else type(points)}")
            return self.current_pose.copy(), False
        
        # 点云预处理
        processed_points = self.point_processor.process_points(points, self.current_pose[:3, 3])
        
        if len(processed_points) < 100:
            logger.debug(f"[RobustKissICP] 点云数量不足: {len(processed_points)}")
            return self.current_pose.copy(), False
        
        # SLAM处理
        success = False
        is_keyframe = False
        
        if self.kiss_icp is not None:
            # 使用官方KISS-ICP
            success, is_keyframe = self._process_with_official_kiss_icp(processed_points, timestamp)
        else:
            # 使用备用实现
            success, is_keyframe = self._process_with_fallback(processed_points, timestamp)
        
        if success:
            self.statistics.successful_registrations += 1
            
            # 更新轨迹
            self.trajectory.append(self.current_pose.copy())
            self.timestamps.append(timestamp)
            
            # 轨迹长度计算
            if len(self.trajectory) > 1:
                distance = np.linalg.norm(self.trajectory[-1][:3, 3] - self.trajectory[-2][:3, 3])
                self.statistics.trajectory_length += distance
            
            # 关键帧处理
            if is_keyframe:
                self._add_keyframe(processed_points, self.current_pose.copy())
                self.statistics.keyframes_count += 1
        
        # 更新统计
        processing_time = time.perf_counter() - start_time
        self.statistics.avg_processing_time = (
            self.statistics.avg_processing_time * 0.9 + processing_time * 0.1
        )
        self.statistics.current_position = self.current_pose[:3, 3].tolist()
        
        return self.current_pose.copy(), is_keyframe
    
    def _process_with_official_kiss_icp(self, points: np.ndarray, timestamp: float) -> Tuple[bool, bool]:
        """使用官方KISS-ICP处理"""
        try:
            # 准备时间戳数组
            timestamps_array = np.full(len(points), timestamp)
            
            # KISS-ICP配准
            frame, source = self.kiss_icp.register_frame(points, timestamps_array)
            
            # 获取最新位姿
            self.last_pose = self.current_pose.copy()
            self.current_pose = self.kiss_icp.last_pose.copy()
            
            # 评估配准质量
            pose_change = np.linalg.norm(self.current_pose[:3, 3] - self.last_pose[:3, 3])
            quality_score = min(1.0, 1.0 / (1.0 + pose_change * 10))  # 启发式质量评估
            self.registration_qualities.append(quality_score)
            self.statistics.quality_score = np.mean(self.registration_qualities)
            
            # 更新地图点云
            with self.local_map_lock:
                if hasattr(self.kiss_icp, 'local_map') and len(self.kiss_icp.local_map.points) > 0:
                    self.local_map_points = np.asarray(self.kiss_icp.local_map.points).copy()
                    self.statistics.map_points_count = len(self.local_map_points)
            
            # 关键帧判断
            is_keyframe = self._should_add_keyframe()
            
            return True, is_keyframe
            
        except Exception as e:
            logger.error(f"[RobustKissICP] 官方KISS-ICP处理失败: {e}")
            return False, False
    
    def _process_with_fallback(self, points: np.ndarray, timestamp: float) -> Tuple[bool, bool]:
        """备用ICP实现"""
        if not _OPEN3D_AVAILABLE:
            logger.warning("[RobustKissICP] Open3D不可用，无法进行配准")
            return False, False
        
        try:
            # 简单的帧间配准
            if len(self.trajectory) == 0:
                # 第一帧
                return True, True
            
            # 获取最近的关键帧作为目标
            target_points = None
            if self.keyframes:
                target_points = self.keyframes[-1][0]
            else:
                return True, True
            
            # Open3D ICP配准
            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(points)
            
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_points)
            
            # 执行ICP
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                max_correspondence_distance=1.0,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # 更新位姿
            self.last_pose = self.current_pose.copy()
            relative_transform = result.transformation
            self.current_pose = self.last_pose @ relative_transform
            
            # 质量评估
            self.registration_qualities.append(result.fitness)
            self.statistics.quality_score = np.mean(self.registration_qualities)
            
            # 关键帧判断
            is_keyframe = self._should_add_keyframe()
            
            return result.fitness > 0.3, is_keyframe
            
        except Exception as e:
            logger.error(f"[RobustKissICP] 备用ICP失败: {e}")
            return False, False
    
    def _should_add_keyframe(self) -> bool:
        """智能关键帧判断"""
        if len(self.keyframes) == 0:
            return True
        
        # 计算位姿变化
        pose_diff = np.linalg.inv(self.last_keyframe_pose) @ self.current_pose
        
        # 位移变化
        translation_distance = np.linalg.norm(pose_diff[:3, 3])
        
        # 旋转变化
        rotation_matrix = pose_diff[:3, :3]
        if _OPEN3D_AVAILABLE:
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_matrix(rotation_matrix)
            angle_change = np.linalg.norm(rotation.as_rotvec())
        else:
            # 简化的角度计算
            angle_change = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
        
        # 基础条件
        distance_condition = translation_distance > self.keyframe_distance_threshold
        angle_condition = angle_change > self.keyframe_angle_threshold
        
        # 质量条件
        quality_condition = False
        if self.registration_qualities:
            recent_quality = np.mean(list(self.registration_qualities)[-5:])
            if recent_quality < 0.5:  # 质量下降时更频繁添加关键帧
                quality_condition = translation_distance > self.keyframe_distance_threshold * 0.5
        
        return distance_condition or angle_condition or quality_condition
    
    def _add_keyframe(self, points: np.ndarray, pose: np.ndarray):
        """添加关键帧"""
        self.keyframes.append((points.copy(), pose.copy()))
        self.last_keyframe_pose = pose.copy()
        
        # 限制关键帧数量
        max_keyframes = getattr(self.config, 'max_keyframes', 150)
        if len(self.keyframes) > max_keyframes:
            # 智能删除策略：保留最近的帧，均匀采样较老的帧
            recent_count = max_keyframes // 2
            uniform_count = max_keyframes - recent_count
            
            recent_frames = self.keyframes[-recent_count:]
            older_frames = self.keyframes[:-recent_count]
            
            if older_frames:
                step = max(1, len(older_frames) // uniform_count)
                uniform_frames = older_frames[::step][:uniform_count]
            else:
                uniform_frames = []
            
            self.keyframes = uniform_frames + recent_frames
        
        logger.debug(f"[RobustKissICP] 添加关键帧 #{len(self.keyframes)}")
    
    def get_trajectory(self) -> List[np.ndarray]:
        """获取轨迹"""
        return [pose.copy() for pose in self.trajectory]
    
    def get_map_points(self) -> Optional[np.ndarray]:
        """获取地图点云"""
        with self.local_map_lock:
            return self.local_map_points.copy() if self.local_map_points is not None else None
    
    def get_keyframes(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取关键帧"""
        return [(points.copy(), pose.copy()) for points, pose in self.keyframes]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取详细统计信息"""
        base_stats = {
            'total_frames': self.statistics.total_frames,
            'successful_registrations': self.statistics.successful_registrations,
            'keyframes': len(self.keyframes),
            'success_rate': (self.statistics.successful_registrations / max(1, self.statistics.total_frames)) * 100,
            'current_position': self.statistics.current_position,
            'trajectory_length': self.statistics.trajectory_length,
            'map_points_count': self.statistics.map_points_count,
            'quality_score': self.statistics.quality_score,
            'avg_processing_time': self.statistics.avg_processing_time
        }
        
        # 添加点云处理统计
        point_stats = self.point_processor.get_statistics()
        base_stats.update({f'point_processor_{k}': v for k, v in point_stats.items()})
        
        # 添加质量统计
        if self.registration_qualities:
            base_stats.update({
                'avg_registration_quality': np.mean(self.registration_qualities),
                'min_registration_quality': np.min(self.registration_qualities),
                'max_registration_quality': np.max(self.registration_qualities)
            })
        
        return base_stats
    
    def save_trajectory(self, filepath: str):
        """保存轨迹到文件"""
        if not self.trajectory:
            logger.warning("[RobustKissICP] 没有轨迹数据可保存")
            return
        
        try:
            trajectory_data = []
            for i, (pose, timestamp) in enumerate(zip(self.trajectory, self.timestamps)):
                position = pose[:3, 3]
                
                # 提取旋转四元数
                if _OPEN3D_AVAILABLE:
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_matrix(pose[:3, :3])
                    quat = rotation.as_quat()  # [x, y, z, w]
                else:
                    quat = [0, 0, 0, 1]  # 默认无旋转
                
                trajectory_data.append([
                    timestamp, position[0], position[1], position[2],
                    quat[0], quat[1], quat[2], quat[3]
                ])
            
            # 保存为TUM格式
            header = "# timestamp tx ty tz qx qy qz qw"
            np.savetxt(filepath, trajectory_data, fmt='%.6f', header=header, comments='')
            logger.info(f"[RobustKissICP] 轨迹已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"[RobustKissICP] 轨迹保存失败: {e}")
    
    def reset(self):
        """重置里程计"""
        self.current_pose = np.eye(4, dtype=np.float64)
        self.last_pose = np.eye(4, dtype=np.float64)
        self.trajectory.clear()
        self.timestamps.clear()
        self.keyframes.clear()
        
        with self.local_map_lock:
            self.local_map_points = None
        
        self.statistics = SLAMStatistics()
        self.frame_count = 0
        self.last_keyframe_pose = np.eye(4, dtype=np.float64)
        self.registration_qualities.clear()
        
        if self.kiss_icp is not None and hasattr(self.kiss_icp, 'reset'):
            try:
                self.kiss_icp.reset()
            except:
                pass
        
        logger.info("[RobustKissICP] 已重置")

# 向后兼容别名
KissICPOdometry = RobustKissICPOdometry
EnhancedKissICPOdometry = RobustKissICPOdometry