"""
KISS-ICP 封装模块
"""
import numpy as np
from typing import List, Optional, Tuple
import open3d as o3d
from scipy.spatial.transform import Rotation
import time

from config import KissICPConfig

class SimpleICP:
    """简化版 ICP 实现"""
    
    def __init__(self, config: KissICPConfig):
        self.config = config
        self.max_iterations = config.max_iterations
        self.threshold = config.convergence_threshold
        self.max_correspondence_distance = config.max_correspondence_distance
    
    def register(self, source: np.ndarray, target: np.ndarray, 
                 initial_transform: np.ndarray = None) -> Tuple[np.ndarray, float]:
        """
        点云配准
        
        Args:
            source: 源点云 (N, 3)
            target: 目标点云 (M, 3)
            initial_transform: 初始变换矩阵 (4, 4)
            
        Returns:
            tuple: (变换矩阵, 配准误差)
        """
        if source.shape[0] < 10 or target.shape[0] < 10:
            return np.eye(4), float('inf')
        
        # 转换为 Open3D 点云
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        # 估计法向量
        source_pcd.estimate_normals()
        target_pcd.estimate_normals()
        
        # 初始变换矩阵
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Point-to-Plane ICP
        try:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, 
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.threshold,
                    relative_rmse=self.threshold
                )
            )
            
            return result.transformation, result.inlier_rmse
            
        except Exception as e:
            print(f"[WARNING] ICP 配准失败: {e}")
            return initial_transform, float('inf')

class KissICPOdometry:
    """KISS-ICP 里程计实现"""
    
    def __init__(self, config: KissICPConfig):
        self.config = config
        self.icp = SimpleICP(config)
        
        # 状态变量
        self.keyframes: List[np.ndarray] = []
        self.keyframe_poses: List[np.ndarray] = []
        self.current_pose = np.eye(4)
        self.last_keyframe_pose = np.eye(4)
        
        # 局部地图
        self.local_map: Optional[np.ndarray] = None
        self.local_map_updated = False
        
        # 统计信息
        self.total_frames = 0
        self.successful_registrations = 0
        
        print("[KissICP] 初始化完成")
    
    def process_frame(self, points: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        处理新的点云帧
        
        Args:
            points: 输入点云 (N, 3)
            
        Returns:
            tuple: (当前位姿, 是否为关键帧)
        """
        self.total_frames += 1
        
        # 点云预处理
        processed_points = self._preprocess_points(points)
        
        if processed_points.shape[0] < 50:
            print(f"[KissICP] 点云数量不足: {processed_points.shape[0]}")
            return self.current_pose, False
        
        # 第一帧处理
        if len(self.keyframes) == 0:
            self._add_keyframe(processed_points, self.current_pose)
            return self.current_pose, True
        
        # 与局部地图配准
        if self.local_map is not None:
            transform, error = self.icp.register(processed_points, self.local_map)
            
            if error < 10.0:  # 配准成功阈值
                self.current_pose = self.current_pose @ transform
                self.successful_registrations += 1
            else:
                print(f"[KissICP] 配准失败，误差: {error:.3f}")
        
        # 检查是否需要添加关键帧
        is_keyframe = self._should_add_keyframe()
        
        if is_keyframe:
            self._add_keyframe(processed_points, self.current_pose)
            self._update_local_map()
        
        return self.current_pose, is_keyframe
    
    def _preprocess_points(self, points: np.ndarray) -> np.ndarray:
        """点云预处理"""
        if len(points) == 0:
            return points
        
        # 距离过滤
        distances = np.linalg.norm(points, axis=1)
        valid_mask = (distances >= self.config.min_range) & (distances <= self.config.max_range)
        filtered_points = points[valid_mask]
        
        if len(filtered_points) < 10:
            return filtered_points
        
        # 体素化降采样
        if self.config.voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            pcd = pcd.voxel_down_sample(self.config.voxel_size)
            filtered_points = np.asarray(pcd.points)
        
        return filtered_points
    
    def _should_add_keyframe(self) -> bool:
        """判断是否应该添加关键帧"""
        if len(self.keyframes) == 0:
            return True
        
        # 计算位移和旋转变化
        pose_diff = np.linalg.inv(self.last_keyframe_pose) @ self.current_pose
        
        # 位移变化
        translation_diff = np.linalg.norm(pose_diff[:3, 3])
        
        # 旋转变化
        rotation_matrix = pose_diff[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        angle_diff = np.abs(rotation.as_rotvec())
        angle_diff_norm = np.linalg.norm(angle_diff)
        
        return (translation_diff > self.config.keyframe_distance or 
                angle_diff_norm > self.config.keyframe_angle)
    
    def _add_keyframe(self, points: np.ndarray, pose: np.ndarray):
        """添加关键帧"""
        self.keyframes.append(points.copy())
        self.keyframe_poses.append(pose.copy())
        self.last_keyframe_pose = pose.copy()
        
        # 限制关键帧数量
        if len(self.keyframes) > self.config.map_size_limit:
            self.keyframes.pop(0)
            self.keyframe_poses.pop(0)
        
        print(f"[KissICP] 添加关键帧 #{len(self.keyframes)}, 位置: {pose[:3, 3]}")
    
    def _update_local_map(self):
        """更新局部地图"""
        if len(self.keyframes) == 0:
            return
        
        # 选择最近的关键帧构建局部地图
        recent_frames = self.keyframes[-self.config.local_map_size:]
        recent_poses = self.keyframe_poses[-self.config.local_map_size:]
        
        # 合并点云
        local_points = []
        for frame_points, pose in zip(recent_frames, recent_poses):
            # 变换到全局坐标系
            homogeneous_points = np.hstack([frame_points, np.ones((len(frame_points), 1))])
            global_points = (pose @ homogeneous_points.T).T[:, :3]
            local_points.append(global_points)
        
        if local_points:
            self.local_map = np.vstack(local_points)
            
            # 降采样局部地图
            if len(self.local_map) > 10000:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(self.local_map)
                pcd = pcd.voxel_down_sample(self.config.voxel_size * 2)
                self.local_map = np.asarray(pcd.points)
            
            self.local_map_updated = True
    
    def get_trajectory(self) -> List[np.ndarray]:
        """获取轨迹"""
        return self.keyframe_poses.copy()
    
    def get_map_points(self) -> Optional[np.ndarray]:
        """获取地图点云"""
        return self.local_map.copy() if self.local_map is not None else None
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        success_rate = (self.successful_registrations / max(1, self.total_frames)) * 100
        return {
            'total_frames': self.total_frames,
            'keyframes': len(self.keyframes),
            'success_rate': success_rate,
            'current_position': self.current_pose[:3, 3].tolist()
        }
    
    def reset(self):
        """重置里程计"""
        self.keyframes.clear()
        self.keyframe_poses.clear()
        self.current_pose = np.eye(4)
        self.last_keyframe_pose = np.eye(4)
        self.local_map = None
        self.total_frames = 0
        self.successful_registrations = 0
        print("[KissICP] 里程计已重置")