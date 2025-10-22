"""
集成SLAM系统 - 全面优化版
与优化后的KISS-ICP、地图管理器和可视化器深度集成
"""
import threading
import time
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

from config import SystemConfig
from slam_processor import AdvancedSLAMProcessor, SLAMProcessingResult
from path_planner import IntelligentPathPlanner, PathQualityMetrics
from map_manager import IntelligentMapManager
from robot_control import RobotController
from visualizer import IntelligentVisualizer
from livox2_python import Livox2

logger = logging.getLogger(__name__)

@dataclass
class SystemStatus:
    """系统状态数据类"""
    is_running: bool = False
    slam_active: bool = False
    planning_active: bool = False
    following_path: bool = False
    lidar_connected: bool = False
    frame_count: int = 0
    current_pose: Optional[np.ndarray] = None
    goal_position: Optional[np.ndarray] = None
    path_length: int = 0
    slam_quality: float = 0.0
    processing_fps: float = 0.0
    uptime: float = 0.0

class OptimizedLivoxAdapter:
    """优化的Livox适配器 - 与Livox2深度集成"""
    
    def __init__(self, config, parent_system):
        self.config = config
        self.parent_system = parent_system
        self.livox_scanner = None
        self.is_running = False
        
        # 数据缓冲
        self.latest_pointcloud = None
        self.latest_imu = None
        self.data_lock = threading.RLock()
        
        # 性能统计
        self.packet_count = 0
        self.point_count = 0
        self.last_packet_time = 0
        
        logger.info("[OptimizedLivoxAdapter] 适配器初始化完成")
    
    def start(self):
        """启动激光雷达"""
        try:
            from livox2_python import Livox2
            
            class LiveoxScannerImpl(Livox2):
                def __init__(self, config_path, host_ip, adapter):
                    super().__init__(config_path, host_ip)
                    self.adapter = adapter
                
                def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                                 tag: np.ndarray, timestamp: int):
                    """处理点云数据"""
                    self.adapter._on_points_received(xyz, reflectivity, tag, timestamp)
                
                def handle_imu(self, imu_data: np.ndarray, timestamp: int):
                    """处理IMU数据"""
                    self.adapter._on_imu_received(imu_data, timestamp)
            
            # 创建Livox2扫描器
            self.livox_scanner = LiveoxScannerImpl(
                self.config.lidar.config_path,
                self.config.lidar.host_ip,
                self
            )
            
            self.is_running = True
            logger.info("[OptimizedLivoxAdapter] 激光雷达启动成功")
            
        except Exception as e:
            logger.error(f"[OptimizedLivoxAdapter] 激光雷达启动失败: {e}")
            raise
    
    def shutdown(self):
        """关闭激光雷达"""
        self.is_running = False
        if self.livox_scanner:
            try:
                self.livox_scanner.shutdown()
                logger.info("[OptimizedLivoxAdapter] 激光雷达已关闭")
            except Exception as e:
                logger.error(f"[OptimizedLivoxAdapter] 关闭激光雷达失败: {e}")
    
    def _on_points_received(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                           tag: np.ndarray, timestamp: int):
        """点云数据回调"""
        try:
            # 基础过滤
            valid_mask = tag == 0  # 只保留正常点
            if np.sum(valid_mask) < 50:  # 点数太少
                return
            
            filtered_xyz = xyz[valid_mask]
            filtered_ref = reflectivity[valid_mask]
            filtered_tag = tag[valid_mask]
            
            # 更新缓冲区
            with self.data_lock:
                self.latest_pointcloud = {
                    'xyz': filtered_xyz.copy(),
                    'reflectivity': filtered_ref.copy(),
                    'tag': filtered_tag.copy(),
                    'timestamp': timestamp
                }
                self.packet_count += 1
                self.point_count += len(filtered_xyz)
                self.last_packet_time = time.time()
            
            # 通知父系统
            if self.parent_system:
                self.parent_system._on_pointcloud_update()
            
        except Exception as e:
            logger.error(f"[OptimizedLivoxAdapter] 点云处理错误: {e}")
    
    def _on_imu_received(self, imu_data: np.ndarray, timestamp: int):
        """IMU数据回调"""
        try:
            with self.data_lock:
                self.latest_imu = {
                    'data': imu_data.copy(),
                    'timestamp': timestamp
                }
            
            # 通知父系统（如果需要）
            if self.parent_system and hasattr(self.parent_system, '_on_imu_update'):
                self.parent_system._on_imu_update()
                
        except Exception as e:
            logger.error(f"[OptimizedLivoxAdapter] IMU处理错误: {e}")
    
    def get_latest_pointcloud(self) -> Optional[Dict]:
        """获取最新点云数据"""
        with self.data_lock:
            return self.latest_pointcloud.copy() if self.latest_pointcloud else None
    
    def get_latest_imu(self) -> Optional[Dict]:
        """获取最新IMU数据"""
        with self.data_lock:
            return self.latest_imu.copy() if self.latest_imu else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取适配器统计信息"""
        with self.data_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_packet_time if self.last_packet_time else 0
            
            return {
                'is_running': self.is_running,
                'packet_count': self.packet_count,
                'total_points': self.point_count,
                'last_packet_time': self.last_packet_time,
                'time_since_last_packet': time_since_last,
                'connection_active': time_since_last < 2.0
            }

class EnhancedIntegratedSLAMSystem:
    """增强版集成SLAM系统 - 与优化组件深度集成"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        
        # 系统状态
        self.status = SystemStatus()
        self.start_time = time.time()
        
        # 线程安全
        self.system_lock = threading.RLock()
        
        # 核心组件初始化
        logger.info("[EnhancedIntegratedSLAM] 初始化核心组件...")
        
        # SLAM处理器 - 使用优化后的AdvancedSLAMProcessor
        self.slam_processor = AdvancedSLAMProcessor(config)
        
        # 地图管理器 - 使用优化后的IntelligentMapManager
        self.map_manager = IntelligentMapManager(
            config.grid,
            max_map_size=config.global_map_size
        )
        
        # 路径规划器 - 修复初始化参数顺序和传递方式
        self.path_planner = IntelligentPathPlanner(
            config.planner,      # PathPlannerConfig
            config.grid,         # GridConfig
            self.slam_processor  # SLAM处理器引用
        )
        
        # 可视化器 - 使用优化后的IntelligentVisualizer
        self.visualizer = IntelligentVisualizer(
            config.visualization,
            system_config=config,
            slam_processor=self.slam_processor,
            path_planner=self.path_planner
        )
        
        # 机器人控制器
        self.robot_controller = None
        if config.robot.enabled:
            try:
                self.robot_controller = RobotController(config.robot)
                logger.info("[EnhancedIntegratedSLAM] 机器人控制器已启用")
            except Exception as e:
                logger.warning(f"[EnhancedIntegratedSLAM] 机器人控制器初始化失败: {e}")
        
        # 激光雷达适配器
        self.lidar_adapter = OptimizedLivoxAdapter(config, self)
        
        # 处理线程
        self.processing_thread = None
        self.processing_enabled = False
        
        # 性能统计
        self.frame_times = []
        self.processing_times = []
        self.last_stats_update = time.time()
        
        # 注册回调
        self._register_callbacks()
        
        logger.info("[EnhancedIntegratedSLAM] 系统初始化完成")
    
    def _register_callbacks(self):
        """注册系统组件间的回调"""
        # SLAM处理器回调已在可视化器中注册
        
        # 路径规划器回调
        if hasattr(self.path_planner, 'register_goal_reached_callback'):
            self.path_planner.register_goal_reached_callback(self._on_goal_reached)
        
        # 地图管理器回调
        if hasattr(self.map_manager, 'register_map_update_callback'):
            self.map_manager.register_map_update_callback(self._on_map_updated)
    
    def start(self):
        """启动系统"""
        if self.status.is_running:
            logger.warning("[EnhancedIntegratedSLAM] 系统已在运行")
            return
        
        logger.info("[EnhancedIntegratedSLAM] 启动系统...")
        
        try:
            # 启动激光雷达
            self.lidar_adapter.start()
            self.status.lidar_connected = True
            
            # 启动处理线程
            self.processing_enabled = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # 启动可视化
            self.visualizer.start_rendering()
            
            # 更新状态
            self.status.is_running = True
            self.status.slam_active = True
            self.start_time = time.time()
            
            logger.info("[EnhancedIntegratedSLAM] 系统启动成功")
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 系统启动失败: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """关闭系统"""
        if not self.status.is_running:
            return
        
        logger.info("[EnhancedIntegratedSLAM] 正在关闭系统...")
        
        # 停止路径跟踪
        self.stop_following()
        
        # 停止处理线程
        self.processing_enabled = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 关闭激光雷达
        self.lidar_adapter.shutdown()
        
        # 关闭可视化
        self.visualizer.stop_rendering()
        
        # 关闭机器人控制
        if self.robot_controller:
            try:
                self.robot_controller.shutdown()
            except Exception as e:
                logger.error(f"[EnhancedIntegratedSLAM] 关闭机器人控制失败: {e}")
        
        # 保存最终数据
        try:
            self._save_final_data()
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 保存最终数据失败: {e}")
        
        # 更新状态
        self.status.is_running = False
        self.status.slam_active = False
        self.status.lidar_connected = False
        
        logger.info("[EnhancedIntegratedSLAM] 系统已关闭")
    
    def _processing_loop(self):
        """主处理循环"""
        logger.info("[EnhancedIntegratedSLAM] 处理循环开始")
        
        while self.processing_enabled:
            start_time = time.perf_counter()
            
            try:
                # 获取最新点云数据
                pointcloud_data = self.lidar_adapter.get_latest_pointcloud()
                
                if pointcloud_data:
                    self._process_pointcloud_data(pointcloud_data)
                
                # 更新路径跟踪
                if self.status.following_path:
                    self._update_path_following()
                
                # 更新性能统计
                processing_time = time.perf_counter() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                # 控制处理频率
                target_interval = 1.0 / self.config.update_frequency
                sleep_time = max(0, target_interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"[EnhancedIntegratedSLAM] 处理循环错误: {e}")
                time.sleep(0.1)
        
        logger.info("[EnhancedIntegratedSLAM] 处理循环结束")
    
    def _process_pointcloud_data(self, pointcloud_data: Dict):
        """处理点云数据 - 使用优化后的处理管道"""
        try:
            xyz = pointcloud_data['xyz']
            timestamp = pointcloud_data.get('timestamp', time.time() * 1e9)
            
            # 使用AdvancedSLAMProcessor处理
            result_grid = self.slam_processor.process_points(xyz, timestamp)
            
            # 更新系统状态
            with self.system_lock:
                self.status.frame_count += 1
                self.status.current_pose = self.slam_processor.get_current_pose()
                
                # 更新SLAM质量
                stats = self.slam_processor.get_comprehensive_statistics()
                self.status.slam_quality = stats.get('slam_processor', {}).get('average_quality', 0.0)
            
            # 更新性能统计
            current_time = time.time()
            if current_time - self.last_stats_update >= 1.0:
                self._update_performance_stats()
                self.last_stats_update = current_time
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 点云数据处理失败: {e}")
    
    def _update_path_following(self):
        """更新路径跟踪"""
        if not self.robot_controller or not self.status.following_path:
            return
        
        try:
            current_pose = self.status.current_pose
            if current_pose is None:
                return
            
            # 获取当前路径和目标
            current_path = self.path_planner.get_current_path()
            goal_pos = self.status.goal_position
            
            if not current_path or goal_pos is None:
                self.stop_following()
                return
            
            # 检查是否到达目标
            current_pos = current_pose[:3, 3][:2]
            goal_distance = np.linalg.norm(current_pos - goal_pos[:2])
            
            if goal_distance < self.config.planner.goal_tolerance:
                logger.info("[EnhancedIntegratedSLAM] 到达目标点")
                self.stop_following()
                return
            
            # 计算控制命令 (这里需要实现路径跟踪算法)
            # TODO: 集成路径跟踪控制器
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 路径跟踪更新失败: {e}")
    
    def _update_performance_stats(self):
        """更新性能统计"""
        if self.processing_times:
            avg_processing_time = np.mean(self.processing_times)
            self.status.processing_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        self.status.uptime = time.time() - self.start_time
    
    def _on_pointcloud_update(self):
        """点云数据更新回调"""
        pass  # 处理在主循环中进行
    
    def _on_goal_reached(self, goal_pos: np.ndarray):
        """目标到达回调"""
        logger.info(f"[EnhancedIntegratedSLAM] 到达目标: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
        self.stop_following()
    
    def _on_map_updated(self, map_data: np.ndarray):
        """地图更新回调"""
        pass  # 地图更新由可视化器处理
    
    def _save_final_data(self):
        """保存最终数据"""
        try:
            # 保存轨迹
            trajectory_path = self.slam_processor.save_trajectory("final_trajectory.txt")
            if trajectory_path:
                logger.info(f"[EnhancedIntegratedSLAM] 轨迹已保存: {trajectory_path}")
            
            # 保存地图
            map_path = self.map_manager.save_map("final_maps")
            if map_path:
                logger.info(f"[EnhancedIntegratedSLAM] 地图已保存: {map_path}")
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 保存最终数据失败: {e}")
    
    # ---------------------------------------------------------------------------
    # 公共接口
    # ---------------------------------------------------------------------------
    
    def set_goal(self, goal_x: float, goal_y: float):
        """设置目标位置"""
        goal_pos = np.array([goal_x, goal_y])
        
        with self.system_lock:
            self.status.goal_position = goal_pos
            self.status.planning_active = True
        
        # 通知路径规划器
        if hasattr(self.path_planner, 'set_goal'):
            self.path_planner.set_goal(goal_pos)
        
        # 通知可视化器
        self.visualizer.set_goal_position(goal_pos)
        
        logger.info(f"[EnhancedIntegratedSLAM] 设置目标: ({goal_x:.2f}, {goal_y:.2f})")
    
    def start_following(self):
        """开始路径跟踪"""
        with self.system_lock:
            self.status.following_path = True
        logger.info("[EnhancedIntegratedSLAM] 开始路径跟踪")
    
    def stop_following(self):
        """停止路径跟踪"""
        with self.system_lock:
            self.status.following_path = False
            self.status.planning_active = False
        
        if self.robot_controller:
            try:
                self.robot_controller.stop()
            except Exception as e:
                logger.error(f"[EnhancedIntegratedSLAM] 停止机器人失败: {e}")
        
        logger.info("[EnhancedIntegratedSLAM] 停止路径跟踪")
    
    def reset_slam(self):
        """重置SLAM系统"""
        try:
            self.slam_processor.reset()
            self.map_manager.reset()
            
            with self.system_lock:
                self.status.frame_count = 0
                self.status.current_pose = None
                self.status.goal_position = None
                self.status.slam_quality = 0.0
            
            logger.info("[EnhancedIntegratedSLAM] SLAM系统已重置")
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 重置SLAM失败: {e}")
    
    def save_data(self):
        """保存当前数据"""
        try:
            timestamp = int(time.time())
            
            # 保存轨迹
            trajectory_path = self.slam_processor.save_trajectory(f"trajectory_{timestamp}.txt")
            
            # 保存地图
            map_path = self.map_manager.save_map(f"maps_{timestamp}")
            
            # 保存截图
            screenshot_path = f"screenshot_{timestamp}.png"
            self.visualizer.save_screenshot(screenshot_path)
            
            logger.info(f"[EnhancedIntegratedSLAM] 数据已保存 (时间戳: {timestamp})")
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 保存数据失败: {e}")
    
    def get_system_status(self) -> SystemStatus:
        """获取系统状态"""
        with self.system_lock:
            # 更新动态状态
            current_pose = self.slam_processor.get_current_pose()
            self.status.current_pose = current_pose
            
            # 更新路径信息
            if hasattr(self.path_planner, 'get_current_path'):
                current_path = self.path_planner.get_current_path()
                self.status.path_length = len(current_path) if current_path else 0
            
            return SystemStatus(
                is_running=self.status.is_running,
                slam_active=self.status.slam_active,
                planning_active=self.status.planning_active,
                following_path=self.status.following_path,
                lidar_connected=self.status.lidar_connected,
                frame_count=self.status.frame_count,
                current_pose=self.status.current_pose.copy() if self.status.current_pose is not None else None,
                goal_position=self.status.goal_position.copy() if self.status.goal_position is not None else None,
                path_length=self.status.path_length,
                slam_quality=self.status.slam_quality,
                processing_fps=self.status.processing_fps,
                uptime=self.status.uptime
            )
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        try:
            # SLAM统计
            slam_stats = self.slam_processor.get_comprehensive_statistics()
            
            # 地图统计
            map_stats = self.map_manager.get_quality_metrics()
            
            # 激光雷达统计
            lidar_stats = self.lidar_adapter.get_statistics()
            
            # 可视化统计
            viz_stats = self.visualizer.get_rendering_stats()
            
            # 系统状态
            system_status = self.get_system_status()
            
            return {
                'slam': slam_stats,
                'map': map_stats,
                'lidar': lidar_stats,
                'visualization': viz_stats,
                'system': {
                    'frame_count': system_status.frame_count,
                    'processing_fps': system_status.processing_fps,
                    'uptime': system_status.uptime,
                    'slam_quality': system_status.slam_quality,
                    'is_running': system_status.is_running
                }
            }
            
        except Exception as e:
            logger.error(f"[EnhancedIntegratedSLAM] 获取统计信息失败: {e}")
            return {}
    
    @property
    def current_pose(self) -> Optional[np.ndarray]:
        """当前位姿属性 (兼容性)"""
        return self.status.current_pose

# 向后兼容别名
IntegratedSLAMSystem = EnhancedIntegratedSLAMSystem