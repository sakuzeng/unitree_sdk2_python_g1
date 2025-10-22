"""
集成系统主模块
"""
import threading
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

# 动态导入 Livox SDK
_SDK2 = False
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except ImportError:
    try:
        from livox_python import Livox as _Livox
        _SDK2 = False
    except ImportError as exc:
        print(f"[ERROR] Livox SDK 未找到: {exc}")
        exit(1)

from config import GridConfig, PathPlannerConfig
from odometry import OdometrySubscriber
from map_manager import StabilizedMapManager
from slam_processor import OptimizedSLAMProcessor
from path_planner import AStarPlanner
from robot_control import PurePursuitController, RobotController

class IntegratedSLAMSystem(_Livox):
    """集成SLAM路径规划系统"""
    
    def __init__(self, config_path: str, host_ip: str, grid_config: GridConfig, 
                 planner_config: PathPlannerConfig, interface: str = "eth0"):
        
        # 初始化 Livox SDK
        if _SDK2:
            super().__init__(config_path, host_ip=host_ip, frame_time=0.1, frame_packets=60)
        else:
            super().__init__()
        
        # 配置参数
        self.grid_config = grid_config
        self.planner_config = planner_config
        self.cell_size = grid_config.grid_size / grid_config.grid_resolution
        
        # 初始化各模块
        self.odometry = OdometrySubscriber(interface) if grid_config.use_odometry else None
        self.map_manager = StabilizedMapManager(grid_config)
        self.slam_processor = OptimizedSLAMProcessor(grid_config)
        self.path_planner = AStarPlanner(planner_config)
        self.controller = PurePursuitController(planner_config)
        self.robot_controller = RobotController(interface)
        
        # 系统状态
        self.frame_count = 0
        self.current_path = []
        self.current_path_index = 0
        self.goal_position = None
        self.is_planning = False
        self.is_following = False
        
        # 线程锁
        self.state_lock = threading.Lock()
        
        print("[IntegratedSystem] 系统初始化完成")
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """处理点云数据"""
        if len(xyz) == 0:
            return
        
        # 获取机器人位姿
        robot_pose = self._get_robot_pose()
        
        # SLAM处理
        local_grid = self.slam_processor.process_points(xyz, robot_pose)
        
        # 更新稳定的地图
        stable_grid = self.map_manager.update_with_local_grid(local_grid, robot_pose, self.frame_count)
        
        # 更新显示网格
        self.slam_processor.occupancy_grid = stable_grid
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"[IntegratedSystem] 处理帧: {self.frame_count}")
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理IMU数据"""
        pass
    
    def _get_robot_pose(self) -> np.ndarray:
        """获取机器人位姿"""
        if self.odometry:
            return self.odometry.get_pose()
        return np.eye(4)
    
    def set_goal(self, goal_x: float, goal_y: float):
        """设置目标点"""
        with self.state_lock:
            self.goal_position = (goal_x, goal_y)
            self.is_planning = True
            self.is_following = False
        
        self.robot_controller.stop()
        print(f"[IntegratedSystem] 设置目标: ({goal_x:.2f}, {goal_y:.2f})")
        
        # 异步路径规划
        threading.Thread(target=self._plan_path, daemon=True).start()
    
    def _plan_path(self):
        """执行路径规划"""
        if not self.goal_position or self.frame_count < 10:
            with self.state_lock:
                self.is_planning = False
            return
        
        # 获取当前网格和位置
        grid = self.slam_processor.occupancy_grid
        robot_pose = self._get_robot_pose()
        robot_grid_pos = self._get_robot_grid_position(robot_pose)
        goal_grid_pos = self._world_to_grid_position(self.goal_position)
        
        # 执行A*规划
        grid_path = self.path_planner.plan_path(grid, robot_grid_pos, goal_grid_pos)
        
        if grid_path:
            # 转换为世界坐标
            world_path = self._grid_to_world_path(grid_path)
            
            with self.state_lock:
                self.current_path = world_path
                self.current_path_index = 0
                self.is_planning = False
                self.is_following = True
            
            print(f"[IntegratedSystem] 路径规划完成: {len(world_path)} 点")
        else:
            with self.state_lock:
                self.is_planning = False
            print("[IntegratedSystem] 路径规划失败")
    
    def update_control(self):
        """更新路径跟踪控制"""
        if not self.is_following or not self.current_path:
            return
        
        robot_pose = self._get_robot_pose()
        robot_pos = robot_pose[:3, 3]
        robot_yaw = math.atan2(robot_pose[1, 0], robot_pose[0, 0])
        current_pose = np.array([robot_pos[0], robot_pos[1], robot_yaw])
        
        # 检查是否到达目标
        if self.goal_position:
            goal_dist = np.linalg.norm(current_pose[:2] - np.array(self.goal_position))
            if goal_dist < self.planner_config.goal_tolerance:
                self.stop_following()
                print("[IntegratedSystem] 到达目标")
                return
        
        # 计算控制指令
        linear_vel, angular_vel, updated_index = self.controller.compute_control(
            current_pose, self.current_path, self.current_path_index
        )
        
        with self.state_lock:
            self.current_path_index = updated_index
        
        self.robot_controller.move(linear_vel, angular_vel)
    
    def stop_following(self):
        """停止路径跟踪"""
        with self.state_lock:
            self.is_planning = False
            self.is_following = False
            self.goal_position = None
        self.robot_controller.stop()
    
    def _get_robot_grid_position(self, robot_pose: np.ndarray) -> Tuple[int, int]:
        """获取机器人网格位置"""
        if not self.slam_processor.origin_set:
            center = self.grid_config.grid_resolution // 2
            return (center, center)
        
        robot_world_pos = robot_pose[:3, 3][:2]
        grid_coords = self.slam_processor._world_to_grid_coord(robot_world_pos.reshape(1, -1))[0]
        return (grid_coords[0], grid_coords[1])
    
    def _world_to_grid_position(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        world_array = np.array([[world_pos[0], world_pos[1], 0]])
        grid_coords = self.slam_processor._world_to_grid_coord(world_array)[0]
        return (grid_coords[0], grid_coords[1])
    
    def _grid_to_world_path(self, grid_path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """网格路径转世界坐标路径"""
        world_path = []
        center = self.grid_config.grid_resolution // 2
        
        for gx, gy in grid_path:
            rel_x = (gx - center) * self.cell_size
            rel_y = -(gy - center) * self.cell_size
            world_x = rel_x + self.slam_processor.origin[0]
            world_y = rel_y + self.slam_processor.origin[1]
            world_path.append((world_x, world_y))
        
        return world_path
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        with self.state_lock:
            return {
                'frame_count': self.frame_count,
                'is_planning': self.is_planning,
                'is_following': self.is_following,
                'goal_position': self.goal_position,
                'path_length': len(self.current_path),
                'motion_state': self.map_manager.motion_state
            }
    
    def save_map(self) -> str:
        """保存完整地图"""
        return self.map_manager.save_complete_map()