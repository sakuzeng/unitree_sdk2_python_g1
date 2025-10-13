"""
机器人控制模块
"""
import math
import numpy as np
from typing import List, Tuple, Optional

from config import PathPlannerConfig

# Unitree SDK 导入
UNITREE_SDK_AVAILABLE = False
try:
    from unitree_sdk2py.go2.sport.sport_client import SportClient
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    pass

class PurePursuitController:
    """Pure Pursuit 路径跟踪控制器"""
    
    def __init__(self, config: PathPlannerConfig):
        self.config = config
    
    def compute_control(self, current_pose: np.ndarray, path: List[Tuple[float, float]], 
                        current_index: int) -> Tuple[float, float, int]:
        """计算控制指令"""
        if not path or current_index >= len(path):
            return 0.0, 0.0, current_index
        
        # 寻找前瞻点
        lookahead_point, updated_index = self._find_lookahead_point(
            current_pose[:2], path, current_index
        )
        
        if lookahead_point is None:
            return 0.0, 0.0, updated_index
        
        # 计算控制量
        dx = lookahead_point[0] - current_pose[0]
        dy = lookahead_point[1] - current_pose[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        target_angle = math.atan2(dy, dx)
        angle_error = self._normalize_angle(target_angle - current_pose[2])
        
        linear_velocity = min(self.config.max_velocity, distance * 2.0)
        angular_velocity = np.clip(angle_error * 2.0, 
                                   -self.config.max_angular_velocity, 
                                   self.config.max_angular_velocity)
        
        # 大角度转弯时减速
        if abs(angle_error) > math.pi / 4:
            linear_velocity *= 0.5
        
        return linear_velocity, angular_velocity, updated_index
    
    def _find_lookahead_point(self, current_pos: np.ndarray, path: List[Tuple[float, float]], 
                              start_index: int) -> Tuple[Optional[Tuple[float, float]], int]:
        """寻找前瞻点"""
        best_point = None
        best_index = start_index
        
        for i in range(start_index, len(path)):
            point = path[i]
            distance = math.sqrt((point[0] - current_pos[0])**2 + (point[1] - current_pos[1])**2)
            
            if distance >= self.config.lookahead_distance:
                best_point = point
                best_index = i
                break
            else:
                best_point = point
                best_index = i
        
        if best_point is None and path:
            best_point = path[-1]
            best_index = len(path) - 1
        
        return best_point, best_index
    
    def _normalize_angle(self, angle: float) -> float:
        """角度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

class RobotController:
    """机器人运动控制器"""
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.sport_client = None
        self.is_connected = False
        
        if UNITREE_SDK_AVAILABLE:
            try:
                self.sport_client = SportClient()
                self.sport_client.SetTimeout(3.0)
                self.sport_client.Init()
                self.is_connected = True
                print(f"[RobotController] 机器人控制器已初始化")
            except Exception as e:
                print(f"[ERROR] 初始化机器人控制器失败: {e}")
        else:
            print(f"[RobotController] 模拟模式运行")
    
    def move(self, linear_velocity: float, angular_velocity: float):
        """控制机器人移动"""
        if not self.is_connected or not self.sport_client:
            return
        
        try:
            self.sport_client.Move(linear_velocity, 0.0, angular_velocity)
        except Exception as e:
            print(f"[ERROR] 机器人移动控制失败: {e}")
    
    def stop(self):
        """停止机器人"""
        self.move(0.0, 0.0)