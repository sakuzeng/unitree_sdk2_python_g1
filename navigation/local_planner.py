#!/usr/bin/env python3
"""
局部路径规划和动态避障模块
实现DWA (Dynamic Window Approach) 算法
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

@dataclass
class RobotState:
    """机器人状态"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 朝向角 (弧度)
    v: float = 0.0      # 线速度 (m/s)
    w: float = 0.0      # 角速度 (rad/s)

@dataclass
class DWAConfig:
    """DWA算法配置参数"""
    # 机器人动力学约束
    max_linear_vel: float = 1.0     # 最大线速度 (m/s)
    max_angular_vel: float = 1.0    # 最大角速度 (rad/s)
    max_linear_acc: float = 2.0     # 最大线加速度 (m/s²)
    max_angular_acc: float = 2.0    # 最大角加速度 (rad/s²)
    
    # 速度分辨率
    linear_vel_resolution: float = 0.1   # 线速度采样分辨率
    angular_vel_resolution: float = 0.1  # 角速度采样分辨率
    
    # 预测时间
    predict_time: float = 3.0  # 轨迹预测时间 (s)
    dt: float = 0.1           # 时间步长 (s)
    
    # 评价函数权重
    goal_weight: float = 1.0      # 目标导向权重
    obstacle_weight: float = 2.0  # 避障权重
    velocity_weight: float = 0.5  # 速度权重
    
    # 安全参数
    robot_radius: float = 0.3    # 机器人半径 (m)
    safety_distance: float = 0.5  # 安全距离 (m)

class DynamicWindowApproach:
    """动态窗口算法实现"""
    
    def __init__(self, config: DWAConfig = None):
        self.config = config or DWAConfig()
        self.last_trajectory = None
        
    def plan_velocity(self, robot_state: RobotState, goal: Tuple[float, float],
                      obstacles: List[Tuple[float, float]], 
                      global_path: List[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        规划最优速度指令
        
        Args:
            robot_state: 当前机器人状态
            goal: 目标点 (x, y)
            obstacles: 障碍物位置列表 [(x1,y1), (x2,y2), ...]
            global_path: 全局路径（可选，用于路径跟踪）
            
        Returns:
            (linear_velocity, angular_velocity) 速度指令
        """
        # 1. 计算动态窗口
        dw = self._calculate_dynamic_window(robot_state)
        
        # 2. 生成速度候选
        velocity_candidates = self._generate_velocity_samples(dw)
        
        # 3. 评价每个速度候选
        best_v, best_w = 0.0, 0.0
        best_score = float('-inf')
        
        for v, w in velocity_candidates:
            # 预测轨迹
            trajectory = self._predict_trajectory(robot_state, v, w)
            
            # 检查碰撞
            if self._check_collision(trajectory, obstacles):
                continue
            
            # 计算评价分数
            score = self._evaluate_trajectory(trajectory, goal, obstacles, global_path)
            
            if score > best_score:
                best_score = score
                best_v, best_w = v, w
                self.last_trajectory = trajectory
        
        # 4. 安全检查
        if best_score == float('-inf'):
            print("[WARNING] DWA: 未找到安全速度，执行紧急停车")
            return 0.0, 0.0
        
        return best_v, best_w
    
    def _calculate_dynamic_window(self, robot_state: RobotState) -> Dict[str, float]:
        """计算动态窗口约束"""
        cfg = self.config
        
        # 机器人动力学约束
        v_min = 0.0
        v_max = cfg.max_linear_vel
        w_min = -cfg.max_angular_vel
        w_max = cfg.max_angular_vel
        
        # 加速度约束
        v_acc_min = robot_state.v - cfg.max_linear_acc * cfg.dt
        v_acc_max = robot_state.v + cfg.max_linear_acc * cfg.dt
        w_acc_min = robot_state.w - cfg.max_angular_acc * cfg.dt
        w_acc_max = robot_state.w + cfg.max_angular_acc * cfg.dt
        
        # 动态窗口
        dw = {
            'v_min': max(v_min, v_acc_min),
            'v_max': min(v_max, v_acc_max),
            'w_min': max(w_min, w_acc_min),
            'w_max': min(w_max, w_acc_max)
        }
        
        return dw
    
    def _generate_velocity_samples(self, dw: Dict[str, float]) -> List[Tuple[float, float]]:
        """生成速度采样点"""
        cfg = self.config
        samples = []
        
        v_range = np.arange(dw['v_min'], dw['v_max'] + cfg.linear_vel_resolution, 
                           cfg.linear_vel_resolution)
        w_range = np.arange(dw['w_min'], dw['w_max'] + cfg.angular_vel_resolution,
                           cfg.angular_vel_resolution)
        
        for v in v_range:
            for w in w_range:
                samples.append((float(v), float(w)))
        
        return samples
    
    def _predict_trajectory(self, robot_state: RobotState, v: float, w: float) -> List[RobotState]:
        """预测机器人轨迹"""
        cfg = self.config
        trajectory = []
        
        # 初始状态
        state = RobotState(robot_state.x, robot_state.y, robot_state.theta, v, w)
        
        # 数值积分预测
        for _ in range(int(cfg.predict_time / cfg.dt)):
            # 运动学模型 (差分驱动)
            state.x += state.v * math.cos(state.theta) * cfg.dt
            state.y += state.v * math.sin(state.theta) * cfg.dt
            state.theta += state.w * cfg.dt
            
            trajectory.append(RobotState(state.x, state.y, state.theta, state.v, state.w))
        
        return trajectory
    
    def _check_collision(self, trajectory: List[RobotState], 
                         obstacles: List[Tuple[float, float]]) -> bool:
        """检查轨迹是否与障碍物碰撞"""
        cfg = self.config
        
        for state in trajectory:
            for obs_x, obs_y in obstacles:
                distance = math.sqrt((state.x - obs_x)**2 + (state.y - obs_y)**2)
                if distance <= cfg.robot_radius + cfg.safety_distance:
                    return True
        return False
    
    def _evaluate_trajectory(self, trajectory: List[RobotState], goal: Tuple[float, float],
                            obstacles: List[Tuple[float, float]], 
                            global_path: List[Tuple[float, float]] = None) -> float:
        """评价轨迹质量"""
        if not trajectory:
            return float('-inf')
        
        cfg = self.config
        final_state = trajectory[-1]
        
        # 1. 目标导向评价 (距离目标越近越好)
        goal_distance = math.sqrt((final_state.x - goal[0])**2 + (final_state.y - goal[1])**2)
        goal_score = 1.0 / (goal_distance + 0.1)  # 避免除零
        
        # 2. 障碍物距离评价 (离障碍物越远越好)
        min_obstacle_dist = float('inf')
        for state in trajectory:
            for obs_x, obs_y in obstacles:
                dist = math.sqrt((state.x - obs_x)**2 + (state.y - obs_y)**2)
                min_obstacle_dist = min(min_obstacle_dist, dist)
        
        obstacle_score = min_obstacle_dist / (cfg.robot_radius + cfg.safety_distance)
        obstacle_score = min(obstacle_score, 1.0)  # 限制最大值
        
        # 3. 速度评价 (速度越大越好，鼓励前进)
        velocity_score = final_state.v / cfg.max_linear_vel
        
        # 4. 路径跟踪评价 (如果有全局路径)
        path_score = 0.0
        if global_path and len(global_path) > 1:
            path_score = self._calculate_path_following_score(final_state, global_path)
        
        # 综合评价
        total_score = (cfg.goal_weight * goal_score +
                      cfg.obstacle_weight * obstacle_score +
                      cfg.velocity_weight * velocity_score +
                      0.5 * path_score)  # 路径跟踪权重
        
        return total_score
    
    def _calculate_path_following_score(self, state: RobotState, 
                                       global_path: List[Tuple[float, float]]) -> float:
        """计算路径跟踪评分"""
        if len(global_path) < 2:
            return 0.0
        
        # 找到最近的路径点
        min_distance = float('inf')
        for px, py in global_path:
            distance = math.sqrt((state.x - px)**2 + (state.y - py)**2)
            min_distance = min(min_distance, distance)
        
        # 距离路径越近评分越高
        path_score = 1.0 / (min_distance + 0.1)
        return min(path_score, 1.0)
    
    def get_last_trajectory(self) -> Optional[List[RobotState]]:
        """获取最后一次规划的轨迹（用于可视化）"""
        return self.last_trajectory