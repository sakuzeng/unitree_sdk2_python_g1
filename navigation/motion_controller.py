#!/usr/bin/env python3
"""
机器人运动控制模块
将导航指令转换为机器人具体的控制指令
"""

import time
import numpy as np
import math
from typing import Tuple, Optional
from dataclasses import dataclass
import threading

# 导入机器人控制SDK
try:
    from unitree_sdk2_python.core.channel import ChannelPublisher, ChannelFactoryInitialize
    from unitree_sdk2_python.idl.default import unitree_go_msg_dds__SportModeCmd_
    from unitree_sdk2_python.idl.unitree_go.msg.dds_ import SportModeCmd_
    UNITREE_AVAILABLE = True
except ImportError:
    print("[WARNING] Unitree SDK 不可用，使用模拟模式")
    UNITREE_AVAILABLE = False

@dataclass
class MotionConfig:
    """运动控制配置参数"""
    max_linear_velocity: float = 1.0    # 最大线速度 (m/s)
    max_angular_velocity: float = 1.0   # 最大角速度 (rad/s)
    velocity_smoothing: float = 0.8     # 速度平滑系数 (0-1)
    control_frequency: float = 20.0     # 控制频率 (Hz)
    emergency_stop_threshold: float = 0.1  # 紧急停车阈值
    
    # PID参数
    linear_kp: float = 1.0
    linear_ki: float = 0.1
    linear_kd: float = 0.05
    angular_kp: float = 1.0
    angular_ki: float = 0.1
    angular_kd: float = 0.05

class PIDController:
    """PID控制器"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
    
    def update(self, error: float) -> float:
        """更新PID控制器"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0.0:
            return 0.0
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        integral_term = self.ki * self.integral
        
        # 微分项
        derivative = (error - self.prev_error) / dt
        derivative_term = self.kd * derivative
        
        # 总输出
        output = proportional + integral_term + derivative_term
        
        # 输出限制
        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 更新状态
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """重置PID控制器"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

class RobotMotionController:
    """机器人运动控制器"""
    
    def __init__(self, config: MotionConfig = None, network_interface: str = "eth0"):
        self.config = config or MotionConfig()
        self.network_interface = network_interface
        
        # 控制状态
        self.is_running = False
        self.emergency_stop = False
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        
        # PID控制器
        self.linear_pid = PIDController(
            self.config.linear_kp, self.config.linear_ki, self.config.linear_kd,
            self.config.max_linear_velocity
        )
        self.angular_pid = PIDController(
            self.config.angular_kp, self.config.angular_ki, self.config.angular_kd,
            self.config.max_angular_velocity
        )
        
        # 机器人控制接口
        self.robot_publisher = None
        self.control_thread = None
        self.control_lock = threading.Lock()
        
        # 初始化机器人连接
        self._initialize_robot_connection()
    
    def _initialize_robot_connection(self):
        """初始化机器人连接"""
        if not UNITREE_AVAILABLE:
            print("[INFO] 运动控制器运行在模拟模式")
            return
        
        try:
            # 初始化DDS
            ChannelFactoryInitialize(0, self.network_interface)
            
            # 创建运动控制发布者
            self.robot_publisher = ChannelPublisher(
                "rt/sportmodecommand", SportModeCmd_
            )
            
            print(f"[INFO] 机器人控制连接成功 (网络接口: {self.network_interface})")
            
        except Exception as e:
            print(f"[ERROR] 机器人控制连接失败: {e}")
            print("[INFO] 运动控制器将运行在模拟模式")
            self.robot_publisher = None
    
    def start(self):
        """启动运动控制器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.emergency_stop = False
        
        # 重置PID控制器
        self.linear_pid.reset()
        self.angular_pid.reset()
        
        # 启动控制线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("[INFO] 运动控制器已启动")
    
    def stop(self):
        """停止运动控制器"""
        self.is_running = False
        
        # 立即停车
        self.set_velocity(0.0, 0.0)
        
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        print("[INFO] 运动控制器已停止")
    
    def emergency_stop_enable(self):
        """启用紧急停车"""
        with self.control_lock:
            self.emergency_stop = True
            self.target_linear_vel = 0.0
            self.target_angular_vel = 0.0
        print("[WARNING] 紧急停车已启用")
    
    def emergency_stop_disable(self):
        """禁用紧急停车"""
        with self.control_lock:
            self.emergency_stop = False
        print("[INFO] 紧急停车已禁用")
    
    def set_velocity(self, linear_vel: float, angular_vel: float):
        """设置目标速度"""
        with self.control_lock:
            if self.emergency_stop:
                linear_vel = 0.0
                angular_vel = 0.0
            
            # 速度限制
            linear_vel = np.clip(linear_vel, 
                                -self.config.max_linear_velocity, 
                                self.config.max_linear_velocity)
            angular_vel = np.clip(angular_vel,
                                 -self.config.max_angular_velocity,
                                 self.config.max_angular_velocity)
            
            self.target_linear_vel = linear_vel
            self.target_angular_vel = angular_vel
    
    def get_current_velocity(self) -> Tuple[float, float]:
        """获取当前速度"""
        with self.control_lock:
            return self.current_linear_vel, self.current_angular_vel
    
    def _control_loop(self):
        """控制循环"""
        control_period = 1.0 / self.config.control_frequency
        
        while self.is_running:
            start_time = time.time()
            
            try:
                with self.control_lock:
                    target_linear = self.target_linear_vel
                    target_angular = self.target_angular_vel
                
                # 速度平滑
                smoothing = self.config.velocity_smoothing
                self.current_linear_vel = (smoothing * self.current_linear_vel + 
                                          (1 - smoothing) * target_linear)
                self.current_angular_vel = (smoothing * self.current_angular_vel + 
                                           (1 - smoothing) * target_angular)
                
                # 发送控制指令
                self._send_robot_command(self.current_linear_vel, self.current_angular_vel)
                
            except Exception as e:
                print(f"[ERROR] 控制循环错误: {e}")
            
            # 控制频率
            elapsed_time = time.time() - start_time
            sleep_time = max(0, control_period - elapsed_time)
            time.sleep(sleep_time)
    
    def _send_robot_command(self, linear_vel: float, angular_vel: float):
        """发送机器人控制指令"""
        if self.robot_publisher is None:
            # 模拟模式：仅打印调试信息
            if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
                print(f"[SIMULATION] 运动指令: 线速度={linear_vel:.3f} m/s, "
                      f"角速度={angular_vel:.3f} rad/s")
            return
        
        try:
            # 创建运动指令
            cmd = SportModeCmd_()
            cmd.mode = 2  # 速度控制模式
            
            # 设置速度指令 (需要根据具体机器人调整)
            cmd.gait_type = 1  # 步态类型
            cmd.speed_level = 0  # 速度等级
            
            # 速度指令 (m/s 和 rad/s)
            cmd.velocity = [float(linear_vel), 0.0, float(angular_vel)]  # [vx, vy, wz]
            cmd.yaw_speed = float(angular_vel)
            
            # 位置指令设为0（纯速度控制）
            cmd.position = [0.0, 0.0]
            cmd.euler = [0.0, 0.0, 0.0]
            
            # 发布指令
            self.robot_publisher.write(cmd)
            
        except Exception as e:
            print(f"[ERROR] 发送机器人指令失败: {e}")
    
    def get_status(self) -> dict:
        """获取控制器状态"""
        with self.control_lock:
            return {
                'is_running': self.is_running,
                'emergency_stop': self.emergency_stop,
                'current_linear_vel': self.current_linear_vel,
                'current_angular_vel': self.current_angular_vel,
                'target_linear_vel': self.target_linear_vel,
                'target_angular_vel': self.target_angular_vel
            }