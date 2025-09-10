#!/usr/bin/env python3
"""
修复后的 Dex3 灵巧手控制测试程序

修复了 MotorCmd_ 构造函数参数问题、关节限位检查问题和 tuple 调用错误。
支持从校准结果加载关节限位数据。
"""

import sys
import argparse
import time
import json
import csv
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import math

# 导入启动序列模块
from hanger_boot_sequence import hanger_boot_sequence

# 导入必要的 SDK 模块
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorCmd_
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"警告: Unitree SDK 不可用: {e}")
    print("将运行在模拟模式下")
    SDK_AVAILABLE = False

# ========================= 关节限位加载器 =========================

class JointLimitsLoader:
    """关节限位数据加载器"""
    
    @staticmethod
    def load_from_calibration_json(config_file: str, hand: str = "right") -> Optional[List[Tuple[float, float]]]:
        """从校准JSON文件加载关节限位"""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"校准配置文件不存在: {config_file}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证是否是正确的手部配置
            if data.get('hand') != hand:
                print(f"警告: 配置文件是为 {data.get('hand')} 手，当前需要 {hand} 手限位")
            
            joint_limits = []
            
            # 按关节ID排序
            limits_data = sorted(data['joint_limits'], key=lambda x: x['joint_id'])
            
            for limit_data in limits_data:
                min_angle = float(limit_data['min_angle'])
                max_angle = float(limit_data['max_angle'])
                joint_limits.append((min_angle, max_angle))
                
                print(f"加载关节{limit_data['joint_id']} ({limit_data['joint_name']}): "
                      f"[{min_angle:.3f}, {max_angle:.3f}] rad "
                      f"(基于 {limit_data['samples_count']} 个样本)")
            
            if len(joint_limits) == 7:
                print(f"✓ 成功从 {config_file} 加载了 {hand} 手关节限位")
                return joint_limits
            else:
                print(f"错误: 限位数据不完整，仅有 {len(joint_limits)} 个关节")
                return None
                
        except Exception as e:
            print(f"加载校准配置失败: {e}")
            return None
    
    @staticmethod
    def load_from_calibration_csv(csv_file: str, hand: str = "right") -> Optional[List[Tuple[float, float]]]:
        """从校准CSV文件加载关节限位"""
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"校准CSV文件不存在: {csv_file}")
            return None
        
        try:
            joint_limits = [None] * 7  # 预分配7个位置
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    joint_id = int(row['joint_id'])
                    if 0 <= joint_id < 7:
                        min_angle = float(row['min_angle'])
                        max_angle = float(row['max_angle'])
                        joint_limits[joint_id] = (min_angle, max_angle)
                        
                        print(f"加载关节{joint_id} ({row['joint_name']}): "
                              f"[{min_angle:.3f}, {max_angle:.3f}] rad "
                              f"(基于 {row['samples_count']} 个样本)")
            
            # 检查是否所有关节都有数据
            if all(limit is not None for limit in joint_limits):
                print(f"✓ 成功从 {csv_file} 加载了 {hand} 手关节限位")
                return joint_limits
            else:
                print(f"错误: CSV限位数据不完整")
                return None
                
        except Exception as e:
            print(f"加载CSV配置失败: {e}")
            return None
    
    @staticmethod
    def load_from_python_module(module_path: str) -> Optional[List[Tuple[float, float]]]:
        """从生成的Python模块加载关节限位"""
        try:
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("joint_limits_module", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'get_dex3_joint_limits'):
                joint_limits = module.get_dex3_joint_limits()
                print(f"✓ 成功从 {module_path} 加载关节限位")
                for i, (min_a, max_a) in enumerate(joint_limits):
                    range_deg = (max_a - min_a) * 180.0 / math.pi
                    print(f"  关节{i}: [{min_a:.3f}, {max_a:.3f}] rad ({range_deg:.1f}°)")
                return joint_limits
            else:
                print(f"Python模块 {module_path} 中未找到 get_dex3_joint_limits 函数")
                return None
                
        except Exception as e:
            print(f"加载Python模块失败: {e}")
            return None
    
    @staticmethod
    def auto_load_limits(hand: str = "right", config_dir: str = "config") -> Optional[List[Tuple[float, float]]]:
        """自动查找并加载关节限位配置"""
        config_path = Path(config_dir)
        
        # 优先级顺序：JSON > Python模块 > CSV
        candidates = [
            (config_path / f"dex3_{hand}_joint_limits.json", "json"),
            (config_path / f"dex3_{hand}_limits.py", "python"),
            (config_path / f"dex3_{hand}_joint_limits.csv", "csv"),
        ]
        
        for file_path, file_type in candidates:
            if file_path.exists():
                print(f"找到校准配置文件: {file_path}")
                
                if file_type == "json":
                    return JointLimitsLoader.load_from_calibration_json(str(file_path), hand)
                elif file_type == "python":
                    return JointLimitsLoader.load_from_python_module(str(file_path))
                elif file_type == "csv":
                    return JointLimitsLoader.load_from_calibration_csv(str(file_path), hand)
        
        print(f"错误: 未找到 {hand} 手的校准配置文件")
        print(f"请确保在 {config_dir}/ 目录下存在以下文件之一:")
        print(f"  - dex3_{hand}_joint_limits.json")
        print(f"  - dex3_{hand}_limits.py")
        print(f"  - dex3_{hand}_joint_limits.csv")
        print(f"请先运行校准程序生成关节限位配置文件")
        return None
    
    @staticmethod
    def validate_limits(joint_limits: List[Tuple[float, float]]) -> bool:
        """验证关节限位数据的有效性"""
        if len(joint_limits) != 7:
            print(f"错误: 关节限位数量不正确，需要7个，实际 {len(joint_limits)} 个")
            return False
        
        for i, (min_val, max_val) in enumerate(joint_limits):
            if min_val >= max_val:
                print(f"错误: 关节{i} 最小值 {min_val} >= 最大值 {max_val}")
                return False
            
            # 检查合理范围（关节角度一般在 -π 到 π 之间）
            if abs(min_val) > math.pi or abs(max_val) > math.pi:
                print(f"警告: 关节{i} 角度超出常规范围 [-π, π]: [{min_val:.3f}, {max_val:.3f}]")
        
        return True

# ========================= 增强的 Dex3 客户端 =========================

class FixedDex3Client:
    """修复后的 Dex3 客户端，支持从校准数据加载关节限位"""
    
    def __init__(self, hand: str = "right", interface: str = "eth0", 
                 config_dir: str = "config", limits_file: str = None):
        if hand not in ["left", "right"]:
            raise ValueError("hand 必须是 'left' 或 'right'")
        
        self.hand = hand
        self.interface = interface
        self.is_connected = False
        
        # DDS 通信
        self._cmd_topic = f"rt/dex3/{hand}/cmd"
        self._state_topic = f"rt/dex3/{hand}/state"
        self._cmd_publisher: Optional[ChannelPublisher] = None
        self._state_subscriber: Optional[ChannelSubscriber] = None
        
        # 状态数据
        self._latest_state = None
        self._state_lock = threading.Lock()
        
        # 加载关节限位
        self.joint_limits = None
        if limits_file:
            # 从指定文件加载
            if limits_file.endswith('.json'):
                self.joint_limits = JointLimitsLoader.load_from_calibration_json(limits_file, hand)
            elif limits_file.endswith('.csv'):
                self.joint_limits = JointLimitsLoader.load_from_calibration_csv(limits_file, hand)
            elif limits_file.endswith('.py'):
                self.joint_limits = JointLimitsLoader.load_from_python_module(limits_file)
            else:
                print(f"不支持的限位文件格式: {limits_file}")
                raise ValueError(f"不支持的限位文件格式: {limits_file}")
        else:
            # 自动查找配置文件
            self.joint_limits = JointLimitsLoader.auto_load_limits(hand, config_dir)
        
        # 验证限位数据
        if self.joint_limits is None:
            raise RuntimeError(f"无法加载 {hand} 手的关节限位配置，请先运行校准程序")
        
        if not JointLimitsLoader.validate_limits(self.joint_limits):
            raise RuntimeError("关节限位数据验证失败")
        
        # 默认控制参数
        self.default_kp = 8.0
        self.default_kd = 1.5
        
        # 初始化连接
        if SDK_AVAILABLE:
            self._init_connection()
    
    def _init_connection(self):
        """初始化DDS连接"""
        try:
            # 确保先初始化 DDS
            ChannelFactoryInitialize(0, self.interface)
            
            # 创建发布者
            self._cmd_publisher = ChannelPublisher(self._cmd_topic, HandCmd_)
            self._cmd_publisher.Init()
            
            # 创建订阅者
            self._state_subscriber = ChannelSubscriber(self._state_topic, HandState_)
            self._state_subscriber.Init(self._state_callback, 100)
            
            self.is_connected = True
            print(f"✓ {self.hand}手连接成功")
            
        except Exception as e:
            print(f"✗ {self.hand}手连接失败: {e}")
            traceback.print_exc()
            self.is_connected = False
    
    def _state_callback(self, msg):
        """状态消息回调"""
        with self._state_lock:
            self._latest_state = msg
    
    def _pack_mode(self, motor_id: int, status: int = 0x01, timeout: bool = True) -> int:
        """打包电机控制模式"""
        mode = 0
        mode |= (motor_id & 0x0F)
        mode |= (status & 0x07) << 4
        mode |= (int(timeout) & 0x01) << 7
        return mode
    
    def _create_motor_cmd(self, motor_id: int, angle: float, kp: float, kd: float):
        """创建单个电机命令 - 修复版本"""
        try:
            # 修复：MotorCmd_ 需要所有参数作为位置参数
            motor_cmd = MotorCmd_(
                self._pack_mode(motor_id),  # mode
                float(angle),               # q
                0.0,                       # dq
                0.0,                       # tau
                float(kp),                 # kp
                float(kd),                 # kd
                [0, 0, 0]                  # reserve
            )
            return motor_cmd
            
        except Exception as e:
            print(f"[Dex3] 创建电机命令失败 (关节{motor_id}): {e}")
            print(f"[Dex3] MotorCmd_ 类型: {type(MotorCmd_)}")
            traceback.print_exc()
            return None
    
    def _create_hand_command(self, angles: List[float], kp: float = None, kd: float = None):
        """创建手部控制命令 - 修复版本"""
        if not SDK_AVAILABLE:
            return None
        
        kp = kp or self.default_kp
        kd = kd or self.default_kd
        
        # 使用校准的关节限位进行角度限制
        limited_angles = []
        for i, angle in enumerate(angles):
            if i < len(self.joint_limits):
                min_val, max_val = self.joint_limits[i]
                if angle < min_val or angle > max_val:
                    limited_angle = max(min_val, min(max_val, angle))
                    print(f"[Dex3] 警告: 关节{i} 角度 {angle:.3f} 超出校准限位 [{min_val:.3f}, {max_val:.3f}]，限制为 {limited_angle:.3f}")
                    limited_angles.append(limited_angle)
                else:
                    limited_angles.append(angle)
            else:
                limited_angles.append(angle)
        
        try:
            # 创建手部命令消息
            motor_cmds = []
            for i in range(7):
                motor_cmd = self._create_motor_cmd(i, limited_angles[i], kp, kd)
                if motor_cmd is None:
                    print(f"[Dex3] 创建电机{i}命令失败")
                    return None
                motor_cmds.append(motor_cmd)
            
            # 修复：HandCmd_ 也需要位置参数
            hand_cmd = HandCmd_(
                motor_cmds,        # motor_cmd
                [0, 0, 0, 0]      # reserve
            )
            
            return hand_cmd
            
        except Exception as e:
            print(f"[Dex3] 创建手部命令失败: {e}")
            traceback.print_exc()
            return None
    
    def set_joint_angles(self, angles: List[float], kp: float = None, kd: float = None) -> bool:
        """设置关节角度 - 修复版本"""
        if not self.is_connected or not SDK_AVAILABLE:
            print(f"[Dex3] 模拟模式: 设置{self.hand}手关节角度: {[f'{a:.2f}' for a in angles]}")
            return True
        
        if len(angles) != 7:
            print(f"[Dex3] 错误: angles 必须包含7个元素，当前为{len(angles)}个")
            return False
        
        try:
            cmd = self._create_hand_command(angles, kp, kd)
            if cmd is not None:
                # 检查发布者是否可用
                if self._cmd_publisher is None:
                    print(f"[Dex3] 错误: 命令发布者未初始化")
                    return False
                
                # 发送命令
                self._cmd_publisher.Write(cmd)
                print(f"[Dex3] {self.hand}手命令发送成功")
                return True
            else:
                print(f"[Dex3] {self.hand}手命令创建失败")
                return False
                
        except Exception as e:
            print(f"[Dex3] 设置关节角度失败: {e}")
            traceback.print_exc()
            return False
    
    def get_current_state(self, timeout: float = 0.1):
        """获取当前状态"""
        if not SDK_AVAILABLE:
            # 模拟状态数据
            return {
                'joint_angles': [0.0] * 7,
                'joint_velocities': [0.0] * 7,
                'pressure_data': {}
            }
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._state_lock:
                if self._latest_state is not None:
                    return self._latest_state
            time.sleep(0.001)
        
        return None
    
    def get_joint_angles(self, timeout: float = 0.1) -> Optional[List[float]]:
        """获取当前关节角度"""
        state = self.get_current_state(timeout)
        
        if not SDK_AVAILABLE:
            return [0.0] * 7
        
        if state and hasattr(state, 'motor_state') and len(state.motor_state) >= 7:
            try:
                return [float(ms.q) for ms in state.motor_state[:7]]
            except Exception as e:
                print(f"[Dex3] 解析关节角度失败: {e}")
        
        return None
    
    def get_pressure_data(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """获取压力传感器数据"""
        state = self.get_current_state(timeout)
        
        if not SDK_AVAILABLE:
            # 模拟压力数据
            return {
                'sensor_0': {'pressure': [0.0] * 12, 'valid_contacts': 0, 'lost': False},
                'sensor_1': {'pressure': [0.0] * 12, 'valid_contacts': 0, 'lost': False},
                'sensor_2': {'pressure': [0.0] * 12, 'valid_contacts': 0, 'lost': False},
            }
        
        if state and hasattr(state, 'press_sensor_state'):
            try:
                pressure_data = {}
                for i, sensor in enumerate(state.press_sensor_state):
                    pressure_data[f'sensor_{i}'] = {
                        'pressure': list(sensor.pressure),
                        'temperature': list(sensor.temperature) if hasattr(sensor, 'temperature') else [],
                        'valid_contacts': sum(1 for p in sensor.pressure if p > 100000),
                        'lost': sensor.lost if hasattr(sensor, 'lost') else False
                    }
                return pressure_data
            except Exception as e:
                print(f"[Dex3] 解析压力数据失败: {e}")
        
        return None
    
    def stop_motors(self):
        """停止所有电机"""
        if not self.is_connected or not SDK_AVAILABLE:
            print(f"[Dex3] 模拟模式: 停止{self.hand}手电机")
            return
        
        try:
            # 发送阻尼控制命令
            current_angles = self.get_joint_angles(timeout=0.1) or [0.0] * 7
            cmd = self._create_hand_command(current_angles, kp=0.0, kd=0.5)
            if cmd is not None and self._cmd_publisher is not None:
                self._cmd_publisher.Write(cmd)
                print(f"[Dex3] {self.hand}手电机已停止")
            else:
                print(f"[Dex3] 停止{self.hand}手电机失败: 命令创建或发布者不可用")
        except Exception as e:
            print(f"[Dex3] 停止电机失败: {e}")
            traceback.print_exc()
    
    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """获取关节限位"""
        return self.joint_limits.copy()
    
    def print_joint_limits_info(self):
        """打印关节限位信息"""
        print(f"\n=== {self.hand.capitalize()} 手关节限位信息 ===")
        joint_names = [
            "thumb_0 (拇指旋转)",
            "thumb_1 (拇指弯曲1)",
            "thumb_2 (拇指弯曲2)",
            "middle_0 (中指弯曲1)",
            "middle_1 (中指弯曲2)",
            "index_0 (食指弯曲1)",
            "index_1 (食指弯曲2)",
        ]
        
        for i, ((min_val, max_val), name) in enumerate(zip(self.joint_limits, joint_names)):
            range_rad = max_val - min_val
            range_deg = range_rad * 180.0 / math.pi
            print(f"关节{i} {name}:")
            print(f"  限位: [{min_val:.3f}, {max_val:.3f}] rad")
            print(f"  范围: {range_deg:.1f}°")

# ========================= 修复后的预定义手势 =========================

class FixedDex3Gestures:
    """修复后的预定义手势库"""
    
    @staticmethod
    def get_gesture(name: str, hand: str = "right", joint_limits: Optional[List[Tuple[float, float]]] = None) -> Optional[List[float]]:
        """获取预定义手势，基于实际关节限位调整"""
        
        # 如果有关节限位，使用保守的手势定义
        if joint_limits:
            # 基于实际限位的保守手势
            gestures = {
                "open": [0.0] + [joint_limits[i][0] * 0.8 for i in range(1, 7)],  # 接近最小值但留有余量
                "closed": [0.0] + [joint_limits[i][1] * 0.8 for i in range(1, 7)],  # 接近最大值但留有余量
                "rest": [0.0] + [(joint_limits[i][0] + joint_limits[i][1]) * 0.2 for i in range(1, 7)],  # 中性位置偏开
            }
            
            # 复杂手势基于限位范围的比例
            thumb_range = joint_limits[1][1] - joint_limits[1][0]
            finger_range = [(joint_limits[i][1] - joint_limits[i][0]) for i in range(2, 7)]
            
            gestures.update({
                "pinch": [
                    joint_limits[0][1] * 0.3,  # 拇指旋转30%
                    joint_limits[1][0] + thumb_range * 0.6,  # 拇指弯曲60%
                    joint_limits[2][0] + finger_range[0] * 0.5,  # 拇指指尖50%
                    joint_limits[3][0] * 0.8,  # 中指根部接近最小
                    joint_limits[4][0] * 0.8,  # 中指指尖接近最小
                    joint_limits[5][0] + finger_range[3] * 0.6,  # 食指根部60%
                    joint_limits[6][0] + finger_range[4] * 0.5,  # 食指指尖50%
                ],
                "point": [
                    0.0,  # 拇指不旋转
                    joint_limits[1][0] + thumb_range * 0.8,  # 拇指弯曲80%
                    joint_limits[2][0] + finger_range[0] * 0.7,  # 拇指指尖70%
                    joint_limits[3][0] + finger_range[1] * 0.8,  # 中指根部80%
                    joint_limits[4][0] + finger_range[2] * 0.8,  # 中指指尖80%
                    joint_limits[5][0] * 0.8,  # 食指根部接近最小
                    joint_limits[6][0] * 0.8,  # 食指指尖接近最小
                ],
                "peace": [
                    0.0,  # 拇指不旋转
                    joint_limits[1][0] + thumb_range * 0.8,  # 拇指弯曲80%
                    joint_limits[2][0] + finger_range[0] * 0.7,  # 拇指指尖70%
                    joint_limits[3][0] * 0.8,  # 中指根部接近最小
                    joint_limits[4][0] * 0.8,  # 中指指尖接近最小
                    joint_limits[5][0] * 0.8,  # 食指根部接近最小
                    joint_limits[6][0] * 0.8,  # 食指指尖接近最小
                ],
                "ok": [
                    joint_limits[0][1] * 0.2,  # 拇指轻微旋转
                    joint_limits[1][0] + thumb_range * 0.6,  # 拇指弯曲60%
                    joint_limits[2][0] + finger_range[0] * 0.5,  # 拇指指尖50%
                    joint_limits[3][0] + finger_range[1] * 0.8,  # 中指根部80%
                    joint_limits[4][0] + finger_range[2] * 0.8,  # 中指指尖80%
                    joint_limits[5][0] + finger_range[3] * 0.6,  # 食指根部60%
                    joint_limits[6][0] + finger_range[4] * 0.5,  # 食指指尖50%
                ]
            })
        else:
            # 如果没有限位信息，返回None，强制使用校准数据
            print("警告: 无关节限位信息，无法生成安全手势")
            return None
        
        angles = gestures.get(name)
        if angles and hand == "left":
            # 左手镜像：仅镜像拇指旋转
            mirrored = angles.copy()
            mirrored[0] = -mirrored[0]
            return mirrored
        
        return angles

# ========================= 姿态管理器（使用修复后的手势） =========================

@dataclass
class HandPose:
    """手部姿态数据类"""
    name: str
    description: str
    angles: List[float]  # 7个关节角度
    kp: float = 8.0
    kd: float = 1.5

class PoseManager:
    """手部姿态管理器"""
    
    def __init__(self, joint_limits: Optional[List[Tuple[float, float]]] = None):
        self.poses: Dict[str, HandPose] = {}
        self.custom_poses: Dict[str, HandPose] = {}
        self.joint_limits = joint_limits
        self._load_default_poses()
    
    def _load_default_poses(self):
        """加载默认手部姿态"""
        if self.joint_limits is None:
            print("警告: 无关节限位信息，跳过默认手势加载")
            return
        
        gesture_names = ["open", "closed", "pinch", "point", "peace", "ok", "rest"]
        
        for name in gesture_names:
            angles = FixedDex3Gestures.get_gesture(name, "right", self.joint_limits)
            if angles:
                self.poses[name] = HandPose(
                    name=name,
                    description=f"基于校准限位的{name}手势",
                    angles=angles
                )
                print(f"✓ 生成基于校准限位的 {name} 手势")
            else:
                print(f"✗ 无法生成 {name} 手势")
    
    def load_poses_from_csv(self, csv_path: str):
        """从CSV文件加载姿态"""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            print(f"CSV文件不存在: {csv_path}")
            return
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pose = HandPose(
                        name=row['label'],
                        description=row['description'],
                        angles=[float(row[f'joint{i}']) for i in range(7)]
                    )
                    self.custom_poses[pose.name] = pose
            
            print(f"从 {csv_path} 加载了 {len(self.custom_poses)} 个自定义姿态")
            
        except Exception as e:
            print(f"加载CSV文件失败: {e}")
    
    def save_poses_to_csv(self, csv_path: str):
        """保存姿态到CSV文件"""
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['label', 'description'] + [f'joint{i}' for i in range(7)]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # 写入所有姿态
                all_poses = {**self.poses, **self.custom_poses}
                for pose in all_poses.values():
                    row = {
                        'label': pose.name,
                        'description': pose.description
                    }
                    for i, angle in enumerate(pose.angles):
                        row[f'joint{i}'] = angle
                    writer.writerow(row)
            
            print(f"姿态已保存到: {csv_path}")
            
        except Exception as e:
            print(f"保存CSV文件失败: {e}")
    
    def get_pose(self, name: str) -> Optional[HandPose]:
        """获取姿态"""
        if name in self.custom_poses:
            return self.custom_poses[name]
        return self.poses.get(name)
    
    def add_custom_pose(self, pose: HandPose):
        """添加自定义姿态"""
        self.custom_poses[pose.name] = pose
    
    def list_poses(self) -> List[str]:
        """列出所有姿态名称"""
        all_poses = list(self.poses.keys()) + list(self.custom_poses.keys())
        return sorted(set(all_poses))
    
    def mirror_pose_for_left_hand(self, pose: HandPose) -> HandPose:
        """为左手镜像姿态"""
        mirrored_angles = pose.angles.copy()
        mirrored_angles[0] = -mirrored_angles[0]  # 拇指旋转镜像
        
        return HandPose(
            name=f"{pose.name}_left",
            description=f"{pose.description} (左手)",
            angles=mirrored_angles,
            kp=pose.kp,
            kd=pose.kd
        )

# ========================= 传感器处理器 =========================

class SensorHandler:
    """传感器数据处理器"""
    
    def __init__(self):
        self.pressure_threshold = 0.1  # 压力阈值 (N)
        self.contact_threshold = 100000  # 原始接触阈值
        self.temperature_threshold = 40.0  # 温度阈值 (°C)
    
    def process_pressure_data(self, pressure_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理压力传感器数据"""
        if not pressure_data:
            return {}
        
        processed_data = {}
        total_contact_points = 0
        total_pressure = 0.0
        
        try:
            for sensor_key, sensor_info in pressure_data.items():
                pressures = sensor_info.get('pressure', [])
                temperatures = sensor_info.get('temperature', [])
                
                # 计算有效接触点
                contact_points = sensor_info.get('valid_contacts', 0)
                avg_pressure = sum(pressures) / len(pressures) if pressures else 0.0
                avg_temperature = sum(temperatures) / len(temperatures) if temperatures else 0.0
                
                sensor_data = {
                    'sensor_key': sensor_key,
                    'pressure_values': pressures,
                    'temperatures': temperatures,
                    'avg_pressure': avg_pressure,
                    'avg_temperature': avg_temperature,
                    'contact_points': contact_points,
                    'is_lost': sensor_info.get('lost', False),
                    'has_contact': contact_points > 0,
                    'over_temperature': avg_temperature > self.temperature_threshold
                }
                
                processed_data[sensor_key] = sensor_data
                total_contact_points += contact_points
                total_pressure += avg_pressure
            
            # 全局统计
            processed_data['global_stats'] = {
                'total_contact_points': total_contact_points,
                'total_pressure': total_pressure,
                'avg_pressure': total_pressure / max(len(pressure_data), 1),
                'has_any_contact': total_contact_points > 0
            }
            
        except Exception as e:
            print(f"处理压力数据失败: {e}")
            return {}
        
        return processed_data
    
    def detect_grasp_quality(self, sensor_data: Dict[str, Any]) -> Tuple[float, str]:
        """
        检测抓握质量
        
        Returns:
            (质量分数 0-1, 质量描述)
        """
        if not sensor_data or 'global_stats' not in sensor_data:
            return 0.0, "无传感器数据"
        
        stats = sensor_data['global_stats']
        contact_points = stats['total_contact_points']
        avg_pressure = stats['avg_pressure']
        
        # 评估抓握质量
        if contact_points == 0:
            return 0.0, "无接触"
        elif contact_points < 3:
            return 0.3, "接触点太少"
        elif avg_pressure < self.pressure_threshold:
            return 0.5, "压力不足"
        elif avg_pressure > 2.0:
            return 0.7, "压力过大"
        else:
            return 0.9, "抓握良好"

# ========================= 自适应抓握控制器 =========================

class AdaptiveGripController:
    """自适应抓握控制器"""
    
    def __init__(self, dex3_client: FixedDex3Client, sensor_handler: SensorHandler):
        self.dex3_client = dex3_client
        self.sensor_handler = sensor_handler
        self.grip_state = "idle"  # idle, gripping, holding, releasing
        self.target_pressure = 0.5  # 目标压力 (N)
        self.pressure_tolerance = 0.1  # 压力容差
        self.grip_speed = 0.05  # 抓握速度 (rad/step)
        
        # PID控制参数
        self.kp_pressure = 2.0
        self.ki_pressure = 0.1
        self.kd_pressure = 0.5
        self.pressure_error_sum = 0.0
        self.last_pressure_error = 0.0
    
    def start_adaptive_grip(self, target_pressure: float = 0.5):
        """开始自适应抓握"""
        self.target_pressure = target_pressure
        self.grip_state = "gripping"
        self.pressure_error_sum = 0.0
        self.last_pressure_error = 0.0
        print(f"开始自适应抓握，目标压力: {target_pressure:.2f} N")
    
    def update_grip(self, timeout: float = 0.1) -> Tuple[bool, str]:
        """
        更新抓握状态
        
        Returns:
            (是否完成抓握, 状态描述)
        """
        if self.grip_state != "gripping":
            return True, self.grip_state
        
        # 获取当前状态
        current_angles = self.dex3_client.get_joint_angles(timeout)
        if not current_angles:
            return False, "无法获取关节角度"
        
        pressure_data = self.dex3_client.get_pressure_data(timeout)
        processed_data = self.sensor_handler.process_pressure_data(pressure_data or {})
        
        if 'global_stats' not in processed_data:
            # 继续闭合手部
            new_angles = self._close_fingers(current_angles)
            self.dex3_client.set_joint_angles(new_angles)
            return False, "无传感器数据，继续闭合"
        
        current_pressure = processed_data['global_stats']['avg_pressure']
        contact_points = processed_data['global_stats']['total_contact_points']
        
        # 检查是否有接触
        if contact_points == 0:
            # 继续闭合手部
            new_angles = self._close_fingers(current_angles)
            self.dex3_client.set_joint_angles(new_angles)
            return False, "无接触，继续闭合"
        
        # PID控制调节压力
        pressure_error = self.target_pressure - current_pressure
        self.pressure_error_sum += pressure_error
        pressure_error_diff = pressure_error - self.last_pressure_error
        
        # PID输出
        control_output = (
            self.kp_pressure * pressure_error +
            self.ki_pressure * self.pressure_error_sum +
            self.kd_pressure * pressure_error_diff
        )
        
        self.last_pressure_error = pressure_error
        
        # 根据控制输出调整关节角度
        if abs(pressure_error) < self.pressure_tolerance:
            # 达到目标压力，保持抓握
            self.grip_state = "holding"
            return True, f"抓握完成，压力: {current_pressure:.2f} N"
        elif pressure_error > 0:
            # 压力不足，继续闭合
            new_angles = self._close_fingers(current_angles, control_output * 0.1)
            self.dex3_client.set_joint_angles(new_angles)
            return False, f"压力不足，继续闭合 (当前: {current_pressure:.2f}N)"
        else:
            # 压力过大，稍微松开
            new_angles = self._open_fingers(current_angles, abs(control_output) * 0.1)
            self.dex3_client.set_joint_angles(new_angles)
            return False, f"压力过大，稍微松开 (当前: {current_pressure:.2f}N)"
    
    def _close_fingers(self, current_angles: List[float], intensity: float = None) -> List[float]:
        """闭合手指"""
        if intensity is None:
            intensity = self.grip_speed
        
        new_angles = current_angles.copy()
        
        # 使用实际的关节限位
        joint_limits = self.dex3_client._get_joint_limits()
        
        # 闭合除拇指旋转外的所有关节
        for i in range(1, 7):  # 跳过拇指旋转 (index 0)
            max_angle = joint_limits[i][1] if i < len(joint_limits) else 1.6
            new_angles[i] = min(new_angles[i] + intensity, max_angle)
        
        return new_angles
    
    def _open_fingers(self, current_angles: List[float], intensity: float = None) -> List[float]:
        """张开手指"""
        if intensity is None:
            intensity = self.grip_speed
        
        new_angles = current_angles.copy()
        
        # 使用实际的关节限位
        joint_limits = self.dex3_client._get_joint_limits()
        
        # 张开除拇指旋转外的所有关节
        for i in range(1, 7):
            min_angle = joint_limits[i][0] if i < len(joint_limits) else -0.2
            new_angles[i] = max(new_angles[i] - intensity, min_angle)
        
        return new_angles
    
    def release_grip(self):
        """释放抓握"""
        self.grip_state = "releasing"
        print("释放抓握")
    
    def is_gripping(self) -> bool:
        """检查是否正在抓握"""
        return self.grip_state in ["gripping", "holding"]

# ========================= 调试工具 =========================

def debug_sdk_types():
    """调试SDK类型信息"""
    if not SDK_AVAILABLE:
        print("SDK不可用，无法调试")
        return
    
    print("=== SDK 类型调试信息 ===")
    print(f"HandCmd_ 类型: {type(HandCmd_)}")
    print(f"HandState_ 类型: {type(HandState_)}")
    print(f"MotorCmd_ 类型: {type(MotorCmd_)}")
    
    try:
        # 测试 MotorCmd_ 创建
        test_motor_cmd = MotorCmd_(
            1,           # mode
            0.0,         # q
            0.0,         # dq
            0.0,         # tau
            8.0,         # kp
            1.5,         # kd
            [0, 0, 0]    # reserve
        )
        print(f"✓ MotorCmd_ 创建成功: {type(test_motor_cmd)}")
    except Exception as e:
        print(f"✗ MotorCmd_ 创建失败: {e}")
    
    try:
        # 测试 HandCmd_ 创建
        test_hand_cmd = HandCmd_([], [0, 0, 0, 0])
        print(f"✓ HandCmd_ 创建成功: {type(test_hand_cmd)}")
    except Exception as e:
        print(f"✗ HandCmd_ 创建失败: {e}")

# ========================= 主测试程序（使用修复后的客户端） =========================

class Dex3ControlTest:
    """Dex3控制测试主程序"""
    
    def __init__(self, interface: str = "eth0", hand: str = "right", 
                 config_dir: str = "config", limits_file: str = None):
        self.interface = interface
        self.hand = hand
        self.config_dir = config_dir
        self.limits_file = limits_file
        
        # 初始化组件
        self.sensor_handler = SensorHandler()
        self.dex3_client: Optional[FixedDex3Client] = None
        self.grip_controller: Optional[AdaptiveGripController] = None
        self.pose_manager: Optional[PoseManager] = None
        
        print(f"Dex3控制测试初始化完成")
    
    def run_startup_sequence(self) -> bool:
        """运行启动序列"""
        print("=== 开始机器人启动序列 ===")
        
        if not SDK_AVAILABLE:
            print("SDK不可用，跳过启动序列")
            return True
        
        try:
            sport_client = hanger_boot_sequence(iface=self.interface)
            print("✓ 机器人已就绪，可以控制灵巧手")
            return True
        except Exception as e:
            print(f"✗ 启动序列失败: {e}")
            return False
    
    def initialize_dex3(self) -> bool:
        """初始化Dex3连接"""
        try:
            self.dex3_client = FixedDex3Client(
                hand=self.hand, 
                interface=self.interface,
                config_dir=self.config_dir,
                limits_file=self.limits_file
            )
            
            # 使用加载的关节限位初始化姿态管理器
            self.pose_manager = PoseManager(joint_limits=self.dex3_client.joint_limits)
            
            # 加载自定义姿态
            data_dir = Path("data")
            csv_file = data_dir / "hand_states.csv"
            if csv_file.exists():
                self.pose_manager.load_poses_from_csv(str(csv_file))
            
            self.grip_controller = AdaptiveGripController(self.dex3_client, self.sensor_handler)
            
            print(f"✓ Dex3 {self.hand}手连接成功")
            print(f"可用姿态: {self.pose_manager.list_poses()}")
            return True
        except Exception as e:
            print(f"✗ Dex3连接失败: {e}")
            traceback.print_exc()
            return False
    
    def test_basic_poses(self):
        """测试基础手部姿态"""
        print("=== 测试基础手部姿态 ===")
        
        if not self.dex3_client or not self.pose_manager:
            print("Dex3客户端或姿态管理器未初始化")
            return
        
        test_poses = ["rest", "open", "closed", "pinch", "point", "peace", "ok"]
        
        for pose_name in test_poses:
            pose = self.pose_manager.get_pose(pose_name)
            if pose:
                print(f"执行姿态: {pose_name} - {pose.description}")
                
                # 调整左手姿态
                if self.hand == "left":
                    pose = self.pose_manager.mirror_pose_for_left_hand(pose)
                
                success = self.dex3_client.set_joint_angles(pose.angles, pose.kp, pose.kd)
                if success:
                    print(f"  ✓ 姿态执行成功")
                else:
                    print(f"  ✗ 姿态执行失败")
                
                time.sleep(3.0)  # 等待姿态执行完成
            else:
                print(f"  ✗ 姿态 {pose_name} 不存在")
    
    def interactive_control(self):
        """交互式控制模式"""
        print("=== 交互式控制模式 ===")
        print("可用命令:")
        print("  pose <姿态名> - 执行指定姿态")
        print("  list - 列出所有可用姿态")
        print("  grip [压力] - 开始自适应抓握")
        print("  release - 释放抓握")
        print("  sensor - 显示传感器数据")
        print("  save <姿态名> - 保存当前姿态")
        print("  stop - 停止电机")
        print("  debug - 显示SDK调试信息")
        print("  limits - 显示关节限位信息")
        print("  quit - 退出")
        
        if not self.dex3_client or not self.pose_manager:
            print("Dex3客户端或姿态管理器未初始化，仅显示帮助信息")
            return
        
        while True:
            try:
                cmd = input(f"\n{self.hand}手控制> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == "quit":
                    break
                elif cmd[0] == "debug":
                    debug_sdk_types()
                elif cmd[0] == "limits":
                    self.dex3_client.print_joint_limits_info()
                elif cmd[0] == "list":
                    poses = self.pose_manager.list_poses()
                    print(f"可用姿态: {', '.join(poses)}")
                elif cmd[0] == "pose" and len(cmd) > 1:
                    pose_name = cmd[1]
                    pose = self.pose_manager.get_pose(pose_name)
                    if pose:
                        if self.hand == "left":
                            pose = self.pose_manager.mirror_pose_for_left_hand(pose)
                        print(f"执行姿态: {pose_name}")
                        success = self.dex3_client.set_joint_angles(pose.angles, pose.kp, pose.kd)
                        if success:
                            print(f"  ✓ {pose_name} 姿态执行成功")
                        else:
                            print(f"  ✗ {pose_name} 姿态执行失败")
                    else:
                        print(f"姿态 {pose_name} 不存在")
                elif cmd[0] == "grip":
                    if not self.grip_controller:
                        print("抓握控制器未初始化")
                        continue
                    
                    target_pressure = 0.3
                    if len(cmd) > 1:
                        try:
                            target_pressure = float(cmd[1])
                        except ValueError:
                            print("无效的压力值，使用默认值 0.3")
                    
                    print(f"开始自适应抓握，目标压力: {target_pressure}")
                    self.grip_controller.start_adaptive_grip(target_pressure)
                    
                    # 执行抓握循环
                    for _ in range(50):
                        completed, status = self.grip_controller.update_grip()
                        print(f"抓握状态: {status}")
                        if completed:
                            break
                        time.sleep(0.1)
                elif cmd[0] == "release":
                    if self.grip_controller:
                        self.grip_controller.release_grip()
                    open_pose = self.pose_manager.get_pose("open")
                    if open_pose:
                        if self.hand == "left":
                            open_pose = self.pose_manager.mirror_pose_for_left_hand(open_pose)
                        self.dex3_client.set_joint_angles(open_pose.angles)
                elif cmd[0] == "sensor":
                    pressure_data = self.dex3_client.get_pressure_data()
                    processed_data = self.sensor_handler.process_pressure_data(pressure_data or {})
                    if processed_data and 'global_stats' in processed_data:
                        quality, description = self.sensor_handler.detect_grasp_quality(processed_data)
                        stats = processed_data.get('global_stats', {})
                        print(f"传感器状态:")
                        print(f"  接触点: {stats.get('total_contact_points', 0)}")
                        print(f"  总压力: {stats.get('total_pressure', 0.0):.3f} N")
                        print(f"  平均压力: {stats.get('avg_pressure', 0.0):.3f} N")
                        print(f"  抓握质量: {quality:.2f} ({description})")
                    else:
                        print("无传感器数据")
                elif cmd[0] == "save" and len(cmd) > 1:
                    pose_name = cmd[1]
                    current_angles = self.dex3_client.get_joint_angles()
                    if current_angles:
                        description = input("请输入姿态描述: ").strip()
                        new_pose = HandPose(
                            name=pose_name,
                            description=description or f"自定义姿态 {pose_name}",
                            angles=current_angles
                        )
                        self.pose_manager.add_custom_pose(new_pose)
                        print(f"姿态 {pose_name} 已保存")
                        
                        # 保存到文件
                        data_dir = Path("data")
                        data_dir.mkdir(exist_ok=True)
                        self.pose_manager.save_poses_to_csv("data/hand_states.csv")
                    else:
                        print("无法获取当前关节角度")
                elif cmd[0] == "stop":
                    print("停止所有电机")
                    self.dex3_client.stop_motors()
                else:
                    print("未知命令")
            
            except KeyboardInterrupt:
                print("\n退出交互模式")
                break
            except Exception as e:
                print(f"命令执行错误: {e}")
                traceback.print_exc()
    
    def cleanup(self):
        """清理资源"""
        print("清理资源...")
        if self.dex3_client:
            self.dex3_client.stop_motors()

# ========================= 主函数 =========================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Dex3 灵巧手控制测试程序 (修复版)")
    parser.add_argument("--iface", default="eth0", help="网络接口名称")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="控制的手")
    parser.add_argument("--mode", choices=["test", "interactive", "calibrate", "debug"], default="interactive", 
                        help="运行模式")
    parser.add_argument("--skip-startup", action="store_true", help="跳过启动序列")
    parser.add_argument("--skip-dex3", action="store_true", help="跳过Dex3初始化（仅测试姿态管理）")
    parser.add_argument("--config-dir", default="config", help="配置文件目录")
    parser.add_argument("--limits-file", type=str, help="指定关节限位配置文件")
    
    args = parser.parse_args()
    
    print("=== Dex3 灵巧手控制测试程序 (修复版) ===")
    print(f"网络接口: {args.iface}")
    print(f"控制手: {args.hand}")
    print(f"运行模式: {args.mode}")
    print(f"配置目录: {args.config_dir}")
    if args.limits_file:
        print(f"限位配置文件: {args.limits_file}")
    print(f"SDK可用: {SDK_AVAILABLE}")
    
    # 调试模式
    if args.mode == "debug":
        debug_sdk_types()
        return
    
    # 创建测试实例
    test = Dex3ControlTest(args.iface, args.hand, args.config_dir, args.limits_file)
    
    try:
        # 执行启动序列
        if not args.skip_startup:
            if not test.run_startup_sequence():
                print("启动序列失败，继续测试...")
        
        # 初始化Dex3连接
        if not args.skip_dex3:
            if not test.initialize_dex3():
                print("Dex3初始化失败，程序退出")
                return
        
        # 根据模式执行不同测试
        if args.mode == "test":
            print("\n执行基础测试...")
            test.test_basic_poses()
        elif args.mode == "interactive":
            test.interactive_control()
        elif args.mode == "calibrate":
            print("校准模式请运行: python3 dex3_joint_limits_calibration.py")
        
    except KeyboardInterrupt:
        print("\n程序被中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
        traceback.print_exc()
    finally:
        test.cleanup()
        print("程序结束")


if __name__ == "__main__":
    main()