#!/usr/bin/env python3
"""G-1 Dex3 Dual Hand Joint Monitor HUD - 修复版本"""

import time
import sys
import curses
import threading
import math
from collections import deque
from pathlib import Path

# Unitree SDK 相关导入 - 修复导入路径
try:
    # 使用本地修复的 dex3_client
    from unitree_sdk2py.dex3.dex3_client import Dex3Client
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, MotorCmd_
    
    DEX3_AVAILABLE = True
    print("Dex3 SDK 加载成功")
except ImportError as e:
    print(f"警告: unitree_sdk2py.dex3 不可用: {e}")
    DEX3_AVAILABLE = False


class Dex3JointIndex:
    """定义 Unitree Dex3 手部关节的索引值和名称。"""
    JOINT_NAMES = [
        "thumb_0",         # 0 - 拇指旋转关节 (范围: -1.57 ~ 1.57 rad)
        "thumb_1",         # 1 - 拇指弯曲1 (范围: -0.5 ~ 1.8 rad)
        "thumb_2",         # 2 - 拇指弯曲2 (范围: -0.2 ~ 1.6 rad)
        "middle_0",        # 3 - 中指弯曲1 (范围: -0.2 ~ 1.6 rad)
        "middle_1",        # 4 - 中指弯曲2 (范围: -0.2 ~ 1.6 rad)
        "index_0",         # 5 - 食指弯曲1 (范围: -0.2 ~ 1.6 rad)
        "index_1",         # 6 - 食指弯曲2 (范围: -0.2 ~ 1.6 rad)
    ]
    
    JOINT_DESCRIPTIONS = [
        "拇指旋转",        # 0 - thumb_0: 拇指旋转
        "拇指弯曲1",       # 1 - thumb_1: 拇指弯曲1
        "拇指弯曲2",       # 2 - thumb_2: 拇指弯曲2
        "中指弯曲1",       # 3 - middle_0: 中指弯曲1
        "中指弯曲2",       # 4 - middle_1: 中指弯曲2
        "食指弯曲1",       # 5 - index_0: 食指弯曲1
        "食指弯曲2",       # 6 - index_1: 食指弯曲2
    ]
    
    JOINT_LIMITS = [
        (-1.57, 1.57),     # thumb_0: 拇指旋转
        (-0.5, 1.8),       # thumb_1: 拇指弯曲1
        (-0.2, 1.6),       # thumb_2: 拇指弯曲2
        (-0.2, 1.6),       # middle_0: 中指弯曲1
        (-0.2, 1.6),       # middle_1: 中指弯曲2
        (-0.2, 1.6),       # index_0: 食指弯曲1
        (-0.2, 1.6),       # index_1: 食指弯曲2
    ]
    
    # 每只手有 9 个压力传感器区域，每个区域多个点
    MAX_PRESSURE_SENSORS = 9


class SafeDex3Client:
    """安全的 Dex3 客户端包装器"""
    
    def __init__(self, hand: str, interface: str):
        self.hand = hand
        self.interface = interface
        self._dex3_client = None
        self._connected = False
        
        try:
            self._connect()
        except Exception as exc:
            print(f"SafeDex3Client 初始化失败: {exc}")
    
    def _connect(self):
        """连接到 Dex3 设备"""
        try:
            # 使用修复的 Dex3Client
            print(f"正在连接 {self.hand} 手...")
            self._dex3_client = Dex3Client(hand=self.hand, interface=self.interface)
            
            # 等待连接稳定
            time.sleep(2.0)
            
            # 测试连接 - 多次尝试读取状态
            success_count = 0
            for attempt in range(5):
                state = self._dex3_client.read_state(timeout=2.0)
                if state is not None:
                    success_count += 1
                    # 验证数据完整性
                    if hasattr(state, 'motor_state') and len(state.motor_state) >= 7:
                        print(f"[SafeDex3] {self.hand} 手状态验证成功 - 电机数量: {len(state.motor_state)}")
                        self._connected = True
                        return
                time.sleep(0.5)
            
            if success_count > 0:
                print(f"[SafeDex3] {self.hand} 手连接成功，但数据不稳定")
                self._connected = True
            else:
                print(f"[SafeDex3] {self.hand} 手无法读取状态数据")
                self._connected = False
                
        except Exception as exc:
            print(f"[SafeDex3] {self.hand} 手连接失败: {exc}")
            import traceback
            traceback.print_exc()
            self._connected = False
    
    def read_state(self, timeout=0.1):
        """读取手部状态 - 增强错误处理"""
        if self._dex3_client is not None and self._connected:
            try:
                state = self._dex3_client.read_state(timeout)
                
                # 验证状态数据的完整性
                if state is not None:
                    if hasattr(state, 'motor_state'):
                        if len(state.motor_state) >= 7:
                            return state
                        else:
                            print(f"[SafeDex3] {self.hand} 手电机状态数据不完整: {len(state.motor_state)}/7")
                    else:
                        print(f"[SafeDex3] {self.hand} 手状态数据缺少 motor_state 字段")
                
                return None
                
            except Exception as exc:
                print(f"[SafeDex3] 读取 {self.hand} 手状态失败: {exc}")
                return None
        return None
    
    def send_command(self, target_angles, kp=6.0, kd=1.0):
        """发送手部控制命令 - 增强参数验证"""
        if self._dex3_client is not None and self._connected:
            try:
                # 验证输入参数
                if not isinstance(target_angles, (list, tuple)):
                    print(f"[SafeDex3] {self.hand} 手角度参数类型错误")
                    return False
                
                # 确保角度数组长度正确
                angles = list(target_angles[:7]) if len(target_angles) >= 7 else list(target_angles) + [0.0] * (7 - len(target_angles))
                
                # 验证角度范围
                for i, angle in enumerate(angles):
                    if i < len(Dex3JointIndex.JOINT_LIMITS):
                        min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[i]
                        if angle < min_limit or angle > max_limit:
                            print(f"[SafeDex3] {self.hand} 手关节{i}角度 {angle:.3f} 超出限位 [{min_limit:.3f}, {max_limit:.3f}]")
                            # 限制在安全范围内
                            angles[i] = max(min_limit, min(max_limit, angle))
                
                success = self._dex3_client.set_joint_angles(
                    angles=angles,
                    kp=kp,
                    kd=kd
                )
                
                if success:
                    print(f"[SafeDex3] {self.hand} 手命令发送成功: {[f'{a:.3f}' for a in angles[:3]]}...")
                else:
                    print(f"[SafeDex3] {self.hand} 手底层命令发送失败")
                
                return success
                
            except Exception as exc:
                print(f"[SafeDex3] {self.hand} 手命令发送失败: {exc}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"[SafeDex3] {self.hand} 手未连接，无法发送命令 (connected: {self._connected})")
            return False
    
    def is_connected(self):
        """检查是否已连接"""
        return self._connected and self._dex3_client is not None


class Dex3DualHandMonitorHUD:
    """修复的 Dex3 双手关节监控 HUD"""
    
    def __init__(self, network_interface: str = "eth0"):
        """初始化双手监控器 HUD"""
        self.network_interface = network_interface
        
        # 左右手客户端
        self._left_client = None
        self._right_client = None
        
        # 状态数据
        self.start_time = time.time()
        self.left_update_count = 0
        self.right_update_count = 0
        
        # 左手数据
        self.left_angles = [0.0] * 7
        self.left_velocities = [0.0] * 7
        self.left_pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
        self.left_status = "connecting"
        self.left_last_time = time.time()
        
        # 右手数据
        self.right_angles = [0.0] * 7
        self.right_velocities = [0.0] * 7
        self.right_pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
        self.right_status = "connecting"
        self.right_last_time = time.time()
        
        # 控制变量
        self._stop_event = threading.Event()
        
        # 预定义手势（更安全的角度范围）
        self._open_pose = [0.0, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1]
        self._closed_pose = [0.0, 0.8, 0.6, 0.8, 0.8, 0.8, 0.8]  # 更保守的闭合角度
        
        self._setup_dex3()

    def _setup_dex3(self):
        """初始化 Dex3 连接"""
        if not DEX3_AVAILABLE:
            print("Dex3 SDK 不可用，使用模拟数据")
            self.left_status = "simulated"
            self.right_status = "simulated"
            self._start_simulation_threads()
            return
            
        try:
            print(f"初始化 DDS 通道工厂: {self.network_interface}")
            # 初始化 DDS 通道工厂
            ChannelFactoryInitialize(0, self.network_interface)
            time.sleep(2.0)  # 增加等待时间确保 DDS 初始化完成
            
            # 连接左右手
            self._connect_hands()
            
            # 启动数据读取线程
            if self._left_client or self._right_client:
                self._start_data_threads()
            else:
                print("警告: 左右手都未连接成功，启动模拟模式")
                self.left_status = "simulated"
                self.right_status = "simulated"
                self._start_simulation_threads()
                
        except Exception as exc:
            print(f"Dex3 设置失败: {exc}")
            import traceback
            traceback.print_exc()
            self.left_status = "error"
            self.right_status = "error"

    def _connect_hands(self):
        """连接左右手设备"""
        # 连接右手 (优先连接右手)
        print("正在连接右手...")
        try:
            self._right_client = SafeDex3Client(hand="right", interface=self.network_interface)
            if self._right_client.is_connected():
                self.right_status = "connected"
                print("右手连接成功")
            else:
                self.right_status = "failed"
                self._right_client = None
                print("右手连接失败")
        except Exception as exc:
            print(f"右手连接异常: {exc}")
            self._right_client = None
            self.right_status = "failed"
        
        # 连接左手
        print("正在连接左手...")
        try:
            self._left_client = SafeDex3Client(hand="left", interface=self.network_interface)
            if self._left_client.is_connected():
                self.left_status = "connected"
                print("左手连接成功")
            else:
                self.left_status = "failed"
                self._left_client = None
                print("左手连接失败")
        except Exception as exc:
            print(f"左手连接异常: {exc}")
            self._left_client = None
            self.left_status = "failed"

    def _start_data_threads(self):
        """启动数据读取线程"""
        if self._right_client:
            self._right_thread = threading.Thread(
                target=self._right_data_loop, 
                daemon=True,
                name="RightHandThread"
            )
            self._right_thread.start()
            print("右手数据线程已启动")
        
        if self._left_client:
            self._left_thread = threading.Thread(
                target=self._left_data_loop, 
                daemon=True,
                name="LeftHandThread"
            )
            self._left_thread.start()
            print("左手数据线程已启动")

    def _start_simulation_threads(self):
        """启动模拟数据线程"""
        self._left_thread = threading.Thread(
            target=self._left_simulation_loop, 
            daemon=True,
            name="LeftSimThread"
        )
        self._right_thread = threading.Thread(
            target=self._right_simulation_loop, 
            daemon=True,
            name="RightSimThread"
        )
        
        self._left_thread.start()
        self._right_thread.start()
        print("模拟数据线程已启动")

    def _left_data_loop(self):
        """左手数据读取循环线程"""
        print("左手数据读取线程启动")
        self.left_status = "reading"
        consecutive_failures = 0
        
        while not self._stop_event.is_set():
            try:
                if self._left_client and self._left_client.is_connected():
                    state = self._left_client.read_state(timeout=0.2)
                    if state is not None:
                        self._process_hand_state(state, "left")
                        self.left_update_count += 1
                        consecutive_failures = 0
                        if self.left_status != "active":
                            self.left_status = "active"
                            print("左手数据流已激活")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            if self.left_status == "active":
                                self.left_status = "no_data"
                                print("左手数据流中断")
                else:
                    self.left_status = "disconnected"
                    print("左手连接丢失")
                    break
                    
                time.sleep(0.02)  # 50Hz
                
            except Exception as exc:
                print(f"左手数据读取错误: {exc}")
                self.left_status = "error"
                time.sleep(0.1)

    def _right_data_loop(self):
        """右手数据读取循环线程"""
        print("右手数据读取线程启动")
        self.right_status = "reading"
        consecutive_failures = 0
        
        while not self._stop_event.is_set():
            try:
                if self._right_client and self._right_client.is_connected():
                    state = self._right_client.read_state(timeout=0.2)
                    if state is not None:
                        self._process_hand_state(state, "right")
                        self.right_update_count += 1
                        consecutive_failures = 0
                        if self.right_status != "active":
                            self.right_status = "active"
                            print("右手数据流已激活")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures > 10:
                            if self.right_status == "active":
                                self.right_status = "no_data"
                                print("右手数据流中断")
                else:
                    self.right_status = "disconnected"
                    print("右手连接丢失")
                    break
                    
                time.sleep(0.02)  # 50Hz
                
            except Exception as exc:
                print(f"右手数据读取错误: {exc}")
                self.right_status = "error"
                time.sleep(0.1)

    def _left_simulation_loop(self):
        """左手模拟数据循环"""
        while not self._stop_event.is_set():
            self._generate_mock_data("left")
            time.sleep(0.02)

    def _right_simulation_loop(self):
        """右手模拟数据循环"""
        while not self._stop_event.is_set():
            self._generate_mock_data("right")
            time.sleep(0.02)

    def _process_hand_state(self, state, hand_side):
        """处理手部状态数据 - 修复版本"""
        try:
            current_time = time.time()
            
            # 验证状态数据
            if not hasattr(state, 'motor_state'):
                print(f"{hand_side} 手状态缺少 motor_state")
                return
            
            motor_states = state.motor_state
            if len(motor_states) < 7:
                print(f"{hand_side} 手电机数据不完整: {len(motor_states)}/7")
                return
            
            # 修复：确保按索引顺序处理关节数据
            if hand_side == "left":
                dt = current_time - self.left_last_time
                if dt > 0:
                    # 关键修复：严格按照索引0-6处理
                    for i in range(7):
                        try:
                            if i < len(motor_states):
                                current_angle = float(motor_states[i].q)
                                
                                # 验证角度值的合理性
                                if -10.0 <= current_angle <= 10.0:
                                    # 计算速度
                                    angle_diff = current_angle - self.left_angles[i]
                                    self.left_velocities[i] = angle_diff / dt
                                    self.left_angles[i] = current_angle
                                else:
                                    print(f"左手关节{i}角度异常: {current_angle}")
                            else:
                                print(f"左手关节{i}数据缺失")
                                
                        except (AttributeError, ValueError, IndexError) as e:
                            print(f"左手关节{i}数据解析错误: {e}")
            
                # 处理压力传感器
                self._process_pressure_sensors(state, "left")
                self.left_last_time = current_time
            
            else:  # right hand
                dt = current_time - self.right_last_time
                if dt > 0:
                    # 关键修复：严格按照索引0-6处理
                    for i in range(7):
                        try:
                            if i < len(motor_states):
                                current_angle = float(motor_states[i].q)
                                
                                # 验证角度值的合理性
                                if -10.0 <= current_angle <= 10.0:
                                    # 计算速度
                                    angle_diff = current_angle - self.right_angles[i]
                                    self.right_velocities[i] = angle_diff / dt
                                    self.right_angles[i] = current_angle
                                else:
                                    print(f"右手关节{i}角度异常: {current_angle}")
                            else:
                                print(f"右手关节{i}数据缺失")
                                
                        except (AttributeError, ValueError, IndexError) as e:
                            print(f"右手关节{i}数据解析错误: {e}")
            
                # 处理压力传感器
                self._process_pressure_sensors(state, "right")
                self.right_last_time = current_time

        except Exception as exc:
            print(f"处理 {hand_side} 手状态数据失败: {exc}") 

    def _process_pressure_sensors(self, state, hand_side):
        """处理压力传感器数据 - 修复版本"""
        # 修复：初始化完整的压力数组
        pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
        
        try:
            if hasattr(state, 'press_sensor_state') and len(state.press_sensor_state) > 0:
                pressure_idx = 0
                
                # 修复：严格控制压力传感器数据处理
                for sensor_group in state.press_sensor_state:
                    if hasattr(sensor_group, 'pressure'):
                        for pressure_val in sensor_group.pressure:
                            if pressure_idx < Dex3JointIndex.MAX_PRESSURE_SENSORS:
                                try:
                                    pressure_float = float(pressure_val)
                                    # 处理不同的压力值格式
                                    if pressure_float >= 100000:
                                        pressure_values[pressure_idx] = pressure_float / 10000.0
                                    elif pressure_float == 30000:
                                        pressure_values[pressure_idx] = 0.0
                                    else:
                                        pressure_values[pressure_idx] = pressure_float / 10000.0
                                except (ValueError, TypeError):
                                    pressure_values[pressure_idx] = 0.0
                                
                                pressure_idx += 1
                            else:
                                break  # 已达到最大传感器数量
                
                    if pressure_idx >= Dex3JointIndex.MAX_PRESSURE_SENSORS:
                        break  # 已达到最大传感器数量
                        
        except Exception as exc:
            print(f"处理 {hand_side} 手压力传感器失败: {exc}")
            
        # 修复：确保数组长度正确
        if hand_side == "left":
            self.left_pressure_values = pressure_values[:Dex3JointIndex.MAX_PRESSURE_SENSORS]
        else:
            self.right_pressure_values = pressure_values[:Dex3JointIndex.MAX_PRESSURE_SENSORS]

    def _generate_mock_data(self, hand_side):
        """生成模拟数据 - 改进的模拟器"""
        current_time = time.time()
        
        if hand_side == "left":
            dt = current_time - self.left_last_time
            if dt > 0:
                for i in range(7):
                    if i < len(Dex3JointIndex.JOINT_LIMITS):
                        min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[i]
                        range_center = (min_limit + max_limit) / 2.0
                        range_amplitude = (max_limit - min_limit) / 4.0
                        
                        # 创建更真实的模拟数据
                        mock_angle = range_center + range_amplitude * math.sin(current_time * 0.5 + i * 0.7)
                        
                        # 计算速度
                        angle_diff = mock_angle - self.left_angles[i]
                        self.left_velocities[i] = angle_diff / dt
                        self.left_angles[i] = mock_angle
            
            # 模拟压力传感器
            for i in range(len(self.left_pressure_values)):
                self.left_pressure_values[i] = abs(8.0 * math.sin(current_time * 2.0 + i * 0.5))
            
            self.left_last_time = current_time
            self.left_update_count += 1
            
        else:  # right hand
            dt = current_time - self.right_last_time
            if dt > 0:
                for i in range(7):
                    if i < len(Dex3JointIndex.JOINT_LIMITS):
                        min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[i]
                        range_center = (min_limit + max_limit) / 2.0
                        range_amplitude = (max_limit - min_limit) / 4.0
                        
                        # 右手使用稍微不同的相位
                        mock_angle = range_center + range_amplitude * math.sin(current_time * 0.3 + i * 0.8 + 3.14)
                        
                        # 计算速度
                        angle_diff = mock_angle - self.right_angles[i]
                        self.right_velocities[i] = angle_diff / dt
                        self.right_angles[i] = mock_angle
            
            # 模拟压力传感器
            for i in range(len(self.right_pressure_values)):
                self.right_pressure_values[i] = abs(6.0 * math.sin(current_time * 1.8 + i * 0.7))
            
            self.right_last_time = current_time
            self.right_update_count += 1

    def _send_hand_command(self, hand_side, target_pose):
        """发送手部控制命令"""
        client = self._left_client if hand_side == "left" else self._right_client
        if client is None or not client.is_connected():
            print(f"{hand_side} 手未连接，无法发送命令")
            return
            
        try:
            print(f"发送 {hand_side} 手命令: {[f'{a:.3f}' for a in target_pose[:3]]}...")
            success = client.send_command(
                target_angles=target_pose[:7],
                kp=6.0,  # 安全的控制参数
                kd=1.0   # 安全的控制参数
            )
            if success:
                print(f"{hand_side} 手命令发送成功")
            else:
                print(f"{hand_side} 手命令发送失败")
                
        except Exception as exc:
            print(f"{hand_side} 手命令发送失败: {exc}")

    def _handle_key_input(self, key):
        """处理键盘输入。"""
        # 左手控制
        if key == ord('a'):  # 左手打开
            self.left_status = "opening"
            self._send_hand_command("left", self._open_pose)
        elif key == ord('s'):  # 左手关闭
            self.left_status = "closing"
            self._send_hand_command("left", self._closed_pose)
        
        # 右手控制
        elif key == ord('k'):  # 右手打开
            self.right_status = "opening"
            self._send_hand_command("right", self._open_pose)
        elif key == ord('l'):  # 右手关闭
            self.right_status = "closing"
            self._send_hand_command("right", self._closed_pose)

    def _safe_addstr(self, stdscr, y, x, text, attr=0):
        """安全地添加字符串，避免越界错误。"""
        height, width = stdscr.getmaxyx()
        if y >= height or x >= width:
            return
        
        max_len = width - x - 1
        if max_len > 0:
            text = text[:max_len]
            try:
                stdscr.addstr(y, x, text, attr)
            except curses.error:
                pass

    def _check_joint_safety(self, joint_id, angle):
        """检查关节角度安全性。"""
        if joint_id >= len(Dex3JointIndex.JOINT_LIMITS):
            return curses.color_pair(3)  # 青色 - 未知关节
            
        min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[joint_id]
        if angle < min_limit or angle > max_limit:
            return curses.color_pair(4)  # 红色 - 超出限位
        elif abs(angle - min_limit) < 0.1 or abs(angle - max_limit) < 0.1:
            return curses.color_pair(2)  # 黄色 - 接近限位
        else:
            return curses.color_pair(1)  # 绿色 - 安全范围

    def draw_hud(self, stdscr):
        """
        使用 curses 绘制终端 HUD 的主函数 - 修复版本
        """
        # 初始化 curses
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

        while True:
            key = stdscr.getch()
            if key == ord('q'):
                break
            elif key in [ord('a'), ord('s'), ord('k'), ord('l')]:
                self._handle_key_input(key)

            stdscr.erase()
            height, width = stdscr.getmaxyx()

            # 检查最小窗口尺寸
            if height < 18 or width < 100:
                self._safe_addstr(stdscr, 0, 0, "Terminal too small. Need at least 100x18.")
                stdscr.refresh()
                time.sleep(0.1)
                continue

            # 绘制标题
            title = "G-1 Dex3 Dual Hand Joint Monitor (HUD)"
            title_x = max(0, (width - len(title)) // 2)
            self._safe_addstr(stdscr, 0, title_x, title, curses.A_BOLD)

            # 修复：重新计算列布局，确保不重叠
            left_col_start = 0
            left_col_width = 30
            right_col_start = left_col_width + 2
            right_col_width = 30
            sensor_col_start = right_col_start + right_col_width + 3
            
            # 绘制列标题
            self._safe_addstr(stdscr, 2, left_col_start, "Left Hand Joints", curses.A_UNDERLINE | curses.color_pair(1))
            self._safe_addstr(stdscr, 2, right_col_start, "Right Hand Joints", curses.A_UNDERLINE | curses.color_pair(1))
            self._safe_addstr(stdscr, 2, sensor_col_start, "Pressure Sensors", curses.A_UNDERLINE | curses.color_pair(1))

            # 绘制压力传感器子标题
            self._safe_addstr(stdscr, 3, sensor_col_start, "L-Hand  R-Hand", curses.color_pair(3))

            # 修复：关节数据显示区域
            joint_start_row = 4
            joint_end_row = joint_start_row + 7  # 第4-10行用于关节数据
            
            for i in range(7):  # 严格限制为7个关节 (0-6)
                display_row = joint_start_row + i
                if display_row >= height - 5:
                    break
                
                # 验证索引范围
                if i >= len(Dex3JointIndex.JOINT_NAMES):
                    print(f"警告: 关节索引{i}超出预定义范围")
                    continue
                
                joint_name = Dex3JointIndex.JOINT_NAMES[i]
                
                # 验证数据数组范围
                if i >= len(self.left_angles) or i >= len(self.right_angles):
                    print(f"警告: 关节索引{i}超出数据数组范围")
                    continue
                
                # 左手关节显示
                left_angle = self.left_angles[i]
                left_color = self._check_joint_safety(i, left_angle)
                left_line = f"{i} {joint_name:<10}: {left_angle:+6.3f}"
                self._safe_addstr(stdscr, display_row, left_col_start, left_line, left_color)
                
                # 右手关节显示
                right_angle = self.right_angles[i]
                right_color = self._check_joint_safety(i, right_angle)
                right_line = f"{i} {joint_name:<10}: {right_angle:+6.3f}"
                self._safe_addstr(stdscr, display_row, right_col_start, right_line, right_color)

            # 修复：压力传感器数据显示在独立区域
            pressure_start_row = 4  # 与关节同行开始，但在右侧独立列
            pressure_max_rows = min(9, height - 5 - pressure_start_row)  # 最多显示9行压力数据
            
            for i in range(min(pressure_max_rows, len(self.left_pressure_values))):
                display_row = pressure_start_row + i
                
                # 验证压力数据数组范围
                if i >= len(self.left_pressure_values) or i >= len(self.right_pressure_values):
                    continue
                
                # 获取左右手压力值
                left_pressure = self.left_pressure_values[i]
                right_pressure = self.right_pressure_values[i]
                
                # 修复：综合左右手压力值决定颜色
                max_pressure = max(left_pressure, right_pressure)
                if max_pressure >= 10.0:
                    pressure_color = curses.color_pair(4)  # 红色 - 高压力
                elif max_pressure >= 5.0:
                    pressure_color = curses.color_pair(2)  # 黄色 - 中等压力
                else:
                    pressure_color = curses.color_pair(1)  # 绿色 - 低压力
                
                # 修复：改进压力传感器显示格式
                pressure_line = f"{i}: {left_pressure:6.3f}  {right_pressure:6.3f}"
                self._safe_addstr(stdscr, display_row, sensor_col_start, pressure_line, pressure_color)

            # 修复：状态信息显示在固定位置，避免重叠
            status_start_row = max(joint_end_row + 1, pressure_start_row + 9 + 1)  # 确保在数据区域之后
            
            if height > status_start_row + 4:
                # 连接状态
                left_conn = "OK" if (self._left_client and self._left_client.is_connected()) else "NG"
                right_conn = "OK" if (self._right_client and self._right_client.is_connected()) else "NG"
                left_conn_color = curses.color_pair(1) if left_conn == "OK" else curses.color_pair(4)
                right_conn_color = curses.color_pair(1) if right_conn == "OK" else curses.color_pair(4)
                
                status_line = f"Left Status: {self.left_status} | Right Status: {self.right_status} | Conn: L:"
                self._safe_addstr(stdscr, height - 5, 0, status_line)
                self._safe_addstr(stdscr, height - 5, len(status_line), left_conn, left_conn_color)
                self._safe_addstr(stdscr, height - 5, len(status_line) + len(left_conn), " R:")
                self._safe_addstr(stdscr, height - 5, len(status_line) + len(left_conn) + 3, right_conn, right_conn_color)

                # 绘制统计信息
                conn_duration = time.time() - self.start_time
                h, m, s = int(conn_duration // 3600), int((conn_duration % 3600) // 60), int(conn_duration % 60)
                left_rate = self.left_update_count / conn_duration if conn_duration > 0 else 0.0
                right_rate = self.right_update_count / conn_duration if conn_duration > 0 else 0.0
                stats_line = f"Stats: Time: {h:02d}:{m:02d}:{s:02d} | Left Rate: {left_rate:.1f}Hz | Right Rate: {right_rate:.1f}Hz"
                self._safe_addstr(stdscr, height - 4, 0, stats_line, curses.color_pair(3))

                # 绘制控制提示
                control_line = "Controls: 'q' quit | Left: 'a' open 's' close | Right: 'k' open 'l' close"
                self._safe_addstr(stdscr, height - 3, 0, control_line, curses.color_pair(3))

                # 绘制操作提示
                tip_line = "注意: 手部动作使用安全控制参数，请确保周围环境安全"
                self._safe_addstr(stdscr, height - 2, 0, tip_line, curses.color_pair(2))

            stdscr.refresh()
            time.sleep(0.05)  # 20Hz 刷新率

    def run(self):
        """启动 curses 应用。"""
        try:
            curses.wrapper(self.draw_hud)
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源。"""
        print("正在清理资源...")
        self._stop_event.set()
        
        if hasattr(self, '_left_thread') and self._left_thread.is_alive():
            print("等待左手线程结束...")
            self._left_thread.join(timeout=2.0)
        
        if hasattr(self, '_right_thread') and self._right_thread.is_alive():
            print("等待右手线程结束...")
            self._right_thread.join(timeout=2.0)
        
        print("资源清理完成")


def main():
    """程序主入口函数。"""
    if len(sys.argv) < 2:
        print("Usage: python3 dex3_gui_hub.py <network_interface>")
        print("Example: python3 dex3_gui_hub.py eth0")
        sys.exit(-1)
    
    print("G-1 Dex3 双手关节监控器 (HUD 版本)")
    
    # 解析命令行参数
    network_interface = sys.argv[1]
    
    print(f"使用网络接口: {network_interface}")
    print("监控双手: 左手 + 右手")
    print("正在启动终端界面...")
    
    # 创建并运行监控器
    try:
        monitor = Dex3DualHandMonitorHUD(network_interface)
        monitor.run()
    except KeyboardInterrupt:
        print("\n程序被中断。")
    except Exception as exc:
        print(f"程序运行错误: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        print("HUD 已关闭。")


if __name__ == '__main__':
    main()