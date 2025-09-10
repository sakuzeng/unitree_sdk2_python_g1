#!/usr/bin/env python3
"""G-1 Dex3 Dual Hand Joint Monitor HUD - Curses 版本

本脚本提供一个基于 curses 的终端平视显示器 (HUD)，用于实时监控
Unitree G-1 机器人 Dex3 左右手部关节的角度、变化速度和压力传感器状态。

Layout
======
┌──────────────── G-1 Dex3 Dual Hand Joint Monitor (HUD) ─────────────────┐
│ Left Hand Joints            Right Hand Joints           Pressure Sensors │
│ 0 thumb_0:     +0.00 rad    0 thumb_0:     +0.00 rad    L-Sensors R-Sensors│
│ 1 thumb_1:     +0.00 rad    1 thumb_1:     +0.00 rad    0: 0.00   0: 0.00 │
│ 2 thumb_2:     +0.00 rad    2 thumb_2:     +0.00 rad    1: 0.00   1: 0.00 │
│ 3 middle_0:    +0.00 rad    3 middle_0:    +0.00 rad    2: 0.00   2: 0.00 │
│ 4 middle_1:    +0.00 rad    4 middle_1:    +0.00 rad    3: 0.00   3: 0.00 │
│ 5 index_0:     +0.00 rad    5 index_0:     +0.00 rad    4: 0.00   4: 0.00 │
│ 6 index_1:     +0.00 rad    6 index_1:     +0.00 rad    5: 0.00   5: 0.00 │
│                                                          6: 0.00   6: 0.00 │
│ Left Status: idle | Right Status: idle | Conn: L:OK R:OK                 │
│ Stats: Time: 00:00:00 | Left Rate: 0.0 Hz | Right Rate: 0.0 Hz           │
│ Controls: 'q' quit | Left: 'a' open 's' close | Right: 'k' open 'l' close │
└──────────────────────────────────────────────────────────────────────────┘

使用方法:
    python3 dex3_gui_hud.py <network_interface>

示例:
    python3 dex3_gui_hud.py eth0
"""

import time
import sys
import curses
import threading
from collections import deque
from pathlib import Path

# Unitree SDK 相关导入
try:
    from unitree_sdk2py.dex3 import Dex3Client
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    # 直接导入 Dex3 相关的消息类型
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, MotorCmd_
    
    DEX3_AVAILABLE = True
except ImportError:
    print("警告: unitree_sdk2py.dex3 不可用，将使用模拟模式")
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
    """安全的 Dex3 客户端包装器，处理 SDK 接口问题"""
    
    def __init__(self, hand: str, interface: str):
        self.hand = hand
        self.interface = interface
        self._dex3_client = None
        self._publisher = None
        self._connected = False
        
        try:
            self._connect()
        except Exception as exc:
            print(f"SafeDex3Client 初始化失败: {exc}")
    
    def _connect(self):
        """连接到 Dex3 设备"""
        try:
            # 尝试使用标准 Dex3Client
            self._dex3_client = Dex3Client(hand=self.hand, interface=self.interface)
            self._connected = True
            print(f"[SafeDex3] {self.hand} 手连接成功")
        except Exception as exc:
            print(f"[SafeDex3] {self.hand} 手连接失败，尝试直接发布器: {exc}")
            
            # 如果 Dex3Client 失败，尝试直接创建发布器
            try:
                topic = f"rt/dex3/{self.hand}/cmd"
                self._publisher = ChannelPublisher(topic, HandCmd_)
                self._publisher.init()
                print(f"[SafeDex3] {self.hand} 手直接发布器创建成功")
            except Exception as exc2:
                print(f"[SafeDex3] {self.hand} 手直接发布器也失败: {exc2}")
    
    def read_state(self, timeout=0.1):
        """读取手部状态"""
        if self._dex3_client is not None:
            try:
                return self._dex3_client.read_state(timeout)
            except Exception as exc:
                print(f"[SafeDex3] 读取 {self.hand} 手状态失败: {exc}")
                return None
        return None
    
    def send_command(self, target_angles, kp=8.0, kd=1.5):
        """发送手部控制命令"""
        try:
            # 方法1: 尝试使用标准 Dex3Client
            if self._dex3_client is not None:
                success = self._dex3_client.set_joint_angles(
                    angles=target_angles[:7],
                    kp=kp,
                    kd=kd
                )
                if success:
                    return True
                else:
                    print(f"[SafeDex3] {self.hand} 手标准方法失败，尝试直接构造命令")
            
            # 方法2: 直接构造 HandCmd_ 消息
            if self._publisher is not None:
                return self._send_direct_command(target_angles, kp, kd)
            
            return False
            
        except Exception as exc:
            print(f"[SafeDex3] {self.hand} 手命令发送失败: {exc}")
            return False
    
    def _send_direct_command(self, target_angles, kp, kd):
        """直接构造并发送 HandCmd_ 消息"""
        try:
            # 创建电机命令数组
            motor_cmds = []
            for i in range(7):
                motor_cmd = MotorCmd_()
                motor_cmd.mode = 0x01  # 位置模式
                motor_cmd.q = float(target_angles[i]) if i < len(target_angles) else 0.0
                motor_cmd.dq = 0.0
                motor_cmd.tau = 0.0
                motor_cmd.kp = float(kp)
                motor_cmd.kd = float(kd)
                motor_cmds.append(motor_cmd)
            
            # 创建手部命令消息（提供必需的参数）
            hand_cmd = HandCmd_(
                motor_cmd=motor_cmds,
                reserve=[0] * 12  # 提供 reserve 数组
            )
            
            # 发布命令
            self._publisher.write(hand_cmd)
            return True
            
        except Exception as exc:
            print(f"[SafeDex3] 直接命令构造失败: {exc}")
            return False
    
    def is_connected(self):
        """检查是否已连接"""
        return self._connected or self._publisher is not None


class Dex3DualHandMonitorHUD:
    """
    Dex3 双手关节监控 HUD 的主类。

    使用 curses 在终端中显示左右手关节状态，并处理 Unitree Dex3 数据。
    """
    def __init__(self, network_interface: str = "eth0"):
        """初始化双手监控器 HUD。"""
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
        self.left_status = "idle"
        self.left_last_time = time.time()
        
        # 右手数据
        self.right_angles = [0.0] * 7
        self.right_velocities = [0.0] * 7
        self.right_pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
        self.right_status = "idle"
        self.right_last_time = time.time()
        
        # 控制变量
        self._stop_event = threading.Event()
        
        # 预定义手势（安全的角度范围）
        self._open_pose = [0.0, -0.3, -0.1, -0.1, -0.1, -0.1, -0.1]
        self._closed_pose = [0.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0]  # 保守的闭合角度
        
        self._setup_dex3()

    def _setup_dex3(self):
        """初始化 Dex3 连接。"""
        if not DEX3_AVAILABLE:
            print("Dex3 SDK 不可用，使用模拟数据")
            return
            
        try:
            # 初始化 DDS 通道工厂
            ChannelFactoryInitialize(0, self.network_interface)
            
            # 连接左右手
            self._connect_hands()
            
            # 启动数据读取线程
            self._left_thread = threading.Thread(
                target=self._left_data_loop, 
                daemon=True
            )
            self._right_thread = threading.Thread(
                target=self._right_data_loop, 
                daemon=True
            )
            
            self._left_thread.start()
            self._right_thread.start()
                
        except Exception as exc:
            print(f"Dex3 设置失败: {exc}")

    def _connect_hands(self):
        """连接左右手设备。"""
        # 连接左手
        try:
            self._left_client = SafeDex3Client(hand="left", interface=self.network_interface)
        except Exception as exc:
            print(f"左手连接失败: {exc}")
            self._left_client = None
        
        # 连接右手
        try:
            self._right_client = SafeDex3Client(hand="right", interface=self.network_interface)
        except Exception as exc:
            print(f"右手连接失败: {exc}")
            self._right_client = None

    def _left_data_loop(self):
        """左手数据读取循环线程。"""
        while not self._stop_event.is_set():
            try:
                if self._left_client is not None and self._left_client.is_connected():
                    state = self._left_client.read_state(timeout=0.1)
                    if state is not None:
                        self._process_hand_state(state, "left")
                        self.left_update_count += 1
                else:
                    self._generate_mock_data("left")
                    
                time.sleep(0.02)  # 50Hz
                
            except Exception as exc:
                print(f"左手数据读取错误: {exc}")
                time.sleep(0.1)

    def _right_data_loop(self):
        """右手数据读取循环线程。"""
        while not self._stop_event.is_set():
            try:
                if self._right_client is not None and self._right_client.is_connected():
                    state = self._right_client.read_state(timeout=0.1)
                    if state is not None:
                        self._process_hand_state(state, "right")
                        self.right_update_count += 1
                else:
                    self._generate_mock_data("right")
                    
                time.sleep(0.02)  # 50Hz
                
            except Exception as exc:
                print(f"右手数据读取错误: {exc}")
                time.sleep(0.1)

    def _process_hand_state(self, state, hand_side):
        """处理手部状态数据。"""
        current_time = time.time()
        
        if hand_side == "left":
            dt = current_time - self.left_last_time
            if dt > 0:
                for i in range(min(7, len(state.motor_state))):
                    current_angle = state.motor_state[i].q
                    angle_diff = current_angle - self.left_angles[i]
                    self.left_velocities[i] = angle_diff / dt
                    self.left_angles[i] = current_angle
            
            # 处理压力传感器
            self.left_pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
            try:
                pressure_idx = 0
                for ps in state.press_sensor_state:
                    for pressure in ps.pressure:
                        if pressure_idx < len(self.left_pressure_values):
                            if pressure >= 100000:
                                self.left_pressure_values[pressure_idx] = pressure / 10000.0
                            elif pressure == 30000:
                                self.left_pressure_values[pressure_idx] = 0.0
                            else:
                                self.left_pressure_values[pressure_idx] = pressure / 10000.0
                            pressure_idx += 1
            except Exception:
                pass
            
            self.left_last_time = current_time
            
        else:  # right hand
            dt = current_time - self.right_last_time
            if dt > 0:
                for i in range(min(7, len(state.motor_state))):
                    current_angle = state.motor_state[i].q
                    angle_diff = current_angle - self.right_angles[i]
                    self.right_velocities[i] = angle_diff / dt
                    self.right_angles[i] = current_angle
            
            # 处理压力传感器
            self.right_pressure_values = [0.0] * Dex3JointIndex.MAX_PRESSURE_SENSORS
            try:
                pressure_idx = 0
                for ps in state.press_sensor_state:
                    for pressure in ps.pressure:
                        if pressure_idx < len(self.right_pressure_values):
                            if pressure >= 100000:
                                self.right_pressure_values[pressure_idx] = pressure / 10000.0
                            elif pressure == 30000:
                                self.right_pressure_values[pressure_idx] = 0.0
                            else:
                                self.right_pressure_values[pressure_idx] = pressure / 10000.0
                            pressure_idx += 1
            except Exception:
                pass
            
            self.right_last_time = current_time

    def _generate_mock_data(self, hand_side):
        """生成模拟数据。"""
        import math
        
        current_time = time.time()
        
        if hand_side == "left":
            dt = current_time - self.left_last_time
            for i in range(7):
                min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[i]
                range_center = (min_limit + max_limit) / 2
                range_amplitude = (max_limit - min_limit) / 4
                
                mock_angle = range_center + range_amplitude * math.sin(current_time * 0.5 + i)
                if dt > 0:
                    self.left_velocities[i] = (mock_angle - self.left_angles[i]) / dt
                self.left_angles[i] = mock_angle
            
            for i in range(len(self.left_pressure_values)):
                self.left_pressure_values[i] = abs(10.0 * math.sin(current_time * 2 + i * 0.5))
            
            self.left_last_time = current_time
            self.left_update_count += 1
            
        else:  # right hand
            dt = current_time - self.right_last_time
            for i in range(7):
                min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[i]
                range_center = (min_limit + max_limit) / 2
                range_amplitude = (max_limit - min_limit) / 4
                
                mock_angle = range_center + range_amplitude * math.sin(current_time * 0.3 + i + 3.14)
                if dt > 0:
                    self.right_velocities[i] = (mock_angle - self.right_angles[i]) / dt
                self.right_angles[i] = mock_angle
            
            for i in range(len(self.right_pressure_values)):
                self.right_pressure_values[i] = abs(8.0 * math.sin(current_time * 1.8 + i * 0.7))
            
            self.right_last_time = current_time
            self.right_update_count += 1

    def _send_hand_command(self, hand_side, target_pose):
        """发送手部控制命令，使用安全的参数。"""
        client = self._left_client if hand_side == "left" else self._right_client
        if client is None or not client.is_connected():
            print(f"{hand_side} 手未连接")
            return
            
        try:
            # 使用安全的控制参数
            success = client.send_command(
                target_angles=target_pose[:7],
                kp=6.0,  # 降低 KP 值以确保安全
                kd=1.0   # 降低 KD 值以确保安全
            )
            if not success:
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
        min_limit, max_limit = Dex3JointIndex.JOINT_LIMITS[joint_id]
        if angle < min_limit or angle > max_limit:
            return curses.color_pair(4)  # 红色 - 超出限位
        elif abs(angle - min_limit) < 0.1 or abs(angle - max_limit) < 0.1:
            return curses.color_pair(2)  # 黄色 - 接近限位
        else:
            return curses.color_pair(1)  # 绿色 - 安全范围

    def draw_hud(self, stdscr):
        """
        使用 curses 绘制终端 HUD 的主函数。
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

            # 计算列宽
            left_col_width = 25
            right_col_width = 25
            sensor_col_start = left_col_width + right_col_width + 2

            # 绘制列标题
            self._safe_addstr(stdscr, 2, 0, "Left Hand Joints", curses.A_UNDERLINE | curses.color_pair(1))
            self._safe_addstr(stdscr, 2, left_col_width, "Right Hand Joints", curses.A_UNDERLINE | curses.color_pair(1))
            self._safe_addstr(stdscr, 2, sensor_col_start, "Pressure Sensors", curses.A_UNDERLINE | curses.color_pair(1))

            # 绘制压力传感器子标题
            self._safe_addstr(stdscr, 3, sensor_col_start, "L-Hand  R-Hand", curses.color_pair(3))

            # 绘制关节数据
            for i in range(7):
                if 4 + i >= height - 5:
                    break
                
                # 左手关节
                joint_name = Dex3JointIndex.JOINT_NAMES[i]
                left_angle = self.left_angles[i]
                left_vel = self.left_velocities[i]
                left_color = self._check_joint_safety(i, left_angle)
                
                left_line = f"{i} {joint_name:<10}: {left_angle:+6.3f}"
                self._safe_addstr(stdscr, 4 + i, 0, left_line, left_color)
                
                # 右手关节
                right_angle = self.right_angles[i]
                right_vel = self.right_velocities[i]
                right_color = self._check_joint_safety(i, right_angle)
                
                right_line = f"{i} {joint_name:<10}: {right_angle:+6.3f}"
                self._safe_addstr(stdscr, 4 + i, left_col_width, right_line, right_color)

            # 绘制压力传感器数据
            for i in range(min(9, len(self.left_pressure_values))):
                if 4 + i >= height - 5:
                    break
                
                # 左手压力
                left_pressure = self.left_pressure_values[i]
                left_pressure_color = curses.color_pair(3)
                if left_pressure >= 10.0:
                    left_pressure_color = curses.color_pair(4)
                elif left_pressure >= 5.0:
                    left_pressure_color = curses.color_pair(2)
                
                # 右手压力
                right_pressure = self.right_pressure_values[i] if i < len(self.right_pressure_values) else 0.0
                right_pressure_color = curses.color_pair(3)
                if right_pressure >= 10.0:
                    right_pressure_color = curses.color_pair(4)
                elif right_pressure >= 5.0:
                    right_pressure_color = curses.color_pair(2)
                
                pressure_line = f"{i}: {left_pressure:5.2f}"
                self._safe_addstr(stdscr, 4 + i, sensor_col_start, pressure_line, left_pressure_color)
                
                pressure_line_r = f"{i}: {right_pressure:5.2f}"
                self._safe_addstr(stdscr, 4 + i, sensor_col_start + 8, pressure_line_r, right_pressure_color)

            # 绘制状态信息
            if height > 8:
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
            if height > 6:
                conn_duration = time.time() - self.start_time
                h, m, s = int(conn_duration // 3600), int((conn_duration % 3600) // 60), int(conn_duration % 60)
                left_rate = self.left_update_count / conn_duration if conn_duration > 0 else 0.0
                right_rate = self.right_update_count / conn_duration if conn_duration > 0 else 0.0
                stats_line = f"Stats: Time: {h:02d}:{m:02d}:{s:02d} | Left Rate: {left_rate:.1f}Hz | Right Rate: {right_rate:.1f}Hz"
                self._safe_addstr(stdscr, height - 4, 0, stats_line)

            # 绘制控制提示
            if height > 4:
                control_line = "Controls: 'q' quit | Left: 'a' open 's' close | Right: 'k' open 'l' close"
                self._safe_addstr(stdscr, height - 3, 0, control_line, curses.color_pair(3))

            # 绘制操作提示
            if height > 3:
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
        self._stop_event.set()
        if hasattr(self, '_left_thread') and self._left_thread.is_alive():
            self._left_thread.join(timeout=1.0)
        if hasattr(self, '_right_thread') and self._right_thread.is_alive():
            self._right_thread.join(timeout=1.0)


def main():
    """程序主入口函数。"""
    if len(sys.argv) < 2:
        print("Usage: python3 dex3_gui_hud.py <network_interface>")
        print("Example: python3 dex3_gui_hud.py eth0")
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
    finally:
        print("HUD 已关闭。")


if __name__ == '__main__':
    main()

"""
Left Status: idle | Right Status: idle | Conn: L:OK R:OK
Stats: Time: 00:03:03 | Left Rate: 48.9Hz | Right Rate: 48.9Hz
Controls: 'q' quit | Left: 'a' open 's' close | Right: 'k' open 'l' close
注意: 手部动作将缓慢执行(安全考虑)，请确保周围环境安全[Dex3] 创建命令消息失败: HandCmd_.__init__() missing 2 required positional arguments: 'motor_cmd' andopening | Right Status: idle | Conn: L:OK R:OK
                   11  [Dex3] 设置关8.9 度失败: HandCmd_.8.9nit__() missing 2 required positional arguments: 'motor_cmd' and 'reserve'
                                                                                                                                      left手命令发送失败                                            [Dex3] 创建命令消息失败: HandCmd_.__init__() missing 2 required positional arguments: 'motor_cmd' andclosserve'
                    7  [Dex3] 设置关9.0 度失败: HandCmd_.9.0nit__() missing 2 required positional arguments: 'motor_cmd' and 'reserve'
                                                                                                                                      left手命令发送失败
"""