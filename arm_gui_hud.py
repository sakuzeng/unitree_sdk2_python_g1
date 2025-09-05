#!/usr/bin/env python3
"""G-1 Arm Joint Monitor HUD - Curses 版本

本脚本提供一个基于 curses 的终端平视显示器 (HUD)，用于实时监控
Unitree G-1 机器人手臂和腰部关节的角度及变化速度。

Layout
======
┌────────────────── G-1 Arm Joint Monitor (HUD) ───────────────────┐
│ Left Arm                    Waist                   Right Arm    │
│ 15 L Shoulder Pitch: +0.00  12 Waist Yaw: +0.00     ...          │
│ ...                                                              │
│                                                                  │
│ Stats: Conn Time: 00:00:00 | Rate: 0.0 Hz | Packets: 0           │
│ Press 'q' to quit                                                │
└──────────────────────────────────────────────────────────────────┘

使用方法:
    python3 arm_gui_hud.py <network_interface>

示例:
    python3 arm_gui_hud.py eth0
"""

import time
import sys
import curses
from collections import deque

# Unitree SDK 相关导入
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# LowState_ 定义了机器人低级状态数据的消息类型，包含关节角度等信息
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


class G1JointIndex:
    """定义 Unitree G1 机器人手臂和腰部关节的索引值。"""
    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28

    # Waist
    WaistYaw = 12


class ArmJointMonitorHUD:
    """
    机器人手臂关节监控 HUD 的主类。

    使用 curses 在终端中显示关节状态，并处理 Unitree SDK 数据。
    """
    def __init__(self):
        """初始化监控器 HUD。"""
        # 机器人状态数据
        self.low_state = None
        self.start_time = time.time()
        self.update_count = 0
        
        # 变化检测相关
        self.previous_angles = {}
        self.angle_velocities = {}
        self.last_angles_time = time.time()
        
        # 定义关节名称和索引的映射
        self.joint_info = [
            # Left arm
            (G1JointIndex.LeftShoulderPitch, "15 L Shoulder Pitch", "left"),
            (G1JointIndex.LeftShoulderRoll, "16 L Shoulder Roll", "left"),
            (G1JointIndex.LeftShoulderYaw, "17 L Shoulder Yaw", "left"),
            (G1JointIndex.LeftElbow, "18 L Elbow", "left"),
            (G1JointIndex.LeftWristRoll, "19 L Wrist Roll", "left"),
            (G1JointIndex.LeftWristPitch, "20 L Wrist Pitch", "left"),
            (G1JointIndex.LeftWristYaw, "21 L Wrist Yaw", "left"),
            # Waist
            (G1JointIndex.WaistYaw, "12 Waist Yaw", "waist"),
            # Right arm
            (G1JointIndex.RightShoulderPitch, "22 R Shoulder Pitch", "right"),
            (G1JointIndex.RightShoulderRoll, "23 R Shoulder Roll", "right"),
            (G1JointIndex.RightShoulderYaw, "24 R Shoulder Yaw", "right"),
            (G1JointIndex.RightElbow, "25 R Elbow", "right"),
            (G1JointIndex.RightWristRoll, "26 R Wrist Roll", "right"),
            (G1JointIndex.RightWristPitch, "27 R Wrist Pitch", "right"),
            (G1JointIndex.RightWristYaw, "28 R Wrist Yaw", "right"),
        ]
        
        # 初始化数据结构
        for joint_idx, _, _ in self.joint_info:
            self.previous_angles[joint_idx] = 0.0
            self.angle_velocities[joint_idx] = 0.0
        
        self._setup_unitree()

    def _setup_unitree(self):
        """初始化 Unitree SDK，订阅机器人低级状态数据。"""
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        """
        处理接收到的机器人低级状态消息的回调函数。

        Args:
            msg (LowState_): 从机器人接收到的状态数据。
        """
        self.low_state = msg
        self.update_count += 1
        
        current_time = time.time()
        dt = current_time - self.last_angles_time
        
        if dt > 0:
            for joint_idx, _, _ in self.joint_info:
                current_angle = msg.motor_state[joint_idx].q
                if joint_idx in self.previous_angles:
                    angle_diff = current_angle - self.previous_angles[joint_idx]
                    self.angle_velocities[joint_idx] = angle_diff / dt
                self.previous_angles[joint_idx] = current_angle
        
        self.last_angles_time = current_time

    def draw_hud(self, stdscr):
        """
        使用 curses 绘制终端 HUD 的主函数。

        Args:
            stdscr: curses 窗口对象。
        """
        # 初始化 curses
        curses.curs_set(0)  # 隐藏光标
        stdscr.nodelay(True) # 非阻塞输入
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)

        # 分组关节
        left_arm = [info for info in self.joint_info if info[2] == "left"]
        waist = [info for info in self.joint_info if info[2] == "waist"]
        right_arm = [info for info in self.joint_info if info[2] == "right"]

        while True:
            # 检查退出键
            key = stdscr.getch()
            if key == ord('q'):
                break

            stdscr.erase()
            height, width = stdscr.getmaxyx()

            # 绘制标题
            title = "G-1 Arm Joint Monitor (HUD)"
            stdscr.addstr(0, (width - len(title)) // 2, title, curses.A_BOLD)

            # 绘制列标题
            col_width = width // 3
            stdscr.addstr(2, 0, "Left Arm", curses.A_UNDERLINE)
            stdscr.addstr(2, col_width, "Waist", curses.A_UNDERLINE)
            stdscr.addstr(2, col_width * 2, "Right Arm", curses.A_UNDERLINE)

            # 绘制关节数据
            if self.low_state:
                for i in range(7):
                    # 左臂
                    if i < len(left_arm):
                        idx, name, _ = left_arm[i]
                        angle = self.low_state.motor_state[idx].q
                        vel = self.angle_velocities.get(idx, 0.0)
                        stdscr.addstr(4 + i, 0, f"{name[:18]:<18}: {angle:+.2f} rad ({vel:+.2f})")
                    
                    # 腰部
                    if i == 0 and waist:
                        idx, name, _ = waist[0]
                        angle = self.low_state.motor_state[idx].q
                        vel = self.angle_velocities.get(idx, 0.0)
                        stdscr.addstr(4 + i, col_width, f"{name[:18]:<18}: {angle:+.2f} rad ({vel:+.2f})")

                    # 右臂
                    if i < len(right_arm):
                        idx, name, _ = right_arm[i]
                        angle = self.low_state.motor_state[idx].q
                        vel = self.angle_velocities.get(idx, 0.0)
                        stdscr.addstr(4 + i, col_width * 2, f"{name[:18]:<18}: {angle:+.2f} rad ({vel:+.2f})")
            else:
                stdscr.addstr(4, (width - 20) // 2, "Waiting for data...", curses.color_pair(2))

            # 绘制统计信息
            conn_duration = time.time() - self.start_time
            h, m, s = int(conn_duration // 3600), int((conn_duration % 3600) // 60), int(conn_duration % 60)
            rate = self.update_count / conn_duration if conn_duration > 0 else 0.0
            stats_line = f"Conn Time: {h:02d}:{m:02d}:{s:02d} | Rate: {rate:.1f} Hz | Packets: {self.update_count}"
            stdscr.addstr(height - 2, 0, stats_line)
            stdscr.addstr(height - 1, 0, "Press 'q' to quit", curses.color_pair(3))

            stdscr.refresh()
            time.sleep(0.05) # 20Hz 刷新率

    def run(self):
        """启动 curses 应用。"""
        curses.wrapper(self.draw_hud)


def main():
    """程序主入口函数。"""
    print("G-1 机器人手臂关节监控器 (HUD 版本)")
    print("正在启动终端界面...")
    
    # 设置默认网口参数为 eth0
    network_interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    print(f"使用网络接口: {network_interface}")
    
    # 初始化 DDS 通道工厂
    ChannelFactoryInitialize(0, network_interface)
    
    # 创建并运行监控器
    try:
        monitor = ArmJointMonitorHUD()
        monitor.run()
    except KeyboardInterrupt:
        print("\n程序被中断。")
    finally:
        print("HUD 已关闭。")

if __name__ == '__main__':
    main()