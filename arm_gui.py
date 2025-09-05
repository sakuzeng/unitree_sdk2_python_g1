#!/usr/bin/env python3
"""G-1 Arm Joint Monitor GUI - 数值监控版本（无绘图）

本脚本提供一个基于 PySide6 的图形用户界面，用于实时监控 Unitree G-1 机器人
手臂和腰部关节的角度及变化速度。

Layout
======
┌─────────────────────────── G-1 Arm Joint Monitor ─────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                           关节角度表格                                    │ │
│ │  Left arm               Waist               Right arm                    │ │
│ │  15 L Shoulder Pitch    12 Waist Yaw       22 R Shoulder Pitch          │ │
│ │  16 L Shoulder Roll                         23 R Shoulder Roll           │ │
│ │  ...                                        ...                          │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                           统计信息                                        │ │
│ │  连接时间: XX:XX:XX     更新频率: XX Hz     数据包计数: XXXX           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘

Requirements
------------
    pip install pyside6
"""

import time
import sys
import os
from typing import Dict, List, Tuple
from collections import deque

# 用于构建 GUI 的 Qt 相关库
try:
    from PySide6 import QtCore, QtWidgets, QtGui
except ImportError as e:
    print(f"请安装必要的依赖: pip install pyside6")
    print(f"错误: {e}")
    sys.exit(1)

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

class ArmJointMonitorGUI(QtCore.QObject):
    """
    机器人手臂关节监控 GUI 的主类。

    负责初始化 GUI 界面、设置 Unitree SDK 数据订阅、处理数据回调
    以及更新界面显示。
    """
    def __init__(self):
        """初始化监控器 GUI。"""
        super().__init__()
        
        # 机器人状态数据
        self.low_state = None
        self.first_update = False
        self.start_time = time.time()
        self.update_count = 0
        self.last_update_time = time.time()
        
        # 变化检测相关
        self.previous_angles = {}  # 存储上一次的角度值
        self.angle_velocities = {}  # 存储角度变化速度
        self.last_angles_time = time.time()
        
        # 变化历史（用于检测活跃度）
        self.angle_history = {}  # 每个关节最近几个值的历史
        self.history_length = 10  # 保持最近10个值
        
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
        
        # 初始化历史数据结构
        for joint_idx, _, _ in self.joint_info:
            self.angle_history[joint_idx] = deque(maxlen=self.history_length)
            self.previous_angles[joint_idx] = 0.0
            self.angle_velocities[joint_idx] = 0.0
        
        self._setup_gui()
        self._setup_unitree()

    def _setup_gui(self):
        """初始化并设置所有 GUI 组件。"""
        # 检查是否已有应用实例
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        
        # 主窗口
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("🤖 G-1 Arm Joint Monitor")
        self.main_window.setGeometry(100, 100, 900, 700)
        
        # 中央控件
        central_widget = QtWidgets.QWidget()
        self.main_window.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 状态栏
        self.status_label = QtWidgets.QLabel("等待机器人连接...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
        main_layout.addWidget(self.status_label)
        
        # 创建表格显示关节数据
        self._setup_joint_table()
        main_layout.addWidget(self.joint_table)
        
        # 创建统计信息显示
        self._setup_stats_widget()
        main_layout.addWidget(self.stats_widget)
        
        # 控制按钮
        self._setup_control_buttons()
        main_layout.addWidget(self.button_widget)
        
        # 提高更新频率 - 从100ms改为50ms (20Hz)
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(50)  # 20Hz 更新，提高响应速度

    def _setup_joint_table(self):
        """设置用于显示关节数据的表格。"""
        self.joint_table = QtWidgets.QTableWidget()
        self.joint_table.setColumnCount(3)
        self.joint_table.setHorizontalHeaderLabels(["Left Arm", "Waist", "Right Arm"])
        self.joint_table.setRowCount(7)  # 最多7行（左臂关节数）
        
        # 设置表格样式
        self.joint_table.setAlternatingRowColors(True)
        self.joint_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        # 设置所有列等宽
        header = self.joint_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)  # 所有列平均分配宽度
        
        # 设置表格的最小高度，确保所有行都能显示
        self.joint_table.setMinimumHeight(300)
        
        # 设置行高自动调整
        self.joint_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        
        # 隐藏行号
        self.joint_table.verticalHeader().setVisible(False)
        
        # 设置字体
        font = QtGui.QFont("Consolas", 9)
        if not font.exactMatch():
            font = QtGui.QFont("Courier New", 9)
        self.joint_table.setFont(font)
        
        # 初始化表格内容
        self._init_table_content()

    def _init_table_content(self):
        """使用占位符初始化表格内容。"""
        left_arm_joints = [info for info in self.joint_info if info[2] == "left"]
        waist_joint = [info for info in self.joint_info if info[2] == "waist"]
        right_arm_joints = [info for info in self.joint_info if info[2] == "right"]
        
        for row in range(7):
            # 左臂
            if row < len(left_arm_joints):
                _, joint_name, _ = left_arm_joints[row]
                item = QtWidgets.QTableWidgetItem(f"{joint_name}: --")
                self.joint_table.setItem(row, 0, item)
            
            # 腰部（只在第一行）
            if row == 0 and waist_joint:
                _, joint_name, _ = waist_joint[0]
                item = QtWidgets.QTableWidgetItem(f"{joint_name}: --")
                self.joint_table.setItem(row, 1, item)
            elif row > 0:
                item = QtWidgets.QTableWidgetItem("")
                self.joint_table.setItem(row, 1, item)
            
            # 右臂
            if row < len(right_arm_joints):
                _, joint_name, _ = right_arm_joints[row]
                item = QtWidgets.QTableWidgetItem(f"{joint_name}: --")
                self.joint_table.setItem(row, 2, item)

    def _setup_stats_widget(self):
        """设置用于显示统计信息的小组件。"""
        self.stats_widget = QtWidgets.QGroupBox("统计信息")
        stats_layout = QtWidgets.QGridLayout(self.stats_widget)
        
        # 连接时间
        self.connection_time_label = QtWidgets.QLabel("连接时间: --")
        stats_layout.addWidget(self.connection_time_label, 0, 0)
        
        # 更新频率
        self.update_rate_label = QtWidgets.QLabel("更新频率: -- Hz")
        stats_layout.addWidget(self.update_rate_label, 0, 1)
        
        # 数据包计数
        self.packet_count_label = QtWidgets.QLabel("数据包计数: 0")
        stats_layout.addWidget(self.packet_count_label, 0, 2)
        
        # 活跃关节数
        self.active_joints_label = QtWidgets.QLabel("活跃关节: 0/15")
        stats_layout.addWidget(self.active_joints_label, 1, 0)
        
        # 最大变化速度
        self.max_velocity_label = QtWidgets.QLabel("最大变化速度: 0.000 rad/s")
        stats_layout.addWidget(self.max_velocity_label, 1, 1)
        
        # 灵敏度设置
        self.sensitivity_label = QtWidgets.QLabel("变化阈值: 0.001 rad")
        stats_layout.addWidget(self.sensitivity_label, 1, 2)

    def _setup_control_buttons(self):
        """设置控制按钮和灵敏度调整小组件。"""
        self.button_widget = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(self.button_widget)
        
        # 重置统计按钮
        reset_btn = QtWidgets.QPushButton("重置统计")
        reset_btn.clicked.connect(self._reset_stats)
        button_layout.addWidget(reset_btn)
        
        # 暂停/继续按钮
        self.pause_btn = QtWidgets.QPushButton("暂停更新")
        self.pause_btn.clicked.connect(self._toggle_pause)
        button_layout.addWidget(self.pause_btn)
        
        # 保存当前数据按钮
        save_btn = QtWidgets.QPushButton("保存当前数据")
        save_btn.clicked.connect(self._save_current_data)
        button_layout.addWidget(save_btn)
        
        # 灵敏度调整
        sensitivity_label = QtWidgets.QLabel("灵敏度:")
        button_layout.addWidget(sensitivity_label)
        
        self.sensitivity_spinbox = QtWidgets.QDoubleSpinBox()
        self.sensitivity_spinbox.setRange(0.0001, 0.1)
        self.sensitivity_spinbox.setSingleStep(0.0001)
        self.sensitivity_spinbox.setDecimals(4)
        self.sensitivity_spinbox.setValue(0.001)
        self.sensitivity_spinbox.valueChanged.connect(self._update_sensitivity)
        button_layout.addWidget(self.sensitivity_spinbox)
        
        button_layout.addStretch()

    def _setup_unitree(self):
        """初始化 Unitree SDK，订阅机器人低级状态数据。"""
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        # 设置回调函数为 LowStateHandler，并指定更新频率为 10Hz
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        """
        处理接收到的机器人低级状态消息的回调函数。

        Args:
            msg (LowState_): 从机器人接收到的状态数据。
        """
        self.low_state = msg
        self.update_count += 1
        
        # 计算角度变化速度
        current_time = time.time()
        dt = current_time - self.last_angles_time
        
        if dt > 0:
            for joint_idx, _, _ in self.joint_info:
                # 从motor_state中提取每个关节的当前角度值
                current_angle = msg.motor_state[joint_idx].q
                
                # 计算变化速度
                if joint_idx in self.previous_angles:
                    angle_diff = current_angle - self.previous_angles[joint_idx]
                    velocity = angle_diff / dt
                    self.angle_velocities[joint_idx] = velocity
                
                # 更新历史数据
                self.angle_history[joint_idx].append(current_angle)
                self.previous_angles[joint_idx] = current_angle
        
        self.last_angles_time = current_time
        
        if not self.first_update:
            self.first_update = True
            self.start_time = time.time()
            self.status_label.setText("✅ 机器人已连接，正在高频监控关节状态...")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def _update_display(self):
        """定时器调用的主更新函数，负责刷新表格和统计信息。"""
        if self.low_state is None:
            return
        
        # 更新表格
        self._update_table()
        
        # 更新统计信息
        if hasattr(self, 'pause_btn') and self.pause_btn.text() != "继续更新":
            self._update_stats()

    def _update_table(self):
        """使用最新的机器人数据更新关节表格。"""
        left_arm_joints = [info for info in self.joint_info if info[2] == "left"]
        waist_joint = [info for info in self.joint_info if info[2] == "waist"]
        right_arm_joints = [info for info in self.joint_info if info[2] == "right"]
        
        sensitivity = self.sensitivity_spinbox.value() if hasattr(self, 'sensitivity_spinbox') else 0.001
        
        for row in range(7):
            # 左臂
            if row < len(left_arm_joints):
                joint_idx, joint_name, _ = left_arm_joints[row]
                angle = self.low_state.motor_state[joint_idx].q
                velocity = self.angle_velocities.get(joint_idx, 0.0)
                
                # 取消颜色编码，改为两位小数显示
                text = f"{joint_name}: {angle:+.2f} ({velocity:+.2f})"
                item = self.joint_table.item(row, 0)
                item.setText(text)
                item.setBackground(QtGui.QColor("white"))  # 设置为白色背景
            
            # 腰部（只在第一行）
            if row == 0 and waist_joint:
                joint_idx, joint_name, _ = waist_joint[0]
                angle = self.low_state.motor_state[joint_idx].q
                velocity = self.angle_velocities.get(joint_idx, 0.0)
                text = f"{joint_name}: {angle:+.2f} ({velocity:+.2f})"
                item = self.joint_table.item(row, 1)
                item.setText(text)
                item.setBackground(QtGui.QColor("white"))  # 设置为白色背景
            
            # 右臂
            if row < len(right_arm_joints):
                joint_idx, joint_name, _ = right_arm_joints[row]
                angle = self.low_state.motor_state[joint_idx].q
                velocity = self.angle_velocities.get(joint_idx, 0.0)
                text = f"{joint_name}: {angle:+.2f} ({velocity:+.2f})"
                item = self.joint_table.item(row, 2)
                item.setText(text)
                item.setBackground(QtGui.QColor("white"))  # 设置为白色背景

    def _save_current_data(self):
        """将当前关节数据快照保存到文本文件。"""
        if self.low_state is None:
            QtWidgets.QMessageBox.information(self.main_window, "提示", "没有数据可保存")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window, "保存当前关节数据", "current_joint_data.txt", "Text files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"G-1 关节数据快照 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 60 + "\n\n")
                    
                    for joint_idx, joint_name, group in self.joint_info:
                        angle = self.low_state.motor_state[joint_idx].q
                        velocity = self.angle_velocities.get(joint_idx, 0.0)
                        # 保存时也使用两位小数
                        f.write(f"{joint_name}: {angle:+.2f} rad (速度: {velocity:+.2f} rad/s)\n")
                    
                    f.write(f"\n统计信息:\n")
                    f.write(f"数据包计数: {self.update_count}\n")
                    f.write(f"连接时长: {time.time() - self.start_time:.1f} 秒\n")
                    
                    # 保存活跃关节信息
                    sensitivity = self.sensitivity_spinbox.value()
                    active_joints = [name for joint_idx, name, _ in self.joint_info 
                                   if abs(self.angle_velocities.get(joint_idx, 0)) > sensitivity]
                    f.write(f"当前活跃关节: {', '.join(active_joints)}\n")
                
                QtWidgets.QMessageBox.information(self.main_window, "成功", f"数据已保存到 {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.main_window, "错误", f"保存失败: {e}")

    def _update_stats(self):
        """更新统计信息标签的内容。"""
        current_time = time.time()
        
        # 连接时间
        connection_duration = current_time - self.start_time
        hours = int(connection_duration // 3600)
        minutes = int((connection_duration % 3600) // 60)
        seconds = int(connection_duration % 60)
        self.connection_time_label.setText(f"连接时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # 更新频率
        if connection_duration > 0:
            update_rate = self.update_count / connection_duration
            self.update_rate_label.setText(f"更新频率: {update_rate:.1f} Hz")
        
        # 数据包计数
        self.packet_count_label.setText(f"数据包计数: {self.update_count}")
        
        # 活跃关节统计
        sensitivity = self.sensitivity_spinbox.value() if hasattr(self, 'sensitivity_spinbox') else 0.001
        active_count = sum(1 for joint_idx, _, _ in self.joint_info 
                          if abs(self.angle_velocities.get(joint_idx, 0)) > sensitivity)
        self.active_joints_label.setText(f"活跃关节: {active_count}/{len(self.joint_info)}")
        
        # 最大变化速度 - 也改为两位小数
        max_velocity = max(abs(v) for v in self.angle_velocities.values()) if self.angle_velocities else 0
        self.max_velocity_label.setText(f"最大变化速度: {max_velocity:.2f} rad/s")

    def _update_sensitivity(self, value: float):
        """
        当灵敏度微调框的值改变时更新标签。

        Args:
            value (float): 新的灵敏度阈值。
        """
        self.sensitivity_label.setText(f"变化阈值: {value:.4f} rad")

    def _reset_stats(self):
        """重置所有统计数据和历史记录。"""
        self.start_time = time.time()
        self.update_count = 0
        self.connection_time_label.setText("连接时间: 00:00:00")
        self.update_rate_label.setText("更新频率: -- Hz")
        self.packet_count_label.setText("数据包计数: 0")
        
        # 清空历史数据
        for joint_idx in self.angle_history:
            self.angle_history[joint_idx].clear()
            self.angle_velocities[joint_idx] = 0.0

    def _toggle_pause(self):
        """切换 GUI 的暂停/继续状态。"""
        if self.pause_btn.text() == "暂停更新":
            self.pause_btn.setText("继续更新")
        else:
            self.pause_btn.setText("暂停更新")

    def run(self):
        """
        显示 GUI 窗口并启动应用程序事件循环。

        Returns:
            int: 应用程序的退出代码。
        """
        # 显示窗口
        self.main_window.show()
        
        # 启动事件循环
        return self.app.exec()

def main():
    """程序主入口函数。"""
    print("G-1 机器人手臂关节监控器 (高灵敏度版本)")
    print("正在启动图形界面...")
    
    # 初始化通道工厂
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    
    # 创建并运行监控器
    monitor = ArmJointMonitorGUI()
    sys.exit(monitor.run())

if __name__ == '__main__':
    main()