#!/usr/bin/env python3
"""G-1 Arm Joint Monitor GUI - 使用 PySide6 和 pyqtgraph 的图形界面版本

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
│ │                           实时角度图表                                    │ │
│ │  [PyQtGraph 绘制的实时关节角度变化曲线]                                   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘

Requirements
------------
    pip install pyside6 pyqtgraph~=0.13
"""

import time
import sys
import os
from collections import deque
from typing import Dict, List, Tuple

# Qt 相关导入
try:
    from PySide6 import QtCore, QtWidgets, QtGui
    import pyqtgraph as pg
except ImportError as e:
    print(f"请安装必要的依赖: pip install pyside6 pyqtgraph~=0.13")
    print(f"错误: {e}")
    sys.exit(1)

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

class G1JointIndex:
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
    def __init__(self):
        super().__init__()
        
        # 机器人状态数据
        self.low_state = None
        self.first_update = False
        
        # 历史数据存储（用于绘图）
        self.max_history = 500  # 保存500个数据点
        self.time_data = deque(maxlen=self.max_history)
        self.joint_data: Dict[int, deque] = {}
        self.start_time = time.time()
        
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
        
        # 初始化历史数据存储
        for joint_idx, _, _ in self.joint_info:
            self.joint_data[joint_idx] = deque(maxlen=self.max_history)
        
        self._setup_gui()
        self._setup_unitree()

    def _setup_gui(self):
        """设置 GUI 界面"""
        # 创建应用
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        
        # 主窗口
        self.main_window = QtWidgets.QMainWindow()
        self.main_window.setWindowTitle("🤖 G-1 Arm Joint Monitor (GUI)")
        self.main_window.setGeometry(100, 100, 1200, 800)
        
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
        
        # 创建实时图表
        self._setup_plot_widget()
        main_layout.addWidget(self.plot_widget)
        
        # 控制按钮
        self._setup_control_buttons()
        main_layout.addWidget(self.button_widget)
        
        # 定时器更新显示
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # 10Hz 更新

    def _setup_joint_table(self):
        """设置关节数据表格"""
        self.joint_table = QtWidgets.QTableWidget()
        self.joint_table.setColumnCount(3)
        self.joint_table.setHorizontalHeaderLabels(["Left Arm", "Waist", "Right Arm"])
        self.joint_table.setRowCount(7)  # 最多7行（左臂关节数）
        
        # 设置表格样式
        self.joint_table.setAlternatingRowColors(True)
        self.joint_table.horizontalHeader().setStretchLastSection(True)
        self.joint_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        
        # 初始化表格内容
        self._init_table_content()

    def _init_table_content(self):
        """初始化表格内容"""
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

    def _setup_plot_widget(self):
        """设置绘图控件"""
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('w')
        
        # 创建绘图区域
        self.plot = self.plot_widget.addPlot(title="关节角度实时变化")
        self.plot.setLabel('left', '角度 (rad)')
        self.plot.setLabel('bottom', '时间 (s)')
        self.plot.addLegend()
        self.plot.showGrid(x=True, y=True)
        
        # 颜色映射
        self.colors = {
            'left': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
            'right': ['#FF7675', '#74B9FF', '#00B894', '#FDCB6E', '#E17055', '#A29BFE', '#FD79A8'],
            'waist': ['#2D3436']
        }
        
        # 创建曲线
        self.curves = {}
        color_idx = {'left': 0, 'right': 0, 'waist': 0}
        
        for joint_idx, joint_name, group in self.joint_info:
            color = self.colors[group][color_idx[group]]
            color_idx[group] += 1
            
            curve = self.plot.plot([], [], pen=color, name=joint_name)
            self.curves[joint_idx] = curve

    def _setup_control_buttons(self):
        """设置控制按钮"""
        self.button_widget = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(self.button_widget)
        
        # 清空历史数据按钮
        clear_btn = QtWidgets.QPushButton("清空历史数据")
        clear_btn.clicked.connect(self._clear_history)
        button_layout.addWidget(clear_btn)
        
        # 暂停/继续按钮
        self.pause_btn = QtWidgets.QPushButton("暂停更新")
        self.pause_btn.clicked.connect(self._toggle_pause)
        button_layout.addWidget(self.pause_btn)
        
        # 保存数据按钮
        save_btn = QtWidgets.QPushButton("保存数据")
        save_btn.clicked.connect(self._save_data)
        button_layout.addWidget(save_btn)
        
        button_layout.addStretch()

    def _setup_unitree(self):
        """设置 Unitree SDK"""
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        """处理机器人状态消息"""
        self.low_state = msg
        if not self.first_update:
            self.first_update = True
            self.status_label.setText("✅ 机器人已连接，正在监控关节状态...")
            self.status_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")

    def _update_display(self):
        """更新显示内容"""
        if self.low_state is None:
            return
        
        # 更新时间数据
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        
        # 更新表格
        self._update_table()
        
        # 更新图表
        if hasattr(self, 'pause_btn') and self.pause_btn.text() != "继续更新":
            self._update_plot()

    def _update_table(self):
        """更新关节数据表格"""
        left_arm_joints = [info for info in self.joint_info if info[2] == "left"]
        waist_joint = [info for info in self.joint_info if info[2] == "waist"]
        right_arm_joints = [info for info in self.joint_info if info[2] == "right"]
        
        for row in range(7):
            # 左臂
            if row < len(left_arm_joints):
                joint_idx, joint_name, _ = left_arm_joints[row]
                angle = self.low_state.motor_state[joint_idx].q
                text = f"{joint_name}: {angle:+.3f}"
                self.joint_table.item(row, 0).setText(text)
            
            # 腰部（只在第一行）
            if row == 0 and waist_joint:
                joint_idx, joint_name, _ = waist_joint[0]
                angle = self.low_state.motor_state[joint_idx].q
                text = f"{joint_name}: {angle:+.3f}"
                self.joint_table.item(row, 1).setText(text)
            
            # 右臂
            if row < len(right_arm_joints):
                joint_idx, joint_name, _ = right_arm_joints[row]
                angle = self.low_state.motor_state[joint_idx].q
                text = f"{joint_name}: {angle:+.3f}"
                self.joint_table.item(row, 2).setText(text)

    def _update_plot(self):
        """更新实时图表"""
        # 存储当前关节角度数据
        for joint_idx, _, _ in self.joint_info:
            angle = self.low_state.motor_state[joint_idx].q
            self.joint_data[joint_idx].append(angle)
        
        # 更新曲线
        if len(self.time_data) > 1:
            time_array = list(self.time_data)
            for joint_idx in self.joint_data:
                angle_array = list(self.joint_data[joint_idx])
                if len(angle_array) == len(time_array):
                    self.curves[joint_idx].setData(time_array, angle_array)

    def _clear_history(self):
        """清空历史数据"""
        self.time_data.clear()
        for joint_idx in self.joint_data:
            self.joint_data[joint_idx].clear()
        self.start_time = time.time()
        
        # 清空曲线
        for curve in self.curves.values():
            curve.setData([], [])

    def _toggle_pause(self):
        """暂停/继续更新"""
        if self.pause_btn.text() == "暂停更新":
            self.pause_btn.setText("继续更新")
        else:
            self.pause_btn.setText("暂停更新")

    def _save_data(self):
        """保存数据到文件"""
        if not self.time_data:
            QtWidgets.QMessageBox.information(self.main_window, "提示", "没有数据可保存")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.main_window, "保存关节数据", "joint_data.csv", "CSV files (*.csv)"
        )
        
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # 写入标题
                    headers = ['Time'] + [name for _, name, _ in self.joint_info]
                    writer.writerow(headers)
                    
                    # 写入数据
                    for i, t in enumerate(self.time_data):
                        row = [t]
                        for joint_idx, _, _ in self.joint_info:
                            if i < len(self.joint_data[joint_idx]):
                                row.append(self.joint_data[joint_idx][i])
                            else:
                                row.append("")
                        writer.writerow(row)
                
                QtWidgets.QMessageBox.information(self.main_window, "成功", f"数据已保存到 {filename}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self.main_window, "错误", f"保存失败: {e}")

    def run(self):
        """运行 GUI"""
        # 等待机器人连接
        while not self.first_update:
            self.app.processEvents()
            time.sleep(0.1)
        
        # 显示窗口
        self.main_window.show()
        
        # 启动事件循环
        return self.app.exec()

def main():
    """主函数"""
    print("G-1 机器人手臂关节监控器 (GUI版本)")
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