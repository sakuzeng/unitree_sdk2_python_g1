#!/usr/bin/env python3
# filepath: /home/sakuzeng/Coding/projects/unitree/unitree_sdk2_python_g1/comprehensive_visualization.py
"""
综合可视化展示系统

功能特性:
- 多窗口可视化架构
- 3D 点云和SLAM地图实时显示
- 2D 占用网格和路径规划可视化
- 实时数据监控面板
- 机器人状态和传感器数据展示
- 交互式控制界面

依赖包:
- open3d>=0.16.0 - 3D可视化
- opencv-python - 2D图像处理
- matplotlib - 图表绘制
- PyQt5/PySide2 - GUI界面 (可选)
- numpy, threading

运行方法:
    python comprehensive_visualization.py [--mode MODE] [--layout LAYOUT]

参数说明:
- --mode: 可视化模式 (3d/2d/dashboard/all)
- --layout: 窗口布局 (single/multi/grid)
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg

# ---------------------------------------------------------------------------
# 可视化数据结构
# ---------------------------------------------------------------------------

@dataclass
class VisualizationData:
    """可视化数据容器"""
    # 点云数据
    raw_points: Optional[np.ndarray] = None
    slam_map: Optional[np.ndarray] = None
    
    # 位姿和轨迹
    current_pose: Optional[np.ndarray] = None
    trajectory: List[np.ndarray] = None
    
    # 占用网格和路径
    occupancy_grid: Optional[np.ndarray] = None
    planned_path: Optional[List[Tuple[float, float]]] = None
    
    # 机器人状态
    robot_velocity: Optional[Tuple[float, float, float]] = None
    sensor_status: Dict[str, Any] = None
    
    # 统计信息
    frame_count: int = 0
    processing_time: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = []
        if self.sensor_status is None:
            self.sensor_status = {}

# ---------------------------------------------------------------------------
# 3D 可视化器 (增强版)
# ---------------------------------------------------------------------------

class Enhanced3DViewer:
    """
    增强的3D可视化器
    
    功能:
    - 多层点云显示 (原始/SLAM/历史)
    - 动态轨迹可视化
    - 路径规划显示
    - 机器人模型可视化
    - 交互式视角控制
    """
    
    def __init__(self, window_name: str = "3D SLAM Visualization", 
                 width: int = 1600, height: int = 900):
        """
        初始化3D可视化器
        
        Args:
            window_name (str): 窗口名称
            width (int): 窗口宽度
            height (int): 窗口高度
        """
        self.window_name = window_name
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name=window_name, width=width, height=height)
        
        # 几何体管理
        self._geometries = {}
        self._initialize_geometries()
        
        # 数据缓冲
        self._data_buffer = deque(maxlen=100)
        self._trajectory_points = deque(maxlen=1000)
        
        # 显示控制
        self._show_raw_points = True
        self._show_slam_map = True
        self._show_trajectory = True
        self._show_path = True
        self._show_grid = True
        
        # 性能监控
        self._fps_counter = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0
        
        self._first_update = True
    
    def _initialize_geometries(self):
        """初始化几何体"""
        # 原始点云 (红色)
        self._geometries['raw_points'] = o3d.geometry.PointCloud()
        self._geometries['raw_points'].paint_uniform_color([1.0, 0.2, 0.2])
        
        # SLAM地图 (绿色)
        self._geometries['slam_map'] = o3d.geometry.PointCloud()
        self._geometries['slam_map'].paint_uniform_color([0.2, 1.0, 0.2])
        
        # 轨迹线 (蓝色)
        self._geometries['trajectory'] = o3d.geometry.LineSet()
        self._geometries['trajectory'].paint_uniform_color([0.2, 0.2, 1.0])
        
        # 规划路径 (黄色)
        self._geometries['planned_path'] = o3d.geometry.LineSet()
        self._geometries['planned_path'].paint_uniform_color([1.0, 1.0, 0.2])
        
        # 机器人位姿坐标系
        self._geometries['robot_pose'] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # 原点坐标系
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        self._geometries['origin'] = origin_frame
        
        # 添加所有几何体到可视化器
        for name, geometry in self._geometries.items():
            self._vis.add_geometry(geometry)
    
    def update_data(self, data: VisualizationData):
        """
        更新可视化数据
        
        Args:
            data (VisualizationData): 新的可视化数据
        """
        self._data_buffer.append(data)
        
        # 更新原始点云
        if data.raw_points is not None and self._show_raw_points:
            self._update_raw_points(data.raw_points)
        
        # 更新SLAM地图
        if data.slam_map is not None and self._show_slam_map:
            self._update_slam_map(data.slam_map)
        
        # 更新位姿和轨迹
        if data.current_pose is not None:
            self._update_robot_pose(data.current_pose)
            if self._show_trajectory:
                self._update_trajectory(data.current_pose)
        
        # 更新路径
        if data.planned_path is not None and self._show_path:
            self._update_planned_path(data.planned_path)
        
        # 更新FPS计数
        self._update_fps()
    
    def _update_raw_points(self, points: np.ndarray):
        """更新原始点云"""
        if points.size == 0:
            return
        
        # 下采样大点云
        if points.shape[0] > 50000:
            indices = np.random.choice(points.shape[0], 50000, replace=False)
            points = points[indices]
        
        self._geometries['raw_points'].points = o3d.utility.Vector3dVector(points)
        self._vis.update_geometry(self._geometries['raw_points'])
    
    def _update_slam_map(self, slam_map: np.ndarray):
        """更新SLAM地图"""
        if slam_map.size == 0:
            return
        
        # 下采样大地图
        if slam_map.shape[0] > 100000:
            step = max(1, slam_map.shape[0] // 100000)
            slam_map = slam_map[::step]
        
        self._geometries['slam_map'].points = o3d.utility.Vector3dVector(slam_map)
        self._vis.update_geometry(self._geometries['slam_map'])
    
    def _update_robot_pose(self, pose: np.ndarray):
        """更新机器人位姿"""
        # 移除旧的位姿
        self._vis.remove_geometry(self._geometries['robot_pose'], reset_bounding_box=False)
        
        # 创建新的位姿坐标系
        size = 1.0
        if len(self._geometries['slam_map'].points) > 0:
            # 根据地图大小调整坐标系大小
            points = np.asarray(self._geometries['slam_map'].points)
            map_extent = np.max(np.linalg.norm(points, axis=1))
            size = max(0.5, min(3.0, map_extent * 0.05))
        
        self._geometries['robot_pose'] = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._geometries['robot_pose'].transform(pose)
        self._vis.add_geometry(self._geometries['robot_pose'], reset_bounding_box=False)
    
    def _update_trajectory(self, pose: np.ndarray):
        """更新轨迹线"""
        position = pose[:3, 3]
        self._trajectory_points.append(position)
        
        if len(self._trajectory_points) > 1:
            # 创建轨迹线段
            points = list(self._trajectory_points)
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            
            self._geometries['trajectory'].points = o3d.utility.Vector3dVector(points)
            self._geometries['trajectory'].lines = o3d.utility.Vector2iVector(lines)
            self._vis.update_geometry(self._geometries['trajectory'])
    
    def _update_planned_path(self, path: List[Tuple[float, float]]):
        """更新规划路径"""
        if len(path) < 2:
            return
        
        # 转换为3D点 (抬高0.1米显示)
        path_3d = [(x, y, 0.1) for x, y in path]
        lines = [[i, i + 1] for i in range(len(path_3d) - 1)]
        
        self._geometries['planned_path'].points = o3d.utility.Vector3dVector(path_3d)
        self._geometries['planned_path'].lines = o3d.utility.Vector2iVector(lines)
        self._vis.update_geometry(self._geometries['planned_path'])
    
    def _update_fps(self):
        """更新FPS计数"""
        self._fps_counter += 1
        current_time = time.time()
        
        if current_time - self._last_fps_time >= 1.0:
            self._current_fps = self._fps_counter / (current_time - self._last_fps_time)
            self._fps_counter = 0
            self._last_fps_time = current_time
    
    def render(self) -> bool:
        """
        渲染一帧
        
        Returns:
            bool: 窗口是否仍然活跃
        """
        # 首次更新时调整视角
        if self._first_update and len(self._data_buffer) > 0:
            self._vis.reset_view_point(True)
            self._first_update = False
        
        # 更新渲染器
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        
        return alive
    
    def toggle_display(self, element: str):
        """
        切换显示元素
        
        Args:
            element (str): 元素名称 ('raw_points', 'slam_map', 'trajectory', 'path', 'grid')
        """
        if element == 'raw_points':
            self._show_raw_points = not self._show_raw_points
        elif element == 'slam_map':
            self._show_slam_map = not self._show_slam_map
        elif element == 'trajectory':
            self._show_trajectory = not self._show_trajectory
        elif element == 'path':
            self._show_path = not self._show_path
        elif element == 'grid':
            self._show_grid = not self._show_grid
    
    def get_status(self) -> Dict[str, Any]:
        """获取可视化器状态"""
        return {
            'fps': self._current_fps,
            'data_buffer_size': len(self._data_buffer),
            'trajectory_length': len(self._trajectory_points),
            'display_settings': {
                'raw_points': self._show_raw_points,
                'slam_map': self._show_slam_map,
                'trajectory': self._show_trajectory,
                'path': self._show_path,
                'grid': self._show_grid
            }
        }
    
    def close(self):
        """关闭可视化器"""
        self._vis.destroy_window()

# ---------------------------------------------------------------------------
# 2D 可视化器
# ---------------------------------------------------------------------------

class Enhanced2DViewer:
    """
    增强的2D可视化器
    
    功能:
    - 多面板布局
    - 实时数据图表
    - 占用网格显示
    - 路径规划可视化
    - 机器人状态监控
    """
    
    def __init__(self, window_name: str = "2D SLAM Dashboard", 
                 width: int = 1400, height: int = 800):
        """
        初始化2D可视化器
        
        Args:
            window_name (str): 窗口名称
            width (int): 窗口宽度
            height (int): 窗口高度
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # 数据缓冲
        self._data_buffer = deque(maxlen=200)
        self._metrics_history = {
            'timestamps': deque(maxlen=100),
            'processing_times': deque(maxlen=100),
            'frame_counts': deque(maxlen=100),
            'velocities': deque(maxlen=100)
        }
        
        # 画布设置
        self._setup_canvas()
        
        # 状态
        self._running = True
        self._current_goal = None
    
    def _setup_canvas(self):
        """设置画布布局"""
        # 创建主画布
        self._canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # 定义面板区域
        self._panels = {
            'occupancy_grid': (0, 0, 600, 600),           # 左上: 占用网格
            'trajectory_plot': (600, 0, 800, 300),        # 右上: 轨迹图
            'metrics_plot': (600, 300, 800, 300),         # 右中: 性能图
            'status_panel': (0, 600, 600, 200),           # 左下: 状态面板
            'control_panel': (600, 600, 800, 200)         # 右下: 控制面板
        }
    
    def update_data(self, data: VisualizationData):
        """
        更新可视化数据
        
        Args:
            data (VisualizationData): 新的可视化数据
        """
        self._data_buffer.append(data)
        
        # 更新指标历史
        current_time = time.time()
        self._metrics_history['timestamps'].append(current_time)
        self._metrics_history['processing_times'].append(data.processing_time)
        self._metrics_history['frame_counts'].append(data.frame_count)
        
        if data.robot_velocity:
            velocity_norm = np.linalg.norm(data.robot_velocity[:2])
            self._metrics_history['velocities'].append(velocity_norm)
        else:
            self._metrics_history['velocities'].append(0.0)
    
    def render(self) -> bool:
        """
        渲染2D界面
        
        Returns:
            bool: 是否继续运行
        """
        # 清空画布
        self._canvas.fill(40)  # 深灰色背景
        
        if len(self._data_buffer) == 0:
            cv2.putText(self._canvas, "Waiting for data...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(self.window_name, self._canvas)
            return cv2.waitKey(1) & 0xFF != ord('q')
        
        latest_data = self._data_buffer[-1]
        
        # 渲染各个面板
        self._render_occupancy_grid_panel(latest_data)
        self._render_trajectory_plot_panel()
        self._render_metrics_plot_panel()
        self._render_status_panel(latest_data)
        self._render_control_panel()
        
        # 绘制面板边框
        self._draw_panel_borders()
        
        # 显示画布
        cv2.imshow(self.window_name, self._canvas)
        
        # 处理鼠标事件
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('r'):  # 重置视图
            self._metrics_history = {k: deque(maxlen=100) for k in self._metrics_history.keys()}
        
        return True
    
    def _render_occupancy_grid_panel(self, data: VisualizationData):
        """渲染占用网格面板"""
        x, y, w, h = self._panels['occupancy_grid']
        panel = self._canvas[y:y+h, x:x+w]
        
        if data.occupancy_grid is not None:
            # 缩放占用网格到面板大小
            grid_resized = cv2.resize(data.occupancy_grid, (w-20, h-80))
            
            # 转换为BGR格式
            if len(grid_resized.shape) == 2:
                grid_bgr = cv2.cvtColor(grid_resized, cv2.COLOR_GRAY2BGR)
            else:
                grid_bgr = cv2.cvtColor(grid_resized, cv2.COLOR_RGB2BGR)
            
            # 放置网格
            panel[60:60+grid_bgr.shape[0], 10:10+grid_bgr.shape[1]] = grid_bgr
            
            # 绘制路径
            if data.planned_path and len(data.planned_path) > 1:
                self._draw_path_on_grid(panel, data.planned_path, (10, 60), 
                                      (grid_bgr.shape[1], grid_bgr.shape[0]))
            
            # 绘制机器人位置
            if data.current_pose is not None:
                robot_pos = data.current_pose[:2, 3]
                self._draw_robot_on_grid(panel, robot_pos, (10, 60), 
                                       (grid_bgr.shape[1], grid_bgr.shape[0]))
        
        # 面板标题
        cv2.putText(panel, "Occupancy Grid", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 图例
        legend_y = h - 40
        cv2.rectangle(panel, (10, legend_y), (30, legend_y+15), (0, 0, 0), -1)
        cv2.putText(panel, "Obstacle", (35, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(panel, (120, legend_y), (140, legend_y+15), (255, 255, 255), -1)
        cv2.putText(panel, "Free", (145, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.rectangle(panel, (200, legend_y), (220, legend_y+15), (128, 128, 128), -1)
        cv2.putText(panel, "Unknown", (225, legend_y+12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _render_trajectory_plot_panel(self):
        """渲染轨迹图面板"""
        x, y, w, h = self._panels['trajectory_plot']
        panel = self._canvas[y:y+h, x:x+w]
        
        # 面板标题
        cv2.putText(panel, "Trajectory", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self._data_buffer) < 2:
            return
        
        # 提取位置数据
        positions = []
        for data in list(self._data_buffer)[-50:]:  # 最近50个点
            if data.current_pose is not None:
                pos = data.current_pose[:2, 3]
                positions.append(pos)
        
        if len(positions) < 2:
            return
        
        positions = np.array(positions)
        
        # 计算显示范围
        margin = 2.0
        min_x, max_x = positions[:, 0].min() - margin, positions[:, 0].max() + margin
        min_y, max_y = positions[:, 1].min() - margin, positions[:, 1].max() + margin
        
        # 转换到面板坐标
        plot_area = (20, 50, w-40, h-70)
        plot_w, plot_h = plot_area[2], plot_area[3]
        
        def world_to_panel(pos):
            px = int((pos[0] - min_x) / (max_x - min_x) * plot_w + plot_area[0])
            py = int((pos[1] - min_y) / (max_y - min_y) * plot_h + plot_area[1])
            return (px, py)
        
        # 绘制轨迹线
        for i in range(len(positions) - 1):
            pt1 = world_to_panel(positions[i])
            pt2 = world_to_panel(positions[i + 1])
            cv2.line(panel, pt1, pt2, (0, 255, 0), 2)
        
        # 绘制当前位置
        if len(positions) > 0:
            current_pt = world_to_panel(positions[-1])
            cv2.circle(panel, current_pt, 5, (0, 0, 255), -1)
        
        # 绘制坐标轴标签
        cv2.putText(panel, f"X: {min_x:.1f} - {max_x:.1f}m", 
                   (plot_area[0], h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, f"Y: {min_y:.1f} - {max_y:.1f}m", 
                   (plot_area[0], h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def _render_metrics_plot_panel(self):
        """渲染性能指标面板"""
        x, y, w, h = self._panels['metrics_plot']
        panel = self._canvas[y:y+h, x:x+w]
        
        # 面板标题
        cv2.putText(panel, "Performance Metrics", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(self._metrics_history['timestamps']) < 2:
            return
        
        # 绘制处理时间图表
        times = list(self._metrics_history['processing_times'])
        velocities = list(self._metrics_history['velocities'])
        
        plot_area = (20, 50, w-40, h-70)
        plot_w, plot_h = plot_area[2], plot_area[3]
        
        if times:
            # 处理时间 (上半部分)
            max_time = max(times) if times else 1.0
            for i in range(len(times) - 1):
                x1 = int(i / len(times) * plot_w + plot_area[0])
                x2 = int((i + 1) / len(times) * plot_w + plot_area[0])
                y1 = int(plot_area[1] + plot_h//2 - times[i] / max_time * plot_h//2)
                y2 = int(plot_area[1] + plot_h//2 - times[i+1] / max_time * plot_h//2)
                cv2.line(panel, (x1, y1), (x2, y2), (255, 100, 100), 2)
            
            cv2.putText(panel, f"Proc Time: {times[-1]:.3f}s", 
                       (plot_area[0], plot_area[1] + plot_h//2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        
        if velocities:
            # 速度 (下半部分)
            max_vel = max(velocities) if velocities else 1.0
            for i in range(len(velocities) - 1):
                x1 = int(i / len(velocities) * plot_w + plot_area[0])
                x2 = int((i + 1) / len(velocities) * plot_w + plot_area[0])
                y1 = int(plot_area[1] + plot_h//2 + velocities[i] / max_vel * plot_h//2)
                y2 = int(plot_area[1] + plot_h//2 + velocities[i+1] / max_vel * plot_h//2)
                cv2.line(panel, (x1, y1), (x2, y2), (100, 255, 100), 2)
            
            cv2.putText(panel, f"Velocity: {velocities[-1]:.2f}m/s", 
                       (plot_area[0], plot_area[1] + plot_h - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        # 中线
        mid_y = plot_area[1] + plot_h//2
        cv2.line(panel, (plot_area[0], mid_y), (plot_area[0] + plot_w, mid_y), 
                (100, 100, 100), 1)
    
    def _render_status_panel(self, data: VisualizationData):
        """渲染状态面板"""
        x, y, w, h = self._panels['status_panel']
        panel = self._canvas[y:y+h, x:x+w]
        
        # 面板标题
        cv2.putText(panel, "System Status", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 状态信息
        status_lines = []
        
        # 基本信息
        status_lines.append(f"Frame: {data.frame_count}")
        status_lines.append(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        
        # 传感器状态
        if data.sensor_status:
            for sensor, status in data.sensor_status.items():
                status_lines.append(f"{sensor}: {status}")
        
        # 机器人状态
        if data.robot_velocity:
            vx, vy, omega = data.robot_velocity
            status_lines.append(f"Vel: [{vx:.2f}, {vy:.2f}, {omega:.2f}]")
        
        # 目标信息
        if self._current_goal:
            status_lines.append(f"Goal: [{self._current_goal[0]:.1f}, {self._current_goal[1]:.1f}]")
        else:
            status_lines.append("Goal: None")
        
        # 渲染状态文本
        for i, line in enumerate(status_lines[:8]):  # 最多8行
            y_pos = 60 + i * 20
            cv2.putText(panel, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _render_control_panel(self):
        """渲染控制面板"""
        x, y, w, h = self._panels['control_panel']
        panel = self._canvas[y:y+h, x:x+w]
        
        # 面板标题
        cv2.putText(panel, "Controls", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 控制说明
        controls = [
            "Left Click: Set Goal",
            "Right Click: Clear Goal", 
            "Q: Quit",
            "R: Reset Metrics",
            "Space: Pause/Resume"
        ]
        
        for i, control in enumerate(controls):
            y_pos = 60 + i * 25
            cv2.putText(panel, control, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_panel_borders(self):
        """绘制面板边框"""
        for panel_name, (x, y, w, h) in self._panels.items():
            cv2.rectangle(self._canvas, (x, y), (x+w, y+h), (100, 100, 100), 1)
    
    def _draw_path_on_grid(self, panel: np.ndarray, path: List[Tuple[float, float]], 
                          offset: Tuple[int, int], size: Tuple[int, int]):
        """在网格上绘制路径"""
        if len(path) < 2:
            return
        
        # 假设网格范围为 [-50, 50] 米
        grid_range = 50.0
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # 转换到面板坐标
            px1 = int((x1 + grid_range) / (2 * grid_range) * size[0] + offset[0])
            py1 = int((-y1 + grid_range) / (2 * grid_range) * size[1] + offset[1])
            px2 = int((x2 + grid_range) / (2 * grid_range) * size[0] + offset[0])
            py2 = int((-y2 + grid_range) / (2 * grid_range) * size[1] + offset[1])
            
            if (0 <= px1 < panel.shape[1] and 0 <= py1 < panel.shape[0] and
                0 <= px2 < panel.shape[1] and 0 <= py2 < panel.shape[0]):
                cv2.line(panel, (px1, py1), (px2, py2), (0, 0, 255), 3)
    
    def _draw_robot_on_grid(self, panel: np.ndarray, robot_pos: np.ndarray, 
                           offset: Tuple[int, int], size: Tuple[int, int]):
        """在网格上绘制机器人位置"""
        grid_range = 50.0
        x, y = robot_pos
        
        px = int((x + grid_range) / (2 * grid_range) * size[0] + offset[0])
        py = int((-y + grid_range) / (2 * grid_range) * size[1] + offset[1])
        
        if 0 <= px < panel.shape[1] and 0 <= py < panel.shape[0]:
            cv2.circle(panel, (px, py), 8, (255, 0, 0), -1)
            cv2.circle(panel, (px, py), 12, (255, 255, 255), 2)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        grid_panel = self._panels['occupancy_grid']
        gx, gy, gw, gh = grid_panel
        
        # 检查是否在占用网格面板内
        if gx <= x <= gx + gw and gy <= y <= gy + gh:
            if event == cv2.EVENT_LBUTTONDOWN:
                # 左键设置目标
                grid_x = x - gx - 10
                grid_y = y - gy - 60
                
                if 0 <= grid_x < gw-20 and 0 <= grid_y < gh-80:
                    # 转换到世界坐标
                    grid_range = 50.0
                    world_x = (grid_x / (gw-20)) * (2 * grid_range) - grid_range
                    world_y = -((grid_y / (gh-80)) * (2 * grid_range) - grid_range)
                    
                    self._current_goal = (world_x, world_y)
                    print(f"[2D Viewer] 设置目标: ({world_x:.2f}, {world_y:.2f})")
            
            elif event == cv2.EVENT_RBUTTONDOWN:
                # 右键清除目标
                self._current_goal = None
                print("[2D Viewer] 清除目标")
    
    def set_goal(self, goal: Optional[Tuple[float, float]]):
        """设置导航目标"""
        self._current_goal = goal
    
    def get_goal(self) -> Optional[Tuple[float, float]]:
        """获取当前目标"""
        return self._current_goal
    
    def close(self):
        """关闭可视化器"""
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# 数据仪表板
# ---------------------------------------------------------------------------

class MetricsDashboard:
    """
    数据仪表板
    
    功能:
    - 实时性能监控
    - 历史数据图表
    - 系统状态显示
    - 数据导出功能
    """
    
    def __init__(self):
        """初始化数据仪表板"""
        # 使用matplotlib创建图表
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('SLAM System Dashboard', fontsize=16)
        
        # 数据存储
        self.metrics_data = {
            'timestamps': deque(maxlen=200),
            'processing_times': deque(maxlen=200),
            'frame_rates': deque(maxlen=200),
            'velocities_x': deque(maxlen=200),
            'velocities_y': deque(maxlen=200),
            'velocities_omega': deque(maxlen=200),
            'trajectory_length': deque(maxlen=200),
            'map_points': deque(maxlen=200)
        }
        
        # 设置子图
        self._setup_subplots()
        
        # 动画
        self.animation = FuncAnimation(self.fig, self._update_plots, 
                                      interval=100, blit=False, cache_frame_data=False)
        
        self.start_time = time.time()
    
    def _setup_subplots(self):
        """设置子图"""
        # 处理时间和帧率
        self.axes[0, 0].set_title('Processing Time & Frame Rate')
        self.axes[0, 0].set_ylabel('Time (s) / FPS')
        self.axes[0, 0].set_xlabel('Time')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        # 机器人速度
        self.axes[0, 1].set_title('Robot Velocities')
        self.axes[0, 1].set_ylabel('Velocity (m/s, rad/s)')
        self.axes[0, 1].set_xlabel('Time')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        # 轨迹长度和地图点数
        self.axes[1, 0].set_title('Map Statistics')
        self.axes[1, 0].set_ylabel('Count')
        self.axes[1, 0].set_xlabel('Time')
        self.axes[1, 0].grid(True, alpha=0.3)
        
        # 系统状态 (文本)
        self.axes[1, 1].set_title('System Status')
        self.axes[1, 1].axis('off')
        
        plt.tight_layout()
    
    def update_data(self, data: VisualizationData):
        """更新仪表板数据"""
        current_time = time.time() - self.start_time
        
        self.metrics_data['timestamps'].append(current_time)
        self.metrics_data['processing_times'].append(data.processing_time)
        
        # 计算帧率
        if len(self.metrics_data['timestamps']) > 1:
            dt = self.metrics_data['timestamps'][-1] - self.metrics_data['timestamps'][-2]
            fps = 1.0 / dt if dt > 0 else 0.0
            self.metrics_data['frame_rates'].append(fps)
        else:
            self.metrics_data['frame_rates'].append(0.0)
        
        # 机器人速度
        if data.robot_velocity:
            vx, vy, omega = data.robot_velocity
            self.metrics_data['velocities_x'].append(vx)
            self.metrics_data['velocities_y'].append(vy)
            self.metrics_data['velocities_omega'].append(omega)
        else:
            self.metrics_data['velocities_x'].append(0.0)
            self.metrics_data['velocities_y'].append(0.0)
            self.metrics_data['velocities_omega'].append(0.0)
        
        # 地图统计
        trajectory_len = len(data.trajectory) if data.trajectory else 0
        map_points = data.slam_map.shape[0] if data.slam_map is not None else 0
        
        self.metrics_data['trajectory_length'].append(trajectory_len)
        self.metrics_data['map_points'].append(map_points)
    
    def _update_plots(self, frame):
        """更新图表"""
        if len(self.metrics_data['timestamps']) < 2:
            return
        
        # 清除所有子图
        for ax in self.axes.flat:
            ax.clear()
        
        self._setup_subplots()
        
        timestamps = list(self.metrics_data['timestamps'])
        
        # 处理时间和帧率
        if self.metrics_data['processing_times']:
            proc_times = list(self.metrics_data['processing_times'])
            self.axes[0, 0].plot(timestamps, proc_times, 'r-', label='Proc Time', linewidth=2)
        
        if self.metrics_data['frame_rates']:
            frame_rates = list(self.metrics_data['frame_rates'])
            ax_twin = self.axes[0, 0].twinx()
            ax_twin.plot(timestamps, frame_rates, 'g-', label='FPS', linewidth=2)
            ax_twin.set_ylabel('FPS')
        
        # 机器人速度
        if self.metrics_data['velocities_x']:
            vx = list(self.metrics_data['velocities_x'])
            vy = list(self.metrics_data['velocities_y'])
            omega = list(self.metrics_data['velocities_omega'])
            
            self.axes[0, 1].plot(timestamps, vx, 'r-', label='Vx', linewidth=2)
            self.axes[0, 1].plot(timestamps, vy, 'g-', label='Vy', linewidth=2)
            self.axes[0, 1].plot(timestamps, omega, 'b-', label='Omega', linewidth=2)
            self.axes[0, 1].legend()
        
        # 地图统计
        if self.metrics_data['trajectory_length']:
            traj_len = list(self.metrics_data['trajectory_length'])
            map_pts = list(self.metrics_data['map_points'])
            
            self.axes[1, 0].plot(timestamps, traj_len, 'c-', label='Trajectory Poses', linewidth=2)
            
            ax_twin = self.axes[1, 0].twinx()
            ax_twin.plot(timestamps, map_pts, 'm-', label='Map Points', linewidth=2)
            ax_twin.set_ylabel('Map Points')
            
            self.axes[1, 0].legend(loc='upper left')
            ax_twin.legend(loc='upper right')
        
        # 系统状态文本
        if len(self.metrics_data['timestamps']) > 0:
            latest_time = timestamps[-1]
            latest_proc_time = self.metrics_data['processing_times'][-1]
            latest_fps = self.metrics_data['frame_rates'][-1]
            
            status_text = f"""
Runtime: {latest_time:.1f}s
Processing Time: {latest_proc_time:.3f}s
FPS: {latest_fps:.1f}
Trajectory Poses: {self.metrics_data['trajectory_length'][-1]}
Map Points: {self.metrics_data['map_points'][-1]}
            """.strip()
            
            self.axes[1, 1].text(0.1, 0.9, status_text, transform=self.axes[1, 1].transAxes,
                                fontsize=12, verticalalignment='top', 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="gray", alpha=0.5))
    
    def save_data(self, filename: str):
        """保存数据到文件"""
        data_dict = {k: list(v) for k, v in self.metrics_data.items()}
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"[Dashboard] 数据已保存到: {filename}")
    
    def show(self):
        """显示仪表板"""
        plt.show()
    
    def close(self):
        """关闭仪表板"""
        plt.close(self.fig)

# ---------------------------------------------------------------------------
# 综合可视化管理器
# ---------------------------------------------------------------------------

class ComprehensiveVisualizationManager:
    """
    综合可视化管理器
    
    统一管理所有可视化组件，提供简单的接口供SLAM系统使用
    """
    
    def __init__(self, mode: str = "all", layout: str = "multi"):
        """
        初始化可视化管理器
        
        Args:
            mode (str): 可视化模式 ('3d', '2d', 'dashboard', 'all')
            layout (str): 窗口布局 ('single', 'multi', 'grid')
        """
        self.mode = mode
        self.layout = layout
        
        # 初始化可视化组件
        self.viewers = {}
        
        if mode in ['3d', 'all']:
            self.viewers['3d'] = Enhanced3DViewer()
        
        if mode in ['2d', 'all']:
            self.viewers['2d'] = Enhanced2DViewer()
        
        if mode in ['dashboard', 'all']:
            self.viewers['dashboard'] = MetricsDashboard()
        
        # 线程控制
        self._running = True
        self._update_thread = None
        
        # 数据队列
        self._data_queue = deque(maxlen=10)
        self._data_lock = threading.Lock()
        
        print(f"[VisMgr] 初始化完成 (模式: {mode}, 布局: {layout})")
    
    def update_data(self, data: VisualizationData):
        """
        更新可视化数据
        
        Args:
            data (VisualizationData): 新的可视化数据
        """
        with self._data_lock:
            self._data_queue.append(data)
    
    def start(self):
        """启动可视化"""
        if self._update_thread is None:
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
        
        # 显示仪表板 (如果存在)
        if 'dashboard' in self.viewers:
            self.viewers['dashboard'].show()
    
    def _update_loop(self):
        """更新循环"""
        while self._running:
            try:
                # 获取最新数据
                latest_data = None
                with self._data_lock:
                    if self._data_queue:
                        latest_data = self._data_queue[-1]
                
                if latest_data is None:
                    time.sleep(0.01)
                    continue
                
                # 更新各个可视化器
                if '3d' in self.viewers:
                    self.viewers['3d'].update_data(latest_data)
                    if not self.viewers['3d'].render():
                        self._running = False
                        break
                
                if '2d' in self.viewers:
                    self.viewers['2d'].update_data(latest_data)
                    if not self.viewers['2d'].render():
                        self._running = False
                        break
                
                if 'dashboard' in self.viewers:
                    self.viewers['dashboard'].update_data(latest_data)
                
                time.sleep(0.01)  # 限制更新频率
                
            except Exception as e:
                print(f"[VisMgr] 更新循环错误: {e}")
                time.sleep(0.1)
    
    def get_current_goal(self) -> Optional[Tuple[float, float]]:
        """获取当前设置的目标"""
        if '2d' in self.viewers:
            return self.viewers['2d'].get_goal()
        return None
    
    def set_goal(self, goal: Optional[Tuple[float, float]]):
        """设置导航目标"""
        if '2d' in self.viewers:
            self.viewers['2d'].set_goal(goal)
    
    def toggle_3d_display(self, element: str):
        """切换3D显示元素"""
        if '3d' in self.viewers:
            self.viewers['3d'].toggle_display(element)
    
    def get_status(self) -> Dict[str, Any]:
        """获取可视化状态"""
        status = {
            'mode': self.mode,
            'layout': self.layout,
            'running': self._running,
            'data_queue_size': len(self._data_queue)
        }
        
        for name, viewer in self.viewers.items():
            if hasattr(viewer, 'get_status'):
                status[f'{name}_status'] = viewer.get_status()
        
        return status
    
    def save_dashboard_data(self, filename: str):
        """保存仪表板数据"""
        if 'dashboard' in self.viewers:
            self.viewers['dashboard'].save_data(filename)
    
    def stop(self):
        """停止可视化"""
        self._running = False
        
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        # 关闭所有可视化器
        for viewer in self.viewers.values():
            if hasattr(viewer, 'close'):
                viewer.close()
        
        print("[VisMgr] 可视化已关闭")

# ---------------------------------------------------------------------------
# 示例集成函数
# ---------------------------------------------------------------------------

def create_sample_data() -> VisualizationData:
    """创建示例数据用于测试"""
    # 生成随机点云
    raw_points = np.random.randn(1000, 3) * 5
    slam_map = np.random.randn(5000, 3) * 10
    
    # 生成示例位姿
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 0.0]
    
    # 生成示例路径
    path = [(i * 0.5, np.sin(i * 0.1) * 2) for i in range(20)]
    
    # 生成示例占用网格
    grid = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    
    return VisualizationData(
        raw_points=raw_points,
        slam_map=slam_map,
        current_pose=pose,
        trajectory=[pose],
        occupancy_grid=grid,
        planned_path=path,
        robot_velocity=(0.5, 0.2, 0.1),
        sensor_status={"lidar": "OK", "camera": "OK"},
        frame_count=123,
        processing_time=0.015,
        timestamp=time.time()
    )

def demo_visualization():
    """演示可视化系统"""
    print("[Demo] 启动可视化演示...")
    
    # 创建可视化管理器
    vis_manager = ComprehensiveVisualizationManager(mode="all", layout="multi")
    vis_manager.start()
    
    try:
        # 模拟数据更新
        for i in range(1000):
            # 创建示例数据
            data = create_sample_data()
            data.frame_count = i
            
            # 更新可视化
            vis_manager.update_data(data)
            
            time.sleep(0.1)  # 10Hz更新
            
            # 检查是否需要退出
            if not vis_manager._running:
                break
    
    except KeyboardInterrupt:
        print("\n[Demo] 收到中断信号")
    
    finally:
        vis_manager.stop()

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="综合可视化展示系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=['3d', '2d', 'dashboard', 'all'], 
                       default='all', help="可视化模式")
    parser.add_argument("--layout", choices=['single', 'multi', 'grid'], 
                       default='multi', help="窗口布局")
    parser.add_argument("--demo", action='store_true', 
                       help="运行演示模式")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_visualization()
    else:
        print(f"[Main] 创建可视化管理器 (模式: {args.mode}, 布局: {args.layout})")
        print("使用 --demo 参数运行演示")

if __name__ == "__main__":
    main()