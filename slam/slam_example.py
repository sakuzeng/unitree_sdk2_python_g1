#!/usr/bin/env python3
# filepath: /home/sakuzeng/Coding/projects/unitree/unitree_sdk2_python_g1/comprehensive_slam.py
"""
全面的 SLAM 应用程序

功能特性:
- 实时激光雷达点云处理和 SLAM 建图
- 原始点云、2D 占用网格和路径规划
- 支持室内/室外场景预设配置
- 3D 可视化地图和位姿显示
- 实时路径规划和导航控制
- 挂载方向自动校正和数值稳定性优化
- 自动保存地图、轨迹和占用网格

依赖包:
- Livox-SDK2 共享库
- Python 包: numpy, open3d>=0.16.0, kiss-icp, opencv-python, scikit-image

运行方法:
    python comprehensive_slam.py [--config CONFIG] [--output_dir OUTPUT]

环境变量配置:
- LIVOX_PRESET: 场景预设 (indoor/outdoor, 默认 indoor)
- LIVOX_MOUNT: 挂载方向 (normal/upside_down, 默认 upside_down)
- LIDAR_TILT_DEG: 倾斜角度校正 (度数, 默认 0)
- SLAM_MAP_RESOLUTION: 占用网格分辨率 (米, 默认 0.05)
- SLAM_MAP_SIZE: 地图尺寸 (米, 默认 200)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import morphology

# ---------------------------------------------------------------------------
# 配置参数
# ---------------------------------------------------------------------------

# 数据保存目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 场景预设
PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

# 地图参数
MAP_RESOLUTION = float(os.environ.get("SLAM_MAP_RESOLUTION", "0.05"))  # 5cm
MAP_SIZE = int(os.environ.get("SLAM_MAP_SIZE", "200"))  # 200m
MAP_CENTER = MAP_SIZE // 2

# 挂载校正
_TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()

# 构建校正矩阵
_R_MOUNT = None
if MOUNT == "upside_down" or abs(_TILT_DEG) > 1e-3:
    _R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)
    
    if abs(_TILT_DEG) > 1e-3:
        _rad = math.radians(-_TILT_DEG)
        c, s = math.cos(_rad), math.sin(_rad)
        
        if _TILT_AXIS == "x":
            _R_TILT = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float)
        elif _TILT_AXIS == "y":
            _R_TILT = np.array([
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, .0, 1.0],
            ], dtype=float)
        else:  # 'z'
            _R_TILT = np.array([
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=float)
    else:
        _R_TILT = np.eye(4)
    
    _R_TOTAL = _R_TILT @ _R_FLIP
    if not np.allclose(_R_TOTAL, np.eye(4)):
        _R_MOUNT = _R_TOTAL

# 场景预设配置
_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        "frame_time": 0.35,
        "frame_packets": 200,
        "voxel_size": 0.4,
        "max_range": 30.0,
        "downsample_limit": 5_000_000,
        "min_motion": 0.03,
        "conv_criterion": 5e-5,
        "max_iters": 800,
        "grid_height_min": -0.5,
        "grid_height_max": 2.0,
        "obstacle_threshold": 0.6,
        "free_threshold": 0.4,
    },
    "outdoor": {
        "frame_time": 0.20,
        "frame_packets": 120,
        "voxel_size": 1.0,
        "max_range": 120.0,
        "downsample_limit": 3_000_000,
        "min_motion": 0.10,
        "conv_criterion": 1e-4,
        "max_iters": 500,
        "grid_height_min": -1.0,
        "grid_height_max": 3.0,
        "obstacle_threshold": 0.65,
        "free_threshold": 0.35,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"未知预设 '{PRESET}'. 请选择 {list(_PRESETS.keys())} 中的一个.")

_P = _PRESETS[PRESET]

# ---------------------------------------------------------------------------
# KISS-ICP 导入
# ---------------------------------------------------------------------------

KissICP = None
try:
    from kiss_icp.pipeline import KissICP
except Exception:
    try:
        from kiss_icp.pybind import KissICP
    except Exception as e:
        raise SystemExit(f"无法导入 KISS-ICP: {e}")

# Livox SDK 导入
try:
    from livox2_python import Livox2 as _Livox
except Exception:
    print("[INFO] livox2_python 不可用 – 回退到 SDK1.")
    from livox_python import Livox as _Livox

# ---------------------------------------------------------------------------
# 占用网格类
# ---------------------------------------------------------------------------

class OccupancyGrid:
    """
    2D 占用网格地图实现
    
    支持:
    - 基于点云的占用概率更新
    - 形态学处理和噪声滤除
    - 路径规划和碰撞检测
    - 地图保存和加载
    """
    
    def __init__(self, size: int = MAP_SIZE, resolution: float = MAP_RESOLUTION):
        """
        初始化占用网格
        
        Args:
            size (int): 网格大小 (像素)
            resolution (float): 分辨率 (米/像素)
        """
        self.size = size
        self.resolution = resolution
        self.center = size // 2
        
        # 占用概率网格 [0, 1] - 0.5为未知
        self.grid = np.full((size, size), 0.5, dtype=np.float32)
        
        # 更新计数器，用于计算平均占用概率
        self.update_count = np.zeros((size, size), dtype=np.int32)
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 形态学内核
        self._erosion_kernel = morphology.disk(2)
        self._dilation_kernel = morphology.disk(1)
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        世界坐标转换为网格坐标
        
        Args:
            x (float): 世界坐标 X
            y (float): 世界坐标 Y
            
        Returns:
            Tuple[int, int]: 网格坐标 (row, col)
        """
        col = int(x / self.resolution + self.center)
        row = int(-y / self.resolution + self.center)  # Y轴翻转
        return row, col
    
    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """
        网格坐标转换为世界坐标
        
        Args:
            row (int): 网格行
            col (int): 网格列
            
        Returns:
            Tuple[float, float]: 世界坐标 (x, y)
        """
        x = (col - self.center) * self.resolution
        y = -(row - self.center) * self.resolution
        return x, y
    
    def update_with_pointcloud(self, points: np.ndarray, sensor_pos: np.ndarray):
        """
        使用点云更新占用网格
        
        Args:
            points (np.ndarray): 点云数据 (N, 3)
            sensor_pos (np.ndarray): 传感器位置 (3,)
        """
        if points.size == 0:
            return
        
        # 高度过滤
        valid_height = (points[:, 2] >= _P["grid_height_min"]) & (points[:, 2] <= _P["grid_height_max"])
        if not valid_height.any():
            return
        
        points_2d = points[valid_height][:, :2]
        sensor_2d = sensor_pos[:2]
        
        with self._lock:
            # 标记障碍物
            for point in points_2d:
                row, col = self.world_to_grid(point[0], point[1])
                if 0 <= row < self.size and 0 <= col < self.size:
                    self.grid[row, col] = min(1.0, self.grid[row, col] + 0.1)
                    self.update_count[row, col] += 1
            
            # 标记自由空间 (射线追踪)
            sensor_row, sensor_col = self.world_to_grid(sensor_2d[0], sensor_2d[1])
            
            # 对部分点进行射线追踪以减少计算量
            sample_points = points_2d[::max(1, len(points_2d) // 100)]
            
            for point in sample_points:
                point_row, point_col = self.world_to_grid(point[0], point[1])
                
                # Bresenham 射线追踪
                ray_points = self._bresenham_line(sensor_row, sensor_col, point_row, point_col)
                
                for r, c in ray_points[:-1]:  # 排除终点
                    if 0 <= r < self.size and 0 <= c < self.size:
                        self.grid[r, c] = max(0.0, self.grid[r, c] - 0.02)
                        self.update_count[r, c] += 1
    
    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """
        Bresenham 直线算法
        
        Args:
            x0, y0: 起点坐标
            x1, y1: 终点坐标
            
        Returns:
            List[Tuple[int, int]]: 直线上的点
        """
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x0 += sx
            
            if e2 < dx:
                err += dx
                y0 += sy
        
        return points
    
    def get_binary_map(self, obstacle_threshold: float = None, free_threshold: float = None) -> np.ndarray:
        """
        获取二值化地图
        
        Args:
            obstacle_threshold (float): 障碍物阈值
            free_threshold (float): 自由空间阈值
            
        Returns:
            np.ndarray: 二值化地图 (0=自由, 1=障碍物, 0.5=未知)
        """
        if obstacle_threshold is None:
            obstacle_threshold = _P["obstacle_threshold"]
        if free_threshold is None:
            free_threshold = _P["free_threshold"]
        
        with self._lock:
            binary_map = np.full_like(self.grid, 0.5)  # 未知区域
            binary_map[self.grid >= obstacle_threshold] = 1.0  # 障碍物
            binary_map[self.grid <= free_threshold] = 0.0  # 自由空间
            
            # 形态学处理
            obstacle_mask = (binary_map == 1.0)
            if obstacle_mask.any():
                # 先膨胀后腐蚀，去除噪声
                obstacle_mask = morphology.binary_dilation(obstacle_mask, self._dilation_kernel)
                obstacle_mask = morphology.binary_erosion(obstacle_mask, self._erosion_kernel)
                binary_map[obstacle_mask] = 1.0
            
            return binary_map
    
    def is_free(self, x: float, y: float, threshold: float = None) -> bool:
        """
        检查指定位置是否为自由空间
        
        Args:
            x (float): 世界坐标 X
            y (float): 世界坐标 Y
            threshold (float): 自由空间阈值
            
        Returns:
            bool: 是否为自由空间
        """
        if threshold is None:
            threshold = _P["free_threshold"]
        
        row, col = self.world_to_grid(x, y)
        if 0 <= row < self.size and 0 <= col < self.size:
            with self._lock:
                return self.grid[row, col] <= threshold
        return False
    
    def get_visualization_image(self) -> np.ndarray:
        """
        获取可视化图像
        
        Returns:
            np.ndarray: RGB 图像
        """
        binary_map = self.get_binary_map()
        
        # 转换为RGB图像
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        
        # 自由空间 - 白色
        img[binary_map == 0.0] = [255, 255, 255]
        
        # 障碍物 - 黑色
        img[binary_map == 1.0] = [0, 0, 0]
        
        # 未知区域 - 灰色
        img[binary_map == 0.5] = [128, 128, 128]
        
        return img
    
    def save(self, file_path: Path) -> bool:
        """
        保存占用网格到文件
        
        Args:
            file_path (Path): 保存路径
            
        Returns:
            bool: 保存成功返回 True
        """
        try:
            data = {
                "grid": self.grid.tolist(),
                "update_count": self.update_count.tolist(),
                "size": self.size,
                "resolution": self.resolution,
                "center": self.center,
                "preset": PRESET,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"[ERROR] 保存占用网格失败: {e}")
            return False

# ---------------------------------------------------------------------------
# 路径规划器
# ---------------------------------------------------------------------------

class PathPlanner:
    """
    基于 A* 算法的路径规划器
    """
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        """
        初始化路径规划器
        
        Args:
            occupancy_grid (OccupancyGrid): 占用网格地图
        """
        self.grid = occupancy_grid
        self.path_cache = {}  # 路径缓存
        self.last_plan_time = 0
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float], 
                  robot_radius: float = 0.3) -> Optional[List[Tuple[float, float]]]:
        """
        规划从起点到终点的路径
        
        Args:
            start (Tuple[float, float]): 起点世界坐标
            goal (Tuple[float, float]): 终点世界坐标
            robot_radius (float): 机器人半径 (米)
            
        Returns:
            Optional[List[Tuple[float, float]]]: 路径点列表，失败返回 None
        """
        # 检查缓存
        cache_key = (round(start[0], 1), round(start[1], 1), 
                    round(goal[0], 1), round(goal[1], 1))
        current_time = time.time()
        
        if (cache_key in self.path_cache and 
            current_time - self.last_plan_time < 2.0):
            return self.path_cache[cache_key]
        
        # 转换为网格坐标
        start_grid = self.grid.world_to_grid(*start)
        goal_grid = self.grid.world_to_grid(*goal)
        
        # 获取二值化地图
        binary_map = self.grid.get_binary_map()
        
        # 膨胀障碍物以考虑机器人半径
        robot_radius_pixels = int(robot_radius / self.grid.resolution)
        if robot_radius_pixels > 0:
            obstacle_mask = (binary_map >= 0.9)
            dilated_obstacles = morphology.binary_dilation(
                obstacle_mask, morphology.disk(robot_radius_pixels)
            )
            binary_map[dilated_obstacles] = 1.0
        
        # A* 路径规划
        path_grid = self._astar_search(start_grid, goal_grid, binary_map)
        
        if path_grid is None:
            self.path_cache[cache_key] = None
            return None
        
        # 转换回世界坐标
        path_world = []
        for row, col in path_grid:
            x, y = self.grid.grid_to_world(row, col)
            path_world.append((x, y))
        
        # 路径平滑
        path_world = self._smooth_path(path_world, binary_map)
        
        # 缓存结果
        self.path_cache[cache_key] = path_world
        self.last_plan_time = current_time
        
        return path_world
    
    def _astar_search(self, start: Tuple[int, int], goal: Tuple[int, int], 
                      binary_map: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        A* 搜索算法
        
        Args:
            start (Tuple[int, int]): 起点网格坐标
            goal (Tuple[int, int]): 终点网格坐标
            binary_map (np.ndarray): 二值化地图
            
        Returns:
            Optional[List[Tuple[int, int]]]: 网格路径，失败返回 None
        """
        from heapq import heappush, heappop
        
        if (not self._is_valid_point(start, binary_map) or 
            not self._is_valid_point(goal, binary_map)):
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                      (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid_point(neighbor, binary_map):
                    continue
                
                # 计算移动代价
                move_cost = 1.4 if abs(dr) + abs(dc) == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _is_valid_point(self, point: Tuple[int, int], binary_map: np.ndarray) -> bool:
        """
        检查点是否有效
        
        Args:
            point (Tuple[int, int]): 网格坐标
            binary_map (np.ndarray): 二值化地图
            
        Returns:
            bool: 是否有效
        """
        row, col = point
        return (0 <= row < binary_map.shape[0] and 
                0 <= col < binary_map.shape[1] and 
                binary_map[row, col] < 0.9)
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        A* 启发式函数 (欧几里得距离)
        
        Args:
            a (Tuple[int, int]): 点 A
            b (Tuple[int, int]): 点 B
            
        Returns:
            float: 距离
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _smooth_path(self, path: List[Tuple[float, float]], 
                     binary_map: np.ndarray) -> List[Tuple[float, float]]:
        """
        路径平滑处理
        
        Args:
            path (List[Tuple[float, float]]): 原始路径
            binary_map (np.ndarray): 二值化地图
            
        Returns:
            List[Tuple[float, float]]: 平滑后的路径
        """
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            # 找到最远的可直达点
            farthest = i + 1
            for j in range(i + 2, len(path)):
                if self._line_of_sight(path[i], path[j], binary_map):
                    farthest = j
                else:
                    break
            
            smoothed.append(path[farthest])
            i = farthest
        
        return smoothed
    
    def _line_of_sight(self, start: Tuple[float, float], end: Tuple[float, float], 
                       binary_map: np.ndarray) -> bool:
        """
        检查两点间是否有视线
        
        Args:
            start (Tuple[float, float]): 起点
            end (Tuple[float, float]): 终点
            binary_map (np.ndarray): 二值化地图
            
        Returns:
            bool: 是否有视线
        """
        start_grid = self.grid.world_to_grid(*start)
        end_grid = self.grid.world_to_grid(*end)
        
        # Bresenham 直线检查
        line_points = self.grid._bresenham_line(*start_grid, *end_grid)
        
        for row, col in line_points:
            if (0 <= row < binary_map.shape[0] and 
                0 <= col < binary_map.shape[1] and 
                binary_map[row, col] >= 0.9):
                return False
        
        return True

# ---------------------------------------------------------------------------
# 3D 可视化器
# ---------------------------------------------------------------------------

class ComprehensiveViewer:
    """
    综合 3D 可视化器，支持原始点云、SLAM地图、占用网格和路径显示
    """
    
    def __init__(self):
        """初始化可视化器"""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Comprehensive SLAM", width=1600, height=900)
        
        # 点云几何体
        self._raw_pcd = o3d.geometry.PointCloud()
        self._slam_pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._raw_pcd)
        self._vis.add_geometry(self._slam_pcd)
        
        # 位姿可视化
        self._pose_frame = None
        
        # 路径可视化
        self._path_lines = None
        
        # 占用网格可视化 (投影到地面)
        self._grid_mesh = None
        
        # 数据队列 (线程安全)
        self._latest_raw = None
        self._latest_slam = None
        self._latest_pose = None
        self._latest_path = None
        self._latest_grid = None
        
        self._first_update = True
    
    def push_raw_points(self, xyz: np.ndarray):
        """推送原始点云数据"""
        self._latest_raw = xyz
    
    def push_slam_map(self, xyz: np.ndarray, pose: np.ndarray):
        """推送SLAM地图和位姿"""
        self._latest_slam = xyz
        self._latest_pose = pose
    
    def push_path(self, path: List[Tuple[float, float]]):
        """推送路径数据"""
        self._latest_path = path
    
    def push_occupancy_grid(self, grid_image: np.ndarray):
        """推送占用网格"""
        self._latest_grid = grid_image
    
    def tick(self) -> bool:
        """更新可视化"""
        updated = False
        
        # 更新原始点云
        if self._latest_raw is not None:
            # 下采样显示
            raw_points = self._latest_raw
            if raw_points.shape[0] > 50000:
                indices = np.random.choice(raw_points.shape[0], 50000, replace=False)
                raw_points = raw_points[indices]
            
            self._raw_pcd.points = o3d.utility.Vector3dVector(raw_points)
            self._raw_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            self._vis.update_geometry(self._raw_pcd)
            self._latest_raw = None
            updated = True
        
        # 更新SLAM地图
        if self._latest_slam is not None:
            slam_points = self._latest_slam
            if slam_points.shape[0] > 100000:
                step = max(1, slam_points.shape[0] // 100000)
                slam_points = slam_points[::step]
            
            self._slam_pcd.points = o3d.utility.Vector3dVector(slam_points)
            self._slam_pcd.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色
            self._vis.update_geometry(self._slam_pcd)
            self._latest_slam = None
            updated = True
        
        # 更新位姿
        if self._latest_pose is not None:
            self._update_pose_visualization(self._latest_pose)
            self._latest_pose = None
            updated = True
        
        # 更新路径
        if self._latest_path is not None:
            self._update_path_visualization(self._latest_path)
            self._latest_path = None
            updated = True
        
        # 更新占用网格
        if self._latest_grid is not None:
            self._update_grid_visualization(self._latest_grid)
            self._latest_grid = None
            updated = True
        
        # 首次更新时调整视角
        if self._first_update and updated:
            self._vis.reset_view_point(True)
            self._first_update = False
        
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive
    
    def _update_pose_visualization(self, pose: np.ndarray):
        """更新位姿可视化"""
        if self._pose_frame is not None:
            self._vis.remove_geometry(self._pose_frame, reset_bounding_box=False)
        
        self._pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self._pose_frame.transform(pose)
        self._vis.add_geometry(self._pose_frame, reset_bounding_box=False)
    
    def _update_path_visualization(self, path: List[Tuple[float, float]]):
        """更新路径可视化"""
        if self._path_lines is not None:
            self._vis.remove_geometry(self._path_lines, reset_bounding_box=False)
        
        if len(path) > 1:
            # 创建路径线段
            path_3d = [(x, y, 0.1) for x, y in path]  # 稍微抬高显示
            
            lines = []
            for i in range(len(path_3d) - 1):
                lines.append([i, i + 1])
            
            self._path_lines = o3d.geometry.LineSet()
            self._path_lines.points = o3d.utility.Vector3dVector(path_3d)
            self._path_lines.lines = o3d.utility.Vector2iVector(lines)
            self._path_lines.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色
            
            self._vis.add_geometry(self._path_lines, reset_bounding_box=False)
    
    def _update_grid_visualization(self, grid_image: np.ndarray):
        """更新占用网格可视化"""
        if self._grid_mesh is not None:
            self._vis.remove_geometry(self._grid_mesh, reset_bounding_box=False)
        
        # 创建网格纹理平面
        height, width = grid_image.shape[:2]
        
        # 创建平面网格
        vertices = []
        triangles = []
        colors = []
        
        for i in range(height):
            for j in range(width):
                # 世界坐标
                x = (j - width // 2) * MAP_RESOLUTION
                y = -(i - height // 2) * MAP_RESOLUTION
                z = -0.1  # 稍微低于地面
                
                vertices.append([x, y, z])
                
                # 颜色映射
                if len(grid_image.shape) == 3:
                    color = grid_image[i, j] / 255.0
                else:
                    gray = grid_image[i, j] / 255.0
                    color = [gray, gray, gray]
                colors.append(color)
                
                # 创建三角形 (每个像素两个三角形)
                if i < height - 1 and j < width - 1:
                    idx = i * width + j
                    triangles.append([idx, idx + 1, idx + width])
                    triangles.append([idx + 1, idx + width + 1, idx + width])
        
        if vertices:
            self._grid_mesh = o3d.geometry.TriangleMesh()
            self._grid_mesh.vertices = o3d.utility.Vector3dVector(vertices)
            self._grid_mesh.triangles = o3d.utility.Vector3iVector(triangles)
            self._grid_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            self._vis.add_geometry(self._grid_mesh, reset_bounding_box=False)
    
    def close(self):
        """关闭可视化器"""
        self._vis.destroy_window()

# ---------------------------------------------------------------------------
# 综合SLAM系统
# ---------------------------------------------------------------------------

class ComprehensiveSLAM(_Livox):
    """
    综合SLAM系统，整合点云处理、建图、占用网格和路径规划
    """
    
    def __init__(self):
        """初始化综合SLAM系统"""
        # 初始化Livox SDK
        _sdk_kwargs = {}
        if _Livox.__name__ == "Livox2":
            _sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])
        
        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.164", **_sdk_kwargs)
        except TypeError:
            super().__init__()
        
        # 初始化KISS-ICP
        try:
            from kiss_icp.config import load_config
            cfg = load_config(config_file=None, max_range=_P["max_range"])
        except Exception as e:
            raise SystemExit(f"KISS-ICP配置失败: {e}")
        
        self._apply_config(cfg)
        self._slam = KissICP(cfg)
        
        # 初始化子系统
        self._occupancy_grid = OccupancyGrid()
        self._path_planner = PathPlanner(self._occupancy_grid)
        self._viewer = ComprehensiveViewer()
        
        # 数据存储
        self._trajectory = []
        self._raw_cloud_buffer = deque(maxlen=10)  # 保持最近10帧原始点云
        
        # 状态变量
        self._frame_count = 0
        self._last_frame_time = time.time()
        self._last_pose = np.eye(4)
        self._current_goal = None
        self._current_path = None
        
        # 统计信息
        self._start_time = datetime.now()
        self._total_frames = 0
        
        print(f"[INFO] 综合SLAM系统已初始化 (预设: {PRESET})")
    
    def _apply_config(self, cfg):
        """应用KISS-ICP配置"""
        try:
            if hasattr(cfg, 'mapping'):
                if hasattr(cfg.mapping, 'voxel_size'):
                    cfg.mapping.voxel_size = _P["voxel_size"]
                if hasattr(cfg.mapping, 'max_points_per_voxel'):
                    cfg.mapping.max_points_per_voxel = 15
        except Exception as e:
            print(f"[WARNING] 设置映射参数失败: {e}")
        
        try:
            if hasattr(cfg, 'adaptive_threshold'):
                if hasattr(cfg.adaptive_threshold, 'min_motion_th'):
                    cfg.adaptive_threshold.min_motion_th = _P["min_motion"]
        except Exception as e:
            print(f"[WARNING] 设置自适应阈值失败: {e}")
        
        try:
            if hasattr(cfg, 'registration'):
                if hasattr(cfg.registration, 'convergence_criterion'):
                    cfg.registration.convergence_criterion = _P["conv_criterion"]
                if hasattr(cfg.registration, 'max_num_iterations'):
                    cfg.registration.max_num_iterations = _P["max_iters"]
        except Exception as e:
            print(f"[WARNING] 设置配准参数失败: {e}")
    
    def handle_points(self, xyz: np.ndarray):
        """处理激光雷达点云数据"""
        current_time = time.time()
        self._frame_count += 1
        
        # 帧率控制
        if current_time - self._last_frame_time < 0.1:  # 最大10Hz
            return
        self._last_frame_time = current_time
        
        # 数据预处理
        xyz = self._preprocess_points(xyz)
        if xyz.size == 0:
            return
        
        # 保存原始点云
        self._raw_cloud_buffer.append(xyz.copy())
        
        # SLAM处理
        try:
            # 生成时间戳
            num_points = xyz.shape[0]
            timestamps = np.linspace(0.0, 0.1, num_points, dtype=np.float64)
            
            # 注册帧
            try:
                self._slam.register_frame(xyz, timestamps)
            except TypeError:
                self._slam.register_frame(xyz)
            
            self._total_frames += 1
            
        except Exception as e:
            print(f"[ERROR] SLAM注册失败: {e}")
            return
        
        # 获取地图和位姿
        try:
            slam_map = self._get_slam_map()
            current_pose = self._get_current_pose()
            
            if slam_map is not None and slam_map.size > 0:
                # 更新占用网格
                sensor_pos = current_pose[:3, 3]
                self._occupancy_grid.update_with_pointcloud(slam_map, sensor_pos)
                
                # 路径规划 (如果有目标)
                if self._current_goal is not None:
                    self._update_path_planning(current_pose)
                
                # 推送到可视化器
                self._update_visualization(xyz, slam_map, current_pose)
            
        except Exception as e:
            print(f"[ERROR] 数据处理失败: {e}")
    
    def _preprocess_points(self, xyz: np.ndarray) -> np.ndarray:
        """点云预处理"""
        if xyz.size == 0:
            return xyz
        
        # 移除无效点
        valid_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid_mask]
        
        if xyz.size == 0:
            return xyz
        
        # 移除机器人自身反射
        r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", "0.20"))
        dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", "0.15"))
        
        dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
        close = dist_xy < r_xy
        near_plane = np.abs(xyz[:, 2]) < dz
        mask = ~(close & near_plane)
        xyz = xyz[mask]
        
        # 应用挂载校正
        if _R_MOUNT is not None:
            xyz = (xyz @ _R_MOUNT[:3, :3].T).astype(xyz.dtype, copy=False)
        
        return xyz
    
    def _get_slam_map(self) -> Optional[np.ndarray]:
        """获取SLAM地图"""
        try:
            if hasattr(self._slam, 'get_map'):
                return self._slam.get_map()
            elif hasattr(self._slam, 'local_map'):
                return self._slam.local_map.point_cloud()
        except Exception as e:
            print(f"[WARNING] 获取SLAM地图失败: {e}")
        return None
    
    def _get_current_pose(self) -> np.ndarray:
        """获取当前位姿"""
        try:
            pose = self._slam.last_pose.copy() if hasattr(self._slam, 'last_pose') else np.eye(4)
            if _R_MOUNT is not None:
                pose = _R_MOUNT @ pose
            
            self._trajectory.append(pose.copy())
            self._last_pose = pose
            return pose
        except Exception:
            return self._last_pose
    
    def _update_path_planning(self, current_pose: np.ndarray):
        """更新路径规划"""
        if self._current_goal is None:
            return
        
        current_pos = current_pose[:3, 3]
        start = (current_pos[0], current_pos[1])
        
        # 规划路径
        path = self._path_planner.plan_path(start, self._current_goal)
        
        if path is not None:
            self._current_path = path
            print(f"[INFO] 路径规划成功，长度: {len(path)} 点")
        else:
            print("[WARNING] 路径规划失败")
            self._current_path = None
    
    def _update_visualization(self, raw_xyz: np.ndarray, slam_map: np.ndarray, pose: np.ndarray):
        """更新可视化"""
        # 推送原始点云
        self._viewer.push_raw_points(raw_xyz)
        
        # 推送SLAM地图
        self._viewer.push_slam_map(slam_map, pose)
        
        # 推送路径
        if self._current_path is not None:
            self._viewer.push_path(self._current_path)
        
        # 推送占用网格
        grid_image = self._occupancy_grid.get_visualization_image()
        self._viewer.push_occupancy_grid(grid_image)
    
    def set_navigation_goal(self, x: float, y: float):
        """设置导航目标"""
        self._current_goal = (x, y)
        print(f"[INFO] 设置导航目标: ({x:.2f}, {y:.2f})")
    
    def clear_navigation_goal(self):
        """清除导航目标"""
        self._current_goal = None
        self._current_path = None
        print("[INFO] 清除导航目标")
    
    def get_current_path(self) -> Optional[List[Tuple[float, float]]]:
        """获取当前路径"""
        return self._current_path
    
    def save_all_data(self) -> bool:
        """保存所有数据"""
        print("[INFO] 正在保存综合SLAM数据...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = DATA_DIR / f"comprehensive_slam_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        
        success_count = 0
        total_saves = 0
        
        # 1. 保存最终SLAM地图
        try:
            slam_map = self._get_slam_map()
            if slam_map is not None and slam_map.size > 0:
                total_saves += 1
                map_file = session_dir / "slam_map.ply"
                if self._save_point_cloud(slam_map, map_file):
                    success_count += 1
        except Exception as e:
            print(f"[ERROR] 保存SLAM地图失败: {e}")
        
        # 2. 保存最近的原始点云
        try:
            if self._raw_cloud_buffer:
                merged_raw = np.concatenate(list(self._raw_cloud_buffer), axis=0)
                total_saves += 1
                raw_file = session_dir / "raw_points.ply"
                if self._save_point_cloud(merged_raw, raw_file):
                    success_count += 1
        except Exception as e:
            print(f"[ERROR] 保存原始点云失败: {e}")
        
        # 3. 保存轨迹
        if self._trajectory:
            total_saves += 1
            trajectory_file = session_dir / "trajectory.txt"
            if self._save_trajectory(self._trajectory, trajectory_file):
                success_count += 1
        
        # 4. 保存占用网格
        total_saves += 1
        grid_file = session_dir / "occupancy_grid.json"
        if self._occupancy_grid.save(grid_file):
            success_count += 1
        
        # 5. 保存占用网格图像
        try:
            grid_image = self._occupancy_grid.get_visualization_image()
            total_saves += 1
            grid_img_file = session_dir / "occupancy_grid.png"
            cv2.imwrite(str(grid_img_file), grid_image)
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 保存占用网格图像失败: {e}")
        
        # 6. 保存元数据
        try:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds()
            
            metadata = {
                "session_info": {
                    "start_time": self._start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "preset": PRESET,
                    "mount": MOUNT,
                    "total_frames": self._total_frames
                },
                "slam_config": _P,
                "grid_config": {
                    "resolution": MAP_RESOLUTION,
                    "size": MAP_SIZE,
                    "center": MAP_CENTER
                },
                "statistics": {
                    "trajectory_poses": len(self._trajectory),
                    "raw_frames_buffered": len(self._raw_cloud_buffer),
                    "current_goal": self._current_goal,
                    "path_length": len(self._current_path) if self._current_path else 0
                }
            }
            
            total_saves += 1
            metadata_file = session_dir / "session_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] 保存元数据失败: {e}")
        
        print(f"[INFO] 数据保存完成: {success_count}/{total_saves} 文件成功")
        print(f"[INFO] 保存位置: {session_dir}")
        
        return success_count > 0
    
    def _save_point_cloud(self, cloud: np.ndarray, file_path: Path) -> bool:
        """保存点云"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            success = o3d.io.write_point_cloud(str(file_path), pcd)
            if success:
                print(f"[INFO] 点云已保存: {file_path.name} ({cloud.shape[0]} 点)")
            return success
        except Exception as e:
            print(f"[ERROR] 保存点云失败: {e}")
            return False
    
    def _save_trajectory(self, poses: List[np.ndarray], file_path: Path) -> bool:
        """保存轨迹"""
        try:
            trajectory_data = []
            for i, pose in enumerate(poses):
                if pose is not None:
                    translation = pose[:3, 3]
                    trajectory_data.append([
                        float(i), translation[0], translation[1], translation[2]
                    ])
            
            np.savetxt(file_path, trajectory_data, fmt="%.6f", delimiter=" ",
                      header="timestamp x y z", comments="# ")
            print(f"[INFO] 轨迹已保存: {file_path.name} ({len(trajectory_data)} 位姿)")
            return True
        except Exception as e:
            print(f"[ERROR] 保存轨迹失败: {e}")
            return False
    
    def shutdown(self):
        """安全关闭系统"""
        print("[INFO] 正在关闭综合SLAM系统...")
        
        try:
            self.save_all_data()
        except Exception as e:
            print(f"[ERROR] 保存数据失败: {e}")
        
        try:
            super().shutdown()
        except Exception as e:
            print(f"[WARNING] 关闭Livox失败: {e}")
        
        try:
            self._viewer.close()
        except Exception as e:
            print(f"[WARNING] 关闭可视化器失败: {e}")
        
        print("[INFO] 综合SLAM系统已关闭")

# ---------------------------------------------------------------------------
# 交互式控制接口
# ---------------------------------------------------------------------------

class SLAMController:
    """
    SLAM系统的交互式控制接口
    """
    
    def __init__(self, slam_system: ComprehensiveSLAM):
        """
        初始化控制器
        
        Args:
            slam_system (ComprehensiveSLAM): SLAM系统实例
        """
        self.slam = slam_system
        self.running = True
        
        # 启动控制线程
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
    
    def _control_loop(self):
        """控制循环"""
        print("\n=== SLAM 控制界面 ===")
        print("命令列表:")
        print("  g <x> <y>  : 设置导航目标到坐标 (x, y)")
        print("  c          : 清除当前目标")
        print("  s          : 保存当前数据")
        print("  i          : 显示系统信息")
        print("  q          : 退出系统")
        print("=====================================\n")
        
        while self.running:
            try:
                command = input("SLAM> ").strip().lower()
                
                if command.startswith('g '):
                    # 设置目标
                    parts = command.split()
                    if len(parts) == 3:
                        try:
                            x, y = float(parts[1]), float(parts[2])
                            self.slam.set_navigation_goal(x, y)
                        except ValueError:
                            print("错误: 无效的坐标格式")
                    else:
                        print("用法: g <x> <y>")
                
                elif command == 'c':
                    # 清除目标
                    self.slam.clear_navigation_goal()
                
                elif command == 's':
                    # 保存数据
                    self.slam.save_all_data()
                
                elif command == 'i':
                    # 显示信息
                    self._show_system_info()
                
                elif command == 'q':
                    # 退出
                    print("正在退出...")
                    self.running = False
                    break
                
                elif command == '':
                    continue
                
                else:
                    print("未知命令，输入 'q' 退出")
            
            except (EOFError, KeyboardInterrupt):
                print("\n正在退出...")
                self.running = False
                break
            except Exception as e:
                print(f"命令处理错误: {e}")
    
    def _show_system_info(self):
        """显示系统信息"""
        print("\n=== 系统信息 ===")
        print(f"总处理帧数: {self.slam._total_frames}")
        print(f"轨迹长度: {len(self.slam._trajectory)}")
        print(f"原始点云缓冲: {len(self.slam._raw_cloud_buffer)} 帧")
        
        if self.slam._current_goal:
            print(f"当前目标: ({self.slam._current_goal[0]:.2f}, {self.slam._current_goal[1]:.2f})")
        else:
            print("当前目标: 无")
        
        if self.slam._current_path:
            print(f"当前路径: {len(self.slam._current_path)} 点")
        else:
            print("当前路径: 无")
        
        # 占用网格统计
        with self.slam._occupancy_grid._lock:
            occupied = (self.slam._occupancy_grid.grid >= _P["obstacle_threshold"]).sum()
            free = (self.slam._occupancy_grid.grid <= _P["free_threshold"]).sum()
            unknown = self.slam._occupancy_grid.grid.size - occupied - free
        
        print(f"占用网格: {occupied} 障碍物, {free} 自由, {unknown} 未知")
        print("==================\n")

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="综合SLAM应用程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, 
                       help="配置文件路径")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR,
                       help="数据输出目录")
    
    args = parser.parse_args()
    
    if args.output_dir:
        global DATA_DIR
        DATA_DIR = args.output_dir
        DATA_DIR.mkdir(exist_ok=True)
    
    print(f"[INFO] 启动综合SLAM系统 (预设: {PRESET}, 挂载: {MOUNT})")
    print(f"[INFO] 数据保存目录: {DATA_DIR.absolute()}")
    print(f"[INFO] 地图参数: 分辨率{MAP_RESOLUTION}m, 尺寸{MAP_SIZE}m")
    
    # 初始化系统
    slam_system = ComprehensiveSLAM()
    controller = SLAMController(slam_system)
    
    # 信号处理
    stop = False
    
    def signal_handler(*_):
        nonlocal stop
        print("\n[INFO] 收到中断信号，正在关闭...")
        stop = True
        controller.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        print("[INFO] 系统已启动，可视化窗口和控制界面已就绪")
        print("[INFO] 在控制台输入命令或按 Ctrl-C 退出")
        
        # 主循环
        while not stop and slam_system._viewer.tick():
            time.sleep(0.01)
    
    finally:
        controller.running = False
        slam_system.shutdown()

if __name__ == "__main__":
    main()