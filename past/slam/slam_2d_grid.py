#!/usr/bin/env python3
"""
slam_2d_grid_simplified.py - 精简版 Livox MID-360 占用网格生成器 + PGM保存功能

主要功能:
- 实时生成2D占用网格
- 自动保存PGM栅格地图文件
- 支持里程计数据融合
- 简化的可视化界面
"""

import argparse
import sys
import threading
import time
import os
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy import ndimage
from PIL import Image

# Unitree SDK 导入
UNITREE_SDK_AVAILABLE = False
try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    UNITREE_SDK_AVAILABLE = True
    print("[INFO] Unitree SDK 已加载")
except ImportError:
    print("[WARNING] Unitree SDK 不可用，将无法使用里程计数据")

# 挂载方向
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

# 动态导入 Livox SDK
_SDK2 = False
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
    print("[INFO] 使用 Livox SDK2")
except ImportError:
    try:
        from livox_python import Livox as _Livox
        _SDK2 = False
        print("[INFO] 使用 Livox SDK1")
    except ImportError as exc:
        print(f"[ERROR] Livox SDK 未找到: {exc}")
        sys.exit(1)

# 全局状态
_state_lock = threading.Lock()
_latest_occupancy_grid = None
_grid_updated = False
_robot_pose = np.eye(4)

@dataclass
class GridConfig:
    """网格配置参数"""
    grid_size: float = 20.0			# 网格大小（米）
    grid_resolution: int = 400		# 网格分辨率（像素）
    hit_threshold: float = 0.65		# 占用概率阈值
    free_threshold: float = 0.35	# 自由概率阈值
    prob_hit: float = 0.7			# 命中概率更新
    prob_miss: float = 0.4			# 未命中概率更新
    max_range: float = 10.0			# 最大有效距离（米）
    min_height: float = -0.5		# 最小高度（米）
    max_height: float = 2.5			# 最大高度（米）
    decay_factor: float = 0.998		# 概率衰减因子
    use_odometry: bool = True		# 是否使用里程计数据
    save_interval: int = 100		# PGM保存间隔（帧数）

class OdometrySubscriber:
    """里程计数据订阅器"""
    
    def __init__(self, interface: str = "eth0"):
        self.position = np.zeros(3)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.lock = threading.Lock()
        
        if not UNITREE_SDK_AVAILABLE:
            return
        
        try:
            ChannelFactoryInitialize(0, interface)
            self.odom_subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
            self.odom_subscriber.Init(self._odom_handler, 10)
            print(f"[INFO] 里程计订阅器已初始化，接口: {interface}")
        except Exception as e:
            print(f"[ERROR] 初始化里程计订阅器失败: {e}")
    
    def _odom_handler(self, msg: SportModeState_):
        """里程计数据处理回调"""
        try:
            with self.lock:
                self.position = np.array(msg.position[:3])
                
                if hasattr(msg, 'imu_state') and hasattr(msg.imu_state, 'quaternion'):
                    quat = msg.imu_state.quaternion
                    self.orientation = np.array([quat[3], quat[0], quat[1], quat[2]])
                
                # 更新全局机器人位姿
                global _robot_pose
                with _state_lock:
                    _robot_pose = self._compute_transform_matrix()
        
        except Exception as e:
            print(f"[ERROR] 里程计数据处理失败: {e}")
    
    def _compute_transform_matrix(self) -> np.ndarray:
        """计算机器人变换矩阵"""
        w, x, y, z = self.orientation
        
        # 四元数到旋转矩阵
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.position
        return T

class PGMSaver:
    """PGM文件保存器"""
    
    def __init__(self, output_dir: str = "maps", name_prefix: str = "slam_map"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        print(f"[PGMSaver] 保存目录: {self.output_dir}")
    
    def save_occupancy_grid(self, grid: np.ndarray, config: GridConfig, 
                           world_origin: np.ndarray, frame_count: int = 0) -> str:
        """
        保存占用网格为PGM格式
        
        Args:
            grid: 占用网格 (0=自由, 128=未知, 255=占用)
            config: 网格配置
            world_origin: 世界坐标系原点
            frame_count: 帧计数
        
        Returns:
            保存的文件路径
        """
        timestamp = int(time.time())
        filename = f"{self.name_prefix}_{timestamp}_{frame_count:06d}"
        
        # 保存PGM文件
        pgm_path = self.output_dir / f"{filename}.pgm"
        
        # 转换网格格式 (PGM: 0=占用, 254=自由, 205=未知)
        pgm_grid = np.zeros_like(grid)
        pgm_grid[grid == 0] = 254		# 自由空间
        pgm_grid[grid == 128] = 205		# 未知区域
        pgm_grid[grid == 255] = 0		# 占用区域
        
        # 保存PGM图像
        image = Image.fromarray(pgm_grid, mode='L')
        image.save(pgm_path)
        
        # 生成YAML配置文件
        yaml_path = self.output_dir / f"{filename}.yaml"
        cell_size = config.grid_size / config.grid_resolution
        
        with open(yaml_path, 'w') as f:
            f.write(f"image: {filename}.pgm\n")
            f.write(f"resolution: {cell_size:.6f}\n")
            f.write(f"origin: [{world_origin[0]:.6f}, {world_origin[1]:.6f}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")
        
        print(f"[PGMSaver] 已保存地图: {pgm_path}")
        return str(pgm_path)
    
    def save_latest_map(self, grid: np.ndarray, config: GridConfig, world_origin: np.ndarray):
        """保存最新的地图文件"""
        latest_pgm = self.output_dir / "latest_map.pgm"
        latest_yaml = self.output_dir / "latest_map.yaml"
        
        # 转换网格格式
        pgm_grid = np.zeros_like(grid)
        pgm_grid[grid == 0] = 254		# 自由空间
        pgm_grid[grid == 128] = 205		# 未知区域
        pgm_grid[grid == 255] = 0		# 占用区域
        
        # 保存PGM图像
        image = Image.fromarray(pgm_grid, mode='L')
        image.save(latest_pgm)
        
        # 生成YAML配置文件
        cell_size = config.grid_size / config.grid_resolution
        
        with open(latest_yaml, 'w') as f:
            f.write(f"image: latest_map.pgm\n")
            f.write(f"resolution: {cell_size:.6f}\n")
            f.write(f"origin: [{world_origin[0]:.6f}, {world_origin[1]:.6f}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")

class OccupancyGridGenerator(_Livox):
    """精简版占用网格生成器"""
    
    def __init__(self, config_path: str = "mid360_config.json", 
                 host_ip: str = "192.168.123.164",
                 config: GridConfig = None,
                 interface: str = "eth0",
                 output_dir: str = "maps"):
        
        # 初始化 Livox SDK
        if _SDK2:
            super().__init__(config_path, host_ip=host_ip, frame_time=0.1, frame_packets=60)
        else:
            super().__init__()
        
        # 网格配置
        self.config = config or GridConfig()
        self.cell_size = self.config.grid_size / self.config.grid_resolution
        
        # 初始化里程计订阅器
        self.odometry_subscriber = None
        if self.config.use_odometry:
            self.odometry_subscriber = OdometrySubscriber(interface)
        
        # 概率占用网格
        self.log_odds = np.zeros((self.config.grid_resolution, self.config.grid_resolution), dtype=np.float32)
        self.occupancy_prob = np.full((self.config.grid_resolution, self.config.grid_resolution), 0.5, dtype=np.float32)
        
        # 离散化占用网格
        self.occupancy_grid = np.full((self.config.grid_resolution, self.config.grid_resolution), 128, dtype=np.uint8)
        
        # 多帧融合缓存
        self.frame_history = deque(maxlen=5)
        
        # 世界坐标系原点
        self.world_origin = np.array([0.0, 0.0])
        self.origin_set = False
        
        # PGM保存器
        self.pgm_saver = PGMSaver(output_dir)
        
        # 统计信息
        self.frame_count = 0
        self.total_points = 0
        self.valid_points = 0
        
        # 预计算概率转换
        self.log_prob_hit = np.log(self.config.prob_hit / (1 - self.config.prob_hit))
        self.log_prob_miss = np.log(self.config.prob_miss / (1 - self.config.prob_miss))
        
        print(f"[OccupancyGrid] 网格大小: {self.config.grid_size}m x {self.config.grid_size}m")
        print(f"[OccupancyGrid] 分辨率: {self.config.grid_resolution}x{self.config.grid_resolution}")
        print(f"[OccupancyGrid] 单元格大小: {self.cell_size:.3f}m")
        print(f"[OccupancyGrid] 里程计数据: {'启用' if self.config.use_odometry else '禁用'}")
        print(f"[OccupancyGrid] PGM保存间隔: {self.config.save_interval} 帧")

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """处理点云数据，生成占用网格"""
        if len(xyz) == 0:
            return
        
        self.total_points += len(xyz)
        
        # 获取当前机器人位姿
        robot_pose = self._get_current_robot_pose()
        
        # 设置世界坐标系原点
        if not self.origin_set:
            self.world_origin = robot_pose[:3, 3][:2].copy()
            self.origin_set = True
            print(f"[OccupancyGrid] 设置世界坐标系原点: ({self.world_origin[0]:.2f}, {self.world_origin[1]:.2f})")
        
        # 应用挂载方向校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0])
        
        # 坐标变换：激光雷达坐标系 -> 世界坐标系
        xyz_world = self._transform_points_to_world(xyz, robot_pose)
        
        # 点云过滤
        valid_mask = self._filter_points(xyz_world, robot_pose[:3, 3])
        xyz_filtered = xyz_world[valid_mask]
        
        if len(xyz_filtered) == 0:
            return
        
        self.valid_points += len(xyz_filtered)
        
        # 更新概率网格
        sensor_position = robot_pose[:3, 3]
        self._update_probability_grid(xyz_filtered, sensor_position)
        
        # 多帧融合
        self._apply_temporal_filtering()
        
        # 转换为离散网格
        self._update_discrete_grid()
        
        # 更新全局状态
        self._update_global_state()
        
        # 保存PGM文件
        self._save_pgm_if_needed()
        
        self.frame_count += 1
        if self.frame_count % 50 == 0:
            self._print_statistics()

    def _get_current_robot_pose(self) -> np.ndarray:
        """获取当前机器人位姿"""
        if self.config.use_odometry and self.odometry_subscriber:
            global _robot_pose
            with _state_lock:
                return _robot_pose.copy()
        else:
            return np.eye(4)

    def _transform_points_to_world(self, xyz: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
        """将激光雷达坐标系下的点云转换到世界坐标系"""
        if len(xyz) == 0:
            return xyz
        
        xyz_homogeneous = np.hstack([xyz, np.ones((len(xyz), 1))])
        xyz_world_homogeneous = (robot_pose @ xyz_homogeneous.T).T
        return xyz_world_homogeneous[:, :3]

    def _filter_points(self, xyz: np.ndarray, sensor_position: np.ndarray) -> np.ndarray:
        """点云过滤"""
        # 距离过滤
        distances = np.linalg.norm(xyz[:, :2] - sensor_position[:2], axis=1)
        distance_mask = (distances > 0.5) & (distances < self.config.max_range)
        
        # 高度过滤
        height_mask = (xyz[:, 2] > self.config.min_height) & (xyz[:, 2] < self.config.max_height)
        
        # 网格边界过滤
        grid_coords = self._world_to_grid_coord(xyz)
        boundary_mask = (
            (grid_coords[:, 0] >= 5) & (grid_coords[:, 0] < self.config.grid_resolution - 5) &
            (grid_coords[:, 1] >= 5) & (grid_coords[:, 1] < self.config.grid_resolution - 5)
        )
        
        return distance_mask & height_mask & boundary_mask

    def _update_probability_grid(self, xyz: np.ndarray, sensor_position: np.ndarray):
        """更新概率网格"""
        # 传感器位置转网格坐标
        sensor_grid = self._world_to_grid_coord(sensor_position.reshape(1, -1))[0]
        
        # 时间衰减
        self.log_odds *= self.config.decay_factor
        
        # 转换点云到网格坐标
        grid_coords = self._world_to_grid_coord(xyz)
        
        # 标记占用点
        for point in grid_coords:
            if self._is_valid_grid_point(point):
                self.log_odds[point[1], point[0]] += self.log_prob_hit
        
        # 光线追踪标记自由空间
        sample_size = min(len(grid_coords), 50)
        if sample_size > 0:
            sampled_indices = np.random.choice(len(grid_coords), sample_size, replace=False)
            for idx in sampled_indices:
                point = grid_coords[idx]
                if self._is_valid_grid_point(point):
                    line_points = self._bresenham_line(sensor_grid[0], sensor_grid[1], point[0], point[1])
                    for lx, ly in line_points[::2]:
                        if self._is_valid_grid_point((lx, ly)) and (lx != point[0] or ly != point[1]):
                            self.log_odds[ly, lx] += self.log_prob_miss
        
        # 限制对数几率范围
        self.log_odds = np.clip(self.log_odds, -5, 5)
        
        # 转换为概率
        self.occupancy_prob = 1.0 / (1.0 + np.exp(-self.log_odds))

    def _world_to_grid_coord(self, xyz: np.ndarray) -> np.ndarray:
        """世界坐标转网格坐标"""
        relative_coords = xyz[:, :2] - self.world_origin
        center = self.config.grid_resolution // 2
        grid_x = (relative_coords[:, 0] / self.cell_size + center).astype(np.int32)
        grid_y = (-relative_coords[:, 1] / self.cell_size + center).astype(np.int32)
        return np.column_stack([grid_x, grid_y])

    def _is_valid_grid_point(self, point: Tuple[int, int]) -> bool:
        """检查网格点是否有效"""
        return (0 <= point[0] < self.config.grid_resolution and 
                0 <= point[1] < self.config.grid_resolution)

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int):
        """Bresenham 直线算法"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def _apply_temporal_filtering(self):
        """应用时间滤波"""
        current_frame = self.occupancy_prob.copy()
        self.frame_history.append(current_frame)
        
        if len(self.frame_history) > 1:
            weights = np.exp(np.linspace(-0.5, 0, len(self.frame_history)))
            weights /= weights.sum()
            
            fused_prob = np.zeros_like(self.occupancy_prob)
            for i, frame in enumerate(self.frame_history):
                fused_prob += weights[i] * frame
            
            self.occupancy_prob = fused_prob

    def _update_discrete_grid(self):
        """更新离散占用网格"""
        smoothed_prob = ndimage.gaussian_filter(self.occupancy_prob, sigma=0.5)
        
        self.occupancy_grid = np.full_like(self.occupancy_grid, 128, dtype=np.uint8)
        self.occupancy_grid[smoothed_prob >= self.config.hit_threshold] = 255
        self.occupancy_grid[smoothed_prob <= self.config.free_threshold] = 0

    def _update_global_state(self):
        """更新全局状态"""
        global _latest_occupancy_grid, _grid_updated
        
        with _state_lock:
            _latest_occupancy_grid = self.occupancy_grid.copy()
            _grid_updated = True

    def _save_pgm_if_needed(self):
        """根据保存间隔保存PGM文件"""
        if self.frame_count % self.config.save_interval == 0:
            # 保存带时间戳的地图
            self.pgm_saver.save_occupancy_grid(
                self.occupancy_grid, 
                self.config, 
                self.world_origin, 
                self.frame_count
            )
        
        # 始终更新最新地图
        self.pgm_saver.save_latest_map(self.occupancy_grid, self.config, self.world_origin)

    def _print_statistics(self):
        """打印统计信息"""
        valid_ratio = self.valid_points / max(self.total_points, 1) * 100
        occupied_cells = np.sum(self.occupancy_grid == 255)
        free_cells = np.sum(self.occupancy_grid == 0)
        unknown_cells = np.sum(self.occupancy_grid == 128)
        
        print(f"[OccupancyGrid] 帧: {self.frame_count}, "
              f"有效点: {valid_ratio:.1f}%, "
              f"占用: {occupied_cells}, "
              f"自由: {free_cells}, "
              f"未知: {unknown_cells}")

    def get_robot_pose_in_grid(self) -> Tuple[int, int]:
        """获取机器人在网格中的位置"""
        if not self.origin_set:
            return (self.config.grid_resolution // 2, self.config.grid_resolution // 2)
        
        robot_pose = self._get_current_robot_pose()
        robot_world_pos = robot_pose[:3, 3][:2]
        grid_coords = self._world_to_grid_coord(robot_world_pos.reshape(1, -1))[0]
        return (grid_coords[0], grid_coords[1])

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理 IMU 数据"""
        pass

def render_occupancy_grid(grid: np.ndarray | None, config: GridConfig) -> np.ndarray:
    """渲染占用网格"""
    if grid is None:
        canvas = np.full((400, 400, 3), 60, dtype=np.uint8)
        cv2.putText(canvas, "No Grid Data", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return canvas
    
    # 创建彩色画布
    canvas = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    canvas[grid == 0] = [0, 255, 0]		# 自由空间 = 绿色
    canvas[grid == 128] = [64, 64, 64]	# 未知 = 深灰色
    canvas[grid == 255] = [0, 0, 255]	# 占用 = 红色
    
    return canvas

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="精简版 Livox MID-360 占用网格生成器 + PGM保存",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="mid360_config.json",
                        help="Livox SDK 配置文件路径")
    parser.add_argument("--host-ip", type=str, default="192.168.123.164",
                        help="主机 IP 地址")
    parser.add_argument("--interface", type=str, default="eth0",
                        help="DDS 通信网络接口")
    parser.add_argument("--grid-size", type=float, default=20.0,
                        help="网格实际尺寸（米）")
    parser.add_argument("--grid-resolution", type=int, default=400,
                        help="网格分辨率（像素）")
    parser.add_argument("--output-dir", type=str, default="maps",
                        help="PGM文件输出目录")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="PGM保存间隔（帧数）")
    parser.add_argument("--no-odometry", action="store_true",
                        help="禁用里程计数据")
    parser.add_argument("--headless", action="store_true",
                        help="无GUI模式")
    args = parser.parse_args()
    
    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 创建网格配置
    grid_config = GridConfig(
        grid_size=args.grid_size,
        grid_resolution=args.grid_resolution,
        use_odometry=not args.no_odometry and UNITREE_SDK_AVAILABLE,
        save_interval=args.save_interval
    )
    
    print(f"[INFO] 使用配置文件: {config_path}")
    print(f"[INFO] 主机 IP: {args.host_ip}")
    print(f"[INFO] DDS 接口: {args.interface}")
    print(f"[INFO] 网格尺寸: {grid_config.grid_size}m x {grid_config.grid_size}m")
    print(f"[INFO] 网格分辨率: {grid_config.grid_resolution}x{grid_config.grid_resolution}")
    print(f"[INFO] 输出目录: {args.output_dir}")
    print(f"[INFO] 保存间隔: {args.save_interval} 帧")
    print(f"[INFO] 里程计数据: {'启用' if grid_config.use_odometry else '禁用'}")
    print(f"[INFO] 无头模式: {'启用' if args.headless else '禁用'}")
    
    # 初始化占用网格生成器
    try:
        grid_generator = OccupancyGridGenerator(
            config_path=str(config_path),
            host_ip=args.host_ip,
            config=grid_config,
            interface=args.interface,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 主循环
    try:
        if args.headless:
            print("[INFO] 无头模式运行，按 Ctrl+C 停止...")
            while True:
                time.sleep(1)
        else:
            print("[INFO] 开始显示占用网格，按 ESC/Q 退出...")
            window_name = "Livox Occupancy Grid - PGM Saver"
            cv2.namedWindow(window_name)
            
            while True:
                global _latest_occupancy_grid, _grid_updated
                
                with _state_lock:
                    grid = _latest_occupancy_grid
                    updated = _grid_updated
                    _grid_updated = False
                
                # 渲染网格
                canvas = render_occupancy_grid(grid, grid_config)
                
                # 添加机器人位置
                if grid is not None:
                    robot_pos = grid_generator.get_robot_pose_in_grid()
                    cv2.circle(canvas, robot_pos, 5, (255, 255, 255), -1)
                    cv2.circle(canvas, robot_pos, 7, (0, 0, 0), 2)
                
                # 添加状态信息
                cv2.putText(canvas, f"Frames: {grid_generator.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(canvas, f"Maps Saved: {grid_generator.frame_count // args.save_interval}", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 添加图例
                cv2.putText(canvas, "Green: Free", (10, canvas.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(canvas, "Red: Occupied", (10, canvas.shape[0] - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(canvas, "Gray: Unknown", (10, canvas.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
                
                cv2.imshow(window_name, canvas)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[INFO] 接收到中断信号")
    except Exception as e:
        print(f"[ERROR] 运行时错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] 正在关闭...")
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] 程序已退出")

if __name__ == "__main__":
    main()