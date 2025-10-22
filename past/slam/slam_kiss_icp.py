#!/usr/bin/env python3
"""
Livox MID-360 雷达点云实时查看器 + KISS-ICP SLAM + 占用网格

本脚本集成了：
1. Livox MID-360 激光雷达实时点云可视化
2. KISS-ICP SLAM 算法进行实时里程计估算
3. 占用网格地图生成和可视化
4. IMU 数据记录和融合

运行前，请确保已正确安装：
- Livox SDK2 和相关 Python 依赖包 (numpy, open3d)
- KISS-ICP: pip install kiss-icp
- 验证 `livox2_python.py` 中导入的 `.so` 文件名称是否正确

效果:
    实时显示雷达点云数据、SLAM轨迹和占用网格，按 'ESC' 键或关闭窗口退出。
    地图数据和轨迹保存为文件到 data/ 目录。

基础流程:
    1. SDK 在后台线程接收 UDP 数据并解析成点云和 IMU 数据
    2. KISS-ICP 处理点云数据进行 SLAM
    3. 生成占用网格地图
    4. 实时可视化显示所有数据

环境变量配置:
- LIVOX_MOUNT: 挂载方向 (normal/upside_down, 默认 upside_down)
"""
from __future__ import annotations

import os
import signal
import time
from typing import Optional, Tuple, List
from pathlib import Path
import csv
import json
import numpy as np
import open3d as o3d
from collections import deque
from threading import Lock

# KISS-ICP 导入
try:
    from kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
except ImportError as e:
    raise ImportError("请安装 KISS-ICP: pip install kiss-icp") from e

# ---------------------------------------------------------------------------
# 配置参数
# ---------------------------------------------------------------------------
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()

if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("环境变量 LIVOX_MOUNT 的值必须是 'normal' 或 'upside_down'")

# 数据保存目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# SLAM 配置
SLAM_CONFIG = {
    "max_range": 30.0,
    "min_range": 0.5,
    "voxel_size": 0.1,
    "adaptive_threshold": True,
    "initial_threshold": 2.0,
    "min_motion_th": 0.1
}

# 占用网格配置
OCCUPANCY_CONFIG = {
    "resolution": 0.1,  # 网格分辨率 (m/cell)
    "map_size": 1000,   # 地图大小 (cells)
    "prob_hit": 0.7,    # 命中概率
    "prob_miss": 0.4,   # 未命中概率
    "prob_occ": 0.8,    # 占用阈值
    "prob_free": 0.2,   # 空闲阈值
    "max_height": 2.0,  # 最大高度
    "min_height": -0.5  # 最小高度
}

# ---------------------------------------------------------------------------
# 占用网格地图类
# ---------------------------------------------------------------------------
class OccupancyGrid:
    """2D 占用网格地图实现"""
    
    def __init__(self, resolution: float = 0.1, map_size: int = 1000):
        self.resolution = resolution
        self.map_size = map_size
        self.origin = np.array([map_size // 2, map_size // 2])
        
        # 使用对数几率表示占用概率
        self.grid = np.zeros((map_size, map_size), dtype=np.float32)
        self.prob_hit = OCCUPANCY_CONFIG["prob_hit"]
        self.prob_miss = OCCUPANCY_CONFIG["prob_miss"]
        self.prob_occ = OCCUPANCY_CONFIG["prob_occ"]
        self.prob_free = OCCUPANCY_CONFIG["prob_free"]
        
        # 转换为对数几率
        self.log_odds_hit = np.log(self.prob_hit / (1 - self.prob_hit))
        self.log_odds_miss = np.log(self.prob_miss / (1 - self.prob_miss))
        
        self.lock = Lock()
    
    def world_to_grid(self, world_coords: np.ndarray) -> np.ndarray:
        """世界坐标转网格坐标"""
        grid_coords = (world_coords / self.resolution + self.origin).astype(int)
        return grid_coords
    
    def grid_to_world(self, grid_coords: np.ndarray) -> np.ndarray:
        """网格坐标转世界坐标"""
        world_coords = (grid_coords - self.origin) * self.resolution
        return world_coords
    
    def is_valid_grid(self, grid_coords: np.ndarray) -> np.ndarray:
        """检查网格坐标是否有效"""
        return ((grid_coords[:, 0] >= 0) & (grid_coords[:, 0] < self.map_size) &
                (grid_coords[:, 1] >= 0) & (grid_coords[:, 1] < self.map_size))
    
    def bresenham_line(self, start: np.ndarray, end: np.ndarray) -> List[np.ndarray]:
        """Bresenham 直线算法生成从起点到终点的网格序列"""
        x0, y0 = start
        x1, y1 = end
        
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            points.append(np.array([x, y]))
            
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
    
    def update_map(self, sensor_pos: np.ndarray, points: np.ndarray):
        """更新占用网格地图"""
        if points.shape[0] == 0:
            return
        
        with self.lock:
            # 传感器位置转网格坐标
            sensor_grid = self.world_to_grid(sensor_pos[:2].reshape(1, -1))[0]
            
            # 点云转网格坐标 (只使用 x, y)
            points_2d = points[:, :2]
            points_grid = self.world_to_grid(points_2d)
            
            # 过滤有效点
            valid_mask = self.is_valid_grid(points_grid)
            valid_points = points_grid[valid_mask]
            
            # 对每个有效点进行射线追踪
            for point_grid in valid_points:
                # 获取从传感器到点的射线上的所有网格
                ray_points = self.bresenham_line(sensor_grid, point_grid)
                
                # 更新射线路径上的网格 (标记为空闲)
                for ray_point in ray_points[:-1]:  # 排除终点
                    if (0 <= ray_point[0] < self.map_size and 
                        0 <= ray_point[1] < self.map_size):
                        self.grid[ray_point[0], ray_point[1]] += self.log_odds_miss
                
                # 更新终点 (标记为占用)
                if (0 <= point_grid[0] < self.map_size and 
                    0 <= point_grid[1] < self.map_size):
                    self.grid[point_grid[0], point_grid[1]] += self.log_odds_hit
    
    def get_occupancy_probability(self) -> np.ndarray:
        """获取占用概率地图"""
        with self.lock:
            prob = 1.0 / (1.0 + np.exp(-self.grid))
            return prob
    
    def save_map(self, filepath: Path):
        """保存地图到文件"""
        with self.lock:
            map_data = {
                "resolution": self.resolution,
                "map_size": self.map_size,
                "origin": self.origin.tolist(),
                "grid": self.grid.tolist()
            }
            with open(filepath, 'w') as f:
                json.dump(map_data, f, indent=2)
            print(f"[INFO] 地图已保存到: {filepath}")

# ---------------------------------------------------------------------------
# 动态导入 SDK 封装
# ---------------------------------------------------------------------------
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except ImportError as _e:
    print(f"[INFO] livox2_python 不可用 ({_e}) – 切换至 SDK1。")
    from livox_python import Livox as _Livox
    _SDK2 = False

# ---------------------------------------------------------------------------
# SLAM 可视化器
# ---------------------------------------------------------------------------
class SLAMViewer:
    """
    集成 SLAM 的点云可视化器
    """
    
    def __init__(self):
        """初始化可视化器"""
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM + Occupancy Grid", width=1600, height=900)
        self._is_alive = True
        
        # 点云数据缓冲
        self._frames: list[np.ndarray] = []
        self._max_frames = 15
        
        # SLAM 相关
        self._trajectory: deque = deque(maxlen=1000)
        self._global_map_points: list[np.ndarray] = []
        
        # 几何体对象
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        
        # 轨迹线条
        self._trajectory_lines = o3d.geometry.LineSet()
        self._vis.add_geometry(self._trajectory_lines)
        
        # 全局地图点云
        self._global_pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._global_pcd)
        
        # 占用网格可视化
        self._occupancy_mesh = o3d.geometry.TriangleMesh()
        self._vis.add_geometry(self._occupancy_mesh)
        
        # 坐标系
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        if MOUNT == "upside_down":
            R180 = np.diag([1.0, -1.0, -1.0, 1.0])
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)
        
        self._first = True
        self._lock = Lock()
    
    def push_frame(self, xyz: np.ndarray):
        """添加新的点云帧"""
        with self._lock:
            self._frames.append(xyz.copy())
            if len(self._frames) > self._max_frames:
                self._frames.pop(0)
    
    def update_slam(self, pose: np.ndarray, local_map: np.ndarray, global_map: Optional[np.ndarray] = None):
        """更新 SLAM 数据"""
        with self._lock:
            # 更新轨迹
            self._trajectory.append(pose[:3, 3].copy())
            
            # 更新全局地图
            if global_map is not None and len(global_map) > 0:
                self._global_map_points.append(global_map.copy())
                if len(self._global_map_points) > 100:  # 限制内存使用
                    self._global_map_points.pop(0)
    
    def update_occupancy_grid(self, occupancy_grid: OccupancyGrid):
        """更新占用网格可视化"""
        prob_map = occupancy_grid.get_occupancy_probability()
        
        # 生成占用网格的 3D 可视化
        occupied_cells = np.where(prob_map > occupancy_grid.prob_occ)
        
        if len(occupied_cells[0]) > 0:
            # 转换为世界坐标
            grid_coords = np.column_stack(occupied_cells)
            world_coords = occupancy_grid.grid_to_world(grid_coords)
            
            # 创建立方体网格
            cubes = []
            for coord in world_coords:
                cube = o3d.geometry.TriangleMesh.create_box(
                    width=occupancy_grid.resolution,
                    height=occupancy_grid.resolution,
                    depth=0.1
                )
                cube.translate([coord[0], coord[1], 0])
                cube.paint_uniform_color([0.8, 0.2, 0.2])  # 红色表示占用
                cubes.append(cube)
            
            if cubes:
                self._occupancy_mesh.clear()
                for cube in cubes[:500]:  # 限制显示数量以保证性能
                    self._occupancy_mesh += cube
    
    def tick(self) -> bool:
        """渲染更新"""
        if not self._is_alive:
            return False
        
        with self._lock:
            # 更新当前点云
            if self._frames:
                merged = np.concatenate(self._frames, axis=0)
                self._pcd.points = o3d.utility.Vector3dVector(merged)
                self._pcd.paint_uniform_color([0.7, 0.7, 0.7])
            
            # 更新轨迹
            if len(self._trajectory) > 1:
                points = np.array(self._trajectory)
                lines = [[i, i + 1] for i in range(len(points) - 1)]
                
                self._trajectory_lines.points = o3d.utility.Vector3dVector(points)
                self._trajectory_lines.lines = o3d.utility.Vector2iVector(lines)
                self._trajectory_lines.paint_uniform_color([0, 1, 0])  # 绿色轨迹
            
            # 更新全局地图
            if self._global_map_points:
                global_points = np.concatenate(self._global_map_points[-10:], axis=0)  # 只显示最近10次的地图
                self._global_pcd.points = o3d.utility.Vector3dVector(global_points)
                self._global_pcd.paint_uniform_color([0.3, 0.3, 1.0])  # 蓝色全局地图
        
        # 更新所有几何体
        self._vis.update_geometry(self._pcd)
        self._vis.update_geometry(self._trajectory_lines)
        self._vis.update_geometry(self._global_pcd)
        self._vis.update_geometry(self._occupancy_mesh)
        
        if self._first:
            self._vis.reset_view_point(True)
            self._first = False
        
        # 处理窗口事件
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        
        if not alive:
            self._is_alive = False
        
        return alive
    
    def close(self):
        """关闭可视化器"""
        self._vis.destroy_window()

# ---------------------------------------------------------------------------
# SLAM 集成的 LiveViewer
# ---------------------------------------------------------------------------
class SLAMLiveViewer(_Livox):
    """集成 KISS-ICP SLAM 的 Livox 查看器"""
    
    def __init__(self):
        """初始化"""
        # 初始化 IMU 数据保存
        self._imu_csv = DATA_DIR / "imu_data.csv"
        self._imu_count = 0
        self._imu_buffer: list[tuple[np.ndarray, int]] = []
        with open(self._imu_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az'])
        
        # SDK 初始化
        if _SDK2:
            super().__init__(
                "mid360_config.json",
                host_ip="192.168.123.164",
                frame_time=0.1,
                frame_packets=60
            )
        else:
            super().__init__()
        
        # SLAM 初始化
        self._slam = KissICP(
            deskew=False,
            max_range=SLAM_CONFIG["max_range"],
            min_range=SLAM_CONFIG["min_range"],
            voxel_size=SLAM_CONFIG["voxel_size"]
        )
        
        # 占用网格地图
        self._occupancy_grid = OccupancyGrid(
            resolution=OCCUPANCY_CONFIG["resolution"],
            map_size=OCCUPANCY_CONFIG["map_size"]
        )
        
        # 可视化器
        self._viewer = SLAMViewer()
        
        # SLAM 状态
        self._current_pose = np.eye(4)
        self._frame_count = 0
        self._trajectory_file = DATA_DIR / "trajectory.txt"
        
        # 轨迹保存
        with open(self._trajectory_file, 'w') as f:
            f.write("# timestamp x y z qx qy qz qw\n")
        
        print("[INFO] SLAM 系统已初始化")
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """处理点云数据"""
        # 坐标转换
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
        
        # 高度过滤 (SLAM 优化)
        height_mask = ((xyz[:, 2] > OCCUPANCY_CONFIG["min_height"]) & 
                       (xyz[:, 2] < OCCUPANCY_CONFIG["max_height"]))
        xyz_filtered = xyz[height_mask]
        
        # 下采样
        if xyz_filtered.shape[0] > 50_000:
            step = xyz_filtered.shape[0] // 50_000
            xyz_filtered = xyz_filtered[::step]
        
        # SLAM 处理
        if xyz_filtered.shape[0] > 100:  # 确保有足够的点进行 SLAM
            try:
                # KISS-ICP 处理
                local_map = self._slam.register_frame(xyz_filtered, timestamp / 1e9)
                self._current_pose = self._slam.poses[-1] if self._slam.poses else np.eye(4)
                
                # 更新占用网格 (使用当前位姿)
                sensor_pos = self._current_pose[:3, 3]
                points_global = (self._current_pose[:3, :3] @ xyz_filtered.T + 
                               self._current_pose[:3, 3:4]).T
                self._occupancy_grid.update_map(sensor_pos, points_global)
                
                # 更新可视化
                global_map = None
                if len(self._slam.local_map) > 0:
                    global_map = np.asarray(self._slam.local_map.points)
                
                self._viewer.update_slam(self._current_pose, local_map, global_map)
                self._viewer.update_occupancy_grid(self._occupancy_grid)
                
                # 保存轨迹
                if self._frame_count % 10 == 0:  # 每10帧保存一次
                    pos = self._current_pose[:3, 3]
                    # 提取旋转矩阵并转换为四元数 (简化版本)
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_matrix(self._current_pose[:3, :3])
                    quat = rot.as_quat()  # [x, y, z, w]
                    
                    with open(self._trajectory_file, 'a') as f:
                        f.write(f"{timestamp/1e9:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                               f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n")
                
                self._frame_count += 1
                
                if self._frame_count % 100 == 0:
                    print(f"[INFO] 已处理 {self._frame_count} 帧，当前位置: "
                         f"[{self._current_pose[0,3]:.2f}, {self._current_pose[1,3]:.2f}, {self._current_pose[2,3]:.2f}]")
                
            except Exception as e:
                print(f"[WARN] SLAM 处理错误: {e}")
        
        # 推送到可视化器
        self._viewer.push_frame(xyz_filtered)
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理 IMU 数据"""
        self._imu_buffer.append((imu_data, timestamp))
        self._imu_count += len(imu_data)
        
        if len(self._imu_buffer) >= 100:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            self._imu_buffer = []
    
    def shutdown(self):
        """关闭系统"""
        print("[INFO] 正在保存 SLAM 数据...")
        
        # 保存剩余 IMU 数据
        if self._imu_buffer:
            with open(self._imu_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for data, ts in self._imu_buffer:
                    for row in data:
                        writer.writerow([ts / 1e9, row[0], row[1], row[2], row[3], row[4], row[5]])
            print(f"[INFO] 已保存 {self._imu_count} 个 IMU 样本")
        
        # 保存地图
        map_file = DATA_DIR / "occupancy_map.json"
        self._occupancy_grid.save_map(map_file)
        
        # 保存最终轨迹和地图点云
        if hasattr(self._slam, 'local_map') and len(self._slam.local_map.points) > 0:
            final_map = np.asarray(self._slam.local_map.points)
            map_pcd_file = DATA_DIR / "final_map.pcd"
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(final_map)
            o3d.io.write_point_cloud(str(map_pcd_file), pcd)
            print(f"[INFO] 最终地图已保存到: {map_pcd_file}")
        
        print(f"[INFO] 轨迹已保存到: {self._trajectory_file}")
        print(f"[INFO] 总共处理 {self._frame_count} 帧")
        
        super().shutdown()
        self._viewer.close()

def main():
    """主函数"""
    print(f"[INFO] 启动 Livox SLAM 系统 (挂载: {MOUNT})")
    print(f"[INFO] 数据保存目录: {DATA_DIR.absolute()}")
    print("[INFO] 按 ESC 或关闭窗口退出")
    
    # 检查依赖
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        print("[WARN] scipy 未安装，轨迹保存功能可能受限")
    
    slam_viewer = SLAMLiveViewer()
    stop = False
    
    def _sigint_handler(*_):
        nonlocal stop
        print("\n[INFO] 收到退出信号，正在关闭...")
        stop = True
    
    signal.signal(signal.SIGINT, _sigint_handler)
    
    try:
        while not stop and slam_viewer._viewer.tick():
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        slam_viewer.shutdown()
        print("[INFO] 系统已关闭")

if __name__ == "__main__":
    main()