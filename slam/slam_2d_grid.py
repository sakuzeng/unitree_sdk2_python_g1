#!/usr/bin/env python3
"""
livox_slam_2d_grid.py - 实时 Livox MID-360 占用网格 2D 可视化

功能:
- 从 Livox MID-360 激光雷达获取点云数据
- 生成实时 2D 占用网格用于路径规划
- 支持 ESC/q 退出

依赖:
- numpy, opencv-python
- livox2_python 或 livox_python (SDK 封装)

运行方法:
    python livox_slam_2d_grid.py [--config PATH] [--host-ip IP] [--grid-size FLOAT] [--grid-resolution INT]

环境变量:
- LIVOX_MOUNT: 'normal' 或 'upside_down' (默认)
"""

from __future__ import annotations
import argparse
import sys
import threading
import time
import os
import numpy as np
import cv2
from pathlib import Path

# 挂载方向
MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in {"normal", "upside_down"}:
    raise SystemExit("环境变量 LIVOX_MOUNT 的值必须是 'normal' 或 'upside_down'")

# 动态导入 SDK
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

# 全局状态锁和数据
_state_lock = threading.Lock()
_latest_occupancy_grid = None
_grid_updated = False

class OccupancyGridGenerator(_Livox):
    """
    占用网格生成器，继承自 Livox SDK
    """
    def __init__(self, config_path: str = "mid360_config.json", 
                 host_ip: str = "192.168.123.164",
                 grid_size: float = 20.0,  # 网格大小（米）
                 grid_resolution: int = 400):  # 网格分辨率（像素）
        
        # 初始化 Livox SDK
        if _SDK2:
            super().__init__(config_path, host_ip=host_ip, frame_time=0.1, frame_packets=60)
        else:
            super().__init__()
        
        # 网格参数
        self.grid_size = grid_size  # 实际尺寸（米）
        self.grid_resolution = grid_resolution  # 像素分辨率
        self.cell_size = grid_size / grid_resolution  # 每个像素代表的实际距离
        
        # 占用网格（0=自由，128=未知，255=占用）
        self.occupancy_grid = np.full((grid_resolution, grid_resolution), 128, dtype=np.uint8)
        self.hit_counts = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
        self.miss_counts = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
        
        # 统计信息
        self.frame_count = 0
        
        print(f"[OccupancyGrid] 网格大小: {grid_size}m x {grid_size}m")
        print(f"[OccupancyGrid] 分辨率: {grid_resolution}x{grid_resolution}")
        print(f"[OccupancyGrid] 单元格大小: {self.cell_size:.3f}m")

    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据，生成占用网格
        """
        if len(xyz) == 0:
            return
        
        # 应用挂载方向校正
        if MOUNT == "upside_down":
            xyz = xyz * np.array([1.0, -1.0, -1.0])
        
        # 过滤有效点（高度和距离）
        valid_mask = (
            (xyz[:, 2] > -2.0) & (xyz[:, 2] < 2.0) &  # 高度过滤
            (np.linalg.norm(xyz[:, :2], axis=1) < self.grid_size / 2)  # 距离过滤
        )
        xyz_filtered = xyz[valid_mask]
        
        if len(xyz_filtered) == 0:
            return
        
        # 转换到网格坐标
        center = self.grid_resolution // 2
        grid_x = (xyz_filtered[:, 0] / self.cell_size + center).astype(np.int32)
        grid_y = (-xyz_filtered[:, 1] / self.cell_size + center).astype(np.int32)  # Y轴翻转
        
        # 清空计数器（简化处理）
        self.hit_counts.fill(0)
        self.miss_counts.fill(0)
        
        # 标记占用点
        for gx, gy in zip(grid_x, grid_y):
            if 0 <= gx < self.grid_resolution and 0 <= gy < self.grid_resolution:
                self.hit_counts[gy, gx] += 1
        
        # 简单光线追踪标记自由空间
        robot_gx, robot_gy = center, center
        for gx, gy in zip(grid_x, grid_y):
            if 0 <= gx < self.grid_resolution and 0 <= gy < self.grid_resolution:
                # Bresenham 线段算法标记从机器人到观测点的自由空间
                line_points = self._bresenham_line(robot_gx, robot_gy, gx, gy)
                for lx, ly in line_points[:-1]:  # 排除终点（占用点）
                    if 0 <= lx < self.grid_resolution and 0 <= ly < self.grid_resolution:
                        self.miss_counts[ly, lx] += 1
        
        # 更新占用网格
        self._update_occupancy_grid()
        
        # 更新全局状态
        global _latest_occupancy_grid, _grid_updated
        with _state_lock:
            _latest_occupancy_grid = self.occupancy_grid.copy()
            _grid_updated = True
        
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            print(f"[OccupancyGrid] 处理第 {self.frame_count} 帧，有效点数: {len(xyz_filtered)}")

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """
        Bresenham 直线算法，返回直线上的所有点
        """
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

    def _update_occupancy_grid(self):
        """
        根据命中和未命中计数更新占用网格
        """
        # 占用阈值
        hit_threshold = 3
        miss_threshold = 2
        
        # 更新占用状态
        occupied_mask = self.hit_counts >= hit_threshold
        free_mask = (self.hit_counts == 0) & (self.miss_counts >= miss_threshold)
        
        self.occupancy_grid[occupied_mask] = 255  # 占用
        self.occupancy_grid[free_mask] = 0        # 自由
        # 其余保持未知（128）

    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理 IMU 数据（暂不使用）
        """
        pass

def render_occupancy_grid(grid: np.ndarray | None, grid_size: float) -> np.ndarray:
    """
    渲染占用网格为彩色图像
    """
    if grid is None:
        canvas = np.full((400, 400, 3), 60, dtype=np.uint8)
        cv2.putText(canvas, "No Grid Data", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return canvas
    
    # 转换为彩色图像
    canvas = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    canvas[grid == 0] = [0, 255, 0]      # 自由空间 = 绿色
    canvas[grid == 128] = [128, 128, 128]  # 未知 = 灰色
    canvas[grid == 255] = [0, 0, 255]    # 占用 = 红色
    
    # 绘制中心点（机器人位置）
    center = grid.shape[0] // 2
    cv2.circle(canvas, (center, center), 3, (255, 255, 255), -1)
    cv2.circle(canvas, (center, center), 5, (0, 0, 0), 2)
    
    # 绘制网格线
    step = grid.shape[0] // 8
    for i in range(0, grid.shape[0], step):
        cv2.line(canvas, (i, 0), (i, grid.shape[0]-1), (64, 64, 64), 1)
        cv2.line(canvas, (0, i), (grid.shape[1]-1, i), (64, 64, 64), 1)
    
    # 绘制边框
    cv2.rectangle(canvas, (0, 0), (grid.shape[1]-1, grid.shape[0]-1), (255, 255, 255), 2)
    
    return canvas

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="Livox MID-360 实时占用网格 2D 可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="mid360_config.json",
                        help="Livox SDK 配置文件路径")
    parser.add_argument("--host-ip", type=str, default="192.168.123.164",
                        help="主机 IP 地址")
    parser.add_argument("--grid-size", type=float, default=20.0,
                        help="网格实际尺寸（米）")
    parser.add_argument("--grid-resolution", type=int, default=400,
                        help="网格分辨率（像素）")
    args = parser.parse_args()
    
    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        print("请确保配置文件存在，或使用 --config 指定正确路径")
        sys.exit(1)
    
    print(f"[INFO] 使用配置文件: {config_path}")
    print(f"[INFO] 主机 IP: {args.host_ip}")
    print(f"[INFO] 网格尺寸: {args.grid_size}m x {args.grid_size}m")
    print(f"[INFO] 网格分辨率: {args.grid_resolution}x{args.grid_resolution}")
    print("[INFO] 按 ESC 或 Q 退出")
    
    # 初始化占用网格生成器
    try:
        grid_generator = OccupancyGridGenerator(
            config_path=str(config_path),
            host_ip=args.host_ip,
            grid_size=args.grid_size,
            grid_resolution=args.grid_resolution
        )
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        sys.exit(1)
    
    # 主显示循环
    try:
        print("[INFO] 开始显示占用网格...")
        last_update_time = time.time()
        
        while True:
            current_time = time.time()
            
            # 获取最新的占用网格
            global _latest_occupancy_grid, _grid_updated
            with _state_lock:
                grid = _latest_occupancy_grid
                updated = _grid_updated
                _grid_updated = False
            
            # 渲染网格
            canvas = render_occupancy_grid(grid, args.grid_size)
            
            # 添加状态信息
            status_text = f"Frames: {grid_generator.frame_count}"
            cv2.putText(canvas, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if updated:
                fps = 1.0 / (current_time - last_update_time) if current_time > last_update_time else 0
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(canvas, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                last_update_time = current_time
            
            legend_y = canvas.shape[0] - 60
            cv2.putText(canvas, "Green: Free", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, "Red: Occupied", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(canvas, "Gray: Unknown", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            cv2.putText(canvas, "ESC/Q to quit", (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示图像
            cv2.imshow("Livox Occupancy Grid", canvas)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):  # ESC 或 Q
                break
            
            time.sleep(0.01)  # 限制刷新率
    
    except KeyboardInterrupt:
        print("\n[INFO] 接收到中断信号")
    except Exception as e:
        print(f"[ERROR] 运行时错误: {e}")
    finally:
        print("[INFO] 正在关闭...")
        try:
            grid_generator.shutdown()
        except:
            pass
        cv2.destroyAllWindows()
        print("[INFO] 程序已退出")

if __name__ == "__main__":
    main()