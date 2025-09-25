#!/usr/bin/env python3
"""
livox_slam_2d.py - 实时 Livox MID-360 SLAM 2D 可视化程序

功能:
- 从 Livox MID-360 激光雷达获取原始点云数据。
- 使用 KISS-ICP 进行实时 SLAM 处理。
- 生成三种 2D 可视化：
  1. 原始点云（x-y 平面散点图，蓝色点）。
  2. SLAM 2D 投影（绿色点，白色边框）。
  3. 占用网格（白色=占用，黑色=自由，灰色=未知）。
- 在 OpenCV 窗口中水平拼接显示（1440x480），支持 ESC/q 退出。

依赖:
- numpy, opencv-python
- live_slam.py (提供 LiveSLAMDemo 类)
- livox2_python.py (Livox SDK2 封装)
- Livox SDK2 已编译并安装

运行方法:
    python livox_slam_2d.py [--config PATH] [--host-ip IP] [--grid-size FLOAT]

环境变量:
- LIVOX_MOUNT: 'normal' 或 'upside_down' (默认)，调整激光雷达方向。
- LIDAR_TILT_DEG: 倾斜角度（度，默认 0）。
- LIDAR_TILT_AXIS: 倾斜轴 ('x', 'y', 'z'，默认 'y')。

安装:
    pip install numpy opencv-python
    参考 livox2_python.py 安装 Livox SDK2
"""

from __future__ import annotations
import argparse
import sys
import threading
import time
import numpy as np
import cv2
from pathlib import Path

# 动态导入依赖
try:
    from live_slam import LiveSLAMDemo
except ImportError as exc:
    print("[livox_slam_2d] 错误: 'live_slam' 未找到。请确保 live_slam.py 存在且正确配置。")
    sys.exit(1)

try:
    from livox2_python import Livox2
except ImportError as exc:
    print("[livox_slam_2d] 错误: 'livox2_python' 未找到。请确保 livox2_python.py 存在且 Livox SDK2 已安装。")
    sys.exit(1)

# 共享状态
_state_lock = threading.Lock()
_state: dict[str, np.ndarray | None] = {
    "slam": None,        # 2D 投影图像
    "raw_points": None,  # 原始点云
    "grid": None         # 占用网格
}

# 2D 渲染器类
class MiniViewer:
    """将 3D 点云渲染为 2D 俯视图和占用网格的类，替换 live_slam._Viewer。"""
    def __init__(self, grid_size: float = 0.1, grid_threshold: int = 5) -> None:
        self._latest_pts: np.ndarray | None = None
        self._grid_size = grid_size  # 网格分辨率（米）
        self._grid_threshold = grid_threshold  # 占用阈值（点数）

    def push(self, xyz: np.ndarray, _pose: np.ndarray) -> None:
        """存储最新的点云数据（线程安全）。"""
        with _state_lock:
            self._latest_pts = xyz.copy()
            _state["raw_points"] = xyz.copy()  # 存储原始点云

    def tick(self) -> bool:
        """渲染点云到 2D 投影和占用网格，返回 True 表示继续运行。"""
        with _state_lock:
            if self._latest_pts is None:
                return True
            pts = self._latest_pts
            self._latest_pts = None

        if pts.shape[0] == 0:
            return True

        # 1. 生成 2D 投影（绿色点）
        x, y = pts[:, 0], pts[:, 1]
        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span  # 留出边距

        slam_img = np.zeros((480, 480, 3), dtype=np.uint8)
        px = ((x - min_x) * scale + 5).astype(np.int32)
        py = ((y - min_y) * scale + 5).astype(np.int32)
        py = 479 - py  # 翻转使 +y 朝上
        slam_img[py.clip(0, 479), px.clip(0, 479)] = (0, 255, 0)
        cv2.rectangle(slam_img, (0, 0), (479, 479), (255, 255, 255), 1)

        # 2. 生成占用网格
        grid_img = np.full((480, 480), 127, dtype=np.uint8)  # 未知为灰色
        grid_span = max(max_x - min_x, max_y - min_y, 1e-6)
        grid_scale = 480 / grid_span
        grid_px = ((x - min_x) * grid_scale).astype(np.int32)
        grid_py = ((y - min_y) * grid_scale).astype(np.int32)
        grid_py = 479 - grid_py  # 翻转

        # 统计网格占用
        grid_counts = np.zeros((480, 480), dtype=np.int32)
        for px, py in zip(grid_px, grid_py):
            if 0 <= px < 480 and 0 <= py < 480:
                grid_counts[py, px] += 1
        grid_img[grid_counts >= self._grid_threshold] = 255  # 占用
        grid_img[grid_counts == 0] = 0  # 自由
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(grid_img, (0, 0), (479, 479), (255, 255, 255), 1)

        with _state_lock:
            _state["slam"] = slam_img
            _state["grid"] = grid_img

        return True

    def close(self) -> None:
        """清理资源。"""
        pass

# 替换 live_slam 的 Viewer
def monkey_patch_slam_viewer(grid_size: float, grid_threshold: int) -> None:
    """替换 live_slam.LiveSLAMDemo 的 _Viewer 为 MiniViewer。"""
    try:
        import live_slam
        live_slam._Viewer = lambda: MiniViewer(grid_size, grid_threshold)
    except Exception as exc:
        print(f"[livox_slam_2d] SLAM viewer 补丁失败: {exc}", file=sys.stderr)
        sys.exit(1)

# 主 SLAM 处理函数
def run_slam(stop: threading.Event, config_path: str | Path, host_ip: str, grid_size: float, grid_threshold: int) -> None:
    """
    运行 Livox SLAM 并将结果存储到共享状态。

    Args:
        stop: 控制线程停止的事件。
        config_path: Livox SDK 配置 JSON 文件路径。
        host_ip: 主机 IP 地址。
        grid_size: 占用网格分辨率（米）。
        grid_threshold: 占用网格阈值（点数）。
    """
    try:
        monkey_patch_slam_viewer(grid_size, grid_threshold)
        demo = LiveSLAMDemo()

        # 初始化 Livox2 以确保点云数据流入
        lidar = Livox2(config_path, host_ip)

        while not stop.is_set():
            if not demo._viewer.tick():
                break
            time.sleep(0.01)

        demo.shutdown()
        lidar.shutdown()

    except Exception as exc:
        print(f"[livox_slam_2d] SLAM 失败: {exc}", file=sys.stderr)

# 合成并显示画布
def compose_canvas() -> np.ndarray | None:
    """
    合成 OpenCV 显示画布，包含原始点云、2D 投影和占用网格。

    Returns:
        合成的画布图像（480x1440x3）。
    """
    with _state_lock:
        slam = _state.get("slam")
        raw_points = _state.get("raw_points")
        grid = _state.get("grid")

    # 初始化三张图像
    raw_img = np.zeros((480, 480, 3), dtype=np.uint8)
    slam_img = np.full((480, 480, 3), 60, dtype=np.uint8)
    grid_img = np.full((480, 480, 3), 60, dtype=np.uint8)

    # 1. 原始点云（蓝色点）
    if raw_points is not None and raw_points.shape[0] > 0:
        x, y = raw_points[:, 0], raw_points[:, 1]
        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span
        px = ((x - min_x) * scale + 5).astype(np.int32)
        py = ((y - min_y) * scale + 5).astype(np.int32)
        py = 479 - py  # 翻转
        raw_img[py.clip(0, 479), px.clip(0, 479)] = (255, 0, 0)  # 蓝色
        cv2.rectangle(raw_img, (0, 0), (479, 479), (255, 255, 255), 1)
    else:
        cv2.putText(raw_img, "No raw points", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 2. 2D 投影
    if slam is not None:
        slam_img = slam
    else:
        cv2.putText(slam_img, "No SLAM data", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 3. 占用网格
    if grid is not None:
        grid_img = grid
    else:
        cv2.putText(grid_img, "No grid data", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 添加标签
    cv2.putText(raw_img, "Raw Point Cloud", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(slam_img, "2D Projection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(grid_img, "Occupancy Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(raw_img, "ESC/Q to quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 水平拼接
    canvas = np.hstack((raw_img, slam_img, grid_img))
    return canvas

# 主函数
def main() -> None:
    """主函数：初始化 SLAM，显示三种 2D 视图。"""
    parser = argparse.ArgumentParser(
        description="Livox MID-360 实时 SLAM 2D 可视化（原始点云、2D 投影、占用网格）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="mid360_config.json",
                        help="Livox SDK 配置 JSON 文件路径")
    parser.add_argument("--host-ip", type=str, default="192.168.123.222",
                        help="主机 IP 地址")
    parser.add_argument("--grid-size", type=float, default=0.1,
                        help="占用网格分辨率（米）")
    parser.add_argument("--grid-threshold", type=int, default=5,
                        help="占用网格点数阈值")
    args = parser.parse_args()

    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[livox_slam_2d] 错误: 配置文件 {config_path} 不存在")
        sys.exit(1)

    stop = threading.Event()

    # 启动 SLAM 线程
    slam_thread = threading.Thread(
        target=run_slam,
        args=(stop, args.config, args.host_ip, args.grid_size, args.grid_threshold),
        daemon=True
    )
    print("[livox_slam_2d] 启动 SLAM 线程...")
    slam_thread.start()

    # 显示 OpenCV 窗口
    try:
        import cv2
        print("[livox_slam_2d] 系统就绪，显示三种 2D 视图...")
        print("按 ESC 或 Q 退出程序")

        while not stop.is_set():
            canvas = compose_canvas()
            if canvas is None:
                time.sleep(0.05)
                continue

            cv2.imshow("Livox SLAM 2D Views", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[livox_slam_2d] 用户请求退出...")
                stop.set()
                break

        cv2.destroyAllWindows()

    except ImportError as e:
        print(f"[livox_slam_2d] OpenCV 依赖缺失: {e}")
        print("请安装: pip install opencv-python")
        sys.exit(1)
    except Exception as e:
        print(f"[livox_slam_2d] 显示窗口时出错: {e}")
        sys.exit(1)
    finally:
        print("[livox_slam_2d] 正在停止 SLAM 线程...")
        stop.set()
        slam_thread.join(timeout=2.0)
        if slam_thread.is_alive():
            print("[livox_slam_2d] 警告: SLAM 线程未能及时停止")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[livox_slam_2d] 接收到中断信号，正在退出...")
    except Exception as e:
        print(f"[livox_slam_2d] 程序异常退出: {e}")
        sys.exit(1)