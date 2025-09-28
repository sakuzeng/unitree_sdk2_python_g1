#!/usr/bin/env python3
"""
livox_slam_2d_projection.py - 实时 Livox MID-360 SLAM 2D 投影可视化

功能:
- 从 Livox MID-360 激光雷达获取点云数据。
- 使用 KISS-ICP 进行 SLAM 处理，渲染 2D 投影（绿色点，480x480）。
- 支持 ESC/q 退出。
- IMU 数据由 live_slam.py 保存到 CSV。

依赖:
- numpy, opencv-python
- live_slam.py (提供 LiveSLAMDemo 类)
- livox2_python.py (Livox SDK2 封装)
- Livox SDK2 已编译并安装

运行方法:
    python livox_slam_2d_projection.py [--config PATH] [--host-ip IP]

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
    print("[livox_slam_2d_projection] 错误: 'live_slam' 未找到。请确保 live_slam.py 存在且正确配置。")
    sys.exit(1)

try:
    from livox2_python import Livox2
except ImportError as exc:
    print("[livox_slam_2d_projection] 错误: 'livox2_python' 未找到。请确保 livox2_python.py 存在且 Livox SDK2 已安装。")
    sys.exit(1)

# 数据保存目录
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# 共享状态
_state_lock = threading.Lock()
_state: dict[str, np.ndarray | None] = {
    "slam": None  # 2D 投影图像
}

# 2D 渲染器类
class MiniViewer:
    """将 3D 点云渲染为 2D 投影的类，替换 live_slam._Viewer。"""
    def __init__(self) -> None:
        self._latest_pts: np.ndarray | None = None

    def push(self, xyz: np.ndarray, pose: np.ndarray) -> None:
        """存储最新的点云数据并渲染 2D 投影（线程安全）。"""
        with _state_lock:
            self._latest_pts = xyz.copy()

    def tick(self) -> bool:
        """渲染点云到 2D 投影，返回 True 表示继续运行。"""
        with _state_lock:
            if self._latest_pts is None:
                return True
            pts = self._latest_pts
            self._latest_pts = None

        if pts.shape[0] == 0:
            return True

        # 生成 2D 投影（绿色点）
        x, y = pts[:, 0], pts[:, 1]
        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span
        slam_img = np.zeros((480, 480, 3), dtype=np.uint8)
        px = ((x - min_x) * scale + 5).astype(np.int32)
        py = ((y - min_y) * scale + 5).astype(np.int32)
        py = 479 - py  # 翻转使 +y 朝上
        slam_img[py.clip(0, 479), px.clip(0, 479)] = (0, 255, 0)
        cv2.rectangle(slam_img, (0, 0), (479, 479), (255, 255, 255), 1)

        with _state_lock:
            _state["slam"] = slam_img

        return True

    def close(self) -> None:
        """清理资源。"""
        pass

# 替换 live_slam 的 Viewer
def monkey_patch_slam_viewer() -> None:
    """替换 live_slam.LiveSLAMDemo 的 _Viewer 为 MiniViewer。"""
    try:
        import live_slam
        live_slam._Viewer = MiniViewer
    except Exception as exc:
        print(f"[livox_slam_2d_projection] SLAM viewer 补丁失败: {exc}", file=sys.stderr)
        sys.exit(1)

# 渲染 SLAM 投影
def render_slam_projection(slam: np.ndarray | None) -> np.ndarray:
    """
    渲染 SLAM 2D 投影（绿色点）。

    Args:
        slam (np.ndarray | None): SLAM 投影图像（480x480x3）。

    Returns:
        np.ndarray: 渲染的图像（480x480x3）。
    """
    slam_img = np.full((480, 480, 3), 60, dtype=np.uint8)
    if slam is not None:
        slam_img = slam
    else:
        cv2.putText(slam_img, "No SLAM data", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(slam_img, "2D Projection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(slam_img, "ESC/Q to quit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return slam_img

# 主 SLAM 处理函数
def run_slam(stop: threading.Event, config_path: str | Path, host_ip: str) -> None:
    """
    运行 Livox SLAM 并将结果存储到共享状态。

    Args:
        stop: 控制线程停止的事件。
        config_path: Livox SDK 配置 JSON 文件路径。
        host_ip: 主机 IP 地址。
    """
    try:
        monkey_patch_slam_viewer()
        demo = LiveSLAMDemo()
        lidar = Livox2(config_path, host_ip, frame_time=0.1, frame_packets=60)

        while not stop.is_set():
            if not demo._viewer.tick():
                break
            time.sleep(0.01)

        demo.shutdown()
        lidar.shutdown()

    except Exception as exc:
        print(f"[livox_slam_2d_projection] SLAM 失败: {exc}", file=sys.stderr)

# 主函数
def main() -> None:
    """主函数：初始化 SLAM，显示 2D 投影。"""
    parser = argparse.ArgumentParser(
        description="Livox MID-360 实时 SLAM 2D 投影可视化",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=str, default="mid360_config.json",
                        help="Livox SDK 配置 JSON 文件路径")
    parser.add_argument("--host-ip", type=str, default="192.168.123.164",
                        help="主机 IP 地址")
    args = parser.parse_args()

    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[livox_slam_2d_projection] 错误: 配置文件 {config_path} 不存在")
        sys.exit(1)

    stop = threading.Event()

    # 启动 SLAM 线程
    slam_thread = threading.Thread(
        target=run_slam,
        args=(stop, args.config, args.host_ip),
        daemon=True
    )
    print(f"[livox_slam_2d_projection] 启动 SLAM 线程 (主机 IP: {args.host_ip})...")
    print(f"[livox_slam_2d_projection] IMU 数据将保存到: {DATA_DIR / 'slam_session_<timestamp>/imu_data.csv'}")
    slam_thread.start()

    # 显示 OpenCV 窗口
    try:
        import cv2
        print("[livox_slam_2d_projection] 系统就绪，显示 SLAM 2D 投影...")
        print("按 ESC 或 Q 退出程序")

        while not stop.is_set():
            with _state_lock:
                slam = _state.get("slam")
            canvas = render_slam_projection(slam)
            cv2.imshow("Livox SLAM 2D Projection", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[livox_slam_2d_projection] 用户请求退出...")
                stop.set()
                break

        cv2.destroyAllWindows()

    except ImportError as e:
        print(f"[livox_slam_2d_projection] OpenCV 依赖缺失: {e}")
        print("请安装: pip install opencv-python")
        sys.exit(1)
    except Exception as e:
        print(f"[livox_slam_2d_projection] 显示窗口时出错: {e}")
        sys.exit(1)
    finally:
        print("[livox_slam_2d_projection] 正在停止 SLAM 线程...")
        stop.set()
        slam_thread.join(timeout=2.0)
        if slam_thread.is_alive():
            print("[livox_slam_2d_projection] 警告: SLAM 线程未能及时停止")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[livox_slam_2d_projection] 接收到中断信号，正在退出...")
    except Exception as e:
        print(f"[livox_slam_2d_projection] 程序异常退出: {e}")
        sys.exit(1)