#!/usr/bin/env python3
"""
g1_navigation.py - Unitree G-1 机器人自主导航系统

功能:
- 融合 Livox MID-360 雷达和 RealSense 摄像头数据生成占用网格。
- 在占用网格上使用 A* 算法规划路径。
- 将路径转换为速度命令控制 G-1 移动。
- 在 OpenCV 窗口中显示 RGB+深度、SLAM 2D 视图、占用网格和路径。
- 鼠标点击设置目标点，实现自主导航。

依赖:
- numpy, opencv-python, pyrealsense2, pynput
- live_slam.py, livox2_python.py, hanger_boot_sequence.py
- Livox SDK2

运行方法:
    python g1_navigation.py [--iface IFACE] [--config PATH] [--host-ip IP]

控制:
- 鼠标左键点击 SLAM 视图设置目标。
- ESC/Q: 退出。
"""

from __future__ import annotations
import argparse
import sys
import threading
import time
import numpy as np
import cv2
from pathlib import Path
from collections import deque
from typing import Any, Dict, Optional, Tuple

# 导入自定义模块
try:
    from live_slam import LiveSLAMDemo
    from livox2_python import Livox2
    from hanger_boot_sequence import hanger_boot_sequence
except ImportError as exc:
    print(f"[g1_navigation] 依赖缺失: {exc}")
    sys.exit(1)

from pynput.keyboard import Listener, Key, KeyCode

# 共享状态
_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "rgbd": None,          # RGB + 深度 (1280x480)
    "slam": None,          # 2D 投影 (480x480)
    "grid": None,          # 占用网格 (480x480)
    "raw_points": None,    # 原始点云
    "vel": (0.0, 0.0, 0.0),# 当前速度
    "pose": np.eye(4),     # 机器人位姿 (4x4)
    "path": [],            # 规划路径 (list of (x,y))
    "goal": None,          # 目标像素 (x,y)
}

# 全局参数
GRID_SIZE = 0.1         # 网格分辨率 (m)
GRID_THRESHOLD = 5      # 占用阈值 (点数)
NAV_SPEED = 0.3         # 最大前进速度 (m/s)
TURN_SPEED = 0.5        # 最大转向速度 (rad/s)
SAFE_DIST = 0.2         # 安全距离 (m)

# RealSense 接收器 (本地)
def rx_realsense_local(stop: threading.Event) -> None:
    try:
        import pyrealsense2 as rs
        WIDTH, HEIGHT, FPS = 640, 480, 30

        def colourise_depth(depth_frame):
            depth_data = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.convertScaleAbs(depth_data, alpha=0.03)
            depth_image_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            return cv2.applyColorMap(depth_image_bgr, cv2.COLORMAP_JET)

        ctx = rs.context()
        pipeline = rs.pipeline(ctx)
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

        spatial_filter = rs.spatial_filter()
        temporal_filter = rs.temporal_filter()
        align = rs.align(rs.stream.color)

        pipeline.start(config)

        last = time.perf_counter()
        while not stop.is_set():
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()

            if not depth or not color:
                time.sleep(0.01)
                continue

            depth = spatial_filter.process(depth)
            depth = temporal_filter.process(depth)

            color_img = np.asanyarray(color.get_data())
            depth_colored = colourise_depth(depth)

            combo = cv2.hconcat([color_img, depth_colored])
            fps = 1.0 / (time.perf_counter() - last)
            last = time.perf_counter()
            cv2.putText(combo, f"RGB+Depth {fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            with _state_lock:
                _state["rgbd"] = combo

        pipeline.stop()

    except Exception as exc:
        print(f"[g1_navigation] RealSense 失败: {exc}", file=sys.stderr)

# SLAM 2D 渲染器
class MiniViewer:
    def __init__(self) -> None:
        self._latest_pts: np.ndarray | None = None

    def push(self, xyz: np.ndarray, pose: np.ndarray) -> None:
        with _state_lock:
            self._latest_pts = xyz.copy()
            _state["raw_points"] = xyz.copy()
            _state["pose"] = pose.copy()

    def tick(self) -> bool:
        with _state_lock:
            if self._latest_pts is None:
                return True
            pts = self._latest_pts
            self._latest_pts = None

        if pts.shape[0] == 0:
            return True

        # 生成 2D 投影
        x, y = pts[:, 0], pts[:, 1]
        min_x, max_x = float(x.min()), float(x.max())
        min_y, max_y = float(y.min()), float(y.max())
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span

        slam_img = np.zeros((480, 480, 3), dtype=np.uint8)
        px = ((x - min_x) * scale + 5).astype(np.int32)
        py = ((y - min_y) * scale + 5).astype(np.int32)
        py = 479 - py
        slam_img[py.clip(0, 479), px.clip(0, 479)] = (0, 255, 0)
        cv2.rectangle(slam_img, (0, 0), (479, 479), (255, 255, 255), 1)

        # 生成占用网格
        grid_img = np.full((480, 480), 127, dtype=np.uint8)
        grid_scale = 480 / span
        grid_px = ((x - min_x) * grid_scale).astype(np.int32)
        grid_py = ((y - min_y) * grid_scale).astype(np.int32)
        grid_py = 479 - grid_py

        grid_counts = np.zeros((480, 480), dtype=np.int32)
        for px, py in zip(grid_px, grid_py):
            if 0 <= px < 480 and 0 <= py < 480:
                grid_counts[py, px] += 1
        grid_img[grid_counts >= GRID_THRESHOLD] = 255
        grid_img[grid_counts == 0] = 0
        grid_img = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(grid_img, (0, 0), (479, 479), (255, 255, 255), 1)

        with _state_lock:
            _state["slam"] = slam_img
            _state["grid"] = grid_img

        return True

# 替换 Viewer
def monkey_patch_slam_viewer() -> None:
    import live_slam
    live_slam._Viewer = MiniViewer

# SLAM 线程
def run_slam(stop: threading.Event, config_path: str, host_ip: str) -> None:
    try:
        monkey_patch_slam_viewer()
        demo = LiveSLAMDemo()
        lidar = Livox2(config_path, host_ip)

        while not stop.is_set():
            if not demo._viewer.tick():
                break
            time.sleep(0.01)

        demo.shutdown()
        lidar.shutdown()

    except Exception as exc:
        print(f"[g1_navigation] SLAM 失败: {exc}", file=sys.stderr)

# 键盘遥控 & 导航控制线程
def control_thread(stop: threading.Event, iface: str):
    try:
        bot = hanger_boot_sequence(iface=iface)

        vx = vy = omega = 0.0
        LIN_STEP, ANG_STEP = 0.05, 0.2
        SEND_PERIOD = 0.05  # 20Hz

        pressed = set()
        def on_press(k):
            if isinstance(k, KeyCode) and k.char:
                pressed.add(k.char.lower())
            else:
                pressed.add(k)

        def on_release(k):
            if isinstance(k, KeyCode) and k.char:
                pressed.discard(k.char.lower())
            else:
                pressed.discard(k)

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        def is_pressed(name):
            if name == "space": return Key.space in pressed
            if name == "esc": return Key.esc in pressed
            return name in pressed

        last_send = 0.0
        while not stop.is_set():
            # 手动模式覆盖
            if is_pressed("w") and not is_pressed("s"):
                vx = min(vx + LIN_STEP, NAV_SPEED)
            elif is_pressed("s") and not is_pressed("w"):
                vx = max(vx - LIN_STEP, -NAV_SPEED)
            else:
                vx = 0.0

            if is_pressed("q") and not is_pressed("e"):
                vy = min(vy + LIN_STEP, NAV_SPEED)
            elif is_pressed("e") and not is_pressed("q"):
                vy = max(vy - LIN_STEP, -NAV_SPEED)
            else:
                vy = 0.0

            if is_pressed("a") and not is_pressed("d"):
                omega = min(omega + ANG_STEP, TURN_SPEED)
            elif is_pressed("d") and not is_pressed("a"):
                omega = max(omega - ANG_STEP, -TURN_SPEED)
            else:
                omega = 0.0

            if is_pressed("space"):
                vx = vy = omega = 0.0

            if is_pressed("z"):
                bot.Damp()
                break
            if is_pressed("esc"):
                bot.StopMove()
                bot.ZeroTorque()
                break

            # 自主导航模式（如果有路径，覆盖手动）
            with _state_lock:
                path = _state["path"]
                pose = _state["pose"]
                grid = _state["grid"]

            if path and len(path) > 1:
                # 下一个 waypoint
                next_wp = path[1]  # (y, x) in grid
                robot_px = (int(pose[0, 3] / GRID_SIZE + 240), int(pose[1, 3] / GRID_SIZE + 240))  # 假设中心原点
                dx = (next_wp[1] - robot_px[1]) * GRID_SIZE
                dy = (next_wp[0] - robot_px[0]) * GRID_SIZE
                dist = np.sqrt(dx**2 + dy**2)
                angle = np.arctan2(dy, dx) - np.arctan2(pose[1, 0], pose[0, 0])  # yaw error

                if dist < SAFE_DIST:
                    path.pop(0)  # 到达，移到下一个
                else:
                    vx = NAV_SPEED * np.cos(angle)
                    omega = TURN_SPEED * angle

                # 安全检查：前方占用
                front_area = grid[robot_px[0]-10:robot_px[0], robot_px[1]:robot_px[1]+20]  # 示例前方区域
                if np.mean(front_area) > 100:  # 占用高
                    vx = 0.0
                    omega = TURN_SPEED  # 转弯避障

                with _state_lock:
                    _state["path"] = path
                    _state["vel"] = (vx, vy, omega)

            now = time.time()
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now

                with _state_lock:
                    _state["vel"] = (vx, vy, omega)

            time.sleep(0.005)

    except Exception as exc:
        print(f"[g1_navigation] 控制失败: {exc}", file=sys.stderr)

# A* 路径规划
def a_star_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> list:
    rows, cols = grid.shape
    queue = deque([start])
    came_from = {start: None}
    g_score = {start: 0}

    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
            neighbor = (current[0] + dy, current[1] + dx)  # 注意坐标系
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor[0], neighbor[1]] < 128:  # 自由
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    queue.append(neighbor)
    return []  # 无路径

# 鼠标回调设置目标
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 假设点击在SLAM视图 (480~960)
        if 480 <= x < 960:
            grid_x = x - 480
            grid_y = y
            with _state_lock:
                _state["goal"] = (grid_y, grid_x)  # (row, col)

# 合成画布
def compose_canvas() -> np.ndarray | None:
    with _state_lock:
        rgbd = _state.get("rgbd")
        slam = _state.get("slam")
        grid = _state.get("grid")
        vx, vy, om = _state.get("vel", (0.0, 0.0, 0.0))
        path = _state["path"]
        pose = _state["pose"]
        goal = _state["goal"]

    # 占位
    if rgbd is None:
        rgbd = np.full((480, 1280, 3), 80, dtype=np.uint8)
        cv2.putText(rgbd, "No RealSense", (380, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if slam is None:
        slam = np.full((480, 480, 3), 60, dtype=np.uint8)
        cv2.putText(slam, "No SLAM", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if grid is None:
        grid = np.full((480, 480, 3), 60, dtype=np.uint8)
        cv2.putText(grid, "No Grid", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # 规划路径
    robot_pos = (240, 240)  # 假设中心 (简化，实际从pose计算)
    if goal and grid is not None:
        path = a_star_path(grid[:, :, 0], robot_pos, goal)  # 灰度通道
        with _state_lock:
            _state["path"] = path

    # 绘制路径到SLAM和网格
    if path:
        path_np = np.array(path, dtype=np.int32)
        cv2.polylines(slam, [path_np], False, (0, 0, 255), 2)
        cv2.polylines(grid, [path_np], False, (0, 0, 255), 2)

    # 合成
    top = rgbd
    bottom = cv2.hconcat([slam, grid])
    canvas = np.vstack([top, bottom])

    # HUD
    txt = f"vx {vx:+.2f} vy {vy:+.2f} omega {om:+.2f} - Click to set goal - ESC: stop"
    cv2.rectangle(canvas, (0, canvas.shape[0]-40), (canvas.shape[1], canvas.shape[0]), (0,0,0), -1)
    cv2.putText(canvas, txt, (10, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return canvas

# 主函数
def main() -> None:
    parser = argparse.ArgumentParser(description="G1 机器人自主导航")
    parser.add_argument("--iface", default="eth0", help="网络接口")
    parser.add_argument("--config", default="mid360_config.json", help="Livox 配置")
    parser.add_argument("--host-ip", default="192.168.123.222", help="主机 IP")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件 {args.config} 不存在")
        sys.exit(1)

    stop = threading.Event()

    # 启动线程
    realsense_t = threading.Thread(target=rx_realsense_local, args=(stop,), daemon=True)
    slam_t = threading.Thread(target=run_slam, args=(stop, args.config, args.host_ip), daemon=True)
    control_t = threading.Thread(target=control_thread, args=(stop, args.iface), daemon=True)

    realsense_t.start()
    slam_t.start()
    control_t.start()

    # OpenCV 窗口
    cv2.namedWindow("G1 Navigation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("G1 Navigation", mouse_callback)

    try:
        while not stop.is_set():
            canvas = compose_canvas()
            if canvas is None:
                time.sleep(0.05)
                continue

            cv2.imshow("G1 Navigation", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop.set()
                break

        cv2.destroyAllWindows()

    finally:
        stop.set()
        realsense_t.join(timeout=2.0)
        slam_t.join(timeout=2.0)
        control_t.join(timeout=2.0)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断退出")
    except Exception as e:
        print(f"异常: {e}")
        sys.exit(1)