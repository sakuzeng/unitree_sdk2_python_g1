#!/usr/bin/env python3
"""run_g1_stack.py – 一站式脚本，启动以下功能模块:

1. Unitree G-1 机器人键盘遥控 (keyboard_controller.py)
2. RealSense 视频流接收器 (receive_realsense_gst.py – RGB & 彩色深度)
3. Livox MID-360 激光雷达实时 SLAM (live_slam.py)

并将上述模块的**视觉输出**合并到一个 OpenCV 窗口中，布局如下::

    ┌──────────────────────────── RGB (640×480) ───────────────────────────┐
    │                                                                     │
    ├────────────────────────── Depth (640×480) ───────────────────────────┤
    │                                                                     │
    └─────────────────────── 2-D SLAM preview (480×480) ───────────────────┘

功能概述:
- **线程管理**: 各子系统运行在后台线程中，最新的 NumPy 图像存储在共享字典中。
- **SLAM 可视化**: 替换 `live_slam._Viewer`，将 3D 地图渲染为 2D 俯视图。
- **键盘遥控**: 从 `keyboard_controller.py` 中复制并精简，支持实时速度显示。

运行方法:
    python run_g1_stack.py [--iface IFACE]

参数说明:
- `--iface`: 指定与 Unitree G-1 连接的网络接口，默认为 `eth0`。

注意事项:
- 如果缺少某些依赖，脚本会打印警告并用灰色背景替代对应的输出。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import subprocess
import grp
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# 共享状态管理
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "rgbd": None,        # RealSense 的 RGB + 深度图像 (1280×480)
    "slam": None,        # SLAM 的 2D 俯视图 (480×480)
    "vel": (0.0, 0.0, 0.0),  # 当前速度 (vx, vy, omega)
}

# ---------------------------------------------------------------------------
# 1. RealSense 接收器 (支持本地设备)
# ---------------------------------------------------------------------------

def _check_device_availability() -> bool:
    """
    检查 RealSense 设备是否被其他进程占用。

    Returns:
        bool: 如果设备可用，返回 True；否则返回 False。
    """
    try:
        result = subprocess.run(['lsof', '/dev/video*'], capture_output=True, text=True)
        if result.stdout:
            print("[run_g1_stack] 警告: 检测到摄像头设备被占用:")
            print(result.stdout)
            return False
        return True
    except FileNotFoundError:
        # lsof 命令不存在，跳过检查
        return True
    except Exception:
        return True  # 如果检查失败，假设设备可用


def _check_video_permissions() -> bool:
    """
    检查当前用户是否有访问视频设备的权限。
    
    Returns:
        bool: 如果有权限返回 True，否则返回 False。
    """
    try:
        # 检查用户是否在 video 组中
        video_gid = grp.getgrnam('video').gr_gid
        user_groups = os.getgroups()
        
        if video_gid in user_groups:
            return True
        else:
            print("[run_g1_stack] 警告: 当前用户不在 video 组中")
            print("建议运行: sudo usermod -a -G video $USER")
            print("然后重新登录或重启系统")
            return False
            
    except KeyError:
        # video 组不存在
        return True
    except Exception:
        # 权限检查失败，假设有权限
        return True


def _reset_usb_devices() -> None:
    """
    重置 USB 摄像头设备（用户级操作）。
    
    注意：某些操作可能需要管理员权限。如果遇到权限问题，
    请考虑将用户添加到 video 组或使用 udev 规则。
    """
    try:
        print("[run_g1_stack] 正在尝试重置 USB 摄像头设备...")
        
        # 尝试通过 RealSense API 重置设备
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            for device in devices:
                try:
                    device.hardware_reset()
                    print(f"[run_g1_stack] 已重置设备: {device.get_info(rs.camera_info.name)}")
                    time.sleep(2)
                except Exception as e:
                    print(f"[run_g1_stack] 重置设备失败: {e}")
            
            print("[run_g1_stack] 设备重置尝试完成")
            
        except ImportError:
            print("[run_g1_stack] pyrealsense2 未安装，跳过设备重置")
        
    except Exception as e:
        print(f"[run_g1_stack] 重置 USB 设备失败: {e}")
        print("提示: 如果经常遇到设备占用问题，请考虑：")
        print("1. 将用户添加到 video 组: sudo usermod -a -G video $USER")
        print("2. 重启系统以应用组权限变更")
        print("3. 检查其他可能占用摄像头的程序")


def _get_first_device(context) -> Optional[Any]:
    """
    返回第一个 RealSense 设备，如果没有设备则返回 None。

    Args:
        context: RealSense 上下文。

    Returns:
        Optional[Any]: 第一个 RealSense 设备。
    """
    devices = context.query_devices()
    if len(devices) == 0:
        return None
    return devices[0]


def _rx_realsense_local(stop: threading.Event) -> None:
    """
    直接从本地 RealSense 设备捕获 RGB 和深度图像。
    参考自 stream_realsense.py，包含重试机制和设备检查。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    try:
        import pyrealsense2 as rs
        import numpy as np
        import cv2

        WIDTH, HEIGHT, FPS = 640, 480, 30

        def colourise_depth(depth_frame: rs.depth_frame) -> cv2.Mat:
            """将深度帧 (16-bit) 转换为伪彩色 8-bit BGR 图像。"""
            depth_data = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.convertScaleAbs(depth_data, alpha=0.03)
            depth_image_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            return cv2.applyColorMap(depth_image_bgr, cv2.COLORMAP_JET)

        # --- 权限检查 ---
        _check_video_permissions()

        # --- 设备检查和重试机制 ---
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[run_g1_stack] 尝试启动 RealSense (第 {attempt + 1}/{max_retries} 次)...")
                
                if not _check_device_availability():
                    print("[run_g1_stack] 设备被占用，尝试重置...")
                    _reset_usb_devices()

                # --- 检查设备连接 ---
                ctx = rs.context()
                device = _get_first_device(ctx)

                if device is None:
                    raise RuntimeError("未找到 RealSense 设备")

                print(f"[run_g1_stack] 找到设备: {device.get_info(rs.camera_info.name)}")
                print(f"[run_g1_stack] 序列号: {device.get_info(rs.camera_info.serial_number)}")
                print(f"[run_g1_stack] 固件版本: {device.get_info(rs.camera_info.firmware_version)}")

                # --- 初始化 RealSense ---
                pipeline = rs.pipeline(ctx)
                config = rs.config()
                config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
                config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

                # 应用深度后处理滤波器
                spatial_filter = rs.spatial_filter()  # 边缘保持平滑
                temporal_filter = rs.temporal_filter()  # 时间降噪

                align_to = rs.stream.color
                align = rs.align(align_to)

                print("[run_g1_stack] 正在启动 RealSense 管道...")
                profile = pipeline.start(config)
                print("[run_g1_stack] RealSense 管道已启动。")

                # 获取相机内参
                colour_intr = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = colour_intr.get_intrinsics()
                print(f"[run_g1_stack] 相机内参: {intr.width}×{intr.height}, fx={intr.fx:.1f}, fy={intr.fy:.1f}")

                break  # 成功初始化，跳出重试循环

            except RuntimeError as e:
                if "Device or resource busy" in str(e) or "xioctl" in str(e):
                    print(f"[run_g1_stack] 设备忙碌错误: {e}")
                    if attempt < max_retries - 1:
                        print("[run_g1_stack] 等待并重试...")
                        time.sleep(2)
                        _reset_usb_devices()
                    else:
                        print("[run_g1_stack] 所有重试都失败了")
                        print("\n故障排除建议:")
                        print("1. 确保没有其他程序在使用摄像头")
                        print("2. 检查 USB 连接是否稳定")
                        print("3. 尝试重新插拔摄像头")
                        print("4. 重启系统以清理设备状态")
                        return
                else:
                    print(f"[run_g1_stack] RealSense 初始化失败: {e}")
                    return

        # --- 主循环 ---
        last = time.perf_counter()

        while not stop.is_set():
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
                color_frame: rs.video_frame = aligned_frames.get_color_frame()

                if not depth_frame or not color_frame:
                    time.sleep(0.01)
                    continue

                # 应用后处理滤波器
                depth_frame = spatial_filter.process(depth_frame)
                depth_frame = temporal_filter.process(depth_frame)

                # --- 转换图像 ---
                color_image = np.asanyarray(color_frame.get_data())
                depth_colored = colourise_depth(depth_frame)

                # --- 合成并更新状态 ---
                combo = cv2.hconcat([color_image, depth_colored])

                fps = 1.0 / (time.perf_counter() - last)
                last = time.perf_counter()
                cv2.putText(combo, f"RGB+Depth  {fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                with _state_lock:
                    _state["rgbd"] = combo

            except Exception as e:
                print(f"[run_g1_stack] 帧处理错误: {e}")
                time.sleep(0.01)
                continue

        # --- 清理 ---
        print("[run_g1_stack] 正在停止 RealSense 管道...")
        pipeline.stop()

    except ImportError:
        print("[run_g1_stack] RealSense 接收器已禁用: 'pyrealsense2' 未安装。", file=sys.stderr)
    except Exception as exc:
        print(f"[run_g1_stack] 本地 RealSense 接收器失败: {exc}", file=sys.stderr)


def _rx_realsense_opencv(stop: threading.Event) -> None:
    """
    使用 OpenCV 接收 RealSense 的 RGB 和深度图像数据。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    try:
        import numpy as np
        import cv2
        import socket
        import struct
        from collections import defaultdict

        class VideoReceiver:
            def __init__(self, rgb_port: int = 5600, depth_port: int = 5602):
                self.rgb_port = rgb_port
                self.depth_port = depth_port
                
                # 创建UDP套接字
                self.rgb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                # 绑定端口
                self.rgb_socket.bind(('', rgb_port))
                self.depth_socket.bind(('', depth_port))
                
                # 设置接收缓冲区和超时
                self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
                self.rgb_socket.settimeout(0.1)
                self.depth_socket.settimeout(0.1)
                
                # 帧缓存
                self.rgb_packets = defaultdict(dict)
                self.depth_packets = defaultdict(dict)
                
                self.latest_rgb = None
                self.latest_depth = None
                self.running = True

            def receive_frame(self, socket_obj: socket.socket, packet_dict: dict) -> Optional[np.ndarray]:
                """接收并重组一个完整帧"""
                try:
                    data, addr = socket_obj.recvfrom(65536)
                    
                    if len(data) < 12:  # 最小包头大小
                        return None
                    
                    # 解析包头
                    packet_id, total_packets, data_len = struct.unpack('!III', data[:12])
                    packet_data = data[12:12+data_len]
                    
                    # 存储包数据
                    frame_id = int(time.time() * 1000) // 100  # 简单的帧ID
                    if frame_id not in packet_dict:
                        packet_dict[frame_id] = {}
                    
                    packet_dict[frame_id][packet_id] = packet_data
                    
                    # 检查是否收到完整帧
                    if len(packet_dict[frame_id]) == total_packets:
                        # 重组帧
                        frame_data = b''.join(packet_dict[frame_id][i] 
                                            for i in range(total_packets))
                        
                        # 解码图像
                        nparr = np.frombuffer(frame_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # 清理旧数据
                        del packet_dict[frame_id]
                        
                        # 清理过期帧
                        current_time = int(time.time() * 1000) // 100
                        expired_frames = [fid for fid in packet_dict.keys() 
                                        if current_time - fid > 10]
                        for fid in expired_frames:
                            del packet_dict[fid]
                        
                        return frame
                    
                except socket.timeout:
                    pass  # 正常超时，继续循环
                except Exception as e:
                    if self.running:
                        print(f"接收包错误: {e}")
                
                return None

            def close(self):
                """关闭接收器"""
                self.running = False
                self.rgb_socket.close()
                self.depth_socket.close()

        receiver = VideoReceiver()
        last = time.perf_counter()

        while not stop.is_set():
            # 尝试接收RGB和深度帧
            rgb_frame = receiver.receive_frame(receiver.rgb_socket, receiver.rgb_packets)
            depth_frame = receiver.receive_frame(receiver.depth_socket, receiver.depth_packets)

            if rgb_frame is not None:
                receiver.latest_rgb = rgb_frame
            if depth_frame is not None:
                receiver.latest_depth = depth_frame

            # 如果两个帧都可用，合成显示
            if receiver.latest_rgb is not None and receiver.latest_depth is not None:
                combo = cv2.hconcat([receiver.latest_rgb, receiver.latest_depth])

                fps = 1.0 / (time.perf_counter() - last)
                last = time.perf_counter()
                cv2.putText(combo, f"RGB+Depth  {fps:5.1f} FPS", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                with _state_lock:
                    _state["rgbd"] = combo

            time.sleep(0.01)

        receiver.close()

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_stack] OpenCV RealSense receiver disabled:", exc, file=sys.stderr)


def _rx_realsense(stop: threading.Event) -> None:
    """
    选择 RealSense 接收器的实现 (优先使用本地设备)。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    print("[run_g1_stack] 使用本地 RealSense 接收器")
    _rx_realsense_local(stop)

# ---------------------------------------------------------------------------
# 2. Livox SLAM (2D 俯视图渲染)
# ---------------------------------------------------------------------------

def _monkey_patch_slam_viewer() -> None:
    """
    替换 `live_slam._Viewer`，将 3D 地图渲染为 2D 俯视图。

    此实现保留了 `push()` 和 `tick()` 的接口，但将渲染结果输出为 NumPy 数组，
    以便在 OpenCV 窗口中显示。
    """

    try:
        import numpy as np  # type: ignore
        import cv2  # type: ignore

        import live_slam as _ls  # type: ignore

        class _MiniViewer:  # pylint: disable=too-few-public-methods
            def __init__(self) -> None:
                self._latest_pts: Optional[np.ndarray] = None
                self._img: Optional[np.ndarray] = None

            # ----------------------------------------------
            def push(self, xyz: np.ndarray, _pose: np.ndarray):
                # Save copy – callback comes from background thread
                self._latest_pts = xyz

            # ----------------------------------------------
            def tick(self) -> bool:  # noqa: D401  – same signature as original
                if self._latest_pts is None:
                    return True  # alive

                pts = self._latest_pts
                self._latest_pts = None

                if pts.shape[0] == 0:
                    return True

                # ----------------  very small & very fast scatter -> canvas
                x, y = pts[:, 0], pts[:, 1]
                min_x, max_x = float(x.min()), float(x.max())
                min_y, max_y = float(y.min()), float(y.max())

                span = max(max_x - min_x, max_y - min_y, 1e-6)
                scale = 470.0 / span  # leave small margin

                img = np.zeros((480, 480, 3), dtype=np.uint8)

                # Map x/y → pixel
                px = ((x - min_x) * scale + 5).astype(np.int32)
                py = ((y - min_y) * scale + 5).astype(np.int32)
                py = 479 - py  # flip so +y is up in the image

                img[py.clip(0, 479), px.clip(0, 479)] = (0, 255, 0)

                # Simple bounding box
                cv2.rectangle(img, (0, 0), (479, 479), (255, 255, 255), 1)

                self._img = img

                with _state_lock:
                    _state["slam"] = self._img

                return True  # keep running

            # ----------------------------------------------
            def close(self):  # kept for compatibility with LiveSLAMDemo.shutdown
                pass

        # Monkey-patch 👍
        _ls._Viewer = _MiniViewer  # type: ignore[attr-defined]

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_stack] SLAM viewer patch failed:", exc, file=sys.stderr)


def _run_slam(stop: threading.Event) -> None:
    """
    运行 Livox SLAM，并将 2D 渲染结果存储到共享状态中。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    try:
        _monkey_patch_slam_viewer()

        import live_slam as _ls  # type: ignore  # now uses patched viewer

        demo = _ls.LiveSLAMDemo()  # type: ignore[attr-defined]

        while not stop.is_set():
            # Tick just to let the patched viewer process latest cloud
            if not demo._viewer.tick():  # type: ignore[attr-defined]
                break
            time.sleep(0.01)

        demo.shutdown()

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_stack] SLAM disabled:", exc, file=sys.stderr)


# ---------------------------------------------------------------------------
# 3. Unitree G-1 键盘遥控
# ---------------------------------------------------------------------------

def _keyboard_thread(stop: threading.Event, iface: str):
    """
    运行 Unitree G-1 的键盘遥控模块。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
        iface (str): 与 Unitree G-1 连接的网络接口。
    """
    try:
        from hanger_boot_sequence import hanger_boot_sequence  # type: ignore
        from pynput.keyboard import Listener, Key, KeyCode  # type: ignore

        bot = hanger_boot_sequence(iface=iface)

        vx = vy = omega = 0.0
        LIN_STEP, ANG_STEP = 0.05, 0.2
        SEND_PERIOD = 0.1

        def _clamp(value: float, limit: float = 0.6) -> float:
            return max(-limit, min(limit, value))

        pressed: set[Any] = set()

        def on_press(k):  # noqa: D401 – callback
            if isinstance(k, KeyCode) and k.char is not None:
                pressed.add(k.char.lower())
            else:
                pressed.add(k)

        def on_release(k):
            if isinstance(k, KeyCode) and k.char is not None:
                pressed.discard(k.char.lower())
            else:
                pressed.discard(k)

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()

        last_send = 0.0

        def _is(name: str) -> bool:
            if name == "space":
                return Key.space in pressed
            if name == "esc":
                return Key.esc in pressed
            return name in pressed

        while not stop.is_set():
            if _is("w") and not _is("s"):
                vx = _clamp(vx + LIN_STEP)
            elif _is("s") and not _is("w"):
                vx = _clamp(vx - LIN_STEP)
            else:
                vx = 0.0

            if _is("q") and not _is("e"):
                vy = _clamp(vy + LIN_STEP)
            elif _is("e") and not _is("q"):
                vy = _clamp(vy - LIN_STEP)
            else:
                vy = 0.0

            if _is("a") and not _is("d"):
                omega = _clamp(omega + ANG_STEP)
            elif _is("d") and not _is("a"):
                omega = _clamp(omega - ANG_STEP)
            else:
                omega = 0.0

            if _is("space"):
                vx = vy = omega = 0.0

            # Exit keys --------------------------------------------------
            if _is("z"):
                bot.Damp()
                break
            if _is("esc"):
                bot.StopMove(); bot.ZeroTorque(); break

            now = time.time()
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now

                with _state_lock:
                    _state["vel"] = (vx, vy, omega)

            time.sleep(0.005)

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_stack] Keyboard / G-1 control disabled:", exc, file=sys.stderr)


# ---------------------------------------------------------------------------
# 网络配置验证
# ---------------------------------------------------------------------------

def check_network_interface(interface: str = "eth0") -> bool:
    """检查网络接口状态"""
    try:
        result = subprocess.run(['ip', 'addr', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[run_g1_stack] 网络接口 {interface} 正常")
            return True
        else:
            print(f"[run_g1_stack] 网络接口 {interface} 不存在或未激活")
            print("请检查网络配置或使用 --iface 参数指定正确的接口")
            return False
    except Exception as e:
        print(f"[run_g1_stack] 检查网络接口时出错: {e}")
        return False


# ---------------------------------------------------------------------------
# OpenCV 窗口合成
# ---------------------------------------------------------------------------

def _compose_canvas() -> Optional[np.ndarray]:
    """
    合成 OpenCV 窗口，将 RGB、深度和 SLAM 数据合并到一个画布中。

    Returns:
        Optional[np.ndarray]: 合成后的画布图像。
    """
    import numpy as np  # local import to avoid hard dep if script is only imported
    import cv2  # type: ignore

    with _state_lock:
        rgbd = _state.get("rgbd")
        slam = _state.get("slam")
        vx, vy, om = _state.get("vel", (0.0, 0.0, 0.0))

    # Place-holders -------------------------------------------------------
    if rgbd is None:
        rgbd = np.full((480, 1280, 3), 80, dtype=np.uint8)
        cv2.putText(rgbd, "No RealSense stream", (380, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if slam is None:
        slam = np.full((480, 480, 3), 60, dtype=np.uint8)
        cv2.putText(slam, "No SLAM data", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Compose – simple vertical stack
    top = rgbd
    bottom = cv2.copyMakeBorder(slam, 0, 0, 0, max(0, top.shape[1] - slam.shape[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0))

    canvas = np.vstack([top, bottom])

    # HUD with current velocities
    txt = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {om:+.2f}   –  Z: quit  ESC: e-stop"
    cv2.rectangle(canvas, (0, canvas.shape[0] - 40), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, txt, (10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return canvas


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    """
    主函数，启动所有模块并显示合成的 OpenCV 窗口。
    """
    parser = argparse.ArgumentParser(
        description="G1 机器人全栈控制系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--iface", default="eth0", 
                      help="与 Unitree G-1 连接的网络接口")
    parser.add_argument("--no-realsense", action="store_true",
                      help="禁用 RealSense 摄像头")
    parser.add_argument("--no-slam", action="store_true",
                      help="禁用 SLAM 功能")
    parser.add_argument("--no-robot", action="store_true",
                      help="禁用机器人控制")
    args = parser.parse_args()

    # 网络接口检查
    if not args.no_robot:
        if not check_network_interface(args.iface):
            print("网络接口检查失败，机器人控制可能无法正常工作")

    stop = threading.Event()

    # 启动后台线程
    workers = []
    
    if not args.no_realsense:
        workers.append(("RealSense", threading.Thread(target=_rx_realsense, args=(stop,), daemon=True)))
    
    if not args.no_slam:
        workers.append(("SLAM", threading.Thread(target=_run_slam, args=(stop,), daemon=True)))
    
    if not args.no_robot:
        workers.append(("G1", threading.Thread(target=_keyboard_thread, args=(stop, args.iface), daemon=True)))

    for name, t in workers:
        print(f"[run_g1_stack] 启动 {name} 线程...")
        t.start()

    # 显示 OpenCV 窗口
    try:
        import cv2

        print("[run_g1_stack] 系统就绪，显示主窗口...")
        print("控制说明:")
        print("  WASD: 移动控制  QE: 侧移  Z: 阻尼模式  ESC: 紧急停止")
        print("  窗口按键: Q 或 ESC 退出程序")

        while not stop.is_set():
            canvas = _compose_canvas()
            if canvas is None:
                time.sleep(0.05)
                continue

            cv2.imshow("G1-Stack", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                print("[run_g1_stack] 用户请求退出...")
                stop.set()
                break

        cv2.destroyAllWindows()

    except ImportError as e:
        print(f"OpenCV 依赖缺失: {e}")
        print("请安装: pip install opencv-python")
    except Exception as e:
        print(f"显示窗口时出错: {e}")
    finally:
        print("[run_g1_stack] 正在停止所有线程...")
        stop.set()
        for name, t in workers:
            print(f"[run_g1_stack] 等待 {name} 线程结束...")
            t.join(timeout=2.0)
            if t.is_alive():
                print(f"[run_g1_stack] 警告: {name} 线程未能及时停止")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[run_g1_stack] 接收到中断信号，正在退出...")
    except Exception as e:
        print(f"[run_g1_stack] 程序异常退出: {e}")
        sys.exit(1)