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
# 1. RealSense 接收器 (支持 GStreamer 和 OpenCV)
# ---------------------------------------------------------------------------

def _rx_realsense_local(stop: threading.Event) -> None:
    """
    直接从本地 RealSense 设备捕获 RGB 和深度图像。
    参考自 stream_realsense.py。

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

        # --- 初始化 RealSense ---
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

        align_to = rs.stream.color
        align = rs.align(align_to)

        print("[run_g1_stack] 正在启动 RealSense 管道...")
        pipeline.start(config)
        print("[run_g1_stack] RealSense 管道已启动。")

        last = time.perf_counter()

        while not stop.is_set():
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            color_frame: rs.video_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                time.sleep(0.01)
                continue

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

        # --- 清理 ---
        print("[run_g1_stack] 正在停止 RealSense 管道...")
        pipeline.stop()

    except ImportError:
        print("[run_g1_stack] RealSense 接收器已禁用: 'pyrealsense2' 未安装。", file=sys.stderr)
    except Exception as exc:
        print(f"[run_g1_stack] 本地 RealSense 接收器失败: {exc}", file=sys.stderr)


def _rx_realsense_gstreamer(stop: threading.Event) -> None:
    """
    使用 GStreamer 接收 RealSense 的 RGB 和深度图像数据。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    try:
        import gi  # type: ignore

        gi.require_version("Gst", "1.0")
        gi.require_version("GstApp", "1.0")
        from gi.repository import Gst, GstApp  # type: ignore

        import numpy as np  # pylint: disable=import-error
        import cv2  # type: ignore

        RGB_PORT, DEPTH_PORT, WIDTH, HEIGHT, FPS = 5600, 5602, 640, 480, 30

        def _build_sink(port: int, payload: int) -> tuple[Any, Any]:
            pipeline = Gst.parse_launch(
                f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload={payload} ! "
                "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=true sync=false drop=true"
            )
            sink = pipeline.get_by_name("sink")
            return sink, pipeline

        Gst.init(None)

        rgb_sink, rgb_pipe = _build_sink(RGB_PORT, 96)
        d_sink, d_pipe = _build_sink(DEPTH_PORT, 97)

        for p in (rgb_pipe, d_pipe):
            p.set_state(Gst.State.PLAYING)

        last = time.perf_counter()

        while not stop.is_set():
            sample_rgb = rgb_sink.emit("try-pull-sample", Gst.SECOND // FPS)
            sample_d = d_sink.emit("try-pull-sample", Gst.SECOND // FPS)

            if not sample_rgb or not sample_d:
                time.sleep(0.005)
                continue

            buf_rgb = sample_rgb.get_buffer()
            buf_d = sample_d.get_buffer()

            rgb = np.frombuffer(buf_rgb.extract_dup(0, buf_rgb.get_size()), dtype=np.uint8)
            rgb = rgb.reshape((HEIGHT, WIDTH, 3))

            depth_bgr = np.frombuffer(buf_d.extract_dup(0, buf_d.get_size()), dtype=np.uint8)
            depth_bgr = depth_bgr.reshape((HEIGHT, WIDTH, 3))

            combo = cv2.hconcat([rgb, depth_bgr])

            fps = 1.0 / (time.perf_counter() - last)
            last = time.perf_counter()
            cv2.putText(combo, f"RGB+Depth  {fps:5.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            with _state_lock:
                _state["rgbd"] = combo

        # Tear-down ------------------------------------------------------
        for p in (rgb_pipe, d_pipe):
            p.set_state(Gst.State.NULL)

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_stack] GStreamer RealSense receiver disabled:", exc, file=sys.stderr)


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
    选择 RealSense 接收器的实现 (优先使用 OpenCV)。

    Args:
        stop (threading.Event): 用于控制线程停止的事件。
    """
    print("[run_g1_stack] 强制使用 OpenCV 版本的 RealSense 接收器")
    _rx_realsense_opencv(stop)
    print("[run_g1_stack] 使用本地 RealSense 接收器。")
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
        print("[run_geoff_stack] SLAM viewer patch failed:", exc, file=sys.stderr)


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
        print("[run_geoff_stack] SLAM disabled:", exc, file=sys.stderr)


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
        print("[run_geoff_stack] Keyboard / G-1 control disabled:", exc, file=sys.stderr)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0", help="与 Unitree G-1 连接的网络接口")
    args = parser.parse_args()

    stop = threading.Event()

    # 启动后台线程
    workers = [
        ("RealSense", threading.Thread(target=_rx_realsense, args=(stop,), daemon=True)),
        ("SLAM", threading.Thread(target=_run_slam, args=(stop,), daemon=True)),
        ("G1", threading.Thread(target=_keyboard_thread, args=(stop, args.iface), daemon=True)),
    ]

    for name, t in workers:
        t.start()

    # 显示 OpenCV 窗口
    try:
        import cv2

        while not stop.is_set():
            canvas = _compose_canvas()
            if canvas is None:
                time.sleep(0.05)
                continue

            cv2.imshow("Geoff-Stack", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                stop.set()
                break

        cv2.destroyAllWindows()

    finally:
        stop.set()
        for name, t in workers:
            t.join(timeout=1.0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass