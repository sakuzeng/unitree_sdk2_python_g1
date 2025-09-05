"""receive_realsense_gst.py – 基于 GStreamer 的视频流接收客户端

本脚本通过 GStreamer 接收 RGB 和深度图像数据流，并使用 OpenCV 进行显示。
它不依赖 OpenCV 的 `cv2.VideoCapture`，因此即使 OpenCV 缺少 GStreamer 支持，
本脚本仍然可以正常工作。

功能:
- 使用 GStreamer 的 `appsink` 接收 RGB 和深度图像数据。
- 将接收到的图像数据转换为 NumPy 数组。
- 使用 OpenCV 显示 RGB 和深度图像，并支持帧率显示。

运行环境:
- 在接收端（如笔记本或工作站）运行本脚本。
- 确保发送端（如 Jetson）正在运行 `jetson_realsense_stream.py`。

依赖:
- GStreamer 和相关插件:
    sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 \
            gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
            gstreamer1.0-plugins-bad gstreamer1.0-libav
- Python 库:
    python3 -m pip install --upgrade numpy opencv-python

默认配置:
- RGB 数据端口: 5600
- 深度数据端口: 5602
- 图像分辨率: 640x480
- 帧率: 30 FPS
"""

from __future__ import annotations

import sys
import time

import numpy as np
import cv2

# GStreamer
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp


# 配置参数
RGB_PORT = 5600
DEPTH_PORT = 5602
WIDTH = 640
HEIGHT = 480
FPS = 30


def build_rgb_sink() -> tuple[GstApp.AppSink, Gst.Pipeline]:
    """
    构建用于接收 RGB 图像数据的 GStreamer 管道。

    Returns:
        tuple: 包含 `appsink` 和 GStreamer 管道的元组。
    """
    pipeline = Gst.parse_launch(
        f"udpsrc port={RGB_PORT} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def build_depth_sink() -> tuple[GstApp.AppSink, Gst.Pipeline]:
    """
    构建用于接收深度图像数据的 GStreamer 管道。

    Returns:
        tuple: 包含 `appsink` 和 GStreamer 管道的元组。
    """
    pipeline = Gst.parse_launch(
        f"udpsrc port={DEPTH_PORT} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def colourise_depth(depth16: np.ndarray) -> np.ndarray:
    """
    将 16 位深度图像转换为伪彩色图像。

    Args:
        depth16 (np.ndarray): 16 位深度图像。

    Returns:
        np.ndarray: 伪彩色深度图像。
    """
    depth_clip = np.clip(depth16, 0, 6000)
    depth8 = cv2.convertScaleAbs(depth_clip, alpha=255.0 / 6000)
    return cv2.applyColorMap(depth8, cv2.COLORMAP_PLASMA)


def main() -> None:
    """
    主函数，初始化 GStreamer 管道并接收和显示 RGB 与深度图像。
    """
    # 初始化 GStreamer
    Gst.init(None)

    # 构建 RGB 和深度图像的 GStreamer 管道
    rgb_sink, rgb_pipeline = build_rgb_sink()
    depth_sink, depth_pipeline = build_depth_sink()

    # 启动管道
    for p in (rgb_pipeline, depth_pipeline):
        p.set_state(Gst.State.PLAYING)

    last = time.perf_counter()

    try:
        while True:
            # 从 GStreamer 管道中获取 RGB 和深度图像样本
            sample_rgb = rgb_sink.emit("try-pull-sample", Gst.SECOND // FPS)
            sample_d = depth_sink.emit("try-pull-sample", Gst.SECOND // FPS)

            if not sample_rgb or not sample_d:
                # 如果未接收到样本，避免忙等待
                time.sleep(0.005)
                continue

            # 提取 RGB 图像缓冲区
            buf_rgb = sample_rgb.get_buffer()
            rgb = np.frombuffer(buf_rgb.extract_dup(0, buf_rgb.get_size()), dtype=np.uint8)
            rgb = rgb.reshape((HEIGHT, WIDTH, 3))

            # 提取深度图像缓冲区
            buf_d = sample_d.get_buffer()
            depth_bgr = np.frombuffer(buf_d.extract_dup(0, buf_d.get_size()), dtype=np.uint8)
            depth_bgr = depth_bgr.reshape((HEIGHT, WIDTH, 3))

            # 合并 RGB 和深度图像
            combo = cv2.hconcat([rgb, depth_bgr])

            # 显示帧率
            now = time.perf_counter()
            fps = 1.0 / (now - last)
            last = now
            cv2.putText(combo, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 显示图像
            cv2.imshow("RGB + Depth", combo)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):  # 按 ESC 或 'q' 退出
                break

    finally:
        # 停止管道并释放资源
        for p in (rgb_pipeline, depth_pipeline):
            p.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)