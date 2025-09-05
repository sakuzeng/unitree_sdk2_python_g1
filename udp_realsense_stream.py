#!/usr/bin/env python3
"""
udp_realsense_stream.py
========================

使用 Intel RealSense 摄像头和 UDP 套接字实现视频流传输。

功能:
1. 使用 RealSense 摄像头采集 RGB 和深度图像。
2. 使用 UDP 套接字传输压缩后的图像数据。
3. 支持图像质量调整和深度图像伪彩色处理。

依赖:
- `pyrealsense2` (`pip install pyrealsense2`)
- `opencv-python` (`pip install opencv-python`)
- `numpy` (`pip install numpy`)

运行方法:
    python3 udp_realsense_stream.py --client-ip <接收端IP地址>

参数说明:
- `--client-ip`: 接收端的 IP 地址。
- `--width` 和 `--height`: 图像分辨率，默认为 640x480。
- `--fps`: 帧率，默认为 30。
- `--rgb-quality` 和 `--depth-quality`: 图像质量，范围为 1-100。

注意:
- 确保接收端已准备好接收 UDP 数据。
- 深度图像默认使用伪彩色显示。
"""

import argparse
import socket
import struct
import cv2
import numpy as np
import pyrealsense2 as rs
import time


class VideoStreamer:
    """
    视频流传输器类，用于通过 UDP 套接字发送 RGB 和深度图像。

    Attributes:
        client_ip (str): 接收端的 IP 地址。
        rgb_port (int): RGB 图像的传输端口。
        depth_port (int): 深度图像的传输端口。
        rgb_socket (socket.socket): 用于发送 RGB 数据的 UDP 套接字。
        depth_socket (socket.socket): 用于发送深度数据的 UDP 套接字。
    """

    def __init__(self, client_ip: str, rgb_port: int = 5600, depth_port: int = 5602):
        """
        初始化视频流传输器。

        Args:
            client_ip (str): 接收端的 IP 地址。
            rgb_port (int): RGB 图像的传输端口，默认为 5600。
            depth_port (int): 深度图像的传输端口，默认为 5602。
        """
        self.client_ip = client_ip
        self.rgb_port = rgb_port
        self.depth_port = depth_port

        # 创建 UDP 套接字
        self.rgb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 设置缓冲区大小
        self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

    def send_frame(self, frame: np.ndarray, socket_obj: socket.socket, port: int, quality: int = 80) -> bool:
        """
        发送压缩后的帧数据。

        Args:
            frame (np.ndarray): 要发送的图像帧。
            socket_obj (socket.socket): 用于发送数据的 UDP 套接字。
            port (int): 目标端口。
            quality (int): 图像压缩质量，范围为 1-100。

        Returns:
            bool: 如果发送成功，返回 True；否则返回 False。
        """
        try:
            # JPEG 压缩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_img = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                return False

            data = encoded_img.tobytes()

            # 分包发送（UDP 包大小限制）
            max_packet_size = 60000
            total_packets = (len(data) + max_packet_size - 1) // max_packet_size

            for i in range(total_packets):
                start = i * max_packet_size
                end = min(start + max_packet_size, len(data))
                packet_data = data[start:end]

                # 包头：包序号(4字节) + 总包数(4字节) + 数据长度(4字节)
                header = struct.pack('!III', i, total_packets, len(packet_data))
                packet = header + packet_data

                socket_obj.sendto(packet, (self.client_ip, port))

            return True

        except Exception as e:
            print(f"发送帧失败: {e}")
            return False

    def colourise_depth(self, depth16: np.ndarray) -> np.ndarray:
        """
        将深度图像转换为伪彩色图像。

        Args:
            depth16 (np.ndarray): 16 位深度图像。

        Returns:
            np.ndarray: 伪彩色深度图像。
        """
        depth_clip = np.clip(depth16, 0, 6000)
        depth8 = cv2.convertScaleAbs(depth_clip, alpha=255.0 / 6000)
        return cv2.applyColorMap(depth8, cv2.COLORMAP_PLASMA)

    def close(self):
        """
        关闭 UDP 套接字。
        """
        self.rgb_socket.close()
        self.depth_socket.close()


def main():
    """
    主函数，初始化 RealSense 摄像头并启动视频流传输。
    """
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--client-ip", required=True, help="接收端 IP 地址")
    ap.add_argument("--width", type=int, default=640, help="图像宽度")
    ap.add_argument("--height", type=int, default=480, help="图像高度")
    ap.add_argument("--fps", type=int, default=30, help="帧率")
    ap.add_argument("--rgb-quality", type=int, default=80, help="RGB 图像质量 (1-100)")
    ap.add_argument("--depth-quality", type=int, default=60, help="深度图像质量 (1-100)")
    args = ap.parse_args()

    # 初始化 RealSense
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipe = rs.pipeline()
    pipe.start(cfg)

    # 初始化流传输器
    streamer = VideoStreamer(args.client_ip)

    # 滤波器
    temp_filter = rs.temporal_filter()

    print(f"开始向 {args.client_ip} 发送视频流...")
    print("按 Ctrl+C 停止")

    try:
        frame_time = 1.0 / args.fps
        last_time = time.time()

        while True:
            current_time = time.time()
            if current_time - last_time < frame_time:
                time.sleep(0.001)
                continue

            frames = pipe.wait_for_frames()

            # RGB 帧
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                streamer.send_frame(color_image, streamer.rgb_socket,
                                    streamer.rgb_port, args.rgb_quality)

            # 深度帧
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_frame = temp_filter.process(depth_frame)
                depth16 = np.asanyarray(depth_frame.get_data())
                depth_colored = streamer.colourise_depth(depth16)
                streamer.send_frame(depth_colored, streamer.depth_socket,
                                    streamer.depth_port, args.depth_quality)

            last_time = current_time

    except KeyboardInterrupt:
        print("\n正在停止视频流...")
    finally:
        streamer.close()
        pipe.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"错误: {exc}")
        exit(1)