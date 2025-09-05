#!/usr/bin/env python3
"""
OpenCV 版本的视频流接收端，无需 gi 依赖

本脚本通过 UDP 套接字接收 RGB 和深度图像数据，并使用 OpenCV 进行解码和显示。

功能:
- 接收压缩后的 RGB 和深度图像数据。
- 支持多线程接收，确保实时性。
- 使用 OpenCV 解码并显示接收到的图像。
- 支持帧率显示和退出控制。

运行方法:
    python3 receive_realsense_udp.py
"""

import socket
import struct
import cv2
import numpy as np
import threading
import time
from collections import defaultdict


class VideoReceiver:
    """
    视频流接收器类，用于通过 UDP 接收 RGB 和深度图像数据。

    Attributes:
        rgb_port (int): RGB 图像接收端口。
        depth_port (int): 深度图像接收端口。
        rgb_socket (socket.socket): 用于接收 RGB 数据的 UDP 套接字。
        depth_socket (socket.socket): 用于接收深度数据的 UDP 套接字。
        rgb_packets (dict): 缓存 RGB 数据包的字典。
        depth_packets (dict): 缓存深度数据包的字典。
        latest_rgb (np.ndarray): 最新的 RGB 图像帧。
        latest_depth (np.ndarray): 最新的深度图像帧。
        running (bool): 接收器运行状态。
    """

    def __init__(self, rgb_port: int = 5600, depth_port: int = 5602):
        """
        初始化视频接收器。

        Args:
            rgb_port (int): RGB 图像接收端口，默认为 5600。
            depth_port (int): 深度图像接收端口，默认为 5602。
        """
        self.rgb_port = rgb_port
        self.depth_port = depth_port

        # 创建 UDP 套接字
        self.rgb_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 绑定端口
        self.rgb_socket.bind(('', rgb_port))
        self.depth_socket.bind(('', depth_port))

        # 设置接收缓冲区
        self.rgb_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        self.depth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        # 帧缓存
        self.rgb_packets = defaultdict(dict)
        self.depth_packets = defaultdict(dict)

        self.latest_rgb = None
        self.latest_depth = None

        self.running = True

    def receive_packets(self, socket_obj: socket.socket, packet_dict: dict, frame_attr: str):
        """
        接收数据包的线程函数。

        Args:
            socket_obj (socket.socket): 用于接收数据的 UDP 套接字。
            packet_dict (dict): 用于缓存数据包的字典。
            frame_attr (str): 存储解码后图像的属性名称。
        """
        while self.running:
            try:
                data, addr = socket_obj.recvfrom(65536)

                if len(data) < 12:  # 最小包头大小
                    continue

                # 解析包头
                packet_id, total_packets, data_len = struct.unpack('!III', data[:12])
                packet_data = data[12:12 + data_len]

                # 存储包数据
                frame_id = int(time.time() * 1000) // 100  # 简单的帧 ID
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

                    if frame is not None:
                        setattr(self, frame_attr, frame)

                    # 清理旧数据
                    del packet_dict[frame_id]

                    # 清理过期帧
                    current_time = int(time.time() * 1000) // 100
                    expired_frames = [fid for fid in packet_dict.keys()
                                      if current_time - fid > 10]
                    for fid in expired_frames:
                        del packet_dict[fid]

            except Exception as e:
                if self.running:
                    print(f"接收包错误: {e}")
                break

    def start(self):
        """
        启动接收线程。
        """
        self.rgb_thread = threading.Thread(
            target=self.receive_packets,
            args=(self.rgb_socket, self.rgb_packets, 'latest_rgb')
        )
        self.depth_thread = threading.Thread(
            target=self.receive_packets,
            args=(self.depth_socket, self.depth_packets, 'latest_depth')
        )

        self.rgb_thread.daemon = True
        self.depth_thread.daemon = True

        self.rgb_thread.start()
        self.depth_thread.start()

    def get_frames(self):
        """
        获取最新的 RGB 和深度图像帧。

        Returns:
            tuple: 最新的 RGB 图像帧和深度图像帧。
        """
        return self.latest_rgb, self.latest_depth

    def close(self):
        """
        关闭接收器，释放资源。
        """
        self.running = False
        self.rgb_socket.close()
        self.depth_socket.close()


def main():
    """
    程序主入口，启动视频接收器并显示接收到的图像。
    """
    receiver = VideoReceiver()
    receiver.start()

    print("开始接收视频流...")
    print("按 ESC 或 'q' 退出")

    last_time = time.perf_counter()

    try:
        while True:
            rgb_frame, depth_frame = receiver.get_frames()

            if rgb_frame is not None and depth_frame is not None:
                # 组合显示
                combo = cv2.hconcat([rgb_frame, depth_frame])

                # FPS 显示
                current_time = time.perf_counter()
                fps = 1.0 / (current_time - last_time) if last_time > 0 else 0
                last_time = current_time

                cv2.putText(combo, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("RGB + Depth (OpenCV)", combo)

            elif rgb_frame is not None:
                cv2.imshow("RGB Only", rgb_frame)

            elif depth_frame is not None:
                cv2.imshow("Depth Only", depth_frame)

            # 退出检查
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC 或 'q'
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        receiver.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()