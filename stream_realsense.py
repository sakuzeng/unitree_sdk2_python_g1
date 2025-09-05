"""
stream_realsense.py
====================

本脚本是 Intel RealSense SDK (librealsense) 的一个简单封装，演示以下功能：

1. 检测连接的 RealSense 设备。
2. 同时流式传输深度和 RGB 图像（分辨率和帧率相同）。
3. 可选地流式传输两个红外 (IR) 通道和 IMU 数据（如陀螺仪和加速度计）。
4. 使用 OpenCV 实时显示图像。
5. 用户按下 **ESC** 或 **q** 键时干净退出。

依赖:
- `pyrealsense2` (`pip install pyrealsense2`)
- `opencv-python` (`pip install opencv-python`)

无需额外的辅助库或 ROS 运行时环境。

作者: OpenAI Codex-CLI helper
"""

from __future__ import annotations

import sys
import time
import subprocess
from typing import Optional

import cv2
import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:  # pragma: no cover – only happens if dependency missing
    raise SystemExit(
        "pyrealsense2 is not installed. Install it with 'pip install pyrealsense2'"
    ) from exc


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def check_device_availability() -> bool:
    """
    检查 RealSense 设备是否被其他进程占用。

    Returns:
        bool: 如果设备可用，返回 True；否则返回 False。
    """
    try:
        result = subprocess.run(['lsof', '/dev/video*'], capture_output=True, text=True)
        if result.stdout:
            print("警告: 检测到摄像头设备被占用:")
            print(result.stdout)
            return False
        return True
    except Exception:
        return True  # 如果检查失败，假设设备可用


def reset_usb_devices() -> None:
    """
    重置 USB 摄像头设备。
    """
    try:
        print("正在重置 USB 摄像头设备...")
        subprocess.run(['sudo', 'modprobe', '-r', 'uvcvideo'], check=False)
        time.sleep(1)
        subprocess.run(['sudo', 'modprobe', 'uvcvideo'], check=False)
        time.sleep(2)
        print("USB 设备重置完成")
    except Exception as e:
        print(f"重置 USB 设备失败: {e}")


def colourise_depth(depth_frame: rs.depth_frame) -> cv2.Mat:
    """
    将深度帧 (16-bit, 毫米) 转换为 8-bit BGR 图像。

    Args:
        depth_frame (rs.depth_frame): RealSense 深度帧。

    Returns:
        cv2.Mat: 伪彩色深度图像。
    """
    depth_data = np.asanyarray(depth_frame.get_data())
    depth_image = cv2.convertScaleAbs(depth_data, alpha=0.03)
    depth_image_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(depth_image_bgr, cv2.COLORMAP_JET)


def get_first_device(context: rs.context) -> Optional[rs.device]:
    """
    返回第一个 RealSense 设备，如果没有设备则返回 None。

    Args:
        context (rs.context): RealSense 上下文。

    Returns:
        Optional[rs.device]: 第一个 RealSense 设备。
    """
    devices = context.query_devices()
    if len(devices) == 0:
        return None
    return devices[0]


# ---------------------------------------------------------------------------
# Main streaming routine
# ---------------------------------------------------------------------------


def run_with_retry(
    rgb_width: int = 640,
    rgb_height: int = 480,
    fps: int = 30,
    enable_infra: bool = False,
    enable_imu: bool = False,
    max_retries: int = 3
) -> None:
    """
    带重试机制的运行函数。

    Args:
        rgb_width (int): RGB 图像宽度。
        rgb_height (int): RGB 图像高度。
        fps (int): 帧率。
        enable_infra (bool): 是否启用红外流。
        enable_imu (bool): 是否启用 IMU 数据流。
        max_retries (int): 最大重试次数。
    """
    for attempt in range(max_retries):
        try:
            print(f"尝试启动摄像头 (第 {attempt + 1}/{max_retries} 次)...")
            if not check_device_availability():
                print("设备被占用，尝试重置...")
                reset_usb_devices()
            run(rgb_width, rgb_height, fps, enable_infra, enable_imu)
            return  # 成功运行，退出重试循环
        except RuntimeError as e:
            if "Device or resource busy" in str(e) or "xioctl" in str(e):
                print(f"设备忙碌错误: {e}")
                if attempt < max_retries - 1:
                    print("等待并重试...")
                    time.sleep(2)
                    reset_usb_devices()
                else:
                    print("所有重试都失败了")
                    raise
            else:
                raise


def run(
    rgb_width: int = 640,
    rgb_height: int = 480,
    fps: int = 30,
    enable_infra: bool = False,
    enable_imu: bool = False,
) -> None:
    """
    打开管道，开始流式传输，并显示帧数据。

    Args:
        rgb_width (int): RGB 图像宽度。
        rgb_height (int): RGB 图像高度。
        fps (int): 帧率。
        enable_infra (bool): 是否启用红外流。
        enable_imu (bool): 是否启用 IMU 数据流。
    """
    ctx = rs.context()
    device = get_first_device(ctx)

    if device is None:
        raise RuntimeError("No RealSense device found. Plug in a camera and try again.")

    print("Found device:", device.get_info(rs.camera_info.name))
    print("  Serial number:", device.get_info(rs.camera_info.serial_number))
    print("  Firmware ver.:", device.get_info(rs.camera_info.firmware_version))

    # 配置管道流
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_stream(rs.stream.depth, rgb_width, rgb_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)

    if enable_infra:
        config.enable_stream(rs.stream.infrared, 1, rgb_width, rgb_height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, rgb_width, rgb_height, rs.format.y8, fps)

    if enable_imu:
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)

    # Apply some recommended depth-postprocessing options to improve quality.
    spatial_filter = rs.spatial_filter()  # edge-preserving smoothing
    temporal_filter = rs.temporal_filter()  # reduces depth noise over time

    align_to = rs.stream.color  # align depth to colour coordinate system
    align = rs.align(align_to)

    # Start streaming
    print("Starting pipeline …")
    profile = pipeline.start(config)

    print("Camera intrinsics (colour stream):")
    colour_intr: rs.video_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colour_intr.get_intrinsics()
    print(f"  Resolution    : {intr.width} × {intr.height}")
    print(f"  Focal length  : fx={intr.fx:.1f}  fy={intr.fy:.1f}")
    print(f"  Principal pt. : cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    # 主循环
    last_time = time.perf_counter()
    try:
        print("摄像头流已启动。按 ESC 或 'q' 键退出。")
        while True:
            frames = pipeline.wait_for_frames()

            # Align depth to colour so that pixel (u,v) matches
            aligned_frames = align.process(frames)

            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            colour_frame: rs.video_frame = aligned_frames.get_color_frame()

            if not depth_frame or not colour_frame:
                # Should rarely happen, but continue gracefully.
                continue

            # Post-process depth
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)

            # Convert RealSense frames to numpy arrays
            colour_image = np.asanyarray(colour_frame.get_data())  # 修复：确保是 numpy 数组
            depth_coloured = colourise_depth(depth_frame)

            # Combine side-by-side for display (make sure both are same height)
            combined = cv2.hconcat([colour_image, depth_coloured])

            # FPS counter
            now = time.perf_counter()
            fps_calc = 1.0 / (now - last_time)
            last_time = now
            cv2.putText(
                combined,
                f"FPS: {fps_calc:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RealSense RGB + Depth", combined)

            if enable_infra:
                ir_left = aligned_frames.get_infrared_frame(1)
                ir_right = aligned_frames.get_infrared_frame(2)
                if ir_left and ir_right:
                    ir_left_img = np.asanyarray(ir_left.get_data())  # 修复：转换为 numpy 数组
                    ir_right_img = np.asanyarray(ir_right.get_data())  # 修复：转换为 numpy 数组
                    cv2.imshow("IR-left", ir_left_img)
                    cv2.imshow("IR-right", ir_right_img)

            if enable_imu:
                gyro: rs.motion_frame = frames.first_or_default(rs.stream.gyro)
                accel: rs.motion_frame = frames.first_or_default(rs.stream.accel)
                if gyro and accel:
                    g_data = gyro.as_motion_frame().get_motion_data()
                    a_data = accel.as_motion_frame().get_motion_data()
                    print(
                        f"Gyro [rad/s]: x={g_data.x:+.3f} y={g_data.y:+.3f} z={g_data.z:+.3f} | "
                        f"Accel [m/s²]: x={a_data.x:+.3f} y={a_data.y:+.3f} z={a_data.z:+.3f}",
                        end="\r",
                    )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # ESC or q to quit
                break
    finally:
        print("\nStopping pipeline, closing windows …")
        pipeline.stop()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# "python stream_realsense.py" entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple RealSense viewer (colour + depth) written in Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--width", type=int, default=640, help="Width of the RGB/depth stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the RGB/depth stream")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--infra", action="store_true", help="Also display the two IR streams")
    parser.add_argument("--imu", action="store_true", help="Print IMU (gyro + accel) readings")
    parser.add_argument("--no-retry", action="store_true", help="禁用重试机制")

    args = parser.parse_args()

    try:
        if args.no_retry:
            run(
                rgb_width=args.width,
                rgb_height=args.height,
                fps=args.fps,
                enable_infra=args.infra,
                enable_imu=args.imu,
            )
        else:
            run_with_retry(
                rgb_width=args.width,
                rgb_height=args.height,
                fps=args.fps,
                enable_infra=args.infra,
                enable_imu=args.imu,
            )
    except RuntimeError as err:
        sys.exit(str(err))