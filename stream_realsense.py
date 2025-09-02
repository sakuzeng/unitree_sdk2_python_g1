"""
stream_realsense.py
====================

A small convenience wrapper around Intel RealSense SDK (librealsense) that
demonstrates how to:

1. Detect a connected RealSense device.
2. Stream depth + colour (RGB) frames at the same resolution / FPS.
3. Optionally stream the two infrared (IR) channels and the IMU (gyroscope & accelerometer) if the
   selected model supports them (e.g. D435i).
4. Display the images live using OpenCV.
5. Exit cleanly when the user presses the **ESC** or **q** key.

This file is 100 % self-contained – the only runtime dependencies are:

* pyrealsense2  (``pip install pyrealsense2``)
* opencv-python  (``pip install opencv-python``)

No additional helper libraries or ROS runtimes are required.

Author: OpenAI Codex-CLI helper
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


def check_device_availability():
    """检查设备是否被其他进程占用"""
    try:
        result = subprocess.run(['lsof', '/dev/video*'], 
                              capture_output=True, text=True)
        if result.stdout:
            print("警告: 检测到摄像头设备被占用:")
            print(result.stdout)
            return False
        return True
    except Exception:
        return True  # 如果检查失败，假设设备可用


def reset_usb_devices():
    """重置 USB 摄像头设备"""
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
    """Converts a depth frame (16-bit, in millimetres) into an 8-bit BGR image.

    The function normalises the depth range to 0-255 and applies the OpenCV
    *JET* colour map so that closer objects appear red and farther objects
    blue.
    """
    # 修复：将 RealSense 数据转换为 numpy 数组
    depth_data = np.asanyarray(depth_frame.get_data())
    
    # 转换为 8-bit 灰度图像
    depth_image = cv2.convertScaleAbs(depth_data, alpha=0.03)
    
    # 转换为 BGR 格式
    depth_image_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
    
    # 应用颜色映射
    depth_coloured = cv2.applyColorMap(depth_image_bgr, cv2.COLORMAP_JET)
    return depth_coloured


def get_first_device(context: rs.context) -> Optional[rs.device]:
    """Return the first RealSense device if any, otherwise *None*."""
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
):
    """带重试机制的运行函数"""
    
    for attempt in range(max_retries):
        try:
            print(f"尝试启动摄像头 (第 {attempt + 1}/{max_retries} 次)...")
            
            # 检查设备可用性
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
                raise  # 其他错误直接抛出


def run(
    rgb_width: int = 640,
    rgb_height: int = 480,
    fps: int = 30,
    enable_infra: bool = False,
    enable_imu: bool = False,
):
    """Open a pipeline, start streaming, and display frames."""

    ctx = rs.context()
    device = get_first_device(ctx)

    if device is None:
        raise RuntimeError("No RealSense device found. Plug in a camera and try again.")

    print("Found device:", device.get_info(rs.camera_info.name))
    print("  Serial number:", device.get_info(rs.camera_info.serial_number))
    print("  Firmware ver.:", device.get_info(rs.camera_info.firmware_version))

    # Configure pipeline streams
    pipeline = rs.pipeline(ctx)
    config = rs.config()

    # If you have multiple cameras, you may specify the serial number here:
    # config.enable_device(<serial>)

    # Depth and colour should have matching resolution + fps when we plan to
    # perform alignment.
    config.enable_stream(rs.stream.depth, rgb_width, rgb_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)

    if enable_infra:
        # Left and right infrared
        config.enable_stream(rs.stream.infrared, 1, rgb_width, rgb_height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, rgb_width, rgb_height, rs.format.y8, fps)

    if enable_imu:
        # D435i exposes gyro at 400 Hz and accel at 250 Hz (but we can ask for
        # any value <= the max).
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)

    # Apply some recommended depth-postprocessing options to improve quality.
    spatial_filter = rs.spatial_filter()  # edge-preserving smoothing
    temporal_filter = rs.temporal_filter()  # reduces depth noise over time

    align_to = rs.stream.color  # align depth to colour coordinate system
    align = rs.align(align_to)

    # Start streaming
    print("Starting pipeline …")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        if "Device or resource busy" in str(e):
            raise RuntimeError(f"摄像头设备被占用。请运行以下命令释放设备:\n"
                             f"sudo pkill -f realsense\n"
                             f"sudo modprobe -r uvcvideo && sudo modprobe uvcvideo")
        else:
            raise

    print("Camera intrinsics (colour stream):")
    colour_intr: rs.video_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colour_intr.get_intrinsics()
    print(f"  Resolution    : {intr.width} × {intr.height}")
    print(f"  Focal length  : fx={intr.fx:.1f}  fy={intr.fy:.1f}")
    print(f"  Principal pt. : cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    # Main loop ----------------------------------------------------------------
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