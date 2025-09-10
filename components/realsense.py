"""
RealSense 摄像头组件
"""
from __future__ import annotations

import time
import subprocess
import grp
import os
from typing import Optional

from .base import ComponentBase, StateManager


class RealSenseComponent(ComponentBase):
    """RealSense 摄像头数据采集组件"""
    
    def __init__(self, state_manager: StateManager, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__("RealSense")
        self.state_manager = state_manager
        self.width = width
        self.height = height
        self.fps = fps
    
    def _run(self) -> None:
        """RealSense 数据采集主循环"""
        try:
            import pyrealsense2 as rs
            import numpy as np
            import cv2
            
            # 权限检查
            self._check_video_permissions()
            
            # 设备检查和重试机制
            max_retries = 3
            pipeline = None
            
            for attempt in range(max_retries):
                try:
                    print(f"[{self.name}] 尝试启动 (第 {attempt + 1}/{max_retries} 次)...")
                    
                    if not self._check_device_availability():
                        print(f"[{self.name}] 设备被占用，尝试重置...")
                        self._reset_usb_devices()
                    
                    # 检查设备连接
                    ctx = rs.context()
                    device = self._get_first_device(ctx)
                    
                    if device is None:
                        raise RuntimeError("未找到 RealSense 设备")
                    
                    print(f"[{self.name}] 找到设备: {device.get_info(rs.camera_info.name)}")
                    print(f"[{self.name}] 序列号: {device.get_info(rs.camera_info.serial_number)}")
                    
                    # 初始化 RealSense
                    pipeline = rs.pipeline(ctx)
                    config = rs.config()
                    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                    
                    # 后处理滤波器
                    spatial_filter = rs.spatial_filter()
                    temporal_filter = rs.temporal_filter()
                    align = rs.align(rs.stream.color)
                    
                    print(f"[{self.name}] 正在启动管道...")
                    profile = pipeline.start(config)
                    print(f"[{self.name}] 管道已启动")
                    
                    # 获取相机内参
                    colour_intr = profile.get_stream(rs.stream.color).as_video_stream_profile()
                    intr = colour_intr.get_intrinsics()
                    print(f"[{self.name}] 相机内参: {intr.width}×{intr.height}, fx={intr.fx:.1f}, fy={intr.fy:.1f}")
                    
                    break  # 成功初始化
                    
                except RuntimeError as e:
                    if "Device or resource busy" in str(e) or "xioctl" in str(e):
                        print(f"[{self.name}] 设备忙碌错误: {e}")
                        if attempt < max_retries - 1:
                            print(f"[{self.name}] 等待并重试...")
                            time.sleep(2)
                            self._reset_usb_devices()
                        else:
                            print(f"[{self.name}] 所有重试都失败了")
                            return
                    else:
                        print(f"[{self.name}] 初始化失败: {e}")
                        return
            
            if pipeline is None:
                print(f"[{self.name}] 无法初始化管道")
                return
            
            # 主循环
            last_time = time.perf_counter()
            
            while self.is_running():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=100)
                    aligned_frames = align.process(frames)
                    
                    depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    
                    if not depth_frame or not color_frame:
                        continue
                    
                    # 应用滤波器
                    depth_frame = spatial_filter.process(depth_frame)
                    depth_frame = temporal_filter.process(depth_frame)
                    
                    # 转换图像
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_colored = self._colorize_depth(depth_frame)
                    
                    # 合成图像
                    combo = cv2.hconcat([color_image, depth_colored])
                    
                    # 添加 FPS 信息
                    fps = 1.0 / (time.perf_counter() - last_time)
                    last_time = time.perf_counter()
                    cv2.putText(combo, f"RGB+Depth {fps:5.1f} FPS", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # 更新状态
                    self.state_manager.set("rgbd", combo)
                    
                except Exception as e:
                    print(f"[{self.name}] 帧处理错误: {e}")
                    time.sleep(0.01)
            
            # 清理
            if pipeline:
                pipeline.stop()
                print(f"[{self.name}] 管道已停止")
            
        except ImportError:
            print(f"[{self.name}] pyrealsense2 未安装，组件禁用")
        except Exception as e:
            print(f"[{self.name}] 组件异常: {e}")
    
    def _colorize_depth(self, depth_frame: "rs.depth_frame") -> "cv2.Mat":
        """将深度数据转换为伪彩色图像"""
        import cv2
        import numpy as np
        
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.convertScaleAbs(depth_data, alpha=0.03)
        depth_image_bgr = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        return cv2.applyColorMap(depth_image_bgr, cv2.COLORMAP_JET)
    
    def _check_device_availability(self) -> bool:
        """检查 RealSense 设备是否被其他进程占用"""
        try:
            result = subprocess.run(['lsof', '/dev/video*'], capture_output=True, text=True)
            if result.stdout:
                print(f"[{self.name}] 警告: 检测到摄像头设备被占用:")
                print(result.stdout)
                return False
            return True
        except FileNotFoundError:
            return True
        except Exception:
            return True
    
    def _check_video_permissions(self) -> bool:
        """检查当前用户是否有访问视频设备的权限"""
        try:
            video_gid = grp.getgrnam('video').gr_gid
            user_groups = os.getgroups()
            
            if video_gid in user_groups:
                return True
            else:
                print(f"[{self.name}] 警告: 当前用户不在 video 组中")
                print("建议运行: sudo usermod -a -G video $USER")
                return False
        except KeyError:
            return True
        except Exception:
            return True
    
    def _reset_usb_devices(self) -> None:
        """重置 USB 摄像头设备"""
        try:
            print(f"[{self.name}] 正在尝试重置 USB 摄像头设备...")
            
            try:
                import pyrealsense2 as rs
                ctx = rs.context()
                devices = ctx.query_devices()
                for device in devices:
                    try:
                        device.hardware_reset()
                        print(f"[{self.name}] 已重置设备: {device.get_info(rs.camera_info.name)}")
                        time.sleep(2)
                    except Exception as e:
                        print(f"[{self.name}] 重置设备失败: {e}")
                
                print(f"[{self.name}] 设备重置尝试完成")
                
            except ImportError:
                print(f"[{self.name}] pyrealsense2 未安装，跳过设备重置")
        
        except Exception as e:
            print(f"[{self.name}] 重置 USB 设备失败: {e}")
    
    def _get_first_device(self, context) -> Optional[Any]:
        """返回第一个 RealSense 设备"""
        devices = context.query_devices()
        if len(devices) == 0:
            return None
        return devices[0]