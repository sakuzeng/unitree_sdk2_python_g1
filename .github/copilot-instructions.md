# Copilot 指令

本文件定义项目的技术栈、编码规范和主题约束，确保 Copilot 生成的代码符合项目需求。

## 项目概述

这是一个基于 Unitree G1 机器人的 Python 开发项目，集成了运动控制、激光雷达、摄像头和语音交互功能。

### 主要技术栈
- **主要语言**: Python 3.8+
- **机器人 SDK**: unitree_sdk2_python
- **激光雷达**: Livox SDK2, Mid-360 LiDAR
- **摄像头**: Intel RealSense, OpenCV
- **点云处理**: Open3D, KISS-ICP
- **通信协议**: DDS (CycloneDX)
- **构建系统**: CMake, ROS

### 关键依赖
- `unitree_sdk2_python` - 宇树机器人控制
- `cyclonedx==0.10.2` - DDS 通信
- `opencv-python` - 图像处理
- `open3d` - 点云处理
- `numpy` - 数值计算
- `livox_ros_driver2` - 激光雷达驱动

## 编码规范

### Python 编码风格
- 使用 Tab 进行缩进，不使用空格
- 行结束符使用 Unix 风格 (LF)
- 遵循 PEP 8 编码规范（除缩进外）
- 函数名使用 snake_case
- 类名使用 PascalCase
- 常量使用 UPPER_SNAKE_CASE

### 文件命名约定
- Python 脚本：`snake_case.py`
- 配置文件：保持原有格式（如 `MID360_config.json`）
- 文档文件：`README.md`, `readme.md`

### 项目结构约定
```
├── unitree_sdk_python/     # 机器人 SDK
├── docs/                   # 文档
├── tools/                  # 工具集
├── Livox-SDK2/            # 激光雷达 SDK
├── librealsense/          # RealSense SDK
└── *.py                   # 主要功能脚本
```

## 代码生成指导

### 机器人控制代码
- 使用 `unitree_sdk2_python` 进行机器人控制
- 网络接口参数使用 `enp2s0` 作为默认值，提示用户修改
- 高级控制使用 sport_mode 服务
- 低级控制需要先关闭 sport_mode

### 激光雷达代码
- 使用 Livox SDK2 处理 Mid-360 LiDAR
- 配置文件路径：`~/livox_cfg/MID360_config.json`
- 默认 IP 配置：雷达 `192.168.123.120`，主机 `192.168.123.164`
- 点云数据使用 Open3D 处理

### 摄像头代码
- Intel RealSense 使用 `librealsense` 库
- OpenCV 用于图像处理和显示
- 前置摄像头示例需要图形界面环境

### 网络配置
- DDS 域 ID 使用默认值
- 激光雷达端口：56301 (点云), 56401 (IMU)
- 机器人通信端口按 SDK 默认配置

## 代码模板

### 基础机器人控制模板
```python
#!/usr/bin/env python3
import sys
from unitree_sdk2_python.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2_python.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2_python.idl.unitree_go.msg.dds_ import SportModeCmd_

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <network_interface>")
        print("Example: python3 script.py enp2s0")
        sys.exit(-1)
    
    network_interface = sys.argv[1]
    # 机器人控制代码
    
if __name__ == "__main__":
    main()
```

### 激光雷达处理模板
```python
#!/usr/bin/env python3
import open3d as o3d
import numpy as np

def process_lidar_data():
    """处理激光雷达点云数据"""
    pass

def visualize_pointcloud(points):
    """可视化点云数据"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
```

### 摄像头集成模板
```python
#!/usr/bin/env python3
import cv2
import pyrealsense2 as rs
import numpy as np

def initialize_camera():
    """初始化 RealSense 摄像头"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    return pipeline

def process_frame(pipeline):
    """处理摄像头帧数据"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None, None
    
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    
    return color_image, depth_image
```

## 文档规范

### 注释规范
- 函数/类使用 docstring 说明功能
```python
def control_robot(cmd: str) -> bool:
    """
    控制机器人执行指定命令
    
    Args:
        cmd (str): 控制命令
        
    Returns:
        bool: 执行成功返回 True，失败返回 False
    """
    pass
```

### 示例代码说明
- 每个示例脚本包含使用说明
- 网络接口参数需要提示用户修改
- 安全提示（如关闭 sport_mode）

### 配置文件规范 (JSON)
```json
{
    "lidar_config": {
        "ip_address": "192.168.123.120",
        "data_port": 56301,
        "imu_port": 56401,
        "point_cloud_coordinate": 0,
        "imu_coordinate": 0
    },
    "host_config": {
        "ip_address": "192.168.123.164",
        "data_port": 56301,
        "imu_port": 56401
    }
}
```

## 错误处理

### 常见问题处理
```python
try:
    # 机器人控制代码
    pass
except ConnectionError as e:
    print(f"网络连接错误: {e}")
    print("请检查网络接口配置")
except ImportError as e:
    print(f"依赖库缺失: {e}")
    print("请安装必要的依赖包")
except Exception as e:
    print(f"未知错误: {e}")
```

### 调试建议
- 使用 `print()` 进行调试输出
- 网络连接检查使用 `ping` 和 `tcpdump`
- 激光雷达连接验证使用 `rostopic hz`

## 安全注意事项

### 低级控制安全提示
```python
# 低级控制前必须关闭高级运动服务
SAFETY_KP = 10.0  # 安全 KP 参数
SAFETY_KD = 1.0   # 安全 KD 参数

def safe_motor_control():
    """安全的电机控制示例"""
    print("警告: 确保机器人处于安全测试环境")
    print("低级控制将覆盖高级运动服务")
    
    # 使用安全参数
    kp = SAFETY_KP
    kd = SAFETY_KD
```

### 摄像头程序退出机制
```python
def camera_loop():
    """摄像头主循环，ESC 键退出"""
    while True:
        # 处理图像
        cv2.imshow('Camera', image)
        
        # ESC 键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
```

## 网络配置验证

### 激光雷达连接验证
```python
import socket

def check_lidar_connection(ip="192.168.123.120", port=56301):
    """检查激光雷达网络连接"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5)
        sock.connect((ip, port))
        print(f"激光雷达连接正常: {ip}:{port}")
        return True
    except Exception as e:
        print(f"激光雷达连接失败: {e}")
        return False
    finally:
        sock.close()
```

### 机器人网络接口检查
```python
import subprocess

def check_network_interface(interface="enp2s0"):
    """检查网络接口状态"""
    try:
        result = subprocess.run(['ip', 'addr', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"网络接口 {interface} 正常")
            return True
        else:
            print(f"网络接口 {interface} 不存在或未激活")
            return False
    except Exception as e:
        print(f"检查网络接口时出错: {e}")
        return False
```

## 最佳实践

### 资源管理
```python
import contextlib

@contextlib.contextmanager
def robot_connection(network_interface):
    """机器人连接上下文管理器"""
    try:
        # 初始化连接
        yield connection
    finally:
        # 清理资源
        pass
```

### 配置文件处理
```python
import json
from pathlib import Path

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)
```