# Copilot 指令

本文件定义项目的技术栈、编码规范和主题约束，确保 Copilot 生成的代码符合项目需求。

## 项目概述

这是一个基于 Unitree G1 机器人的 Python 开发项目，集成了运动控制、激光雷达、摄像头、语音交互和SLAM自主导航功能。

### 主要技术栈
- **主要语言**: Python 3.8+
- **机器人 SDK**: unitree_sdk2_python
- **激光雷达**: Livox SDK2, Mid-360 LiDAR
- **摄像头**: Intel RealSense, OpenCV
- **点云处理**: Open3D, KISS-ICP
- **语音识别**: SenseVoiceSmall
- **语音合成**: pyttsx3, gTTS, Azure TTS
- **通信协议**: DDS (CycloneDX)
- **构建系统**: CMake, ROS

### 关键依赖
- `unitree_sdk2_python` - 宇树机器人控制
- `cyclonedx==0.10.2` - DDS 通信
- `opencv-python` - 图像处理
- `open3d` - 点云处理
- `numpy` - 数值计算
- `livox_ros_driver2` - 激光雷达驱动
- `sensevoice` - SenseVoiceSmall语音识别
- `pyaudio` - 音频输入输出
- `librosa` - 音频处理
- `pyttsx3` - 本地TTS引擎

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
- 配置文件：保持原有格式（如 `MID360_config.json`, `audio_config.yaml`）
- 文档文件：`README.md`

## 代码生成指导

### 参考项目文档
在生成或修改代码时，请务必参考 `docs/` 目录下的以下关键文档，以确保符合项目设计和需求：
- **`docs/contents.md`**: 项目的总体结构、功能模块和技术栈说明。
- **`docs/requirements.md`**: 详细的功能性和非功能性需求。
- **`docs/slam_autonomous_navigation_requirements.md`**: SLAM自主导航系统专项需求。
- **`docs/audio_interaction_requirements.md`**: 音频交互系统专项需求。
- **`docs/g1_arm_control_requirements.md`**: G1手臂控制系统专项需求。
- **`docs/dex3_control_requirements.md`**: Dex3灵巧手控制系统专项需求。
- **`docs/g1_edu.md`**: G1 EDU 版本的特定说明和开发指南。
- **`unitree_sdk2_python.md`**: 详解 `unitree_sdk2_python` SDK 的结构和使用方法。

### 机器人控制代码
- 使用 `unitree_sdk2_python` 进行机器人控制
- 网络接口参数使用 `eth0` 作为默认值，提示用户修改
- 高级控制使用 sport_mode 服务
- 低级控制需要先关闭 sport_mode
- SLAM导航控制需要集成路径规划和避障

### 激光雷达代码
- 使用 Livox SDK2 处理 Mid-360 LiDAR
- 配置文件路径：`~/livox_cfg/MID360_config.json`
- 默认 IP 配置：雷达 `192.168.123.120`，主机 `192.168.123.164`
- 点云数据使用 Open3D 处理
- SLAM集成使用 KISS-ICP

### 摄像头代码
- Intel RealSense 使用 `librealsense` 库
- OpenCV 用于图像处理和显示
- 前置摄像头示例需要图形界面环境

### 音频交互代码
- 使用 SenseVoiceSmall 进行语音识别
- 集成 G1 机器人音频硬件接口
- 支持本地和云端 TTS 服务
- 音频预处理包括降噪、回声消除、AGC

### 网络配置
- DDS 域 ID 使用默认值
- 激光雷达端口：56301 (点云), 56401 (IMU)
- 机器人通信端口按 SDK 默认配置
- 音频流处理使用本地接口

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
- 音频设备配置说明

### 配置文件规范 (JSON/YAML)
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
    },
    "audio_config": {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_size": 1024,
        "device_name": "G1_microphone_array"
    },
    "slam_config": {
        "max_range": 30.0,
        "voxel_size": 0.1,
        "adaptive_threshold": true
    }
}
```

## 错误处理

### 调试建议
- 使用 `print()` 进行调试输出
- 网络连接检查使用 `ping` 和 `tcpdump`