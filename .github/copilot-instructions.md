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

### 项目结构约定
```
├── unitree_sdk_python/     # 机器人 SDK
├── docs/                   # 文档
│   ├── requirements.md     # 总体需求
│   ├── slam_autonomous_navigation_requirements.md
│   ├── audio_interaction_requirements.md
│   ├── g1_arm_control_requirements.md
│   └── dex3_control_requirements.md
├── src/                    # 源代码
│   ├── slam/              # SLAM相关
│   ├── navigation/        # 导航规划
│   ├── audio/             # 音频处理
│   ├── sensors/           # 传感器
│   └── control/           # 控制模块
├── config/                # 配置文件
├── tools/                 # 工具集
├── Livox-SDK2/            # 激光雷达 SDK
├── librealsense/          # RealSense SDK
└── *.py                   # 主要功能脚本
```

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
        print("Example: python3 script.py eth0")
        sys.exit(-1)
    
    network_interface = sys.argv[1]
    # 机器人控制代码
    
if __name__ == "__main__":
    main()
```

### SLAM导航控制模板
```python
#!/usr/bin/env python3
import numpy as np
from kiss_icp.pipeline import OdometryPipeline
from unitree_sdk2_python.core.channel import ChannelPublisher

class SLAMNavigationController:
    """SLAM导航控制器"""
    
    def __init__(self, network_interface: str):
        """
        初始化SLAM导航控制器
        
        Args:
            network_interface (str): 网络接口名称
        """
        self.network_interface = network_interface
        self.slam_pipeline = OdometryPipeline()
        self.current_pose = np.eye(4)
        self.target_pose = None
        
    def update_slam(self, pointcloud: np.ndarray):
        """
        更新SLAM状态
        
        Args:
            pointcloud (np.ndarray): 输入点云数据
        """
        self.current_pose = self.slam_pipeline.register_frame(pointcloud)
        
    def plan_path(self, target: np.ndarray):
        """
        规划路径到目标点
        
        Args:
            target (np.ndarray): 目标位置 [x, y, theta]
        """
        # 路径规划逻辑
        pass
        
    def navigate_to_target(self):
        """执行导航到目标点"""
        # 导航控制逻辑
        pass
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

### 音频交互系统模板
```python
#!/usr/bin/env python3
import pyaudio
import numpy as np
import librosa
from typing import Optional, Callable

class AudioInteractionSystem:
    """音频交互系统"""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        """
        初始化音频交互系统
        
        Args:
            sample_rate (int): 采样率
            chunk_size (int): 音频块大小
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        
    def initialize_audio_devices(self):
        """初始化G1音频设备"""
        # 查找G1机器人的音频设备
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            devices.append(device_info)
        return devices
        
    def start_recording(self, callback: Optional[Callable] = None):
        """
        开始录音
        
        Args:
            callback: 音频数据回调函数
        """
        def audio_callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            if callback:
                callback(audio_data)
            return (in_data, pyaudio.paContinue)
            
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        self.stream.start_stream()
        self.is_recording = True
        
    def stop_recording(self):
        """停止录音"""
        if hasattr(self, 'stream') and self.is_recording:
            self.stream.stop_stream()
            self.stream.close()
            self.is_recording = False
            
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        音频预处理
        
        Args:
            audio_data (np.ndarray): 原始音频数据
            
        Returns:
            np.ndarray: 预处理后的音频数据
        """
        # 降噪
        audio_data = librosa.effects.preemphasis(audio_data)
        
        # 归一化
        audio_data = librosa.util.normalize(audio_data)
        
        return audio_data
        
    def play_audio(self, audio_data: np.ndarray):
        """
        播放音频
        
        Args:
            audio_data (np.ndarray): 要播放的音频数据
        """
        stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True
        )
        stream.write(audio_data.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()
        
    def __del__(self):
        """清理资源"""
        if self.is_recording:
            self.stop_recording()
        self.audio.terminate()

class SenseVoiceASR:
    """SenseVoiceSmall语音识别"""
    
    def __init__(self, model_path: str = "models/SenseVoiceSmall"):
        """
        初始化SenseVoice语音识别
        
        Args:
            model_path (str): 模型路径
        """
        self.model_path = model_path
        # 加载SenseVoiceSmall模型
        self.model = self._load_model()
        
    def _load_model(self):
        """加载SenseVoiceSmall模型"""
        # TODO: 实现模型加载逻辑
        pass
        
    def recognize(self, audio_data: np.ndarray) -> tuple[str, float]:
        """
        语音识别
        
        Args:
            audio_data (np.ndarray): 音频数据
            
        Returns:
            tuple[str, float]: 识别文本和置信度
        """
        # TODO: 实现语音识别逻辑
        text = "识别结果"
        confidence = 0.95
        return text, confidence

class TTSEngine:
    """文本转语音引擎"""
    
    def __init__(self, engine_type: str = "local"):
        """
        初始化TTS引擎
        
        Args:
            engine_type (str): 引擎类型 ("local", "azure", "google")
        """
        self.engine_type = engine_type
        self.engine = self._initialize_engine()
        
    def _initialize_engine(self):
        """初始化TTS引擎"""
        if self.engine_type == "local":
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # 语速
            engine.setProperty('volume', 0.8)  # 音量
            return engine
        else:
            # TODO: 实现云端TTS集成
            pass
            
    def synthesize(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        文本转语音
        
        Args:
            text (str): 要合成的文本
            output_path (Optional[str]): 输出文件路径
            
        Returns:
            np.ndarray: 音频数据
        """
        if self.engine_type == "local":
            if output_path:
                self.engine.save_to_file(text, output_path)
                self.engine.runAndWait()
                # 加载生成的音频文件
                audio_data, _ = librosa.load(output_path, sr=16000)
                return audio_data
            else:
                # 直接播放
                self.engine.say(text)
                self.engine.runAndWait()
                return np.array([])  # 返回空数组，因为是直接播放
```

### 语音控制集成模板
```python
#!/usr/bin/env python3
from typing import Dict, Callable
import re

class VoiceCommandProcessor:
    """语音控制指令处理器"""
    
    def __init__(self):
        """初始化语音控制处理器"""
        self.command_patterns = {
            'move_forward': [r'前进', r'向前', r'go forward', r'move forward'],
            'move_backward': [r'后退', r'向后', r'go back', r'move back'],
            'turn_left': [r'左转', r'向左转', r'turn left'],
            'turn_right': [r'右转', r'向右转', r'turn right'],
            'stop': [r'停止', r'停下', r'stop', r'halt'],
            'navigate_to': [r'导航到', r'去', r'navigate to', r'go to'],
            'arm_up': [r'抬起手臂', r'手臂向上', r'raise arm'],
            'arm_down': [r'放下手臂', r'手臂向下', r'lower arm'],
            'grasp': [r'抓取', r'抓住', r'grab', r'grasp'],
            'release': [r'释放', r'松开', r'release', r'let go']
        }
        self.command_handlers: Dict[str, Callable] = {}
        
    def register_command_handler(self, command: str, handler: Callable):
        """
        注册指令处理函数
        
        Args:
            command (str): 指令名称
            handler (Callable): 处理函数
        """
        self.command_handlers[command] = handler
        
    def parse_command(self, text: str) -> tuple[str, dict]:
        """
        解析语音指令
        
        Args:
            text (str): 识别的文本
            
        Returns:
            tuple[str, dict]: 指令类型和参数
        """
        text = text.lower().strip()
        
        for command, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # 提取参数
                    params = self._extract_parameters(command, text)
                    return command, params
                    
        return 'unknown', {}
        
    def _extract_parameters(self, command: str, text: str) -> dict:
        """
        从文本中提取命令参数
        
        Args:
            command (str): 命令类型
            text (str): 原始文本
            
        Returns:
            dict: 提取的参数
        """
        params = {}
        
        if command == 'navigate_to':
            # 提取目标位置
            location_patterns = [r'客厅', r'厨房', r'卧室', r'bathroom', r'kitchen', r'living room']
            for pattern in location_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    params['target'] = pattern
                    break
                    
        elif command in ['move_forward', 'move_backward']:
            # 提取距离参数
            distance_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:米|m|meter)', text)
            if distance_match:
                params['distance'] = float(distance_match.group(1))
            else:
                params['distance'] = 1.0  # 默认1米
                
        return params
        
    def execute_command(self, command: str, params: dict) -> bool:
        """
        执行语音指令
        
        Args:
            command (str): 指令类型
            params (dict): 指令参数
            
        Returns:
            bool: 执行是否成功
        """
        if command in self.command_handlers:
            try:
                self.command_handlers[command](params)
                return True
            except Exception as e:
                print(f"执行指令 {command} 时出错: {e}")
                return False
        else:
            print(f"未知指令: {command}")
            return False
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
except OSError as e:
    print(f"音频设备错误: {e}")
    print("请检查音频设备配置")
except Exception as e:
    print(f"未知错误: {e}")
```

### 调试建议
- 使用 `print()` 进行调试输出
- 网络连接检查使用 `ping` 和 `tcpdump`
- 激光雷达连接验证使用 `rostopic hz`
- 音频设备检查使用 `arecord -l` 和 `aplay -l`

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

### 手臂姿态变换安全指导
```python
import time

def safe_arm_pose_transition(from_pose, to_pose, duration=3.0):
    """
    安全的手臂姿态变换
    
    Args:
        from_pose: 起始姿态
        to_pose: 目标姿态
        duration: 变换持续时间（秒），默认3秒，建议不少于2秒
    
    注意事项:
        - 手臂姿态变换必须缓慢进行，避免突然运动造成机械损伤
        - 建议变换时间不少于2-3秒
        - 大幅度姿态变换应分解为多个小步骤
        - 监控关节角度变化速度，避免超出安全范围
    """
    print("警告: 手臂姿态变换中，请确保周围环境安全")
    print(f"姿态变换将持续 {duration} 秒")
    
    # 计算插值步数（50Hz控制频率）
    steps = int(duration * 50)
    
    for i in range(steps):
        # 线性插值计算中间姿态
        alpha = i / steps
        current_pose = interpolate_pose(from_pose, to_pose, alpha)
        
        # 发送控制命令
        send_arm_command(current_pose)
        
        # 20ms 控制周期
        time.sleep(0.02)

def interpolate_pose(pose1, pose2, alpha):
    """
    姿态线性插值
    
    Args:
        pose1: 起始姿态
        pose2: 目标姿态
        alpha: 插值系数 [0, 1]
    
    Returns:
        插值后的姿态
    """
    # 确保平滑过渡
    return pose1 + alpha * (pose2 - pose1)

# 手臂控制安全常量
ARM_MAX_ANGULAR_VELOCITY = 0.5  # rad/s，最大角速度限制
ARM_TRANSITION_MIN_TIME = 2.0   # 秒，最小变换时间
ARM_SAFETY_MARGIN = 0.1         # 安全余量

def validate_arm_motion(current_joints, target_joints, dt):
    """
    验证手臂运动是否安全
    
    Args:
        current_joints: 当前关节角度
        target_joints: 目标关节角度
        dt: 时间间隔
    
    Returns:
        bool: 运动是否安全
    """
    for i, (current, target) in enumerate(zip(current_joints, target_joints)):
        angular_velocity = abs(target - current) / dt
        if angular_velocity > ARM_MAX_ANGULAR_VELOCITY:
            print(f"警告: 关节 {i} 角速度 {angular_velocity:.2f} rad/s 超出安全限制")
            return False
    return True
```

### SLAM导航安全指导
```python
def safe_navigation_control():
    """安全的导航控制指导"""
    print("警告: SLAM导航模式启动")
    print("确保导航区域无人员和重要物品")
    print("准备随时使用紧急停止")
    
    # 安全参数
    MAX_LINEAR_VELOCITY = 0.8   # m/s，最大线速度
    MAX_ANGULAR_VELOCITY = 1.0  # rad/s，最大角速度
    SAFETY_DISTANCE = 0.3       # m，安全距离
    
def validate_navigation_command(linear_vel, angular_vel):
    """验证导航指令是否安全"""
    if abs(linear_vel) > MAX_LINEAR_VELOCITY:
        print(f"警告: 线速度 {linear_vel:.2f} m/s 超出安全限制")
        return False
    if abs(angular_vel) > MAX_ANGULAR_VELOCITY:
        print(f"警告: 角速度 {angular_vel:.2f} rad/s 超出安全限制")
        return False
    return True
```

### 音频系统安全指导
```python
def safe_audio_interaction():
    """安全的音频交互指导"""
    print("音频交互系统启动")
    print("确保周围环境相对安静以提高识别准确率")
    print("紧急情况下可通过特定关键词或物理按钮停止系统")
    
    # 音频安全参数
    MAX_VOLUME_LEVEL = 0.8      # 最大音量
    NOISE_THRESHOLD = 0.1       # 噪声阈值
    CONFIDENCE_THRESHOLD = 0.7  # 置信度阈值
    
def validate_audio_command(text, confidence):
    """验证语音指令是否可信"""
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"警告: 语音识别置信度 {confidence:.2f} 过低，忽略指令")
        return False
    
    dangerous_commands = ['紧急停止', 'emergency stop', '立即停止']
    if any(cmd in text.lower() for cmd in dangerous_commands):
        print("检测到紧急停止指令，立即执行")
        return True
        
    return True
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

def check_network_interface(interface="eth0"):
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

### 音频设备验证
```python
import pyaudio

def check_audio_devices():
    """检查音频设备状态"""
    audio = pyaudio.PyAudio()
    
    print("可用音频输入设备:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"  {i}: {device_info['name']}")
    
    print("可用音频输出设备:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxOutputChannels'] > 0:
            print(f"  {i}: {device_info['name']}")
    
    audio.terminate()

def test_microphone():
    """测试麦克风功能"""
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("录音测试 3 秒...")
        for _ in range(int(16000 / 1024 * 3)):
            data = stream.read(1024)
            # 检查是否有音频输入
            if max(np.frombuffer(data, dtype=np.float32)) > 0.01:
                print("麦克风工作正常")
                break
        else:
            print("麦克风可能有问题，未检测到音频输入")
            
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"麦克风测试失败: {e}")
    finally:
        audio.terminate()
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

@contextlib.contextmanager
def audio_system():
    """音频系统上下文管理器"""
    audio_manager = None
    try:
        audio_manager = AudioInteractionSystem()
        audio_manager.initialize_audio_devices()
        yield audio_manager
    finally:
        if audio_manager:
            audio_manager.stop_recording()
```

### 配置文件处理
```python
import json
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")

def save_config(config: dict, config_path: str):
    """保存配置文件"""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.endswith('.json'):
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent='\t', ensure_ascii=False)
    elif config_path.endswith(('.yaml', '.yml')):
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
```

### 多线程音频处理
```python
import threading
import queue
import time

class AudioProcessor:
    """多线程音频处理器"""
    
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=100)
        self.is_running = False
        self.processing_thread = None
        
    def start_processing(self):
        """开始音频处理"""
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.start()
        
    def stop_processing(self):
        """停止音频处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def add_audio_data(self, audio_data):
        """添加音频数据到处理队列"""
        try:
            self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            print("警告: 音频处理队列已满，丢弃数据")
            
    def _process_audio_loop(self):
        """音频处理主循环"""
        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                self._process_single_audio(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音频处理错误: {e}")
                
    def _process_single_audio(self, audio_data):
        """处理单个音频数据"""
        # 实现具体的音频处理逻辑
        pass
```

### 系统集成最佳实践
```python
class RobotSystem:
    """机器人系统集成"""
    
    def __init__(self, config_path: str):
        """
        初始化机器人系统
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config = load_config(config_path)
        self.slam_controller = None
        self.audio_system = None
        self.voice_commander = None
        self.is_initialized = False
        
    def initialize(self):
        """初始化所有子系统"""
        try:
            # 初始化SLAM系统
            if self.config.get('slam_enabled', True):
                self.slam_controller = SLAMNavigationController(
                    self.config['network_interface']
                )
                
            # 初始化音频系统
            if self.config.get('audio_enabled', True):
                self.audio_system = AudioInteractionSystem(
                    sample_rate=self.config['audio']['sample_rate'],
                    chunk_size=self.config['audio']['chunk_size']
                )
                
            # 初始化语音控制
            if self.config.get('voice_control_enabled', True):
                self.voice_commander = VoiceCommandProcessor()
                self._register_voice_commands()
                
            self.is_initialized = True
            print("机器人系统初始化完成")
            
        except Exception as e:
            print(f"系统初始化失败: {e}")
            self.cleanup()
            
    def _register_voice_commands(self):
        """注册语音控制指令"""
        if self.voice_commander and self.slam_controller:
            self.voice_commander.register_command_handler(
                'move_forward', 
                lambda params: self.slam_controller.move_forward(params.get('distance', 1.0))
            )
            self.voice_commander.register_command_handler(
                'navigate_to',
                lambda params: self.slam_controller.navigate_to(params.get('target'))
            )
            
    def run(self):
        """运行机器人系统"""
        if not self.is_initialized:
            print("系统未初始化，请先调用 initialize()")
            return
            
        try:
            print("机器人系统开始运行...")
            while True:
                # 系统主循环
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("收到停止信号，正在关闭系统...")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """清理系统资源"""
        if self.audio_system:
            self.audio_system.stop_recording()
        print("系统资源清理完成")
```