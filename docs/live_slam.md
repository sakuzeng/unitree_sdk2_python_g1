# `live_slam.py` 代码分析文档

## 文件概述

`live_slam.py` 是一个专为 Livox MID-360 激光雷达设计的实时 SLAM（同步定位与地图构建）演示程序。该文件实现了基于 KISS-ICP 算法的实时点云处理和位姿估计功能，并提供了 Open3D 可视化界面来显示构建的点云地图和机器人轨迹。

### 主要用途和应用场景
- **实时 SLAM 建图**：使用 Livox MID-360 激光雷达进行环境感知和地图构建
- **机器人导航**：为 Unitree G1 机器人提供定位和地图服务
- **环境监测**：实时显示周围环境的 3D 点云地图
- **算法验证**：测试和验证 SLAM 算法性能

### 与项目整体的关系
该文件是 Unitree G1 机器人项目的核心 SLAM 模块，与以下组件紧密集成：
- `livox_python.py` / `livox2_python.py`：激光雷达数据采集
- `run_g1_stack.py`：多传感器融合系统
- `mid360_config.json`：激光雷达配置文件

## 依赖关系

### 标准库
```python
import signal          # 信号处理（Ctrl-C 中断）
import time           # 时间控制和延迟
import math           # 数学计算（三角函数）
import os             # 环境变量读取
from pathlib import Path     # 路径处理
from typing import Optional, Dict, Any  # 类型注解
```

### 第三方库
```python
import numpy as np          # 数值计算和数组操作 (>=1.19.0)
import open3d as o3d       # 3D 数据处理和可视化 (>=0.16.0)
from kiss_icp.pipeline import KissICP    # SLAM 算法核心 (>=1.2.0)
from kiss_icp.config import load_config  # 配置加载
```

### 项目内部依赖
```python
from livox2_python import Livox2 as _Livox  # Livox SDK2 接口（优先）
from livox_python import Livox as _Livox    # Livox SDK1 接口（备用）
```

### 系统级依赖
- **Livox SDK2**：激光雷达驱动程序和通信库
- **网络配置**：UDP 端口 56301（点云数据）、56401（IMU 数据）
- **硬件要求**：Livox MID-360 激光雷达
- **操作系统**：Linux（推荐 Ubuntu 20.04+）

## 代码结构分析

### 主要类

#### 1. `_Viewer` 类
Open3D 可视化器，负责显示点云地图和机器人位姿：
```python
class _Viewer:
    """Open3D visualiser that shows both the map *and* the current pose."""
    
    def __init__(self):           # 初始化可视化窗口
    def push(self, xyz, pose):    # 接收新的点云和位姿数据
    def tick(self) -> bool:       # 更新显示（主线程调用）
    def close(self):              # 关闭可视化窗口
```

#### 2. `LiveSLAMDemo` 类
核心 SLAM 处理类，继承自 Livox 基类：
```python
class LiveSLAMDemo(_Livox):
    def __init__(self):                    # 初始化 SLAM 算法和可视化器
    def handle_points(self, xyz):          # 处理激光雷达点云数据
    def shutdown(self):                    # 优雅关闭所有资源
```

### 核心函数

#### 1. `main()` 函数
程序入口点，负责启动 SLAM 演示：
```python
def main():
    demo = LiveSLAMDemo()
    # 设置 Ctrl-C 信号处理
    # 主循环：更新可视化并处理用户输入
    # 优雅关闭
```

### 关键变量和常量

#### 环境变量配置
```python
LIVOX_MOUNT = "upside_down"     # 激光雷达安装方向
LIVOX_PRESET = "indoor"         # 预设模式（indoor/outdoor）
LIDAR_TILT_DEG = "0"           # 倾斜角度校正
LIDAR_TILT_AXIS = "y"          # 倾斜轴向
LIDAR_SELF_FILTER_RADIUS = "0.30"  # 自身过滤半径
LIDAR_SELF_FILTER_Z = "0.24"       # 垂直过滤范围
```

#### 预设参数
```python
_PRESETS = {
    "indoor": {
        "frame_time": 0.35,           # 帧聚合时间
        "frame_packets": 200,         # 每帧数据包数量
        "voxel_size": 0.4,           # 体素大小
        "max_range": 30.0,           # 最大检测距离
        "downsample_limit": 5_000_000, # 可视化点数限制
        "min_motion": 0.03,          # 最小运动阈值
        "conv_criterion": 5e-5,       # 收敛准则
        "max_iters": 800,            # 最大迭代次数
    },
    "outdoor": {
        # 户外场景优化参数
    }
}
```

### 模块导入结构

程序采用渐进式导入策略，确保兼容性：

1. **KISS-ICP 导入**：尝试新版本路径，失败时回退到旧版本
2. **Livox SDK 导入**：优先使用 SDK2，不可用时使用 SDK1
3. **错误处理**：记录所有导入错误并提供详细的安装指导

## 核心功能说明

### 主要算法实现

#### 1. SLAM 算法（KISS-ICP）
```python
# 初始化 SLAM 配置
cfg = load_config(config_file=None, max_range=_P["max_range"])
cfg.adaptive_threshold.min_motion_th = _P["min_motion"]
cfg.registration.convergence_criterion = _P["conv_criterion"]
cfg.registration.max_num_iterations = _P["max_iters"]

# 创建 SLAM 实例
self._slam = KissICP(cfg)
```

#### 2. 坐标变换和校正
支持两种独立的坐标校正：
- **倒置安装校正**：180° 绕 X 轴旋转
- **固定倾斜校正**：补偿机器人头部倾斜

```python
# 构建校正矩阵
_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)
_R_TOTAL = _R_TILT @ _R_FLIP
```

### 数据处理流程

1. **点云接收** → `handle_points()` 接收原始点云数据
2. **自身过滤** → 移除机器人自身反射点
3. **SLAM 处理** → KISS-ICP 算法进行位姿估计和地图更新
4. **坐标校正** → 应用安装方向和倾斜校正
5. **可视化更新** → 更新 Open3D 显示

### 错误处理机制

#### 依赖检查和回退
```python
try:
    from kiss_icp.pipeline import KissICP
except Exception as e:
    _IMPORT_ERRORS.append(e)
    try:
        from kiss_icp.pybind import KissICP  # 回退到旧版本
    except Exception as e:
        # 提供详细错误信息和解决方案
```

#### 版本兼容性处理
```python
try:
    self._slam.register_frame(xyz)  # 旧版本 API
except TypeError:
    # 新版本需要时间戳参数
    ts = np.linspace(0.0, period, num=xyz.shape[0])
    self._slam.register_frame(xyz, ts)
```

### 线程安全设计

- **数据传递**：使用简单的变量赋值而非队列（只关心最新数据）
- **内存管理**：及时清空已处理的数据引用
- **状态同步**：最小化线程间共享状态

## 使用方法

### 命令行参数
```bash
python live_slam.py
```
无需命令行参数，所有配置通过环境变量控制。

### 环境变量配置

#### 基础配置
```bash
# 激光雷达安装方向（默认：upside_down）
export LIVOX_MOUNT=upside_down  # 或 normal

# 场景预设（默认：indoor）
export LIVOX_PRESET=indoor      # 或 outdoor
```

#### 高级校正
```bash
# 倾斜校正（默认：0度无校正）
export LIDAR_TILT_DEG=15        # 倾斜角度
export LIDAR_TILT_AXIS=y        # 倾斜轴向（x/y/z）

# 自身过滤调节
export LIDAR_SELF_FILTER_RADIUS=0.30  # 水平过滤半径（米）
export LIDAR_SELF_FILTER_Z=0.24       # 垂直过滤范围（米）
```

### 前置条件和准备工作

#### 1. 安装依赖包
```bash
pip install -r requirements.txt
# 包含：numpy, open3d>=0.16.0, kiss-icp
```

#### 2. 编译安装 Livox SDK2
```bash
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) && sudo make install
```

#### 3. 网络配置
```bash
# 设置网络接口 IP
sudo ip addr add 192.168.123.222/24 dev enp2s0

# 验证连接
ping 192.168.123.120  # 激光雷达 IP
```

#### 4. 配置文件
确保 `mid360_config.json` 存在并正确配置：
```json
{
    "lidar_config": {
        "ip_address": "192.168.123.120",
        "data_port": 56301,
        "imu_port": 56401
    },
    "host_config": {
        "ip_address": "192.168.123.222",
        "data_port": 56301,
        "imu_port": 56401
    }
}
```

### 运行示例

#### 基础运行
```bash
python live_slam.py
```

#### 室外场景
```bash
LIVOX_PRESET=outdoor python live_slam.py
```

#### 正常安装（非倒置）
```bash
LIVOX_MOUNT=normal python live_slam.py
```

#### 头部倾斜校正
```bash
LIDAR_TILT_DEG=15 LIDAR_TILT_AXIS=y python live_slam.py
```

## 配置说明

### 预设模式说明

#### Indoor 模式（室内）
- **帧时间**：0.35秒（较长，提高稳定性）
- **体素大小**：0.4米（高精度）
- **最大距离**：30米
- **收敛准则**：5e-5（高精度）
- **适用场景**：办公室、家庭、实验室

#### Outdoor 模式（户外）
- **帧时间**：0.20秒（较短，提高响应）
- **体素大小**：1.0米（降低计算量）
- **最大距离**：120米
- **收敛准则**：1e-4（平衡精度和速度）
- **适用场景**：广场、停车场、道路

### 参数调优建议

#### 性能优化
```bash
# 减少可视化点数（提升性能）
# 修改代码中的 downsample_limit 参数

# 调整帧聚合参数
export LIVOX_PRESET=outdoor  # 更快的处理速度
```

#### 精度优化
```bash
# 提高处理精度
export LIVOX_PRESET=indoor   # 更高的地图精度

# 调整运动阈值
# 在代码中修改 min_motion 参数
```

## 技术要点

### 算法原理简述

#### KISS-ICP SLAM
- **核心思想**：Keep It Small and Simple - Iterative Closest Point
- **配准方法**：点到点 ICP 迭代配准
- **地图表示**：体素哈希地图（VoxelHashMap）
- **位姿估计**：增量式位姿计算

#### 坐标系变换
```python
# 应用校正变换矩阵
if _R_MOUNT is not None:
    cloud = (cloud @ _R_MOUNT[:3, :3].T)  # 点云校正
    pose = _R_MOUNT @ pose               # 位姿校正
```

### 性能优化策略

#### 1. 点云降采样
```python
if cloud.shape[0] > self._vis_max_points:
    step = int(cloud.shape[0] / self._vis_max_points) + 1
    cloud = cloud[::step]  # 等间隔采样
```

#### 2. 自身过滤
```python
# 移除机器人自身反射，提高 SLAM 质量
dist_xy = np.linalg.norm(xyz[:, :2], axis=1)
close = dist_xy < r_xy
near_plane = np.abs(xyz[:, 2]) < dz
mask = ~(close & near_plane)
```

#### 3. 内存管理
- **及时清理**：处理完数据后立即清空引用
- **避免复制**：使用视图和原地操作
- **预分配**：合理设置缓冲区大小

### 多线程/异步处理

#### 线程模型
- **主线程**：GUI 更新和用户交互
- **后台线程**：激光雷达数据接收和 SLAM 处理
- **数据传递**：通过实例变量进行最新数据传递

#### 同步机制
```python
def push(self, xyz: np.ndarray, pose: np.ndarray):
    """后台线程调用，传递最新数据"""
    self._latest_pts = xyz
    self._latest_pose = pose

def tick(self) -> bool:
    """主线程调用，更新显示"""
    if self._latest_pts is not None:
        # 更新点云显示
        self._latest_pts = None  # 清空引用
```

## 安全注意事项

### 硬件安全提醒

#### 激光雷达操作安全
- **功率设置**：确保激光功率在安全范围内
- **人员保护**：避免激光直射眼部
- **安装检查**：确认雷达安装牢固，避免高速运动时脱落

#### 机器人集成安全
- **校准验证**：运行前验证坐标校正参数正确性
- **运动限制**：SLAM 运行时限制机器人高速运动
- **紧急停止**：确保 Ctrl-C 能够立即停止程序

### 网络安全配置

#### 防火墙设置
```bash
# 开放 Livox 通信端口
sudo ufw allow 56301/udp  # 点云数据
sudo ufw allow 56401/udp  # IMU 数据
```

#### 网络隔离
- **专用网段**：使用 192.168.123.0/24 专用网段
- **访问控制**：限制激光雷达网络访问权限

### 数据安全处理

#### 敏感信息保护
- **环境信息**：点云数据可能包含敏感的环境信息
- **存储加密**：如需保存地图数据，考虑加密存储
- **访问控制**：限制点云数据的访问权限

### 异常处理机制

#### 优雅关闭
```python
def shutdown(self):
    super().shutdown()    # 关闭 Livox 连接
    self._viewer.close()  # 关闭可视化窗口
```

#### 错误恢复
- **连接丢失**：自动重连激光雷达
- **内存不足**：自动降低点云密度
- **计算超时**：跳过当前帧继续处理

## 故障排除

### 常见错误及解决方案

#### 1. 导入错误
```bash
# 错误：Could not import KISS-ICP
# 解决：升级或安装 KISS-ICP
pip install --upgrade kiss-icp
```

#### 2. 网络连接问题
```bash
# 错误：激光雷达无响应
# 检查网络配置
ping 192.168.123.120

# 检查端口占用
sudo netstat -ulnp | grep 56301
```

#### 3. 可视化问题
```bash
# 错误：Open3D 窗口无法显示
# 检查图形环境
echo $DISPLAY

# 安装图形支持
sudo apt install python3-opengl
```

#### 4. 内存不足
```bash
# 减少可视化点数
export downsample_limit=1000000  # 在代码中修改

# 监控内存使用
htop
```

### 调试方法建议

#### 1. 日志输出
```python
# 添加调试输出
print(f"[DEBUG] Points received: {xyz.shape[0]}")
print(f"[DEBUG] SLAM pose: {pose[:3, 3]}")  # 位置信息
```

#### 2. 数据验证
```python
# 检查点云数据有效性
if xyz.size > 0:
    print(f"Point cloud range: X[{xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}]")
    print(f"                  Y[{xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}]")
    print(f"                  Z[{xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f}]")
```

#### 3. 性能分析
```python
import time

start_time = time.time()
self._slam.register_frame(xyz)
end_time = time.time()
print(f"SLAM processing time: {(end_time - start_time)*1000:.2f} ms")
```

### 性能问题诊断

#### CPU 使用率过高
1. **降低帧率**：增加 `frame_time` 参数
2. **减少点数**：降低 `downsample_limit`
3. **简化算法**：减少 `max_iters` 迭代次数

#### 内存使用过多
1. **清理缓存**：定期清空地图缓存
2. **降低精度**：增大 `voxel_size`
3. **限制范围**：减小 `max_range`

### 网络连接问题

#### 数据包丢失
```bash
# 检查网络丢包率
sudo tcpdump -i enp2s0 udp port 56301 -c 100

# 优化网络缓冲区
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
```

#### 延迟过高
```bash
# 检查网络延迟
ping -i 0.01 192.168.123.120

# 优化网络接口
sudo ethtool -G enp2s0 rx 4096 tx 4096
```

## 扩展和定制

### 可配置参数

#### 运行时参数调整
可以通过修改 `_PRESETS` 字典来调整算法参数：

```python
# 自定义预设
custom_preset = {
    "frame_time": 0.25,        # 平衡速度和稳定性
    "frame_packets": 150,      # 适中的数据聚合
    "voxel_size": 0.6,        # 平衡精度和性能
    "max_range": 50.0,        # 中等检测距离
    "downsample_limit": 2_000_000,  # 适中的可视化点数
    "min_motion": 0.05,       # 适中的运动阈值
    "conv_criterion": 1e-4,    # 平衡精度和速度
    "max_iters": 600,         # 适中的迭代次数
}
```

#### 环境变量扩展
可以添加新的环境变量来控制更多参数：

```python
# 添加新的配置选项
SLAM_DEBUG = os.environ.get("SLAM_DEBUG", "false").lower() == "true"
SAVE_MAP = os.environ.get("SAVE_MAP", "false").lower() == "true"
MAP_SAVE_INTERVAL = int(os.environ.get("MAP_SAVE_INTERVAL", "100"))
```

### 扩展接口说明

#### 1. 数据处理钩子
```python
class LiveSLAMDemo(_Livox):
    def pre_process_points(self, xyz: np.ndarray) -> np.ndarray:
        """点云预处理钩子，子类可重写"""
        return xyz
    
    def post_process_map(self, cloud: np.ndarray) -> np.ndarray:
        """地图后处理钩子，子类可重写"""
        return cloud
```

#### 2. 可视化扩展
```python
class ExtendedViewer(_Viewer):
    def add_custom_geometry(self, geometry):
        """添加自定义几何体"""
        self._vis.add_geometry(geometry)
    
    def update_custom_info(self, info: str):
        """更新自定义信息显示"""
        # 实现文本信息显示
        pass
```

### 自定义回调函数

#### 地图保存回调
```python
def map_save_callback(self, cloud: np.ndarray, pose: np.ndarray, frame_count: int):
    """定期保存地图数据"""
    if frame_count % MAP_SAVE_INTERVAL == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"map_{timestamp}.ply"
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Map saved: {filename}")
```

#### 性能监控回调
```python
def performance_monitor_callback(self, processing_time: float, point_count: int):
    """监控处理性能"""
    fps = 1.0 / processing_time if processing_time > 0 else 0
    print(f"Processing: {processing_time*1000:.1f}ms, Points: {point_count}, FPS: {fps:.1f}")
```

### 插件机制

#### 插件接口定义
```python
class SLAMPlugin:
    """SLAM 插件基类"""
    
    def on_points_received(self, xyz: np.ndarray) -> np.ndarray:
        """点云接收时调用"""
        return xyz
    
    def on_map_updated(self, cloud: np.ndarray, pose: np.ndarray):
        """地图更新时调用"""
        pass
    
    def on_shutdown(self):
        """程序关闭时调用"""
        pass
```

#### 插件加载机制
```python
class PluginManager:
    def __init__(self):
        self.plugins = []
    
    def load_plugin(self, plugin_class):
        """加载插件"""
        plugin = plugin_class()
        self.plugins.append(plugin)
    
    def call_hook(self, hook_name: str, *args, **kwargs):
        """调用所有插件的钩子函数"""
        for plugin in self.plugins:
            if hasattr(plugin, hook_name):
                getattr(plugin, hook_name)(*args, **kwargs)
```

### 集成示例

#### 与机器人控制系统集成
```python
from unitree_sdk2_python.core.channel import ChannelPublisher
from unitree_sdk2_python.idl.unitree_go.msg.dds_ import SportModeCmd_

class RobotIntegratedSLAM(LiveSLAMDemo):
    def __init__(self, network_interface: str):
        super().__init__()
        self.cmd_pub = ChannelPublisher(
            "rt/sport_mode_cmd", SportModeCmd_
        )
        
    def handle_points(self, xyz: np.ndarray):
        super().handle_points(xyz)
        
        # 获取当前位姿并发送给机器人
        pose = self._slam.last_pose
        position = pose[:3, 3]
        
        # 发送位置信息给机器人控制系统
        self.publish_position_to_robot(position)
```

#### 与 ROS 集成
```python
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

class ROSIntegratedSLAM(LiveSLAMDemo):
    def __init__(self):
        super().__init__()
        rospy.init_node('livox_slam')
        
        self.cloud_pub = rospy.Publisher(
            '/slam/pointcloud', PointCloud2, queue_size=1
        )
        self.pose_pub = rospy.Publisher(
            '/slam/pose', PoseStamped, queue_size=1
        )
    
    def handle_points(self, xyz: np.ndarray):
        super().handle_points(xyz)
        
        # 发布点云和位姿到 ROS
        self.publish_to_ros()
```

通过这些扩展接口和插件机制，用户可以根据具体需求定制 SLAM 系统的功能，实现与其他系统的深度集成