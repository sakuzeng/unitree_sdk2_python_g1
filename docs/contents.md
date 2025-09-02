<!--
 * @Author: sakuzeng1213
 * @Date: 2025-09-02 10:30:19
 * @LastEditTime: 2025-09-02 10:30:22
 * @LastEditors: sakuzeng1213
 * @FilePath: /unitree_sdk2_python_g1/docs/contents.md
 * @Description: Unitree G1 机器人项目结构与功能说明
-->
# Unitree G1 机器人项目结构与功能说明

## 项目概述

这是一个基于 Unitree G1 人形机器人的 Python 开发项目，集成了运动控制、激光雷达、摄像头和语音交互功能。项目旨在提供完整的机器人控制和传感器集成解决方案，实现实时环境感知、自主导航和人机交互功能。

## 核心技术栈

- **主要语言**: Python 3.8+
- **机器人控制**: unitree_sdk2_python
- **激光雷达处理**: Livox SDK2, Mid-360 LiDAR
- **计算机视觉**: Intel RealSense, OpenCV
- **点云处理**: Open3D, KISS-ICP
- **通信协议**: DDS (CycloneDX)
- **构建系统**: CMake, ROS

## 项目目录结构详解

### 根目录核心文件

```
├── README.md                    # 项目主要说明文档
├── requirements.txt             # Python 依赖包列表
├── .gitignore                   # Git 版本控制忽略配置
├── .gitmodules                  # Git 子模块配置
├── mid360_config.json          # Livox Mid-360 激光雷达配置文件
```

### 主要功能脚本

#### 运动控制模块
- **`g1_loco_client_example.py`** - G1机器人运动控制示例，使用RPC进行控制
- **`keyboard_controller.py`** - 键盘控制机器人运动的交互式程序
- **`hanger_boot_sequence.py`** - 机器人启动序列控制脚本

#### 传感器数据处理
- **`live_points.py`** - 激光雷达实时点云数据处理和可视化
- **`live_slam.py`** - 基于激光雷达的实时SLAM系统
- **`livox_python.py`** - Livox激光雷达Python接口封装
- **`livox2_python.py`** - Livox SDK2的Python集成实现

#### 视觉处理模块
- **`jetson_realsense_stream.py`** - Jetson平台上的RealSense摄像头数据流处理
- **`stream_realsense.py`** - RealSense摄像头数据流管理
- **`receive_realsense_gst.py`** - 基于GStreamer的RealSense数据接收

#### 用户界面与交互
- **`run_g1_gui.py`** - G1机器人图形用户界面主程序
- **`run_g1_stack.py`** - G1机器人完整功能栈启动脚本
- **`arm_gui.py`** - 机械臂传感器测试和控制界面
- **`audio_example.py`** - 语音交互功能示例

### 子目录详细说明

#### `.github/` - GitHub 项目配置
```
.github/
├── copilot-instructions.md      # GitHub Copilot 编码指令和规范
└── prompts/                     # 提示词模板目录
    ├── analysis.prompt.md       # 代码分析提示模板
    ├── generate_contents.prompt.md  # 内容生成提示模板
    └── thinkharder.prompt.md    # 深度思考提示模板
```

#### `.vscode/` - VSCode 开发环境配置
```
.vscode/
└── settings.json               # VSCode 编辑器配置（Tab缩进等）
```

#### `docs/` - 项目文档目录
```
docs/
├── requirements.md             # 详细需求文档
├── FSM_README.md              # 有限状态机说明文档
└── lidar_cheatsheet.md        # Livox Mid-360 激光雷达快速配置指南
```

**文档功能说明**:
- [`requirements.md`](docs/requirements.md) - 包含功能性需求、非功能性需求、技术约束等完整需求规格
- [`lidar_cheatsheet.md`](docs/lidar_cheatsheet.md) - 提供激光雷达从网络配置到ROS集成的完整操作指南

#### `unitree_sdk_python/` - 机器人控制核心SDK
```
unitree_sdk_python/
├── README.md                   # SDK 使用说明（英文）
├── README zh.md               # SDK 使用说明（中文）
├── unitree_sdk2py/            # Python SDK 核心模块
│   ├── __init__.py
│   ├── core/                  # 核心通信模块
│   ├── rpc/                   # RPC 远程调用模块
│   ├── utils/                 # 工具函数模块
│   └── idl/                   # 数据定义语言模块
│       ├── default.py         # 默认数据结构定义
│       ├── unitree_go/        # Go2/B2/H1 机器人消息定义
│       ├── unitree_hg/        # G1/H1-2 机器人消息定义
│       └── unitree_api/       # API 消息定义
└── example/                   # 示例代码目录
    └── g1/                    # G1 机器人专用示例
        └── readme.md          # G1 示例说明
```

**SDK 功能模块**:
- **通信层**: 基于 DDS (CycloneDX) 的高性能通信
- **控制接口**: 高级运动控制和低级电机控制
- **数据结构**: 完整的机器人状态和命令消息定义
- **示例程序**: 涵盖运动控制、传感器读取、避障等功能

#### `Livox-SDK2/` - 激光雷达SDK
完整的 Livox Mid-360 激光雷达开发套件，包含:
- C++ 核心库和示例程序
- 配置文件模板和说明文档
- 多雷达升级和日志记录工具
- Ubuntu 和 Windows 平台支持

#### `librealsense/` - Intel RealSense SDK
Intel RealSense 深度摄像头完整开发环境:
```
librealsense/
├── examples/                   # 各种功能示例
│   ├── cmake/                 # CMake 构建示例
│   ├── post-processing/       # 后处理算法示例
│   ├── hdr/                   # HDR 处理示例
│   └── gl/                    # OpenGL 渲染示例
├── tools/                     # 实用工具集
│   ├── convert/               # 格式转换工具
│   ├── data-collect/          # 数据采集工具
│   └── embed/                 # 嵌入式开发工具
├── unit-tests/                # 单元测试框架
├── doc/                       # 技术文档
└── src/                       # 源代码目录
```

#### `tools/` - 开发工具集
```
tools/
└── kiss-icp-v1.2.3/          # KISS-ICP SLAM 算法库
    └── kiss-icp-1.2.3/
        ├── python/            # Python 接口
        └── README.md          # SLAM 算法说明
```

#### `rl_arm/` - 强化学习机械臂模块
机械臂强化学习训练和控制相关代码（具体内容需进一步分析）

## 功能模块映射

### 1. 机器人运动控制系统
- **核心文件**: [`g1_loco_client_example.py`](g1_loco_client_example.py), [`keyboard_controller.py`](keyboard_controller.py)
- **SDK支持**: [`unitree_sdk_python/`](unitree_sdk_python/)
- **功能**: 高级运动控制、低级电机控制、避障与安全控制

### 2. 传感器数据处理系统
- **激光雷达**: [`live_points.py`](live_points.py), [`Livox-SDK2/`](Livox-SDK2/)
- **深度摄像头**: [`jetson_realsense_stream.py`](jetson_realsense_stream.py), [`librealsense/`](librealsense/)
- **数据融合**: [`live_slam.py`](live_slam.py), [`tools/kiss-icp-v1.2.3/`](tools/kiss-icp-v1.2.3/)

### 3. SLAM和地图构建系统
- **实时SLAM**: [`live_slam.py`](live_slam.py)
- **算法库**: [`tools/kiss-icp-v1.2.3/`](tools/kiss-icp-v1.2.3/)
- **点云处理**: [`live_points.py`](live_points.py)

### 4. 语音交互系统
- **语音处理**: [`audio_example.py`](audio_example.py)
- **多模态交互**: 与视觉和运动控制系统集成

### 5. 用户界面系统
- **图形界面**: [`run_g1_gui.py`](run_g1_gui.py), [`arm_gui.py`](arm_gui.py)
- **系统集成**: [`run_g1_stack.py`](run_g1_stack.py)

## 配置文件说明

### 网络配置
- **激光雷达配置**: [`mid360_config.json`](mid360_config.json)
  - 雷达IP: 192.168.123.120
  - 主机IP: 192.168.123.164  
  - 数据端口: 56301 (点云), 56401 (IMU)

### 开发环境配置
- **编码规范**: [`.github/copilot-instructions.md`](.github/copilot-instructions.md)
  - Tab 缩进, Unix 换行符
  - Python PEP 8 规范
  - 文件命名约定

## 部署需求

### 硬件要求
- Unitree G1 机器人
- Livox Mid-360 激光雷达
- Intel RealSense D435i 摄像头
- Ubuntu 20.04/22.04 开发环境

### 软件依赖
- Python 3.8+
- CyclonDX DDS 通信库
- OpenCV, Open3D, NumPy
- ROS Noetic/Humble (可选)

### 网络配置
- 机器人网络接口: enp2s0 (可配置)
- 激光雷达网段: 192.168.123.0/24
- DDS 域配置: 默认域

## 开发指南

### 快速开始
1. 配置网络连接 (参考 [`docs/lidar_cheatsheet.md`](docs/lidar_cheatsheet.md))
2. 安装 SDK (参考 [`unitree_sdk_python/README.md`](unitree_sdk_python/README.md))
3. 运行基础示例 (参考 [`g1_loco_client_example.py`](g1_loco_client_example.py))

### 代码规范
- 使用 Tab 进行缩进
- 遵循项目编码指令 ([`.github/copilot-instructions.md`](.github/copilot-instructions.md))
- 完整的错误处理和安全检查

### 测试验证
- 网络连接测试
- 传感器数据验证  
- 运动控制安全测试
- 系统集成测试

## 项目特色

1. **完整性**: 涵盖运动控制、感知、决策、交互的完整机器人系统
2. **模块化**: 清晰的模块划分，便于开发和维护
3. **实时性**: 基于 DDS 的高性能实时通信
4. **可扩展**: 支持多传感器融合和功能扩展
5. **跨平台**: 支持 Ubuntu 和 Jetson 平台部署

此项目为 Unitree G1 机器人提供了完整的开发框架，适合机器人研究、教学和应