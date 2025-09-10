# 项目分析文档

## 项目概述

本项目是基于 Unitree G1 机器人的 Python SDK 开发环境，主要用于控制 Unitree G1 机器人的运动、传感器数据处理（如激光雷达、摄像头等）以及语音交互等功能。项目基于 [unitree_g1_vibes](https://github.com/Sentdex/unitree_g1_vibes/tree/main) 进行开发。

## 目录结构

- **components/**: 包含机器人各功能模块的实现，如电池管理、SLAM、传感器集成等。
- **data/**: 存放训练数据或模型文件。
- **docs/**: 文档目录。
- **librealsense/**: Intel RealSense 相机的相关代码和测试脚本。
- **Livox-SDK2/**: Livox 激光雷达的 SDK 和集成代码。
- **past/**: 历史代码存档，包含早期版本的传感器测试代码。
- **rl_arm/**: 机械臂强化学习相关代码。
- **tools/**: 工具脚本。
- **unitree_sdk_python/**: Unitree 官方 Python SDK 的核心代码和示例。

## 关键文件

1. **g1_loco_client_example.py**: 演示如何使用 RPC 控制 G1 机器人的运动。
2. **live_points.py** 和 **live_slam.py**: 激光雷达点云处理和 SLAM 实现。
3. **arm_gui_hud.py**: 机械臂传感器测试和 GUI 界面。
4. **audio_client_example.py**: 语音交互示例代码。
5. **run_g1_stack.py**: 机器人主控制栈的实现。

## 依赖项

1. **CycloneDDS**: 用于机器人通信。
2. **Livox-SDK2**: 激光雷达数据处理。
3. **GStreamer**: 多媒体处理（如摄像头数据流）。
4. **Intel RealSense**: 深度相机支持。

## 开发重点

1. **激光雷达点云处理**: 实时点云地图构建和 SLAM。
2. **机械臂控制**: 传感器集成和动作规划。
3. **语音交互**: ASR/TTS 功能测试和优化。
4. **数据流处理**: 实时传感器数据流的高效处理。

## 环境配置

详见 `CODEBUDDY.md` 中的环境配置部分。

## 运行示例

```bash
# 运行 G1 运动控制示例
python3 g1_loco_client_example.py
```

## 扩展功能

1. **强化学习**: 在 `rl_arm/` 目录中提供了机械臂强化学习的训练代码。
2. **多传感器融合**: 结合激光雷达、摄像头和 IMU 数据进行环境感知。
3. **语音控制**: 通过语音指令控制机器人动作。