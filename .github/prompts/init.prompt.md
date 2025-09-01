# 项目初始化提示

## 提示名称
`init` - 生成项目 Copilot 指令文档

## 用途
为Unitree G1 机器人项目生成标准化的 `copilot-instructions.md` 文件，确保代码生成符合项目技术栈和编码规范。

## 提示内容

请为此 Unitree G1 机器人项目生成一个完整的 `copilot-instructions.md` 文件，包含以下内容：

### 项目技术栈
- Python 3.8+ 作为主要开发语言
- unitree_sdk2_python 机器人控制 SDK
- Livox SDK2 和 Mid-360 LiDAR 激光雷达系统
- Intel RealSense 摄像头集成
- Open3D 点云处理
- OpenCV 图像处理
- DDS (CycloneDX) 通信协议
- CMake 和 ROS 构建系统

### 编码规范要求
- 遵循 PEP 8 Python 编码规范
- snake_case 函数命名，PascalCase 类命名
- UPPER_SNAKE_CASE 常量命名
- 标准项目目录结构

### 代码模板需求
- 机器人控制基础模板（包含网络接口参数处理）
- 激光雷达数据处理模板
- 摄像头集成模板
- 错误处理和安全注意事项

### 文档规范
- 标准化注释和 docstring 格式
- 示例代码使用说明
- 配置文件规范 (JSON 格式)

### 安全和调试指导
- 低级控制安全提示
- 常见问题解决方案
- 调试方法建议
- 网络配置验证

生成的指令文件应确保 GitHub Copilot 能够为此项目生成符合技术栈要求、遵循编码规范且安全可靠的代码。

## 调用方式
```
#prompt:init
```

## 输出文件
生成的内容应保存为项目家目录下的.github目录下的的 `copilot-instructions.md` 文件。

## 注意事项
- 确保涵盖项目的所有主要技术组件
- 包含实用的代码模板和最佳实践
- 提供清晰的安全操作指导
- 适配 Unitree G1 机器人的特定需求