# GUI 模块化重构说明

本文档描述了 `run_g1_gui.py` 的模块化重构过程和新的项目结构。

## 概述

原始的 `run_g1_gui.py` 文件包含 2647 行代码，是一个庞大的单体应用程序。为了提高代码的可维护性和可读性，我们将其重构为多个专门的模块。

## 新的项目结构

```
gui/
├── __init__.py           # 包初始化
├── logging.py           # 日志配置和流重定向
├── state.py             # 全局状态管理
├── threads.py           # 后台工作线程
├── utils.py             # 工具函数
└── main_window.py       # 主窗口类 (G1Windows)

run_g1_gui_modular.py    # 新的模块化入口点
test_modular_gui.py      # 模块化结构测试脚本
run_g1_gui.py            # 原始文件（保持不变）
```

## 模块说明

### 1. `gui/logging.py` - 日志管理
- **功能**: 设置统一的日志配置，包括文件轮转和控制台输出重定向
- **核心类**: `_StreamToLogger` - 将标准输出/错误重定向到日志系统
- **主要函数**: `setup_logging()` - 配置日志级别和输出格式

### 2. `gui/state.py` - 状态管理
- **功能**: 管理全局共享状态和线程同步
- **核心变量**: 
  - `_state` / `_state_lock` - 主要应用状态（摄像头数据、电池等）
  - `_slam_latest` / `_slam_lock` - SLAM 数据状态
- **导入**: 从 `run_g1_stack` 导入现有状态变量

### 3. `gui/threads.py` - 后台线程
- **功能**: 电池监控和SLAM处理的后台工作线程
- **核心函数**:
  - `rx_battery()` - 电池状态监控线程
  - `run_slam()` - SLAM 处理线程
  - `patch_slam_viewer()` - SLAM 查看器集成

### 4. `gui/utils.py` - 工具函数
- **功能**: 共用的工具函数和数据转换
- **核心函数**:
  - `numpy_to_qpix()` - NumPy 数组到 Qt QPixmap 转换
  - `clamp()` - 数值范围限制

### 5. `gui/main_window.py` - 主窗口
- **功能**: GUI 主窗口类的完整实现
- **核心类**: `G1Windows` - 主要的 GUI 控制器
- **特性**: 机器人控制、摄像头显示、SLAM 可视化、路径规划

### 6. `run_g1_gui_modular.py` - 新入口点
- **功能**: 模块化版本的主入口程序
- **特性**: 命令行参数解析、模块导入、用户友好的错误处理

## 使用方法

### 安装依赖

```bash
# 安装 GUI 依赖
pip install pyside6 pyqtgraph

# 安装其他依赖（如果尚未安装）
pip install -r requirements.txt
```

### 运行模块化 GUI

```bash
# 基本使用
python3 run_g1_gui_modular.py <网络接口>

# 示例
python3 run_g1_gui_modular.py eth0

# 带选项
python3 run_g1_gui_modular.py eth0 --ground-clear 6.0 --hand right --grip-force 0.5
```

### 运行测试

```bash
# 测试模块化结构
python3 test_modular_gui.py
```

## 命令行选项

- `network_interface`: 连接到机器人的网络接口（必需）
- `--ground-clear`: 地面间隙（英寸），默认 4.0
- `--hand`: 手部选择（left/right），默认 left
- `--grip-force`: 抓取力（N·m），默认 0.3
- `--log-level`: 日志级别（DEBUG/INFO/WARNING/ERROR），默认 INFO

## 控制说明

### 移动控制
- **W/S**: 前进/后退
- **A/D**: 左/右移动  
- **Q/E**: 左/右旋转

### 手臂控制
- **H**: 手臂回到默认位置
- **B**: 卸力手臂 & 腰部回中

### 手部控制
- **G**: 快速抓取
- **O**: 打开手部
- **C**: 关闭手部
- **F**: 指向手势
- **T**: 点赞手势

### 其他功能
- **点击2D地图**: 路径规划
- **ESC**: 退出程序

## 重构优势

### 1. 代码组织
- **模块化**: 功能按逻辑分组到不同模块
- **单一职责**: 每个模块有明确的职责
- **可读性**: 较小的文件更容易理解和维护

### 2. 维护性
- **隔离变更**: 修改特定功能时只需要修改对应模块
- **测试友好**: 可以单独测试各个模块
- **调试简化**: 问题定位更容易

### 3. 可扩展性
- **新功能**: 可以轻松添加新模块
- **重用性**: 模块可以在其他项目中重用
- **配置化**: 参数配置更加灵活

## 兼容性

- **向后兼容**: 原始 `run_g1_gui.py` 文件保持不变
- **依赖相同**: 使用相同的依赖包和配置
- **功能等价**: 模块化版本提供完全相同的功能

## 依赖要求

### 必需依赖
```
PySide6 >= 6.0.0          # Qt GUI 框架
pyqtgraph >= 0.12.0       # 图形绘制
numpy >= 1.20.0           # 数值计算
opencv-python >= 4.5.0    # 图像处理
open3d >= 0.15.0          # 点云处理
```

### 机器人相关依赖
```
unitree_sdk2_python       # 宇树机器人 SDK
cyclonedx == 0.10.2       # DDS 通信
```

### 可选依赖
```
pyrealsense2              # Intel RealSense 摄像头
```

## 故障排除

### 常见问题

1. **ModuleNotFoundError: No module named 'PySide6'**
   ```bash
   pip install pyside6 pyqtgraph
   ```

2. **网络连接问题**
   - 检查网络接口名称是否正确
   - 确保机器人和主机在同一网络
   - 验证防火墙设置

3. **摄像头连接问题**
   - 检查 RealSense 摄像头连接
   - 验证 pyrealsense2 安装
   - 确保有足够的 USB 带宽

4. **GUI 显示问题**
   - 确保有 X11 或 Wayland 显示环境
   - 检查 DISPLAY 环境变量设置
   - 验证 Qt 平台插件

### 调试技巧

1. **增加日志详细程度**:
   ```bash
   python3 run_g1_gui_modular.py enp2s0 --log-level DEBUG
   ```

2. **检查模块导入**:
   ```bash
   python3 test_modular_gui.py
   ```

3. **网络连接测试**:
   ```bash
   ping <机器人IP>
   tcpdump -i <网络接口> port 56301
   ```

## 开发说明

### 添加新功能

1. **创建新模块**: 在 `gui/` 目录下创建新的 `.py` 文件
2. **更新导入**: 在相关模块中添加必要的导入
3. **测试集成**: 更新 `test_modular_gui.py` 包含新模块测试
4. **文档更新**: 更新本文档说明新功能

### 代码风格

- 使用 Tab 进行缩进
- 遵循 PEP 8 编码规范
- 添加适当的类型注解
- 提供详细的文档字符串

### 调试建议

- 使用 `print()` 进行调试输出
- 利用日志系统记录重要事件
- 在适当的位置添加异常处理
- 使用测试脚本验证功能

## 未来改进

1. **进一步模块化**: 考虑将 `G1Windows` 类拆分为更小的组件
2. **配置文件**: 添加 JSON/YAML 配置文件支持
3. **插件系统**: 实现插件架构以支持扩展功能
4. **单元测试**: 添加完整的单元测试套件
5. **文档生成**: 使用 Sphinx 生成 API 文档

---

*此文档描述了从单体应用到模块化架构的重构过程。如有问题或建议，请查阅项目文档或联系开发团队。*
