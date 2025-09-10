# CODEBUDDY.md

## 开发命令

### 环境配置
```bash
# 安装依赖
sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 \
                 gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad gstreamer1.0-libav

# 创建虚拟环境
mkvirtualenv --system-site-packages unitree

# 安装 cyclonedds
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# 安装 unitree_sdk2_python
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="*/install"
pip3 install -e .

# 安装 Livox-SDK2
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && make -j$(nproc)
sudo make install
```

### 运行示例
```bash
# 运行 G1 运动控制示例
python3 g1_loco_client_example.py
```

## 架构概述

- **依赖项**:
  - 使用 CycloneDDS 进行通信
  - Livox-SDK2 用于激光雷达集成
  - GStreamer 用于多媒体处理

- **关键文件**:
  - `g1_loco_client_example.py`: 演示基于 RPC 的 G1 机器人控制

- **开发重点**:
  - 激光雷达点云处理 (`live_points.py`, `live_slam.py`)
  - 机械臂传感器集成 (`arm_gui.py`)
  - 数据流处理
  - 语音交互 (ASR/TTS)