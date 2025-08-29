# 前提
基于[unitree_g1_vibes](https://github.com/Sentdex/unitree_g1_vibes/tree/main)进行开发。
# list
- 更新readme.md
   - ~~环境配置~~
- 雷达点云代码(live_points.py, live_slam.py)
   - 测试移动时点云地图的构建
- arm传感器测试(arm_gui.py)
- 数据输入流代码
- 语音对话demo编写

# 环境配置

1. **安装gi**
   ```bash
   sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 \
                    gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
                    gstreamer1.0-plugins-bad gstreamer1.0-libav
   ```

2. **创建虚拟环境（包含系统包的环境，gi需要）**
   ```bash
   mkvirtualenv --system-site-packages unitree
   ```

3. **安装unitree_sdk2_python**
   
   a. 安装cyclonedds
   ```bash
   git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
   cd cyclonedds && mkdir build install && cd build
   cmake .. -DCMAKE_INSTALL_PREFIX=../install
   cmake --build . --target install
   ```
   
   b. 进入unitree_sdk2_python目录并安装
   ```bash
   cd ~/unitree_sdk2_python
   export CYCLONEDDS_HOME="*/install"
   pip3 install -e .
   ```
4. **升级cmake版本(用户级)**
   ```bash
   wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-aarch64.sh
   chmod +x cmake-3.22.1-linux-aarch64.sh
   sudo ./cmake-3.22.1-linux-aarch64.sh --prefix=$HOME/.local --exclude-subdir
   # 测试
   export PATH="$HOME/.local/bin:$PATH"
   cmake --version
   # 写入配置文件（可选）
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```
5. **安装livox-sdk2**
   git clone https://github.com/Livox-SDK/Livox-SDK2.git
   cd Livox-SDK2 && mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && make -j$(nproc)
   sudo make install          # installs liblivox_lidar_sdk.so → /usr/local/lib
# 代码功能
- g1_loco_client_example.py:g1使用rpc来进行控制示例


