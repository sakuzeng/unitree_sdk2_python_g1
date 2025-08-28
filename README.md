# 前提
基于[unitree_g1_vibes](https://github.com/Sentdex/unitree_g1_vibes/tree/main)进行开发。
# 环境配置
- 1. 安装gi
```
sudo apt install python3-gi gir1.2-gst-plugins-base-1.0 \
                 gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good \
                 gstreamer1.0-plugins-bad gstreamer1.0-libav
```
- 2. 创建虚拟环境（包含系统包的环境，gi需要）
```
mkvirtualenv --system-site-packages unitree
```
- 3. 安装unitree_sdk2_python
        安装cyclonedds
```
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```

        进入unitree_sdk2_python目录
        
```
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="*/install"
pip3 install -e .
```