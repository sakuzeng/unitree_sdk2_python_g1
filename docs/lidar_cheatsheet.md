# Unitree G-1 — Mid-360 LiDAR Quick-Start

This document condenses the full bring-up procedure into a repeatable checklist. Follow it on any fresh machine to get a live `/livox/lidar` PointCloud2 stream in less than 15 minutes.

## 1. Network & Prerequisites
1. Ethernet from G-1 shoulder port to your PC.
2. Assign PC an IP in `192.168.123.0/24`, e.g.
   ```
   sudo ip addr add 192.168.123.222/24 dev enp68s0f1
   ```
3. Verify link:
   ```
   ping 192.168.123.120
   ```
4. Base packages:
   ```
   sudo apt update
   sudo apt install git build-essential cmake tcpdump
   ```

**Note:** Firewall: allow or disable UDP 56100-56500.

## 2. Livox SDK 2
```
# clone & build
cd ~
git clone https://github.com/Livox-SDK/Livox-SDK2.git
cd Livox-SDK2 && mkdir build && cd build
cmake ..
make -j$(nproc)

# install
sudo make install
sudo ldconfig
```

## 3. livox_ros_driver2 (ROS Noetic)
```
# workspace skeleton
source /opt/ros/noetic/setup.bash
mkdir -p ~/catkin_ws/src && cd ~/catkin_ws/src

# driver
git clone https://github.com/Livox-SDK/livox_ros_driver2.git

# switch to ROS-1 flavour & build
cd livox_ros_driver2
./build.sh ROS1            # copies package_ROS1.xml → package.xml

cd ~/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release -DROS_EDITION=ROS1
```

Add to `~/.bashrc`:
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

## 4. Configure `MID360_config.json`
```
mkdir -p ~/livox_cfg
cp ~/catkin_ws/src/livox_ros_driver2/config/MID360_config.json ~/livox_cfg/
nano ~/livox_cfg/MID360_config.json
```

Set:
| Field                        | Value                                         |
|------------------------------|-----------------------------------------------|
| `lidar_configs[0].ip`       | 192.168.123.120                              |
| `host_net_info.*_ip`        | Your PC IP (e.g. 192.168.123.222)           |

```
{
  "lidar_summary_info" : {
    "lidar_type": 8
  },
  "MID360": {
    "lidar_net_info" : {
      "cmd_data_port": 56100,
      "push_msg_port": 56200,
      "point_data_port": 56300,
      "imu_data_port": 56400,
      "log_data_port": 56500
    },
    "host_net_info" : {
      "cmd_data_ip" : "192.168.123.222",
      "cmd_data_port": 56101,
      "push_msg_ip": "192.168.123.222",
      "push_msg_port": 56201,
      "point_data_ip": "192.168.123.222",
      "point_data_port": 56301,
      "imu_data_ip" : "192.168.123.222",
      "imu_data_port": 56401,
      "log_data_ip" : "",
      "log_data_port": 56501
    }
  },
  "lidar_configs" : [
    {
      "ip" : "192.168.123.120",
      "pcl_data_type" : 1,
      "pattern_mode" : 0,
      "extrinsic_parameter" : {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "x": 0,
        "y": 0,
        "z": 0
      }
    }
  ]
}
```

## 5. Run the Driver
```
roslaunch livox_ros_driver2 msg_MID360.launch \
    user_config_path:=/home/$USER/livox_cfg/MID360_config.json \
    xfer_format:=0  data_src:=0
```
```
[ INFO] Connect lidar: 192.168.123.120 …
[ INFO] Init lidar success
```

## 6. Verify
```
rostopic hz /livox/lidar       # ~10 Hz
sudo tcpdump -i enp68s0f1 udp port 56300 -n -c 5
```

## 7. Python Live Viewer
```
pip3 install --user open3d ros_numpy
python3 -m g1_lidar.live_view   # scroll wheel to zoom
```

## 8. Troubleshooting
- **bind failed** → JSON host IP wrong or ports busy.
- **CustomMsg mismatch** → use `xfer_format:=0`.
- **White viewer** → camera inside cloud; zoom out.
- **Cannot import rospy** → forgot to source `setup.bash`.

**Done.** You now have a live `sensor_msgs/PointCloud2` stream ready for SLAM, mapping, or your own NumPy processing.