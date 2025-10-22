"""
Livox2 扫描器适配器
"""
import numpy as np
import threading
import time
from typing import Optional
from pathlib import Path

from livox2_python import Livox2

class Livox2Scanner(Livox2):
    """Livox2扫描器，继承自Livox2并添加数据缓存功能"""
    
    def __init__(self, config_path: str, host_ip: str, 
                 *, frame_time: float = 0.20, frame_packets: int = 120):
        super().__init__(config_path, host_ip, frame_time=frame_time, frame_packets=frame_packets)
        
        # 数据缓存
        self.latest_pointcloud = None
        self.latest_imu_data = None
        self.data_lock = threading.Lock()
        
        # 统计信息
        self.point_frame_count = 0
        self.imu_frame_count = 0
        self.last_point_time = 0
        self.last_imu_time = 0
        
        print("[Livox2Scanner] 扫描器初始化完成")
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, tag: np.ndarray, timestamp: int):
        """
        处理点云数据的回调函数
        
        Args:
            xyz: 点云坐标 (N, 3)，单位：米
            reflectivity: 反射强度 (N,)，范围 0-255
            tag: 标签 (N,)，用于噪声过滤（0为正常点）
            timestamp: 时间戳，单位：ns
        """
        # 过滤有效点
        valid_mask = tag == 0  # 只保留正常点
        if np.sum(valid_mask) > 0:
            valid_points = xyz[valid_mask]
            
            # 更新缓存
            with self.data_lock:
                self.latest_pointcloud = valid_points.copy()
                self.point_frame_count += 1
                self.last_point_time = timestamp
            
            # 打印调试信息（限频）
            if self.point_frame_count % 10 == 0:
                print(f"[Livox2Scanner] 点云帧 #{self.point_frame_count}: {len(xyz)} -> {len(valid_points)} 点")
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """
        处理IMU数据的回调函数
        
        Args:
            imu_data: IMU数据 (N, 6)，包含 [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z]
            timestamp: 时间戳，单位：ns
        """
        with self.data_lock:
            self.latest_imu_data = imu_data.copy()
            self.imu_frame_count += 1
            self.last_imu_time = timestamp
        
        # 打印调试信息（限频）
        if self.imu_frame_count % 50 == 0:
            print(f"[Livox2Scanner] IMU帧 #{self.imu_frame_count}: {len(imu_data)} 样本")
    
    def get_latest_pointcloud(self) -> Optional[np.ndarray]:
        """获取最新点云数据"""
        with self.data_lock:
            return self.latest_pointcloud.copy() if self.latest_pointcloud is not None else None
    
    def get_latest_imu(self) -> Optional[np.ndarray]:
        """获取最新IMU数据"""
        with self.data_lock:
            return self.latest_imu_data.copy() if self.latest_imu_data is not None else None
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        with self.data_lock:
            return {
                'point_frames': self.point_frame_count,
                'imu_frames': self.imu_frame_count,
                'last_point_timestamp': self.last_point_time,
                'last_imu_timestamp': self.last_imu_time,
                'latest_point_count': len(self.latest_pointcloud) if self.latest_pointcloud is not None else 0
            }
    
    def wait_for_data(self, timeout: float = 5.0) -> bool:
        """等待数据到达"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.latest_pointcloud is not None:
                return True
            time.sleep(0.1)
        return False

def create_default_config(host_ip: str = "192.168.123.164") -> Path:
    """创建默认配置文件"""
    import json
    
    cfg_path = Path("mid360_config.json")
    
    if not cfg_path.exists():
        data = {
            "MID360": {
                "lidar_net_info": {
                    "lidar_ip": "192.168.123.120",
                    "cmd_data_port": 56100,
                    "push_msg_port": 56200,
                    "point_data_port": 56300,
                    "imu_data_port": 56400,
                    "log_data_port": 56500,
                },
                "host_net_info": [
                    {
                        "host_ip": host_ip,
                        "multicast_ip": "224.1.1.5",
                        "cmd_data_port": 56101,
                        "push_msg_port": 56201,
                        "point_data_port": 56301,
                        "imu_data_port": 56401,
                        "log_data_port": 56501,
                    }
                ]
            }
        }
        cfg_path.write_text(json.dumps(data, indent=2))
        print(f"[Livox2Scanner] 默认配置文件已生成: {cfg_path}")
    
    return cfg_path

if __name__ == "__main__":
    import sys
    
    # 创建默认配置
    host_ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.123.164"
    config_path = create_default_config(host_ip)
    
    # 测试扫描器
    scanner = Livox2Scanner(str(config_path), host_ip)
    
    print("[Livox2Scanner] 等待数据...")
    if scanner.wait_for_data():
        print("[Livox2Scanner] 数据接收正常")
        
        # 运行一段时间收集数据
        try:
            time.sleep(10)
            stats = scanner.get_statistics()
            print(f"[Livox2Scanner] 统计信息: {stats}")
        except KeyboardInterrupt:
            print("[Livox2Scanner] 用户中断")
    else:
        print("[Livox2Scanner] 数据接收超时")
    
    scanner.shutdown()