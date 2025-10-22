"""
Livox2 数据查看器
简单查看 MID-360 雷达的点云和 IMU 数据输出，并保存到文件
"""
import time
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from path_planner.livox2_python import Livox2, create_default_config

class SimpleDataViewer(Livox2):
    """简单的数据查看器，继承 Livox2 类"""
    
    def __init__(self, *args, save_data=True, save_dir="data", **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = 0
        self.total_points = 0
        self.imu_count = 0
        self.start_time = time.time()
        
        # 数据保存设置
        self.save_data = save_data
        self.save_dir = Path(save_dir)
        
        if self.save_data:
            # 创建保存目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.save_dir / f"livox_session_{timestamp}"
            self.session_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            self.pointcloud_dir = self.session_dir / "pointcloud"
            self.imu_dir = self.session_dir / "imu"
            self.pointcloud_dir.mkdir(exist_ok=True)
            self.imu_dir.mkdir(exist_ok=True)
            
            # 数据存储列表
            self.pointcloud_data = []
            self.imu_data = []
            self.metadata = {
                "session_start": timestamp,
                "config": {
                    "host_ip": kwargs.get('host_ip', '192.168.123.164'),
                    "frame_time": kwargs.get('frame_time', 0.1),
                    "frame_packets": kwargs.get('frame_packets', 50),
                    "enable_filter": kwargs.get('enable_filter', True),
                    "max_range": kwargs.get('max_range', 30.0),
                    "voxel_size": kwargs.get('voxel_size', 0.1)
                }
            }
            
            print(f"数据将保存到: {self.session_dir}")
    
    def handle_points(self, xyz, reflectivity, tag, timestamp):
        """点云数据回调"""
        self.frame_count += 1
        self.total_points += len(xyz)
        
        print(f"\n=== 点云帧 #{self.frame_count} ===")
        print(f"时间戳: {timestamp}")
        print(f"点数: {len(xyz)}")
        
        if len(xyz) > 0:
            # 保存点云数据
            if self.save_data:
                frame_data = {
                    "frame_id": self.frame_count,
                    "timestamp": timestamp,
                    "point_count": len(xyz),
                    "xyz": xyz.tolist(),
                    "reflectivity": reflectivity.tolist(),
                    "tag": tag.tolist()
                }
                self.pointcloud_data.append(frame_data)
                
                # 每10帧保存一次到磁盘
                if self.frame_count % 10 == 0:
                    self._save_pointcloud_batch()
            
            # 显示坐标范围
            print(f"X 范围: [{xyz[:, 0].min():.3f}, {xyz[:, 0].max():.3f}] 米")
            print(f"Y 范围: [{xyz[:, 1].min():.3f}, {xyz[:, 1].max():.3f}] 米") 
            print(f"Z 范围: [{xyz[:, 2].min():.3f}, {xyz[:, 2].max():.3f}] 米")
            
            # 显示距离统计
            distances = np.linalg.norm(xyz, axis=1)
            print(f"距离范围: [{distances.min():.3f}, {distances.max():.3f}] 米")
            print(f"平均距离: {distances.mean():.3f} 米")
            
            # 显示反射强度统计
            print(f"反射强度范围: [{reflectivity.min()}, {reflectivity.max()}]")
            print(f"平均反射强度: {reflectivity.mean():.1f}")
            
            # 显示噪声点统计
            noise_points = np.sum(tag != 0)
            valid_points = len(xyz) - noise_points
            print(f"有效点: {valid_points}, 噪声点: {noise_points}")
            
            # 显示前5个点的详细信息
            print(f"前5个点详情:")
            for i in range(min(5, len(xyz))):
                print(f"  点{i+1}: XYZ=({xyz[i,0]:.3f}, {xyz[i,1]:.3f}, {xyz[i,2]:.3f}), "
                     f"反射强度={reflectivity[i]}, 标签={tag[i]}")
    
    def handle_imu(self, imu_data, timestamp):
        """IMU 数据回调"""
        self.imu_count += len(imu_data)
        
        print(f"\n--- IMU 数据 ---")
        print(f"时间戳: {timestamp}")
        print(f"样本数: {len(imu_data)}")
        
        if len(imu_data) > 0:
            # 保存 IMU 数据
            if self.save_data:
                imu_frame = {
                    "timestamp": timestamp,
                    "sample_count": len(imu_data),
                    "gyro_x": imu_data[:, 0].tolist(),
                    "gyro_y": imu_data[:, 1].tolist(),
                    "gyro_z": imu_data[:, 2].tolist(),
                    "acc_x": imu_data[:, 3].tolist(),
                    "acc_y": imu_data[:, 4].tolist(),
                    "acc_z": imu_data[:, 5].tolist()
                }
                self.imu_data.append(imu_frame)
            
            # 分离陀螺仪和加速度计数据
            gyro = imu_data[:, :3]  # 前3列：陀螺仪 (rad/s)
            acc = imu_data[:, 3:]   # 后3列：加速度计 (m/s²)
            
            print(f"陀螺仪 (rad/s):")
            print(f"  X: [{gyro[:, 0].min():.4f}, {gyro[:, 0].max():.4f}]")
            print(f"  Y: [{gyro[:, 1].min():.4f}, {gyro[:, 1].max():.4f}]")
            print(f"  Z: [{gyro[:, 2].min():.4f}, {gyro[:, 2].max():.4f}]")
            
            print(f"加速度计 (m/s²):")
            print(f"  X: [{acc[:, 0].min():.4f}, {acc[:, 0].max():.4f}]")
            print(f"  Y: [{acc[:, 1].min():.4f}, {acc[:, 1].max():.4f}]")
            print(f"  Z: [{acc[:, 2].min():.4f}, {acc[:, 2].max():.4f}]")
            
            # 显示第一个样本的详细信息
            print(f"第一个样本:")
            print(f"  陀螺仪: ({gyro[0,0]:.4f}, {gyro[0,1]:.4f}, {gyro[0,2]:.4f})")
            print(f"  加速度: ({acc[0,0]:.4f}, {acc[0,1]:.4f}, {acc[0,2]:.4f})")
    
    def _save_pointcloud_batch(self):
        """批量保存点云数据到文件"""
        if not self.pointcloud_data:
            return
        
        # 保存为JSON格式
        batch_file = self.pointcloud_dir / f"pointcloud_batch_{self.frame_count//10:04d}.json"
        with open(batch_file, 'w') as f:
            json.dump(self.pointcloud_data, f, indent=2)
        
        # 同时保存为NumPy格式（便于后续处理）
        for frame in self.pointcloud_data:
            frame_id = frame['frame_id']
            xyz = np.array(frame['xyz'])
            reflectivity = np.array(frame['reflectivity'])
            tag = np.array(frame['tag'])
            
            # 保存点云数据为.npz格式
            npz_file = self.pointcloud_dir / f"frame_{frame_id:06d}.npz"
            np.savez_compressed(npz_file,
                               xyz=xyz,
                               reflectivity=reflectivity,
                               tag=tag,
                               timestamp=frame['timestamp'])
        
        print(f"已保存 {len(self.pointcloud_data)} 帧点云数据到 {batch_file}")
        self.pointcloud_data.clear()  # 清空缓存
    
    def _save_imu_data(self):
        """保存 IMU 数据到文件"""
        if not self.imu_data:
            return
        
        # 保存为JSON格式
        imu_file = self.imu_dir / "imu_data.json"
        with open(imu_file, 'w') as f:
            json.dump(self.imu_data, f, indent=2)
        
        # 同时保存为NumPy格式
        if self.imu_data:
            # 合并所有IMU数据
            all_timestamps = []
            all_gyro_x, all_gyro_y, all_gyro_z = [], [], []
            all_acc_x, all_acc_y, all_acc_z = [], [], []
            
            for frame in self.imu_data:
                frame_timestamps = [frame['timestamp']] * frame['sample_count']
                all_timestamps.extend(frame_timestamps)
                all_gyro_x.extend(frame['gyro_x'])
                all_gyro_y.extend(frame['gyro_y'])
                all_gyro_z.extend(frame['gyro_z'])
                all_acc_x.extend(frame['acc_x'])
                all_acc_y.extend(frame['acc_y'])
                all_acc_z.extend(frame['acc_z'])
            
            # 保存为NumPy数组
            imu_npz = self.imu_dir / "imu_data.npz"
            np.savez_compressed(imu_npz,
                               timestamps=np.array(all_timestamps),
                               gyro_x=np.array(all_gyro_x),
                               gyro_y=np.array(all_gyro_y),
                               gyro_z=np.array(all_gyro_z),
                               acc_x=np.array(all_acc_x),
                               acc_y=np.array(all_acc_y),
                               acc_z=np.array(all_acc_z))
        
        print(f"已保存 {len(self.imu_data)} 组 IMU 数据到 {imu_file}")
    
    def save_session_summary(self):
        """保存会话摘要信息"""
        if not self.save_data:
            return
        
        # 更新元数据
        elapsed_time = time.time() - self.start_time
        self.metadata.update({
            "session_end": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "duration_seconds": elapsed_time,
            "statistics": {
                "total_pointcloud_frames": self.frame_count,
                "total_points": self.total_points,
                "total_imu_samples": self.imu_count,
                "average_points_per_frame": self.total_points / max(self.frame_count, 1),
                "pointcloud_frame_rate": self.frame_count / max(elapsed_time, 1),
                "sdk_stats": self.get_stats()
            }
        })
        
        # 保存元数据
        metadata_file = self.session_dir / "session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # 保存剩余的点云数据
        if self.pointcloud_data:
            self._save_pointcloud_batch()
        
        # 保存IMU数据
        self._save_imu_data()
        
        # 创建数据说明文件
        readme_file = self.session_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"""# Livox MID-360 数据采集会话

## 会话信息
- 开始时间: {self.metadata['session_start']}
- 结束时间: {self.metadata['session_end']}
- 持续时间: {elapsed_time:.1f} 秒

## 数据统计
- 点云帧数: {self.frame_count}
- 总点数: {self.total_points:,}
- IMU 样本数: {self.imu_count:,}
- 平均帧率: {self.frame_count / max(elapsed_time, 1):.1f} Hz
- 平均每帧点数: {self.total_points / max(self.frame_count, 1):.0f}

## 文件结构
- `pointcloud/`: 点云数据文件
  - `*.json`: JSON格式的点云数据
  - `*.npz`: NumPy压缩格式的点云数据
- `imu/`: IMU数据文件
  - `imu_data.json`: JSON格式的IMU数据
  - `imu_data.npz`: NumPy压缩格式的IMU数据
- `session_metadata.json`: 会话元数据

## 数据格式说明
### 点云数据 (.npz)
- `xyz`: 点坐标 (N×3 array, 单位: 米)
- `reflectivity`: 反射强度 (N×1 array, 0-255)
- `tag`: 点标签 (N×1 array, 0=有效点, 1=噪声点)
- `timestamp`: 时间戳

### IMU数据 (.npz)
- `gyro_x/y/z`: 陀螺仪数据 (rad/s)
- `acc_x/y/z`: 加速度计数据 (m/s²)
- `timestamps`: 时间戳数组
""")
        
        print(f"\n数据保存完成:")
        print(f"  会话目录: {self.session_dir}")
        print(f"  点云帧数: {self.frame_count}")
        print(f"  IMU 样本: {self.imu_count}")
        print(f"  说明文件: {readme_file}")
    
    def print_statistics(self):
        """打印统计信息"""
        elapsed = time.time() - self.start_time
        frame_rate = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"运行统计 ({elapsed:.1f} 秒)")
        print(f"{'='*50}")
        print(f"点云帧数: {self.frame_count}")
        print(f"总点数: {self.total_points}")
        print(f"IMU 样本数: {self.imu_count}")
        print(f"点云帧率: {frame_rate:.1f} Hz")
        
        if self.frame_count > 0:
            avg_points = self.total_points / self.frame_count
            print(f"平均每帧点数: {avg_points:.0f}")
        
        # 获取SDK统计信息
        stats = self.get_stats()
        print(f"\nSDK 统计:")
        print(f"  总数据包: {stats['total_packets']}")
        print(f"  丢包数: {stats['dropped_packets']}")
        print(f"  处理时间: {stats['processing_time_ms']:.2f} ms")
        
        if self.save_data:
            print(f"\n数据保存状态:")
            print(f"  保存目录: {self.session_dir}")
            print(f"  点云批次: {len(list(self.pointcloud_dir.glob('*.json')))}")
            print(f"  IMU 记录: {len(self.imu_data)}")

def main():
    """主函数"""
    print("Livox MID-360 数据查看器 (带数据保存)")
    print("="*50)
    
    # 检查配置文件
    config_path = Path("mid360_config.json") 
    if not config_path.exists():
        print("配置文件不存在，创建默认配置...")
        create_default_config(config_path)
        print(f"已创建配置文件: {config_path}")
    
    print("\n网络配置要求:")
    print("1. MID-360 雷达 IP: 192.168.123.120")
    print("2. 主机网络接口 IP: 192.168.123.164")
    print("3. 确保网络连通性")
    
    # 检查网络连通性
    import subprocess
    try:
        result = subprocess.run(['ping', '-c', '1', '192.168.123.120'], 
                               capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✓ 雷达网络连通正常")
        else:
            print("⚠️  雷达网络不通，请检查网络配置")
    except Exception:
        print("⚠️  无法检查网络连通性")
    
    # 询问是否保存数据
    save_data = input("\n是否保存数据到文件? (Y/n): ").strip().lower()
    save_data = save_data != 'n'
    
    if save_data:
        save_dir = input("保存目录 (默认: ./data): ").strip()
        if not save_dir:
            save_dir = "./data"
        print(f"数据将保存到: {save_dir}")
    
    try:
        print(f"\n正在初始化 Livox2...")
        viewer = SimpleDataViewer(config_path, 
                                 save_data=save_data,
                                 save_dir=save_dir if save_data else "data",
                                 host_ip="192.168.123.164",
                                 frame_time=0.1,      # 100ms 帧聚合时间
                                 frame_packets=50,    # 每帧最多50个数据包
                                 enable_filter=True,  # 启用点云过滤
                                 max_range=30.0,      # 最大距离30米
                                 voxel_size=0.1       # 体素化大小0.1米
                                 )
        
        with viewer:
            print("✓ Livox2 初始化成功")
            print("等待雷达连接和数据接收...")
            
            # 数据收集时间
            collection_time = 20.0  # 收集20秒数据
            start_time = time.time()
            
            print(f"开始收集数据（{collection_time}秒）...")
            print("按 Ctrl+C 可提前停止\n")
            
            try:
                while time.time() - start_time < collection_time:
                    time.sleep(0.5)
                    
                    # 每5秒显示一次统计
                    elapsed = time.time() - start_time
                    if int(elapsed) % 5 == 0 and elapsed > 0:
                        viewer.print_statistics()
                        
            except KeyboardInterrupt:
                print("\n用户中断数据收集")
            
            # 最终统计
            print("\n数据收集完成")
            viewer.print_statistics()
            
            # 保存会话数据
            if save_data:
                print("\n正在保存数据...")
                viewer.save_session_summary()
            
            # 数据质量评估
            print(f"\n{'='*50}")
            print("数据质量评估")
            print(f"{'='*50}")
            
            if viewer.frame_count > 0:
                print("✓ 成功接收点云数据")
                
                frame_rate = viewer.frame_count / (time.time() - viewer.start_time)
                if frame_rate >= 5.0:
                    print("✓ 点云帧率正常")
                else:
                    print("⚠️  点云帧率偏低")
                    
                avg_points = viewer.total_points / viewer.frame_count
                if avg_points >= 1000:
                    print("✓ 点云密度正常")
                else:
                    print("⚠️  点云密度偏低")
            else:
                print("✗ 未接收到点云数据")
            
            if viewer.imu_count > 0:
                print("✓ 成功接收 IMU 数据")
            else:
                print("⚠️  未接收到 IMU 数据")
            
            stats = viewer.get_stats()
            if stats['dropped_packets'] == 0:
                print("✓ 无数据包丢失")
            else:
                drop_rate = stats['dropped_packets'] / stats['total_packets'] * 100
                print(f"⚠️  数据包丢失率: {drop_rate:.1f}%")
            
    except Exception as e:
        print(f"✗ 初始化或运行失败: {e}")
        print("\n可能的原因:")
        print("1. Livox-SDK2 动态库未安装或路径不正确")
        print("2. 网络配置不正确")
        print("3. 雷达未连接或未上电")
        print("4. 权限不足（需要sudo运行）")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()