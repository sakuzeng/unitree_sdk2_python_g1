"""
Livox2 点云数据可视化器

使用 Open3D 实时可视化 MID-360 雷达的点云数据
支持颜色映射、坐标轴显示、统计信息展示
"""

import time
import numpy as np
import open3d as o3d
import threading
from pathlib import Path
from collections import deque
from typing import Optional
from livox2_python import Livox2, create_default_config

class PointCloudVisualizer(Livox2):
    """
    点云可视化器，继承 Livox2 类
    提供实时点云显示和交互功能
    """
    
    def __init__(self, *args, **kwargs):
        # 可视化参数
        self.max_points_display = kwargs.pop('max_points_display', 50000)
        self.color_mode = kwargs.pop('color_mode', 'height')  # height, intensity, distance
        self.point_size = kwargs.pop('point_size', 1.0)
        self.show_coordinate_frame = kwargs.pop('show_coordinate_frame', True)
        
        super().__init__(*args, **kwargs)
        
        # 可视化状态
        self.vis = None
        self.pcd = None
        self.coordinate_frame = None
        self.stats_text = None
        self.vis_thread = None
        self.vis_running = False
        
        # 点云数据缓存
        self.point_buffer = deque(maxlen=10)  # 保持最近10帧
        self.display_lock = threading.Lock()
        
        # 统计信息
        self.frame_count = 0
        self.total_points = 0
        self.last_update_time = time.time()
        self.fps = 0.0
        
        # 初始化可视化器
        self._init_visualizer()
    
    def _init_visualizer(self):
        """初始化 Open3D 可视化器"""
        try:
            print("初始化点云可视化器...")
            
            # 创建可视化窗口
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Livox MID-360 点云可视化",
                width=1200,
                height=800
            )
            
            # 创建空点云对象
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)
            
            # 添加坐标系（如果启用）
            if self.show_coordinate_frame:
                self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=2.0, origin=[0, 0, 0]
                )
                self.vis.add_geometry(self.coordinate_frame)
            
            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
            render_option.background_color = np.array([0.05, 0.05, 0.05])  # 深灰色背景
            
            # 设置视角
            view_control = self.vis.get_view_control()
            view_control.set_front([0, 0, 1])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.3)
            
            print("✓ 可视化器初始化完成")
            
        except Exception as e:
            print(f"✗ 可视化器初始化失败: {e}")
            raise
    
    def handle_points(self, xyz: np.ndarray, reflectivity: np.ndarray, 
                     tag: np.ndarray, timestamp: int):
        """处理点云数据并更新可视化"""
        self.frame_count += 1
        self.total_points += len(xyz)
        
        # 计算FPS
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        if elapsed > 1.0:  # 每秒更新一次FPS
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.last_update_time = current_time
            self.frame_count = 0
        
        # 过滤噪声点
        if len(xyz) > 0:
            valid_mask = tag == 0  # 只保留有效点
            xyz_filtered = xyz[valid_mask]
            reflectivity_filtered = reflectivity[valid_mask]
            
            if len(xyz_filtered) > 0:
                # 限制显示点数
                if len(xyz_filtered) > self.max_points_display:
                    indices = np.random.choice(
                        len(xyz_filtered), 
                        self.max_points_display, 
                        replace=False
                    )
                    xyz_filtered = xyz_filtered[indices]
                    reflectivity_filtered = reflectivity_filtered[indices]
                
                # 添加到缓冲区
                with self.display_lock:
                    self.point_buffer.append({
                        'xyz': xyz_filtered.copy(),
                        'reflectivity': reflectivity_filtered.copy(),
                        'timestamp': timestamp
                    })
        
        # 打印统计信息
        if self.frame_count % 10 == 0:
            avg_points = self.total_points / max(self.frame_count, 1)
            print(f"帧 #{self.frame_count}: {len(xyz)} 点, "
                 f"平均: {avg_points:.0f} 点/帧, FPS: {self.fps:.1f}")
    
    def handle_imu(self, imu_data: np.ndarray, timestamp: int):
        """处理 IMU 数据（可选显示）"""
        # IMU 数据可以用于显示运动状态
        pass
    
    def _get_point_colors(self, xyz: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
        """根据颜色模式生成点云颜色"""
        if len(xyz) == 0:
            return np.array([]).reshape(0, 3)
        
        if self.color_mode == 'height':
            # 基于高度着色（Z轴）
            z_vals = xyz[:, 2]
            z_min, z_max = z_vals.min(), z_vals.max()
            if z_max > z_min:
                normalized = (z_vals - z_min) / (z_max - z_min)
            else:
                normalized = np.zeros_like(z_vals)
            
            # 使用彩虹色图：蓝色(低) -> 绿色 -> 红色(高)
            colors = np.zeros((len(xyz), 3))
            colors[:, 0] = np.clip(2 * normalized - 1, 0, 1)  # 红色
            colors[:, 1] = np.clip(2 * (1 - np.abs(normalized - 0.5)), 0, 1)  # 绿色
            colors[:, 2] = np.clip(2 * (1 - normalized) - 1, 0, 1)  # 蓝色
            
        elif self.color_mode == 'intensity':
            # 基于反射强度着色
            normalized = reflectivity / 255.0
            colors = np.zeros((len(xyz), 3))
            colors[:, 0] = normalized  # 红色通道
            colors[:, 1] = normalized  # 绿色通道
            colors[:, 2] = normalized  # 蓝色通道（灰度）
            
        elif self.color_mode == 'distance':
            # 基于距离着色
            distances = np.linalg.norm(xyz, axis=1)
            d_min, d_max = distances.min(), distances.max()
            if d_max > d_min:
                normalized = (distances - d_min) / (d_max - d_min)
            else:
                normalized = np.zeros_like(distances)
            
            # 近距离蓝色，远距离红色
            colors = np.zeros((len(xyz), 3))
            colors[:, 0] = normalized  # 红色
            colors[:, 1] = 1.0 - normalized  # 绿色
            colors[:, 2] = 1.0 - normalized  # 蓝色
            
        else:
            # 默认白色
            colors = np.ones((len(xyz), 3))
        
        return colors
    
    def _update_visualization(self):
        """更新可视化显示"""
        try:
            if not self.point_buffer:
                return
            
            # 合并最近的点云数据
            with self.display_lock:
                if not self.point_buffer:
                    return
                
                # 使用最新的几帧数据
                recent_frames = list(self.point_buffer)[-3:]  # 最近3帧
                
                all_xyz = []
                all_reflectivity = []
                
                for frame in recent_frames:
                    all_xyz.append(frame['xyz'])
                    all_reflectivity.append(frame['reflectivity'])
                
                if not all_xyz:
                    return
                
                combined_xyz = np.concatenate(all_xyz, axis=0)
                combined_reflectivity = np.concatenate(all_reflectivity, axis=0)
            
            # 生成颜色
            colors = self._get_point_colors(combined_xyz, combined_reflectivity)
            
            # 更新点云
            self.pcd.points = o3d.utility.Vector3dVector(combined_xyz)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 更新几何体
            self.vis.update_geometry(self.pcd)
            
            # 更新标题显示统计信息
            stats_info = (
                f"点数: {len(combined_xyz):,} | "
                f"FPS: {self.fps:.1f} | "
                f"模式: {self.color_mode} | "
                f"范围: {self._max_range:.1f}m"
            )
            
            # 渲染
            self.vis.poll_events()
            self.vis.update_renderer()
            
        except Exception as e:
            print(f"可视化更新异常: {e}")
    
    def _visualization_loop(self):
        """可视化循环线程"""
        print("启动可视化循环...")
        
        try:
            while self.vis_running and self._running:
                self._update_visualization()
                time.sleep(0.05)  # 20 FPS 更新率
                
        except Exception as e:
            print(f"可视化循环异常: {e}")
        finally:
            print("可视化循环退出")
    
    def start_visualization(self):
        """启动可视化"""
        if self.vis_running:
            return
        
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.vis_thread.start()
        print("✓ 可视化线程已启动")
    
    def stop_visualization(self):
        """停止可视化"""
        if not self.vis_running:
            return
        
        print("正在停止可视化...")
        self.vis_running = False
        
        if self.vis_thread and self.vis_thread.is_alive():
            self.vis_thread.join(timeout=2.0)
        
        if self.vis:
            self.vis.destroy_window()
            self.vis = None
        
        print("✓ 可视化已停止")
    
    def change_color_mode(self, mode: str):
        """切换颜色模式"""
        if mode in ['height', 'intensity', 'distance']:
            self.color_mode = mode
            print(f"颜色模式切换为: {mode}")
        else:
            print(f"无效的颜色模式: {mode}")
    
    def set_point_size(self, size: float):
        """设置点大小"""
        self.point_size = max(0.1, min(10.0, size))
        if self.vis:
            render_option = self.vis.get_render_option()
            render_option.point_size = self.point_size
        print(f"点大小设置为: {self.point_size}")
    
    def shutdown(self):
        """安全关闭"""
        self.stop_visualization()
        super().shutdown()

def main():
    """主函数 - 演示点云可视化"""
    print("Livox MID-360 点云可视化器")
    print("="*50)
    
    # 检查配置文件
    config_path = Path("mid360_config.json")
    if not config_path.exists():
        print("配置文件不存在，创建默认配置...")
        create_default_config(config_path)
        print(f"已创建配置文件: {config_path}")
    
    # 显示使用说明
    print("\n使用说明:")
    print("- 鼠标左键拖动: 旋转视角")
    print("- 鼠标右键拖动: 平移视角")
    print("- 滚轮: 缩放")
    print("- 按 'q' 或关闭窗口: 退出")
    
    print("\n可视化设置:")
    print("1. 颜色模式: height (基于高度)")
    print("2. 最大显示点数: 50,000")
    print("3. 更新频率: 20 FPS")
    
    # 可视化参数配置
    viz_config = {
        'max_points_display': 50000,    # 最大显示点数
        'color_mode': 'height',         # 颜色模式
        'point_size': 2.0,              # 点大小
        'show_coordinate_frame': True,  # 显示坐标轴
    }
    
    try:
        print(f"\n正在初始化可视化器...")
        visualizer = PointCloudVisualizer(
            config_path,
            host_ip="192.168.123.164",
            frame_time=0.1,         # 100ms 帧聚合
            frame_packets=50,       # 每帧50包
            enable_filter=True,     # 启用过滤
            max_range=50.0,         # 最大50米
            voxel_size=0.1,         # 体素大小
            **viz_config
        )
        
        with visualizer:
            print("✓ 可视化器初始化成功")
            print("等待雷达连接...")
            
            # 启动可视化
            visualizer.start_visualization()
            time.sleep(2.0)  # 等待初始化
            
            print("✓ 开始可视化，等待点云数据...")
            print("按 Ctrl+C 退出")
            
            # 交互式控制
            try:
                while visualizer.is_running() and visualizer.vis_running:
                    time.sleep(0.1)
                    
                    # 简单的键盘控制（可选扩展）
                    # 这里可以添加键盘输入处理来切换可视化模式
                    
            except KeyboardInterrupt:
                print("\n收到退出信号...")
            
            print("\n可视化统计:")
            stats = visualizer.get_stats()
            print(f"  总数据包: {stats['total_packets']}")
            print(f"  总点数: {stats['total_points']:,}")
            print(f"  处理时间: {stats['processing_time_ms']:.2f} ms")
            print(f"  丢包率: {stats['dropped_packets']}")
            
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        print("\n可能的原因:")
        print("1. Open3D 未正确安装")
        print("2. 图形界面环境不可用")
        print("3. Livox-SDK2 库未找到")
        print("4. 雷达网络连接问题")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()