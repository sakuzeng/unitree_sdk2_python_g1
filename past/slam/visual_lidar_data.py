"""
从保存的 Livox MID-360 点云数据 (.npy) 绘制点云
功能：
- 加载单个或多个 .npy 文件
- 使用 Open3D 可视化点云
- 可选：根据 reflectivity 设置颜色
使用方法：
    python visualize_saved_points.py
"""

import numpy as np
import open3d as o3d
from glob import glob
from pathlib import Path

def visualize_point_cloud(file_path: str, use_reflectivity: bool = False):
    """
    加载并可视化单个点云文件
    :param file_path: .npy 文件路径
    :param use_reflectivity: 是否使用 reflectivity 设置颜色
    """
    # 加载数据
    data = np.load(file_path)
    xyz = data[:, :3]  # 提取 x, y, z

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 可选：使用 reflectivity 设置颜色
    if use_reflectivity and data.shape[1] >= 4:
        reflectivity = data[:, 3]
        colors = np.zeros_like(xyz)
        colors[:, 0] = reflectivity / 255.0  # 红色通道表示强度
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {file_path}", width=1280, height=720)
    vis.add_geometry(pcd)

    # 添加坐标系
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(origin_frame)

    # 渲染
    vis.run()
    vis.destroy_window()

def visualize_multiple_point_clouds(directory: str, use_reflectivity: bool = False):
    """
    合并并可视化多个点云文件
    :param directory: 包含 .npy 文件的目录
    :param use_reflectivity: 是否使用 reflectivity 设置颜色
    """
    files = sorted(glob(str(Path(directory) / "point_cloud_*.npy")))
    if not files:
        print(f"目录 {directory} 中未找到点云文件")
        return

    # 合并所有点云
    all_data = []
    for f in files[:10]:  # 限制最多 10 帧以避免内存溢出
        data = np.load(f)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    xyz = all_data[:, :3]

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 可选：使用 reflectivity 设置颜色
    if use_reflectivity and all_data.shape[1] >= 4:
        reflectivity = all_data[:, 3]
        colors = np.zeros_like(xyz)
        colors[:, 0] = reflectivity / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Merged Point Clouds: {directory}", width=1280, height=720)
    vis.add_geometry(pcd)
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(origin_frame)

    # 渲染
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 示例：可视化单个文件
    single_file = "lidar_data/point_cloud_1634567890.123.npy"  # 替换为实际文件路径
    if Path(single_file).exists():
        visualize_point_cloud(single_file, use_reflectivity=True)

    # 示例：可视化目录中的所有点云
    directory = "lidar_data"
    visualize_multiple_point_clouds(directory, use_reflectivity=True)