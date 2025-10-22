#!/usr/bin/env python3
"""
convert_slam_to_pgm.py - 将SLAM生成的栅格地图转换为PGM格式的独立工具

功能:
- 读取保存的numpy数组格式的栅格地图
- 转换为ROS兼容的PGM格式
- 生成对应的YAML配置文件
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import json
import time

def convert_grid_to_pgm(grid_data: np.ndarray, output_path: str, 
                       resolution: float = 0.05, origin: tuple = (0.0, 0.0)):
    """
    将栅格数据转换为PGM格式
    
    Args:
        grid_data: 栅格数据 (0=自由, 128=未知, 255=占用)
        output_path: 输出文件路径（不含扩展名）
        resolution: 栅格分辨率 (m/pixel)
        origin: 地图原点坐标 (x, y)
    """
    output_path = Path(output_path)
    
    # 转换网格格式 (PGM: 0=占用, 254=自由, 205=未知)
    pgm_grid = np.zeros_like(grid_data)
    pgm_grid[grid_data == 0] = 254		# 自由空间
    pgm_grid[grid_data == 128] = 205	# 未知区域
    pgm_grid[grid_data == 255] = 0		# 占用区域
    
    # 保存PGM文件
    pgm_path = output_path.with_suffix('.pgm')
    image = Image.fromarray(pgm_grid, mode='L')
    image.save(pgm_path)
    
    # 生成YAML配置文件
    yaml_path = output_path.with_suffix('.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"image: {pgm_path.name}\n")
        f.write(f"resolution: {resolution:.6f}\n")
        f.write(f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, 0.0]\n")
        f.write("negate: 0\n")
        f.write("occupied_thresh: 0.65\n")
        f.write("free_thresh: 0.196\n")
    
    print(f"[INFO] PGM文件已保存: {pgm_path}")
    print(f"[INFO] YAML文件已保存: {yaml_path}")
    
    return str(pgm_path), str(yaml_path)

def load_numpy_grid(file_path: str) -> np.ndarray:
    """加载numpy格式的栅格数据"""
    try:
        grid = np.load(file_path)
        print(f"[INFO] 加载栅格数据: {file_path}")
        print(f"[INFO] 栅格尺寸: {grid.shape}")
        print(f"[INFO] 数据类型: {grid.dtype}")
        print(f"[INFO] 数值范围: [{grid.min()}, {grid.max()}]")
        return grid
    except Exception as e:
        print(f"[ERROR] 加载栅格数据失败: {e}")
        return None

def create_test_grid(size: int = 400) -> np.ndarray:
    """创建测试栅格数据"""
    grid = np.full((size, size), 128, dtype=np.uint8)  # 初始化为未知
    
    # 添加一些占用区域（障碍物）
    center = size // 2
    
    # 中央正方形障碍物
    grid[center-20:center+20, center-20:center+20] = 255
    
    # 边界墙
    grid[10:20, 50:size-50] = 255  # 上墙
    grid[size-20:size-10, 50:size-50] = 255  # 下墙
    grid[50:size-50, 10:20] = 255  # 左墙
    grid[50:size-50, size-20:size-10] = 255  # 右墙
    
    # 自由空间
    grid[50:size-50, 50:size-50] = 0
    grid[center-20:center+20, center-20:center+20] = 255  # 恢复中央障碍物
    
    print(f"[INFO] 创建测试栅格: {size}x{size}")
    print(f"[INFO] 占用像素: {np.sum(grid == 255)}")
    print(f"[INFO] 自由像素: {np.sum(grid == 0)}")
    print(f"[INFO] 未知像素: {np.sum(grid == 128)}")
    
    return grid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将SLAM栅格地图转换为PGM格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", "-i", type=str, help="输入栅格文件路径 (.npy)")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出文件路径（不含扩展名）")
    parser.add_argument("--resolution", "-r", type=float, default=0.05, help="栅格分辨率 (m/pixel)")
    parser.add_argument("--origin-x", type=float, default=0.0, help="地图原点X坐标")
    parser.add_argument("--origin-y", type=float, default=0.0, help="地图原点Y坐标")
    parser.add_argument("--create-test", action="store_true", help="创建测试栅格数据")
    parser.add_argument("--test-size", type=int, default=400, help="测试栅格大小")
    
    args = parser.parse_args()
    
    # 加载或创建栅格数据
    if args.create_test:
        print("[INFO] 创建测试栅格数据...")
        grid_data = create_test_grid(args.test_size)
    elif args.input:
        grid_data = load_numpy_grid(args.input)
        if grid_data is None:
            return
    else:
        print("[ERROR] 请指定输入文件 (--input) 或使用 --create-test 创建测试数据")
        return
    
    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为PGM格式
    origin = (args.origin_x, args.origin_y)
    pgm_path, yaml_path = convert_grid_to_pgm(
        grid_data, 
        str(output_path), 
        args.resolution, 
        origin
    )
    
    print(f"[INFO] 转换完成!")
    print(f"[INFO] 可以使用ROS map_server加载: rosrun map_server map_server {yaml_path}")

if __name__ == "__main__":
    main()