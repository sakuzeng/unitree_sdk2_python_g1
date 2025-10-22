#!/usr/bin/env python3
"""
主程序入口
"""
import argparse
import sys
import time
import cv2
from pathlib import Path

from config import GridConfig, PathPlannerConfig
from integrated_system import IntegratedSLAMSystem
from visualizer import render_slam_system, add_status_info, mouse_callback

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化的集成SLAM路径规划系统")
    parser.add_argument("--config", default="mid360_config.json", help="Livox配置文件")
    parser.add_argument("--host-ip", default="192.168.123.164", help="主机IP")
    parser.add_argument("--interface", default="eth0", help="网络接口")
    parser.add_argument("--grid-size", type=float, default=20.0, help="网格大小（米）")
    parser.add_argument("--max-velocity", type=float, default=0.3, help="最大速度")
    parser.add_argument("--no-odometry", action="store_true", help="禁用里程计")
    args = parser.parse_args()
    
    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        possible_paths = [
            Path.home() / "livox_cfg" / "MID360_config.json",
            Path.cwd() / "config" / "mid360_config.json"
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            print(f"[ERROR] 找不到配置文件")
            sys.exit(1)
    
    # 创建配置
    grid_config = GridConfig(
        grid_size=args.grid_size,
        use_odometry=not args.no_odometry
    )
    
    planner_config = PathPlannerConfig(
        max_velocity=args.max_velocity
    )
    
    print(f"[INFO] 配置文件: {config_path}")
    print(f"[INFO] 网格尺寸: {grid_config.grid_size}m")
    print(f"[INFO] 最大速度: {planner_config.max_velocity} m/s")
    
    # 初始化系统
    try:
        slam_system = IntegratedSLAMSystem(
            str(config_path), args.host_ip, grid_config, planner_config, args.interface
        )
        print("[INFO] 系统初始化成功")
    except Exception as e:
        print(f"[ERROR] 初始化失败: {e}")
        sys.exit(1)
    
    # 主循环
    try:
        print("[INFO] 系统启动，左键设置目标，ESC退出")
        window_name = "Optimized SLAM System"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback, slam_system)
        
        while True:
            # 更新控制
            slam_system.update_control()
            
            # 渲染显示
            canvas = render_slam_system(slam_system)
            status = slam_system.get_status()
            canvas = add_status_info(canvas, status)
            
            cv2.imshow(window_name, canvas)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC或Q退出
                break
            elif key == ord(' '):  # 空格停止
                slam_system.stop_following()
            elif key == ord('s'):  # S保存地图
                slam_system.save_map()
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[INFO] 接收到中断信号")
    finally:
        print("[INFO] 正在关闭...")
        slam_system.stop_following()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()