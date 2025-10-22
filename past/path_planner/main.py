#!/usr/bin/env python3
"""
SLAM路径规划系统主程序 - 深度优化版
与增强的系统组件完全集成
"""
import sys
import time
import signal
import logging
import threading
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from config import (
    SystemConfig, GridConfig, PathPlannerConfig,
    CoordinateConfig, RobotConfig, LidarConfig,
    KissICPConfig, VisualizationConfig
)
from integrated_system import EnhancedIntegratedSLAMSystem, SystemStatus
from visualizer import IntelligentVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('slam_system.log')
    ]
)
logger = logging.getLogger(__name__)

class AdvancedSLAMApplication:
    """高级SLAM应用程序 - 完整功能版本"""
    
    def __init__(self, args):
        self.args = args
        self.system: Optional[EnhancedIntegratedSLAMSystem] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # 性能监控
        self.performance_monitor = None
        self.monitor_thread = None
        
        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("[AdvancedSLAMApp] 应用程序初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"[AdvancedSLAMApp] 接收到信号 {signum}，正在关闭...")
        self.shutdown()
    
    def create_optimized_config(self) -> SystemConfig:
        """创建优化的系统配置"""
        # 根据命令行参数调整配置
        
        # 网格配置 - 高性能版本
        grid_config = GridConfig(
            grid_size=self.args.grid_size,
            grid_resolution=400,  # 高分辨率
            resolution=0.05,      # 5cm分辨率
            height_filter_min=-0.3,
            height_filter_max=2.5,
            prob_occ=0.8,
            prob_free=0.2,
            prob_prior=0.5,
            hit_threshold=0.7,
            free_threshold=0.3,
            max_range=self.args.max_range,
            use_odometry=not self.args.no_odometry,
            voxel_size=0.03,      # 更小的体素大小
            outlier_removal=True,
            ground_removal=False,
            map_update_threshold=0.02
        )
        
        # KISS-ICP配置 - 高精度版本
        kiss_icp_config = KissICPConfig(
            voxel_size=0.03,
            max_range=self.args.max_range,
            min_range=0.1,
            keyframe_distance=0.3,
            keyframe_angle=0.15,
            max_iterations=100,
            convergence_threshold=5e-4,
            max_correspondence_distance=0.3,
            local_map_size=15,
            map_size_limit=50,
            adaptive_threshold=True,
            initial_threshold=1.5,
            min_motion_th=0.02,
            use_deskew=True,
            max_num_threads=0
        )
        
        # 路径规划配置
        planner_config = PathPlannerConfig(
            max_velocity=self.args.max_velocity,
            max_angular_velocity=1.5,
            lookahead_distance=1.5,
            goal_tolerance=0.2,
            safety_margin=0.25,
            replan_frequency=3.0,
            obstacle_inflation=0.4,
            dynamic_obstacle_threshold=3
        )
        
        # 坐标系配置
        coordinate_config = CoordinateConfig(
            coordinate_frame="kiss_icp",
            origin_auto_set=True,
            coordinate_publish_rate=20.0,
            use_lidar_odometry=True
        )
        
        # 机器人配置
        robot_config = RobotConfig(
            interface=self.args.interface,
            control_frequency=20.0,
            wheelbase=0.3,
            max_acceleration=1.5,
            enabled=not self.args.no_robot_control
        )
        
        # 激光雷达配置
        lidar_config = LidarConfig(
            config_path=self.args.config,
            host_ip=self.args.host_ip,
            min_distance=0.3,
            max_distance=self.args.max_range,
            angle_filter_enabled=False,
            angle_min=-180.0,
            angle_max=180.0
        )
        
        # 可视化配置 - 修复参数
        visualization_config = VisualizationConfig(
            window_width=1200,
            window_height=800,
            render_frequency=30.0,
            grid_line_spacing=1.0,
            show_grid_lines=True,
            grid_line_color=(80, 80, 80),
            grid_line_thickness=1,
            trajectory_color=(0, 255, 0),
            path_thickness=3,
            robot_size=8,
            goal_size=12,
            coordinate_text_size=0.5,
            show_coordinates=True,
            show_trajectory=True,
            show_keyframes=False,
            show_map_points=False,
            show_path=True,
            show_robot_orientation=True,
            show_quality_info=True,
            show_occupancy_grid=True,  # 现在这个参数存在了
            auto_center=True,
            view_range=15.0
        )
        
        return SystemConfig(
            grid=grid_config,
            planner=planner_config,
            coordinate=coordinate_config,
            robot=robot_config,
            lidar=lidar_config,
            kiss_icp=kiss_icp_config,
            visualization=visualization_config,
            debug_mode=self.args.debug,
            update_frequency=20.0,
            visualization_scale=4,
            global_map_size=50.0,
            mount_correction=self.args.mount_correction
        )
    
    def validate_configuration(self, config: SystemConfig) -> bool:
        """验证配置有效性"""
        try:
            # 检查配置文件
            config_path = Path(config.lidar.config_path)
            if not config_path.exists():
                logger.error(f"[AdvancedSLAMApp] 激光雷达配置文件不存在: {config_path}")
                return False
            
            # 检查网络配置
            if not config.lidar.host_ip or not config.robot.interface:
                logger.error("[AdvancedSLAMApp] 网络配置不完整")
                return False
            
            # 检查参数范围
            if config.grid.resolution <= 0 or config.grid.grid_size <= 0:
                logger.error("[AdvancedSLAMApp] 网格参数无效")
                return False
            
            logger.info("[AdvancedSLAMApp] 配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"[AdvancedSLAMApp] 配置验证失败: {e}")
            return False
    
    def start(self) -> bool:
        """启动系统"""
        if self.is_running:
            logger.warning("[AdvancedSLAMApp] 系统已在运行")
            return False
        
        logger.info("[AdvancedSLAMApp] 启动高级SLAM系统...")
        
        try:
            # 创建和验证配置
            config = self.create_optimized_config()
            if not self.validate_configuration(config):
                return False
            
            # 打印配置摘要
            self._print_config_summary(config)
            
            # 初始化系统
            self.system = EnhancedIntegratedSLAMSystem(config)
            
            # 启动系统
            self.system.start()
            
            # 启动性能监控
            if self.args.monitor_performance:
                self._start_performance_monitor()
            
            self.is_running = True
            logger.info("[AdvancedSLAMApp] 系统启动成功")
            
            # 启动主循环
            self._run_main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"[AdvancedSLAMApp] 系统启动失败: {e}")
            self.shutdown()
            return False
    
    def _print_config_summary(self, config: SystemConfig):
        """打印配置摘要"""
        logger.info("=== 系统配置摘要 ===")
        logger.info(f"网格大小: {config.grid.grid_size}m x {config.grid.grid_size}m")
        logger.info(f"网格分辨率: {config.grid.resolution}m/cell")
        logger.info(f"激光雷达范围: {config.grid.max_range}m")
        logger.info(f"KISS-ICP体素大小: {config.kiss_icp.voxel_size}m")
        logger.info(f"最大速度: {config.planner.max_velocity}m/s")
        logger.info(f"激光雷达IP: {config.lidar.host_ip}")
        logger.info(f"网络接口: {config.robot.interface}")
        logger.info(f"机器人控制: {'启用' if config.robot.enabled else '禁用'}")
        logger.info(f"调试模式: {'启用' if config.debug_mode else '禁用'}")
        logger.info(f"坐标校正: {config.mount_correction}")
        logger.info("================")
    
    def _start_performance_monitor(self):
        """启动性能监控"""
        def monitor_loop():
            logger.info("[AdvancedSLAMApp] 性能监控启动")
            
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    if self.system:
                        stats = self.system.get_comprehensive_statistics()
                        self._log_performance_stats(stats)
                    
                    time.sleep(10.0)  # 每10秒监控一次
                    
                except Exception as e:
                    logger.error(f"[AdvancedSLAMApp] 性能监控错误: {e}")
                    time.sleep(5.0)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def _log_performance_stats(self, stats: Dict[str, Any]):
        """记录性能统计"""
        try:
            system_stats = stats.get('system', {})
            slam_stats = stats.get('slam', {})
            viz_stats = stats.get('visualization', {})
            
            logger.info(f"[性能监控] "
                       f"FPS: {system_stats.get('processing_fps', 0):.1f}, "
                       f"帧数: {system_stats.get('frame_count', 0)}, "
                       f"SLAM质量: {system_stats.get('slam_quality', 0):.2f}, "
                       f"渲染FPS: {viz_stats.get('fps', 0):.1f}")
            
        except Exception as e:
            logger.error(f"[AdvancedSLAMApp] 性能统计记录失败: {e}")
    
    def _run_main_loop(self):
        """运行主循环"""
        logger.info("[AdvancedSLAMApp] 进入主循环")
        
        # 如果启用了可视化，则不需要额外的主循环
        # IntelligentVisualizer 会在其渲染线程中处理用户交互
        if self.system and hasattr(self.system, 'visualizer'):
            try:
                # 等待用户退出
                logger.info("[AdvancedSLAMApp] 系统运行中，按 ESC 或 Ctrl+C 退出")
                
                while self.is_running and not self.shutdown_event.is_set():
                    time.sleep(0.1)
                    
                    # 检查系统状态
                    if not self.system.status.is_running:
                        logger.warning("[AdvancedSLAMApp] 系统组件已停止，退出主循环")
                        break
            
            except KeyboardInterrupt:
                logger.info("[AdvancedSLAMApp] 用户中断")
            except Exception as e:
                logger.error(f"[AdvancedSLAMApp] 主循环异常: {e}")
        else:
            # 如果没有可视化，提供简单的命令行界面
            self._run_cli_mode()
    
    def _run_cli_mode(self):
        """运行命令行模式"""
        logger.info("[AdvancedSLAMApp] 命令行模式启动")
        logger.info("可用命令: status, save, reset, goal <x> <y>, stop, quit")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    cmd = input("> ").strip().lower().split()
                    if not cmd:
                        continue
                    
                    if cmd[0] == 'quit' or cmd[0] == 'q':
                        break
                    elif cmd[0] == 'status':
                        self._print_system_status()
                    elif cmd[0] == 'save':
                        self.system.save_data()
                        print("数据已保存")
                    elif cmd[0] == 'reset':
                        self.system.reset_slam()
                        print("SLAM系统已重置")
                    elif cmd[0] == 'goal' and len(cmd) >= 3:
                        try:
                            x, y = float(cmd[1]), float(cmd[2])
                            self.system.set_goal(x, y)
                            print(f"目标已设置: ({x:.2f}, {y:.2f})")
                        except ValueError:
                            print("无效的坐标格式")
                    elif cmd[0] == 'stop':
                        self.system.stop_following()
                        print("路径跟踪已停止")
                    else:
                        print("未知命令")
                
                except EOFError:
                    break
                except Exception as e:
                    logger.error(f"[AdvancedSLAMApp] 命令处理错误: {e}")
        
        except KeyboardInterrupt:
            logger.info("[AdvancedSLAMApp] 用户中断")
    
    def _print_system_status(self):
        """打印系统状态"""
        if not self.system:
            print("系统未启动")
            return
        
        try:
            status = self.system.get_system_status()
            stats = self.system.get_comprehensive_statistics()
            
            print(f"\n=== 系统状态 ===")
            print(f"运行状态: {'运行中' if status.is_running else '停止'}")
            print(f"SLAM状态: {'活跃' if status.slam_active else '非活跃'}")
            print(f"激光雷达: {'连接' if status.lidar_connected else '断开'}")
            print(f"处理帧数: {status.frame_count}")
            print(f"处理FPS: {status.processing_fps:.1f}")
            print(f"SLAM质量: {status.slam_quality:.2f}")
            print(f"运行时间: {status.uptime:.1f}s")
            
            if status.current_pose is not None:
                pos = status.current_pose[:3, 3]
                print(f"当前位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            
            if status.following_path:
                print(f"路径跟踪: 进行中 (路径长度: {status.path_length})")
            else:
                print("路径跟踪: 停止")
            
            print("===============\n")
            
        except Exception as e:
            logger.error(f"[AdvancedSLAMApp] 获取系统状态失败: {e}")
    
    def shutdown(self):
        """关闭系统"""
        if not self.is_running:
            return
        
        logger.info("[AdvancedSLAMApp] 正在关闭应用程序...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # 关闭性能监控
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        
        # 关闭系统
        if self.system:
            try:
                self.system.shutdown()
                logger.info("[AdvancedSLAMApp] SLAM系统已关闭")
            except Exception as e:
                logger.error(f"[AdvancedSLAMApp] 关闭SLAM系统失败: {e}")
        
        logger.info("[AdvancedSLAMApp] 应用程序关闭完成")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="高级SLAM路径规划系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基础配置
    parser.add_argument("--config", default="mid360_config.json",
                       help="Livox配置文件路径")
    parser.add_argument("--host-ip", default="192.168.123.164",
                       help="主机IP地址")
    parser.add_argument("--interface", default="eth0",
                       help="网络接口")
    
    # 系统参数
    parser.add_argument("--grid-size", type=float, default=20.0,
                       help="网格大小(米)")
    parser.add_argument("--max-range", type=float, default=30.0,
                       help="最大激光雷达范围(米)")
    parser.add_argument("--max-velocity", type=float, default=0.5,
                       help="最大速度(m/s)")
    
    # 功能开关
    parser.add_argument("--no-odometry", action="store_true",
                       help="禁用里程计")
    parser.add_argument("--no-robot-control", action="store_true",
                       help="禁用机器人控制")
    parser.add_argument("--no-visualization", action="store_true",
                       help="禁用可视化")
    
    # 高级选项
    parser.add_argument("--mount-correction", default="identity",
                       choices=["identity", "upside_down", "rotated_90", "simple_flip_y"],
                       help="激光雷达挂载校正")
    parser.add_argument("--debug", action="store_true",
                       help="启用调试模式")
    parser.add_argument("--monitor-performance", action="store_true",
                       help="启用性能监控")
    
    # 日志配置
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    return parser.parse_args()

def print_banner():
    """打印启动横幅"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                  Unitree G1 SLAM Navigation System           ║
    ║                     高级路径规划系统 v2.0                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  功能特性:                                                    ║
    ║  • Livox Mid-360 激光雷达 SLAM 建图                          ║
    ║  • KISS-ICP 高精度位姿估计                                   ║
    ║  • 智能地图管理和可视化                                      ║
    ║  • A* + Pure Pursuit 路径规划                                ║
    ║  • 实时性能监控和统计                                        ║
    ║  • 多线程并行处理架构                                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 打印启动信息
    print_banner()
    logger.info("[Main] 启动高级SLAM路径规划系统")
    
    # 创建应用程序
    app = AdvancedSLAMApplication(args)
    
    try:
        # 启动系统
        success = app.start()
        
        if success:
            logger.info("[Main] 系统运行完成")
            return 0
        else:
            logger.error("[Main] 系统启动失败")
            return 1
    
    except Exception as e:
        logger.error(f"[Main] 程序异常: {e}", exc_info=True)
        return 1
    
    finally:
        # 确保系统正确关闭
        app.shutdown()

if __name__ == "__main__":
    sys.exit(main())