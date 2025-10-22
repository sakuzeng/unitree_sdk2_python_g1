"""
系统配置模块 - 深度优化版
支持智能SLAM系统的完整配置管理
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path

@dataclass
class GridConfig:
    """网格配置"""
    grid_size: float = 20.0                    # 网格大小(米)
    grid_resolution: int = 400                 # 网格分辨率(格数)
    resolution: float = 0.05                   # 单元格大小(米/格)
    height_filter_min: float = -0.5            # 最低高度过滤(米)
    height_filter_max: float = 2.0             # 最高高度过滤(米)
    prob_occ: float = 0.7                      # 占用概率
    prob_free: float = 0.3                     # 自由概率
    prob_prior: float = 0.5                    # 先验概率
    hit_threshold: float = 0.7                 # 击中阈值
    free_threshold: float = 0.3                # 自由阈值
    max_range: float = 30.0                    # 最大激光雷达范围(米)
    use_odometry: bool = True                  # 使用里程计
    voxel_size: float = 0.1                    # 体素大小(米)
    outlier_removal: bool = True               # 离群点移除
    ground_removal: bool = False               # 地面移除
    map_update_threshold: float = 0.1          # 地图更新阈值

@dataclass
class PathPlannerConfig:
    """路径规划配置"""
    max_velocity: float = 0.5                  # 最大速度(m/s)
    max_angular_velocity: float = 1.0          # 最大角速度(rad/s)
    lookahead_distance: float = 1.0            # 前瞻距离(米)
    goal_tolerance: float = 0.3                # 目标容差(米)
    safety_margin: float = 0.2                 # 安全边距(米)
    replan_frequency: float = 2.0              # 重规划频率(Hz)
    obstacle_inflation: float = 0.3            # 障碍物膨胀(米)
    dynamic_obstacle_threshold: int = 5        # 动态障碍物阈值
    heuristic_weight: float = 1.0              # 启发式权重
    smoothing_iterations: int = 3              # 平滑迭代次数

@dataclass
class CoordinateConfig:
    """坐标系配置"""
    coordinate_frame: str = "kiss_icp"         # 坐标系框架
    origin_auto_set: bool = True               # 自动设置原点
    coordinate_publish_rate: float = 20.0      # 坐标发布频率(Hz)
    use_lidar_odometry: bool = True            # 使用激光雷达里程计

@dataclass
class RobotConfig:
    """机器人配置"""
    interface: str = "eth0"                    # 网络接口
    control_frequency: float = 20.0            # 控制频率(Hz)
    wheelbase: float = 0.3                     # 轴距(米)
    max_acceleration: float = 1.0              # 最大加速度(m/s²)
    max_velocity: float = 0.5                  # 最大速度(m/s)
    max_angular_velocity: float = 1.0          # 最大角速度(rad/s)
    enabled: bool = True                       # 启用机器人控制

@dataclass
class LidarConfig:
    """激光雷达配置"""
    config_path: str = "mid360_config.json"    # 配置文件路径
    host_ip: str = "192.168.123.164"          # 主机IP
    min_distance: float = 0.1                  # 最小距离(米)
    max_distance: float = 30.0                 # 最大距离(米)
    angle_filter_enabled: bool = False         # 角度过滤启用
    angle_min: float = -180.0                  # 最小角度(度)
    angle_max: float = 180.0                   # 最大角度(度)

@dataclass
class KissICPConfig:
    """KISS-ICP配置"""
    voxel_size: float = 0.1                    # 体素大小(米)
    max_range: float = 30.0                    # 最大范围(米)
    min_range: float = 0.1                     # 最小范围(米)
    keyframe_distance: float = 0.5             # 关键帧距离(米)
    keyframe_angle: float = 0.2                # 关键帧角度(弧度)
    max_iterations: int = 50                   # 最大迭代次数
    convergence_threshold: float = 1e-3        # 收敛阈值
    max_correspondence_distance: float = 0.5   # 最大对应距离(米)
    local_map_size: int = 10                   # 局部地图大小
    map_size_limit: int = 100                  # 地图大小限制
    adaptive_threshold: bool = True            # 自适应阈值
    initial_threshold: float = 2.0             # 初始阈值
    min_motion_th: float = 0.1                 # 最小运动阈值
    use_deskew: bool = True                    # 使用去偏斜
    max_num_threads: int = 4                   # 最大线程数

@dataclass
class VisualizationConfig:
    """可视化配置"""
    window_width: int = 1200                   # 窗口宽度
    window_height: int = 800                   # 窗口高度
    render_frequency: float = 30.0             # 渲染频率(Hz)
    grid_line_spacing: float = 1.0             # 网格线间距(米)
    show_grid_lines: bool = True               # 显示网格线
    grid_line_color: Tuple[int, int, int] = (80, 80, 80)  # 网格线颜色
    grid_line_thickness: int = 1               # 网格线粗细
    trajectory_color: Tuple[int, int, int] = (0, 255, 0)  # 轨迹颜色
    path_thickness: int = 3                    # 路径粗细
    robot_size: int = 8                        # 机器人大小
    goal_size: int = 12                        # 目标大小
    coordinate_text_size: float = 0.5          # 坐标文本大小
    show_coordinates: bool = True              # 显示坐标
    show_trajectory: bool = True               # 显示轨迹
    show_keyframes: bool = False                # 显示关键帧
    show_map_points: bool = False               # 显示地图点
    show_path: bool = True                     # 显示路径
    show_robot_orientation: bool = True        # 显示机器人方向
    show_quality_info: bool = True             # 显示质量信息
    show_occupancy_grid: bool = True          # 添加占用网格显示选项
    auto_center: bool = True                   # 自动居中
    view_range: float = 15.0                   # 视图范围(米)

@dataclass
class MapManagerConfig:
    """地图管理器配置"""
    max_map_size: float = 200.0                # 最大地图大小(米)
    map_resolution: float = 0.05               # 地图分辨率(米/像素)
    keyframe_distance_threshold: float = 1.0   # 关键帧距离阈值(米)
    loop_closure_threshold: float = 2.0        # 回环检测阈值(米)
    map_compression_enabled: bool = True       # 启用地图压缩
    map_save_frequency: float = 10.0           # 地图保存频率(Hz)

@dataclass
class PerformanceConfig:
    """性能配置"""
    max_cpu_usage: float = 80.0                # 最大CPU使用率(%)
    max_memory_usage: float = 80.0             # 最大内存使用率(%)
    processing_threads: int = 4                # 处理线程数
    visualization_threads: int = 2             # 可视化线程数
    enable_profiling: bool = False             # 启用性能分析

@dataclass
class LoggingConfig:
    """日志配置"""
    log_level: str = "INFO"                    # 日志级别
    log_to_file: bool = True                   # 写入文件
    log_file_path: str = "slam_system.log"     # 日志文件路径
    max_log_file_size: int = 10 * 1024 * 1024  # 最大日志文件大小(字节)
    log_backup_count: int = 5                  # 日志备份数量

@dataclass
class SystemConfig:
    """系统总配置 - 深度集成版"""
    # 核心组件配置
    grid: GridConfig = field(default_factory=GridConfig)
    planner: PathPlannerConfig = field(default_factory=PathPlannerConfig)
    coordinate: CoordinateConfig = field(default_factory=CoordinateConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    kiss_icp: KissICPConfig = field(default_factory=KissICPConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    map_manager: MapManagerConfig = field(default_factory=MapManagerConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 系统级参数
    debug_mode: bool = False                   # 调试模式
    update_frequency: float = 20.0             # 更新频率(Hz)
    visualization_scale: int = 3               # 可视化缩放比例
    global_map_size: float = 200.0             # 全局地图大小(米)
    mount_correction: str = "identity"         # 激光雷达挂载校正
    coordinate_correction: str = "livox_standard"  # 坐标校正类型

def create_default_config() -> SystemConfig:
    """创建默认配置"""
    return SystemConfig()

def create_high_performance_config() -> SystemConfig:
    """创建高性能配置"""
    config = SystemConfig()
    
    # 高性能参数
    config.update_frequency = 30.0
    config.kiss_icp.max_num_threads = 0  # 自动使用所有核心
    config.grid.resolution = 0.03        # 更高分辨率
    config.visualization.render_frequency = 60.0
    config.performance.processing_threads = 8
    config.performance.visualization_threads = 4
    
    return config

def create_low_resource_config() -> SystemConfig:
    """创建低资源配置"""
    config = SystemConfig()
    
    # 低资源参数
    config.update_frequency = 10.0
    config.kiss_icp.max_num_threads = 2
    config.grid.resolution = 0.1         # 降低分辨率
    config.visualization.render_frequency = 15.0
    config.visualization.show_map_points = False
    config.performance.processing_threads = 2
    config.performance.visualization_threads = 1
    
    return config

def create_outdoor_config() -> SystemConfig:
    """创建户外配置"""
    config = SystemConfig()
    
    # 户外参数
    config.grid.max_range = 50.0         # 更大范围
    config.grid.grid_size = 100.0        # 更大网格
    config.kiss_icp.max_range = 50.0
    config.kiss_icp.keyframe_distance = 1.0
    config.planner.max_velocity = 1.0    # 更高速度
    config.planner.lookahead_distance = 2.0
    
    return config

def create_indoor_config() -> SystemConfig:
    """创建室内配置"""
    config = SystemConfig()
    
    # 室内参数
    config.grid.max_range = 15.0         # 较小范围
    config.grid.grid_size = 30.0         # 较小网格
    config.kiss_icp.max_range = 15.0
    config.kiss_icp.keyframe_distance = 0.3
    config.planner.max_velocity = 0.3    # 较低速度
    config.planner.safety_margin = 0.3   # 更大安全边距
    
    return config