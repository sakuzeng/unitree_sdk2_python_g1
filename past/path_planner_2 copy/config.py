"""
配置参数模块
"""
from dataclasses import dataclass

@dataclass
class GridConfig:
    """网格配置参数"""
    grid_size: float = 20.0			# 网格大小（米）
    grid_resolution: int = 400		# 网格分辨率（像素）
    hit_threshold: float = 0.65		# 占用概率阈值
    free_threshold: float = 0.35	# 自由概率阈值
    prob_hit: float = 0.7			# 命中概率更新
    prob_miss: float = 0.4			# 未命中概率更新
    max_range: float = 10.0			# 最大有效距离（米）
    min_height: float = -0.5		# 最小高度（米）
    max_height: float = 2.5			# 最大高度（米）
    decay_factor: float = 0.999		# 概率衰减因子
    use_odometry: bool = True		# 是否使用里程计数据
    save_interval: int = 200		# PGM保存间隔（帧数）
    confidence_threshold: float = 0.8	# 高置信度更新阈值

@dataclass
class PathPlannerConfig:
    """路径规划器配置参数"""
    goal_tolerance: float = 0.3			# 目标点容差（米）
    obstacle_inflation: int = 3			# 障碍物膨胀像素
    max_planning_distance: float = 10.0	# 最大规划距离（米）
    replan_threshold: float = 0.5		# 重规划阈值（米）
    max_velocity: float = 0.5			# 最大速度（m/s）
    max_angular_velocity: float = 1.0	# 最大角速度（rad/s）
    lookahead_distance: float = 1.0		# 前瞻距离（米）
    safety_distance: float = 0.5		# 安全距离（米）