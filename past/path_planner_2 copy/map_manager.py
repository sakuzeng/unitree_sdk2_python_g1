"""
地图管理模块 - 解决移动时显示混乱问题
"""
import numpy as np
import threading
from scipy import ndimage
from typing import Tuple
from pathlib import Path
from PIL import Image
import time

from config import GridConfig

class StabilizedMapManager:
    """稳定的地图管理器 - 防止移动时显示混乱"""
    
    def __init__(self, config: GridConfig, global_map_size: float = 100.0):
        self.config = config
        self.cell_size = config.grid_size / config.grid_resolution
        self.global_map_size = global_map_size
        self.global_resolution = int(global_map_size / self.cell_size)
        
        # 全局地图存储
        self.global_log_odds = np.zeros((self.global_resolution, self.global_resolution), dtype=np.float32)
        self.global_confidence = np.zeros((self.global_resolution, self.global_resolution), dtype=np.float32)
        self.global_occupancy_grid = np.full((self.global_resolution, self.global_resolution), 128, dtype=np.uint8)
        
        # 稳定显示网格
        self.stable_display_grid = np.full((config.grid_resolution, config.grid_resolution), 128, dtype=np.uint8)
        self.last_stable_update = 0
        self.stability_threshold = 5  # 稳定更新阈值（帧数）
        
        # 地图原点和运动状态
        self.origin = np.array([0.0, 0.0])
        self.origin_set = False
        self.last_robot_pos = np.zeros(2)
        self.motion_state = "stationary"
        self.motion_threshold = 0.02
        
        # 访问区域记录
        self.visited_regions = set()
        
        # 线程锁
        self.update_lock = threading.Lock()
        
        print(f"[StabilizedMap] 全局地图初始化: {global_map_size}m, 分辨率: {self.global_resolution}")
    
    def update_with_local_grid(self, local_grid: np.ndarray, robot_pose: np.ndarray, frame_count: int) -> np.ndarray:
        """
        使用局部网格更新全局地图，返回稳定的显示网格
        
        Args:
            local_grid: 局部占用网格
            robot_pose: 机器人当前位姿
            frame_count: 当前帧数
            
        Returns:
            稳定的显示网格
        """
        with self.update_lock:
            # 设置原点
            if not self.origin_set:
                self.origin = robot_pose[:3, 3][:2].copy()
                self.origin_set = True
                self.last_robot_pos = self.origin.copy()
                print(f"[StabilizedMap] 设置地图原点: ({self.origin[0]:.2f}, {self.origin[1]:.2f})")
            
            # 检测运动状态
            current_pos = robot_pose[:3, 3][:2]
            pos_change = np.linalg.norm(current_pos - self.last_robot_pos)
            
            if pos_change > self.motion_threshold:
                self.motion_state = "moving"
                self.last_robot_pos = current_pos.copy()
            else:
                self.motion_state = "stationary"
            
            # 更新全局地图
            self._update_global_map(local_grid, robot_pose)
            
            # 根据运动状态决定是否更新显示
            if (self.motion_state == "stationary" or 
                frame_count - self.last_stable_update >= self.stability_threshold):
                
                self.stable_display_grid = self._get_stable_local_view(robot_pose)
                self.last_stable_update = frame_count
            
            return self.stable_display_grid
    
    def _update_global_map(self, local_grid: np.ndarray, robot_pose: np.ndarray):
        """更新全局地图"""
        robot_pos_2d = robot_pose[:3, 3][:2]
        rel_pos = robot_pos_2d - self.origin
        
        # 计算在全局地图中的位置
        global_center = self.global_resolution // 2
        global_x = int(rel_pos[0] / self.cell_size + global_center)
        global_y = int(-rel_pos[1] / self.cell_size + global_center)
        
        # 计算映射范围
        local_half = local_grid.shape[0] // 2
        g_x_start = max(0, global_x - local_half)
        g_x_end = min(self.global_resolution, global_x + local_half)
        g_y_start = max(0, global_y - local_half)
        g_y_end = min(self.global_resolution, global_y + local_half)
        
        l_x_start = max(0, local_half - (global_x - g_x_start))
        l_x_end = l_x_start + (g_x_end - g_x_start)
        l_y_start = max(0, local_half - (global_y - g_y_start))
        l_y_end = l_y_start + (g_y_end - g_y_start)
        
        # 更新对应区域
        if (g_x_end > g_x_start and g_y_end > g_y_start and
            l_x_end > l_x_start and l_y_end > l_y_start):
            
            # 转换局部网格为对数几率
            local_region = local_grid[l_y_start:l_y_end, l_x_start:l_x_end]
            local_log_odds = np.where(local_region == 255, 2.0,
                                    np.where(local_region == 0, -2.0, 0.0))
            
            # 累积更新全局地图
            self.global_log_odds[g_y_start:g_y_end, g_x_start:g_x_end] += local_log_odds * 0.3
            self.global_confidence[g_y_start:g_y_end, g_x_start:g_x_end] += 0.1
            
            # 限制范围
            self.global_log_odds = np.clip(self.global_log_odds, -8.0, 8.0)
            self.global_confidence = np.clip(self.global_confidence, 0.0, 3.0)
            
            # 记录访问区域
            region_key = (global_x // 50, global_y // 50)
            self.visited_regions.add(region_key)
    
    def _get_stable_local_view(self, robot_pose: np.ndarray) -> np.ndarray:
        """获取稳定的局部视图"""
        robot_pos_2d = robot_pose[:3, 3][:2]
        rel_pos = robot_pos_2d - self.origin
        
        # 计算在全局地图中的位置
        global_center = self.global_resolution // 2
        global_x = int(rel_pos[0] / self.cell_size + global_center)
        global_y = int(-rel_pos[1] / self.cell_size + global_center)
        
        # 提取局部视图
        view_size = self.config.grid_resolution
        half_view = view_size // 2
        
        g_x_start = max(0, global_x - half_view)
        g_x_end = min(self.global_resolution, global_x + half_view)
        g_y_start = max(0, global_y - half_view)
        g_y_end = min(self.global_resolution, global_y + half_view)
        
        # 创建局部视图
        local_view = np.full((view_size, view_size), 128, dtype=np.uint8)
        
        if g_x_end > g_x_start and g_y_end > g_y_start:
            # 从全局地图提取区域
            global_region = self.global_log_odds[g_y_start:g_y_end, g_x_start:g_x_end]
            confidence_region = self.global_confidence[g_y_start:g_y_end, g_x_start:g_x_end]
            
            # 置信度加权平滑
            weighted_region = global_region * (confidence_region + 0.1)
            smoothed_region = ndimage.gaussian_filter(weighted_region, sigma=0.8)
            
            # 转换为概率并生成占用网格
            prob = 1.0 / (1.0 + np.exp(-smoothed_region))
            region_grid = np.full_like(smoothed_region, 128, dtype=np.uint8)
            region_grid[prob >= self.config.hit_threshold] = 255
            region_grid[prob <= self.config.free_threshold] = 0
            
            # 映射到局部视图
            l_x_start = max(0, half_view - (global_x - g_x_start))
            l_x_end = l_x_start + region_grid.shape[1]
            l_y_start = max(0, half_view - (global_y - g_y_start))
            l_y_end = l_y_start + region_grid.shape[0]
            
            if (l_x_end <= view_size and l_y_end <= view_size):
                local_view[l_y_start:l_y_end, l_x_start:l_x_end] = region_grid
        
        return local_view
    
    def save_complete_map(self, output_dir: str = "maps") -> str:
        """保存完整的探索地图"""
        if not self.origin_set:
            return ""
        
        timestamp = int(time.time())
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 转换为概率并生成占用网格
        global_prob = 1.0 / (1.0 + np.exp(-self.global_log_odds))
        smoothed_prob = ndimage.gaussian_filter(global_prob, sigma=1.5)
        
        # 生成PGM格式网格
        pgm_grid = np.full_like(smoothed_prob, 205, dtype=np.uint8)  # 未知
        pgm_grid[smoothed_prob >= 0.65] = 0   # 占用
        pgm_grid[smoothed_prob <= 0.35] = 254 # 自由
        
        # 保存文件
        filename = f"complete_map_{timestamp}"
        pgm_path = output_path / f"{filename}.pgm"
        Image.fromarray(pgm_grid, mode='L').save(pgm_path)
        
        # 保存配置文件
        yaml_path = output_path / f"{filename}.yaml"
        with open(yaml_path, 'w') as f:
            f.write(f"image: {filename}.pgm\n")
            f.write(f"resolution: {self.cell_size:.6f}\n")
            f.write(f"origin: [{self.origin[0]:.6f}, {self.origin[1]:.6f}, 0.0]\n")
            f.write("negate: 0\n")
            f.write("occupied_thresh: 0.65\n")
            f.write("free_thresh: 0.196\n")
        
        print(f"[StabilizedMap] 已保存完整地图: {pgm_path}")
        print(f"[StabilizedMap] 访问区域数: {len(self.visited_regions)}")
        return str(pgm_path)