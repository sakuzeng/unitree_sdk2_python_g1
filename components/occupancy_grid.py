"""
SLAM 占用网格可视化组件
"""
from __future__ import annotations

import numpy as np
import cv2
import threading
import time
from typing import Optional, Tuple, Any
from .base import ComponentBase, StateManager


class OccupancyGridComponent(ComponentBase):
    """SLAM 占用网格可视化组件，生成2D俯视图地图"""
    
    def __init__(self, state_manager: StateManager, ground_clearance: float = 0.1):
        super().__init__("OccupancyGrid")
        self.state_manager = state_manager
        self.ground_clearance = ground_clearance  # 地面间隙阈值 (米)
        
        # 地图参数
        self.map_size = 480  # 地图分辨率
        self.margin = 5      # 边界边距
        
        # 地面估计
        self._ground_z_smooth: Optional[float] = None
        self._ALPHA = 0.05  # 地面平滑系数
        
        # 占用网格数据
        self._occ_map: Optional[np.ndarray] = None
        self._map_meta: Optional[Tuple[float, float, float]] = None
        self._robot_px: Optional[Tuple[int, int]] = None
        
        # 自身传感器过滤参数
        self._R_XY = 0.30  # 水平过滤半径 (米)
        self._DZ = 0.24    # 垂直过滤范围 (米)
    
    def _run(self) -> None:
        """占用网格生成主循环"""
        try:
            while self.is_running():
                # 获取SLAM数据
                slam_data = self.state_manager.get("slam_raw")
                if slam_data is not None:
                    xyz, pose = slam_data
                    if xyz is not None and xyz.shape[0] > 0:
                        # 生成占用网格
                        grid_image = self._generate_occupancy_grid(xyz, pose)
                        if grid_image is not None:
                            self.state_manager.set("occupancy_grid", grid_image)
                
                time.sleep(0.05)  # 20Hz 更新频率
                
        except Exception as e:
            print(f"[{self.name}] 组件异常: {e}")
    
    def _generate_occupancy_grid(self, xyz: np.ndarray, pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """生成占用网格地图"""
        if xyz.shape[0] == 0:
            return None
        
        # 鲁棒的地面估计
        ground_z_inst = float(np.percentile(xyz[:, 2], 5.0))
        
        if self._ground_z_smooth is None:
            self._ground_z_smooth = ground_z_inst
        else:
            self._ground_z_smooth = ((1.0 - self._ALPHA) * self._ground_z_smooth + 
                                    self._ALPHA * ground_z_inst)
        
        ground_z = float(self._ground_z_smooth)
        
        # 自身传感器抑制
        if pose is not None and pose.shape == (4, 4):
            xyz = self._filter_self_sensor(xyz, pose)
        
        # 定义地图边界
        min_x, max_x = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        min_y, max_y = float(xyz[:, 1].min()), float(xyz[:, 1].max())
        
        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = (self.map_size - 2 * self.margin) / span
        
        # 存储映射元数据
        self._map_meta = (min_x, min_y, scale)
        
        # 创建画布
        canvas = np.full((self.map_size, self.map_size, 3), 30, dtype=np.uint8)
        
        # 障碍物检测：高于地面+间隙的点
        thresh = ground_z + self.ground_clearance
        obstacle_pts = xyz[xyz[:, 2] > thresh]
        
        # 创建二进制占用网格
        occ = np.zeros((self.map_size, self.map_size), dtype=bool)
        
        if obstacle_pts.shape[0] > 0:
            px_obs, py_obs = self._world_to_px(obstacle_pts[:, 0], obstacle_pts[:, 1], 
                                               min_x, min_y, scale)
            valid = ((px_obs >= 0) & (px_obs < self.map_size) & 
                    (py_obs >= 0) & (py_obs < self.map_size))
            px_obs, py_obs = px_obs[valid], py_obs[valid]
            
            # 更新占用网格和可视化
            occ[py_obs, px_obs] = True
            canvas[py_obs, px_obs] = (255, 255, 255)  # 白色障碍物
        
        # 绘制边框
        cv2.rectangle(canvas, (0, 0), (self.map_size - 1, self.map_size - 1), 
                     (255, 255, 255), 1)
        
        # 绘制机器人位置和朝向
        if pose is not None:
            self._draw_robot(canvas, pose, min_x, min_y, scale, occ)
        
        # 存储占用网格数据
        self._occ_map = occ.copy()
        
        return canvas
    
    def _filter_self_sensor(self, xyz: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """过滤机器人自身传感器噪声"""
        rob_pos = pose[:3, 3]
        diff = xyz - rob_pos
        dist_xy = np.linalg.norm(diff[:, :2], axis=1)
        close = dist_xy < self._R_XY
        near_plane = np.abs(diff[:, 2]) < self._DZ
        keep_mask = ~(close & near_plane)
        return xyz[keep_mask] if keep_mask.sum() != xyz.shape[0] else xyz
    
    def _world_to_px(self, xw: np.ndarray, yw: np.ndarray, 
                    min_x: float, min_y: float, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """将世界坐标转换为像素坐标"""
        # 水平：+y 向右，垂直：+x 向上（翻转）
        px = ((yw - min_y) * scale + self.margin).astype(np.int32)
        py = ((xw - min_x) * scale + self.margin).astype(np.int32)
        py = (self.map_size - 1) - py  # 翻转Y轴，使+x向上
        return px, py
    
    def _draw_robot(self, canvas: np.ndarray, pose: np.ndarray, 
                   min_x: float, min_y: float, scale: float, occ: np.ndarray) -> None:
        """在地图上绘制机器人位置和朝向"""
        rob_pos = pose[:3, 3]
        rx, ry = self._world_to_px(np.array([rob_pos[0]]), np.array([rob_pos[1]]), 
                                  min_x, min_y, scale)
        rx, ry = int(rx[0]), int(ry[0])
        
        # 确保机器人周围区域为自由空间
        rr0, rr1 = max(0, ry - 1), min(self.map_size, ry + 2)
        rc0, rc1 = max(0, rx - 1), min(self.map_size, rx + 2)
        occ[rr0:rr1, rc0:rc1] = False
        
        # 存储机器人像素位置
        self._robot_px = (rx, ry)
        
        # 计算朝向箭头
        fwd_m = 0.25  # 25厘米箭头长度
        fwd_vec = pose[:3, 0] * fwd_m  # 机器人前向向量
        tip_world = rob_pos + fwd_vec
        tx, ty = self._world_to_px(np.array([tip_world[0]]), np.array([tip_world[1]]), 
                                  min_x, min_y, scale)
        tx, ty = int(tx[0]), int(ty[0])
        
        # 绘制机器人箭头
        cv2.arrowedLine(canvas, (rx, ry), (tx, ty), (0, 255, 0), 2, tipLength=0.8)
        
        # 绘制机器人中心点
        cv2.circle(canvas, (rx, ry), 3, (0, 255, 0), -1)
    
    def get_occupancy_data(self) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float]], Optional[Tuple[int, int]]]:
        """获取占用网格数据"""
        return self._occ_map, self._map_meta, self._robot_px