"""
SLAM 组件
"""
from __future__ import annotations

import time
from typing import Optional

from .base import ComponentBase, StateManager


class SLAMComponent(ComponentBase):
    """SLAM 数据处理和可视化组件"""
    
    def __init__(self, state_manager: StateManager):
        super().__init__("SLAM")
        self.state_manager = state_manager
    
    def _run(self) -> None:
        """SLAM 主循环"""
        try:
            self._patch_slam_viewer()
            import live_slam as ls
            
            demo = ls.LiveSLAMDemo()
            
            while self.is_running():
                if not demo._viewer.tick():
                    break
                time.sleep(0.01)
            
            demo.shutdown()
            
        except Exception as e:
            print(f"[{self.name}] 组件异常: {e}")
    
    def _patch_slam_viewer(self) -> None:
        """修补 SLAM 可视化器，输出 2D 俯视图和原始数据"""
        try:
            import numpy as np
            import cv2
            import live_slam as ls
            
            # 创建对 state_manager 的引用
            state_manager = self.state_manager
            
            class MiniViewer:
                def __init__(self):
                    self._latest_pts: Optional[np.ndarray] = None
                
                def push(self, xyz: np.ndarray, pose: np.ndarray):
                    self._latest_pts = xyz.copy()
                    
                    # 同时存储原始数据供占用网格使用
                    state_manager.set("slam_raw", (xyz.copy(), pose.copy() if pose is not None else None))
                
                def tick(self) -> bool:
                    if self._latest_pts is None:
                        return True
                    
                    pts = self._latest_pts
                    self._latest_pts = None
                    
                    if pts.shape[0] == 0:
                        return True
                    
                    # 生成简单的 2D 俯视图（保持兼容性）
                    img = self._render_2d_map(pts)
                    state_manager.set("slam", img)
                    
                    return True
                
                def close(self):
                    pass
                
                def _render_2d_map(self, pts: np.ndarray) -> np.ndarray:
                    """渲染简单的 2D 地图"""
                    x, y = pts[:, 0], pts[:, 1]
                    min_x, max_x = float(x.min()), float(x.max())
                    min_y, max_y = float(y.min()), float(y.max())
                    
                    span = max(max_x - min_x, max_y - min_y, 1e-6)
                    scale = 470.0 / span
                    
                    img = np.zeros((480, 480, 3), dtype=np.uint8)
                    
                    # 映射坐标到像素
                    px = ((x - min_x) * scale + 5).astype(np.int32)
                    py = ((y - min_y) * scale + 5).astype(np.int32)
                    py = 479 - py  # 翻转 Y 轴
                    
                    img[py.clip(0, 479), px.clip(0, 479)] = (0, 255, 0)
                    cv2.rectangle(img, (0, 0), (479, 479), (255, 255, 255), 1)
                    
                    return img
            
            # 修补 SLAM 可视化器
            ls._Viewer = MiniViewer
            
        except Exception as e:
            print(f"[{self.name}] 修补 SLAM 可视化器失败: {e}")