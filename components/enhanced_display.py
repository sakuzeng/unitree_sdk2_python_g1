"""
增强的显示组件，集成占用网格
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional
from .base import ComponentBase, StateManager


class EnhancedDisplayComponent(ComponentBase):
    """增强的显示组件，包含占用网格地图"""
    
    def __init__(self, state_manager: StateManager):
        super().__init__("EnhancedDisplay")
        self.state_manager = state_manager
    
    def _run(self) -> None:
        """显示组件不需要独立线程"""
        pass
    
    def compose_display(self) -> Optional[np.ndarray]:
        """合成增强的显示画面，包含占用网格"""
        rgbd = self.state_manager.get("rgbd")
        slam = self.state_manager.get("slam")
        occupancy_grid = self.state_manager.get("occupancy_grid")
        vx, vy, omega = self.state_manager.get("vel", (0.0, 0.0, 0.0))
        soc = self.state_manager.get("soc")
        voltage = self.state_manager.get("voltage")
        
        # 创建占位图像
        if rgbd is None:
            rgbd = np.full((480, 1280, 3), 80, dtype=np.uint8)
            cv2.putText(rgbd, "No RealSense stream", (380, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 使用占用网格或回退到SLAM
        map_display = occupancy_grid if occupancy_grid is not None else slam
        if map_display is None:
            map_display = np.full((480, 480, 3), 60, dtype=np.uint8)
            cv2.putText(map_display, "No SLAM/Grid data", (120, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 在占用网格上添加标签
        if occupancy_grid is not None:
            cv2.putText(map_display, "Occupancy Grid", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 合成画面
        top = rgbd
        bottom = cv2.copyMakeBorder(map_display, 0, 0, 0, max(0, top.shape[1] - map_display.shape[1]), 
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        canvas = np.vstack([top, bottom])
        
        # 添加增强的 HUD
        status_text = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {omega:+.2f}"
        
        # 添加电池信息
        if soc is not None:
            status_text += f"   Battery {soc:3d}%"
        elif voltage is not None:
            status_text += f"   V {voltage:5.1f}"
        
        status_text += "   –  Z: quit  ESC: e-stop"
        
        # 绘制状态栏
        cv2.rectangle(canvas, (0, canvas.shape[0] - 40), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.putText(canvas, status_text, (10, canvas.shape[0] - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 添加系统信息
        sys_info = f"Components: {len([c for c in [rgbd, slam, occupancy_grid] if c is not None])}/3 active"
        cv2.putText(canvas, sys_info, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return canvas