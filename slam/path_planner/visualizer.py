"""
可视化模块
"""
import cv2
import numpy as np
from typing import Tuple

def render_slam_system(slam_system) -> np.ndarray:
    """渲染SLAM系统可视化"""
    grid = slam_system.slam_processor.occupancy_grid
    
    if grid is None or slam_system.frame_count < 5:
        canvas = np.full((400, 400, 3), 60, dtype=np.uint8)
        cv2.putText(canvas, "Initializing...", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return canvas
    
    # 创建彩色画布
    canvas = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    canvas[grid == 0] = [0, 255, 0]		# 自由 = 绿色
    canvas[grid == 128] = [64, 64, 64]	# 未知 = 灰色
    canvas[grid == 255] = [0, 0, 255]	# 占用 = 红色
    
    # 绘制机器人
    robot_pos = slam_system._get_robot_grid_position(slam_system._get_robot_pose())
    cv2.circle(canvas, robot_pos, 6, (255, 255, 255), -1)
    cv2.circle(canvas, robot_pos, 8, (0, 0, 0), 2)
    
    # 绘制路径
    if slam_system.current_path:
        path_points = []
        for world_x, world_y in slam_system.current_path:
            grid_pos = slam_system._world_to_grid_position((world_x, world_y))
            if (0 <= grid_pos[0] < grid.shape[1] and 0 <= grid_pos[1] < grid.shape[0]):
                path_points.append(grid_pos)
        
        if len(path_points) > 1:
            path_array = np.array(path_points, dtype=np.int32)
            cv2.polylines(canvas, [path_array], False, (255, 255, 0), 2)
    
    # 绘制目标点
    if slam_system.goal_position:
        goal_grid = slam_system._world_to_grid_position(slam_system.goal_position)
        if (0 <= goal_grid[0] < grid.shape[1] and 0 <= goal_grid[1] < grid.shape[0]):
            cv2.circle(canvas, goal_grid, 8, (255, 0, 255), 3)
    
    return canvas

def add_status_info(canvas: np.ndarray, status: dict) -> np.ndarray:
    """添加状态信息"""
    info_y = 30
    cv2.putText(canvas, f"Frames: {status['frame_count']}", 
               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    info_y += 20
    
    if status['is_planning']:
        cv2.putText(canvas, "Status: Planning...", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    elif status['is_following']:
        cv2.putText(canvas, "Status: Following", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        cv2.putText(canvas, "Status: Ready", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 操作说明
    cv2.putText(canvas, "Left Click: Set Goal | Space: Stop | ESC: Exit", 
               (10, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return canvas

def mouse_callback(event, x, y, flags, slam_system):
    """鼠标回调函数"""
    if event == cv2.EVENT_LBUTTONDOWN:
        grid = slam_system.slam_processor.occupancy_grid
        
        if grid is None or slam_system.frame_count < 10:
            print("[MouseCallback] 系统未完全初始化")
            return
        
        # 检查范围
        if x < 0 or x >= grid.shape[1] or y < 0 or y >= grid.shape[0]:
            return
        
        # 检查是否为障碍物
        if grid[y, x] == 255:
            print("[MouseCallback] 不能在障碍物上设置目标")
            return
        
        # 转换为世界坐标
        center = slam_system.grid_config.grid_resolution // 2
        rel_x = (x - center) * slam_system.cell_size
        rel_y = -(y - center) * slam_system.cell_size
        
        world_x = rel_x + slam_system.slam_processor.origin[0]
        world_y = rel_y + slam_system.slam_processor.origin[1]
        
        slam_system.set_goal(world_x, world_y)