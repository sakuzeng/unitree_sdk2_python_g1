"""
智能可视化模块 - 简化高效版本
提供实时、高效的SLAM可视化界面
"""
import numpy as np
import cv2
import threading
import time
import logging
from typing import Optional, List, Tuple, Dict, Any, Callable
from collections import deque
from dataclasses import dataclass

from config import VisualizationConfig, SystemConfig
from slam_processor import AdvancedSLAMProcessor
from path_planner import IntelligentPathPlanner, PathQualityMetrics
from kiss_icp_wrapper import RobustKissICPOdometry
from map_manager import IntelligentMapManager

logger = logging.getLogger(__name__)

@dataclass
class VisualizationState:
    """可视化状态管理"""
    current_pose: np.ndarray = None
    current_path: List[np.ndarray] = None
    trajectory: List[np.ndarray] = None
    keyframes: List[Tuple[np.ndarray, np.ndarray]] = None
    map_points: Optional[np.ndarray] = None
    occupancy_grid: Optional[np.ndarray] = None
    goal_position: Optional[np.ndarray] = None
    robot_status: str = "IDLE"
    slam_quality: float = 0.0
    path_quality: PathQualityMetrics = None
    processing_stats: Dict[str, Any] = None

@dataclass
class RenderingMetrics:
    """渲染性能指标"""
    frame_count: int = 0
    fps: float = 0.0
    avg_render_time: float = 0.0
    dropped_frames: int = 0
    last_update_time: float = 0.0

class SimplifiedVisualizer:
    """简化高效可视化器"""
    
    def __init__(self, config: VisualizationConfig, system_config: SystemConfig = None,
                slam_processor: AdvancedSLAMProcessor = None,
                path_planner: IntelligentPathPlanner = None):
        self.config = config
        self.system_config = system_config or SystemConfig()
        self.slam_processor = slam_processor
        self.path_planner = path_planner
        
        # 窗口参数
        self.window_name = "G1 SLAM Navigation"
        self.window_width = config.window_width
        self.window_height = config.window_height
        
        # 视图管理 - 简化参数
        self.view_center = np.array([0.0, 0.0])
        self.view_range = config.view_range
        self.auto_center = config.auto_center
        
        # 状态管理
        self.visualization_state = VisualizationState()
        self.rendering_metrics = RenderingMetrics()
        
        # 修复颜色配置 - 符合用户要求
        self.colors = {
            'background': (20, 20, 20),         # 深色背景
            'grid_lines': (60, 60, 60),         # 网格线
            'robot': (0, 150, 255),             # 蓝色机器人
            'robot_direction': (255, 255, 0),   # 黄色方向指示
            'trajectory': (0, 200, 255),        # 蓝色轨迹
            'path': (255, 100, 255),            # 紫色路径
            'goal': (255, 100, 100),            # 红色目标
            'keyframes': (100, 255, 255),       # 青色关键帧
            'text': (255, 255, 255),            # 白色文字
            'obstacle': (0, 0, 255),            # 红色障碍物 (修改)
            'free_space': (0, 255, 0),          # 绿色自由空间 (修改)
            'unknown_space': (128, 128, 128),   # 灰色未知空间 (新增)
            'status_good': (0, 255, 0),         # 状态良好
            'status_warning': (255, 165, 0),    # 状态警告
            'status_error': (255, 0, 0)         # 状态错误
        }
        
        # 渲染控制
        self.rendering_enabled = False
        self.render_thread = None
        self.render_lock = threading.RLock()
        self.target_fps = min(config.render_frequency, 20.0)  # 限制最大FPS提高效率
        self.frame_interval = 1.0 / self.target_fps
        
        # 窗口初始化状态
        self.window_initialized = False
        self.canvas = None
        
        # 交互控制
        self.mouse_pos = (0, 0)
        self.mouse_dragging = False
        self.mouse_drag_start = None
        
        # 简化显示选项
        self.show_options = {
            'grid_lines': True,
            'trajectory': True,
            'path': True,
            'robot_orientation': True,
            'coordinates': False,  # 默认关闭坐标显示
            'keyframes': False,    # 默认关闭关键帧显示
            'map_points': False,   # 默认关闭地图点显示
            'occupancy_grid': True,
            'info_panel': True
        }
        
        # 信息面板
        self.info_panel_width = 250  # 减小面板宽度
        
        # 性能优化参数
        self.pixels_per_meter = 0
        self.last_grid_update = 0
        self.grid_update_interval = 0.1  # 网格更新间隔
        self.trajectory_max_points = 200  # 限制轨迹点数
        
        # 注册回调
        if self.slam_processor:
            self._register_slam_callbacks()
        if self.path_planner:
            self._register_planner_callbacks()
        
        logger.info(f"[SimplifiedVisualizer] 简化可视化器初始化完成")
        logger.info(f"[SimplifiedVisualizer] 目标FPS: {self.target_fps}")
    
    def _init_window(self):
        """初始化OpenCV窗口"""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            
            # 创建画布
            self.canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
            self.canvas.fill(20)
            
            # 计算像素比例
            map_width = self.window_width - self.info_panel_width
            self.pixels_per_meter = min(map_width, self.window_height) / (2 * self.view_range)
            
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)
            
            self.window_initialized = True
            logger.info("[SimplifiedVisualizer] 窗口初始化完成")
            
        except Exception as e:
            logger.error(f"[SimplifiedVisualizer] 窗口初始化失败: {e}")
            self.window_initialized = False
    
    def _register_slam_callbacks(self):
        """注册SLAM回调"""
        if hasattr(self.slam_processor, 'register_callback'):
            self.slam_processor.register_callback('pose_update', self._on_pose_update)
            self.slam_processor.register_callback('map_update', self._on_map_update)
            self.slam_processor.register_callback('quality_update', self._on_quality_update)
    
    def _register_planner_callbacks(self):
        """注册路径规划回调"""
        if hasattr(self.path_planner, 'register_path_update_callback'):
            self.path_planner.register_path_update_callback(self._on_path_update)
    
    def _on_pose_update(self, pose: np.ndarray, is_keyframe: bool):
        """位姿更新回调"""
        with self.render_lock:
            self.visualization_state.current_pose = pose.copy()
            
            # 更新轨迹（限制点数）
            if self.slam_processor and hasattr(self.slam_processor, 'get_trajectory'):
                trajectory = self.slam_processor.get_trajectory()
                if trajectory and len(trajectory) > self.trajectory_max_points:
                    # 智能采样保留关键点
                    step = len(trajectory) // self.trajectory_max_points
                    self.visualization_state.trajectory = trajectory[::step]
                else:
                    self.visualization_state.trajectory = trajectory
            
            # 自动居中
            if self.auto_center:
                self.view_center = pose[:2, 3].copy()
    
    def _on_map_update(self, occupancy_grid: np.ndarray):
        """地图更新回调"""
        current_time = time.time()
        if current_time - self.last_grid_update > self.grid_update_interval:
            with self.render_lock:
                if occupancy_grid is not None:
                    self.visualization_state.occupancy_grid = occupancy_grid.copy()
                self.last_grid_update = current_time
    
    def _on_quality_update(self, stats: Dict[str, Any]):
        """质量更新回调"""
        with self.render_lock:
            slam_stats = stats.get('slam_processor', {})
            self.visualization_state.slam_quality = slam_stats.get('average_quality', 0.0)
            self.visualization_state.processing_stats = stats
    
    def _on_path_update(self, path: List[np.ndarray], quality):
        """路径更新回调"""
        with self.render_lock:
            self.visualization_state.current_path = [p.copy() for p in path]
            self.visualization_state.path_quality = quality
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        self.mouse_pos = (x, y)
        map_width = self.window_width - self.info_panel_width
        
        if x >= map_width:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_dragging = True
            self.mouse_drag_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging:
            if self.mouse_drag_start:
                dx = x - self.mouse_drag_start[0]
                dy = y - self.mouse_drag_start[1]
                
                world_dx = -dx / self.pixels_per_meter
                world_dy = dy / self.pixels_per_meter
                
                self.view_center[0] += world_dx
                self.view_center[1] += world_dy
                self.mouse_drag_start = (x, y)
                self.auto_center = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键设置目标
            world_pos = self._screen_to_world(x, y, map_width)
            self.set_goal_position(world_pos)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮缩放
            if flags > 0:
                self.view_range *= 0.9
            else:
                self.view_range *= 1.1
            self.view_range = np.clip(self.view_range, 2.0, 50.0)
            self.pixels_per_meter = min(map_width, self.window_height) / (2 * self.view_range)
    
    def start_rendering(self):
        """启动渲染"""
        if self.render_thread is None or not self.render_thread.is_alive():
            self.rendering_enabled = True
            self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
            self.render_thread.start()
            logger.info("[SimplifiedVisualizer] 渲染线程启动")
    
    def stop_rendering(self):
        """停止渲染"""
        self.rendering_enabled = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        logger.info("[SimplifiedVisualizer] 渲染已停止")
    
    def _render_loop(self):
        """渲染主循环"""
        self._init_window()
        
        if not self.window_initialized:
            logger.error("[SimplifiedVisualizer] 窗口初始化失败")
            return
        
        while self.rendering_enabled:
            start_time = time.perf_counter()
            
            try:
                self._render_frame()
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_keyboard_input(key):
                        break
                
                # 帧率控制
                render_time = time.perf_counter() - start_time
                self._update_metrics(render_time)
                
                sleep_time = max(0, self.frame_interval - render_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.rendering_metrics.dropped_frames += 1
                    
            except Exception as e:
                logger.error(f"[SimplifiedVisualizer] 渲染错误: {e}")
                time.sleep(0.1)
    
    def _render_frame(self):
        """渲染单帧"""
        if not self.window_initialized:
            return
        
        with self.render_lock:
            # 清空画布
            self.canvas.fill(20)
            
            # 获取地图区域
            map_width = self.window_width - self.info_panel_width
            map_canvas = self.canvas[:, :map_width]
            
            # 简化的绘制顺序
            if self.show_options['grid_lines']:
                self._draw_simple_grid(map_canvas)
            
            if (self.show_options['occupancy_grid'] and 
                self.visualization_state.occupancy_grid is not None):
                self._draw_simple_occupancy_grid(map_canvas)
            
            if (self.show_options['trajectory'] and 
                self.visualization_state.trajectory):
                self._draw_simple_trajectory(map_canvas)
            
            if (self.show_options['path'] and 
                self.visualization_state.current_path):
                self._draw_simple_path(map_canvas)
            
            if self.visualization_state.current_pose is not None:
                self._draw_simple_robot(map_canvas)
            
            if self.visualization_state.goal_position is not None:
                self._draw_simple_goal(map_canvas)
            
            if self.show_options['coordinates']:
                self._draw_simple_coordinates(map_canvas)
            
            if self.show_options['info_panel']:
                self._draw_simple_info_panel()
            
            # 显示
            try:
                cv2.imshow(self.window_name, self.canvas)
            except Exception as e:
                logger.error(f"[SimplifiedVisualizer] 显示失败: {e}")
    
    def _draw_simple_grid(self, canvas: np.ndarray):
        """绘制简化网格"""
        spacing = 2.0  # 固定2米间距
        color = self.colors['grid_lines']
        
        # 计算网格范围
        half_range = self.view_range
        start_x = int((self.view_center[0] - half_range) // spacing) * spacing
        end_x = int((self.view_center[0] + half_range) // spacing + 1) * spacing
        start_y = int((self.view_center[1] - half_range) // spacing) * spacing
        end_y = int((self.view_center[1] + half_range) // spacing + 1) * spacing
        
        # 绘制主要网格线
        x = start_x
        while x <= end_x:
            screen_x = int(self._world_to_screen_x(x, canvas.shape[1]))
            if 0 <= screen_x < canvas.shape[1]:
                cv2.line(canvas, (screen_x, 0), (screen_x, canvas.shape[0]-1), color, 1)
            x += spacing
        
        y = start_y
        while y <= end_y:
            screen_y = int(self._world_to_screen_y(y))
            if 0 <= screen_y < canvas.shape[0]:
                cv2.line(canvas, (0, screen_y), (canvas.shape[1]-1, screen_y), color, 1)
            y += spacing
    
    def _draw_simple_occupancy_grid(self, canvas: np.ndarray):
        """绘制简化占用网格 - 修复颜色映射和坐标系"""
        grid = self.visualization_state.occupancy_grid
        if grid is None or grid.size == 0:
            return
        
        grid_size = grid.shape[0]
        # 确保有分辨率配置
        resolution = getattr(self.system_config.grid, 'resolution', 0.05)
        
        # 优化步长 - 根据缩放级别调整
        step = max(1, int(2 / max(self.pixels_per_meter, 0.1)))
        
        for i in range(0, grid_size, step):
            for j in range(0, grid_size, step):
                # 坐标转换 - 与地图管理器保持一致
                world_x = (j - grid_size // 2) * resolution
                world_y = (grid_size // 2 - i) * resolution  # Y轴翻转
                
                # 视图裁剪优化
                if (abs(world_x - self.view_center[0]) > self.view_range or
                    abs(world_y - self.view_center[1]) > self.view_range):
                    continue
                
                occupancy_value = grid[i, j]
                
                # 标准ROS占用网格颜色映射
                if occupancy_value == 255:  # 确定占用
                    color = (0, 0, 255)  # 红色障碍物
                elif occupancy_value >= 200:  # 高概率占用/软占用
                    color = (0, 50, 200)  # 深红色
                elif occupancy_value == 0:  # 确定自由
                    color = (0, 255, 0)  # 绿色自由空间
                elif occupancy_value <= 50:   # 高概率自由
                    color = (50, 255, 50)  # 浅绿色
                elif occupancy_value == 128:  # 未知
                    continue  # 跳过未知区域，保持背景色
                else:
                    # 中间值处理
                    if occupancy_value > 128:
                        # 倾向占用 - 红色渐变
                        intensity = min(255, int((occupancy_value - 128) * 2))
                        color = (0, 0, intensity)
                    else:
                        # 倾向自由 - 绿色渐变
                        intensity = min(255, int((128 - occupancy_value) * 2))
                        color = (0, intensity, 0)
                
                screen_x = int(self._world_to_screen_x(world_x, canvas.shape[1]))
                screen_y = int(self._world_to_screen_y(world_y))
                
                if (0 <= screen_x < canvas.shape[1] and 0 <= screen_y < canvas.shape[0]):
                    rect_size = max(1, int(resolution * self.pixels_per_meter * step))
                    
                    # 绘制占用网格单元
                    cv2.rectangle(canvas,
                                (screen_x - rect_size//2, screen_y - rect_size//2),
                                (screen_x + rect_size//2, screen_y + rect_size//2),
                                color, -1)
                    
                    # 为障碍物添加边框以增强可见性
                    if occupancy_value >= 200:
                        cv2.rectangle(canvas,
                                    (screen_x - rect_size//2, screen_y - rect_size//2),
                                    (screen_x + rect_size//2, screen_y + rect_size//2),
                                    (255, 255, 255), 1)
    
    def _draw_simple_trajectory(self, canvas: np.ndarray):
        """绘制简化轨迹"""
        trajectory = self.visualization_state.trajectory
        if not trajectory or len(trajectory) < 2:
            return
        
        color = self.colors['trajectory']
        
        # 简化线条绘制
        for i in range(1, len(trajectory)):
            pos1 = trajectory[i-1][:3, 3][:2]
            pos2 = trajectory[i][:3, 3][:2]
            
            screen_pos1 = (int(self._world_to_screen_x(pos1[0], canvas.shape[1])),
                          int(self._world_to_screen_y(pos1[1])))
            screen_pos2 = (int(self._world_to_screen_x(pos2[0], canvas.shape[1])),
                          int(self._world_to_screen_y(pos2[1])))
            
            cv2.line(canvas, screen_pos1, screen_pos2, color, 2)
    
    def _draw_simple_path(self, canvas: np.ndarray):
        """绘制简化路径"""
        path = self.visualization_state.current_path
        if not path or len(path) < 2:
            return
        
        color = self.colors['path']
        
        # 绘制路径线
        for i in range(1, len(path)):
            pos1 = path[i-1][:2] if len(path[i-1]) > 2 else path[i-1]
            pos2 = path[i][:2] if len(path[i]) > 2 else path[i]
            
            screen_pos1 = (int(self._world_to_screen_x(pos1[0], canvas.shape[1])),
                          int(self._world_to_screen_y(pos1[1])))
            screen_pos2 = (int(self._world_to_screen_x(pos2[0], canvas.shape[1])),
                          int(self._world_to_screen_y(pos2[1])))
            
            cv2.line(canvas, screen_pos1, screen_pos2, color, 3)
        
        # 绘制起点和终点
        if path:
            start_pos = path[0][:2] if len(path[0]) > 2 else path[0]
            end_pos = path[-1][:2] if len(path[-1]) > 2 else path[-1]
            
            start_screen = (int(self._world_to_screen_x(start_pos[0], canvas.shape[1])),
                           int(self._world_to_screen_y(start_pos[1])))
            end_screen = (int(self._world_to_screen_x(end_pos[0], canvas.shape[1])),
                         int(self._world_to_screen_y(end_pos[1])))
            
            cv2.circle(canvas, start_screen, 6, (0, 255, 0), -1)  # 绿色起点
            cv2.circle(canvas, end_screen, 6, (255, 0, 0), -1)    # 红色终点
    
    def _draw_simple_robot(self, canvas: np.ndarray):
        """绘制简化机器人图标 - 修复方向指示"""
        pose = self.visualization_state.current_pose
        if pose is None:
            return
            
        pos = pose[:3, 3][:2]
        
        screen_x = int(self._world_to_screen_x(pos[0], canvas.shape[1]))
        screen_y = int(self._world_to_screen_y(pos[1]))
        
        # 检查是否在画布范围内
        if not (0 <= screen_x < canvas.shape[1] and 0 <= screen_y < canvas.shape[0]):
            return
        
        # 动态调整大小
        robot_size = max(8, int(self.pixels_per_meter * 0.3))
        robot_color = self.colors['robot']
        
        # 机器人主体 - 实心圆
        cv2.circle(canvas, (screen_x, screen_y), robot_size, robot_color, -1)
        cv2.circle(canvas, (screen_x, screen_y), robot_size + 1, (255, 255, 255), 1)
        
        # 方向指示 - 只在有效时显示
        if self.show_options['robot_orientation']:
            try:
                # 从旋转矩阵提取角度
                angle = np.arctan2(pose[1, 0], pose[0, 0])
                
                # 限制方向线长度
                direction_length = min(robot_size + 15, int(self.pixels_per_meter * 0.5))
                end_x = int(screen_x + direction_length * np.cos(angle))
                end_y = int(screen_y + direction_length * np.sin(angle))
                
                # 确保终点在画布内
                end_x = np.clip(end_x, 0, canvas.shape[1] - 1)
                end_y = np.clip(end_y, 0, canvas.shape[0] - 1)
                
                cv2.line(canvas, (screen_x, screen_y), (end_x, end_y), 
                        self.colors['robot_direction'], 3)
                
                # 在方向线末端添加小圆点
                cv2.circle(canvas, (end_x, end_y), 3, self.colors['robot_direction'], -1)
                
            except Exception as e:
                logger.debug(f"[SimplifiedVisualizer] 绘制机器人方向失败: {e}")
    
    def _draw_simple_goal(self, canvas: np.ndarray):
        """绘制简化目标点"""
        goal_pos = self.visualization_state.goal_position
        
        screen_x = int(self._world_to_screen_x(goal_pos[0], canvas.shape[1]))
        screen_y = int(self._world_to_screen_y(goal_pos[1]))
        
        goal_size = max(8, int(self.pixels_per_meter * 0.25))
        color = self.colors['goal']
        
        # 简化的目标标记 - 圆圈加十字
        cv2.circle(canvas, (screen_x, screen_y), goal_size, color, 2)
        cv2.line(canvas, (screen_x - goal_size//2, screen_y), 
                (screen_x + goal_size//2, screen_y), color, 2)
        cv2.line(canvas, (screen_x, screen_y - goal_size//2), 
                (screen_x, screen_y + goal_size//2), color, 2)
    
    def _draw_simple_coordinates(self, canvas: np.ndarray):
        """绘制简化坐标信息"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = self.colors['text']
        
        # 只显示关键信息
        if self.visualization_state.current_pose is not None:
            pos = self.visualization_state.current_pose[:3, 3]
            coord_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})"
            cv2.putText(canvas, coord_text, (10, 25), font, 0.5, color, 1)
        
        range_text = f"Range: {self.view_range:.1f}m"
        cv2.putText(canvas, range_text, (10, 50), font, 0.5, color, 1)
    
    def _draw_simple_info_panel(self):
        """绘制简化信息面板"""
        panel_x = self.window_width - self.info_panel_width
        
        # 绘制背景
        cv2.rectangle(self.canvas, (panel_x, 0), (self.window_width, self.window_height), 
                     (35, 35, 35), -1)
        cv2.line(self.canvas, (panel_x, 0), (panel_x, self.window_height), (70, 70, 70), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = self.colors['text']
        y_pos = 30
        x_pos = panel_x + 10
        
        # 标题
        cv2.putText(self.canvas, "G1 SLAM", (x_pos, y_pos), font, 0.6, 
                   (100, 255, 255), 1)
        y_pos += 40
        
        # SLAM状态
        if self.visualization_state.current_pose is not None:
            pos = self.visualization_state.current_pose[:3, 3]
            cv2.putText(self.canvas, f"X: {pos[0]:6.2f}m", (x_pos, y_pos), 
                       font, 0.4, text_color, 1)
            y_pos += 20
            cv2.putText(self.canvas, f"Y: {pos[1]:6.2f}m", (x_pos, y_pos), 
                       font, 0.4, text_color, 1)
            y_pos += 25
        
        # 质量指示
        quality = self.visualization_state.slam_quality
        quality_color = (self.colors['status_good'] if quality > 0.7 else
                        self.colors['status_warning'] if quality > 0.3 else
                        self.colors['status_error'])
        cv2.putText(self.canvas, f"Quality: {quality:.2f}", (x_pos, y_pos), 
                   font, 0.4, quality_color, 1)
        y_pos += 25
        
        # 轨迹信息
        if self.visualization_state.trajectory:
            cv2.putText(self.canvas, f"Traj: {len(self.visualization_state.trajectory)}", 
                       (x_pos, y_pos), font, 0.4, text_color, 1)
            y_pos += 20
        
        # 路径信息
        if self.visualization_state.current_path:
            cv2.putText(self.canvas, f"Path: {len(self.visualization_state.current_path)}", 
                       (x_pos, y_pos), font, 0.4, text_color, 1)
            y_pos += 25
        
        # 性能
        cv2.putText(self.canvas, f"FPS: {self.rendering_metrics.fps:.0f}", 
                   (x_pos, y_pos), font, 0.4, text_color, 1)
        y_pos += 30
        
        # 简化的控制说明
        controls = [
            "Right: Set goal",
            "Drag: Pan",
            "Wheel: Zoom",
            "G: Grid",
            "T: Trajectory",
            "Q: Quit"
        ]
        
        y_pos = self.window_height - len(controls) * 15 - 20
        for control in controls:
            cv2.putText(self.canvas, control, (x_pos, y_pos), 
                       font, 0.3, (150, 150, 150), 1)
            y_pos += 15
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """处理键盘输入"""
        if key == ord('q') or key == 27:  # Q或ESC退出
            self.stop_rendering()
            return False
        elif key == ord('g'):  # 切换网格
            self.show_options['grid_lines'] = not self.show_options['grid_lines']
        elif key == ord('t'):  # 切换轨迹
            self.show_options['trajectory'] = not self.show_options['trajectory']
        elif key == ord('p'):  # 切换路径
            self.show_options['path'] = not self.show_options['path']
        elif key == ord('o'):  # 切换占用网格
            self.show_options['occupancy_grid'] = not self.show_options['occupancy_grid']
        elif key == ord('c'):  # 切换坐标
            self.show_options['coordinates'] = not self.show_options['coordinates']
        elif key == ord('r'):  # 重置视图
            self.reset_view()
        elif key == ord(' '):  # 切换自动居中
            self.auto_center = not self.auto_center
        
        return True
    
    def _update_metrics(self, render_time: float):
        """更新性能指标"""
        self.rendering_metrics.frame_count += 1
        self.rendering_metrics.avg_render_time = (
            self.rendering_metrics.avg_render_time * 0.9 + render_time * 0.1
        )
        
        # 计算FPS
        current_time = time.time()
        if current_time - self.rendering_metrics.last_update_time >= 1.0:
            time_diff = current_time - self.rendering_metrics.last_update_time
            self.rendering_metrics.fps = self.rendering_metrics.frame_count / time_diff
            self.rendering_metrics.frame_count = 0
            self.rendering_metrics.last_update_time = current_time
    
    def _world_to_screen_x(self, world_x: float, canvas_width: int) -> float:
        """世界坐标X转屏幕坐标X"""
        return (world_x - self.view_center[0]) * self.pixels_per_meter + canvas_width / 2
    
    def _world_to_screen_y(self, world_y: float) -> float:
        """世界坐标Y转屏幕坐标Y"""
        return -(world_y - self.view_center[1]) * self.pixels_per_meter + self.window_height / 2
    
    def _screen_to_world(self, screen_x: float, screen_y: float, canvas_width: int) -> np.ndarray:
        """屏幕坐标转世界坐标"""
        world_x = (screen_x - canvas_width / 2) / self.pixels_per_meter + self.view_center[0]
        world_y = -(screen_y - self.window_height / 2) / self.pixels_per_meter + self.view_center[1]
        return np.array([world_x, world_y])
    
    # 公共接口
    def set_goal_position(self, goal_pos: np.ndarray):
        """设置目标位置"""
        with self.render_lock:
            self.visualization_state.goal_position = goal_pos.copy()
        
        if self.path_planner and hasattr(self.path_planner, 'set_goal'):
            self.path_planner.set_goal(goal_pos)
        
        logger.info(f"[SimplifiedVisualizer] 目标: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    
    def reset_view(self):
        """重置视图"""
        self.view_center = np.array([0.0, 0.0])
        self.view_range = self.config.view_range
        map_width = self.window_width - self.info_panel_width
        self.pixels_per_meter = min(map_width, self.window_height) / (2 * self.view_range)
        self.auto_center = True
        logger.info("[SimplifiedVisualizer] 视图已重置")
    
    def get_rendering_stats(self) -> Dict[str, Any]:
        """获取渲染统计"""
        return {
            'fps': self.rendering_metrics.fps,
            'avg_render_time': self.rendering_metrics.avg_render_time,
            'dropped_frames': self.rendering_metrics.dropped_frames,
            'view_center': self.view_center.tolist(),
            'view_range': self.view_range
        }

# 为了兼容性，保留原类名作为别名
IntelligentVisualizer = SimplifiedVisualizer