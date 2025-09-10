#!/usr/bin/env python3
"""
G1 机器人全栈控制系统

使用组件化架构实现模块化管理，包含占用网格地图显示。
"""
from __future__ import annotations

import argparse
import time
import sys
import subprocess
from typing import List, Optional

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"依赖缺失: {e}")
    print("请安装: pip install opencv-python numpy")
    sys.exit(1)

from components.base import ComponentBase, StateManager
from components.realsense import RealSenseComponent
from components.slam import SLAMComponent
from components.robot_control import RobotControlComponent
from components.occupancy_grid import OccupancyGridComponent
from components.enhanced_display import EnhancedDisplayComponent
from components.battery import BatteryComponent


class G1StackSystem:
    """G1 机器人全栈系统"""
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.state_manager = StateManager()
        self.components: List[ComponentBase] = []
        self.display_component: Optional[EnhancedDisplayComponent] = None
        self.running = False
    
    def add_component(self, component: ComponentBase) -> None:
        """添加组件"""
        self.components.append(component)
    
    def set_display_component(self, display_component: EnhancedDisplayComponent) -> None:
        """设置显示组件"""
        self.display_component = display_component
    
    def start(self) -> None:
        """启动所有组件"""
        print("[G1-Stack] 启动系统组件...")
        for component in self.components:
            component.start()
        
        # 显示组件不需要单独启动，它是在主线程中运行的
        self.running = True
        print("[G1-Stack] 所有组件已启动")
    
    def stop(self) -> None:
        """停止所有组件"""
        print("[G1-Stack] 停止系统组件...")
        
        for component in reversed(self.components):
            component.stop()
        
        self.running = False
        print("[G1-Stack] 所有组件已停止")
    
    def compose_display(self) -> Optional[np.ndarray]:
        """合成显示画面，集成所有新组件的输出"""
        # 获取所有状态数据
        rgbd = self.state_manager.get("rgbd")
        slam = self.state_manager.get("slam")
        occupancy_grid = self.state_manager.get("occupancy_grid")  # 新增：占用网格
        vx, vy, omega = self.state_manager.get("vel", (0.0, 0.0, 0.0))
        soc = self.state_manager.get("soc")  # 新增：电池电量
        voltage = self.state_manager.get("voltage")  # 新增：电池电压
        active_arm = self.state_manager.get("active_arm", "right")  # 新增：活动手臂
        balance_mode = self.state_manager.get("balance_mode", -1)  # 新增：平衡模式
        
        # 创建占位图像
        if rgbd is None:
            rgbd = np.full((480, 1280, 3), 80, dtype=np.uint8)
            cv2.putText(rgbd, "No RealSense stream", (380, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 优先使用占用网格，回退到普通SLAM
        map_display = occupancy_grid if occupancy_grid is not None else slam
        if map_display is None:
            map_display = np.full((480, 480, 3), 60, dtype=np.uint8)
            cv2.putText(map_display, "No SLAM/Grid data", (120, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # 在地图上添加类型标签
        if occupancy_grid is not None:
            cv2.putText(map_display, "Occupancy Grid", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        elif slam is not None:
            cv2.putText(map_display, "SLAM Points", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 合成画面布局
        top = rgbd
        bottom = cv2.copyMakeBorder(map_display, 0, 0, 0, max(0, top.shape[1] - map_display.shape[1]), 
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
        canvas = np.vstack([top, bottom])
        
        # 构建增强的状态文本
        status_text = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {omega:+.2f}"
        
        # 添加电池信息
        if soc is not None:
            status_text += f"   Battery {soc:3d}%"
        elif voltage is not None:
            status_text += f"   V {voltage:5.1f}"
        
        # 添加手臂和平衡模式信息
        status_text += f"   Arm:{active_arm[0].upper()}"
        if balance_mode >= 0:
            status_text += f"   Bal:{balance_mode}"
        
        status_text += "   –  Z: quit  ESC: e-stop"
        
        # 绘制状态栏
        cv2.rectangle(canvas, (0, canvas.shape[0] - 40), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.putText(canvas, status_text, (10, canvas.shape[0] - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 添加系统信息
        active_components = []
        if rgbd is not None and not np.all(rgbd == 80):
            active_components.append("RealSense")
        if occupancy_grid is not None:
            active_components.append("OccGrid")
        elif slam is not None:
            active_components.append("SLAM")
        if soc is not None or voltage is not None:
            active_components.append("Battery")
        
        sys_info = f"Active: {', '.join(active_components) if active_components else 'None'}"
        cv2.putText(canvas, sys_info, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 添加时间戳
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(canvas, timestamp, (canvas.shape[1] - 100, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return canvas
    
    def run_display_loop(self) -> None:
        """运行显示循环"""
        print("[G1-Stack] 系统就绪，显示主窗口...")
        print("控制说明:")
        print("  WASD: 移动控制  QE: 侧移  Z: 阻尼模式  ESC: 紧急停止")
        print("  窗口按键: Q 或 ESC 退出程序")
        print("  地图显示: 白色=障碍物, 绿色箭头=机器人位置和朝向")
        
        try:
            while self.running:
                # 直接调用系统的compose_display方法，而不是显示组件的
                canvas = self.compose_display()
                if canvas is not None:
                    cv2.imshow("G1-Stack (Enhanced)", canvas)
                
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC 或 Q 键
                    print("[G1-Stack] 用户请求退出...")
                    break
                
                time.sleep(0.01)
        
        finally:
            cv2.destroyAllWindows()


def check_network_interface(interface: str = "eth0") -> bool:
    """检查网络接口状态"""
    try:
        result = subprocess.run(['ip', 'addr', 'show', interface], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[G1-Stack] 网络接口 {interface} 正常")
            return True
        else:
            print(f"[G1-Stack] 网络接口 {interface} 不存在或未激活")
            print("请检查网络配置或使用 --iface 参数指定正确的接口")
            return False
    except Exception as e:
        print(f"[G1-Stack] 检查网络接口时出错: {e}")
        return False


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="G1 机器人全栈控制系统 (增强版)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--iface", default="eth0", help="网络接口")
    parser.add_argument("--no-realsense", action="store_true", help="禁用 RealSense")
    parser.add_argument("--no-slam", action="store_true", help="禁用 SLAM")
    parser.add_argument("--no-robot", action="store_true", help="禁用机器人控制")
    parser.add_argument("--clear", type=float, default=0.15, 
                       help="地面间隙阈值（米），高于此值的点被视为障碍物")
    args = parser.parse_args()
    
    # 网络接口检查
    if not args.no_robot:
        if not check_network_interface(args.iface):
            print("网络接口检查失败，机器人控制可能无法正常工作")
    
    # 创建系统
    system = G1StackSystem(interface=args.iface)
    
    # 添加核心组件
    if not args.no_realsense:
        print("[G1-Stack] 启用 RealSense 组件")
        system.add_component(RealSenseComponent(system.state_manager))
    
    if not args.no_slam:
        print("[G1-Stack] 启用 SLAM 组件")
        system.add_component(SLAMComponent(system.state_manager))
        
        print(f"[G1-Stack] 启用占用网格组件 (地面间隙: {args.clear}m)")
        system.add_component(OccupancyGridComponent(system.state_manager, args.clear))
    
    if not args.no_robot:
        print("[G1-Stack] 启用机器人控制组件")
        system.add_component(RobotControlComponent(system.state_manager, args.iface))
        
        print("[G1-Stack] 启用电池监控组件")
        system.add_component(BatteryComponent(system.state_manager, args.iface))
    
    print(f"[G1-Stack] 总计启用 {len(system.components)} 个组件")
    
    try:
        # 启动系统
        system.start()
        
        # 运行显示循环
        system.run_display_loop()
    
    except KeyboardInterrupt:
        print("\n[G1-Stack] 接收到中断信号...")
    
    finally:
        system.stop()


if __name__ == "__main__":
    main()