#!/usr/bin/env python3
"""
Dex3 灵巧手控制示例（带启动序列）
"""
import time
import threading
from unitree_sdk2py.dex3.dex3_client import (
    Dex3Client, Dex3Gestures, dex3_connection
)
from hanger_boot_sequence import hanger_boot_sequence

def main():
    print("=== Dex3 灵巧手控制示例（带启动序列）===")
    
    # 首先执行启动序列，进入主运动控制模式
    print("执行机器人启动序列...")
    sport_client = hanger_boot_sequence(iface="eth0")
    print("机器人已进入主运动控制模式")
    
    # 使用上下文管理器确保资源正确释放
    with dex3_connection(hand="right", interface="eth0") as dex3:
        # 读取当前状态
        angles = dex3.get_joint_angles()
        if angles:
            print(f"初始关节角度: {angles}")
        
        # 打开手掌
        print("打开手掌...")
        open_hand = Dex3Gestures.get_gesture("open", "right")
        dex3.set_joint_angles(open_hand)
        time.sleep(2)
        
        # 读取触觉传感器
        print("读取触觉传感器...")
        pressure_data = dex3.get_fingertip_pressures()
        if pressure_data:
            print(f"指尖压力: {pressure_data}")
        
        # 执行抓取
        print("执行抓取...")
        dex3.grip_hand(grip_strength=1.5)
        time.sleep(2)
        
        # 再次打开手掌
        print("再次打开手掌...")
        dex3.set_joint_angles(open_hand)
        time.sleep(2)
        
        # 阻尼模式（可手动移动）
        print("进入阻尼模式，可手动移动手指（5秒）...")
        dex3.damp_motors(kd=0.5)
        time.sleep(5)
        
        print("测试完成")

if __name__ == "__main__":
    main()