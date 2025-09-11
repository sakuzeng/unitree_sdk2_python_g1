#!/usr/bin/env python3
"""
Dex3 API 简单测试脚本

测试内容：
- 读取当前关节角度
- 设置关节角度
- 抓握动作
- 停止电机
"""

from unitree_sdk_python.unitree_sdk2py.dex3.dex3_api import Dex3FingerAPI
import time

def main():
    print("=== Dex3 API 简单测试 ===")
    hand = "right"
    interface = "eth0"

    # 初始化API
    api = Dex3FingerAPI(hand=hand, interface=interface)

    # 读取当前关节角度
    angles = api.get_joint_angles()
    print(f"当前关节角度: {angles}")

    # 设置一个简单的目标角度
    if angles:
        target = [a + 0.1 for a in angles]
        print(f"设置目标角度: {target}")
        api.set_joint_angles(target)
        time.sleep(2)

    # 抓握动作
    print("执行抓握动作")
    api.grip_hand()
    time.sleep(2)

    # 停止电机
    print("停止电机")
    api.stop_motors()

    # 关闭资源
    api.close()
    print("测试结束")

if __name__ == "__main__":
    main()