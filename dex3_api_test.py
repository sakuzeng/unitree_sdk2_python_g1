#!/usr/bin/env python3
"""
Dex3FingerAPI 测试脚本

用法:
    python3 dex3_api_test.py [eth0]

- 获取当前关节角度并保存到文件
- 设置关节角度为目标值（带限位检查）
- 测试抓握功能
"""

import sys
import time
from unitree_sdk2py.dex3.dex3_api import Dex3FingerAPI

def main():
    interface = sys.argv[1] if len(sys.argv) > 1 else "eth0"
    dex3_api = Dex3FingerAPI(hand="right", interface=interface)

    try:
        # 获取当前关节角度
        print("正在获取当前关节角度...")
        angles = dex3_api.get_joint_angles(timeout=2.0)
        if angles is None:
            print("获取关节角度失败，请检查网络和DDS配置")
            return
        print("当前关节角度:", [f"{a:.6f}" for a in angles])

        # 保存到文件
        with open("dex3_jointangles.txt", "w") as f:
            f.write(",".join([f"{a:.6f}" for a in angles]) + "\n")
        print("已保存到 dex3_jointangles.txt")

        # 设置目标关节角度（示例：略微张开）
        target_angles = [min(a + 0.1, dex3_api.joint_limits[i][1]) for i, a in enumerate(angles)]
        print("设置目标关节角度:", [f"{a:.6f}" for a in target_angles])
        ok = dex3_api.set_joint_angles(target_angles)
        if ok:
            print("关节角度设置成功")
        else:
            print("关节角度设置失败")

        # 等待2秒后恢复原始角度
        time.sleep(2)
        ok = dex3_api.set_joint_angles(angles)
        if ok:
            print("已恢复原始关节角度")
        else:
            print("恢复原始关节角度失败")

        # 测试抓握功能
        print("测试抓握功能...")
        ok = dex3_api.grip_hand()
        if ok:
            print("抓握命令发送成功")
        else:
            print("抓握命令发送失败")

        # 等待2秒后获取当前角度
        time.sleep(2)
        angles = dex3_api.get_joint_angles()
        if angles:
            print("抓握后的关节角度:", [f"{a:.6f}" for a in angles])
        else:
            print("获取抓握后关节角度失败")

        # 停止电机
        dex3_api.stop_motors()
        print("已发送停止电机命令")

    finally:
        dex3_api.close()

if __name__ == "__main__":
    main()