import sys
import time
import signal
from typing import Optional
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.arm.arm_client import G1ArmClient
from unitree_sdk2py.dex3.dex3_client import Dex3Client

def cleanup(arm_client: Optional[G1ArmClient], hand_client: Optional[Dex3Client]):
    """清理函数，确保在中断或异常时停止控制"""
    print("\n[清理] 正在停止手臂和手指控制...")
    try:
        if arm_client:
            arm_client.stop_control()
            print("[清理] 手臂控制已停止")
    except Exception as e:
        print(f"[清理] 停止手臂控制失败: {e}")
    try:
        if hand_client:
            hand_client.stop_control()
            print("[清理] 手指控制已停止")
    except Exception as e:
        print(f"[清理] 停止手指控制失败: {e}")

def signal_handler(sig, frame, arm_client: Optional[G1ArmClient], hand_client: Optional[Dex3Client]):
    """处理 SIGINT 信号（Ctrl+C）"""
    print("\n[中断] 检测到 Ctrl+C，清理并退出...")
    cleanup(arm_client, hand_client)
    sys.exit(0)

def main():
    # 初始化 DDS 通信
    interface = "eth0"  # 根据实际网络接口修改，例如 "wlan0"
    ChannelFactoryInitialize(0, interface)
    
    # 创建手臂和手指客户端
    arm_client = None
    hand_client = None
    try:
        arm_client = G1ArmClient(interface=interface)
        hand_client = Dex3Client(hand="left", interface=interface)
    except Exception as e:
        print(f"[错误] 初始化客户端失败: {e}")
        cleanup(arm_client, hand_client)
        return
    
    # 注册信号处理
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, arm_client, hand_client))
    
    try:
        # 步骤 1: 初始化手臂和手指
        print("步骤 1: 初始化手臂和手指")
        input("按回车键开始初始化...")
        if not arm_client.initialize_arms():
            print("[错误] 手臂初始化失败")
            cleanup(arm_client, hand_client)
            return
        if not hand_client.initialize_hand():
            print("[错误] 手指初始化失败")
            cleanup(arm_client, hand_client)
            return
        print("手臂和手指初始化完成")
        time.sleep(1.0)
        
        # 步骤 2: 将手臂设置到 press_0
        print("步骤 2: 设置手臂到 press_0")
        input("按回车键继续...")
        if not arm_client.set_arm_pose("press_0"):
            print("[错误] 设置手臂到 press_0 失败")
            cleanup(arm_client, hand_client)
            return
        time.sleep(1.0)
        
        # 步骤 3: 将手指设置到 press
        print("步骤 3: 设置手指到 press")
        input("按回车键继续...")
        if not hand_client.set_gesture("press"):
            print("[错误] 设置手指到 press 失败")
            cleanup(arm_client, hand_client)
            return
        time.sleep(1.0)
        
        # 步骤 4: 将手臂设置到 press_1
        print("步骤 4: 设置手臂到 press_1")
        input("按回车键继续...")
        if not arm_client.set_arm_pose("press_1"):
            print("[错误] 设置手臂到 press_1 失败")
            cleanup(arm_client, hand_client)
            return
        time.sleep(1.0)
        
        # 步骤 5: 将手指设置到 nature
        print("步骤 5: 设置手指到 nature")
        input("按回车键继续...")
        if not hand_client.set_gesture("nature"):
            print("[错误] 设置手指到 nature 失败")
            cleanup(arm_client, hand_client)
            return
        time.sleep(1.0)
        
        # 步骤 6: 将手臂设置到 nature
        print("步骤 6: 设置手臂到 nature")
        input("按回车键继续...")
        if not arm_client.set_arm_pose("nature"):
            print("[错误] 设置手臂到 nature 失败")
            cleanup(arm_client, hand_client)
            return
        time.sleep(1.0)
        
        # 步骤 7: 停止控制手臂和手指
        print("步骤 7: 停止控制手臂和手指")
        input("按回车键继续...")
        cleanup(arm_client, hand_client)
        print("手臂和手指控制已停止")
        
        print("所有步骤完成！")
    
    except Exception as e:
        print(f"[错误] 程序异常: {e}")
        cleanup(arm_client, hand_client)
        sys.exit(1)

if __name__ == "__main__":
    main()