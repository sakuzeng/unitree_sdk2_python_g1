"""
悬挂启动程序 (Hanger Boot Sequence)

本脚本为 Unitree G1 机器人提供一个从被动悬挂状态到自主平衡站立的安全启动序列。
它会引导用户完成机器人触地的过程，并自动切换到平衡站立模式。

使用方法:
    python3 hanger_boot_sequence.py --iface <network_interface>

例如:
    python3 hanger_boot_sequence.py eth0
"""
from __future__ import annotations

import time
import sys
import json
from typing import Optional
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


def hanger_boot_sequence(
    iface: str = "eth0",
    step: float = 0.02,
    max_height: float = 0.5,
) -> LocoClient:
    """
    执行机器人从悬挂状态到站立的启动序列。

    Args:
        iface (str): 用于 DDS 通信的网络接口名称，默认为 "eth0"。
        step (float): 每次增加的站立高度步长（米），默认为 0.02。
        max_height (float): 尝试达到的最大站立高度（米），默认为 0.5。

    Returns:
        LocoClient: 初始化并进入平衡站立模式后的运动控制客户端实例。
    """
    # 1. 初始化 DDS 通信和运动控制客户端
    ChannelFactoryInitialize(0, iface)

    sport_client = LocoClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    def get_mode(val) -> Optional[int]:
        """
        解析运动控制客户端返回的值，提取模式（mode）或状态ID（FSM ID）。
        返回值可能是 JSON 字符串或字典。
        """
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        if isinstance(val, dict) and "data" in val:
            return int(val["data"])
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    def show(tag: str) -> None:
        """打印当前操作和机器人的 FSM 及平衡状态信息。"""
        fsm_id = get_mode(sport_client.GetFsmId())
        fsm_mode = get_mode(sport_client.GetFsmMode())
        balance_mode = get_mode(sport_client.GetBalanceMode())
        print(f"{tag:<12} → FSM {fsm_id}   mode {fsm_mode}   balance {balance_mode}")
    show("initial")
    
    # 检查机器人是否已处于站立或运动状态 (FSM ID 200, FSM mode 0/1)
    # 如果是，则跳过启动序列，直接返回。
    print("=== 检查当前机器人状态 ===")
    try:
        # 使用已有的 get_mode 函数进行解析，而不是直接 int() 转换
        cur_id = get_mode(sport_client.GetFsmId())
        cur_mode = get_mode(sport_client.GetFsmMode())
        
        # 添加原始值调试输出
        raw_id = sport_client.GetFsmId()
        raw_mode = sport_client.GetFsmMode()
        print(f"原始 FSM ID: {raw_id} (类型: {type(raw_id)})")
        print(f"原始 FSM Mode: {raw_mode} (类型: {type(raw_mode)})")
        print(f"解析后 FSM ID: {cur_id}, FSM Mode: {cur_mode}")
        
        if cur_id == 200 and cur_mode != None and cur_mode != 2:
            print(f"✓ 机器人已处于平衡站立状态 (FSM {cur_id}, mode {cur_mode}) – 跳过启动序列。")
            return sport_client
        else:
            print(f"✗ 机器人未处于目标状态，执行完整启动序列")
            print(f"  当前状态: FSM {cur_id}, mode {cur_mode}")
            print(f"  目标状态: FSM 200, mode 0")
            
    except Exception as e:
        print(f"检查机器人状态时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("=== 状态检查完成 ===\n")




    # 2. 进入阻尼模式 (Damp, FSM ID: 1)
    # 此时关节有阻力，但不会主动运动。
    sport_client.Damp()
    show("Damp")

    input("请确认机器人双足已触地，然后按回车键继续...")

    # 3. 进入预备站立模式 (StandUp, FSM ID: 4)
    # 机器人会进入一个固定的站立姿态，但足底可能还未承重。
    sport_client.StandUp()
    show("StandUp")

    # 4. 自动检测触地与调整高度
    # 循环检测机器人是否已承重站立 (mode 0)，如果未达到则逐步增加高度。
    while True:
        # 如果当前已是站立模式 (mode 0)，则退出循环。
        if get_mode(sport_client.GetFsmMode()) == 0:
            print("机器人已处于站立状态 (mode=0)，无需重复检测。")
            break

        # 提示用户进行物理确认
        input("请再次确认机器人双足已触地，然后按回车键开始高度调整...")

        # 逐步增加目标站立高度，直到检测到足底承重 (mode 0)。
        height = 0.0
        while height < max_height:
            height += step
            sport_client.SetStandHeight(height)
            show(f"height {height:.2f} m")
            if get_mode(sport_client.GetFsmMode()) == 0:
                print(f"检测到机器人进入站立状态 (mode=0)，当前高度：{height:.2f} m")
                break

        # 如果成功站立，则跳出外层循环。
        if get_mode(sport_client.GetFsmMode()) == 0:
            break

        # 如果达到最大高度仍未站立，提示用户手动调整并重试。
        print(
            f"在达到 {height:.2f} 米高度后，足底仍未承重 (mode {get_mode(sport_client.GetFsmMode())})。\n"
            "请调整悬挂架高度，确保双足刚好接触地面，然后按回车重试…"
        )
        try:
            sport_client.SetStandHeight(0.0)
            show("reset")
        except Exception:
            pass
        input()  # 等待用户操作

    # 5. 设置为平衡站立模式 (BalanceMode 0)
    sport_client.SetBalanceMode(0)
    show("balance")

    # 6. 启动主运动控制器 (Start, FSM ID: 200)
    # 机器人将进入可以自主平衡和移动的状态。
    input("警告：即将启动主运控。请确保机器人姿态大致直立，否则初始平衡动作可能较大。按回车继续...")
    sport_client.Start()
    show("Start")

    return sport_client


def main():
    """
    主函数，用于解析命令行参数并执行悬挂启动序列。
    """
    parser = argparse.ArgumentParser(description="Unitree G1 悬挂启动测试程序")
    parser.add_argument("--iface", default="eth0", help="连接到机器人的网络接口")
    args = parser.parse_args()

    sport_client = hanger_boot_sequence(iface=args.iface)
    print("\n启动序列完成。现在可以向机器人发送运动指令。")
    print(f"当前 FSM ID: {sport_client.GetFsmId()}, FSM Mode: {sport_client.GetFsmMode()}")


if __name__ == "__main__":
    # 导入 argparse 仅在直接运行脚本时需要
    import argparse

    main()

# 定义当使用 from hanger_boot_sequence import * 时，哪些名称会被导入
__all__ = ["hanger_boot_sequence"]



