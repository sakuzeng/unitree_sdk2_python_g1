'''
@Author: sakuzeng1213
@Date: 2025-08-25 12:28:19
@LastEditTime: 2025-08-26 15:53:27
@LastEditors: sakuzeng1213
@FilePath: /unitree_sdk2_python_g1/hanger_boot_sequence.py
@Description: 
悬挂启动程序

'''
from __future__ import annotations

import time
import sys
import json
from typing import Optional
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

#--------------------------------------------------------------------
# return: LocoClient instance,fsm id:200,fsm mode 0/1
#--------------------------------------------------------------------
def hanger_boot_sequence(
    iface: str = "eth0",
    step: float = 0.02,
    max_height: float = 0.5,
) -> LocoClient:    

    # DDS initialisation ---------------------------------------------------
    ChannelFactoryInitialize(0, iface)

    sport_client = LocoClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    #------------------------------------------------------------------------------------------------
    # 当前已经处于主运控(fsm id:200)且当前fsm mode为0/1（处于站立状态或移动状态）时直接返回当前client
    #------------------------------------------------------------------------------------------------
    try:
        cur_id = int(sport_client.GetFsmId())
        cur_mode = int(sport_client.GetFsmMode())
        if cur_id == 200 and cur_mode is not None and cur_mode != 2:
            print(
                f"Rosport_client already in balanced stand (FSM 200, mode {cur_mode}) – skipping boot sequence."
            )
            return sport_client
    except Exception:
        pass

    # 解析函数返回值，返回int型  ----------------------------------------------------------------
    def get_mode(val):
        # 如果是字符串且是json格式，先解析
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                pass
        # 如果是字典，取data
        if isinstance(val, dict) and "data" in val:
            return int(val["data"])
        # 如果已经是数字
        try:
            return int(val)
        except Exception:
            return val
    # 打印FSM信息  ----------------------------------------------------------------
    def show(tag: str) -> None:
        print(f"{tag:<12} → FSM {get_mode(sport_client.GetFsmId())}   mode {get_mode(sport_client.GetFsmMode())}")

    # - 1. Damp 阻尼模式 fsm:1 --------------------------------------------------------------
    sport_client.Damp(); show("Damp")

    input("确认机器人双足已触地...")

    # - 2. Stand-up 预备模式（锁定站立）fsm:4 ----------------------------------------------------------
    sport_client.StandUp(); show("StandUp")

    #------------------------------------------------------------------------------------------------
    # 3. 检测 mode 是否为 0，如果是则直接跳出循环，否则提示用户确认机器人触地，然后逐步增加站立高度直到 mode 为 0
    #------------------------------------------------------------------------------------------------
    while True:
        # 先检测当前 mode
        if get_mode(sport_client.GetFsmMode()) == 0:
            print("机器人已处于站立状态（mode=0），无需重复检测。")
            break

        # 否则提示用户确认机器人双足已触地
        input("请确认机器人双足已触地后按回车继续...")

        # 逐步增加站立高度，直到 mode 为 0
        height = 0.0
        while height < max_height:
            height += step
            sport_client.SetStandHeight(height)
            show(f"height {height:.2f} m")
            if get_mode(sport_client.GetFsmMode()) == 0:
                print(f"检测到机器人进入站立状态（mode=0），当前高度：{height:.2f} m")
                break

        # 如果已经进入 mode=0，跳出外层循环
        if get_mode(sport_client.GetFsmMode()) == 0:
            break

        # 如果还未进入 mode=0，提示用户调整悬挂架并重试
        print(
            f"Feet still unloaded (mode {get_mode(sport_client.GetFsmMode())}) after reaching {height:.2f} m.\n"
            "请调整悬挂架高度（升高或降低），确保双足刚好接触地面，然后按回车重试…"
        )
        try:
            sport_client.SetStandHeight(0.0)
            show("reset")
        except Exception:
            pass
        input()  # 等待用户

    # - 4. BalanceMode: 0(平衡站立)  ----------------------------------------------------------------
    sport_client.SetBalanceMode(0); show("balance")
    # sport_client.SetStandHeight(height); show("height✔")

    # - 5. Start 主运控  fsm id: 200  ----------------------------------------------------------------
    input("尽量将机器人正常直立，否则刚开始确立平衡动静大...")
    sport_client.Start(); show("Start")
    # - 6. BalanceMode: 1(连续步态)  ----------------------------------------------------------------
    # sport_client.SetBalanceMode(1); show("ContinuousGait")

    return sport_client

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test hanger_boot_sequence for Unitree G1")
    parser.add_argument("--iface", default="eth0", help="network interface connected to rosport_client")
    args = parser.parse_args()

    sport_client = hanger_boot_sequence(iface=args.iface)
    print("Boot sequence finished. You can now send velocity commands to the rosport_client.")
    print(f"Current FSM ID: {sport_client.GetFsmId()}, FSM Mode: {sport_client.GetFsmMode()}")

if __name__ == "__main__":
    main()

# 用于定义当使用 from hanger_boot_sequence import * 时，哪些名称会被导入。
__all__ = ["hanger_boot_sequence"]



