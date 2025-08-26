'''
@Author: sakuzeng1213
@Date: 2025-08-22 12:13:16
@LastEditTime: 2025-08-26 18:04:38
@LastEditors: sakuzeng1213
@FilePath: /unitree_sdk2_python_g1/g1_loco_client_example.py
@Description: 
    # g1运动模块的测试
    # 测试启动运动过程
    # damp(1)->stand_up(2)->start(3)->move forward(4)
'''
import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
# from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from dataclasses import dataclass

@dataclass
class TestOption:
    name: str
    id: int
    
option_list = [
    TestOption(name="zero torque", id=0),      # 零力矩模式
    TestOption(name="damp", id=1),             # 阻尼模式
    TestOption(name="StandUp", id=2),          # 进入站立模式
    TestOption(name="Start", id=3),            # 主运控模式
    TestOption(name="move forward", id=4),     # 向前移动
    TestOption(name="move back", id=5),        # 向后移动
    TestOption(name="move rotate", id=6),      # 旋转移动
    TestOption(name="low stand", id=7),        # 站的低
    TestOption(name="high stand", id=8),       # 站的高
    TestOption(name="wave hand1", id=9),       # wave hand without turning around
    TestOption(name="wave hand2", id=10),      # wave hand and trun around  
    TestOption(name="shake hand", id=11),      # 握手    
    TestOption(name="Lie2StandUp", id=12),     # 躺到站立
    TestOption(name="Squat2StandUp", id=13),   # 蹲下到站立
    TestOption(name="StandUp2Squat", id=14),   # 站立到蹲下
]

class UserInterface:
    def __init__(self):
        # 初始化时，定义一个属性 test_option_，用于存储当前选中的测试选项
        self.test_option_ = None
    # 字符串转整数方法
    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        # 提示用户输入动作名称或编号。
        input_str = input("Enter id or name: \n")
        # 如果输入 list，则打印所有可选动作及编号，方便用户查阅
        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return
        # 遍历 option_list，如果输入与某个选项的名称或编号匹配，则更新当前选项并输出选中信息。
        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        print("No matching test option found.")

if __name__ == "__main__":
    # 需要参数：g1有线网口：eth0
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = LocoClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    while True:
        user_interface.terminal_handle()

        print(f"Updated Test Option: Name = {test_option.name}, ID = {test_option.id}")

        if test_option.id == 0:
            sport_client.ZeroTorque()
        elif test_option.id == 1:
            sport_client.Damp()
        elif test_option.id == 2:
            sport_client.StandUp()
        elif test_option.id == 3:
            sport_client.Start()
        elif test_option.id == 4:
            sport_client.Move(0.3,0,0)
        elif test_option.id == 5:
            sport_client.Move(-0.3,0,0)
        elif test_option.id == 6:
            sport_client.Move(0,0,0.3)
        elif test_option.id == 7:
            sport_client.LowStand()
        elif test_option.id == 8:
            sport_client.HighStand()
        elif test_option.id == 9:
            sport_client.WaveHand()
        elif test_option.id == 10:
            sport_client.WaveHand(True)
        elif test_option.id == 11:
            sport_client.ShakeHand()
            time.sleep(3)
            sport_client.ShakeHand()
        elif test_option.id == 12:
            sport_client.Damp()
            time.sleep(0.5)
            sport_client.Lie2StandUp()
        elif test_option.id == 13:
            sport_client.Squat2StandUp()
        elif test_option.id == 14:
            sport_client.StandUp2Squat()
        time.sleep(1)
        # 每次动作后输出当前 FSM ID 和 Mode
        fsm_id = sport_client.GetFsmId()
        fsm_mode = sport_client.GetFsmMode()
        print(f"Current FSM ID: {fsm_id}, FSM Mode: {fsm_mode}")