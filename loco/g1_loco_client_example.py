"""
Unitree G1 机器人高级运动控制 (LocoClient) 交互式测试程序

本脚本提供一个命令行界面，用于测试 G1 机器人的各种高级运动能力。
用户可以通过输入指令ID或名称来调用对应的机器人动作，例如站立、行走、挥手等。

使用方法:
    python3 g1_loco_client_example.py <network_interface>

示例:
    python3 g1_loco_client_example.py eth0
"""
import time
import sys
from dataclasses import dataclass
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


@dataclass
class TestOption:
    """存储一个测试选项的名称和ID。"""
    name: str
    id: int

# 定义所有可用的测试选项列表
option_list = [
    TestOption(name="zero torque", id=0),      # 零力矩模式
    TestOption(name="damp", id=1),             # 阻尼模式
    TestOption(name="StandUp", id=2),          # 进入站立模式
    TestOption(name="Start", id=3),            # 启动主运控模式
    TestOption(name="move forward", id=4),     # 向前移动
    TestOption(name="move back", id=5),        # 向后移动
    TestOption(name="move rotate", id=6),      # 旋转移动
    TestOption(name="low stand", id=7),        # 降低站立高度
    TestOption(name="high stand", id=8),       # 升高站立高度
    TestOption(name="wave hand1", id=9),       # 原地挥手
    TestOption(name="wave hand2", id=10),      # 转身并挥手
    TestOption(name="shake hand", id=11),      # 握手
    TestOption(name="Lie2StandUp", id=12),     # 从躺姿站立
    TestOption(name="Squat2StandUp", id=13),   # 从蹲姿站立
    TestOption(name="StandUp2Squat", id=14),   # 从站姿蹲下
    TestOption(name="getFsmStatus", id=15),    # 获取FSM状态
]


class UserInterface:
    """处理用户在终端的输入，并将其解析为对应的测试选项。"""
    def __init__(self):
        """初始化用户界面，准备存储用户选择的测试选项。"""
        self.test_option_: TestOption = None

    def convert_to_int(self, input_str: str):
        """
        尝试将字符串转换为整数。

        Args:
            input_str (str): 用户输入的字符串。

        Returns:
            int or None: 如果转换成功，返回整数；否则返回 None。
        """
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        """
        从终端获取用户输入，并更新 test_option_。
        - 输入 'list' 可查看所有可用指令。
        - 输入指令ID或名称可选择一个指令。
        """
        input_str = input("Enter id or name (or 'list' to show all options): \n")

        # 如果输入 'list'，则打印所有选项
        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"id: {option.id}, name: {option.name}")
            return

        # 遍历列表，查找匹配的指令
        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Selected: {self.test_option_.name} (id: {self.test_option_.id})")
                return

        print("No matching test option found.")


if __name__ == "__main__":
    # 检查是否提供了网络接口参数
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface>")
        print("Example: python3 g1_loco_client_example.py eth0")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    # 1. 初始化DDS通信
    ChannelFactoryInitialize(0, sys.argv[1])

    # 2. 初始化用户界面和机器人控制客户端
    test_option = TestOption(name=None, id=None)
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = LocoClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    # 3. 进入主循环，等待用户输入并执行指令
    while True:
        user_interface.terminal_handle()

        if test_option.id is None:
            continue

        # 根据用户选择的ID执行相应的机器人动作
        if test_option.id == 0:
            sport_client.ZeroTorque()
        elif test_option.id == 1:
            sport_client.Damp()
        elif test_option.id == 2:
            sport_client.StandUp()
        elif test_option.id == 3:
            sport_client.Start()
        elif test_option.id == 4:
            sport_client.Move(0.3, 0, 0)
        elif test_option.id == 5:
            sport_client.Move(-0.3, 0, 0)
        elif test_option.id == 6:
            sport_client.Move(0, 0, 0.3)
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
        elif test_option.id == 15:
            fsm_id = sport_client.GetFsmId()
            fsm_mode = sport_client.GetFsmMode()
            balance_mode = sport_client.GetBalanceMode()
            print(f"Current FSM Status - ID: {fsm_id}, Mode: {fsm_mode}, Balance Mode: {balance_mode}")
            continue  # 跳过后续的延时和状态打印

        # 等待1秒，以便观察动作效果
        time.sleep(1)

        # 每次动作后输出当前 FSM 状态，帮助调试
        fsm_id = sport_client.GetFsmId()
        fsm_mode = sport_client.GetFsmMode()
        balance_mode = sport_client.GetBalanceMode()
        print(f"Current FSM ID: {fsm_id}, FSM Mode: {fsm_mode}, Balance Mode: {balance_mode}")