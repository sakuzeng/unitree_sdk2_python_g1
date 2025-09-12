#!/usr/bin/env python3
import time
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from unitree_sdk2py.core.channel import ChannelSubscriber
import threading

class HandStatePrinter:
    """
    订阅 HandState_ 数据并打印所有字段内容
    """
    def __init__(self, topic: str = "rt/dex3/right/state", domain_id: int = 0):
        """
        初始化 HandState_ 订阅者

        Args:
            topic (str): 订阅的话题名称
            domain_id (int): DDS 域 ID
        """
        self.topic = topic
        self.domain_id = domain_id
        self.subscriber = None
        self.latest_state = None
        self.state_lock = threading.Lock()

    def initialize(self):
        """初始化 DDS 订阅者"""
        try:
            self.subscriber = ChannelSubscriber(self.topic, HandState_)
            self.subscriber.Init(self._state_callback, 10)
            print(f"[INFO] 成功订阅话题: {self.topic}")
        except Exception as e:
            print(f"[ERROR] 初始化订阅者失败: {e}")
            raise

    def _state_callback(self, msg: HandState_):
        """
        状态消息回调函数

        Args:
            msg (HandState_): 接收到的 HandState_ 数据
        """
        with self.state_lock:
            self.latest_state = msg

    def get_latest_state(self):
        """
        获取最新的 HandState_ 数据

        Returns:
            HandState_: 最新的 HandState_ 数据
        """
        with self.state_lock:
            return self.latest_state

    def print_hand_state(self):
        """打印 HandState_ 的所有字段内容"""
        state = self.get_latest_state()
        if not state:
            print("[WARNING] 未接收到 HandState_ 数据")
            return

        print("=== HandState_ 数据 ===")

        # 打印电机状态
        print("电机状态 (motor_state):")
        for i, motor in enumerate(state.motor_state):
            print(f"  电机 {i}: 角度={motor.q:.3f}, 速度={motor.dq:.3f}, 扭矩={motor.tau:.3f}")

        # 打印触觉传感器状态
        print("触觉传感器状态 (press_sensor_state):")
        for i, sensor in enumerate(state.press_sensor_state):
            print(f"  传感器 {i}: 压力={sensor.pressure:.3f}")

        # 打印 IMU 状态
        print("IMU 状态 (imu_state):")
        print(f"  加速度: x={state.imu_state.acc_x:.3f}, y={state.imu_state.acc_y:.3f}, z={state.imu_state.acc_z:.3f}")
        print(f"  角速度: x={state.imu_state.gyro_x:.3f}, y={state.imu_state.gyro_y:.3f}, z={state.imu_state.gyro_z:.3f}")

        # 打印电源信息
        print("电源信息:")
        print(f"  电源电压 (power_v): {state.power_v:.3f} V")
        print(f"  电源电流 (power_a): {state.power_a:.3f} A")
        print(f"  系统电压 (system_v): {state.system_v:.3f} V")
        print(f"  设备电压 (device_v): {state.device_v:.3f} V")

        # 打印错误信息
        print("错误信息 (error):")
        print(f"  错误代码: {list(state.error)}")

        # 打印保留字段
        print("保留字段 (reserve):")
        print(f"  保留字段: {list(state.reserve)}")

    def run(self):
        """运行订阅者并定期打印数据"""
        try:
            while True:
                self.print_hand_state()
                time.sleep(1.0)  # 每秒打印一次
        except KeyboardInterrupt:
            print("[INFO] 程序已终止")

if __name__ == "__main__":
    # 配置参数
    TOPIC = "rt/dex3/right/state"  # 替换为实际的话题名称
    DOMAIN_ID = 0  # DDS 域 ID

    # 初始化订阅者
    subscriber = HandStatePrinter(topic=TOPIC, domain_id=DOMAIN_ID)
    subscriber.initialize()

    # 运行订阅者
    subscriber.run()