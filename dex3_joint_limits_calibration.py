#!/usr/bin/env python3
"""
Dex3 灵巧手关节限位校准和保存工具

通过读取实际硬件状态来获取关节限位，并保存到配置文件。
"""

import sys
import time
import json
import csv
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# 导入启动序列模块
from hanger_boot_sequence import hanger_boot_sequence

# 导入必要的 SDK 模块
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, MotorCmd_
    SDK_AVAILABLE = True
except ImportError as e:
    print(f"警告: Unitree SDK 不可用: {e}")
    print("将运行在模拟模式下")
    SDK_AVAILABLE = False

@dataclass
class JointLimitData:
    """关节限位数据结构"""
    joint_id: int
    joint_name: str
    min_angle: float
    max_angle: float
    current_angle: float
    samples_count: int
    description: str = ""

@dataclass
class CalibrationSession:
    """校准会话数据"""
    timestamp: str
    hand: str
    interface: str
    joint_limits: List[Dict[str, Any]]
    calibration_method: str
    notes: str = ""

class Dex3JointLimitsCalibrator:
    """Dex3 关节限位校准器"""
    
    def __init__(self, hand: str = "right", interface: str = "eth0"):
        self.hand = hand
        self.interface = interface
        self.is_connected = False
        
        # DDS 通信
        self._cmd_topic = f"rt/dex3/{hand}/cmd"
        self._state_topic = f"rt/dex3/{hand}/state"
        self._cmd_publisher: Optional[ChannelPublisher] = None
        self._state_subscriber: Optional[ChannelSubscriber] = None
        
        # 状态数据
        self._latest_state = None
        self._state_lock = threading.Lock()
        
        # 校准数据
        self.joint_names = [
            "thumb_0",    # 拇指旋转
            "thumb_1",    # 拇指弯曲1
            "thumb_2",    # 拇指弯曲2
            "middle_0",   # 中指弯曲1
            "middle_1",   # 中指弯曲2
            "index_0",    # 食指弯曲1
            "index_1",    # 食指弯曲2
        ]
        
        self.joint_descriptions = [
            "拇指旋转关节 - 控制拇指左右旋转",
            "拇指弯曲关节1 - 控制拇指根部弯曲",
            "拇指弯曲关节2 - 控制拇指指尖弯曲",
            "中指弯曲关节1 - 控制中指根部弯曲",
            "中指弯曲关节2 - 控制中指指尖弯曲",
            "食指弯曲关节1 - 控制食指根部弯曲",
            "食指弯曲关节2 - 控制食指指尖弯曲",
        ]
        
        # 校准数据存储
        self.joint_angle_samples: Dict[int, List[float]] = {i: [] for i in range(7)}
        self.is_calibrating = False
        
        # 默认控制参数
        self.default_kp = 8.0
        self.default_kd = 1.5
        
        # 初始化连接
        if SDK_AVAILABLE:
            self._init_connection()
    
    def _init_connection(self):
        """初始化DDS连接"""
        try:
            # 确保先初始化 DDS
            ChannelFactoryInitialize(0, self.interface)
            
            # 创建发布者
            self._cmd_publisher = ChannelPublisher(self._cmd_topic, HandCmd_)
            self._cmd_publisher.Init()
            
            # 创建订阅者
            self._state_subscriber = ChannelSubscriber(self._state_topic, HandState_)
            self._state_subscriber.Init(self._state_callback, 100)
            
            self.is_connected = True
            print(f"✓ {self.hand}手连接成功，开始校准程序")
            
        except Exception as e:
            print(f"✗ {self.hand}手连接失败: {e}")
            traceback.print_exc()
            self.is_connected = False
    
    def _state_callback(self, msg):
        """状态消息回调"""
        with self._state_lock:
            self._latest_state = msg
            
            # 如果正在校准，记录关节角度
            if self.is_calibrating and hasattr(msg, 'motor_state') and len(msg.motor_state) >= 7:
                for i in range(7):
                    angle = float(msg.motor_state[i].q)
                    self.joint_angle_samples[i].append(angle)
    
    def get_current_joint_angles(self, timeout: float = 0.5) -> Optional[List[float]]:
        """获取当前关节角度"""
        if not SDK_AVAILABLE:
            # 模拟数据
            return [0.0] * 7
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._state_lock:
                if self._latest_state and hasattr(self._latest_state, 'motor_state'):
                    if len(self._latest_state.motor_state) >= 7:
                        try:
                            angles = [float(ms.q) for ms in self._latest_state.motor_state[:7]]
                            return angles
                        except Exception as e:
                            print(f"解析关节角度失败: {e}")
            time.sleep(0.01)
        
        print("获取关节角度超时")
        return None
    
    def start_calibration_recording(self):
        """开始记录校准数据"""
        print("开始记录关节角度数据...")
        self.is_calibrating = True
        
        # 清空之前的数据
        for i in range(7):
            self.joint_angle_samples[i].clear()
    
    def stop_calibration_recording(self):
        """停止记录校准数据"""
        print("停止记录关节角度数据")
        self.is_calibrating = False
    
    def get_calibration_statistics(self) -> List[JointLimitData]:
        """获取校准统计数据"""
        joint_limits = []
        
        for i in range(7):
            angles = self.joint_angle_samples[i]
            if len(angles) > 0:
                min_angle = float(np.min(angles))
                max_angle = float(np.max(angles))
                current_angle = float(np.mean(angles[-10:])) if len(angles) >= 10 else float(np.mean(angles))
                
                joint_limit = JointLimitData(
                    joint_id=i,
                    joint_name=self.joint_names[i],
                    min_angle=min_angle,
                    max_angle=max_angle,
                    current_angle=current_angle,
                    samples_count=len(angles),
                    description=self.joint_descriptions[i]
                )
                joint_limits.append(joint_limit)
            else:
                # 使用默认值
                joint_limit = JointLimitData(
                    joint_id=i,
                    joint_name=self.joint_names[i],
                    min_angle=-1.57 if i == 0 else -0.2,
                    max_angle=1.57 if i == 0 else 1.6,
                    current_angle=0.0,
                    samples_count=0,
                    description=self.joint_descriptions[i]
                )
                joint_limits.append(joint_limit)
        
        return joint_limits
    
    def interactive_calibration(self):
        """交互式校准过程"""
        print("=== Dex3 灵巧手关节限位交互式校准 ===")
        print("请按照提示手动移动每个关节到极限位置")
        print("按 Enter 键开始...")
        input()
        
        if not self.is_connected:
            print("未连接到灵巧手，使用模拟数据")
        
        # 开始记录
        self.start_calibration_recording()
        
        try:
            for i, (joint_name, description) in enumerate(zip(self.joint_names, self.joint_descriptions)):
                print(f"\n--- 校准关节 {i}: {joint_name} ---")
                print(f"描述: {description}")
                print("请手动移动此关节到最小位置，然后按 Enter...")
                input()
                
                # 记录最小位置
                min_angles = []
                for _ in range(10):  # 记录10个样本
                    angles = self.get_current_joint_angles()
                    if angles:
                        min_angles.append(angles[i])
                    time.sleep(0.1)
                
                if min_angles:
                    min_angle = np.mean(min_angles)
                    print(f"记录最小角度: {min_angle:.3f} rad ({min_angle*180/np.pi:.1f}°)")
                
                print("现在请移动此关节到最大位置，然后按 Enter...")
                input()
                
                # 记录最大位置
                max_angles = []
                for _ in range(10):  # 记录10个样本
                    angles = self.get_current_joint_angles()
                    if angles:
                        max_angles.append(angles[i])
                    time.sleep(0.1)
                
                if max_angles:
                    max_angle = np.mean(max_angles)
                    print(f"记录最大角度: {max_angle:.3f} rad ({max_angle*180/np.pi:.1f}°)")
                
                print(f"关节 {i} 校准完成")
        
        except KeyboardInterrupt:
            print("\n校准被中断")
        finally:
            self.stop_calibration_recording()
    
    def automatic_calibration(self, duration: float = 60.0):
        """自动校准过程 - 持续记录指定时间内的关节运动"""
        print(f"=== Dex3 灵巧手关节限位自动校准 ===")
        print(f"将持续记录 {duration} 秒内的关节运动")
        print("请在此期间手动操作灵巧手，让每个关节都运动到极限位置")
        print("按 Enter 键开始...")
        input()
        
        if not self.is_connected:
            print("未连接到灵巧手，使用模拟数据")
        
        # 开始记录
        self.start_calibration_recording()
        
        try:
            start_time = time.time()
            last_print_time = start_time
            
            print(f"开始自动校准，剩余时间: {duration:.0f} 秒")
            
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # 每5秒打印一次进度
                if current_time - last_print_time >= 5.0:
                    remaining_time = duration - (current_time - start_time)
                    print(f"校准进行中，剩余时间: {remaining_time:.0f} 秒")
                    
                    # 显示当前统计
                    for i in range(7):
                        samples = len(self.joint_angle_samples[i])
                        if samples > 0:
                            angles = self.joint_angle_samples[i]
                            min_a, max_a = np.min(angles), np.max(angles)
                            print(f"  关节{i}: {samples}个样本, 范围: [{min_a:.3f}, {max_a:.3f}] rad")
                    
                    last_print_time = current_time
                
                time.sleep(0.1)
            
            print("自动校准完成")
            
        except KeyboardInterrupt:
            print("\n校准被中断")
        finally:
            self.stop_calibration_recording()
    
    def save_calibration_results(self, output_dir: str = "config"):
        """保存校准结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取校准统计数据
        joint_limits = self.get_calibration_statistics()
        
        # 创建校准会话数据
        from datetime import datetime
        session = CalibrationSession(
            timestamp=datetime.now().isoformat(),
            hand=self.hand,
            interface=self.interface,
            joint_limits=[asdict(limit) for limit in joint_limits],
            calibration_method="interactive" if hasattr(self, '_interactive_mode') else "automatic",
            notes=f"校准了 {len(joint_limits)} 个关节的限位"
        )
        
        # 保存为 JSON 格式
        json_file = output_path / f"dex3_{self.hand}_joint_limits.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(session), f, indent='\t', ensure_ascii=False)
        print(f"✓ 校准结果已保存到: {json_file}")
        
        # 保存为 CSV 格式
        csv_file = output_path / f"dex3_{self.hand}_joint_limits.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['joint_id', 'joint_name', 'min_angle', 'max_angle', 
                         'current_angle', 'samples_count', 'description']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for limit in joint_limits:
                writer.writerow(asdict(limit))
        print(f"✓ CSV 格式已保存到: {csv_file}")
        
        # 保存原始采样数据
        raw_data_file = output_path / f"dex3_{self.hand}_raw_samples.json"
        raw_data = {
            'timestamp': session.timestamp,
            'hand': self.hand,
            'samples': {str(i): angles for i, angles in self.joint_angle_samples.items()}
        }
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent='\t')
        print(f"✓ 原始采样数据已保存到: {raw_data_file}")
        
        return session
    
    def load_calibration_results(self, config_file: str) -> Optional[CalibrationSession]:
        """加载校准结果"""
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"配置文件不存在: {config_file}")
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            session = CalibrationSession(**data)
            print(f"✓ 校准结果已加载: {config_file}")
            print(f"  时间戳: {session.timestamp}")
            print(f"  手部: {session.hand}")
            print(f"  关节数量: {len(session.joint_limits)}")
            
            return session
            
        except Exception as e:
            print(f"加载校准结果失败: {e}")
            return None
    
    def print_calibration_summary(self):
        """打印校准结果摘要"""
        joint_limits = self.get_calibration_statistics()
        
        print("\n=== 关节限位校准结果摘要 ===")
        print(f"手部: {self.hand}")
        print(f"总关节数: {len(joint_limits)}")
        
        print("\n详细信息:")
        for limit in joint_limits:
            range_rad = limit.max_angle - limit.min_angle
            range_deg = range_rad * 180.0 / np.pi
            print(f"关节{limit.joint_id} ({limit.joint_name}):")
            print(f"  限位: [{limit.min_angle:.3f}, {limit.max_angle:.3f}] rad")
            print(f"  范围: {range_deg:.1f}°")
            print(f"  当前: {limit.current_angle:.3f} rad")
            print(f"  样本: {limit.samples_count} 个")
            print(f"  描述: {limit.description}")
            print()
    
    def export_to_python_code(self, output_file: str = "dex3_joint_limits_generated.py"):
        """将校准结果导出为 Python 代码"""
        joint_limits = self.get_calibration_statistics()
        
        code = f'''#!/usr/bin/env python3
"""
自动生成的 Dex3 {self.hand}手关节限位配置

生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import List, Tuple

# Dex3 {self.hand}手关节限位配置
DEX3_{self.hand.upper()}_JOINT_LIMITS = [
'''
        
        for limit in joint_limits:
            code += f'\t({limit.min_angle:.6f}, {limit.max_angle:.6f}),  # {limit.joint_name}: {limit.description}\n'
        
        code += ''']

def get_dex3_joint_limits() -> List[Tuple[float, float]]:
    """获取 Dex3 关节限位"""
    return DEX3_{self.hand.upper()}_JOINT_LIMITS.copy()

def get_joint_limit(joint_index: int) -> Tuple[float, float]:
    """获取指定关节的限位"""
    if not 0 <= joint_index < len(DEX3_{self.hand.upper()}_JOINT_LIMITS):
        raise IndexError(f"关节索引超出范围: {{joint_index}}")
    return DEX3_{self.hand.upper()}_JOINT_LIMITS[joint_index]

if __name__ == "__main__":
    print("Dex3 {self.hand}手关节限位:")
    for i, (min_a, max_a) in enumerate(DEX3_{self.hand.upper()}_JOINT_LIMITS):
        range_deg = (max_a - min_a) * 180.0 / 3.14159
        print(f"  关节{{i}}: [{{min_a:.3f}}, {{max_a:.3f}}] rad ({{range_deg:.1f}}°)")
'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"✓ Python 代码已导出到: {output_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dex3 灵巧手关节限位校准工具")
    parser.add_argument("--iface", default="eth0", help="网络接口名称")
    parser.add_argument("--hand", choices=["left", "right"], default="right", help="要校准的手")
    parser.add_argument("--mode", choices=["interactive", "auto"], default="interactive", 
                        help="校准模式: interactive(交互式) 或 auto(自动)")
    parser.add_argument("--duration", type=float, default=60.0, 
                        help="自动校准持续时间(秒)")
    parser.add_argument("--output-dir", default="config", 
                        help="输出目录")
    parser.add_argument("--load", type=str, 
                        help="加载已有的校准结果文件")
    parser.add_argument("--skip-startup", action="store_true", 
                        help="跳过机器人启动序列")
    
    args = parser.parse_args()
    
    print("=== Dex3 灵巧手关节限位校准工具 ===")
    print(f"手部: {args.hand}")
    print(f"模式: {args.mode}")
    print(f"网络接口: {args.iface}")
    
    # 如果只是加载现有结果
    if args.load:
        calibrator = Dex3JointLimitsCalibrator(args.hand, args.iface)
        session = calibrator.load_calibration_results(args.load)
        if session:
            # 重新创建校准数据用于显示
            for limit_data in session.joint_limits:
                limit = JointLimitData(**limit_data)
                print(f"关节{limit.joint_id} ({limit.joint_name}): "
                      f"[{limit.min_angle:.3f}, {limit.max_angle:.3f}] rad")
        return
    
    try:
        # 执行启动序列
        if not args.skip_startup and SDK_AVAILABLE:
            print("执行机器人启动序列...")
            try:
                sport_client = hanger_boot_sequence(iface=args.iface)
                print("✓ 机器人启动完成")
            except Exception as e:
                print(f"启动序列失败: {e}")
                print("继续校准程序...")
        
        # 创建校准器
        calibrator = Dex3JointLimitsCalibrator(args.hand, args.iface)
        
        if not calibrator.is_connected and SDK_AVAILABLE:
            print("无法连接到灵巧手，请检查连接")
            return
        
        # 执行校准
        if args.mode == "interactive":
            calibrator._interactive_mode = True
            calibrator.interactive_calibration()
        else:
            calibrator.automatic_calibration(args.duration)
        
        # 显示校准结果
        calibrator.print_calibration_summary()
        
        # 保存结果
        session = calibrator.save_calibration_results(args.output_dir)
        
        # 导出 Python 代码
        python_file = f"{args.output_dir}/dex3_{args.hand}_limits.py"
        calibrator.export_to_python_code(python_file)
        
        print("\n=== 校准完成 ===")
        print(f"配置文件已保存到: {args.output_dir}/")
        print("你现在可以在 dex3_test.py 中使用这些限位数据")
        
    except KeyboardInterrupt:
        print("\n校准被用户中断")
    except Exception as e:
        print(f"校准过程中出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()