#!/usr/bin/env python3
"""
手臂运动数据收集器 (智能语音版)
用于收集不同方向的手臂运动数据，支持语音提示和自动化采集
"""

import csv
import time
import threading
import random
from pathlib import Path
from collections import defaultdict

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except ImportError:
    print("请确保已安装 unitree_sdk2_python")
    exit(1)

class ArmTrainingRecorder:
    def __init__(self, arm: str = "left", network_interface: str = "eth0"):
        """
        初始化手臂训练数据记录器
        
        Args:
            arm: 要记录的手臂 ("left" 或 "right")
            network_interface: 网络接口名称
        """
        self._arm = arm
        self._iface = network_interface
        
        # 关节索引配置
        self._waist_idx = 12
        if arm == "left":
            self._arm_joint_indices = list(range(15, 22))  # 左臂关节
        else:
            self._arm_joint_indices = list(range(22, 29))  # 右臂关节
        
        # 数据存储
        self._training_samples = []
        self._current_joints = {}
        self._recording = False
        self._direction = None
        self._start_joints = None
        
        # 方向映射 (英文 -> 中文)
        self._direction_map = {
            "up": "上",
            "down": "下", 
            "left": "左",
            "right": "右",
            "forward": "前",
            "back": "后"
        }
        
        # 语音提示文本
        self._voice_prompts = {
            "up": "上",
            "down": "下",
            "left": "左", 
            "right": "右",
            "forward": "前",
            "back": "后"
        }
        
        # 初始化语音客户端
        self._audio_client = None
        
        # 初始化SDK连接
        self._init_sdk_connection()
        
        # 加载已有数据
        self._load_existing_data()

    def _init_sdk_connection(self):
        """初始化SDK连接和关节状态订阅"""
        try:
            # 初始化DDS通道
            ChannelFactoryInitialize(0, self._iface)
            
            # 初始化语音客户端
            try:
                self._audio_client = AudioClient()
                self._audio_client.SetTimeout(10.0)
                self._audio_client.Init()
                print("语音客户端初始化成功")
            except Exception as e:
                print(f"语音客户端初始化失败: {e}")
                self._audio_client = None
            
            # 设置关节状态回调
            def joint_state_callback(msg: LowState_):
                """处理关节状态消息"""
                for idx in [*self._arm_joint_indices, self._waist_idx]:
                    try:
                        self._current_joints[idx] = msg.motor_state[idx].q
                    except (IndexError, AttributeError):
                        pass
            
            # 创建订阅器
            self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
            self._subscriber.Init(joint_state_callback, 100)
            
            print(f"已连接到 {self._arm} 手臂，网络接口: {self._iface}")
            
        except Exception as e:
            print(f"SDK连接失败: {e}")
            raise

    def _load_existing_data(self):
        """加载已有的训练数据"""
        csv_path = self._get_csv_path()
        
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                # 只加载当前手臂的数据
                arm_data = df[df['arm'] == self._arm]
                self._training_samples = arm_data.to_dict('records')
                print(f"已加载 {len(self._training_samples)} 个已有样本")
            except Exception as e:
                print(f"加载已有数据失败: {e}")
                self._training_samples = []
        else:
            self._training_samples = []

    def _get_csv_path(self):
        """获取CSV文件路径"""
        data_dir = Path("data/arms") / self._arm
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "training_data_with_waist.csv"

    def _speak(self, text: str):
        """语音播报"""
        if self._audio_client:
            try:
                print(f"🔊 语音提示: {text}")
                code = self._audio_client.TtsMaker(text, 0)  # 0表示中文
                if code != 0:
                    print(f"语音播报失败，错误码: {code}")
                time.sleep(2)  # 等待语音播放完成
            except Exception as e:
                print(f"语音播报异常: {e}")
        else:
            print(f"📢 提示: {text} (语音功能不可用)")
    
    def _generate_random_sequence(self, total_samples: int) -> list:
        """
        生成随机的采集序列，确保每个方向的样本数平衡且相邻方向不重复
        
        Args:
            total_samples: 总样本数
            
        Returns:
            list: 方向序列，确保相邻方向不同
        """
        directions = ["up", "down", "left", "right", "forward", "back"]
        
        # 计算每个方向需要采集的样本数
        samples_per_direction = total_samples // len(directions)
        remaining = total_samples % len(directions)
        
        # 构建基础序列
        base_sequence = []
        for direction in directions:
            base_sequence.extend([direction] * samples_per_direction)
        
        # 分配剩余的样本
        for i in range(remaining):
            base_sequence.append(directions[i])
        
        # 生成不重复相邻方向的序列
        sequence = self._arrange_non_adjacent_sequence(base_sequence, directions)
        
        print(f"生成序列完成，相邻方向检查通过")
        return sequence

    def _arrange_non_adjacent_sequence(self, base_sequence: list, directions: list) -> list:
        """
        重新排列序列，确保相邻方向不同
        
        Args:
            base_sequence: 基础方向序列
            directions: 所有可用方向
            
        Returns:
            list: 重新排列后的序列
        """
        # 统计每个方向的数量
        direction_counts = defaultdict(int)
        for direction in base_sequence:
            direction_counts[direction] += 1
        
        # 转换为可操作的列表
        available_directions = []
        for direction, count in direction_counts.items():
            available_directions.extend([direction] * count)
        
        # 使用贪心算法生成不重复相邻的序列
        result = []
        max_attempts = 1000  # 最大尝试次数
        
        for attempt in range(max_attempts):
            result = []
            remaining = available_directions.copy()
            random.shuffle(remaining)
            
            while remaining:
                # 查找与上一个方向不同的候选方向
                candidates = []
                if not result:
                    # 第一个方向可以是任意方向
                    candidates = remaining
                else:
                    last_direction = result[-1]
                    candidates = [d for d in remaining if d != last_direction]
                
                if not candidates:
                    # 如果没有合适的候选方向，重新开始
                    break
                
                # 随机选择一个候选方向
                chosen = random.choice(candidates)
                result.append(chosen)
                remaining.remove(chosen)
            
            # 检查是否成功生成完整序列
            if len(result) == len(base_sequence):
                # 验证序列
                if self._validate_sequence(result):
                    print(f"序列生成成功，尝试次数: {attempt + 1}")
                    return result
        
        # 如果无法生成完全不重复的序列，使用备选方案
        print("⚠️ 无法生成完全不重复相邻的序列，使用备选方案")
        return self._generate_fallback_sequence(base_sequence)

    def _validate_sequence(self, sequence: list) -> bool:
        """
        验证序列是否满足相邻方向不重复的要求
        
        Args:
            sequence: 待验证的序列
            
        Returns:
            bool: 是否有效
        """
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                return False
        return True

    def _generate_fallback_sequence(self, base_sequence: list) -> list:
        """
        生成备选序列，尽可能减少相邻重复
        
        Args:
            base_sequence: 基础序列
            
        Returns:
            list: 备选序列
        """
        directions = ["up", "down", "left", "right", "forward", "back"]
        
        # 简单的交替策略
        result = []
        remaining = base_sequence.copy()
        random.shuffle(remaining)
        
        while remaining:
            if not result:
                # 第一个元素随机选择
                chosen = remaining.pop(0)
            else:
                # 尝试找到与上一个不同的方向
                last_direction = result[-1]
                different_options = [d for d in remaining if d != last_direction]
                
                if different_options:
                    chosen = random.choice(different_options)
                    remaining.remove(chosen)
                else:
                    # 如果没有不同的选项，选择第一个可用的
                    chosen = remaining.pop(0)
            
            result.append(chosen)
        
        # 统计相邻重复的数量
        adjacent_repeats = sum(1 for i in range(1, len(result)) if result[i] == result[i-1])
        print(f"备选序列生成完成，相邻重复数量: {adjacent_repeats}")
        
        return result

    def wait_for_joint_data(self, timeout: float = 5.0) -> bool:
        """等待关节数据可用"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(self._current_joints) >= len(self._arm_joint_indices) + 1:  # +1 for waist
                return True
            time.sleep(0.1)
        return False

    def auto_collect_samples(self, total_samples: int):
        """自动采集指定数量的样本"""
        print(f"\n=== 自动采集 {total_samples} 个样本 ===")
        
        # 获取当前统计
        current_stats = self.get_current_stats()
        current_total = sum(current_stats.values())
        
        if current_total >= total_samples:
            print(f"已有 {current_total} 个样本，达到目标数量 {total_samples}")
            self._print_stats()
            return
        
        # 计算还需要采集的样本数
        remaining = total_samples - current_total
        print(f"当前已有 {current_total} 个样本，还需采集 {remaining} 个")
        
        # 生成采集序列（确保相邻方向不重复）
        sequence = self._generate_random_sequence(remaining)
        
        print(f"采集序列: {[self._direction_map[d] for d in sequence]}")
        print(f"序列验证: {'✅ 相邻方向不重复' if self._validate_sequence(sequence) else '⚠️ 存在相邻重复'}")
        
        # 初始化提示 - 只在开始时提示一次
        self._speak("开始自动数据采集")
        self._speak("请将手臂调整到初始位置")
        input("请将手臂调整到舒适的起始位置，然后按回车键开始采集...")
        time.sleep(1)
        
        for i, direction in enumerate(sequence):
            print(f"\n--- 第 {i+1}/{len(sequence)} 个样本 ---")
            print(f"方向: {self._direction_map[direction]} ({direction})")
            
            # 显示上一个方向信息
            if i > 0:
                prev_direction = sequence[i-1]
                print(f"上一个方向: {self._direction_map[prev_direction]} -> 当前方向: {self._direction_map[direction]}")
            
            # 等待关节数据可用
            if not self.wait_for_joint_data():
                print("❌ 无法获取关节数据，跳过此样本")
                continue
            
            # 开始记录（不再提示调整位置）
            if self.start_recording(direction):
                # 语音提示运动方向
                prompt_text = f"请向{self._voice_prompts[direction]}移动手臂"
                self._speak(prompt_text)
                
                # 等待用户完成运动
                input(f"请完成{self._direction_map[direction]}向运动，然后按回车键...")
                
                # 停止记录
                self.stop_recording()
                
                # 播放完成提示
                self._speak("样本记录完成")
                
                # 短暂休息
                time.sleep(0.5)
            else:
                print("❌ 记录启动失败，跳过此样本")
        
        # 采集完成
        self._speak("数据采集全部完成")
        print(f"\n🎉 采集完成！总共采集了 {len(sequence)} 个新样本")
        self._print_stats()

    def start_recording(self, direction: str) -> bool:
        """
        开始记录指定方向的运动
        
        Args:
            direction: 运动方向 ("up", "down", "left", "right", "forward", "back")
        """
        if not self.wait_for_joint_data():
            print("错误: 无法获取关节数据，请检查机器人连接")
            return False
        
        # 记录起始关节位置
        self._start_joints = []
        for idx in sorted([*self._arm_joint_indices, self._waist_idx]):
            self._start_joints.append(self._current_joints.get(idx, 0.0))
        
        self._direction = direction
        self._recording = True
        
        print(f"开始记录 {self._direction_map[direction]} 方向运动...")
        return True

    def stop_recording(self):
        """停止记录并保存当前样本"""
        if not self._recording:
            print("当前没有正在进行的记录")
            return
        
        # 记录结束关节位置
        end_joints = []
        for idx in sorted([*self._arm_joint_indices, self._waist_idx]):
            end_joints.append(self._current_joints.get(idx, 0.0))
        
        # 创建训练样本
        sample = {
            'direction': self._direction,
            'arm': self._arm,
        }
        
        # 添加起始关节位置
        joint_names = [self._waist_idx] + self._arm_joint_indices
        for i, (start_val, end_val) in enumerate(zip(self._start_joints, end_joints)):
            joint_idx = joint_names[i]
            sample[f'start_{joint_idx}'] = start_val
            sample[f'end_{joint_idx}'] = end_val
        
        self._training_samples.append(sample)
        self._recording = False
        
        # 保存到文件
        self._save_to_csv()
        
        print(f"已保存 {self._direction_map[self._direction]} 运动样本")
        self._print_current_stats()

    def _save_to_csv(self):
        """保存数据到CSV文件"""
        csv_path = self._get_csv_path()
        
        if self._training_samples:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = self._training_samples[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._training_samples)
            
            print(f"数据已保存到: {csv_path}")

    def _print_current_stats(self):
        """打印当前样本的简要统计"""
        stats = self.get_current_stats()
        total = sum(stats.values())
        
        # 显示各方向的中文统计
        print(f"当前统计: ", end="")
        direction_stats = []
        for direction in ["up", "down", "left", "right", "forward", "back"]:
            count = stats.get(direction, 0)
            direction_stats.append(f"{self._direction_map[direction]}:{count}")
        
        print(" | ".join(direction_stats) + f" | 总计:{total}")

    def _print_stats(self):
        """打印详细的数据统计"""
        stats = defaultdict(int)
        for sample in self._training_samples:
            stats[sample['direction']] += 1
        
        print("\n📊 详细数据统计:")
        total = 0
        for direction in ["up", "down", "left", "right", "forward", "back"]:
            count = stats[direction]
            print(f"  {self._direction_map[direction]} ({direction}): {count} 个样本")
            total += count
        print(f"  总计: {total} 个样本")
        
        # 检查数据平衡性
        if total > 0:
            max_count = max(stats.values())
            min_count = min(stats.values()) if stats.values() else 0
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            if balance_ratio >= 0.8:
                print(f"  ✅ 数据平衡度良好 ({balance_ratio:.2f})")
            elif balance_ratio >= 0.6:
                print(f"  ⚠️  数据轻微不平衡 ({balance_ratio:.2f})")
            else:
                print(f"  ❌ 数据不平衡 ({balance_ratio:.2f})，建议补充样本较少的方向")

    def get_current_stats(self) -> dict:
        """获取当前数据统计"""
        stats = defaultdict(int)
        for sample in self._training_samples:
            stats[sample['direction']] += 1
        return dict(stats)

def interactive_collection_session():
    """交互式数据收集会话"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 arm_training_recorder.py <network_interface> [arm] [samples]")
        print("示例: python3 arm_training_recorder.py eth0 left 60")
        return
    
    iface = sys.argv[1]
    arm = sys.argv[2] if len(sys.argv) > 2 else "left"
    target_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if arm not in ["left", "right"]:
        print("手臂参数必须是 'left' 或 'right'")
        return
    
    # 创建记录器
    try:
        recorder = ArmTrainingRecorder(arm, iface)
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    directions = ["up", "down", "left", "right", "forward", "back"]
    direction_names = [recorder._direction_map[d] for d in directions]
    
    print(f"\n=== {arm.upper()} 手臂训练数据收集器 (智能语音版) ===")
    print("✨ 新功能: 自动确保相邻采集方向不重复")
    
    # 显示当前数据状态
    recorder._print_stats()
    
    if target_samples:
        print(f"\n🎯 目标样本数: {target_samples}")
        print("\n选择采集模式:")
        print("1. 自动采集 (语音提示，随机顺序，相邻方向不重复)")
        print("2. 手动采集 (交互式选择)")
        
        choice = input("\n请选择模式 (1/2): ").strip()
        
        if choice == "1":
            # 自动采集模式
            confirm = input(f"\n开始自动采集 {target_samples} 个样本？(y/N): ").strip().lower()
            if confirm == 'y':
                recorder.auto_collect_samples(target_samples)
            return
    
    # 手动采集模式
    print("\n操作说明:")
    print("1. 将手臂调整到舒适的起始位置")
    print("2. 选择运动方向")
    print("3. 程序开始记录起始位置")
    print("4. 手动移动手臂到目标位置")
    print("5. 确认完成记录")
    print("6. 重复以上步骤收集更多样本")
    
    # 手动模式的初始位置提示
    input("\n请先将手臂调整到舒适的起始位置，然后按回车键继续...")
    
    while True:
        print(f"\n可用方向: {', '.join(direction_names)}")
        print("输入 'stats' 查看统计，'auto <数量>' 自动采集，'quit' 退出")
        
        choice = input("\n请选择方向: ").strip().lower()
        
        if choice == 'quit':
            break
        elif choice == 'stats':
            recorder._print_stats()
            continue
        elif choice.startswith('auto '):
            try:
                auto_count = int(choice.split()[1])
                recorder.auto_collect_samples(auto_count)
            except (ValueError, IndexError):
                print("用法: auto <数量>，例如: auto 30")
            continue
        
        # 检查中文输入
        direction = None
        for d, name in recorder._direction_map.items():
            if choice == name or choice == d:
                direction = d
                break
        
        if direction is None:
            print("无效选择，请重新输入")
            continue
        
        # 开始记录（不再提示调整位置）
        if recorder.start_recording(direction):
            input(f"请移动手臂到{recorder._direction_map[direction]}向目标位置，完成后按回车...")
            recorder.stop_recording()
        else:
            print("记录启动失败，请检查机器人连接")

if __name__ == "__main__":
    interactive_collection_session()