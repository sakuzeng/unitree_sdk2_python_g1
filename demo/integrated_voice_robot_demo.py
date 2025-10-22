import sys
import time
import signal
import socket
import struct
import threading
import netifaces
import os
import json
import numpy as np
import wave
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.arm.arm_client import G1ArmClient
from unitree_sdk2py.dex3.dex3_client import Dex3Client
import webrtcvad
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 常量定义
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000
RECORDING_DURATION = 3  # 录音时长（秒）
SILENCE_TIMEOUT = 2.0
MAX_SPEECH_DURATION = 30

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
is_recording = False
recording_start_time = None

# 音频缓冲区
audio_buffer = deque(maxlen=16000 * 30)

# 初始化语音识别模型
model_dir = "/home/unitree/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
print("🧠 正在加载语音识别模型...")
asr_model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    hub="hf",
    device="cuda",
    disable_update=True,
)
print("✅ 语音识别模型加载完成")

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# 预定义手臂位姿数据
ARM_POSES = {
    "nature": [0.243, 0.173, -0.016, 0.796, 0.090, 0.027, -0.008, 0.250, -0.175, 0.025, 0.801, -0.111, 0.035, 0.009],
    "hello1": [0.243, 0.173, -0.016, 0.796, 0.090, 0.027, -0.008,
               -0.567, -0.226, -0.418, -0.150, -1.308, 0.003, -0.315],
    "hello2": [0.243, 0.173, -0.016, 0.796, 0.090, 0.027, -0.008,
               -0.567, -0.226, -0.787, -0.073, -1.141, 0.064, -0.161],
    "hello3": [0.243, 0.173, -0.016, 0.796, 0.090, 0.027, -0.008, 
               -0.567, -0.226, 0.137, -0.257, -1.615, -0.112, -0.189],
}

# 预定义灵巧手位姿数据
HAND_POSES = {
    "nature": [-0.029, -1.019, -1.667, 1.551, 1.702, 1.568, 1.710],
    "hello1": [-0.027, -1.022, -1.668, -0.059, -0.057, -0.040, -0.070],
}

class IntegratedRobotDemo:
    """
    G1机器人集成语音交互演示类
    
    集成语音播放、录音、语音识别和机器人控制功能，
    支持前进、后退和打招呼动作的语音控制。
    
    Args:
        interface_name (str): 网络接口名称，默认为 "eth0"
    """
    
    def __init__(self, interface_name="eth0"):
        self.interface_name = interface_name
        
        # 各模块客户端
        self.audio_client = None
        self.loco_client = None
        self.arm_client = None
        self.hand_client = None
        
        # 音频相关
        self.socket = None
        
        # 控制状态
        self.is_arm_hand_initialized = False
        self.emergency_stop = False
        self.cleanup_executed = False  # 添加清理状态标志
        
    def initialize(self):
        """
        初始化所有模块客户端
        
        Returns:
            bool: 初始化成功返回 True，失败抛出异常
        """
        try:
            # 初始化通道
            ChannelFactoryInitialize(0, self.interface_name)
            
            # 初始化音频客户端
            print("🔊 初始化音频模块...")
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            code = self.audio_client.SetVolume(60)
            if code == 0:
                print("✅ 音频模块初始化成功，音量设置为60")
            
            # 初始化运动控制客户端
            print("🦿 初始化运动控制模块...")
            self.loco_client = LocoClient()
            self.loco_client.Init()
            print("✅ 运动控制模块初始化成功")
            
            # 初始化手臂控制客户端
            print("🦾 初始化手臂控制模块...")
            self.arm_client = G1ArmClient(interface=self.interface_name)
            print("✅ 手臂控制模块初始化成功")
            
            # 初始化灵巧手控制客户端（右手）
            print("🤲 初始化灵巧手控制模块...")
            self.hand_client = Dex3Client(hand="right", interface=self.interface_name)
            print("✅ 灵巧手控制模块初始化成功")
            
            # 设置音频接收器
            self.setup_audio_receiver()

            # 初始化手臂和灵巧手到自然位姿
            self.initialize_arm_and_hand()
            
            print("✅ 所有模块初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    def setup_audio_receiver(self):
        """设置音频接收器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('', MULTICAST_PORT))
            
            local_ip = self.get_local_ip_for_multicast()
            if local_ip is None:
                raise Exception("无法找到192.168.123.x网段的网络接口")
                
            mreq = struct.pack("4s4s",
                               socket.inet_aton(MULTICAST_GROUP),
                               socket.inet_aton(local_ip))
            self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            self.socket.settimeout(1.0)
            
            print(f"📡 音频接收器设置完成: {MULTICAST_GROUP}:{MULTICAST_PORT}")
            
        except Exception as e:
            print(f"❌ 音频接收器设置失败: {e}")
            raise
            
    def get_local_ip_for_multicast(self):
        """
        获取192.168.123.x网段的IP地址
        
        Returns:
            str: 找到的IP地址，未找到返回 None
        """
        for interface in netifaces.interfaces():
            try:
                addresses = netifaces.ifaddresses(interface)
                if netifaces.AF_INET in addresses:
                    for addr_info in addresses[netifaces.AF_INET]:
                        ip = addr_info['addr']
                        if ip.startswith('192.168.123.'):
                            return ip
            except:
                continue
        return None
        
    def play_tts(self, text, speaker_id=0):
        """
        播放TTS语音
        
        Args:
            text (str): 要播放的文本
            speaker_id (int): 说话人ID，默认为0
        """
        try:
            print(f"🔊 播放语音: {text}")
            code = self.audio_client.TtsMaker(text, speaker_id)
            if code == 0:
                print("✅ TTS播放成功")
                time.sleep(len(text) * 0.15 + 2)  # 估算播放时间
            else:
                print(f"❌ TTS播放失败，错误码: {code}")
        except Exception as e:
            print(f"❌ TTS播放错误: {e}")
            
    def process_audio_frame(self, audio_data):
        """处理音频帧"""
        global is_recording
        
        if not is_recording:
            return
            
        frame_duration = 10
        frame_size = int(FRAME_RATE * frame_duration / 1000 * 2)
        frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]
        
        for frame in frames:
            if len(frame) < frame_size:
                continue
                
            frame_np = np.frombuffer(frame, dtype=np.int16)
            audio_buffer.extend(frame_np)
            
    def listen_for_audio(self):
        """监听音频数据"""
        global audio_receiver_running
        
        while audio_receiver_running:
            try:
                data, addr = self.socket.recvfrom(2048)
                self.process_audio_frame(data)
            except socket.timeout:
                continue
            except Exception as e:
                if audio_receiver_running:
                    print(f"❌ 音频接收错误: {e}")
                break
                
    def record_audio(self, duration=RECORDING_DURATION):
        """
        录音指定时长
        
        Args:
            duration (int): 录音时长（秒）
            
        Returns:
            np.ndarray: 录制的音频数据
        """
        global is_recording, recording_start_time
        
        audio_buffer.clear()
        is_recording = True
        recording_start_time = time.time()
        
        print(f"🎤 开始录音 {duration} 秒...")
        
        # 显示录音进度
        for i in range(duration):
            print(f"⏺️  录音中... {i+1}/{duration}秒", end="\r")
            time.sleep(1)
            
        is_recording = False
        print(f"\n✅ 录音完成，共录制 {len(audio_buffer)/FRAME_RATE:.2f} 秒")
        
        # 返回录制的音频数据
        if audio_buffer:
            return np.array(list(audio_buffer), dtype=np.int16)
        else:
            return np.array([], dtype=np.int16)
            
    def save_audio_data(self, audio_data, session_id):
        """
        保存原始音频数据
        
        Args:
            audio_data (np.ndarray): 原始音频数据
            session_id (int): 会话ID
            
        Returns:
            str: 保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("data/sessions", exist_ok=True)
            
            # 保存原始音频
            audio_wav = f"data/sessions/session_{session_id}_{timestamp}_original.wav"
            with wave.open(audio_wav, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(FRAME_RATE)
                f_wav.writeframes(audio_data.tobytes())
            
            print(f"💾 音频文件已保存: {os.path.basename(audio_wav)}")
            return audio_wav
            
        except Exception as e:
            print(f"❌ 保存音频失败: {e}")
            return ""
            
    def recognize_speech(self, audio_data):
        """
        语音识别
        
        Args:
            audio_data (np.ndarray): 音频数据
            
        Returns:
            str: 识别的文本
        """
        if audio_data.size == 0:
            return ""
            
        print(f"🧠 开始语音识别...")
        
        # 转换为模型需要的格式
        audio_float32 = audio_data.astype(np.float32) / 32768.0
        
        try:
            res = asr_model.generate(
                input=audio_float32,
                language="zn",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=MAX_SPEECH_DURATION,
                sampling_rate=16000,
            )
            
            if res and len(res) > 0 and "text" in res[0]:
                text = rich_transcription_postprocess(res[0]["text"])
                return text.strip()
            else:
                return ""
                
        except Exception as e:
            print(f"❌ 语音识别错误: {e}")
            return ""
            
    def classify_text(self, text):
        """
        分类文本并返回匹配的指令
        
        Args:
            text (str): 待分类的文本
            
        Returns:
            str: 匹配的指令，无匹配返回 None
        """
        if not text:
            return None
            
        text_lower = text.lower()
        
        if "前进" in text or "前" in text or "进" in text:
            return "前进"
        elif "后退" in text or "后" in text or "退" in text:
            return "后退"
        elif "打招呼" in text or "招呼" in text:
            return "打招呼"
        else:
            return None
            
    def execute_forward_movement(self):
        """
        执行前进1米动作
        
        Returns:
            bool: 执行成功返回 True，失败返回 False
        """
        try:
            print("🚶 开始执行前进1米动作...")
            
            # 前进1米，速度0.5m/s，持续2秒
            self.loco_client.SetVelocity(vx=0.5, vy=0.0, omega=0.0, duration=2.0)
            time.sleep(2.5)
            
            # 停止移动
            self.loco_client.StopMove()
            time.sleep(1.0)
            
            print("✅ 前进1米动作执行完成")
            return True
            
        except Exception as e:
            print(f"❌ 前进动作执行失败: {e}")
            return False
            
    def execute_backward_movement(self):
        """
        执行后退1米动作
        
        Returns:
            bool: 执行成功返回 True，失败返回 False
        """
        try:
            print("🚶 开始执行后退1米动作...")
            
            # 后退1米，速度-0.5m/s，持续2秒
            self.loco_client.SetVelocity(vx=-0.5, vy=0.0, omega=0.0, duration=2.0)
            time.sleep(2.5)
            
            # 停止移动
            self.loco_client.StopMove()
            time.sleep(1.0)
            
            print("✅ 后退1米动作执行完成")
            return True
            
        except Exception as e:
            print(f"❌ 后退动作执行失败: {e}")
            return False
            
    def initialize_arm_and_hand(self):
        """
        初始化手臂和灵巧手到自然位姿
        
        Returns:
            bool: 初始化成功返回 True，失败返回 False
        """
        try:
            if self.is_arm_hand_initialized:
                return True
                
            print("🤖 初始化手臂和灵巧手...")
            
            # 初始化手臂
            if not self.arm_client.initialize_arms():
                print("❌ 手臂初始化失败")
                return False
            
            # 初始化灵巧手
            if not self.hand_client.initialize_hand():
                print("❌ 灵巧手初始化失败")
                return False
            
            self.is_arm_hand_initialized = True
            print("✅ 手臂和灵巧手初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 手臂和灵巧手初始化失败: {e}")
            return False
            
    def execute_hello_gesture(self):
        """
        执行打招呼动作序列
        
        Returns:
            bool: 执行成功返回 True，失败返回 False
        """
        try:
            print("👋 开始执行打招呼动作...")
            
            # 确保手臂和灵巧手已初始化
            if not self.initialize_arm_and_hand():
                return False
            
            # 步骤1: 手臂到 hello1 位姿
            print("📍 步骤1: 手臂移动到 hello1 位姿")
            if not self.arm_client.set_joint_positions(ARM_POSES["hello1"], duration=2.0):
                print("❌ 手臂移动到 hello1 失败")
                return False
            # time.sleep(1.0)
            
            # 步骤2: 灵巧手到 hello1 位姿
            print("🤲 步骤2: 灵巧手移动到 hello1 位姿")
            if not self.hand_client.set_joint_positions(HAND_POSES["hello1"], duration=1.0):
                print("❌ 灵巧手移动到 hello1 失败")
                return False
            # time.sleep(1.0)
            
            # 步骤3: 手臂到 hello2 位姿
            print("📍 步骤3: 手臂移动到 hello2 位姿")
            if not self.arm_client.set_joint_positions(ARM_POSES["hello2"], duration=1.0):
                print("❌ 手臂移动到 hello2 失败")
                return False
            # time.sleep(1.0)
            
            # 步骤4: 手臂到 hello3 位姿
            print("📍 步骤4: 手臂移动到 hello3 位姿")
            if not self.arm_client.set_joint_positions(ARM_POSES["hello3"], duration=1.0):
                print("❌ 手臂移动到 hello3 失败")
                return False
            
            # 步骤5: 维持2秒
            print("⏱️ 步骤5: 维持打招呼姿态2秒")
            time.sleep(2.0)
            
            # 步骤6: 恢复到自然位姿
            print("🔄 步骤6: 恢复到自然位姿")
            
            # 灵巧手恢复自然位姿
            if not self.hand_client.set_joint_positions(HAND_POSES["nature"], duration=2.0):
                print("❌ 灵巧手恢复自然位姿失败")
                return False
            time.sleep(1.0)
            
            # 手臂恢复自然位姿
            if not self.arm_client.set_joint_positions(ARM_POSES["nature"], duration=3.0):
                print("❌ 手臂恢复自然位姿失败")
                return False
            time.sleep(1.0)
            
            print("✅ 打招呼动作执行完成")
            return True
            
        except Exception as e:
            print(f"❌ 打招呼动作执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def emergency_stop_arm_hand(self):
        """紧急停止手臂和灵巧手（幂等操作）"""
        # 如果已经执行过清理，直接返回，避免重复操作
        if self.cleanup_executed:
            return
            
        self.cleanup_executed = True  # 设置清理标志
        
        try:
            print("🚨 执行紧急停止...")
            
            if self.arm_client and self.is_arm_hand_initialized:
                try:
                    # 手臂恢复自然位姿并停止控制
                    current_positions = self.arm_client.get_current_joint_positions()
                    if current_positions:
                        self.arm_client.smooth_transition(
                            current_positions,
                            ARM_POSES["nature"],
                            2.0,
                            "紧急恢复手臂到自然位姿"
                        )
                    self.arm_client.stop_control()
                    print("✅ 手臂已安全停止")
                except Exception as e:
                    print(f"⚠️  手臂停止过程中出现错误: {e}")
            
            if self.hand_client and self.is_arm_hand_initialized:
                try:
                    # 灵巧手恢复自然位姿并停止控制
                    current_positions = self.hand_client.get_current_joint_positions()
                    if current_positions:
                        self.hand_client.smooth_transition(
                            current_positions,
                            HAND_POSES["nature"],
                            2.0,
                            "紧急恢复灵巧手到自然位姿"
                        )
                    self.hand_client.stop_control()
                    print("✅ 灵巧手已安全停止")
                except Exception as e:
                    print(f"⚠️  灵巧手停止过程中出现错误: {e}")
            
            # 停止运动
            if self.loco_client:
                try:
                    self.loco_client.StopMove()
                    print("✅ 机器人运动已停止")
                except Exception as e:
                    print(f"⚠️  运动停止过程中出现错误: {e}")
                    
        except Exception as e:
            print(f"❌ 紧急停止失败: {e}")
            
    def classify_and_respond(self, text):
        """
        对识别的文本进行分类并响应
        
        Args:
            text (str): 识别的文本
        """
        print(f"🎯 识别结果: {text}")
        
        # 检查识别结果
        command = self.classify_text(text)
        
        if command:
            print(f"✅ 识别到指令: {command}")
            
            if command == "前进":
                self.play_tts("我将前进1米")
                self.execute_forward_movement()
            elif command == "后退":
                self.play_tts("我将后退1米")
                self.execute_backward_movement()
            elif command == "打招呼":
                self.play_tts("我将做出打招呼的动作")
                self.execute_hello_gesture()
        else:
            self.play_tts("抱歉，我不理解您的指令，请说前进、后退或打招呼")
            print("❓ 未识别的指令")
            
    def run_demo(self):
        """运行演示主循环"""
        global audio_receiver_running, audio_receiver_thread
        
        # 启动音频监听线程
        audio_receiver_running = True
        audio_receiver_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        audio_receiver_thread.start()
        
        session_counter = 0
        
        try:
            while True:
                session_counter += 1
                print(f"\n{'='*80}")
                print(f"🎯 G1机器人集成语音交互演示 - 第 {session_counter} 轮")
                print(f"{'='*80}")
                
                # 1. 播放提示语音
                self.play_tts("有什么需要我做的")
                
                # 2. 录音
                original_audio = self.record_audio(RECORDING_DURATION)
                
                if original_audio.size == 0:
                    print("⚠️  没有录制到音频数据")
                    continue
                
                # 3. 保存音频数据
                self.save_audio_data(original_audio, session_counter)
                
                # 4. 语音识别
                text = self.recognize_speech(original_audio)
                
                # 5. 分类和响应（包括执行机器人动作）
                self.classify_and_respond(text)
                
                print(f"\n{'='*80}")
                user_input = input("按回车继续下一轮，输入'q'退出: ")
                if user_input.lower() == 'q':
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 接收到退出信号")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """清理资源"""
        global audio_receiver_running
        
        print("🧹 正在清理资源...")
        audio_receiver_running = False
        
        # 仅在未执行过紧急停止时调用，避免重复执行
        if not self.cleanup_executed:
            self.emergency_stop_arm_hand()
        
        # 等待音频线程结束
        if audio_receiver_thread and audio_receiver_thread.is_alive():
            audio_receiver_thread.join(timeout=2)
            
        # 关闭网络连接
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
                
        print("✅ 资源清理完成")

def signal_handler(signum, frame):
    """
    信号处理函数，处理 Ctrl+C 中断
    
    Args:
        signum: 信号编号
        frame: 当前栈帧
    """
    global audio_receiver_running, is_recording
    print("\n🛑 接收到退出信号，正在触发紧急停止...")
    audio_receiver_running = False
    is_recording = False
    
    # 如果有全局的demo实例，执行紧急停止（幂等）
    if 'demo' in globals() and demo:
        demo.emergency_stop_arm_hand()
    
    # 不在 signal handler 中调用 sys.exit，允许主流程进入 finally 并执行 cleanup
    print("📋 紧急停止已触发，程序将安全退出。")

def main():
    """主函数"""
    global demo
    
    if len(sys.argv) < 2:
        print("🔧 未提供网络接口名称，使用默认值: eth0")
        print("💡 提示: 根据实际情况修改网络接口参数")
        interface_name = "eth0"
    else:
        interface_name = sys.argv[1]
        print(f"🔧 使用网络接口: {interface_name}")
        
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建演示实例
    demo = IntegratedRobotDemo(interface_name)
    
    try:
        print("🎉 G1机器人集成语音交互演示启动")
        print("🔧 功能: 语音播放 + 录音 + 语音识别 + 机器人控制")
        print("📝 支持指令:")
        print("   - 前进: 控制机器人前进1米")
        print("   - 后退: 控制机器人后退1米") 
        print("   - 打招呼: 执行完整的打招呼动作序列")
        print(f"⏱️  录音时长: {RECORDING_DURATION}秒")
        print("⚠️  安全提示: 确保机器人处于安全环境，程序会自动处理紧急停止")
        
        print("\n📋 手臂位姿数据说明:")
        print("   当前使用的是采集的实际数据:")
        print("   - ARM_POSES['hello1']: 打招呼位姿1")
        print("   - ARM_POSES['hello2']: 打招呼位姿2") 
        print("   - ARM_POSES['hello3']: 打招呼位姿3")
        print("   - HAND_POSES['hello1']: 灵巧手打招呼位姿")
        
        demo.initialize()
        demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n🛑 接收到键盘中断信号")
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    main()