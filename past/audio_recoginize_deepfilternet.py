#!/usr/bin/env python3
# filepath: /home/sakuzeng/Coding/projects/unitree/unitree_sdk2_python_g1/audio_recoginize_deepfilternet.py
"""
使用 DeepFilterNet 的交互式语音助手
功能：
1. TTS语音输出询问"有什么需要帮助的呢"
2. 按回车开始接收音频数据
3. 检测语音活动，语音结束后停止录音
4. 使用 DeepFilterNet 对完整音频进行降噪
5. 保存降噪前和降噪后的音频数据
6. 对降噪后的音频进行语音识别并输出
"""

import sys
import time
import signal
import socket
import struct
import threading
import netifaces
import os
import json
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
import webrtcvad
import numpy as np
from collections import deque
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import wave
from datetime import datetime

# DeepFilterNet 导入
from df.enhance import enhance, init_df
import torch
import torchaudio

# 音频参数
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
SILENCE_TIMEOUT = 2.0
MAX_SPEECH_DURATION = 30

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
is_recording = False
recording_started = False
session_counter = 0

# 初始化语音识别模型
model_dir = "/home/unitree/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
asr_model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    hub="hf",
    device="cuda",
    disable_update=True,
)

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# 音频缓冲区
audio_buffer = deque(maxlen=16000 * 30)  # 音频缓冲区，最多30秒

class DeepFilterNetProcessor:
    """DeepFilterNet 音频处理器"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model = None
        self.df_state = None
        self.target_sample_rate = None
        self.resampler_to_target = None
        self.resampler_from_target = None
        self.initialized = False
        
    def initialize(self):
        """初始化 DeepFilterNet"""
        try:
            print("🧠 正在初始化 DeepFilterNet...")
            
            # 初始化模型
            self.model, self.df_state, _ = init_df()
            
            # 获取模型所需采样率
            self.target_sample_rate = self.df_state.sr()
            
            # 创建重采样器（如果需要）
            if self.sample_rate != self.target_sample_rate:
                self.resampler_to_target = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, 
                    new_freq=self.target_sample_rate
                )
                self.resampler_from_target = torchaudio.transforms.Resample(
                    orig_freq=self.target_sample_rate, 
                    new_freq=self.sample_rate
                )
            
            print(f"✅ DeepFilterNet 初始化成功")
            print(f"   模型采样率: {self.target_sample_rate}Hz")
            print(f"   输入采样率: {self.sample_rate}Hz")
            
            self.initialized = True
            
        except Exception as e:
            print(f"❌ DeepFilterNet 初始化失败: {e}")
            print("将回退到不使用降噪的模式")
            self.initialized = False
            
    def process_audio(self, audio_np):
        """
        使用 DeepFilterNet 处理音频
        
        Args:
            audio_np (np.ndarray): 输入音频数据 (int16)
            
        Returns:
            np.ndarray: 降噪后的音频数据 (int16)
        """
        if not self.initialized or audio_np.size == 0:
            return audio_np
            
        try:
            # 转换为 float32 格式 (-1.0 到 1.0) 并添加批次维度
            audio_float = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
            
            # 如果需要重采样到目标采样率
            if self.resampler_to_target is not None:
                audio_float = self.resampler_to_target(audio_float)
            
            # 使用 DeepFilterNet 进行降噪
            with torch.no_grad():
                enhanced_audio = enhance(self.model, self.df_state, audio_float)
            
            # 如果需要重采样回原始采样率
            if self.resampler_from_target is not None:
                enhanced_audio = self.resampler_from_target(enhanced_audio)
            
            # 转换回 numpy 数组并移除批次维度
            enhanced_np = enhanced_audio.squeeze(0).numpy()
            
            # 转换回 int16 格式
            enhanced_int16 = (enhanced_np * 32767.0).astype(np.int16)
            
            # 防止溢出
            enhanced_int16 = np.clip(enhanced_int16, -32768, 32767)
            
            return enhanced_int16
            
        except Exception as e:
            print(f"❌ DeepFilterNet 处理错误: {e}")
            import traceback
            traceback.print_exc()
            return audio_np

class VoiceAssistant:
    def __init__(self, interface_name="eth0"):
        self.interface_name = interface_name
        self.audio_client = None
        self.socket = None
        self.is_speaking_detected = False
        self.silence_start_time = None
        self.deepfilter_processor = DeepFilterNetProcessor()
        
    def initialize(self):
        """初始化语音助手"""
        print("🚀 初始化交互式语音助手（DeepFilterNet版本）...")
        
        # 初始化通道
        ChannelFactoryInitialize(0, self.interface_name)
        
        # 初始化音频客户端
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(10.0)
        self.audio_client.Init()
        
        # 初始化 DeepFilterNet
        self.deepfilter_processor.initialize()
        
        # 初始化音频接收器
        self.setup_audio_receiver()
        
        print("✅ 语音助手初始化完成")
        
    def get_local_ip_for_multicast(self):
        """获取192.168.123.x网段的IP地址"""
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
            
    def speak(self, text, speaker_id=0):
        """语音输出"""
        if not self.audio_client:
            print(f"📢 语音输出: {text}")
            return
            
        try:
            print(f"🔊 语音输出: {text}")
            code = self.audio_client.TtsMaker(text, speaker_id)
            if code != 0:
                print(f"❌ TTS失败，错误码: {code}")
            else:
                # 等待语音播放完成（估算时间）
                wait_time = max(2, len(text) * 0.15)  # 每个字约0.15秒
                time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 语音输出异常: {e}")
            
    def save_audio_session(self, original_audio, denoised_audio, session_id):
        """保存音频会话数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 确保目录存在
        os.makedirs("data/sessions", exist_ok=True)
        
        # 保存原始音频
        original_raw_path = f"data/sessions/session_{session_id}_{timestamp}_original.raw"
        original_wav_path = f"data/sessions/session_{session_id}_{timestamp}_original.wav"
        
        with open(original_raw_path, "wb") as f:
            f.write(original_audio.tobytes())
            
        self.convert_raw_to_wav(original_raw_path, original_wav_path, original_audio)
        
        # 保存降噪音频
        denoised_raw_path = f"data/sessions/session_{session_id}_{timestamp}_denoised_deepfilter.raw"
        denoised_wav_path = f"data/sessions/session_{session_id}_{timestamp}_denoised_deepfilter.wav"
        
        with open(denoised_raw_path, "wb") as f:
            f.write(denoised_audio.tobytes())
            
        self.convert_raw_to_wav(denoised_raw_path, denoised_wav_path, denoised_audio)
        
        print(f"💾 音频会话 {session_id} 已保存:")
        print(f"   原始音频: {original_wav_path}")
        print(f"   DeepFilterNet降噪音频: {denoised_wav_path}")
        
        return denoised_wav_path
        
    def convert_raw_to_wav(self, raw_path, wav_path, audio_data):
        """转换RAW到WAV格式"""
        try:
            with wave.open(wav_path, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(FRAME_RATE)
                f_wav.writeframes(audio_data.tobytes())
        except Exception as e:
            print(f"❌ 音频转换错误: {e}")
            
    def recognize_speech(self, audio_data):
        """语音识别"""
        if audio_data.size == 0:
            return ""
            
        print(f"🎯 开始语音识别，音频长度: {len(audio_data)/FRAME_RATE:.2f}秒")
        
        # 转换为模型需要的格式
        audio_float32 = audio_data.astype(np.float32) / 32768.0
        
        try:
            res = asr_model.generate(
                input=audio_float32,
                language="auto",
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
            
    def process_audio_frame(self, audio_data):
        """处理音频帧 - 只接收数据和VAD检测"""
        global recording_started, is_recording
        
        frame_duration = 10  # 10ms帧
        frame_size = int(FRAME_RATE * frame_duration / 1000 * 2)
        frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]
        
        for frame in frames:
            if len(frame) < frame_size:
                continue
                
            # 转换为numpy数组
            frame_np = np.frombuffer(frame, dtype=np.int16)
            
            # 如果正在录音，将数据添加到缓冲区
            if is_recording:
                audio_buffer.extend(frame_np)
            
            # VAD检测（使用原始音频）
            try:
                is_speech = vad.is_speech(frame, FRAME_RATE)
            except Exception as e:
                is_speech = False
            
            # 语音活动检测逻辑
            if is_recording:
                if is_speech:
                    # 检测到语音
                    if not self.is_speaking_detected:
                        print("🎤 开始检测到语音")
                        recording_started = True
                    
                    self.is_speaking_detected = True
                    self.silence_start_time = None
                    print(".", end="", flush=True)
                else:
                    # 检测到静音
                    if self.is_speaking_detected:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                            print(" [静音开始]", end="", flush=True)
                        elif time.time() - self.silence_start_time > SILENCE_TIMEOUT:
                            # 语音结束
                            print(f"\n📝 语音结束，停止录音...")
                            is_recording = False
                            self.process_complete_speech()
                            self.is_speaking_detected = False
                            self.silence_start_time = None
                            recording_started = False
                            
    def process_complete_speech(self):
        """处理完整的语音"""
        global session_counter
        session_counter += 1
        
        if not audio_buffer:
            print("⚠️  没有检测到有效语音")
            return
            
        # 转换为numpy数组
        original_audio = np.array(list(audio_buffer), dtype=np.int16)
        
        # 清空缓冲区
        audio_buffer.clear()
        
        # 使用 DeepFilterNet 对完整语音段进行降噪
        print("🧠 对完整语音段进行 DeepFilterNet 降噪...")
        denoised_audio = self.deepfilter_processor.process_audio(original_audio)
        
        # 保存音频数据
        self.save_audio_session(original_audio, denoised_audio, session_counter)
        
        # 语音识别
        recognized_text = self.recognize_speech(denoised_audio)
        
        if recognized_text:
            print(f"🎤 识别结果: {recognized_text}")
            # 语音输出识别结果
            response_text = f"我听到您说的是：{recognized_text}"
            self.speak(response_text)
        else:
            print("⚠️  语音识别结果为空")
            self.speak("抱歉，我没有听清楚，请再说一遍")
            
    def listen_for_audio(self):
        """监听音频数据"""
        global audio_receiver_running
        
        print("👂 开始监听音频数据...")
        
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
                
        print("👂 音频监听已停止")
        
    def start_recording(self):
        """开始录音"""
        global is_recording, recording_started
        
        # 清空缓冲区
        audio_buffer.clear()
        
        # 重置状态
        self.is_speaking_detected = False
        self.silence_start_time = None
        recording_started = False
        
        print("🔴 开始录音，请说话...")
        is_recording = True
        
        # 等待录音结束
        while is_recording:
            time.sleep(0.1)
            
        print("⏹️  录音结束")
        
    def start_conversation(self):
        """开始对话"""
        global audio_receiver_running, audio_receiver_thread
        
        # 启动音频接收线程
        audio_receiver_running = True
        audio_receiver_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        audio_receiver_thread.start()
        
        try:
            while True:
                # 语音输出询问
                self.speak("有什么需要帮助的呢？")
                
                # 等待用户按回车开始录音
                input("按回车键开始录音...")
                
                # 开始录音
                self.start_recording()
                
                # 询问是否继续
                print("\n" + "="*50)
                user_input = input("按回车继续对话，输入'q'退出: ")
                if user_input.lower() == 'q':
                    break
                    
        except KeyboardInterrupt:
            print("\n接收到退出信号")
        finally:
            audio_receiver_running = False
            if audio_receiver_thread and audio_receiver_thread.is_alive():
                audio_receiver_thread.join(timeout=2)
                
    def cleanup(self):
        """清理资源"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("🧹 资源清理完成")

def signal_handler(signum, frame):
    """信号处理"""
    global audio_receiver_running, is_recording
    print("\n🛑 接收到退出信号，正在关闭...")
    audio_receiver_running = False
    is_recording = False
    sys.exit(0)

def main():
    if len(sys.argv) < 2:
        print("未提供网络接口名称，使用默认值: eth0")
        interface_name = "eth0"
    else:
        interface_name = sys.argv[1]
        print(f"使用网络接口: {interface_name}")
        
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建语音助手
    assistant = VoiceAssistant(interface_name)
    
    try:
        # 初始化
        assistant.initialize()
        
        # 开始对话
        print("\n🎉 交互式语音助手已启动 (DeepFilterNet版本)")
        print("💡 功能说明：")
        print("   1. 语音询问'有什么需要帮助的呢'")
        print("   2. 按回车键开始录音")
        print("   3. 检测语音活动，语音结束后自动停止录音")
        print("   4. 使用 DeepFilterNet 对完整音频进行降噪")
        print("   5. 保存原始和降噪后的音频文件")
        print("   6. 进行语音识别并语音回复")
        
        if assistant.deepfilter_processor.initialized:
            print("🧠 DeepFilterNet 降噪已激活")
        else:
            print("⚠️  DeepFilterNet 未激活，使用原始音频")
            
        print("\n开始对话...")
        
        assistant.start_conversation()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
    finally:
        assistant.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    main()