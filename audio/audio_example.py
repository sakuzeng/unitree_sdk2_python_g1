import sys
import time
import signal
import socket
import struct
import threading
import netifaces
import os
import wave
from datetime import datetime
import numpy as np
from collections import deque
import torch
import torchaudio
import argparse

# 导入 DeepFilterNet 和 FunASR 相关模块
try:
    from df.enhance import enhance, init_df
except ImportError:
    print("⚠️  DeepFilterNet 未安装，降噪功能将不可用")

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except ImportError:
    print("⚠️  FunASR 未安装，语音识别功能将不可用")

# 音频参数
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
MAX_SPEECH_DURATION = 30

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
is_recording = False
session_counter = 0

# 音频缓冲区
audio_buffer = deque(maxlen=16000 * 30)  # 最多30秒

class DeepFilterNetProcessor:
    """DeepFilterNet 音频处理器"""
    
    def __init__(self, sample_rate=FRAME_RATE):
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
            
            # 创建重采样器（如果输入采样率 != 模型采样率）
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
            
            # 如果需要重采样到模型采样率
            if self.resampler_to_target is not None:
                audio_float = self.resampler_to_target(audio_float)
            
            # 使用 DeepFilterNet 进行降噪
            with torch.no_grad():
                enhanced_audio = enhance(self.model, self.df_state, audio_float)
            
            # 如果需要重采样回输入采样率
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
            return audio_np

class SpeechRecognizer:
    """语音识别器"""
    
    def __init__(self, model_dir=None, device="cuda"):
        self.model = None
        self.initialized = False
        self.model_dir = model_dir or "FunAudioLLM/SenseVoiceSmall"
        self.device = device
        
    def initialize(self):
        """初始化语音识别模型"""
        try:
            print("🧠 正在初始化语音识别模型...")
            
            self.model = AutoModel(
                model=self.model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                hub="hf",
                device=self.device,
                disable_update=True,
            )
            
            print("✅ 语音识别模型初始化成功")
            self.initialized = True
            
        except Exception as e:
            print(f"❌ 语音识别模型初始化失败: {e}")
            self.initialized = False
            
    def recognize(self, audio_np, sample_rate=FRAME_RATE):
        """语音识别"""
        if not self.initialized or audio_np.size == 0:
            return ""
            
        try:
            # 转换为模型需要的格式
            audio_float32 = audio_np.astype(np.float32) / 32768.0
            
            # 执行语音识别
            res = self.model.generate(
                input=audio_float32,
                language="zh",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=MAX_SPEECH_DURATION,
                sampling_rate=sample_rate,
            )
            
            if res and len(res) > 0 and "text" in res[0]:
                text = rich_transcription_postprocess(res[0]["text"])
                return text.strip()
            else:
                return ""
                
        except Exception as e:
            print(f"❌ 语音识别错误: {e}")
            return ""

class AudioRecorder:
    def __init__(self, interface_name="eth0", enable_denoise=False, save_denoised=True, enable_recognition=True):
        self.interface_name = interface_name
        self.enable_denoise = enable_denoise
        self.save_denoised = save_denoised
        self.enable_recognition = enable_recognition
        self.socket = None
        self.recording_start_time = None
        self.denoise_processor = None
        self.recognizer = None
        
        # 初始化处理器
        if self.enable_denoise:
            self.denoise_processor = DeepFilterNetProcessor()
            self.denoise_processor.initialize()
            
        if self.enable_recognition:
            self.recognizer = SpeechRecognizer()
            self.recognizer.initialize()
        
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
            
    def save_audio_session(self, original_audio, denoised_audio, session_id):
        """保存音频会话数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("data/sessions", exist_ok=True)
        
        # 保存原始音频
        original_wav_path = f"data/sessions/session_{session_id}_{timestamp}_original.wav"
        self.save_wav(original_audio, original_wav_path, FRAME_RATE)
        
        # 保存降噪音频（如果启用）
        denoised_wav_path = None
        if self.enable_denoise and denoised_audio is not None and self.save_denoised:
            denoised_wav_path = f"data/sessions/session_{session_id}_{timestamp}_denoised.wav"
            self.save_wav(denoised_audio, denoised_wav_path, FRAME_RATE)
        
        return original_wav_path, denoised_wav_path
        
    def save_wav(self, audio_data, file_path, sample_rate):
        """保存WAV文件"""
        try:
            with wave.open(file_path, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(sample_rate)
                f_wav.writeframes(audio_data.tobytes())
            print(f"💾 音频已保存: {file_path}")
        except Exception as e:
            print(f"❌ 保存音频失败 {file_path}: {e}")
            
    def process_audio_frame(self, audio_data):
        """处理音频帧"""
        global is_recording
        
        frame_duration = 10  # 10ms帧
        frame_size = int(FRAME_RATE * frame_duration / 1000 * 2)
        frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]
        
        for frame in frames:
            if len(frame) < frame_size:
                continue
                
            # 转换为numpy数组
            frame_np = np.frombuffer(frame, dtype=np.int16)
            
            # 如果正在录音，存储到缓冲区
            if is_recording:
                audio_buffer.extend(frame_np)
                            
    def process_complete_speech(self):
        """处理并保存音频"""
        global session_counter
        session_counter += 1
        
        if not audio_buffer:
            print("⚠️  没有检测到有效音频数据，缓冲区为空")
            return None, None
            
        original_audio = np.array(list(audio_buffer), dtype=np.int16)
        
        print(f"📊 缓冲区包含 {len(original_audio)} 个样本（约 {len(original_audio)/FRAME_RATE:.2f}秒）")
        
        # 清空缓冲区
        audio_buffer.clear()
        
        # 降噪处理
        denoised_audio = None
        if self.enable_denoise and self.denoise_processor and self.denoise_processor.initialized:
            print("🧠 正在进行降噪处理...")
            denoised_audio = self.denoise_processor.process_audio(original_audio)
            print("✅ 降噪处理完成")
        
        # 保存音频
        original_path, denoised_path = self.save_audio_session(original_audio, denoised_audio, session_counter)
        
        # 语音识别
        if self.enable_recognition and self.recognizer and self.recognizer.initialized:
            print("🎤 正在进行语音识别...")
            # 使用降噪后的音频进行识别（如果可用），否则使用原始音频
            recognition_audio = denoised_audio if denoised_audio is not None else original_audio
            recognized_text = self.recognizer.recognize(recognition_audio)
            
            if recognized_text:
                print(f"🎯 识别结果: {recognized_text}")
                
                # 保存识别结果到文本文件
                text_file_path = f"data/sessions/session_{session_counter}_transcript.txt"
                with open(text_file_path, 'w', encoding='utf-8') as f:
                    f.write(recognized_text)
                print(f"💾 识别结果已保存: {text_file_path}")
            else:
                print("⚠️  语音识别结果为空")
        
        return original_path, denoised_path
        
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
        global is_recording
        
        audio_buffer.clear()
        self.recording_start_time = time.time()
        
        print(f"🔴 开始录音: {datetime.now().strftime('%H:%M:%S')}，按回车停止...")
        is_recording = True
        
        # 等待用户按回车停止录音
        try:
            input()
        except KeyboardInterrupt:
            pass
            
        is_recording = False
        duration = time.time() - self.recording_start_time
        print(f"⏹️  录音结束，持续时间: {duration:.2f}秒")
        self.process_complete_speech()
        
    def start(self):
        """开始录音过程"""
        global audio_receiver_running, audio_receiver_thread
        
        audio_receiver_running = True
        audio_receiver_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        audio_receiver_thread.start()
        
        try:
            while True:
                input("按回车键开始录音...")
                self.start_recording()
                print("\n" + "="*50)
                user_input = input("按回车继续录音，输入'q'退出程序: ")
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
    parser = argparse.ArgumentParser(description="音频录制、降噪和语音识别系统")
    parser.add_argument("--interface", default="eth0", help="网络接口名称 (默认: eth0)")
    parser.add_argument("--denoise", action="store_true", help="启用降噪处理")
    parser.add_argument("--no-save-denoised", action="store_true", help="不保存降噪后的音频文件")
    parser.add_argument("--no-recognition", action="store_true", help="禁用语音识别")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="推理设备 (默认: cuda)")
    
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 显示配置信息
    print("🎯 音频处理系统配置:")
    print(f"   网络接口: {args.interface}")
    print(f"   降噪处理: {'启用' if args.denoise else '禁用'}")
    print(f"   保存降噪音频: {'启用' if not args.no_save_denoised else '禁用'}")
    print(f"   语音识别: {'启用' if not args.no_recognition else '禁用'}")
    print(f"   推理设备: {args.device}")
    print()
    
    recorder = AudioRecorder(
        interface_name=args.interface,
        enable_denoise=args.denoise,
        save_denoised=not args.no_save_denoised,
        enable_recognition=not args.no_recognition
    )
    
    try:
        recorder.setup_audio_receiver()
        print("\n🎉 音频处理系统已启动")
        recorder.start()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
    finally:
        if is_recording and audio_buffer:
            print("💾 强制保存未完成的音频...")
            recorder.process_complete_speech()
        recorder.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    main()