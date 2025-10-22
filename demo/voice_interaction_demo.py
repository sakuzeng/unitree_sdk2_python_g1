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
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
import webrtcvad
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import noisereduce as nr
import torch
import torchaudio

# 导入DeepFilterNet（可选）
try:
    from df.enhance import enhance, init_df
    DEEPFILTERNET_AVAILABLE = True
    print("✅ DeepFilterNet 可用")
except ImportError:
    DEEPFILTERNET_AVAILABLE = False
    print("⚠️  DeepFilterNet 不可用，将使用 noisereduce 进行降噪")

# 常量定义
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000
RECORDING_DURATION = 5  # 录音时长（秒）
SILENCE_TIMEOUT = 2.0
MAX_SPEECH_DURATION = 30

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
is_recording = False
recording_start_time = None
global_noise_profile = None
noise_reduction_buffer = deque(maxlen=FRAME_RATE // 2)

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

class DeepFilterNetProcessor:
    """DeepFilterNet 音频处理器"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.model = None
        self.df_state = None
        self.target_sample_rate = None
        self.resampler_to_target = None
        self.resampler_from_target = None
        self.initialized = False
        
    def initialize(self):
        """初始化 DeepFilterNet"""
        if not DEEPFILTERNET_AVAILABLE:
            self.initialized = False
            return
            
        try:
            print("🧠 正在初始化 DeepFilterNet...")
            
            self.model, self.df_state, _ = init_df()
            self.target_sample_rate = self.df_state.sr()
            
            if self.sample_rate != self.target_sample_rate:
                self.resampler_to_target = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, 
                    new_freq=self.target_sample_rate
                )
                self.resampler_from_target = torchaudio.transforms.Resample(
                    orig_freq=self.target_sample_rate, 
                    new_freq=self.sample_rate
                )
            
            print(f"✅ DeepFilterNet 初始化成功 (模型采样率: {self.target_sample_rate}Hz)")
            self.initialized = True
            
        except Exception as e:
            print(f"❌ DeepFilterNet 初始化失败: {e}")
            self.initialized = False
            
    def process_audio(self, audio_np):
        """使用 DeepFilterNet 处理音频"""
        if not self.initialized or audio_np.size == 0:
            return audio_np
            
        try:
            audio_float = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
            
            if self.resampler_to_target is not None:
                audio_float = self.resampler_to_target(audio_float)
            
            with torch.no_grad():
                enhanced_audio = enhance(self.model, self.df_state, audio_float)
            
            if self.resampler_from_target is not None:
                enhanced_audio = self.resampler_from_target(enhanced_audio)
            
            enhanced_np = enhanced_audio.squeeze(0).numpy()
            enhanced_int16 = (enhanced_np * 32767.0).astype(np.int16)
            enhanced_int16 = np.clip(enhanced_int16, -32768, 32767)
            
            return enhanced_int16
            
        except Exception as e:
            print(f"❌ DeepFilterNet 处理错误: {e}")
            return audio_np

class VoiceInteractionDemo:
    """语音交互演示类"""
    
    def __init__(self, interface_name="eth0"):
        self.interface_name = interface_name
        self.audio_client = None
        self.socket = None
        self.deepfilter_processor = None
        
    def initialize(self):
        """初始化音频客户端和网络连接"""
        try:
            # 初始化通道
            ChannelFactoryInitialize(0, self.interface_name)
            
            # 创建音频客户端
            self.audio_client = AudioClient()
            self.audio_client.SetTimeout(10.0)
            self.audio_client.Init()
            
            # 设置音量
            code = self.audio_client.SetVolume(40)
            if code == 0:
                print("🔊 音量设置为40")
            
            # 设置音频接收器
            self.setup_audio_receiver()
            
            # 初始化 DeepFilterNet 处理器
            self.deepfilter_processor = DeepFilterNetProcessor(FRAME_RATE)
            self.deepfilter_processor.initialize()
            
            print("✅ 语音交互系统初始化完成")
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
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
        
    def play_tts(self, text, speaker_id=0):
        """播放TTS语音"""
        try:
            print(f"🔊 播放语音: {text}")
            code = self.audio_client.TtsMaker(text, speaker_id)
            if code == 0:
                print("✅ TTS播放成功")
                # 等待播放完成
                time.sleep(len(text) * 0.15 + 2)  # 估算播放时间
            else:
                print(f"❌ TTS播放失败，错误码: {code}")
        except Exception as e:
            print(f"❌ TTS播放错误: {e}")
            
    def apply_noisereduce_suppression(self, audio_np, sample_rate=16000):
        """使用 noisereduce 进行音频降噪处理"""
        global global_noise_profile, noise_reduction_buffer
        
        if audio_np.size == 0:
            return audio_np
        
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        try:
            if global_noise_profile is not None and global_noise_profile.size > 0:
                noise_float = global_noise_profile.astype(np.float32) / 32768.0
                denoised_audio = nr.reduce_noise(
                    y=audio_float, 
                    y_noise=noise_float, 
                    sr=sample_rate,
                    stationary=True,
                    prop_decrease=0.8
                )
            else:
                noise_reduction_buffer.extend(audio_np)
                
                if len(noise_reduction_buffer) >= FRAME_RATE // 4:
                    noise_sample = np.array(list(noise_reduction_buffer)[-FRAME_RATE//4:], dtype=np.float32) / 32768.0
                    denoised_audio = nr.reduce_noise(
                        y=audio_float,
                        y_noise=noise_sample,
                        sr=sample_rate,
                        stationary=False,
                        prop_decrease=0.5
                    )
                else:
                    denoised_audio = nr.reduce_noise(
                        y=audio_float,
                        sr=sample_rate,
                        stationary=False,
                        prop_decrease=0.3
                    )
        except Exception as e:
            print(f"❌ noisereduce 降噪处理出错: {e}")
            denoised_audio = audio_float
        
        denoised_audio_int16 = (denoised_audio * 32768.0).astype(np.int16)
        denoised_audio_int16 = np.clip(denoised_audio_int16, -32768, 32767)
        
        return denoised_audio_int16
        
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
        """录音指定时长"""
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
            
    def save_audio_data(self, original_audio, denoised_noisereduce, denoised_deepfilter, session_id):
        """保存原始音频和降噪后的音频数据"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("data/sessions", exist_ok=True)
            
            # 保存原始音频
            original_wav = f"data/sessions/session_{session_id}_{timestamp}_original.wav"
            with wave.open(original_wav, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(FRAME_RATE)
                f_wav.writeframes(original_audio.tobytes())
            
            # 保存 noisereduce 降噪音频
            noisereduce_wav = f"data/sessions/session_{session_id}_{timestamp}_noisereduce.wav"
            with wave.open(noisereduce_wav, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(FRAME_RATE)
                f_wav.writeframes(denoised_noisereduce.tobytes())
            
            saved_files = [original_wav, noisereduce_wav]
            
            # 保存 DeepFilterNet 降噪音频（如果可用）
            if denoised_deepfilter is not None:
                deepfilter_wav = f"data/sessions/session_{session_id}_{timestamp}_deepfilter.wav"
                with wave.open(deepfilter_wav, 'wb') as f_wav:
                    f_wav.setnchannels(CHANNELS)
                    f_wav.setsampwidth(SAMPLE_WIDTH)
                    f_wav.setframerate(FRAME_RATE)
                    f_wav.writeframes(denoised_deepfilter.tobytes())
                saved_files.append(deepfilter_wav)
            
            print(f"💾 音频文件已保存:")
            for file_path in saved_files:
                print(f"   {os.path.basename(file_path)}")
            
            return saved_files
            
        except Exception as e:
            print(f"❌ 保存音频失败: {e}")
            return []
            
    def recognize_speech(self, audio_data, description=""):
        """语音识别"""
        if audio_data.size == 0:
            return ""
            
        print(f"🧠 开始语音识别{description}...")
        
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
            print(f"❌ 语音识别错误{description}: {e}")
            return ""
            
    def classify_text(self, text):
        """分类文本并返回匹配的指令"""
        if not text:
            return None
            
        text_lower = text.lower()
        
        if "前进" in text:
            return "前进"
        elif "后退" in text:
            return "后退"
        elif "打招呼" in text or "招呼" in text:
            return "打招呼"
        else:
            return None
            
    def classify_and_respond(self, original_text, noisereduce_text, deepfilter_text=None):
        """对识别的文本进行分类并响应"""
        print(f"🎯 识别结果对比:")
        print(f"   原始音频: {original_text}")
        print(f"   noisereduce降噪: {noisereduce_text}")
        if deepfilter_text is not None:
            print(f"   DeepFilterNet降噪: {deepfilter_text}")
        
        # 分别检查每个结果
        original_command = self.classify_text(original_text)
        noisereduce_command = self.classify_text(noisereduce_text)
        deepfilter_command = None
        if deepfilter_text is not None:
            deepfilter_command = self.classify_text(deepfilter_text)
        
        # 找到第一个有效指令
        final_command = None
        source = ""
        
        if original_command:
            final_command = original_command
            source = "原始音频"
        elif noisereduce_command:
            final_command = noisereduce_command
            source = "noisereduce降噪"
        elif deepfilter_command:
            final_command = deepfilter_command
            source = "DeepFilterNet降噪"
        
        if final_command:
            print(f"✅ 识别到指令: {final_command} (来源: {source})")
            
            if final_command == "前进":
                self.play_tts("我将前进2米")
                print("🤖 执行动作: 前进2米")
            elif final_command == "后退":
                self.play_tts("我将后退2米")
                print("🤖 执行动作: 后退2米")
            elif final_command == "打招呼":
                self.play_tts("我将做出打招呼的动作")
                print("🤖 执行动作: 打招呼")
        else:
            self.play_tts("抱歉，我不理解您的指令，请说前进、后退或打招呼")
            print("❓ 未识别的指令")
            
    def run_demo(self):
        """运行演示"""
        global audio_receiver_running, audio_receiver_thread
        
        # 启动音频监听线程
        audio_receiver_running = True
        audio_receiver_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        audio_receiver_thread.start()
        
        session_counter = 0
        
        try:
            while True:
                session_counter += 1
                print(f"\n{'='*60}")
                print(f"🎯 语音交互演示 - 第 {session_counter} 轮")
                print(f"{'='*60}")
                
                # 1. 播放提示语音
                self.play_tts("有什么需要我做的")
                
                # 2. 录音
                original_audio = self.record_audio(RECORDING_DURATION)
                
                if original_audio.size == 0:
                    print("⚠️  没有录制到音频数据")
                    continue
                
                # 3. 使用两种方法进行降噪
                print("🧹 正在进行音频降噪处理...")
                
                # noisereduce 降噪
                denoised_noisereduce = self.apply_noisereduce_suppression(original_audio.copy())
                
                # DeepFilterNet 降噪（如果可用）
                denoised_deepfilter = None
                if self.deepfilter_processor and self.deepfilter_processor.initialized:
                    denoised_deepfilter = self.deepfilter_processor.process_audio(original_audio.copy())
                
                # 4. 保存音频数据
                self.save_audio_data(original_audio, denoised_noisereduce, denoised_deepfilter, session_counter)
                
                # 5. 对三个版本的音频分别进行语音识别
                print("🧠 开始多版本语音识别...")
                
                original_text = self.recognize_speech(original_audio, "(原始音频)")
                noisereduce_text = self.recognize_speech(denoised_noisereduce, "(noisereduce降噪)")
                
                deepfilter_text = None
                if denoised_deepfilter is not None:
                    deepfilter_text = self.recognize_speech(denoised_deepfilter, "(DeepFilterNet降噪)")
                
                # 6. 分类和响应
                self.classify_and_respond(original_text, noisereduce_text, deepfilter_text)
                
                print(f"\n{'='*60}")
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
        
        if audio_receiver_thread and audio_receiver_thread.is_alive():
            audio_receiver_thread.join(timeout=2)
            
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
                
        print("✅ 资源清理完成")

def load_noise_profile(filepath="data/noise_profile.npy"):
    """从文件加载预先录制的噪声样本"""
    global global_noise_profile
    try:
        global_noise_profile = np.load(filepath)
        print(f"✅ 成功加载噪声样本文件: {filepath}")
    except FileNotFoundError:
        print(f"⚠️  噪声样本文件未找到: {filepath}")
        print("将使用自适应降噪模式")
    except Exception as e:
        print(f"❌ 加载噪声样本时发生错误: {e}")

def signal_handler(signum, frame):
    """信号处理函数"""
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
    
    # 加载噪声样本
    load_noise_profile()
    
    # 创建演示实例
    demo = VoiceInteractionDemo(interface_name)
    
    try:
        print("🎉 G1 多重降噪语音交互演示启动")
        print("🔧 功能: 语音播放 + 录音 + 双重降噪 + 多版本识别 + 指令分类")
        print("📝 支持指令: 前进、后退、打招呼")
        print(f"⏱️  录音时长: {RECORDING_DURATION}秒")
        print("🎯 降噪方法: noisereduce + DeepFilterNet(可选)")
        
        demo.initialize()
        demo.run_demo()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    main()