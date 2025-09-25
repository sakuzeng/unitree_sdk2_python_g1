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

# 音频参数
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000  # 默认采样率
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

class AudioRecorder:
    def __init__(self, interface_name="eth0"):
        self.interface_name = interface_name
        self.socket = None
        self.recording_start_time = None
        
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
            
    def save_audio_session(self, original_audio, session_id):
        """保存音频会话数据"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("data/sessions", exist_ok=True)
        
        original_raw_path = f"data/sessions/session_{session_id}_{timestamp}_original.raw"
        original_wav_path = f"data/sessions/session_{session_id}_{timestamp}_original.wav"
        
        try:
            with open(original_raw_path, "wb") as f:
                f.write(original_audio.tobytes())
                
            self.convert_raw_to_wav(original_raw_path, original_wav_path, original_audio)
            
            print(f"💾 音频会话 {session_id} 已保存:")
            print(f"   原始音频: {original_wav_path}")
            
            return original_wav_path
        except Exception as e:
            print(f"❌ 保存音频会话 {session_id} 失败: {e}")
            return None
        
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
            
    def process_audio_frame(self, audio_data):
        """处理音频帧 - 直接存储"""
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
            return
            
        original_audio = np.array(list(audio_buffer), dtype=np.int16)
        
        print(f"📊 缓冲区包含 {len(original_audio)} 个样本（约 {len(original_audio)/FRAME_RATE:.2f}秒）")
        
        audio_buffer.clear()
        
        self.save_audio_session(original_audio, session_counter)
        
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
    if len(sys.argv) < 2:
        print("未提供网络接口名称，使用默认值: eth0")
        interface_name = "eth0"
    else:
        interface_name = sys.argv[1]
        print(f"使用网络接口: {interface_name}")
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    recorder = AudioRecorder(interface_name)
    
    try:
        recorder.setup_audio_receiver()
        print("\n🎉 录音模块已启动")
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