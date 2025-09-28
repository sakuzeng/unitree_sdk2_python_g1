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

# 音频参数
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000  # 默认采样率
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
is_recording = False
session_counter = 0

class AudioRecorder:
    def __init__(self, interface_name="eth0"):
        self.interface_name = interface_name
        self.socket = None
        self.recording_start_time = None
        self.recording_file = None
        self.current_session_file = None
        
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
        
    def create_session_file(self, session_id):
        """创建新的录音会话文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("data/sessions", exist_ok=True)
        
        # 创建RAW文件路径
        raw_filename = f"data/sessions/session_{session_id}_{timestamp}.raw"
        return raw_filename
        
    def convert_raw_to_wav(self, raw_path, wav_path):
        """将RAW文件转换为WAV格式"""
        try:
            # 读取原始数据
            with open(raw_path, 'rb') as f_raw:
                raw_data = f_raw.read()
            
            # 计算音频时长
            sample_count = len(raw_data) // SAMPLE_WIDTH
            duration = sample_count / FRAME_RATE
            
            # 写入WAV文件
            with wave.open(wav_path, 'wb') as f_wav:
                f_wav.setnchannels(CHANNELS)
                f_wav.setsampwidth(SAMPLE_WIDTH)
                f_wav.setframerate(FRAME_RATE)
                f_wav.writeframes(raw_data)
            
            print(f"💾 音频转换完成: {wav_path}")
            print(f"📊 音频时长: {duration:.2f}秒, 文件大小: {len(raw_data)}字节")
            return True
            
        except Exception as e:
            print(f"❌ 音频转换错误: {e}")
            return False
            
    def process_audio_data(self, audio_data):
        """直接存储音频数据"""
        global is_recording
        
        if is_recording and audio_data and self.recording_file:
            try:
                # 直接写入文件
                self.recording_file.write(audio_data)
                self.recording_file.flush()  # 确保数据写入磁盘
            except Exception as e:
                print(f"❌ 写入音频数据失败: {e}")
                            
    def listen_for_audio(self):
        """监听音频数据"""
        global audio_receiver_running
        
        print("👂 开始监听音频数据...")
        bytes_received = 0
        packet_count = 0
        
        while audio_receiver_running:
            try:
                data, addr = self.socket.recvfrom(2048)
                bytes_received += len(data)
                packet_count += 1
                
                # 处理音频数据
                self.process_audio_data(data)
                
                # 每100个包显示一次统计信息
                if packet_count % 100 == 0:
                    print(f"📊 已接收 {packet_count} 个数据包, 总计 {bytes_received} 字节")
                    
            except socket.timeout:
                continue
            except Exception as e:
                if audio_receiver_running:
                    print(f"❌ 音频接收错误: {e}")
                break
                
        print("👂 音频监听已停止")
        
    def start_recording(self):
        """开始录音"""
        global is_recording, session_counter
        
        session_counter += 1
        
        # 创建录音文件
        raw_filename = self.create_session_file(session_counter)
        
        try:
            self.recording_file = open(raw_filename, 'wb')
            self.current_session_file = raw_filename
            
            self.recording_start_time = time.time()
            print(f"🔴 开始录音: {datetime.now().strftime('%H:%M:%S')}")
            print(f"💾 保存到: {raw_filename}")
            print("按回车键停止录音...")
            
            is_recording = True
            
            # 等待用户停止录音
            try:
                input()
            except KeyboardInterrupt:
                pass
                
        except Exception as e:
            print(f"❌ 开始录音失败: {e}")
            return
            
        finally:
            is_recording = False
            
            # 关闭文件
            if self.recording_file:
                self.recording_file.close()
                self.recording_file = None
            
            duration = time.time() - self.recording_start_time
            print(f"⏹️  录音结束，持续时间: {duration:.2f}秒")
            
            # 转换为WAV格式
            if self.current_session_file and os.path.exists(self.current_session_file):
                wav_filename = self.current_session_file.replace('.raw', '.wav')
                self.convert_raw_to_wav(self.current_session_file, wav_filename)
        
    def start(self):
        """开始录音过程"""
        global audio_receiver_running, audio_receiver_thread
        
        audio_receiver_running = True
        audio_receiver_thread = threading.Thread(target=self.listen_for_audio, daemon=True)
        audio_receiver_thread.start()
        
        try:
            while True:
                print("\n" + "="*50)
                input("按回车键开始录音...")
                self.start_recording()
                
                user_input = input("\n按回车继续录音，输入'q'退出程序: ")
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
        # 关闭录音文件
        if self.recording_file:
            try:
                self.recording_file.close()
            except:
                pass
        
        # 关闭socket
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
        print("✨ 简化版 - 直接存储原始音频数据")
        recorder.start()
        
    except Exception as e:
        print(f"❌ 程序运行错误: {e}")
    finally:
        recorder.cleanup()
        print("👋 程序已退出")

if __name__ == "__main__":
    main()