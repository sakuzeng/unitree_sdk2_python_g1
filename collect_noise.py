#!/usr/bin/env python3
import sys
import time
import socket
import struct
import os
import numpy as np
import netifaces
import wave

# 从 audio_example.py 复制过来的常量
MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
DATA_DIR = "data"
NOISE_RAW_PATH = os.path.join(DATA_DIR, "noise_profile.raw")
NOISE_WAV_PATH = os.path.join(DATA_DIR, "noise_profile.wav")
NOISE_NPY_PATH = os.path.join(DATA_DIR, "noise_profile.npy")
RECORD_DURATION = 5 # 录制5秒以获得更稳定的噪声样本

# 音频参数 (单通道, 16-bit, 16kHz)
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000

def get_local_ip_for_multicast():
    """获取192.168.123.x网段的本地IP地址"""
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

def main():
    """主函数，用于采集并保存噪声样本"""
    if len(sys.argv) < 2:
        print("未提供网络接口名称，使用默认值: eth0")
        network_interface = "eth0"
    else:
        network_interface = sys.argv[1]
        print(f"使用提供的网络接口名称: {network_interface}")
    
    print("请使用APP或遥控器将机器人切换到唤醒模式以开启麦克风。")
    print(f"准备采集环境噪声，请在 {RECORD_DURATION} 秒内保持绝对安静...")
    time.sleep(2) # 给用户准备时间

    noise_data = bytearray()
    start_time = time.time()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('', MULTICAST_PORT))
            
            local_ip = get_local_ip_for_multicast()
            if not local_ip:
                print("错误：无法获取本地IP，无法采集噪声。")
                return
                
            mreq = struct.pack("4s4s", socket.inet_aton(MULTICAST_GROUP), socket.inet_aton(local_ip))
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(1.0)
            
            print("正在采集中...")
            while time.time() - start_time < RECORD_DURATION:
                try:
                    data, _ = sock.recvfrom(2048)
                    noise_data.extend(data)
                except socket.timeout:
                    print("...仍在等待音频数据...")
                    continue
            
            if len(noise_data) > 0:
                # 确保目录存在
                os.makedirs(DATA_DIR, exist_ok=True)
                
                # 1. 保存为 .raw 文件
                with open(NOISE_RAW_PATH, "wb") as f_raw:
                    f_raw.write(noise_data)
                print(f"\n原始噪声数据已保存到: {NOISE_RAW_PATH}")

                # 2. 保存为 .wav 文件
                with wave.open(NOISE_WAV_PATH, 'wb') as f_wav:
                    f_wav.setnchannels(CHANNELS)
                    f_wav.setsampwidth(SAMPLE_WIDTH)
                    f_wav.setframerate(FRAME_RATE)
                    f_wav.writeframes(noise_data)
                print(f"WAV 格式噪声数据已保存到: {NOISE_WAV_PATH}")

                # 3. 保存为 .npy 文件 (供 audio_example.py 使用)
                noise_profile = np.frombuffer(noise_data, dtype=np.int16)
                np.save(NOISE_NPY_PATH, noise_profile)
                print(f"Numpy 格式噪声数据已保存到: {NOISE_NPY_PATH}")

            else:
                print("\n错误：未能采集到任何音频数据。请检查机器人麦克风是否已开启。")

    except Exception as e:
        print(f"采集噪声时发生错误: {e}")

if __name__ == "__main__":
    main()