import sys
import time
import signal
import socket
import struct
import threading
import netifaces
import os
import json
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
import webrtcvad
import numpy as np
from collections import deque
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import noisereduce as nr
import wave

# 导入正确的ASR消息类型
from unitree_sdk2py.idl.std_msgs.msg.dds_._String_ import String_

# 常量定义
AUDIO_SUBSCRIBE_TOPIC = "rt/audio_msg"  # ASR结果订阅话题
MULTICAST_GROUP = "239.168.123.161"  # 组播地址
MULTICAST_PORT = 5555  # 组播端口
SILENCE_TIMEOUT = 2.0  # 静音超时（秒），对于长句可以适当加长
MAX_SPEECH_DURATION = 30 # 支持的最大语音时长（秒）

# 音频参数 (单通道, 16-bit, 16kHz)
CHANNELS = 1
SAMPLE_WIDTH = 2
FRAME_RATE = 16000

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None
global_noise_profile = None # 用于存储全局噪声样本

# 初始化语音识别模型
model_dir = "/home/unitree/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
asr_model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    # 这是传递给 VAD 的参数，设置单个语音片段的最大时长为 30000 毫秒（即 30 秒）。
    vad_kwargs={"max_single_segment_time": 30000},
    hub="hf",
    device="cpu",
)

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # 设置 VAD 模式，0-3，数字越大越敏感

# 缓冲区和状态
audio_buffer = deque(maxlen=16000 * 30)  # 缓存最多 30 秒的音频数据
is_speaking = False
silence_start_time = None

def asr_handler(msg):
    """ASR结果处理回调函数"""
    try:
        # 解析 JSON 数据
        asr_data = json.loads(msg.data)
        
        # 检查是否包含完整字段
        if "text" in asr_data:
            text = asr_data["text"]
            angle = asr_data.get("angle", "未知")
            speaker_id = asr_data.get("speaker_id", "未知")
            sense = asr_data.get("sense", "未知")
            confidence = asr_data.get("confidence", "未知")
            language = asr_data.get("language", "未知")
            is_final = asr_data.get("is_final", "未知")
            
            # 打印完整结果
            print(f"ASR识别结果: {text}")
            print(f"方位角度: {angle}, 说话人: {speaker_id}, 情绪: {sense}")
            print(f"置信度: {confidence}, 语言: {language}, 是否结束: {is_final}")
        else:
            print(f"ASR识别结果: 未知数据 - {asr_data}")
    except json.JSONDecodeError:
        print(f"ASR识别结果: 无法解析的消息 - {msg.data}")

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

def apply_noise_suppression(audio_np, noise_profile_np, sample_rate=16000):
    """
    对音频数据应用 noisereduce 降噪处理。
    
    Args:
        audio_np (np.ndarray): 原始音频数据的 numpy 数组 (dtype=np.int16)。
        noise_profile_np (np.ndarray): 噪声样本的 numpy 数组 (dtype=np.int16)。
        sample_rate (int): 采样率，默认 16kHz。
        
    Returns:
        np.ndarray: 降噪后的音频数据 numpy 数组 (dtype=np.int16)。
    """
    # 检查输入是否为空
    if audio_np.size == 0 or noise_profile_np.size == 0:
        return audio_np
    
    # 使用 noisereduce 进行降噪
    # 注意：y_noise 参数现在由外部提供
    denoised_audio = nr.reduce_noise(y=audio_np, y_noise=noise_profile_np, sr=sample_rate)
    
    return denoised_audio.astype(np.int16)


def load_noise_profile(filepath="data/noise_profile.npy"):
    """从文件加载预先录制的噪声样本"""
    global global_noise_profile
    try:
        global_noise_profile = np.load(filepath)
        print(f"成功加载噪声样本文件: {filepath}")
    except FileNotFoundError:
        print(f"警告: 噪声样本文件未找到: {filepath}")
        print("降噪功能将不会启用。请先运行 collect_noise.py 脚本。")
    except Exception as e:
        print(f"加载噪声样本时发生错误: {e}")


def convert_raw_to_wav(raw_path, wav_path):
    """将 .raw 音频文件转换为 .wav 文件"""
    try:
        with open(raw_path, 'rb') as f_raw:
            raw_data = f_raw.read()
        
        if not raw_data:
            return # 如果文件为空则不处理

        with wave.open(wav_path, 'wb') as f_wav:
            f_wav.setnchannels(CHANNELS)
            f_wav.setsampwidth(SAMPLE_WIDTH)
            f_wav.setframerate(FRAME_RATE)
            f_wav.writeframes(raw_data)
        print(f"已将 {os.path.basename(raw_path)} 转换为 {os.path.basename(wav_path)}")
    except FileNotFoundError:
        # 如果 raw 文件不存在，则静默处理
        pass
    except Exception as e:
        print(f"转换 {raw_path} 到 .wav 时出错: {e}")


def audio_receiver(interface_name):
    """音频数据接收线程"""
    global audio_receiver_running

    try:
        # 创建UDP套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 绑定到组播端口
        sock.bind(('', MULTICAST_PORT))

        # 获取本地IP地址（192.168.123.164）
        local_ip = get_local_ip_for_multicast()
        if local_ip is None:
            print("无法找到192.168.123.x网段的网络接口")
            return

        print(f"本地IP地址: {local_ip}")

        # 加入组播组 - 使用正确的本地接口
        mreq = struct.pack("4s4s",
                           socket.inet_aton(MULTICAST_GROUP),
                           socket.inet_aton(local_ip))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # 设置接收缓冲区大小
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        # 设置超时
        sock.settimeout(1.0)

        print(f"音频接收器已启动，监听 {MULTICAST_GROUP}:{MULTICAST_PORT}")
        print("音频格式: 单通道/16K采样率/16bit")

        # 打开文件以追加模式保存音频数据
        pcm_file_path = "data/received_audio.raw"
        os.makedirs(os.path.dirname(pcm_file_path), exist_ok=True)  # 确保目录存在

        with open(pcm_file_path, "ab") as audio_file:
            while audio_receiver_running:
                try:
                    data, addr = sock.recvfrom(2048)  # 接收音频数据
                    
                    # 保存原始音频数据到文件
                    audio_file.write(data)

                    # 直接处理音频流
                    process_audio_stream(data)

                except socket.timeout:
                    continue  # 超时继续循环
                except Exception as e:
                    if audio_receiver_running:
                        print(f"音频接收错误: {e}")
                    break

    except Exception as e:
        print(f"音频接收器初始化失败: {e}")
    finally:
        try:
            sock.close()
        except:
            pass
        print("音频接收器已关闭")

def signal_handler(signum, frame):
    """信号处理函数"""
    global audio_receiver_running
    print("\n接收到退出信号，正在关闭...")
    # 关闭音频接收线程
    audio_receiver_running = False
    if audio_receiver_thread and audio_receiver_thread.is_alive():
        audio_receiver_thread.join(timeout=2)
    
    # 在退出前转换音频文件
    print("正在转换录制的音频文件...")
    convert_raw_to_wav("data/received_audio.raw", "data/received_audio.wav")
    convert_raw_to_wav("data/denoised_audio.raw", "data/denoised_audio.wav")

    sys.exit(0)

def process_audio_stream(audio_data, sample_rate=16000):
    """
    处理音频流，检测语音活动并调用语音识别
    Args:
        audio_data (bytes): 接收到的音频数据
        sample_rate (int): 采样率，默认 16kHz
    """
    global is_speaking, silence_start_time

    # 将音频数据分割为 10ms 的帧
    frame_duration = 10  # 每帧时长 10ms
    frame_size = int(sample_rate * frame_duration / 1000 * 2)  # 每帧字节数
    frames = [audio_data[i:i + frame_size] for i in range(0, len(audio_data), frame_size)]

    for frame in frames:
        if len(frame) < frame_size:
            continue

        # 判断当前帧是否为语音
        is_speech = vad.is_speech(frame, sample_rate)

        if is_speech:
            # 如果检测到语音，加入缓冲区
            audio_buffer.extend(np.frombuffer(frame, dtype=np.int16))
            is_speaking = True
            silence_start_time = None
        else:
            # 如果检测到静音
            if is_speaking:
                if silence_start_time is None:
                    # 记录静音开始时间
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:  # 使用常量判断静音超时
                    # 停止说话，调用语音识别
                    is_speaking = False
                    silence_start_time = None
                    recognize_speech()

def recognize_speech():
    """
    调用语音识别模型处理缓冲区中的音频数据
    """
    global audio_buffer, global_noise_profile

    if not audio_buffer:
        return

    # 将缓冲区中的音频数据转换为 numpy 数组
    audio_np = np.array(list(audio_buffer), dtype=np.int16)
    audio_buffer.clear()  # 清空缓冲区

    # 如果有噪声样本，则应用降噪
    if global_noise_profile is not None and global_noise_profile.size > 0:
        denoised_audio_np = apply_noise_suppression(audio_np, global_noise_profile)
    else:
        denoised_audio_np = audio_np # 如果没有噪声样本，则使用原始音频

    # 将降噪后的音频数据追加保存到文件
    try:
        denoised_file_path = "data/denoised_audio.raw"
        # 确保目录存在，这在第一次写入时很有用
        os.makedirs(os.path.dirname(denoised_file_path), exist_ok=True)
        with open(denoised_file_path, "ab") as f:
            f.write(denoised_audio_np.tobytes())
    except Exception as e:
        print(f"保存降噪音频时出错: {e}")

    # 转换为模型需要的格式
    audio_float32 = denoised_audio_np.astype(np.float32) / 32768.0

    # 调用语音识别模型
    res = asr_model.generate(
        input=audio_float32,
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=MAX_SPEECH_DURATION,
        sampling_rate=16000,
    )
    if res and "text" in res[0]:
        text = rich_transcription_postprocess(res[0]["text"])
        print(f"识别结果: {text}")

def main():
    global audio_receiver_running, audio_receiver_thread
    
    if len(sys.argv) < 2:
        print(f"未提供网络接口名称，使用默认值: eth0")
        interface_name = "eth0"
    else:
        interface_name = sys.argv[1]
        print(f"使用提供的网络接口名称: {interface_name}")
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化通道
    ChannelFactoryInitialize(0, interface_name)

    # 从文件加载噪声样本
    load_noise_profile()
    
    # 创建音频客户端
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    
    print("=== G1 音频功能测试开始 ===")
        
    # # 1. 音量控制测试
    # print("\n1. 音量控制测试")
    # code, volume = audio_client.GetVolume()
    # if code == 0:
    #     print(f"当前音量: {volume}")
    # else:
    #     print(f"获取音量失败，错误码: {code}")
    
    # print("设置音量为50")
    # code = audio_client.SetVolume(50)
    # if code == 0:
    #     code, volume = audio_client.GetVolume()
    #     if code == 0:
    #         print(f"设置后音量: {volume}")
    # else:
    #     print(f"设置音量失败，错误码: {code}")
    
    # # 2. TTS测试
    # print("\n2. TTS测试")
    # print("播放中文TTS...")
    # code = audio_client.TtsMaker("你好。我是宇树科技的机器人。例程启动成功", 0)
    # if code == 0:
    #     print("中文TTS播放成功")
    # else:
    #     print(f"中文TTS播放失败，错误码: {code}")
    # time.sleep(5)
    
    # print("播放英文TTS...")
    # code = audio_client.TtsMaker("Hello. I'm a robot from Unitree Robotics. The example has started successfully.", 1)
    # if code == 0:
    #     print("英文TTS播放成功")
    # else:
    #     print(f"英文TTS播放失败，错误码: {code}")
    # time.sleep(8)
    
    # # 3. LED控制测试
    # print("\n3. LED控制测试")
    # led_colors = [
    #     (0, 255, 0, "绿色"),
    #     (0, 0, 0, "关闭"),
    #     (0, 0, 255, "蓝色")
    # ]
    
    # for r, g, b, name in led_colors:
    #     print(f"LED设置为{name}")
    #     code = audio_client.LedControl(r, g, b)
    #     if code != 0:
    #         print(f"LED控制失败，错误码: {code}")
    #     time.sleep(1)
    
    # # 4. ASR测试
    # print("\n4. ASR语音识别测试")
    # print("初始化ASR消息订阅...")
    # try:
    #     subscriber = ChannelSubscriber(AUDIO_SUBSCRIBE_TOPIC, String_)
    #     subscriber.Init(asr_handler)
    #     print("ASR订阅初始化成功")
    # except Exception as e:
    #     print(f"ASR订阅初始化失败: {e}")

    # print("ASR系统已启动，请对着机器人说话...")
    # print("程序将持续运行，按Ctrl+C退出")

    # 5. 启动音频数据接收器
    print("\n5. 启动音频数据接收器")
    print("请使用APP或遥控器将机器人切换到唤醒模式以开启麦克风")
    audio_receiver_running = True
    audio_receiver_thread = threading.Thread(target=audio_receiver, args=(interface_name,), daemon=True)
    audio_receiver_thread.start()

    # 等待接收器启动
    time.sleep(1) 


    try:
        # 主循环，等待ASR消息和音频数据
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n接收到退出信号")
    finally:
        audio_receiver_running = False
        if audio_receiver_thread and audio_receiver_thread.is_alive():
            audio_receiver_thread.join(timeout=2)
        
        # 在退出前转换音频文件
        print("正在转换录制的音频文件...")
        convert_raw_to_wav("data/received_audio.raw", "data/received_audio.wav")
        convert_raw_to_wav("data/denoised_audio.raw", "data/denoised_audio.wav")

        print("程序已退出")

if __name__ == "__main__":
    main()