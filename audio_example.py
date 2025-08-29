import sys
import time
import signal
import socket
import struct
import threading
import netifaces
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

# 导入正确的ASR消息类型 - 根据C++代码，应该是String_类型
from unitree_sdk2py.idl.std_msgs.msg.dds_._String_ import String_

# 常量定义
AUDIO_SUBSCRIBE_TOPIC = "rt/audio_msg"  # ASR结果订阅话题
MULTICAST_GROUP = "239.168.123.161"  # 组播地址
MULTICAST_PORT = 5555  # 组播端口

# 全局变量
audio_receiver_running = False
audio_receiver_thread = None

def asr_handler(msg):
    """ASR结果处理回调函数"""
    print(f"ASR识别结果: {msg.data}")

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
        
        audio_data_count = 0
        total_bytes = 0
        
        while audio_receiver_running:
            try:
                data, addr = sock.recvfrom(2048)  # 接收音频数据
                audio_data_count += 1
                total_bytes += len(data)
                
                # 每收到50个数据包打印一次统计信息
                if audio_data_count % 50 == 0:
                    print(f"已接收音频数据包: {audio_data_count}, 当前包大小: {len(data)} 字节, 总计: {total_bytes} 字节")
                
                # 这里可以添加音频数据处理逻辑
                # 例如保存到文件、进行音频分析等
                
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
    audio_receiver_running = False
    if audio_receiver_thread and audio_receiver_thread.is_alive():
        audio_receiver_thread.join(timeout=2)
    sys.exit(0)

def main():
    global audio_receiver_running, audio_receiver_thread
    
    if len(sys.argv) < 2:
        print(f"用法: python3 {sys.argv[0]} <网络接口名称>")
        print("例如: python3 audio_example.py eth0")
        sys.exit(1)
    
    interface_name = sys.argv[1]
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 初始化通道
    ChannelFactoryInitialize(0, interface_name)
    
    # 创建音频客户端
    audio_client = AudioClient()
    audio_client.SetTimeout(10.0)
    audio_client.Init()
    
    print("=== G1 音频功能测试开始 ===")
    
    # ASR消息订阅 - 使用正确的String_类型
    print("初始化ASR消息订阅...")
    try:
        subscriber = ChannelSubscriber(AUDIO_SUBSCRIBE_TOPIC, String_)
        subscriber.Init(asr_handler)
        print("ASR订阅初始化成功")
    except Exception as e:
        print(f"ASR订阅初始化失败: {e}")
    
    # 1. 音量控制测试
    print("\n1. 音量控制测试")
    code, volume = audio_client.GetVolume()
    if code == 0:
        print(f"当前音量: {volume}")
    else:
        print(f"获取音量失败，错误码: {code}")
    
    print("设置音量为50")
    code = audio_client.SetVolume(50)
    if code == 0:
        code, volume = audio_client.GetVolume()
        if code == 0:
            print(f"设置后音量: {volume}")
    else:
        print(f"设置音量失败，错误码: {code}")
    
    # 2. TTS测试
    print("\n2. TTS测试")
    print("播放中文TTS...")
    code = audio_client.TtsMaker("你好。我是宇树科技的机器人。例程启动成功", 0)
    if code == 0:
        print("中文TTS播放成功")
    else:
        print(f"中文TTS播放失败，错误码: {code}")
    time.sleep(5)
    
    print("播放英文TTS...")
    code = audio_client.TtsMaker("Hello. I'm a robot from Unitree Robotics. The example has started successfully.", 1)
    if code == 0:
        print("英文TTS播放成功")
    else:
        print(f"英文TTS播放失败，错误码: {code}")
    time.sleep(8)
    
    # 3. LED控制测试
    print("\n3. LED控制测试")
    led_colors = [
        (0, 255, 0, "绿色"),
        (0, 0, 0, "关闭"),
        (0, 0, 255, "蓝色")
    ]
    
    for r, g, b, name in led_colors:
        print(f"LED设置为{name}")
        code = audio_client.LedControl(r, g, b)
        if code != 0:
            print(f"LED控制失败，错误码: {code}")
        time.sleep(1)
    
    # 4. 启动音频数据接收器
    print("\n4. 启动音频数据接收器")
    print("请使用APP或遥控器将机器人切换到唤醒模式以开启麦克风")
    audio_receiver_running = True
    audio_receiver_thread = threading.Thread(target=audio_receiver, args=(interface_name,), daemon=True)
    audio_receiver_thread.start()
    
    print("\n=== 音频API测试完成，开始ASR监听和音频数据接收... ===")
    
    # 5. ASR测试
    print("\n5. ASR语音识别测试")
    print("ASR系统已启动，请对着机器人说话...")
    print("程序将持续运行，按Ctrl+C退出")
    
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
        print("程序已退出")

if __name__ == "__main__":
    main()