#!/usr/bin/env python3

import os
import wave

def save_pcm_to_wav(pcm_file, wav_file, channels=1, sample_width=2, frame_rate=16000):
    """
    将 PCM 数据转换为 WAV 格式并保存

    Args:
        pcm_file (str): 原始 PCM 文件路径
        wav_file (str): 输出 WAV 文件路径
        channels (int): 声道数，默认单声道
        sample_width (int): 采样宽度（字节），默认 2 字节（16 位）
        frame_rate (int): 采样率，默认 16000 Hz
    """
    try:
        with open(pcm_file, 'rb') as pcm_f:
            pcm_data = pcm_f.read()
        
        with wave.open(wav_file, 'wb') as wav_f:
            wav_f.setnchannels(channels)
            wav_f.setsampwidth(sample_width)
            wav_f.setframerate(frame_rate)
            wav_f.writeframes(pcm_data)
        
        print(f"PCM 数据已成功转换为 WAV 格式并保存到 {wav_file}")
    except Exception as e:
        print(f"PCM 转换为 WAV 时出错: {e}")

def main():
    # 定义 PCM 文件和目标 WAV 文件的路径
    pcm_file_path = "data/received_audio.raw"
    wav_file_path = "data/converted_audio.wav"

    # 检查 PCM 文件是否存在
    if not os.path.exists(pcm_file_path):
        print(f"PCM 文件不存在: {pcm_file_path}")
        return

    # 调用转换函数
    print(f"开始将 PCM 文件转换为 WAV 文件...")
    save_pcm_to_wav(pcm_file_path, wav_file_path)

    # 检查 WAV 文件是否生成
    if os.path.exists(wav_file_path):
        print(f"WAV 文件已成功生成: {wav_file_path}")
    else:
        print("WAV 文件生成失败")

if __name__ == "__main__":
    main()