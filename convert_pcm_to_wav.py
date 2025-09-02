#!/usr/bin/env python3

import os
from audio_example import save_pcm_to_wav  # 从 audio_example.py 导入函数

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