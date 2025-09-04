#!/usr/bin/env python3
'''
@Author: sakuzeng1213
@Date: 2025-09-03
@Description: 使用 noisereduce 对音频进行降噪处理
'''

import wave
import os
import numpy as np
import noisereduce as nr

def read_wave(file_path):
    """读取 WAV 文件"""
    with wave.open(file_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "仅支持单声道音频"
        assert wf.getsampwidth() == 2, "仅支持 16 位 PCM 格式"
        audio_data = wf.readframes(wf.getnframes())
        return np.frombuffer(audio_data, dtype=np.int16), wf.getframerate()

def write_wave(file_path, audio_data, sample_rate):
    """写入 WAV 文件"""
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16 位 PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.astype(np.int16).tobytes())

def apply_noisereduce(audio_data, sample_rate):
    """使用 noisereduce 对音频数据进行降噪"""
    # 假设前 1 秒为噪声样本
    noise_sample = audio_data[:sample_rate]
    denoised_audio = nr.reduce_noise(y=audio_data, y_noise=noise_sample, sr=sample_rate)
    return denoised_audio

def main():
    input_file = "data/converted_audio.wav"
    output_file = "data/converted_audio_denoised_noisereduce.wav"

    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在: {input_file}")
        return

    # 读取 WAV 文件
    print(f"读取输入文件: {input_file}")
    audio_data, sample_rate = read_wave(input_file)

    # 应用 noisereduce 降噪处理
    print("应用 noisereduce 降噪处理...")
    denoised_data = apply_noisereduce(audio_data, sample_rate)

    # 保存降噪后的音频
    print(f"保存降噪后的音频到: {output_file}")
    write_wave(output_file, denoised_data, sample_rate)

    print("noisereduce 降噪处理完成！")

if __name__ == "__main__":
    main()