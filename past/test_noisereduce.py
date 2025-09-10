#!/usr/bin/env python3
"""
test_noisereduce.py
====================

使用 `noisereduce` 库对音频文件进行降噪处理的脚本。

功能:
1. 读取输入的 WAV 文件。
2. 使用 `noisereduce` 库对音频数据进行降噪。
3. 保存降噪后的音频到指定输出文件。

依赖:
- `noisereduce` (`pip install noisereduce`)
- `numpy` (`pip install numpy`)

注意:
- 仅支持单声道、16 位 PCM 格式的 WAV 文件。
- 假设音频的前 1 秒为噪声样本。

运行方法:
    python3 test_noisereduce.py
"""

import wave
import os
import numpy as np
import noisereduce as nr


def read_wave(file_path: str) -> tuple[np.ndarray, int]:
    """
    读取 WAV 文件并返回音频数据和采样率。

    Args:
        file_path (str): 输入 WAV 文件路径。

    Returns:
        tuple[np.ndarray, int]: 音频数据 (NumPy 数组) 和采样率。

    Raises:
        AssertionError: 如果音频不是单声道或不是 16 位 PCM 格式。
    """
    with wave.open(file_path, 'rb') as wf:
        assert wf.getnchannels() == 1, "仅支持单声道音频"
        assert wf.getsampwidth() == 2, "仅支持 16 位 PCM 格式"
        audio_data = wf.readframes(wf.getnframes())
        return np.frombuffer(audio_data, dtype=np.int16), wf.getframerate()


def write_wave(file_path: str, audio_data: np.ndarray, sample_rate: int) -> None:
    """
    将音频数据写入 WAV 文件。

    Args:
        file_path (str): 输出 WAV 文件路径。
        audio_data (np.ndarray): 音频数据 (NumPy 数组)。
        sample_rate (int): 音频采样率。
    """
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16 位 PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.astype(np.int16).tobytes())


def apply_noisereduce(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    使用 `noisereduce` 对音频数据进行降噪处理。

    Args:
        audio_data (np.ndarray): 输入音频数据。
        sample_rate (int): 音频采样率。

    Returns:
        np.ndarray: 降噪后的音频数据。
    """
    # 假设前 1 秒为噪声样本
    noise_sample = audio_data[:sample_rate]
    denoised_audio = nr.reduce_noise(y=audio_data, y_noise=noise_sample, sr=sample_rate)
    return denoised_audio


def main() -> None:
    """
    主函数，执行音频降噪处理。
    """
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