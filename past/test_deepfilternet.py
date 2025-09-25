#!/usr/bin/env python3
"""
DeepFilterNet 音频降噪简单示例
直接对文件进行降噪处理
"""

import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
from df.enhance import enhance, init_df

def deepfilter_denoise_simple():
    """简单的 DeepFilterNet 降噪"""
    
    # 输入文件路径
    input_file = "data/sessions/session_1_20250924_171834_original.wav"
    output_dir = "data/sessions/deepfilter_results"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return
    
    print(f"🎯 处理文件: {input_file}")
    
    try:
        # 1. 初始化 DeepFilterNet
        print("🧠 初始化 DeepFilterNet...")
        model, df_state, _ = init_df()
        
        # 2. 使用 torchaudio 读取音频
        audio, sample_rate = torchaudio.load(input_file)
        
        # 转换为单声道
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # 重采样到模型需要的采样率
        target_sr = df_state.sr()
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            audio = resampler(audio)
            sample_rate = target_sr
        
        print(f"📊 音频信息: 采样率={sample_rate}Hz, 时长={audio.shape[1]/sample_rate:.2f}s")
        
        # 3. 进行降噪处理
        print("🔊 开始降噪...")
        with torch.no_grad():
            enhanced_audio = enhance(model, df_state, audio)
        
        # 4. 保存结果
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_deepfilter.wav")
        
        # 保存降噪后的音频
        torchaudio.save(output_file, enhanced_audio, sample_rate)
        
        print(f"✅ 降噪完成!")
        print(f"📁 输出文件: {output_file}")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    deepfilter_denoise_simple()