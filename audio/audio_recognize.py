import sys
import os
import numpy as np
import wave
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 音频参数
FRAME_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
MAX_SPEECH_DURATION = 30

# 初始化语音识别模型
# model_dir = "FunAudioLLM/SenseVoiceSmall"
# model_dir = "/home/sakuzeng/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"
model_dir = "/home/unitree/.cache/huggingface/hub/models--FunAudioLLM--SenseVoiceSmall/snapshots/3eb3b4eeffc2f2dde6051b853983753db33e35c3"

asr_model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad", # 集成 VAD 模型，用于长音频切割（检测语音段落），<30s可设为none。
    vad_kwargs={"max_single_segment_time": 30000}, #设置最大单段音频时长为 30000 ms（30s），防止过长段落导致内存溢出。
    hub="hf", # 指定模型来源
    device="cuda",
    # device="cpu",
    disable_update=True, # 禁用自动更新模型
)

def convert_audio_to_target_format(audio_path, target_channels=CHANNELS, target_sample_width=SAMPLE_WIDTH, target_sample_rate=FRAME_RATE, output_path=None):
    """
    将音频文件转换为目标格式（通道数、采样宽度、采样率）
    
    Args:
        audio_path (str): 输入音频文件路径
        target_channels (int): 目标通道数
        target_sample_width (int): 目标采样宽度（字节）
        target_sample_rate (int): 目标采样率
        output_path (str, optional): 保存转换后的音频文件路径。如果为 None，则不保存
    
    Returns:
        tuple: (转换后的音频数据 (np.int16), 实际采样率)
    """
    try:
        # 使用 torchaudio 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 检查参数
        current_channels = waveform.shape[0]
        current_sample_width = waveform.dtype.itemsize
        current_sample_rate = sample_rate
        
        need_conversion = (
            current_channels != target_channels or
            current_sample_width != target_sample_width or
            current_sample_rate != target_sample_rate
        )
        
        if need_conversion:
            print(f"⚠️  音频参数不匹配，正在转换...")
            print(f"   当前: channels={current_channels}, sample_width={current_sample_width}, sample_rate={current_sample_rate}")
            print(f"   目标: channels={target_channels}, sample_width={target_sample_width}, sample_rate={target_sample_rate}")
            
            # 1. 通道数转换（立体声转单声道）
            if current_channels != target_channels:
                if current_channels > 1 and target_channels == 1:
                    waveform = waveform.mean(dim=0, keepdim=True)  # 取平均值转换为单声道
                else:
                    print(f"❌ 不支持的通道转换: {current_channels} -> {target_channels}")
                    return None, None
            
            # 2. 采样率重采样
            if current_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=current_sample_rate,
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
            
            # 3. 采样宽度转换
            if current_sample_width != target_sample_width:
                if target_sample_width == 2:  # 转换为 16 位
                    waveform = (waveform * 32767).clamp(-32768, 32767).to(torch.int16)
                else:
                    print(f"❌ 不支持的采样宽度转换: {current_sample_width} -> {target_sample_width}")
                    return None, None
            
            # 转换为 numpy 数组
            audio_data = waveform.numpy().astype(np.int16)
            if target_channels == 1:
                audio_data = audio_data[0]  # 单声道，去掉通道维度
            
            # 保存转换后的音频（可选）
            if output_path:
                torchaudio.save(
                    output_path,
                    waveform,
                    target_sample_rate,
                    channels_first=True
                )
                print(f"💾 转换后的音频已保存: {output_path}")
            
            print("✅ 音频转换完成")
            return audio_data, target_sample_rate
        
        else:
            # 参数匹配，直接转换为 numpy 数组
            audio_data = (waveform.numpy() * 32767).astype(np.int16)
            if current_channels == 1:
                audio_data = audio_data[0]
            return audio_data, sample_rate
    
    except Exception as e:
        print(f"❌ 音频转换错误: {e}")
        return None, None

def recognize_speech(audio_path):
    """语音识别"""
    # 读取并转换音频
    print(f"📥 加载音频文件: {audio_path}")
    
    # 转换音频到目标格式
    audio_data, sample_rate = convert_audio_to_target_format(audio_path)
    
    if audio_data is None or sample_rate is None:
        print("❌ 无法加载或转换音频文件")
        return ""
    
    if audio_data.size == 0:
        print("⚠️ 音频文件为空")
        return ""
    
    print(f"🎯 开始语音识别，音频长度: {len(audio_data)/FRAME_RATE:.2f}秒")
    
    # 转换为模型需要的格式
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    
    try:
        # 执行语音识别推理，返回结果列表（res[0]["text"] 为原始文本）
        res = asr_model.generate(
            input=audio_float32, # 支持文件路径、数组或列表
            language="zh",
            use_itn=True, # 启用逆文本规范化（添加标点、情感标签，如 <happy>）
            batch_size_s=60, # 动态批量大小 
            merge_vad=True, #合并 VAD 切割的短片段
            merge_length_s=MAX_SPEECH_DURATION, # 合并后的最大时长，单位秒
            sampling_rate=16000, # 指定输入采样率
        )
        
        if res and len(res) > 0 and "text" in res[0]:
            text = rich_transcription_postprocess(res[0]["text"])
            return text.strip()
        else:
            return ""
    
    except Exception as e:
        print(f"❌ 语音识别错误: {e}")
        return ""

def main():
    if len(sys.argv) < 2:
        print("用法: python audio_recognize.py <input_wav_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    recognized_text = recognize_speech(input_path)
    
    if recognized_text:
        print(f"🎤 识别结果: {recognized_text}")
    else:
        print("⚠️  语音识别结果为空")

if __name__ == "__main__":
    main()