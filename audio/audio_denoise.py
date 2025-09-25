import sys
import numpy as np
import wave
import os
from datetime import datetime
from df.enhance import enhance, init_df
import torch
import torchaudio

# 音频参数（移除固定采样率）
SAMPLE_WIDTH = 2
CHANNELS = 1

class DeepFilterNetProcessor:
    """DeepFilterNet 音频处理器"""
    
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate  # 动态采样率
        self.model = None
        self.df_state = None
        self.target_sample_rate = None
        self.resampler_to_target = None
        self.resampler_from_target = None
        self.initialized = False
        
    def initialize(self):
        """初始化 DeepFilterNet"""
        try:
            print("🧠 正在初始化 DeepFilterNet...")
            
            # 初始化模型
            self.model, self.df_state, _ = init_df()
            
            # 获取模型所需采样率
            self.target_sample_rate = self.df_state.sr()
            
            # 创建重采样器（如果输入采样率 != 模型采样率）
            if self.sample_rate != self.target_sample_rate:
                self.resampler_to_target = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, 
                    new_freq=self.target_sample_rate
                )
                self.resampler_from_target = torchaudio.transforms.Resample(
                    orig_freq=self.target_sample_rate, 
                    new_freq=self.sample_rate
                )
            
            print(f"✅ DeepFilterNet 初始化成功")
            print(f"   模型采样率: {self.target_sample_rate}Hz")
            print(f"   输入采样率: {self.sample_rate}Hz")
            
            self.initialized = True
            
        except Exception as e:
            print(f"❌ DeepFilterNet 初始化失败: {e}")
            print("将回退到不使用降噪的模式")
            self.initialized = False
            
    def process_audio(self, audio_np):
        """
        使用 DeepFilterNet 处理音频
        
        Args:
            audio_np (np.ndarray): 输入音频数据 (int16)
            
        Returns:
            np.ndarray: 降噪后的音频数据 (int16)
        """
        if not self.initialized or audio_np.size == 0:
            return audio_np
            
        try:
            # 转换为 float32 格式 (-1.0 到 1.0) 并添加批次维度
            audio_float = torch.from_numpy(audio_np.astype(np.float32) / 32768.0).unsqueeze(0)
            
            # 如果需要重采样到模型采样率
            if self.resampler_to_target is not None:
                audio_float = self.resampler_to_target(audio_float)
            
            # 使用 DeepFilterNet 进行降噪
            with torch.no_grad():
                enhanced_audio = enhance(self.model, self.df_state, audio_float)
            
            # 如果需要重采样回输入采样率
            if self.resampler_from_target is not None:
                enhanced_audio = self.resampler_from_target(enhanced_audio)
            
            # 转换回 numpy 数组并移除批次维度
            enhanced_np = enhanced_audio.squeeze(0).numpy()
            
            # 转换回 int16 格式
            enhanced_int16 = (enhanced_np * 32767.0).astype(np.int16)
            
            # 防止溢出
            enhanced_int16 = np.clip(enhanced_int16, -32768, 32767)
            
            return enhanced_int16
            
        except Exception as e:
            print(f"❌ DeepFilterNet 处理错误: {e}")
            import traceback
            traceback.print_exc()
            return audio_np

def convert_audio_to_target_format(audio_path, target_channels=CHANNELS, target_sample_width=SAMPLE_WIDTH, target_sample_rate=None, output_path=None):
    """
    将音频文件转换为目标格式（通道数、采样宽度、采样率）
    如果 target_sample_rate 为 None，则不转换采样率，使用原采样率。
    """
    try:
        # 使用 torchaudio 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 检查参数
        current_channels = waveform.shape[0]
        current_sample_width = waveform.dtype.itemsize
        current_sample_rate = sample_rate
        
        # 如果 target_sample_rate 为 None，使用原采样率
        target_sample_rate = current_sample_rate if target_sample_rate is None else target_sample_rate
        
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
            
            # 2. 采样率重采样（仅当 target_sample_rate != current 时）
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

def save_denoised_audio(denoised_audio, input_path, sample_rate):
    """保存降噪音频，使用动态采样率"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_path).replace("_original.wav", "")
    
    # 确保目录存在
    os.makedirs("data/sessions", exist_ok=True)
    
    # denoised_raw_path = f"data/sessions/{base_name}_denoised_deepfilter_{timestamp}.raw"
    denoised_wav_path = f"data/sessions/{base_name}_denoised_deepfilter_{timestamp}.wav"
    
    # with open(denoised_raw_path, "wb") as f:
    #     f.write(denoised_audio.tobytes())
        
    with wave.open(denoised_wav_path, 'wb') as f_wav:
        f_wav.setnchannels(CHANNELS)
        f_wav.setsampwidth(SAMPLE_WIDTH)
        f_wav.setframerate(sample_rate)  # 使用动态采样率
        f_wav.writeframes(denoised_audio.tobytes())
    
    print(f"💾 降噪音频已保存: {denoised_wav_path}")
    return denoised_wav_path

def main():
    if len(sys.argv) < 2:
        print("用法: python audio_denoise.py <input_wav_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    
    # 读取并转换音频（不指定 target_sample_rate，使用原采样率）
    print(f"📥 加载音频文件: {input_path}")
    audio_data, sample_rate = convert_audio_to_target_format(input_path, target_sample_rate=None)
    if audio_data is None or sample_rate is None:
        print("❌ 无法加载或转换音频文件")
        sys.exit(1)
    
    # 初始化处理器，使用动态 sample_rate
    processor = DeepFilterNetProcessor(sample_rate=sample_rate)
    processor.initialize()
    
    # 降噪
    print("🧠 开始 DeepFilterNet 降噪...")
    denoised_audio = processor.process_audio(audio_data)
    
    # 保存，使用动态 sample_rate
    save_denoised_audio(denoised_audio, input_path, sample_rate)

if __name__ == "__main__":
    main()