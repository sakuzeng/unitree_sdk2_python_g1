#!/bin/bash
# 完整的手臂模型创建工作流程 (智能语音版)

set -e

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <network_interface> <arm> [samples]"
    echo "示例: $0 eth0 left 60"
    exit 1
fi

INTERFACE=$1
ARM=$2
SAMPLES=${3:-60}  # 默认60个样本

echo "=== 手臂模型创建工作流程 (智能语音版) ==="
echo "网络接口: $INTERFACE"
echo "目标手臂: $ARM"
echo "目标样本数: $SAMPLES"

# 1. 检查数据目录
DATA_DIR="data/arms/$ARM"
CSV_FILE="$DATA_DIR/training_data_with_waist.csv"

echo ""
echo "检查现有数据..."

if [ -f "$CSV_FILE" ]; then
    SAMPLE_COUNT=$(tail -n +2 "$CSV_FILE" | wc -l)
    echo "找到已有训练数据: $SAMPLE_COUNT 个样本"
else
    SAMPLE_COUNT=0
    echo "未找到已有数据，将从零开始"
fi

if [ $SAMPLE_COUNT -ge $SAMPLES ]; then
    echo "✅ 已有足够样本 ($SAMPLE_COUNT >= $SAMPLES)，跳过数据收集"
else
    NEED_SAMPLES=$((SAMPLES - SAMPLE_COUNT))
    echo "还需收集 $NEED_SAMPLES 个样本"
    
    echo ""
    echo "🎙️ 启动智能语音数据收集器..."
    echo "请确保:"
    echo "1. 机器人已正确连接"
    echo "2. 机器人处于安全的工作环境"
    echo "3. 音响设备工作正常"
    echo ""
    
    read -p "是否开始自动采集? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 自动采集模式
        python3 arm_training_recorder.py $INTERFACE $ARM $SAMPLES
    else
        echo "请手动运行数据收集:"
        echo "  python3 arm_training_recorder.py $INTERFACE $ARM $SAMPLES"
        exit 1
    fi
fi

# 2. 验证数据质量
if [ -f "$CSV_FILE" ]; then
    FINAL_COUNT=$(tail -n +2 "$CSV_FILE" | wc -l)
    echo ""
    echo "最终数据统计: $FINAL_COUNT 个样本"
    
    if [ $FINAL_COUNT -lt 30 ]; then
        echo "⚠️  警告: 样本数量较少 ($FINAL_COUNT)，建议至少60个样本"
        read -p "是否继续训练? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "❌ 数据收集失败"
    exit 1
fi

# 3. 检查Python依赖
echo ""
echo "检查Python依赖..."
python3 -c "
import pandas, numpy, sklearn, joblib
print('✅ 基础依赖已安装')
try:
    import matplotlib
    print('✅ matplotlib 已安装')
except ImportError:
    print('⚠️  matplotlib 未安装 (可选，用于绘图)')
"

# 4. 训练模型
echo ""
echo "开始训练模型..."
python3 train_arm_model.py $ARM "$CSV_FILE"

# 5. 验证模型
MODEL_FILE="data/artifacts/$ARM-arm/arm_mlp.joblib"
if [ -f "$MODEL_FILE" ]; then
    echo ""
    echo "✅ 模型训练完成!"
    echo "模型文件: $MODEL_FILE"
    echo ""
    echo "测试推理功能..."
    python3 data/inference_arm.py
    echo ""
    echo "🎉 手臂模型已就绪，可以在GUI中使用箭头键进行智能控制!"
    echo ""
    echo "使用方法:"
    echo "1. 启动GUI: python3 run_g1_gui.py $INTERFACE"
    echo "2. 使用箭头键 (↑↓←→) 或 F/B 键控制手臂运动"
else
    echo ""
    echo "❌ 模型创建失败"
    exit 1
fi
