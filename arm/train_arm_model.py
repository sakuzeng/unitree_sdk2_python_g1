#!/usr/bin/env python3
"""
手臂运动模型训练脚本
从收集的数据训练MLP回归模型
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 不可用，跳过绘图功能")

# 常量定义
HIDDEN_LAYER_SIZES = (64, 32)  # 从图片中的配置
RANDOM_STATE = 42

class ArmMovementTrainer:
    def __init__(self, arm: str = "left"):
        """
        初始化训练器
        
        Args:
            arm: 要训练的手臂 ("left" 或 "right")
        """
        self.arm = arm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 使用图片中的模型配置
        self.model_config = {
            'hidden_layer_sizes': HIDDEN_LAYER_SIZES,
            'activation': 'relu',
            'solver': 'adam',
            'learning_rate_init': 1e-3,
            'max_iter': 1,  # 我们将使用迭代式训练
            'warm_start': True,
            'random_state': RANDOM_STATE,
            'early_stopping': False,  # 关闭早停，使用手动迭代控制
        }
        
        # 迭代式训练参数
        self.max_training_iterations = 1000
        self.tolerance = 1e-6
        self.patience = 50

    def load_data(self, csv_path: str) -> bool:
        """
        加载训练数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            self.df = pd.read_csv(csv_path)
            print(f"已加载数据: {len(self.df)} 个样本")
            
            # 数据验证
            required_cols = ['direction', 'arm']
            if not all(col in self.df.columns for col in required_cols):
                raise ValueError(f"缺少必要列: {required_cols}")
            
            # 过滤指定手臂的数据
            self.df = self.df[self.df['arm'] == self.arm]
            print(f"过滤后 {self.arm} 手臂数据: {len(self.df)} 个样本")
            
            if len(self.df) < 10:
                raise ValueError(f"数据量太少 ({len(self.df)}),建议至少50个样本")
            
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def preprocess_data(self):
        """数据预处理"""
        print("\n开始数据预处理...")
        
        # 1. 数据清理
        self._clean_data()
        
        # 2. 特征工程
        self._extract_features()
        
        # 3. 数据标准化
        self._normalize_features()
        
        print(f"预处理完成 - 特征维度: {self.X.shape}, 标签维度: {self.y.shape}")

    def _clean_data(self):
        """清理异常数据"""
        initial_count = len(self.df)
        
        # 移除重复记录
        self.df = self.df.drop_duplicates()
        
        # 检查关节角度范围（合理的关节限制）
        joint_cols = [col for col in self.df.columns if col.startswith(('start_', 'end_'))]
        
        for col in joint_cols:
            # 移除极端异常值
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            self.df = self.df[
                (self.df[col] >= lower_bound) & 
                (self.df[col] <= upper_bound)
            ]
        
        print(f"数据清理: {initial_count} -> {len(self.df)} 个样本")

    def _extract_features(self):
        """提取特征和标签"""
        # 获取起始和结束关节列
        start_cols = sorted([col for col in self.df.columns if col.startswith('start_')])
        end_cols = sorted([col for col in self.df.columns if col.startswith('end_')])
        
        if len(start_cols) != len(end_cols):
            raise ValueError(f"起始和结束关节数量不匹配: {len(start_cols)} vs {len(end_cols)}")
        
        # 方向编码
        directions = self.df['direction'].values
        direction_encoded = self.label_encoder.fit_transform(directions)
        
        # 构建特征矩阵 (起始关节角度 + 方向编码)
        start_features = self.df[start_cols].values
        self.X = np.column_stack([start_features, direction_encoded])
        
        # 构建标签矩阵 (结束关节角度)
        self.y = self.df[end_cols].values
        
        # 保存特征和标签列名
        self.start_cols = start_cols
        self.end_cols = end_cols
        
        print(f"方向编码映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

    def _normalize_features(self):
        """特征标准化"""
        # 只标准化关节角度特征，保持方向编码不变
        joint_features = self.X[:, :-1]  # 所有列除了最后的方向编码
        direction_features = self.X[:, -1:]  # 方向编码列
        
        # 标准化关节角度
        joint_features_scaled = self.scaler.fit_transform(joint_features)
        
        # 重新组合特征
        self.X = np.column_stack([joint_features_scaled, direction_features])

    def train_model(self, test_size: float = 0.2):
        """
        使用迭代式训练模型（按图片中的配置）
        
        Args:
            test_size: 测试集比例
        """
        print(f"\n开始迭代式训练模型...")
        print(f"模型配置:")
        print(f"  隐藏层尺寸: {HIDDEN_LAYER_SIZES}")
        print(f"  激活函数: relu")
        print(f"  求解器: adam")
        print(f"  学习率: 1e-3")
        print(f"  热启动: True")
        print(f"  随机种子: {RANDOM_STATE}")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=RANDOM_STATE, stratify=self.X[:, -1]
        )
        
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"测试集: {X_test.shape[0]} 样本")
        
        # 创建模型
        self.model = MLPRegressor(**self.model_config)
        
        # 迭代式训练
        best_loss = float('inf')
        patience_counter = 0
        loss_history = []
        
        print("\n开始迭代式训练...")
        for iteration in range(self.max_training_iterations):
            # 训练一次迭代
            self.model.fit(X_train, y_train)
            
            # 计算当前损失
            train_pred = self.model.predict(X_train)
            current_loss = mean_squared_error(y_train, train_pred)
            loss_history.append(current_loss)
            
            # 每10次迭代打印一次进度
            if (iteration + 1) % 10 == 0:
                test_pred = self.model.predict(X_test)
                test_loss = mean_squared_error(y_test, test_pred)
                print(f"迭代 {iteration + 1:3d}: 训练损失 {current_loss:.6f}, 测试损失 {test_loss:.6f}")
            
            # 早停检查
            if current_loss < best_loss - self.tolerance:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"早停: {self.patience} 次迭代无改善")
                break
        
        # 最终评估
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\n=== 训练结果 ===")
        print(f"总迭代次数: {iteration + 1}")
        print(f"训练集 MSE: {train_mse:.6f}")
        print(f"测试集 MSE: {test_mse:.6f}")
        print(f"训练集 R²: {train_r2:.6f}")
        print(f"测试集 R²: {test_r2:.6f}")
        
        # 检查过拟合
        if test_mse > train_mse * 2:
            print("⚠️  警告: 可能存在过拟合!")
        
        if test_r2 < 0.7:
            print("⚠️  警告: 模型性能较低，建议收集更多数据或调整模型参数")
        
        # 保存损失历史
        self.loss_history = loss_history
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'iterations': iteration + 1,
            'loss_history': loss_history
        }

    def save_model(self, output_dir: str = None):
        """
        保存训练好的模型
        
        Args:
            output_dir: 输出目录路径
        """
        if self.model is None:
            print("错误: 模型尚未训练")
            return False
        
        # 默认输出路径
        if output_dir is None:
            output_dir = f"data/artifacts/{self.arm}-arm"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据包
        model_bundle = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'start_cols': self.start_cols,
            'end_cols': self.end_cols,
            'arm': self.arm,
            'config': self.model_config,
            'loss_history': getattr(self, 'loss_history', [])
        }
        
        # 保存模型
        model_path = output_path / "arm_mlp.joblib"
        joblib.dump(model_bundle, model_path)
        
        print(f"✅ 模型已保存到: {model_path}")
        return True

    def plot_training_stats(self):
        """绘制训练统计图"""
        if not MATPLOTLIB_AVAILABLE:
            print("跳过绘图: matplotlib 不可用")
            return
            
        if hasattr(self.df, 'direction'):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # 子图1: 方向分布
            direction_counts = self.df['direction'].value_counts()
            axes[0].bar(direction_counts.index, direction_counts.values)
            axes[0].set_title(f'{self.arm.title()} 手臂 - 方向数据分布')
            axes[0].set_xlabel('方向')
            axes[0].set_ylabel('样本数量')
            axes[0].tick_params(axis='x', rotation=45)
            
            # 子图2: 训练损失曲线
            if hasattr(self, 'loss_history') and self.loss_history:
                axes[1].plot(self.loss_history)
                axes[1].set_title('迭代式训练损失曲线')
                axes[1].set_xlabel('迭代次数')
                axes[1].set_ylabel('MSE 损失')
                axes[1].grid(True)
                axes[1].set_yscale('log')  # 使用对数刻度更好地显示损失变化
            
            # 子图3: 模型架构可视化
            layer_sizes = [self.X.shape[1]] + list(HIDDEN_LAYER_SIZES) + [self.y.shape[1]]
            layer_positions = np.arange(len(layer_sizes))
            
            axes[2].bar(layer_positions, layer_sizes)
            axes[2].set_title('MLP 模型架构')
            axes[2].set_xlabel('网络层')
            axes[2].set_ylabel('神经元数量')
            axes[2].set_xticks(layer_positions)
            axes[2].set_xticklabels(['输入层'] + [f'隐藏层{i+1}' for i in range(len(HIDDEN_LAYER_SIZES))] + ['输出层'])
            axes[2].tick_params(axis='x', rotation=45)
            
            # 添加架构参数文本
            arch_text = f"配置参数:\n"
            arch_text += f"隐藏层: {HIDDEN_LAYER_SIZES}\n"
            arch_text += f"激活函数: ReLU\n"
            arch_text += f"求解器: Adam\n"
            arch_text += f"学习率: 1e-3\n"
            arch_text += f"热启动: True"
            axes[2].text(0.02, 0.98, arch_text, transform=axes[2].transAxes, 
                        verticalalignment='top', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            plt.tight_layout()
            
            # 保存图表
            plot_dir = Path(f"data/artifacts/{self.arm}-arm")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_dir / "training_stats.png", dpi=150, bbox_inches='tight')
            print(f"训练统计图已保存到: {plot_dir / 'training_stats.png'}")
            
            try:
                plt.show()
            except:
                print("无法显示图表 (可能是无图形环境)")

def main():
    """主训练流程"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python3 train_arm_model.py <arm> [csv_path]")
        print("示例: python3 train_arm_model.py left")
        print("      python3 train_arm_model.py right data/arms/right/training_data_with_waist.csv")
        print("\n使用图片中的模型配置:")
        print(f"  - 隐藏层尺寸: {HIDDEN_LAYER_SIZES}")
        print(f"  - 激活函数: ReLU")
        print(f"  - 求解器: Adam")
        print(f"  - 学习率: 1e-3")
        print(f"  - 热启动: True")
        print(f"  - 迭代式训练: 是")
        return
    
    arm = sys.argv[1]
    if arm not in ["left", "right"]:
        print("手臂参数必须是 'left' 或 'right'")
        return
    
    # 确定数据文件路径
    if len(sys.argv) > 2:
        csv_path = sys.argv[2]
    else:
        csv_path = f"data/arms/{arm}/training_data_with_waist.csv"
    
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"❌ 数据文件不存在: {csv_path}")
        print("请先运行 arm_training_recorder.py 收集训练数据")
        return
    
    print(f"=== 训练 {arm.upper()} 手臂运动模型 ===")
    print(f"使用图片中的模型配置 - 迭代式训练")
    
    # 创建训练器
    trainer = ArmMovementTrainer(arm)
    
    # 加载数据
    if not trainer.load_data(csv_path):
        return
    
    # 数据预处理
    trainer.preprocess_data()
    
    # 训练模型
    results = trainer.train_model()
    
    # 保存模型
    if trainer.save_model():
        print(f"\n✅ {arm} 手臂模型训练完成!")
        print(f"模型文件: data/artifacts/{arm}-arm/arm_mlp.joblib")
        print(f"训练迭代: {results['iterations']} 次")
    
    # 绘制统计图
    try:
        trainer.plot_training_stats()
    except Exception as e:
        print(f"绘图失败: {e}")

if __name__ == "__main__":
    main()
