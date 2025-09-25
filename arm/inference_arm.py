#!/usr/bin/env python3
"""
手臂运动推理接口
为GUI提供模型推理功能
"""

import numpy as np
import joblib
from pathlib import Path

def load_bundle(bundle_path=None):
	"""
	加载训练好的模型包
	
	Args:
		bundle_path: 模型文件路径，如果为None则使用默认路径
		
	Returns:
		dict: 包含模型和预处理器的字典
	"""
	if bundle_path is None:
		bundle_path = Path("data/artifacts/left-arm/arm_mlp.joblib")
	
	try:
		bundle = joblib.load(bundle_path)
		
		# 验证必要组件
		required_keys = ['model', 'scaler', 'label_encoder']
		for key in required_keys:
			if key not in bundle:
				raise ValueError(f"模型包缺少必要组件: {key}")
		
		return bundle
		
	except Exception as e:
		print(f"模型加载失败 ({bundle_path}): {e}")
		raise

def predict_end_positions(direction: str, start_joints: list, arm: str, bundle):
	"""
	预测手臂运动的目标关节位置
	
	Args:
		direction: 运动方向 ("up", "down", "left", "right", "forward", "back")
		start_joints: 起始关节角度列表 (包含腰部+手臂关节)
		arm: 手臂名称 ("left" 或 "right")
		bundle: 加载的模型包
		
	Returns:
		list: 预测的目标关节角度
	"""
	try:
		# 获取模型组件
		model = bundle['model']
		scaler = bundle['scaler']
		label_encoder = bundle['label_encoder']
		
		# 验证方向
		if direction not in label_encoder.classes_:
			available_directions = list(label_encoder.classes_)
			raise ValueError(f"未知方向 '{direction}'，可用方向: {available_directions}")
		
		# 编码方向
		direction_encoded = label_encoder.transform([direction])[0]
		
		# 构建特征向量
		features = np.array([start_joints + [direction_encoded]])
		
		# 标准化关节角度特征 (保持方向编码不变)
		joint_features = features[:, :-1]  # 除最后一列
		direction_features = features[:, -1:]  # 方向编码
		
		joint_features_scaled = scaler.transform(joint_features)
		features_scaled = np.column_stack([joint_features_scaled, direction_features])
		
		# 预测
		prediction = model.predict(features_scaled)
		
		return prediction[0].tolist()
		
	except Exception as e:
		print(f"推理失败: {e}")
		raise

def test_inference():
	"""测试推理功能"""
	print("=== 测试手臂运动推理 ===")
	
	# 测试左臂
	try:
		bundle_path = Path("data/artifacts/left-arm/arm_mlp.joblib")
		if bundle_path.exists():
			bundle = load_bundle(bundle_path)
			
			# 示例起始关节角度 (腰部 + 7个手臂关节)
			start_joints = [0.0, 0.2, 0.1, -0.3, 0.7, -0.4, -0.8, 0.0]
			
			# 测试所有方向
			directions = ["up", "down", "left", "right", "forward", "back"]
			
			print("\n左臂推理测试:")
			for direction in directions:
				try:
					result = predict_end_positions(direction, start_joints, "left", bundle)
					print(f"  {direction:8s}: {[f'{x:.3f}' for x in result[:3]]}...")
				except Exception as e:
					print(f"  {direction:8s}: 失败 - {e}")
		else:
			print("左臂模型文件不存在")
			
	except Exception as e:
		print(f"左臂测试失败: {e}")
	
	# 测试右臂
	try:
		bundle_path = Path("data/artifacts/right-arm/arm_mlp.joblib")
		if bundle_path.exists():
			bundle = load_bundle(bundle_path)
			
			# 示例起始关节角度 (腰部 + 7个手臂关节)
			start_joints = [0.0, 0.087, -0.271, 0.323, 0.691, 0.240, -0.771, -0.176]
			
			# 测试所有方向
			directions = ["up", "down", "left", "right", "forward", "back"]
			
			print("\n右臂推理测试:")
			for direction in directions:
				try:
					result = predict_end_positions(direction, start_joints, "right", bundle)
					print(f"  {direction:8s}: {[f'{x:.3f}' for x in result[:3]]}...")
				except Exception as e:
					print(f"  {direction:8s}: 失败 - {e}")
		else:
			print("右臂模型文件不存在")
			
	except Exception as e:
		print(f"右臂测试失败: {e}")

if __name__ == "__main__":
	test_inference()
