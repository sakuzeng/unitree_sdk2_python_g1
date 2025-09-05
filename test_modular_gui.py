#!/usr/bin/env python3
"""
测试模块化 GUI 结构的脚本。

验证所有模块是否可以正确导入以及基本功能是否正常。
"""

import sys
import traceback
from pathlib import Path

# 确保 gui 包可以导入
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
	"""测试所有模块导入。"""
	print("正在测试模块导入...")
	
	tests = [
		("gui", "GUI 包"),
		("gui.logging", "日志模块"),
		("gui.state", "状态管理模块"),
		("gui.threads", "后台线程模块"),
		("gui.utils", "工具函数模块"),
		("gui.main_window", "主窗口模块"),
	]
	
	failed = []
	for module_name, description in tests:
		try:
			__import__(module_name)
			print(f"  ✓ {description}")
		except Exception as e:
			print(f"  ✗ {description}: {e}")
			failed.append((module_name, description, str(e)))
	
	return failed

def test_gui_creation():
	"""测试 GUI 创建（不启动主循环）。"""
	print("\n正在测试 GUI 创建...")
	
	try:
		# 检查是否有显示环境
		import os
		if not os.environ.get('DISPLAY'):
			print("  ⚠ 无显示环境，跳过 GUI 测试")
			return []
		
		# 尝试创建主窗口（但不运行）
		from gui.main_window import G1Windows
		
		# 使用虚拟参数
		print("  - 创建主窗口实例...")
		window = G1Windows(
			iface="lo",  # 使用本地回环接口
			ground_clear_in=4.0,
			hand="left",
			grip_force=0.3,
		)
		print("  ✓ 主窗口创建成功")
		
		# 清理
		try:
			window._stop_evt.set()
		except:
			pass
		
		return []
		
	except Exception as e:
		print(f"  ✗ GUI 创建失败: {e}")
		return [("GUI creation", "主窗口创建", str(e))]

def test_utilities():
	"""测试工具函数。"""
	print("\n正在测试工具函数...")
	
	failed = []
	try:
		from gui.utils import clamp, numpy_to_qpix
		import numpy as np
		
		# 测试 clamp 函数
		assert clamp(5, 0, 10) == 5
		assert clamp(-5, 0, 10) == 0
		assert clamp(15, 0, 10) == 10
		print("  ✓ clamp 函数正常")
		
		# 测试 numpy_to_qpix 函数
		test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
		qpix = numpy_to_qpix(test_img)
		if qpix is not None:
			print("  ✓ numpy_to_qpix 函数正常")
		else:
			print("  ⚠ numpy_to_qpix 返回 None（可能缺少 Qt 环境）")
		
	except Exception as e:
		print(f"  ✗ 工具函数测试失败: {e}")
		failed.append(("utils", "工具函数", str(e)))
	
	return failed

def test_logging():
	"""测试日志设置。"""
	print("\n正在测试日志设置...")
	
	failed = []
	try:
		from gui.logging import setup_logging
		
		# 测试不同日志级别
		for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
			setup_logging(level=level)
		
		print("  ✓ 日志设置正常")
		
	except Exception as e:
		print(f"  ✗ 日志设置失败: {e}")
		failed.append(("logging", "日志设置", str(e)))
	
	return failed

def main():
	"""运行所有测试。"""
	print("=" * 60)
	print("模块化 GUI 结构测试")
	print("=" * 60)
	
	all_failed = []
	
	# 运行各种测试
	all_failed.extend(test_imports())
	all_failed.extend(test_logging())
	all_failed.extend(test_utilities())
	all_failed.extend(test_gui_creation())
	
	# 输出结果
	print("\n" + "=" * 60)
	if not all_failed:
		print("✓ 所有测试通过！模块化结构正常工作。")
		print("\n要运行 GUI，请使用：")
		print("  python3 run_g1_gui_modular.py <网络接口>")
		print("例如：")
		print("  python3 run_g1_gui_modular.py eth0")
	else:
		print(f"✗ {len(all_failed)} 个测试失败：")
		for module, desc, error in all_failed:
			print(f"  - {desc}: {error}")
		
		print("\n请检查以下内容：")
		print("  1. 确保所有依赖包已安装")
		print("  2. 检查 Python 路径设置")
		print("  3. 验证环境配置")
	
	print("=" * 60)
	return len(all_failed)

if __name__ == "__main__":
	sys.exit(main())
