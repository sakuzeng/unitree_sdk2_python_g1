"""GUI 后台工作线程 (电池, SLAM)。"""

import sys
import threading
import time

import numpy as np

from gui.state import _slam_latest, _slam_lock, _state, _state_lock


def rx_battery(stop: "threading.Event", iface: str):
	"""在后台工作，将最新的电池百分比保存在共享的 _state 中。"""

	try:
		from unitree_sdk2_python.core.channel import ChannelSubscriber, ChannelFactoryInitialize

		# 将 SOC 写入共享状态的帮助函数
		def _publish(soc_val: int | None = None, voltage: float | None = None):
			with _state_lock:
				if soc_val is not None:
					_state["soc"] = soc_val
				if voltage is not None:
					_state["voltage"] = voltage

		def _attempt_sub(name: str, msg_type, cb):
			try:
				sub = ChannelSubscriber(name, msg_type)
				sub.Init(cb, 50)
				return True
			except Exception:
				return False

		# 1) Unitree Go/G1 – LowState
		ok = False
		try:
			from unitree_sdk2_python.idl.unitree_go.msg.dds_ import LowState_

			def _cb_go(msg: LowState_):
				soc_val = getattr(getattr(msg, 'bms_state', None), 'soc', None)
				if soc_val is not None and soc_val > 0:
					_publish(int(soc_val))
				else:
					_publish(voltage=float(msg.power_v))

			ok = _attempt_sub("rt/lowstate", LowState_, _cb_go)
		except Exception:
			ok = False

		# 2) 人形机器人 HG – BmsState 主题
		if not ok:
			try:
				from unitree_sdk2_python.idl.unitree_hg.msg.dds_ import BmsState_

				def _cb_hg(msg: BmsState_):
					_publish(int(msg.soc))

				ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)
			except Exception:
				ok = False

		# 如果两者都失败，可能是工厂未初始化 – 执行初始化并重试
		if not ok:
			try:
				ChannelFactoryInitialize(0, iface)
			except Exception:
				pass  # 如果早先初始化失败，这里仍可能失败

			if not ok:
				# 再次重试两个订阅
				ok = _attempt_sub("rt/lowstate", LowState_, _cb_go) if 'LowState_' in locals() else False
				if not ok and 'BmsState_' in locals():
					ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)

		if not ok:
			raise RuntimeError("无法订阅任何电池 SOC 主题")

		# 空闲 – 回调已处理更新
		while not stop.is_set():
			time.sleep(0.5)

	except Exception as exc:
		print("[gui.threads] 电池监视器已禁用:", exc, file=sys.stderr)


def _patch_live_slam_for_pyqt() -> None:
	"""猴子补丁 live_slam._Viewer，使其不再打开 GLFW 窗口。"""

	class _QtViewer:
		def __init__(self):
			self._latest_pts: np.ndarray | None = None
			self._latest_pose: np.ndarray | None = None

		# -------- 从 SLAM 线程调用 --------------------------------
		def push(self, xyz: np.ndarray, pose: np.ndarray):
			global _slam_latest
			with _slam_lock:
				_slam_latest = (xyz, pose)

		# -------- tick() 签名保留以兼容 -----------------------
		def tick(self) -> bool:
			"""tick 方法，保持 SLAM 主循环存活。"""
			# 无事可做 – 返回 True 以保持 SLAM 主循环存活。
			return True

		def close(self):
			pass

	import live_slam as _ls

	_ls._Viewer = _QtViewer

	# ------------------------------------------------------------------
	#  安全补丁 – 使 LiveSLAMDemo 对偶尔的 KISS-ICP
	#  初始化问题具有鲁棒性。我们包装其 handle_points()，以便捕获
	#  原始实现中的*任何*异常（例如由于第一帧扫描时
	#  未初始化的位姿），并且我们仍然将原始 xyz 转发给查看器。
	#  这保证了 Qt GUI 总是能收到一些东西来显示，因此永远不会
	#  保持空白。
	# ------------------------------------------------------------------

	try:
		_orig_hp = _ls.LiveSLAMDemo.handle_points

		def _safe_hp(self, xyz):
			try:
				_orig_hp(self, xyz)
			except Exception as exc:
				# 推送没有有效位姿的原始点。GL 散点图仍然
				# 渲染；位姿轴只是保持缺失，直到 KISS-ICP
				# 恢复。
				try:
					self._viewer.push(xyz, None)
				except Exception:
					pass
				print("[gui.threads] KISS-ICP 第一帧失败:", exc)

		_ls.LiveSLAMDemo.handle_points = _safe_hp
	except Exception:
		pass


def run_slam(stop_evt: threading.Event):
	"""运行 Livox SLAM 管道的后台工作线程。

	我们让*驱动程序*在其自己的 ``spin()`` 方法内阻塞，以便 SDK
	线程可以不间断地推送点云帧。一旦 Qt
	应用程序请求关闭（设置了 ``stop_evt``），我们将优雅地
	拆卸所有东西。
	"""

	try:
		_patch_live_slam_for_pyqt()

		import live_slam as _ls

		demo = _ls.LiveSLAMDemo()

		# ------------------------------------------------------------------
		# 运行 SDK 循环 – 通过提供的 .spin() 帮助程序 (SDK2) 或
		# 当包装器不公开它时的简单睡眠循环回退。
		# ------------------------------------------------------------------

		spin_fn = getattr(demo, "spin", None)

		# 在另一个守护线程中启动 SDK spin-loop（如果存在），
		# 以便我们仍然可以监视 *stop_evt* 并自己调用 ``shutdown``。
		if callable(spin_fn):
			t_spin = threading.Thread(target=spin_fn, daemon=True)
			t_spin.start()

		try:
			while not stop_evt.is_set():
				# 尽管我们的自定义 _QtViewer 不需要调用其 tick()
				# 方法（它只是返回 True），但原始的
				# LiveSLAMDemo 主循环期望能够在该函数内
				# 执行一些周期性的内务处理。在这里调用它
				# 重新建立了与上游脚本的完全行为对等，
				# 并且 – 至关重要的是 – 确保未来版本添加的任何
				# 副作用仍将运行。

				try:
					demo._viewer.tick()
				except Exception:
					pass

				time.sleep(0.05)
		finally:
			try:
				demo.shutdown()
			except Exception:
				pass

	except Exception as exc:
		print("[gui.threads] SLAM 线程已禁用:", exc, file=sys.stderr)
