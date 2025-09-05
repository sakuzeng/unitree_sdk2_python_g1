"""GUI 主窗口类。"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Tuple

# Qt 导入必须在 *类* 定义时可用，因为我们现在
# 从 QtCore.QObject 派生 G1Windows，使其可以作为全局
# 事件过滤器。

try:
	from PySide6 import QtCore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – missing optional dep.
	print("需要 PySide6 – 通过 'pip install pyside6 pyqtgraph' 安装")
	raise SystemExit(
		"run_g1_gui.py 需要 PySide6 – 请使用以下命令安装:\n"
		"    pip install pyside6 pyqtgraph"
	) from exc

# 导入其他模块
import numpy as np
from run_g1_stack import _rx_realsense_local
from gui.state import _state, _state_lock, _slam_latest, _slam_lock
from gui.threads import run_slam, rx_battery
from gui.utils import numpy_to_qpix, clamp


class G1Windows(QtCore.QObject):  # type: ignore[misc]
	def __init__(
		self,
		iface: str,
		ground_clear_in: float,
		*,
		hand: str = "left",
		grip_force: float | None = None,
	):
		"""
		创建主 GUI 窗口。

		Args:
			iface (str): 连接到机器人的网络接口。
			ground_clear_in (float): 在检测到的地平面上方的间隙（以**英寸**为单位），
				超过该间隙的点被视为障碍物（转发给 SLAM 障碍物过滤器）。
			hand (str): 哪个 Dex3 手物理连接到机器人。默认为 ``"left"`` 以保留原始行为。
			grip_force (float | None): 在连续*抓取*模式下应用的可选前馈扭矩（约 **N·m**）。
		"""

		super().__init__()

		from PySide6 import QtWidgets, QtGui  # type: ignore

		# 在检测到的地平面上方，一个点被视为障碍物之前的间隙（米）。
		self._clear_m = ground_clear_in * 0.0254  # 英寸 → 米

		# 存储 CLI 抓取力，以便构造函数的下半部分
		# 即使在特定手部控制属性初始化之前也能读取它。
		self._cli_grip_force = grip_force if grip_force is not None else 0.3
		import pyqtgraph.opengl as gl  # type: ignore

		self.app = QtWidgets.QApplication(sys.argv)

		# ------------------------------------------------------------------
		#  使 Ctrl-C (SIGINT) 立即关闭应用程序。
		# ------------------------------------------------------------------
		import signal

		try:
			signal.signal(signal.SIGINT, lambda *_: self.app.quit())
		except Exception:
			pass

		# 初始化 UI 组件
		self._init_ui_components()
		
		# 初始化遥控状态
		self._init_remote_control()
		
		# 初始化机器人控制
		self._init_robot_control(iface)
		
		# 初始化手臂控制
		self._init_arm_control(iface)
		
		# 初始化手部控制
		self._init_hand_control(hand, iface)
		
		# 安装为全局事件过滤器
		self.app.installEventFilter(self)

		# 启动后台工作线程
		self._init_background_threads(iface)

		# 优雅退出
		self.app.aboutToQuit.connect(self._on_quit)

		# 为初始的*左*默认值最终确定手臂特定的帮助程序。
		try:
			self._configure_arm_variables()
		except Exception as exc:
			print("[run_g1_gui] 初始手臂配置失败:", exc, file=sys.stderr)

	def _init_ui_components(self):
		"""初始化 UI 组件。"""
		from PySide6 import QtWidgets
		import pyqtgraph as pg
		import pyqtgraph.opengl as gl

		# ---------------- 主控件 ----------------------------------
		self.rgb_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
		self.depth_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)

		# 不裁剪传入的 640×480 RealSense 流 – 显示完整图像（带黑边）
		self.rgb_lbl.setMinimumSize(640, 320)
		self.depth_lbl.setMinimumSize(640, 320)

		# 明确设置黑色背景
		for _lbl in (self.rgb_lbl, self.depth_lbl):
			_lbl.setStyleSheet("background-color: black")

		# 2D 占据栅格预览
		self.map_view = pg.GraphicsLayoutWidget()
		self.map_view.setMinimumSize(640, 320)

		# 使用专用的 ViewBox
		self._map_vb = self.map_view.addViewBox(lockAspect=True, enableMouse=True)
		self._map_vb.setMenuEnabled(False)
		self._map_vb.invertY(True)  # 匹配常规图像坐标

		# ImageItem 将在每一帧中使用由 _update_2d_map() 生成的新渲染的占据栅格画布进行更新
		self._map_img = pg.ImageItem()
		self._map_vb.addItem(self._map_img)

		# ------------------------------------------------------------------
		#  路径规划状态
		# ------------------------------------------------------------------

		# 最新的二进制占据栅格（True = 障碍物），图像坐标。
		self._occ_map: "np.ndarray | None" = None  # type: ignore[name-defined]

		# 元数据 (min_x, min_y, scale)，用于在世界坐标 ↔ 图像像素之间映射。
		self._map_meta: tuple[float, float, float] | None = None

		# 上次规划的路径，作为像素位置 (x, y) 的列表 – 图像坐标。
		self._path_px: list[tuple[int, int]] | None = None

		# 将场景图上的鼠标点击转发到我们的处理器
		self.map_view.scene().sigMouseClicked.connect(self._on_map_click)

		# 用于点云的 GL 查看器
		self.gl_view = gl.GLViewWidget()
		# 从稍远一点的地方开始，以便整个地图都能在视图中显示。
		self.gl_view.opts["distance"] = 30
		self.gl_view.setCameraPosition(distance=30, elevation=20, azimuth=45)
		# 确保 GL 面板以合理的宽度启动
		self.gl_view.setMinimumWidth(640)

		# 散点图项 – 增量更新
		self._scatter = gl.GLScatterPlotItem()
		self.gl_view.addItem(self._scatter)

		# 当前持有代表机器人位姿的 3 个彩色轴线的列表。
		self._pose_items: list[gl.GLLinePlotItem] = []

		# -------- 布局 -----------------------------------------------
		splitter = QtWidgets.QSplitter()
		left = QtWidgets.QWidget()
		v = QtWidgets.QVBoxLayout(left)
		v.addWidget(self.rgb_lbl)
		v.addWidget(self.depth_lbl)
		v.addWidget(self.map_view)
		splitter.addWidget(left)
		splitter.addWidget(self.gl_view)
		splitter.setStretchFactor(1, 2)
		# 初始化分割器大小 (左, 右)
		splitter.setSizes([640, 640])

		self.win = QtWidgets.QMainWindow()
		self.win.setWindowTitle("G1-Stack")
		self.win.setCentralWidget(splitter)

		# 给主窗口一个初始大小
		self.win.resize(1600, 760)

		self.status = QtWidgets.QLabel()
		self.win.statusBar().addWidget(self.status)

		# ------------------------------------------------------------------
		#  用户控件 – 卸力手臂 / 腰部回中
		# ------------------------------------------------------------------

		self._btn_damp = QtWidgets.QPushButton("卸力手臂 & 腰部回中")
		self._btn_damp.setToolTip("将上半身切换到被动模式，并将腰部设置为 0 rad")
		self._btn_damp.clicked.connect(self._on_damp_pressed)  # type: ignore[arg-type]
		self.win.statusBar().addPermanentWidget(self._btn_damp)

		# ------------------------------------------------------------------
		#  手臂选择器 – 左 / 右
		# ------------------------------------------------------------------

		self._arm_selector = QtWidgets.QComboBox()
		self._arm_selector.addItems(["左臂", "右臂"])
		self._arm_selector.setCurrentIndex(0)  # 默认 → 左臂
		self._arm_selector.setToolTip("选择要控制和运行推理的手臂")

		# 保留一个易于访问的文本标志
		self._active_arm: str = "left"

		def _on_sel_changed(idx: int):
			self._active_arm = "left" if idx == 0 else "right"
			try:
				self._configure_arm_variables()
			except Exception as exc:
				print("[run_g1_gui] 重新配置手臂失败:", exc, file=sys.stderr)

		self._arm_selector.currentIndexChanged.connect(_on_sel_changed)  # type: ignore[arg-type]
		self.win.statusBar().addPermanentWidget(self._arm_selector)

		# 初始化按键覆盖层
		self._init_key_overlay()

		# 初始化定时器
		self._refresh = QtCore.QTimer()
		self._refresh.setInterval(30)  # ms
		self._refresh.timeout.connect(self._on_tick)
		self._refresh.start()

	def _init_key_overlay(self):
		"""初始化按键视觉反馈覆盖层。"""
		from PySide6 import QtWidgets

		# 左上角的一个小型半透明覆盖层
		self._key_overlay = QtWidgets.QWidget(self.gl_view)
		self._key_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
		self._key_overlay.move(10, 10)

		# 容器样式提供通用的半透明背景。
		self._key_overlay.setStyleSheet(
			"background-color: rgba(0, 0, 0, 150);"
			"border-radius: 6px;"
		)

		_lay = QtWidgets.QVBoxLayout(self._key_overlay)
		_lay.setContentsMargins(8, 6, 8, 6)
		_lay.setSpacing(0)

		# 标题 – 始终保持完全不透明。
		self._header_lbl = QtWidgets.QLabel("按键输入:")
		self._header_lbl.setStyleSheet(
			"color: #ffff00; font: 12pt 'Consolas', 'Monaco', 'Courier New', monospace;"
		)
		_lay.addWidget(self._header_lbl)

		# 动态按键列表标签。
		self._keys_lbl = QtWidgets.QLabel("–")
		self._keys_lbl.setStyleSheet(
			"color: #ffff00; font: bold 24pt 'Consolas', 'Monaco', 'Courier New', monospace;"
		)
		_lay.addWidget(self._keys_lbl)

		self._key_overlay.adjustSize()
		self._key_overlay.show()

		# 仅应用于按键标签的透明度效果和淡出动画。
		self._keys_opacity = QtWidgets.QGraphicsOpacityEffect(self._keys_lbl)
		self._keys_lbl.setGraphicsEffect(self._keys_opacity)

		self._fade_anim = QtCore.QPropertyAnimation(self._keys_opacity, b"opacity", self)
		self._fade_anim.setDuration(600)  # ms

		def _on_fade_finished():
			# 为下一个周期重置。
			self._keys_opacity.setOpacity(1.0)
			self._keys_lbl.setText("–")
			self._key_overlay.adjustSize()

		self._fade_anim.finished.connect(_on_fade_finished)  # type: ignore[arg-type]

	def _init_remote_control(self):
		"""初始化遥控操作状态。"""
		self._stop_evt = threading.Event()

		# 按下的键集合，保存 Qt.Key 枚举 / 小写字符
		self._pressed: set[object] = set()

		# 将发送给机器人的当前目标速度
		self._vx = 0.0
		self._vy = 0.0
		self._omega = 0.0

		# 跟踪当前平衡模式
		self._bal_mode: int = -1

	def _init_robot_control(self, iface: str):
		"""初始化机器人控制。"""
		# 尝试启动 Unitree G-1
		try:
			from hanger_boot_sequence import hanger_boot_sequence  # type: ignore
			self._bot = hanger_boot_sequence(iface=iface)
		except Exception as exc:
			print("[run_g1_gui] 遥控操作已禁用:", exc, file=sys.stderr)
			self._bot = None

		# 以 10 Hz 更新速度并发送 Move 的定时器
		self._drive_timer = QtCore.QTimer()
		self._drive_timer.setInterval(100)  # ms  (10 Hz)
		self._drive_timer.timeout.connect(self._on_drive_tick)
		self._drive_timer.start()

	def _init_arm_control(self, iface: str):
		"""初始化手臂控制。"""
		self._arm_pub = None  # type: ignore[assignment]
		try:
			from unitree_sdk2_python.core.channel import ChannelPublisher
			from unitree_sdk2_python.idl.unitree_hg.msg.dds_ import LowCmd_
			from unitree_sdk2_python.idl.default import unitree_hg_msg_dds__LowCmd_
			from unitree_sdk2_python.utils.crc import CRC

			# 关节索引定义
			_WAIST_YAW_IDX = 12
			_LEFT_IDX = {idx: 0 for idx in range(15, 22)}
			_RIGHT_IDX = {idx: 0 for idx in range(22, 29)}
			_ARM_IDX = _LEFT_IDX if self._active_arm == "left" else _RIGHT_IDX

			# 存储帮助程序的索引
			self._arm_joint_idx: list[int] = list(_ARM_IDX.keys())
			self._waist_idx: int = _WAIST_YAW_IDX
			_NOT_USED_IDX = 29

			self._crc = CRC()

			# 持久完整消息
			self._arm_cmd = unitree_hg_msg_dds__LowCmd_()
			self._arm_cmd.motor_cmd[_NOT_USED_IDX].q = 1

			# 目标位姿序列
			self._init_arm_pose_sequences(_WAIST_YAW_IDX)

			# 每关节命令状态
			self._cmd_q: dict[int, float] = {idx: 0.0 for idx in _ARM_IDX}
			self._cmd_q[_WAIST_YAW_IDX] = 0.0

			# 实时反馈
			self._joint_cur: dict[int, float] = {}
			self._ls_sub = None

			# 延迟的 LowState 订阅
			self._init_lowstate_subscription()

			# 序列进度跟踪器
			self._initialised_from_state = False
			self._seq_idx = 0
			self._SEQ_EPS = 0.01
			self._STEP = 0.02

			# DDS 发布者
			self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
			self._arm_pub.Init()

			# 50 Hz 的定时器，应用斜坡
			self._arm_timer = QtCore.QTimer()
			self._arm_timer.setInterval(20)  # ms
			self._arm_timer.timeout.connect(self._on_arm_tick)
			self._arm_timer.start()

			# 启动时准备两个手臂
			self._prepare_both_arms(_WAIST_YAW_IDX)

		except Exception as exc:
			print("[run_g1_gui] 手臂控制已禁用:", exc, file=sys.stderr)

	def _init_arm_pose_sequences(self, waist_idx: int):
		"""初始化手臂位姿序列。"""
		if self._active_arm == "right":
			# 两步启动序列
			self._pose_seq: list[list[tuple[int, float]]] = [
				[
					(waist_idx, 0.0),
					(22, -0.023), (23, -0.225), (24, +0.502), (25, +1.317),
					(26, +0.185), (27, +0.125), (28, -0.182),
				],
				[
					(waist_idx, 0.0),
					(22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
					(26, +0.240), (27, -0.771), (28, -0.176),
				],
			]
		else:
			# 左臂的单步初始位姿
			self._pose_seq = [
				[
					(waist_idx, 0.0),
					(15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
					(19, -0.379), (20, -0.852), (21, -0.019),
				]
			]

	def _init_lowstate_subscription(self):
		"""初始化 LowState 订阅。"""
		def _init_ls_sub():
			from unitree_sdk2_python.core.channel import ChannelSubscriber

			_candidates = [
				"unitree_sdk2_python.idl.unitree_hg.msg.dds_.LowState_",
				"unitree_sdk2_python.idl.unitree_go.msg.dds_.LowState_",
			]

			for dotted in _candidates:
				try:
					mod_path, cls_name = dotted.rsplit(".", 1)
					mod = __import__(mod_path, fromlist=[cls_name])
					LowState_ = getattr(mod, cls_name)

					def _ls_cb(msg):
						for j_idx in (*self._arm_joint_idx, self._waist_idx):
							try:
								self._joint_cur[j_idx] = msg.motor_state[j_idx].q
							except Exception:
								pass

					sub = ChannelSubscriber("rt/lowstate", LowState_)
					sub.Init(_ls_cb, 200)
					self._ls_sub = sub
					return
				except Exception:
					continue

		threading.Thread(target=_init_ls_sub, daemon=True).start()

	def _prepare_both_arms(self, waist_idx: int):
		"""启动时准备两个手臂。"""
		def _apply_pose_once(pose: list[tuple[int, float]]):
			for j_idx, q_val in pose:
				mc = self._arm_cmd.motor_cmd[j_idx]
				mc.q = q_val
				mc.dq = 0.0
				mc.tau = 0.0
				mc.kp = 60.0
				mc.kd = 1.5

		# 为另一只手臂构建准备位姿
		if self._active_arm == "left":
			_ready_other = [
				(22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
				(26, +0.240), (27, -0.771), (28, -0.176),
			]
		else:
			_ready_other = [
				(15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
				(19, -0.379), (20, -0.852), (21, -0.019),
			]

		_apply_pose_once(_ready_other)

		# 重新计算 CRC 并立即传输
		self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
		try:
			self._arm_pub.Write(self._arm_cmd)
		except Exception:
			pass

	def _init_hand_control(self, hand: str, iface: str):
		"""初始化手部控制。"""
		self._dex3 = None
		try:
			from unitree_sdk2_python.dex3 import Dex3Client
			import csv
			from pathlib import Path

			# 尝试连接到请求的手
			try:
				self._dex3 = Dex3Client(hand=hand, interface=iface)
			except Exception as exc:
				print(f"[run_g1_gui] 在接口 '{iface}' 上的 Dex3 连接失败:", exc, file=sys.stderr)
				try:
					self._dex3 = Dex3Client(hand=hand, interface=None)
					print("[run_g1_gui] Dex3 通过自动检测的 NIC 连接。")
				except Exception as exc2:
					print("[run_g1_gui] Dex3 自动检测失败:", exc2, file=sys.stderr)
					self._dex3 = None

			# 加载可选的手部姿势 CSV
			self._hand_poses: dict[str, list[float]] = {}
			try:
				csv_path = Path("data/hand_states.csv")
				if csv_path.exists():
					with csv_path.open("r", newline="") as fp:
						rdr = csv.DictReader(fp)
						for row in rdr:
							label = row.get("label")
							if label:
								try:
									vals = [float(row[f"joint{i}"]) for i in range(7)]
									self._hand_poses[label.lower()] = vals
								except Exception:
									pass
			except Exception as exc:
				print("[run_g1_gui] 无法加载 hand_states.csv:", exc, file=sys.stderr)

			# 手部运动斜坡帮助程序
			self._init_hand_motion_helpers()

		except Exception as exc:
			print("[run_g1_gui] Dex3 手部控制已禁用:", exc, file=sys.stderr)
			self._dex3 = None

	def _init_hand_motion_helpers(self):
		"""初始化手部运动帮助程序。"""
		import logging

		self._hand_cmd_q: list[float] = [0.0] * 7
		self._hand_pose_seq: list[list[float]] = []
		self._hand_seq_idx: int = 0
		self._HAND_STEP = 0.1

		self._hand_timer = QtCore.QTimer()
		self._hand_timer.setInterval(20)  # ms – 50 Hz
		self._hand_timer.timeout.connect(self._on_hand_tick)
		self._hand_timer.start()

		# 简化的打开/关闭关键姿势
		self._simple_open_pose = [
			-0.15717165172100067, -0.41322529315948486, 0.02846403606235981,
			0.17782948911190033, -0.025226416066288948, 0.17983606457710266,
			-0.027690349146723747,
		]

		self._simple_closed_pose = [
			0.07452802360057831, 0.9478388428688049, 1.766921877861023,
			-1.4442411661148071, -1.4384468793869019, -1.5298594236373901,
			-1.4153316020965576,
		]

		# 当前高级目标姿势
		self._hand_target: list[float] = list(self._simple_open_pose)
		self._hand_mode: str = "idle"

		# 连续抓取配置
		self._GRAB_PRIMARY_IDX: list[int] = [1, 4, 6]
		self._GRAB_TAU: float = getattr(self, "_cli_grip_force", 0.3)
		self._grab_stage: int = 0

		# 预先计算关节的闭合方向
		self._hand_open_pose = self._hand_poses.get("open", [0.0] * 7)
		self._hand_closed_pose = self._hand_poses.get("closed", [0.0] * 7)
		self._close_dir = [
			1.0 if (c - o) >= 0 else -1.0 
			for o, c in zip(self._hand_open_pose, self._hand_closed_pose)
		]

		# 自适应抓取调整
		self._PRESS_TARGET = 0.4
		self._PRESS_HYST = 0.05

		# 用于详细抓取调试的日志记录器帮助程序
		self._log_hand = logging.getLogger("g1_gui.hand")
		self._PRESS_THR = 0.5
		self._PRESS_MIN_COUNT = 3

	def _init_background_threads(self, iface: str):
		"""初始化后台工作线程。"""
		self._threads = [
			threading.Thread(target=_rx_realsense_local, args=(self._stop_evt,), daemon=True),
			threading.Thread(target=run_slam, args=(self._stop_evt,), daemon=True),
			threading.Thread(target=rx_battery, args=(self._stop_evt, iface), daemon=True),
		]
		for t in self._threads:
			t.start()

	# ------------------------------------------------------------------
	#  动态手臂配置帮助程序
	# ------------------------------------------------------------------

	def _configure_arm_variables(self):
		"""在用户切换选择器后（重新）初始化每臂状态。"""
		_WAIST_YAW_IDX = 12

		self._arm_joint_idx = list(range(15, 22)) if self._active_arm == "left" else list(range(22, 29))

		# 确保命令字典为每个受控关节都有一个条目
		if not hasattr(self, "_cmd_q"):
			self._cmd_q = {}
		for idx in self._arm_joint_idx:
			self._cmd_q.setdefault(idx, 0.0)
		self._cmd_q.setdefault(_WAIST_YAW_IDX, 0.0)

		# 从先前选择的手臂中删除任何过时的关节条目
		for idx in list(self._cmd_q):
			if idx not in self._arm_joint_idx and idx != _WAIST_YAW_IDX:
				self._cmd_q.pop(idx, None)

		# 构建适当的启动位姿序列
		self._init_arm_pose_sequences(_WAIST_YAW_IDX)

		# 重置进度
		self._seq_idx = 0
		self._initialised_from_state = False

	def run(self):
		"""运行 GUI 主循环。"""
		self.win.show()
		sys.exit(self.app.exec())

	# ------------------------------------------------------------------
	#  Qt 事件过滤器 (覆盖)
	# ------------------------------------------------------------------

	def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
		"""
		全局事件过滤器，用于处理键盘输入。
		"""
		from PySide6 import QtCore

		if event.type() == QtCore.QEvent.KeyPress:
			self._on_key_down(event)
		elif event.type() == QtCore.QEvent.KeyRelease:
			self._on_key_up(event)

		return super().eventFilter(source, event)

	def _on_key_down(self, evt: "QtCore.QEvent"):
		"""处理按键按下事件。"""
		from PySide6 import QtCore

		key = evt.key()
		self._pressed.add(key)
		self._update_movement()

	def _on_key_up(self, evt: "QtCore.QEvent"):
		"""处理按键释放事件。"""
		from PySide6 import QtCore

		key = evt.key()
		self._pressed.discard(key)
		self._update_movement()

		# 对于抓取手势的即时键盘绑定
		if key == QtCore.Qt.Key_G:
			self._hand_grab_quick()
		elif key == QtCore.Qt.Key_O:
			self._hand_open()
		elif key == QtCore.Qt.Key_C:
			self._hand_close()
		elif key == QtCore.Qt.Key_F:
			self._hand_point()
		elif key == QtCore.Qt.Key_T:
			self._hand_thumbs_up()

		# 关键手势提示显示按键列表
		from PySide6 import QtCore
		_gest_keys = [QtCore.Qt.Key_G, QtCore.Qt.Key_O, QtCore.Qt.Key_C, QtCore.Qt.Key_F, QtCore.Qt.Key_T]
		if key in _gest_keys:
			_key_names = {
				QtCore.Qt.Key_G: "G", QtCore.Qt.Key_O: "O", QtCore.Qt.Key_C: "C",
				QtCore.Qt.Key_F: "F", QtCore.Qt.Key_T: "T",
			}
			self._show_key_overlay(_key_names.get(key, "?"))

	def _show_key_overlay(self, text: str):
		"""显示按键视觉反馈。"""
		self._keys_lbl.setText(text)
		self._key_overlay.adjustSize()

		self._fade_anim.stop()
		self._keys_opacity.setOpacity(1.0)
		self._fade_anim.setStartValue(1.0)
		self._fade_anim.setEndValue(0.0)
		self._fade_anim.start()

	def _update_movement(self):
		"""根据按下的键更新运动速度。"""
		from PySide6 import QtCore

		# 腰部卸力
		if QtCore.Qt.Key_B in self._pressed:
			self._start_damp()

		# 手臂默认关节位置
		if QtCore.Qt.Key_H in self._pressed:
			self._arm_home()

		# 运动控制
		old_v = (self._vx, self._vy, self._omega)

		self._vx = 0.0
		self._vy = 0.0
		self._omega = 0.0

		# 前进/后退
		if QtCore.Qt.Key_W in self._pressed:
			self._vx += 0.5
		if QtCore.Qt.Key_S in self._pressed:
			self._vx -= 0.5

		# 左/右
		if QtCore.Qt.Key_A in self._pressed:
			self._vy += 0.4
		if QtCore.Qt.Key_D in self._pressed:
			self._vy -= 0.4

		# 旋转
		if QtCore.Qt.Key_Q in self._pressed:
			self._omega += 0.5
		if QtCore.Qt.Key_E in self._pressed:
			self._omega -= 0.5

		# 记录速度变化
		new_v = (self._vx, self._vy, self._omega)
		if new_v != old_v:
			print(f"[Vel] vx={self._vx:.2f}, vy={self._vy:.2f}, ω={self._omega:.2f}")

	# ------------------------------------------------------------------
	#  机器人运动控制定时器
	# ------------------------------------------------------------------

	def _on_drive_tick(self):
		"""定期向机器人发送运动命令。"""
		if self._bot is None:
			return

		try:
			self._bot.Move(self._vx, self._vy, self._omega)
		except Exception as exc:
			print(f"[drive] 错误: {exc}", file=sys.stderr)

	# ------------------------------------------------------------------
	#  手臂运动定时器
	# ------------------------------------------------------------------

	def _on_arm_tick(self):
		"""50 Hz 运动生成 – 根据活动手臂的身体状态进行斜坡化。"""
		if self._arm_pub is None:
			return

		self._maybe_init_from_lowstate()
		self._run_body_sequence()

		for joint_idx in self._arm_joint_idx:
			# 从命令关节角度到当前关节角度的轻微斜坡
			current_q = self._joint_cur.get(joint_idx, 0.0)
			target_q = self._cmd_q.get(joint_idx, 0.0)

			dq = target_q - current_q
			if abs(dq) > self._STEP:
				current_q += self._STEP if dq > 0 else -self._STEP

			# 应用到 DDS 消息
			mc = self._arm_cmd.motor_cmd[joint_idx]
			mc.q = current_q
			mc.dq = 0.0
			mc.tau = 0.0
			mc.kp = 60.0
			mc.kd = 1.5

		# 腰部卸力 - 特殊处理
		wmc = self._arm_cmd.motor_cmd[self._waist_idx]
		wmc.q = self._cmd_q.get(self._waist_idx, 0.0)
		wmc.dq = 0.0
		wmc.tau = 0.0
		if hasattr(self, "_waist_damped") and self._waist_damped:
			wmc.kp = 0.0
			wmc.kd = 10.0
		else:
			wmc.kp = 60.0
			wmc.kd = 1.5

		# 计算 CRC 并发送
		self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
		try:
			self._arm_pub.Write(self._arm_cmd)
		except Exception:
			pass

	def _maybe_init_from_lowstate(self):
		"""从当前低状态初始化手臂关节位置。"""
		if self._initialised_from_state:
			return

		for joint_idx in self._arm_joint_idx:
			if joint_idx in self._joint_cur:
				self._cmd_q[joint_idx] = self._joint_cur[joint_idx]

		if self._waist_idx in self._joint_cur:
			self._cmd_q[self._waist_idx] = self._joint_cur[self._waist_idx]

		self._initialised_from_state = True

	def _run_body_sequence(self):
		"""运行启动的身体位姿序列。"""
		if self._seq_idx >= len(self._pose_seq):
			return  # 已完成

		target_pose = self._pose_seq[self._seq_idx]
		
		# 检查当前位姿是否接近目标
		all_close = True
		for joint_idx, target_q in target_pose:
			current_q = self._joint_cur.get(joint_idx, 0.0)
			if abs(current_q - target_q) > self._SEQ_EPS:
				all_close = False

		if all_close:
			self._seq_idx += 1
			if self._seq_idx < len(self._pose_seq):
				# 设置下一个目标
				next_pose = self._pose_seq[self._seq_idx]
				for joint_idx, target_q in next_pose:
					self._cmd_q[joint_idx] = target_q
		else:
			# 设置当前目标
			for joint_idx, target_q in target_pose:
				self._cmd_q[joint_idx] = target_q

	# ------------------------------------------------------------------
	#  手臂控制用户命令
	# ------------------------------------------------------------------

	def _arm_home(self):
		"""将活动手臂移动到默认位置。"""
		if not self._arm_joint_idx:
			return

		print(f"[手臂] 回到 {self._active_arm} 手臂的默认位置")

		if self._active_arm == "right":
			target_poses = [
				(22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
				(26, +0.240), (27, -0.771), (28, -0.176),
			]
		else:
			target_poses = [
				(15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
				(19, -0.379), (20, -0.852), (21, -0.019),
			]

		for joint_idx, target_q in target_poses:
			self._cmd_q[joint_idx] = target_q

	def _start_damp(self):
		"""卸力手臂并回中腰部。"""
		print("[手臂] 卸力 & 腰部回中")

		# 将腰部设置为 0 rad
		self._cmd_q[self._waist_idx] = 0.0

		# 设置卸力标志
		self._waist_damped = True

		# 为活动手臂设置松弛的命令关节位置
		for joint_idx in self._arm_joint_idx:
			current_q = self._joint_cur.get(joint_idx, 0.0)
			self._cmd_q[joint_idx] = current_q

	def _on_damp_pressed(self):
		"""按钮处理程序，用于卸力。"""
		self._start_damp()

	# ------------------------------------------------------------------
	#  手部运动定时器
	# ------------------------------------------------------------------

	def _on_hand_tick(self):
		"""50 Hz 手部关节控制。"""
		if self._dex3 is None:
			return

		try:
			self._hand_update_sequence()
			self._hand_run_grab_mode()
			self._dex3.sendCommand(pos=self._hand_cmd_q, tau=[0.0] * 7)
		except Exception as exc:
			print(f"[手部] 控制错误: {exc}", file=sys.stderr)

	def _hand_update_sequence(self):
		"""从序列中更新手部关节位置。"""
		if not self._hand_pose_seq or self._hand_seq_idx >= len(self._hand_pose_seq):
			return

		target = self._hand_pose_seq[self._hand_seq_idx]
		
		# 计算从当前位置到目标的距离
		all_close = all(
			abs(self._hand_cmd_q[i] - target[i]) < 0.05 
			for i in range(7)
		)

		if all_close:
			self._hand_seq_idx += 1
		else:
			# 向目标移动
			for i in range(7):
				dq = target[i] - self._hand_cmd_q[i]
				if abs(dq) > self._HAND_STEP:
					self._hand_cmd_q[i] += self._HAND_STEP if dq > 0 else -self._HAND_STEP
				else:
					self._hand_cmd_q[i] = target[i]

	def _hand_run_grab_mode(self):
		"""处理连续抓取模式。"""
		if self._hand_mode != "grab" or self._dex3 is None:
			return

		try:
			# 获取压力反馈
			state = self._dex3.getState()
			pressures = [state.tau[i] for i in self._GRAB_PRIMARY_IDX]
			
			# 根据压力调整关节位置
			for i, idx in enumerate(self._GRAB_PRIMARY_IDX):
				press = pressures[i]
				
				if press < self._PRESS_TARGET - self._PRESS_HYST:
					# 继续关闭
					self._hand_cmd_q[idx] += self._close_dir[idx] * 0.02
				elif press > self._PRESS_TARGET + self._PRESS_HYST:
					# 稍微松开
					self._hand_cmd_q[idx] -= self._close_dir[idx] * 0.01

		except Exception as exc:
			print(f"[手部] 抓取模式错误: {exc}", file=sys.stderr)

	# ------------------------------------------------------------------
	#  手部姿势命令
	# ------------------------------------------------------------------

	def _hand_grab_quick(self):
		"""快速抓取。"""
		print("[手部] 快速抓取")
		self._hand_mode = "grab"
		self._grab_stage = 0

	def _hand_open(self):
		"""打开手部。"""
		print("[手部] 打开")
		self._hand_mode = "pose"
		self._hand_target = list(self._simple_open_pose)
		self._hand_pose_seq = [self._hand_target]
		self._hand_seq_idx = 0

	def _hand_close(self):
		"""关闭手部。"""
		print("[手部] 关闭")
		self._hand_mode = "pose"
		self._hand_target = list(self._simple_closed_pose)
		self._hand_pose_seq = [self._hand_target]
		self._hand_seq_idx = 0

	def _hand_point(self):
		"""指向手势。"""
		print("[手部] 指向")
		if "point" in self._hand_poses:
			self._hand_mode = "pose"
			self._hand_target = list(self._hand_poses["point"])
			self._hand_pose_seq = [self._hand_target]
			self._hand_seq_idx = 0

	def _hand_thumbs_up(self):
		"""点赞手势。"""
		print("[手部] 点赞")
		if "thumbs_up" in self._hand_poses:
			self._hand_mode = "pose"
			self._hand_target = list(self._hand_poses["thumbs_up"])
			self._hand_pose_seq = [self._hand_target]
			self._hand_seq_idx = 0

	# ------------------------------------------------------------------
	#  2D 地图点击处理器
	# ------------------------------------------------------------------

	def _on_map_click(self, event):
		"""处理 2D 地图上的鼠标点击，用于路径规划。"""
		import numpy as np

		if (
			self._occ_map is None
			or self._map_meta is None
			or not hasattr(event, "scenePos")
		):
			return

		scene_pos = event.scenePos()
		if scene_pos is None:
			return

		click_x, click_y = scene_pos.x(), scene_pos.y()

		min_x, min_y, scale = self._map_meta
		h, w = self._occ_map.shape

		# 限制点击位置在地图范围内
		click_x = clamp(click_x, 0, w - 1)
		click_y = clamp(click_y, 0, h - 1)

		# 将图像像素转换为世界坐标
		world_x = min_x + click_x / scale
		world_y = min_y + click_y / scale

		print(f"[路径] 规划到世界位置 ({world_x:.2f}, {world_y:.2f})")

		# 运行 A* 路径规划
		try:
			with _slam_lock:
				slam_data = _slam_latest
			
			if slam_data and len(slam_data) >= 3:
				robot_pose = slam_data[:3]  # [x, y, yaw]
				self._plan_path_to_goal(robot_pose, (world_x, world_y))

		except Exception as exc:
			print(f"[路径] 规划错误: {exc}", file=sys.stderr)

	def _plan_path_to_goal(self, robot_pose: list[float], goal: tuple[float, float]):
		"""使用 A* 算法规划路径。"""
		import numpy as np

		if self._occ_map is None or self._map_meta is None:
			return

		min_x, min_y, scale = self._map_meta
		h, w = self._occ_map.shape

		# 将机器人位置转换为像素坐标
		robot_px = int((robot_pose[0] - min_x) * scale)
		robot_py = int((robot_pose[1] - min_y) * scale)

		# 将目标转换为像素坐标
		goal_px = int((goal[0] - min_x) * scale)
		goal_py = int((goal[1] - min_y) * scale)

		# 边界检查
		robot_px = clamp(robot_px, 0, w - 1)
		robot_py = clamp(robot_py, 0, h - 1)
		goal_px = clamp(goal_px, 0, w - 1)
		goal_py = clamp(goal_py, 0, h - 1)

		# 简化的 A* 实现
		try:
			path = self._astar_search(
				(robot_px, robot_py), 
				(goal_px, goal_py), 
				self._occ_map
			)
			self._path_px = path
			print(f"[路径] 找到 {len(path) if path else 0} 个路径点")
		except Exception as exc:
			print(f"[路径] A* 搜索失败: {exc}", file=sys.stderr)
			self._path_px = None

	def _astar_search(self, start: tuple[int, int], goal: tuple[int, int], occ_map: "np.ndarray") -> list[tuple[int, int]] | None:
		"""简化的 A* 路径搜索算法。"""
		import heapq
		import numpy as np

		h, w = occ_map.shape
		
		# A* 数据结构
		open_set = [(0, start)]
		came_from = {}
		g_score = {start: 0}
		f_score = {start: self._heuristic(start, goal)}

		while open_set:
			current = heapq.heappop(open_set)[1]

			if current == goal:
				# 重构路径
				path = []
				while current in came_from:
					path.append(current)
					current = came_from[current]
				path.append(start)
				return path[::-1]

			# 检查邻居
			for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
				neighbor = (current[0] + dx, current[1] + dy)
				
				# 边界检查
				if not (0 <= neighbor[0] < w and 0 <= neighbor[1] < h):
					continue
				
				# 障碍物检查
				if occ_map[neighbor[1], neighbor[0]]:  # True = 障碍物
					continue

				tentative_g = g_score[current] + (1.4 if abs(dx) + abs(dy) == 2 else 1.0)

				if neighbor not in g_score or tentative_g < g_score[neighbor]:
					came_from[neighbor] = current
					g_score[neighbor] = tentative_g
					f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
					heapq.heappush(open_set, (f_score[neighbor], neighbor))

		return None  # 未找到路径

	def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
		"""A* 启发式函数（欧几里得距离）。"""
		return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

	# ------------------------------------------------------------------
	#  定期 UI 更新定时器
	# ------------------------------------------------------------------

	def _on_tick(self):
		"""定期更新 GUI 组件。"""
		try:
			self._update_cameras()
			self._update_slam_viz()
			self._update_2d_map()
			self._update_status()
		except Exception as exc:
			print(f"[UI] 更新错误: {exc}", file=sys.stderr)

	def _update_cameras(self):
		"""更新摄像头显示。"""
		with _state_lock:
			rgb_img = _state.get("rgb")
			depth_img = _state.get("depth")

		if rgb_img is not None:
			qpix = numpy_to_qpix(rgb_img)
			if qpix:
				qpix = qpix.scaled(
					self.rgb_lbl.size(),
					QtCore.Qt.KeepAspectRatio,
					QtCore.Qt.SmoothTransformation,
				)
				self.rgb_lbl.setPixmap(qpix)

		if depth_img is not None:
			# 将深度转换为可视化图像
			import numpy as np
			depth_vis = np.clip(depth_img / 5000.0 * 255, 0, 255).astype(np.uint8)
			depth_rgb = np.stack([depth_vis] * 3, axis=-1)
			
			qpix = numpy_to_qpix(depth_rgb)
			if qpix:
				qpix = qpix.scaled(
					self.depth_lbl.size(),
					QtCore.Qt.KeepAspectRatio,
					QtCore.Qt.SmoothTransformation,
				)
				self.depth_lbl.setPixmap(qpix)

	def _update_slam_viz(self):
		"""更新 3D SLAM 可视化。"""
		import numpy as np
		import pyqtgraph.opengl as gl

		with _state_lock:
			points = _state.get("slam_points")

		if points is not None and len(points) > 0:
			# 限制点数以提高性能
			if len(points) > 10000:
				step = len(points) // 10000
				points = points[::step]

			# 更新散点图
			self._scatter.setData(
				pos=points,
				color=(0.5, 0.8, 1.0, 0.6),
				size=2.0
			)

		# 更新机器人位姿
		with _slam_lock:
			slam_data = _slam_latest

		if slam_data and len(slam_data) >= 6:
			x, y, z, qx, qy, qz, qw = slam_data[:7]
			
			# 清除之前的位姿线条
			for item in self._pose_items:
				self.gl_view.removeItem(item)
			self._pose_items.clear()

			# 绘制位姿轴
			axis_length = 0.5
			poses = [
				([x, x + axis_length], [y, y], [z, z], (1, 0, 0, 1)),  # X 轴 - 红色
				([x, x], [y, y + axis_length], [z, z], (0, 1, 0, 1)),  # Y 轴 - 绿色
				([x, x], [y, y], [z, z + axis_length], (0, 0, 1, 1)),  # Z 轴 - 蓝色
			]

			for pos_x, pos_y, pos_z, color in poses:
				line_item = gl.GLLinePlotItem(
					pos=np.array([pos_x, pos_y, pos_z]).T,
					color=color,
					width=3.0
				)
				self.gl_view.addItem(line_item)
				self._pose_items.append(line_item)

	def _update_2d_map(self):
		"""更新 2D 占据栅格地图。"""
		import numpy as np

		with _state_lock:
			occ_data = _state.get("occupancy_map")

		if occ_data is None:
			return

		# 解包占据栅格数据
		occ_map, min_x, min_y, scale = occ_data
		self._occ_map = occ_map
		self._map_meta = (min_x, min_y, scale)

		# 创建可视化图像
		h, w = occ_map.shape
		vis_img = np.zeros((h, w, 3), dtype=np.uint8)

		# 自由空间 = 白色，障碍物 = 黑色，未知 = 灰色
		vis_img[occ_map == 0] = [255, 255, 255]  # 自由空间
		vis_img[occ_map == 1] = [0, 0, 0]        # 障碍物
		vis_img[occ_map == -1] = [128, 128, 128] # 未知

		# 绘制规划的路径
		if self._path_px:
			for px, py in self._path_px:
				if 0 <= px < w and 0 <= py < h:
					vis_img[py, px] = [255, 0, 0]  # 红色路径

		# 绘制机器人位置
		with _slam_lock:
			slam_data = _slam_latest

		if slam_data and len(slam_data) >= 2:
			robot_x, robot_y = slam_data[:2]
			robot_px = int((robot_x - min_x) * scale)
			robot_py = int((robot_y - min_y) * scale)
			
			if 0 <= robot_px < w and 0 <= robot_py < h:
				# 绘制机器人位置为绿色圆点
				for dx in range(-2, 3):
					for dy in range(-2, 3):
						px, py = robot_px + dx, robot_py + dy
						if 0 <= px < w and 0 <= py < h and dx*dx + dy*dy <= 4:
							vis_img[py, px] = [0, 255, 0]  # 绿色

		# 更新图像显示
		self._map_img.setImage(vis_img.transpose(1, 0, 2))

	def _update_status(self):
		"""更新状态栏信息。"""
		with _state_lock:
			battery = _state.get("battery_pct", 0)

		status_text = f"电池: {battery}%"

		# 添加 SLAM 状态信息
		with _slam_lock:
			slam_data = _slam_latest

		if slam_data and len(slam_data) >= 3:
			x, y, yaw = slam_data[:3]
			status_text += f" | 位置: ({x:.2f}, {y:.2f}, {np.rad2deg(yaw):.1f}°)"

		self.status.setText(status_text)

	# ------------------------------------------------------------------
	#  清理
	# ------------------------------------------------------------------

	def _on_quit(self):
		"""应用程序退出时的清理。"""
		print("[run_g1_gui] 正在关闭...")
		self._stop_evt.set()

		# 等待线程结束
		for t in self._threads:
			if t.is_alive():
				t.join(timeout=1.0)

		# 清理手部连接
		if self._dex3:
			try:
				self._dex3.close()
			except Exception:
				pass

		print("[run_g1_gui] 已关闭。")
