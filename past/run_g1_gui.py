#!/usr/bin/env python3.10
"""
run_g1_gui.py – 单窗口 PySide6 GUI

布局
======
┌─────────────────────────── MainWindow ─────────────────────────────┐
│ ┌─────────────┐  ┌──────────────────────────────────────────────┐ │
│ │   RGB       │  │        3D SLAM (pyqtgraph.GLViewWidget)      │ │
│ │   640×480   │  │  – 计划支持旋转/缩放/点击拾取 –              │ │
│ └─────────────┘  │                                              │ │
│ ┌─────────────┐  │                                              │ │
│ │  深度图     │  │                                              │ │
│ │  640×480    │  │                                              │ │
│ └─────────────┘  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘

键盘遥控、RealSense 和 Livox 的工作线程在后台运行（从 *run_geoff_stack* 导入），
未作更改。SLAM 点云在 **GLViewWidget** 中渲染，因此在 Qt 布局内部时仍可交互。

依赖需求
------------
    pip install pyside6 pyqtgraph~=0.13

(pyqtgraph 使用 *qtpy*，因此可自动与 PySide6 配合使用。)
"""

# noqa: D301
# pylint: disable=attribute-defined-outside-init

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Tuple

# ---------------------------------------------------------------------------
#  日志记录 – 将所有控制台输出捕获到滚动文件中，以便每次运行都从
#  一个新的日志开始，同时保留少量先前会话的历史记录
#  (``run_g1_gui.log.1`` …)。活动日志位于项目
#  的 "logs" 子目录中。
# ---------------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging() -> logging.Logger:
    """初始化根日志记录器并返回 GUI 特定的子日志记录器。"""

    # 总是从一个新的日志文件开始，这样我们只捕获当前运行的输出。
    # 当文件大小超过 *maxBytes* 时，它会被轮转为 ``run_g1_gui.log.1``
    # (旧版本会根据 *backupCount* 被丢弃)。

    # 将日志文件放在项目目录内，使其保持自包含。
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)

    log_path = log_dir / "run_g1_gui.log"

    # 文件处理器 – 每次启动时覆盖，大约 2 MB 后轮转。
    fh = RotatingFileHandler(
        log_path, mode="w", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )

    # 控制台处理器 – 继续打印到 *原始* stderr，以便用户
    # 从终端启动时仍能看到消息。
    _orig_stderr = sys.stderr
    ch = logging.StreamHandler(_orig_stderr)

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)

    root.addHandler(ch)

    # ------------------------------------------------------------------
    #  屏蔽 NumPy ctypeslib 的 PEP3118 警告
    # ------------------------------------------------------------------
    # Unitree 的 Livox C-wrapper 返回 ctypes 结构，其内部的
    # *PEP 3118* 缓冲区格式字符串偶尔会错误地报告
    # 真实的 itemsize。因此，每次我们将这样的结构转换为
    # ndarray 时，NumPy 都会发出一个 RuntimeWarning。我们的 stderr→logger
    # 桥接器将该警告升级到 *ERROR* 级别，这会吓到用户，
    # 即使该消息是无害的。
    #
    # 全局过滤掉它，以保持日志清洁，同时不影响其他警告。
    # ------------------------------------------------------------------
    import warnings  # 局部延迟导入 – 仅在此处需要

    warnings.filterwarnings(
        "ignore",
        message=r"A builtin ctypes object gave a PEP3118 format string that does not match its itemsize.*",
        category=RuntimeWarning,
        module=r"numpy\.ctypeslib",
    )

    # 重定向所有对 sys.stdout / sys.stderr 的写入，以便来自
    # 第三方库的零散打印也最终进入日志（同时通过上面的
    # StreamHandler 出现在控制台中）。

    class _StreamToLogger:  # pylint: disable=too-few-public-methods
        def __init__(self, level: int):
            self._level = level
            self._logger = logging.getLogger("g1_gui")

        def write(self, msg: str):
            """流接口的 write 方法。"""
            msg = msg.rstrip()
            if msg:
                self._logger.log(self._level, msg)

        def flush(self):
            """流接口的 flush 方法。"""
            pass

    sys.stdout = _StreamToLogger(logging.INFO)  # type: ignore[assignment]
    sys.stderr = _StreamToLogger(logging.ERROR)  # type: ignore[assignment]

    # 我们自己消息的子日志记录器 – 从根继承处理器。
    return logging.getLogger("g1_gui")


# 立即初始化，以便在导入期间执行的任何内容都被捕获
log = _setup_logging()

# Qt 导入必须在 *类* 定义时可用，因为我们现在
# 从 QtCore.QObject 派生 G1Windows，使其可以作为全局
# 事件过滤器。

try:
    from PySide6 import QtCore  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover – missing optional dep.
    log.error("需要 PySide6 – 通过 'pip install pyside6 pyqtgraph' 安装")
    raise SystemExit(
        "run_g1_gui.py 需要 PySide6 – 请使用以下命令安装:\n"
        "    pip install pyside6 pyqtgraph"
    ) from exc

# ------------------------------------------------------------------------
# 复用 run_g1_stack 中的 RealSense 接收器和遥控线程
# ------------------------------------------------------------------------

# 注意：我们仍然从 *run_g1_stack* 导入 RealSense 接收器和共享状态帮助程序，
#       但 **不再** 启动键盘线程。相反，我们直接通过 Qt 处理按键，
#       因此监听器位于主 GUI 线程中，并在所有平台/显示服务器上可靠工作。

from run_g1_stack import (  # type: ignore
    _rx_realsense_local,
    _state,
    _state_lock,
)

# ---------------------------------------------------------------------------
# 电池监视器 – 订阅 LowState 并在 _state 中发布 SOC 百分比。
# ---------------------------------------------------------------------------


def _rx_battery(stop: "threading.Event", iface: str):
    """在后台工作，将最新的电池百分比保存在共享的 _state 中。"""

    try:
        import time
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

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
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

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
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_

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

    except Exception as exc:  # pylint: disable=broad-except
        import sys

        print("[run_g1_gui] 电池监视器已禁用:", exc, file=sys.stderr)


# ------------------------------------------------------------------------
# 为 live_slam 提供一个*仅推送*的查看器，它只将最新的地图存储
# 在一个共享变量中。Qt 线程将使用 pyqtgraph 对其进行可视化。
# ------------------------------------------------------------------------


_slam_latest: Tuple[Any, Any] | None = None  # (xyz ndarray, pose ndarray)
_slam_lock = threading.Lock()


def _patch_live_slam_for_pyqt() -> None:
    """猴子补丁 live_slam._Viewer，使其不再打开 GLFW 窗口。"""

    import numpy as np  # pylint: disable=import-error

    class _QtViewer:  # pylint: disable=too-few-public-methods
        def __init__(self):
            self._latest_pts: np.ndarray | None = None
            self._latest_pose: np.ndarray | None = None

        # -------- 从 SLAM 线程调用 --------------------------------
        def push(self, xyz: np.ndarray, pose: np.ndarray):
            global _slam_latest  # noqa: PLW0603
            with _slam_lock:
                _slam_latest = (xyz, pose)

        # -------- tick() 签名保留以兼容 -----------------------
        def tick(self) -> bool:
            """tick 方法，保持 SLAM 主循环存活。"""
            # 无事可做 – 返回 True 以保持 SLAM 主循环存活。
            return True

        def close(self):
            pass

    import live_slam as _ls  # type: ignore

    _ls._Viewer = _QtViewer  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    #  安全补丁 – 使 LiveSLAMDemo 对偶尔的 KISS-ICP
    #  初始化问题具有鲁棒性。我们包装其 handle_points()，以便捕获
    #  原始实现中的*任何*异常（例如由于第一帧扫描时
    #  未初始化的位姿），并且我们仍然将原始 xyz 转发给查看器。
    #  这保证了 Qt GUI 总是能收到一些东西来显示，因此永远不会
    #  保持空白。
    # ------------------------------------------------------------------

    try:
        _orig_hp = _ls.LiveSLAMDemo.handle_points  # type: ignore[attr-defined]

        def _safe_hp(self, xyz):  # type: ignore[no-self-use]
            try:
                _orig_hp(self, xyz)  # type: ignore[misc]
            except Exception as exc:  # pylint: disable=broad-except
                # 推送没有有效位姿的原始点。GL 散点图仍然
                # 渲染；位姿轴只是保持缺失，直到 KISS-ICP
                # 恢复。
                try:
                    self._viewer.push(xyz, None)
                except Exception:
                    pass
                print("[run_g1_gui] KISS-ICP 第一帧失败:", exc)

        _ls.LiveSLAMDemo.handle_points = _safe_hp  # type: ignore[assignment]
    except Exception:
        pass


# ------------------------------------------------------------------------
# SLAM 工作线程 – 在打补丁后启动
# ------------------------------------------------------------------------


def _run_slam(stop_evt: threading.Event):  # pragma: no cover – needs HW
    """运行 Livox SLAM 管道的后台工作线程。

    我们让*驱动程序*在其自己的 ``spin()`` 方法内阻塞，以便 SDK
    线程可以不间断地推送点云帧。一旦 Qt
    应用程序请求关闭（设置了 ``stop_evt``），我们将优雅地
    拆卸所有东西。
    """

    try:
        _patch_live_slam_for_pyqt()

        import live_slam as _ls  # type: ignore

        demo = _ls.LiveSLAMDemo()  # type: ignore[attr-defined]

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
                    demo._viewer.tick()  # type: ignore[attr-defined]
                except Exception:
                    pass

                time.sleep(0.05)
        finally:
            try:
                demo.shutdown()
            except Exception:
                pass

    except Exception as exc:  # pylint: disable=broad-except
        print("[run_g1_gui] SLAM 线程已禁用:", exc, file=sys.stderr)


# ------------------------------------------------------------------------
# Qt GUI
# ------------------------------------------------------------------------


class G1Windows(QtCore.QObject):  # type: ignore[misc]  # pylint: disable=too-few-public-methods
    def __init__(
        self,
        iface: str,
        ground_clear_in: float,
        *,
        hand: str = "right",
        grip_force: float | None = None,
    ):
        """
        创建主 GUI 窗口。

        Args:
            iface (str): 连接到机器人的网络接口。
            ground_clear_in (float): 在检测到的地平面上方的间隙（以**英寸**为单位），
                超过该间隙的点被视为障碍物（转发给 SLAM 障碍物过滤器）。
            hand (str): 哪个 Dex3 手物理连接到机器人。默认为 ``"right"`` 以保留原始行为。
            grip_force (float | None): 在连续*抓取*模式下应用的可选前馈扭矩（约 **N·m**）。
        """
        super().__init__()
        
        # 存储配置参数
        self._store_config_parameters(iface, ground_clear_in, hand, grip_force)
        
        # 初始化核心组件
        self._init_qt_application()
        self._init_ui_components()
        self._init_control_components()
        self._init_robot_connections(iface, hand)
        
        # 启动后台线程和定时器
        self._start_background_threads(iface)
        self._configure_event_handling()
        
        # 完成初始化
        self._finalize_initialization()

    def _store_config_parameters(self, iface: str, ground_clear_in: float, hand: str, grip_force: float | None):
        """存储配置参数到实例变量。"""
        self._iface = iface
        self._clear_m = ground_clear_in * 0.0254  # 英寸 → 米
        self._hand_config = hand
        self._cli_grip_force = grip_force if grip_force is not None else 0.3

    def _init_qt_application(self):
        """初始化 Qt 应用程序和信号处理。"""
        from PySide6 import QtWidgets
        import signal
        
        self.app = QtWidgets.QApplication(sys.argv)
        
        # 配置 Ctrl-C 信号处理
        try:
            signal.signal(signal.SIGINT, lambda *_: self.app.quit())
        except Exception:
            pass

    def _init_ui_components(self):
        """初始化用户界面组件。"""
        self._init_main_widgets()
        self._init_layout()
        self._init_status_controls()
        self._init_visual_feedback()
        self._init_timers()

    def _init_main_widgets(self):
        """初始化主要 UI 控件。"""
        from PySide6 import QtWidgets, QtCore
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        
        # RGB/深度图像标签
        self.rgb_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.depth_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        
        # 设置最小尺寸和背景样式
        for lbl in (self.rgb_lbl, self.depth_lbl):
            lbl.setMinimumSize(640, 320)
            lbl.setStyleSheet("background-color: black")
        
        # 2D 地图视图
        self.map_view = pg.GraphicsLayoutWidget()
        self.map_view.setMinimumSize(640, 320)
        self._map_vb = self.map_view.addViewBox(lockAspect=True, enableMouse=True)
        self._map_vb.setMenuEnabled(False)
        self._map_vb.invertY(True)
        self._map_img = pg.ImageItem()
        self._map_vb.addItem(self._map_img)
        
        # 3D 点云视图
        self.gl_view = gl.GLViewWidget()
        self.gl_view.opts["distance"] = 30
        self.gl_view.setCameraPosition(distance=30, elevation=20, azimuth=45)
        self.gl_view.setMinimumWidth(640)
        self._scatter = gl.GLScatterPlotItem()
        self.gl_view.addItem(self._scatter)
        self._pose_items: list[gl.GLLinePlotItem] = []

    def _init_layout(self):
        """设置主窗口布局。"""
        from PySide6 import QtWidgets
        
        # 创建分割器布局
        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(left)
        
        v.addWidget(self.rgb_lbl)
        v.addWidget(self.depth_lbl)
        v.addWidget(self.map_view)
        
        splitter.addWidget(left)
        splitter.addWidget(self.gl_view)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 640])
        
        # 主窗口配置
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("G1-Stack")
        self.win.setCentralWidget(splitter)
        self.win.resize(1600, 760)

    def _init_status_controls(self):
        """初始化状态栏和控制按钮。"""
        from PySide6 import QtWidgets
        
        # 状态标签
        self.status = QtWidgets.QLabel()
        self.win.statusBar().addWidget(self.status)
        
        # 卸力按钮
        self._btn_damp = QtWidgets.QPushButton("卸力手臂 & 腰部回中")
        self._btn_damp.setToolTip("将上半身切换到被动模式，并将腰部设置为 0 rad")
        self._btn_damp.clicked.connect(self._on_damp_pressed)
        self.win.statusBar().addPermanentWidget(self._btn_damp)
        
        # 手臂选择器
        self._arm_selector = QtWidgets.QComboBox()
        self._arm_selector.addItems(["左臂", "右臂"])
        self._arm_selector.setCurrentIndex(0)
        self._arm_selector.setToolTip("选择要控制和运行推理的手臂")
        self._arm_selector.currentIndexChanged.connect(self._on_arm_selection_changed)
        self.win.statusBar().addPermanentWidget(self._arm_selector)
        
        self._active_arm: str = "right"

    def _on_arm_selection_changed(self, index: int):
        """处理手臂选择器的变化事件。"""
        try:
            # 更新活动手臂
            self._active_arm = "left" if index == 0 else "right"
            
            # 重新配置手臂变量
            self._configure_arm_variables()
            
            print(f"[run_g1_gui] 已切换到 {self._active_arm} 臂控制")
            
        except Exception as exc:
            print(f"[run_g1_gui] 手臂切换失败: {exc}", file=sys.stderr)

    def _on_fade_finished(self):
        """按键覆盖层淡出动画完成时的回调。"""
        if self._fade_anim.state() != QtCore.QAbstractAnimation.Running:
            self._keys_lbl.setText("–")
            self._key_overlay.adjustSize()
    def _init_visual_feedback(self):
        """初始化按键视觉反馈覆盖层。"""
        from PySide6 import QtWidgets, QtCore
        
        # 按键覆盖层
        self._key_overlay = QtWidgets.QWidget(self.gl_view)
        self._key_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._key_overlay.move(10, 10)
        self._key_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150);"
            "border-radius: 6px;"
        )
        
        # 覆盖层布局
        lay = QtWidgets.QVBoxLayout(self._key_overlay)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(0)
        
        # 标题和按键标签
        self._header_lbl = QtWidgets.QLabel("按键输入:")
        self._header_lbl.setStyleSheet(
            "color: #ffff00; font: 12pt 'Consolas', 'Monaco', 'Courier New', monospace;"
        )
        lay.addWidget(self._header_lbl)
        
        self._keys_lbl = QtWidgets.QLabel("–")
        self._keys_lbl.setStyleSheet(
            "color: #ffff00; font: bold 24pt 'Consolas', 'Monaco', 'Courier New', monospace;"
        )
        lay.addWidget(self._keys_lbl)
        
        self._key_overlay.adjustSize()
        self._key_overlay.show()
        
        # 透明度动画
        self._keys_opacity = QtWidgets.QGraphicsOpacityEffect(self._keys_lbl)
        self._keys_lbl.setGraphicsEffect(self._keys_opacity)
        self._fade_anim = QtCore.QPropertyAnimation(self._keys_opacity, b"opacity", self)
        self._fade_anim.setDuration(600)
        self._fade_anim.finished.connect(self._on_fade_finished)

    def _init_timers(self):
        """初始化定时器。"""
        from PySide6 import QtCore
        
        # 主刷新定时器
        self._refresh = QtCore.QTimer()
        self._refresh.setInterval(30)
        self._refresh.timeout.connect(self._on_tick)
        self._refresh.start()
        
        # 驱动控制定时器
        self._drive_timer = QtCore.QTimer()
        self._drive_timer.setInterval(100)
        self._drive_timer.timeout.connect(self._on_drive_tick)
        self._drive_timer.start()

    def _init_control_components(self):
        """初始化控制状态变量。"""
        self._stop_evt = threading.Event()
        self._pressed: set[object] = set()
        
        # 速度控制状态
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._bal_mode: int = -1
        
        # 路径规划状态
        self._occ_map: "np.ndarray | None" = None
        self._map_meta: tuple[float, float, float] | None = None
        self._path_px: list[tuple[int, int]] | None = None
        
        # 连接地图点击信号
        self.map_view.scene().sigMouseClicked.connect(self._on_map_click)

    def _init_robot_connections(self, iface: str, hand: str):
        """初始化机器人连接。"""
        self._init_robot_control(iface)
        self._init_arm_control()
        self._init_hand_control(hand, iface)

    def _init_robot_control(self, iface: str):
        """初始化机器人基础控制连接。"""
        try:
            from hanger_boot_sequence import hanger_boot_sequence
            self._bot = hanger_boot_sequence(iface=iface)
        except Exception as exc:
            print("[run_g1_gui] 遥控操作已禁用:", exc, file=sys.stderr)
            self._bot = None

    def _init_arm_control(self):
        """初始化手臂控制系统。"""
        self._arm_pub = None
        
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            
            # 初始化手臂控制变量
            self._init_arm_variables()
            
            # 创建发布者和定时器
            self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._arm_pub.Init()
            
            self._arm_timer = QtCore.QTimer()
            self._arm_timer.setInterval(20)
            self._arm_timer.timeout.connect(self._on_arm_tick)
            self._arm_timer.start()
            
            # 启动 LowState 订阅
            self._start_lowstate_subscription()
            
            # 准备非活动手臂
            self._prepare_inactive_arm()
            
        except Exception as exc:
            print("[run_g1_gui] 手臂控制已禁用:", exc, file=sys.stderr)

    def _init_arm_variables(self):
        """初始化手臂控制变量。"""
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        
        # 关节索引配置
        self._WAIST_YAW_IDX = 12
        self._LEFT_IDX = {idx: 0 for idx in range(15, 22)}
        self._RIGHT_IDX = {idx: 0 for idx in range(22, 29)}
        
        # 初始化关节状态
        self._arm_joint_idx: list[int] = []
        self._waist_idx: int = self._WAIST_YAW_IDX
        self._cmd_q: dict[int, float] = {}
        self._joint_cur: dict[int, float] = {}
        
        # 控制参数 - 减小步长以实现更慢的运动
        self._SEQ_EPS = 0.01
        self._STEP = 0.008  # 从 0.02 减小到 0.008，使运动速度降低约 60%
        self._seq_idx = 0
        self._initialised_from_state = False
        
        # DDS 消息和 CRC
        self._crc = CRC()
        self._arm_cmd = unitree_hg_msg_dds__LowCmd_()
        self._arm_cmd.motor_cmd[29].q = 1  # 启用 arm_sdk
        
        # 位姿序列
        self._pose_seq: list[list[tuple[int, float]]] = []

    def _init_hand_control(self, hand: str, iface: str):
        """初始化 Dex3 手部控制。"""
        self._dex3 = None
        
        try:
            from unitree_sdk2py.dex3 import Dex3Client
            
            # 尝试连接 Dex3
            self._connect_dex3(hand, iface)
            
            if self._dex3 is not None:
                # 加载手部姿势配置
                self._load_hand_poses()
                
                # 初始化手部控制变量
                self._init_hand_variables()
                
                # 启动手部控制定时器
                self._hand_timer = QtCore.QTimer()
                self._hand_timer.setInterval(20)
                self._hand_timer.timeout.connect(self._on_hand_tick)
                self._hand_timer.start()
                
        except Exception as exc:
            print("[run_g1_gui] Dex3 手部控制已禁用:", exc, file=sys.stderr)
            self._dex3 = None

    def _connect_dex3(self, hand: str, iface: str):
        """连接到 Dex3 手部设备。"""
        from unitree_sdk2py.dex3 import Dex3Client
        
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

    def _load_hand_poses(self):
        """从 CSV 文件加载手部姿势配置。"""
        import csv
        
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

    def _init_hand_variables(self):
        """初始化手部控制变量。"""
        # 运动控制 - 减小步长以实现更慢的手部运动
        self._hand_cmd_q: list[float] = [0.0] * 7
        self._hand_pose_seq: list[list[float]] = []
        self._hand_seq_idx: int = 0
        self._HAND_STEP = 0.04  # 从 0.1 减小到 0.04，使手部运动更加平滑
        
        # 预定义姿势
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
        
        # 控制状态
        self._hand_target: list[float] = list(self._simple_open_pose)
        self._hand_mode: str = "idle"
        
        # 抓取配置
        self._GRAB_PRIMARY_IDX: list[int] = [1, 4, 6]
        self._GRAB_TAU: float = self._cli_grip_force
        self._grab_stage: int = 0
        
        # 计算关节闭合方向
        self._hand_open_pose = self._hand_poses.get("open", [0.0] * 7)
        self._hand_closed_pose = self._hand_poses.get("closed", [0.0] * 7)
        self._close_dir = [
            1.0 if (c - o) >= 0 else -1.0 
            for o, c in zip(self._hand_open_pose, self._hand_closed_pose)
        ]
        
        # 压力控制参数
        self._PRESS_TARGET = 0.4
        self._PRESS_HYST = 0.05
        self._PRESS_THR = 0.5
        self._PRESS_MIN_COUNT = 3
        
        # 日志记录器
        self._log_hand = logging.getLogger("g1_gui.hand")

    def _start_background_threads(self, iface: str):
        """启动后台工作线程。"""
        self._threads = [
            threading.Thread(target=_rx_realsense_local, args=(self._stop_evt,), daemon=True),
            threading.Thread(target=_run_slam, args=(self._stop_evt,), daemon=True),
            threading.Thread(target=_rx_battery, args=(self._stop_evt, iface), daemon=True),
        ]
        
        for t in self._threads:
            t.start()

    def _configure_event_handling(self):
        """配置事件处理。"""
        # 安装全局事件过滤器
        self.app.installEventFilter(self)
        
        # 连接退出信号
        self.app.aboutToQuit.connect(self._on_quit)

    def _finalize_initialization(self):
        """完成初始化过程。"""
        try:
            self._configure_arm_variables()
        except Exception as exc:
            print("[run_g1_gui] 初始手臂配置失败:", exc, file=sys.stderr)

    def _start_lowstate_subscription(self):
        """启动 LowState 订阅线程。"""
        def _init_ls_sub():
            from unitree_sdk2py.core.channel import ChannelSubscriber
            
            candidates = [
                "unitree_sdk2py.idl.unitree_hg.msg.dds_.LowState_",
                "unitree_sdk2py.idl.unitree_go.msg.dds_.LowState_",
            ]
            
            for dotted in candidates:
                try:
                    mod_path, cls_name = dotted.rsplit(".", 1)
                    mod = __import__(mod_path, fromlist=[cls_name])
                    LowState_ = getattr(mod, cls_name)
                    
                    def _ls_cb(msg):
                        for j_idx in (*self._arm_joint_idx, self._WAIST_YAW_IDX):
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

    def _prepare_inactive_arm(self):
        """为非活动手臂设置待机姿势。"""
        def _apply_pose_once(pose: list[tuple[int, float]]):
            for j_idx, q_val in pose:
                mc = self._arm_cmd.motor_cmd[j_idx]
                mc.q = q_val
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 60.0
                mc.kd = 1.5

        # 为另一只手臂构建准备位姿
        if self._active_arm == "right":
            ready_other = [
                (22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
                (26, +0.240), (27, -0.771), (28, -0.176),
            ]
        else:
            ready_other = [
                (15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
                (19, -0.379), (20, -0.852), (21, -0.019),
            ]

        _apply_pose_once(ready_other)

        # 重新计算 CRC 并立即传输，以便机器人在
        # 第一个 50 Hz 定时器滴答发生之前开始保持位姿。
        self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
        try:
            self._arm_pub.Write(self._arm_cmd)
        except Exception:
            pass

    # ------------------------------------------------------------------
    #  动态手臂配置帮助程序
    # ------------------------------------------------------------------

    def _configure_arm_variables(self):
        """
        在用户切换选择器后（重新）初始化每臂状态。

        所有需要手臂特定列表的地方都查询 *self._arm_joint_idx*，
        因此更新该属性以及启动位姿序列
        就足够了。如果在运行时调用配置，我们还
        重置当前序列索引，以便新手臂平滑地移动到
        其初始姿态。
        """

        _WAIST_YAW_IDX = 12

        self._arm_joint_idx = list(range(15, 22)) if self._active_arm == "left" else list(range(22, 29))

        # 确保命令字典为每个受控
        # 关节都有一个条目，以便后续的斜坡更新不会出现 KeyError。
        if not hasattr(self, "_cmd_q"):
            self._cmd_q = {}
        for idx in self._arm_joint_idx:
            self._cmd_q.setdefault(idx, 0.0)
        self._cmd_q.setdefault(_WAIST_YAW_IDX, 0.0)

        # 从先前选择的手臂中删除任何过时的关节条目，
        # 以便后续的 LowCmd 消息*仅*触及当前活动的
        # 一侧（加上腰部）。
        for idx in list(self._cmd_q):
            if idx not in self._arm_joint_idx and idx != _WAIST_YAW_IDX:
                self._cmd_q.pop(idx, None)

        # 构建适当的启动位姿序列。
        if self._active_arm == "right":
            # 与之前相同的两步序列。
            self._pose_seq = [
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (22, -0.023),
                    (23, -0.225),
                    (24, +0.502),
                    (25, +1.317),
                    (26, +0.185),
                    (27, +0.125),
                    (28, -0.182),
                ],
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (22, +0.087),
                    (23, -0.271),
                    (24, +0.323),
                    (25, +0.691),
                    (26, +0.240),
                    (27, -0.771),
                    (28, -0.176),
                ],
            ]
        else:
            self._pose_seq = [
                [
                    (_WAIST_YAW_IDX, 0.0),
                    (15, +0.211),
                    (16, +0.181),
                    (17, -0.284),
                    (18, +0.672),
                    (19, -0.379),
                    (20, -0.852),
                    (21, -0.019),
                ]
            ]

        # 重置进度，以便新手臂立即开始移动。
        self._seq_idx = 0

        # 强制基于 LowState 的新初始化，以避免
        # 在运行中的机器人上切换手臂时发生跳变。
        self._initialised_from_state = False

    # ------------------------------------------------------------------
    def _numpy_to_qpix(self, bgr):
        import numpy as np  # 局部
        from PySide6 import QtGui  # type: ignore

        if bgr is None or bgr.dtype != np.uint8:
            return None
        h, w, _ = bgr.shape
        qimg = QtGui.QImage(bgr.data.tobytes(), w, h, 3 * w, QtGui.QImage.Format_BGR888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    # ------------------------------------------------------------------
    def _on_tick(self):
        import numpy as np  # type: ignore

        # 始终刷新按键覆盖层，以便用户获得即时反馈。
        self._update_key_overlay()

        with _state_lock:
            rgbd = _state.get("rgbd")
            vx, vy, om = _state.get("vel", (0.0, 0.0, 0.0))
            soc = _state.get("soc")

        if rgbd is not None and rgbd.shape == (480, 1280, 3):
            rgb, depth = rgbd[:, :640], rgbd[:, 640:]
            px1, px2 = self._numpy_to_qpix(rgb), self._numpy_to_qpix(depth)
            if px1:
                from PySide6 import QtCore  # 局部导入 – 仅在此处需要

                # 缩放像素图以*适应*当前标签大小，
                # 同时保持原始宽高比。这避免了
                # 之前的行为，即 480 像素高的相机图像
                # 被简单地裁剪到约 320 像素以适应堆叠
                # 布局。用户现在会看到轻微的黑边，但
                # 永远不会丢失帧的任何内容。
                scaled = px1.scaled(
                    self.rgb_lbl.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.rgb_lbl.setPixmap(scaled)
            if px2:
                from PySide6 import QtCore  # 局部导入 – 仅在此处需要
                scaled = px2.scaled(
                    self.depth_lbl.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.depth_lbl.setPixmap(scaled)

        status_txt = f"vx {vx:+.2f}  vy {vy:+.2f}  omega {om:+.2f}"
        if soc is not None:
            status_txt += f"   电池 {soc:3d}%"
        else:
            with _state_lock:
                volt = _state.get("voltage")
            if volt is not None:
                status_txt += f"   V {volt:5.1f}"
        self.status.setText(status_txt)

        # ----------- 点云更新 --------------------------------
        with _slam_lock:
            data = _slam_latest
        if data is None:
            return

        xyz, pose = data
        if xyz.shape[0] == 0:
            return

        # 为 UI 速度进行下采样
        if xyz.shape[0] > 200_000:
            xyz = xyz[:: int(xyz.shape[0] / 200_000) + 1]

        # -------- 带有强调地标的连续梯度 ----------
        # 1) 高度（英尺）（相对于当前最小值，因此地面 = 0）
        z_ft = xyz[:, 2] * 3.28084
        z_rel = z_ft - z_ft.min()

        # 2) 在稍宽的 0–9 英尺范围内归一化到 0-1，以便红色
        #    仅出现在非常高的天花板上；可根据喜好调整 _SPAN_FT。
        _SPAN_FT = 9.0
        v = np.clip(z_rel / _SPAN_FT, 0.0, 1.0)

        # 3) gamma (<1) – 值越高 => 梯度越柔和
        _GAMMA = 0.35
        v_gamma = v ** _GAMMA

        # 4) 映射到颜色 – 使用 pyqtgraph 附带的感知均匀的 "turbo" 颜色图
        #    （如果找不到，则回退到简单的 HSV）。
        try:
            import pyqtgraph as pg  # type: ignore

            cmap = pg.colormap.get("turbo")
            colors = cmap.map(v_gamma, mode="float")  # 返回 Nx4 float
        except Exception:  # pragma: no cover – 最小回退
            # 回退到之前的 HSV 彩虹
            h = 0.66 * (1.0 - v_gamma)
            s = np.ones_like(h)
            val = np.ones_like(h)

            i = np.floor(h * 6).astype(int)
            f = h * 6 - i
            p = val * (1 - s)
            q = val * (1 - f * s)
            t = val * (1 - (1 - f) * s)

            r = np.choose(i % 6, [val, q, p, p, t, val])
            g = np.choose(i % 6, [t, val, val, q, p, p])
            b = np.choose(i % 6, [p, p, t, val, val, q])
            colors = np.stack([r, g, b, np.ones_like(r)], axis=1)

        self._scatter.setData(pos=xyz, size=1.0, color=colors)

        # ---------------- 2D 占据栅格图 -----------------------------
        self._update_2d_map(xyz, pose)

        # ---------------- 位姿可视化 ---------------------------
        if pose is not None and pose.shape == (4, 4):
            self._update_pose_axes(pose, xyz)

    # ------------------------------------------------------------------
    # Qt 原生键盘处理
    # ------------------------------------------------------------------

    # 与 keyboard_controller.py 相同的帮助常量
    _LIN_STEP = 0.05
    _ANG_STEP = 0.2

    # 速度上限 (m/s)，取决于用户是否按住 <Shift>。
    _SPEED_LIMIT_NORMAL = 0.6
    _SPEED_LIMIT_FAST = 1.2

    def _current_speed_limit(self) -> float:
        """返回用于当前按键状态的速度限制。"""

        return (
            self._SPEED_LIMIT_FAST
            if self._is_pressed("shift")
            else self._SPEED_LIMIT_NORMAL
        )

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        """将 *val* 限制在 ±*limit* 范围内。"""

        return max(-limit, min(limit, val))

    # ------------------------------------------------------------------
    #  按键覆盖层帮助程序
    # ------------------------------------------------------------------

    _DISPLAY_NAMES = {
        "space": "␣",
        "shift": "⇧",
        "esc": "⎋",
        "up_arrow": "↑",
        "down_arrow": "↓",
        "left_arrow": "←",
        "right_arrow": "→",
    }

    def _display_name(self, key: str) -> str:
        """返回 *key* 的简短用户友好表示。"""

        if key in self._DISPLAY_NAMES:
            return self._DISPLAY_NAMES[key]
        # 单个字符，如 'w', 'a', … – 为清晰起见显示大写。
        if len(key) == 1:
            return key.upper()
        return key

    def _update_key_overlay(self) -> None:
        """刷新屏幕上当前按下的键的列表。"""

        # 标题始终存在，因此覆盖层永远不会消失 – 帮助
        # 新手在第一次按键之前发现可用的控件。

        if self._pressed:
            # 如果当前有按键按下，确保任何正在运行的淡出
            # 被中止，并且覆盖层再次完全不透明。
            if self._fade_anim.state() == QtCore.QAbstractAnimation.Running:
                self._fade_anim.stop()
                self._keys_opacity.setOpacity(1.0)

            keys_txt = "  ".join(self._display_name(k) for k in sorted(self._pressed))
            self._keys_lbl.setText(keys_txt)
            self._key_overlay.adjustSize()
            self._key_overlay.move(10, 10)
            self._key_overlay.raise_()

        else:
            # 没有按键按下 → 如果不在运行中且当前
            # 文本不是占位符破折号，则开始淡出。
            if (
                self._fade_anim.state() != QtCore.QAbstractAnimation.Running
                and self._keys_lbl.text() != "–"
            ):
                self._fade_anim.stop()
                self._fade_anim.setStartValue(1.0)
                self._fade_anim.setEndValue(0.0)
                self._fade_anim.start()

    # Qt 为*所有*事件调用此方法，一旦我们将对象安装为过滤器
    def eventFilter(self, _obj, ev):
        from PySide6 import QtCore  # 局部导入以避免存根问题

        if ev.type() == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return False  # 让默认处理器运行

            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                    # 存储按键状态，以便 Drive 控制继续为
                    # 原始遥控键 (w/a/s/d/…) 工作。对于新
                    # 添加的箭头 + f/b 键，我们另外触发一次
                    # 基于推理的手臂运动。

                    self._pressed.add(name)

                    # ----------------------------------------------------
                    # 手臂运动推理触发器
                    # ----------------------------------------------------
                    self._maybe_arm_inference(name)

                    # ----------------------------------------------------
                    # 手 (Dex3) 打开/关闭触发器
                    # ----------------------------------------------------
                    self._maybe_hand_control(name)

                    return True  # 已处理

        elif ev.type() == QtCore.QEvent.KeyRelease:
            if ev.isAutoRepeat():
                return False

            key = ev.key()
            name = self._qt_key_name(key, ev.text())

            if name is None:
                return False

            # 从按下的集合中移除*所有*键，以保持 GUI 覆盖层
            # 准确。（在简化手部控制逻辑后，
            # 不再需要 p/o 的先前特殊情况。）

            self._pressed.discard(name)
            return True

        return False  # 其他事件继续正常处理

    # ------------------------------------------------------------------
    @staticmethod
    def _qt_key_name(key: int, text: str | None) -> str | None:
        """将 Qt 键码映射到我们的规范名称 (w,a,s,space,…)。"""
        from PySide6 import QtCore  # 局部

        mapping = {
            QtCore.Qt.Key_Space: "space",
            QtCore.Qt.Key_Escape: "esc",
            QtCore.Qt.Key_Z: "z",
            QtCore.Qt.Key_Shift: "shift",

            # 箭头键 – 映射到专用名称，以便我们稍后可以触发
            # 学习到的手臂运动推理。
            QtCore.Qt.Key_Up: "up_arrow",
            QtCore.Qt.Key_Down: "down_arrow",
            QtCore.Qt.Key_Left: "left_arrow",
            QtCore.Qt.Key_Right: "right_arrow",
        }

        if key in mapping:
            return mapping[key]

        if text:
            ch = text.lower()
            if ch in ("w", "a", "s", "d", "q", "e", "u", "j", "f", "b", "p", "o"):
                return ch

            # 手部控制键 (右 Dex3)
            if ch in ("g", "h"):
                return ch
        return None

    # ------------------------------------------------------------------
    def _is_pressed(self, name: str) -> bool:
        return name in self._pressed

    # ------------------------------------------------------------------
    #  推理引导的手臂运动
    # ------------------------------------------------------------------

    def _maybe_arm_inference(self, key_name: str) -> None:
        """
        当 *key_name* 对应于配置的箭头 / f / b 命令之一时，
        触发单个推理步骤。当前测量的活动臂关节角度
        作为*起始*位姿，而按下的键定义了输入到 MLP
        回归器的高级*方向*。然后通过重用
        *_on_arm_tick* 内现有的斜坡逻辑平滑地达到
        预测的*结束*关节目标。
        """

        # 将我们的规范键名映射到 ML 接受的方向字符串
        dir_map = {
            "up_arrow": "up",
            "down_arrow": "down",
            "left_arrow": "left",
            "right_arrow": "right",
            "f": "forward",
            "b": "back",
        }

        direction = dir_map.get(key_name)
        if direction is None:
            return  # 无关的键

        # ----------------------------------------------------------------
        #  先决条件 – 我们需要 SDK 发布者和至少一个
        #  反馈样本，以便知道当前的关节位置。
        # ----------------------------------------------------------------
        if self._arm_pub is None:
            # 手臂控制不可用 (SDK 缺失或早先失败)。
            return

        # 如果还没有反馈，我们无法根据确切的
        # 当前配置定制*起始*位姿。我们不完全放弃，
        # 而是回退到最后命令的角度，以便推理仍然有效 –
        # 这只是重新引入了基于反馈的初始化旨在避免的
        # 小初始跳变，但大大提高了在 LowState 主题
        # 不可用的机器人/PC 上的可用性。

        try:
            # 懒加载，以便在 ML 模型的依赖项缺失时 GUI 启动不会失败。
            # 它们是轻量级的 (joblib, numpy, pandas, scikit-learn)，
            # 因此导入时间可以忽略不计。
            from data.inference_arm import predict_end_positions, load_bundle  # type: ignore

            # 为每个手臂缓存加载的 bundle，以便重复推理是即时的。
            if not hasattr(self, "_arm_bundle_cache"):
                self._arm_bundle_cache = {}

            if self._active_arm not in self._arm_bundle_cache:
                try:
                    from pathlib import Path

                    bundle_path = Path(f"data/artifacts/{self._active_arm}-arm/arm_mlp.joblib")
                    self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                except TypeError as exc:
                    # ------------------------------------------------------------------
                    # 向后兼容性垫片：用较新版本的 scikit-learn 训练的模型
                    # 使用一个双参数帮助程序来 pickle NumPy RandomState 对象，
                    # 而旧版本的 NumPy 会拒绝该帮助程序。
                    # 猴子补丁构造函数，以便 *joblib.load()* 成功。
                    # ------------------------------------------------------------------
                    if "__randomstate_ctor" in str(exc):
                        try:
                            import numpy.random._pickle as _np_pickle  # type: ignore

                            def _rs_ctor(*_args, **_kwargs):
                                import numpy as _np

                                # 返回一个默认的 RandomState – 确切的
                                # 种子对于*推理*无关紧要。
                                return _np.random.RandomState()

                            _np_pickle.__randomstate_ctor = _rs_ctor

                            # 再试一次。
                            self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                        except Exception:
                            # 仍然失败 – 传播到下面的通用处理器。
                            raise
                    else:
                        raise
                except Exception:
                    # 如果显式路径失败（相对 cwd 等），则回退到帮助程序内的默认路径。
                    # 此处的任何错误都将由外部的 *except* 处理。
                    self._arm_bundle_cache[self._active_arm] = load_bundle()

            bundle = self._arm_bundle_cache[self._active_arm]

            # 当前*起始*关节角度（手臂特定）。优先使用
            # 测量的 LowState 样本；如果尚不可用，则回退
            # 到最后命令的值，以便回归器仍然
            # 接收到一个合理的位姿向量。

            start_joints = [
                self._joint_cur.get(j_idx, self._cmd_q.get(j_idx, 0.0))
                for j_idx in sorted(self._arm_joint_idx)
            ]

            preds = predict_end_positions(
                direction,
                start_joints,
                arm=self._active_arm,
                bundle=bundle,
            )

            # 构建新的一步位姿，以便 *_on_arm_tick* 以已经调整好的速度
            # 向新预测的目标斜坡。
            target_pose = [(self._waist_idx, self._cmd_q.get(self._waist_idx, 0.0))]
            target_pose += list(zip(sorted(self._arm_joint_idx), preds))

            self._pose_seq = [target_pose]
            self._seq_idx = 0

        except Exception as exc:  # pylint: disable=broad-except
            # 任何失败都不应使 GUI 崩溃 – 我们只报告它。
            import sys

            print("[run_g1_gui] 手臂推理失败:", exc, file=sys.stderr)

    # ------------------------------------------------------------------
    #  Dex3 手部控制
    # ------------------------------------------------------------------

    def _maybe_hand_control(self, key_name: str) -> None:
        """
        为 Dex3 手发出打开/关闭命令。

        键盘映射:
            g – 关闭 (抓握)
            h – 打开 (释放)
        """

        if self._dex3 is None:
            return  # 手部控制不可用

        # -------------------------------------------------------------
        # 简化的直接打开 (o) / 关闭 (p)
        # -------------------------------------------------------------

        if key_name in ("p", "o"):
            # 获取当前测量的关节角度 – 回退到最后的命令。
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)

            if key_name == "p":  # 关闭
                target = self._simple_closed_pose
                self._hand_mode = "closing"
                self._hand_target = list(target)
            else:  # 'o' 打开
                target = self._simple_open_pose
                self._hand_mode = "opening"
                self._hand_target = list(target)

            # 清除任何旧的序列控制 – 我们现在只依赖于
            # _hand_target。定时器循环将每滴答驱动 *self._hand_cmd_q*
            # 朝向该目标。

            self._hand_pose_seq.clear()
            self._hand_seq_idx = 0
            return

        # -------------------------------------------------------------
        #  传统 / 高级模式 (g/h 自适应、抓取等)
        # -------------------------------------------------------------

        # 确保我们有可用于传统处理的捕获姿势。
        if not getattr(self, "_hand_poses", None):
            return

        # ------------------------------------------------------------------
        # 1) 压力自适应抓取 (传统 – 按键 p / o)
        # ------------------------------------------------------------------
        if key_name in ("p", "o"):
            if self._hand_mode in ("closing", "holding") and key_name == "p":
                return  # 已经在关闭/保持
            if self._hand_mode == "opening" and key_name == "o":
                return  # 已经在打开

            if key_name == "p":
                self._log_hand.info("自适应抓取已触发")
                # 准备朝向关闭的序列，但启用自适应标志
                target_label = "closed"
                middle = self._hand_poses.get("middle")
                target = self._hand_poses.get(target_label)
                if middle is None or target is None:
                    return

                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)

                self._hand_pose_seq = [cur]  # 从当前开始
                self._hand_seq_idx = 0
                self._hand_mode = "adaptive"
            else:  # 'o' 完全释放到打开姿势
                print("[Dex3] 自适应打开", file=sys.stderr)
                target = self._hand_poses.get("open")
                if target is None:
                    return
                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)

                self._hand_pose_seq = [cur, target]
                self._hand_seq_idx = 1
                self._hand_mode = "opening"
            return

        # ------------------------------------------------------------------
        # 2) 连续抓取 / 释放 (新增 – 按键 g / h)
        # ------------------------------------------------------------------

        if key_name == "g":
            # 如果尚未激活，则开始新的连续抓取。
            if self._hand_mode != "grabbing":
                self._hand_mode = "grabbing"
                # 连续模式 – 所有关节立即向存储的
                # *闭合*姿势移动。没有阶段门控，因此每个关节
                # 即使其他关节被物体阻塞，也继续其自己的
                # 接近。
                # 清除任何先前活动的脚本化姿势序列，以便
                # 抓取逻辑可以完全控制关节目标。
                self._hand_pose_seq = []
                self._hand_seq_idx = 0
                self._log_hand.info("连续抓取已启动 (扭矩=%.2f N·m)", self._GRAB_TAU)
            return

        if key_name == "h":
            # 中止任何正在进行的抓取并执行正常的打开序列。
            target_label = "open"
            middle = self._hand_poses.get("middle")
            target = self._hand_poses.get(target_label)

            if target is None or middle is None:
                print("[run_g1_gui] 缺少 'open' 或 'middle' 的手部姿势。")
                return

            # 获取当前测量的关节角度 – 回退到最后的命令。
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)

            # 构建新的姿势序列：当前 -> 中间 -> 打开
            self._hand_pose_seq = [cur, middle, target]
            self._hand_seq_idx = 1
            self._hand_mode = "opening"
            self._log_hand.info("手部打开序列已排队。")
            return

        # 忽略所有其他键
        return

    # ------------------------------------------------------------------
    def _on_drive_tick(self):
        # 根据当前按下的键更新目标速度。

        lim = self._current_speed_limit()

        if self._is_pressed("w") and not self._is_pressed("s"):
            self._vx = self._clamp(self._vx + self._LIN_STEP, lim)
        elif self._is_pressed("s") and not self._is_pressed("w"):
            self._vx = self._clamp(self._vx - self._LIN_STEP, lim)
        else:
            self._vx = 0.0

        if self._is_pressed("q") and not self._is_pressed("e"):
            self._vy = self._clamp(self._vy + self._LIN_STEP, lim)
        elif self._is_pressed("e") and not self._is_pressed("q"):
            self._vy = self._clamp(self._vy - self._LIN_STEP, lim)
        else:
            self._vy = 0.0

        if self._is_pressed("a") and not self._is_pressed("d"):
            self._omega = self._clamp(self._omega + self._ANG_STEP, lim)
        elif self._is_pressed("d") and not self._is_pressed("a"):
            self._omega = self._clamp(self._omega - self._ANG_STEP, lim)
        else:
            self._omega = 0.0

        # 空格键强制完全停止
        if self._is_pressed("space"):
            self._vx = self._vy = self._omega = 0.0

        # 退出键
        if self._is_pressed("z"):
            if self._bot is not None:
                try:
                    self._bot.Damp()
                except Exception:
                    pass
            self.app.quit()
            return

        if self._is_pressed("esc"):
            if self._bot is not None:
                try:
                    self._bot.StopMove()
                    self._bot.ZeroTorque()
                except Exception:
                    pass
            self.app.quit()
            return

        # 每滴答发送命令 (10 Hz)
        if self._bot is not None:
            try:
                self._bot.Move(self._vx, self._vy, self._omega, continous_move=True)

                # 当没有运动命令时，保持机器人在静态平衡模式，
                # 当操作员请求运动时，切换到连续步态。
                # 这避免了当控制器保持在模式-1 但目标速度为零时
                # 有时观察到的“原地踏步”行为。
                desired_mode = 0 if (self._vx == self._vy == self._omega == 0.0) else 1
                if desired_mode != self._bal_mode:
                    try:
                        self._bot.SetBalanceMode(desired_mode)
                        self._bal_mode = desired_mode
                    except Exception:
                        pass
            except Exception as exc:
                print("[run_g1_gui] 移动失败:", exc, file=sys.stderr)
                self._bot = None  # 禁用进一步尝试

        # 发布到 HUD
        with _state_lock:
            _state["vel"] = (self._vx, self._vy, self._omega)

    # ------------------------------------------------------------------
    #  手臂控制帮助程序
    # ------------------------------------------------------------------

    def _on_arm_tick(self) -> None:
        """
        周期性发布器，驱动选定的手臂通过预定义的启动姿势。

        应用一个温和的每关节斜坡（每 40 毫秒 ``self._STEP`` rad），
        以便运动看起来平滑，没有任何突然的颠簸。
        """

        if self._arm_pub is None:
            return  # SDK 不可用或早先失败

        # 最好等待一个 *LowState* 反馈样本，以便命令的
        # 轨迹可以精确地从**当前**关节位置开始 –
        # 这可以防止在高刚度位置
        # 控制器接合时发生突然的跳变。然而，在某些部署中，
        # ``rt/lowstate`` 主题不可用，这以前意味着整个手臂例程
        # 永远**禁用**。为了在这种情况下保持 GUI 功能，
        # 我们现在在短暂的宽限期后回退到预初始化的*零*姿势，
        # 而不是提前退出。

        # 允许最多 2 秒的时间来接收第一个反馈包，增加等待时间以确保稳定初始化
        if not self._joint_cur and not getattr(self, "_no_fb_deadline", None):
            # 记住我们第一次注意到缺少反馈的时刻，并
            # 定义一个截止日期，之后我们停止等待 rt/lowstate
            # 并无论如何都运行手臂序列。

            self._no_fb_deadline = time.time() + 2.0  # 从 1.0 秒增加到 2.0 秒


        if not self._joint_cur and time.time() < self._no_fb_deadline:
            return  # 再等一会儿 LowState

        # ------------------------------------------------------------------
        #  确定活动序列步骤的当前*目标*位姿
        # ------------------------------------------------------------------

        if self._seq_idx >= len(self._pose_seq):
            target_pose = self._pose_seq[-1]  # 保持最终位姿
        else:
            target_pose = self._pose_seq[self._seq_idx]

        # ------------------------------------------------------------------
        #  将命令的关节角度朝目标位姿推进 - 使用更小的步长
        # ------------------------------------------------------------------

        all_reached = True
        # --------------------------------------------------------------
        #  从*测量*的关节位置进行一次性初始化（如果我们
        #  已经收到至少一个 LowState 样本）。在这里这样做
        #  – 在斜坡逻辑之前 – 确保我们从
        #  实际配置开始序列，从而避免
        #  由高刚度命令 0 rad 引起的突然跳变。
        # --------------------------------------------------------------

        if not self._initialised_from_state and self._joint_cur:
            for j_idx, q_val in self._joint_cur.items():
                self._cmd_q[j_idx] = q_val
            self._initialised_from_state = True

        # ----------------------------------------------------------------
        #  现在将每个命令的关节朝当前目标推进，使用更小的步长以确保平滑运动
        # ----------------------------------------------------------------
        for idx, tgt in target_pose:
            cur = self._cmd_q.get(idx, 0.0)
            diff = tgt - cur
            if abs(diff) <= self._SEQ_EPS:
                self._cmd_q[idx] = tgt
            else:
                # 使用更小的步长，并根据角度差异动态调整步长
                dynamic_step = min(self._STEP, abs(diff) * 0.1)  # 限制步长为差值的10%
                step = dynamic_step if diff > 0 else -dynamic_step
                if abs(step) > abs(diff):
                    step = diff  # 无过冲
                self._cmd_q[idx] = cur + step
                all_reached = False

        # 当所有都在容差范围内时，前进到下一个位姿。
        if all_reached and self._seq_idx < len(self._pose_seq):
            self._seq_idx += 1

        # ------------------------------------------------------------------
        #  构建并传输 LowCmd 消息 - 使用更温和的增益
        # ------------------------------------------------------------------

        try:
            # 为我们接触的每个关节应用命令的 q/kp/kd。其他关节
            # 保持其默认的 kp=kd=0 → 固件将它们视为被动
            # (Damp)。
            for idx, q in self._cmd_q.items():
                mc = self._arm_cmd.motor_cmd[idx]
                mc.q = q
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 40.0  # 从 60.0 降低到 40.0，减少刚度以实现更平滑的运动
                mc.kd = 1.0   # 从 1.5 降低到 1.0，减少阻尼

            # 发布前重新计算 CRC (固件要求)。
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)

            self._arm_pub.Write(self._arm_cmd)
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_g1_gui] 手臂发布失败:", exc, file=sys.stderr)


    # ------------------------------------------------------------------
    #  Dex3 手部控制帮助程序
    # ------------------------------------------------------------------

    def _on_hand_tick(self):
        """
        平滑地将 Dex3 手通过当前活动的位姿序列
        （由 *_maybe_hand_control* 构建）。使用与手臂帮助程序
        相同的每关节斜坡方法，以便运动看起来是渐进的，
        没有突然的颠簸。
        """

        if self._dex3 is None:
            return

        # 完全空闲时无工作（未设置目标且无序列）。
        if self._hand_mode not in ("closing", "opening", "grabbing") and not self._hand_pose_seq:
            return

        # ------------------------------------------------------------------
        #  简化的打开/关闭处理 – 朝 *self._hand_target* 移动
        # ------------------------------------------------------------------

        if self._hand_mode in ("closing", "opening"):
            target = list(self._hand_target)

        # ------------------------------------------------------------------
        #  连续抓取模式覆盖标准的位姿序列逻辑。
        # ------------------------------------------------------------------

        if self._hand_mode == "grabbing":
            # 连续收紧 – 将*所有*关节驱动到记录的
            # 闭合角度，每个关节根据当前
            # 差异以自己的速度进行。没有可能阻塞
            # 其他手指进一步运动的全局阶段。

            closed_pose = self._hand_closed_pose

            # 增量构建目标，以便*每个*关节独立地向
            # 其闭合角度前进 – 即使另一个手指
            # 已经被物体阻塞。因此，我们显式计算
            # 下一步，而不是依赖于之前使用的静态 *target =
            # closed_pose* 方法。

            target = list(self._hand_cmd_q)
            for j in range(7):
                cur = self._hand_cmd_q[j]
                tgt = closed_pose[j]
                diff = tgt - cur

                # 如果我们尚未达到目标（在 epsilon 内），则向闭合方向
                # 移动一步。这保证了即使在临时障碍物
                # 移开后，我们也会继续“尝试”关闭。
                if abs(diff) > self._SEQ_EPS:
                    step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                    # 避免过冲。
                    if abs(step) > abs(diff):
                        step = diff
                    target[j] = cur + step
                else:
                    target[j] = tgt

        else:
            # 从脚本化序列确定当前目标位姿。
            if self._hand_seq_idx >= len(self._hand_pose_seq):
                target = self._hand_pose_seq[-1] if self._hand_pose_seq else list(self._hand_cmd_q)
            else:
                target = self._hand_pose_seq[self._hand_seq_idx]

        # --------------------------------------------------------------
        #  自适应抓取 – 根据压力覆盖目标进程
        # --------------------------------------------------------------
        if self._hand_mode == "adaptive":
            # 读取压力
            state = self._dex3.read_state(timeout=0.0)
            pressures = []
            if state is not None:
                try:
                    for ps in state.press_sensor_state:
                        pressures.extend(list(ps.pressure))
                except Exception:
                    pass

            self._log_hand.debug("press=%s cmd=%s", pressures[:12], [round(q,2) for q in self._hand_cmd_q])

            # 决定调整：如果平均压力低于目标，则稍微关闭所有关节
            avg_press = (sum(pressures)/len(pressures)) if pressures else 0.0

            if avg_press < self._PRESS_TARGET:
                # 朝闭合位姿关闭关节
                target = self._hand_poses.get("closed", self._hand_cmd_q)
            else:
                target = self._hand_cmd_q  # 保持

            try:
                state = self._dex3.read_state(timeout=0.0)
                if state is not None:
                    # 压力数组的长度可能可变；将最后 4 个
                    # 元素视为指尖垫（拇指、食指、中指、无名指）。
                    # Unitree 文档提到 9 个传感器；使用可用的最大索引。
                    pressures = []
                    for ps in state.press_sensor_state:
                        try:
                            pressures.extend(list(ps.pressure))
                        except Exception:
                            pass

                    # 选择代表指尖的子集（如果需要，调整索引）
                    tip_idx = [2, 5, 8, 11] if len(pressures) >= 12 else list(range(len(pressures)))
                    cnt = sum(1 for i in tip_idx if i < len(pressures) and pressures[i] >= self._PRESS_THR)

                    if cnt >= self._PRESS_MIN_COUNT:
                        # 足够的接触 – 在当前姿势停止。
                        # 保持当前姿势并切换到保持模式
                        self._hand_pose_seq = [list(self._hand_cmd_q)]
                        self._hand_seq_idx = 1
                        self._hand_mode = "holding"
                        target = list(self._hand_cmd_q)
            except Exception as exc:
                print("[run_g1_gui] 自适应抓取传感器读取失败:", exc, file=sys.stderr)

        # 斜坡每个关节值
        all_reached = True
        for i, tgt in enumerate(target):
            cur = self._hand_cmd_q[i]
            diff = tgt - cur
            if abs(diff) <= self._HAND_STEP:
                self._hand_cmd_q[i] = tgt
            else:
                step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                self._hand_cmd_q[i] = cur + step
                all_reached = False

        # 如果当前已到达，则前进到下一个位姿。
        if self._hand_mode == "grabbing":
            # 无限期地继续运行，以便控制器在物体
            # 被移除后继续施加扭矩并跟踪任何释放/滑动。
            all_reached = False
        else:
            if all_reached and self._hand_seq_idx < len(self._hand_pose_seq):
                self._hand_seq_idx += 1

        # 当打开完成时，切换回空闲。
        if self._hand_mode == "opening" and all_reached and self._hand_seq_idx >= len(self._hand_pose_seq):
            self._hand_mode = "idle"

        # ------------------------------------------------------------------
        #  发布命令
        # ------------------------------------------------------------------
        try:
            cmd = self._dex3._make_zero_cmd()
            mins, maxs = self._dex3._limits()

            # 使用温和的增益。
            # 增加增益，以便所有关节，特别是高负载的基础
            # 关节，获得足够的权限以达到其目标角度。
            kp = 8.0  # 更强的比例增益，用于权威的位置保持
            kd = 1.5

            for i, q in enumerate(self._hand_cmd_q):
                mode = self._dex3._pack_mode(i, status=0x01, timeout=False)
                mc = cmd.motor_cmd[i]
                mc.mode = mode
                mc.kp = kp
                mc.kd = kd

                # 连续抓取模式的前馈扭矩，以便手指
                # 即使在达到期望的关节角度后也继续施加
                # 闭合力。
                if self._hand_mode in ("grabbing", "closing", "holding"):
                    mc.tau = max(0.3, self._GRAB_TAU) * self._close_dir[i]
                else:
                    mc.tau = 0.0

                # 在 URDF 限制内夹紧以避免故障。
                mc.q = max(min(q, maxs[i]), mins[i])

            ok = self._dex3._publish(cmd)

            if not ok:
                # 第一次检测到 – 警告用户一次。
                if not getattr(self, "_dex3_no_match_warned", False):
                    print(
                        "[Dex3] 警告 – rt/dex3 cmd 没有匹配的订阅者；正在重试…",
                        file=sys.stderr,
                    )
                    self._dex3_no_match_warned = True

                # 执行几次快速重试，以便启动后的前几个定时器
                # 滴答仍然至少提供一次成功的
                # 发布，一旦 DDS 主题匹配（模仿抓取/打开
                # 行为，循环约 20 次）。
                for _ in range(3):
                    if self._dex3._publish(cmd):
                        break
        except Exception as exc:  # pylint: disable=broad-except
            print("[run_g1_gui] Dex3 发布失败:", exc, file=sys.stderr)
            self._arm_pub = None

    # ------------------------------------------------------------------
    #  GUI 按钮 – 卸力上半身 & 腰部回中
    # ------------------------------------------------------------------

    def _on_damp_pressed(self):
        """
        使双臂被动 (kp=kd=0)，同时保持腿部处于
        平衡站立状态，然后将腰部偏航关节设置为 0 rad，使躯干
        再次朝前。在用户点击按钮时执行一次。
        """

        if self._arm_pub is None:
            print("[run_g1_gui] 卸力请求 – SDK 不可用")
            return

        # 停止自动手臂定时器，使其不再覆盖我们的
        # 一次性被动命令。
        try:
            if hasattr(self, "_arm_timer"):
                self._arm_timer.stop()
        except Exception:
            pass

        try:
            # 1) 为所有右臂关节设置 kp=kd=0，使其变软。
            # 使*双*臂变软，以便用户可以安全地交互，无论
            # 当前哪一侧是活动的。
            for idx in (*range(15, 22), *range(22, 29)):
                mc = self._arm_cmd.motor_cmd[idx]
                # 保持当前角度以避免刚度下降时突然跳动。
                mc.q = self._cmd_q.get(idx, 0.0)
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0

            # 2) 以正常刚度将腰部居中于 0 rad，使躯干
            #    保持朝前。
            waist_idx = getattr(self, "_waist_idx", 12)
            mc_w = self._arm_cmd.motor_cmd[waist_idx]
            mc_w.q = 0.0
            mc_w.dq = 0.0
            mc_w.tau = 0.0
            mc_w.kp = 60.0
            mc_w.kd = 1.5

            # 更新内部 cmd_q，以便后续调用（如果用户重新启用
            # 手臂序列）从居中姿势开始。
            self._cmd_q[waist_idx] = 0.0

            # 重新计算 CRC 并发布一次。
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            self._arm_pub.Write(self._arm_cmd)

            print("[run_g1_gui] 手臂已卸力，腰部已回中。")

        except Exception as exc:  # pylint: disable=broad-except
            print("[run_g1_gui] 卸力请求失败:", exc, file=sys.stderr)

    # ------------------------------------------------------------------
    #  地图点击 → 路径规划
    # ------------------------------------------------------------------

    def _on_map_click(self, ev):
        """
        处理 2D 占据栅格图上的鼠标点击。

        GraphicsScene 转发所有鼠标事件；我们将场景
        位置转换为*视图*坐标（在 ViewBox 变换后与我们的图像像素匹配），
        并从当前机器人位置到点击的目标开始路径规划运行，
        如果两者都在地图内。
        """

        import numpy as np  # 局部导入

        if self._occ_map is None or self._map_meta is None:
            return  # 地图尚未准备好

        # 转换点击 → 图像像素
        pos = ev.scenePos()
        # 映射到 ViewBox 坐标 (float)
        view_pt = self._map_vb.mapSceneToView(pos)
        gx, gy = int(view_pt.x()), int(view_pt.y())

        if not (0 <= gx < 480 and 0 <= gy < 480):
            return  # 画布外

        # 从最后存储的位姿元数据中定位当前机器人像素 (rx, ry)。
        rob_px = getattr(self, "_robot_px", None)
        if rob_px is None:
            return  # 没有机器人位置无法规划

        rx, ry = rob_px

        # 仅在*双击*左键时触发，以便常规单击
        # 保留用于在 ViewBox 内平移/缩放，并且不会在
        # 用户每次仅选择或拖动地图时启动
        # 昂贵的 A* 搜索。

        if not getattr(ev, "double", lambda: False)():  # pyqtgraph 帮助程序
            return

        # 规划路径 (返回 (x,y) 列表，包括起点+终点)
        path = self._plan_path(rx, ry, gx, gy, self._occ_map)

        if path is not None and len(path) > 1:
            self._path_px = path
        else:
            print("[run_g1_gui] 未找到到点击目标的路径。")

        # 触发立即刷新，以便用户无需等待
        # 下一个定时器滴答即可看到结果 – 安全，因为我们在 Qt 线程内。
        self._on_tick()

    # ------------------------------------------------------------------
    @staticmethod
    def _plan_path(sx: int, sy: int, gx: int, gy: int, occ: "np.ndarray") -> list[tuple[int, int]] | None:
        """
        在 2D 占据栅格上进行 A* 搜索，偏好宽阔的间隙。

        Args:
            sx (int): 起点 x 坐标 (图像坐标)。
            sy (int): 起点 y 坐标 (图像坐标)。
            gx (int): 终点 x 坐标 (图像坐标)。
            gy (int): 终点 y 坐标 (图像坐标)。
            occ (np.ndarray): 布尔数组，True 表示障碍物，形状为 (H, W)。

        Returns:
            list[tuple[int, int]] | None: [(x0,y0), …, (xn,yn)] 列表，如果无法到达则为 None。
        """

        import heapq
        import math
        import numpy as np  # 局部导入
        import cv2  # type: ignore

        h, w = occ.shape

        if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
            return None

        if occ[sy, sx] or occ[gy, gx]:
            return None  # 起点或终点被阻塞

        # 预计算距离变换 (像素 → 最近的障碍物)
        free_uint8 = (~occ).astype(np.uint8)  # 1 = 自由
        dist = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) or 1.0

        # 偏向搜索远离障碍物、朝向地图中心的权重。
        # 越大 => 对间隙的偏好越强。
        _BIAS = 3.0

        def cell_cost(x: int, y: int) -> float:
            d_norm = dist[y, x] / max_dist  # 0 … 1
            # 当 d_norm 高时（远离障碍物），成本较低
            return 1.0 + _BIAS * (1.0 - d_norm)

        # A* 搜索
        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, (sx, sy)))

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = { (sx, sy): 0.0 }

        def heuristic(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)

        while open_set:
            _, current = heapq.heappop(open_set)
            cx, cy = current

            if current == (gx, gy):
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # 探索邻居 (8-连通)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if occ[ny, nx]:
                        continue

                    step = math.hypot(dx, dy) * cell_cost(nx, ny)
                    tentative = g_score[current] + step

                    if tentative < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative
                        f = tentative + heuristic(nx, ny)
                        heapq.heappush(open_set, (f, (nx, ny)))

        return None  # 无法到达

    # ------------------------------------------------------------------
    def _on_quit(self):
        """处理退出事件。"""
        self._stop_evt.set()
        self._drive_timer.stop()
        if hasattr(self, "_arm_timer"):
            self._arm_timer.stop()
        for t in self._threads:
            t.join(timeout=1.0)

    # ------------------------------------------------------------------
    # 位姿轴帮助程序
    # ------------------------------------------------------------------

    def _update_pose_axes(self, pose: "np.ndarray", pts: "np.ndarray") -> None:
        """在机器人位姿处渲染一个小的 RGB 坐标系。"""

        import numpy as np  # 局部
        import pyqtgraph.opengl as gl  # 局部复用

        # 首先移除任何先前的坐标系
        for item in self._pose_items:
            self.gl_view.removeItem(item)
        self._pose_items.clear()

        # 从地图大小推导出一个合理的轴长度
        size = 0.5
        if pts.shape[0] > 0:
            span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            size = max(0.2, min(span * 0.03, 2.0))  # 3 % 对角线, 夹紧

        origin = pose[:3, 3]
        rot = pose[:3, :3]

        # -------------------------------------------------------------
        # 应用与 live_slam 使用的相同的 LiDAR 安装校正，
        # 以便在头部倾斜时视觉位姿轴看起来是水平的。
        # -------------------------------------------------------------
        try:
            from live_slam import _R_MOUNT as _LS_R_MOUNT  # type: ignore

            if _LS_R_MOUNT is not None:
                rot = rot @ _LS_R_MOUNT[:3, :3]
        except Exception:  # pragma: no cover – live_slam 在测试中缺失
            pass

        axes = {
            (1.0, 0.0, 0.0, 1.0): rot @ np.array([size, 0, 0]),  # X 红色
            (0.0, 1.0, 0.0, 1.0): rot @ np.array([0, size, 0]),  # Y 绿色
            (0.0, 0.0, 1.0, 1.0): rot @ np.array([0, 0, size]),  # Z 蓝色
        }

        for color, vec in axes.items():
            pts_arr = np.vstack([origin, origin + vec])
            item = gl.GLLinePlotItem(pos=pts_arr, color=color, width=2, antialias=True)
            self.gl_view.addItem(item)
            self._pose_items.append(item)

    # ------------------------------------------------------------------
    # 2D 占据栅格帮助程序
    # ------------------------------------------------------------------

    def _update_2d_map(self, xyz: "np.ndarray", pose: "np.ndarray" | None) -> None:
        """推导简单的鸟瞰占据栅格图，忽略地面。"""

        import numpy as np  # 局部
        import cv2  # type: ignore

        if xyz.shape[0] == 0:
            return  # 还没有数据

        # 从完整点云定义整体边界，以确保机器人始终在内
        min_x, max_x = float(xyz[:, 0].min()), float(xyz[:, 0].max())
        min_y, max_y = float(xyz[:, 1].min()), float(xyz[:, 1].max())

        span = max(max_x - min_x, max_y - min_y, 1e-6)
        scale = 470.0 / span  # 边距 5 px

        # 存储映射，以便点击处理器可以在像素 ↔ 世界之间转换
        # 注意：我们有意将*世界 y* → 水平像素和*世界 x*
        # → 垂直，以便“前进”（正 x）在
        # 占据栅格视图中显示为**向上**，这与自上而下
        # 地图的直观映射（北/上 = 前进）相匹配。
        self._map_meta = (min_x, min_y, scale)

        # 帮助闭包
        def world_to_px(xw: "np.ndarray", yw: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
            """向量化转换世界 (x, y) → 图像 (px, py)。"""

            # 水平：+y 向*右* – 如果您的物理
            # 坐标系不同，请在此处调整。我们*不*应用反转，因此正
            # 世界-Y 出现在右侧。垂直轴仍然翻转，
            # 以便前进 (+X) 是向上。
            px = ((yw - min_y) * scale + 5).astype(np.int32)
            py = ((xw - min_x) * scale + 5).astype(np.int32)
            py = 479 - py  # 翻转，以便 +x (前进) 在图像中是向上
            return px, py

        canvas = np.full((480, 480, 3), 30, dtype=np.uint8)

        # ------------------------------------------------------------------
        #  鲁棒的地面估计
        # ------------------------------------------------------------------
        # 简单地使用 *min(z)* 对单个噪声尖峰或
        # 偶尔比真实地板稍近的反射非常敏感。
        # 这种抖动导致动态阈值提升得恰到好处，
        # 以至于真正的地板点穿透并被显示为
        # 障碍物。

        # 1) 鲁棒的*瞬时*估计 – 取第 5 百分位数，以便
        #    少数虚假的低读数不会拉低地面估计。
        ground_z_inst = float(np.percentile(xyz[:, 2], 5.0))

        # 2) 随时间指数平滑 – 机器人在行走时会轻微倾斜，
        #    因此感知的地板距离会有些变化。保持一个
        #    缓慢适应的全局值，以便瞬间的颠簸不会
        #    每帧都使点在间隙阈值上下翻转。

        _ALPHA = 0.05  # 平滑因子 0 → 关闭, 1 → 无平滑
        if not hasattr(self, "_ground_z_smooth"):
            # 第一帧 – 直接从瞬时值开始。
            self._ground_z_smooth = ground_z_inst
        else:
            self._ground_z_smooth = (
                (1.0 - _ALPHA) * self._ground_z_smooth + _ALPHA * ground_z_inst
            )

        ground_z = float(self._ground_z_smooth)

        # ------------------------------------------------------------------
        #  自身传感器抑制 – 忽略几乎与
        #  LiDAR 平面齐平*且*非常靠近机器人中心的返回
        #  （主要是 G-1 自己的头部/安装支架）。完全相同的逻辑
        #  已经在 live_slam.handle_points() 中为 SLAM 前端运行，但我们
        #  在这里重复它，以清理可能
        #  在早期扫描中溜过的任何残留点，这些点仍然存在于
        #  聚合的局部地图中。
        # ------------------------------------------------------------------

        import os as _os

        try:
            _R_XY = float(_os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.30))
            _DZ = float(_os.environ.get("LIDAR_SELF_FILTER_Z", 0.24))
        except ValueError:
            _R_XY, _DZ = 0.08, 0.05

        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]

            diff = xyz - rob_pos  # 广播减法
            dist_xy = np.linalg.norm(diff[:, :2], axis=1)
            close = dist_xy < _R_XY
            near_plane = np.abs(diff[:, 2]) < _DZ
            keep_mask = ~(close & near_plane)

            if keep_mask.sum() != xyz.shape[0]:
                xyz = xyz[keep_mask]

        # 任何高于 (地面 + 间隙) 的点都被标记为障碍物。
        thresh = ground_z + self._clear_m

        # 间隙上方的障碍物
        pts = xyz[xyz[:, 2] > thresh]

        # 二进制占据栅格缓冲区 (True = 障碍物)
        occ = np.zeros((480, 480), dtype=bool)

        if pts.shape[0] > 0:
            x_obs, y_obs = pts[:, 0], pts[:, 1]
            px_obs, py_obs = world_to_px(x_obs, y_obs)
            valid = (px_obs >= 0) & (px_obs < 480) & (py_obs >= 0) & (py_obs < 480)
            px_obs, py_obs = px_obs[valid], py_obs[valid]

            # 更新占据栅格
            occ[py_obs, px_obs] = True

            # 在画布中绘制障碍物以进行可视化
            canvas[py_obs, px_obs] = (255, 255, 255)

        cv2.rectangle(canvas, (0, 0), (479, 479), (255, 255, 255), 1)

        # ---------------- 机器人箭头 ---------------------------------
        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            rx, ry = world_to_px(np.array([rob_pos[0]]), np.array([rob_pos[1]]))
            rx, ry = int(rx[0]), int(ry[0])

            # 持久化机器人像素，以便规划器知道从哪里开始
            self._robot_px = (rx, ry)

            # 保证机器人自己的单元格被视为*自由*，
            # 用于规划目的，即使基于距离的障碍物掩码
            # （地面以上阈值）由于激光雷达/
            # 重投影噪声而将其标记为被占据。我们还清除 8-邻域，
            # 以便规划器永远不会在第一步就被困住。

            rr0, rr1 = max(0, ry - 1), min(480, ry + 2)
            rc0, rc1 = max(0, rx - 1), min(480, rx + 2)
            occ[rr0:rr1, rc0:rc1] = False

            # 航向角 (偏航) 来自旋转矩阵 – 机器人 x 轴
            # 通过将正前方的一个点（机器人 +
            # 沿局部 +x 0.25 米）转换为像素坐标来推导端点。这避免了
            # 任何手动三角函数，一旦我们改变地图
            # 投影，这些函数就会失效。

            fwd_m = 0.25  # 世界空间中 25 厘米的箭头长度
            # 世界坐标中的前向向量是旋转的第一列
            fwd_vec = pose[:3, 0] * fwd_m
            tip_world = rob_pos + fwd_vec
            tx, ty = world_to_px(np.array([tip_world[0]]), np.array([tip_world[1]]))
            tx, ty = int(tx[0]), int(ty[0])

            cv2.arrowedLine(canvas, (rx, ry), (tx, ty), (0, 255, 0), 2, tipLength=0.8)

        # ------------------------------------------------------------------
        #  覆盖规划的路径（如果有）
        # ------------------------------------------------------------------

        if self._path_px and len(self._path_px) > 1:
            cv2.polylines(
                canvas,
                [np.array(self._path_px, dtype=np.int32)],
                isClosed=False,
                color=(0, 0, 255),
                thickness=2,
            )

            # 用一个实心红点突出显示目标，以便无论
            # 缩放级别如何，它都保持可见。列表中的最后一个元素
            # 始终是点击的目标像素。
            gx, gy = self._path_px[-1]
            cv2.circle(canvas, (gx, gy), 4, (0, 0, 255), -1)

        # 为规划器存储占据栅格（我们复制以避免别名）
        self._occ_map = occ.copy()

        # 更新交互式图像 – pyqtgraph 期望图像的
        # 第一个轴是 *y*。生成的画布已经遵循该
        # 约定，因此我们可以逐字传递它。

        try:
            self._map_img.setImage(canvas, levels=(0, 255))
        except Exception:
            # 如果 pyqtgraph 由于某种原因（例如无头
            # 测试运行器上缺少 OpenGL）未能初始化，则回退到非交互式 QLabel。
            # 我们保留旧代码路径作为优雅降级。
            px = self._numpy_to_qpix(canvas)
            if px and hasattr(self, "rgb_lbl"):  # 确保 GUI 已构建
                from PySide6 import QtWidgets as _QtW  # 局部导入

                if not hasattr(self, "_legacy_lbl"):
                    self._legacy_lbl = _QtW.QLabel(alignment=QtCore.Qt.AlignCenter)
                    self._legacy_lbl.setMinimumSize(640, 320)
                    self._map_vb.hide()
                    # 在布局中替换 map_view – 安全，因为此
                    # 代码仅在罕见的回退路径中执行一次。
                    self.map_view.setParent(None)
                    self.rgb_lbl.parentWidget().layout().addWidget(self._legacy_lbl)

                self._legacy_lbl.setPixmap(px)

    # ------------------------------------------------------------------
    def run(self):
        """运行 GUI 主循环。"""
        self.win.show()
        sys.exit(self.app.exec())


# ------------------------------------------------------------------------


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0", help="连接到 Unitree G-1 的网卡")
    parser.add_argument(
        "--clear",
        type=float,
        default=18.0,
        help="在检测到的地板上方，一个点被标记为障碍物之前的间隙（英寸）",
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right"],
        default="right",
        help="启动时控制哪只手臂 (默认: right)",
    )

    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="right",
        help="连接了哪个 Dex3 手 (默认: right。当连接了右手单元时，"
        "传递 --hand right。",
    )

    parser.add_argument(
        "--grip-force",
        type=float,
        dest="grip_force",
        metavar="N·m",
        default=0.3,
        help="在连续抓取期间应用的前馈扭矩（约 N·m）(默认: 0.3)",
    )
    args = parser.parse_args()

    window = G1Windows(
        args.iface,
        args.clear,
        hand=args.hand,
        grip_force=args.grip_force,
    )
    window._active_arm = args.arm
    # 使用命令行选择重新初始化手臂变量。
    window._arm_selector.setCurrentIndex(0 if args.arm == "left" else 1)

    try:
        window._configure_arm_variables()
    except Exception as exc:  # pragma: no cover – config failures should not crash
        print("[run_g1_gui] 初始手臂切换失败:", exc, file=sys.stderr)

    window.run()


if __name__ == "__main__":
    main()