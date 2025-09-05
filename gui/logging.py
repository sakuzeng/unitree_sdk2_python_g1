"""GUI 日志记录模块。"""

import logging
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _StreamToLogger:
	"""将 stdout/stderr 重定向到日志记录器的流。"""
	
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


def setup_logging() -> logging.Logger:
	"""初始化根日志记录器并返回 GUI 特定的子日志记录器。"""

	# 总是从一个新的日志文件开始，这样我们只捕获当前运行的输出。
	# 当文件大小超过 *maxBytes* 时，它会被轮转为 ``run_g1_gui.log.1``
	# (旧版本会根据 *backupCount* 被丢弃)。

	# 将日志文件放在项目目录内，使其保持自包含。
	log_dir = Path(__file__).resolve().parent.parent / "logs"
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
	warnings.filterwarnings(
		"ignore",
		message=r"A builtin ctypes object gave a PEP3118 format string that does not match its itemsize.*",
		category=RuntimeWarning,
		module=r"numpy\.ctypeslib",
	)

	# 重定向所有对 sys.stdout / sys.stderr 的写入，以便来自
	# 第三方库的零散打印也最终进入日志（同时通过上面的
	# StreamHandler 出现在控制台中）。
	sys.stdout = _StreamToLogger(logging.INFO)  # type: ignore[assignment]
	sys.stderr = _StreamToLogger(logging.ERROR)  # type: ignore[assignment]

	# 我们自己消息的子日志记录器 – 从根继承处理器。
	return logging.getLogger("g1_gui")
