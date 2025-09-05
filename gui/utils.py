"""GUI 工具函数。"""

import numpy as np
from typing import Optional
from PySide6 import QtGui


def numpy_to_qpix(bgr: Optional[np.ndarray]) -> Optional[QtGui.QPixmap]:
	"""将 NumPy BGR 图像转换为 Qt QPixmap。"""
	if bgr is None or bgr.dtype != np.uint8:
		return None
	h, w, _ = bgr.shape
	qimg = QtGui.QImage(bgr.data.tobytes(), w, h, 3 * w, QtGui.QImage.Format_BGR888)
	return QtGui.QPixmap.fromImage(qimg.copy())


def clamp(val: float, limit: float) -> float:
	"""将 *val* 限制在 ±*limit* 范围内。"""
	return max(-limit, min(limit, val))
