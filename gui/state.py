"""GUI 共享状态管理。"""

import threading
from typing import Any, Tuple

# 从主堆栈导入基础状态
from run_g1_stack import _state, _state_lock

# SLAM 特定的共享状态
_slam_latest: Tuple[Any, Any] | None = None  # (xyz ndarray, pose ndarray)
_slam_lock = threading.Lock()
