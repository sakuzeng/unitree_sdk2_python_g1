"""
组件基类定义
"""
from __future__ import annotations

import abc
import threading
import time
from typing import Any, Dict, Optional


class ComponentBase(abc.ABC):
    """组件基类，定义统一的生命周期接口"""
    
    def __init__(self, name: str):
        self.name = name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    @abc.abstractmethod
    def _run(self) -> None:
        """组件主运行逻辑，子类必须实现"""
        pass
    
    def start(self) -> None:
        """启动组件"""
        if self._running:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{self.name}-thread")
        self._thread.start()
        self._running = True
        print(f"[{self.name}] 组件已启动")
    
    def stop(self, timeout: float = 2.0) -> None:
        """停止组件"""
        if not self._running:
            return
        
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print(f"[{self.name}] 警告: 组件未能在 {timeout}s 内停止")
        
        self._running = False
        print(f"[{self.name}] 组件已停止")
    
    def is_running(self) -> bool:
        """检查组件是否正在运行"""
        return self._running and not self._stop_event.is_set()


class StateManager:
    """共享状态管理器"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> None:
        """设置状态值"""
        with self._lock:
            self._state[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        with self._lock:
            return self._state.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新状态"""
        with self._lock:
            self._state.update(updates)