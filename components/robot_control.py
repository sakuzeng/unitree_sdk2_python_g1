"""
机器人控制组件
"""
from __future__ import annotations

import time
from typing import Any, Set

from .base import ComponentBase, StateManager


class RobotControlComponent(ComponentBase):
    """机器人键盘遥控组件"""
    
    def __init__(self, state_manager: StateManager, interface: str = "eth0"):
        super().__init__("RobotControl")
        self.state_manager = state_manager
        self.interface = interface
        self.lin_step = 0.05
        self.ang_step = 0.2
        self.send_period = 0.1
    
    def _run(self) -> None:
        """机器人控制主循环"""
        try:
            from hanger_boot_sequence import hanger_boot_sequence
            from pynput.keyboard import Listener, Key, KeyCode
            
            bot = hanger_boot_sequence(iface=self.interface)
            
            vx = vy = omega = 0.0
            last_send = 0.0
            pressed_keys: Set[Any] = set()
            
            def on_press(key):
                if isinstance(key, KeyCode) and key.char:
                    pressed_keys.add(key.char.lower())
                else:
                    pressed_keys.add(key)
            
            def on_release(key):
                if isinstance(key, KeyCode) and key.char:
                    pressed_keys.discard(key.char.lower())
                else:
                    pressed_keys.discard(key)
            
            def is_pressed(name: str) -> bool:
                if name == "space":
                    return Key.space in pressed_keys
                if name == "esc":
                    return Key.esc in pressed_keys
                return name in pressed_keys
            
            listener = Listener(on_press=on_press, on_release=on_release)
            listener.start()
            
            try:
                while self.is_running():
                    # 速度控制逻辑
                    vx = self._update_velocity(vx, is_pressed("w"), is_pressed("s"), self.lin_step)
                    vy = self._update_velocity(vy, is_pressed("q"), is_pressed("e"), self.lin_step)
                    omega = self._update_velocity(omega, is_pressed("a"), is_pressed("d"), self.ang_step)
                    
                    if is_pressed("space"):
                        vx = vy = omega = 0.0
                    
                    # 退出控制
                    if is_pressed("z"):
                        bot.Damp()
                        break
                    if is_pressed("esc"):
                        bot.StopMove()
                        bot.ZeroTorque()
                        break
                    
                    # 发送控制指令
                    now = time.time()
                    if now - last_send >= self.send_period:
                        bot.Move(vx, vy, omega, continous_move=True)
                        last_send = now
                        
                        # 更新状态
                        self.state_manager.set("vel", (vx, vy, omega))
                    
                    time.sleep(0.005)
            
            finally:
                listener.stop()
        
        except Exception as e:
            print(f"[{self.name}] 组件异常: {e}")
    
    def _update_velocity(self, current: float, forward: bool, backward: bool, step: float) -> float:
        """更新速度值"""
        if forward and not backward:
            return self._clamp(current + step)
        elif backward and not forward:
            return self._clamp(current - step)
        else:
            return 0.0
    
    def _clamp(self, value: float, limit: float = 0.6) -> float:
        """速度限制"""
        return max(-limit, min(limit, value))