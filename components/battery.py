"""
电池监控组件
"""
from __future__ import annotations

import time
import threading
from .base import ComponentBase, StateManager


class BatteryComponent(ComponentBase):
    """电池监控组件"""
    
    def __init__(self, state_manager: StateManager, interface: str):
        super().__init__("Battery")
        self.state_manager = state_manager
        self.interface = interface
    
    def _run(self) -> None:
        """电池监控主循环"""
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
            
            def _publish(soc_val: int = None, voltage: float = None):
                if soc_val is not None:
                    self.state_manager.set("soc", soc_val)
                if voltage is not None:
                    self.state_manager.set("voltage", voltage)
            
            def _attempt_sub(name: str, msg_type, cb):
                try:
                    sub = ChannelSubscriber(name, msg_type)
                    sub.Init(cb, 50)
                    return True
                except Exception:
                    return False
            
            # 尝试订阅 Unitree Go/G1 LowState
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
            
            # 尝试订阅人形机器人 HG BmsState
            if not ok:
                try:
                    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_
                    
                    def _cb_hg(msg: BmsState_):
                        _publish(int(msg.soc))
                    
                    ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)
                except Exception:
                    ok = False
            
            # 如果失败，尝试初始化工厂
            if not ok:
                try:
                    ChannelFactoryInitialize(0, self.interface)
                except Exception:
                    pass
                
                # 重试订阅
                if 'LowState_' in locals():
                    ok = _attempt_sub("rt/lowstate", LowState_, _cb_go)
                if not ok and 'BmsState_' in locals():
                    ok = _attempt_sub("rt/bmsstate", BmsState_, _cb_hg)
            
            if not ok:
                raise RuntimeError("无法订阅任何电池SOC主题")
            
            # 监控循环
            while self.is_running():
                time.sleep(0.5)
                
        except Exception as e:
            print(f"[{self.name}] 电池监控失败: {e}")