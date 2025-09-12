#!/usr/bin/env python3
"""
G1 机器人手臂控制组件

集成ML推理、平滑运动控制和安全保护功能
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

from .base import ComponentBase, StateManager


class G1ArmControlComponent(ComponentBase):
    """G1机器人手臂控制组件"""
    
    def __init__(self, state_manager: StateManager, interface: str = "eth0"):
        """
        初始化手臂控制组件
        
        Args:
            state_manager: 状态管理器
            interface: 网络接口名称
        """
        super().__init__("ArmControl", state_manager)
        self.interface = interface
        
        # 手臂控制状态
        self.active_arm = "right"
        self.control_enabled = False
        self.ml_enabled = False
        
        # 初始化控制组件
        self._init_arm_control_components()
        self._init_robot_connections()
        self._init_ml_inference()
        
        # 控制线程
        self._control_thread: Optional[threading.Thread] = None
        self._control_running = False
        
    def _init_arm_control_components(self):
        """初始化手臂控制组件"""
        # 关节索引配置
        self._WAIST_YAW_IDX = 12
        self._LEFT_IDX = {idx: 0 for idx in range(15, 22)}
        self._RIGHT_IDX = {idx: 0 for idx in range(22, 29)}
        
        # 初始化关节状态
        self._arm_joint_idx: List[int] = []
        self._waist_idx: int = self._WAIST_YAW_IDX
        self._cmd_q: Dict[int, float] = {}
        self._joint_cur: Dict[int, float] = {}
        
        # 控制参数
        self._SEQ_EPS = 0.01
        self._STEP = 0.008  # 较小步长实现平滑运动
        self._seq_idx = 0
        self._initialised_from_state = False
        
        # 位姿序列
        self._pose_seq: List[List[Tuple[int, float]]] = []
        
        # 配置当前手臂
        self._configure_arm_variables()
        
        # 日志记录器
        self._log = logging.getLogger("g1_arm_control")
        
    def _init_robot_connections(self):
        """初始化机器人连接"""
        self._arm_pub = None
        self._ls_sub = None
        
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            
            # DDS 消息和 CRC
            self._crc = CRC()
            self._arm_cmd = unitree_hg_msg_dds__LowCmd_()
            self._arm_cmd.motor_cmd[29].q = 1  # 启用 arm_sdk
            
            # 创建发布者
            self._arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._arm_pub.Init()
            
            # 启动 LowState 订阅
            self._start_lowstate_subscription()
            
            # 准备非活动手臂
            self._prepare_inactive_arm()
            
            self.control_enabled = True
            self.logger.info(f"手臂控制初始化成功，控制 {self.active_arm} 臂")
            
        except Exception as exc:
            self.logger.error(f"手臂控制初始化失败: {exc}")
            self._arm_pub = None
            self.control_enabled = False
            
    def _init_ml_inference(self):
        """初始化ML推理系统"""
        try:
            # 懒加载ML模型，避免启动时失败
            self._arm_bundle_cache = {}
            self.ml_enabled = True
            self.logger.info("ML推理系统初始化成功")
        except Exception as exc:
            self.logger.warning(f"ML推理系统初始化失败: {exc}")
            self.ml_enabled = False
            
    def _configure_arm_variables(self):
        """配置手臂变量"""
        # 设置当前手臂的关节索引
        if self.active_arm == "left":
            self._arm_joint_idx = list(range(15, 22))
        else:
            self._arm_joint_idx = list(range(22, 29))
        
        # 初始化命令字典
        for idx in self._arm_joint_idx:
            self._cmd_q.setdefault(idx, 0.0)
        self._cmd_q.setdefault(self._WAIST_YAW_IDX, 0.0)
        
        # 构建启动位姿序列
        if self.active_arm == "right":
            self._pose_seq = [
                [
                    (self._WAIST_YAW_IDX, 0.0),
                    (22, -0.023), (23, -0.225), (24, +0.502), (25, +1.317),
                    (26, +0.185), (27, +0.125), (28, -0.182),
                ],
                [
                    (self._WAIST_YAW_IDX, 0.0),
                    (22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
                    (26, +0.240), (27, -0.771), (28, -0.176),
                ],
            ]
        else:
            self._pose_seq = [
                [
                    (self._WAIST_YAW_IDX, 0.0),
                    (15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
                    (19, -0.379), (20, -0.852), (21, -0.019),
                ]
            ]
        
        # 重置进度
        self._seq_idx = 0
        self._initialised_from_state = False
        
    def _start_lowstate_subscription(self):
        """启动 LowState 订阅线程"""
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
                    self.logger.info("LowState 订阅成功")
                    return
                    
                except Exception:
                    continue
                    
            self.logger.warning("LowState 订阅失败，将使用回退模式")
        
        threading.Thread(target=_init_ls_sub, daemon=True).start()
        
    def _prepare_inactive_arm(self):
        """为非活动手臂设置待机姿势"""
        def _apply_pose_once(pose: List[Tuple[int, float]]):
            for j_idx, q_val in pose:
                mc = self._arm_cmd.motor_cmd[j_idx]
                mc.q = q_val
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 60.0
                mc.kd = 1.5

        # 为另一只手臂构建准备位姿
        if self.active_arm == "right":
            ready_other = [
                (15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672),
                (19, -0.379), (20, -0.852), (21, -0.019),
            ]
        else:
            ready_other = [
                (22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691),
                (26, +0.240), (27, -0.771), (28, -0.176),
            ]

        _apply_pose_once(ready_other)

        # 重新计算 CRC 并立即传输
        self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
        try:
            if self._arm_pub:
                self._arm_pub.Write(self._arm_cmd)
        except Exception:
            pass
            
    def _on_arm_tick(self):
        """周期性手臂控制更新"""
        if not self.control_enabled or self._arm_pub is None:
            return
        
        # 等待反馈的超时处理
        if not self._joint_cur and not getattr(self, "_no_fb_deadline", None):
            self._no_fb_deadline = time.time() + 2.0
            
        if not self._joint_cur and time.time() < self._no_fb_deadline:
            return
        
        # 确定当前目标位姿
        if self._seq_idx >= len(self._pose_seq):
            target_pose = self._pose_seq[-1] if self._pose_seq else []
        else:
            target_pose = self._pose_seq[self._seq_idx]
        
        # 从测量的关节位置进行一次性初始化
        if not self._initialised_from_state and self._joint_cur:
            for j_idx, q_val in self._joint_cur.items():
                self._cmd_q[j_idx] = q_val
            self._initialised_from_state = True
            self.logger.info("已从 LowState 初始化关节位置")
        
        # 将每个关节朝目标推进
        all_reached = True
        for idx, tgt in target_pose:
            cur = self._cmd_q.get(idx, 0.0)
            diff = tgt - cur
            if abs(diff) <= self._SEQ_EPS:
                self._cmd_q[idx] = tgt
            else:
                # 动态调整步长
                dynamic_step = min(self._STEP, abs(diff) * 0.1)
                step = dynamic_step if diff > 0 else -dynamic_step
                if abs(step) > abs(diff):
                    step = diff
                self._cmd_q[idx] = cur + step
                all_reached = False
        
        # 当所有关节到达目标时，前进到下一个位姿
        if all_reached and self._seq_idx < len(self._pose_seq):
            self._seq_idx += 1
            self.logger.debug(f"前进到位姿序列 {self._seq_idx}/{len(self._pose_seq)}")
        
        # 构建并发送 LowCmd 消息
        try:
            for idx, q in self._cmd_q.items():
                mc = self._arm_cmd.motor_cmd[idx]
                mc.q = q
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 40.0  # 温和的增益
                mc.kd = 1.0
            
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            self._arm_pub.Write(self._arm_cmd)
            
        except Exception as exc:
            self.logger.error(f"手臂命令发送失败: {exc}")
            
    def _maybe_arm_inference(self, direction: str) -> bool:
        """
        使用ML推理预测手臂运动
        
        Args:
            direction: 运动方向 ("up", "down", "left", "right", "forward", "back")
            
        Returns:
            bool: 推理是否成功
        """
        if not self.ml_enabled or not self.control_enabled:
            return False
            
        try:
            # 懒加载ML模型
            if self.active_arm not in self._arm_bundle_cache:
                from data.inference_arm import load_bundle
                bundle_path = Path(f"data/artifacts/{self.active_arm}-arm/arm_mlp.joblib")
                if not bundle_path.exists():
                    self.logger.warning(f"ML模型文件不存在: {bundle_path}")
                    return False
                self._arm_bundle_cache[self.active_arm] = load_bundle(bundle_path)
            
            # 获取当前关节角度作为输入
            start_joints = [
                self._joint_cur.get(j_idx, self._cmd_q.get(j_idx, 0.0))
                for j_idx in sorted(self._arm_joint_idx)
            ]
            
            # 调用ML推理
            from data.inference_arm import predict_end_positions
            preds = predict_end_positions(
                direction,
                start_joints,
                arm=self.active_arm,
                bundle=self._arm_bundle_cache[self.active_arm]
            )
            
            # 构建目标位姿序列
            target_pose = [(self._waist_idx, self._cmd_q.get(self._waist_idx, 0.0))]
            target_pose += list(zip(sorted(self._arm_joint_idx), preds))
            
            # 设置新的运动目标
            self._pose_seq = [target_pose]
            self._seq_idx = 0
            
            self.logger.info(f"ML推理成功: {direction} -> {len(preds)} 个关节目标")
            return True
            
        except Exception as exc:
            self.logger.error(f"ML推理失败: {exc}")
            return False
            
    def _control_loop(self):
        """控制循环主函数"""
        self.logger.info("手臂控制循环启动")
        
        while self._control_running:
            try:
                self._on_arm_tick()
                
                # 更新状态管理器
                self.state_manager.set("active_arm", self.active_arm)
                self.state_manager.set("arm_joints", dict(self._joint_cur))
                self.state_manager.set("arm_cmd_joints", dict(self._cmd_q))
                self.state_manager.set("arm_pose_reached", self._seq_idx >= len(self._pose_seq))
                
                time.sleep(0.02)  # 50Hz 控制频率
                
            except Exception as exc:
                self.logger.error(f"控制循环错误: {exc}")
                time.sleep(0.1)
                
        self.logger.info("手臂控制循环停止")
        
    def switch_arm(self, arm: str) -> bool:
        """
        切换控制的手臂
        
        Args:
            arm: 手臂选择 ("left" 或 "right")
            
        Returns:
            bool: 切换是否成功
        """
        if arm not in ["left", "right"]:
            self.logger.error("手臂选择必须是 'left' 或 'right'")
            return False
        
        if arm == self.active_arm:
            return True
        
        self.logger.info(f"切换手臂控制: {self.active_arm} -> {arm}")
        self.active_arm = arm
        self._configure_arm_variables()
        return True
        
    def move_to_pose(self, pose_sequence: List[List[Tuple[int, float]]]) -> bool:
        """
        移动到指定位姿序列
        
        Args:
            pose_sequence: 位姿序列
            
        Returns:
            bool: 设置是否成功
        """
        if not self.control_enabled:
            return False
            
        if not pose_sequence:
            self.logger.warning("空位姿序列")
            return False
            
        self._pose_seq = pose_sequence
        self._seq_idx = 0
        self.logger.info(f"设置新位姿序列，共 {len(pose_sequence)} 个位姿")
        return True
        
    def set_single_pose(self, joint_angles: Dict[int, float]) -> bool:
        """
        设置单个目标位姿
        
        Args:
            joint_angles: 关节角度字典
            
        Returns:
            bool: 设置是否成功
        """
        target_pose = [(idx, angle) for idx, angle in joint_angles.items()]
        return self.move_to_pose([target_pose])
        
    def damp_arm(self) -> bool:
        """
        使手臂进入卸力模式
        
        Returns:
            bool: 卸力是否成功
        """
        if not self.control_enabled or self._arm_pub is None:
            self.logger.warning("卸力请求失败 - 控制不可用")
            return False
        
        try:
            # 为所有手臂关节设置 kp=kd=0
            for idx in (*range(15, 22), *range(22, 29)):
                mc = self._arm_cmd.motor_cmd[idx]
                mc.q = self._cmd_q.get(idx, 0.0)
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0
            
            # 腰部归零
            waist_idx = self._WAIST_YAW_IDX
            mc_w = self._arm_cmd.motor_cmd[waist_idx]
            mc_w.q = 0.0
            mc_w.dq = 0.0
            mc_w.tau = 0.0
            mc_w.kp = 60.0
            mc_w.kd = 1.5
            
            self._cmd_q[waist_idx] = 0.0
            
            # 发送命令
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            self._arm_pub.Write(self._arm_cmd)
            
            self.logger.info("手臂已卸力，腰部已归零")
            return True
            
        except Exception as exc:
            self.logger.error(f"卸力操作失败: {exc}")
            return False
            
    def handle_keyboard_input(self, key: str) -> bool:
        """
        处理键盘输入进行手臂控制
        
        Args:
            key: 按键名称
            
        Returns:
            bool: 是否处理了该按键
        """
        # 手臂切换
        if key == "tab":
            new_arm = "left" if self.active_arm == "right" else "right"
            return self.switch_arm(new_arm)
        
        # ML推理控制
        direction_map = {
            "up": "up",
            "down": "down",
            "left": "left", 
            "right": "right",
            "f": "forward",
            "b": "back",
        }
        
        if key in direction_map and self.ml_enabled:
            return self._maybe_arm_inference(direction_map[key])
        
        # 卸力模式
        if key == "z":
            return self.damp_arm()
        
        return False
        
    def get_status(self) -> Dict[str, Any]:
        """获取手臂控制状态"""
        return {
            "active_arm": self.active_arm,
            "control_enabled": self.control_enabled,
            "ml_enabled": self.ml_enabled,
            "pose_reached": self._seq_idx >= len(self._pose_seq) if self._pose_seq else True,
            "current_joints": dict(self._joint_cur),
            "command_joints": dict(self._cmd_q),
            "pose_sequence_progress": f"{self._seq_idx}/{len(self._pose_seq)}"
        }
        
    def start(self) -> None:
        """启动手臂控制组件"""
        if not self.control_enabled:
            self.logger.warning("手臂控制未启用，跳过启动")
            return
            
        self._control_running = True
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        self.logger.info("手臂控制组件已启动")
        
    def stop(self) -> None:
        """停止手臂控制组件"""
        self._control_running = False
        
        if self._control_thread:
            self._control_thread.join(timeout=1.0)
            
        # 停止订阅
        if hasattr(self, '_ls_sub') and self._ls_sub:
            try:
                self._ls_sub.Close()
            except Exception:
                pass
                
        self.logger.info("手臂控制组件已停止")