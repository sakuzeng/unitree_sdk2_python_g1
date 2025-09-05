"""
Unitree G-1 机器人键盘遥控程序

通过 WASD 风格的按键实现对机器人的远程操作。

控制按键
---------
    W / S : 向前/向后移动
    A / D : 向左/向右转向
    Q / E : 向左/向右平移
    Space : 停止 (所有速度清零)
    Z     : 进入阻尼模式并退出
    Esc   : 紧急停止 (零力矩) 并退出

工作原理
----------
程序通过 `pynput` 库实时监听键盘的按下和释放事件，持续更新目标速度。
速度指令以 10Hz 的频率发送给机器人，保证了控制的连续性。
同时，使用 `curses` 库在终端上显示一个简易的平视显示器 (HUD)，
实时展示当前的速度状态。
"""
from __future__ import annotations

import argparse
import time
import curses

# 导入 pynput 用于跨平台键盘监听。
# pynput 在桌面环境 (X11, Win32, Quartz) 下无需特殊权限。
# 在 Wayland 环境下可能回退到读取 /dev/input，此时可能需要相应权限。
try:
    from pynput.keyboard import Listener, Key, KeyCode  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "需要 'pynput' 依赖包。\n"
        "请使用以下命令安装: pip install pynput"
    ) from exc

# 导入机器人启动序列函数
from hanger_boot_sequence import hanger_boot_sequence


# --- 参数默认值 ---
LIN_STEP = 0.05      # m/s: 每次按键调整的线速度步长
ANG_STEP = 0.2       # rad/s: 每次按键调整的角速度步长
SEND_PERIOD = 0.1    # seconds (10 Hz): 向机器人发送速度指令的周期


def clamp(value: float, limit: float = 0.4) -> float:
    """
    将速度值限制在 [-limit, +limit] 范围内，防止超出安全阈值。

    Args:
        value (float): 输入的速度值。
        limit (float): 速度的绝对值上限。

    Returns:
        float: 限制后的速度值。
    """
    return max(-limit, min(limit, value))


def drive_loop(stdscr: "curses._CursesWindow", bot) -> None:
    """
    键盘遥控主循环。

    使用 curses 在终端绘制 HUD，并通过 pynput 监听键盘事件来控制机器人运动。

    Args:
        stdscr ("curses._CursesWindow"): curses 窗口对象，用于绘制 HUD。
        bot: 已初始化的 LocoClient 实例。
    """
    # 初始化 Curses HUD
    curses.cbreak()
    stdscr.nodelay(True)

    # 初始化速度变量 (vx:前后, vy:左右, omega:旋转)
    vx = vy = omega = 0.0
    last_send = 0.0

    # --- pynput 键盘监听器设置 ---
    pressed_keys: set[object] = set()  # 存储当前按下的键

    def _on_press(key):
        """按键按下时的回调函数，将键存入 pressed_keys 集合。"""
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.add(key.char.lower())
        else:
            pressed_keys.add(key)

    def _on_release(key):
        """按键释放时的回调函数，从 pressed_keys 集合中移除键。"""
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.discard(key.char.lower())
        else:
            pressed_keys.discard(key)

    listener = Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    def key(name: str) -> bool:
        """辅助函数，检查特定键是否被按下。"""
        if name == "space":
            return Key.space in pressed_keys
        if name == "esc":
            return Key.esc in pressed_keys
        return name in pressed_keys

    try:
        while True:
            # 1. 根据按键状态更新目标速度
            if key("w") and not key("s"):
                vx = clamp(vx + LIN_STEP)
            elif key("s") and not key("w"):
                vx = clamp(vx - LIN_STEP)
            else:
                vx = 0.0

            if key("q") and not key("e"):
                vy = clamp(vy + LIN_STEP)
            elif key("e") and not key("q"):
                vy = clamp(vy - LIN_STEP)
            else:
                vy = 0.0

            if key("a") and not key("d"):
                omega = clamp(omega + ANG_STEP)
            elif key("d") and not key("a"):
                omega = clamp(omega - ANG_STEP)
            else:
                omega = 0.0

            # 紧急停止按键
            if key("space"):
                vx = vy = omega = 0.0

            # 退出按键
            if key("z"):
                bot.Damp()  # 进入阻尼模式
                break

            if key("esc"):
                bot.StopMove()
                bot.ZeroTorque()  # 进入零力矩模式（紧急）
                break

            # 2. 按固定频率向机器人发送指令并更新 HUD
            now = time.time()
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now

                # 刷新 HUD 显示
                stdscr.erase()
                stdscr.addstr(0, 0, "Hold keys to drive – Z: 进入阻尼模式并退出 — Esc: 紧急停止 (零力矩) 并退出")
                stdscr.addstr(1, 0, "Ctrl+C中断会进入阻尼模式，请确认机器人已正确悬挂")
                stdscr.addstr(3, 0, f"vx: {vx:+.2f}  vy: {vy:+.2f}  omega: {omega:+.2f}")
                stdscr.refresh()

            # 短暂休眠以降低 CPU 使用率
            time.sleep(0.005)

    finally:
        # 确保在退出 curses 上下文之前停止监听器线程
        listener.stop()


def main() -> None:
    """
    程序主入口。

    解析命令行参数，执行机器人启动序列，并启动键盘遥控循环。
    """
    parser = argparse.ArgumentParser(description="Unitree G-1 键盘遥控程序")
    parser.add_argument("--iface", default="eth0", help="连接到机器人的网络接口")
    args = parser.parse_args()

    # 执行启动序列，返回初始化完成的 LocoClient 实例
    bot = hanger_boot_sequence(iface=args.iface)
    
    # 使用 curses.wrapper 启动终端 HUD 并进入遥控主循环
    curses.wrapper(drive_loop, bot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被中断 – 发送 Damp 指令...")
        try:
            # 尝试让机器人进入安全的阻尼模式
            bot.Damp()  # type: ignore[name-defined]
        except Exception:
            pass