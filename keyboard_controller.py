'''
simple WASD-style tele-op for Unitree G-1.
Controls
---------
    W / S : forward / backward velocity 
    A / D : yaw left / right (turn)
    Q / E : lateral left / right (optional, G-1 supports side-step)
    Space : stop (zero velocities)
    Z      : Damp (soft) and exit
    Esc    : emergency stop & exit (ZeroTorque)

Velocities are applied continuously – every key-press adjusts the target
values which are sent to the robot at 10 Hz.
'''
from __future__ import annotations

import argparse


# * Continuous command _while a key is physically held_. As soon as the key is
#   released the corresponding velocity is reset to **zero**.
# * Supports holding several keys together – e.g. **W + A** to move forward
#   while turning left.
#
# This requires real “key-up” events which the `curses` module cannot provide.
# We now use the lightweight third-party `pynput` package to poll the current
# key state.  It communicates with the X-server (Linux), Win32, or Quartz and
# therefore works for normal users on typical desktop sessions (no sudo).
# Curses is kept only for drawing the tiny on-screen HUD.
#
# When running under Wayland `pynput` might still fall back to reading
# `/dev/input/event*`; in that corner-case you’d again need the permissions or
# group/udev tweaks previously mentioned.
# ------------------------------------------------------------------------

import time

# Third-party ---------------------------------------------------------------
# * 用于在终端显示 HUD（Head-Up Display，抬头显示）速度和提示信息。
import curses

# We now use the cross-platform `pynput` library which reads key events via the
# X server / Win32 API / Quartz, so it works unprivileged on desktop Linux,
# macOS and Windows.  On Wayland sessions `pynput` falls back to /dev/input and
# may again need the permissions discussed earlier—but on X11 (the default on
# many distros) it works out-of-the-box.

# * 导入 pynput 键盘监听库
try:
    from pynput.keyboard import Listener, Key, KeyCode  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'pynput' package is required for keyboard_controller.py.\n"
        "Install with:  pip install pynput"
    ) from exc

from hanger_boot_sequence import hanger_boot_sequence


# ---------------------------------------------------------------------------
# Parameter defaults
# ---------------------------------------------------------------------------
# * 每次按键调整线速度的步长（0.05米/秒）;每次按键调整角速度的步长（0.2弧度/秒）;向机器人发送速度指令的周期（0.1秒，即10Hz）
LIN_STEP = 0.05  # m/s per press
ANG_STEP = 0.2   # rad/s per press

SEND_PERIOD = 0.1  # seconds (10 Hz)

# * 用于限制速度值在 -limit 到 +limit 之间，防止速度超出机器人安全范围。
def clamp(value: float, limit: float = 0.4) -> float:
    return max(-limit, min(limit, value))


def drive_loop(stdscr: "curses._CursesWindow", bot) -> None:
    # * Curses HUD 初始化;设置终端为非阻塞模式，便于实时刷新 HUD（速度和提示信息）。
    curses.cbreak()
    stdscr.nodelay(True)

    # - vx:前后速度; vy:左右速度; omega:角速度
    vx = vy = omega = 0.0
    # * 上次发送指令时间
    last_send = 0.0

    # ------------------------------------------------------------------
    # Keyboard listener setup (pynput)
    # 使用 pynput 监听键盘按下和释放事件，实时更新 pressed_keys 集合。
    # ------------------------------------------------------------------

    pressed_keys: set[object] = set()  # holds Key / single-char strings
    # * 键盘监听器的按键按下事件回调
    # * 如果按下的是普通字符键（如字母、数字），则将其转换为小写后加入 pressed_keys 集合。
    # * 如果按下的是特殊键（如空格、ESC），则直接将 Key 对象加入 pressed_keys 集合。
    def _on_press(key):  # noqa: D401 – tiny helper
        """Callback – store the key object / char in *pressed_keys*."""
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.add(key.char.lower())
        else:
            pressed_keys.add(key)
    # * 键盘监听器的按键释放事件回调
    # * 如果释放的是普通字符键（如字母、数字），则将其小写形式从 pressed_keys 集合中移除。
    # * 如果释放的是特殊键（如空格、ESC），则直接移除对应的 Key 对象。
    def _on_release(key):
        if isinstance(key, KeyCode) and key.char is not None:
            pressed_keys.discard(key.char.lower())
        else:
            pressed_keys.discard(key)

    listener = Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    # * 通用判断函数
    def key(name: str) -> bool:  # helper similar to keyboard.is_pressed
        if name == "space":
            return Key.space in pressed_keys
        if name == "esc":
            return Key.esc in pressed_keys
        return name in pressed_keys

    try:
        while True:
            # ------------------------------------------------------------------
            # 1. 基于当前状态更新g1速度
            # ------------------------------------------------------------------

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

            # ! 紧急停止
            if key("space"):
                vx = vy = omega = 0.0

            if key("z"):
                bot.Damp()
                break

            if key("esc"):
                bot.StopMove()
                bot.ZeroTorque()
                break

            # ------------------------------------------------------------------
            # 3.以目标速率更新hud
            # ------------------------------------------------------------------
            now = time.time()
            # * 每隔 SEND_PERIOD 秒（即 0.1 秒，10Hz），将当前速度（vx, vy, omega）发送给机器人，保持运动的实时性和流畅性。
            if now - last_send >= SEND_PERIOD:
                bot.Move(vx, vy, omega, continous_move=True)
                last_send = now

                stdscr.erase()
                stdscr.addstr(0, 0, "Hold keys to drive – Z: quit  ESC: e-stop")
                stdscr.addstr(2, 0, f"vx: {vx:+.2f}  vy: {vy:+.2f}  omega: {omega:+.2f}")
                stdscr.refresh()

            # A very small sleep keeps CPU usage civilised (<1 % on typical PCs).
            time.sleep(0.005)

    finally:
        # Ensure the listener thread is stopped before leaving curses context.
        listener.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0", help="network interface connected to robot")
    args = parser.parse_args()

    # Boot sequence – returns initialised LocoClient in FSM-200
    bot = hanger_boot_sequence(iface=args.iface)
    # * 使用 curses.wrapper 启动终端 HUD，并进入键盘遥控主循环 drive_loop，实现实时键盘控制机器人运动。
    curses.wrapper(drive_loop, bot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – sending Damp …")
        try:
            bot.Damp()  # type: ignore[name-defined]
        except Exception:
            pass