#!/usr/bin/env python3
"""
Unitree G1 GUI 控制器 - 模块化版本

这是原始 run_g1_gui.py 的模块化重构版本。
主要功能已拆分到 gui/ 目录中的单独模块中。

使用方法:
    python3 run_g1_gui_modular.py [选项]

示例:
    python3 run_g1_gui_modular.py
    python3 run_g1_gui_modular.py --ground-clear 6.0 --hand right
"""

import argparse
import sys
from pathlib import Path

# 确保 gui 包可以导入
sys.path.insert(0, str(Path(__file__).parent))

from gui.logging import setup_logging
from gui.main_window import G1Windows


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Unitree G1 机器人 GUI 控制器 (模块化版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                            # 使用默认设置
  %(prog)s --ground-clear 6.0         # 设置地面间隙为 6 英寸
  %(prog)s --hand right               # 使用右手
  %(prog)s --grip-force 0.5           # 设置抓取力为 0.5 N·m
        """,
    )

    parser.add_argument(
        "--network-interface",
        default="eth0",
        help="连接到机器人的网络接口 (默认: eth0)",
    )

    parser.add_argument(
        "--ground-clear",
        type=float,
        default=4.0,
        help="地面间隙 (英寸) - 超过此高度的点被视为障碍物 (默认: 4.0)",
    )

    parser.add_argument(
        "--hand",
        choices=["left", "right"],
        default="left",
        help="物理连接的 Dex3 手 (默认: left)",
    )

    parser.add_argument(
        "--grip-force",
        type=float,
        default=0.3,
        help="连续抓取模式的前馈扭矩 (N·m) (默认: 0.3)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)",
    )

    return parser.parse_args()


def main():
    """主入口点。"""
    args = parse_args()

    # 设置日志记录
    setup_logging(level=args.log_level)

    print(f"[run_g1_gui] 启动 Unitree G1 GUI 控制器 (模块化版本)")
    print(f"[run_g1_gui] 网络接口: {args.network_interface}")
    print(f"[run_g1_gui] 地面间隙: {args.ground_clear} 英寸")
    print(f"[run_g1_gui] 手部: {args.hand}")
    print(f"[run_g1_gui] 抓取力: {args.grip_force} N·m")

    try:
        # 创建并运行主窗口
        window = G1Windows(
            iface=args.network_interface,
            ground_clear_in=args.ground_clear,
            hand=args.hand,
            grip_force=args.grip_force,
        )
        
        # 显示控制说明
        print("\n" + "="*60)
        print("控制说明:")
        print("  移动:")
        print("    W/S     - 前进/后退")
        print("    A/D     - 左/右移动")
        print("    Q/E     - 左/右旋转")
        print("  手臂:")
        print("    H       - 手臂回到默认位置")
        print("    B       - 卸力手臂 & 腰部回中")
        print("  手部:")
        print("    G       - 快速抓取")
        print("    O       - 打开手部")
        print("    C       - 关闭手部")
        print("    F       - 指向手势")
        print("    T       - 点赞手势")
        print("  其他:")
        print("    点击2D地图 - 路径规划")
        print("    ESC     - 退出程序")
        print("="*60 + "\n")

        # 运行 GUI
        window.run()

    except KeyboardInterrupt:
        print("\n[run_g1_gui] 用户中断，正在退出...")
        sys.exit(0)
    except Exception as exc:
        print(f"[run_g1_gui] 启动失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
