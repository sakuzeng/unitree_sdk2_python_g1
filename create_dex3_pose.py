#!/usr/bin/env python3
"""
创建 Dex3 灵巧手姿势配置文件

根据宇树 Dex3-1 文档中的关节限位和典型手势，生成标准的手部姿势配置。
关节顺序：thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
"""

import csv
import json
from pathlib import Path


def create_hand_poses_config():
    """创建手部姿势配置文件"""
    
    # 确保数据目录存在
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 基于 run_g1_gui.py 中实际使用的姿势值
    hand_poses = {
        # GUI 中使用的预定义姿势
        "open": {
            "description": "完全打开，与 GUI 中 _simple_open_pose 对应",
            "joints": [-0.15717165172100067, -0.41322529315948486, 0.02846403606235981,
                      0.17782948911190033, -0.025226416066288948, 0.17983606457710266,
                      -0.027690349146723747]
        },
        
        "closed": {
            "description": "完全闭合，与 GUI 中 _simple_closed_pose 对应",
            "joints": [0.07452802360057831, 0.9478388428688049, 1.766921877861023,
                      -1.4442411661148071, -1.4384468793869019, -1.5298594236373901,
                      -1.4153316020965576]
        },
        
        # 标准中间位置
        "middle": {
            "description": "中间位置，准备抓取",
            "joints": [0.0, 0.65, 0.8, 0.7, 0.7, 0.7, 0.7]
        },
        
        # 精确抓取 (捏取)
        "pinch": {
            "description": "拇指和食指捏取小物体",
            "joints": [0.5, 1.0, 0.8, -0.1, -0.1, 1.0, 0.8]
        },
        
        # 指向手势
        "point": {
            "description": "食指指向，其他手指闭合",
            "joints": [0.0, 1.4, 1.2, 1.2, 1.2, -0.1, -0.1]
        },
        
        # 胜利手势 (V字)
        "peace": {
            "description": "食指和中指伸出，拇指和其他手指闭合",
            "joints": [0.0, 1.4, 1.2, -0.1, -0.1, -0.1, -0.1]
        },
        
        # OK手势
        "ok": {
            "description": "拇指和食指形成圆圈",
            "joints": [0.5, 1.0, 0.8, 1.2, 1.2, 1.0, 0.8]
        },
        
        # 休息位置
        "rest": {
            "description": "自然休息位置，轻微弯曲",
            "joints": [0.0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
        },
        
        # 轻握状态
        "light_grip": {
            "description": "轻握，适合易碎物品",
            "joints": [0.0, 0.6, 0.9, 0.8, 0.8, 0.8, 0.8]
        },
        
        # 强握状态
        "strong_grip": {
            "description": "强力抓握，适合重物",
            "joints": [0.0, 1.2, 1.4, 1.2, 1.2, 1.2, 1.2]
        },
        
        # 展示手势 (手掌朝外)
        "show": {
            "description": "展示手势，手掌朝外",
            "joints": [-0.1, -0.3, 0.1, 0.0, -0.1, 0.0, -0.1]
        },
        
        # 握拳
        "fist": {
            "description": "握拳状态",
            "joints": [0.2, 1.5, 1.5, 1.4, 1.4, 1.4, 1.4]
        }
    }
    
    return hand_poses


def save_csv_format(hand_poses: dict, file_path: Path):
    """保存为 CSV 格式，兼容现有的加载函数"""
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # CSV 表头 - 与 GUI 中的 _load_hand_poses 完全兼容
        fieldnames = ['label', 'description'] + [f'joint{i}' for i in range(7)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # 写入每个姿势
        for pose_name, pose_data in hand_poses.items():
            row = {
                'label': pose_name,
                'description': pose_data['description']
            }
            
            # 添加关节角度
            for i, joint_value in enumerate(pose_data['joints']):
                row[f'joint{i}'] = joint_value
            
            writer.writerow(row)
    
    print(f"CSV 格式配置已保存到: {file_path}")


def save_json_format(hand_poses: dict, file_path: Path):
    """保存为 JSON 格式，便于程序读取"""
    
    # 转换为更简洁的格式
    json_data = {
        "metadata": {
            "description": "Dex3-1 灵巧手标准姿势配置",
            "joint_count": 7,
            "joint_order": [
                "thumb_0 (拇指旋转)",
                "thumb_1 (拇指弯曲1)", 
                "thumb_2 (拇指弯曲2)",
                "middle_0 (中指弯曲1)",
                "middle_1 (中指弯曲2)", 
                "index_0 (食指弯曲1)",
                "index_1 (食指弯曲2)"
            ],
            "unit": "radians",
            "coordinate_system": "unitree_hg_msg_dds",
            "created_by": "dex3_pose_generator"
        },
        "poses": {}
    }
    
    for pose_name, pose_data in hand_poses.items():
        json_data["poses"][pose_name] = {
            "description": pose_data["description"],
            "joints": pose_data["joints"]
        }
    
    with open(file_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"JSON 格式配置已保存到: {file_path}")


def validate_joint_limits(hand_poses: dict):
    """验证关节角度是否在安全限位内"""
    
    # Dex3-1 关节限位
    joint_limits = [
        (-1.57, 1.57),   # thumb_0
        (-0.5, 1.8),     # thumb_1  
        (-0.2, 1.6),     # thumb_2
        (-0.2, 1.6),     # middle_0
        (-0.2, 1.6),     # middle_1
        (-0.2, 1.6),     # index_0
        (-0.2, 1.6),     # index_1
    ]
    
    warnings = []
    
    for pose_name, pose_data in hand_poses.items():
        joints = pose_data['joints']
        
        for i, (joint_val, (min_limit, max_limit)) in enumerate(zip(joints, joint_limits)):
            if not (min_limit <= joint_val <= max_limit):
                warning = (
                    f"警告: 姿势 '{pose_name}' 关节{i} "
                    f"值 {joint_val:.3f} 超出限位 [{min_limit:.3f}, {max_limit:.3f}]"
                )
                warnings.append(warning)
    
    if warnings:
        print("关节限位验证结果:")
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("✓ 所有姿势关节角度均在安全限位内")
    
    return len(warnings) == 0


def create_calibration_sequence():
    """创建校准序列配置"""
    
    calibration_sequence = {
        "description": "Dex3 校准序列，用于验证关节运动范围",
        "sequence": [
            {
                "step": 1,
                "pose": "rest",
                "duration": 2.0,
                "description": "初始休息位置"
            },
            {
                "step": 2, 
                "pose": "open",
                "duration": 3.0,
                "description": "完全打开测试"
            },
            {
                "step": 3,
                "pose": "middle", 
                "duration": 2.0,
                "description": "中间位置"
            },
            {
                "step": 4,
                "pose": "closed",
                "duration": 3.0, 
                "description": "完全闭合测试"
            },
            {
                "step": 5,
                "pose": "pinch",
                "duration": 2.0,
                "description": "精确抓取测试"
            },
            {
                "step": 6,
                "pose": "rest",
                "duration": 2.0,
                "description": "返回休息位置"
            }
        ]
    }
    
    cal_file = Path("data/dex3_calibration_sequence.json")
    with open(cal_file, 'w', encoding='utf-8') as f:
        json.dump(calibration_sequence, f, indent=2, ensure_ascii=False)
    
    print(f"校准序列配置已保存到: {cal_file}")


def main():
    """主函数"""
    print("=== Dex3 灵巧手姿势配置生成器 ===")
    
    # 创建姿势配置
    hand_poses = create_hand_poses_config()
    
    # 验证关节限位
    validate_joint_limits(hand_poses)
    
    # 保存为 CSV 格式 (兼容现有加载函数)
    csv_file = Path("data/hand_states.csv")
    save_csv_format(hand_poses, csv_file)
    
    # 保存为 JSON 格式 (便于程序读取)
    json_file = Path("data/dex3_poses.json") 
    save_json_format(hand_poses, json_file)
    
    # 创建校准序列
    create_calibration_sequence()
    
    print(f"\n生成了 {len(hand_poses)} 个标准手部姿势:")
    for pose_name, pose_data in hand_poses.items():
        print(f"  - {pose_name}: {pose_data['description']}")
    
    print("\n配置文件已就绪，可在 GUI 中使用。")
    print("\n使用方法:")
    print("1. 运行此脚本生成配置文件")
    print("2. 在 GUI 中将自动加载 data/hand_states.csv")
    print("3. 支持的手势键盘控制:")
    print("   - p: 压力自适应抓取")
    print("   - o: 打开手部")
    print("   - g: 连续抓取模式")
    print("   - h: 释放并打开")


if __name__ == "__main__":
    main()