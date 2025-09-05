# G1-edu

## 1. 遥控器操控

悬挂开机：短按一次再长按2s。

进入阻尼 L2+B,进入预备状态L2+up

下降悬挂绳使双足着地，进入常规运控R1+x

走跑运控R2+a

start控制G1在战力和行走之间切换

意外状况L2+b(长按5s)进入阻尼状态缓慢倒地。

悬挂关机：悬挂好之后进入阻尼状态L2+b,电源短按一次，长按2s

## 2. 灯带含义

| 运控模式 | LED灯带颜色 |
| --- | --- |
| **正常运控（start）** | **蓝灯常亮** |
| **阻尼模式（damp）** | **橙灯常亮** |
| 落座模式 | 绿灯常亮 |
| 调试模式 | 黄灯常亮 |
| **零力矩模式（zero Torque）** | **紫灯常亮** |
| **预备模式（stand up）** | **深蓝色灯常亮** |
| 异常状态 | 红灯常亮 |

## 3. G1-edu基本构造

### 3.1 颈部接口

### 3.2 LIVOX-MID360激光雷达

雷达ip: `192.168.123.120`

### 3.3 D435i深度相机

### 3.4 PC2开发计算单元信息

初始用户unitree,密码123

| 参数 | 开发计算单元（ PC2 ) |
| --- | --- |
| 型号 | Jetson Orin NX |
| CPU | Arm® Cortex®-A78AE |
| 内核数 | 8 |
| 线程数 | 8 |
| 最大睿频频率 | 2GHz |
| 显存 | 16G |
| 内存 | 16G |
| 缓存 | 2MB L2 + 4MB L3 |
| 存储 | 2T |
| 英特尔® 图像处理单元 | 6.0 |
| GPU | 搭载 32 个 Tensor Core 的 1024 核 NVIDIA Ampere 架构 GPU |
| 显卡最大动态频率 | 918MHz |
| 高斯和神经加速器 | 3.0 |
| 英特尔®深度学习提升 | 是 |
| 英特尔®Adaptix™ 技术 | 是 |
| 英特尔®超线程技术 | 是 |
| 指令集 | 64bit |
| OpenGL | 4.6 |
| OpenCL | 3.0 |
| DirectX | 12.1 |
| IP 地址 | 192.168.123.164 |

### **3.5 关节电机**

| **关节序号** | **关节名称 (英文)** | **关节名称 (中文)** | **限位(弧度)** |
| --- | --- | --- | --- |
| 0 | L_LEG_HIP_PITCH | 左腿髋关节俯仰 | -2.5307~2.8798 |
| 1 | L_LEG_HIP_ROLL | 左腿髋关节侧摆 | -0.5236~2.9671 |
| 2 | L_LEG_HIP_YAW | 左腿髋关节偏航 | -2.7576~2.7576 |
| 3 | L_LEG_KNEE | 左腿膝关节 | -0.087267~2.8798 |
| 4 | L_LEG_ANKLE_PITCH | 左腿踝关节俯仰 | -0.87267~0.5236 |
| 5 | L_LEG_ANKLE_ROLL | 左腿踝关节侧摆 | -0.2618~0.2618 |
| 6 | R_LEG_HIP_PITCH | 右腿髋关节俯仰 | -2.5307~2.8798 |
| 7 | R_LEG_HIP_ROLL | 右腿髋关节侧摆 | -2.9671~0.5236 |
| 8 | R_LEG_HIP_YAW | 右腿髋关节偏航 | -2.7576~2.7576 |
| 9 | R_LEG_KNEE | 右腿膝关节 | -0.087267~2.8798 |
| 10 | R_LEG_ANKLE_PITCH | 右腿踝关节俯仰 | -0.87267~0.5236 |
| 11 | R_LEG_ANKLE_ROLL | 右腿踝关节侧摆 | -0.2618~0.2618 |
| 12 | WAIST_YAW | 腰部偏航 | -2.618~2.618 |
| 13 | WAIST_ROLL | 腰部侧摆 | -0.52~0.52 |
| 14 | WAIST_PITCH | 腰部俯仰 | -0.52~0.52 |
| 15 | L_SHOULDER_PITCH | 左肩俯仰 | -3.0892~2.6704 |
| 16 | L_SHOULDER_ROLL | 左肩侧摆 | -1.5882~2.2515 |
| 17 | L_SHOULDER_YAW | 左肩偏航 | -2.618~2.618 |
| 18 | L_ELBOW | 左肘关节 | -1.0472~2.0944 |
| 19 | L_WRIST_ROLL | 左腕侧摆 | -1.972222054~1.972222054 |
| 20 | L_WRIST_PITCH | 左腕俯仰 | -1.614429558~1.614429558 |
| 21 | L_WRIST_YAW | 左腕偏航 | -1.614429558~1.614429558 |
| 22 | R_SHOULDER_PITCH | 右肩俯仰 | -3.0892~2.6704 |
| 23 | R_SHOULDER_ROLL | 右肩侧摆 | -2.2515~1.5882 |
| 24 | R_SHOULDER_YAW | 右肩偏航 | -2.618~2.618 |
| 25 | R_ELBOW | 右肘关节 | -1.0472~2.0944 |
| 26 | R_WRIST_ROLL | 右腕侧摆 | -1.972222054~1.972222054 |
| 27 | R_WRIST_PITCH | 右腕俯仰 | -1.614429558~1.614429558 |
| 28 | R_WRIST_YAW | 右腕偏航 | -1.614429558~1.614429558 |

# G1参数说明

## FSM（有限状态机，Finite State Machine）

### fsm ID

| **ID** | **名称/动作** | 备注 |
| --- | --- | --- |
| 0 | **Zero Torque** | **零力矩模式** Motors off, gravity-droop allowed. |
| 1 | **Damp** | **阻尼模式** Motors apply viscous damping only – legs are “soft”. |
| 2 | Squat | **蹲下** Low squat posture (static). |
| 3 | Sit | **落座** Dog-sit (hips flexed). |
| 4 | **Stand-up** | **预备模式（锁定站立）在阻尼模式后使用**Raises body to nominal height; used after Damp. |
| 200 | **Start (balance / gait)** | **主运控 （平衡站立/连续步态）**Main balance controller & gait planner; enables walking. |
| 702 | Lie-to-Stand | From lying on the back. |
| 706 | Squat-to-Stand-up | Smooth stand from deep squat. |

### fsm Mode

| **Mode** | 备注 |
| --- | --- |
| 0 | 脚负荷，站立状态 Feet loaded, **static stand**. |
| 1 | 脚负荷，移动状态  Feet loaded, **dynamic / gait active**. |
| 2 | 脚未负荷  Feet **un-loaded** (hanging or airborne). |

`mode:0`  表示机器人站立但不尝试迈步

`mode:1`  表示可以行走

`mode:2`  表示机器人为“脚空载”状态

### BalanceMode

| **BalanceMode** | 备注 |
| --- | --- |
| 0 | 平衡站立 |
| 1 | 持续移动 |
| 2 | 强制站立（外力作用下不会自平衡踏步，请谨慎使用） |