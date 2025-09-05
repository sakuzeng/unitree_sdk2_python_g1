# Unitree Python SDK (unitree_sdk_python) 详细说明

## 1. 概述

`unitree_sdk_python` 目录是宇树科技为其系列机器人（包括 G1, H1, Go2, B2 等）提供的官方 Python SDK。它封装了与机器人底层通信、控制和数据交换的复杂性，为开发者提供了一套高级和低级的 Python API。

该 SDK 的核心是基于 DDS (Data Distribution Service) 的实时通信机制，并通过 RPC (Remote Procedure Call) 提供了方便的控制接口。

## 2. 根目录核心文件

-   **`README.md` / `README zh.md`**: SDK 的官方中英文使用说明，是入门的首选文档。
-   **`setup.py`**: 标准的 Python 包安装脚本，用于将 `unitree_sdk2py` 安装到 Python 环境中。
-   **`requirements.txt`**: SDK 运行所需的 Python 依赖库列表。
-   **`pyproject.toml`**: 现代 Python 项目的配置文件，定义了构建系统和项目元数据。
-   **`LICENSE`**: 项目的开源许可证文件。

## 3. 核心模块详解 (`unitree_sdk2py/`)

这是 SDK 的 Python 源码核心，开发者在代码中 `import` 的就是这个模块。它通过清晰的子模块划分，实现了通信、数据结构和机器人功能服务的解耦。

```
unitree_sdk2py/
├── core/                  # 核心通信模块 (DDS封装)
├── rpc/                   # RPC 远程调用模块
├── idl/                   # 数据接口定义语言 (Data Definition Language)
├── g1/                    # G1 机器人专用服务客户端
├── go2/                   # Go2 机器人专用服务客户端
├── h1/                    # H1 机器人专用服务客户端
├── comm/                  # 通用服务客户端
└── utils/                 # 实用工具函数
```

### 3.1. 通信核心 (`core/`)

此目录是整个 SDK 的基石，封装了底层的 DDS 通信细节。
-   **`channel.py`**: 提供了 `ChannelPublisher` 和 `ChannelSubscriber` 类，是进行 DDS 消息发布和订阅的直接接口。低级控制和状态订阅会直接使用它们。
-   **`channel_name.py`**: 定义了 DDS 通信所使用的 Topic 名称常量。
-   **`channel_config.py`**: 允许配置 DDS 的服务质量（QoS）和绑定的网络接口。

### 3.2. RPC 模块 (`rpc/`)

实现了客户端-服务器模式的远程过程调用，将复杂的 DDS 请求/回复模式封装成简单的函数调用。这是实现高级指令（如让机器人“行走”）的主要方式。
-   **`client.py`**: 提供了 `Client` 类，是所有高级服务客户端的基类。它管理着与 RPC 服务器的连接和通信。
-   **`server.py`**: RPC 服务器的实现，通常在机器人端运行。
-   **`client_stub.py` / `server_stub.py`**: RPC 的存根（Stub）代码，处理序列化和底层通信细节。

### 3.3. 数据定义 (`idl/`)

这是 SDK 中至关重要的部分，它定义了所有与机器人通信时使用的数据结构（消息类型）。这些 Python 文件是由 `.idl` 文件自动生成的。
-   **`unitree_hg/`**: **G1 和 H1** 等人形/通用（Humanoid/General）机器人的消息定义。**这是 G1 开发最核心的参考**。
-   **`unitree_go/`**: Go2, B2 等四足机器人的消息定义。
-   **`unitree_api/`**: 通用的 API 消息定义，如服务请求、响应等。
-   `sensor_msgs/`, `geometry_msgs/` 等: 兼容 ROS 的标准消息类型，便于与 ROS 系统集成。
-   **`default.py`**: 导入了所有常用的消息类型，方便开发者使用。

### 3.4. 机器人专用服务客户端 (`g1/`, `go2/`, `h1/` 等)

这些目录是开发者最常直接使用的模块。它们为特定机器人的特定功能（如运动、机械臂）提供了封装好的、易于使用的客户端。

以 `g1/` 为例，其结构体现了 SDK 的设计模式：
-   **`loco/` (运动控制)**
    -   `g1_loco_api.py`: 定义了 G1 运动服务（Locomotion Service）的 API 常量，如服务名称、版本号、Topic 名称等。
    -   `g1_loco_client.py`: 提供了 `G1LocoClient` 类。开发者通过实例化这个类，就可以调用如 `walk()`、`stand()` 等高级运动指令。这个类内部使用了 `rpc/client.py` 来发送指令。
-   **`arm/` (机械臂控制)**
    -   `g1_arm_action_api.py`: 定义了机械臂服务的 API 常量。
    -   `g1_arm_action_client.py`: 提供了 `G1ArmActionClient` 类，用于控制机械臂。
-   **`audio/` (音频服务)**
    -   `g1_audio_api.py`: 定义了音频服务的 API 常量。
    -   `g1_audio_client.py`: 提供了 `G1AudioClient` 类，用于播放音频或进行语音交互。

**工作流程**: 当你调用 `G1LocoClient().walk(vx, vy)` 时，`g1_loco_client.py` 会使用 `g1_loco_api.py` 中定义的 Topic 名称，将你的参数打包成 `idl/unitree_hg/` 中定义的 `WalkCmd` 消息，然后通过 `rpc/client.py` 发送给机器人端的 RPC 服务器。

## 4. 示例代码详解 (`example/`)

`example/` 目录是学习如何使用 SDK 的最佳资源，它为不同机器人型号和功能提供了可直接运行的示例代码。

### 4.1. G1 机器人示例 (`g1/`)

这是 G1 开发者最需要关注的目录。
-   **`readme.md`**: G1 示例的说明文档。
-   **`high_level/`**: 高级控制示例，演示如何使用 `g1/loco` 等目录下的高级服务客户端。这是最推荐的入门方式。
-   **`low_level/`**: 低级控制示例，直接使用 `core/channel.py` 发布 DDS 消息来控制关节，风险高，需要非常谨慎。
-   **`audio/`**: 语音功能的客户端示例。

### 4.2. 通用功能示例

-   **`helloworld/`**: 演示了最基础的 DDS 发布者 (`publisher.py`) 和订阅者 (`subscriber.py`) 的工作流程，有助于理解底层通信。
-   **`motionSwitcher/`**: 演示了如何在不同运动模式（如站立、行走、低级控制）之间安全切换。

## 5. 总结与开发流程

`unitree_sdk_python` 目录提供了一个结构清晰、功能完整的开发环境。对于 G1 机器人的开发者来说，核心工作流程如下：
1.  **阅读文档**: 首先阅读根目录的 `README zh.md` 和 `example/g1/readme.md`。
2.  **运行高级示例**: 运行 `example/g1/high_level/` 下的示例，理解如何通过 RPC 服务客户端与机器人交互。
3.  **查阅数据定义**: 在开发自己的应用时，频繁查阅 `unitree_sdk2py/idl/unitree_hg/` 来了解可以获取哪些状态数据以及可以发送哪些命令的数据结构。
4.  **二次开发**:
    -   **高级控制**: `from unitree_sdk2py.g1.loco import G1LocoClient`，然后实例化并调用其方法。
    -   **状态订阅**: `from unitree_sdk2py.core import ChannelSubscriber` 和 `from unitree_sdk2py.idl.unitree_hg import RobotState_`，然后创建订阅者来接收实时状态数据。
5.  **低级控制 (仅限专家)**: 如果需要，参考 `example/g1/low_level/` 示例，使用 `ChannelPublisher` 直接发送关节指令。