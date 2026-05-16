# 🤖 Legged Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab/v2.3.1/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## 目录

- [概述](#概述)
- [演示](#演示)
- [新闻与更新](#新闻与更新)
- [安装](#安装)
  - [前置要求](#前置要求)
  - [安装步骤](#安装步骤)
  - [Docker 使用 (Dockerfile + Bash 脚本)](#docker-使用)
- [使用方法](#使用方法)
  - [准备运动数据](#准备运动数据)
  - [训练与运行](#训练与运行)
- [路线图](#路线图)
- [致谢](#致谢)

<a id="概述"></a>
## 📖 概述

本项目是基于 Isaac Lab 的足式机器人强化学习扩展,允许在 Isaac Lab 核心仓库之外的独立环境中进行开发。强化学习算法基于 [forked RSL-RL 库](https://github.com/zitongbai/rsl_rl/tree/feature/amp)。

**核心特性:**

- `DeepMimic` 用于人形机器人,包括宇树 G1。
- `AMP` 对抗性运动先验 (Adversarial Motion Priors) 用于人形机器人,包括宇树 G1。我们建议使用 [GMR](https://github.com/YanjieZe/GMR) 对人类运动数据进行重定向。

<a id="演示"></a>
## 演示

* 宇树 G1 的对抗性运动先验:

https://github.com/user-attachments/assets/ed84a8a3-f349-44ac-9cfd-2baab2265a25

<a id="新闻与更新"></a>
## 🔥 新闻与更新

- 2026/02/09: 添加 Dockerfile + bash 脚本工作流,包括本地 `rsl_rl` 的主机路径要求。
- 2025/12/16: 在 Isaac Lab 2.3.1 和 RSL-RL 3.2.0 中测试。
- 2025/12/05: 使用 git lfs 存储大文件,包括运动数据和机器人模型。
- 2025/11/23: 在 AMP 训练中添加对称性数据增强。
- 2025/11/22: AMP 的新实现。
- 2025/11/19: 为 G1 添加 DeepMimic。
- 2025/10/14: 更新以支持 rsl_rl v3.1.1。目前仅支持平地行走。
- 2025/08/24: 支持在 AMP 训练中使用更多步数的观测和运动数据。
- 2025/08/22: 兼容 Isaac Lab 2.2.0。
- 2025/08/21: 添加通过 [GMR](https://github.com/YanjieZe/GMR) 重定向人类运动数据的支持。

<a id="安装"></a>
## ⚙️ 安装

<a id="前置要求"></a>
### 前置要求

- **Isaac Lab**: 确保已安装 Isaac Lab `v2.3.1`。请遵循[官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)。
- **Git LFS**: 下载大型模型文件所需。

<a id="安装步骤"></a>
### 安装步骤

1.  **克隆仓库**
    将本仓库克隆到现有 `IsaacLab` 目录*之外*,以保持隔离。

    ```bash
    # 选项 1: HTTPS
    git clone https://github.com/zitongbai/legged_lab

    # 选项 2: SSH
    git clone git@github.com:zitongbai/legged_lab.git

    cd legged_lab
    ```

2.  **拉取 Git LFS 资源**
    在机器上安装并初始化 `git-lfs`(一次性操作),然后为本仓库拉取大型资源(USD 模型和运动数据)。

    ```bash
    git lfs install
    git lfs pull
    ```

3.  **安装包**
    使用与 Isaac Lab 安装关联的 Python 解释器。

    ```bash
    python -m pip install -e source/legged_lab
    ```

4.  **安装 RSL-RL (Forked 版本)**
    我们使用定制版本的 `rsl_rl` 来支持 AMP 等高级功能。

    ```bash
    # 在 IsaacLab 和 legged_lab 目录之外克隆
    git clone -b feature/amp https://github.com/zitongbai/rsl_rl.git

    cd rsl_rl
    python -m pip install -e .
    ```

<a id="docker-使用"></a>
### Docker 使用 (Dockerfile + Bash 脚本)

如果使用提供的 Docker 工作流,容器将在启动时自动挂载本地源代码并安装包。

#### `rsl_rl` 的主机目录要求

默认情况下,`docker/.env.base` 期望 `rsl_rl` 放置在 `legged_lab` 旁边:

```text
.../lab_dev/
├── legged_lab/
└── rsl_rl/
```

如果你的 `rsl_rl` 在其他位置,请更新 `docker/.env.base` 中的 `RSL_RL_PATH`。

默认情况下,Isaac Sim 缓存、日志、数据和文档使用官方 Docker 目录布局,位于 `~/docker/isaac-sim` 下。

#### 构建镜像

```bash
bash docker/build.sh
```

#### 启动容器

```bash
# xhost +
bash docker/run.sh
```

启动时,容器将:
- 用容器内置的 VS Code 设置覆盖 `.vscode/settings.json`
- 以可编辑模式安装挂载的 `rsl_rl` (`/workspace/rsl_rl`)
- 以可编辑模式安装挂载的 `legged_lab` (`/workspace/legged_lab/source/legged_lab`)

#### 进入容器

```bash
bash docker/enter.sh
```

默认工作目录是 `/workspace/legged_lab`。

#### 停止/删除容器

```bash
bash docker/stop.sh
```

#### Dockerfile 更改后重新构建镜像

```bash
bash docker/stop.sh
bash docker/build.sh
bash docker/run.sh
```

<a id="使用方法"></a>
## 🚀 使用方法

<a id="准备运动数据"></a>
### 1. 准备运动数据

我们已经在 `source/legged_lab/legged_lab/data/MotionData` 文件夹中提供了一些现成的运动数据用于测试。

如果你想添加更多运动数据,可以按照以下步骤操作。

1. 将人类运动数据重定向到机器人模型。我们推荐使用 [GMR](https://github.com/YanjieZe/GMR) 来重定向人类运动数据。
2. 将重定向后的运动数据放入 `temp/gmr_data` 文件夹。
3. 使用辅助脚本将运动数据转换为所需格式:

    ```bash
    python scripts/tools/retarget/dataset_retarget.py \
        --robot g1 \
        --input_dir temp/gmr_data/ \
        --output_dir temp/lab_data/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml \
        --loop clamp
    ```
4. 将转换后的数据从 `temp/lab_data` 移动到 `source/legged_lab/legged_lab/data/MotionData`,并在配置文件中设置 `MotionDataCfg`,例如 `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`。

有关参数的更多详细信息,请参阅脚本中的注释,并参考 `scripts/tools/retarget/gmr_to_lab.py` 了解本仓库使用的数据格式。

<a id="训练与运行"></a>
### 2. 训练与运行

#### 🎭 DeepMimic

<details>
<summary>训练</summary>

要训练 DeepMimic 算法,可以运行以下命令:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
```

`max_iterations` 可以根据你的需求进行调整。有关参数的更多详细信息,运行 `python scripts/rsl_rl/train.py -h`。

</details>

<details>
<summary>运行</summary>

你可以以无头模式运行训练好的模型并录制视频:

```bash
# 将检查点路径替换为你的训练模型路径
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-Deepmimic-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

</details>


#### 🏃 对抗性运动先验 (AMP)

<details>
<summary>训练</summary>

要训练 AMP 算法,可以运行以下命令:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000
```

如果你想在非默认 GPU 上训练,可以向命令传递更多参数:

```bash
# 将 `x` 替换为你想使用的 GPU ID
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000 --device cuda:x agent.device=cuda:x
```

有关参数的更多详细信息,运行 `python scripts/rsl_rl/train.py -h`。

</details>

<details>
<summary>运行</summary>

你可以以无头模式运行训练好的模型并录制视频:

```bash
# 将检查点路径替换为你的训练模型路径
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

视频将保存在 `logs/rsl_rl/experiment_name/run_name/videos/play` 目录中。

</details>

<a id="路线图"></a>
## 🗺️ 路线图

- [ ] 添加更多足式机器人,如宇树 H1
- [x] AMP 中的自接触惩罚
- [x] AMP 中的非对称 Actor-Critic
- [x] 对称奖励
- [ ] Mujoco 中的 Sim2sim
- [ ] 添加图像观测支持
- [ ] 在复杂地形上使用 AMP 行走

<a id="致谢"></a>
## 🙏 致谢

我们要向以下开源项目表示感谢:

- [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) - 本项目的基础。
- [**RSL-RL**](https://github.com/leggedrobotics/rsl_rl) - 足式机器人的强化学习算法。
- [**AMP_for_hardware**](https://github.com/Alescontrela/AMP_for_hardware) - AMP 实现的灵感来源。
- [**GMR**](https://github.com/YanjieZe/GMR) - 优秀的运动重定向库。
- [**MimicKit**](https://github.com/xbpeng/MimicKit) - 模仿学习的参考。