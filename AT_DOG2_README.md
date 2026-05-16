# ATDog2 AMP 强化学习训练指南

## 目录结构

```
legged_lab/
├── source/legged_lab/legged_lab/
│   ├── assets/atdog2.py                             # 机器人配置（URDF路径、默认关节角、执行器参数）
│   ├── data/
│   │   ├── Robots/atdog2/urdf/                      # URDF + STL 文件
│   │   └── MotionData/atdog2/amp/walk_and_run/      # 训练动作数据 (.pkl)
│   └── tasks/locomotion/amp/config/atdog2/
│       ├── atdog2_amp_env_cfg.py                    # 环境配置（观测、奖励、事件、命令等）
│       └── agents/rsl_rl_ppo_cfg.py                 # PPO-AMP 算法超参
├── scripts/
│   ├── tools/retarget/
│   │   ├── csv_to_gmr.py                            # CSV → GMR pkl
│   │   ├── single_retarget.py                       # GMR pkl → Lab pkl（含 key_body_pos）
│   │   ├── dataset_retarget.py                      # 批量转换
│   │   └── config/atdog2.yaml                       # 关节名映射配置
│   └── rsl_rl/
│       ├── train.py                                 # 训练入口
│       └── play.py                                  # 推理入口
└── temp/urdf/atdog2/                                # URDF 源文件
```

---

## 快速开始

### 1. 准备 URDF 文件（容器内执行，仅需一次）

```bash
mkdir -p /workspace/legged_lab/source/legged_lab/legged_lab/data/Robots/atdog2/urdf
cp /workspace/legged_lab/temp/urdf/atdog2/* /workspace/legged_lab/source/legged_lab/legged_lab/data/Robots/atdog2/urdf/
```

> **重要**：必须使用 `UrdfFileCfg` 直接加载 URDF（当前配置已设置好），不要用 USD。直接加载 URDF 能保留原始关节轴方向，使训练/部署符号一致。

### 2. 录制 & 转换动作数据

```

**CSV 格式要求**（列名必须匹配）：
```
time_sec, root_pos_x, root_pos_y, root_pos_z,
root_quat_w, root_quat_x, root_quat_y, root_quat_z,
base_lin_vel_x, base_lin_vel_y, base_lin_vel_z,
base_ang_vel_x, base_ang_vel_y, base_ang_vel_z,
FR_hip_joint, FR_thigh_joint, FR_calf_joint,
FL_hip_joint, FL_thigh_joint, FL_calf_joint,
RR_hip_joint, RR_thigh_joint, RR_calf_joint,
RL_hip_joint, RL_thigh_joint, RL_calf_joint,
<后续列为关节速度，会被忽略>
```

```
# Step 1: CSV -> GMR pkl
python scripts/tools/retarget/csv_to_gmr.py \
    --input_csv Vz0.8.csv \
    --output_pkl temp/gmr_walk.pkl

# Step 1（批量）: 多个 CSV -> 多个 GMR pkl
python scripts/tools/retarget/csv_to_gmr.py \
    --input_dir . \
    --output_dir temp/gmr_data

# Step 2（单个）: GMR pkl -> Legged Lab pkl (含 key_body_pos)
python scripts/tools/retarget/single_retarget.py \
    --robot atdog2 \
    --input_file temp/gmr_data/L0.5.pkl \
    --output_file source/legged_lab/legged_lab/data/MotionData/atdog2/amp/walk_and_run/L0.5_walk.pkl \
    --config_file scripts/tools/retarget/config/atdog2.yaml \
    --loop clamp --headless
# Step 2（批量）: GMR pkl -> Legged Lab pkl (含 key_body_pos)
python scripts/tools/retarget/dataset_retarget.py \
    --robot atdog2 \
    --input_dir temp/gmr_data \
    --output_dir source/legged_lab/legged_lab/data/MotionData/atdog2/amp/walk_and_run \
    --config_file scripts/tools/retarget/config/atdog2.yaml \
    --output_suffix _walk \
    --loop clamp --headless


```

### 3. 训练
xhost +local:docker
```bash
# 小规模测试训练
python scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-ATDog2-v0 \
    --num_envs 100 \
    --max_iterations 2000

# 正式训练
python scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-ATDog2-v0 \
    --num_envs 1000 \
    --max_iterations 20000 \
    --headless

python scripts/rsl_rl/train.py \
    --task LeggedLab-Isaac-AMP-ATDog2-v0 \
    --num_envs 3000 \
    --max_iterations 2000 \
    --headless

--resume \
    --load_run 2026-05-14_10-47-03 \
    --checkpoint model_2800.pt \
```

### 4. 推理

```bash
python scripts/rsl_rl/play.py \
    --task LeggedLab-Isaac-AMP-ATDog2-Play-v0 \
    --num_envs 48 \
    --checkpoint /workspace/legged_lab/logs/rsl_rl/atdog2_amp/2026-05-14_14-33-31/model_19999.pt

```

---

## 可调参数

### 机器人配置 (`source/legged_lab/legged_lab/assets/atdog2.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `init_state.pos` | `(0, 0, 0.35)` | 机器人初始高度 |
| `init_state.joint_pos` | 见下方 | 初始关节角度 |
| `stiffness` | `25.0` | PD 控制器刚度 |
| `damping` | `0.5` | PD 控制器阻尼 |
| `friction` | `0.01` | 执行器摩擦 |
| `soft_joint_pos_limit_factor` | `0.9` | 关节软限位缩放 |

**默认关节角度**（MuJoCo/CSV 符号）：
```python
".*_hip_joint": 0.0,        # 所有 hip 初始 0
".*L_thigh_joint": 0.8,     # 左侧大腿向后弯
".*R_thigh_joint": -0.8,    # 右侧大腿向后弯（符号相反）
".*_calf_joint": 0.0,       # 所有小腿初始 0
```

> **符号约定**：由于 URDF 中 L/R 大腿/小腿关节的 RPY 不同，MuJoCo 保留这些差异，导致右侧 thigh/calf 正方向与左侧相反。**训练输出的 pt 文件使用此符号约定，可直接用于实际部署，无需翻转。**

### 环境配置 (`atdog2_amp_env_cfg.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `episode_length_s` | 20.0 | 每个 episode 时长（秒） |
| `decimation` | 4 | 控制频率降采样（dt=0.005s × 4 = 0.02s = 50Hz） |
| `lin_vel_x` | (-0.5, 3.0) | 前向速度命令范围 (m/s) |
| `lin_vel_y` | (-0.5, 0.5) | 横向速度命令范围 |
| `ang_vel_z` | (-1.0, 1.0) | 角速度命令范围 (rad/s) |
| `height_offset` | 0.05 | 从参考动画重置时的高度偏移 |
| `AMP_NUM_STEPS` | 4 | 判别器取参考动画的步数 |

### 奖励权重 (`ATDog2AmpRewards`)

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `track_lin_vel_xy_exp` | 1.0 | 线速度跟踪 |
| `track_ang_vel_z_exp` | 1.0 | 角速度跟踪 |
| `flat_orientation_l2` | -1.0 | 惩罚身体倾斜 |
| `lin_vel_z_l2` | -2.0 | 惩罚垂直速度 |
| `dof_pos_limits` | -1.0 | 关节限位惩罚（仅 calf） |
| `joint_deviation_hip` | -0.1 | hip 关节偏离默认值惩罚 |
| `feet_air_time` | 0.5 | 足部腾空时间奖励 |
| `termination_penalty` | -50.0 | 终止惩罚 |

### 算法超参 (`agents/rsl_rl_ppo_cfg.py`)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `num_steps_per_env` | 24 | 每次更新的步数 |
| `max_iterations` | 50000 | 最大迭代次数 |
| `save_interval` | 200 | 保存间隔 |
| `learning_rate` | 1e-4 | 学习率 |
| `clip_param` | 0.2 | PPO clip 参数 |
| `entropy_coef` | 0.01 | 熵系数 |
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `actor_hidden_dims` | [512, 256, 128] | Actor 网络结构 |
| `disc_hidden_dims` | [1024, 512] | 判别器结构 |
| `symmetry_cfg` | 启用 | 左右对称数据增强 |

---

## 对称性配置

对称性在 `source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/symmetry/atdog2.py` 中定义。

**关键逻辑**：由于 URDF 中 L/R thigh/calf 的 RPY 不同，MuJoCo/CSV 符号下 L/R 正方向相反。因此 L↔R swap 后**所有关节都需要取反**（不仅仅是 hip）。

如需修改对称性（例如加入新的观测维度），需要同步更新：
1. `_transform_policy_obs_left_right` — 观测维度偏移量
2. `_switch_atdog2_joints_left_right` — 关节 swap 逻辑
3. `_switch_atdog2_key_body_pos_left_right` — 关键体位置 swap

---

## 导入新的机器人

### 1. 准备 URDF

将 URDF 和 STL 文件放到 `data/Robots/<robot_name>/urdf/` 下：

```bash
mkdir -p source/legged_lab/legged_lab/data/Robots/<robot_name>/urdf
cp <your_urdf_and_stl_files> source/legged_lab/legged_lab/data/Robots/<robot_name>/urdf/
```

### 2. 创建机器人配置

在 `source/legged_lab/legged_lab/assets/` 下新建或修改配置文件，参照 `atdog2.py`：

```python
NEW_DOG_CFG = ATDog2ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/<robot_name>/urdf/<urdf_file>.urdf",
        activate_contact_sensors=True,
        rigid_props=...,
        articulation_props=...,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, <height>),
        joint_pos={...},  # 默认关节角，用 MuJoCo/CSV 符号
        joint_vel={".*": 0.0},
    ),
    actuators={...},
    joint_sdk_names=[...],  # 关节顺序，影响观测/动作维度映射
)
```

> **必须使用 `UrdfFileCfg`**，不要用 `UsdFileCfg`。USD 转换会统一 L/R 关节轴方向，导致与 MuJoCo 符号不一致。

### 3. 创建 retarget 配置

在 `scripts/tools/retarget/config/` 下新建 `<robot_name>.yaml`，参照 `atdog2.yaml`：

```yaml
# GMR (MuJoCo/CSV) 关节名和顺序
gmr_dof_names:
  - <left_front_hip>
  - <left_front_thigh>
  - <left_front_calf>
  - <right_front_hip>
  - ...  # 与 CSV 列名一致

# Lab (Isaac Lab) 关节名和顺序
lab_dof_names:
  - <right_front_hip>
  - <right_front_thigh>
  - <right_front_calf>
  - <left_front_hip>
  - ...  # 与 joint_sdk_names 一致

# 关键体名称（用于 key_body_pos）
lab_key_body_names:
  - FR_calf
  - FL_calf
  - RR_calf
  - RL_calf
```

### 4. 创建环境配置

在 `tasks/locomotion/amp/config/<robot_name>/` 下创建：

- `__init__.py` — 注册环境
- `<robot_name>_amp_env_cfg.py` — 环境配置（参照 `atdog2_amp_env_cfg.py`）
- `agents/rsl_rl_ppo_cfg.py` — 算法配置（参照 `atdog2/agents/rsl_rl_ppo_cfg.py`）

### 5. 创建对称性配置

在 `tasks/locomotion/amp/mdp/symmetry/` 下新建 `<robot_name>.py`，参照 `atdog2.py`。

**注意**：如果新机器人的 URDF 也有 L/R thigh/calf RPY 差异（大多数四足机器人都有），则 L↔R swap 后所有关节都需要取反。

### 6. 录制动作数据

使用 VMC 或 MuJoCo 录制 CSV，然后按上述「快速开始」的步骤 2 转换为 pkl。

---

## 符号约定说明

| 约定 | 左侧 thigh 正值含义 | 右侧 thigh 正值含义 |
|------|---------------------|---------------------|
| MuJoCo/CSV（URDF 原始） | 大腿向后弯 | 大腿**向前伸**（正方向相反） |
| 部署代码 | 以此为准 | 以此为准 |
| Isaac Lab（直接加载 URDF） | 与 CSV 一致 | 与 CSV 一致 |

当前项目使用 `UrdfFileCfg` 直接加载 URDF，**三种场景的符号完全一致**，无需任何翻转。

---

## 常见问题

### Q: 训练时腿蜷缩/绷直，动作不自然？

检查 `atdog2.py` 中的默认关节角度是否使用了正确的符号约定。如果使用 USD 模型而非 URDF，可能出现 L/R 符号不匹配。

### Q: 如何更换动作数据？

1. 录制新 CSV（格式同上）
2. 运行 `csv_to_gmr.py` → `single_retarget.py` 生成新的 pkl
3. 将 pkl 放到 `data/MotionData/atdog2/amp/walk_and_run/` 下
4. 在 `atdog2_amp_env_cfg.py` 的 `motion_data_weights` 中添加/修改权重

### Q: 如何调整训练速度？

- 增大 `num_envs`（默认 4096，显存够的话可以加到 8192）
- 调整 `num_steps_per_env`（越大越稳定但越慢）
- 修改 `decimation`（减小可提高控制频率，但训练更慢）

### Q: 训练结果如何部署？

训练输出的 pt 文件中的关节角使用 MuJoCo/CSV 符号约定（即与 URDF 一致），**直接用于部署代码即可，无需翻转任何符号**。
