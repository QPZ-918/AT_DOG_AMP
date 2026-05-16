# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and tool expectations

- This repository is an Isaac Lab extension. Python entrypoints expect an Isaac Lab / Isaac Sim-capable environment, not a generic system Python.
- Install large tracked assets after clone with:
  - `git lfs install`
  - `git lfs pull`
- Per the repository's existing agent guidance in `AGENTS.md`, prefer running `python`, `pytest`, `pip`, `pre-commit`, and repository Python entrypoints through the repository's configured runner/skill rather than assuming a machine-specific interpreter path.
- The package itself is installed editable from `source/legged_lab`:
  - `python -m pip install -e source/legged_lab`
- This repo also expects a forked `rsl_rl` with AMP support:
  - `git clone -b feature/amp https://github.com/zitongbai/rsl_rl.git`
  - `python -m pip install -e /path/to/rsl_rl`
- If using Docker, `docker/.env.base` expects `rsl_rl` to live next to `legged_lab` unless `RSL_RL_PATH` is changed.

## Common commands

### Install / bootstrap

```bash
git lfs install
git lfs pull
python -m pip install -e source/legged_lab
```

### Format / lint

```bash
pre-commit run --all-files
black --line-length 120 .
isort --profile black .
flake8 .
```

### Run tests

```bash
pytest -q
pytest source/legged_lab/test -q
pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -q
pytest source/legged_lab/test/test_manager_based_amp_env_terminal_disc.py -k terminal_obs -q
```

### Docker workflow

```bash
bash docker/build.sh
bash docker/run.sh
bash docker/enter.sh
bash docker/stop.sh
```

### Training / playback

DeepMimic:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac--Deepmimic-G1-v0 --headless --max_iterations 50000
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-Deepmimic-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

AMP:

```bash
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000
python scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --max_iterations 50000 --device cuda:x agent.device=cuda:x
python scripts/rsl_rl/play.py --task LeggedLab-Isaac-AMP-G1-v0 --headless --num_envs 64 --video --checkpoint logs/rsl_rl/experiment_name/run_name/model_xxx.pt
```

### Motion retargeting / animation utilities

Batch retarget GMR motions into Legged Lab format:

```bash
python scripts/tools/retarget/dataset_retarget.py \
  --robot g1 \
  --input_dir temp/gmr_data/ \
  --output_dir temp/lab_data/ \
  --config_file scripts/tools/retarget/config/g1_29dof.yaml \
  --loop clamp
```

Single-motion retarget:

```bash
python scripts/tools/retarget/single_retarget.py \
  --robot g1 \
  --input_file temp/gmr_walk.pkl \
  --output_file temp/lab_walk.pkl \
  --config_file scripts/tools/retarget/config/g1_29dof.yaml \
  --loop clamp
```

Animation-only playback utility:

```bash
python scripts/play_anim.py --robot g1_29dof --num_envs 4
```

## High-level architecture

### Repository role

`legged_lab` is an Isaac Lab extension package developed outside the core Isaac Lab repository. It adds legged-robot tasks, custom manager-based environments, robot assets, motion-data tooling, and RSL-RL training/playback entrypoints.

### Main package layout

- `source/legged_lab/legged_lab/assets/`: robot articulation and actuator configs (for example Unitree and ATDog2). These configs define the robot model, startup joint defaults, actuator behavior, and any repository-specific metadata such as `joint_sdk_names`.
- `source/legged_lab/legged_lab/envs/`: custom environment classes layered on Isaac Lab manager-based envs.
- `source/legged_lab/legged_lab/managers/`: motion-data loading, reference animation playback, and AMP-specific observation helpers.
- `source/legged_lab/legged_lab/tasks/`: task registration plus task-specific config trees for locomotion variants such as AMP, DeepMimic, and animation-only playback.
- `source/legged_lab/legged_lab/rsl_rl/`: repository-specific RL config objects that complement the scripts in `scripts/rsl_rl/`.
- `source/legged_lab/legged_lab/data/`: checked-in robot resources and motion datasets consumed by tasks.
- `scripts/rsl_rl/`: train/play entrypoints.
- `scripts/tools/retarget/`: CSV/GMR-to-Legged-Lab motion conversion pipeline.
- `source/legged_lab/test/`: tests, including pure-Python env logic tests and tests that boot an Isaac Lab test app.

### How tasks are discovered and launched

Task registration is import-driven:

- `source/legged_lab/legged_lab/tasks/__init__.py` uses `isaaclab_tasks.utils.import_packages(...)` to import config subpackages.
- Each robot/task package registers Gym environments in its own `__init__.py` (for example AMP G1 config packages call `gym.register(...)`).
- `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` import `legged_lab.tasks` for these side effects, then resolve the selected task via `@hydra_task_config(args_cli.task, args_cli.agent)`.

When adding a new task, the usual pattern is:
1. Create a robot/task config package under `tasks/.../config/...`.
2. Register a Gym id in that package’s `__init__.py`.
3. Point the registration at an env config class and an RSL-RL config entry point.

### Environment stack

The custom env hierarchy is central to the codebase:

- `ManagerBasedAnimationEnv` extends Isaac Lab’s `ManagerBasedRLEnv` and injects two custom managers before the normal manager load path:
  - `MotionDataManager`
  - `AnimationManager`
- `ManagerBasedAmpEnv` extends `ManagerBasedAnimationEnv` and swaps in `PreviewObservationManager`, then augments `step()` so it can preserve pre-reset discriminator observations in `extras["terminal_obs"]` for terminated environments.

This means AMP behavior is not just config-level; it depends on the custom env implementation in `envs/` plus the managers in `managers/`.

### Motion-data and animation pipeline

A lot of repository-specific behavior lives in the reference-motion path:

1. External motion data is first converted into repository format via `scripts/tools/retarget/`.
2. The final `.pkl` motion files live under `source/legged_lab/legged_lab/data/MotionData/...`.
3. `MotionDataManager` loads these files, computes derived quantities such as velocities, samples motions/timesteps, and serves interpolated reference states.
4. `AnimationManager` consumes those sampled states and exposes them to:
   - visualization through `robot_anim`
   - reset-from-reference events
   - discriminator/demo observations for AMP and DeepMimic-style tasks

Important implication: if a bug concerns reference poses, joint ordering, discriminator demo observations, or reset-from-reference behavior, inspect both the retarget scripts and the runtime manager path; the issue may be in either layer.

### Task config layering

Task configs follow a base-plus-robot-override pattern.

- Base task files such as `tasks/locomotion/amp/amp_env_cfg.py` define the scene schema, observation groups, action terms, reward terms, termination terms, motion-data config, and animation config.
- Robot-specific files such as `tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py` or `tasks/locomotion/amp/config/atdog2/atdog2_amp_env_cfg.py` override:
  - `scene.robot`
  - motion dataset path/weights
  - key-body configuration
  - event wiring like `reset_from_ref`
  - command ranges, curriculum, and robot-specific reward/termination details
- Matching `agents/rsl_rl_ppo_cfg.py` files hold the RSL-RL runner/algorithm hyperparameters for that task.

When changing behavior, first decide whether it belongs in:
- shared base task logic,
- robot-specific task config,
- custom env/manager runtime code,
- or asset metadata.

### Training and playback flow

`scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` are thin orchestration layers around Isaac Lab + RSL-RL:

- launch simulator via `AppLauncher`
- resolve env/agent config with Hydra task config decorators
- `gym.make(task_id, cfg=env_cfg, ...)`
- wrap with `RslRlVecEnvWrapper`
- choose runner class (`OnPolicyRunner`, `AMPRunner`, or `DistillationRunner`)
- write logs under `logs/rsl_rl/<experiment>/<timestamp>_*`

This is why most algorithm-specific changes are configuration changes, while environment semantics changes usually live under `source/legged_lab/legged_lab/`.

### Retargeting pipeline

The motion conversion path is intentionally separate from runtime training:

- `csv_to_gmr.py`: CSV to GMR-style intermediate format
- `single_retarget.py` / `dataset_retarget.py`: command-line entrypoints for one or many motions
- `gmr_to_lab.py`: core conversion/replay logic that maps joint order, converts quaternion format, replays the motion in Isaac Lab, and records `key_body_pos`

If a motion issue appears both in offline conversion playback and during training, debug `scripts/tools/retarget/gmr_to_lab.py` first. If it appears only after loading datasets inside envs, debug `managers/motion_data_manager.py` and `managers/animation_manager.py`.

## Repository-specific conventions

- Python formatting/linting is governed by `.pre-commit-config.yaml` and `pyproject.toml`:
  - Black line length is 120.
  - isort uses the Black profile and treats `legged_lab` as first-party.
  - Pyright is configured but `typeCheckingMode = "off"`.
- Tests use `pytest`; `pytest.ini` defines the `isaacsim_ci` marker.
- Large assets and sample motion/model files are tracked with Git LFS.
- Generated outputs usually appear in `logs/`, `outputs/`, and `temp/`; avoid treating them as source unless the user is explicitly updating tracked sample assets.
