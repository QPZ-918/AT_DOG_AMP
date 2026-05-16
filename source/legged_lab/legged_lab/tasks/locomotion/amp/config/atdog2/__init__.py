import gymnasium as gym

from legged_lab.envs import ManagerBasedAmpEnv

from . import agents

gym.register(
    id="LeggedLab-Isaac-AMP-ATDog2-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atdog2_amp_env_cfg:ATDog2AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDog2RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="LeggedLab-Isaac-AMP-ATDog2-Play-v0",
    entry_point="legged_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atdog2_amp_env_cfg:ATDog2AmpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ATDog2RslRlOnPolicyRunnerAmpCfg",
    },
)
