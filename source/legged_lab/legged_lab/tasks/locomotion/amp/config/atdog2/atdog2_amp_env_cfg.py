import math
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.atdog2 import ATDOG2_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg

# Key body names for the quadruped: calf links (foot contact points after merge_fixed_joints)
# if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.atdog2 as well!
KEY_BODY_NAMES = [
    "FR_calf",
    "FL_calf",
    "RR_calf",
    "RL_calf",
]
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class ATDog2AmpRewards:
    """Reward terms for the ATDog2 AMP MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-8)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0025)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    # -- quadruped-specific
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_calf"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh"]),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)


@configclass
class ATDog2AmpEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the ATDog2 AMP environment."""

    rewards: ATDog2AmpRewards = ATDog2AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ATDOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "atdog2", "amp", "walk_and_run"
        )
        self.motion_data.motion_dataset.motion_data_weights = {
            "TB_walk": 3.0,
            "Vx0.4_walk": 2.0,
            "Vx0.8_walk": 2.0,
            "Vx-0.4_walk": 0.25,
            "Vx-0.8_walk": 0.1,
            "Vz0.4_walk": 0.1,
            "Vz0.8_walk": 0.05,
            "Vz-0.4_walk": 0.1,
            "Vz-0.8_walk": 0.05,
            "L0.25_walk": 0.1,
            "L0.5_walk": 0.05,
            "R0.25_walk": 0.1,
            "R0.5_walk": 0.05,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        # critic observations
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        # discriminator observations
        self.observations.disc.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demonstration observations
        self.observations.disc_demo.ref_root_local_rot_tan_norm.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_key_body_pos_b.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_from_ref.params = {"animation": ANIMATION_TERM_NAME, "height_offset": 0.02}

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # Curriculum
        # ------------------------------------------------------
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        # Disable base_contact termination for quadruped (torso contact is normal in some gaits)
        self.terminations.base_contact = None


@configclass
class ATDog2AmpEnvCfg_PLAY(ATDog2AmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        self.events.reset_from_ref = None
