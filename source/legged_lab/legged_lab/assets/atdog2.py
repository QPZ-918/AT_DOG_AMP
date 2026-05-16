"""Configuration for ATDog2 robot.

Uses UrdfFileCfg to directly load the URDF at runtime, which preserves the
original joint axis directions from the URDF (including the different RPY
for left vs right leg thigh/calf joints). This ensures that the joint sign
convention in Isaac Lab matches MuJoCo/CSV, so no sign flipping is needed
in the training pipeline or deployment code.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets import unitree_actuators


@configclass
class ATDog2ArticulationCfg(ArticulationCfg):
    """Configuration for ATDog2 articulations."""

    joint_sdk_names: list[str] = None

    soft_joint_pos_limit_factor = 0.9


ATDOG2_CFG = ATDog2ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{LEGGED_LAB_ROOT_DIR}/data/Robots/atdog2/urdf/dog2.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.28),
        joint_pos={
            ".*_hip_joint": 0.0,
            ".*R_thigh_joint": -0.8,
            ".*R_calf_joint": 0.0,
            ".*L_thigh_joint": 0.8,
            ".*L_calf_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            min_delay=1,  # physics steps (sim.dt=0.005s): 1 * 5ms = 5ms
            max_delay=2,  # physics steps (fixed delay): 1 * 5ms = 5ms
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ],
    # fmt: on
)
