"""Functions to specify the symmetry in the observation and action space for ATDog2 quadruped."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying left-right symmetry.

    This function creates augmented versions of the provided observations and actions
    by applying a left-right symmetry transformation. For a quadruped robot, this
    swaps the left and right legs.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)

        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(
            env.unwrapped, obs["policy"][:]
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    For the ATDog2 quadruped, this swaps FL↔FR and RL↔RR joint data,
    and negates the y-component of angular velocity and commands.

    Joint ordering (12 joints, matching GO2 SDK order):
      0 - FR_hip_joint
      1 - FR_thigh_joint
      2 - FR_calf_joint
      3 - FL_hip_joint
      4 - FL_thigh_joint
      5 - FL_calf_joint
      6 - RR_hip_joint
      7 - RR_thigh_joint
      8 - RR_calf_joint
      9 - RL_hip_joint
     10 - RL_thigh_joint
     11 - RL_calf_joint
    """
    obs = obs.clone()
    device = obs.device
    joint_num = 12
    key_body_num = 4

    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    KEY_BODY_POS_DIM = key_body_num * 3

    end_idx = 0
    # ang vel: negate y
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([-1, 1, -1], device=device)
    # root rot tan norm
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, 1, 1, -1, 1], device=device)
    # velocity command: negate y and heading
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor([1, -1, -1], device=device)
    # joint pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_atdog2_joints_left_right(obs[:, start_idx:end_idx])
    # joint vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_atdog2_joints_left_right(obs[:, start_idx:end_idx])
    # last actions
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_atdog2_joints_left_right(obs[:, start_idx:end_idx])
    # key body pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_atdog2_key_body_pos_left_right(obs[:, start_idx:end_idx])

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor."""
    actions = actions.clone()
    actions[:] = _switch_atdog2_joints_left_right(actions[:])
    return actions


"""
Joint ordering for ATDog2 (matching GO2 SDK order: FR, FL, RR, RL):
  0 - FR_hip_joint
  1 - FR_thigh_joint
  2 - FR_calf_joint
  3 - FL_hip_joint
  4 - FL_thigh_joint
  5 - FL_calf_joint
  6 - RR_hip_joint
  7 - RR_thigh_joint
  8 - RR_calf_joint
  9 - RL_hip_joint
 10 - RL_thigh_joint
 11 - RL_calf_joint

Left-right symmetry swaps:
  FR (0,1,2) <-> FL (3,4,5)
  RR (6,7,8) <-> RL (9,10,11)

The policy uses MuJoCo/CSV sign convention where right-side thigh/calf positive direction
is opposite to left-side (due to URDF RPY differences). Therefore, after swapping L↔R,
ALL joint values need sign flipping — not just hip/roll joints.
"""


def _switch_atdog2_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor.

    Swaps FR↔FL and RR↔RL joints, and negates all joint values after swapping.
    The policy uses MuJoCo/CSV sign convention where L/R thigh/calf have opposite
    positive directions, so swapping a left value to a right joint requires sign flip
    and vice versa.
    """
    joint_data_switched = joint_data.clone()

    # Swap FR <-> FL
    fr_indices = [0, 1, 2]
    fl_indices = [3, 4, 5]
    joint_data_switched[..., fr_indices] = joint_data[..., fl_indices]
    joint_data_switched[..., fl_indices] = joint_data[..., fr_indices]

    # Swap RR <-> RL
    rr_indices = [6, 7, 8]
    rl_indices = [9, 10, 11]
    joint_data_switched[..., rr_indices] = joint_data[..., rl_indices]
    joint_data_switched[..., rl_indices] = joint_data[..., rr_indices]

    # Negate ALL joints after swap:
    # - Hip (roll) joints: L/R have opposite sign convention
    # - Thigh/calf joints: MuJoCo/CSV convention has opposite positive directions on L/R sides,
    #   so swapping a left value to a right joint requires sign flip and vice versa
    joint_data_switched *= -1.0

    return joint_data_switched


def _switch_atdog2_key_body_pos_left_right(key_body_pos: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor.

    Key bodies for quadruped (in order, matching GO2 SDK order):
      0 - FR_calf
      1 - FL_calf
      2 - RR_calf
      3 - RL_calf

    Left-right swaps FR↔FL and RR↔RL, and negates the y-coordinate.
    """
    key_body_pos_switched = key_body_pos.clone()

    # Swap FR_calf <-> FL_calf
    key_body_pos_switched[..., 0:3] = key_body_pos[..., 3:6]
    key_body_pos_switched[..., 3:6] = key_body_pos[..., 0:3]

    # Swap RR_calf <-> RL_calf
    key_body_pos_switched[..., 6:9] = key_body_pos[..., 9:12]
    key_body_pos_switched[..., 9:12] = key_body_pos[..., 6:9]

    # Negate y-coordinate for all key bodies
    key_body_pos_switched[..., 1::3] *= -1.0

    return key_body_pos_switched
