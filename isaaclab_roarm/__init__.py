# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""RoArm-M3-Pro Isaac Lab configurations."""

from .roarm_cfg import ROARM_M3_CFG
from .roarm_reach_env_cfg import RoArmReachEnvCfg, RoArmReachEnvCfg_PLAY
from .roarm_reach_env_cfg_sim2real import RoArmReachEnvCfg_Sim2Real, RoArmReachEnvCfg_Sim2Real_PLAY
from .roarm_grasp_env_cfg import RoArmGraspEnvCfg, RoArmGraspEnvCfg_PLAY

# Register environments with gymnasium
from . import register_envs

__all__ = [
    "ROARM_M3_CFG",
    "RoArmReachEnvCfg",
    "RoArmReachEnvCfg_PLAY",
    "RoArmReachEnvCfg_Sim2Real",
    "RoArmReachEnvCfg_Sim2Real_PLAY",
    "RoArmGraspEnvCfg",
    "RoArmGraspEnvCfg_PLAY",
]
