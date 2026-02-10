# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""Register RoArm environments with Gymnasium."""

import gymnasium as gym


def register_roarm_envs():
    """Register all RoArm environments."""

    # RoArm Reach - Joint Position Control (Base)
    gym.register(
        id="Isaac-Reach-RoArm-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_roarm:RoArmReachEnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_roarm.agents:RoArmReachPPORunnerCfg",
        },
    )

    # RoArm Reach - Play/Evaluation (Base)
    gym.register(
        id="Isaac-Reach-RoArm-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_roarm:RoArmReachEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "isaaclab_roarm.agents:RoArmReachPPORunnerCfg",
        },
    )

    # RoArm Reach - Sim2Real with Domain Randomization
    gym.register(
        id="Isaac-Reach-RoArm-Sim2Real-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_roarm:RoArmReachEnvCfg_Sim2Real",
            "rsl_rl_cfg_entry_point": "isaaclab_roarm.agents:RoArmReachPPORunnerCfg",
        },
    )

    # RoArm Reach - Sim2Real Play/Evaluation
    gym.register(
        id="Isaac-Reach-RoArm-Sim2Real-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_roarm:RoArmReachEnvCfg_Sim2Real_PLAY",
            "rsl_rl_cfg_entry_point": "isaaclab_roarm.agents:RoArmReachPPORunnerCfg",
        },
    )


# Auto-register when module is imported
register_roarm_envs()
