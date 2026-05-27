"""roarm_rl package — gym env registration only.

NOTE: do NOT import env classes at top-level. They transitively import `pxr`
(USD) which requires AppLauncher to be initialized FIRST. gym.register's
entry_point string is resolved lazily by gym.make() after AppLauncher init.
"""
import gymnasium as gym

gym.register(
    id="RoArm-Pick-Direct-v0",
    entry_point="roarm_rl.roarm_pick_env:RoArmPickEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "roarm_rl.roarm_pick_env:RoArmPickEnvCfg",
        "rsl_rl_cfg_entry_point": "roarm_rl.agents.rsl_rl_ppo_cfg:RoArmPickPPORunnerCfg",
    },
)

gym.register(
    id="RoArm-Stack-Direct-v0",
    entry_point="roarm_rl.roarm_stack_env:RoArmStackEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "roarm_rl.roarm_stack_env:RoArmStackEnvCfg",
        "rsl_rl_cfg_entry_point": "roarm_rl.agents.rsl_rl_ppo_cfg:RoArmPickPPORunnerCfg",
    },
)

gym.register(
    id="RoArm-CubePush-Direct-v0",
    entry_point="roarm_rl.roarm_cube_push_env:RoArmCubePushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "roarm_rl.roarm_cube_push_env:RoArmCubePushEnvCfg",
        "rsl_rl_cfg_entry_point": "roarm_rl.agents.rsl_rl_ppo_cfg:RoArmPickPPORunnerCfg",
    },
)
