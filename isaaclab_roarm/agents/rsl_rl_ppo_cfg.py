# Copyright (c) 2024, RoArm Project
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for RoArm-M3-Pro tasks."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RoArmReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for RoArm reach task."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "roarm_reach"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=8,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RoArmGraspPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for RoArm grasping task.

    Grasping requires longer training and larger networks due to:
    1. Multi-phase behavior (reach → grasp → lift)
    2. More complex reward shaping
    3. Object dynamics and contact handling
    """

    num_steps_per_env = 32  # More steps per env for multi-phase learning
    max_iterations = 3000   # Longer training for grasping
    save_interval = 100
    experiment_name = "roarm_grasp"
    run_name = ""
    logger = "tensorboard"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,  # Lower initial noise for stability
        actor_obs_normalization=True,  # Enable obs normalization for stability
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],  # Larger network for grasping
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Slightly higher entropy for exploration
        num_learning_epochs=5,  # Fewer epochs to prevent overfitting
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Lower learning rate for stability
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,  # Slightly looser KL constraint
        max_grad_norm=0.5,  # Stricter gradient clipping
    )
