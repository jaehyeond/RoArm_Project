"""
sim_isaac_failure_analysis.py
[A2 SIM2REAL] Isaac Lab → RoArm M3 sim-to-real failure analysis

This is a DOCUMENTATION file, not executable code.
It records why the Isaac Lab reaching policy failed and what would be needed
for a correct sim-to-real transfer.

Analysis date: 2026-03-26
Training artifact: /home/cgxr/Documents/Robotics/isaac_roarm_m3/logs/rsl_rl/roarm_m3_reach/2026-02-20_15-30-43/
"""

# =============================================================================
# FAILURE ROOT CAUSES (ranked by severity)
# =============================================================================

FAILURE_ANALYSIS = {

    "1_training_not_converged": {
        "severity": "CRITICAL",
        "evidence": {
            "iterations_run": 100,
            "position_error_at_100iter": "0.097m",
            "expected_convergence_iter": "1000-2000",
            "franka_reach_baseline_convergence": "~1500 iter for <0.02m error",
        },
        "explanation": (
            "100 iterations = 100 * 24 steps/env * 512 envs = 1,228,800 total steps. "
            "At 0.097m error the policy is still in early exploration phase. "
            "A fully trained policy needs <0.02m to be deployable. "
            "At 49K steps/sec this was only 26 seconds of training."
        ),
        "fix": "Train for minimum 1000 iterations (~4 minutes). Check play.py error < 0.02m before deploying.",
    },

    "2_action_space_mismatch": {
        "severity": "CRITICAL",
        "sim_action": {
            "type": "JointPositionAction",
            "mode": "relative_delta",
            "scale": 0.5,
            "unit": "radians",
            "use_default_offset": True,
            # output = default_joint_pos + scale * network_output
        },
        "real_robot_api": {
            "function": "arm.joints_angle_ctrl(angles=[...], speed=500, acc=200)",
            "mode": "absolute",
            "unit": "degrees",
        },
        "required_conversion": """
def sim_action_to_real_command(
    sim_action_output: np.ndarray,  # shape (5,), radians delta
    current_joint_pos_deg: np.ndarray,  # shape (6,) from arm.joints_angle_get()
    default_joint_pos_rad: np.ndarray,  # from ArticulationCfg.init_state.joint_pos
    scale: float = 0.5,
) -> np.ndarray:
    # sim policy outputs: delta from default, in radians, pre-scaled by 0.5
    # NOTE: scale is already baked in by JointPositionAction before we receive output
    # The raw NN output * scale = what we get
    target_rad = default_joint_pos_rad[:5] + sim_action_output  # absolute rad
    target_deg = np.degrees(target_rad)
    return np.clip(target_deg, JOINT_MIN_DEG[:5], JOINT_MAX_DEG[:5])
        """,
    },

    "3_control_frequency_mismatch": {
        "severity": "HIGH",
        "sim_frequency_hz": 30,  # 60Hz physics / decimation=2
        "sim_dt_s": 0.03333,
        "real_robot_expected_latency_ms": "~15-50ms (USB serial + internal CPU)",
        "explanation": (
            "Sim policy expects to run at exactly 30Hz. "
            "Real deployment loop has variable latency from USB serial + SDK overhead. "
            "Without rate limiting, the policy runs too fast and accumulates position error "
            "differently than during sim training."
        ),
        "fix": """
import time
CONTROL_HZ = 30
dt = 1.0 / CONTROL_HZ
while True:
    t_start = time.time()
    obs = get_observation()
    action = policy.act(obs)
    send_to_robot(action)
    elapsed = time.time() - t_start
    time.sleep(max(0, dt - elapsed))
        """,
    },

    "4_no_domain_randomization": {
        "severity": "HIGH",
        "what_was_randomized": ["initial joint positions (position_range 0.5-1.5x default)"],
        "what_was_NOT_randomized": [
            "link masses (no MassPropertiesCfg)",
            "joint friction (friction=null in actuator config)",
            "actuator delays (no delay simulation)",
            "external disturbances",
            "ground friction (static=0.5, dynamic=0.5 fixed)",
        ],
        "minimum_dr_for_transfer": {
            "mass_variation": "±20% on each link",
            "joint_friction": "uniform(0.05, 0.3) per joint",
            "action_delay": "0-2 steps delay buffer",
            "observation_noise": "already present (Unoise -0.01 to +0.01)",
        },
    },

    "5_actuator_gains_not_validated": {
        "severity": "MEDIUM",
        "sim_gains": {
            "base_link_to_link1": {"stiffness": 200.0, "damping": 80.0},
            "link1_to_link2": {"stiffness": 170.0, "damping": 65.0},
            "link2_to_link3": {"stiffness": 120.0, "damping": 45.0},
            "link3_to_link4": {"stiffness": 80.0, "damping": 30.0},
            "link4_to_link5": {"stiffness": 50.0, "damping": 20.0},
        },
        "real_servo": "ST3235 with internal PID (P~32, I~0, D~0 default)",
        "explanation": (
            "ImplicitActuator gains in Isaac Lab are NOT the servo's internal PID. "
            "They are physics engine parameters controlling the PhysX joint drive. "
            "The actual ST3235 has its own closed-loop control. "
            "To match them properly, system identification is needed: "
            "command step inputs, measure response, fit 2nd-order system model."
        ),
        "impact": (
            "Policy trained with these gains will have different response dynamics "
            "than the real servo. The real servo may overshoot or undershoot "
            "differently than simulated."
        ),
    },

    "6_observation_pose_command_unavailable": {
        "severity": "HIGH",
        "sim_obs_vector": {
            "joint_pos_rel": "5 dims, relative to default, radians",
            "joint_vel_rel": "5 dims, relative to default, rad/s",
            "pose_command": "7 dims, target EE pose (xyz + quat) in robot frame",
            "last_action": "5 dims, previous policy output",
            "total": "22 dims",
        },
        "problem": (
            "In sim, pose_command is auto-generated from UniformPoseCommandCfg. "
            "In real deployment, you need to provide the target EE pose. "
            "This requires either: "
            "(1) a pre-defined target position, or "
            "(2) camera-based object detection + coordinate transform. "
            "Without (2), the RL policy is effectively blind to object location."
        ),
        "conclusion": (
            "This is why VLA is better for pick-and-place: "
            "VLA takes camera image as input and infers where to go. "
            "RL reach policy requires explicit target coordinates."
        ),
    },
}


# =============================================================================
# TRAINING ADEQUACY ANALYSIS
# =============================================================================

TRAINING_ANALYSIS = {
    "actual_training": {
        "iterations": 100,
        "steps": 100 * 24 * 512,  # 1,228,800
        "wall_time_s": 26,
        "final_position_error_m": 0.097,
        "checkpoint_saved": ["model_0.pt", "model_50.pt", "model_99.pt"],
    },
    "minimum_required": {
        "iterations": 1000,
        "steps": 1000 * 24 * 512,  # 12,288,000
        "wall_time_s": 260,  # ~4.3 minutes at 49K steps/sec
        "expected_position_error_m": "~0.02-0.04",
    },
    "recommended": {
        "iterations": 2000,
        "steps": 2000 * 24 * 512,  # 24,576,000
        "wall_time_s": 520,  # ~9 minutes
        "expected_position_error_m": "<0.02",
    },
    "verdict": (
        "Training was terminated at roughly 8% of the minimum required iterations. "
        "The policy had not learned meaningful reaching behavior. "
        "Deploying a 100-iter policy is equivalent to deploying random noise."
    ),
}


# =============================================================================
# IS IT WORTH RETURNING TO ISAAC LAB?
# =============================================================================

RETURN_ASSESSMENT = {
    "current_stage": "Stage 1 (basic pick-and-place with vision)",
    "verdict": "NO - not worth returning now",
    "reasoning": [
        "Stage 1 goal requires vision (camera → object location). "
        "Isaac Lab RL reach policy has no vision input.",
        "VLA solves the full pipeline (image + language → action) directly.",
        "Current bottleneck is data quantity, not physics simulation accuracy.",
        "Isaac Lab RL would require a separate object detection pipeline "
        "to provide pose_command, effectively duplicating VLA's job.",
    ],
    "when_to_return": [
        "Stage 2+: after basic pick-and-place works reliably",
        "When real data collection is the bottleneck (not enough demos)",
        "If researching sim-to-real transfer itself (CoRL paper direction)",
        "If attempting trajectory optimization or contact-rich tasks",
    ],
    "what_abandoning_was_NOT": (
        "Abandoning Isaac Lab was not wrong in direction, but wrong in method. "
        "Should have: trained to convergence, built proper action conversion, "
        "tested in sim play mode, then analyzed the gap. "
        "Instead: ran 100 iters (26 sec), saw it fail, moved on. "
        "No diagnostic information was gathered."
    ),
}


# =============================================================================
# COMBINING ISAAC LAB + VLA (future consideration)
# =============================================================================

COMBINATION_STRATEGIES = {
    "strategy_1_trajectory_priors": {
        "feasibility": "MEDIUM",
        "description": "Use Isaac Lab to generate reaching trajectories → add to VLA training data",
        "pipeline": [
            "1. Train Isaac Lab reach policy to convergence (2000+ iter)",
            "2. Run policy in sim, export HDF5 episodes",
            "3. Convert HDF5 → LeRobot v3 format (parquet + video)",
            "4. Combine with real demos for SmolVLA training",
        ],
        "blocker": (
            "SigLIP vision encoder is trained on real images. "
            "Sim rendering features may not align with real features. "
            "UNTESTED: need to compare SigLIP embeddings of sim vs real frames."
        ),
        "effort_estimate": "2-3 weeks for pipeline + validation",
    },

    "strategy_2_demonstration_generation": {
        "feasibility": "LOW-MEDIUM",
        "description": "SplatSim-style: scripted IK demos in sim with photorealistic rendering",
        "blockers": [
            "No 3DGS reconstruction of real scene done yet",
            "Need domain randomization for textures/lighting",
            "Photorealistic rendering = slow data generation",
        ],
        "effort_estimate": "4-6 weeks minimum",
    },

    "strategy_3_sim_as_safety_filter": {
        "feasibility": "LOW",
        "description": "VLA proposes action → Isaac Lab validates feasibility → execute if safe",
        "blockers": [
            "30Hz VLA inference + Isaac Lab query latency unknown",
            "Sim model may not match real kinematics well enough for safety",
        ],
    },

    "recommendation": (
        "None of these strategies are worth pursuing before Stage 1 baseline works. "
        "Revisit after achieving >60% pick-and-place success with real VLA."
    ),
}


# =============================================================================
# WHAT SHOULD HAVE BEEN DONE (lessons learned)
# =============================================================================

CORRECT_SIM_TO_REAL_PROTOCOL = """
Correct Isaac Lab → RoArm M3 transfer protocol (for future reference):

Step 1: TRAIN TO CONVERGENCE IN SIM
    - Run for 2000+ iterations (not 100)
    - Verify play.py shows position_error < 0.02m consistently
    - Check tensorboard: reward should plateau, not still rising

Step 2: VALIDATE ACTION SPACE
    - Print sim action output range: should be [-1, 1] after scale=0.5
    - Map to real joint delta: sim_rad_delta → real_deg_target
    - Test with zero_agent.py first to verify kinematics match

Step 3: ADD DOMAIN RANDOMIZATION
    - Mass: ±20%, Friction: uniform(0.05, 0.3)
    - Re-train with DR, verify sim performance doesn't collapse
    - If performance drops >30% with DR, gains/reward need tuning

Step 4: SYSTEM IDENTIFICATION (minimum version)
    - Command a step input to each joint: 0° → 30° → 0°
    - Record actual trajectory (joints_angle_get at 30Hz)
    - Compare to sim predicted trajectory
    - Adjust stiffness/damping until sim matches real step response

Step 5: BUILD DEPLOYMENT WRAPPER
    - 30Hz control loop with time.sleep synchronization
    - Observation vector construction matching sim exactly
    - Action conversion: sim delta_rad → real absolute_deg
    - Hard joint limits (JOINT_LIMITS from CLAUDE.md)
    - Emergency stop on large velocity commands

Step 6: INCREMENTAL TESTING
    - Test in simulation play mode first (play.py)
    - Then test on real robot with low-amplitude targets (±10° from rest)
    - Gradually expand target range
    - Log all failures for gap analysis
"""

if __name__ == "__main__":
    print("This file is documentation, not executable.")
    print("See FAILURE_ANALYSIS, TRAINING_ANALYSIS, RETURN_ASSESSMENT for details.")
