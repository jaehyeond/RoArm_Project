# Path D — Task Decomposition (P6v14a pick + release BC) Design

## Hypothesis
P6v14a/model_499 already solves PICK (iter 0 stage4=0.37). The 4-attempt PPO failure
is specifically in the RELEASE path (gripper_open → sponge falls → stage4_success).
Solution: **freeze P6v14a as pick policy, train a tiny release-only BC, hardcode handoff**.

By construction:
- No forgetting (pick policy never updated)
- Release task is trivially simple (~10 ctrl steps: gripper open from closed)
- No PPO finetune needed → no catastrophic 1-iter shift

## 3-Phase Pipeline

### Phase D.1: Generate Release Demos
**Method A (Recommended)**: P6v14a rollout sweep, save only successful (jackpot=1) episodes.
- Run 200 episodes via curriculum_pregrasp init (TCP 5cm above target, sponge attached, gripper closed).
- Filter episodes where `place_success_flag==True` at any step.
- Save (obs, action) trajectory for steps T_grasp_release → T_release+5.
- ~74 demos expected from 200 rollouts (P6v14a iter 499 stage4≈0.37).

**Method B**: Synthetic scripted demos (Isaac Sim-free).
- Random init pose (sponge in hand + TCP near target).
- Scripted action [0, 0, 0, 0, 0, -1.0] for 10 steps (gripper open delta).
- Pure kinematics, no physics.
- Faster but less realistic (no contact dynamics).

→ Use Method A. Script: `roarm_rl/scripts/gen_release_demos_from_rollout.py` (~120 LOC)

### Phase D.2: Train Release BC
- MLP: 28 → 64 → 6, ELU, MSE loss.
- ~70 demos × 10 steps = 700 (obs, action) pairs.
- 1000 epochs, batch=32, Adam lr=1e-3.
- Validation: 80/20 split, train ~3min on B200 CPU.

→ Script: `roarm_rl/scripts/train_release_bc.py` (~80 LOC)

### Phase D.3: Eval State Machine Deploy
- Two policies: P6v14a/model_499 (pick), release_bc.pt (release).
- State machine:
  - Mode 1 (PICK): obs → pick policy → action. Trigger transition: `_was_grasped & sponge_z > target_z+thresh & d_xy < 0.10`.
  - Mode 2 (RELEASE): obs → release_bc → action. Episode-locked once entered.
- 500 episodes eval, record stage4_success_rate, jackpot_fire_rate.

→ Script: `roarm_rl/scripts/eval_statemachine.py` (~100 LOC) or add `--state_machine` flag
  to existing `eval_policy.py`.

## Launch Scripts (3 sequential)
1. `launch_p6v17a_release_demos.sh` — Phase D.1, ETA ~5min (200 episodes B200)
2. `launch_p6v17b_release_bc.sh` — Phase D.2, ETA ~3min
3. `launch_p6v17c_statemachine_eval.sh` — Phase D.3, ETA ~5min eval

Total ~15min wall-time.

## Critical Risks
1. **Handoff brittleness**: Transition trigger threshold sensitive. If pick policy hovers
   at slightly different position than BC demos saw, BC sees OOD state. Mitigation:
   train release BC on actual P6v14a trajectories (Method A), not synthetic.
2. **Pick policy gripper bias**: P6v14a learned grasp-hold; at handoff, gripper may
   already be releasing accidentally. Mitigation: log gripper q at transition.
3. **Sponge orientation drift during pick**: If P6v14a tips sponge, release BC trained
   on upright trajectories may fail. Mitigation: filter Phase D.1 successful episodes
   for `upright==True` at transition.

## Success Criteria (vs P6v15 baseline stage4=0.011)
- Phase D.3 stage4_success_rate ≥ 0.30 → SIGNIFICANT improvement
- ≥ 0.50 → SUCCESS (publishable result)
- < 0.10 → Path D FAIL, root cause is not "release skill missing"

## Dependencies
- P6v14a/model_499.pt (B200, ready)
- roarm_stack_env.py with curriculum_pregrasp (already exists, md5 ff31c5a...)
- new: gen_release_demos_from_rollout.py, train_release_bc.py, eval_statemachine.py
- new: 3 launch scripts

## Decision Gate
Skip Path D if:
- Path B (p6v16 RPL) shows stage4 ≥ 0.30 at iter 50 → RPL is the answer, no need to decompose
- Path D is FALLBACK if RPL also fails (forgetting was symptom, not root cause)
