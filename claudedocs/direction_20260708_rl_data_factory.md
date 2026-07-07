# Direction 2026-07-08 - RL Data Factory for Cube10cm Top-View

Created on 2026-07-07 KST. The filename preserves the requested 20260708
direction label.

## Top-Level Research Chain

The advisor-confirmed chain is:

```text
script push -> rendered pair dataset -> RL training -> large-scale RL policy
data generation (tens of thousands of episodes, GPU parallel) -> VLA training
at the end
```

RL is not the final research artifact in this chain. RL is the data-factory
engine. The 10cm cube tap/push task is not the destination; it is the fixture
used to validate the factory.

## Literature Anchor

RLDG, "Robotic Generalist Policy Distillation via Reinforcement Learning"
(arXiv:2412.09858), uses task-specialist RL policies to generate high-quality
fine-tuning data for generalist policies such as OpenVLA and Octo. The relevant
positioning for this repo is:

```text
RLDG-style specialist-RL data generation, but with sim-rendered top-view
trajectory data, perturbation robustness rows, and zero-action/scripted
baseline ablations.
```

The mm-level stop-band is not the essence of the final VLA-data objective. For
data generation, varied push displacement can be useful coverage. Quality
control should be enforced by post-hoc, reward-independent label filters, not
only by the online reward.

Primary source checked: https://arxiv.org/abs/2412.09858

## Four Risks Installed as Gates

1. Diversity collapse
   - Risk: a converged policy produces homogeneous trajectories.
   - Gate: every data-generation pilot must report condition-bin quotas,
     displacement variance, push-direction histogram, and contact/proxy
     distribution against the existing script dataset.

2. Reward defects replicated at scale
   - Risk: a subtle reward bug becomes tens of thousands of bad demonstrations.
   - Gate: generated episodes enter the training corpus only through the existing
     reward-independent label validator/filter.

3. Missing control baseline
   - Risk: without script-only VLA training, the claim "RL-generated data improves
     VLA" is not falsifiable.
   - Gate: script-only VLA baseline is a parallel experimental requirement, not
     an inversion of the advisor chain.

4. Critical path depends on perfect RL
   - Risk: the whole pipeline stalls until a deployable RL policy is promoted.
   - Gate: use generator criteria for data production. A bin with label-filter
     pass rate >=30% can produce data; promotion criteria for deployment remain
     separate.

## Phase Structure

- Phase 0, this week in parallel:
  - A: secure a data generator. D318 effectively provides baseline v2.
  - B: prepare VLA control baseline and evaluation harness.
- Phase 1:
  - Run a 500-1000 episode data-conveyor pilot.
- Phase 2:
  - Pre-registered comparison: script-only VLA vs script+RL-data VLA on the
    perturbation matrix. Proceed only if perturbed rows improve without nominal
    regression.
- Phase 3:
  - Scale to 5k-10k episodes, run VLA main training, then RoArm real checks.

## D318 Strategic Consequences

1. Baseline v2 is:

```text
candidate8 zero-action + candidate8_hybrid_stop_after_useful
```

It is distinct from D314/D310 baseline v1 (`tap_push_primitive`).

2. The current candidate8 residual PPO setup is degenerate under hybrid stop:
the hybrid latch dominates behavior, all checkpoints match zero-action, and
policy contribution is unproven. Longer PPO with the same setting is banned.

3. RL should be moved to parameters where zero-action is not sufficient:
goal-conditioned primitive displacement, push direction, stop margin, or
approach/contact offset. Every evaluation must include zero-action.

4. The data conveyor can start from baseline v2 because D318 showed useful rates
around 94-100% on the tested low-friction/nominal rows, even though it did not
prove learned-policy contribution.

## Weekly Plan

1. D319:
   - run a conveyor pilot with baseline v2;
   - audit high-friction D314 artifact status;
   - measure bin pass rates and diversity;
   - convert a small accepted sample through replay render -> LeRobot.
2. D320:
   - script-only VLA baseline and evaluation harness.
3. D321:
   - goal-conditioned RL where zero-action cannot solve the task.
4. Next week:
   - Phase 2 comparison: script-only versus script+RL-data.
