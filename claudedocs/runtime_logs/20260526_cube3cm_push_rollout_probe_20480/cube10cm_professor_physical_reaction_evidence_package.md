# Professor 10cm Cube Physical-Reaction Evidence Package

## Scope

- Branch: professor 10cm/0.72kg cube push/tap DiffIK reaction-window + quality-tier.
- Local package over existing logs and visual audits only.
- This is evidence of weak physical object reaction, not action-teacher, dataset, RL, or RoArm readiness.

## Evidence

- Status: `READY_PROFESSOR_EVIDENCE_ONLY`.
- Direct-IK professor physical evidence: `PASS`.
- Max displacement along push: `0.000922590m`.
- Max speed: `0.008078601m/s`.
- Overshoot: `0.0`.
- Contact/tap success remains `0.0` / `0.0`.

## Metadata Link

- Event-label metadata: `READY_LOCAL_ONLY`.
- Events/contact/reaction/overshoot: `16` / `16` / `16` / `0`.
- Quality tiers: `{'B_REACTION_VALID_FOLLOW_OK_CLIP_HIGH': 16}`.
- Metadata is label/quality-tier only; no action payloads are included.

## Caveats

- clean_tap_visual_verified=`false`.
- grazing_or_outside_face_behavior=`true`.
- contact_gated_rl_success=`FAIL`.
- action_teacher=`BLOCKED`.
- dataset/RL/RoArm=`BLOCKED` / `BLOCKED` / `BLOCKED`.

## Pipeline Position

- P0 evidence checkpoint: this package is one checkpoint inside the integrated professor-report pipeline, not an alternative to learning/RL.
- P1/P2 learning and RL path: continue by resolving the local RL/learning blockers: contact-gated positive-control, clean teacher or explicit noisy Tier-B exception gate, tiny dry run, large dataset, PPO/RL training, and only then RoArm/generalization.

## Line Evidence

- Direct audit summary: line 4 for weak reaction PASS metrics, line 8 for RL/dataset/RoArm blockers.
- Preflight summary: line 3 for READY_PROFESSOR_EVIDENCE_ONLY, line 6 for RL/RoArm blockers.
- Event-label metadata summary: lines 1-6 for metadata-only counts and blocked downstream gates.
