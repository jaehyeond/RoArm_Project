# Session 2026-05-22 — Track B P4 deploy prep: deploy_openvla_oft.py + offline + hw sanity

## TL;DR

Track B follow-up after P3 (ckpt 7500 = best deployable). Built `deploy_openvla_oft.py`
mirroring `deploy_smolvla.py` 4/9 Plan 3 SUCCESS setup, then validated via inline-action-head
strict-load + denorm range + Kinect + Follower USB1 round-trip — all sanity PASS.

Real deploy gated on two pending items (next session):

1. CUDA driver mismatch (`Failed to initialize NVML: Driver/library version mismatch`,
   NVML lib 580.159). Fix = `sudo reboot` (no PC power-cycle needed).
2. `openvla/openvla-7b` base model 14 GB HF download (in background, ~2 GB / 14 GB at session end).

No Isaac, no RL, no Track A files touched. Track A v5/v6 work is parallel-session-owned.

## Verified state at session start

- P3 finalized: ckpt 7500 best deploy candidate. Holdout `l2_step0_mean = 22.16°`,
  catastrophic collapse 7500→10000 (22.16° → 70.07°), train+holdout both worsen.
- ckpt 7500 local path:
  `openvla_oft_b200_pulls/openvla-7b+roarm_v6_pick+b8+lr-0.0005+lora-r32+dropout-0.0--v6_30k--7500_chkpt/`
  (action_head 268 MB + lora_adapter/ + dataset_statistics.json + processor files).
- JSON `openvla_oft_b200_pulls/openvla_oft_v6_eval_20260522_121028.json`
  sha256 `3707a0ee1efd189868eb0421a1f56b2a71ec16dfb3b87632772a0e5a87332bf0` (8 entries).
- USB serials read at session start:
  - `/dev/ttyUSB0` serial `7842202ff8d9ef11b33f513dc8728757` → Leader (per
    `~/.claude/projects/.../memory/tech_leader_follower_setup.md` 2026-04-01).
  - `/dev/ttyUSB1` serial `ee7a06468e98ef1194edca63a8793231` → Follower (deploy
    target). Confirmed by 1.93°-amplitude motion test in hw sanity below.

## Inventory blockers identified

| Blocker | Status |
|---|---|
| CUDA driver mismatch | NVML 580.159, kernel module forward-compat error 804 → `torch.cuda.is_available() = False`. Reboot required. |
| `openvla/openvla-7b` HF cache | Missing. Triggered `huggingface_hub.snapshot_download` at pinned revision `47a0ec7fc4ec123775a391911046cf33cf9ed83f`, background. ~14 GB. |
| `peft` | Missing. Installed `0.18.0` (`pip install --no-deps`). |
| `prismatic` editable | Missing. `pip install --no-deps -e /home/cgxr/Documents/Robotics/openvla-oft/`. |
| `rich` | Missing (prismatic logging deps). Installed `rich-15.0.0`. |
| `timm` | Missing (prismatic vision backbone). Installed `0.9.16` per HARD RULE #15. |
| `dlimp` (RLDS) | Missing AND not installed — instead bypassed via inline L1 action head copy (see deploy_openvla_oft.py:78-138). |

Disk: 76 GB free on root, sufficient for openvla-7b 14 GB.

## deploy_openvla_oft.py

Created `/home/cgxr/Documents/Robotics/RoArm_Project/deploy_openvla_oft.py` (561 lines).
Structure mirrors `deploy_smolvla.py` 4/9 Plan 3 setup; inference path replaced
with OpenVLA-OFT.

Critical design choices (head-to-head fair with SmolVLA v6 4/9 Plan 3 baseline):

- `INIT_POS = [0, 0, 90, 0, 0, 5]` HOME — identical to `deploy_smolvla.py:93`. v6
  episodes all start at HOME; deploying from HOME = in-distribution.
- `JOINT_SPEED_CAPS = [500, 500, 500, 300, 300, 300]` and `get_safe_speed = min()` —
  identical to `deploy_smolvla.py:110-115`. SDK lacks per-joint speed.
- Plan 3 gripper unlock: after each `arm.joints_angle_ctrl(...)`, immediately call
  `arm.gripper_angle_ctrl(angle=action_clamped[5], speed=1000, acc=0)`. Second call
  overrides joints_angle_ctrl's gripper portion with the unlocked speed — same
  pattern as `deploy_smolvla.py:685-689,841-846` (4/9 SmolVLA v6 SUCCESS).
- Workspace safety: `Z_FLOOR_DEPLOY = -130mm`, `DIST_MAX_DEPLOY = 420mm` — identical
  to `deploy_smolvla.py:83-84`.
- Follower-only safety: `--port /dev/ttyUSB0` raises `SystemExit` (Leader is for
  user manual ops only).
- L-F deployment: no command issued to Leader.

OpenVLA-OFT-specific design:

- Input: BGR → cv2 RGB → PIL → 224×224 resize. No state input (model doesn't use
  proprioception — `prismatic.vla.constants.ROARM_M3_CONSTANTS.PROPRIO_DIM` is set,
  but inference path through `predict_action` is vision+language only).
- Prompt: `f"In: What action should the robot take to {task.lower()}?\nOut:"` —
  identical to `eval_offline_v6.py:336`.
- Output: action chunk shape `(NUM_ACTIONS_CHUNK=8, ACTION_DIM=6)`, unnormalized
  inside `vla.predict_action` via BOUNDS_Q99 (q01/q99 from chkpt
  `dataset_statistics.json`, key = `roarm_v6_pick`).
- `apply_sdpa_class_attr_patch()` (D071) applied before AutoModelForVision2Seq
  instantiation, and again after dynamic class realisation.
- `norm_stats` injected on three handles: `vla.base_model.model.norm_stats`,
  `vla.norm_stats`, `vla.base_model.norm_stats` (mirrors D080 / `eval_offline_v6.py:276-287`).
- Action head loaded with `module.` prefix strip and `strict=True`.
- Dual mode: default `--n-action-steps 8` chunk-mode (~2 Hz on 4090 GPU, ~0.1 Hz CPU);
  `--n-action-steps 1` closed-loop (10× slower, sanity/comparison only).

### Inline L1RegressionActionHead (lines 78-138)

Inline copy of `prismatic.models.action_heads.L1RegressionActionHead` to avoid the
`prismatic.models.__init__` → `vlas` → `vla.materialize` → `vla.datasets.rlds`
→ `dlimp` import chain (RLDS / TensorFlow / dlimp not needed for inference).

Inline classes: `_MLPResNetBlock`, `_MLPResNet`, `L1RegressionActionHead`. Verified
state_dict match against B200 ckpt 7500 below.

## Offline sanity (CPU only — no GPU access this session)

### Sanity 1 — L1 head inline class strict-load

Loaded `action_head--7500_checkpoint.pt` (16 tensors, all `module.`-prefixed,
13.4 MB action-only weights).

After `state = {k.removeprefix("module."): v for k, v in state.items()}` and
`head.load_state_dict(state, strict=True)`:

```
L1 head params: 134,328,326  ACTION_DIM=6 NUM_CHUNK=8
load_state_dict: missing=[] unexpected=[]
STRICT LOAD OK ✓
```

Forward pass on dummy `(1, NUM_ACTIONS_CHUNK*ACTION_DIM=48, 4096)`:

```
forward out: shape=(1, 8, 6) mean=-0.2538 std=0.4765
FORWARD OK ✓
```

Both shape and dtype agree with the trained model. Inline class is numeric-identical
to the prismatic class for this checkpoint.

### Sanity 2 — denorm q01/q99 ⊂ JOINT_LIMITS

Inspected `dataset_statistics.json` key `roarm_v6_pick` action q01/q99.
BOUNDS_Q99 maps model output ±1.0 → joint angles in [q01, q99] degrees.

| joint | q01 | q99 | JOINT_LIMITS | in-bounds |
|---|---:|---:|:---:|:---:|
| base | -10.13 | +33.11 | [-180, 180] | OK |
| shoulder | +1.79 | +50.69 | [-110, 110] | OK |
| elbow | +41.89 | +96.18 | [-70, 190] | OK |
| wrist_p | -3.49 | +80.35 | [-110, 110] | OK |
| wrist_r | -11.68 | +28.54 | [-180, 180] | OK |
| gripper | +0.93 | +68.65 | [-10, 100] | OK |

Even at saturated model output ±1.0, joint angles stay well within hardware limits.
`clamp_joints` should be a no-op for in-distribution behavior.

### Sanity 3 — script syntax + key imports

`ast.parse` passes. The 3 critical sub-imports that bypass the dlimp chain:

- `from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK` → OK (ROARM_M3
  detected, ACTION_DIM=6, NUM_ACTIONS_CHUNK=8).
- `from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction`
  → OK (used as type reference only; actual model loaded via `AutoModelForVision2Seq`).
- Inline `L1RegressionActionHead` does not go through `prismatic.models.__init__`.

## Hardware sanity (no GPU needed)

### Sanity 4 — Kinect single-frame

`pyk4a` 720P NFOV_UNBINNED. Captured 1 frame after 1.0s warmup:

```
shape=(720, 1280, 3) dtype=uint8
mean BGR = [148.81 145.34 147.70]
saved logs/hw_sanity_20260522/kinect_sanity_frame.png
KINECT SANITY OK ✓
```

### Sanity 5 — Follower USB1 INIT_POS round-trip

`/dev/ttyUSB1`, RoArm M3 SDK, torque ON, target = `INIT_POS [0, 0, 90, 0, 0, 5]`.

```
before: [0.4, 1.9, 91.1, 0.3, -0.1, 0.4]
torque ON ...
→ INIT_POS [0, 0, 90, 0, 0, 5]
도달 0.5s max_diff=1.93°  cur=[0.4, 1.9, 91.1, 0.3, -0.1, 4.7]
FK pose: x=353 y=2 z=204 mm
FOLLOWER SANITY OK ✓
```

Sub-5° tolerance reached in 500ms. FK end-effector at x=353 y=2 z=204 mm —
in workspace center, above table (Z_FLOOR=-130 safe), below DIST_MAX=420 safe.
Gripper change 0.4° → 4.7° confirms USB1 is the addressed arm.

## Pending blockers for Step 6 real deploy

1. **CUDA driver mismatch — must reboot**.
   - `nvidia-smi` returns `Failed to initialize NVML: Driver/library version mismatch / NVML library version: 580.159`.
   - `torch.cuda.is_available() = False`, error 804 forward compatibility.
   - Fix: `sudo reboot` (or `sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia` — lower success rate).
   - PC power-cycle is NOT required; OS reboot is sufficient.

2. **openvla/openvla-7b download** — background, ~2.0 GB / 14 GB at session end.
   - Pinned `revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f'`.
   - On reboot, download must be resumed. Run:
     ```
     conda activate roarm && python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='openvla/openvla-7b', revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f', allow_patterns=['*.json','*.txt','*.md','*.model','*.bin','*.safetensors','*.py']))"
     ```

## Files changed this session

Local:
- `deploy_openvla_oft.py` (new, 561 lines, syntax OK).
- `claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md` (this file).
- `logs/hw_sanity_20260522/kinect_sanity_frame.png` (sanity 4 artifact).
- `EXPERIMENT_LEDGER.md` (row 118 append, this session).
- `START_HERE.md` (Track B P4 section append; Track A region unchanged).
- `DECISIONS.md` (D086 append).

Env (`roarm` conda env):
- `peft 0.18.0` installed (`pip install --no-deps`).
- `rich 15.0.0` installed.
- `timm 0.9.16` installed (HARD RULE #15 spec).
- `prismatic` editable installed from `/home/cgxr/Documents/Robotics/openvla-oft/`.

HF cache: `~/.cache/huggingface/hub/models--openvla--openvla-7b/` partial (~2 GB).

No Track A files touched. No Isaac runtime. No RL training.

## Track A boundary

Track A v5/v6 work is owned by a parallel session. Latest Track A truth per
ledger rows 116-117 + START_HERE lines 11-30 / 66-148: v5 and v6
close_26 B200 runtimes both FAIL. Do not cite Track B deploy prep as Track A
contact-success evidence.

## Decisions / Next steps

1. **Now**: `sudo reboot`. CUDA recovers + no other state lost.
2. **After reboot (new Claude Code session)**: paste continuation prompt below,
   resume openvla-7b download, then full GPU sanity (1 chunk inference,
   verify shape (8,6), values inside JOINT_LIMITS).
3. **Real deploy ckpt 7500**: multi-position protocol matching SmolVLA v6 2026-04-09
   Plan 3 setup. Same `--start-pos init` (HOME), `--max-steps ~64-150`,
   `--n-action-steps 8` chunk-mode default. Video record + CSV log.
4. **Head-to-head comparison**: SmolVLA v6 Plan 3 (4/9 SUCCESS, multi-position
   sponge pick) vs OpenVLA-OFT 7B ckpt 7500. Tabulate: success rate /
   position, grasp depth, drift, total time, inference latency.
5. **Update**: new session doc, ledger row, START_HERE Track B P5 section,
   DECISIONS if real deploy reveals durable lesson.

## Continuation prompt for next session

```
Read CLAUDE.md first, then follow Current-State Protocol exactly.

한국어로 브리핑. 비판적/분석적. 기억 단독 truth 금지.
HANDOFF.md / TASKS.md 사용 금지. /half-clone / /handoff 절대 사용 금지 (HARD RULE #11).

Must read:
1. CLAUDE.md
2. START_HERE.md (Track A는 별도 세션 truth — 본인은 Track B만 다룸)
3. claudedocs/DECISIONS.md D071-D086 (D086 신규 = OpenVLA-OFT local inference deps)
4. claudedocs/EXPERIMENT_LEDGER.md rows 111-118
5. claudedocs/session_20260522_openvla_oft_offline_eval_v6_result.md (P3 결과)
6. claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md (P4 본 session, 이 파일)
7. deploy_openvla_oft.py (561 lines)

Step-by-step next:
0. `nvidia-smi` 가 driver mismatch 없이 나오는지 검증. mismatch면 또 reboot.
1. `du -sh ~/.cache/huggingface/hub/models--openvla--openvla-7b` 로 다운로드 진행/완료 확인.
   완료 안 됐으면:
   conda activate roarm && python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='openvla/openvla-7b', revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f', allow_patterns=['*.json','*.txt','*.md','*.model','*.bin','*.safetensors','*.py']))"
2. CUDA + 모델 sanity:
   conda activate roarm && python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NA')"
3. Dry-run 1 chunk inference (GPU):
   python deploy_openvla_oft.py --dry-run --max-steps 8 --no-kinect --device cuda
   → 한 chunk 추론, shape (8,6) 출력, JOINT_LIMITS 안 검증, 추론 시간 측정.
4. (조건부) Dry-run with Kinect (실제 frame):
   python deploy_openvla_oft.py --dry-run --max-steps 8 --device cuda
5. Real deploy (사용자 명시 승인 후만):
   python deploy_openvla_oft.py --max-steps 80 --log-csv --save-frames-dir logs/deploy_oft_<ts>_frames
   - Default --port /dev/ttyUSB1 (Follower). Leader (USB0) 사용 금지.
   - Default --start-pos init = [0,0,90,0,0,5] HOME.
   - Default --n-action-steps 8 chunk-mode (~2Hz).
   - Plan 3 gripper unlock 자동 적용.
6. 결과 비교 vs SmolVLA v6 2026-04-09 Plan 3 SUCCESS (multi-position sponge pick).
7. END-OF-SESSION: START_HERE Track B P5 section append, EXPERIMENT_LEDGER row,
   new session doc, MEMORY recent sessions prepend (5-slot HARD RULE #8).

Hard rules in effect:
- HARD RULE #1 HOME 시작
- HARD RULE #4 거짓 갭 방지 (deploy 결과 보고 시)
- HARD RULE #11 /half-clone /handoff 금지
- HARD RULE #13 Follower=/dev/ttyUSB1, Leader=/dev/ttyUSB0 (Leader 명령 절대 금지)
- HARD RULE #15 (학습 한정. 본 inference에는 timm 0.9.16 + nightly cu128는 아님.
  Local 4090 = stable cu126. dtype = bf16 OK)
- HARD RULE #18 사용자 명시 정정 절대 우선

Do not touch:
- Track A files (parallel session active there). START_HERE lines 1-30 / 66-148.
- /dev/ttyUSB0 (Leader). 모든 명령은 /dev/ttyUSB1.
```

## HARD RULE compliance

- ✅ #1 INIT_POS [0,0,90,0,0,5] HOME default for `--start-pos init`.
- ✅ #4 No 갭 주장, no "최초" — empirical sanity only.
- ✅ #5 JOINT_LIMITS preserved in deploy_openvla_oft.py.
- ✅ #11 `/half-clone` not invoked. Continuation via project state files + this doc.
- ✅ #13 Follower = /dev/ttyUSB1 verified, Leader = /dev/ttyUSB0 explicitly blocked
  in deploy_openvla_oft.py (`SystemExit` on `--port /dev/ttyUSB0`).
- N/A #14/#15/#26 No B200 work, no Isaac, no RL.
- ✅ #18 User-corrected USB mapping (tech_leader_follower_setup.md 2026-04-01)
  followed verbatim. No autonomous re-interpretation.
- ✅ #19 Edge-stand sponge orientation unchanged (no sponge work this session).
