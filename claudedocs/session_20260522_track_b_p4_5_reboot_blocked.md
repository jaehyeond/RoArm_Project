# Session 2026-05-22 — Track B P4.5 post-P4 verification: reboot still pending

## TL;DR

Follow-up to P4 deploy prep
(`claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md`).
Session was invoked under the premise "reboot already done, proceed to Step 0
CUDA verification → GPU dry-run → real deploy".

Verified at session start that the premise is **false**:

- `uptime` = 1 day, 20:02; `who -b` = 2026-05-20 22:53. The P4 prep session ran
  today (2026-05-22). No reboot has occurred between P4 prep and this session.
- `nvidia-smi` still returns `Failed to initialize NVML: Driver/library version
  mismatch / NVML library version: 580.159`. Kernel module
  `/proc/driver/nvidia/version` = `580.126.09`. Userspace `libnvidia-ml.so.1`
  → `libnvidia-ml.so.580.159.03`.
- `conda run -n roarm python -c "import torch; print(torch.cuda.is_available())"`
  → `False`, `Error 804: forward compatibility was attempted on non supported HW`.

Per HARD RULE (verify from current state before citing), session paused at
Step 0 — did not run dry-run, did not load any model weights, did not issue any
robot command, did not touch any Track A file. User explicitly chose
"Reboot 후 새 세션 (권장)" via AskUserQuestion.

No code changes. No env changes. No Isaac. No RL. No Track A files touched.

## Verified state at session start

### Premise check — FAIL

| Check | Expected | Actual | Verdict |
|---|---|---|---|
| `uptime` | recent boot (post-P4 reboot) | `1 day, 20:02` | Stale |
| `who -b` | 2026-05-22 ~17:00+ | `2026-05-20 22:53` | No reboot since 5/20 |
| `last reboot \| head -2` | second-newest = 5/22 | second-newest = `2026-04-24 16:13` | No reboot between P4 and now |
| `nvidia-smi` | clean device listing | `Failed to initialize NVML: Driver/library version mismatch / NVML library version: 580.159` | Same as P4 prep blocker |
| `torch.cuda.is_available()` (roarm env) | `True` | `False` (Error 804) | GPU inference unavailable |
| Kernel module | matches userspace | kernel `580.126.09` vs userspace `580.159.03` | Mismatch persists |

### openvla-7b cache — partial check

| Check | Result |
|---|---|
| `du -sh ~/.cache/huggingface/hub/models--openvla--openvla-7b` | `14G` (expected total ~14 GB) |
| Pinned revision dir present | `snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/` exists |
| `model.safetensors.index.json` symlink | present |
| `model-00003-of-00003.safetensors` symlink | present |
| `model-00001-of-00003.safetensors` symlink | **NOT VISIBLE** in `ls` |
| `model-00002-of-00003.safetensors` symlink | **NOT VISIBLE** in `ls` |
| Total blobs in `blobs/` | 17 files, 14 GB total |

Interpretation: 14 GB on disk strongly suggests download is **byte-complete** in
blobs, but the 3-shard symlink set inside `snapshots/<revision>/` is currently
incomplete. Most likely cause is in-progress symlink finalization by an earlier
`snapshot_download` call that was interrupted, or `.incomplete` rename races.
The next session must verify all 3 safetensors symlinks exist (or re-run
`snapshot_download` to make idempotent fixups) before instantiating the model.

### Other state — unchanged from P4

- `deploy_openvla_oft.py` 561 lines, syntax PASS (no edit this session).
- `roarm` env: `peft 0.18.0`, `rich 15.0.0`, `timm 0.9.16`, prismatic editable,
  torch 2.7.1+cu126 (no edit this session).
- Track A region of `START_HERE.md` (lines 11-30 truth + lines 79-148 Track A
  B200 evidence + lines 194-233 Track A continuation prompt): not modified.

## What this session DID not do

- Did not run `sudo reboot`. Out of scope: I cannot autonomously execute sudo
  on the user's host.
- Did not run `sudo modprobe -r/modprobe nvidia*` userspace fix. P4 session
  noted lower success rate; user picked clean reboot path.
- Did not run any `python deploy_openvla_oft.py ...` (Step 1+ blocked on
  Step 0 CUDA verification fail).
- Did not touch any Track A file (parallel session).
- Did not modify `deploy_openvla_oft.py`, `openvla_oft_roarm/*`, or any env.
- Did not append to MEMORY HARD RULES (no new failure pattern; this is the
  same NVML mismatch P4 already documented as Blocker (a)).
- Did not append to DECISIONS (no durable lesson — reboot omission is a
  one-off operational miss, not a new rule).

## Why this is logged as its own session

A common-sense alternative would be to amend the P4 session doc with a
"verification attempt 2 — still blocked" footnote rather than create a new
file. Two reasons for a separate file:

1. **Append-only convention** — `claudedocs/session_*.md` files are
   append-only per `CLAUDE.md` Current-State Protocol. Modifying P4 to add a
   post-hoc verification would either violate append-only or require an
   "addendum" block; either way, less discoverable than a stand-alone session
   doc with its own timestamp.
2. **Future-proofing the Track B chain** — when P5 (real deploy after reboot)
   eventually lands, its "previous step" reference is naturally
   `session_20260522_track_b_p4_5_reboot_blocked.md`, which encodes both the
   blocker and the unblocking action (reboot). This is more honest than
   pretending P4 → P5 is direct.

## Decisions / Next steps

Decision: end session, defer real deploy to a fresh session after the user
reboots.

User reboots from their terminal (not me):

```
sudo reboot
```

After ~1 min boot + login, open a new Claude Code session and paste the
continuation prompt below.

## Continuation prompt for next session (post-reboot)

```
Read CLAUDE.md first, then follow Current-State Protocol exactly.

한국어로 브리핑. 비판적/분석적. 기억 단독 truth 금지.
HANDOFF.md / TASKS.md 사용 금지. /half-clone / /handoff 절대 사용 금지 (HARD RULE #11).

본 세션 = Track B P5 real deploy (reboot 직후 첫 세션).

Step 0 (CUDA 정상화 검증 — premise 검증):
- `uptime`, `who -b`로 reboot 시각이 직전 ~수분-수십분 임을 확인.
- `nvidia-smi` mismatch 없이 device list 출력.
- `conda run -n roarm python -c "import torch; print(torch.cuda.is_available(),
  torch.cuda.get_device_name(0))"` → True + RTX 4090.

Must read:
1. CLAUDE.md
2. START_HERE.md (Track B P4 + P4.5 sections; Track A는 별도 세션 truth)
3. claudedocs/DECISIONS.md D071-D086
4. claudedocs/EXPERIMENT_LEDGER.md rows 111-119
5. claudedocs/session_20260522_openvla_oft_offline_eval_v6_result.md (P3 best=7500)
6. claudedocs/session_20260522_track_b_p4_deploy_prep_offline_hw_sanity.md (P4 prep)
7. claudedocs/session_20260522_track_b_p4_5_reboot_blocked.md (P4.5, 이 세션)
8. deploy_openvla_oft.py
9. ~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/memory/tech_leader_follower_setup.md

Step 1 (HF cache 완전성 검증):
- `du -sh ~/.cache/huggingface/hub/models--openvla--openvla-7b` ≈ 14 GB.
- `ls ~/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f/ | grep safetensors`
  → 4줄 (3 shard + 1 index). 4줄 미만이면:
  conda activate roarm && python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='openvla/openvla-7b', revision='47a0ec7fc4ec123775a391911046cf33cf9ed83f', allow_patterns=['*.json','*.txt','*.md','*.model','*.bin','*.safetensors','*.py']))"

Step 2 (1-chunk GPU dry-run):
conda activate roarm && python deploy_openvla_oft.py --dry-run --max-steps 8 --no-kinect --device cuda
- 검증: chunk shape (8,6), 6 joints all in JOINT_LIMITS, 추론 latency 측정,
  log하단 success.

Step 3 (Kinect dry-run, real frame):
python deploy_openvla_oft.py --dry-run --max-steps 8 --device cuda

Step 4 (real deploy, 사용자 명시 승인 후만):
python deploy_openvla_oft.py --max-steps 80 --log-csv --save-frames-dir logs/deploy_oft_$(date +%Y%m%d_%H%M%S)_frames
- default --port /dev/ttyUSB1 (Follower). /dev/ttyUSB0 (Leader) 사용 금지.
- default --start-pos init = [0,0,90,0,0,5] HOME.
- default --n-action-steps 8 chunk-mode (~2Hz).

Step 5 (multi-position):
sponge 위치 3-5번 변경하며 반복. SmolVLA v6 2026-04-09 Plan 3 SUCCESS (multi-position)
와 동일 protocol. 결과 비교 표 작성.

Step 6 (end-of-session):
- START_HERE Track B P5 section append (P4.5 section은 P5 결과 안에서 link).
- EXPERIMENT_LEDGER row 120 append.
- 새 claudedocs/session_20260522_track_b_p5_real_deploy_<verdict>.md.
- DECISIONS append if durable lesson.
- MEMORY recent sessions prepend (5-slot HARD RULE #8 — 12+ entries, archive
  권장).

Hard rules in effect:
- HARD RULE #1 HOME 시작
- HARD RULE #4 거짓 갭/최초 주장 금지
- HARD RULE #5 JOINT_LIMITS 절대 제거 금지
- HARD RULE #11 /half-clone /handoff 절대 사용 금지
- HARD RULE #13 Follower=/dev/ttyUSB1, Leader=/dev/ttyUSB0 (Leader 명령 절대 금지)
- HARD RULE #18 사용자 명시 정정 절대 우선

Do not touch:
- Track A files (parallel session active). START_HERE Track A regions.
- /dev/ttyUSB0 (Leader).
- 새 학습/PPO/dataset generation/Isaac runtime — Track B P5는 deploy 전용.
```

## HARD RULE compliance

- ✅ #1 No data collection / training. No HOME/start-pos change.
- ✅ #4 No "최초"/"없다" claims. Premise-failure brief is empirical (uptime,
  who -b, nvidia-smi, torch.cuda).
- ✅ #5 JOINT_LIMITS unchanged.
- ✅ #11 `/half-clone` / `/handoff` not used. Session-end continuation handled
  via project state files (this doc + START_HERE + ledger + MEMORY index).
- N/A #13 No USB command issued (Leader or Follower).
- N/A #14/#15/#26 No B200 work. No Isaac. No RL.
- ✅ #18 User explicit answer ("Reboot 후 새 세션 (권장)") honored. No
  autonomous sudo. No autonomous deviation.

## Track A boundary

This session did not touch any Track A file or directory. Track A v5/v6
projected-guard close_26 FAIL state (parallel-session-owned, START_HERE lines
11-30 truth + 79-148 evidence + 194-233 continuation prompt, DECISIONS D082-D085,
ledger rows 113-117) remains the latest verified Track A truth.
