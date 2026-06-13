# Session 2026-06-13 - Cube10cm Top-View Professor Packet + Storage Root + RunPod/H100 AV1 Gate D234

## Scope

User approved proceeding with:

- professor view/format confirmation packet;
- storage/output-root decision;
- RunPod/H100 AV1 LeRobot dataloader decode/speed verification.

This session did not run Isaac Sim render, generate a 100 episode chunk, generate
1000/10000 episodes, delete/archive/move local files, train PPO/L2/Large PPO,
start VLA/action-teacher work, deploy to RoArm, use SSH JHPark/B200, pull from
B200, copy `.ssh`, or mix with Track A.

## Verified Starting Truth

- `START_HERE.md` D233 says the active branch is professor 10cm / 0.72kg cube
  top-view visual trajectory dataset camera-contract, not Track A or PPO.
- D233 local smoke passed: 5 episodes, 975 frames, `1280x720`, reprojection
  median/max `3.074639061891291px` / `9.956731449704932px`, all-frame visibility
  `975/975` full, contact-window visibility `882/882` full.
- D233 local LeRobot passed with `observation.images.top`, AV1, `yuv420p`,
  `30fps`, 975 frame match, sampled decode avg/max
  `0.016793251037597656s` / `0.06672263145446777s`.
- D233 blocked 100 episode chunk until professor confirmation,
  storage/output-root decision, RunPod/H100 codec gate if applicable, and
  explicit approval.
- Local disk after D233 remained tight: about `590G` total, `529G` used, `32G`
  free, `95%`.

## Professor View/Format Packet

Created:

`claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`

Created visual packet artifact:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/professor_review_contact_sheet_d234.png`

The contact sheet is a `1326x1442` RGB PNG with 5 smoke episodes and frame
samples `0`, `97`, and `194` per episode.

Existing PNG extraction proof remains:

`claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/extract_ep003_frame050.png`

The extraction proof is a `1280x720` RGB PNG.

Professor-facing framing:

> 프레임별 이미지-상태 페어 구조는 그대로 유지하되, 저장은 업계 표준인
> LeRobot 포맷(mp4+parquet)으로 하겠습니다. 임의 frame을 PNG로 즉시 추출
> 가능합니다.

Important status:

- User approved proceeding with the view/format gate.
- Direct professor response is not present in repo docs yet, so this session
  records the packet as prepared and ready, not professor-signed.

## Storage / Output-Root Decision

Primary storage remains LeRobot MP4+parquet:

- video key: `observation.images.top`;
- raw image shape: `[720, 1280, 3]`;
- per-frame state/action metadata in parquet;
- PNG only for smoke/debug/extraction.

Codec decision:

- AV1 is selected for the next chunk by current evidence.
- H.264 remains the fallback only if a future tool/environment cannot decode AV1
  or if speed becomes unacceptable.

Output-root decision:

- Next 100 episode chunk, if explicitly launched later, should use a fresh local
  root:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_chunk100_d235`.
- This local root is acceptable only for one 100 episode chunk because D233
  measured retained debug PNG at about `52.3344778MB/episode`, projecting to
  about `5.2334477800000005GB` for 100 episodes, plus about
  `0.059648780000000005GB` for AV1 video.
- No cleanup is required for that one 100 episode chunk if pre-run free space is
  still at least about `25GB`.
- Stop before running if free space has dropped below that threshold.
- 1000/10000 episodes remain blocked on local disk. They require external/RunPod
  storage, a no-full-PNG-retention pipeline, or an explicitly approved cleanup/
  archive action.

No deletion/archive/move was performed.

## RunPod/H100 AV1 Dataloader Gate

Created RunPod pod:

- pod id: `86qyuxeldab9h4`;
- name: `roarm-cube10cm-av1-dataloader-d234`;
- image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`;
- GPU: `NVIDIA H100 80GB HBM3`;
- cloud: secure;
- volume: `20GB` at `/workspace`;
- container disk: `30GB`;
- reported cost rate: `$3.29/hr`.

Runtime timeline from RunPod response:

- created: `2026-06-12 16:58:49 UTC` (`2026-06-13 01:58:49 KST`);
- stopped: `2026-06-12 17:16:07 UTC` (`2026-06-13 02:16:07 KST`);
- approximate active duration: 17 minutes.

The pod was stopped after results were copied back. It was not deleted.

Cost rule for future RunPod use:

- Always stop RunPod pods immediately after the approved work finishes.
- Stopping releases GPU compute, but stopped pods can still accrue persistent
  volume-storage charges.
- If all outputs are copied back and the remote environment is not needed,
  delete/terminate the pod to avoid ongoing storage charges.
- If keeping the pod for reproducibility or quick resume, document that choice
  and the remaining storage-cost exposure.

RunPod environment:

- system Python: `3.12.3`;
- base torch: `2.8.0+cu128`;
- GPU check: `NVIDIA H100 80GB HBM3`, `81559 MiB`, used `0 MiB` before test;
- `/workspace`: `20G` total, initially about `20G` free.

Installed an isolated venv under `/workspace/roarm_d234/venv` because system pip
was blocked by PEP 668:

- `lerobot 0.4.4`;
- `torch 2.10.0`;
- `torchcodec 0.10.0`;
- `av 15.1.0`;
- `imageio-ffmpeg 0.6.0`.

Uploaded only:

- the 3.0M smoke LeRobot dataset tar;
- `sim_scripts/cube10cm_lerobot_codec_preflight.py`.

No render, training, or dataset generation ran on RunPod.

## RunPod/H100 Decode Results

Local copies:

- 50-sample result:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_d234.json`
- full-frame result:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_full_d234.json`

50-sample gate:

- status: `PASS`;
- codec: `av1`;
- pix_fmt: `yuv420p`;
- fps: `30`;
- shape: `[720, 1280, 3]`;
- frames: `975`;
- episodes: `5`;
- samples: `50`;
- decoded image shape: `[3, 720, 1280]`;
- decoded image dtype: `torch.float32`;
- state/action shape: `[6]` / `[6]`;
- avg decode: `0.032439069747924806s`;
- max decode: `0.11921572685241699s`;
- elapsed: `4.6211488246917725s`.

Full 975-frame gate:

- status: `PASS`;
- codec: `av1`;
- pix_fmt: `yuv420p`;
- fps: `30`;
- shape: `[720, 1280, 3]`;
- frames: `975`;
- episodes: `5`;
- samples: `975`;
- decoded image shape: `[3, 720, 1280]`;
- decoded image dtype: `torch.float32`;
- state/action shape: `[6]` / `[6]`;
- avg decode: `0.017871856689453125s`;
- max decode: `0.10865616798400879s`;
- elapsed: `26.368597984313965s`.

Interpretation:

- RunPod/H100 LeRobot dataloader can decode the smoke AV1 dataset.
- AV1 remains the selected codec for the next 100 episode chunk.
- OpenCV AV1 compatibility is irrelevant to training as long as LeRobot
  dataloader decode remains the training path.

## Decision

Verdict:

`PROFESSOR_PACKET_READY_STORAGE_ROOT_DECIDED_RUNPOD_H100_AV1_DATALOADER_PASS_100EP_NOT_RUN`

Closed gates:

- professor-facing view/format packet prepared;
- PNG extraction proof available;
- storage/output-root policy selected for the next 100 episode chunk;
- RunPod/H100 AV1 LeRobot dataloader decode/speed gate passed;
- RunPod compute pod stopped after result retrieval.
- RunPod cost rule documented: stop immediately; delete/terminate when remote
  volume/environment retention is unnecessary.

Still blocked:

- Direct professor response is not recorded yet.
- 100 episode chunk was not run in this session.
- 100 episode chunk still requires a fresh explicit run instruction if the user
  wants to launch it.
- 1000/10000 episode generation remains blocked.
- Any deletion/archive/move remains blocked without explicit approval.
- PPO/L2/Large PPO, VLA/action-teacher, RoArm deployment, SSH JHPark/B200, pull,
  `.ssh` copy, and Track A remain out of scope.

## Sources

- `START_HERE.md`
- `claudedocs/DECISIONS.md#d233`
- `claudedocs/session_20260612_cube10cm_top_view_visual_smoke_lerobot_d233.md`
- `claudedocs/professor_view_format_packet_cube10cm_top_view_d234.md`
- `claudedocs/storage_plan_cube10cm_visual_dataset_d232.md`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_d234.json`
- `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_full_d234.json`
