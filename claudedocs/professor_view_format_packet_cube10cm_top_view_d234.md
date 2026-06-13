# Professor View/Format Packet - Cube10cm Top-View Visual Dataset D234

Status: prepared for professor-facing confirmation after D233 local smoke. This
file records the packet and wording to use; it does not claim an independent
professor response unless that response is later added to repo docs.

## One-Line Confirmation

Korean:

> 프레임별 이미지-상태 페어 구조는 그대로 유지하되, 저장은 업계 표준인
> LeRobot 포맷(mp4+parquet)으로 하겠습니다. 임의 frame을 PNG로 즉시 추출
> 가능합니다.

English:

> We will keep the frame-by-frame image-state pair structure, but store it in the
> industry-standard LeRobot format, MP4 plus parquet. Any arbitrary frame can be
> extracted immediately as PNG when needed.

## View Assets

Use these local artifacts for the view/format check:

- contact-sheet overview:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/professor_review_contact_sheet_d234.png`
- arbitrary-frame PNG extraction proof:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/extract_ep003_frame050.png`
- render summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/render_summary.json`
- local LeRobot validation summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/cube10cm_top_view_visual_smoke_d232/lerobot_validation_summary.json`
- RunPod/H100 LeRobot AV1 decode summary:
  `claudedocs/runtime_logs/20260526_cube3cm_push_rollout_probe_20480/runpod_d234/cube10cm_runpod_h100_av1_decode_preflight_full_d234.json`

## Minimal Ask

Ask only these points:

1. Is this top-view framing acceptable for the 10cm / 0.72kg cube tap/push visual
   trajectory dataset?
2. Is LeRobot MP4+parquet acceptable as the primary storage format if arbitrary
   frames can be extracted immediately as PNG?
3. Is PNG needed for any concrete reason beyond visual inspection or external
   tool compatibility?

## Current Evidence

- Local smoke rendered 5 episodes / 975 frames at `1280x720`.
- Camera contract id: `cube10cm_top_view_v1_candidate`.
- Reprojection centroid median/max: `3.074639061891291px` /
  `9.956731449704932px`.
- All-frame visibility: `975/975` full, `0` partial, `0` full occlusion.
- Contact-window visibility: `882/882` full, `0` partial, `0` full occlusion.
- Local LeRobot AV1 conversion/load/decode passed.
- RunPod/H100 LeRobot AV1 full-frame decode passed.
- Extracted PNG proof is `1280x720`.

## Format Policy

- Primary storage: LeRobot MP4+parquet.
- Video key: `observation.images.top`.
- Raw image shape: `[720, 1280, 3]`.
- Per-frame state/action shapes observed through LeRobot: `[6]` / `[6]`.
- PNG role: smoke/debug/extraction only, not full scale-up storage.
- Codec: AV1 is acceptable locally and on the RunPod/H100 gate. If a future
  external tool or dataloader cannot decode AV1, switch the new dataset codec to
  H.264 for that environment.

## Confirmation Status

- User approved proceeding with the view/format gate on 2026-06-13 KST.
- This packet is ready to present.
- Direct professor response is still not recorded in repo docs.
