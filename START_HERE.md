# START_HERE.md

Last updated: 2026-08-13 KST — 56th Claude 재인수 부트 검증 완료 (55th 핀 전부 재확인,
불일치 0, 실험 0, 사용자 A/B/C 결정 대기).

## Active Case — single source of truth

- Case: `g0b_d420`, fixed-base RoArm-M3, D29×H50 mm / 24.83 g cylinder.
- 이번 case의 신규 변수: `[파지점=side midpoint, SDG candidate pose]`.
- Active run folder: `claudedocs/runtime_logs/grasp_track/g0b_d420/`.
- Detailed current session:
  `claudedocs/session_20260813_55th_g0b_t3u_side_midpoint_p13_runpod_render.md`.
- 56th boot re-verification (Codex→Claude handoff, 실험 0, 불일치 0, RunPod Pod 0 실측):
  `claudedocs/session_20260813_56th_g0b_boot_reverify_claude_handoff.md`.
- Robot hardware 0. P13 local Isaac physics completed once. Cloud physics rerun 0.
- Current scientific direction is not advanced beyond this failed-grasp observation.
  Any change to target/depth/trajectory/gripper/candidate requires a new approved case.

## Current verified truth

### P13 physics/grasp

- Frozen P13 completed 2 diagnostic + 2,340 task callbacks; task physics wall 74.140 s.
- Result semantics: 30/30 PASS; numeric and measurement validity: 5/5 true.
- Grasp success: **0/5**.
- Population verdict: **`NO_BILATERAL_SIDE_CONTACT`**.
- Rows: `c05_o00=premature_jaw_contact`; `c05_o01..o04=no_bilateral_close`.
- Arrival error: 1.525311–1.525837 mm, all five PASS.
- Close fixed/moving/bilateral force: 0/0/0 N for every row.
- Corrected object lift: −0.0002366 to +0.0007916 mm; required gate is >6 mm.
- TCP rose 24.0486–24.1584 mm while the cylinder stayed on the support.
- Authority: `t3u_side_preflight13_results.json`, SHA
  `8324ed7a9682ccb297985dd733c9e91c480bed9ce65bb02672d5b40226eea6d5`.
- Trace: `t3u_side_preflight13_trace.npz`, SHA
  `ee67d3516a1c7871e5f48d455b420c3f5985ae889bceb097536904548e8134ee`.
- This is valid preflight failure evidence, not a canonical scientific promotion.

### Visual evidence

- Local native RTX original and bounded render1 repair both reached capture but emitted
  zero frames and timed out. They are consumed prefixes and must not be retried.
- Meeting fallback exists and is verified:
  `t3u_side_meeting1_trace_video.mp4`, H.264/yuv420p 1280×720, 20 fps,
  234 frames / 11.7 s, full decode PASS, SHA
  `14a9b6d9ef6dee9fae0210c7f7eda524692548d3d62e3a3608972f10b51f8414`.
- The meeting MP4 is an exact-trace CPU schematic, not an Isaac RTX camera view and not
  independent scientific authority. It must be shown as failed-grasp diagnostics only.
- Current delivery bundle: `t3u_side_meeting1_lab_bundle_v3.zip`, 4,172,534 bytes,
  9/9 members verified, SHA
  `2bcdc926e60b1848026cf4c6bcd62610e04f1047bedabe355f92cb953ce67ac1`.

### RunPod RTX PRO 6000 A/B

- Prior A100 Pod was deleted. Secure RTX PRO 6000 Blackwell Server Edition 96 GB Pod
  `aoyagwoz7blwiv` was created in US-NE-1 at $2.09/h and later permanently deleted.
- Driver 580.126.16, `cuInit=0`, PyTorch CUDA, Isaac Lab 2.3.0, Isaac Sim 5.1.0.0,
  `numpy==1.26.0`, and `psutil==5.9.8` all passed.
- Exact input recovery required the original absolute repo path and nanosecond mtimes.
  After restoration the P13 30/30 semantic and 245/245 dependency gates passed.
- Exact render1 still reached capture and produced zero PNGs before its 90 s no-progress
  deadline.
- Root infrastructure boundary: the Pod exposed `/dev/nvidia3`, `/dev/nvidiactl`, and
  `/dev/nvidia-uvm`, but not `/dev/nvidia-modeset`; `vulkaninfo` could not create an
  instance even after matching userspace GL libraries were supplied. CUDA compute was
  available, but the selected Pod runtime was not a usable Vulkan/RTX graphics runtime.
- Raw cloud evidence: `t3u_side_rendercloud1_runpod_evidence.tar.gz`, 59 entries,
  73,679 bytes, SHA
  `5469c2fcf6eaed522c6a670ebc1731e8b0da360b15972237cf288d80e91e0610`.
- RunPod account was rechecked after deletion: active Pods = 0.

## Active pivot and reserve pivots

- Active pivot: present the truthful 0/5 failure and identify why bilateral close never
  occurred. Do not tune the grasp within `g0b_d420`.
- Immediate non-scientific attachment: use the verified CPU trace MP4/contact sheet.
- Reserve infrastructure pivot: a future cloud RTX replay must use an official/prepared
  Isaac Sim 5.1 graphics runtime and pass `/dev/nvidia-modeset` + `vulkaninfo` + one-frame
  smoke gates before environment installation or project transfer.
- Reserve science pivot: a new approved case may change one or two variables such as
  side-depth/trajectory or gripper geometry; record it in `claudedocs/BACKLOG.md` first.

## Open risks / do-not-repeat

- RT-capable GPU model does not prove the container exposes a Vulkan/RTX graphics device.
  CUDA PASS is insufficient. See DECISIONS D443.
- Do not use A100/H100 for Isaac RTX rendering; they lack RT cores.
- Do not retry the deleted RunPod setup or the two consumed local render prefixes.
- Do not claim successful grasp, bilateral close, or >6 mm lift from lift-phase forces.
- Do not use `decision_snapshot.png` separation as TCP error; target and actual markers use
  different origins there.
- Do not call the CPU meeting MP4 an RTX viewport render.
- Preserve `numpy==1.26.0` and `psutil==5.9.8` in every Isaac Lab environment.
- Do not edit old output prefixes; all future folders/tags are forward-only.
- Single-copy risk (56th 관측): `t3u_side_preflight13_trace.npz`는 `.gitignore`(`*.npz`)
  대상이고 bundle v3에도 없음 — 이 디스크 단일 사본. `*.mp4/*.png/*.log/*.csv`도 g0b
  whitelist 없음(커밋해도 repo 밖).

## Next concrete action / authorization boundary

1. For the lab meeting, deliver `t3u_side_meeting1_lab_bundle_v3.zip` and state 0/5.
2. If the user approves a new grasp case, preregister only one or two changed variables and
   run a failure-capable physics perturbation; do not silently tune the current case.
3. If the user approves another RTX cloud attempt, first provision a graphics-ready Isaac
   container and run only the device/Vulkan/one-frame smoke gate. Transfer the 300 MiB
   dependency payload only after that gate passes.

## Must read first

1. `AGENTS.md`
2. `claudedocs/session_20260813_55th_g0b_t3u_side_midpoint_p13_runpod_render.md`
3. `claudedocs/DECISIONS.md` tail, especially D441–D443
4. `claudedocs/EXPERIMENT_LEDGER.md` latest row
5. `claudedocs/runtime_logs/grasp_track/g0b_d420/t3u_side_meeting1_brief.md`

## Do not trust as current

- `HANDOFF.md`, `TASKS.md`.
- Earlier `START_HERE.md` snapshots embedded in chat/session history.
- `t3u_side_meeting1_lab_bundle.zip` and `_v2.zip`; `_v3.zip` is current.
- Any statement that P13 produced an RTX video or a successful grasp.
