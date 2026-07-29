# D403 [d402_host_boundary_git_repin_rerun] — 정적 준비 완료, runtime 직전 중단 (context 비상)

Date: 2026-07-28 저녁 KST. 이번 case의 신규 변수:
`[git_baseline_repin_to_a69a96d36219268e4bc5e25065cc234da9d99674,
gpu_visible_host_boundary_execution_v1]` (정확히 2개).

## 1. 무엇을 왜

사용자가 GPU 근본원인 정정 확인 후 `a69a96d`("gpu상태 확인 완료")를
master에 push하고 **순차 고속 실행을 지시** (per-step tuple-SHA 인용 절차
생략 승인, one-controller/one-worker/retry0 유지). D403 = 동결된 D402 계약
(D400 gripper_link A64→SDF res256 load/cook/readback + D402의 harness 수리
2건)을 새 Git baseline에서 **호스트 경계**로 재실행하는 forward-only wrapper
case. 과학/geometry 변경 0.

## 2. 이번 세션에서 완료된 것

1. **Preregistration** 작성:
   `claudedocs/runtime_logs/grasp_track/g0a_d403/attempt1_d402_host_boundary_git_repin_rerun/d403_preregistration.json`
   SHA-256 `fd403c6633ddd9f0f01615c4da35463547ade2319f0860af5b2db5cfe7e919f0`.
   상속 계약 13개 파일 + NVIDIA 설치 소스 5개 + D401/D402 attempt1 증거
   13개 해시 전부 이번 세션에서 재계산·일치 확인.
2. **Thin wrappers** 작성 (동결 체인 D403→D402(af1940a5)→D401(2807353b)→D400
   에 경로/provenance 재바인딩 + pre-delegation host-boundary gate만 추가):
   - `sim_scripts/cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_controller.py`
   - `sim_scripts/cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_worker.py`
   Gate: `/dev/nvidiactl|nvidia0|nvidia-uvm` 존재 + pid>10, 실패 시 어떤
   forward-only 쓰기도 전에 raise (attempt 소모 없음).
3. **정적 검사 실제 실행 22/22 PASS** (scratchpad runner, 휘발 — 결과 전사):
   - AST parse 2/2, 금지 top-level import 0/2, `__pycache__` 생성 0
   - 해시 pin 4종 exact (D402 ctrl/worker, prereg×2, ctrl-worker pin 일치)
   - `-B` 없이 실행 → controller/worker 모두 거부 (fixture 2 reject PASS)
   - host_boundary_gate on host → accept PASS
   - D400 counter 상수 AST 추출(36키=14 exact+21 zero+1 pump) 후
     counter gate fixture 11종 (accept 3 / reject 8) 전부 PASS
   - Item accessor fixture 8종 (accept 2 / reject 6) 전부 PASS
4. **적대적 리뷰 3-lens 워크플로우 발사** (위임 메커니즘/계약 스키마/worker
   체인). 결과 미회수 — journal:
   `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/cfceb787-c5e1-46ac-bc3e-1a26fc41bc07/subagents/workflows/wf_8c57c767-be5/journal.jsonl`

## 3. 아직 안 한 것 (다음 세션 순서)

1. 위 journal에서 리뷰 3건 회수 → blocker 있으면 스크립트/prereg 수리
   (수리 시 prereg sha 변경 → wrapper 내 `EXPECTED_PREREG_SHA256` 2곳 갱신).
2. `d403_reviewed_script_attestation.json` 작성 — **D402 attestation을
   템플릿으로** (`g0a_d402/attempt1_*/d402_reviewed_script_attestation.json`).
   D400 controller `_validate_approval_tuple`(라인 462-595)이 요구:
   5개 static true 필드 / `negative_static_fixture_results` ≥30개 전원
   expected=reject·observed=rejected·pass=true·id는
   `REGISTERED_STATIC_NEGATIVE_IDS` 18종(D400 controller 102-121행) superset /
   `static_stage_zero_counters` 10키 정확히 0 /
   controller·worker path+sha 바인딩 / prereg sha 바인딩.
   정직성: 18종 science fixture는 전 체인 해시 동일성 재확인 기반 재승인,
   D402-layer 14종은 본 세션 재실행 결과 (본 문서 §2.3) — 근거 분리 명기.
3. `d403_proposed_runtime_hash_tuple.json` 작성 (4필드 정확한 순서:
   preregistration/attestation/controller/worker sha) → tuple 파일 sha 계산.
4. **호스트 경계 실행 1회** (retry 0):
   `cd RoArm_Project && /home/cgxr/miniconda3/envs/isaaclab/bin/python -B
   sim_scripts/cyl34_top_view_d403_d402_host_boundary_git_repin_rerun_controller.py
   --approved-tuple-sha256 <tuple sha>` (background 실행 권장, watchdog
   300s+300s). 실행 전 `nvidia-smi` 정상 + 잔존 Isaac 프로세스 0 확인.
5. technical PASS 시 controller가 `D400_MANUAL_INSPECTION_REQUIRED` JSON을
   출력하고 300s 대기 → `d400_rerun_viewer_1920x1080.png`을 **실제로 Read로
   열람** 후 요구 스키마 그대로
   `d400_manual_visual_inspection.json` 작성 (subjects_visible 5종은 실제
   보이는 것만 true — D400 controller 3052-3149행이 스키마 검증).
6. 결과 보고(한국어 step-by-step) + 상태 문서 갱신. PASS 시 다음 case 설계:
   **SDF collider 물리 A/B** (D362 전도 재측정, 34×90 동결 계약 유지) —
   사용자 "연구 목표까지 순차 고속" 지시의 다음 rung.

## 4. 주의

- D402의 세 인프라 문서는 오진 — 정정:
  `session_20260728_grasp_g0a_d402_sandbox_misdiagnosis_root_cause_correction.md`.
- **모든 Isaac/GPU 명령은 호스트 경계에서** (샌드박스 Bash 금지, D402-R1).
- prereg `allowed_dirty_paths`는 상태문서 3종 + 세션문서 2종 + D403 파일
  5종만 허용 — 다른 파일을 dirty로 만들면 runtime freeze gate FAIL.
- commit/push 미승인. 스크립트 실행 자체는 사용자 지시로 승인됨.
- `scientific_or_physics_verdict=null`, `g0a_pass=false` (변동 없음).

## 5. (추가) 3-lens 적대적 리뷰 결과 — 세션 종료 후 도착

**blockers 0 / 0 / 0, 3인 전원 PASS** (위임 메커니즘 / 계약 스키마 / worker 체인).
counter 게이트 정적 재검증: baseline accept + offline fixture 7종 reject 확인 →
`_offline_negative_controls`에서 false fail-stop 없음 예상. 전체 결과:
wf_8c57c767-be5 journal (§2.4 경로).

**실행 전 반드시 지킬 경고 (리뷰어 지적):**
1. attestation/tuple 미작성 상태 (알려진 잔여 작업) — 없이 실행하면
   `_validate_approval_tuple`(preflight.py:467-470)에서 쓰기 0으로 fail-stop
   (attempt 미소모).
2. **prereg를 지금부터 절대 수정 금지** (wrapper 2곳이 해시 pin) / tuple 작성
   후에는 controller·worker도 수정 금지 (tuple이 최종 해시를 담아야 함).
3. **tuple 생성~실행 사이 commit/push 금지** — HEAD 변경 시 `head_exact`
   (d401 controller:350-355) + `head_still_exact`(preflight.py:1748-1753) FAIL.
4. 실행 직전, D403 스크립트 파일명 또는 OUT_DIR 문자열을 cmdline에 가진
   비조상 프로세스(에디터/tail/grep) 종료 — conflict scan(preflight.py:837-848)
   이 fail-stop시킴.
5. isaaclab conda env 건드리지 말 것 — installed NVIDIA 5+7개 해시 pin이
   manifest gate에서 재검증됨 (현재 전부 exact 확인됨).
6. `/dev/nvidia-uvm`는 lazy 생성 노드 — 갓 부팅한 호스트에선 CUDA 앱 1회
   전까지 없을 수 있음 (게이트가 쓰기 전에 멈추므로 비용 0, 현재는 존재).
7. 산출물 내부 id는 동결 상속상 `g0a_d400` 라벨을 가짐 (예: raw summary
   `case: g0a_d400`, Rerun id `roarm_g0a_d400_sdf_preflight`) — 양쪽 동결
   코드가 일관 매칭하므로 실행엔 무해, provenance 독해 시 참고.
8. prereg `worktree_at_case_boot='clean'`은 case boot 시점 기술로 유지
   (현재 dirty 8경로는 전부 allowed 목록 내 — 게이트 무관).
