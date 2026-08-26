# LEDGER_RECENT.md — 최근 실험 20건 요약 (부팅 read)

Last updated: 2026-08-25 — 생성 시점 원본 `EXPERIMENT_LEDGER.md` = 531줄 / 1,062,466 B / 표 행 509개
(줄당 평균 약 2 KB, 최근 행은 5~7 KB — **단일 Read로 열면 토큰 초과**).

## 0. 이 파일의 권위와 사용법

- **이 파일은 권위가 아니다.** 수치·판정의 정본은 `claudedocs/EXPERIMENT_LEDGER.md`의 해당 줄이고,
  그 원장조차 스스로 이렇게 경고한다: *"Do not use this as the only source for metrics;
  verify from the linked session/data files before making claims."* → **인용 전에 세션문서/데이터 파일까지 내려갈 것.**
- 존재 이유: 부트 절차 3단계의 원장 통째 read가 **물리적으로 불가능**하다(줄당 2 KB × 531줄).
  그래서 조용히 생략되고, 최근 실험이 뭐였는지 모르는 채 세션이 시작됐다.
- 쓰는 법: 여기서 대상 행을 고르고 → `EXPERIMENT_LEDGER.md:<줄>`만 `offset`/`limit`으로 on-demand read.
- **중복 금지 지도**:

  | 알고 싶은 것 | 읽을 곳 |
  |---|---|
  | 지금 뭘 하고 있나 / 다음 행동 | `START_HERE.md` (여기 아님) |
  | 규칙 원문 | `AGENTS.md` (자동 로드 — 여기 아님) |
  | 어떤 결정이 살아 있나 | `claudedocs/DECISIONS_ACTIVE.md` |
  | **최근에 뭘 돌렸고 판정이 뭐였나 + 그 앵커** | **이 파일** |
  | 실험 상세·수치·재현 절차 | `EXPERIMENT_LEDGER.md:<줄>`, `claudedocs/session_*.md` |

## 1. 선정 기준 (재현 가능 — 기억으로 판단하지 말 것)

원장 **마지막 20행 = `:512`~`:531`**, 정렬은 append 순(=시간순). 재확인 명령:
```bash
wc -l claudedocs/EXPERIMENT_LEDGER.md          # 531 이면 아래 앵커 그대로 유효
sed -n '512,531p' claudedocs/EXPERIMENT_LEDGER.md | awk -F'|' '{print NR+511": "substr($2,1,120)}'
```
⚠️ 원장은 append-only라 **줄이 늘면 앵커가 전부 밀린다.** 원장 줄수가 531이 아니면 이 파일부터 갱신할 것.

## 2. 🔴 원장 무결성 결함 4건 (요약하다 발견 — 원본은 손대지 않았다)

**① 등재 누락 3건** — 세션문서는 있는데 원장에 행이 없다. 원장에서 이 세션들은 **존재하지 않는다**:
```
56th  session_20260813_56th_g0b_boot_reverify_claude_handoff.md      원장 언급 0회
57th  session_20260813_57th_g0b_d444_flying_gripper_case_open.md     원장 언급 0회  (D444 개시 세션)
70th  session_20260817_70th_cold_archive_t1_t2_migration.md          원장 언급 0회  (콜드 아카이브 이관)
```
→ 원장 최종 등재 = **69th**, 세션문서 최신 = **70th**. 원장은 하루 뒤처져 있다.

**② 스키마 드리프트 — 하필 최근 3행.** 표 헤더는 6열(`Date/Label | Run/Path | Goal | Key Result | Verdict | Source`)인데
`:529`~`:531`(**67th·68th·69th = 현재 야드 피벗 전체**)은 **4열**이고, `Verdict` 칸에 판정 토큰 없이 `**D453**`/`**D454**`/`**D455**`만 있다.
앞선 17행은 `FG1_ALL_13_FAIL_...` 같은 기계 판독 토큰 + 비주장 한정어를 달고 있다.
→ **가장 최근이고 가장 필요한 3행이 가장 정보가 적다.** 판정 내용은 `DECISIONS.md`(D453~D455)로만 도달 가능.
(전체 분포: 6열 494행 / 4열 3행 / 나머지는 셀 안 `|` 때문에 필드 수 7·9~14)

**③ 표가 두 블록으로 쪼개져 있다.** 표 헤더는 `:7-8`에 한 번뿐인데 `:105`~`:531`(427행)이
헤더 없이 이어진다 → 마크다운에서 **두 번째 블록은 표로 렌더되지 않는다.**

**④ 원장 한복판에 죽은 상태 12줄.** `:92-103` `## Current Next Experiment Candidate`가
**"Active pivot (2026-05-21): Track A P7/Branch B ..."** 라고 단언한다 — 현재 피벗(포스코 야드, 63rd~)과 **정면 모순**.
AGENTS.md에서 걷어낸 결함 B(죽은 상태가 규칙/참조 파일에 상주)와 **같은 패턴**이다.
→ 부트 3단계가 이 파일 대신 `LEDGER_RECENT.md`를 읽게 되면서 **이 12줄은 더 이상 자동 주입되지 않는다.**
원본은 append-only 정책상 **삭제하지 않았다.**

## 3. 최근 20건 (신 → 구)

> 형식: `앵커 · 세션` — 무엇을 돌렸나 → **판정** · 근거. 세션문서는 전부 `claudedocs/` 아래.
> 4열 행(`:529`~`:531`)은 원장에 판정 토큰이 없어 `DECISIONS.md`를 근거로 표시했다.

### 현행 피벗 — 포스코 야드 (63rd~)

- **`:531` · 69th** (08-16) `y3_d455` 정책 비교층 v1 — 규칙 정책 8종 완주 에피소드 + a1 rep2 (물리 O ×9)
  → **D455** · `session_20260816_69th_y3_d455_policy_compare.md` · `runtime_logs/yard_track/y3_d455/`
- **`:530` · 68th** (08-16) `y2_d454` pick-place 전이 — yp1 spread · yp2 stack · yp1 rep2 (32-cycle 전량 이송 ×3, 물리 O ×3)
  → **D454** · `session_20260816_68th_y2_d454_pick_place_transfer.md` · `runtime_logs/yard_track/y2_d454/`
- **`:529` · 67th** (08-16) `y1_d453` 야드 테스트베드 v1 — 설계 p26 + 더미 정착/높이맵 probe yt1·yt3 + rep2 (물리 O ×3)
  → **D453** · `session_20260816_67th_y1_d453_testbed_pile_heightmap.md` · `runtime_logs/yard_track/y1_d453/`
- **`:528` · 66th** (08-16) `o1` O-step 물체 생성기 — **저작 전용**(물리 0, Isaac 0)
  → **`O1_ROCK_SET_52_AUTHORED`** (DECISIONS 신규 0) · `session_20260816_66th_o1_posco_rock_generator.md` · `sim_assets/posco_rocks_o1/`
  ※ **질량은 실측 기록 전 sim 주장 금지**(manifest 규약). 파일럿 이관은 `E:\posco-pilot` 미마운트로 blocked.
- **`:527` · 65th** (08-16) `g0f_d452` 조 슬리브 설계 + gs1 완전닫힘 13pose + gs2 폭-정지 창 56평가 (물리 O ×2)
  → **`GS2_SLEEVE_WIDTH_STOP_WINDOW_MEASURED`** (**D452**) · `session_20260816_65th_g0f_d452_gs1_gs2_sleeve_design_probes.md`
  ※ 실기 처방 = 슬리브 + **전류-제한 stall 닫힘**. 잔여: rim 0/5 · 29~14° 미측정.
- **`:526` · 64th** (08-16) `fg2` 폭-정지 닫힘 정책 40 평가 (물리 O)
  → **`FG2_WIDTH_STOP_SOME_HOLD_SW_POLICY_VIABLE_SIM`** (**D451**) · `session_20260816_64th_g0e_d451_fg2_width_stop_probe.md`
  ※ 성공 창 ~2° → 고정각 정책은 폭 지식·캘리브 오차에 취약. 비주장: 실로봇·마찰 현실성.
- **`:525` · 63rd** (08-16) 포스코 야드 pivot recon — **조사 전용**(물리 0). 실험 부재 사유 = 교수님 기각발 pivot 재설계
  → **`PIVOT_RECON_COMPLETE__GAP_NARROWED_TO_3_COMBO__GTSU_ANCHOR_CONFIRMED`** (**D450**) · `session_20260816_63rd_posco_yard_pivot_domain_recon.md`

### Frozen — grasp track (재실행 금지, 인용 전용)

- **`:524` · 62nd** (08-14) `ba2` B601 full-arm side pick→carry→place+release probe (물리 O, RTX 키프레임 9장)
  → **`BA2_TCP_TRACK_FAIL`** (**D449**) · `session_20260814_62nd_g0d_d449_ba2_full_arm_side_place_probe.md`
  ※ **place 실패가 아니라 다중 지지물 장면의 손목 후행 체적 클리어런스 설계 실패.** ba3 재시도는 사용자 승인 대기.
- **`:523` · 61st** (08-13) `ba1` B601 full-arm side 파지+리프트 + RTX mp4 (물리 O, RTX O)
  → **`BA1_FULL_ARM_SIDE_GRASP_LIFT_SUCCESS`** (**D448**) · `session_20260813_61st_g0d_d448_ba1_full_arm_side_grasp_mp4.md`
  ※ 벤더 자산 결함 2호(중첩 계층 미시뮬). 비주장: 실물 파지/게인 현실성.
- **`:522` · 60th** (08-13) `bg1v` 시각화 전용 상태-복원 렌더 (물리 0, 렌더만)
  → **`VIZ_ONLY_OK`** (판정 신규 0) · `session_20260813_60th_g0c_bg1v_b601_grasp_render_snapshots.md`
  ※ **"쥐고 유지" 증명은 렌더가 아니라 hang 수치 캡션이 한다.** 권위 = `bg1_results.json`.
- **`:521` · 59th** (08-13) `bg1` B601 flying-gripper 2변형 판별 (물리 O)
  → **`BG1_REAL_GEOM_HOLDS_USD_COLLISION_BLOCKS`** (**D446**) · `session_20260813_59th_g0c_d446_bg1_b601_flying_gripper_run.md`
  ※ 병목은 실기하가 아니라 벤더 USD 1-hull 근사. 비주장: 실물 파지. **B601 트랙은 이후 교수님 기각으로 종료.**
- **`:520` · 58th** (08-13) `fg1` flying-gripper 13 pose 물리 판별 (물리 O)
  → **`FG1_ALL_13_FAIL_GRIPPER_GEOMETRY_BOTTLENECK_SUPPORTED`** (**D445**) · `session_20260813_58th_g0b_d444_fg1_flying_gripper_run.md`
  ※ 병목 층위가 "접촉 불가"→"유지 불가"로 이동. ⚠️ 폭-정지 정책 미시험 = **"어떤 정책으로도 불가" 아님** (→ 64th가 해소).
- **`:519` · 55th** (08-13) `t3u` P13 side-midpoint physics + local/cloud render A/B
  → **`GRASP_FAIL_0_OF_5__CPU_MEETING_VIDEO_VALID_NONRTX__RUNPOD_COMPUTE_ONLY_VULKAN_UNAVAILABLE`** (**D443**) · `session_20260813_55th_g0b_t3u_side_midpoint_p13_runpod_render.md`
- **`:518` · 54th-b** (08-11) `t3y_workspace1` 광역 workspace 병렬 PhysX (Isaac O, GPU O)
  → **`BILATERAL_CONTACT_ONLY_DURING_LIFT_NO_VALID_GRASP`** (**D441**) · `session_20260811_54th_g0b_t3x_t3y_workspace_physics.md`
  ※ 확대 금지 목록 명시: θ0 · 측면 중점 · 연속 workspace · 하드웨어 · force closure.
- **`:517` · 54th-a** (08-11) `t3x_bite81` IK-conditioned finite-cylinder bite audit (CPU)
  → **`NO_BILATERAL_WINDOW_IN_SPAWN_ENVELOPE`** (**D441**) · 같은 세션문서
  ※ **정적 물림은 force closure/lift가 아니다.**
- **`:516` · 53rd** (08-11) 반경별 도달 경계 스윕 (사용자 승인 1-NEXT ⓐ)
  → **`REACH_CEILING_IS_POSE_SPECIFIC__BUT_THE_75DEG_BRANCH_IS_UNUSABLE`** (**D440**) · `session_20260811_53rd_g0b_t3w_reach_boundary_sweep.md`
  ※ ≥75° 가지 = 480셀 중 2셀 · 모든 스폰 영역 밖.
- **`:515` · 52nd** (08-11) `t3p` 랜덤화 병렬 **접촉력 계측** 물리 시행 (물리 O, Isaac O)
  → **`CONTACT_MEASURED_FIRST_TIME__BOTH_JAWS_LOAD__ZERO_LIFT_IN_1024__MECHANISM_IS_PRESS_INTO_TABLE_NOT_PINCH`** (**D439**) · `session_20260811_52nd_g0b_t3p_randomized_parallel_sweep.md`

### 감사·패널 (물리 재실행 0 — 문서 무결성 축)

- **`:514` · 51st-b** (08-11) 적대 패널 `wf_46941a6d-04e` 회수 13/13 (2,185,034 tok · 633 calls)
  → **`PANEL_CONFIRMS_D437R1_CORE__REFUTES_8_OF_51ST_OWN_REDERIVATIONS__DOCINT_SELF_INVALIDATED`** (**D438-R1**)
  ※ **⛔ 진행 승인 아님.** 12 패스에서 신규 물리 측정 0건. **세션문서 없음** — 근거는 `DECISIONS.md` D438-R1 + `g0b_d420/t3d_panel_wf46941a6d_*`.
- **`:513` · 51st** (08-11) 부트 재검증 + N-1 프록시 동결 로그 재도출 + 패널 발사·미회수
  → **`N1_PROXY_REPRODUCED__FRAMING_CORRECTED__PHASE_UNDISCRIMINABLE__REPLICATES_ZERO`** (**D438**) · `session_20260811_51st_g0b_boot_n1_proxy_rederivation.md`
- **`:512` · 50th-b** (08-10) 적대 패널 `wf_84c6e3b4-92a` 회수 13/13 (1,883,325 tok · 542 calls)
  → **D427~D437 헤드라인 전부 불변 · D433 `LIFT_FAIL`×3 전건 재현** (**D437-R1**) · `session_20260810_50th_g0b_boot_reverify_panel_launched.md`
  ※ **"무너진 것은 물리 결론이 아니라 그 결론을 받친다던 근거·범위·선례다."**

## 4. 원장 항해 인덱스 (통째 read 금지)

```
:1-5     머리말 (원장 자신의 경고 — "이것만으로 수치 인용하지 말 것")
:7-8     표 헤더 (6열, 파일 전체에서 여기 한 번뿐)
:9-90    표 블록 1 — 82행 (~2026-05-21)
:92-103  ⚠️ 죽은 상태 "Current Next Experiment Candidate" (2026-05-21)
:105-531 표 블록 2 — 427행 (헤더 없음 → 표로 렌더 안 됨)
:512-531 ← 이 파일이 요약한 최근 20행
```
- 특정 세션 행 찾기: `grep -n '(<번호>th,' claudedocs/EXPERIMENT_LEDGER.md`
- 행 1개만 읽기: `sed -n '<줄>p' claudedocs/EXPERIMENT_LEDGER.md` (한 행이 최대 7 KB)

## 5. 이 파일의 갱신 규칙

- 종료 세션 1개가 쓴다. `EXPERIMENT_LEDGER.md`에 행을 append한 세션은 **이 파일도 같은 턴에 갱신**한다.
- 갱신 = 맨 위에 새 항목 추가 + 20건 넘으면 가장 오래된 것 삭제. **원장은 절대 건드리지 않는다**(append-only).
- ⚠️ **행을 추가하면 `:512`~`:531` 앵커가 전부 밀린다.** §1의 `wc -l` 확인 명령을 먼저 돌리고 앵커를 다시 적을 것.
- 새 행은 **6열 스키마를 지켜라**(§2 ②). `Verdict` 칸에 `Dxxx`만 적지 말고 **판정 토큰과 비주장 한정어를 함께** 쓴다.
- 상한 **200줄**. 넘으면 오래된 항목의 ※ 주석부터 접는다.
