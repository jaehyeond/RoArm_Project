# LEDGER_RECENT.md — 최근 실험 20건 요약 (부팅 read)

Last updated: 2026-09-02 (76th 연장) — 현재 원본 `EXPERIMENT_LEDGER.md` = 567줄 / 표 블록 512행
(`:9-90` 82행 + `:105-536` 430행. 줄당 평균 약 2 KB, 최근 행은 5~9 KB — **단일 Read로 열면 토큰 초과**).
표 블록 끝 = `:536`(76th 연장 행 추가), `## Schema errata` = `:538`.

**2026-08-26 변경 (사용자 승인, append만 — 원본 삭제·수정 0건)**: 소급 등재 2행(`:532` 57th · `:533` 70th)
\+ 표 밖 `## Schema errata` 절 append (2026-09-02 기준 `:537~`). 앞 1,062,466 B는 **바이트 불변**(md5 `0a6d7071…` 대조 PASS).

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

원장 **표 블록의 마지막 20행 = `:516`~`:535`**, 정렬은 append 순. ⚠️ 75th·76th 가 2행을 더해 앵커가 2칸 밀렸다. 재확인 명령:
```bash
grep -n '^## Schema errata' claudedocs/EXPERIMENT_LEDGER.md   # 537 → 표 블록 끝 = :535
sed -n '516,535p' claudedocs/EXPERIMENT_LEDGER.md | awk -F'|' '{print NR+515": "substr($2,1,120)}'
```
⚠️ **2026-08-26부터 append 순 ≠ 시간 순이다.** 소급 등재로 `:532`(57th, 08-13)가 `:531`(69th, 08-16)보다
뒤에 있다. 시간순이 필요하면 앵커가 아니라 각 행의 Date 셀을 봐야 한다.
⚠️ 원장은 append-only라 **줄이 늘면 앵커가 전부 밀린다.** 위 `grep`이 535를 주지 않으면 이 파일부터 갱신할 것.

## 2. 🔴 원장 무결성 결함 4건 (요약하다 발견 — 원본은 손대지 않았다)

**① 등재 누락 — ✅ 2026-08-26 해소 (2건 소급 등재 / 1건은 결함 아님으로 재판정).**
초판은 이것을 "등재 누락 3건"이라고 썼는데, 세션문서를 열어 보니 **세 건이 같은 성질이 아니었다**:

```
56th  session_20260813_56th_g0b_boot_reverify_claude_handoff.md
      → doc `:72` "LEDGER append 0 (실험 없음), DECISIONS append 0" = 명시적·정당화된 미등재.
        순수 부트 검증이라 산출물 0. 결함 아님 → 등재하지 않음 (원 세션 결정 존중).
57th  session_20260813_57th_g0b_d444_flying_gripper_case_open.md
      → doc `:67-68` "LEDGER append 0 (물리 실행 없음 — 실행 세션에서 fg1 row 기록 예정)" = 명시적 결정.
        그러나 그 결과 **D444 case 개시가 원장에서 소실**됐다 → **`:532`로 소급 등재.**
70th  session_20260817_70th_cold_archive_t1_t2_migration.md
      → 원장/LEDGER 언급 0회. **유일하게 사유조차 무기록인 누락** → **`:533`으로 소급 등재.**
```
→ 원장 최종 등재 = **70th**(`:533`), 세션문서 최신 = **70th**. 뒤처짐 해소.
⚠️ 소급 행의 **판정 토큰은 2026-08-26 부여**이며 원 세션문서·`DECISIONS.md`에는 없다(각 행이 스스로 명시).

**①-b 등재 관행 자체가 비일관이었다 (신규 발견).** 물리 0인데 등재된 행이 이미 있다 — `:525` 63rd(조사 전용),
`:528` 66th(저작 전용). 즉 "실험 0이면 미등재"는 지켜진 적 없는 암묵 규칙이다.
새 기준은 원장 `### 등재 관행 메모`(`:553~`)에 기재: **`Dxxx`를 낳았거나 되짚어야 할 산출물·상태 변경을
만든 세션은 물리 실행 여부와 무관하게 등재**하고, 등재하지 않을 때는 세션문서에 사유를 남긴다.

**② 스키마 드리프트 — 하필 최근 3행.** 표 헤더는 6열(`Date/Label | Run/Path | Goal | Key Result | Verdict | Source`)인데
`:529`~`:531`(**67th·68th·69th = 현재 야드 피벗 전체**)은 **4열**이고, `Verdict` 칸에 판정 토큰 없이 `**D453**`/`**D454**`/`**D455**`만 있다.
앞선 17행은 `FG1_ALL_13_FAIL_...` 같은 기계 판독 토큰 + 비주장 한정어를 달고 있다.
→ **가장 최근이고 가장 필요한 3행이 가장 정보가 적다.** 판정 내용은 `DECISIONS.md`(D453~D455)로만 도달 가능했다.
(전체 분포: 6열 494행 / 4열 3행 / 나머지는 셀 안 `|` 때문에 필드 수 7·9~14)

**✅ 2026-08-26 보정 (원행 무수정).** append-only라 `:529`~`:531` 자체는 고칠 수 없으므로, 원장 **표 밖**에
`## Schema errata`(2026-09-02 기준 `:537~`) 절을 신설해 세 행의 누락된 `Run/Path`·`Goal`과 **소급 판정 토큰**을 보정 기재했다.
표 블록이 아니므로 마크다운 렌더에 영향 0. ⚠️ 소급 토큰은 검색·기계 판독용 보조 표기일 뿐이고
**판정의 정본은 언제나 `DECISIONS.md` D453~D455 원문**이다(어긋나면 원문이 이긴다).
신규 행 `:532`·`:533`은 **6열 스키마 준수**(필드 8 = 6열) — `awk -F'|' 'NR>=532&&NR<=533{print NF}'`로 확인 가능.

**③ 표가 두 블록으로 쪼개져 있다.** 표 헤더는 `:7-8`에 한 번뿐인데 `:105`~`:531`(427행)이
헤더 없이 이어진다 → 마크다운에서 **두 번째 블록은 표로 렌더되지 않는다.**

**④ 원장 한복판에 죽은 상태 12줄.** `:92-103` `## Current Next Experiment Candidate`가
**"Active pivot (2026-05-21): Track A P7/Branch B ..."** 라고 단언한다 — 현재 피벗(포스코 야드, 63rd~)과 **정면 모순**.
AGENTS.md에서 걷어낸 결함 B(죽은 상태가 규칙/참조 파일에 상주)와 **같은 패턴**이다.
→ 부트 3단계가 이 파일 대신 `LEDGER_RECENT.md`를 읽게 되면서 **이 12줄은 더 이상 자동 주입되지 않는다.**
원본은 append-only 정책상 **삭제하지 않았다.**

## 3. 최근 20건 (신 → 구)

> 형식: `앵커 · 세션` — 무엇을 돌렸나 → **판정** · 근거. 세션문서는 전부 `claudedocs/` 아래.
> 4열 행(`:529`~`:531`)은 원장 표에 판정 토큰이 없어 `DECISIONS.md`를 근거로 표시했다
> (2026-08-26 `## Schema errata` `:535~`에 소급 토큰 보정 기재됨).
> ⚠️ **앵커 순서 ≠ 시간 순서**: 소급 등재된 `:532`(57th, 08-13)는 앵커상 뒤에 있지만 시간상으로는 58th 앞이다.
> 아래는 **시간순**으로 배열했다.

### 현행 피벗 — 포스코 야드 (63rd~)

- **`:537` · 76th 연장** (09-03 새벽) **Phase 3 자산화**: 구동 1축 URDF → 로봇 합성(link5 부착) → Isaac 5.1 USD → RTX 렌더
  → **`PHASE3_URDF_USD_MATERIALIZED__PER_PIECE_CONVEX_COLLISION_D446_AVOIDED__MIMIC_TO_NORMAL__CUSTOM_GRAB_ADDED_NOT_REPLACING_STOCK_GRIPPER`**
  (**D474** `:29848`) · `local_assets/roarm_m3/`
  ※ 🔴 **정정: 커스텀 그랩은 순정 그리퍼 "교체"가 아니라 "추가"**(브래킷=순정 고정 조, 서보크랭크=순정 가동 조에 볼트, 순정 서보로 구동).
  ※ 🔴 collider=조각별 볼록350(D446 회피) · mimic→독립(Isaac 구동) · 카메라 AABB→1.14m(근접이면 프레임 놓침).
  ※ ⚠️ **물리 시뮬 0**(펠릿=DEME 별개, USD↔DEME 브리핑만) · 구동 인출 실물 미검증(Phase 2 잔여).
- **`:536` · 76th 연장** (09-02 심야) Phase 1 요크 양단지지 구현 + **실물 로봇 장착검증** + 손목롤 제약
  → **`PHASE1_YOKE_BOTH_END_SUPPORT__G2_ATTACH_OK_ON_REAL_ROBOT__MASS_61G_ALU_BOLTS__WRIST_ROLL_X_OPEN_LINK4_CONSTRAINT_DOCUMENTED`**
  (**D473** `:29764`) · `PHASE1_ASSEMBLY_DEFINITION.md` · `runtime_logs/grab_track/g17_yoke_alu/`
  ※ 정본 형상 = `g17_yoke_alu/`(g9_sidefix·g16* 낡음). 커밋 `bcde1df`·`541cda3`·`d58f6b6`.
  ※ 🔴 **실물 로봇 장착을 사용자가 지적** — p37 `G2_ATTACH_OK`, 백포스트 0.263→x-shift로 기존값 복귀(요크 무악화).
  ※ 🔴 **손목롤×개폐 link4 간섭** = servocrank(링크 선재)가 완전개방+롤±14°밖서 침범. 요크 무관. 문서화+G8b.
  ※ ⚠️ **출력 0 · URDF·USD 0(Phase 3 미착수) · 서보 인출 미검증(Phase 2 잔여)**. 반치·백래시·유격 미실측.
- **`:535` · 76th** (09-02) 셸 L·R 실물 완주 2건 + 워커 P4 트랙 코디네이터 독립 검증
  → **`G10_G11_ADHESION_CAUSE_CONFIRMED_BY_CONTROLLED_SLICE_ONLY_CHANGE__CONTACT_PER_GRAM_GATE_CALIBRATED_ON_REAL_SUCCESS_FAILURE_PAIR__D_D_RECOMPUTED_ON_CURRENT_SHELL_6PCT_TO_62PCT`**
  (**D469** `:29349` · **D470** `:29477` · **D471** `:29548`) · `session_20260902_76th_g10_g11_print_and_worker_adjudication.md`
  ※ 형상 무수정으로 슬라이스만 바꿔 완주 = **통제 대조**. 임계 125 mm²/g 는 **성공 1·실패 1** 표본이다.
  ※ 🔴 워커의 그랩 포획 판정이 **낡은 셸**(08-31)로 계산돼 있었다 — 수정본은 **6% 가 아니라 62%**.
  ※ ⚠️ Newton "38배·VRAM 424 MiB" 는 **미검증**(입력이 산출물에 없음). 인용 금지.
- **`:534` · 75th** (09-01) 출력 파이프라인 수리 + P1 n=5 + 게이트 정비 (실물 출력 4회: 3실패 1완주)
  → **`PRINT_PIPELINE_REPAIRED__GATE_BLINDSPOTS_8_ALL_INTENT_NOT_RESULT`** (**D465** `:28960` · **D466** `:29050`)
  · `session_20260901_75th_print_pipeline_repair_p1_n5.md`

- **`:533` · 70th** (08-17) 콜드 아카이브 T1/T2 이관 — git 비추적 대형 14폴더 ≈176GB를 외장으로 검증-사본 후
  move-only (**연구 실험 0 · 물리 0 · 로봇 0**, 스토리지 인프라 전용). ⚠️ **2026-08-26 소급 등재**
  → **`T1_T2_COLD_ARCHIVE_MIGRATED__SHA256_ALL_MATCH__SINGLE_COPY_NOT_BACKUP__T3_PENDING`**
  (판정 토큰 소급 부여 · DECISIONS 신규 0) · `session_20260817_70th_cold_archive_t1_t2_migration.md` · `ARCHIVE_INDEX.md`
  ※ 이관분은 **사본 1개 = 백업 아님**(외장 분실 = 데이터 손실). **외장 미마운트 시 심링크 14개 끊김.**
  **T3 45G(`b200_backup_*` 2종 + `openvla_oft_b200_pulls`)는 유일 사본으로 내장 유지 — 2사본화 결정 대기.**
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
- **`:532` · 57th** (08-13) `g0b_d444` case 개시 — prereg 13 pose 동결 + Grasping SDG 1.0.9 소스 감사
  (**물리 0 · Isaac 0**, git commit/push `b9020fd`). ⚠️ **2026-08-26 소급 등재** (원 세션은 "LEDGER append 0"을
  명시 결정했으나 그 결과 case 개시가 원장에서 소실)
  → **`G0B_D444_CASE_OPENED__PREREG_FROZEN__NO_PHYSICS`** (**D444** `:27616`, 판정 토큰 소급 부여) ·
  `session_20260813_57th_g0b_d444_flying_gripper_case_open.md`
  ※ **D444는 개시 선언이지 물리 verdict가 아니다.** 이 case의 물리 판정은 바로 아래 58th `fg1`(D445).
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
- **`:515` · 52nd** (08-11) `t3p` 접촉력 계측 물리 시행 → **`..._ZERO_LIFT_IN_1024__MECHANISM_IS_PRESS_INTO_TABLE_NOT_PINCH`** (**D439**) · `session_20260811_52nd_g0b_t3p_randomized_parallel_sweep.md`

### 감사·패널 (물리 재실행 0 — 문서 무결성 축)

- **`:514` · 51st-b** (08-11) 적대 패널 `wf_46941a6d-04e` 회수 13/13 (2,185,034 tok · 633 calls)
  → **`PANEL_CONFIRMS_D437R1_CORE__REFUTES_8_OF_51ST_OWN_REDERIVATIONS__DOCINT_SELF_INVALIDATED`** (**D438-R1**)
  ※ **⛔ 진행 승인 아님.** 12 패스에서 신규 물리 측정 0건. **세션문서 없음** — 근거는 `DECISIONS.md` D438-R1 + `g0b_d420/t3d_panel_wf46941a6d_*`.
> **회전 이탈 2건 (2026-08-26, 소급 등재 2행이 들어오면서 20건 상한 밖으로 밀림 — 앵커는 계속 유효):**
> **`:513` · 51st** (08-11) 부트 재검증 + N-1 프록시 재도출 → **D438** ·
> `session_20260811_51st_g0b_boot_n1_proxy_rederivation.md` /
> **`:512` · 50th-b** (08-10) 적대 패널 `wf_84c6e3b4-92a` 13/13 → **D437-R1**(D427~D437 헤드라인 전부 불변,
> D433 `LIFT_FAIL`×3 전건 재현) · `session_20260810_50th_g0b_boot_reverify_panel_launched.md`.
> 이 둘의 교훈 원문은 `DECISIONS_ACTIVE.md` §8(D437-R1 `:26932` · D438-R1 `:27181`)이 계속 보유한다 —
> **"무너진 것은 물리 결론이 아니라 그 결론을 받친다던 근거·범위·선례다."**

## 4. 원장 항해 인덱스 (통째 read 금지)

```
:1-5     머리말 (원장 자신의 경고 — "이것만으로 수치 인용하지 말 것")
:7-8     표 헤더 (6열, 파일 전체에서 여기 한 번뿐)
:9-90    표 블록 1 — 82행 (~2026-05-21)
:92-103  ⚠️ 죽은 상태 "Current Next Experiment Candidate" (2026-05-21) — 원본 append-only라 삭제 안 함.
         부트 3단계가 이 파일을 대신 읽으므로 **자동 주입은 멎었다.**
:105-533 표 블록 2 — 429행 (헤더 없음 → 표로 렌더 안 됨)
:514-533 ← 이 파일이 요약한 최근 20행
:532-533 ← 2026-08-26 소급 등재 (57th · 70th, 6열 준수). **시간순 아님**
:535-552 🆕 ## Schema errata — :529~:531의 누락 열 + 소급 판정 토큰 (표 밖, 렌더 영향 0)
:553-564 🆕 ### 등재 관행 메모 — 물리 0 세션의 등재 기준
```
- 특정 세션 행 찾기: `grep -n '(<번호>th,' claudedocs/EXPERIMENT_LEDGER.md`
- 행 1개만 읽기: `sed -n '<줄>p' claudedocs/EXPERIMENT_LEDGER.md` (한 행이 최대 9 KB)

## 5. 이 파일의 갱신 규칙

- 종료 세션 1개가 쓴다. `EXPERIMENT_LEDGER.md`에 행을 append한 세션은 **이 파일도 같은 턴에 갱신**한다.
- 갱신 = 맨 위에 새 항목 추가 + 20건 넘으면 가장 오래된 것 삭제. **원장은 절대 건드리지 않는다**(append-only).
- ⚠️ **행을 추가하면 `:514`~`:533` 앵커가 전부 밀린다.** §1의 `grep -n '^## Schema errata'` 명령을 먼저 돌려
  표 블록 끝을 잡고 앵커를 다시 적을 것. **`wc -l`은 더 이상 표 끝이 아니다** — 2026-08-26에 표 밖 절이 생겼다.
- ⚠️ 새 행은 표 블록 **끝(`## Schema errata` 절 바로 위)** 에 넣는다. 절 뒤에 붙이면 표에서 이탈한다.
- 새 행은 **6열 스키마를 지켜라**(§2 ②). `Verdict` 칸에 `Dxxx`만 적지 말고 **판정 토큰과 비주장 한정어를 함께** 쓴다.
- 상한 **200줄**. 넘으면 오래된 항목의 ※ 주석부터 접는다.
