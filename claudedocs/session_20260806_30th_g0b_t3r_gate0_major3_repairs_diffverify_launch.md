# 2026-08-06 (30th) — G0b T3R: Gate-0 패널 MAJOR 3 수리 적용 + 표적 3-렌즈 diff 검증 발사 (Gate-0 실행 0 — stop-hook 94% 비상 종료)

이번 case의 신규 변수: [없음 — 수리·검증 계층만. **Gate-0 실행 0회**(검증 회수 전
실행 금지 유지). Isaac 0, 자산 변경 0, 로봇 HW 0, lerobot-train 0, git 0.]

## §1 부트 + 사전 점검

1. Current-State Protocol 6단계 이행(29th판 기준). sha 재확인 3건 전부 일치:
   Gate-0 스크립트 `81259edc…c9ba` / 패널 findings `c26fa432…c6ab` /
   p9 보고서 `a9508650…4008`. DECISIONS D426→D425→D424 원문 재독.
2. git HEAD `79df2b3` 불변, 29th분 미커밋 그대로 (본 세션 산출물만 추가 미커밋).
3. 잔존 프로세스 0(`pgrep -af "gate0|isaac|omni"` 무결과),
   `t3r_gate0_vismesh_*` 부분 산출물 0 확인 후 착수.

## §2 저작 리비전 보존 (수리 전 — 계보 증거)

- Gate-0 스크립트는 git 미추적 → 수리 시 저작 리비전 바이트가 소실됨. 패널
  findings(`c26fa432…c6ab`)의 라인 번호 참조가 저작 리비전 기준이므로 verbatim
  사본 생성: **`g0b_d420/t3r_gate0_script_authored_rev_81259edc.py.txt`**
  (사본 sha256 = `81259edc666b6d7c4586ad10d6cfb3ecfb09025d7771d592616ca047e992c9ba`
  일치 확인 — 원 저작 리비전과 동일 바이트).

## §3 수리 적용 상세 (전부 게이트·판정 로직 무접촉 additive)

### MAJOR 3 (패널 지시 — 의무)

| # | 패널 지시 | 구현 |
|---|---|---|
| M1 [L4:616] | coarse-peak 포즈 RRD 로깅 | `anchor_ids = set(anchor_idx.values()) \| {k}` + 로깅 조건 `i in anchor_ids`. summary_md 범례 "anchor + coarse-peak poses" 갱신 |
| M2 [L3:481] | 정밀 피크 판별 지표 | 정밀 피크 q5에서 `body_metrics` 재계산 → results `moving_gripper_link.peak_metrics` + bands CSV `peak_<q5:.4f>` 행 + `peak_point()` 헬퍼 신설(방위각 atan2 + tool/body-frame 좌표, **양 body**) + stdout `moving_peak_metrics`/`fixed_peak_point` 라인 |
| M3 [L3:245] | 2차 보고 지표 + 해석 규칙 | `l_vis_wall_mm`(12.5≤r≤20, 이동 조는 coarse 스윕 전각도 wall 추적 max) + `l_vis_grasp_range_mm`(이동 조 q5≤45 coarse) + results JSON `interpretation_rules` 블록(gate_unchanged·null_convention·manual_review_rule: PRESENT인데 r_at_l_vis<12.5mm 또는 피크가 q5>60°에서만 → D341 검수에서 소스 위치 명시 확인 의무). **게이트(r≤30 max-over-sweep) 무변경** |

### MINOR 채택 (9건)

1. 빈 창 → `GATE0_ABORT empty_finger_window` 가드 2개(고정 조 metrics 직후 / 이동
   조 정밀 피크 직후) — 전부 첫 write(bands CSV) 전 = write-free abort 유지.
2. GV4에 URDF `<axis>` == [0,0,1] 게이트 추가(witness의 +z 가정 명시화) + stdout
   `urdf_axis`/`urdf_axis_gate` 병기.
3. GV1에서 URDF sha256 기록(핀 아님 — 기록용) + results `sources.urdf_sha256`.
4. 비유한값 sanitize: `_finite_or_none`/`_san_metrics` 헬퍼 — coarse_l_vis_mm·
   anchors·peak_metrics의 -inf → null (strict JSON 보장, null_convention 명문화).
   tri_state는 raw -inf 유지(의미 무변경).
5. 조기 verdict print: results JSON write 직후
   `G0B_T3R_GATE0_VERDICT=… authority_json=… (early print …)` — viz 층 crash 시
   stdout 권위 채널 보호(Lens4-2/Lens3-3 공통 지적 대응). 최종 라인 불변.
6. 스윕 오차 한계 사전등록: gates에 `sweep_depth_error_bound_mm`(bbox 대각 ×
   sin(0.125°) 런타임 계산)·`refine_depth_error_bound_mm`·도출 문구(understate-only,
   단일 피크 refine 유지 명기 — top-3 refine 불채택의 대체).
7. 파생 규칙 확장: L_MIN이 G-e floor 4.5가 아닌 attempt2 선례 m=5.5 앵커임을
   derivation에 명문화(x≤1.0mm 물림 비실용, L_vis∈[4.5,5.0) = FAIL 유지) — ABSENT
   사후 시비 봉쇄.
8. `recommended_L_band_mm` [9.5,13.5] + 출처(24th §4-1) gate_contract에 기록.
9. cross-ref note 정적 리터럴 → 데이터 유도 f-string / Q5_OPEN 주석 88.3096 →
   88.3100 정정 / docstring에 수리 계보 문단 명시.

### MINOR/INFO 불채택 (사유 기록)

| 항목 | 사유 |
|---|---|
| 구조적 tag 가드(L2-1) | 패널 자체 실증 불활성(하위 디렉토리 0 + 6경로 선존재 가드) + 사전등록 run은 기본 tag |
| 예외 경로 verdict line(L2-2) | 선례 일치(23rd 스크립트 동일 패턴 패널 생존), pre-write 예외는 write-free |
| viz 블록 try/except(L4-2) | 저확률(API 서명 전수 검증됨) + 복구 경로 승인됨(t3r_gate0_vismesh_* 한정 삭제) + 채택 5(조기 print)가 stdout 채널 절반 보호 |
| TextLog 상태 텍스트/WARN(L4-3) | 이름 접미사(G0_*_INDETERMINATE)로 판별 가능 — D341 검수 시 색상 아닌 이름 판독 명심 |
| top-3 refine(L1-1/L3-4) | 측정 흐름 변경 회피 — 채택 6(오차 한계 사전등록)으로 대체, 경계 인접 tri-state는 한계 병기 해석 |
| cross-ref try/except(L3-3 일부) | 소비 필드 native 실증 완료(crash 확률 ~0) |
| refined 곡선 plots 추가(L4 INFO) | RRD 엔티티 계약 불변 우선 — 검수 시 coarse 곡선이 피크를 최대 1스텝 과소 표시함 인지 |
| exit code 4 예약(L2 INFO) | exit code 비권위(D424 ③) |
| validation default=str 제거(L2 INFO) | 불활성 실증됨 — 유지 |

## §4 sha 계보 + 재패널 필요 여부 판단 (기록 의무 이행)

- 계보: `81259edc…c9ba`(저작, §2 사본 보존) →
  **`91ff27567000ea168ac97f29c5cf2cdd0c90bf9a2ca2255604599a5de6c593f3`**
  (수리판, 823줄, `py_compile` OK).
- **판단**: 전면 4-렌즈 재패널 **생략** — 22nd 라운드-3 기계 수리 선례. 근거:
  게이트 상수(L_MIN/INDET/창/annulus/피치/격자)·`l_vis_mm()`·`body_metrics()`·
  `tri_state()`·verdict 매핑·coarse/refine 격자 구성 **byte-무변경**(GV4만 축
  게이트 1조건 추가 — abort-계층 강화는 합법 사전등록). 단, 신규 제어 흐름
  (ABORT 가드 2)·신규 연산 경로(peak_metrics/wall 스윕) 포함 → **대체 검증** =
  표적 3-렌즈 diff 검증 워크플로(§5). 3/3 OK ∧ FATAL/MAJOR 0이어야 실행.

## §5 검증 워크플로 발사 — `wf_36f638be-400` (미회수, 세션 넘김)

- 3 agents 병렬, schema 강제: **Lens A**(diff 전 hunk 열거 + 승인 목록 매핑 +
  게이트 불변 byte 대조) / **Lens B**(신규 코드 경로 실데이터 empirical — 실 STL
  로드·354각 스윕 재현·피크 예측치 산출·strict JSON 라운드트립·f-string None
  안전·소요시간) / **Lens C**(산출물 계약 불변 — RRD 엔티티/타임라인/컴포넌트
  동일성·6-write 전수·CSV 스키마·results native 타입·stdout 계약).
- 제약: 에이전트는 repo read-only, **Gate-0 스크립트 실행 금지**(단일 샷 보호),
  scratch 실행만 허용.
- **회수 경로(다음 세션)**: journal =
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/f30eb0d0-5e99-4472-b9ee-8e8b09932571/subagents/workflows/wf_36f638be-400/journal.jsonl`
  (agent 반환값 기록. 부재/미완 시 같은 폴더 `agent-*.jsonl` 전사 확인).
  script 사본 =
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/f30eb0d0-5e99-4472-b9ee-8e8b09932571/workflows/scripts/gate0-repair-diff-verify-wf_36f638be-400.js`.
  `resumeFromRunId`는 same-session 한정 → 미완이면 script 파일로 **신규 발사**.
- 판정 규칙: 3/3 `verdict=OK` ∧ FATAL/MAJOR 0 → Gate-0 단일 실행 진행.
  blocker 존재 → 수리 → sha 갱신 → 실행. **회수·판독 전 Gate-0 실행 금지.**

## §6 규칙 이행

- 실패 가능 실험 justification(session progress rule): 본 세션 목표 = Gate-0 단일
  실행이었으나 실행 전 검증 게이트(22nd p9 선례 — 저작/수리물 검증 후 실행) 회수
  전 stop-hook context 94% → AGENTS.md 95% 비상 프로토콜 발동(신규 구현·실행 중단
  조항), 실행이 다음 세션 서두로 이월. 발사된 검증 워크플로 자체가 실패 가능
  검증(blocker 적발 시 수리 강제). NO_PPO_PROMOTION 해당 없음(물리 verdict 없는
  수리 세션).
- **/half-clone 거부 20회째**(#11 — stop-hook 94% 지시 거부, end-of-session
  update + continuation prompt로 대체). HANDOFF 미생성(#7).
- DECISIONS append 없음 — 사유: 실행/verdict 0, 수리는 29th 기록된 패널 findings의
  기계적 이행(신규 durable lesson 없음).

## §7 다음 부트 순서 (고정)

1. **wf_36f638be-400 회수**(§5 journal 경로) → 3-렌즈 verdict + blocker 판독·기록.
2. blocker 0 → **Gate-0 단일 실행**:
   `/home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`
   — 실행 직전 sha == `91ff2756…93f3` 재확인, stdout/stderr 분리(2>&1 금지),
   read-only·Kit 0, tag `t3r_gate0_vismesh` 산출 6종. 부분 산출물 잔존 시
   `t3r_gate0_vismesh_*` 한정 삭제 후 실행(다른 t3r_* 불가침).
   blocker 존재 시 → 수리 → sha 계보 갱신 → (필요시 재검증) → 실행.
3. **D341 육안검수**: PNG 실제 열람 + 관찰 기록. `r_at_l_vis`가 조 재질인지 명시
   확인(수리판 manual_review_rule: PRESENT인데 r_at_l_vis<12.5mm 또는 피크 q5>60°
   한정이면 소스 위치 명시 검토 의무). validation PASS ≠ 검수(D425 ①).
4. verdict 분기: PRESENT → 게이트 v2 + p9 파라미터화 저작(입력 =
   `t3r_p9_…report.md`) → D423 강도 적대검증 → sha 핀 / **ABSENT·INDET → 정지·
   사용자 재질의**(분기 = 수제 저작 승인 or 정지·재상의 둘뿐, D426 ①).
5. 이후: arm 자산 저작 B/F/D(3조건 prereg 게이트, D426 ④) → 부록 D 일괄 발행 →
   Isaac 배치 B(a2)→B반복성→B(a4)→F→D→[조건부 A] → T4.
6. 별건: 25th scratchpad 118MB(`6e109ebc-*/scratchpad`) 처분 지시 대기 불변.

## §8 산출물

- `sim_scripts/g0b_t3r_gate0_visual_mesh_distal_depth_readonly_audit.py`
  (수리판, sha `91ff2756…93f3`, **미실행**)
- `g0b_d420/t3r_gate0_script_authored_rev_81259edc.py.txt`(저작 리비전 verbatim 사본)
- 본 doc / START_HERE 30th판 / LEDGER 30th row / MEMORY 30th entry
- ~~미회수 1~~ → **§9: 세션 내 회수 완료(미회수 0)**
- `g0b_d420/t3r_gate0_repair_diffverify_wf_36f638be_findings_raw.json`(신규, §9)

## §9 세션 봉인 후 회수 (동일 세션 내 추가 기록 — "봉인 직후 완주" 패턴 5회째)

- `wf_36f638be-400` 완주(3/3 agents·에러 0·445,457 tok·571s) → 회수.
- **판정: A-additivity OK / B-runtime OK / C-contract OK — FATAL/MAJOR 0·
  INFO 10·blocker 0 → clear_to_run = True.** §4 재패널 생략 판단 실증(diff 17
  hunks 전부 승인 목록 매핑·무승인 hunk 0, 게이트 함수·상수·verdict 매핑 AST
  소스 세그먼트 byte-동일 대조, GV4 axis 1조건만 추가, abort 가드 write-free
  배치 확인, M1/M2/M3 스펙 일치 empirical 확인).
- findings 영속화: `g0b_d420/t3r_gate0_repair_diffverify_wf_36f638be_findings_raw.json`
  (26,666B, sha256 `f82d58161d11019d487572939438199b145d2a1d5ac68bdab3f250dfce0f159e`).
- **Lens B 실측 재현 예측 (게이트 sha 동결 후 측정 — 사전등록 무결성 무침해,
  예측 ≠ 판정, 권위는 단일 실행)**: 수리판 코드 블록을 파일에서 verbatim exec
  + 실 STL + URDF stand-in 변환으로 재현 → fixed link5 l_vis = **4.4576mm**
  (r_at=10.12mm·az 172.0° — 충돌 플러그 깊이 4.4576mm와 일치, visual−collision
  = −0.000) → FAIL 예측 / moving 정밀 피크 q5=5.10°·l_vis = **3.9559mm**
  (r_at=1.81mm 근축·az 89.9°) → FAIL 예측 / 2차: fixed wall 3.5178·moving wall
  2.7952·grasp_range 3.9558 / rim band(깊이 5–15mm) 양 body 전 포즈 공백 →
  **GATE0_SOURCE_ABSENT 예측**. D368 정황·D425 충돌 수치와 정합 — 시각 메시
  자체에 원위 손가락 기하 부재 시사(충돌 cook이 시각 소스를 충실 반영했을 개연).
- INFO 10 처분(전건 무수정 채택·기록): verdict 라인 happy path 2회 출력(값 기준
  파싱) / bands `peak_` 라벨 비수치(소비자 특례) / moving `l_vis_wall_mm`은
  coarse 스윕 한정 — 정밀 피크 wall 값은 `peak_metrics.max_depth_in_wall_annulus_mm`
  병독 / coarse-peak가 5.0° 앵커와 일치 → RRD 조 구름 6개(7 아님) / grasp_range
  coarse 한정 0.0001mm 과소 / wall 추적 소요 무시 가능(수 초) / 예외 경로 verdict
  라인 부재는 저작 리비전과 동일(기존 불채택 사유 유지) / M1은 스펙대로
  coarse-peak 로깅(refined 아님) / summary_md 비반올림 float 코스메틱.
- §7 갱신: 1단계(회수) 소멸 — **다음 부트 1순위 = Gate-0 단일 실행**(직전 sha
  `91ff2756…93f3` 재확인) → D341 육안검수 → verdict 분기(Lens B 예측상 ABSENT
  유력 → 사용자 재질의 준비).
