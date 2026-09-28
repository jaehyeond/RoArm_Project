# session 2026-09-28 (밤) — W25: 그리퍼 "덜 그려짐" 원인 규명 · 실물 정렬 전체 사이클 재실행 준비(rev34) · worktree 보관 2차 · 랩미팅 정정

> 이번 case 의 신규 변수: **없음(물리·GPU·RunPod·실물·설치·commit 0)**. 준비·CPU 검증·문서·보관만.
> 메인 = Claude Code Fable 5.1(1M). 워커 = Orca Run `run_d0eba99ce41b`(코디네이터 터미널 `term_82e35414-…`), A `w25-rev34-fullcycle`(claude-opus-5-5) · B `w25-render-cad`(claude-opus-5-5) · C `w25-pile-runpod-plan`(codex gpt-6-astra). 과제서 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/coordinator/TASK_SPEC_w25{A,B,C}_*.md`(기대 답 없이 입력·요구만).
> **Session progress rule 사유**: 실패할 수 있는 검사 = 메인 독립 검산(CAD 배치식 vs 저장 노드 4 sync, 더미·비용 산술 6행 재계산), 워커 CPU 검사(post05 단위 18/18, rev34 사전검토). 물리 실행은 사용자 GPU/RunPod 승인 전이라 범위 밖.

## 1. 사용자 피드백(랩미팅 자료)과 부팅
- 사용자: 첫 완주 렌더(`figures/02~04`)의 그리퍼가 "이전 것 아니냐 / 덜 그려진 것 아니냐", 실물은 상자를 돌려 놓는다 → 실물 조건으로 전체 사이클 재실행(RunPod)·Isaac 렌더·worktree 정리 2단계 직접·교차 검토 후 브리핑.
- 부팅: START_HERE → DECISIONS_ACTIVE(D490~D497) → LEDGER_RECENT → relay(from_claude/from_codex) → W22 세션 §8~§9 → `git status`(미커밋 64건, worktree 20).

## 2. 그리퍼 판정 — "덜 그려진 것"이 맞다 (메인 원자료 확인 → B 워커가 직접 원인 특정)
| 사실 | 근거 |
|---|---|
| 시뮬 물리 툴 = 해석적 **충돌 셸**(보울 벽+캡만, 고정 204·문 204 삼각형, 108 정점) | `sim_deme_scoop_s1.py:100-140 half_bowl()`, `:170-193 load_tool()`(STL 은 검사에만) · W19 `_obj/fixed_seed460.obj` V108/F204 |
| S1 v1 STL(실물 최신, D481) = `fixed_ALL.stl` 1,704 tri · `door_ALL.stl` 4,832 tri; 출력본 `s1_v1_print/fixed.stl` 은 같은 V/F(출력 방향 회전) | sha256 `2fea1c37…`(fixed_ALL) / `beef6062…`(print) — 바이트 다름, 위상 동일 |
| 후속 설계 버전 없음(`s1_v2` 는 "전단 립 미설계" 언급뿐) | `DECISIONS.md:30150`, 폴더 `g19_servo_direct/` 목록 |
| 재생(post04)은 셸 노드를 그대로 그리고, USD 의 **S1 v1 문 CAD(`gripper_link`)를 `gripper` 이름 필터로 숨김**, 고정부 CAD(`grab_fixed`)는 보임 → 문은 파란 셸만 | `isaac_replay_w13.py:740-772, :834-852`, B `usd_prims.md`(USDC 토큰 해독), post04 `render_manifest.json` 숨김 목록 |
| 사용자 질문 "옆 부분 렌더 덜 된 것?" → 예. 그리퍼 설계는 최신(S1 v1)이고 표시층 문제 | 위 전부 |
- 메인 비교 그림: `claudedocs/research/labmeeting_20260929/lab_pc_bundle/figures/08_gripper_shell_vs_cad.png`(A 셸 vs B CAD, link5 프레임).

## 3. 상자 방향 사실 확정 (기준 좌표 명시)
- W19 A 실행 로그 `run_01/runA_full_cycle.stdout.txt:2` `t_robot [0.350013, -0.0081, -0.283241]` = DEME 원점(상자 바닥 중심)의 로봇 좌표; `w13_fk.py Adapter` "축 평행·원점만 이동" → 로봇 베이스 = DEME (−0.350, +0.008) = **상자 x(31 cm) 축 위** → 시뮬은 **22 cm 벽이 로봇을 마주 봄**. 실물 = 31 cm 벽. 더미 NPZ `box_bounds_m` = x ±0.155·y ±0.11(메인이 NPZ 직접 확인).
- `w13_kinematics.py:10-16` 도크스트링의 "(0, +R)" 는 낡은 선언(FK 확정 전) — 실행값과 다르다(rev34 에서 정정 대상).
- 전체 사이클의 툴 자세는 `Adapter.owner_pose()` 가 실물 FK `R_l5` 를 쓰므로 베이스각을 따라 돈다(W23 F 의 `tool_yaw` 패치는 스쿱 셀 전용) — A 워커가 확인 보고 예정.

## 4. 워커 산출과 메인 교차 검산
### C `w25-pile-runpod-plan`(codex gpt-6-astra, 완료·release) — `…/w25-pile-runpod-plan/claudedocs/research/w25_pile_runpod_plan_20260928/{PLAN,REPORT}.md`
- NTC106 301×198 평평한 층: 30/35/40 mm × 0.456/0.55 g/cm³ → **40,247 ~ 64,725알**(815~1,311 g). 기존 20k 를 NTC106 전면에 펴면 12.4~14.9 mm(< 30 mm) → 재사용 불가.
- 전체 사이클(W19 A 20,977.836 s 기준, 세 모형): **10.30 ~ 22.60 h, 7.6 ~ 16.7 $**(0.74 $/h) — 여섯 조건 전부 현행 cap 32,400 s(9 h) 초과 → **현 규약으로 GO 불가, cap 은 새 revision·criteria 로 사전 결정 필요**. 09-28 밤 기동 시 회의 전 완주 불가.
- 함정: slab 생성기 가장자리 여백(13.5 mm/변, 목표 발자국 상한 274×171) → 전면 평탄층은 생성 후 높이 대조 관문 필요 · 부트스트랩 v5 경로 하드코딩 → W25 사본 필요 · A2 과금 4.08 $ vs 4.31 $ 계산 불일치(원인 미확인).
- **메인 검산**: NPZ 템플릿 질량 0.020257382129428684 g 동일 · 6행 알 수/시간/비용 소수 2자리까지 일치(`budget_w25.py --check` PASS, `verify_budget.py` PASS, 메인 독립 산술 일치).
### B `w25-render-cad`(claude-opus-5-5, 완료·release) — `…/w25-render-cad/claudedocs/research/w25_render_cad_20260928/REPORT.md`
- CAD 배치식(고정: `p + R(q)·R_W·(v−L5)`, 문: `p_d + R(q_d)·R_W·roty(27.5°)·(v−H5)`, 상수는 결과 JSON 에서) → W19 A **16,813 sync 전부** 셸 노드 재현 최대 7.2e-5 mm, 음성 대조 3.0/56.0/119.3 mm.
- post05 초안: `/World/s1_cad/{fixed,door}` 원시 포즈, USD S1 비주얼 2개 숨김(이중 표시 방지), 셸 반투명 옵션, fixtures 일치 게이트, 단위 18/18, `--ik-only` 실경로 rc0(관절 계획 post04 동일). 그림 8장. Rerun 동반 RRD 표본 `rrd verify` PASS.
- 한계: Isaac 실제 렌더는 GPU 승인 후(명령 REPORT §6) · IK 프레임 56장에서 CAD 가 팔 끝과 최대 8.7 mm 어긋날 수 있음(원시 포즈 vs 표시 IK, post04 셸도 동일).
- **메인 검산**: 워커 코드를 쓰지 않고 sync 0/3000/11549/16812 에서 셸 owner 포즈 재현 최대 3.5e-5 mm, 음성 대조(+3 mm) 3.000 mm — 일치. 그림 `fig_f201_{overlay,context}.png` 육안 확인(문 27.5° 열림, 셸이 CAD 보울을 감쌈).
### A `w25-rev34-fullcycle`(claude-opus-5-5) — (아래 §4-A 에 완료 후 append)

## 5. worktree 정리 (직접)
- 09-28 저녁 6개(w19-audit·w19-rev33·w19-role-agents·w20-particle-count·w21-p2-runs·w21-smoke-evidence) **2단계 완료 재검증**: `archive.final.json` = `via_original_path.json` 6/6, `HEAD.bundle` verify 6/6, `orca_remove.json ok` 6/6, 태그 `archive/<이름>` = `fc557db0f1c2`, 심링크 6/6 정상.
- 신규 보관 후보 10개(W23 7 + W24 3, 전부 완료·산출 START_HERE/세션 반영, HEAD `3267dcb`, 미커밋 2~93건, 각 3.41~3.43 GB): **1단계 복사+전량 SHA 매니페스트 대조 완료**(`_plan_20260928_<이름>/{source.before,source.after,archive}.json` cmp 일치, 22:16~22:18) + **태그 `archive/<이름>`(3267dcb) 10개 + bundle 10개 verify OK**(각 1.29 GB).
- **미완(사용자 권한 필요)**: `orca-ide worktree rm --force` + 원경로 심링크 + 심링크 경유 재대조 — 자동 모드 분류기가 "Irreversible Local Destruction" 으로 거부(09-28 저녁과 동일). 스크립트 `scratchpad/archive/stage2_finalize.sh`(K 계획 COMMANDS §3 그대로, 대상 10개).
- 남은 정리 후보(결정 대기): 옛 보관분 브랜치 4개(`jaehyeond/{research-survey,w12-input-audit,w12-isaac-replay,w13-full-cycle}` — 태그로 보존됨) · `scoop_v0/contract-wave` 브랜치 · `w23-sim-reference` 잔류 Codex 터미널은 종료함.

## 6. 랩미팅 자료 정정(피드백 반영)
- 슬라이드(https://claude.ai/artifact/WGdzNzRDqr2CsawRucz7K2, v4→v5): `fullcycle` 그림 캡션 경고 · `fullcycle_views` 경고 문단 · **새 슬라이드 `gripper_note`**(A 셸 vs B CAD 그림, 원인·조치) · `mismatch` 표 "상자 방향" 정정(시뮬 = 22 cm 벽이 로봇 쪽) + "그랩 표시" 행 · `plan` 시뮬 항목(rev34 재실행). 11장.
- `lab_pc_bundle/LAB_PC_UPDATE_20260928.md` §1·§3·§4 정정, `figures/08_gripper_shell_vs_cad.png` 추가, 새 zip `lab_pc_bundle_20260928b.zip`(구 zip 은 그대로 둠). `OUTLINE.md` 3장에 정정 줄.

## 7. 사용자 추가 입력(09-28 밤 2차, HARD RULE #18)과 메인 분석
- **실물 층(사용자 실측)**: 현재 상자(임시 A4 종이 상자, 윗단 바깥 31×22 cm·벽 0.2) 안에 펠릿이 **벽면까지 꽉 찬 상태**, 부분 높이 차는 있으나 **상자 바닥→펠릿 윗면 ≈ 4 cm**. → 재실행 더미 = 310×220(기존 NPZ 규약, 안쪽 306×216 과 4 mm 차·선언값) × 40 mm 평평한 층. 메인 산술: 0.456 g/cm³ 61,408알(1,244 g) / 0.55 74,067알(1,500 g)(W23 F 표 61,462/74,067 과 일치). 안쪽 306×216 이면 59,514/71,782.
- **전체 사이클 비용(40 mm, 310×220)**: 세 모형 **15.0~26.4 h · 11~20 $**(20k 기준 20,977.836 s, D493 고정부담 a=6,403 / 비례 / 지수 1.154). cap 은 새 revision 에서 사전 결정(≥ 비관×1.2 = 32 h 제안, 비용 상한 25 $ 제안).
- **흘림(spill) 원인 — 원자료 재계산**: 충돌 셸 립 기하로 문 각도별 입구 이음새(고정 립↔문 립 최소 거리): 0.8° → 1.65 mm · 1.2° → 2.47 · 2.88° → 5.92 · **3.054°(W19 A 재닫기) → 6.28** · 3.549°(W13) → 7.30(W18 실측 7.30 일치) · 8° → 16.4 mm. 렌즈 알 두께 2.5·폭 3.8·길이 4.5 mm. → 시뮬 문은 서보 정지 모델(1.96 N·m×0.9)에서 끼인 알 때문에 **≈3° 에서 멈춰 이음새 6~7 mm > 알 크기** → 들어올림·운반 첫 1 s 에 입구로 빠져나감(W18: 144개 중 136개가 `post_lift_travel`, 129개 입구 이음새, 가속도 상관 −0.016). W19 A: 재닫기 후 공구 내부 292 → 운반 종료 153(용기 18·유출 123), 최종 spill 124. **실물은 채터링(8° 열었다 재닫기 ≤3회)로 관절 ≈0.8° 까지 닫혀 이음새 1.65 mm < 알 두께 → 유출 0~1알**. 즉 원인 = (1) 시뮬에 채터링 절차 없음 (2) 표면 10 mm 위에서 문을 연 채 시작 (3) 물성 임시값(E 5 MPa 무른 알이 립에 끼임). rev34 는 (1)(2)를 params 스위치로 넣는다(A 워커). (3)은 별도 case.
- **dt 근거(교수 질문 대비)**: 생산 dt = 1 µs(W11·W13·W19), 2 µs(W10). 10 µs 는 폭주(W15). 근거 사슬: ① 안정성 — 관측된 가장 뻣뻣한 접촉 무리(문 립–알 3개–고정 립, 구-구 16~17쌍) 감쇠 명시적 적분 한계 **7.5~8.1 µs**(W24 L, 증폭률 10 µs 1.50/1.73 >1, 1·2 µs <1) → 안전계수: 5 µs = 1.5×, 2 µs = 3.8×, 1 µs = 7.5×. ② 일반 관행(출처 미확인) — 레일리 시간 85.4 µs(메인 재계산 일치)의 10~20 % = 8.5~17 µs, 헤르츠 접촉 시간(우리 임시 물성, v 0.1~2.5 m/s) 173~330 µs 의 1/20~1/50 → 둘 다 10 µs 근처를 상한으로 주지만 ①이 더 엄격하고 우리 데이터다. ③ 비용 — 물리 1초당 1 µs 866 s·2 µs 595 s·10 µs 802 s(탐색 여유 ∝ dt 로 큰 dt 가 오히려 느림) → 5 µs 비용은 **미측정**. ④ 수렴 — 1 vs 2 µs 포획 517 vs 541 은 실행 간 변동(517/517/489, ±5 %) 안이라 n=1 로는 구별 불가(D490). **"5 µs 왜 안 했나"의 정직한 답 = 안 했고, 한계의 1.5× 안쪽이라 안정할 수 있으나 검사 없이는 채택 불가 → 같은 입력 반복 3회로 {2, 5} µs 사다리를 사전 등록해 측정한다**(E 워커가 사전등록·비용 작성).
- **RunPod 사용 계획(단계, 승인 뒤)**: P0 로컬 GPU 8분 — post05 로 W19 A 재렌더(CAD 그리퍼). P1 RunPod dt 사다리 — 스쿱 셀 {2 µs ×2, 5 µs ×3}(1 µs 는 n=3 기존), 셀당 pod ≈1,220 s(2 µs, 0.69× 실측비)·5 µs 미측정(상한 1,772 s) → 총 ≤ 2.2 GPU-h ≤ 1.7 $, pod 2대면 벽시계 ≈1 h → 회의 전 결과 가능. P2 RunPod 전체 사이클 rev34 — 40 mm 층 61~74k 알, 15~26 h·11~20 $, 선행: 더미 생성(로컬 GPU ≈1~2 h, D 워커 패치 필요: 시드 여백 13.5 mm 제거) + 격자 증거(fail-closed 가능성 D 워커 확인) + cap/비용 결정. 회의 전 완주 불가(사실대로).
### E `w25-dt-basis`(codex gpt-6-astra, 완료·release) — `…/w25-dt-basis/claudedocs/research/w25_dt_basis_20260928/{DT_BASIS_MEMO,PREREG_dt_ladder,REPORT}.md`, `dt_cost_plan.json`
- 교수용 메모(2쪽): 안정 한계 7.483~8.130 µs(W24 저장 접촉망 선형화) → 여유 1 µs 7.5~8.1× · 2 µs 3.7~4.1× · **5 µs 1.50~1.63×**; 독립 구–구 헤르츠 접촉 시간 328/238/207 µs(v 0.1/0.5/1 m/s), 레일리 85.15 µs(일반 관행 10~20 % = 8.5~17 µs, 출처 미확인 표기); 1 µs 반복 517/517/489 → 평균 507.67·SD 16.17·2 SD 32.3알이라 W10 2 µs 와의 24알 차(4.64 %)는 변동 안 → n=1 로 dt 효과 판정 불가. 비용: 1 µs 866 · 2 µs 595 · 10 µs 802 s/물리초(10 µs 는 프로세스 전체 시간이라 동종 아님).
- 사전 등록: 같은 입력(W11 params·20k NPZ·seed 460·`sim_deme_scoop_s1.py` sha `2e40f7ed…`)에서 2 µs ×2 + 5 µs ×3(선택 3 µs ×3), 판정 = 2·s_pooled 선별(포획 수·질량·문 각/정지 사유·최대 속도·5 m/s 초과·발산·높이지도 RMSE), 발산 1회 = 그 dt REJECT, cap = 비관×2(2 µs 2,454 s·5 µs 3,561 s), 통과해도 `SCREEN_PASS_NOT_PRODUCTION_APPROVAL`.
- 비용(0.74 $/h, 준비 1,800 s·회수 900 s/pod 가정): 필수 5셀 **1대 1.83~2.91 h·1.35~2.16 $ / 2대 1.36~2.08 h·2.01~3.08 $**(cap 시 3.76/5.05 $). 5 µs 셀 시간은 미측정(낙관 2 µs×2/5, 비관 = 1 µs pod 시간).
- **메인 검산**: `dt_cost.py --check` PASS(JSON 정확 재생성·입력 sha 6/6) · 메인 독립 산술 헤르츠 t_c 330/239/208 µs·레일리 85.4 µs(반경 1.16 vs 1.1564 mm 차)·안전계수 일치 · 5셀 비용 산술 일치.
### D `w25-pile-flat40`(claude-opus-5-5, 완료·release) — `…/w25-pile-flat40/claudedocs/research/w25_pile_flat40_20260928/REPORT.md`
- 생성기 사본 `--seed-margin-mm`(기본 None = 기존 13.5 mm 와 **바이트 동일** 12/12, AST 변경 6정의). **여백 0~2 mm 는 생성기 자체 검사(상자 ≥ 봉투 + 2×지름, `sim_deme_pile.py:489-492`)에 걸려 불가능**, 4.55 mm 부터 통과 → **4.6 mm** 채택 시 시드 알 표면–벽 간격 0.15~2.7 mm(기존 9~11.7 mm) = "벽까지 꽉 찬" 층에 근접.
- 알 수(40 mm, 벽까지): 310×220 **61,408(0.456)/74,067(0.55)**, 306×216 59,514/71,782, NTC106 53,663/64,725 — 메인·W23 F·W25-C 산술 일치. 시드 검사 6/6 통과(초기 관통 0). 생성 시간 2,491~4,791 s 추정(cap 6,660~9,600 s), VRAM 미확인.
- 정착 관문(40±5 mm 중앙값·p5/p95·벽 칸 커버리지 ≥0.95): 기존 20k 둔덕 → **FLAT40_FAIL**(중앙값 7.9 mm, 벽 2.9 mm, 커버리지 0) = 음성 대조 정상, 합성 대조 15/15.
- **격자 증거**: 양성 대조 2(W21 ROI-300 stdout 바이트 동일·rev34 TEMP binary64 동일)·음성 대조 1(W23 F 거부문 동일) PASS → rev34 후보 도메인 5개 **전부 인증 분기**(경계 15 cm 이상 이동해야 이탈) → fail-closed 위험 낮음(실제 증거는 새 NPZ 로 재작성).
- NPZ ±90° 회전 유틸 + 테스트 PASS(rev34 가 원할 때만).
- **메인 검산**: 알 수 6행 일치 · 생성기 `:489-492` 검사문 직접 확인 · default_identity pass 확인.

## 8. 사용자 3차 지시(23:2x)와 실행
- **worktree rm 권한 부여** → 2단계 실행 완료(23:33~23:36): 10개 전부 `ARCHIVE_VERIFIED`(최종 대조 cmp 일치·태그 HEAD 일치·bundle verify·`orca_remove ok`·원경로 심링크·심링크 경유 재대조). `ARCHIVE_INDEX.md` 10행 append. 남은 실제 worktree = 메인 + pellet-model·w13-cycle-audit·w19-postprocess·w19-replay·w20-rev33b·w22-×4 + **w25-×5**(이번 워커) = 15(전부 사용자 retained/미merge 산출 보유).
- **RunPod 관리 지시**: 남의 pod 안 건드림·우리 것만 생성/종료·완료 시 회수 후 종료·대기 시 과금 방지·MCP 로 관리. **읽기 전용 확인(23:3x, `list-pods`·`list-billing`)**: 계정(랩)에 pod 7개 — RUNNING 1(`t41m6mb79u45h6` "[Jihoon]-migration", RTX PRO 6000, 2.09 $/h, 타인) + EXITED 6(DK-planm ×3·[Jihoon]·Geena ×2, 타인). **우리 pod 0**(W19 pod 3개는 9/17 종료 확인됨). 최근 14일 랩 과금 295.57 $(GPU 282.3·디스크 5.9·스토리지 7.0), 09-28 9.97 $. → 규칙: 우리 pod 이름은 `roarm_w25_*` 로만, 조작 대상은 이 세션이 만든 pod ID 만, 완료 즉시 회수·terminate, 대기가 필요하면 **stop**(GPU 과금 0, 디스크만) 후 재개, 30 GB persistent 는 쓰지 않거나 종료 시 함께 삭제.
- **dt 사다리 GPU 실행 보류**(사용자: CPU 근거로 충분, 결과 검토 뒤 브리핑) → P1 은 문서(E)로 종결, 필요 시 별도 승인.
### A `w25-rev34-fullcycle`(claude-opus-5-5, 완료 → 후속 dispatch 재사용) — `…/w25-rev34-fullcycle/claudedocs/research/w25_rev34_20260928/REPORT.md`, rev34 `…/runtime_logs/grasp_track/w25_realign_fullcycle_d498/rev34/`
- rev32 바이트 사본(34/34 sha) → **rev34**: 6개 파일 패치(`w13_fk.py` 어댑터 `R_robot_box`·`sim_w13_full_cycle.py` run()·preflight 3·verify), **18개 바이트 동일**(메인 cmp 재확인), `criteria.json` 동일(메인 cmp), AST 범위 PASS(run() 내부 변경 = `door_move`·`pose_of` 뿐).
- **OFF = rev32 증명**: 러너 자체를 CPU 운동학 스텁으로 돌려 NPZ 96키 중 92 바이트 동일(경과시간·메타 추가 키 제외), 정지 주입 시나리오까지 IDENTICAL(19,526/17,526 sync).
- **ON(규약 A)**: `R_robot_box = [[0,1,0],[−1,0,0],[0,0,1]]`, `t_robot = (0.25, 0, −0.2552)`, 상자 발자국 로봇 x×y = 198×301(긴 변 좌우 ✓), 앞벽 0.151 m, 취점 립 (8.10, 0.006) mm(옆 오프셋), 컵 = 로봇 (8.1, 250) mm. JOINT_LIMITS 위반 0(840+240 자세), 트레이 최소 54.3 mm·컵 26.1 mm, 취점 기둥 IK 오차 0.046 mm(rev32 0.889). 절차 스위치: 표면에서 열기·plunge 25·채터링(서보 >3.6° → 8° 열고 재닫기 ≤3회, 스텁 주입 3.2→5.5→2.0→5.5→0.5°)·+80 mm 상승 후 항상 재닫기 — 실물 `hw_s1_manual.py:157-169` 순서 재현(그림 fig2 육안 확인). 단위시험 9/9(규약 A 축·Rz(90+θ) 등변성·트레이 fail-closed·채터링).
- 음성 대조 3건 정상 거부(선언 트레이≠더미 발자국 / 도메인≠증거 / 증거 없음 → CLEARANCE_UNCERTIFIED). 수치 증거 생성기 양성 대조 7/7. CAD 변환 W19 A sync0 재현 ≤8.9e-5 mm(B 와 독립 일치).
- **FAIL·미해결(그대로 보고)**: C4 취점 재현 0.112 mm > 0.1 mm(rev32 `r0 = hypot` 편향 상속, r 0.25 에서 커짐 — 허용치/식 결정 대기) · C13 TEMP 더미 310×220 ≠ 선언 301×198(평평한 층 NPZ 로 재실행 필요; **사용자 3차 지시로 상자 = 현재 종이 상자 310×220 → 후속 A2 에서 params 변형**) · `verify_w13_self --check` 가 rev32 부터 자기 산출에서 CRASH(`bridge_precheck_columns` 키 불일치, 범위 밖) · Isaac 재생은 규약 A 미패치(BLOCKED 표기, B post05 와 합쳐야) · 열기 토크 200 미반영 · 어깨축 CAD 123.06 vs FK 체인 122.06 mm 1 mm 차(미해결).
- **메인 검산**: 18/6 파일 cmp·criteria cmp·equality JSON(96키)·preflight C10/C11/C13 수치·fig1/fig2 육안(상자 긴 변 로봇 y, 채터링 순서) — 일치. Variable Ladder: 이번 재실행은 [배치·방향, 절차] 2묶음 + 더미(4 cm 층) = **3변수** → 사용자 명시 요청("실물 조건으로 다시")으로 case 변경 승인된 것으로 기록, 스위치별 A/B 가능.
### A 후속(같은 터미널, 종이 상자 변형, 완료·release) — REPORT §11
- `params_w25_paperbox.json`(310×220·벽 230 mm·두께 2 mm·펠릿면 23.9 cm = 받침 19.7 + 바닥 0.2 + 층 4.0, 규약 A·0.25 m·절차 ON) + `…_inner306.json`. rev34 `w13_fk.w25_tray_bounds` 2건 수정(선언 문자열 수용·벽 접촉 tol 0.5 mm — 정착 더미가 벽에 µm 급으로 닿아 필요, 음성 대조 2 mm 차는 여전히 거부), 단위시험 9/9.
- TEMP 20k 더미로 CPU 사전검토: **JOINT_LIMITS 위반 0**, 툴 셸–벽 최소 **33.9/33.7 mm**, 컵 26.1 mm, 팔 링크 OBB–벽 겹침 0(최소 56.3 mm), S1 전체 STL–벽 37.4 mm, 도메인 x[−0.353, 0.217] y[−0.345, 0.208] z[0, 0.561] 격자 인증 분기, rigid_hinge PASS. **벽 윗단 42.78 cm(바닥판보다 9.58 cm 위), 운반 높이(립 45 cm)가 벽 윗단보다 22 mm 위뿐** → 실물 처짐(~2 cm 주석) 고려 시 `travel_cm` 상향 결정 필요. `all_pass=false` = C4 0.135 mm(rev32 r0 상속) + inner306 의 C13(TEMP 더미 불일치, 설계된 거부).
- 메인 검산: preflight JSON 수치 재확인(C10 33.886/33.653, C11 [], 상자 로봇 xy 220×310, 벽 42.78 cm, 펠릿 23.9 cm) 일치. 워커 5개 전부 release, reclaimable 0.

## 9. 세션 종료 정리
- 원장: D498 append(백업·md5), DECISIONS_ACTIVE·LEDGER_RECENT 갱신, EXPERIMENT_LEDGER 1행, START_HERE overwrite(백업 `.bak_20260928_pre_w25`), relay overwrite(백업), MEMORY.md prepend(회전 archive `MEMORY_archive_20260928.md`).
- 승인 대기(사용자): ① 로컬 GPU post05 재렌더(8분) ② 40 mm 층 더미 생성(로컬 GPU ~1 h) ③ RunPod rev34 전체 사이클(알 수·cap·비용·기동 시각) ④ C4 처리·travel_cm 상향·규약 A 부호·상자 y=0 ⑤ 옛 브랜치 4개 삭제 여부.
