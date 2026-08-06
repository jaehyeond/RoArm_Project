# t3_prereg.md — G0b T3 attempt1 사전등록 / 해시 / attestation (③)

발행: 2026-08-06 (22nd 세션). Case `g0b_d420`. 승인 근거: 사용자 "T2/T3 진행 승인"(19th 수령).
본 문서 발행 후 p9 소스·기본 인자 무변경 실행만 유효하다. 변경 시 재발행 필수.

## 1. 질문 (실패 가능 실험)

D419 고정 파지 방식(수직 상부 접근, 상면 중심)으로 D29×H50 기립 원통(24.83 g)을
PhysX 물리에서 APPROACH→DESCEND→LATCH(닫힘 스윕)→HOLD→LIFT(+10 mm) 체인으로 잡아
들어올릴 수 있는가. 성공 정의 = `G0B_T3_GRASP_VERDICT=GRASP_PASS`
(모든 phase 게이트 통과 ∧ LIFT 물체 추종 ≥ 6 mm ∧ kinematic attach/posewrite 0).
실패 시 해당 `*_FAIL` phase가 다음 수리 지점이다 — **재시도 면허가 아니다.**

## 2. 실행 tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp_stderr.log
```

- 인자 없음 = argparse 기본값 전체가 아래 §5 사전등록 값이다 (tag `t3_grasp`).
- stdout/stderr 분리 (2>&1 금지 규칙). exit code: 0=GRASP_PASS∧rerun 검증 PASS /
  2=과학 FAIL 또는 rerun 검증 FAIL / 3=가드 abort(산출물 선존재·USD 가드·버전 불일치).
- env pins (실행 직전 재확인 완료): isaaclab env — rerun-sdk 0.34.1 / numpy 1.26.0 /
  psutil 5.9.8 (D326). GPU RTX 4090 Laptop (사용 1.3/16.4 GB), 디스크 여유 56 G.

## 3. 대상 스크립트 핀

- `sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py`
- **sha256 = `939a5bd0639332afc2572bee8cd7a7e735ad2ed7080193d4f605114cc1e2f5fb`**
  (1,747줄, py_compile OK, isaaclab env python)
- 잠정 sha `1e7f1907…fa31`(21st)은 재검증 생존 이슈로 폐기됨(D423-R1 ①). 본 sha는
  §8 검증 계보 전체(감사→수리→재검증→라운드-2 수리→라운드-3 재검증→기계 수리 3건)
  가 선행 완료된 리비전이다.

## 4. 스폰 / 자산 / 물성

| 항목 | 값 | 근거 |
|---|---|---|
| 물체 | 원통 D29×H50 mm, axis Z 기립 | HARD RULE #18, D419 |
| 질량 | 0.02483 kg (50 g 분동 교정 실측) | D420 ③ |
| 마찰 주 leg | μs 0.40 / μd 0.30 / rest 0.0 — **미실측 사전등록 가정** | `t3_mass_friction_contract.md` §주 leg (sha `31750694…7ce05`). D362 1.5/1.2 전이 금지 |
| 감도 leg | leg-low 0.25/0.19 · leg-high 0.60/0.45 — **주 leg 결과 수령 후 별도 tag 별도 실행만** | 동 계약 |
| 스폰 위치 | `--pose_label seed0_S1` (+0.213696, −0.195719) | **supersession S-1**: 설계 D-4의 (0.300, 0)은 T2 annulus 경계(r≈0.30)라 폐기 → T2/T2b PASS 후보(D421/D422; S1 descend tilt 0.199°/0.366°) 중 권고 S1 |
| 로봇 충돌체 | 동결 attempt3 64+64 USD — root `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff` / physics `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`, import 전 env-var 주입 + 스테이지 64+64 감사 hard-fail | **supersession S-2**: 설계 D-6의 local_assets(convex_hull 1개/링크)는 조 목구멍 폐색이라 폐기 → attempt3 재사용(D420-R1, 재분해 금지 D415 ③) |
| episode | 60 s = 6000 control steps | **supersession S-3**: 설계 D-8 "10→20 s"는 최악 예산(실측 3435~3735) 미달 — 감사 MAJOR-c로 60 s 승격 |

## 5. 게이트/인자 전수 (argparse 기본값 = 사전등록 값)

close 스윕 88.31→24.0° 11각(D-1 역전 규약: 큼=열림, 마지막 각 24° > env 해제 임계
22.92°) / marker = dist<0.030 m ∧ q5≤41.40°(D-2 패치 — LATCH 게이트 전용, 물리 증거는
LIFT follow) / target_error 3 mm / plan tilt 5° / max_tcp_step 10 mm /
**transit_tcp_step 20 mm(approach phase 한정, §9-i)** / drift 6 mm / speed 0.08
(lift 0.25) m/s / tilt 12° / upright 0.95 / lift_follow 6 mm / stall 0.02°/step×5 /
settle 2 / substep 60 / close 45 step/각 / hold 30 / initial settle 30 /
**waypoint 관절 trust region 12° + transit 폴리시 2°** / **transit 선택 밴드 0.5 mm** /
continue_close 기본 ON / episode 60 s / tag `t3_grasp` / log_every 10 /
resample fraction 0.80(=명령 간격 8 mm).
전체 수치는 실행 시 results JSON `gates` 섹션이 자체 기록한다(재현성 계약,
`path_ok` 3종 포함).

## 6. 산출물 (전부 `claudedocs/runtime_logs/grasp_track/g0b_d420/`)

`t3_grasp_timeline.rrd` / `t3_grasp_timeline.rbl` / `t3_grasp_inspection.png` /
`t3_grasp_rerun_validation.json` / `t3_grasp_results.json` / `t3_grasp_steps.csv`
+ `t3_grasp_stdout.log` / `t3_grasp_stderr.log`(셸 리다이렉션).
어느 하나라도 선존재 시 Isaac 기동 전 abort(감사 MAJOR-d 가드) — 실행 직전 확인
완료(t3_grasp_* 0건). **T2/T2b 산출물·기존 증거 덮어쓰기 금지 유지.**

## 7. D341 Rerun 계약

save-only RecordingStream(app_id `roarm_g0b_t3_grasp` / recording_id
`g0b_d420_t3_grasp` — tag 파생), 전체 실행 step 타임라인(physics_step/sim_time_s)
+ TCP/물체 위치·쿼터니언·q5 actual/command·게이트 스칼라·marker + verdict +
고정 blueprint 임베드 + `.rbl` export + footer `rrd verify` + exact 엔티티 15종/
타임라인 4종/컴포넌트 계약(validate_rerun_artifact, CLI 0.34.1 핀) + 2400×1400
헤드리스 PNG. **완료 조건은 산출이 아니라 육안검수까지다** — 관찰 기록을 세션 doc에
남기기 전 "inspected" 보고 금지.
**접촉력 화살표 생략 정당화**: 어떤 게이트도 접촉력을 소비하지 않으며(판정 권위 =
LIFT follow + drift/tilt/upright), 접촉 벡터 시각화는 본 probe 계약 외 — D341 이탈
항목으로 사전 고지한다. **[사용자 확인 플래그]** 접촉 화살표가 필요하면 실행 전에
지시할 것(그 경우 p9 개정 + 본 문서 재발행).

## 8. 검증 계보 (본 sha에 선행 완료)

1. **6렌즈 적대감사 `wf_78b1adfd-20d`** (20th): confirmed 10(FATAL 2)·refuted 0 →
   전건 수리(21st doc §3). 전문 = `p9_audit_wf_78b1adfd_findings_raw.json`
   (sha `c3f606e2fc283efb3295db8568996e9fa597010c259ad02137b570b51bf915c1`).
2. **수리판 재검증 `wf_3cea04db-7c2`** (10 agents, 21st 발사 → 22nd 회수 완료):
   수리 유효 전건 인정·notFixed 0; **신규 MAJOR 2종**(transit min-tilt 랭킹 칼끝 —
   D423-R1) + MINOR 다수. 전문 = `p9_reverify_wf_3cea04db_findings_raw.json`
   (sha `ca348e58d275800efe611de2e7b4557febee2cd440b66f2db36783029a3a95ec`).
3. **라운드-2 수리** (22nd): transit 선택 밴드 0.5 mm + bias-free 폴리시(≤2°,
   비악화 가드) / 관절 trust region 12° / transit_tcp_step 게이트 20 mm(approach
   한정) / `_close_all` 핸들별 try + 선클리어 / tag 파생 recording 식별자 / verdict
   lift_path_ok 소비 / aggregate lift_follow NaN-sanitize / 솔버 밴드 = min(3.0,
   gate) / docstring 3건(D-8 60 s supersession·미래형 provenance·transit 서술).
   설계 근거 = `t3_trust_region_dev_sweep_evidence.md`(sha `3a128e76…3c957`):
   잔차 1.2~1.5 mm는 trust region 크기 무관(12~24° 동일) → 폴리시 필요 실증.
4. **사전비행**: v1 PASS(sha `471a6d0a024de0adba25c0fa5640b0ef848374cf83957ff1662df5aeac386a07`,
   21st §4) + **v2 PASS**(`t3_preflight_ik_chain_v2.py`, sha
   `f44c5a45c9a90c903d1ca8b97fdd146bf9bdc1cd1c30bbd83eb1c6e4abed4c16`):
   4포즈 × (REACH + approach/descend/lift 체인) — worst 명령 pe 0.468~0.492 mm
   (게이트 0.6) / 관절 dev ≤12.05°(≤14 허용) / 비관 슬루 approach ≤9.59 mm(마진
   게이트 14, 하드 20)·회랑 ≤8.07 mm(마진 게이트 9, 하드 10) / 도착 tilt ≤0.213° /
   예산 3435~3735 < 6000 / planned 높이 REACH PASS.
5. **라운드-2 수리판 적대 재검증 `wf_9b819983-97c`** (22nd, 9 agents: R1~R7 수리
   반박 + 런타임/사전비행-정합 회귀 2렌즈, 에러 0): **수리 7/7 유효·notFixed 0·
   회귀 렌즈 발견 0건.** 하이라이트 — R1: 폴리시 채택 22~25/40~45 transit wp,
   tilt 악화 최대 +0.261°(수직화 후퇴 없음), **plan 15해가 라운드-1과 비트 동일
   (T2/D421 REACH 계약 불변)**; R2: property test 400회 joint-limit/dev 위반 0;
   R3: 명령 경로가 원통 풋프린트 진입 시점 z=상면+55.7 mm→도착 +40.0 mm로 접촉
   불가 논증. 잔존 = **MINOR 3**(전문 = `p9_reverify2_wf_9b819983_findings_raw.json`,
   sha `c09665462e9939317577ebbc074d1b6caeb7f108d1ad13557ff3b8aef570b641`).
6. **라운드-3 기계 수리 3건** (22nd, 동작 무변경 — 직렬화/문서 수치만):
   ① results JSON에 `path_ok`(approach/descend/lift) 직렬화(JSON 단독 verdict
   재계산 복원) ② 예산 주석/docstring 3675 → 실측 최악 3735 정정 ③ gates JSON에
   `transit_polish_dev_deg` 수록 + summary "12+2deg" 표기.
   **4차 전면 재검증 생략 정당화**: 세 수정 모두 제어 흐름·게이트·IK 무접촉
   (diff 라인 자체검증), py_compile OK + 사전비행 v2 재실행 PASS(수치 동일)로
   행동 동일성 확인. D423 ①의 재검증 요구는 행동 변경 수리에 대한 것으로 해석 —
   이 해석 자체를 본 조항으로 사전 고지한다.

## 9. p7-parity 의도적 델타 (전건 사전 고지)

라운드-1 (감사 wf_78b1adfd 수리 계열):
- (a) approach 도착점 + descend/lift 회랑 waypoint IK = 수직 제약 DLS(q4=0) —
  p7 위치 전용 ik_dls는 수직 회랑 재사용 금지(D422); transit는 위치 전용 수용
  (FATAL-2 정정, vertical_scope="arrival").
- (b) descend_path_ok 소비 + descend 실패 = APPROACH_FAIL 승계 표기.
- (c) plan 3타깃 IK 전건 게이트(REACH_FAIL).
- (d) continue_close 플래그 기본 ON(p7 기본 False에서 의도적 상향 — 전 밴드 증거).
- (e) stall-aware reached(+`gripper_stalled` 기록).
- (f) grasped_seen phase-any 집계(p7 등가 의미론, 물리 증거는 LIFT follow 불변).

라운드-2 (재검증 wf_3cea04db 생존 이슈 수리 계열):
- (g) transit 선택 밴드 0.5 mm + bias-free 위치 폴리시(2단 솔브, 비악화 가드).
- (h) waypoint IK 관절 trust region 12°(+폴리시 2°) — 명령 연속성 계약.
- (i) TCP-step runaway 게이트 phase 스코핑: approach 20 mm / 그 외 10 mm.
  근거: trust region은 hop 길이만 줄이고 다관절 동시 슬루 순간 속도(|J·q̇|)는 못
  줄임 — 비관(속도클램프) 모델 피크 9.59~9.90 mm가 10 mm 게이트와 칼끝. approach
  중 물체는 ≥40 mm 이격(R3 기하 논증)·물체 보호 게이트(drift/speed/tilt/upright/
  done/nan)는 전부 불변. 현실(effort 2.5 N·m/damping 4) 평형 슬루는 ~0.5 rad/s로
  클램프의 ~1/6이라 실측 피크는 훨씬 낮을 것으로 예상 — 20 mm는 모델 불확실성
  보험이다.
- (j) verdict가 lift path_ok 소비(IK-절단 lift의 GRASP_PASS 차단) + `path_ok`
  3종 JSON 직렬화.
- (k) app_id/recording_id tag 파생(기본 tag에서 기존 문자열과 동일).

## 10. 잔존 MINOR 처분 (코드 무변경 — 사전 고지)

| 항목 (출처) | 처분 근거 |
|---|---|
| stall 채터/저속 crawl 보수 창 — 최종 각 한정 false-FAIL 가능 (wf_3cea04db) | 방향 보수적(false-FAIL). close_records `gripper_stalled/reached` + CSV q5 궤적으로 즉시 진단 가능. HARD RULE #3 유형 재판독 대상 |
| latch 세그먼트-상대 drift 기준 — 수평 슬라이드 누적 사각 (wf_3cea04db) | 판정 권위는 LIFT follow(phase-누적). per-angle close_records drift로 사후 재구성 가능. 오귀속(LATCH→LIFT_FAIL) 방향만 존재 — verdict 부풀림 없음 |
| nan_seen run의 phases/close_records NaN 토큰 (wf_3cea04db) | aggregate는 sanitize 완료. nan_seen=true run은 증거 사용 금지(폐기 표지). Python json.load는 파싱 가능 |

## 11. 층위 선언

본 시행은 **sim 물리 probe verdict**다. 실물 파지력·서보 폐지력·자율 재현성 주장이
아니다("T1이 파지력 증명" 표현 금지 유지). `g0a_pass=false` 불변. GRASP_PASS여도
실물 권위 라벨은 T4 실물 재현이 준다(D419 라벨 사다리 — sim은 발견 도구, 라벨은
실물 시행).

---

## 부록 A — attempt2 leg (`t3_grasp2`) 사전등록 (attempt1 결과 수령 후 발행)

### A1. attempt1 결과 요약 (본 leg의 유일한 설계 입력)

attempt1 (§1~§11 tuple, sha `939a5bd0…f5fb`) = **`G0B_T3_GRASP_VERDICT=APPROACH_FAIL`**
(descend 실패의 사전등록 승계 표기, §9-b). 기전 실측: approach 44wp 완주(도착 vertical
PASS) → descend wp005(z=0.0572)까지 정상 → **wp006(목표 z=0.0505 = top+0.5mm)에서 sim
TCP가 z≈0.05440에 정지**(tcp_step→2e-6, 오차 3.917mm 포화, 60스텝 소진). 정지 시점부터
물체 speed 지터 20~30배(0.0005→0.016 m/s)·tilt 지터(0.002→0.044°) 급증, drift 26μm 유지
= **열린 그리퍼(88.31°) 구조물이 원통 상면(z=0.050)에 접촉, 수직 압입은 지면이 지지**.
→ 결론: **controller/IK 실패가 아니라 descend 목표 자체가 기하 위반** — 그리퍼 최저
구조물이 TCP frame 아래 ~4.4mm에 있어 도달 가능 최저 TCP는 top+4.4mm(압입 평형 기준).
증거 = t3_grasp_stdout.log(wp006 이벤트), t3_grasp_results.json, RRD/PNG 육안검수 완료.
**attempt1 산출물 8종은 동결** — 본 leg는 별도 tag로만 실행.

부수 관측 1건: verdict FAIL인데 프로세스 exit code 0 — `sim_app.close()`(Kit) 종료
동작이 `return 2` 도달 전 프로세스를 끝내는 것으로 추정. 증거 파일은 close 전 전부
기록되므로 무손상. **판정 권위는 exit code가 아니라 stdout verdict 라인 + results
JSON**(기존 §2 조항 유지, exit code는 참고 불가로 격하).

### A2. attempt2 질문

descend 목표를 실측 접촉 한계 위(top+5.5mm)로 재유도하면 (a) descend가 통과하는가,
(b) 그 높이의 close 스윕에서 marker fire + LIFT follow ≥ 6mm가 성립하는가. LATCH에서
물체 무접촉 실패 시 = **"조 팁이 상면 위에서 닫힘"(개방 폭/팁 깊이 문제) 판별 증거**로,
성공 시 = 상면 rim 핀치 성립 증거로 기록한다(T1 실물 rim 핀치 0~12mm와 대조).

### A3. attempt2 tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py \
  --grasp_surface_margin_m 0.0055 --marker_dist_m 0.035 --tag t3_grasp2 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp2_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp2_stderr.log
```

- **p9 sha(attempt2 리비전) = `1ef8a4117ec3aa2fe82f693cbc5ee9d4929d8ab7b7216afec47ca7a452b94a55`**
  (1,758줄). attempt1 sha 대비 델타 = `--marker_dist_m` CLI 인자화(기본 0.030 =
  attempt1 값 그대로 — 기본 경로 의미 불변) + 구조적 발화 불가 가드
  (`marker_dist_m > H/2 + margin` 강제; attempt1 기본값도 통과: 0.030>0.0255) +
  사용처 4곳(marker patch·게이트 로그·summary·gates JSON)을 인자로 연결.
  검증 = py_compile OK + 사용처 전수 grep + attempt2 수치 예행(아래).
- 파라미터 유도(전부 attempt1 실측 기반):
  - `grasp_surface_margin_m 0.0055`: 접촉 압입 평형 z=0.05440 → 목표 z=0.0555,
    여유 +1.1mm(자유 구간). 나머지 인자 전부 §5 기본값 유지.
  - `marker_dist_m 0.035`: descend에서 TCP-중심 거리 = H/2+margin = 0.0305 →
    D-2와 동일 논법(+4.5mm 헤드룸) = 0.035. q5 조건(≤41.40°) 불변.
- 수치 예행 PASS (isaaclab python, 본 리비전): REACH 3타깃 ok(descend tilt 0.609°),
  approach 44wp(worst pe 0.468mm)·descend 5wp·lift 2wp 전부 ok, 예산 3615<6000,
  marker 가드 0.035>0.0305 성립.

### A4. attempt2 게이트/판정/D341

§5~§7, §9~§11 전부 승계(변경 = margin·marker_dist·tag 3개뿐). 산출물 =
`t3_grasp2_*` 6종 + stdout/stderr 로그. `*_FAIL` = 다음 수리 지점(재시도 면허 아님).
D341 육안검수 의무 동일.

---

## 부록 B — attempt3 leg (`t3_grasp3`) 사전등록 (attempt2 결과 수령 후 발행)

### B1. attempt2 결과 요약 (본 leg의 설계 입력)

attempt2 (부록 A tuple, sha `1ef8a411…4a55`) = **`G0B_T3_GRASP_VERDICT=LIFT_FAIL`
(5/5 phase 완주)**. 기전 실측: descend 통과(margin 5.5mm 적중) → **close 스윕
88.31→24° 전 각도에서 접촉 증거 0** — 전 각도 `gripper_stalled=NO`·reached=YES,
drift 2~3μm, 속도/tilt 지터 없음, 즉 조가 24°(env 해제 임계 22.92° 직상)까지 닫히는
동안 원통 무접촉 → HOLD 통과 → LIFT에서 TCP 상승·물체 z=0.025 부동(follow −0.0mm).
marker는 39°부터 발화(거리+각도 휴리스틱 — 접촉 증거 아님이 실증됨; attach 시도
144회 전부 무력화 감시 하, posewrite 0). → 결론: **완전 개방(88.31°) 하강의 접촉
한계 깊이에서는 닫힘 면이 원통 상면 위를 지나간다** — attempt1의 "하강을 막는 최저
구조물"과 "닫힘 평면"이 같은 깊이대에 있어 상면 rim이 조 사이에 들어오지 못함.
attempt1/2를 합치면: 이 그리퍼의 top-down 파지는 **부분 개방 하강**(팁을 미리
내려 rim을 양옆으로 통과)이 필요하다는 판별이 완성된다 — T1 실물 rim 핀치(상면
0~12mm)와 정합. attempt2 산출물 8종 동결(tag-abort 가드 실증 완료).

### B2. attempt3 질문

부분 개방(45°) 하강이면 (a) 팁이 상면 rim을 간섭 없이 통과해 descend가 통과하는가
(팁 간극 반경 추정 ~18mm > r=14.5mm), (b) close 45→24° 스윕이 **접촉 stall**
(stall-aware reached, 라운드-2 수리)을 보이며 (c) LIFT follow ≥ 6mm가 성립하는가.
descend에서 팁-상면 간섭 stall 발생 시 = 팁 깊이 D_tip(45°) 실측 증거. close에서
24°까지 재차 무접촉 시 = "D29 첫 접촉각 < 24°" 증거(개방 폭 프로파일 재설계 지시).
어느 쪽도 판별 증거다.

### B3. attempt3 tuple (정확히 이대로 1회)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py \
  --descend_open_deg 45.0 \
  --close_deg 45.0 41.40 39.0 37.0 35.0 33.0 31.65 28.0 24.0 \
  --grasp_surface_margin_m 0.0055 --marker_dist_m 0.035 --tag t3_grasp3 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp3_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp3_stderr.log
```

- **p9 sha(attempt3 리비전) = `99c99c65da75d5b77fff5c777ebf6d5628c6cbf3cdd528b156ff461d79dc2412`**
  (1,780줄). attempt2 sha 대비 델타 = `--descend_open_deg` 인자화(기본 = 동결 OPEN
  88.31° — 기본 경로 의미 불변): approach/descend/초기 home의 q5 명령 + plan q5 +
  settle 로그 표기 + close_deg[0] 연속성 검사(D-1 검사를 "OPEN 고정"에서
  "descend_open 일치"로 일반화) + 가드 2종(≤ 동결 OPEN / > marker q5 게이트 41.40°).
  **D-1 부분 개정 고지**: q5 규약(큼=열림)·스윕 방향·d409 밴드는 불변, "하강 시
  개방값"만 인자화 — 근거는 attempt2 무접촉 실측.
- 검증(비례 원칙): py_compile OK + Q5_OPEN_DEG 사용처 전수 grep 후 명령 경로 5곳
  전환(상수 정의·close_deg 기본값·가드 비교만 잔존) + 스모크 3종(attempt3 인자
  수용→tag-abort 정상 / descend_open 40 거부 / 기본 인자 불변) PASS. arm IK는 q5
  무관(FK가 q0..q3만 소비)이라 사전비행 수치 불변 — 별도 재실행 생략을 고지.
  전면 적대 라운드는 생략(단일-값 파라미터화 + 가드, attempt1/2와 동일 harness) —
  이 판단 자체를 사전 고지한다.
- 파라미터 유도: descend_open 45° = d409 첫 접촉 밴드 상단(41.40°) + 3.6° 여유
  (marker 게이트 위, D29 예상 접촉각 ~35° 위), close 밴드 = 45 → 기존 밴드 그대로
  (60.0만 제외 — 45 위 각도는 불가). margin·marker_dist = 부록 A 유지.
- 산출물 = `t3_grasp3_*` 6종 + stdout/stderr. `*_FAIL` = 다음 수리 지점.
  D341 육안검수 의무 동일. §9~§11 승계.

---

## 부록 C — attempt4 leg (`t3_grasp4`) 사전등록 (attempt3 결과 수령 후 발행)

### C1. attempt3 결과 요약 (본 leg의 설계 입력)

attempt3 (부록 B tuple, p9 sha `99c99c65…2412` — 본 leg와 동일 리비전) =
**`G0B_T3_GRASP_VERDICT=LIFT_FAIL` (5/5 완주), attempt2와 동일 무접촉 시그니처**:
45→24° 전 각도 stall=NO·drift 2μm·지터 0, q5 실측 24.03°까지 추종, LIFT에서 물체
부동. → **판별 갱신**: 부분 개방(45°)으로도 닫힘 전 구간에서 파지면이 TCP−5.5mm
평면에 도달하지 못한다(도달 깊이 <5.5mm 또는 그 깊이 간극 >29mm). T1 실물 rim
핀치(팁이 상면 0~12mm 아래 물림)와 정합하려면 **TCP가 상면 아래로 내려가는 하강**이
필요하다. attempt1의 하강 한계(+4.4mm)가 (a) 개방각 의존(88.31° 팁/구조물)인지
(b) 축 근방 고정 구조물(개방 무관)인지가 미판별 — 본 leg가 가른다.

### C2. attempt4 질문

45° 개방으로 TCP를 top−7.5mm(z=0.0425)까지 하강 명령하면: (a) 팁이 rim을 양옆
통과해 하강이 통과하는가(→ close 스윕에서 원통 측벽 접촉 stall → LIFT follow 검사
— 성립 시 GRASP_PASS 경로), (b) 재차 접촉 stall로 하강이 멈추는가(→ 그 정지 z =
45° 개방의 하강 한계 실측; 한계가 +4.4mm와 같으면 축 근방 고정 구조물 확정 —
top-down 파지의 sim 기하 불가능성 증거로 사용자 에스컬레이션). 어느 쪽도 판별 증거.

### C3. attempt4 tuple (정확히 이대로 1회 — **코드 무변경, sha `99c99c65…2412` 동일**)

```
cd /home/cgxr/Documents/Robotics/RoArm_Project
/home/cgxr/miniconda3/envs/isaaclab/bin/python \
  sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py \
  --descend_open_deg 45.0 \
  --close_deg 45.0 41.40 39.0 37.0 35.0 33.0 31.65 28.0 24.0 \
  --grasp_surface_margin_m -0.0075 --marker_dist_m 0.035 --tag t3_grasp4 \
  > claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp4_stdout.log \
  2> claudedocs/runtime_logs/grasp_track/g0b_d420/t3_grasp4_stderr.log
```

- 파라미터 유도: margin −0.0075 → descend TCP z = 0.0425 = top−7.5mm — T1 실물
  rim 핀치 깊이 범위(0~12mm)의 중앙부. marker 검사 0.035 > 0.025+(−0.0075)=0.0175 ✓.
  descend 왕복 +8mm(waypoint +1) — 예산 여유 내. 하강 중 접촉 시 물체 보호 게이트
  (drift 6mm/speed 0.08/tilt 12°/upright 0.95) 전부 유효 — attempt1 실측상 수직
  압입 stall은 drift 26μm 수준으로 무해(지면 지지).
- 검증: 코드 무변경(전 인자 CLI) — 수치 예행은 plan REACH 게이트가 런타임 자체
  검증(REACH_FAIL 시 무실행 종료). D341 육안검수 의무 동일. §9~§11 승계.
