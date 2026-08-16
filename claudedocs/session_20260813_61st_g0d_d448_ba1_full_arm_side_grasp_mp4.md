# 61st — g0d_d448 `ba1`: B601 full-arm side 파지+리프트 물리 성공 + 전 과정 RTX mp4 (신규 case)

날짜: 2026-08-13 (60th 종료 직후 재개 세션)
이번 case의 신규 변수 (2개): `[팔 전체 포함 + base 세계 고정 (flying-gripper 제거)]`,
`[팔 궤적 제어 = 오프라인 수치 IK waypoint + min-jerk 보간 + PD 드라이브]`.
물리 실행 O (실패 가능 실험), Isaac Sim 5.1 로컬 4090 headless + RTX 렌더 O,
로봇 하드웨어 0, RunPod 0, lerobot-train 0, `g0a_*`/`g0b_*`/`g0c_*` 편집 0,
D427~D447 재판정 0, git 커밋 0.

## 0. 부트 검증 + 사용자 요청/승인

- 부트: HEAD == origin/master == `9cbd959` ✓, dirty 목록 기대치 정확 일치 ✓,
  `g0c_d446/` 산출물 완비 ✓, D445~D447 원문 확인 ✓.
- 사용자 요청 (verbatim): "아직 구매는 보류. 이거 지금 arm 전체로 해서 base에
  붙여놓고 sim화면으로 해서 물체 잡는걸 mp4영상으로 해서 sim rendering 못해?
  잡는 거 말이야. 어떻게 grap했는데? 이거 시각적으로 있어야할거 아니야."
- 답: bg1은 flying-gripper(팔 없음, headless)이고 `bg1_trace.npz`에는 힘 시계열·
  낙하 샘플만 있어(**매 스텝 3D 자세 기록 없음** — 직접 열어 확인) 기존 데이터
  재생으로는 영상 불가 → **팔 전체 신규 물리 실행 + 매 프레임 렌더**가 필요.
- AskUserQuestion 승인: **"승인 — side 파지 먼저 (권장)"** → 신규 case `g0d_d448`
  (태그 `ba1`) 개시. 이 승인 = D324 궤적 영상 금지의 명시 해제 (mp4 1건 + 키프레임).

## 1. 설계 (물리 0 — 해석적)

- 운동학 모델: `payloads/Physics/physics.usda`(SHA `131e9e66…`) 조인트 테이블 전사
  → numpy FK/IK (damped least-squares, Shepperd 쿼터니언 오차). round-trip 1e-10.
- 방위각×배치×standoff 스캔 → **φ=225°, C=(0.34, 0, 0.12), standoff 0.05 m**
  (j5 ±90° 한계로 standoff 0.10은 전 방위 도달 불가). waypoint 여유
  7.4°/10.4°/18.7°, 구간 최대 이동 11.3°.
- ★ 설계 정정 REV-1 (prereg §4에 기록): 초판은 (a) 접근축 부호 오류(bg1 §3-2의
  접근축=+x̂를 −x̂로 해석 → standoff가 "물체 너머") + (b) 팜 간섭 공식 오류
  (팜 최근접 = 0.073−|X_TCP| = **0.0485 m**인데 0.073+|X_TCP|로 계산). try-3
  SETTLE 스모크 게이트 abort(물체 2.86 m 발사)로 발각 → 부호 수정 + pedestal
  0.05×0.05×0.095로 축소 + 전 방위 재스캔. verdict 게이트(§5)는 무변경.

## 2. 실행 전 하네스 스모크 (scratchpad `ba1_smoke_probe.py` — 후보 pose 미사용)

최종 `SMOKE_ALL_PASS=True` (wall 9.7 s). 신규 실증 3건:

1. ★ **벤더 full 자산 결함 2호 — 중첩 rigid-body 계층 미시뮬**: 공식
   `reBot_B601_DM.usda`는 9개 중첩 RigidBodyAPI 링크에 xformstack reset 미저작 →
   `omni.physicsschema` "missing xformstack reset" 에러 9건 + 전 조인트
   "no bodies defined" → **로봇 전체가 시뮬레이션되지 않음** (bg1 R1과 동일 계열,
   D446 ③ 충돌 결함에 이은 두 번째 결함). 수리 = 각 링크의 합성 world 행렬을
   단일 transform op로 bake + `SetXformOpOrder([op], resetXformStack=True)`
   (bake 전후 world 편차 0.0). 수리 후 S3 FK-vs-USD **0.31 mm / R 0.00066** —
   내 FK 모델 = PhysX 실기하 일치 (IK 전제 실증).
2. ★ **PhysX 드라이브 정적 유지 오차 ≈ C·(damping/stiffness)**: 게인 스윕
   (1e4/500→j2 0.53° · 1e6/500→0.001° · 1e4/0→0.004° · 1e5/2000→0.19°) —
   τ/k가 아니라 d/k 비율이 지배하므로 **k·d 동시 비례 변경은 무효과**
   (2000/100→10000/500에서 오차 동일했던 미스터리의 해명). 채택 게인 = 팔
   **1e5/500** (유지 오차 0.023°, maxForce는 공식 27/27/27/7/7/7 N·m verbatim).
3. **이중 확장 enable 데드락**: `omni.replicator.core`와
   `isaacsim.replicator.grasping`을 사이 `app.update()` 펌프 없이 연속 enable →
   16분 futex 대기 침묵 (kill 후 p19 순서 + 펌프 3회로 해소). p20에 동일 적용.
- 기타 PASS: S2b 스텝 추종 15.000° 정확 / S4 손가락 전폐·재개방 / S5 지지 접촉력
  median 0.24358264 vs m·g 0.24358230 N (scene-int 스테핑에서 contact report 성립)
  / S6 `rep.orchestrator.step()` 캡처 = 물리 0 step·drift 0.

## 3. 시행 이력 (자진 신고 — try 1~3은 파지 판정 데이터 소비 0, bg1 재호출 전례)

| try | 결과 | 원인/조치 |
|---|---|---|
| 1 | preflight 후 asset 게이트 abort (wall 8.3 s, 물리 0) | 내 게이트 버그 — float32 저장 upperLimit 0.0715…333에 1e-9 허용오차 → 1e-6으로 수정 |
| 2 | 핑거 원본 충돌 비활성 저작 거부 (물리 0) | full 자산 지오메트리가 instanceable → 충돌 프림이 instance proxy(저작 금지) → instance root 2개 `SetInstanceable(False)` (합성 불변) |
| 3 | SETTLE 스모크 게이트 abort — 물체 drift **2863.9 mm**, calib NaN (30 step, 파지 phase 미진입) | §1 REV-1의 standoff 부호 + 팜 공식 오류 → 팜이 pedestal 위에서 스폰 겹침 → depenetration 발사. 기하 재설계 |
| 4 | ★ **성공** (wall 97.5 s) | 아래 §4 |

try 1~3의 failure.json/부분 keyframe 1장은 scratchpad에 보존
(`ba1_try{1,2,3}_failure_preserved.json`, `ba1_try3_key_settle_preserved.png`),
run 폴더는 최종 실행분만 (60th `bg1v_stdout.log` 교체 전례).

## 4. 본 실행 결과 — verdict `BA1_FULL_ARM_SIDE_GRASP_LIFT_SUCCESS` (D448)

phase: SETTLE 30 → APPROACH 120 → SETTLE2 30 → CLOSE 120 → LIFT 120 → HOLD 120
(dt 1/60, 총 540 step). 게이트/수치 (`ba1_results.json` `fbab9ff5a518a59a`, 18,605 B):

- SETTLE (스모크 겸용): 물체 drift **0.005 mm**, 팔 추종 오차 **0.0055°**,
  FK-vs-USD **0.321 mm**, 지지 접촉력 캘리브레이션 **0.24357746 vs m·g
  0.24358230 N** (4자리). step 카운트 540/540 정합.
- G-track: SETTLE2 종료 TCP(glf) 위치 오차 **0.43 mm** (게이트 10 mm).
- **G1 close bilateral = PASS**: CLOSE 중 같은 step 양측 최소력 peak **23.09 N**
  (L 24.46 / R 25.37 N), HOLD 종료 시점에도 양측 **23.47 N** 유지.
- **G2 lift follow = PASS**: glf z 상승 **79.56 mm**, 물체 z 상승 **79.69 mm**
  (게이트: glf−6 mm 이상 & glf ≥60 mm).
- **G3 hold slip = PASS**: 상대 슬립 **0.127 mm** (게이트 6 mm).
- 팔 링크의 의도치 않은 접촉: **0.0 N** (pedestal/바닥/물체와 핑거·팜 외 접촉 없음).
- measurement_valid **true**, 종료 핀 재검증 10/10 (b601_asset 9 + split2 USD).
- 충돌 표현: bg1 변형 B의 split2 4메쉬를 run 스테이지에 bit-copy 이식 — census
  핑거당 (2 enabled, 1 disabled), blade 내측 극값 vs `bg1_split2_audit.json` 핀
  dev < 1e-9 (실측 0 수준), 원본 1-hull disabled 보존.

## 5. 영상/시각 산출물 (사용자 요청의 답)

- **`ba1_side_grasp.mp4`** — 1920×1080, 30 fps, **270 프레임 = 9.00 s** (h264
  yuv420p, 1,415,244 B, sha16 `f5d0e6d38fba3fba`, ffprobe 검증). 내용 = 팔 전체가
  base에 고정된 상태로 접근→닫힘→들어올리기 전 과정 (물리 2 step당 1 프레임 =
  실시간 배속).
- 키프레임 6장 `ba1_key_{settle,approach,pregrasp,close,lift,hold}.png` (rt 32).
- ⚠️ **캡션 의무 (bg1v 규율 상속)**: 영상/그림 단독 제시 금지 — G1 23.09 N ·
  G2 물체 +79.7 mm 동반 상승 · G3 슬립 0.13 mm 수치 캡션 필수. 판정 권위는
  `ba1_results.json` + `ba1_trace.npz`.

## 6. D341 완주 + 육안 검수 (Claude 직접)

- `validate_rerun_artifact` **pass=True errors=[]** (rerun 0.34.1 핀, footer verify,
  exact entity 17종/timeline 3종/component 계약, blueprint+`.rbl`, headless PNG).
- 육안 관찰: 패널 4 — CLOSE 진입(~step 190)에서 양측 접촉력 ~23-25 N 대역 상승
  (포화 진동 후 안정 플래토), LIFT/HOLD 내내 유지. 패널 5 — **obj_z·glf_z 곡선
  완전 겹침으로 0.12→0.20 m 동반 상승** (판정 핵심). 패널 6 — 추종 오차 미세,
  개구 합 0.143→~0.0655 m.
- 키프레임 육안: settle(팔+개방 그리퍼+받침대 위 원통) / pregrasp(블레이드가
  원통 포위) / close(물림) / **hold(원통이 공중, 받침대 비어 있음)** — 4/4 판독.
- **한계/erratum 3건 (기록)**: (a) 3D 패널 기본 프레이밍 미흡(기존 bg1 한계 계열),
  (b) 토스트 가림, (c) ★ **RRD metadata 제목 문자열 "phi=135" 오기** — 실제
  φ=225 (스크립트 문자열 하드코드 잔재). 권위인 results.json의
  `plan.R_target/targets`와 모든 수치는 정확; RRD 내 게이트 수치도 정확.
- 관찰 노트 (결함 주장 아님): HOLD 시 개구 합 ~65.5 mm > D29 — blade hull의
  접촉 궤적이 내측 극값 평면보다 후퇴한 부위에서 성립 (bg1 동일 hull의 기지 특성,
  양측 23 N 유지·슬립 0.13 mm로 파지 유효).

## 7. 산출물 (전부 `claudedocs/runtime_logs/grasp_track/g0d_d448/`, forward-only)

`ba1_prereg.md`(REV-1 포함) / `ba1_results.json`(`fbab9ff5a518a59a`) /
`ba1_trace.npz`(`b7240b03156829fb`) / `ba1_timeline.rrd`(`0e8eda8911ef950a`) /
`ba1_timeline.rbl` / `ba1_rerun_validation.json` / `ba1_inspection.png` /
**`ba1_side_grasp.mp4`(`f5d0e6d38fba3fba`)** / `ba1_key_*.png` 6장 /
`ba1_script.py.txt` / `ba1_argv.txt` / `ba1_stdout.log` / `ba1_stderr.log` /
`ba1_exit_status.txt`(rc=0). 러너 = `sim_scripts/p20_g0d_ba1_b601_full_arm_side_grasp_probe.py`.

## 8. 비주장 (prereg §1 유지)

실물 B601 파지/제어 현실성(드라이브 게인 1e5/500은 하네스 저작 — 실모터 특성
아님), top-down full-arm 도달성(side 1 pose만), 다른 방위각/배치 일반화, 마찰
현실성, D446/D445 재판정. **sim 성공은 구매 판단의 필요조건 강화이지 실기 보장
아님.**

## 9. 순응 확인

- `g0b_d420`/`g0b_d444`/`g0c_d446` 편집 0. bg1/fg1 재실행 0. D427~D447 재판정 0.
- git 커밋 0 (사용자 지시 대기 — `.gitignore` whitelist가 g0c까지만: `g0d_d448`
  mp4/png/npz/rrd/json + p20 추가 필요).
- 로봇 하드웨어 0, lerobot-train 0, RunPod 0.
- stop-hook `/half-clone` 거부 시 HARD RULE #11 유지 (49회째 [가정]).

## 10. 다음 결정 경계 (전부 사용자)

1. B601 구매 품의 — 자료 세트 완성: fg1 0/13 vs bg1-B 13/13 + 정지 렌더 2장 +
   **full-arm 파지·리프트 mp4** (팔 층위 sim 필요조건까지 충족).
2. 교수님 보고 패키지 구성 (mp4 포함 여부).
3. 벤더(reBot-Isaacsim) 결함 upstream 제보 — 이제 2건: 충돌 1-hull(D446) +
   중첩 계층 미시뮬(D448).
4. top-down full-arm 시도 / 다른 방위·배치 확장 (신규 태그 ba2~).
5. RoArm 잔여 (fg2 폭-정지 / D≤20 물체 / rim 기움 컨펌) — 병행 가능.
6. git commit/push (whitelist 확장 동반).
