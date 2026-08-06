# 2026-08-05 (20th) — G0b: T2b 완주(T2B_PASS) + p9 저작 완료 + 적대감사 발사 (context 비상 종료)

이번 case의 신규 변수: [기존 ①②(19th) 범위 내 — T2b는 ①의 높이 provenance 부속, p9는 ②의 저작]

Case: `g0b_d420` 계속. 로봇 HW 0 · lerobot-train 0 · git commit/push 0 · **Isaac 기동 0**
(p9는 저작만, 실행 전). 승인 신규 0 (기존 "T2/T3 진행 승인" 범위 내).
git 참고: 사용자가 16th~19th분을 `fe2de19`로 commit/push 완료 (부트 시 clean tree 확인).

## §1 부트 과업 #0 — MEMORY.md 압축 (HARD RULE #8)

- 28,014B → 21,457B(헤더 정정 후 소폭 변동). D418/D417(+R1/R2)/D416 3개 엔트리를
  `MEMORY_archive_20260712.md` "2026-08-05 20th 세션 압축" 블록으로 **verbatim 이동**
  (sed 추출 append, **md5 대조 일치 확인** `1dabb58562982858e640d323cb28e6e8`) 후 본문을
  2줄 요약+포인터로 교체. 19th 엔트리 헤더 "T2 발사·미완" → "T2_PASS"로 본문과 정합화.

## §2 소스 추출 (에이전트 2건, verbatim 회수)

- **env 렌즈**: `roarm_stack_env.py` — USD 기본값 = B200 `/NHNHOME` 경로(:97-100, env-var로
  오버라이드, roarm_rl/__init__은 지연 import라 gym.make 전 설정이면 안전);
  `_grasp_condition`(:1192-1195) dist<0.025 ∧ q5≥0.4rad; latch(:1184-1190) release=q5<0.4rad;
  sim dt 1/200·decimation 2(control dt 0.01s) → episode 20s = 2,000 step; actuator arm/gripper
  동일 80/4/`effort_limit_sim`2.5; 물체 슬롯명 legacy `Sponge`; 카메라 없음(state-only);
  attach = kinematic pose-write(:1216-1236) — p7 방식 무력화 필수 재확인.
- **동결 자산 렌즈**: attempt3 폴더 6파일 실측 — root `roarm_m3.usd` sha
  `a4be58e87b1f9790f2a2ed600f0620c79d3cfb95c608b3c598308d52f5e46fff`(local과 **bit-동일**),
  physics 레이어 `configuration/roarm_m3_physics.usd` sha
  `043a5d35aa425c2589c77a34fcf415401ce9f9e7155e84ed75f6737df79fb503`(33,705B, 기대값 완전
  일치; local은 `1df07d38…` 4,242B convex_hull) → **attempt3 폴더 root를 가리켜야 64-part
  로드**(sublayer 상대경로). d334 `_usd_collision_inventory`(:197-239)·d349 body_checks
  (:921-934)·d337 `Q5_OPEN_RAD=1.5413`(:59) verbatim 확보. Rerun 스텝 타임라인 정본 패턴 =
  d362(:2888-2903, physics_step/sim_time_s + reset_time), d355는 전량 static(스텝 타임라인 아님).

## §3 T2b (사전등록 → 발사 → 미완)

- p8 수정: `--z_offset_m`(기본 0.0)·`--tag`(기본 t2) 추가. 기본값 실행 = T2와 동일 동작
  (솔버·게이트·격자·자기검증 무변경). offset≠0 ∧ tag=t2 → abort(exit 3) 가드로 T2 산출물
  보호 — 스모크 확인(산출물 0). **신규 sha `bde79c01f4b01d2ecdca503404593edddc4a219b20e14e11725a677c4df7093b`.**
- `t2b_prereg.md` 부속서 발행(t2_prereg 본문 승계 + 델타 = 높이 시프트 1건 + 판독 계획
  사전 고정: PASS 4후보 유지→스폰 그대로 / 일부 탈락→잔존 후보 / 0/4→p9 prereg 중단).
- 고정 CLI 실행: `--z_offset_m 0.012117 --tag t2b` → descend +0.050500 / approach +0.090500.
  자기검증 PASS(tilt 23.0195° / tcp_z 0.013521 — 높이 무관 게이트).
- **세션 말미 완주(exit 0): `T2B_VERTICAL_IK_VERDICT=T2B_PASS`.** 실높이에서 후보 4/4 유지
  (S1 descend/approach tilt 0.366°/1.314° · S2 0.208°/0.492° · R1c 0.525°/0.200° ·
  R2c 0.420°/0.383°), 외곽 4 FAIL 유지(R4c 16.86°/S3 20.17°/R3c 20.12°/S4 33.56° — 전부
  pos<3mm, D323 동형), 격자 URDF 264/513·v6-clip 250/513(T2 272/256 대비 −8/−6 = 외곽 경계
  소폭 내측 이동), best_named=seed0_S2(기록만), rerun_validation pass=True errors=[],
  rrd sha `536c4c7ccac7e339…`. 산출물 6종 전부 존재 확인.
- **판독(사전 계획 적용)**: "PASS 4후보 유지 → 스폰 권고 그대로" → **p9 스폰 = seed0_S1
  확정**. **잔여 의무: 육안검수(D341) 미수행**(context 비상) — 다음 세션에서
  `t2b_ik_reachability_inspection.png` 판독 + 관찰 기록 전까지 "검증기 PASS·육안 미검수"
  층위로만 인용.

## §4 p9 저작 (T3 프로브 — 실행 전, sha 핀·③ prereg 미발행)

`sim_scripts/p9_g0b_t3_cyld29h50_top_center_vertical_close_sweep_grasp_probe.py`, 1,082줄,
py_compile OK. p7 골격 승계 + 델타:

| 델타 | 구현 |
|---|---|
| D-1 | `Q5_OPEN_RAD=1.5413`; APPROACH/DESCEND q5=OPEN; close 스윕 기본 [88.31,60,45,41.40,39,37,35,33,31.65,28,24] **내림차순 검증**(첫값=OPEN±0.02°, 끝값≥23.0>release 0.4rad) |
| D-2 | `base_env._grasp_condition` monkeypatch: dist<0.030 ∧ q5≤41.40°. 파지 실증거는 여전히 LIFT follow≥6mm |
| D-3 | attempt3 USD를 `ROARM_M3_USD_PATH`로 주입(Isaac import 전) + root/physics sha 핀 assert + gym.make 직후·물리 스텝 전 64+64 스테이지 감사(d334 inventory 복사본; enabled part_ 정확 64/body + enabled 총수=64 + disabled `node_STL_BINARY_` 정확 1) 불일치 시 `USD_AUDIT_FAIL` hard-fail |
| D-4 | `CylinderCfg(radius 0.0145, height 0.050, axis Z)` + mass 0.02483 + μs0.40/μd0.30/rest0.0(계약 주 leg; 감도 leg는 CLI 인자로 별도 실행 가능) + 스폰 기본 seed0_S1 (설계 doc의 (0.300,0)은 D421 스폰 권고로 갱신 — ③ prereg에 명시 예정) |
| D-5 | p7 settled replan 승계(스폰 z=TABLE_Z 기반 → 지면 매립 12.117mm → settle → replan; T2b가 그 높이 커버) |
| D-6 | `/NHNHOME` 문자열 가드 + 자산 실존/sha 검사 → `USD_GUARD_FAIL` |
| D-7 | Rerun 스텝 타임라인(physics_step seq + sim_time_s ts, d362 패턴), 엔티티 15종(world/tcp·object·targets·cylinder + plots 8종 + metadata 2 + events), 고정 blueprint, validate_rerun_artifact(exact entities/timelines/components), CSV 전 스텝 기록. 접촉력 화살표 생략-정당화: 게이트가 접촉력 불소비 |
| D-8 | 단계 체인/verdict 세트(`latch.reached` 포함 p7 원형)/물리 게이트 6종/marker-only attach/posewrite watch/set_target watch 승계. episode 20s |
| **T2 귀결(설계 doc 외 추가)** | **waypoint IK = p8 수직 제약 DLS(q4=0) 복사 승계** — p7 위치 전용 `ik_dls`(+q4=90°)는 수직 미보장(D421 Impl ③: 동결 기움 가족은 위치 전용 IK의 선택). 전 waypoint pos≤3mm ∧ tilt≤5°, 미충족 시 경로 실패. 계약 §3 재료 기록: cfg 선언값 콘솔 출력 + 스테이지 MaterialAPI/PhysxSchema 전수 질의 → 콘솔+JSON+RRD metadata |

verdict 토큰 `G0B_T3_GRASP_VERDICT=` {REACH/APPROACH/LATCH/HOLD/LIFT}_FAIL | GRASP_PASS
(+USD_GUARD_FAIL/USD_AUDIT_FAIL). 산출물 = `g0b_d420/t3_grasp_*` 7종.

**사전 인지 리스크(감사 대조용)**: ① LATCH verdict가 p7 원형(latch.reached: TCP 유지 ∧
gripper_err≤0.75°)이라 첫 접촉이 41.40° 위면 stall→reached=False→LATCH_FAIL 가능 — D-8
attempt1 원안 유지 조항에 따라 수리 지점으로 취급. ② `_material_report`가 Isaac의 실제
material prim 구조를 잡는지 미검증. ③ rr API(Scalars/set_time/TimeSeriesView)는 repo
전례(d355/d362) 기반 — 감사 렌즈 5가 검증 담당.

## §5 적대감사 워크플로우 (발사 → 세션 말미 완주 회수)

`wf_78b1adfd-20d` — 6렌즈 Find(D-1/D-2 규약, D-3/D-6 자산·가드, D-4/D-5 계약+재료 기록,
p7-parity 행별 대조, Rerun/D341, 런타임 버그) → serious 10건 적대 Verify.
**16/16 agents 완주(에러 0, 1,895,560 tok, 1206.5s) — findings 18 수집, serious 10 전건
적대검증 생존(refuted 0), minor 8.** 전문(구조: {confirmed, refuted, minor}) repo 영속 사본 =
`claudedocs/runtime_logs/grasp_track/g0b_d420/p9_audit_wf_78b1adfd_findings_raw.json` (82.5KB).
17th 교훈("미완 워크플로우는 죽지 않는다") 재현 — 상태 문서를 "미수령"으로 봉인한 직후 완주
통지 도착, 세션 내 회수·정정 완료.

**Confirmed 요지** (수정 전 p9 실행 금지):
- **FATAL ① (3렌즈 독립)**: `object_follow_delta_m`이 run_to_q 호출별 시작점 대비라 lift가
  waypoint 2분할되면 마지막 구간 ~5mm만 측정 → min_lift_follow 6mm 구조적 미달, 완벽
  파지도 LIFT_FAIL. D-2의 "파지 실증거 = LIFT follow ≥6mm" 체인 붕괴. 수정 = lift 전
  구간 누적 z 추적(또는 phase 단위 follow 집계).
- **FATAL ②**: lead의 수직 IK 게이트를 HOME→approach **transit waypoint 전부**에 적용 —
  transit 경유점들은 수직 도구축이 기구학적으로 불가한 영역을 지나므로 매 실행
  APPROACH_FAIL. 수정 = transit는 p7 위치 전용 ik_dls, 수직 게이트는 approach 도착점 +
  descend/lift 수직 회랑에만. **D422 규칙의 적용 범위를 "수직 회랑 구간"으로 정정.**
- MAJOR: ⓐ close reached(gripper_err≤0.75°) vs 접촉 stall — 잔존 위험 창 (39.75,41.40°] +
  41.40 float knife-edge(marker 임계와 스윕 명령이 동일 리터럴) ⓑ marker 거리항이 descend
  자세에서 상시 참(0.0255+3mm<0.030) → 밴드 스윕이 첫 fire 각도(41.40/39)에서 무조건
  절단, 실조임은 HOLD 41.4→24° 단일 점프로 이동(속도/드리프트 게이트 저촉 시 HOLD_FAIL
  오귀속; p7 continue 플래그 제거 회귀) ⓒ episode 20s=1999 control step < 최악 3615 step
  ⓓ 고정 산출 경로 — 재실행 시 이전 증거 덮어쓰기+검증 충돌(p8식 tag 가드 부재)
  ⓔ RERUN_VERSION_MISMATCH 조기 return에 sim_app.close() 누락 → Isaac Kit 행.
- MINOR 8: /NHNHOME 가드 vacuous / 스폰 seed0_S1 vs 설계 doc (0.300,0) — ③ prereg 명시
  의무 / descend FAIL의 APPROACH_FAIL 표기(p7 상속) / REACH_FAIL 경로 JSON Infinity/NaN /
  D341 접촉 증거 생략이 funnel 진단과 긴장 / 기타 전문 참조.

## §6 산출물

- `g0b_d420/t2b_prereg.md`(신규), t2b_ik_stdout.log(진행 중), p8 수정판, p9 신규,
  START_HERE 전면 갱신, LEDGER 1행, DECISIONS D422, 본 doc, MEMORY 압축+20th 엔트리.

## §7 다음 세션 (순서 고정)

1. T2b 육안검수(D341) — `t2b_ik_reachability_inspection.png` 판독 + 관찰 기록
   (verdict/스폰은 확정 완료: T2B_PASS, seed0_S1).
2. `wf_78b1adfd-20d` 회수 → p9 수정 → py_compile → **p9 최종 sha 확정**.
3. ③ prereg/hash/attestation 발행(p9 sha + 전체 CLI + 게이트 + D341 계약 + 스폰 근거) →
   T3 Isaac 실행(승인 기수령; 실행 직전 tuple 요약 브리핑) → `*_FAIL` = 다음 수리 지점.
