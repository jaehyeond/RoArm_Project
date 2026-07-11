# Session 2026-07-11 (late) — D332 검증 브리핑 + D333 precheck 권고

Verdict: `D332_VERIFICATION_BRIEFING_D333_PRECHECKS` (audit, no sim run)

이번 case의 신규 변수: `[]` — D332 결과의 라인 단위 검증/브리핑 + D333 설계
보완 권고만. 코드/target/게이트 변경 없음.

## Session progress rule 준수 노트

실험 없음. 정당화: 사용자 요청으로 D332 결과(별도 세션 산출)를 독립
검증했고, D333 실행 전 반영해야 할 precheck 2건과 판독 기준 3건을 도출했다.
D329/D331 감사 선례와 동일 구조.

## 검증 결과 — D332 보고 전부 사실

- git: HEAD == origin/master == `d5d2060` (사용자 직접 commit/push).
- 수치: `g0a_d332_static_collision_summary.json`과 전부 일치 — raw STL
  `+4.273819mm` / 수학 hull `-6.363467mm` / mirror recook `-6.236272mm`
  (35v/48p), commanded TCP `0.817812mm`, gripper_link `66.866266N`@step0,
  link4/link5 `0N`, XY `10.282285/10.452925mm`, tilt `9.235/9.440deg`,
  baseline net `7.0632007N` (m·g 오차 `7.1e-7N` — net reporter 신뢰 근거),
  baseline max XY `0.459mm`/tilt `0.673deg`.
- 씬 구조 독립 확인: `/World/ground` 전역 plane z=0
  (`roarm_rl/roarm_stack_env.py:133-135`, TerrainImporterCfg) + TapTable
  (`roarm_rl/roarm_cube_push_env.py:230`) 공존 — `12.117mm` 매립(=|TABLE_Z|)
  은 측정 아티팩트가 아니라 씬 구조.
- 코드 라인: d330 probe `:482` `_try_make_sensor`(post-PLAY 수동 생성),
  d332 probe `:504` zero-reporter spawn / `:643` sensor contract / `:705`
  mirror recook 실재. `activate_contact_sensors(prim_path, threshold=0.0)`
  시그니처 확인 (installed isaaclab `schemas.py:488`) — bool→`1.0N`
  threshold 버그 메커니즘 부합.
- 산출물: PNG 3 + RRD 1 + attempt0/1 보존 확인. MIXED 판정은 pre-registered
  matrix 규약대로이며 중간 CONFIRMED 철회는 올바른 재감사.

## 신규 해석 3건 (D333 판독 기준)

1. **D330 소급 혼입**: D330 CSV `target_tcp_z=0.032883` → 캡처 시점 원통이
   아직 매립 상태 → 접근 중 실제 중심은 pop 이후 ~`0.045m`인데 TCP는
   `0.0329`를 조준 = **12mm 낮게 조준한 실험**. D330의 gap/contact-height
   게이트도 잘못된 가정 자세 기준.
2. **step-0 gripper 이벤트의 방향성 의심**: pop(+12.7mm, 125N)이 원통을
   그리퍼로 밀어올린 아티팩트일 가능성 — "정렬 pose 자체가 충돌"의 증거
   아님. D332의 coupled 판정과 정합.
3. **gripper 66.9N vs link5 0N 긴장** (D231: gripper collision = 4mm proxy):
   후보 ① pop 이후 자세에서 link5 hull 비겹침(mirror 계산은 설계 자세 기준)
   ② link5 filter 채널 결함(독립 양성 대조 없음) ③ 4mm proxy가 상승 원통에
   걸림. **D333 클린 재시험이 판별**: ground 제거 후 원통이 설계 z에 머물면
   mirror가 맞을 경우 teleport 즉시 link5 접촉/depenetration이 나와야 하고,
   support 양성 대조가 살아있는데 link5가 0N이면 mirror≠live cook 또는
   filter 채널 결함 확정.

## D333 precheck 권고 2건 (사용자 승인 시 명세에 포함)

1. TapTable collision top z == TABLE_Z 사전 확인 + settle 첫 post-step
   z-보정 ≈ 0mm를 precheck gate로 등록 — "혼입 제거 성공" 자체의 검증.
2. 전역 ground collision 비활성화가 로봇 base(고정 articulation root)
   지지에 영향 없는지 1줄 확인 기록.

## Non-goals (불변)

D332/D333 non-goals 전부 유지. collision mesh 재저작, target/게이트 수정,
G0b, RL/PPO, 렌더, 질량/마찰 변경, VLA, RoArm, B200, 큐브, /half-clone 금지.

## Next

D333 실행: START_HERE.md D333 명세(`support_domain_global_ground_collision_disabled`
단일 변수, TapTable sole support, 동일 1-env 200+200 static retest) + 위
precheck 2건 + 신규 해석 3건을 판독 기준으로 사용.

## D333 후속 correction (2026-07-11, runtime 이후 append)

이 문서의 D333 전 해석 중 다음 두 문장은 과도했으며 D333 원자료로
교정한다.

1. D330의 `target_tcp_z=0.032883`은 매립 reset pose 캡처를 증명하지만, D330은
   object z/quaternion을 기록하지 않았다. `실제 중심보다 약 12mm 낮게 조준`은
   D332 재현을 결합한 강한 구조 추론이지 D330 per-trial 직접 측정이 아니다.
   또한 D330 gap gate는 XY-only라 vertical pop으로 오염됐다는 표현은
   부정확하다. 실제 이동한 원통 대신 start XY를 쓴 별도 한계가 있다.
2. `support PASS + link5 0N -> mirror/live cook 또는 filter 결함 확정`은 잘못된
   이분법이었다. D333 clean run은 gripper_link step-0 `76.412755N`, link5
   `0N`, max XY `12.598179mm`의 제3 branch를 관측했다. 판정은
   `D333_G0A_CLEAN_STATIC_BODY_ATTRIBUTION_MIXED_STOP`이며 collision shape/owner
   parity 전에는 body-specific repair가 금지된다.

D333은 ground pop을 제거했지만 clean gripper event와 object disturbance를
제거하지 못했다. 따라서 이 문서의 `pop-into-gripper artifact`는 가능한 일부
기여였을 뿐 sole explanation으로는 반증됐다.
