# Session 2026-07-11 — D331: External critique audit of the D330 analysis + D332 static collision discriminator design

Verdict: `D331_G0A_ANALYSIS_AUDIT_D332_DESIGN` (audit, no sim run)

이번 case의 신규 변수: `[]` — 감사/설계 세션. 물체·목표·게이트 변경 없음.

## Session progress rule 준수 노트

이 세션은 sim 실험을 돌리지 않았다. 명시적 정당화: 사용자가 제공한 외부(제2 AI)
critique가 D330 해석의 근거 강도를 문제 삼았고, 6개 수정 주장의 진위가 다음
failable probe(D332)의 설계를 직접 바꾸므로 새 실험 전에 결정-변경 감사가
선행되어야 했다 (D329 선례와 동일 구조). 감사 결과 D332 설계가 강화되었다.

## 검증 결과 — 외부 critique 6건 전부 사실

1. **Regime 분류 = 5 low + 1 intermediate + 4 stall**: trial 1은 TCP 27.233mm,
   actual z 51.865mm(목표 +19mm), displacement 39.456mm(최대)라 "정상 z"
   클러스터가 아니다. stall regime = trials 5/7/9/10 (TCP 70.3–80.5mm,
   z 0.090–0.098m, joint err 0.142–0.170rad). n=10이므로 "bimodal"이 아니라
   "두 regime + 중간 사례 1"이 정확. (`g0a_d330_cyl_alignment_trials.csv`)
2. **Reset jitter — "동일 초기상태·동일 명령" 아님**: HOME + uniform(±0.02rad)
   jitter (`roarm_rl/roarm_cube_push_env.py:1636`; D330은
   `RoArmCubeTap10cmEnvCfg` + `ik_endpoint_reset=False`라 이 브랜치,
   probe:324,327,366). IK seed = jitter된 실제 관절값 (probe:655), 위치-only IK
   (probe:678-679, orientation 축 None) → 5관절·3구속의 2-dim null space에서
   env별 wrist 구성이 갈라질 수 있음. **per-env commanded joint 벡터는
   CSV/JSON/rrd 어디에도 미기록** (rrd는 env0만). "병렬 env 갈라짐 = 접촉
   카오스 증거" 추론은 철회.
3. **link5 collision = convexHull, 직접 stage 증거로 승격**: D231은 binary USD
   미개봉 inference였다 (D231 doc :42-44). 본 세션 pxr instance-proxy
   traversal로 직접 확인: `/roarm_m3/link5/collisions/link5/node_STL_BINARY_`
   및 `/roarm_m3/gripper_link/collisions/gripper_link/node_STL_BINARY_`에
   `PhysicsCollisionAPI`+`PhysicsMeshCollisionAPI`,
   `physics:approximation=convexHull` 적용. 주의: 이 prim들은 instance proxy
   뒤에 있어 기본 `Stage.Traverse()`에 나오지 않는다 —
   `Usd.TraverseInstanceProxies` 필요. (asset:
   `local_assets/roarm_m3/usd/roarm_m3.usd`)
4. **힘 역산 불가 + tipping이 기본 교란 모드**: 8.48N(μ_d)/10.59N(μ_s)은 평면
   슬라이딩 이상화 값. 서있는 D34×H90 (CoM 45mm, base r 17mm)의 중심높이 수평
   push tipping 임계는 F≈mgr/h≈**2.67N** — 슬라이딩 시작의 약 1/4. D330은 물체
   최종 z/quaternion을 기록하지 않아 (trial row keys 감사) 19mm XY
   displacement가 슬라이딩인지 기울어짐/전도인지 판별 불가. gap/contact-height/
   penetration proxy 게이트는 전부 물체 직립을 가정하므로 기울면 조용히 무효.
5. **D326은 명시 위반 아님 + 그대로 복제해도 무용**: 규칙 범위는 "D325 target의
   execution repair 전" (DECISIONS D326). 더 중요한 사실: d326
   `_teleport_check`는 `write_joint_state_to_sim` 후 physics step 0회 — proxy
   수학만 평가하므로 hull 접촉/물체 교란은 원리적으로 검출 불가 (d326
   probe:429-449). D332는 settle step이 필요하다.
6. **100-120g 추정치가 커밋 artifact에 "real spec"으로 유입**: probe:964
   `mass_note: "...replace with real 100-120g spec"` + summary JSON
   `/object_contract/mass_note`. 사용자 확인: 제2 AI의 미실측 추정치. Durable
   correction: **실물 원통 질량은 미측정, 100-120g은 추정치 — G0b 전 실측
   필수.** 커밋된 artifact는 append-only 원칙상 수정하지 않고 본 기록으로 정정.

## 외부 미기록 파일럿 (사용 조건부)

Critique 제공자의 scratch 계산 — env0 최종 commanded pose에서 raw link5 STL은
원통과 +4.04mm clearance, **단일 convex hull은 ~6.545mm penetration** — 은
gap-fill artifact 가설을 지지하지만 repo 밖 미기록 + env0 단일 pose다. D332에서
기록된 산출물로 재계산되기 전에는 결정 근거로 쓰지 않는다.

## git

감사 시점 HEAD == origin/master == `84d0934` (D330). 본 세션 코드/커밋 없음
(상태 문서 갱신만).

## D332 사전 등록 설계 (failable; 구현/실행 전 사용자 확인)

목표: "link5 convex hull이 D330 commanded pose에서 원통과 겹치는가"만 확정.
collision mesh 재저작·target/게이트 수정·waypoint 탐색 금지.

1. **오프라인 actual-hull overlap** (Isaac 불필요): link5 collision mesh(STL,
   USD와 동일 소스)의 convex hull을 commanded pose에 배치, 원통 D34×H90과
   penetration/clearance 계산. AABB 판정 금지. 기록된 산출물로 저장.
2. **고정 reset (jitter=0) + per-env commanded joint 벡터 CSV 기록** — D330의
   기록 공백을 닫는다.
3. **Teleport + controlled settle steps**: N physics step 후 물체 full pose
   (xy, z, quaternion) 및 displacement 기록 — d326 proxy-only 방식 복제 금지.
4. **Contact witness**: scene-owned·pre-PLAY·전체 env-domain 원통 ContactSensor
   + robot-link filter. 알려진 접촉에서 force>0이 실제 보고되는지 검증 단계
   포함 (init 성공 ≠ 보고 검증, D328 교훈).
5. **Null-space family 스캔**: wrist pitch 범위에서 overlap의 자세 민감도 —
   D330 regime 분열(6 근접 vs 4 stall)의 설명 후보 판별.

판정 semantics: commanded pose에서 hull overlap 확정 → blocker = collision
geometry class → 수리 옵션(진짜 조 형상 collision 저작 vs 정렬 family 변경)은
사용자 결정으로 회부. overlap 없음 + static 교란 없음 → runtime/drive 상호작용
감사 재개.

## BACKLOG 기록

- `tool_surface_union` (D231): 가동 조 gripper_link collision이 4mm proxy
  (convexHull, 본 세션 직접 확인)라 G0a alignment PASS 후에도 G0b 파지 물리가
  성립하지 않는다. G0b 진입 전 필수 준비물. (지금 착수 금지, 기록만.)

## Non-goals (불변)

G0b 파지/lift, 그리퍼 닫힘, RL/PPO, 위치 랜덤화, 렌더(단일 프레임 viz 제외),
마찰/재질/질량 변경, VLA, RoArm 실기, B200, 큐브 재등장, waypoint 탐색,
collision mesh 재저작, 기존 파일 이동/개명.

## Next steps

1. 사용자: D332 설계 확인 (특히 jitter=0 고정 reset은 "D330 정확 재현"이
   아니라 "판별 우선" 선택이라는 점).
2. D332 구현 + 실행 (failable). 산출물:
   `claudedocs/runtime_logs/grasp_track/g0a_d332/` + Visualization DoD.
3. 결과에 따라 collision geometry 수리 옵션 논의 또는 runtime/drive 감사 재개.
