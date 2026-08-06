# 2026-08-06 (23rd) — G0b T3: 조 목구멍 폐색 읽기 전용 정점 감사 (JAW_AUDIT_CONSISTENT — 중앙 플러그 특정 + 조 원위부 부재 실증)

이번 case의 신규 변수: [없음 — D424 Implication ②가 지시한 읽기 전용 진단. Isaac 기동 0,
자산 변경 0, 물체/질량/마찰/스폰 변경 0. 분석 계층 노브만 신설(샘플링 0.5mm, 게이트
1.0/0.5mm, VIEW_STRIDE 16).]

Case `g0b_d420` 계속. 로봇 HW 0 · lerobot-train 0 · git commit/push 0(사용자가 직접
`702580f` "D424·22nd" 커밋·푸시 — 부트 시 START_HERE Git 섹션만 현재 사실로 정정).
세션 진행 규칙: 실패 가능 실험 = 본 감사 자체(consistency 게이트 CA1/CA4/ANGLE_INV/CB —
기하 예측이 Isaac 4회 실측을 재현 못 하면 GEOMETRY_MISMATCH).

## §0 부트 검증

boot 6단계 이행. 원본 교차검증: 4회 verdict(stdout `G0B_T3_GRASP_VERDICT=` 라인),
a1 정지 z=0.054398/a4 z=0.054394(steps CSV 최종 descend 행), p9 sha `99c99c65…2412`
(1,780줄), 산출물 32파일 — 전건 상태 문서와 일치. 불일치 1건 = START_HERE.md Git 섹션
(구 "fe2de19 미커밋") → 사용자 push 반영 정정.

## §1 목적 / 층위

D424 ② 지시: attempt3 physics 레이어 link5/gripper_link 64+64 파트 정점을 링크
좌표로 덤프 → q5별 FK 변환 → (a) 축 근방 최저점 = "+4.4mm 바닥" 파트 특정
(b) 조 면 간극-깊이 프로파일. **읽기 전용 진단 — 재분해 아님(D415 ③ 무저촉),
sim 충돌 기하 사실만 주장(실물/파지력/시각 메시 주장 아님).**

## §2 방법

- 스크립트: `sim_scripts/g0b_t3_attempt3_jaw_throat_occlusion_readonly_vertex_audit.py`
  (최종 sha `bca4f898023f63f21d540483a169499760038c582ce3a7919d7622e77946e1c3`, 774줄 —
  적대검증 후 기계 수리 6건 반영판. 실행 3-leg의 리비전은 §3 참조).
- pxr 부트스트랩: Kit 기동 없이 `omni.usd.libs-1.0.1` 번들 pxr을 재-exec로 로드
  (LD_LIBRARY_PATH는 프로세스 시작 시 읽히므로 in-process 설정 불가 → execve 1회).
- 파트 열거: p9 `_audit_collision_bodies`와 동일 기준(TraverseInstanceProxies +
  CollisionAPI + `part_` + enabled, legacy `node_STL_BINARY_` 정확 1건 disabled).
- 조인트: `/roarm_m3/joints/link5_to_gripper_link` revolute frame(localPos0=(0,0.018821,
  0.052035), axis Z, limits [0,90.01]°)에서 X_l5_g(θ)=T0·Rz(+θ)·T1⁻¹. 부호 근거 =
  UsdPhysics revolute 규약(한계값은 범위만 고정 — 방향 아님; 적대검증이 반대 부호 시
  자기-관통을 실증해 교차 확인).
- TCP frame: link5 +z로 115.428mm (roarm_kinematics `_CHAIN` == env `TCP_LOCAL_OFFSET_M`,
  G5 게이트). 자세 = 각 attempt results JSON의 plan q_descend R + steps CSV 실측 TCP.
- 샘플링: 파트별 scipy ConvexHull 면을 0.5mm 피치로 표면 샘플 + hull 정점(정점은
  극값에 대해 정확 — min-z는 정점에서 달성됨을 검증 에이전트가 별도 재계산으로 확인).
- 게이트: G1 sha 핀 / G2 64+64 구조 / G3 조인트 frame 증인(authored=θ0 잔차 1.2e-7) /
  G4 FK 분할 자기일관 / G5 env TCP 상수 → 실패 시 AUDIT_ABORT.
  CA1·CA4(바닥 |clearance|≤1.0mm)·ANGLE_INV(≤0.5mm)·CB(a2 밴드 무접촉) → 실패 시
  GEOMETRY_MISMATCH.

## §3 실행 3-leg + supersession (전부 보존 — 삭제 0)

| leg | tag | 결과 | 처분 |
|---|---|---|---|
| 1 | `t3_jaw_audit` | 과학 verdict CONSISTENT, **rerun validation FAIL** — `plots/min_horiz_clearance_mm`가 전 각도 None이라 엔티티 미생성(exact 계약 위반) | SUPERSEDED (계약 버그 증거로 보존) |
| 2 | `t3_jaw_audit2` | 계약 수리(None→NaN 상시 로깅 + `vertical_gap_mm` 지표 신설) → validation PASS. **그러나 inspection PNG = 뷰어 로딩 중 캡처(빈 viewport + "Loading" 토스트) — 결함**. 원인 = RRD 27.9MB(p9의 24×)가 헤드리스 스크린샷 로더와 레이스 | SUPERSEDED — **validation PASS를 인용할 때 반드시 이 결함 병기.** 검증기 계약은 mid-load PNG를 감지 못함 — 육안검수만이 잡음(D341 실증) |
| 3 | `t3_jaw_audit3` | 뷰 데시메이션 1/16(RRD 4.32MB) → validation PASS(footer verify 양 파일, exact 엔티티 14/타임라인 3/컴포넌트 계약) + **PNG 정상 + 육안검수 완료(§5)** | **권위 leg** |

- leg 간 무결성(컴플라이언스 에이전트 검증): leg2 vs leg3 results JSON은 tag만 상이;
  parts.csv는 leg1 vs leg3 md5 동일 — 결정 계층 3-leg 안정.
- leg1/2 생성 시점의 스크립트 리비전은 in-place 수리로 미보존(MINOR) — 위 동등성
  증거로 갈음. 최종 수리판 sha만 핀(§2).

## §4 결과 (권위 = t3_jaw_audit3_stdout.log + t3_jaw_audit3_results.json)

**가드**: G1~G5 전부 PASS (sha 핀 일치 / 64+64·legacy 1·convexHull 전건 / joint frame
잔차 1.2e-7 / FK 자기일관 0 / env TCP 일치). 샘플 736,206(link5)+153,197(gripper).

**CA1 (attempt1 재현)**: 바닥 z=+0.049953 vs 원통 top +0.050000 → clearance **−0.047mm**
(접촉 ≈0 예측 적중), d_floor **4.445mm** (실측 stop 4.398mm). **바닥 파트 =
`link5/collisions/d338_convex_parts/part_029 + part_030`**, 원통 축에서 r=11.77mm
(footprint 14.5mm 내부 중앙 플러그. 차상위 파트는 +5.9mm 위 — 특정 견고).

**CA4 (attempt4 재현)**: clearance −0.048mm, d_floor 4.442mm, r=12.03mm — 동일 파트.
**ANGLE_INV delta 0.003mm** (실측 4μm와 부합) → "개방각 무관 축 근방 고정 구조물" =
link5 고정체의 part_029/030으로 확정.

**CB (attempt2 재현)**: a2 close 밴드 11각 전부 무접촉. footprint 내 구조물 최저점이
원통 top 위 **+0.233mm(mean 자세)** — 방법론 감사 정정: close 중 최저 실측 자세 기준
**+0.117mm** (mean은 초기 과도 10행이 +0.113mm 상향 편의; 부호 불변·무접촉 유지).
marker는 39°부터 발화했으나 0.1~0.2mm 갭이 남아 접촉 0 — a2의 "marker 발화·무접촉"
미스터리가 기하로 닫힘.

**Throat 프로파일**: assembly_max_depth = **4.458mm 전 각도 상수**(link5 지배).
이동 조(gripper_link) 최심점: 88.31° = TCP **위** 53.07mm / 45° = 위 11.73mm /
24° = 아래 0.324mm(r=21.8mm) / 0°(완전 닫힘) = 아래 3.69mm(r=4.5mm).
rim 깊이 밴드(5~15mm) 구조물 = **전 각도 None**. 검증 에이전트의 0.02° 정밀 스윕:
gripper 최심 피크 +3.956mm @ q5=5.10°(샘플 간 보간 피크) → 전역 여유 0.502mm —
**"어느 각도·어느 반경에서도 TCP−4.458mm 아래 구조물 없음" 확정**(link5 플러그가 항상 최심).

**진단 종합**: 동결 attempt3 충돌 자산은 ① 중앙 플러그(part_029/030)가 TCP−4.46mm에서
원통 top을 막고 ② 실물 rim 핀치(상면 0~12mm 물림, T1)에 필요한 **조 원위 손가락
형상이 충돌 레이어에 부재**(스윕 밴드에서 TCP 아래 ≤0.32mm) — D424의 "목구멍 폐색"
기전이 두 성분으로 분해·특정됨.

## §5 D341 육안검수 기록 (t3_jaw_audit3_inspection.png, 5.24MB)

관찰: ① 패널1 헤더(verdict/sha/바닥 수치/각도 불변 0.003mm/floor parts/sweep
contact-free) — stdout 전건 일치, 수치-시각 불일치 0 ② 3D 뷰: 회색 link5 클라우드
하단 끝 **빨간 floor 파트가 주황 원통 top 링 바로 위 중앙에 돌출**(플러그-top 접촉
기전 시각 확증), 파란 이동 조는 축 우측 단일 조로 top 링 위에 머묾, 초록 TCP 마커+
수직 tool axis가 링 중심 통과 ③ 패널4 q5 곡선 88→0 + footprint 바닥 깊이 4.458mm
평탄선 ④ 패널5 assembly_max_depth 4.458mm 평탄선, min_horiz_clearance NaN(정의상 정상).
사소(표시 전용): 우상단 gRPC/Loading 토스트 오버레이(p9 관례와 동일), **gates TextLog
패널이 band_index 타임라인에서 빈 표시**(gates가 reset_time 후 log_time에만 기록됨 —
엔티티/검증 계약은 PASS, 권위는 stdout; 스크립트 사후 수리로 band_index 스탬프 추가),
3D 기본 카메라는 이번엔 결정 대상 전체를 담음.

## §6 적대 검증 4-렌즈 (병렬 agents, FATAL 0 — verdict 4/4 생존)

| 렌즈 | 결과 |
|---|---|
| ① 수학/frame (accc8f75) | 5항목 전부 SURVIVES. FK-USD 대조 7.1e-6, Gf/quat 왕복 0. **강화**: 부호 −1이면 이동 조가 legal 각도에서 link5 자기-관통(불성립 기구) → +1 유일. **정정**: 부호 근거 문구(한계값은 범위만 고정), 독립 앵커는 사실상 1개(4.4mm 바닥 — CB·ANGLE_INV는 비판별적), r=11.5mm(CSV, link5 frame 축 기준)와 r=11.77/12.03mm(원통 축 기준)는 다른 양 |
| ② USD 파싱 (ae768ec5) | 5항목 전부 SURVIVES(누락 0, metersPerUnit 1.0/스케일 없음, physics 레이어 합성 확인, cook-faithful — 전 파트 ≤13정점 < hullVertexLimit 64, rest/contactOffset 미저작=0). **신규 발견**: world/link1~4에 legacy 전체-메시 convex hull collider **enabled 잔존**(link2 28,092점) — 바닥 후보로는 배제 실증(a1 자세에서 최저가 top 위 120.6mm) — T3 verdict 무영향, 향후 접촉 추론·자산 게이트 문구에 중대. omni.physx의 Xform-level CollisionAPI 실체화 여부 1건은 향후 확인 |
| ③ 방법론 (ad6e8154) | 결론 전부 생존 + 독립 재계산 2건(0.1mm 피치 재샘플링 bit-일치 / 0.02° 스윕 §4). **정정 수용**: "오차 한계=피치" 문구는 일반 정리 아님(이번엔 정점-달성이라 실오차≈0) / CB 마진 +0.117mm 병기 / 게이트 수용창 ±0.95mm는 약함 — 실일치 0.047mm가 증거이지 게이트가 아님 / 자세 orientation 비기록 → 정직 정밀도 ~±0.2mm/°. **비기하 원인 배제 확정**: a4의 "12mm 아래 목표"는 wp006 stall로 미발행(실발행 깊이 4.98mm) — 그래도 명령 1.09mm 차이 vs 정지 4μm 차이로 기하 원인 유일 선택. 미인용 물리 채널 발견: stall 중 물체 진동 20~30배(접촉 하중 증거) |
| ④ 컴플라이언스 (a18cc1a3) | 읽기전용 PASS(변이 API 0, 자산 sha/mtime 불변), 동결 산출물 PASS(t3_grasp*/t2* 무변경 — audit는 read만), overclaim PASS(층위 문구 전건), 변수 사다리 PASS. MAJOR-1(본 doc 작성으로 해소) / **MAJOR-2: audit3 results JSON의 `gates.ANGLE_INV`가 문자열 "True"**(np.bool_→default=str; "False"도 truthy — 소비자는 native 캐스팅 필수. verdict 계산은 in-memory 값이라 무영향) / MINOR 5(§5 gates 패널·cook 문구·리비전 미보존·밴드 마커·단독 저작 caveat) |

**스크립트 사후 기계 수리 6건**(동작 무변경 — bool/float 캐스팅, cook·부호 문구,
beyond-band 마커, gates band_index 스탬프, docstring): py_compile OK, 최종 sha
`bca4f898…1c3`. audit3 재실행 생략 — 3개 렌즈 일치로 verdict 무영향 + 수리 전건이
직렬화/문구/표시 계층(본 조항으로 사전 고지, 22nd §4 관례).

**단독 저작 caveat**(렌즈④ MINOR-5): 본 감사는 p9류 사전 적대감사 없이 저작→실행
후 4-렌즈 사후 검증을 받았다. 에스컬레이션 인용 시 이 순서를 명시한다.

## §7 판정·한계

- **G0B_T3_JAW_AUDIT_VERDICT=JAW_AUDIT_CONSISTENT** — 기하 예측이 물리 실측(4.4mm
  바닥·각도 불변·a2 무접촉)과 정합. 판별 강도의 정직 표기: 독립 앵커 1개(바닥 깊이,
  실일치 0.047mm) + 물리 채널 보강(stall 진동·명령-정지 해리) — CB/ANGLE_INV는 재현
  확인이지 추가 판별이 아님.
- 한계: sim 충돌 기하 사실만(시각 메시·실물·파지력 아님, `g0a_pass=false` 불변) /
  각도 14점+0.02° 스윕 보간 / 자세 orientation 비기록(±0.2mm/°) / scipy hull=cooked
  hull은 이번 자산에서 cook-faithful(≤13정점)임을 근거로 한 등치.

## §8 다음 (사용자 결정 대기 — D415 ③ 해제 권한은 사용자)

자산 수리 없이 T3 GRASP_PASS 불가(D424) + 이번 감사로 수리 대상이 특정됨:
(A) 재분해 금지 해제 → 조 원위부 포함 재분해, (B) 최소 수리 — part_029/030 비활성
(플러그 제거)만으로는 **불충분**(조 원위부 부재로 여전히 무접촉 — 본 감사 §4가 실증),
(C) TCP-타깃 재유도(파지 의미 변경 — D419 고정 방식과 충돌 소지), (D) 손가락 충돌
파트 추가 저작(재분해 아닌 증분 — D415 ③ 해석 필요). 결정 후 attempt5 재사전등록
(prereg 부록 D) → T4 실물 재현 대조.

## §9 산출물

`g0b_d420/`: t3_jaw_audit{,2,3}_{results.json,parts.csv,timeline.rrd,timeline.rbl,
inspection.png,rerun_validation.json,stdout.log,stderr.log} (3-leg 전건 보존),
스크립트(sha `bca4f898…1c3`), 본 doc, START_HERE/LEDGER/DECISIONS(D425)/MEMORY 갱신.
적대검증 4 agents 전문은 세션 전사 내(accc8f75/ae768ec5/ad6e8154/a18cc1a3).

**/half-clone 거부 11회째(#11, stop-hook 지시 — context 169% 경고에도 불구.
상태 문서 롤업 기완료 + continuation prompt 출력으로 대체).**
