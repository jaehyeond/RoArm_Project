# 59th — `g0c_d446` bg1 실행: reBot B601 평행 그리퍼 flying-gripper 판별 = **A(공식 USD) 0/13 vs B(충실 충돌) 13/13** — 실기하는 전 pose에서 쥐고 유지, 병목은 벤더 USD의 충돌 근사

- 날짜: 2026-08-13 KST (58th 직후 동일 날짜 세션)
- Active case: **`g0c_d446`** (본 세션 개시, 사용자 명시 승인 "1번 진행해. 승인할게" —
  교수님 보고 패키지는 사용자가 명시 보류)
- 이번 case의 신규 변수 (2개): `[그리퍼 기종 = reBot B601-DM 평행 그리퍼 (팔 제거
  유지)]`, `[충돌 표현 = A(공식 USD 1-hull/핑거) vs B(동일 공식 점군의 blade/mount
  2-piece 분해)]`
- Session progress rule: 실패 가능 물리 실험 **1회 실행** (bg1, 26 run × 420 step,
  SUCCESS 게이트 실패 가능) — 충족.
- 성격: 부트 재검증 → 사용자 질문(하드웨어 교체) 분석·웹 검증 → case 개시 → 자산
  확보·감사 → 변형 저작(R1~R4) → 스모크/진단 5종 → **본 실행 26 run** → D341 → 상태 갱신.

## 0. 배경 — 사용자 질문과 case 성립

사용자: "비대칭 jaw + 5+1 DOF 저가 가동범위로 될 수 있는 grasp이 많이 실패하지
않아? [reBot B601로] 바꿔서 test해본다면? 어차피 물리엔진 sim에서 되어야 뭐든 한다.
됐다고 하면 사지/만들지." → 웹 검증 브리핑(§1) 후 **sim 판별 case만 승인**.
전제는 기존 증거와 정합: D430(수직 수용 밴드 D<20.25mm vs 과제 D29),
D441(3,476 pose 유효 파지 0), D445(fg1: 이상 배치에서도 물었다가 배출 0/13).

## 1. B601 사전 조사 (웹, 지식 컷오프 이후 제품이라 전부 외부 소스 검증)

- Seeed Studio reBot Arm B601 (2026-04 출시): **6+1 DOF**, **평행 그리퍼**
  (prismatic 2지, URDF에서 직접 확인: `gripper_joint1/2` 각 [0, 0.0715] m),
  Damiao QDD 모터(DM-J4310×4 + DM-J4340P×3), reach 767mm, payload 1.5kg,
  반복정밀도 <0.2mm. 가격: Complete Bundle $1,197 / 조립품 $1,499 / DIY 구조 $169
  + 모터 $829 + 그리퍼 $199 (모터가 BOM 지배 — 3D 프린팅 절약 ~$300뿐).
- 생태계 (GitHub API 직접 확인, 4월 기사보다 진전): `Seeed-Projects/reBot-Isaacsim`
  (공식 USD DM/RS + URDF + MJCF, 8/11 갱신), `lerobot-robot-seeed-b601`(follower) +
  `lerobot-teleoperator-seeed-b601`(leader), 8/12 갱신. 라이선스 CERN-OHL-W/Apache-2.0.
- 출처: CNX Software 2026-04-17 기사, seeedstudio.com 제품/블로그, GitHub
  Seeed-Projects (커밋 `cb824be1`), URDF/usda 원문 직접 파싱.

## 2. 자산 확보 + 감사 (`b601_asset/`, `bg1_asset_audit.json`)

- `reBot-Isaacsim` 커밋 **`cb824be157fdd5db7d6153b644b9b8ce85775bef`** (2026-08-10)의
  `usd/reBot_B601_DM/` 9파일 verbatim 복사 + SHA 핀 (`b601_asset/UPSTREAM.md`).
  루트 doc: "URDF USD Converter v0.1.3" 산출물, Physics variant 기본 = physx.
- 구조: 중첩 링크 체인(base_link→…→link6→gripper_link→gripper_left/right),
  `gripper_joint1/2` = prismatic axis X(조인트 프레임), limits [0, 0.0715] m,
  **maxForce 100 N**, **drive stiffness/damping 미저작**(스키마 기본 0 = 무력),
  질량 팜 0.1818 / 핑거 각 0.0423 kg, 물리 재질 미저작.
- **패드 기하 실측** (glf 프레임, 점군 19,481/핑거 + 투영 PNG 육안):
  접근축 = +x̂(팁 x=0, 블레이드 x∈[-0.04896, 0], 팜 x∈[-0.151,-0.073]),
  개구축 = ŷ(q=0 = 닫힘, 개구 = qL+qR, 최대 143mm), 블레이드 z 폭 ±0.0196 m,
  블레이드 두께 ~13.6mm(이빨 패턴), **마운트가 반대편 +58.6mm까지 교차**(interlock 설계).
- ★★ **hull-fill 쐐기 발견 (변형 A 예측 근거, LP 단면 실측)**: 충돌이 핑거당
  convex hull **1개**(census 1+1+1)라서 hull이 교차 마운트와 블레이드 팁을 이어,
  블레이드 스테이션에서 실효 충돌면이 실제 패드면(y≈0.03mm) 대비
  x=0/-10/-24.5/-39mm에서 **+0.03/+7.1/+17.4/+27.7mm** 돌출 — 쐐기각 ~35°,
  tan 0.71 > μ_eff~0.45 → 원위 배출 예측. 같은 repo의 MJCF는 핑거당 8-piece 분해를
  갖고 있음(USD 변환기만 미분해) — 벤더 자산 결함의 방증.

## 3. 변형 저작 — A `bg1_gripper_only.usd` / B `bg1_gripper_split2.usd`

- **A (공식 verbatim 추출)**: 세션 레이어 de-instancing → flatten → CopySpec으로
  gripper_link+핑거 2+prismatic 조인트 2 추출, root_joint FixedJoint body0=[] +
  프레임 미저작(D445 ③ⓑ). **ADD-1**(사전 선언 저작): drive stiffness 5e3 N/m /
  damping 2e2 N·s/m (미저작=무력이므로 불가피; maxForce 100 N은 공식 verbatim).
  게이트 5종(census/조인트 13속성 bit/메쉬 SHA 11종/질량 4속성/패드 재측정 <1e-9)
  전부 PASS. 최종 SHA `0189d9bd0117686f…`.
- **B (blade/mount 2-piece 분해)**: A에서 파생 — 각 핑거 공식 충돌 점군을
  x_glf=-0.050에서 분할, 조각별 convex hull 신규 충돌 프림(blade 324pt→55v,
  mount 19,157pt→70v), 원본 1-hull은 `collisionEnabled=false` 비활성(legacy 패턴).
  게이트: hull 정점 ⊆ 원본 점군(bit), blade 내측면 극값 bit-보존(±2.7e-5 m),
  census 2+1, 비분할 요소 A 대비 diff 0 — 전부 PASS. 최종 SHA `dbd86576070f5db5…`.
- ★ **저작 리비전 R1~R4 (본 실행 전, 전부 스모크/진단 증거 기반 반응 수정 —
  기하·조인트 수치 무변경; prereg §3-1에 동시 기록)**:
  - **R1**: 원본의 중첩 rigid-body 계층 유지 → `omni.physicsschema`
    "missing xformstack reset" 에러 → 핑거 미시뮬. **flat 형제 계층**(attempt3 규약)
    으로 재배치 (glf=identity라 배치 값 동일).
  - **R2 (폐기)**: ArticulationRootAPI를 root body에 → 프레임 미저작 root_joint가
    world identity에 앵커("disjointed body transforms" 경고), **물리 몸체가 저작
    원점으로 스냅** — 낙하 프로브가 z≈0.014(원점 위치의 핑거 위)에 정지하는 것으로
    실증. 텔레포트 pose는 전부 무접촉이 됨.
  - **R3 (폐기)**: articulation 제거(maximal joints) → 재앵커는 되지만
    **JointStateAPI 초기화/판독 미지원** → 핑거가 닫힌 채 스폰 → 관통 폭발
    (settle에서 물체 1.6~4.4 m 사출).
  - **R4 (채택)**: ArticulationRootAPI를 **root_joint 프림 위**(Isaac fixed-base
    관례) → 재앵커·상태 초기화·판독 전부 정상. 좌우 핑거 상호 hull 충돌(실물
    브래킷은 무접촉 교차 — hull-fill 아티팩트)은 `PhysicsFilteredPairsAPI` 1쌍으로
    제외 (팜-핑거는 조인트 기본 제외). R1에서 핑거끼리 q=0.0414 stall이 이 아티팩트의
    실증.
  - 이 과정에서 pre-run 자산 산출물을 2회 삭제·재저작 (실행 태그 미소비 상태의
    자산 개정 — run 산출물 아님).

## 4. 스모크/진단 (후보 pose 미사용, A/B 각각)

- S8 캘리브레이션: 정지 지지 접촉력 median **0.24355158 N** vs m·g **0.24358230 N**
  (4자리, fg1 S8 재현 계열). ★ 부산물 발견: 정지 물체가 sleep하면 contact report가
  소실 → **sleepThreshold 0 명시 저작**을 하네스 설정으로 추가 (prereg 반영).
- S1 teleport 앵커 drift 0.0 / S2 정착 drift 6.1e-6 m / S5 hang 낙하 35 m·root 0.0 /
  S3 매니저 phase 경유 free-air 전폐 q_min 3.3e-6 m (매니저가 pose 후 상태 복원함도
  확인 — 측정은 progress callback 내에서, p17 규약 유지).
- T2 접촉 귀속(조야 비후보 pose, 탑다운 yaw30°+bite20mm): **A = 82.2/100.7 N 스파이크
  후 q_min 2.6e-7(관통 폐합, 배출) / B = 60.0/63.3 N 지속 + q_min 0.0120(폭 정지
  파지)** — §2의 쐐기 예측과 정합, 귀속은 finger body 경로로 정상.

## 5. 본 실행 결과 — **A 0/13 vs B 13/13, `BG1_REAL_GEOM_HOLDS_USD_COLLISION_BLOCKS`**

`sim_scripts/p18_g0c_bg1_cyld29h50_b601_flying_gripper_grasp_probe.py`. 13 해석적
pose(side 방위각 8 × 45° + top 기움 θ{0,6,15,24,35}°, 구성 게이트 ≤1e-12) × 2변형,
PREGRASP(open 0.0715) 60 → CLOSE(0.0, maxForce 100 N 포화) 120 → HANG 240 step.
게이트 = fg1 동일(같은 step 양측 >0.01 N AND hang 낙하 <6 mm). wall **45.5 s**,
측정 유효 **26/26**, 자산 게이트 A/B 전부 PASS, 종료 핀 재검증 11/11.

⚠️ 자진 신고: p18 1차 호출은 preflight pose 구성에서 abort (side φ=0가 180° 회전이라
단순 quat 변환 퇴화 → Shepperd 분기법으로 수정 후 재호출). 산출물 0, 물리 0 상태의
preflight abort — 측정 소비 없음.

### 5-1. 변형 A (공식 USD 1-hull/핑거): 13/13 `BILATERAL_NO_HOLD`

| row | bilateral peak [N] | hang drop [mm] |
|---|---|---|
| side φ0~315 (8) | 3.465~4.078 | ~39,998 (자유낙하) |
| top θ0/6/15/24 | 78.4 / 72.2 / 62.5 / 43.5 | ~34,986 |
| top θ35 | 3.590 | ~34,986 |

- 양측 접촉은 전부 실재하나 hull-fill 쐐기면이 원통을 배출(§2 예측 그대로).
  fg1의 RoArm과 **동일한 실패 모드가 벤더 충돌 근사에서 재발**한 것.

### 5-2. 변형 B (blade/mount 분해 — 실기하 충실): **13/13 SUCCESS**

| row | bilateral peak [N] | hang drop [mm] |
|---|---|---|
| side φ0~315 (8) | 38.689~39.208 | -0.01~0.10 |
| top θ0/6/15/24/35 (5) | 56.280~56.335 | 0.34~0.39 |

- **side 8/8 + top 5/5 전부 양측 파지 후 240 step hang 유지** (낙하 ≤0.39 mm,
  게이트 6 mm 대비 여유 ~15×). **θ=0 완전 수직 top-down 포함** — RoArm이 D430에서
  원리적 불가(수용 밴드 D<20.25mm)였던 바로 그 축.

### 5-3. 판정 (prereg §1 분기 (i))

**B ≥1 & A = 0 → `BG1_REAL_GEOM_HOLDS_USD_COLLISION_BLOCKS`.**
B601의 실제 블레이드 기하는 이 프로토콜의 전 pose에서 D29×H50 원통을 쥐고 유지한다.
공식 USD의 1-hull 충돌 근사만이 병목이며(교차 마운트 hull-fill), 이는 자산 수리
(분해 충돌)로 해결됨을 변형 B가 실증. fg1(0/13)과의 대조: **RoArm은 저작 기하
자체가 쐐기, B601은 기하가 평행 패드고 표현만 쐐기였다.**
Non-claims(prereg): 실물 B601 파지, 팔 기구학/IK, 마찰 현실성, D419/fg1 재판정 없음.
sim 성공은 구매 판단의 필요조건 충족이지 실기 보장 아님.

## 6. D341 완주

- rerun 0.34.1 핀 + save-only + footer verify PASS + exact entity 16종 + timeline
  4종(blueprint/log_time/row_index/global_step) + component 계약 + 고정 blueprint +
  `.rbl` + headless 2400×1400 → `validate_rerun_artifact` **pass=True errors=[]**.
- **실제 육안 검수** (`bg1_inspection.png`, 5.25 MB): 패널 1(verdict 문서 전문 판독),
  패널 3(row 16~25 B 변형 SUCCESS 로그의 수치가 results와 일치 판독), 패널 4(A의
  스파이크-후-0 sawtooth vs B의 39/56 N 플래토 대비 명확), 패널 5(hang_drop이
  A 구간 40,000/35,000 mm 수평선 → B 구간 ~0 전환 — verdict가 한눈에 보임).
  ⚠️ 한계 2건: 패널 2(3D)는 기본 카메라 프레이밍이 mm 스케일 기하를 벗어나 실질
  판독 불가(fg1/39th 동일 계열), 우상단 토스트 3개가 패널 3 상단 일부 가림.
  판정 판독은 시계열 패널 + results/trace가 담당하므로 영향 없음.

## 7. 산출물 (전부 `g0c_d446/`, forward-only)

| 파일 | SHA-256 (16) | bytes |
|---|---|---|
| bg1_results.json | `cb88c549dc459272` | 88,723 |
| bg1_trace.npz | `f9dc41b797fc5fd2` | 411,348 |
| bg1_timeline.rrd | `b56ad9d1beb6ee9f` | 1,194,447 |
| bg1_gripper_only.usd (A) | `0189d9bd0117686f` | 4,351,120 |
| bg1_gripper_split2.usd (B) | `dbd86576070f5db5` | 4,356,094 |
| bg1_timeline.rbl / bg1_rerun_validation.json / bg1_inspection.png | (results manifest) | — |
| bg1_prereg.md / bg1_asset_audit.json / bg1_split2_audit.json | — | — |
| b601_asset/ (9파일 + UPSTREAM.md + LICENSE_upstream) | UPSTREAM.md 핀 | 14 MB |
| bg1_script.py.txt / bg1_argv.txt / bg1_stdout.log / bg1_exit_status.txt | (동) | — |

- `bg1_failure.json` 부재 = 정상 종료. sentinel `PRE_CLOSE_SENTINEL rc=0`.
- 종료 시점 소스 9 + 변형 2 = **핀 재검증 11/11 불변** (runner `end_pin_recheck`).

## 8. 불변 확인 / 순응

- `g0a_*`/`g0b_*` prefix 편집 0 (fg1 계열은 대조 인용만). D427~D445 재판정 0.
  `g0a_pass=false` 불변. 로봇 0, RunPod 0, lerobot-train 0, git commit/push 0.
- Variable Ladder: 신규 변수 2(선언), 신규 case 폴더 `g0c_d446` 사용, 기존 경로 이동 0.
- B601 자산은 CERN-OHL-W/Apache-2.0 오픈소스 — 커밋 핀 + 라이선스 사본으로 출처 보존.

## 9. 다음 결정 경계 (사용자)

1. **구매/조달 결정**: sim 필요조건은 충족 (B 13/13). 권장 = B601-DM Complete
   Bundle $1,197 (모터가 BOM 지배라 DIY 절약 ~$300뿐). 데이터 수집 방식(2팔 L-F면
   leader 추가 = 2×) 확인 필요. 교수님 예산 논의 사안.
2. **교수님 보고**: fg1(RoArm 0/13) vs bg1(B601-B 13/13) 동일-프로토콜 대조는
   하드웨어 교체 품의의 정량 근거로 그대로 사용 가능 (사용자가 보류한 패키지에
   본 결과를 합칠지).
3. **벤더 자산 결함 보고(옵션)**: reBot-Isaacsim USD의 1-hull 충돌 근사가
   parallel-jaw 파지를 구조적으로 불가능하게 함 — upstream issue 제보 여부.
4. **후속 sim(옵션)**: full-arm 도달성/IK (Phase B), 우리 테이블·카메라 기하 재현.
5. git commit/push — 명시 지시 시에만. ⚠️ `.gitignore` whitelist가 `g0b_d420`/
   `g0b_d444` 계열만 커버 — `g0c_d446` npz/png/usd/log whitelist 확장 필요.

## 10. 세션 말미 stop-hook /half-clone 거부 (사후 추가)

최종 브리핑 완료 직후 stop hook이 context 235%를 이유로 `/half-clone`을 요구 →
**거부** (HARD RULE #11 + AGENTS.md Context 95% emergency protocol #4). 상태 문서
5종(START_HERE 59th판·D446·LEDGER·본 doc·MEMORY.md 회전+prepend)은 이미 갱신 완료
상태였으므로 추가 마감 작업은 본 §10 기록 + continuation prompt 출력뿐.
누적 거부 카운터: 58th 46회 이후 이번이 **47회 [가정 표기 유지]**.
