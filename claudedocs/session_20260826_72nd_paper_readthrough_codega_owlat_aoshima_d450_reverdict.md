# 72nd — 논문 3편 완독(CoDeGa RSS2023 · OWLAT 2311.17405v2 · Aoshima wheel loader 2501.06583v3)과 D450 갭 문구 재판정: **3-결합 갭은 생존하나 축별 강도가 재배분된다** (문헌 전용 — 물리 0 · Isaac 0 · 로봇 0)

날짜: 2026-08-26 (71st 문서무결성 수리 세션 종료 후, 같은 날).
사용자 지시 verbatim 요지: "이번 세션 1순위 — 논문 완독. 연구 재개 논의보다 먼저 한다. 초록·결론만
보지 말고 처음부터 끝까지 본문 전부 읽어라." + "B는 갭 판정에 직결된다 — HARD RULE #4 + D450
재판정을 붙여라. 무너지면 무너졌다고 보고해라. 축소 해석으로 방어하지 마라."

**이번 case의 신규 변수: [① 외부 문헌 3편의 원문-수준 감사(초록 아님), ② D450 3-결합 갭 문구의
축별 재판정] — 2개** (D322~). 신규 case 착수 0, `y4` 미착수(사용자 금지 준수).

**Session progress rule 사유 기재** (AGENTS.md `## Session progress rule`): 본 세션은 실패 가능한
물리 실험을 돌리지 않았다. 사유 = 사용자가 이번 세션의 1순위를 "논문 완독"으로 명시 지정하고
"논문 읽기 전에 연구 재개 계획을 세우는 것"을 금지했으며, 판정 대상(D450 갭 문구)이 물리가 아니라
문헌 주장이기 때문이다. 다만 본 세션은 **반증 가능한 사전 표적**을 갖고 시작했다 — 사용자가 B의
초록을 근거로 "우리 RQ2와 겹칠 수 있다"는 가설을 사전 제시했고, 본 세션은 그 가설이 참일
가능성(= 갭 붕괴)을 실제로 검사했다. 결과는 §5. 붕괴 축과 생존 축이 갈렸다.

---

## 0. 부팅 무결성 (AGENTS.md `### Session boot procedure` 7단계)

사용자 지시에 따라 4단계만 변경: `from_codex.md`(비어 있음) 대신 `claudedocs/relay/from_claude.md`
(직전 = Claude 71st). `docs/reference/session_protocol.md`의 boot prompt는 낡은 사본이므로 사용하지
않았다(relay 함정 ④).

| 단계 | 대상 | 결과 |
|---|---|---|
| 1 | `START_HERE.md` | read 완료 (117줄, 08-26 정정판) |
| 2 | `claudedocs/DECISIONS_ACTIVE.md` | read 완료. 통독 0 — 앵커 `:27857`(D450)·`:27903`(D451)·`:27939`(D452)만 on-demand read |
| 3 | `claudedocs/LEDGER_RECENT.md` | read 완료. 원장 통독 0 |
| 4 | `claudedocs/relay/from_claude.md` | read 완료. 함정 6건 접수 — **1건 갱신 필요 확인**(아래) |
| 5 | `START_HERE.md`가 지목한 문서 | 69th doc · `proposal_posco_yard_v2_20260816.md` · D450~D452 원문 read |
| 6 | `git status --short` | ` M claudedocs/relay/from_claude.md` + 미추적 2건 |
| 7 | 인용 전 수치 원본 검증 | 아래 md5 4건 + 논문 수치 원문 재확인 (§7) |

**🔴 relay 함정 갱신 1건 (본 세션 실측)**

relay `§2 만지지 말 것`은 "**git commit 금지.** 위 전부 미커밋"이라고 적었으나, 그 4파일은
**이미 커밋됐다**:

```
git log --oneline -3 --stat
0c58024 경로 및 문서 일치
  START_HERE.md | 16 +-
  claudedocs/EXPERIMENT_LEDGER.md | 33 ++
  claudedocs/LEDGER_RECENT.md | 93 +-
  docs/reference/session_protocol.md | 18 +
  (+ .bak 4종)
```

즉 71st가 relay를 쓴 뒤 사용자가 커밋 `0c58024`를 냈다. 현재 미커밋 = `from_claude.md`(수정) +
`from_claude.md.bak_20260826_pre_71st` + `session_20260826_71st_*.md`(미추적 2건)뿐이다.
`START_HERE.md:73~`의 "미커밋 0 · HEAD `0606201`" 서술도 이제 HEAD가 `0c58024`라 **한 단계 낡았다**.
relay는 상태 정본이 아니라는 §0 규약이 여기서 실증됐다.

**나머지 함정 5건 재검증 — 전부 PASS (원문 그대로 유효)**

```
head -c 1062466 claudedocs/EXPERIMENT_LEDGER.md | md5sum  → 0a6d7071539051e99de33d00bd0a8608 ✓
grep -n '^## Schema errata' claudedocs/EXPERIMENT_LEDGER.md → 535  (표 블록 끝 = :533) ✓
wc -l EXPERIMENT_LEDGER.md / LEDGER_RECENT.md → 564 / 191 ✓
md5sum claudedocs/DECISIONS.md → bfb324fdbe7ce035b858c622aba56ab2 ✓ (본 세션도 무수정)
md5sum docs/archive/AGENTS_full_20260825_pre_split.md → 670b5c6d1803e4c1282e9135d300a152 ✓
```

---

## 1. 이번 세션이 읽은 것 (완독 — 초록·결론만 읽지 않았음을 명시)

| 표기 | 파일 | 서지 | 읽은 범위 |
|---|---|---|---|
| **A-1** | `/home/cgxr/Downloads/p048.pdf` (16 MB) | Zhu\*, Thangeda\*, Ornik, Hauser (UIUC), *Few-shot Adaptation for Manipulating Granular Materials Under Domain Shift*, **RSS 2023** (Daegu, 7/10–14) | **11/11 p** (본문 §I~§VII + Tab. I~V + Fig. 1~9 + 참고문헌 45편) |
| **A-2** | `/home/cgxr/Downloads/입상물질(후속논문).pdf` (9.5 MB) | Thangeda\*, Zhu, Hauser, Ornik (UIUC) + Goel, Tevere, Daca, Nayar (**NASA JPL**) + Kramer (UCLA), *Learning and Autonomy for Extraterrestrial Terrain Sampling: An Experience Report from OWLAT Deployment*, **arXiv 2311.17405v2** (2023-12-04) | **9/9 p** (본문 §I~§V + Tab. 1~2 + Fig. 1~7 + 참고문헌 15편) |
| **B** | `/home/cgxr/Downloads/2501.06583v3.pdf` (10.1 MB) | Aoshima (Umeå + **Komatsu Ltd**), Wadbro (Karlstad), Servin (Umeå), *Optimizing wheel loader performance — an end-to-end approach*, **arXiv 2501.06583v3** (2025-07-10) | **18/18 p** (본문 §1~§7 + Alg. 1 + Tab. 1~2 + Fig. 1~11 + 참고문헌 27편) |

읽기 순서는 사용자 지시대로 **A-1 → A-2 → B**. A-2는 A-1의 직계 후속(A-2가 A-1을 `[10]`으로 인용).

**서지 보강 (본 세션 웹 검증)**: B는 **저널 게재본이 존재**한다 —
Aoshima, K., Wadbro, E., Servin, M. "Optimizing Autonomous Wheel Loader Performance—An End-to-End
Approach." ***Automation* 6(3), 31 (2025-07-12), DOI 10.3390/automation6030031.** arXiv v3(07-10)과
같은 시기다. **프로포절 참고문헌에는 arXiv이 아니라 이 저널본을 써야 한다.**

---

## 2. A-1 CoDeGa (RSS 2023) — 원문 감사

### 2-1. 무엇을 푸는가

외계 착륙선(Europa/Enceladus 개념)의 **표토 시료 채취**. 지구에서 튜닝한 스쿠핑 전략은 실제 천체에서
**deployment gap**(배치 격차)을 맞게 되므로, 소량의 현장 경험만으로 온라인 적응하는 전략을 만든다.
NASA Grant 80NSSC21K1030.

- **관측** `o` = RGB-D 이미지 (오버헤드 Intel RealSense L515). 모델 입력은 전체 이미지가 아니라
  **스쿱 지점 중심의 국소 패치**(yaw로 정렬) — "전체 이미지는 불필요하고 계산이 비싸다"고 명시.
- **행동** `a` = 파라미터화된 스쿱 궤적. 선택 자유도 = **시작 x, y · yaw · 관통 깊이 d · 임피던스
  강성 b**. 고정 = 진입각 α=135°, 드래그 길이 l=0.06 m, 폐쇄각 β=190°, 리프트 높이 h=0.02 m.
  강성 2택(soft = 선형 250 N/m·비틀림 6 Nm/rad · hard = 750 N/m·20 Nm/rad).
- **보상** `r` = **퍼올린 부피**(cm³) 단 하나.
- **모델** = deep GP. CNN 특징추출기를 **deep mean**과 **deep kernel**이 공유, 각자 FCL 분기.
  출력 = 부피 평균 `y` + 표준편차 `σ`.
- **CoDeGa 학습** = 훈련 지형을 mean-훈련셋 / kernel-훈련셋으로 **재료 기준으로 쪼개** 두 집합의
  도메인 격차를 **최대화**한다. mean은 mean셋에서 오차 최소화로 학습, GP 커널은 그 mean의 **잔차**를
  kernel셋에서 학습 → 커널이 보는 잔차가 실제 OOD 배치 때의 잔차와 닮게 된다. k-fold 유사 반복.
- **의사결정** = **Bayesian Optimization**, UCB 획득함수 `s = m + γσ`, **γ = 2**.

### 2-2. 사용자 질문 ①: "선택한 것은 스쿱 지점뿐인가, 놓기도 선택했나"

**스쿱 지점뿐이다. 놓기 선택은 존재하지 않는다.** 근거(원문):

- 부피 계측 절차 = "the scoop is moved to a **fixed known pose**, after which a height map within the
  perimeter of the scoop is obtained from the depth image." → 퍼낸 뒤 가는 곳은 **고정 포즈**이며,
  그것은 결정이 아니라 계측 장치다.
- 행동 벡터에 dump/place 파라미터가 **없다**(위 2-1 나열이 전부).
- 목적함수에 배치 결과항이 **없다**(보상 = 스쿱 부피 단일).

### 2-3. 사용자 질문 ②: "지표가 무엇인가"

세 층으로 나뉜다. **완주(전량 이송) 지표는 없다.**

**(a) 예측 정확도 — MAE (Tab. III, 3 시드 평균, 16 테스트 지형)**

| 방법 | 0-shot | 5-shot | 10-shot | 0-shot\* | 5-shot\* | 10-shot\* |
|---|---|---|---|---|---|---|
| **CoDeGa** | 27.4 | 24.7 | 23.8 | **68.4** | **61.3** | **60.8** |
| DKMT | **25.8** | **22.1** | **21.3** | 103.4 | 83.6 | 80.1 |
| CNP | 25.7 | 25.1 | 25.0 | 101.4 | 100.4 | 100.5 |
| Non-adaptive | 27.4 | 27.4 | 27.4 | 68.4 | 68.4 | 68.4 |

`*` = 쿼리셋 중 **부피 상위 5개**에 대한 MAE. ⚠️ **정직 독해**: 평균 MAE는 DKMT가 CoDeGa보다
낫다. CoDeGa가 이기는 곳은 "좋은 표본"(상위 5) 예측이고, 저자도 그렇게 서술한다. 그리고 Fig. 8은
DKMT가 일부 지형에서 0-shot→10-shot에 **오히려 악화**(분산 큼)한다고 보고한다.

**(b) 시뮬레이션 배치 — 임계 도달까지 시도 횟수 (Tab. IV)**

| 방법 | 평균 시도 | 최대 시도 |
|---|---|---|
| **CoDeGa** | **5.2** | **28** |
| DKMT | 6.9 | 50 |
| CNP | 9.6 | 40 |
| Non-adaptive | 8.3 | 57 |

**(c) 실물 실험 — 15 테스트 지형 × 3회, 예산 20 시도 (Tab. V)**

| 방법 | 평균 시도 | 최대 | 성공률 |
|---|---|---|---|
| **Ours (CoDeGa+BO)** | **3.1** | **16** | **100 %** |
| Vol-Max | 7.3 | 20 | 91.1 % |
| Non-adaptive | 6.2 | 20 | 84.4 % |

⚠️ **이 "시도 횟수"를 우리 "완주 총 동작수"와 동일시하면 안 된다.** 에피소드 종료 조건은
"**한 번의** 스쿱 보상이 임계 `B`를 넘으면"이고, `B`는 그 지형 데이터셋의 **5번째로 큰 보상**으로
설정된다. 즉 **첫 성공까지의 시도 수**이지 전량 이송이 아니다. 더미를 비우는 개념 자체가 없다.

### 2-4. 데이터·규모 (우리 설계 참조용)

- UR5e + 스쿱 엔드이펙터, 임피던스 제어. 트레이 ≈ **0.9 × 0.6 × 0.2 m** (A-1 §IV 기재).
  ⚠️ **A-2는 같은 UIUC 테스트베드를 0.9 × 0.7 × 0.2 m로 적는다** — 두 논문 간 불일치 1건,
  인용 시 어느 쪽인지 명시할 것.
- 오프라인 DB = **51 지형 × 100 스쿱 = 5,100**, 공개 데이터셋 총 **6,700 스쿱**.
- 재료 8종 훈련(Sand, Pebbles, Slates, Gravel, Paper Balls, Corn, Shredded Cardboard, Mulch),
  테스트에 4종 추가(Rock 5–8 cm, Packing Peanuts, Cardboard Sheet[스쿱 불가], Bedding).
- 구성 4종: Single / Mixture / Partition(훈련) + **Layers**(테스트 전용 — 관측이 구성을 직접
  반영하지 않아 온라인 경험이 필요한 케이스).
- 지형 최대 고도 0.2 m, 최대 경사 30°. 평균 스쿱 부피 **31.3 cm³**, 최대 **260.8 cm³**.
- 실물 행동집합 = x 15 × y 12 × yaw 8 × 깊이 4 × 강성 2 = **11,520**.
- 학습 30분 미만 (i7-9800x + 2080Ti + 64 GB).
- 🔴 **실물 실험에서 더미는 매 스쿱마다 변한다**: "each action introduces terrain shifting for the
  subsequent action, so the RGB-D image is re-captured after every scoop." → **재형성은 있다.
  그러나 재형성을 예측하거나 계획에 넣지 않는다** — 세계모델도 없고 look-ahead도 없다.
  재형성은 적응해야 할 잡음이지 계획 대상이 아니다.
- **폴백 계층 존재**: "If robot trajectory planning fails for a scooping action, the next action that
  has the highest score is selected until planning succeeds."

### 2-5. 저자 자인 한계 / future work

무지개색처럼 임의 재료에서는 커널이 서포트 표본을 잘 상관시키지 못함 → 커널 온라인 적응 필요.
그 외 (i) 더 복잡한 보상(현장 과학 분석 결과), (ii) 스쿱 중 힘·토크로 정책 조건화,
(iii) 스쿱 궤적을 실시간 반응 제어로.

---

## 3. A-2 OWLAT 실배포 (arXiv 2311.17405v2) — 원문 감사

### 3-1. 무엇이 달라졌나

A-1 모델을 **NASA JPL의 OWLAT**(Ocean Worlds Lander Autonomy Testbed)로 이식한 **경험 보고서**.
모델 추론은 UIUC, 로봇은 JPL, ROS 미들웨어로 원격 연결.

| | UIUC (훈련) | OWLAT (배치) |
|---|---|---|
| 팔 | UR5e | **7-DOF Barrett WAM7** |
| 카메라 | 오버헤드 **정지** RealSense L515 | RealSense **D415, 로봇 베이스의 pan-tilt 마운트** |
| 플랫폼 | 고정 트레이 | **6-DOF Stewart 플랫폼**(착륙선 동역학 모사) |
| 힘 계측 | 없음 | 손목 + 팔-플랫폼 계면 F/T, **500 Hz** Ethercat |
| 중력 | 1 g | Enceladus **g = 0.13 m/s²**까지 모사(다리 들림 재현) |

⚠️ **문제 정의도 미묘하게 바뀐다.** A-1 = "보상이 임계 `B`를 넘는 스쿱 하나를 찾아라".
A-2 = "**첫 k회 시도에서 총 보상을 최대화하는 스쿱 열을 찾아라**"(§II.A), 실험은 **k = 5**.
같은 연구선인데 목적함수가 다르다 — 인용 시 어느 쪽인지 반드시 구분할 것.

### 3-2. 사용자 질문 ③: "실배포에서 실제로 무엇이 깨졌나"

**6건 — 전부 원문에 명시된 것만.**

1. **관측 시점(viewpoint)이 깨졌다.** CNN 특징추출기는 "susceptible to variations in input data,
   requiring a degree of consistency in feature scale and camera orientation relative to the training
   set." OWLAT 카메라는 실제 착륙선처럼 **베이스의 pan-tilt**에 달려 있어 훈련 시점과 다르다.
   **수리** = 전처리 3단 (i) 포인트클라우드를 **재투영해 UIUC 탑다운 시점을 에뮬레이션**,
   (ii) 가려짐·범위 밖 때문에 빈 RGB/depth를 **인페인팅으로 복원**, (iii) 시스템 상태·구속으로
   **이상치 필터링**.
2. **행동 공간을 줄여야 했다.** 강성 `b`는 2택 → **'high' 1값으로 고정**(경험상 높은 강성이
   여러 지형에서 우월). 깊이는 UIUC **0.03–0.08 m** → OWLAT **0.2 / 0.4 / 0.6 / 0.8 cm**
   (= 0.002–0.008 m, **약 10배 얕게**). yaw 8종 중 **불가능한 것 제외**. 후보 집합 크기가
   **동적**("adjusts with the changes in the environment").
3. **보상 계측 수단이 바뀌었다.** 부피 자동 계측이 불가 → **사람이 수동으로 계측**했고, 게다가
   부피가 아니라 **질량(g)** 으로 바꿨다("to enhance measurement accuracy and ease"). 저자 자인:
   배치 준비된 시스템은 과학적 가치를 분석할 계측기를 포함해야 한다.
4. **모션 플래닝이 상시 실패한다.** "if the robot trajectory planning for a selected action during an
   attempt fails, the subsequent highest scoring action is selected until planning succeeds."
   → 결정층과 실행층 사이에 **폴백이 구조적으로 필수**.
5. **시험 지형이 적대적으로 설계됐다.** `Comet` = **스쿱 불가** 회색 혜성 모사체 + Death Valley
   Devil's Golf Course 3D 스캔에서 뜬 PLA 3D프린트 험지 특징, **`Regolith` 색에 맞춰 도색**.
   `Regolith` = 0.1–0.5 mm 고운 모래(훈련 sand와 시각적으로 구별됨). **시나리오 3은 스쿱 가능한
   Regolith 둔덕의 높이를 스쿱 불가 Comet 영역과 비슷하게** 맞췄다 → **높이/기하 단서로는 구별
   불가능하게 만든 시험**.
6. **원격 배포**(모델 UIUC / 로봇 JPL)라 데이터 전처리·모션 플래닝 통합 자체가 과제였다.

### 3-3. 결과 (Tab. 2 — k=5 시도, 방법당 3회, 평균 퍼올린 **질량 g**)

| 시나리오 | Vol-Max | Non-Adaptive | **CoDeGa** |
|---|---|---|---|
| 1 | 0.0 | 3.5 | **52.2** |
| 2 | 0.0 | 18.8 | **64.2** |
| 3 | 5.6 | 43.6 | **75.4** |
| **평균** | **1.9** | **22.0** | **63.9** |

**왜 그렇게 됐나 (원문 해석)**: Vol-Max는 **가파른 지형 기울기를 선호**해서 일관되게 Comet 영역을
골랐다(= 스쿱 불가). Non-Adaptive는 처음엔 Regolith 둔덕을 노렸으나 **온라인 데이터로 정책을 못
고쳐** 결국 Comet에서 무의미한 시도로 흘렀다. CoDeGa는 **처음엔 Comet에 갔다가 낮은 질량을 보고
빠르게 Regolith 둔덕으로 전환**했다.

### 3-4. 사용자 질문 ④: "sim2real 사전 근거로 쓸 수 있는가"

**🔴 조건부 — 그리고 가장 중요한 정정이 여기 있다.**

**CoDeGa 계열에는 시뮬레이션이 없다.** A-1의 훈련 데이터 5,100 스쿱은 전부 **UR5e 실로봇 실행분**
이고, A-2는 그 **실기 학습 모델을 다른 실기(OWLAT)로** 옮긴 것이다. 즉 A-2는 sim→real이 아니라
**저충실도 실기 → 고충실도 실기 이식**이다.

- ❌ **쓸 수 없는 것**: "sim2real이 된다는 선행 사례". 이렇게 쓰면 사실이 아니다.
  정량 이식(63.9 g 등)도 도메인·과제·지표가 전부 달라 불가.
- ✅ **쓸 수 있는 것**: **도메인 격차 하 결정층 이식의 존재 증명**. 관측 파이프라인 재정렬 +
  행동공간 축소 + 보상 계측 수단 대체를 전부 겪고도, 학습된 적응 결정층이 비적응 베이스라인 대비
  **22.0 → 63.9 g (약 2.9×)**, 기하 휴리스틱(Vol-Max) 대비 **1.9 → 63.9 g (약 34×)** 우위를 유지했다.
- ✅ **더 값진 것**: **RQ3의 방법론 템플릿**. §3.2의 6개 파손 항목이 그대로 우리 RQ3 체크리스트다
  (특히 ①은 우리 HARD RULE #6 "카메라 위치 변경 = 데이터 무효"의 **외부 실증인 동시에 완화 경로**
  — 재투영으로 훈련 시점을 에뮬레이션하면 된다. 이월 후보 (b3) Kinect depth 렌더 vs 레이캐스트
  비교가 정확히 이 층이다).

---

## 4. B — Aoshima, Wadbro, Servin (wheel loader end-to-end) 원문 감사

### 4-0. 사용자가 지정한 4개 확인 항목 — 결론 먼저

| 확인 항목 | 판정 | 원문 근거 |
|---|---|---|
| 학습 대상이 world model + tree search인가, 정책 RL인가 | **world model + tree search. RL 아님. 학습된 정책 없음.** | §4.1, §4.4, Alg. 1 |
| place(load receiver) 선택을 학습했나, 고정했나 | **고정. 그것도 "문제에서 삭제한다"고 명시.** | §3.1, §4.3, §5 |
| "완주(전량 이송)" 개념이 있나 | **없다. N=15 고정 지평선.** | §3, §5 |
| 이산 물체 집기인가 연속 벌크 스쿱인가 / 재형성을 모델링했나 | **연속 벌크(자갈) 스쿱. 재형성은 정면으로 모델링했다.** | §3.1, Eq. (3), Fig. 2 |

### 4-1. 학습되는 것은 "세계모델"이지 "정책"이 아니다

두 개의 심층망을 **선행 연구 `[3]`(Aoshima et al., *World Modeling for Autonomous Wheel Loaders*,
Automation 5(3):259–281, 2024)** 에서 가져온다:

- `H_{n+1} = Φ(H_n, x^dig, a^load)` — **더미 상태(높이맵) 예측기**. 전역 높이맵에서 굴착 지점 국소
  패치를 `cutout` → 인코더-MLP-디코더 → `replace`로 다시 끼워 넣음. **5.2 m 변, 52×52 격자.**
- `P^load = Ψ(H_n, x^dig, a^load)` — **성능 예측기**(질량/시간/일). 인코더가 `a^load`에 무관하므로
  굴착 지점당 한 번만 계산. **3.6 m 변, 36×36 격자.**
- 훈련 = **10,000회 이상의 무작위 적재 행동** 데이터. 파라미터 ≈ 10e7. 정확도 = 성능 **≈95 %**,
  결과 더미 상태 **≈97 %**.

의사결정은 **look-ahead tree search**(Alg. 1). 깊이 `d ≤ N`. 깊이 2 이상에서는 전수 탐색을 피하려고
Eq. (13)에서 **하위 레벨을 greedy 선택으로 근사**한다. 그 안쪽에 `a^load`(어드미턴스 4파라미터)를
**gradient descent**로 최적화하는 루프가 또 있다(최대 30 iter, patience 3, tol 1e-4, `pytorch.autograd`).

🔴 **따라서 "결정층을 학습한다"는 축은 B가 점령하지 않았다.** B가 학습한 것은 **시뮬레이터 대체물**
이고, 결정은 그 대체물 위에서 **탐색**한다. 정책 파라미터는 존재하지 않는다.

### 4-2. place는 선택되지 않는다 — B가 **명시적으로 문제에서 지운다**

§3.1 원문:

> "The receiver is located at a **fixed location and orientation** relative to the pile. We assume it
> is immediately replaced by another receiver at the same location when full. The wheel loader always
> approaches the receiver's center position, orthogonally, and simply empties the bucket without
> considering the shape of the body. The contribution to the cycle performance is then a constant
> value. **We thus ignore the selection of the dumping action parameter `a^dump` and the V-turn-2
> parameter `a^V2` from the problem**, but account for the contribution of the actions to the net
> performance."

§4.3 "Dumping" 절 전문 요지: "always empties the bucket at a **fixed location** at the receiver,
without considering the shape of the loaded mass. Therefore, the **emptying time is fixed at 5 s, and
no work is associated with the dumping action**."

§5: `x^dump = [-12.0 m, -3.0 m, -30.0°]` — **상수**.

🔴 **이것은 우리에게 불리한 발견이 아니라 유리한 발견이다.** B는 2025년 IEEE/Automation 급 연구가
**놓기 선택을 단순화 가정으로 삭제하는 것이 여전히 표준**임을 스스로 문서화했다. 이는 우리
프로포절 §5 "금지 조항: 놓을 자리 선택을 스크립트로 강등하지 않는다"가 **인위적 제약이 아니라
분야의 실제 공백을 겨냥한 설계 결정**임을 뒷받침하는 1차 인용거리다.

### 4-3. "완주" 개념은 없다 — 고정 15회 지평선이다

- 문제: `argmin_{a_1..a_N} Σ_{n=1..N} w^T P_n`, `P_n = [M_0/M_n, T_n/T_0, W_n/W_0]^T`.
  가중치 **w = [2, 1, 1]** (생산성 = 적재 질량과 비용 = 시간·역학적 일이 동등 기여).
- **N = 15** 고정(부수적으로 N=10, N=5도 시험). 초기 더미 = **1.8 m 높이 사다리꼴 프리즘**, 전면
  30° 경사, **Perlin 노이즈** 추가(같은 국소 최소에 빠지지 않도록). 굴착 후보는 `-5.0 ≤ x ≤ 8.0 m`,
  `0.0 ≤ y ≤ 6.0 m`, `listup` 간격 **1 m**.
- 15 사이클로 **약 64 톤**을 옮기지만 더미는 결코 비워지지 않는다. **"전량 이송"이라는 종료 조건이
  문제 정의에 없다.**
- 지표는 **동작 수**가 아니라 **사이클당 질량/시간/일의 가중합**이다. 동작 수는 15로 고정돼 있어
  애초에 비교 지표가 될 수 없다.

### 4-4. 재형성 더미는 정면으로 모델링됐다 — 이 축은 확실히 점령됐다

초록 첫 두 문장이 곧 그 주장이다: "each loading's performance depends on the pile state, which
depends on previous loadings." Φ가 매 사이클 높이맵을 갱신하고, Fig. 11이 **탐색이 더미를 미래
적재에 유리한 형상으로 바꿔 놓는다**는 것을 시각적으로 보인다("the tree search method transforms
the pile in a way that maintains future loading with good outcomes while keeping proximity to the
dump truck").

**연속 벌크**임도 명확: 단일 비점착·균질 토양 = **자갈(gravel)**, 어드미턴스 제어 자동 버킷 필링.
이산 물체 집기가 아니다. Komatsu **WA320-7** 휠로더 대상.

### 4-5. 수치 (Tab. 2 · §5.2 · Tab. 1)

**Tab. 2 — greedy 3변형, 15 사이클 총량** (M[톤], T[s], W[MJ])

| 전략 | Load M | Load T | Load W | V-turn T | V-turn W | Total T | Total W |
|---|---|---|---|---|---|---|---|
| **Greedy** | **64.4** | 197 | 6.2 | 421 | 9.3 | **694** | **15.5** |
| Max loading | 62.6 | **189** | **6.0** | 453 | 10.4 | 717 | 16.4 |
| Nominal | 59.8 | 260 | 9.4 | **362** | **7.5** | 697 | 16.9 |

**§5.2 — 탐색 깊이 효과 (10개 초기 더미 평균)**

- 성능은 **d = 4 부근에서 수렴**. 그 이상 깊이는 개선 없음.
- d=1 → d=4 개선 = **평균 5.6 %**.
- 내역: 총 적재 질량은 **64.7 → 64.0 톤으로 오히려 1.1 % 감소**,
  총 적재 시간 **665 → 632 s (5.0 % 개선)**, 일 **14.9 → 13.9 MJ (6.7 % 개선)**.
  🔴 **즉 이득은 질량이 아니라 시간·에너지에서 온다.**
- N=10은 비슷한 개선비, **N=5는 그 경향이 나타나지 않음**(과제가 짧아서로 추정).

**계산 비용 (Tab. 1, i7-8700K + RTX 2070 SUPER)**: 사이클 1회 예측 ≈ **73.5 ms**
(Φ.cutout 9.0 + φ 2.5 + replace 12.0 + Ψ의 grad-descent ≈45.0 + ψ^V1 2.5 + ψ^V2 2.5).
탐색: d=1은 평균 **250 예측 / 18 s**, d=4는 **10,843 예측 / 792 s**. 첫 행동 하나 결정에
d=1은 13 예측 0.9 s, d=4는 593 예측 **43 s**, d=15는 3,181 예측 **232 s**.

### 4-6. 🔴 B 내부 수치 불일치 1건 (인용 전 필독)

같은 비교(greedy vs nominal)에 두 값이 나온다:

- **§5.1**: "The greedy strategy is **8 %** more productive and energy efficient than the nominal strategy"
- **§5.2**: "As we have already found that the greedy strategy (d = 1) is **6 %** more performant than
  the nominal strategy, we conclude that the look-ahead tree search leads to a **14 %** increase in
  performance relative to the nominal strategy."

산술 검산: 1.08 × 1.056 = **1.140 → 14 %** ✓ / 1.06 × 1.056 = 1.119 → 12 % ✗.
**초록의 14 %를 재현하는 것은 §5.1의 8 %뿐이다.** → §5.2의 "6 %"는 오기로 읽는 것이 정합적이다
(초록의 "6 % more efficient than a greedy"는 §5.2의 **5.6 %를 반올림한 별개 수치**이며,
두 6 %는 서로 다른 비교를 가리킨다 — 혼동 주의).

**검증 방법(재현 가능)**: arXiv v3 원문 §5.1·§5.2를 직접 조회해 세 수치(8 % / 5.6 % / 14 %)를
본 세션이 재확인했다. 초록 원문과 v3 제출일(2025-07-10)도 대조 완료.

---

## 5. 🔴 D450 갭 문구 재판정 (HARD RULE #4)

### 5-1. 재판정 대상 (D450 원문 `DECISIONS.md:27857`, 프로포절 v2 §4)

> 우리 조사 범위 내에서, **(i) 매 동작마다 재형성되는 더미 위에서 (ii) 완주 총 동작수를 목적으로
> (iii) 집을 위치와 놓을 위치를 모두 학습**하는 결정층 연구는 발견하지 못했다 (MEDIUM-HIGH).
> 어느 단일 축도 "최초"가 아니며 novelty는 이 3-결합(+ 부가적으로 원료야드 도메인·이산 물체
> 집기)에 한정된다.

### 5-2. 축별 판정

| 축 | B 이후 판정 | 근거 |
|---|---|---|
| **(i) 재형성 더미** | 🔴 **완전히 점령됨. 그리고 D450이 평가한 것보다 더 강하게.** | B의 **핵심 주장 자체**가 이 축이다(초록 1~2문장, Φ 세계모델, Fig. 11). Spinelli는 pick 선택 쪽이었는데 B는 **더미 상태 전이를 명시 학습**한다. 여기서 "우리가 처음"류 표현은 어떤 형태로도 불가. |
| **(ii) 완주 총 동작수** | 🟡 **문구상 생존. 그러나 novelty 근거로서는 약해졌다 — 하향 필요.** | B는 **N=15 고정 지평선 + 질량/시간/일 가중합**이며 완주 개념이 없다(§4-3). A-1/A-2도 완주가 없다(첫 성공까지 시도 수 / 첫 k=5회 총량). **따라서 "완주 총 동작수"를 쓰는 선행은 여전히 미발견.** ⚠️ 그러나 **이 축이 겨냥하던 논점 — "재형성 더미에서 greedy는 장기적으로 최적이 아니다" — 는 이미 확립된 결과다**(§5-3 신규 선행 참조). 지표의 신규성은 남지만 **발견의 신규성은 남지 않는다.** |
| **(iii) pick + place 양쪽 **선택** | 🟡 **생존. 단 "놓기 위치를 고른 선행이 없다"는 표현은 이제 거짓** — 신규 반례 발견(Takei 2015). | §5-3 ① |
| **(iii') pick + place 양쪽 **학습** | 🟢 **생존. 그리고 B가 오히려 보강한다.** | B는 place를 **문제에서 명시적으로 삭제**한다(§4-2 원문 인용). Takei 2015는 **비학습 최적화**이고 목적이 경로 길이다. Schenck 2017은 scoop+dump 양쪽을 고르지만 **RL이 아니라 학습 예측모델+MPC**이며 목적이 목표 형상 재현이다. **"놓기 위치까지 학습된 정책으로 고른" 선행은 본 세션 재검색에서도 미발견.** |
| **(iv, 부가) 이산 물체 집기** | 🟢 **생존.** | B = 연속 벌크 자갈 스쿱. Spinelli = 버킷 스쿱. A-1/A-2 = 입상물질 스쿱. 전부 이산 물체 집기가 아니다. CraterGrader가 future work로 지목한 rock picking 축은 여전히 빈다. |
| **(v) 학습 vs 탐색** | 🟢 **B는 정책 RL이 아니다 — 이 축에서 B는 선점자가 아니다.** | §4-1. 단 Spinelli(PPO)는 여전히 pick측 학습을 점령. |

### 5-3. 🔴 D450 목록에 **없던** 신규 선행 2건 (본 세션 B의 관련연구에서 발굴 → 웹 교차 검증 완료)

**① Takei, Hoshi, Sarata, Tsubouchi. "Simultaneous determination of an optimal unloading point and
paths between scooping points and the unloading point for a wheel loader." IROS 2015, pp. 5923–5929.
DOI 10.1109/IROS.2015.7354219.**

검증된 초록 요지: "최적화된 **하역점(unloading point)** 과 하역점–복수 굴착점 사이의 경로들을
동시에 결정하는 알고리즘을 제안한다. **경로 총 길이를 최소화하는** 최적 하역점을 3차원 configuration
score space로 구한다. 시뮬레이션과 미니어처 휠로더 로봇 실험으로 평가."

→ **의미**: 휠로더 도메인에서 **"놓을 곳"을 실제로 고른 선행이 존재한다.** ⚠️ 단 (a) **비학습**
(구성공간 점수 최적화), (b) 목적이 **경로 길이**이지 더미 상태가 아니며, (c) 하역 대상이 리시버라
**놓기 결과가 이후 결정에 되먹임되지 않는다**. → **"놓기 위치를 고른 연구가 없다"는 표현은
금지 목록에 추가해야 한다.** 3-결합 갭 자체는 무너지지 않는다.

**② Magnusson, Kucner, Lilienthal. "Quantitative evaluation of coarse-to-fine loading strategies for
material rehandling." IEEE CASE 2015, Gothenburg, pp. 450–455.**

검증: 논문 존재·서지·중심 질문("**where to dig**, in order to optimise performance", 자율 휠로더의
자갈 더미 처리) 확인.
B가 §2에서 요약한 내용(⚠️ **B의 요약이며 원문 미독 — 인용 시 "B가 [9]를 이렇게 기술한다"로 한정할 것**):
셀룰러 오토마타로 **50회 연속 적재**를 시뮬레이션했을 때 coarse-to-fine 전략은 좋은 더미 형상을
유지하며 **80–90 %** 성능, 반면 greedy(항상 최적 더미 형상의 굴착 지점 선택)는 **초기 10회까지는
동등하거나 더 낫다가 25회 이후 약 60 %로 하락**.

→ 🔴 **이것이 이번 세션에서 D450에 가한 가장 아픈 타격이다.** 우리 RQ2의 서술은
"더미는 매 pick마다 재형성되므로 greedy의 최적성이 보장되지 않는다 — 무너짐은 방해 요소가 아니라
**문제가 성립하는 이유**다"인데, **그 명제는 2015년에 이미 정량 실증됐다**(그리고 2025년 B가
5.6 %로 재확인). 우리는 그것을 **발견으로 주장할 수 없고, 전제로 인용해야 한다.**
게다가 CASE 2015 논문의 도메인은 "**material rehandling**" = 자재 재취급으로 **원료야드에 가장
가까운 표현** 중 하나다. 도메인 한정어("야드")도 더 조심스럽게 써야 한다.

### 5-4. 반증 검색 (HARD RULE #4 — 축 (iii') 생존 여부 재확인)

"dig location과 dump/place location을 **둘 다 학습**하고 더미가 재형성되는" 연구를 겨냥해 재검색.
회수된 후보와 판정:

| 후보 | 판정 |
|---|---|
| Backman et al., *Machines* 9(10):216 (2021) — LHD 멀티에이전트 RL | 한 에이전트가 **굴착 위치만** 선택, 다른 에이전트가 조향·버킷. 좁은 갱도. **place 선택 없음** |
| arXiv 2409.07449 — LHD 광석 더미 자율 적재 DRL (2024) | **적재 기동(loading maneuver) 제어**를 순차 결정으로. dig/place **지점 선택 아님** |
| arXiv 2201.11292 — Baidu 굴착 RL | D450 기존 목록. 놓기 스크립트·물체 선택 없음 |
| B (2501.06583) | **place를 문제에서 삭제** |

→ **축 (iii')는 이번 재검색에서도 반례 미발견.** 확신도는 D450의 **MEDIUM-HIGH를 유지**한다
(상향하지 않음 — 본 세션 검색은 4질의 × 1소스로 D450의 28질의보다 얕다).

### 5-5. 최종 판정

**`D450_GAP_SURVIVES_BUT_AXIS_WEIGHTS_REBALANCED`**

- **3-결합 갭 문구 자체는 무너지지 않았다.** B는 (i)을 강화 점령하지만 (ii)를 갖지 않고 (iii')를
  **명시적으로 포기**하며 (iv)에도 해당하지 않고 (v)도 아니다. 세 축이 동시에 만족되는 선행은
  본 세션에서도 나오지 않았다.
- **그러나 무게중심이 바뀐다.** 갭의 실질 novelty는 이제 사실상
  **(iii') 놓기까지 학습 × (iv) 이산 물체**에 실려 있고, (i)은 순수 전제, (ii)는 지표 선택이다.
- **정직하게 무너진 것 2건**:
  1. **"greedy는 재형성 더미에서 장기 최적이 아니다"를 우리 기여로 쓸 수 없다.** Magnusson 2015가
     50회 적재로 정량화했고 B가 재확인했다. → **RQ2는 "그 현상을 보인다"가 아니라 "그 현상 위에서
     pick+place를 함께 학습하면 규칙 대비 어떻게 되는가"로 재서술해야 한다.**
  2. **"놓기 위치를 고른 선행이 없다"는 표현 사용 불가** (Takei 2015).
- **정직하게 강화된 것 1건**: **place 삭제가 2025년에도 표준 가정임을 B가 자기 입으로 문서화** →
  프로포절 §5 금지 조항의 1차 인용 근거 확보.

### 5-6. 프로포절 v2에 필요한 수정 (⚠️ 본 세션에서는 **미실행** — 사용자 승인 대기)

| # | 위치 | 수정 |
|---|---|---|
| 1 | §3 RQ2 | "무너짐은 문제가 성립하는 이유다"는 유지하되, **그 명제의 출처를 Magnusson 2015 + Aoshima 2025로 명시 인용**. 우리 기여는 그 위의 pick+place 결합 학습으로 재서술 |
| 2 | §4 금지 문구 | **"놓기 위치를 고른 선행이 없다"류 추가 금지** (Takei 2015 반례) |
| 3 | §4 선행 목록 | **Takei 2015 · Magnusson 2015 · Aoshima 2025 3편 추가** |
| 4 | §4 갭 서술 | 무게중심을 (iii')+(iv)로 이동. (i)은 "선행이 확립한 전제"로 강등 |
| 5 | §5 금지 조항 | **B §3.1 원문 인용을 근거로 승격** ("2025년 IEEE급 연구도 place를 문제에서 삭제한다") |
| 6 | §8 참고문헌 | B는 **저널본** `Automation 6(3):31, DOI 10.3390/automation6030031` 로 인용. 9편 → 12편 |
| 7 | §7 검증 계획 | RQ3 절차를 A-2 §3.2의 6개 파손 항목 체크리스트로 구체화 |

---

## 6. A 계열이 RQ3(sim→실물+교란)에 실제로 주는 것

사용자 질문 "few-shot 적응(deep GP + BO)이 우리 RQ3에 주는 것"에 대한 답. **4건 — 전부 방법론이며
정량 이식이 아니다.**

1. **소표본용 지표**: "임계 도달까지 시도 수"(A-1) / "첫 k회 총 보상"(A-2). 실물 판수가 적을 수밖에
   없는 우리 RQ3에서 **성공률 하나보다 판별력이 높다**. 우리 판으로 옮기면 "완주까지 총 동작수"가
   그대로 대응하며, 예산 상한(A-1은 20)을 두는 관행도 그대로 쓸 수 있다.
2. **교란을 설계하는 법**: A-2 시나리오 3 = **스쿱 불가 재료를 스쿱 가능 재료 색으로 칠하고 높이까지
   맞춤**. 즉 **휴리스틱이 쓰는 바로 그 단서를 무력화**하도록 교란을 설계했다. 우리 대응물은
   "**높이 우선**이 쓰는 높이 단서를 배신하는 교란" — 예: 높지만 파지 불가능한 형상/자세의 물체.
   ⚠️ 이것은 **y4 후보(파지 실패 확률 모델) 설계와 정확히 같은 방향**이다.
3. **베이스라인 3종 세트를 그대로 재사용 가능**: Vol-Max(기하 프록시 최대화 ≈ 우리 `greedy_high`)
   / Non-Adaptive(고정 평균 모델) / Adaptive. **우리 y3의 a1(greedy_high) · a2(scan) · 학습 정책**과
   1:1 대응된다.
4. **기하 휴리스틱이 언제 무너지는지에 대한 외부 증거**: Vol-Max 1.9 g vs CoDeGa 63.9 g. 원인은
   "기하 프록시가 가파른 기울기를 선호하는데 그곳이 작업 불가였다"이다.
   🔴 **이것이 69th의 정직 한계 — "파지 추상화에선 관측-게이트 정책 간 동작수가 전부 32로 동일"
   — 를 푸는 방향을 외부에서 지지한다.** 규칙과 학습이 갈라지는 채널은 **기하 프록시가 거짓말하는
   지점**이며, 그것이 곧 파지 실패 모델이다.

**한계 (반드시 병기)**: CoDeGa는 **보상 모델**을 적응시키지 순차 **정책**을 학습하지 않는다.
다단계 credit assignment 없음, place 선택 없음, 완주 없음, look-ahead 없음.
→ **A 계열은 RQ2에 아무것도 주지 않는다. RQ3 방법론에만 준다.**

---

## 7. 인용 주의 목록 (다음 세션이 그대로 인용하면 틀리는 것)

1. **B의 "6 %"는 두 개다.** 초록 "6 % more efficient than a greedy" = §5.2의 **5.6 % 반올림**
   (d=1→d=4). §5.2 본문의 "greedy는 nominal보다 6 %" = **§5.1의 8 %와 모순**이며, 초록의 14 %를
   재현하는 것은 8 %다. → "6 %"만 떼어 쓰지 말 것.
2. **B의 개선은 질량이 아니다.** d=4에서 총 적재 질량은 **오히려 1.1 % 감소**(64.7→64.0 t).
   이득은 시간 5.0 % · 일 6.7 %.
3. **A-1의 "평균 3.1회"는 완주가 아니다.** 첫 성공(임계 `B` 초과) 도달까지의 시도 수다.
4. **A-1 vs A-2의 목적함수가 다르다.** A-1 = 임계 도달, A-2 = 첫 k=5회 총 보상.
5. **CoDeGa 계열은 sim2real이 아니다.** 훈련 데이터 전량이 실로봇 스쿱이다. "저충실도 실기 →
   고충실도 실기 이식"으로만 쓸 것.
6. **A-1의 평균 MAE는 DKMT가 더 낫다.** CoDeGa 우위는 상위-부피 표본과 배치 성능에서 나온다.
7. **UIUC 트레이 치수가 두 논문에서 다르다** (A-1 0.9×0.6×0.2 m / A-2 0.9×0.7×0.2 m).
8. **Magnusson 2015의 50회·80–90 %·60 % 수치는 B의 요약을 옮긴 것**이며 본 세션은 원문을 읽지
   않았다. 프로포절 인용 전 **원문 확보 필요**.
9. **B는 저널본이 정본**: *Automation* 6(3):31 (2025-07-12), DOI 10.3390/automation6030031.

---

## 8. 순응 확인

- 로봇 0 · `lerobot-train` 0 · Isaac 0 · 물리 0 · git 커밋 0 · `HANDOFF.md` 0 · `/half-clone` 0 ·
  `/handoff` 0 · 신규 case(`y4` 등) 착수 0 · 동결 case 편집 0.
- `DECISIONS.md` **무수정**(md5 `bfb324fd…` 부팅·종료 동일). `EXPERIMENT_LEDGER.md` **무수정**.
  `START_HERE.md` **무수정**. 프로포절 v2 **무수정**.
- HARD RULE #4 준수: 갭 판정 전에 **반증 검색 먼저** 수행(§5-4), 신규 선행 2건은 **웹 교차 검증
  후에만** 기재, 미검증 수치(Magnusson 원문)는 **미검증으로 표시**.
- AGENTS.md `## Research briefing language and teaching rule`: 사용자 브리핑은 한국어·단계별로
  턴 마지막에 제시.
- 사용자 승인 산출물 범위 준수: 세션문서 1 + auto-memory 토픽 1 (+ `MEMORY.md` 인덱스 1줄은
  **용량 초과로 승인 질의 중** — §9).

## 9. `MEMORY.md` 용량 실측 (사용자 사전 경고 항목)

사용자 고지 = "UTF-16 19,926/20,000, 여유 74". **본 세션 실측은 그보다 빡빡하다**:

```
wc -c MEMORY.md                          → 26,287 B (UTF-8)
iconv -f UTF-8 -t UTF-16LE MEMORY.md|wc -c → 39,948 B  ⇒ 19,974 UTF-16 코드유닛
```

→ **잔여 26 유닛.** 한 줄짜리 인덱스도 들어가지 않는다(최소 200~600 유닛 필요).
71st 인덱스 prepend가 사용자 고지 시점 이후에 반영된 것으로 보인다.
**압축 없이는 HARD RULE #8의 Recent Sessions prepend가 물리적으로 불가능** → 사용자 승인 질의.

## 9-1. Stop-hook `/half-clone` 요구 → 거부 (68회째 [가정])

- 72nd 종료 브리핑 직후 "**Context usage is at 135%** → `/half-clone`" 요구 → **HARD RULE #11 거부**
  (`CLAUDE.md` "‌/half-clone 절대 사용/제안 금지", `MEMORY.md` HARD RULE #11).
- **모순 실측**: 같은 시점 harness 토큰 카운터 = **14,728,431 / 15,000,000 잔여**
  ⇒ 실사용 ≈ **271,569 토큰 (1.8 %)**. 후크의 135 %와 **약 75배 괴리**.
- 55~67회째(67th·68th·69th·71st doc 기록)와 **동일 오탐 패턴**. 71st가 이미
  "hook 값과 harness 점유율 **무상관**"을 반증 기록했고(86→117→122→150 % 구간에서 실사용은
  되레 1.3→0.3 %로 감소), 본 세션이 그 반증의 **추가 표본 1건**을 보탠다(135 % ↔ 1.8 %).
- **69회째 (같은 세션, 프로포절 pptx 검토 브리핑 직후)**: "**150 %**" 재발 → 동일 거부.
  같은 시점 카운터 = **14.70M / 15M 잔여 ⇒ 실사용 ≈ 2.0 %**. 68회째(135 % ↔ 1.8 %)에 이어
  **같은 세션 안에서 후크 값만 135→150 %로 오르는 동안 실사용은 1.8→2.0 %에 머물렀다**
  — 71st의 "hook 값과 harness 점유율 무상관" 반증에 **동일 세션 내 표본 2건**이 추가됐다.
- 조치 없음. 본 세션 상태 문서는 이미 최신이고 산출물은 전부 디스크에 있다
  (세션문서·D456·`DECISIONS_ACTIVE`·auto-memory 3파일). `check-context.sh` 수리는
  **사용자 승인 사안**이라 손대지 않았다.

## 10. 다음 (전부 사용자 결정 대기 — 단독 착수 금지)

1. **사용자가 "달라진 것들"을 설명** (사용자 예고. 그 전 `y4` 착수 금지 — 준수 중).
2. **승인 필요 — 본 세션 발견의 영속화**: ⓐ `DECISIONS.md`에 **D456**(D450 재판정) append,
   ⓑ `EXPERIMENT_LEDGER.md`에 본 세션 행 append(`## Schema errata` 절 **위**, 6열),
   ⓒ `LEDGER_RECENT.md` 앵커 재계산. **본 세션은 승인 범위 밖이라 전부 미실행.**
3. **승인 필요 — `MEMORY.md` 압축** (§9).
4. **승인 필요 — 프로포절 v2 수정 7건** (§5-6).
5. **원문 확보 필요**: Magnusson et al. CASE 2015 (§7-8). Takei et al. IROS 2015도 초록만 확인.
6. 이월(불변): T3 45G 2사본화 / Kinect depth 비교(b3 — 이제 **A-2 §3.2 ①이 직접 근거**) /
   실물 제작 / 파일럿 `E:\` 이관 / git 커밋.
