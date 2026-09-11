# R1 — 문헌 조사: 입상 재료 퍼내기·굴착 예측에서 재료 보정(calibration)과 초기 상태

- 작성: 2026-09-10 (Orca 워커 task_a7f63402b672, 조사 시작 16:36 KST)
- 산출 위치: `claudedocs/research/survey_20260910/R1_papers.md`
- 규칙 준수: 로봇·시뮬 실행 0, 코드 수정 0, 상태 원장 미접촉, 커밋 0. 출처 40건(§9 표). 검색 로그 58행(§10).
- 한 줄 결론: **"둘 다(병행)"**. 실험실 안식각 측정은 유지하되 **단독 보정 목표로 쓰면 안 되고**, 벽 있는 상자에서 **벽 제거(렛지) 각 + 한 입 퍼낸 뒤 절단면 각**을 Kinect로 재어 두 번째·세 번째 목표로 추가한다. 근거는 §7.

---

## 0. 요약 (먼저 읽을 것)

사용자 반론("현장은 벽 있는 선창에 가득 차 있고 처음엔 평평하다. 안식각은 퍼내면서 생긴다. 그러면 부어 만든 더미로 안식각을 재서 DEM을 보정하는 절차가 맞는가?")에 대한 문헌 답은 다음 다섯 가지다.

1. **부은 더미 안식각 하나만으로 DEM을 보정하면 답이 여러 개 나온다(비유일성).** 2026년 두 편이 정량으로 확인했다. Katagiri 2026은 DEM 10,000회로 "안식각 하나를 목표로 두면 구름마찰이 지배적이면서도 추정이 가장 불안정하다"고 했고(S1, 초록), Gaboriault 2026은 같은 더미 사진에서 안식각 재는 법 3가지가 48.65°·49.34°·41.50°로 갈리고 각 방법이 서로 다른 파라미터 군집으로 수렴함을 보였다(S2, 본문). 벌크 재료 표준 절차(Roessler 2019 S3, Coetzee 2020 S4)는 그래서 **배출 시험(draw-down)처럼 한 시험에서 여러 값을 얻거나, 시험 2개 이상을 조합**하라고 한다.
2. **"퍼낸 뒤 남는 사면"을 재는 시험은 이미 벌크 취급(그랩·선창) 분야의 표준 보정 시험 중 하나다.** 이름은 렛지(ledge) 시험 = 전단 상자(shear box) = 직사각 용기 시험. 벽(문)을 열어 재료가 흘러내린 뒤 남는 각이다. 철광석 펠릿에서 렛지 41° vs 자유 원뿔(부은 더미) 26°(Lommen 2019, S10), 습윤 철광석 미분에서 렛지 55~70°(한 시료 84°)(Mohajeri 2020, S9), Mohajeri 2021 APT 32(5)에서는 렛지 각 63~84°를 측정하되 **최적화 목표에서는 빼고 검증(7.1% 오차)에만 썼다**(S8, 본문). 즉 "부은 각"과 "절단면 각"은 같은 재료에서 10~15° 이상 다를 수 있고, 후자가 우리 예측 대상(한 입 퍼낸 뒤 남는 형상)과 기하학적으로 같다.
3. **높이맵 기반 퍼내기 학습 논문들은 대부분 안식각을 재지 않는다.** CoDeGa(S18)는 손으로 만든 지형(경사 ≤30°)을 매 시도 뒤 다시 촬영할 뿐 붕괴를 모델링하지 않고, L-GBND(S17)는 시뮬 사전학습 + 실물 100궤적 미세조정으로 재료 파라미터 자체를 맞추지 않으며 "산사태(장거리 붕괴)는 한계"라고 명시한다. 반대로 휠로더 계열(Servin S12, Aoshima S13–S15)은 **시뮬 안에 안식각 규칙(셀룰러 오토마타 붕괴)을 넣고**, 현장 안식각 32°·벌크밀도 1727 kg/m³를 맞춘 뒤 굴착력으로 2개 파라미터를 추가 보정해 sim2real 격차 약 10%를 얻었다. **초기 상태는 평평(Schenck S22, Kreis S23, DDBot S16, OWLAT S19 시나리오 1·2)과 더미(Aoshima 계열) 둘 다 흔하고, 사용자가 말한 "벽 있는 더미"는 Aoshima & Servin 2024 현장 시험이 정확히 그 조건(옆·뒤 수직벽)이었다.**
4. **로봇이 실제로 판 결과(높이맵)로 직접 보정하는 방식은 존재한다.** DDBot(S16)은 0.28×0.28 m 상자에서 삽질 1회의 점군만으로 미분가능 MPM 시뮬의 4개 파라미터를 5~20분에 동정했고, Matl 2020(S25)은 부은 더미·고리의 깊이 영상 통계 16개로 마찰·구름마찰·반발계수를 베이즈 추론했으며, FLIP 2025(S26)는 안식각 측정값을 베이즈 최적화 목표로 삼아 시뮬을 맞춘 뒤 분말 계량 오차를 6.11→2.12 mg로 줄였다. 우리 규모(수십 g 한 입)와 가장 닮은 실험은 Takahashi 2021(S28, 커피콩 17~27 g 목표 파지)과 DDBot이다.
5. **우리 절차에 대한 판정**: (b) 유지 = 실험실 부은 더미 안식각 측정(값싸고 표준이며 목표 1로 필요). (a) 바꾼다 = ① 그 값을 **단독** 목표로 쓰지 않는다 ② 실제 30×22 cm 상자에서 **벽 제거 렛지 각**과 **한 입 뒤 절단면 각**을 Kinect 높이맵으로 재어 목표 2·검증 3으로 쓴다 ③ 시뮬에 붕괴(안식각) 규칙과 "한 입보다 넓은 높이맵 창"을 둔다 ④ 벽 효과를 명시적으로 처리한다. (c) 추가 실험 = PP 알에서 부은 각·렛지 각·절단면 각 세 값의 차이를 먼저 잰다(차이가 ±3° 안이면 부은 각만으로 충분, Lommen 펠릿처럼 15° 벌어지면 렛지 필수).

---

## 1. 용어 풀이 (처음 나오는 순서, 한 줄씩)

| 용어 | 뜻 |
|---|---|
| DEM (이산요소법) | 알갱이 하나하나를 공(또는 공 묶음)으로 두고 충돌·마찰을 계산하는 시뮬레이션. 입력 파라미터(마찰계수·구름저항·강성·응집력)를 실험과 맞추는 일이 "보정(calibration)". |
| 보정 비유일성(ambiguity) | 서로 다른 파라미터 조합이 같은 실험값(예: 안식각 32°)을 만들어 어느 것이 맞는지 정할 수 없는 문제. |
| 안식각(angle of repose, AoR) | 재료를 쌓았을 때 사면이 수평과 이루는 각. 재는 방법에 따라 값이 다르다(§5). |
| 정적/동적 안식각 | 정적 = 멈춘 더미의 각. 동적 = 회전 드럼 등에서 계속 흐를 때의 각. 동적이 보통 3~10° 작다(S31). |
| 부은 각(poured) / 배출 각(drained) | 부은 각 = 위에서 부어 쌓인 바깥 사면. 배출 각 = 아래 구멍으로 빠진 뒤 남는 안쪽 사면(크레이터 벽). 배출 각이 더 크다(S31). |
| 렛지(ledge) 각 = 전단 상자 = 직사각 용기 시험 | 상자에 재료를 채우고 한쪽 벽(문)을 열어 흘러내린 뒤 남는 사면 각. "지지벽이 사라진 뒤 남는 절단면"이라 우리 한 입 뒤 형상과 같은 기하. |
| 배출 시험(draw-down test) | 위 칸에서 아래 칸으로 재료를 빼내며 안식각·전단각·질량유량·잔류질량 4개를 한 번에 얻는 보정 시험(S3, S7). |
| 전단 시험(ring shear / shear box) | 재료에 수직 하중을 주고 밀어 파괴될 때 힘을 재는 시험. 응집·내부마찰각을 준다. 가루·습윤 재료용. |
| 구름저항(rolling resistance) | 공 모양 DEM 입자가 실제 모난 알갱이처럼 잘 구르지 않게 넣는 가짜 저항. 안식각을 가장 크게 좌우한다(S1). |
| JKR 모델 | 습기·응집이 있는 재료의 달라붙는 힘을 넣는 DEM 접촉 모델(S39). |
| 높이맵(heightmap) | 깊이 카메라 점군을 격자별 "높이"로 바꾼 2D 지도. 퍼내기 학습의 표준 관측. |
| 붕괴(avalanche) / 안식각 규칙 | 사면이 안식각보다 가파르면 알갱이가 흘러내려 낮아지는 현상. 시뮬에서는 "격자 간 경사 > 안식각이면 이웃으로 옮긴다"는 셀룰러 오토마타로 흉내 낸다(S12, S23). |
| sim2real 격차 | 시뮬과 실물의 결과 차이. Aoshima & Servin 2024는 굴착 시계열·적재질량·일로 정량화해 약 10%(S13). |
| MPM (물질점법) | 입자 + 배경 격자로 연속체처럼 푸는 시뮬. DDBot이 미분가능 형태로 씀(S16). |
| 세계모델(world model) | 현재 더미 + 행동 → 다음 더미·성능을 예측하는 학습 모델(S14). |

---

## 2. Q1 — DEM/입상 시뮬 논문들은 재료 파라미터를 무엇으로 보정하는가

### 2.1 보정 방식 카탈로그

| 방식 | 대표 출처 | 무엇을 맞추나 | 장점 | 단점 | "퍼낸 뒤 남는 형상" 예측에 대한 적합도 |
|---|---|---|---|---|---|
| A. 부은 더미 안식각 단독 (원뿔·리프팅 실린더) | S1 Katagiri 2026, S2 Gaboriault 2026, S5 Coetzee 2010 | 스칼라 1개 | 값싸고 표준. 마찰 계열 감도 큼 | **비유일성**: S1 = 10,000회 DEM에서 안식각 단독 목표는 역문제가 불량조건(ill-conditioned). 구름마찰이 기여 최대이면서 σ_AoR 0.5→3.0°에 대해 추정 분산이 가장 크게 증폭. 정지마찰·영률의 CV가 σ에 무관 = "안정"이 아니라 "제약 부족". S2 = 같은 더미에서 방법 3종이 48.65/49.34/41.50°, 최적 파라미터가 방법마다 다른 군집(μt 0.25/0.35/0.15/0.60) | 낮음(단독일 때). 형상은 "각" 하나로 요약되지 않음(S2: 오목 사면·뾰족 정점을 놓침) |
| B. 배출 시험(draw-down) | S3 Roessler 2019 Part I(+Richter 2020 Part II), S4 Coetzee 2020, S7 Marín Pérez 2024, S6 Grima 2011 | 안식각 + 전단각 + 질량유량 + 잔류질량 | 한 시험으로 3~4개 값 → S3 "거의 유일한 파라미터 집합", S4 "배출 시험만으로 미끄럼·구름마찰 둘 다 결정", S7 다중구 입자 + 구멍 3종으로 최대 편차 5.9% | 장치 제작 필요. 응집 재료는 별도 처리(S3 Part I은 비응집 전용) | 중간~높음. 배출 각(drained)이 "빠져나간 뒤 남는 크레이터 벽"이라 절단면과 가까움 |
| C. 렛지(ledge) 각 = 벽 제거 뒤 남는 사면 | S10 Lommen 2019, S9 Mohajeri 2020, S8 Mohajeri 2021, S11 Mohajeri 박사논문 | 지지 제거 후 정적 각 | **우리 예측 대상과 같은 기하**. 그랩·선창 분야 표준. 펠릿 렛지 41° vs 원뿔 26°(S10) — 같은 재료가 방법에 따라 15° 차이. 습윤 미분 55~70°(84°)(S9). S8 표 2: 63~84° | 역시 스칼라 1개. S8은 "정의성(definiteness)" 기준 때문에 최적화 목표에서 **제외**하고 검증에만 씀(시뮬 90° vs 실측 84°, 7.1%). 직사각 상자라 ≤90°로 포화. 벽 효과 있음(S10은 폭 = 입경 18배, 벽 근처 입자 제외) | **가장 높음**(검증 목표로). 단독 최적화 목표로는 A와 같은 비유일성 |
| D. 구속 압축 / (가상) 삼축 시험 | S5 Coetzee 2010, S12 Servin 2021 | 강성·내부마찰·응집 | S5: "압축 시험으로 강성 먼저, 그 다음 안식각으로 마찰" — 파라미터를 나눠서 잡음. S12: 15종 흙을 가상 삼축으로 사전 보정한 라이브러리 | 강성은 퍼내기 형상에 둔감(S1). 장비 필요 | 낮음(형상), 중간(굴착력) |
| E. 더미 전체 형상(full-field) 영상 매칭 | S2 Gaboriault 2026 | 더미 실루엣 픽셀 차이(SAD) | 안식각이 놓치는 오목/볼록·정점 재현. 군집이 뚜렷 | 실린더 1 cm 분말 규모. 재료 소량 | 높음(개념). 우리 Kinect 높이맵으로 확장 가능 |
| F. 부은 형상 깊이영상 + 베이즈 추론 | S25 Matl 2020, S26 FLIP 2025 | S25: 더미·고리 요약통계 16개 → μs, μr, e. S26: 안식각 측정 → BO로 시뮬 파라미터(마찰·응집·부착 등 9개) 맞춤(오차 ≤1.5°) | 로봇·깊이카메라만으로 자동. S26은 하류 과제(분말 계량)에서 6.11→2.12 mg | S25는 반발계수 등 3개만, 공 입자 가정. S26은 PBD(속도 우선) 시뮬 | 중간. "부은 형상"이라 절단면은 아님 |
| G. **로봇이 실제로 판 결과(높이맵/점군)로 직접 동정** | S16 DDBot 2025, S13 Aoshima & Servin 2024, S17 L-GBND(파라미터 아님, 모델 미세조정) | S16: 삽질 궤적 1개의 실측 점군 vs 시뮬 점군(EMD/높이맵 거리)로 E·ν·ρ·φ 4개. S13: 굴착력 시계열로 버킷-흙 마찰 0.2, 골재 강성 배수 0.01 | **예측 대상과 목표가 동일**(파낸 뒤 형상). S16은 5~20분 수렴, 검증 궤적 별도. S13은 격차 ~10% | 시뮬이 미분가능(MPM)이거나 빠른 대체모델이어야 함. S16은 φ가 흙·모래 모두 ≈19°로 수렴(물리값이라기보다 "형상 재현용" 값) | **가장 높음**. 단 초기 추정치는 A/B/C에서 가져오는 게 수렴에 유리(S16 "ManInit") |

### 2.2 각 방식의 근거 수치 (본문 확인분)

- **S2 Gaboriault 2026 (본문)**: GranuHeap(내경 1.00 cm 실린더, 1.00 cm/s 리프팅), Ti6Al4V 1.50 g, 시뮬 670회. 방법 1(높이/반경) 34.90°, 방법 2(전체 사면 회귀) 33.73°, 방법 3(하반부 회귀) 31.34°를 각각 맞춘 최적 파라미터가 (μt, μr, γ) = (0.35, 0.001, 65e-5)/(0.15, 0.30, 35e-5)/(0.60, 0.10, 5e-5)로 완전히 다름. 형상 전체(SAD) 최적은 (0.25, 0.001, 40e-5). 결론 원문: "AOR-based measurement methods do not guarantee finding a DEM model parameter set that adequately reproduce the real powder behaviour."
- **S8 Mohajeri 2021 APT 32(5):1532–1548 (본문, 로컬 PDF 텍스트 추출)**: 표 2 "Ledge angle of repose, α_M, ° : 63 (min) – 84 (max)" (σ_pre ≤ 20 kPa, 함수율 ±2%). 렛지 상자 높이 250 mm, 문 높이 200 mm까지 충전, 문 개방 후 사면에 선형회귀로 각 산정. 3단계에서 "정의성 기준을 고려해 렛지 각을 목표에서 제외하고 관통 시험 W80,65·W70,300을 넣음". 표 10: 최적 파라미터로 렛지 시뮬 90° vs 목표 84°, |e| 7.1%. 유의 변수 = 정지마찰계수·입자 전단탄성률·표면에너지·소성비.
- **S9 Mohajeri 2020 Powder Technol. 367 (exa 본문 발췌)**: 렛지 상자 250 mm 높이 × 215 mm 길이 × 80 mm 폭, 약 10 cm 높이에서 천천히 부어 다짐 최소화, 문 개방 후 수평 사진에서 10점 좌표로 각 산정. 세 응집 철광석 시료 55~70°, 시료 I2(함수율 +2%)는 평균 84°. "[1](Lommen)은 자유유동 펠릿에서 같은 렛지법으로 40°".
- **S10 Lommen 2019 Powder Technol. 352 (exa 본문 발췌)**: 펠릿 "ledge 41°, cone 26°". 렛지 상자 300×200×300 mm, 폭 = 평균 입경 18배(권장 최소보다 작아 벽 근처 입자 제외로 보정). 보정 목표 = 안식각 시험 2종 + 쐐기 관통 시험.
- **S11 Mohajeri 박사논문 (exa 발췌)**: "굴착(그랩) 응용에서 모이는 부피는 주로 안식각으로 결정된다 — 버킷이 닫힐 때 여분 재료가 열린 옆면으로 흘러나가므로 안식각이 높을수록 더 많이 모인다." → 안식각은 우리 "한 입 질량"에도 직접 들어가는 양.
- **S13 Aoshima & Servin 2024 (본문)**: 현장 자갈 30~40 mm, 벌크밀도 1727 kg/m³. DEM 입자 ρ 2590, μ 0.3, μr 0.02, e 0 = "현장 벌크밀도 1727과 안식각 32°에 가장 잘 맞는 사전 보정 흙". 지형 모델 벌크 파라미터: 내부마찰각 32°, 팽창각 8°(유효 40°), E 4.6 MPa. **굴착력에 맞춰 보정한 것은 2개뿐**(버킷-흙 마찰 0.2, 골재 강성 배수 0.01). 8단계 충실도 전부 격차 ~10%(표 4·5 평균 0.07~0.22). 실물 적재 3회(3.46/2.70/2.10 t). ⚠️ 32°가 현장에서 어떻게 측정됐는지는 본문에 없음.
- **S16 DDBot (본문)**: 상자 0.28×0.28 m, 깊이 0.11 m, 충전 0.07 m, UR5e + 3D 프린트 삽, Zivid 카메라. DPSI(미분가능 시스템 동정)로 E 50~200 MPa, ν 0.1~0.4, ρ 1200~2200, φ 10~40° 범위에서 최적화·검증 궤적 각 1개. 결과 흙 (182,683 kPa, 0.242, 1566, 18.88°), 모래 (121,378 kPa, 0.198, 1974, 19.02°). 높이맵 거리(HMD) 손실이 EMD보다 매끄럽고 수렴 안정. 초기값(ManInit)이 있으면 더 빠르고 분산 작음.
- **S25 Matl 2020 (본문)**: 쿠스쿠스 4 mm 2,000알, 깔때기 12 cm, RealSense D435 40 cm 높이, Isaac 시뮬 2,000구. "대부분의 DEM 보정은 원뿔 더미의 안식각으로 파라미터 1개를 추정하지만, 부으면 더미가 아니라 고리가 생기기도 하고 얕은 더미는 깊이영상에서 안식각 측정이 어렵다" → 16개 요약통계. 추정 (μs, μr, e) = (0.6687, 8.15e-7, 0.7689). 2~10 cm 낙하 높이로 일반화.
- **S26 FLIP 2025 (본문)**: ISO 8398 방식 안식각을 로봇으로 자동 측정(수동 대비 평균 |오차| 0.84°, 모래 28.55 vs 27.69°). BO로 시뮬 안식각 오차 ≤1.5°가 되는 파라미터 10세트 확보(재료별 sim2real 오차 0.22~1.37°). 실물 계량 오차: FLIP 커리큘럼 2.12±1.53 mg vs 도메인 무작위화 6.11±3.92 mg.
- **S12 Servin 2021 (본문)**: "지역 경사가 흙의 안식각 δ_b를 넘으면 셀룰러 오토마타로 붕괴". "비응집 재료에서 δ_b는 내부마찰각과 잘 일치, 응집이 있으면 더 클 수 있음". 입자 파라미터는 가상 삼축 시험으로 사전 보정. 검증은 현장이 아니라 고해상 DEM 참조 시뮬 대비 10~25%.

### 2.3 Q1 소결

- "부어 만든 더미 안식각 → DEM 보정"은 **틀린 절차가 아니라 불충분한 절차**다. 벌크 취급·분말 공학 문헌은 2018~2026년에 걸쳐 일관되게 "목표 2개 이상"을 요구한다(S1, S2, S3, S4, S7, S8).
- 우리 예측 대상(퍼낸 뒤 남는 형상)에 기하학적으로 가장 가까운 표준 시험은 **렛지(벽 제거) 시험**이고, 이는 정확히 사용자가 말한 "벽 있는 상자에서 절단면 각을 재는" 방식이다. 문헌은 이를 **최적화 목표보다 검증 목표**로 쓴 선례(S8)를 남겼다.
- 로봇이 판 결과로 직접 동정하는 방식(S16)은 우리 규모(0.28 m 상자)에서 이미 작동했고, 초기값으로 A/B/C의 값을 넣는 하이브리드가 권장된다.

---

## 3. Q2 — 높이맵 기반 퍼내기/굴착 학습 논문의 초기 상태·안식각·붕괴 처리

| 출처 | 규모·재료 | 초기 재료 상태 | 안식각 측정? | 사면 붕괴(avalanche) 처리 | 비고 |
|---|---|---|---|---|---|
| S18 CoDeGa RSS 2023 (본문) | 트레이 0.9×0.6×0.2 m, UR5e, L515. 모래~자갈·옥수수·종이공 12종 | **손으로 만든 지형**(경사·능선), 최대 높이 0.2 m, **최대 경사 30°**. 배포 시도마다 수동 리셋 | 안 잼 | 모델링 안 함. "각 행동이 지형을 바꾸므로 매 스쿱 뒤 RGB-D 재촬영" | 깊이 3~8 cm, 평균 31.3 cm³, 최대 260.8 cm³. 실로봇 5,100 스쿱(sim2real 아님) |
| S19 OWLAT 2023 (본문) | 빈 0.9×0.7×0.2 m, WAM7, D415 팬틸트 | 시나리오 1·2 = **평평한 Regolith** + 리셋 시 자연 발생 소요철; 시나리오 3 = 인공 둔덕 3개 | 안 잼 | 안 함 | 깊이 0.2~0.8 cm(UIUC 3~8 cm의 1/10). 질량 수동 계측. 결과 1.9/22.0/63.9 g |
| S17 L-GBND (본문) | 시뮬 SAPIEN, 실물 상자 0.9×0.6×0.2 m, 자갈 0.8~1.0 cm·모래 | 시뮬: **다이아몬드-스퀘어 무작위 지형**(~3,000 큐브). 실물: 자기놀이 100궤적 | 값 언급만("안식각 약 25~40°라 각진 목표 형상은 물리적으로 불가") | RoI 국소 예측이라 **"산사태 같은 장거리 효과는 한계"라고 명시**. 재료 바뀌면 전체 미세조정 필요 | 시뮬 파라미터 보정 없음. 2D CNN 대비 37% 낮은 오차 |
| S14 Aoshima WM 2024 (본문) | AGX 시뮬, WA320-7, 자갈(1727, 32°, 0 kPa, 8°) | **시드 더미 6종**(삼각·원뿔·쐐기 × 경사 20°/30°) + Perlin 노이즈, 연속 30회 적재 × 60반복 = 10,718표본 | 시뮬 입력값으로 사용(32°) | 시뮬 내부 붕괴 + **"붕괴 때문에 더미상태 예측기는 3.6 m가 아니라 5.2 m 높이맵이 필요"**. 저차원 모델은 셀룰러 오토마타로 안식각까지 재분배 | 더미 MAE 0.75 m³(버킷 용량 25%, 창 부피 3%), MRE 3.04%. 40회 누적 시 안식각보다 가파른 비물리 상태로 발산 → CA 후처리 권고 |
| S15 Aoshima 2025 (본문) | 위와 동일 세계모델 | **사다리꼴 프리즘 1.8 m, 전면 30°** + Perlin, 10개 더미 | 32° 입력 | 시뮬 내부. 논의: "자갈 더미는 항상 자연 안식각 근처로 가라앉지만, **응집·비균질 흙은 더미 상태가 행동에 훨씬 더 의존**할 것" | 이득 = 시간 5.0%·일 6.7%, 질량 −1.1%(함정 ③ 재확인) |
| S13 Aoshima & Servin 2024 현장 (본문) | 실물 WA320-7 15.2 t, 자갈 30~40 mm | **옆·뒤에 수직벽으로 구속된 자갈 더미**, 전면 자유사면. 적재 전마다 2D 레이저로 표면 스캔 | 32°를 현장값으로 채택(측정법 미기재) | 시뮬(DEM/다중스케일) 내부 | 사용자 "벽 있는 현장" 주장의 직접 선례 |
| S20 Sandzimier & Asada RA-L 2020 (초록) | 로봇 굴착기 실험실 | 명시 없음(초록). "흙 표면 형상 직전 상태에 따라 분산이 달라진다" → 이분산 GP | 안 잼(초록 기준) | 언급 없음 | 본문 미확보(dspace 차단) |
| S21 AES Science Robotics 2021 (초록+보도) | 6.5~49 t 굴삭기, 폐기물장·자갈 등 | 자연 더미(현장). LiDAR+카메라 융합, 재료·질감 분류 | 안 잼 | 언급 없음 | 24 h 무개입, 소형기 67.1 m³/h(보도자료) |
| S22 Schenck CoRL 2017 (본문) | 트레이(칸막이), 핀토콩 **3.75 kg**, KUKA 7대 | **한쪽 칸에 평평하게 고른 상태**에서 시작, 반대칸에 목표 더미 | 안 잼 | 안 함(다음 높이맵을 CNN이 직접 예측) | ~15,000 스쿱&덤프, 1 cm 셀, 다단계 예측 오차 수 mm(그래프) |
| S23 Kreis 2025 (본문) | 상자, UR5e, 2×2×15 cm 큐브 EE, ZED 2i | **평평한 바닥 6 cm**(시뮬), 실물은 "완전히 평평하진 않음" | 시뮬 규칙의 입력(값 미기재) | **안식각 기반 높이맵 붕괴 모델**(Kim/Pavlov)을 시뮬로 채택, 제로샷 실물 전이 | 목표 ≤10×10 cm, 깊이 ≤3 cm, 잔차 3.4 mm |
| S24 GRAIN CoRL 2024 (본문) | 60×60×20 cm 탱크, 6 mm 플라스틱 BB, RHex 다리 | **경사 18°로 평탄화 = "재료 안식각에 가까운 값"**(붕괴 연구 목적) | 안식각 근처로 초기화 | 붕괴 자체를 ViT가 학습(실물 100회×10 굴착) | MAE 1.13 cm, 성공 ≥80%. 시뮬 없음 |
| S16 DDBot (본문) | 0.28×0.28×0.11 m, 충전 7 cm | **평평 충전** | 안 잼(φ를 동정) | MPM 내부 | 위 §2 |

### 3.1 Q2 소결

- 초기 상태는 **평평**(Schenck, Kreis, DDBot, OWLAT 1·2)과 **더미**(CoDeGa 손지형, Aoshima 계열) 둘 다 흔하다. 사용자의 "처음엔 평평" 설정은 로봇 규모 실험의 다수 관행과 일치하고, "벽 구속"은 휠로더 현장 시험(S13)에 선례가 있다.
- **퍼내기·굴착 학습 논문 중 안식각을 실측한 사례는 이번 검색 범위에서 없었다**(GRAIN이 "안식각 근처 18°"로 초기화한 것이 유일한 준-측정). 분말 계량 과제(S26 FLIP)에서는 로봇이 안식각을 실측해 시뮬을 맞춘 선례가 있다. 안식각은 (i) 시뮬 붕괴 규칙의 입력(S12, S14, S23) 또는 (ii) 목표 도달 불가의 설명(S17)으로만 등장한다.
- 붕괴 처리는 두 갈래: (i) 모델링 안 하고 매 행동 뒤 재관측(S18, S19, S22) (ii) 시뮬에 안식각 규칙 내장(S12, S14, S23). S14의 정량 교훈 두 개가 우리 설계에 직접 쓰인다: **붕괴 때문에 예측 창을 한 입보다 넓게(3.6→5.2 m, 약 1.44배) 잡아야 했고**, **누적 예측은 안식각보다 가파른 비물리 상태로 발산하므로 안식각 후처리(CA)가 필요**했다.

---

## 4. Q4 — 우리 축소 실험(30×22 cm 상자, PP 렌즈형 알 4.5×3.8×2.5 mm, 한 입 12~18 g)과 닮은 소형 실험

| 출처 | 용기 | 재료·알 크기 | 한 행동당 양 | 센서·계측 | 우리와의 거리 |
|---|---|---|---|---|---|
| S16 DDBot | 0.28×0.28×0.11 m, 충전 7 cm | 모래·흙 | 삽질 1회(질량 미보고) | Zivid 점군 → 높이맵 40×40 | **용기 규모 최근접**. 판 결과로 직접 동정 |
| S28 Takahashi ICRA 2021 | 트레이 603×377×145 mm | 커피콩·쌀·오트밀·땅콩 | **커피콩 17/22/27 g, 쌀 45/60/75 g**(2 cm 삽입 파지) | RGB-D(Ensenso N35) 150×150 px 패치, 저울 1 g | **한 입 질량 최근접**. 1,000회 자기지도 수집. "남은 양이 적어지면 더미를 모아야 한다"고 언급 |
| S22 Schenck 2017 | 칸막이 트레이 | 핀토콩 3.75 kg | 스푼 1회 | 깊이 카메라, 1 cm 셀 | 알 크기·형상(콩) 유사. 높이맵 다음상태 예측 = 우리 과제 ② |
| S29 Clarke 2019 (초록) | 미상 | 두 종 입상 | 7,380 스쿱 | 높이맵 + 질량 | **질량 RMSE 5.8 g, 높이맵 RMSE 0.38 cm** — 우리 목표 오차의 기준선 후보 |
| S18/S19 CoDeGa/OWLAT | 0.9×0.6(0.7)×0.2 m | 모래~자갈 | 평균 31.3 cm³ | L515/D415 | 3배 큰 트레이, 부피 계측 |
| S24 GRAIN | 60×60×20 cm | 6 mm BB | 다리 굴착 | D435i 15 Hz | 알 크기 유사(6 mm vs 4.5 mm), 붕괴 학습 |
| S25 Matl 2020 | 58×58 cm 시야 | 쿠스쿠스 4 mm 2,000알 | 붓기 | D435 40 cm | 알 크기·형상(둥근 원기둥) 유사, 부은 형상으로 보정 |
| S9/S10 렛지 상자 | 250×215×80 mm / 300×200×300 mm | 철광석 미분 / 펠릿 | — | 사진·선형회귀 | **우리 30×22 cm 상자와 같은 급**. 렛지 시험을 그대로 옮길 수 있음 |
| S27 Kadokawa 2023, S26 FLIP, S30 Kang 2024 (초록/본문) | 실험실 스푼·바이알 | 분말 | 5~20 mg | 저울 | 규모 1,000배 작지만 "안식각→시뮬 보정→정책" 파이프라인(S26)은 동형 |
| S35 Datta 2024 (초록) | 리프팅 실린더 | **m&m·핀토콩 모양 슈퍼쿼드릭** 30종 | — | — | 우리 렌즈형 알의 안식각을 종횡비로 예측하는 모델(R² 98.6%) |

소결: 우리 설정은 **DDBot(용기)·Takahashi(질량)·렛지 상자(기하)**의 교집합이고, 완전히 같은 실험은 검색 범위에서 없다. 각 논문에서 가져올 측정 항목은 (i) 행동 전후 높이맵과 질량을 항상 쌍으로 기록(S22, S29), (ii) 질량 계측 해상도와 목표 오차를 먼저 정의(S28: 5%/10% 허용), (iii) 한 입 뒤 형상을 시뮬과 같은 손실(높이맵 거리)로 비교(S16).

---

## 5. Q3 — "안식각" vs "절단면 각/동적 안식각/붕괴 후 각"의 구분

| 각 | 정의·측정 | 크기 관계 (문헌 수치) | 출처·확인 |
|---|---|---|---|
| 정적 안식각(부은 각, poured/external) | 부어 쌓은 더미 바깥 사면 | 기준값 | S31 §2 (본문) |
| 배출 각(drained/internal) | 아래로 빼낸 뒤 남는 안쪽 사면 | 부은 각보다 "significantly greater"(Cho et al. 인용) | S31 (본문) |
| 렛지 각(벽 제거 후) | 문 개방 후 남는 사면 | 펠릿 41° vs 원뿔 26°(S10); 응집 미분 55~90°(S9); ≤90° 포화(S8) | 본문 발췌 |
| 동적 안식각 | 회전 드럼에서 흐르는 중 | 정적보다 **3~10° 작다**; 상부(붕괴 시작)·하부(붕괴 멈춤) 각의 평균 ≈ 동적(S33: 0.28~4.38 mm 모래, 차이는 입경 커질수록 증가) | S31, S33 (초록) |
| 붕괴 후 각 / 최대 안정각 | 정적 각을 넘으면 붕괴 → 동적 각에서 멈춤 | 감소 중력에서 정적 +5°, 동적 −10° → 붕괴 규모 증가; 모난 알 ~40°, 둥근 알 ~25°(S32) | S32 (초록; ⚠ Consensus 저널 메타데이터 오류, 실제는 JGR-Planets) |
| 내부마찰각과의 관계 | 전단 시험 | "안식각 = 가장 느슨한 상태의 내부마찰각"(Metcalf, S31); 비응집이면 일치·응집이면 안식각이 더 큼(S12); S37: 안식각은 구성적(constitutive) 양이 아니고 초기 공극비에 무관하며 **임계 마찰각이 하한** (클럼프 35.95±0.88° vs 볼록 단순화 31.26±0.95°) | S31, S12 (본문), S37 (초록) |
| 기둥 붕괴 후 잔류 형상 | 수직 기둥을 놓아 무너뜨림 | 중앙 미교란 원뿔 **약 59°**(종횡비 1.7), 런아웃 r∞ = r_i(1+1.24a) | S38 (초록) |

영향 인자(우리 PP 알에 해당하는 것만):

- **입자 형상**: 종횡비가 안식각을 좌우(S35, m&m·핀토콩 모양 포함). 둥근 알 ~25° vs 모난 알 ~40°(S32). → 렌즈형 PP 알은 "둥근 쪽"이라 낮은 안식각·잘 흐르는 편으로 예상되며, 이 경우 렛지-원뿔 차이가 펠릿(S10)과 비슷할 가능성.
- **응집·함수율**: 안식각 30.83→37.13°(S39, JKR); 함수율 +2%에서 렛지 84°(S9); 응집비가 임계를 넘으면 원뿔→불규칙 더미(S34). → 건조 PP 알에는 해당 없음(장점). 단 정전기·습도는 별도.
- **벽 효과**: 간격이 좁을수록 더미 각 증가(S36); S10은 폭 = 입경 18배에서 벽 근처 입자를 제외. 우리 상자 22 cm / 4.5 mm ≈ 49배, 30 cm / 4.5 mm ≈ 67배라 중앙부는 안전하지만 **벽에 붙은 절단면은 벽 마찰 때문에 더 가파르게 서는 것이 정상**이고, 이를 "재료 안식각"으로 읽으면 안 된다.
- **리프팅 속도·낙하 높이**: 속도·질량·높이 증가 → 안식각 감소, 바닥 거칠기 증가 → 증가(S31). → 렛지 문 여는 속도와 붓는 높이(S9는 ~10 cm)를 고정해야 재현된다.
- **IMSBC 안식각(선박 화물 규정)**: MSC.1/Circ.1453 지침은 "비응집 화물에만 안식각을 기재, 응집이면 Not applicable"(S40 원문). IRON ORE 일정표는 Not applicable로 확인. 철광석 펠릿 일정표의 Not applicable은 코디네이터 제공 사실이며 이번 조사에서 원문을 직접 열지는 못했다(2차). 어느 쪽이든 IMSBC 값은 규제용 단일 수치라 DEM 보정 목표로 쓰면 안 된다.

---

## 6. 인용 함정 재확인 (코디네이터 ③ 4건 + 이번에 발견 4건)

| 함정 | 확인 결과 |
|---|---|
| CoDeGa = 실로봇 5,100 스쿱, sim2real 아님 | **확인**(S18 본문: 51 지형×100 = 5,100, 공개 6,700, 시뮬 0) |
| Aoshima 2025 이득 = 시간 5%·에너지 6.7%, 질량 아님 | **확인**(S15 본문: 질량 64.7→64.0 t = −1.1%, 시간 665→632 s, 일 14.9→13.9 MJ) |
| L-GBND = 시뮬 사전학습 + 실물 미세조정으로 "상호작용 후 지형" 예측 | **확인**(S17 본문: SAPIEN ~1,000궤적 사전학습, 실물 100궤적 자기놀이 미세조정, 0.1 s 스텝) |
| IMSBC 안식각 = 비응집 전용, 철광석 펠릿 N/A | **지침·IRON ORE 일정표 확인**, 펠릿 일정표 본문은 미열람(2차) |
| (신규) Kleinhans 2011 저널 | Consensus가 "Molecular and Cellular Biochemistry"로 표기 — 오류. J. Geophys. Res. Planets 116, E11004 |
| (신규) Katagiri 2026 DOI | 저널은 Computational Particle Mechanics 14:334–345인데 DOI가 10.1016/j.cpms.2026.03.008(Elsevier 접두). 출판사 기록대로 인용, 본문 미열람 |
| (신규) Sandzimier & Asada 본문 | alphaXiv 제목 검색이 ExACT(2405.05861)로 오해석됨. dspace 405/404. **초록만** 인용 가능 |
| (신규) Kreis 2025 안식각 값 | "안식각 기반 붕괴 모델"을 쓴다고만 하고 값·실물 보정은 본문 반환 페이지에 없음. "보정했다"고 쓰면 안 됨 |

---

## 7. 결론 — 이 조사가 우리 절차를 바꾸는가

### (b) 유지

- **실험실 부은 더미 안식각 측정은 유지한다.** 이유: ① 값싸고 표준(S31)이며 시뮬 붕괴 규칙(S12·S14·S23)과 그랩 적재량(S11 "부피는 주로 안식각으로 결정")에 직접 들어가는 입력. ② 안식각 실측 → 베이즈 최적화로 시뮬을 맞추면 하류 과제 정확도가 실제로 좋아진다는 실증(S26, 6.11→2.12 mg). ③ 휠로더 현장 sim2real도 안식각 32°·벌크밀도를 1차 목표로 썼다(S13).
- **초기 상태 "평평"도 유지한다.** 로봇 규모 실험 다수(S16, S22, S23, S19)와 일치하고, 사용자 논리(현장은 가득 차서 평평)와 어긋나는 선행이 없다.

### (a) 바꾼다

1. **부은 각을 단독 보정 목표로 쓰지 않는다.** 단독이면 파라미터가 비유일(S1, S2, S3). 최소 2개 목표(예: 부은 각 + 렛지 각, 또는 배출 시험 4값)를 동시에 만족시키는 조합을 찾고, 남는 하나는 검증에 쓴다(S8의 "정의성" 절차).
2. **벽 있는 30×22 cm 상자에서 렛지(벽 제거) 각과 한 입 뒤 절단면 각을 Kinect 높이맵으로 잰다.** 이것이 문헌의 "shear box/rectangular container test"(S8, S9, S10)와 같은 시험이고, 우리 예측 대상과 기하가 같다. 절차 고정 항목: 붓는 높이(~10 cm), 문 개방 속도, 벽에서 입경 몇 배 안쪽을 제외할지(S10: 폭/입경 18에서 벽 근처 제외).
3. **한 입 뒤 형상은 실물 높이맵으로 직접 동정·검증한다(DDBot 방식).** 초기값은 1·2에서, 손실은 높이맵 거리(S16에서 EMD보다 안정). Aoshima 교훈에 따라 예측 창은 한 입 폭보다 넓게(1.4배 이상), 누적 예측 뒤에는 안식각 후처리를 둔다(S14).
4. **시뮬 검증 지표를 분리한다.** "굴착력/질량"과 "남는 형상"은 서로 다른 파라미터에 민감하다(S5: 드래그 힘은 초반 1/3만 맞음; S1: 강성은 형상에 둔감). 우리 지표(총 시간·횟수·실패율)는 형상 쪽이므로 형상 검증을 우선한다.

### (c) 추가 실험 필요 (결정 분기점)

- **E1. 세 각 비교**: 같은 PP 알로 (i) 부은 원뿔 각 (ii) 렛지 각 (iii) 한 입 퍼낸 뒤 절단면 각(Kinect, 벽에서 입경 20배 이상 안쪽). 판정: (ii)−(i) ≤ 3°이면 부은 각만으로 충분(둥근 알 가설 채택). 펠릿처럼 10~15° 벌어지면(S10) 렛지 각이 보정 필수.
- **E2. 벽 인접 절단면 vs 중앙 절단면 각 차이**: 벽 마찰 효과(S36)의 크기를 우리 상자에서 수치화. 차이가 크면 시뮬에 벽 마찰계수를 별도 목표로 둔다.
- **E3. 한 입 12~18 g의 재현성**: 평평 상태 vs 더미 상태에서 각 10회. Takahashi(S28)처럼 5%/10% 허용오차 성공률로 보고. 이 값이 시뮬 검증의 잣대가 된다.
- E1~E3 모두 로봇·시뮬 실행이 필요하므로 이 보고서 범위 밖이며, 실행 전 사용자 승인 대상이다.

---

## 8. 검색 범위 내 "미발견" 진술과 그 근거

- **진술 1**: "높이맵 기반 로봇 퍼내기 학습 논문 중 절단면(렛지) 각을 DEM/시뮬 보정 목표로 쓴 사례는 **이번 검색 범위 내에서 미발견**." 근거 검색: §10 행 2, 7, 38, 39, 40, 46, 47, 48, 49 + 3, 4, 5, 19 (arXiv MCP 5건·Consensus 5건·alphaXiv 3건·exa 3건 = 16 검색어 × 4 소스). 반례 후보로 검토한 것: DDBot(판 결과로 동정하지만 렛지 각은 안 잼), FLIP(부은 각), Aoshima & Servin(현장 안식각 32°, 측정법 미기재).
- **진술 2**: "로봇 퍼내기·굴착 학습 논문이 안식각을 실측한 사례 미발견(GRAIN의 '안식각 근처 18°' 초기화만 준-사례; 분말 계량 S26은 과제가 달라 제외)." 근거: 위와 동일 + 본문 확인 12편(S16–S19, S22–S24, S13–S15, S25, S26).
- 두 진술 모두 "없다/최초"가 아니라 "검색 범위 내 미발견"으로 한정한다(HARD RULE #4).

---

## 9. 출처 표 (40건 이내)

열: 출처 · URL/DOI · 연도 · 확인 방식[본문/초록/2차] · 핵심 수치 · 우리 질문과의 관계

| # | 출처 | URL/DOI | 연도 | 확인 | 핵심 수치(확인한 것) | 관계 |
|---|---|---|---|---|---|---|
| S1 | Katagiri, Shoji, Yoneda, Takeya. Limits of identifiability… angle of repose as the objective function. Comput. Part. Mech. 14:334–345 | doi:10.1016/j.cpms.2026.03.008 | 2026 | 초록(+S2 인용으로 교차) | DEM 10,000회, 파라미터 7개, σ_AoR 0.5–3.0°; 구름마찰 기여 최대·불확실성 증폭; 안식각 단독 = 불량조건 | Q1 A 방식의 비유일성 |
| S2 | Gaboriault et al. From Angle of Repose to Heap Morphology | arXiv:2605.09371 | 2026 | 본문 | 3방법 48.65/49.34/41.50°; 최적 파라미터 군집 상이; SAD 픽셀 매칭; 670 시뮬 | Q1 A·E |
| S3 | Roessler & Katterfeld. Standard calibration procedure… Part I (ambiguous combos); (Part II Richter 2020 GSMC) | Powder Technol. (2019); (2020) | 2019 | 초록 | 개조 배출 시험 = 한 시험에서 다중 기준값 → "거의 유일" | Q1 B |
| S4 | Coetzee. Calibration of DEM: strategies for spherical/non-spherical | Powder Technol. 364:851–878 | 2020 | 초록 | 배출 시험만으로 μs·μr 결정, 강성은 구속 압축 | Q1 B·D |
| S5 | Coetzee & Els. DEM calibration and dragline bucket filling | J. Terramech. 47 | 2010 | 초록 | 구속 압축→강성, 안식각→마찰; 드래그 힘 초반 1/3만 정확 | Q1 D, 힘 vs 형상 분리 |
| S6 | Grima & Wypych. Calibration methods for DEM | Granular Matter 13 | 2011 | 초록 | 부은/배출 안식각 + 호퍼 유량으로 검증 | Q3 부은 vs 배출 |
| S7 | Marín Pérez et al. Sliding/rolling friction of cohesionless bulk | Particuology (2024) | 2024 | 초록 | 배출 시험 4기준, 구멍 3종 → 작은 가용영역, 최대 편차 5.9% | Q1 B |
| S8 | Mohajeri, van Rhee, Schott. Feasibility and definiteness in DEM calibration | doi:10.1016/j.apt.2021.02.044 (APT 32(5):1532–1548) | 2021 | 본문(로컬 PDF 추출) | 렛지 α_M 63–84°(표 2); 최적화 목표에서 렛지 제외, 검증 시뮬 90° vs 84° (7.1%); 상자 250 mm(문 200 mm) | Q1 C, Q3 |
| S9 | Mohajeri, van den Bos, van Rhee, Schott. Bulk properties variability… cohesive iron ore | doi:10.1016/j.powtec.2020.04.018 | 2020 | 본문 발췌(exa) | 렛지 상자 250×215×80 mm, 붓기 ~10 cm; 55–70°, I2 +2% 84°; 펠릿 40°(Lommen) | Q1 C, Q4 상자 규모 |
| S10 | Lommen, Mohajeri, Lodewijks, Schott. DEM particle upscaling… grab | doi:10.1016/j.powtec.2019.04.034 | 2019 | 본문 발췌(exa) | 펠릿 렛지 41° / 원뿔 26°; 상자 300×200×300 mm; 폭=18d, 벽 근처 제외 | Q1 C, Q3 벽효과 |
| S11 | Mohajeri. Grabs and Cohesive Bulk Solids (TU Delft PhD) | doi:10.4233/uuid:b232e542-4881-4b02-8677-a7b1dd37b6b0 | 2021 | 본문 발췌(exa) | "그랩 적재 부피는 주로 안식각이 결정"; 선창 실물 검증 | 그랩 전이(슬라이드 16) |
| S12 | Servin, Berglund, Nystedt. Multiscale terrain dynamics | arXiv:2011.00459 / AMSES 8:11 | 2021 | 본문 | 안식각 δ_b 셀룰러 오토마타 붕괴; 비응집이면 내부마찰각≈안식각; 가상 삼축 사전 보정; 참조 DEM 대비 10–25% | Q2 붕괴, Q3 |
| S13 | Aoshima & Servin. Sim-to-reality gap of a wheel loader | arXiv:2310.05765 | 2024 | 본문 | 옆·뒤 수직벽 더미; 자갈 30–40 mm, 1727 kg/m³, 안식각 32°; 보정 2개(μ 0.2, 강성배수 0.01); 격차 ~10%; 적재 3회 | Q1 G, Q2 벽 구속 |
| S14 | Aoshima, Fälldin, Wadbro, Servin. World modeling for autonomous wheel loaders | arXiv:2309.12016 / Automation 5(3):259–281 | 2024 | 본문 | 시드 더미 6종(20°/30°)+Perlin; 10,718표본; 붕괴로 5.2 m 창; MAE 0.75 m³, MRE 3.04%; 40회 누적 발산→CA | Q2 초기상태·붕괴 |
| S15 | Aoshima, Wadbro, Servin. Optimizing wheel loader performance | arXiv:2501.06583 / Automation 6(3):31 | 2025 | 본문 | 사다리꼴 1.8 m 30°; d=4 수렴 5.6%; 질량 −1.1%, 시간 5.0%, 일 6.7% | Q2, 함정 ③ |
| S16 | Yang, Wei, Lai, Ji. DDBot | arXiv:2510.17335 | 2025 | 본문 | 상자 0.28×0.28×0.11 m, 충전 7 cm; 삽질 1회로 E·ν·ρ·φ 동정 5–20분; φ≈18.9°/19.0° | Q1 G, Q4 |
| S17 | Liu, Li, Hauser. L-GBND | arXiv:2503.23270 (ICRA 2026) | 2026 | 본문 | SAPIEN 다이아몬드-스퀘어; 실물 0.9×0.6×0.2 m, 100궤적 미세조정; 안식각 25–40° 언급; 산사태 한계; 2D CNN 대비 −37% | Q2, 함정 ③ |
| S18 | Zhu, Thangeda, Ornik, Hauser. CoDeGa (RSS 2023) | arXiv:2303.02893 | 2023 | 본문 | 트레이 0.9×0.6×0.2 m; 손지형 ≤30°, ≤0.2 m; 5,100 실스쿱; 31.3/260.8 cm³ | Q2, Q4, 함정 ③ |
| S19 | Thangeda et al. OWLAT deployment | arXiv:2311.17405 | 2023 | 본문 | 빈 0.9×0.7×0.2 m; 깊이 0.2–0.8 cm; 평평 Regolith+둔덕; 질량 1.9/22.0/63.9 g | Q2 평평 초기 |
| S20 | Sandzimier & Asada. Data-driven bucket-filling | IEEE RA-L 5(2):2682–2689; hdl.handle.net/1721.1/128006 | 2020 | 초록 | 이분산 GP; 분산이 궤적과 "직전 흙 표면 형상"에 의존 | Q2(본문 미확보) |
| S21 | Zhang et al. Autonomous excavator system (AES) | doi:10.1126/scirobotics.abc3164 | 2021 | 초록+2차(EurekAlert) | LiDAR+카메라; 24 h 무개입; 소형 67.1 m³/h | Q2 현장 |
| S22 | Schenck, Tompson, Fox, Levine. Learning robotic manipulation of granular media (CoRL 2017) | arXiv:1709.02833 | 2017 | 본문 | 핀토콩 3.75 kg 한쪽 칸 평평; ~15,000 행동; 1 cm 셀 | Q2, Q4 |
| S23 | Kreis et al. Interactive shaping of granular media using RL | arXiv:2509.06469 | 2025 | 본문 | 평평 6 cm; 안식각 기반 붕괴 시뮬(Kim/Pavlov); 제로샷; 3.4 mm | Q2 붕괴 규칙 |
| S24 | Hu, Qian, Seita. GRAIN (CoRL 2024) | arXiv:2407.01898 | 2024 | 본문 | 60×60×20 cm, 6 mm BB, 경사 18°≈안식각; 실물 100회; MAE 1.13 cm | Q2 붕괴 학습, Q4 |
| S25 | Matl, Narang, Bajcsy, Ramos, Fox. Inferring material properties of granular media (ICRA 2020) | arXiv:2003.08032 | 2020 | 본문 | 쿠스쿠스 4 mm; 요약통계 16; (μs, μr, e)=(0.6687, 8.15e-7, 0.7689); 2–10 cm 일반화 | Q1 F |
| S26 | Radulov et al. FLIP | arXiv:2506.03896 | 2025 | 본문 | 자동 안식각 |오차| 0.84°; BO ≤1.5°; 계량 2.12 vs 6.11 mg | Q1 F, 절차 동형 |
| S27 | Kadokawa, Hamaya, Tanaka. Robotic powder weighing from simulation (IROS 2023) | IEEE IROS 2023 pp. 2932–2939 | 2023 | 초록 | DR 시뮬; 0.1–0.2 mg 오차, 목표 5–15 mg | Q4 |
| S28 | Takahashi, Ko, Ummadisingu, Maeda. Target-mass grasping of granular foods (ICRA 2021) | arXiv:2105.12946 | 2021 | 본문 | 트레이 603×377×145 mm; 커피콩 17/22/27 g, 쌀 45/60/75 g; 저울 1 g; 1,000회; 5%/10% 성공률 | Q4 한 입 질량 |
| S29 | Clarke. Robot learning for manipulation of granular materials using vision and sound (thesis) | Consensus 항목 0c4f9ef1… | 2019 | 초록 | 7,380 스쿱; 질량 RMSE 5.8 g, 높이맵 RMSE 0.38 cm | Q4 기준선 |
| S30 | Kang et al. Learning framework… dispense granular material on-demand (ASME CIE) | ASME IDETC-CIE 2024 Vol. 2B | 2024 | 초록 | mg급; 잔량 적으면 더미 생성 후 스쿱; 시간 55.2% 단축 | Q4 잔량 효과 |
| S31 | Al-Hashemi & Al-Amoudi. A review on the angle of repose of granular materials | doi:10.1016/j.powtec.2018.02.003 (330:397–417) | 2018 | 본문(OA PDF 추출) | 동적 = 정적 −(3~10°); 배출>부은; 안식각 = 최이완 상태 내부마찰각; 속도·질량·높이↑→각↓ | Q3 정의 |
| S32 | Kleinhans et al. Static and dynamic angles of repose under reduced gravity | J. Geophys. Res. Planets 116 E11004 | 2011 | 초록(저널명 정정) | 모난 ~40°, 둥근 ~25°; 저중력에서 정적 +5°, 동적 −10° | Q3 붕괴 |
| S33 | Cheng & Zhao. Static vs dynamic angle of repose of uniform sediment | Int. J. Sediment Res. | 2017 | 초록 | 0.28–4.38 mm; (상+하)/2 ≈ 동적; 차이는 입경↑ | Q3 |
| S34 | Elekes & Parteli. Expression for angle of repose of dry cohesive granular materials | PNAS 118 | 2021 | 초록 | 원뿔→불규칙 전이(응집/중력비 임계) | Q3 응집 |
| S35 | Datta et al. Angle of repose for superquadric particles | Comput. Geotech. (2024) | 2024 | 초록 | 30 형상, m&m·핀토콩 모사; 종횡비 모델 R² 98.6% | Q3 형상, Q4 PP 알 |
| S36 | Pont, Gondret, Perrin, Rabaud. Wall effects on granular heap stability | EPL (Europhys. Lett.) | 2002 | 초록 | 간격↓ → 안정각·안식각↑ | Q3 벽 효과 |
| S37 | Duverger et al. Investigation techniques and physical aspects of the angle of repose | Granular Matter 26 | 2024 | 초록 | 클럼프 35.95±0.88° vs 볼록 31.26±0.95°; 안식각 비구성적, 임계마찰각 = 하한 | Q3 정의 |
| S38 | Lube, Huppert, Sparks, Hallworth. Axisymmetric collapses of granular columns | J. Fluid Mech. 508 | 2004 | 초록 | 미교란 중앙 원뿔 ~59°(a=1.7); r∞ = r_i(1+1.24a) | Q3 붕괴 후 형상 |
| S39 | Wei et al. DEM parameters for cohesive soils at different moisture contents (loader) | PLOS One | 2026 | 초록 | 안식각 30.83→37.13°(함수율↑); JKR; PSO 오차 ≤2.2% | Q3 함수율 |
| S40 | IMO MSC.1/Circ.1453 Rev.2 지침 + IMSBC IRON ORE 일정표(imorules) | IMO 문서 / imorules.com | — | 원문 발췌(exa) | "비응집 재료만 안식각 기재, 응집이면 Not applicable"; IRON ORE: Not applicable | 함정 ③ |

---

## 10. 검색 로그 (검색어 · 소스 · 건수), HARD RULE #4 준수 증빙

| # | 소스 | 검색어/대상 | 건수 | 메모 |
|---|---|---|---|---|
| 1 | arXiv MCP | all:"discrete element" AND all:calibration AND (excavation OR bucket OR scooping) | 1 | S12 |
| 2 | arXiv MCP | all:granular AND all:heightmap AND (scoop OR excavat OR "wheel loader") | 0 | |
| 3 | Consensus | DEM parameter calibration angle of repose bucket filling excavation simulation | 10 | S1, S5, S39 |
| 4 | Consensus | ledge angle of repose iron ore pellets DEM calibration Mohajeri | 10 | |
| 5 | Consensus | draw down test versus poured angle of repose DEM calibration bulk material | 10 | S3, S4, S6, S7 |
| 6 | alphaXiv discover | DEM calibration excavation scooping granular | 7 | S2 |
| 7 | alphaXiv discover | heightmap scooping granular learning excavator wheel loader | 12 | S16, S23, S14 |
| 8 | exa | Mohajeri "ledge angle of repose" iron ore pellets APT | n/a | S8–S11 발췌 |
| 9 | WebSearch | Sandzimier Asada 2020 bucket-filling RA-L | 10 | S20 |
| 10 | WebSearch | "autonomous excavator system" Science Robotics 2021 Zhang | 10 | S21 |
| 11–16 | alphaXiv pdf | S17, S16, S18, S15, S12, S2 본문 질의 | 본문 | |
| 17 | Consensus | infer granular material properties robot interaction depth camera pouring heap Bayesian | 10 | S25, S29 |
| 18 | Consensus | static vs dynamic angle of repose avalanche particle shape moisture cohesion review | 10 | S32–S35 |
| 19 | Consensus | robot learning scooping granular media heightmap prediction tray small scale | 10 | S22, S29, S30 |
| 20 | WebFetch | TU Delft 리포지토리 S8 PDF | 이미지 PDF | 후속 로컬 추출(56) |
| 21–25 | alphaXiv pdf | S13, S22, S14, S23, Lu/Zhu/Zhang 2201.11292(이산 물체라 표 제외) | 본문 | |
| 26 | Consensus | target mass grasping granular food pile gripper self-supervised | 10 | S28 |
| 27 | Consensus | robotic powder weighing scooping spoon simulation sim-to-real lab automation | 10 | S27, S26 |
| 28 | Consensus | angle of repose heap against wall confined container boundary effect wall friction | 9 | S36, S37 |
| 29 | WebSearch | Sandzimier Asada bucket-filling pdf dspace | 5 | |
| 30 | exa | Al-Hashemi 2018 review angle of repose full text | 5 | S31 OA PDF |
| 31 | python | exa 결과에서 'ledge' 발췌 | n/a | 상자 치수·각 |
| 32 | WebFetch | dspace handle 1721.1/128006 | 405 | 차단 |
| 33 | WebFetch+pdftotext | S31 OA PDF | 본문 | |
| 34 | WebFetch | Baidu Research 블로그(AES) | n/a | 점검 중 페이지 |
| 35 | alphaXiv pdf | S24 GRAIN | 본문 | |
| 36 | Consensus | granular column collapse final deposit slope angle experiments run-out | 10 | S38 |
| 37 | Consensus | grab clamshell bulk handling DEM validation scaled experiment iron ore pellets ship unloading | 10 | S10, S11 |
| 38 | Consensus | excavation cut slope face angle after digging bucket DEM validation heightmap remaining terrain shape | 10 | 로봇 퍼내기 무관 결과만 |
| 39 | arXiv MCP | all:"angle of repose" AND all:robot AND (scoop OR excavat OR heightmap) | 0 | |
| 40 | arXiv MCP | ("cut face" OR ledge OR "trench wall") AND granular AND (robot OR excavat) | 0 | |
| 41 | exa | IMSBC angle of repose non-cohesive iron ore pellets not applicable | 5 | S40 |
| 42 | exa | Lommen Schott ledge angle pellets wedge penetration | 5 | S10 |
| 43 | alphaXiv pdf | 제목 'A Data-Driven Approach…Bucket-Filling' | 오해석 | ExACT 반환 → 미인용 |
| 44 | exa fetch | science.org abc3164 + EurekAlert + dspace | 초록/2차 | S21 |
| 45 | alphaXiv pdf | S23 안식각 값 질의 | 본문 | 값 없음 |
| 46 | arXiv MCP | all:scooping AND ("discrete element" OR DEM) AND robot | 0 | |
| 47 | arXiv MCP | granular AND manipulation AND calibration AND simulator AND real | 3 | 무관 |
| 48 | alphaXiv discover | angle of repose sim-to-real scooping calibration granular | 10 | S26, S16 재확인 |
| 49 | exa | robot scooping sim-to-real angle of repose measured to calibrate simulation heightmap | 8 | S25 |
| 50 | exa | Katagiri 2026 identifiability OA pdf | 4 | 초록만 |
| 51–54 | alphaXiv pdf | S19, S25, S26, S28 본문 | 본문 | |
| 55 | alphaXiv pdf | 제목 'Learning Robotic Powder Weighing…' | 오해석 | FLIP 반환 → S27 초록 유지 |
| 56 | pdftotext | S8 로컬 PDF 표 2·표 10 | 본문 | 63–84°, 90 vs 84 |
| 57 | pdftotext | S31 로컬 PDF | 본문 | 정의 발췌 |
| 58 | curl | dspace REST | 404 | S20 초록 유지 |

소스 종류 6종(arXiv MCP, Consensus, alphaXiv discover/pdf, exa, WebSearch/WebFetch, 로컬 PDF 추출). 원 로그 파일: 세션 스크래치패드 `search_log.tsv`(58행).
