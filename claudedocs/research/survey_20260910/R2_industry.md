# R2 — 산업 조사: 벌크 원료 그랩 크레인 하역의 실제 운용

- 작성일: 2026-09-10 · 작성: Claude(Orca worker, task_24691eb73394) · 워크트리 `research-survey`
- 목적: 우리 축소 실물(로봇팔 + PP 펠릿 상자 + 위 고정 깊이 카메라)이 재현하려는 **원료야드·선창 그랩 하역**이 실제로 어떻게 돌아가는지 확인하고, 실험 설계(초기 표면, 벽 있는 상자, 관측 주기, 실패 정의)를 산업과 맞추는 데 필요한 근거를 모은다.
- 조사 규칙 준수: 로봇·시뮬 실행 0, 코드 수정 0, 상태 원장(START_HERE/DECISIONS*/EXPERIMENT_LEDGER/session_*/relay) 미접촉, 커밋 0. 출처 40건(§8 표). 검색 로그 §9. "없다/최초/유일" 단정은 **쓰지 않았고**, 업체·논문이 스스로 주장하는 "최초"는 주장 주체를 밝혀 그대로 옮겼다(§1.6).
- 검증 표기: 표의 "확인 방식" 열에서 **본문** = 원문 본문(PDF·제품 문서·규정 원문)을 직접 읽고 수치를 확인, **초록** = 초록·요약만 확인, **2차** = 뉴스·포럼·블로그 등 2차 자료.

---

## 0. 용어 한 줄 풀이 (처음 나오는 순서)

| 용어 | 뜻 |
|---|---|
| 그랩(grab, 抓斗) | 크레인 줄 끝에 달려 벌어졌다 오므라들며 원료를 한 입씩 퍼 올리는 집게형 버킷. 조개형(clamshell)·가위형(scissors) 등. |
| 선창(cargo hold) | 벌크선의 화물칸. 위는 해치(hatch, 뚜껑 열리는 구멍)로 열리고, 옆·아래는 경사진 탱크 벽으로 둘러싸인다. |
| 코밍(hatch coaming) | 해치 구멍 둘레의 세로 벽. 그랩이 이 밑으로 들어가야 가장자리 원료를 퍼낼 수 있다. |
| 호퍼(hopper) | 크레인이 퍼 올린 원료를 부어 넣는 깔때기. 위치가 고정돼 있다. 우리 "고정 배출 용기"와 같은 역할. |
| 트리밍(trimming) | 적재 후 화물 표면을 고르게 펴는 작업. 하역 때는 불도저로 구석 원료를 가운데로 밀어 그랩이 닿게 하는 작업도 같은 말로 부른다. |
| 안식각(angle of repose) | 마른 알갱이를 쌓았을 때 저절로 서 있는 최대 경사각. |
| 응집성(cohesive) / 자유유동(free-flowing) | 젖거나 미세해서 서로 달라붙어 수직 벽이 서는 재료 / 마른 알갱이처럼 흘러내려 안식각 이상으로 못 서는 재료. |
| 프리 디깅(free digging) | 그랩이 표면 어디든 자유롭게 퍼낼 수 있는 하역 초반 단계. |
| 클린업(clean-up) | 그랩이 못 닿는 구석·바닥 잔량을 불도저(페이로더)와 사람이 모아 퍼내는 마지막 단계. |
| 사이클 타임 · 충진율(fill ratio) | 한 번 퍼서 붓고 돌아오는 데 걸리는 시간 · 그랩 용량 대비 실제 담긴 양의 비율. |
| LiDAR / 3D 레이저 스캐너 | 레이저를 쏘아 거리를 재서 표면의 3차원 점구름(point cloud)을 만드는 센서. |
| RTK-GPS · INS · TOF 카메라 | 센티미터급 위성 측위 · 관성항법장치(가속도·자이로로 위치·자세 추정) · 빛의 왕복시간으로 거리를 재는 3D 카메라. |
| heightmap(높이 지도) | 위에서 내려다본 격자마다 표면 높이를 적은 2차원 배열. 우리 예측 모델의 입력. |
| DEM(이산요소법) | 알갱이 하나하나를 입자로 두고 충돌·마찰을 계산하는 시뮬레이션. |
| 사전압밀(pre-consolidation) | 위에 쌓인 화물 무게로 아래층이 눌려 단단해진 상태. 하역이 깊어질수록 그랩 침투가 어려워진다. |
| IMSBC Code · BLU Code | 국제해사기구(IMO)의 고체 벌크 화물 규정 · 벌크선 안전 적재/하역 실무 규정. |
| 안티스웨이(anti-sway) | 줄에 매달린 그랩의 흔들림을 억제하는 제어. |

---

## 1. 한눈에 답 — 사용자 세 질문

**Q1. 실제 현장은 벽이 있는 공간에 가득 차 있고 처음엔 평평한가?**
대체로 **그렇다, 단 "해치 아래는 평평, 해치 바깥(코밍·탱크 밑)은 안식각으로 비스듬"**이 정확한 그림이다.
- IMSBC Code 5.1.1은 "화물은 필요한 만큼 **적당히 평평하게 트리밍**해야 한다", 5.1.2는 "화물칸은 **가능한 한 가득**, 화물칸 경계까지 **가능한 한 넓게** 펴야 한다"고 규정한다[33].
- 비응집(마른 알갱이) 화물만 안식각으로 트리밍 등급을 나눈다(≤30°는 곡물 규정, 30–35°는 표면 높낮이차 Δh ≤ B/10·최대 1.5 m, >35°는 최대 2 m)[33]. **응집성 화물(석탄·젖은 철광석)은 안식각을 아예 쓰지 않고** 일반 조항만 적용한다(5.3.2)[33]. 철광석 펠릿·석탄·철광석 세 스케줄 모두 안식각 = "Not applicable"[34].
- 1979년 옛 BC Code(IMO A.434) 3.2.2는 물리적 그림을 직접 적어 두었다: "**해치 사각형 안은 평평하게 고르고, 나머지 화물은 화물칸 옆벽과 끝벽 쪽으로 거의 일정하게 경사**지게 둔다"[35].
- 따라서 우리 실험의 초기 상태는 "벽 있는 상자에 가득 채워 평평하게 고른 표면"이 산업 표준에 가장 가깝다. 봉긋한 더미(pile)는 적재 직후가 아니라 **야드 저장 더미나 하역 중간 상태**에 해당한다.

**Q2. 퍼내면서 사면·크레이터가 생기는가?**
**생긴다. 그리고 그 사면을 얼마나 가파르게 두느냐가 안전·효율 규칙의 핵심이다.**
- IMO BLU 매뉴얼 Annex 2는 하역을 4단계로 나눈다: ① 프리 디깅(사다리·코밍 파손 위험) ② 옆벽·호퍼 탱크 단계(**"선창 전체 면적을 고르게 퍼서 옆구리(wings)에 가파른 둑(steep banks)이 생기지 않게 하라"**, "**항상 화물의 가장 높은 지점에서 퍼라**") ③ 바닥판 단계 ④ 페이로더(불도저) 단계[36].
- 중국 항만 작업규정은 수치를 준다: "**선창 안 원료와 수평면의 각도는 70°를 넘기면 안 된다**", "두 그랩 깊이만큼 파면 한 그랩 너비만큼 옮겨 왕복 작업"[16]. 상하이 지역 규정은 "**그랩이 원료에 묻히면 즉시 보고하고 무턱대고 들어 올리지 말 것**"[18].
- iSAM(함부르크 Hansaport 완전무인 하역기)은 기능 목록에 "**무너지는 원료 벽(collapsing material walls)에 그랩이 묻혔을 때 탈출**"을 명시했고, 초기 과제로 "**'가파른' 재료를 어떻게 다룰 것인가**"를 꼽았다[1][3].
- 응집성 화물은 수직 벽이 서다가 한꺼번에 무너진다. 캐나다 TSB 보고서: 석탄(안식각 37°, 수분 11.3%) 잔여 더미 4.6–5.5 m 높이의 **면이 갑자기 갈라져 흘러내려** 선원이 1 m 깊이로 매몰 사망[37]. 미국: 굳은 석고 **벽 6 m×10.7 m(20×35 ft)**가 무너져 작업자 매몰 사망, 당시 선창은 85% 비운 상태[38].
- 자유유동 재료(펠릿)는 반대로 그랩이 자른 자국이 **곧바로 흘러내려 메워진다**. TU Delft 실선 실험·DEM: 응집성 철광석은 그랩 칼날 자국과 가파른 경사가 그대로 남지만, 자유유동 재료는 "**경사 안정성이 낮아 그랩이 자르자마자 입자가 흘러 자국이 흐트러진다**"[29].
- 즉 **우리 PP 펠릿은 "펠릿형(자유유동) 영역"**이라 크레이터가 얕고 금방 완만해지며, 석탄·젖은 철광석의 "벽 붕괴"는 재현되지 않는다(§4).

**Q3. 운전자는 어디를 먼저 퍼는가?**
공개 문서에서 확인되는 원칙은 다섯 가지이고, **재료에 따라 순서가 바뀐다**.
1. **높은 곳부터** — BLU 매뉴얼 "항상 가장 높은 지점"[36], 중국 규정 "抓高留高(높은 곳을 퍼서 평평하게)"[17], 자동화 표준 "작업 중인 열(row)의 최고 취점"[14], 창원대 실험실 자동 GTSU도 "원료가 가장 많은 곳 = LiDAR에 가장 가까운 점"[23].
2. **전체를 고르게, 층(layer)·열(row) 단위로** — "선창 전체 면적을 고르게"[36], "체계적이고 고른 패턴"(자동 모드 요건)[36-b], 자동화 표준 "**스캔한 원료 높이에 따라 분층 작업**"[14], 중국 특허 "잡기 방향은 아래로 한 단계씩, 대차 방향은 해측/육측으로 한 단계씩, 고랑(furrow) 단위 전진 + **그랩 매몰 방지 전략**"[22], 포스코DX "**한 줄씩 차례대로**"[20], Siwertell 반자동 "열 간격·층 깊이를 파라미터로 두고 왕복"[11].
3. **재료별로 다르게** — iSAM: "**펠릿처럼 잘 흐르는 재료는 사이클을 줄이는 '성능 지향' 전략**, **석탄·철광석처럼 잘 안 흐르는 재료는 처음부터 '구석에서부터(out of the corners)' 퍼내야 한다**"[1]. Hansaport는 광석 18종·석탄 12종을 다루며 "각 재료의 흐름·구름 특성이 달라 자동화를 그에 맞춘다"[4].
4. **좌우 균형** — 선박이 기울지 않도록 좌현·우현을 맞추어 퍼낸다[36-b].
5. **숙련자는 꼭 꼭대기를 노리지 않는다** — ETH·Liebherr 40 t 머티리얼 핸들러 실험: "전문 운전자는 봉우리가 아니라 **사면을 공략해 더 가득 담는** 경우가 많다", 최고 충진율 76%는 "평평한 표면·완전 접촉·충분한 깊이·**느슨한 재료**"라는 이상 조건에서 나왔다[31].

한 줄 결론: **초기 = 평평하게 채운 벽 있는 상자, 표면은 하역 중 층·열 단위로 내려가며 70° 이하 사면을 유지, 결정 규칙은 "가장 높은 곳/현재 열의 최고점"이 산업 표준 휴리스틱**이다. 이 휴리스틱이 최적이 아니라는 것은 ETH(2025)가 실기·시뮬 양쪽에서 수치로 보였다(§1.5) — 우리 연구의 "어디를 퍼야 총 시간·횟수·실패가 주는가"가 바로 그 지점이다.

---

## 1.1 I1 — 자동화 제품·기능 (선박 하역기 + 야드/옥내 그랩 크레인)

| 업체·시스템 | 무엇을 자동화 | 센서 | 선창 표면 스캔 | 그랩 위치 결정 논리(공개 범위) | 사고·복구 기능 | 확인 방식 |
|---|---|---|---|---|---|---|
| **iSAM AG — Hansaport 함부르크** (2010-01 운전 시작, 2011-01 준공; 35 t급 4기) | 완전 무인(원격 조종 아님), 중앙 관제 1명이 4기 감독 | RIEGL 3D 레이저 스캐너(선박·해치·원료 분포, 석탄을 100 m 거리에서 감지), RTK-GPS 2기(기계 위치), 그랩 위치: 초기엔 레이저링 자이로 INS를 그랩에 직접 장착(2010 문서), 현행은 실시간 3D 스캐너로 줄·그랩 추적(2025 문서), TOF 3D 카메라 4대 | 같은 스캐너로 해치 위치 + 해치 안 원료 프로파일 생성 | "선박 설계·재료 데이터·현재 센서 데이터에 따라 **최적 하역 전략을 산업용 PC가 결정**". 공개된 것은 **재료별 모드**(펠릿 = 성능 지향 짧은 사이클 / 석탄·광석 = 처음부터 구석에서) 뿐, 알고리즘 자체는 문서에 없음. 착지 정밀도 ≈0.5 m, 코밍 밑 자동 하역 | "**무너지는 원료 벽에 묻힌 그랩 탈출**", 예측 에너지·위치 모델로 급정지 시에도 충돌 0, 해치 교체 자동 | 본문[1][2][3][4] |
| iSAM — EMO 로테르담·말레이시아·바레인·캐나다 | 위와 동일 패키지 | 3D LiDAR + GPS 충돌 방지, 실시간 그랩 추적 | 동일 | "스마트 알고리즘이 **선박 균형을 유지하도록 최적 원료 분포**에 초점" (본 문장은 하역·적재 공통 서술) | — | 본문[5] |
| **ABB GSU Automation** (200기 이상 납품) | 운전자 보조형 반자동: 해치 안 안티스웨이, 호퍼 위 비행 투하, 그랩 닫힘/권상 제어, **그랩 경로 최적화**, "**해치·원료 스캐닝으로 지능형 자동 하역**" | 명시 없음(스캐닝 존재만) | 있음 | 페이지에 알고리즘 미기재 | — | 본문[6] |
| **Konecranes AGD 그랩 언로더** (Koper항: 32 t·1,500 t/h·아웃리치 40 m) | 4드럼 전기 동기 그랩, 수동/반자동/자동 모드 표준, 신호 30 ms 간격 기록 | 명시 없음 | 미기재 | 미기재 | — | 본문[7] |
| Konecranes Gottwald 모바일 하버 크레인 | "**자기학습 그랩 충진 수준 점검**", 해치별 카운터, 검증 계량 | 센서·카메라 | 미기재 | 충진 최적화(위치 선택 아님) | — | 본문[8] |
| **Konecranes 폐기물·바이오매스 벙커 크레인** (Hamburger Hungária 등) | **완전 무인** 수취/저장/혼합/투입 사이클 | 크레인 장착 **레이저 스캐너로 벙커 높이 3D 시각화**, 레이더 레벨 센서 | 있음(벙커 전체) | 사이클 규칙(수취→저장, 투입 최우선)만 공개 | 원격 조종석(ROS) 폴백 | 본문[9] |
| **Liebherr SmartGrip / Cycoptronic / Teach-in** (모바일 하버 크레인) | 그랩 **충진 자기학습**(5회 만에 재료 밀도·압축·입도 학습, 현장 평균 충진 70% → 최대 +30%), 안티스웨이, 해치→호퍼 반자동 왕복 | 크레인 하중·기하 | **선창 스캔·취점 선택 기능은 읽은 3개 문서에서 확인 못 함** | 없음(충진·궤적만) | 과부하 방지 | 본문[10] |
| **Bruks Siwertell** (스크류식 연속 하역기, 그랩 아님) | 반자동: 운전자가 시작점 지정 → **열(row)·층(layer) 파라미터로 왕복**, 저속 구역에서 방향 전환 | 충돌 방지 시스템, 토크/하중 | 없음(규칙 기반) | "**표면을 평평하게 유지해 선창 내 '원료 눈사태(cargo avalanches)'를 줄인다**" | 토크·하중 초과 시 감속 | 본문[11] |
| **중국 단체표준 T/CIN 012—2023** (山东港口 등 기초) | 청소(클린업) 제외 전 과정 자동, 원격 개입 | 레이저 스캐너(선형·해치·원료), 정위 2중화, 안티스웨이 ±200 mm | 원료 영상 정밀도 **±300 mm**, 렌더링 갱신 **≤60 s**; 실시간 최고점·최저점·**현재 열 최고 취점**, 갱신 **≤40 s**, 위치 오차 ±200 mm, **그랩 사이클마다 1회 재스캔** | "유효 원료 분포를 **행렬 데이터 모델**로 만들어 취료 전략 산출, **스캔 높이에 따라 분층 작업**", 선체 기울기(±0.2°) 반영해 최적 취점 자동 선택 | 이상 시 원격 모드 자동 전환 | 본문[14] |
| 중국 자동 하역기 **안전작업규정**(中国航海学会 2025) | 전자동은 클린업 층 위까지, **정지 30분 초과 시 재스캔**, 벽 50 cm 이내는 원격 전환 | 스캔 시스템 자기학습 확인 | 있음 | — | 자동 이상 시 즉시 정지 | 본문[15] |
| **중국 특허 CN116101901A** (2023) | 무인 그랩 하역기 제어 | 영상 감시 + 해치·원료 스캔 | 있음 | "**잡기 방향은 아래로 단계적, 대차 방향은 해측·육측 단계적, 그랩 매몰 방지 제어 전략, 고랑(furrow) 단계 작업**, 단일 선창 작업 계획"; 시작 전 모든 작업 단위의 적재 상황을 보고 **초기 단위 선택 → 중앙 구역부터** | 클린업은 원격으로 청소기 투입 | 본문(청구항·상세설명)[22] |
| **포스코DX GTSU 무인화** (포항·광양 원료부두) | 2026-03: 카메라+LiDAR 융합으로 원료 형상·높이·해치 크기 인식, **강화학습으로 효율 최대 지점 분석, 한 줄씩 하역**, 1명이 4기, 효율은 안전상 사람의 80%로 설정, 1/35 축소 모형 시연. 2026-06: 18–20만 t급, 그랩 20–25 t + 원료 20 t = 40–45 t, 안티스웨이 0.6–0.7°, Isaac Sim 가상환경, 7월부터 무인 시운전. 2026-07: 그랩 작업 80% 무인 목표 | 카메라 + LiDAR | 있음 | RL 기반(구체 미공개) | — | 2차(뉴스 3건)[19][20][21] |
| 현대제철 당진 | 원료부두는 **밀폐형 연속식 하역기(CSU)** — 그랩 방식이 아니어서 본 조사 대상 밖 | — | — | — | — | 2차(스틸데일리)[40] |
| 창원대 **실험실 자동 GTSU** (Ngo et al. 2024) | 146×146 cm 모형 선창(돌), VLP-16 LiDAR + QHD 카메라, YOLOv3 선창 검출 96%, 충돌 경고 90% | LiDAR + 카메라 | 있음 | **작업점 = LiDAR에 가장 가까운 점(=가장 높은 곳, "원료가 가장 많은 곳")**, 손 측정 대비 5–10% 오차 | — | 본문[23] |
| 상하이 등 **Sci. Rep. 2025** (안전 최적화) | RGB-D(D455)+Livox Mid-70+IMU SLAM, 3D 지도 RMSE 실험실 2.8 / 주간 4.3 / 야간 6.1 cm, 14–20 FPS, 충돌 예측 91.2%·1.4 s, 7일간 경보 68건 중 오경보 3 | 위 | 있음(안전용) | 취점 선택 없음 | 충돌 예측·경보 | 본문[24] |
| 우한이공대 **Xiao et al. 2026 IEEE T-TE** | 전 과정 지능 하역(WPIU): **3D 특징 분포의 슬라이딩 윈도 검출로 실시간 선창·원료 모델 → 더 안전·효율적인 파지 영역 예측**, 곡물·광석 항만 실증 | 다중 센서 + 청소 로봇 협동 측위 | 있음 | 위(초록 수준) | 청소 로봇 충돌 회피 | 초록[25] |
| 중국 선행 (2010–2013) | Qing 2010: 상하이 뤄징 광석부두 완전자동 그랩 하역기(**저자 스스로 "세계 최초" 주장**); Yang 2011: 3D 레이저 스캔으로 PLC가 원료 3D 형상·**취점·깊이** 결정; Hu 2013: 3D 윤곽 인식으로 **연속 자동 취점** | 3D 레이저 | 있음 | 취점 알고리즘(초록만) | — | 초록[26][27] |
| IHI **연속식 하역기 자동화** (JSME 2026) | 다중 LiDAR 복셀 모델 + **휴리스틱 경로 계획** + 피드포워드 굴착률 제어, 실항만 20분 연속·정격 대비 97% | 다중 LiDAR | 있음 | 휴리스틱(초록) | 근접 위험 0건 | 초록[28] |

정리: **센서는 3D 레이저 스캐너(LiDAR)가 표준**이고 카메라는 보조(먼지·야간 때문). **표면 스캔은 그랩 사이클마다 갱신(≤40–60 s)**하며 정밀도는 ±200–300 mm 수준이다. **결정 논리 중 공개된 것은 "최고점/현재 열 최고점 + 층·열 단위 진행 + 재료별 모드"**까지이고, iSAM·ABB의 실제 최적화 로직과 포스코DX의 RL 세부는 읽은 문서에서 확인되지 않았다(§9 로그 참조). 학술 쪽에서 취점 **예측**을 명시한 것은 Xiao 2026(파지 영역 예측)·ETH 2025(RL 취점, §1.5)·Hu 2013 정도다.

### 1.5 "가장 높은 곳" 휴리스틱은 최적이 아니다 — ETH Zürich + Liebherr 2025 (본문 확인)

40 t 머티리얼 핸들러(유압 크레인형, 조개형 그랩 1.5 t)로 6×6 m 더미(36–41 m³, 봉우리 2.3–2.6 m)를 옮긴 실기·시뮬 결과[31]:

| 조건 | 정책 | 평균 충진율 | 평균 횟수/성공률 |
|---|---|---|---|
| 시뮬 더미, 관측 잡음 있음 | RL 취점 | 62.7% | 16.0회 |
| 시뮬 더미, 잡음 있음 | 최고점 휴리스틱 | 14.6% | 60.0회 |
| 시뮬 더미, **잡음 없음(이상 관측)** | 최고점 휴리스틱 | 57.7% | 17.8회 |
| 시뮬 컨테이너(벽 있음), 잡음 있음 / 없음 | RL / 휴리스틱 / 휴리스틱 | 62.9 / 33.5 / 61.2% | 21.6 / 35.6 / 23.0회 |
| **실기** | RL 취점 | 67.3% | 성공 100% |
| 실기 | 최고점 휴리스틱 | 47.9% (성공분만 62.9%) | 성공 76% |
| 실기 사람 | Expert Fast / Expert Slow / Practiced / Novice | 66.7 / **76.0** / 73.3 / 58–65% | 사이클 17.4 / 21.6 / 24.2 / 21.9–34.1 s |

- 잡음이 없어도 최고점 휴리스틱은 RL보다 못했다(57.7 < 62.7%, 17.8 > 16.0회). 실기에서는 낙하 먼지가 점구름 위에 잡혀 "공중"을 취점으로 골라 **빈 그랩 24%**가 났다.
- 시뮬은 그랩 단면으로 heightmap을 파고 **셀룰러 오토마타로 임계 경사 이하가 될 때까지 흘러내림**을 계산한다(물리 솔버 없음). 우리 "퍼낸 뒤 남을 형상" 예측과 같은 문제를 규칙으로 푼 것.
- 해석: 우리가 비교하려는 "양만 보기(방법①) vs 양+남을 형상(방법②)"에서 방법①의 산업 대응물이 바로 이 최고점 휴리스틱이며, 산업 표준 규정(BLU 매뉴얼·T/CIN)이 그것을 명문화하고 있다. 즉 **베이스라인은 임의의 greedy가 아니라 "최고점 + 층/열 진행"으로 두어야 산업과 맞다.**

### 1.6 "최초" 주장 목록 (주장 주체 표기, 본 조사는 판정하지 않음)
- Hansaport 홈페이지: "세계 최초 해측(ship-to-shore) 자동 운전", "세계 유일 완전자동 벌크 항만"(운영사 주장)[2].
- Qing 2010(PLA理工大学 학보): 상하이 뤄징 광석부두 "세계 최초 완전자동 그랩 하역 솔루션"(저자 주장)[26].
- Banno et al. 2026(IHI): "우리가 아는 한 실항만 통합 CSU 자동화 실증은 처음"(저자 주장)[28].
- ETH 2025: "우리가 아는 한 실물 크기 머티리얼 핸들링 완전 자동화는 처음"(저자 주장)[31].
서로 겹치는 주장이 있으므로 우리 문서에는 어느 쪽도 "최초"로 인용하지 않는다.

---

## 2. I2 — 재료의 초기 상태와 하역 중 표면 변화, 붕괴·매몰

### 2.1 적재 직후: 규정이 요구하는 표면

| 조항 | 내용 | 우리에게 주는 뜻 |
|---|---|---|
| IMSBC 5.1.1[33] | 트리밍은 화물 이동과 공기 유입(자연발열)을 줄인다. **"필요한 만큼 적당히 평평하게"** | 초기 표면 = 평평 |
| IMSBC 5.1.2[33] | 화물칸은 **가능한 한 가득**, 바닥 과하중 없이, **경계까지 가능한 한 넓게** | 벽까지 채움 |
| IMSBC 5.3.2[33] | **응집성 화물은 안식각이 안정성 지표가 아니며 스케줄에도 없다** | 석탄·젖은 광석엔 안식각 논리 금지 |
| IMSBC 5.4.3–5.4.5[33] | 비응집: ≤30° 곡물 규정 / 30–35° Δh ≤ B/10 & ≤1.5 m / >35° Δh ≤ B/10 & ≤2 m (B = 선폭 m) | 우리 PP 펠릿의 안식각을 **직접 재야** 어느 등급인지 안다 |
| 스케줄[34] | 철광석 펠릿 10–40 mm, 안식각 **N/A**, 1,800–2,400 kg/m³, Group C (MSC.575(110) 2025 개정) · 석탄 ≤50 mm, N/A, 654–1,266 kg/m³, Group B(and A), "**적절히 트리밍하지 않으면 석탄 몸체에 수직 균열이 생겨** 산소가 순환" · 철광석 ≤250 mm, N/A, 1,250–3,500 kg/m³ | **"IMSBC 안식각 = 비응집 전용, 펠릿은 N/A"** 인용 함정 ④ 확인 |
| 1979 BC Code A.434 3.2.2[35] | "해치 사각형 안을 고르고 나머지는 옆벽·끝벽으로 거의 일정한 경사" | 해치 밖 사면은 원래부터 존재 |
| BLU 원칙[36-b] | 좌현/우현 균형, "체계적이고 고른 패턴"(자동·반자동 모드 요건) | 자동 모드의 명시 요건 |

### 2.2 하역 중 표면이 바뀌는 방식 (단계별)

| 단계 | 표면 상태 | 근거 |
|---|---|---|
| ① 프리 디깅 | 평평한 표면을 **높은 곳부터 층 단위**로 내린다. 사다리·플랫폼이 화물에 묻혀 안 보임. 자동화 표준은 매 사이클 재스캔 | BLU Annex 2[36], T/CIN 5.2.3[14] |
| ② 옆벽·호퍼 단계 | 해치 밑은 내려가고 코밍·탱크 밑 원료가 **둑(banks)**으로 남는다. "고르게 퍼서 가파른 둑을 만들지 말라", 규정상 **사면 ≤70°**. 그랩을 세로로 내린 뒤 **옆으로 던져 넣어** 코밍 밑을 판다 | BLU[36], 중국 규정[16], HHLA[4] |
| ③ 바닥판 단계 | 바닥 곳곳에 **둔덕(mounds)**이 남아 그랩 한쪽이 맨 강판에 떨어질 위험 | BLU[36] |
| ④ 클린업 | 불도저 1–2대가 **옆·구석 원료를 가운데로 밀어** 그랩이 집게 하고, 사람은 6 m 대나무 스크레이퍼로 긁어낸다. 한 선창 15,000 t 규모 | JTSB[39], BLU[36] |
| 응집성 특유 | 깊이에 따라 **사전압밀**(7 m 깊이 ≈ 200 kPa)이 커져 그랩 초기 침투가 0.67 → 0.37 m로 준다(사전압밀 0→300 kPa); 바닥 z/z_max ≥ 0.8은 젖은 "wet bottom". 자른 자국이 **가파른 경사로 남는다** | Mohajeri 2021[29] |
| 자유유동 특유 | 자른 자국이 **즉시 흘러내려 흐트러진다**; 사전압밀은 침투에 영향 없음 | Mohajeri 2021[29] |
| 그랩 사이클 5단계 | 하강 → 안착(줄 느슨) → 닫기 → 권상 → 현수 | Mohajeri 2021[29] |

### 2.3 붕괴·매몰 사고 (그랩·사람)

| 사례 | 무엇이 무너졌나 | 수치 | 출처 |
|---|---|---|---|
| CSL Atlas, 1996, 캐나다 | 자체하역선 석탄 잔량 더미의 **면(face)이 갑자기 갈라져 흘러내림** | 더미 4.6–5.5 m, 석탄 안식각 37°, 수분 11.3%, 기온 −14~−17 °C, 매몰 깊이 ≈1 m, 사망 1 | TSB M96M0002, 본문[37] |
| Pioneer, 2006, 미국 | 습기로 굳은 **석고 벽 6 m×10.7 m** 붕괴 | 선창 85% 비운 시점, 질식 사망 1 | Professional Mariner, 2차[38] |
| Hansaport 운용 | "**무너지는 원료 벽으로 그랩이 묻히는** 특수 상황에서 탈출" 기능 | — | iSAM 본문[1] |
| 중국 안전규정 | "그랩이 화물(특히 밀도 1.2 t/m³ 초과)에 묻히면 즉시 보고, **무턱대고 권상 금지**(기계 손상)" | — | 본문[18] |
| Siwertell | 표면을 평평하게 유지해 "**선창 내 원료 눈사태**"를 줄임 | — | 본문[11] |
| Giga 2, 1996, 호주 | (참고) 무너진 것은 원료가 아니라 **격벽** — 인접 선창 밸러스트수 유입 | — | ATSB, 2차 |

주의: 위 붕괴 사례는 전부 **응집성 재료(젖은 석탄·굳은 석고)**다. 마른 펠릿 계열에서 벽이 서다 무너지는 사례는 본 조사에서 확인하지 못했고, TU Delft 실험이 오히려 "자유유동은 즉시 흘러내린다"고 보고한다[29]. 우리 재료로는 "벽 붕괴"가 아니라 "**사면 흘러내림(avalanche)**"만 재현된다.

---

## 3. I3 — 운전자 관행: 순서, 사이클, 한 입, 실패

### 3.1 퍼내는 순서 (재료·단계별)

| 재료/단계 | 관행 | 출처 |
|---|---|---|
| 펠릿(자유유동) | **성능 지향 모드**: 사이클을 줄이는 쪽으로, 어디서 퍼도 흘러내려 메워지므로 순서 제약 약함 | iSAM[1] |
| 석탄·철광석(응집) | **처음부터 구석에서부터** — 나중에 벽이 서서 무너지는 것을 막기 위함(문서의 명시 이유는 "잘 흐르지 않기 때문") | iSAM[1] |
| 일반 원칙 | 가장 높은 곳 / 전체 면적 고르게 / 양 끝을 번갈아 / 좌우 균형 / 가파른 둑 금지 | BLU[36], bulkcarrierguide[36-b] |
| 중국 항만 | 선수→선미 방향, 평균 잡기, **두 그랩 깊이 → 한 그랩 너비 이동 왕복**, 사면 ≤70°, 두 선창 이상은 번갈아; 바닥 보이면 "바닥 모드" | [16] |
| 중국 문형 크레인 | "抓高留高"로 평평하게 만든 뒤 **"回"자형(동심 사각형) 순서**로 집기 | [17] |
| 자동화 표준 | 스캔 높이에 따른 **분층**, 현재 열 최고 취점 | T/CIN[14] |
| 특허 | 중앙 구역 → 아래로 한 단계, 해측/육측 한 단계, 고랑 단위 | CN116101901A[22] |
| 포스코DX | "한 줄씩 차례대로" | [20] |
| 숙련자 | 봉우리보다 **사면**을 공략해 더 담음; 느슨한 재료·평평한 면·완전 접촉·충분한 깊이일 때 최고 76% | ETH[31] |

### 3.2 사이클 시간·한 입 질량·처리량

| 항목 | 수치 | 출처·확인 |
|---|---|---|
| 프리 디깅 사이클 | 갠트리형 하역기 **약 60사이클/시간(≈60 s)** | Nemag 블로그(제조사, 2차)[12] |
| 한 입 | Hansaport **최대 25 t/그랩**, 그랩 10 m³ | Dry Cargo 2010[3], HHLA[4], 본문 |
| 한 입 | EMO 로테르담 석탄 그랩 90 t(순 60 t) | Dry Cargo[5] |
| 한 입·처리량 | Koper 32 t 리프트, 1,500 t/h | Konecranes[7] |
| 한 입 | 포스코 그랩 20–25 t + 원료 20 t | etnews[19], 2차 |
| 총 횟수 | **15만 DWT 1척 = 그랩 5,000–7,500 사이클** | Mohajeri 2021b 본문[30] |
| 선박 단위 | Hansaport: 292 m 벌크선 126,000 t 광석 ≈ **2일** | HHLA[4] |
| 충진율 | 현장 평균 70%(Liebherr 실측), SmartGrip로 최대 +30%; ETH 사람 58–76%, RL 67% | [10][31] |
| 효율 | JIS 기준 하역기 효율 50–60%; 화물 60%는 정격, 25%는 정격의 50%, **마지막 15%는 정격의 20%**; "트리밍(클린업)에서 시간이 가장 많이 샌다" | bulk-online 포럼·Nemag(2차)[13][12] |
| 사이클(다른 기계) | 40 t 머티리얼 핸들러 사람 17–34 s, RL 33–41 s | ETH[31] |

### 3.3 실패의 정의 (산업에서 세는 것)

| 실패 유형 | 설명 | 출처 |
|---|---|---|
| **빈 그랩 / 저충진** | 관측 잡음(낙하 먼지)으로 공중을 취점 → 성공률 76%; 충진율이 지표 | ETH[31], Liebherr[10] |
| **과적재** | 넘치면 살짝 열어 덜어냄 → 사이클 손실; 재료가 깊어지며 밀도·수분 변화로 과부하 | bulk-online[13], Liebherr[10] |
| **흘림(spillage)** | 선창을 나가기 전 완전 폐합 확인, 잠시 멈춰 가장자리 원료를 떨군 뒤 이동, 권상 초속 ≤30% | 중국 규정[16][17] |
| **매몰(buried grab)** | 벽 붕괴로 그랩이 묻힘 → 보고·탈출 절차 | iSAM[1], [18] |
| **구조물 충돌** | 코밍·사다리(1단계), 프레임·호퍼(2단계), 바닥판(3단계) 손상 | BLU[36] |
| **줄 이완·전도** | 줄을 너무 풀면 그랩이 기울어 전복 | [18] |
| 재스캔 누락 | 30분 이상 정지 후 재스캔 없이 재개 금지 | [15] |

---

## 4. I4 — 우리 축소 실물 ↔ 산업 대응표

우리 쪽 값 출처: `START_HERE.md`(Active Case), auto-memory 79th(09-07) — S1 그랩(고정 반쪽 보울 + 서보 문, 0–30°), PP 펠릿(쌀알형, 긴쪽 3.8±0.3 mm, 반투명), 5회 사이클 5/5·93 s/회, 회당 적재 질량 **미계량**(설계값 27.1 g, 립 36 mm), 펠릿 밀도·안식각·마찰 **미실측**, Kinect DK 위 고정(hand-eye RMSE 10.13 mm), 배출 용기 = 컵(고정).

| 축 | 산업(선창 그랩 하역) | 우리 축소 실물 | 대응 여부 |
|---|---|---|---|
| 기하 스케일 | 해치 15–20 m × 16–19 m, 선창 1개 15,000 t, 그랩 10 m³/25 t, 침투 0.4–0.7 m | 상자 수십 cm, 그랩 립 36 mm, 한 입 ≈27 g(설계) | 선형 비율 대략 1/300–1/500. 포스코DX는 1/35 모형, 창원대 1.46 m, UIUC(CoDeGa)·L-GBND 0.9×0.6×0.2 m 상자를 썼다 — 우리도 축소 실물 계보 안에 있음 |
| 벽·경계 | 수직 코밍 + **경사 호퍼 탱크 + 데크 밑 오버행** → 해치 바깥 원료는 그랩이 직접 못 봄·못 닿음, 불도저 필요 | **수직 벽·위가 완전 개방**된 상자 | ⚠️ 부분 대응. 우리 상자는 "해치 사각형 안"만 재현. 오버행·클린업 단계는 없음 |
| 재료 응집성 | 석탄·철광석 미분 = 응집·사전압밀·젖은 바닥·벽 붕괴 / 펠릿 = 자유유동 | 마른 PP 펠릿 3.8 mm = **자유유동** | ✅ "펠릿 모드"에만 대응. 석탄·젖은 광석의 벽 붕괴는 재현 불가(§2.3) |
| 초기 표면 | 트리밍된 평평 표면, 벽까지 가득 | 현재 설계는 "더미"인지 "평평"인지 미확정 | ⚠️ §5 (a) |
| 표면 진화 | 층·열 단위 하강, 사면 ≤70°, 자유유동은 자국이 즉시 흘러내림 | 펠릿 안식각 미실측 | (c) 안식각·흘러내림 실측 필요 |
| 관측 | 크레인 장착 3D 레이저 스캐너, **매 그랩 사이클 재스캔**, ±200–300 mm, 갱신 ≤40–60 s, 먼지·그랩 가림으로 단일 스캔이 듬성듬성(다중 프레임 정합 필요) | Kinect 위 고정, mm급, 팔이 가림 | ✅ 관측 주기(사이클당 1회)는 일치. ⚠️ 산업의 관측 잡음(먼지·가림)이 최고점 휴리스틱을 무너뜨리는 주 요인(ETH) — 우리 잡음은 훨씬 작다 |
| 배출 위치 | 호퍼 고정 | 컵 고정 | ✅ |
| 결정 규칙(산업 표준) | 최고점/현재 열 최고점 + 층·열 진행 + 재료별 모드 | 학습 예측 모델 | 베이스라인을 산업 규칙으로 두어야 함 |
| 성능 지표 | t/h, 사이클/h, 충진율, 선박 손상 0, 안전 | 총 시간·총 횟수·실패율 | ✅ 대응(충진율 = 회당 질량/최대치) |
| 종료 조건 | 자동은 클린업 층 위까지; 마지막 15%는 정격 20% 속도 | "정해진 양 완료" | ⚠️ 우리 "완주"는 산업의 "자동 구간 종료 + 클린업"에 해당. 바닥 근처 저효율 구간을 별도로 봐야 함 |
| 그랩 기구 | 4줄 조개형, 줄 장력으로 폐합, 자중으로 침투 | 서보 문 + 고정 보울, 폐합 토크 900 | ⚠️ 침투 물리가 다름(자중 vs 서보). 단 "한 입 양과 남는 형상"이라는 함수 관계는 동형 |
| 실패 정의 | 빈 그랩·저충진, 과적재, 흘림, 매몰, 충돌 | 미정의 | (a) 산업 정의 채택 |

---

## 5. 권고 — 이 조사가 우리 절차를 바꾸는가

### (a) 바꾼다

1. **초기 표면 = "벽 있는 상자에 가득 채워 평평하게 고른 표면"을 1차 조건으로.** 근거: IMSBC 5.1.1–5.1.2[33]가 요구하고, A.434[35]와 BLU[36]가 "해치 안 평평 → 하역 중 둑·사면 발생"이라는 진화를 명시한다. "봉긋한 더미"는 초기 조건이 아니라 **실험이 진행되면 저절로 생기는 중간 상태**다. 더미로 시작하면 산업의 첫 단계(프리 디깅, 층 하강)를 건너뛰는 셈이다. 더미는 야드 저장 더미(reclaimer 문제)에 해당하므로 2차 조건으로 남긴다.
2. **벽 있는 상자 유지 + 한 변에 경사 벽(호퍼 탱크 모사) 추가 검토.** 근거: 산업의 "어디를 퍼야 하나" 난이도는 벽 근처 둑과 사면에서 생긴다[36][16]. 수직 벽만 있으면 둑 문제가 약해진다. 단 이는 (c)의 실험으로 효과를 본 뒤 결정.
3. **관측 주기 = 그랩 사이클마다 1회, 결정 직전.** 근거: T/CIN 5.2.3 "매 그랩 사이클 1회 스캔"[14], 안전규정 "30분 정지 후 재스캔"[15]. 우리 현재 계획과 같으므로 "산업 근거 있음"으로 문서화만 하면 된다. 추가로 **팔이 화면에서 빠진 뒤 촬영**(Shen 2025: 그랩 가림으로 단일 스캔 결손)[27].
4. **실패 정의를 산업 목록으로 교체**: ① 빈 그랩/저충진(충진율 < 임계, 회당 질량 계측 필수) ② 흘림(집은 양 − 부은 양) ③ 매몰·정지(서보 전류 스톨) ④ 벽 충돌. 근거 §3.3. 회당 적재 질량 미계량(79th)은 이 정의를 막는 첫 병목이다.
5. **베이스라인을 "최고점 휴리스틱 + 층/열 진행"으로 명시.** 근거: BLU "항상 최고점"[36], T/CIN "현재 열 최고 취점·분층"[14], 창원대 "가장 가까운 점"[23], 포스코 "한 줄씩"[20]. ETH[31]가 이 휴리스틱이 잡음 없이도 RL보다 못함을 보였으므로 우리 방법①(양만 보기)의 가장 강한 산업 대응물이다. 순수 greedy만 두면 산업 관행과 어긋난다.
6. **재료 영역을 문서에 못 박기**: 우리 결과는 "자유유동 펠릿 영역"(iSAM의 성능 지향 모드)에 한정되며 석탄·젖은 철광석의 벽 붕괴·사전압밀은 다루지 않는다. 근거 [1][29][37]. 프로포절 서술에서 "석탄·철광석 하역"을 예시로 들 때 이 한정어를 붙인다.

### (b) 유지

- **heightmap 파이프라인·위 고정 깊이 카메라**: 산업도 위에서 내려다보는 3D 스캔이 표준[1][14][23]. 카메라가 크레인에 달려 움직인다는 차이는 우리 목적(결정 모델)에 영향이 없다.
- **배출 위치 고정**: 호퍼 고정과 동형[4][11].
- **지표 = 총 시간·총 횟수·실패율**: 산업의 t/h·사이클/h·충진율·손상 0과 1:1 대응(§4).
- **축소 실물 접근 자체**: 포스코DX 1/35 모형, 창원대 1.46 m, UIUC 0.9 m 상자 등 동일 계보가 있어 방법론 정당화에 쓸 수 있다.
- **로봇 동작(정해진 경로)·학습 대상 한정**: 산업 자동화도 궤적·안티스웨이는 규칙/제어이고 "어디를"만 전략 층에 둔다[1][6][14].

### (c) 추가 실험 필요 (결정을 바꿀 수 있는 것만)

1. **펠릿 안식각·흘러내림 실측** (이미 START_HERE 미실측 항목): IMSBC 등급(≤30° / 30–35° / >35°)이 갈리고, 사면 ≤70° 규칙이 우리 재료에서 의미 있는지 결정한다. 실측 없이는 "크레이터가 남는가"에 답할 수 없다.
2. **벽 효과 실험**: 상자 폭/그랩 폭 비를 2단계로 바꾸어 벽 근처 둑이 형성되는지, 최고점 휴리스틱이 벽 근처에서 실패(빈 그랩·충돌)하는지 본다. 결과에 따라 (a)-2 경사 벽 추가 여부 결정.
3. **관측 잡음 주입 실험**: ETH가 보인 "잡음이 휴리스틱을 무너뜨림"[31]이 우리 스케일에서도 재현되는지 — Kinect 원본 vs 인위 잡음(드롭아웃·스파이크) 비교. 재현되지 않으면 우리 방법②의 이득 근거를 "잡음 강건성"이 아니라 "재형성 예측"에만 두어야 한다.
4. **바닥 근접 구간 분리 계측**: 산업은 마지막 15%가 정격 20%[13]. 우리 "완주" 시간의 어느 비율이 바닥 근접 구간에서 나오는지 보고, 지표 정의(총 시간 vs 자동 구간 시간)를 정한다.

---

## 6. 인용 함정 준수 확인 (조사 지시 ③)

| 함정 | 본 보고서 처리 |
|---|---|
| CoDeGa(RSS 2023) = 실로봇 5,100 스쿱, sim2real 아님 | 본문에서 sim2real로 쓰지 않음. 축소 상자 계보(0.9×0.6×0.2 m)로만 언급. 정본은 auto-memory `research_scooping_loader_priors_d450_reverdict.md`(출처 40건에 미포함, 프로젝트 내 기검증) |
| Aoshima 2025 이득 = 시간 5%·에너지 6.7%, 질량 아님 | 본 보고서에서 수치 인용 없음 |
| L-GBND(arXiv 2503.23270) = 시뮬 사전학습 + 실물 미세조정, "상호작용 후 지형" 예측 | §8 [32]에 그대로 기재(SAPIEN 시뮬 ≈1,000궤적 사전학습 → 실물 100궤적 미세조정, 상자 0.9×0.6×0.2 m, 자갈 0.8–1.0 cm·모래, 안식각 25–40°). 본문 확인 |
| IMSBC 안식각 = 비응집 전용, 철광석 펠릿 N/A | §2.1에서 원문 조항·스케줄로 확인[33][34] |

---

## 7. 결론 한 단락

실제 그랩 하역은 **"벽까지 가득, 평평하게 고른 표면"에서 시작해 "최고점부터 층·열 단위로" 내려가며, 벽 근처에 둑과 사면(≤70°)이 생기고, 마지막 15%는 불도저와 사람이 정리**한다. 자동화 제품은 3D 레이저 스캐너로 **매 사이클 표면을 다시 재고**, 공개된 결정 규칙은 "최고점 + 재료별 모드"까지다. 이 휴리스틱이 최적이 아니라는 것은 2025년 실기 실험이 수치로 보였다. 우리 축소 실물은 **자유유동 펠릿 영역의 "해치 사각형 안"**을 재현하는 장치이므로, (a) 초기 표면을 평평·가득으로, 실패 정의와 베이스라인을 산업 규칙으로 바꾸고, (b) 관측 주기·고정 배출·지표는 유지하며, (c) 펠릿 안식각·벽 효과·관측 잡음 실험으로 나머지 설계를 확정하면 된다.

---

## 8. 출처 표 (40건)

| # | 출처 | URL / DOI | 연도 | 확인 방식 | 핵심 수치·문장 | 우리 질문과의 관계 |
|---|---|---|---|---|---|---|
| 1 | iSAM AG, "Automation of Grab Ship Unloaders (GSU) for Bulk Materials" 적용 보고서(Hansaport) | https://www.isam-ag.de/wp-content/uploads/2025/12/02_EN_Applik_Report_Automation-of-GSUs.pdf | 2011/2025판 | 본문 | 3D 스캐너·RTK-GPS 2·TOF 4; 착지 0.5 m; 석탄 100 m 감지; **펠릿=성능지향, 석탄·광석=구석부터**; 붕괴 벽 매몰 탈출; 1명이 4기 | I1 센서·논리, I3 순서, 사고 복구 |
| 2 | iSAM AG 웹 "Autonomous grab ship unloaders" · Hansaport "Automation" | https://www.isam-ag.com/services/autonomous-ship-loaders-and-advanced-collision-protection/autonomous-grab-ship-unloaders/ · https://www.hansaport.de/en/automation/ | 2023–2025 | 본문 | 함부르크 4·로테르담 4·말레이시아 3, 바레인·캐나다 준비; "세계 최초"(운영사 주장) | I1 보급 범위, §1.6 |
| 3 | Dry Cargo International, "iSAM AG helps the Port of Hamburg to become 'operatorless'" | https://www.drycargomag.com/isam-ag-helps-the-port-of-hamburg-to-become-operatorless | 2010 | 2차(업계지) | 35 t 하역기 4기, 2010-01-01 운전, INS(레이저링 자이로) 그랩 장착, **최대 25 t/그랩, 0.5 m**; 초기 과제 "가파른 재료" | I1, I3 한 입 |
| 4 | HHLA 매거진, "Specialists for difficult cases" | https://hhla.de/en/magazine/specialists-for-difficult-cases | ~2022 | 2차(운영사 매거진) | 그랩 10 m³·25 t, 292 m선 126,000 t ≈ 2일, 그랩을 세로로 내려 옆으로 던짐, 광석 18·석탄 12종 흐름 특성별 자동화 | I3 사이클·순서, I2 코밍 밑 |
| 5 | Dry Cargo International, "Coal handling with iSAM's advanced terminal automation solutions" | https://www.drycargomag.com/coal-handling-with-isams-advanced-terminal-automation-solutions | n.d. | 2차 | EMO BR3/BR4 90 t 그랩(순 60 t); GSU 7기·SL 6기·S/R 40기+ | I1, I3 한 입 |
| 6 | ABB, "Grab Ship Unloader Automation" | https://new.abb.com/ports/solutions-for-marine-terminals/our-offerings/automation-for-bulk-handling/grab-ship-unloader-automation | n.d. | 본문(제품 페이지) | 200기+; 안티스웨이·경로 최적화·"해치·원료 스캐닝으로 지능형 자동 하역" | I1 |
| 7 | Konecranes, "AGD Grab Unloader" 브로슈어 | https://www.konecranes.com/sites/default/files/download/konecranes_brochure_agd_grab_unloader_2014_en_vfinal_0.pdf | 2014 | 본문 | Koper 32 t·1,500 t/h·40 m; 자동 모드 표준; 30 ms 기록 | I1, I3 처리량 |
| 8 | Konecranes, "Smart Crane Features" (Gottwald MHC) | https://www.konecranes.com/port-equipment-services/mobile-harbor-cranes/smart-crane-features | n.d. | 본문 | 자기학습 그랩 충진 점검, 해치별 카운터 | I1 충진 |
| 9 | Konecranes, WTE/바이오매스 크레인 자동화 · Hamburger Hungária 사례 | https://www.konecranes.com/industries/waste-to-energy-and-biomass/waste-to-energy-and-biomass-crane-automation · https://www.konecranes.com/discover/waste-cranes-help-mill-power-plant-meet-the-demands-of-paper-production | 2023 | 본문 | 완전 무인, 레이저 스캐너 벙커 높이 3D, 8 m³ 그랩 | I1 옥내 그랩 크레인 |
| 10 | Liebherr, SmartGrip 브로슈어 · 웹 | https://www-assets.liebherr.com/media/bu-media/lhbu-mcc/downloads-and-brochures/brochures/liebherr-lhm-mobile-harboure-crane-bulk-handling-smart-grip-brochure.pdf | n.d. | 본문 | 현장 평균 충진 70%, 최대 +30%, 5회 만에 학습 | I3 충진·실패(과적재) |
| 11 | Bruks Siwertell, "Anti-collision systems enable semi-automatic ship unloading" | https://bruks-siwertell.com/anti-collision-systems-enable-semi-automatic-ship-unloading | n.d. | 본문 | 열·층 파라미터 왕복, 표면 평평 유지로 "cargo avalanches" 감소 | I1, I2 표면 |
| 12 | Nemag, "How faster grab cycles improve bulk terminal productivity" | https://www.nemag.com/blogs/how-faster-grab-cycles-improve-bulk-terminal-productivity | 2025–2026 | 2차(제조사 블로그) | 갠트리 하역기 ≈60사이클/h; 트리밍에서 시간 최다 손실 | I3 사이클 |
| 13 | bulk-online 포럼, "Efficiency of Grab-type Ship Unloader" | https://www.bulk-online.com/en/forum/loading-unloading/efficiency-grab-type-ship-unloader | 2013 | 2차(포럼) | JIS 효율 50–60%; 60/25/15% 구간; "standing walls" | I3 효율, I2 벽 |
| 14 | 中国航海学会 단체표준 T/CIN 012—2023 自动化桥式抓斗卸船机技术要求 | http://www.cinnet.cn/zh-hans/system/files/files/12.zi_dong_hua_qiao_shi_zhua_dou_xie_chuan_ji_ji_zhu_yao_qiu_.pdf | 2023 | 본문 | ±300 mm, ≤60 s; 최고·최저·현재 열 최고 취점, ≤40 s, ±200 mm, **사이클마다 재스캔**; 행렬 모델·분층; 안티스웨이 ±200 mm | I1 스캔 주기·논리 |
| 15 | 中国航海学会, 自动化桥式抓斗卸船机安全作业操作规程 | https://www.cinnet.cn/zh-hans/system/files/files/61.zi_dong_hua_qiao_shi_zhua_dou_xie_chuan_ji_an_quan_zuo_ye_cao_zuo_gui_cheng_.pdf | 2025 | 본문 | 클린업 제외 자동; 30분 정지 후 재스캔; 벽 50 cm 이내 원격 | I1, 관측 주기 |
| 16 | 郑州豫森, "卸船机的详细使用步骤" (항만 작업 절차) | http://www.hnsenmiao.com/article/63.html | n.d. | 2차(업체 게시) | 선수→선미, 평균 잡기, 2그랩 깊이→1그랩 폭 이동, **사면 ≤70°**, 바닥 모드 | I3 순서, I2 사면 |
| 17 | 门机操作规程汇编 (항만 문형 크레인 작업규정) | http://www.35331.cn/lhd_4xsuj2r4jq670et7c26i4qfr01784a016ik_1.html | n.d. | 2차 | "抓高留高", "回"자형 순서, 흘림 방지(정지 후 떨구기) | I3 순서·실패 |
| 18 | 卸船机安全技术措施 (制度大全) | http://www.qiquha.com/html/201702/225544.html | 2017 | 2차 | 그랩 매몰 시 보고·무리 권상 금지; 과부하 시 열어 덜기; 줄 과다 이완 금지 | I3 실패 |
| 19 | 전자신문, "'45톤 쇳덩이 0.1도까지 조정'… 포스코DX 피지컬 AI" | https://www.etnews.com/20260624000367 | 2026-06-24 | 2차 | GTSU 무인 시운전(7월), 18–20만 t급, 그랩 20–25 t+원료 20 t, 안티스웨이 0.6–0.7°, Isaac Sim | I1 국내 |
| 20 | 산업일보(kidd), "포스코DX, 비전 AI로 철강 원료 항만 하역 '무인화'" | https://kidd.co.kr/news/245144 | 2026-03-04 | 2차 | 카메라+LiDAR, RL로 효율 최대 지점, **한 줄씩**, 1명:4기, 효율 80%, 1/35 모형 | I1 논리, I4 축소 계보 |
| 21 | Seoul Economic Daily, "POSCO DX Automates Giant Cranes…" · 시사저널e 2026-07-29 | https://en.sedaily.com/technology/2026/07/29/posco-dx-automates-giant-cranes-to-build-smart-steel-mills · https://www.sisajournal-e.com/news/articleView.html?idxno=422750 | 2026-07-29 | 2차 | 그랩 작업 80% 무인 목표, 시험 운전 중 | I1 |
| 22 | 특허 CN116101901A, 无人化抓斗卸船机控制系统及控制方法 | https://patents.google.com/patent/CN116101901A/en | 2023 | 본문(청구항·상세) | 잡기 방향 하향 단계, 해측/육측 단계, **그랩 매몰 방지 전략**, 고랑 단계, 초기 단위 선택→중앙 구역 | I1 논리 |
| 23 | Ngo, Lee, Kim, Dinh, Park, "Design of an AI Model for a Fully Automatic Grab-Type Ship Unloader System", JMSE 12(2):326 | https://doi.org/10.3390/jmse12020326 | 2024 | 본문 | 146×146 cm, VLP-16, YOLOv3 96%, 작업점 = LiDAR 최근접점, 5–10% | I1 논리, I4 축소 계보 |
| 24 | "Safety optimization of grab unloaders based on machine vision and 3D coordinate system reconstruction", Sci. Rep. | https://www.nature.com/articles/s41598-025-19944-1 | 2025 | 본문 | D455+Mid-70+MTi-30, RMSE 2.8/4.3/6.1 cm, 14–20 FPS, 91.2%, 1.4 s, 3/68 | I1 센서·정밀도 |
| 25 | Xiao et al., "Design and Application of Whole-Process Intelligent Unloading System for Bulk Carriers", IEEE T-TE 12(3):5526–5539 | https://doi.org/10.1109/tte.2026.3678401 | 2026 | 초록 | 3D 특징 분포 슬라이딩 윈도로 파지 영역 예측; 곡물·광석 항만 실증 | I1 논리(예측형) |
| 26 | Qing L., "Novel full-automatic grab ship unloader system", J. PLA Univ. Sci. Tech. · Yang S., "Application of 3D Laser Scanning System in Automatic Grabbing of Ship Unloader", J. Wuhan Polytechnic | Consensus 색인(원문 미확보) | 2010 · 2011 | 초록 | 뤄징 광석부두 "세계 최초" 주장; 3D 스캔으로 취점·깊이 결정 | §1.6, I1 |
| 27 | Hu J., "Continuous Automatic Positioning for Reclaiming Point of Grab Ship Unloader Based on 3D Perception", J. Wuhan Univ. Tech. · Shen Y. et al., "Point Cloud Surface Registration for Cargo Hold Material Matching…", IEEE Sensors J. | Consensus 색인 | 2013 · 2025 | 초록 | 3D 윤곽 기반 연속 취점; 단일 스캔은 진동·그랩 가림으로 결손 → 다중 프레임 정합 | I1 논리, 관측 가림 |
| 28 | Banno et al.(IHI), "Development of an automated system for continuous ship unloader operation in coal handling", Trans. JSME 92(958) | https://www.jstage.jst.go.jp/article/transjsme/92/958/92_25-00233/_article/-char/en | 2026 | 초록 | 다중 LiDAR 복셀, 휴리스틱 경로, 97%, 20분 | I1(CSU 대조) |
| 29 | Mohajeri, de Kluijver, Helmons, van Rhee, Schott, "A validated co-simulation of grab and moist iron ore cargo…", Adv. Powder Technol. | https://doi.org/10.1016/j.apt.2021.02.017 | 2021 | 본문 | 7 m 깊이 침투 0.39±0.05 m; 사전압밀 0→300 kPa에 0.67→0.37 m; 응집 자국 유지 vs 자유유동 즉시 흘러내림; wet bottom z/z_max≥0.8; 5단계 | I2 표면 변화·재료 차이 |
| 30 | Mohajeri, van den Bergh, Jovanova, Schott, "Systematic design optimization of grabs considering bulk cargo variability", Adv. Powder Technol. | https://doi.org/10.1016/j.apt.2021.03.027 | 2021 | 본문 | **15만 DWT = 5,000–7,500 사이클** | I3 총 횟수 |
| 31 | Spinelli, Zhai, …, Hutter (ETH) + Liebherr, "Large Scale Robotic Material Handling: Learning, Planning, and Control", arXiv 2508.09003v2 | https://arxiv.org/abs/2508.09003 | 2025/2026 | 본문(PDF 표 1·3·6) | RL 62.7%/16.0회 vs 최고점 14.6%/60.0(잡음)·57.7%/17.8(무잡음); 실기 100%/67.3% vs 76%/47.9%; 사람 58–76%, 사이클 17–34 s; 셀룰러 오토마타 흘러내림; 전문가는 사면 공략 | §1.5 베이스라인, I3 숙련자 |
| 32 | Liu, Li, Hauser, "Localized Graph-Based Neural Dynamics Models for Terrain Manipulation (L-GBND)", arXiv 2503.23270v3 | https://arxiv.org/abs/2503.23270 | 2026 | 본문 | 시뮬(SAPIEN ≈1,000궤적) 사전학습 + 실물 100궤적 미세조정; 0.9×0.6×0.2 m 상자; 자갈 0.8–1.0 cm·모래; 안식각 25–40°; 2D heightmap CNN 대비 오차 −37%; 체적최대 휴리스틱은 초기엔 빠르나 침하를 예측 못함 | I4 축소 계보, 함정 ③ |
| 33 | IMSBC Code Section 5 "Trimming procedures" (ClassNK T845e 첨부 / MPA Singapore 결의문 사본) | https://www.classnk.or.jp/hp/pdf/tech_info/tech_img/T845e.pdf · https://www.mpa.gov.sg/api/media/6bb9fd9c-df7b-46bb-9d13-6c522c056f37/sc08-14c.pdf | 현행 | 본문 | 5.1.1–5.1.3, 5.3.2, 5.4.3–5.4.5 (30°/35°, B/10, 1.5/2 m) | I2 초기 상태 |
| 34 | IMSBC 개별 스케줄: IRON ORE PELLETS(MSC.575(110) 개정), COAL, IRON ORE (imorules 사본) | https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC.575(110).pdf · https://www.imorules.com/GUID-BFAC9206-D183-4924-B8CA-47E9D803D717.html · https://imorules.com/GUID-577D7807-7645-43A7-9CE4-20BE2FDABD91.html | 2025 | 본문 | 펠릿 10–40 mm·N/A·1,800–2,400 kg/m³·Group C; 석탄 ≤50 mm·N/A·654–1,266·B(and A)·"수직 균열"; 철광석 ≤250 mm·N/A | 함정 ④ |
| 35 | IMO Resolution A.434(XI), Code of Safe Practice for Solid Bulk Cargoes (1979) §3.2.2 | https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/AssemblyDocuments/A.434(11).pdf | 1979 | 본문(구 규정) | 해치 사각형 안 평평, 나머지는 옆·끝벽으로 일정 경사 | I2 초기 기하 |
| 36 | IMO MSC/Circ.1160 BLU Manual, Annex 2 "Avoidance of Damage During Cargo Handling" (36-b: bulkcarrierguide "Terminal duties while unloading") | https://www.imorules.com/GUID-6E148B64-B1D5-43F0-A156-447F9BB943AF.html · https://bulkcarrierguide.com/terminal-duties-unloading-cargo.html | 2005 | 본문 (36-b 2차) | 4단계; "가파른 둑 금지"; "**항상 최고점**"; 좌우 균형; 자동 모드 "체계적·고른 패턴" | I2 진화, I3 순서 |
| 37 | TSB Canada M96M0002, CSL Atlas | https://www.tsb.gc.ca/eng/rapports-reports/marine/1996/m96m0002/m96m0002.html | 1996 | 본문 | 석탄 37°·수분 11.3%, 더미 4.6–5.5 m, 면 파단·매몰 1 m, 사망 | I2 붕괴 |
| 38 | Professional Mariner, "Shifting cargo kills crewman during unloading operation" (MV Pioneer, 석고) | https://professionalmariner.com/shifting-cargo-kills-crewman-during-unloading-operation/ | 2007 | 2차 | 굳은 석고 벽 20×35 ft 붕괴, 85% 비운 시점 | I2 붕괴 |
| 39 | JTSB(일본 운수안전위원회) 해양사고보고서, ISHIZUCHI 니이하마항 | https://jtsb.mlit.go.jp/eng-mar_report/2020/2019tk0004e.pdf | 2020 | 본문 | 한 선창 15,000 t; 불도저가 옆 석탄을 가운데로 모음; 6 m 스크레이퍼; 그랩·불도저 접촉 우려 | I2 클린업, I3 |
| 40 | 스틸데일리, "현대제철의 밀폐형 원료처리시스템은?" | https://www.steeldaily.co.kr/news/articleView.html?idxno=49838 | n.d. | 2차 | 당진 원료부두 = 밀폐형 CSU + 밀폐 컨베이어 | I1 국내(그랩 아님) |

---

## 9. 검색 로그 (검색어 × 소스 × 건수)

"없다/최초/유일" 단정은 본문에 쓰지 않았다. 아래 로그는 **한정적 '확인 못함'** 세 항목(① iSAM·ABB 알고리즘 수준 공개, ② Liebherr MHC의 선창 스캔·취점 기능, ③ 현대제철 그랩 하역기 자동화)에 대한 탐색 범위를 남기기 위한 것이다. 셋 다 "읽은 문서 범위에서"라는 한정어를 붙였다.

| # | 도구 | 검색어 | 결과 수 | 채택 |
|---|---|---|---|---|
| 1 | exa | Konecranes automated grab ship unloader autonomous unloading laser scanner hold cargo profile | 8 | [7][8][1] |
| 2 | exa | iSAM AG Hansaport fully automated grab ship unloader 3D laser scanning hold coal iron ore | 8 | [1][2][3][4][5] |
| 3 | exa | Liebherr LiSIM or ABB crane automation bulk grab unloader cargo hold scanning system remote operation | 8 | [6] (Liebherr는 시뮬레이터 페이지만) |
| 4 | brave | grab ship unloader automation 3D laser scanner hold surface profile grab position determination patent | 15 | [22][23][24] |
| 5 | brave | bulk carrier hold unloading grab operator practice cargo collapse trimming digging sequence | 15 | [36-b], IACS 안내 |
| 6 | exa | IMSBC Code section 5 trimming procedures "trimmed reasonably level" … | 8 | [33][35] |
| 7 | exa | TU Delft grab unloading DEM co-simulation iron ore cargo bed penetration Lommen Mohajeri Schott | 8 | [29][30] |
| 8 | exa | accident investigation report worker buried engulfed by collapsing bulk cargo in ship hold during discharge | 8 | [37][38][39] |
| 9 | exa | grab crane operator unloading technique bulk carrier hold: free digging… cycle time seconds | 8 | [36][12][13][30] |
| 10 | exa | patent grab ship unloader automatic material surface scanning select grab digging position highest point | 8 | [22] + CN121894458A 등 5건(미채택) |
| 11 | brave | grab ship unloader cycle time seconds per cycle … "free digging" "clean-up" | 1 | [13] |
| 12 | brave | automated waste bunker grab crane 3D pile scanning Konecranes … | 10 | [9] |
| 13 | brave | 포스코 현대제철 무인 하역기 자동화 언로더 3D 스캐너 원료부두 grab unloader 자동화 | 10 | [19][21] |
| 14 | exa | 抓斗卸船机 司机 操作 取料 顺序 舱内 先中间后两边 分层 抓取 料面 塌方 经验 | 8 | [14][15][16][17][18] |
| 15 | exa | Xiao 2026 IEEE T-TE whole-process intelligent unloading sliding window 3-D feature distribution | 5 | [25] |
| 16 | exa | 현대제철 당진 원료부두 하역기 무인 자동화 라이다 3D 스캔 그랩 언로더 원격 | 6 | [20] (현대제철 관련 0) |
| 17 | exa | Liebherr MHC bulk grab automation semi-automatic hold scanning; Bruks Siwertell automation; ABB brochure | 8 | [10][11] |
| 18 | exa | grab buried in cargo hold collapsing material wall coal iron ore unloader grab stuck | 8 | [29][38][39], MHI 1975 특허(미채택) |
| 19 | exa | IMSBC schedule IRON ORE PELLETS "angle of repose" "not applicable"; COAL schedule | 6 | [34] |
| 20 | exa | bulk carrier grab discharge: percentage of cargo the grab can reach … clean-up time share | 8 | [36], DNV GRAB(미채택), HandyBulk(미채택) |
| 21 | brave | 현대제철 당진제철소 원료 하역기 자동화 무인 grab unloader 3D 스캐너 | 10 | [40] (CSU 방식 확인) |
| 22 | exa | Hyundai Steel Dangjin raw material port ship unloader automation remote unmanned LiDAR grab crane | 6 | 현대제철 0, [21] |
| 23 | Consensus | grab ship unloader automatic digging point selection cargo hold material surface | 10 | [23][25][26][27][30] |
| 24 | alphaXiv discover | grab ship unloader / bulk cargo hold / digging point / unloading strategy | 9 | [31], 2405.16774·2509.13890(미채택) |
| 25 | alphaXiv PDF | 2508.09003 표 1·3·6, 흘러내림 모델, 전문가 행동 | 본문 6쪽 | [31] 수치 검증 |
| 26 | alphaXiv | 2503.23270 전문 | 본문 | [32] |
| 27 | WebFetch | Isbester, Bulk Carrier Practice(armcol.org PDF) | 접속 거부(ECONNREFUSED) | 미채택 |
| 28 | exa fetch | iSAM PDF·MDPI·Nature·imorules·bulkcarrierguide·Konecranes·jstage·bulk-online·etnews·sisajournal·T/CIN PDF·HHLA·DryCargo | 14 페이지 | 본문 확인용 |

소스 종류: 제조사·운영사 원문 6(iSAM, ABB, Konecranes, Liebherr, Siwertell, Nemag) · 규정 원문 4(IMSBC, A.434, BLU 매뉴얼, T/CIN + 안전규정) · 학술 10(MDPI, Nature Sci. Rep., IEEE T-TE, IEEE Sensors, JSME, APT×2, arXiv×2, 중국 학보×3) · 사고조사 3(TSB, JTSB, 업계지) · 특허 1 · 뉴스 4 · 포럼/블로그 3. 검색 도구: exa 16회, brave 6회(429 제한 1회·빈 결과 1회 포함), Consensus 1회, alphaXiv 3회, WebFetch 1회(실패). 소요 약 55분.

---

## 10. 이 조사가 우리 절차를 바꾸는가 — 최종 판정

| 구분 | 항목 | 근거(§) |
|---|---|---|
| **(a) 바꾼다** | 초기 표면 = 벽까지 가득·평평(더미는 2차) · 실패 정의 4종 채택(회당 질량 계측 선행) · 베이스라인 = 최고점 휴리스틱 + 층/열 진행 · 재료 영역 한정어("자유유동 펠릿") · 촬영 시점 = 팔 퇴장 후 | §1 Q1, §3.3, §1.5, §2.3, [27] |
| **(b) 유지** | heightmap + 위 고정 깊이 카메라 · 사이클당 1회 관측 · 배출 위치 고정 · 지표 총 시간/횟수/실패 · 축소 실물 접근 · 동작 고정·학습은 "어디를"만 | §4, §5(b), [14][1][23] |
| **(c) 추가 실험 필요** | 펠릿 안식각·흘러내림 실측(IMSBC 등급·70° 규칙 적용 여부) · 벽 효과(상자 폭/그랩 폭 2단계) → 경사 벽 추가 여부 · 관측 잡음 주입(휴리스틱 붕괴 재현 여부) · 바닥 근접 구간 분리 계측(마지막 15% 저효율) | §5(c), [31][13][33] |
