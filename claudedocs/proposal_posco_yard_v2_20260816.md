# 연구 프로포절 v2 — 원료야드 축소판: 재형성 더미 위 순차 pick-place 결정층 학습

작성: 2026-08-16 (63rd 조사 + D450 반영 개정판. 원안은
`session_20260816_63rd_posco_yard_pivot_domain_recon.md` §0에 영속 기록)
지위: 사용자 검토 대기 초안. 갭 문구는 D450 금지/한정 규칙을 따름.

---

## 1. 산업 배경과 시나리오 매핑

제철 원료야드는 철광석 fines·원료탄 등 벌크 원료(bulk material)를 더미
(stockpile)로 적치·불출하는 공정이다. 포스코DX는 2026년 현재 이 공정의 두
장비 — 리클레이머와 GTSU(그랩식 원료하역기, 그랩 20~25t) — 를 무인화 중이며
(하역 80% 무인화 목표, 2026-07 무인 시험운전), 개발·검증에 NVIDIA Isaac Sim
기반 Sim2Real을 사용한다. 공개된 AI의 역할은 인지(높이맵 계측)와 실행·제어
계층이고, **"어디서 집고 어디에 놓을지"를 정하는 결정 계층은 업계 전반에서
규칙 라이브러리로 남아 있다** (ABB 야드 자동화 = chevron/windrow 등 적치
규칙; BHP South Flank 자율 스태커-리클레이머 = GPS+기하 규칙).

본 연구의 축소판 시나리오는 원료 체인의 두 결정을 하나의 탁상 사이클로
압축한다:

- **pick측 = GTSU 그랩 사이클**: 선창/더미를 높이맵으로 계측한 뒤 이산적
  pick-and-dump를 반복한다 (RIST 공개 영상 t=00:53에 LiDAR/카메라 높이맵
  계측 실물 장면). "어디를 집을 것인가"가 매 사이클의 결정이다.
- **place측 = 스태커 적치**: 투하 위치 선택에 따라 더미가 안식각으로
  자기-적치된다. "어디에 놓을 것인가"가 실재하는 층이다 (GTSU 자체의 투하는
  고정 호퍼라 놓기 선택이 없음 — 그래서 GTSU 단독이 아니라 pick+place 결합
  축소판이다).

주의: 본 축소판의 "높이 우선" 목적함수는 실조업 목적(배합비 준수·잔량
최소화)이 아니라 연구용 **대리 휴리스틱**이다. 이 단순화는 본문에 명기한다.

## 2. 문제 정의 (한 줄)

**자기-재형성(self-reshaping) 더미 위의 순차 pick-place 순서 결정 문제를
정의하고, 결정층(집을 위치 + 놓을 위치) 하나만 강화학습으로 학습하여, 완주
(전량 이송) 총 동작수 기준으로 규칙·계획·회당-최대 정책과 비교한다.**

조작 기술(파지·궤적)은 기여가 아니다 — 팔 동작은 스크립트로 고정하고,
학습은 결정층 한 층에 국한한다. 이 국한이 기각 방어의 2차 축이다 (1차 축 =
문제 자체가 선행 3편의 원문 future work로 문서화되어 있음, §4).

## 3. 연구 질문

- **RQ1 (데모)**: depth 높이맵을 보고 가장 높은 곳에서 집어 지정 위치에
  놓는 greedy 규칙 한 줄로 완주가 되는가 — 성립 데모이자 베이스라인.
- **RQ2 (본논문)**: "가장 높은 곳부터"가 완주까지 **최소 총 동작수**인가?
  같은 초기 더미에서 greedy vs 학습된 순서를 비교한다. 더미는 매 pick마다
  재형성되므로(무너짐·흘러내림) greedy의 최적성이 보장되지 않는다 —
  무너짐은 연구의 방해 요소가 아니라 **문제가 성립하는 이유**다.
- **RQ3 (검증)**: sim 결론이 실물 RoArm-M3-Pro + 사람 교란(더미 재배치·물체
  추가/제거)에서 유지되는가 (Sim2Real).

## 4. 관련 연구와 갭 (2026-08-16 조사 기준, D450)

베이스 3편과 그 원문 future work:

- **ETH Terenzi & Hutter** (arXiv 2308.11478, IEEE T-FR): 자율 굴착 계획 —
  greedy 스쿱 시퀀스, 결론부에서 학습 기반 계획을 future work로 지목.
- **Baidu Lu/Zhu/Zhang** (arXiv 2201.11292, RA-L 2022): Franka+미니버킷으로
  비정형 강체(나무블록 282개) 굴착 RL + sim2real. 단 (a) 회당 부피에서 RL이
  휴리스틱을 이기지 못했고 (b) 완주 개념이 없으며 (c) 놓기는 스크립트,
  (d) 물체 선택이 없다.
- **CMU CraterGrader** (arXiv 2311.01697): EMD 최적수송 기반 지형 계획.
  future work로 **이산 모달리티(rock picking)**를 명시 — 본 연구가 정확히
  그 이산 집기 축이다.

단, 다음 선행이 존재하므로 아래 문구는 사용하지 않는다:

- **Spinelli et al. 2025** (arXiv 2508.09003, ETH Hutter 그룹, IEEE T-FR,
  DOI 10.1109/TFR.2026.3662619): 40t 머티리얼 핸들러 실기에서 PPO로
  높이맵→파지점(attack point)을 순차 선택, 보상 = 버킷 충전 최대화(= 총
  사이클 수 최소화). **pick측 결정 학습은 이미 존재한다.** 단 place는
  사용자 지정 고정 좌표(미학습)이고, 대상은 버킷 스쿱(연속 매체)이지 이산
  물체 집기가 아니며, 도메인은 핸들러이지 야드가 아니다.
- **Schenck et al. 2017** (CoRL): 높이맵 상태에서 scoop 위치와 dump 위치를
  **둘 다** 선택 (학습 예측모델+MPC, RL 아님, 목적 = 목표 형상 재현).
  ⇒ "놓기 선택 학습은 최초" 단독 주장 불가.
- **Lu & Myo** (Adelaide, 2010-2018): 실제 원료야드 BWR 도메인에서 "어느
  voxel을 퍼낼지 + 이동 최소화"라는 동일 결정 문제를 정수계획법(비학습)으로
  해결 — 도메인 최근접 선행.
- **Three Springs 2019**: 야드 리클레이머 **제어층** RL(DDPG) 상용 실증
  ⇒ "야드에 학습 적용 없음"이라는 문장은 거짓.
- 제어·형상 계층 학습 계열: Backman et al. 2021 (arXiv 2103.01283, 지하
  로더 연속 제어 RL), AGPNet (arXiv 2112.10877, 도저 정지 작업 MDP).

**갭 서술 (한정어 포함)**: 우리 조사 범위 내(2026-08-16, 28질의 × 4소스;
Semantic Scholar 미조회)에서, **(i) 매 동작마다 재형성되는 더미 위에서
(ii) 완주 총 동작수를 목적으로 (iii) 집을 위치와 놓을 위치를 모두 학습**하는
결정층 연구는 발견하지 못했다 (확신도 MEDIUM-HIGH). 어느 단일 축도 "최초"가
아니며, novelty는 이 **3-결합**(+ 부가적으로 원료야드 도메인·이산 물체 집기
설정)에 한정된다.

## 5. 구축물 (기존 자산 재사용)

- **테스트베드**: Isaac Sim (기존 `isaaclab` env + RoArm URDF/USD + 테이블
  캘리브레이션 자산 재사용). 더미 = 비정형 convex 다면체(파지 창 22~35mm,
  그리퍼 개구 40~45mm 기준) 40~60개 + 지정 적치 존. sim↔real 메쉬 동일
  원칙(3D 프린트 물체 = 동일 CAD 메쉬 = USD 충돌 기하; D446 교훈).
- **관측**: 탑다운 depth 높이맵 (Kinect ToF 축소판; 실광석 흑색 대신 무광
  밝은 색 물체 사용 — 센서 충실도를 위한 의도적 변경으로 명기).
- **판단 정책 4종 비교**: ① greedy(최고점 우선) ② 최적수송 계획(EMD,
  CraterGrader 계열) ③ RL(학습 결정층) ④ Baidu식 회당-최대.
- **2×2 분해**: {집기 선택, 놓기 선택} × {규칙, 학습} — 어느 층의 학습이
  이득을 만드는지 분리.
- **지표**: 완주 총 동작수(주지표) · 실패율(파지 실패/이탈) · 적치 평탄도.
- **금지 조항**: 놓을 자리 선택을 스크립트로 강등하지 않는다 — 강등하는
  순간 본 연구는 Baidu 2201.11292의 부분집합이 된다.

## 6. 로봇 플랫폼 (사양만 기재)

RoArm-M3-Pro (5+1축, 비대칭 평행 그리퍼). 그리퍼는 파지 신뢰성을 위해
탈착식 3D 프린트 조 슬리브(접촉면 기하 수정 + 탄성 인서트)를 장착하며,
sim(USD)과 실물에 동일 CAD 메쉬를 적용한다. 조작 기술은 기여가 아니므로
본문에는 사양만 기재한다.

## 7. 검증 계획

- Sim: 동일 초기 더미 시드에서 정책 4종 × N판 완주 비교 (판별 통계 포함).
- Real: 소규모 판수로 sim 결론의 순위 보존 여부 확인 + 사람 교란(더미 재배치,
  물체 추가/제거) 하 재평가 (RQ3).
- 파일럿: MuJoCo 예비 파일럿(물체 40개·20판)이 존재하나 본 repo 증거 체인
  밖(미검증)이므로 **본 프로포절에서는 그 수치를 인용하지 않는다**. v1
  (100판+상수 스윕)을 repo로 이관·재현한 뒤 별도 보고한다. 또한 파일럿은
  규칙 vs 규칙 비교이지 "RL이 이긴다"는 증거가 아니다 (Baidu 선례와 함께
  정직 프레임 유지).

## 8. 참고문헌 (필수 9편 — 서지 검증 2026-08-16, arXiv API)

1. Terenzi, L., Hutter, M. "Towards Autonomous Excavation Planning."
   arXiv:2308.11478, IEEE T-FR.
2. Lu, Q., Zhu, Y., Zhang, L. "Excavation Reinforcement Learning Using
   Geometric Representation." arXiv:2201.11292, RA-L 2022.
3. Lee, R., Younes, B., Pletta, A., et al. "CraterGrader: Autonomous Robotic
   Terrain Manipulation for Lunar Site Preparation and Earthmoving."
   arXiv:2311.01697.
4. Spinelli, F. A., Zhai, Y., Nan, F., et al. "Large Scale Robotic Material
   Handling: Learning, Planning, and Control." arXiv:2508.09003, IEEE T-FR,
   DOI 10.1109/TFR.2026.3662619.
5. Schenck, C., et al. "Learning Robotic Manipulation of Granular Media."
   CoRL 2017.
6. Lu, T.-F., Myo, M. T. R. 원료야드 스톡파일 리클레이밍 voxel 선택 정수계획
   연구군 (Univ. of Adelaide, 2010-2018).
7. Backman, S., Lindmark, D., Bodin, K., et al. "Continuous Control of an
   Underground Loader Using Deep Reinforcement Learning." arXiv:2103.01283.
8. Ross, C., Miron, Y., Goldfracht, Y., et al. "AGPNet — Autonomous Grading
   Policy Network." arXiv:2112.10877.
9. Three Springs (2019) — 야드 리클레이머 제어층 RL(DDPG) 상용 실증 사례.

---

*개정 이력: v1 = 사용자 구두 원안 (63rd doc §0 영속화). v2 = D450 갭 문구
3금지 반영(Spinelli/Schenck/Three Springs 반례 명시), GTSU/스태커 매핑 서두
추가, 필수 인용 9편 서지 검증 완료(Backman=2103.01283, AGPNet=2112.10877
신규 확보), 파일럿 수치 인용 제거.*
