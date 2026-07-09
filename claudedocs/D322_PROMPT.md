# D322 실행 프롬프트 — grasp 트랙 개시 (G0a) + 플랜·규칙 영구 설치

> 사용법 (둘 중 하나):
> ① 아래 코드블록 내용을 Claude Code에 그대로 붙여넣기.
> ② 이 파일을 레포에 넣고 (권장 위치: `claudedocs/D322_PROMPT.md`) Claude Code에
>   "claudedocs/D322_PROMPT.md 읽고 그대로 실행해" 라고 지시.
> 출처: 채팅 인수인계 문서 v5 (2026-07-08) 11절과 동일 전문.

```
CLAUDE.md Current-State Protocol 준수. 이번 세션 = D322: (0) 플랜 정본·규칙 영구
설치, (1) grasp 트랙 Active Case G0a 구현.
failable 실험 = Step 2 (실패 조건: 정렬 판정 4조건 중 하나라도 10회 중 3회 이상 미달).

═══ Step 0. 플랜 정본 + 재발 방지 규칙 영구 설치 (최우선) ═══
0-1. claudedocs/direction_20260708_grasp_pivot.md 생성 — grasp 트랙 플랜의
  레포 내 정본. 아래 내용을 그대로 수록:
  [교수님 지시 (2026-07-08 랩미팅)]
  ① 변수 최소화 — 완결된 case 하나를 먼저 성공시키고, 성공 시 변수를 1개씩
     추가한다 (기존 마찰 randomization 선행은 순서 역전이었음).
  ② VLA는 추후. 지금 보고 싶은 것은 RL이 되는 것.
  ③ 마찰 보정/randomization은 최후 단계.
  ④ grasp 피벗: 비대칭 그리퍼(고정 조+가동 조)를 물체 옆에 정확히 정렬하는
     것이 1차 관문.
  ⑤ 핵심 목표 능력 = 위치가 랜덤해도 잡기 (고정 위치 태스크의 랜덤 위치
     일반화 — 커피머신 예시).
  ⑥ 원통 등 잘 잡히는 형태부터, 형태 다양화는 나중 ("다양한 물체가 되는가"가
     장기 질문).
  ⑦ 로봇은 나중에 바뀔 수 있다 → 최종 산출물은 정책이 아니라 파이프라인과
     데이터 스펙.
  ⑧ 색상은 당분간 무시. 재질/마찰 변경 금지.
  [G-사다리] G0a 정렬(신규 변수: grasp 기하) → G0b 원통 D34×H90 파지+들어올림
  (= 첫 완결 case, 프로포절 트리거) → G1a 위치 grid 민감도 곡선 → G1b
  standalone PPO 스크래치 0%→X% 커브(+zero-action 대조, 학습 전후 영상 = "RL로
  되는 것" 증명) → G2 형태 다양화 → G3 grid place → 실기 전이(캘리퍼 보정 선행).
  [그리퍼 실측] 실기 최대 88.3°(URDF 1.571rad), 접촉 반경 ~43mm(0.75mm/°),
  실용 개구 ~40~45mm, 오프셋 = 물체 중심을 TCP에서 가동 조 방향 +x로 (D/2−8mm),
  cmd 0~5° 금지(서보 stall), 30mm 앵커 stall 37.88°. sim 계약(G0b~): joint 하한
  0.09rad + effort limit로 stall 재현.
  [tap 트랙] 종결. D321(1,920ep/96.0%)이 최종 산출물, 자산 동결. 인수: DiffIK
  접근·D256 reset·검증기+물리성 게이트·컨베이어·평가 규약·script 0~999 대조군.
0-2. CLAUDE.md에 durable 섹션 "Variable Ladder Protocol (D322~)":
  (a) case당 신규 변수 1~2개만, 세션 문서 상단에 "이번 case의 신규 변수: [...]"
      명시 의무.
  (b) 미래 대비 아이디어는 구현 금지 — claudedocs/BACKLOG.md에 append만 하고
      임계 경로 복귀.
  (c) START_HERE "Active Case" 섹션이 단일 진실. 밖은 전부 Non-goal, 변경은
      사용자 승인으로만.
  (d) 폴더 전방 전용: 기존 파일/폴더 이동·개명 금지(경로 참조 보존). 신규
      grasp 산출물은 claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/ 에만
      생성하고, Active Case 섹션에 경로 병기.
0-3. claudedocs/BACKLOG.md 생성, 시드 append: TCP/EEF 이중 기록 / 캘리퍼 mm
  보정(실기 전이 선행) / 렌더 해상도(저장 448², 입력 224²) / 방향 다양화·
  goal-conditioned(design_d321 연결) / upper friction bin(RL target 예약) /
  overshoot 167ep HER 재라벨 / 데이터 스펙 v1 / VLA 학습(보류).
0-4. START_HERE에 "Active Case: G0a" 섹션 신설 (성공 기준 4조건 + 산출물 경로
  grasp_track/g0a_d322/ 병기).

═══ Step 1. G0a 구현 ═══
이번 case의 신규 변수: grasp pose 기하 1개 (base yaw 정렬 + 비대칭 오프셋).
불변: 기존 10cm 큐브(개구 ~45mm < 100mm — 파지 불가, 정렬만 검증), 고정 위치
1곳, 마찰 1.5/1.2, state 기반, 렌더 없음, candidate6 DiffIK 접근 재사용.
구현:
  (a) grasp pose 계산기: base yaw = atan2(cube_y, cube_x), TCP 목표 = 물체
      측면에 고정 조 파지면이 닿는 pose, 오프셋 = 물체 중심이 TCP 기준 가동 조
      방향 +x로 (D/2 − 8mm), D=0.10m → 42mm.
  (b) 시퀀스: pre-approach(측면 4cm 밖) → DiffIK 직선 접근 → 정렬 pose 도달
      → 정지. 그리퍼 열림 유지(닫힘/파지 없음).
  (c) 그리퍼 sim 계약(G0b 대비): joint 하한 0.09rad + effort limit — config
      주석으로만 준비, 활성화 금지.
산출물 위치: claudedocs/runtime_logs/grasp_track/g0a_d322/.

═══ Step 2. 판정 (사전 등록) ═══
4조건: ① TCP pose 오차 ≤5mm/3° ② 고정 조 파지면-큐브 면 간극 ≤3mm & 무관통
③ 큐브 변위 <5mm ④ 동일 조건 10회 전부 충족.
보고: 10회 각각의 (pose 오차, 간극, 변위) 표 + 실패 시 어느 조건·어느 시행.
verdict는 runtime 결과 단어로.
문서 갱신: session doc(신규 변수 명시) + START_HERE + DECISIONS(규칙 durable)
+ LEDGER.
Non-goals: 원통/신규 물체 스폰, 그리퍼 닫힘·파지·들어올림, RL, 렌더, 위치
랜덤, TCP 스키마 변경, 마찰/재질 변경, 기존 파일·폴더 이동/개명, B200/실기.
(필요해 보이면 BACKLOG에 적고 진행하지 말 것.)
```
