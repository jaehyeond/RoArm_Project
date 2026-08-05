# Session 2026-08-05 (16th) — 외부 "프리미너리 테스트 v2" 인계 + lead 단독 부분 검증 (감사 워크플로우 미완, context 98% 중단)

이번 case의 신규 변수: [없음 — case 미개시. 외부 문서 v2 수령 + lead 단독 수치 검증 세션.
repo 코드 무변경, 동결 D409/D362/D400~D408 침범 0, Isaac/로봇/GPU/lerobot-train 실행 0, git commit/push 0.
읽기 전용 명령(`nvidia-smi`, `grep`, `sed`)만 실행.]

## 0. 세션 성격 / Session progress rule 정당화

- 사용자 입력 = 외부 AI 대화에서 받아온 **"프리미너리 테스트 — 연구 설계서 v2"** 전문 + "제대로 파악 후 상세
  브리핑 + 승인받을 것 브리핑" 지시.
- **실패 가능한 검증을 실행했고 실제로 문서 주장이 깨졌다**: lead가 직접 repo 파일을 열어 대조한 결과
  부록 E 3건 중 **2건이 붕괴**, §15 #11 하드웨어 1건 불일치, 동결 상수 1건 불일치를 확인했다(§2).
- **미완 항목(정직 기록)**: 5-lens 적대 감사 워크플로우 `ww413jhfc` / `wf_3f4d6079-4e7`를 발사했으나
  **context 98% 도달로 결과 수령 전에 세션을 종료**한다. **그 산출물은 본 doc에 반영되지 않았고, 인용해서도
  안 된다.** transcript:
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/eb0914aa-.../subagents/workflows/wf_3f4d6079-4e7/`
  script: `.../workflows/scripts/prelim-v2-adversarial-audit-wf_3f4d6079-4e7.js`
- 새 case 개시 0. 승인 획득 0건. repo 코드 0. 로봇 0. Isaac 0. GPU 연산 0.

## 1. 부트 시 확인한 git 상태 (14th·15th doc의 "미커밋" 해소)

- HEAD = `c1b7679` ("15th, D416"). `git status --short` **클린**.
- 즉 `START_HERE.md:114`의 "미커밋: 15th 세션분"은 **해소됨** — 사용자가 push 완료.

## 2. 외부 v2 문서 — lead 단독 검증으로 **확정 붕괴한 항목** (전부 lead가 파일 직접 판독)

### 2-1. [높음] 부록 E "SmolVLA 450M **LoRA**: 스펀지 태스크 open-loop **74 에피소드 100% 성공**" — 2겹 붕괴

(a) **"100%"의 실체는 5회 시행이고, repo가 스스로 FALSE POSITIVE로 기록해 뒀다.**
`claudedocs/BASELINE.md:13-17` = "데이터 74 episodes, **1개 위치(Base ~45°)**, 스펀지 1개 / 배포 open-loop
4-chunk **5/5 (100%) 성공** / 한계 **1개 위치, 1개 물체, 1개 동작만 가능**".
`claudedocs/EAAS_PROPOSAL.md:30` = "v3 (good quality 74ep) → 100%, **하지만 FALSE POSITIVE
(base ~45° 한 구간만)**".
→ 이 수치를 "이전 트랙 자산"으로 제시하는 것은 repo 자신의 판정을 뒤집는 인용이다.

(b) **"LoRA"는 이 태스크에 존재하지 않는다.** `BASELINE.md:80-84` 학습 명령 =
`lerobot-train --policy.pretrained_path=lerobot/smolvla_base --batch_size=64 --steps=200000`
= 전체 파인튜닝. repo에서 LoRA가 붙는 것은 **OpenVLA-OFT 7B(Track B)**뿐
(`EXPERIMENT_LEDGER.md:105` "OpenVLA-OFT 7B 30K LoRA"). 모델·학습법 오귀속.

### 2-2. [중] 부록 E "Hand-Eye Calibration: 평균 오차 **2cm**, 검출률 **85.8%**, 30 FPS" — superseded 자료

- 85.8% / 30 FPS는 실재한다: `CALIBRATION_LOG.md:204`(감지율 85.8%), `:206`(평균 FPS 30.1),
  `:263`. **단 2026-01-19 테스트**다(`:195`).
- 같은 문서 `:255-263`이 스스로 "좌표 변환 ⚠️ **제한적**(좁은 범위에서만 정확) / 캘리브레이션 ⚠️
  **재수행 필요**(범위가 너무 좁음)"로 판정하고 `:265` 이하에서 "옵션 A: 캘리브레이션 재수행(권장)"을 적었다.
- **정본은 4/15 재수행분 RMSE 10.13mm**(`AGENTS.md:619`, git commit `a217cd3`) + 4/24 table plane
  RMSE 1.24mm(`AGENTS.md:620`).
- "평균 오차 2cm"의 출처는 `CALIBRATION_LOG.md:210-213`의 **좌표 표준편차**(X 1.61 / Y 2.60 / Z 4.18 cm)로
  보이나 정확도 지표가 아니다 → **lead 추정, 미확정**.

### 2-3. [중] §15 #11 "연산 자원 (4070 Ti Super 12GB / 32GB)" — 이 머신이 아니다

- `nvidia-smi` 실측(본 세션) = **NVIDIA GeForce RTX 4090 Laptop GPU, 16376 MiB**.
- `AGENTS.md:297` = "RTX 4090 Laptop (15.6 GB VRAM), Driver 580, CUDA 12.6".
- → 문서가 다른 머신을 전제로 병목을 계산했거나, 사용자에게 별도 데스크톱이 있다. **사용자 확인 항목**.

### 2-4. [중] §3 "D349: link5 4.27mm, **gripper 11.18mm**" — 동결 상수와 불일치

- 동결 D409 worker 상수(lead 직접 판독):
  `sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_worker.py`
  `ANCHOR_REF_LINK5_MM_REPR = "4.272736580324082"` / `ANCHOR_REF_GRIPPER_MM_REPR = "11.340262326338637"`.
- link5 4.27은 일치하나 **gripper는 11.34이지 11.18이 아니다.** 15th doc:110이 이미 "D349 raw
  11.175088374613944 / live 11.340262326338637 / D372 P34 10.9714602318 — **단일 상수 아님**"으로 판정했고,
  동결 harness가 채택한 값은 **live 11.340262326338637**이다.
- → 부록 C "0.5mm는 D349 간극(4.27 / 11.18mm) 대비 작으므로 형상은 무시 가능"은 **틀린 baseline 위에 서 있다.**
  더 결정적으로 D415 ④(`DECISIONS.md:24625-24630`)의 실제 비교 대상은 D349 간극이 아니라
  **세로 3셀 여유 2.014/1.475/1.305mm 대비 Δz 0.336~0.388mm = 17~30% 소모**이고,
  신규 비용은 **yaw 허용 ±10° → ±6~9°** 축소다.

### 2-5. [확인] 문서가 **맞은** 항목

- **sim 원통 치수**: `worker:250-251` `CYL_RADIUS_M = 0.0145` / `CYL_HEIGHT_M = 0.050`,
  `:297-298` `OLD_CYL_RADIUS_M = 0.017  # calibration-only (anchor gate), no D362 physics transfer`
  → D416 ①이 확정한 대로 **sim은 이미 D29×H50**. 문서 §15 #2("재계약 미결")는 **이미 끝난 일**.
- **부록 B 그리퍼 계약**은 `direction_20260708_grasp_pivot.md:26-32`와 문자 그대로 일치한다:
  88.3° / "URDF 기준 1.571rad" / 접촉 반경 43mm·0.75mm/deg / 실용 개구 40~45mm / `(D/2 − 8mm)` /
  cmd 0~5° 금지 / 30mm anchor stall 37.88° / "G0b 이후 sim 계약: gripper joint lower 0.09rad".
  **단 두 가지 결함을 그대로 물려받았다**: (a) `direction:26` "1.571rad"는 D414 ③ 부수(`:24550`)가
  **오기**로 판정(동결 1.5413 rad)했고 `START_HERE.md:85`가 3중 충돌을 미해소 결함으로 기록 중,
  (b) `direction:32`가 0.09rad을 "**D322 G0a에서는 활성화하지 않는다**"로 못박았는데 문서는 현행
  sim 계약처럼 적었다(15th doc:111 지적 반복).
- **부록 E tap 트랙 "1,920 에피소드, 96%"**는 정확하다 — `direction_20260708_grasp_pivot.md` "Tap Track
  Freeze" 절 "1,920 accepted episodes, combined acceptance 96.0%".

## 3. 문서 자체의 구조적 문제 2건 (lead 판독)

1. **사용자 붙여넣기가 §6-4 중간부터 §12-1 직전까지 절단됐다** — **§7·§8·§9·§10·§11 원문 미수령**.
   그런데 문서가 이들을 반복 참조한다: §0("§7~9 실험 설계"), §12-5("§9 조건 3"), §13 P0(§7 시험 순서 전제),
   §15 #3("§10"), #2("§11"), #7("§9-2"), #10("§7-4"), §16(§7-4). → **실험 설계의 심장부가 없다.**
   현 상태로는 "무엇을 어떤 순서로 시험하는가"와 "성공 판정이 무엇인가"를 감사할 수 없다.
2. **§15 #12가 참조하는 `§6-5`는 본문에 존재하지 않는다**(§6은 6-1~6-4까지). 문서 내부 참조 깨짐.

## 4. 문서에서 실제로 **새롭고 검토 가치가 있는** 것 (lead 1차 판단 — 적대 검증 미수행)

**§6 "특징의 두 블록 분리"**가 v1 대비 유일한 실질 신규 내용이고, §2 연구 질문도 여기에 걸려 있다.

- **§6-2 라벨 오염 논증은 repo 규칙과 독립 수렴한 것으로 보인다** — `direction_20260708_grasp_pivot.md:125-126`
  ("기하 라벨 단독 학습 승격 금지 — Kappler/Rubert") + D409 Implication ④(`DECISIONS.md:24379-24381`,
  "A 665는 admission+A-band 통과이지 grasp 후보 아님"). **채택 후보.**
- **§6-4가 v1의 "탈락 사유에 물체 탓/로봇 탓 태그" 를 철회한 것은 개선**이다. "어느 쪽 요인인지는 이 실험으로
  알아내려는 것이지 전제로 깔 것이 아니다"는 논증이 자기일관적이다.
- **다만 lead가 즉시 지목하는 미해결 모순 1건**: §6-3 A블록이 `antipodal_score`("두 접촉면 법선이 서로
  마주보는 정도")와 `pair_distance`를 물체 고유 특징으로 넣는다. 그러나 **D409가 `B 0 / A∧B 0`으로
  옆면 대향 접촉의 부재를 확정했고**(`DECISIONS.md:24368-24369`), D412가 이 그리퍼의 두 영역을
  **"평행한 두 상단 모서리 쌍"**으로 재정의했으며(`:24442`, `:24453-24455`), 문서 자신도 §5-1·부록 A에서
  "대향 두 점이 아니라 한 면 + 한 선"이라고 쓴다. **문서가 부정한 표현을 스키마가 채용한 자기모순 후보** —
  적대 검증 대상이었으나 미완.
- **§5-3 → §15 #4("jaw_depth를 탐색 축으로 / 필수")는 두 조문에 걸린다**: (a) D416 ④
  (`:24721-24722`) "(7,11)mm은 **정렬 standoff**이고 파지 flush는 `D/2−8`, `direction:74-77`이 두 목표
  혼동을 명시 금지" — 문서는 (7,11)을 파지 타깃으로 읽었다. (b) `START_HERE.md:77` + D415 ⑧
  (`:24669-24671`) "**n=3으로 지배 변수 순위를 정하지 말 것**" — §5-3은 사진 n≈2에서 곧바로
  "radial = 지배 변수"로 승격한다. 15th doc §3-7이 v1에 대해 이미 같은 판정을 내렸고 **v2에 미반영**.

## 5. 절차상 즉시 걸리는 항목 (lead 판독, 적대 검증 미수행)

- **§14 "8/20까지 실기 실험 금지(시뮬 전용)"** ↔ D415 ①(`:24647-24649`) "실물 캠페인 개시의 실제 게이트는
  **HW 명시 승인과 속도 정책 결정**뿐" + `direction:118-126` 라벨 사다리 3단이 **실물을 권위 라벨**로 규정 +
  `START_HERE.md:47` 사용자 지시 "**8/20에 매이지 말고 더 빨리 진행**". → 3중 충돌. **사용자 결정 사항.**
- **§13 P0 ③ "마찰을 현실값(0.4/0.35)으로 내려 재시도" + "②나 ③에서 해결되면 그 값이 곧 동결 계약이 된다"**
  → 결과에 맞춰 파라미터를 정하고 계약으로 승격하는 절차. D354/D405(통과 목적 tolerance 수리 금지) 및
  `START_HERE.md:96`(material/physics 변경 금지)와의 관계 판정이 필요하다. 0.4/0.35는 D416 §3-2가
  **출처 없는 신규 수치**로 판정했고 v2에도 출처가 없다.
- **§14 "P-ladder 선언, 기존 G-사다리 동결"** → G-사다리는 교수님 지시 산물이고
  `direction:11` G0b = "원통 파지 + 들어올림 = 첫 완결 case이며 **프로포절 트리거**"다. G-사다리 동결은
  프로포절 트리거를 동결한다는 뜻이 될 수 있다. **교수 확인 사항.**
- **case id / dNNN / 출력 폴더가 v2에 없다** — `AGENTS.md:99-102`는 신규 grasp 산출물을
  `claudedocs/runtime_logs/grasp_track/<case>_<dNNN>/`에만 허용한다.

## 6. C1(물리 경로 개통)용으로 **이미 존재하는 자산** (lead 발견, 본 세션 신규)

D416 ⑤가 지목한 "D407 worker + D409 harness S0~S2 재사용"에 더해:

- **`sim_scripts/p7_branch_b_cube2cm_local_grasp_close_sweep_probe.py`** 가 이미
  **닫힘 스윕 → hold → lift + 게이트** 전체 골격을 구현하고 있다:
  `:287 --object_drift_gate_m(0.006)`, `:293 --min_lift_follow_m(0.006)`, `:299 --hold_steps(30)`,
  `:300 --lift_steps(80)`, `:265` lift 판정(`reached` ∧ ¬`early_kill` ∧ `object_follow_delta_m ≥ min_lift_follow_m`).
  docstring `:1-8`이 **"refuses to count the env's hidden kinematic pose-write attach path as physical
  grasp evidence"**를 명시 — D416 ⑨의 kinematic attach 문제를 이미 회피한 설계다.
- 즉 C1은 **백지 저작이 아니라 (a) 대상 물체를 cube2cm → 원통 D29×H50으로, (b) 자세를 D409 동결 격자 1셀로,
  (c) 임계값을 사전등록으로 바꾸는 작업**에 가깝다. 단 이 probe는 P7/Branch B 시대 산물이며
  `EXPERIMENT_LEDGER.md:105`가 "P7/Branch B 5/14~5/21 evidence shows **rigid jaw grasp fails**"를
  기록하고 있으므로, **재사용 시 그 실패 이력을 먼저 판독해야 한다.**

## 7. 한계 / 권위

- 본 doc은 **lead 단독 판독 권위**다. **적대 검증 미수행**(워크플로우 미완). 15th doc이 5/5 PARTIAL_ADOPT를
  받은 것과 달리, 본 doc의 §4·§5는 **반박을 거치지 않았다** — 다음 세션에서 검증 대상이다.
- §2는 lead가 파일을 직접 열어 확인한 것만 담았고 확신도 HIGH다. §4·§5는 MEDIUM.
- 신규 runtime 수치 0. 물리 verdict 아님. `g0a_pass=false` 불변.
- lead 파생 미검증: "2cm = 좌표 표준편차 유래" 추정 1건.
- 파일 변경: 본 doc + START_HERE + DECISIONS D417 + LEDGER 1행 + MEMORY.md. repo 코드 0.
  git commit/push 0, 로봇 0, Isaac 0, GPU 연산 0, D409 재실행 0, 동결 침범 0.

## 8. [세션 연장 추기] 적대 감사 **완주** — §0·§7의 "미완" 표기는 해소됨

- `wf_3f4d6079-4e7` 완주: 10 agents(ground 5 + verify 5), 에러 0, subagent 1,719,835 tokens,
  tool_uses 203, 1371.3s. **5/5 전부 `PARTIAL_ADOPT`**(v1과 동일 판정, 무조건 채택 0건).
- 결과와 자체 정정은 **`DECISIONS.md` D417-R1**에 전부 기록했다. 본 doc §2~§5보다 D417-R1이 우선한다.
- **본 doc §2-4 / D417 ③은 폐기**: "정본 비교 대상 = 셀 여유 2.014/1.475/1.305mm + yaw ±6~9°"는
  젠가 세로 배치 값이고 yaw는 원통에 없는 자유도(D411 ④). **원통 테이프 기하 비용 = 미계산**.
- 신규 생존 4건: (7,11)은 D409 격자의 **positive control 셀**(`worker:276`)이고 radial은 0~14.5mm
  전수 탐색 소진(`:270-275`) / **radial ≠ 목구멍 깊이, 지배축은 z → 신규 case 승인 대상** /
  **v2에 D341 언급 0건** / 스키마에 Δh·법선 부호·μ 미기재(단 파생량이므로 "결손" 아닌 "기록 권고").
- 처방 변경 1건: 마찰은 "금지"가 아니라 **사전등록 후 저작**(D416 ① "실제 공백 = 질량·마찰 계약뿐").
- 따라서 본 세션은 **적대 검증을 거친 세션**이며 §7의 "적대 검증 미수행" 한계 표기는 해소된다.
