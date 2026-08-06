# 2026-08-06 (27th) — G0b T3: 사용자 결정 수신(Arm-C 완전 제거) + tie-break 적대 패널(양 렌즈 CONDITIONAL_D 수렴) + 채점표 방법론 1차 소스 검증

이번 case의 신규 변수: [없음 — 설계·검증 계층만. Isaac 0, 자산·코드 변경 0, 로봇 HW 0,
lerobot-train 0, git 0. T3 본선은 잔여 확인 2건(②′·③)에 게이트된 상태로 종료.]

## §1 부트 + 사용자 결정 수신

1. Current-State Protocol 부트(26th판 기준) + 권위 JSON 스팟체크 2건 **전건 일치**:
   `t3_jaw_audit3_results.json` verdict=JAW_AUDIT_CONSISTENT, floor clearance a1 −0.0472mm /
   a4 −0.0483mm, angle_inv delta 0.00296mm; `anygrasp_..._findings_raw.json` 논문 35/3/1 ·
   전사 15/15 · impact NO_CHANGE×2. 북키핑 불일치 1건 flag: /half-clone 거부 횟수
   (START_HERE:77 "12회" vs 26th doc §4 "14회" vs 부트 프롬프트 "15회") → 본 세션 16회로 정합화.
2. **사용자 결정(verbatim 요지)**: "C는 강등 말고 완전 빼는거에 동의. 근데 tie-break는
   어떻게 하는게 좋은지 헷갈리네, 이거에 대해서 고민해봐. step-by-step으로 … 원래 이렇게
   sim에서 물리엔진으로 할때 우리처럼 이러한 채점표 기반으로 되는지 안되는지 기준을 두고
   하는거야?" → ① **Arm-C 완전 제거 확정**(예비 강등 아님) ② tie-break = lead 분석 +
   적대 패널 위임 ③ D426 = 미답(잔여).
3. C 완전 제거 파급 1건 사용자 고지: Gate-0(시각 메시 원위부 실재 감사) 실패 분기에서
   기존 "수제 저작 승인 or **C 재평가**" 중 C 폴백 소멸 → 잔여 분기 = 수제 저작 승인 or
   정지·재상의 둘뿐. 사용자 무답 시 완전 제거 유지, Gate-0 실패 시 재질의 예정.

## §2 워크플로우 `wf_29eb2529-df7` (세션 내 발사·완주·회수)

- 6 agents / 2 phase, 에러 0, 571,617 tok, 81 tool uses, 377s.
  - **Verify 4렌즈**(1차 소스 강제): isaac-official / benchmarks / grasp-datasets /
    counter-lens — 전원 schema 강제, URL+파일/줄+버전 인용 의무.
  - **TieBreak 적대 2렌즈**: F 옹호 vs D 옹호 — 각자 최강 변호 후 정직 자기판정 의무,
    repo file:line 인용 의무(24th doc / DECISIONS D419·D424·D425 / t3_prereg / AGENTS.md).
- 전문 영속화: `g0b_d420/t3r_tiebreak_scorecard_verify_wf_29eb2529_findings_raw.json`
  (75,712 B, sha256 `1bb3c85656f3a56ef156873c93f2d55f0aaed8708488d7672988dce688b5fcdd`).

## §3 tie-break 판정 — 양 렌즈 CONDITIONAL_D 수렴 (권고, 미발효)

- **F 옹호 렌즈**: F 논거 6종(최소 diff / 고정 조 seed-plane 4캐리어 보존 / 기존 시그니처
  비교 가능성 / 원안 준수 / touch-only-what-you-must / F 물림 x≈5~9mm ⊂ T1 0~12mm) 구축
  후 자기판정 = **CONDITIONAL_D**. 특기 조건: F 폴백 채택 시 "**T5 진입 차단 재검증 게이트
  의무** — T7 RL 궤적은 튜플로 구속 불가, 유령(플러그) 접촉 신호는 게이트 불가" 명시.
- **D 옹호 렌즈**: **CONDITIONAL_D**. 핵심 = ① 채택≠귀속 범주 분리(귀속은 (B,F,D) 요인
  매트릭스가 이미 확보 — 24th §4-3의 범주 오류 지적과 동종 논리) ② "최소 diff"의 기준을
  동결 자산이 아닌 **실물 그리퍼**로 두면 D가 최소(F는 실증된 불일치 지점[조 목구멍]을
  보존) ③ F 성공 = margin≥+4.5mm 우회 하 sub-band(x≈5~9mm vs 실물 0~12mm) 한정 인증
  ④ T6 랜덤화/T7 RL 탐색 궤적의 유령 접촉 오염(반경 r≈11.77mm, TCP−4.458mm) ⑤ Variable
  Ladder상 늦은 자산 교체 = 최고가 경로(T4 기준선 무효화) ⑥ D 부작용(커버리지 4→2)은
  측정 가능 → 게이트 변환 가능 vs F 잔존 결함의 미래 간섭면은 비한정.
- **권고안(②′ 확인 대기)**: F∧D 동시 성공 시 **기본 채택 = D**, 가드 5종 전부 사전등록
  (부록 D) 후 통과 조건, 실패 시 F 폴백:
  - (G-a) Arm-B가 예측 클래스대로 실패(a2 튜플 무접촉 LIFT_FAIL + a4 튜플 part_031 정지
    top−1.6mm ±1.0mm/±1스텝) ∧ F·D 모두 off-prediction 0. B 예상외 성공 = 전면 정지,
    채택 논의 무효.
  - (G-b) **잔여 커버리지 게이트**: 027/031이 인증 물림 깊이의 원통 접촉 영역을 커버함을
    첫 Isaac 전에 수치 문턱값과 함께 사전등록·통과(소급 변경 무효).
  - (G-c) **F-D 접촉 일치 게이트**: 동일 튜플에서 D의 고정 조 접촉 관측치(첫 접촉 높이 /
    접촉 시작 각도 / lift follow)가 F와 ±1.0mm·±1이산스텝 내 일치. 이탈 시 채택 보류 +
    사전등록 판별 앵커(24th §4-8)로 원인 귀속 선행.
  - (G-d) F 자산·manifest·성공 증거 전량 보존(forward-only). T4에서 고정 조 예측외 차이
    발생 시 "물리 갭" 결론 전 **F 대조 sim 재실행 1회 의무**.
  - (G-e) **F 폴백 시**: 마개 잔존을 자산 불변식으로 명기(전 미래 튜플 margin≥+4.5mm
    의무, 물림 상한 x≤L−4.5mm 문서화) + **T6(랜덤화) 진입 전 마개 제거 재검증 의무 게이트**.
- 절차 적법성: 부록 D 미발행 + Isaac 실행 전 = **사전 개정**(사후 변경 아님). 원안
  "F 기본"도 미비준 lead 제안이었음(24th §6 ② 대기 상태). 발효는 D426 기록으로만.

## §4 채점표 방법론 검증 — "gate 기반 pass/fail = 분야 표준" 확인 (1차 소스)

- **표준 확인**(전부 원본 코드/논문 줄 단위 인용, 상세 = findings raw JSON):
  Isaac Lab lift `minimal_height 0.04m + success_threshold 0.05m + 낙하 종료 −0.05m`
  (release/3.0.0-beta2 @6a7acb0, rewards.py·lift_env_cfg.py·terminations.py, blob sha 포함) /
  IsaacGymEnvs AllegroHand successTolerance 0.1rad·fallDistance 0.24m / ManiSkill3 PickCube
  `dist≤0.025m ∧ qvel<0.2`(pick_cube.py:147-160) / Meta-World PickPlace `0.07m`·lift 2cm
  (sawyer_pick_place_v3.py:98-105) / robosuite Lift `table+0.04m`(lift.py:433-444) / RLBench
  DetectedCondition boolean / ACRONYM: FleX 흔들기 후 양손가락 접촉 유지 = 성공 라벨
  (arXiv:2011.09584 §III, 17.744M 중 59.21% pass) / DexGraspNet: 6중력방향×100스텝 접촉
  유지 ∧ 관통<1mm(isaac_validator.py:71,108-130,280-304 — 논문·코드 상호 확인).
- **두 전통**: 물리 시행 채점표(현 대세) vs 해석적 점수(GraspIt! ε/v — 물리 시행 없음,
  GraspNet-1B force-closure 11등급). Kappler ICRA 2015(~50만 파지)가 ε-임계 라벨이 물리
  시행 라벨보다 부정확함을 실증 → 물리 시행 채점 우위(단 검증 대상은 crowdsourced 인간
  판정, 실로봇 대규모 상관 연구는 미검증 lead만 확보 — caveat 명기).
- **함정 4종**(counter-lens): ① 문턱값=저작 가정(자체 전례: HARD RULE #3 VGST, 이번
  FATAL-1 "잡혀도 실패 코딩") ② **충돌 기하가 verdict 반전** — CoACD(SIGGRAPH 2022)
  동일 태스크 49%→80%(V-HACD vs CoACD; ⚠️ 중계 요약 경유 인용 등급) + NVIDIA Factory
  (RSS 2022) "convex 분해 spatial artifacts" 자인 → **유령 마개 현상의 문헌 정합** ③
  SIMPLER(CoRL 2024, 쌍 평가 1500+): sim 절대 성공률 실물 비이전·순위만(MMRV) →
  D419 ④("sim=발견, 권위=실물") 문헌 정합 ④ 이진 성공이 실패 구조 은폐(RoboEval
  arXiv:2507.00435; "Beyond Binary Success" arXiv:2605.19986 — ⚠️ 2026 preprint 미심사)
  → 우리 단계 분류(APPROACH/LATCH/HOLD/LIFT_FAIL)가 선제 이행.
- **종합**: 우리 방식 = 표준 채점표 + 비표준 강화 3종(**사전등록** — 로보틱스 희귀,
  HRI 서브필드 소수뿐[Gunes 2022 survey] / **적대검증+sha 핀** / **D341 육안검수**) —
  강화 3종이 함정 ①②④를 정조준. 사용자 질문에 대한 답 = "표준 맞음, 단 우리는
  문턱값 사후 조정이 불가능하도록 스스로 묶은 강화판".

## §5 산출물

- `g0b_d420/t3r_tiebreak_scorecard_verify_wf_29eb2529_findings_raw.json`(§2, 신규)
- 본 doc / START_HERE 27th판 / LEDGER 27th row / MEMORY 27th entry(21st entry 회전).
- **DECISIONS append 없음** — 사유: C 제거·tie-break 개정의 공식 발효는 D426(확인 ③
  대기)에 귀속. 미비준 상태에서 조항 저작 금지(26th 선례와 동일 논리).

## §6 규칙 이행

- 실패 가능 실험 = tie-break 적대 패널(F 옹호 렌즈가 F를 방어해냈다면 권고 반전됐음 —
  실제로는 F 옹호조차 CONDITIONAL_D 자기판정) + 방법론 검증(표준 아님 판정 가능했음).
- Isaac 미실행 사유 = 본선이 확인 잔여 2건에 게이트(#18).
- **/half-clone 거부 16회째(#11, stop-hook context 90% 지시에도 — 본 end-of-session
  update + continuation prompt로 대체).** HANDOFF 미생성(#7).

## §7 다음 부트

잔여 확인 2건이 유일 블로커: **②′** F-arm 신설 + tie-break 조건부 D 권고안(§3) 동의? /
**③** D426 기록(해금 발언 + C 완전 제거 + tie-break 개정 통합, 3중 앵커 scoped-supersede)
진행? + 별건 scratchpad 118MB 처분. 수령 후 = D426 저작 → Gate-0 → p9 파라미터화+게이트
v2 → D423 적대검증·sha 핀 → arm 자산 저작(B/F/D) → 부록 D 일괄 발행(tie-break 가드 포함)
→ Isaac 순차 B(a2)→B반복성→B(a4)→F→D→[조건부 A] → T4.
