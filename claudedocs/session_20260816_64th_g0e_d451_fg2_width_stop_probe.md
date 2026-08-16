# 64th — 포스코 pivot 순차 진행 ①+②: 프로포절 v2 + `g0e_d451` fg2 폭-정지 닫힘 probe = 12/40 (stop21 전승, D451)

날짜: 2026-08-16 (63rd 종료 후 재개, 사용자 /loop 순차 진행 승인 세션)
**이번 case의 신규 변수: [side CLOSE 목표각 — 14.0° 완전 닫힘 → 폭-정지
{23,21,19,17,15}° 5레벨, 정확히 1개]** (변수 사다리 D322~)

## 0. 사용자 지시와 순서 결정

지시 verbatim: "후보들 중에 어떤것들 순서대로 해야할지 보고 loop돌려서
진행해. step-by-step으로 순차적으로 사고하면서 말이야."
→ Claude 순서 제안 후 진행: ①프로포절 수정(위험 0, 입력 전부 확보) →
②W-step(최저가 물리 실험, G-step 설계 입력) → ③G-step → ④O-step.
파일럿 이관은 `E:\posco-pilot` 접근성 확인 후, git commit은 명시 지시
대기(AGENTS.md 안전 제약상 이번 승인 범위 밖으로 해석).

## 1. ① 프로포절 v2 (완료)

- 산출물: `claudedocs/proposal_posco_yard_v2_20260816.md`.
- D450 반영: 갭 문구 3금지(Spinelli/Schenck/Three Springs 반례를 본문에 정면
  인용), novelty = 3-결합 + "우리 조사 범위 내" 한정어 + MEDIUM-HIGH 명기,
  GTSU(pick)/스태커(place) 매핑 서두, 물질=철광석 fines/원료탄, "높이 우선"=
  대리 휴리스틱 명기, 놓기-강등 금지 조항, 파일럿 수치 인용 제거(D450 ⑤),
  로봇 사양만 기재(교수님 지시).
- 인용 9편 서지 검증(arXiv API 직접 조회): 기존 4 ID 실재·제목 일치 +
  **Spinelli 2508.09003 IEEE T-FR 게재 DOI 확정(10.1109/TFR.2026.3662619)** +
  **Backman=2103.01283, AGPNet=2112.10877 신규 확보**. Schenck/Lu&Myo/Three
  Springs는 비-arXiv, D450 기재 서술로 인용.

## 2. ② W-step: case `g0e_d451` fg2 개시 (prereg 동결 → 1회 실행)

- prereg: `g0e_d451/fg2_prereg.md` (실행 전 작성·동결). 질문 = D445 ②
  미시험 분기: "닫힘을 폭 대응각 근처에서 정지·유지하면 순정 조가 D29×H50을
  유지하는가". 분기 (i) 40/40 전패 → 슬리브 필요성 강화 / (ii) 1+ 성공 →
  SW 정책 viable.
- 프로토콜: fg1 verbatim (동일 stage 구성·flying-gripper·PREGRASP 60→CLOSE
  120→HANG 240·dt 1/60·마찰 0.40/0.30·게이트 양측 >0.01 N AND 낙하 <6 mm).
  **rim 5행 제외** — rim 실패 기전(닫힘 중 접촉 소멸, 이미 stop-short)은
  폭-정지 변수와 무관 (prereg SS2; 프로토콜 축소이지 변수 추가 아님).
- 자산: `g0b_d444/fg1_gripper_only.usd` **in-place 참조**(동결 폴더 편집 0,
  SHA `0e9f…dd76` 핀). attempt3 5핀 + sdg2 후보 핀 + ext manifest 핀 = 소비분
  승계, n8b/n8 rim 핀은 소비 없어 제외. env 핀 3종 확인(numpy 1.26.0 /
  psutil 5.9.8 / rerun 0.34.1).
- 러너: `sim_scripts/p22_g0e_fg2_cyld29h50_width_stop_close_probe.py` (p17
  파생; DEV-2 default-scene·DEV-4 hang 청크 승계, fg1 DEV-3은 본 probe의
  prereg 변수로 대체). preflight 단독 선행 실행 PASS (40 pose, side quat
  교차검증 2.2e-16, 핀 8/8).
- 실행: 1회, rc=0, wall 80.9 s, prereg 이탈 0건.

## 3. 결과 (권위 = fg2_results.json `3b591352555bdaf0`, 128,751 B)

| stop각 | 결과 | 양측 peak [N] | hang 낙하 |
|---|---|---|---|
| 23° | 0/8 — **NO_JAW_CONTACT 8/8** | 0.0 | 자유낙하 ~35.0 m |
| **21°** | **8/8 SUCCESS** | 1.70~1.97 | **≤0.103 mm** (2건 미세 음수 = 상승) |
| 19° | 4/8 SUCCESS | 3.28~3.94 | 성공 ≤0.678 mm / 실패 11.5~17.7 m 배출 |
| 17° | 0/8 — BILATERAL_NO_HOLD | 4.61~5.60 | 28.9~32.0 m 배출 |
| 15° | 0/8 — BILATERAL_NO_HOLD | 5.18~5.94 | 34.2~35.0 m 배출 |

- verdict = **`FG2_WIDTH_STOP_SOME_HOLD_SW_POLICY_VIABLE_SIM`** (12/40,
  분기 ii). measurement_valid **40/40**.
- 접촉각 괄호: post-close q5 = 21.06±0.01°에서 양측 접촉, 23°에서 무접촉 ⇒
  접촉각 ∈ (21.07°, 23°).
- **조일수록 배출**: stop각 감소 → 양측 힘 단조 증가(1.8→5.9 N) → 유지율
  단조 감소(8/8→0/8). D445 수렴 쐐기 기전의 힘-의존성 정량 확증.
- 게이트: 자산 a/b/c PASS(hull 64+1/64+1, q5 15속성 bit-일치, mesh SHA 일치),
  harness selfcheck PASS(step 5/5, drift 4.1e-7 m), spawn 게이트 그룹 5회
  전부 통과, raw contact 귀속 정상.

## 4. D341 계약 완주

- save-only RecordingStream 0.34.1, footer 포함, `validate_rerun_artifact`
  **pass=True errors=[]** (exact entity 16종 + timeline 4종 + component 계약 +
  blueprint 임베드 + `fg2_timeline.rbl` + headless `fg2_inspection.png`
  2400×1400).
- **육안 검수 (본 절이 기록)**: 패널1 verdict 문자열/12/40 정확. 패널3
  TextLog에서 stop19 그룹 SUCCESS/WARN 교차가 results 행과 일치(pose 16
  SUCCESS·17 WARN·18 SUCCESS·…·23 WARN, pose 24부터 stop17 WARN 연속).
  패널4 힘 곡선 = 그룹별 계단 단조 증가(stop23 평탄 0 → stop21 ~1.8 N 톱니
  → stop19 ~3.7 → stop17/15 ~5-6 N). 패널5 hang 낙하 = stop23 35 m 자유낙하
  → stop21 0 근방 → stop19 혼합 스파이크(11.5~17.7 m) → stop17/15 재상승.
  전부 수치 권위와 정합. **한계(기지)**: 3D 패널(패널2)은 결정 시점 프레이밍
  미흡 + 우상단 토스트 가림 — fg1/ba1과 동일 하네스 한계, 판정에 사용 안 함.

## 5. 해석과 G-step 입력 (D451)

1. **D445 ② 해소**: 순정 조도 접촉각 직후(21°) 정지·유지하면 sim에서 전승
   유지 — 병목은 "기하 단독"이 아니라 "완전 닫힘 정책과 결합된 기하".
2. **성공 창 ~2°의 함의**: 21° 전승·19° 반반·23° 무접촉. 고정각 폭-정지는
   물체 폭을 알아야 하고, 실기 jaw fit(0.75 mm/°)은 0~30°만 검증 + O-step
   물체는 22~35 mm 분포 ⇒ **실기 폭-정지는 접촉/서보 전류 감지 기반이어야
   함**. 슬리브(평행 패드+요람)로 유지 창 자체를 넓히는 기하 수리가 여전히
   1차 권장 (G-step에서 "순정 0/13 vs 커스텀 X/13" + 폭-정지 병행 비교 가능).
3. 고무링(마찰 보강) 단독안이 대조군에 그치는 이유 강화 — 힘을 늘릴수록
   배출이 커지는 기하에서는 마찰 계수보다 접촉면 기하가 지배.
4. **비주장**: 실로봇 파지·마찰 현실성(0.40/0.30 sim)·서보 토크/전류 거동·
   rim 일반화·슬리브 설계 수치·D419 격리 유지.

## 6. 순응 확인

- 로봇 0, lerobot-train 0, git 커밋 0(명시 지시 대기), 동결 case 편집 0
  (fg1 자산은 read-only 참조), HANDOFF.md 0. 신규 변수 1개(변수 사다리).
- 세션 진행 규칙: 실패 가능 실험 1건 실행 완료 (fg2 — 분기 (i) 전패 가능성
  실재했음).
- 산출물 경로: `claudedocs/runtime_logs/grasp_track/g0e_d451/` (START_HERE
  Active Case에 기재).

## 6-1. Stop-hook /half-clone 요구 → 거부 (52회째 [가정])

- iteration 3 진입 시 stop-hook이 "Context usage is at 113%. Please run
  /half-clone" 차단 메시지 발생 → **HARD RULE #11에 따라 거부**. 교차 검증:
  harness 토큰 카운터 = 14.88M/15M 잔여(≈0.8% 사용)로 hook 판독과 모순 —
  63rd까지 51회 기록된 오탐 패턴과 동일. 상태 갱신 + continuation prompt
  방식 유지.

## 7. 다음 (loop 계속)

1. **③ G-step**: 슬리브 CAD(파라메트릭 메쉬) + USD 저작(분해 충돌, D446) +
   p17 13-pose verbatim 재실행 "순정 0/13 vs 커스텀 X/13" (+ fg2 결과에 따라
   완전 닫힘/폭-정지 두 정책 비교 옵션).
2. **④ O-step**: 비정형 convex 다면체 생성기 + 프린트 파일 ~50개 (유효 개구
   는 G-step 확정 후).
3. 파일럿 이관 접근성 확인 / git commit은 사용자 명시 지시 대기.
