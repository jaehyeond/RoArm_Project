# START_HERE.md

Last updated: 2026-09-29 21:2x — **W25(3일차) 완료: RunPod 병행 본 실행 2회 완주·회수·종료**(podB PRO 6000×2 36,372.8 s·40.21 $ / podA 4090 52,707.9 s·10.62 $, 둘 다 completed_rc0·abort None·회수 14/14 sha·terminate 204, **우리 pod 0**). **R_full 1.449**(스모크 2.556 은 접촉 단계 한정). **기하 라벨(lift_end 들린 알) 836 vs 832(0.5 %)**. 원시 배출 585~665 vs 251~297(문 재닫기 2.39° vs 3.04°, 승격 아님). podB 회계 불일치 0·정착 cadence FAIL. **교수님 지시로 학습 단계 진입** — 정본 `claudedocs/research/w25_learning_plan_20260929/`. D500(`DECISIONS.md:30350`), 원장 :607. 세션 문서 §13.

> 용어: "벽시계" 대신 **실제 경과 시간(wall-clock time)**.

## Active Case — single source of truth

- **방향(W22 유지)**: 결과물 병목 = ① DEM 1회 퍼내기 30~45분·가속 수단 막힘 ② 핵심 가정(지금 선택이 다음 퍼내기를 나쁘게 만든다) 미검증 → **실물 2회 연속 퍼내기로 핵심 가정 확인**이 주 경로, 시뮬은 보조.
- **W25 사용자 요청(09-28 밤)**: 첫 완주(W19 A)를 **실물 조건(최신 그리퍼 형상 표시·상자 방향·배치·평평한 층·실물 절차)** 으로 RunPod 재실행 + Isaac 렌더 → **09-29 06:1x 본 실행 기동됨(아래 진행 중)**.
- **RunPod(09-29) 종료**: 우리 pod 0(podA `5pzwyyqhi9f1gd`·podB `th14bds3fsjg8n` 모두 terminate). 원자료 `exec_rev34_paperbox_20260929/runs/{podA_4090,podB_pro6000x2}/run_01`(회수 영수증 포함), podB 회계 `runs/podB_pro6000x2/postprocess_20260929/`, post06 재생 준비 `<case>/replay_post06_convA_20260929/`(렌더 미실행). 영수증 `exec/receipts/`, 로그 `exec/RUNPOD_LOG_W25.md`. 타인 pod 불가침 유지.
- **좌표·카메라 규칙(D494~D497, `AGENTS.md` #6-a)**: 데이터 = 상자 바닥 중심 좌표(5 mm·62×44), 실행 = 로봇 좌표. 카메라 고정 거치대, 상자 = 바닥 표시 + 멈춤판, 매 촬영 상자 위치 측정(3 mm). 높이지도 max·median 둘 다. 손-눈 합격 = RMSE ≤ 5 mm AND 조건수 ≤ 70 AND 기울임 ≥ 8° 자세 ≥ 12. 핵심 실험 최소 8쌍.
- **실물 환경 사실**: 바닥→책상 윗면 33.2 cm · 어깨축 바닥 45.5 cm(CAD 일치) · 상자 중심 = 회전축 앞 25 cm · 받침 19.7 cm · 펠릿면 ≈24 cm(선언) · 새 상자 후보 NTC106 안쪽 301×198×105 mm · **실물은 31 cm 벽이 로봇을 마주 봄**. 카메라 NFOV, 아랫면 바닥 0.82~0.85 m, 촬영 P1.
- **W25 확정 사실(D498·D499)**: 🔴 그리퍼 "덜 그려짐" = 표시층(물리 툴 = 보울+캡 충돌 셸, post04 가 S1 문 CAD 를 `gripper` 필터로 숨김; 설계는 최신 S1 v1) → post05 재렌더 완료 · 🔴 시뮬 상자 = 22 cm 벽이 로봇 쪽(W19 `t_robot (0.350,−0.008,−0.283)`), 실물과 90° 다름 → rev34 규약 A(`R_robot_box=Rz(−90°)`, `t_robot (0.25,0,−0.2542)`) · 4 cm 층 더미 67,737알(정착 밀도 0.503 실측) · rev34 종이 상자 사전검토 PASS(C4 0.135 mm 제외, 사용자 수용) · 스텁 전 사이클 22.40 s 완주.
- **smoke 실측(09-29, 원자료 `exec/runs/<tag>/smoke_0*/`)**: 67,737알 settle 0.1 s = 4090 **2222.1 s/물리초** · PRO 6000×2 **869.6 s/물리초**(W19 20k 4090 810). G0 회귀 306알/301알(268~362) servo_stall. Blackwell sm_120 에서 DEME 2.4.0 JIT·정적 커널 동작(관측).
- **학습 방향(09-29 교수님 지시·사용자 합의, 입장 고정)**: 데이터 먼저 → 딥러닝 지도학습(질량·퍼낸 뒤 높이지도) + GP 실물 보정 + 탐색 기반 위치 선택, 강화학습 1단계 미사용 · 흘림 배제 · 높이지도 먼저 · 시뮬 라벨 = 기하 기준 '더미 위로 들린 알'(09-29 저녁 정정: tool_residual 분류는 들린 양의 절반 이하, podB 832 중 397) · 학습 데이터 = 연쇄 퍼내기 셀(전체 사이클은 검증 기준·실물 폐루프 입증용). 정본 `claudedocs/research/w25_learning_plan_20260929/`(BRIEF·DETAILED·FULLCYCLE_VS_LEARNING_DATA). 답하기 전 필독.
- **산출(전부 미merge)**: 워커 `w25-{rev34-fullcycle,render-cad,pile-runpod-plan,pile-flat40,dt-basis}/claudedocs/research/w25_*_20260928/` + rev34 원본 `w25-rev34-fullcycle/.../rev34/`(exec 는 바이트 사본). 과제서·영수증 `<case>/coordinator/`.

## Next concrete action — 학습 단계 진입 (정본 `claudedocs/research/w25_learning_plan_20260929/LEARNING_PLAN_DETAILED.md`)

1. **0단계**: podA 회계(podB 와 같은 절차) · 학습 case 새 폴더 사전등록(목표 M\*, 라벨 = 기하 기준 lift_end 들린 알, 지표, 기준선 3개, 에피소드·초기 더미 단위 분할) · 라벨 잡음 폭(836/832, 구덩이 RMS 1.93 mm) 기록.
2. **1단계**: 연쇄 퍼내기 셀 제작(rev34 사본, 들림 후 들린 알 삭제 → 짧은 정착 → 다음 위치, 참/렌더 높이지도 저장) · 관문(셀 lift_end 들린 알이 836/832 변동 폭 안, 질량 보존, 셀 단가 실측) · 단가 줄이기 후보 하나씩(ROI·dt 2 µs·채터링) · GPU 는 4090 다수 병렬 우선(D500). 병행: 실물 M1(알 1개 질량 — 0.06 g 미확정 vs 시뮬 20.26 mg)·M4 부피밀도·M5 안식각(`pellet-model/.../MEASUREMENT_PROTOCOL_20260909.md`).
3. **2단계 결정 실험**(학습 없음): 초기 더미 3종 × 정책 3개(무작위·최고점·발자국 부피 최대) × 연속 8회 × 시드 3 + 위치 민감도.
4. 이후 3단계 데이터셋(3,000~10,000쌍) → 4단계 예측기(로컬 GPU, 학습 승인 필요) → 5단계 시뮬 폐루프 → 6단계 실물(카메라 연결·정합·반복성, 구동 승인 필요).
5. 보류: post06 렌더(규약 A 패치 CPU 자체검사 9/9, 렌더·육안 미실행) · podA/podB 독립 감사 · 옛 브랜치 4개 삭제 여부.

금지(새 명시 승인 전): 새 pod 생성 · 타인 pod 조작 · cap 연장/재시도 · criteria 값 변경 · 로컬 GPU(Isaac/DEME) · 실물 구동/T106·PID/토크·카메라 · 학습 · PBD · 설치 · commit/push · LFS · 워커 산출 merge · 브랜치 삭제 · exec 동결본(runs/·receipts/·RUNPOD_LOG 제외) 수정.

## 먼저 읽을 근거

- AGENTS.md(#6-a) → DECISIONS_ACTIVE.md(D493~D499) → 세션 문서 §12~§13 → `exec/EXEC_PIN.json`·`exec/receipts/*`·`exec/RUNPOD_LOG_W25.md` → 필요 시 워커 REPORT.md 5건 + `coordinator/RUNPOD_PLAN_W25.md`.
- 관측은 JSON/NPZ 정본, 그림·Rerun·Isaac 은 검사층(표시층 ≠ 물리 기하).

## 이 프로젝트에서 되풀이하지 말 것 (최근 교훈)

- 🔴 **표시층 ≠ 물리 기하** — prim 이름 부분 문자열로 숨기지 않기, 숨긴 목록은 매니페스트로 실측(W25).
- 🔴 **좌표 방향은 실행값(`t_robot`)으로 확정**, 도크스트링·파일명·그림 인상 금지(W22·W25).
- 🔴 **cap 은 결과 전에** — 새 revision·새 criteria 파일(D492·W25 실행: `criteria_w25_paperbox_cap32h.json`).
- 🔴 **GPU 비교는 사전등록 규칙으로**(R ≥ 1.3, 결과 본 뒤 변경 금지) · "장비 혼합 반복"은 동일 장비 반복이 아님(W25).
- ssh 로 `A && B && setsid … &` 보내면 원격 서브셸이 채널을 잡아 로컬 ssh 가 매달린다 → 서브셸 감싸기/스크립트 업로드(W25) · pgrep 자기매치 금지 → child_pid kill -0(W19·W25).
- 검증 결과가 다음 결정을 바꾸지 못하면 돌리지 않는다(W22) · 워커 과제문에 기대 답 금지(W22~W25) · 손-눈은 RMSE 만으로 합격 금지(D497) · (W21) step 경계 미도달 시 소프트 상한 미발동 → 바깥 watchdog(러너 하드 상한).

## 과거 완료 결과 — 유지

- W11 dt 1 µs 517알/10.4731 g · W16 run_02 기준선 11,021.63 s · W19 A 물리 24.807 s 완주(러너 20,977.836 s), 배출 272~327알(정착 미증명) · W21 P2 n=5,000 `CELL_UNRUNNABLE`(D493) · 재닫기 문 각도 2.880/3.054/3.549°(W16/W19A/W13) · 10 µs 폭주 = 감쇠 적분 불안정(W24).

## 실물 종료 정본 / 유지할 관찰

- `claudedocs/session_20260911_hardware_closeout_next_sim.md`; HOME [0,0,90,0,0,0]. S1 v1 5/5 사이클 93 s(D481), 배출 19.95 g 1건(원장 :585). 9/18 사용자 관찰: 운반 중 유출 ≈0~1알. 실물 카메라 **미연결**.

## 신뢰하지 않을 과거 상태 / 환경

- CONTINUE_2026091x·20260923 계열 GO/COMMANDS, HANDOFF.md/TASKS.md, `coordinator/RUNPOD_PLAN_W25.md` 의 단가·P1/P2′ 안(실행 전 계획). RunPod 랩 계정: 타인 pod 8개(RUNNING 2) — 조작 금지. 09-28 랩 과금 26.08 $.
- `w13_kinematics.py:10-16` "(0,+R)" 도크스트링(낡음) · post04 주석 "순정 그리퍼 숨김"(실제는 S1 문 CAD) · 9/7 "베이스판 38 cm"(현재 33.2) · 8/31 `roarm_base` 격자(D496 개정, 코드 미반영) · exec `COPY_RECEIPT.rule` 산문 "56"(실제 54).
- isaaclab numpy 1.26.0·psutil 5.9.8·Rerun 0.34.1·DEME 2.4.0 유지. Codex 0.158.0. pod 이미지 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 드라이버 podA 580.159.03·podB 595.91.07.
