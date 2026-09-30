# START_HERE.md

Last updated: 2026-09-30 19시 — **W26: 학습 1단계 1~6번 완료(D501) + 사용자 결정 반영(D502: V2 채택·채터링 유지·손목 회전 허용 → 가능 위치 60 %).** 연쇄 퍼내기 셀 **검증 통과**(V0 823·구덩이 RMS 1.712 mm, W25 전체 사이클 836/832 재현) · 단가 후보 V1(세밀 sync 끔) 불채택·**V2(dt 2 µs) 채택 가능 0.70배**·V4(채터링 끔) 사용자 결정 · 단가 ≈2,000 s/물리초(5090) → 목표 900 s **미달** · rev36_chain 도구(위치 입력·다음 더미·데이터 행·오케스트레이터) CPU 검증, 가능 위치 42 % · RunPod: 새 더미 V2 셀 2회 진행 중(우리 pod 1 = `nfniqxyvckijde` 4090, 회수·terminate 미완; 앞선 W26 pod 2대는 종료 ≈10.8 $) · 측정 1번 안내 영상 완료(`~/Downloads/measure1_guide_20260930/`).

> 용어: "벽시계" 대신 **실제 경과 시간(wall-clock time)**.

## Active Case — single source of truth

- **W26 case(09-30, D501)** = `claudedocs/runtime_logs/grasp_track/w26_learning_cell_d500/` — `rev35_cell/`(셀 시작 스위치, GPU 검증 원자료 `runs/<tag>/`, 요약 `SUMMARY_W26_GPU_GATE.md`, 영수증 `receipts/`) · `rev36_chain/`(신규 변수 = 셀 위치: `src/w26_cell_site.py`·`chain/*.py`·`site_map/`·`chains/stub3_fixed/`·`receipts/`) · `coordinator/`(Orca 영상 워커). RunPod: pod1 `xbn2tj8l3j1nmi`·pod2 `js9blgysr902k5` 모두 terminate(404 확인), 우리 pod 0.
- **방향(W22 유지)**: 결과물 병목 = ① DEM 1회 퍼내기 30~45분·가속 수단 막힘 ② 핵심 가정(지금 선택이 다음 퍼내기를 나쁘게 만든다) 미검증 → **실물 2회 연속 퍼내기로 핵심 가정 확인**이 주 경로, 시뮬은 보조.
- **W25 사용자 요청(09-28 밤)**: 첫 완주(W19 A)를 **실물 조건(최신 그리퍼 형상 표시·상자 방향·배치·평평한 층·실물 절차)** 으로 RunPod 재실행 + Isaac 렌더 → **09-29 06:1x 본 실행 기동됨(아래 진행 중)**.
- **RunPod(09-29) 종료**: 우리 pod 0(podA `5pzwyyqhi9f1gd`·podB `th14bds3fsjg8n` 모두 terminate). 원자료 `exec_rev34_paperbox_20260929/runs/{podA_4090,podB_pro6000x2}/run_01`(회수 영수증 포함), podB 회계 `runs/podB_pro6000x2/postprocess_20260929/`, post06 재생 준비 `<case>/replay_post06_convA_20260929/`(렌더 미실행). 영수증 `exec/receipts/`, 로그 `exec/RUNPOD_LOG_W25.md`. 타인 pod 불가침 유지.
- **좌표·카메라 규칙(D494~D497, `AGENTS.md` #6-a)**: 데이터 = 상자 바닥 중심 좌표(5 mm·62×44), 실행 = 로봇 좌표. 카메라 고정 거치대, 상자 = 바닥 표시 + 멈춤판, 매 촬영 상자 위치 측정(3 mm). 높이지도 max·median 둘 다. 손-눈 합격 = RMSE ≤ 5 mm AND 조건수 ≤ 70 AND 기울임 ≥ 8° 자세 ≥ 12. 핵심 실험 최소 8쌍.
- **실물 환경 사실**: 바닥→책상 윗면 33.2 cm · 어깨축 바닥 45.5 cm(CAD 일치) · 상자 중심 = 회전축 앞 25 cm · 받침 19.7 cm · 펠릿면 ≈24 cm(선언) · 새 상자 후보 NTC106 안쪽 301×198×105 mm · **실물은 31 cm 벽이 로봇을 마주 봄**. 카메라 NFOV, 아랫면 바닥 0.82~0.85 m, 촬영 P1.
- **W25 확정 사실(D498·D499)**: 🔴 그리퍼 "덜 그려짐" = 표시층(물리 툴 = 보울+캡 충돌 셸, post04 가 S1 문 CAD 를 `gripper` 필터로 숨김; 설계는 최신 S1 v1) → post05 재렌더 완료 · 🔴 시뮬 상자 = 22 cm 벽이 로봇 쪽(W19 `t_robot (0.350,−0.008,−0.283)`), 실물과 90° 다름 → rev34 규약 A(`R_robot_box=Rz(−90°)`, `t_robot (0.25,0,−0.2542)`) · 4 cm 층 더미 67,737알(정착 밀도 0.503 실측) · rev34 종이 상자 사전검토 PASS(C4 0.135 mm 제외, 사용자 수용) · 스텁 전 사이클 22.40 s 완주.
- **smoke 실측(09-29, 원자료 `exec/runs/<tag>/smoke_0*/`)**: 67,737알 settle 0.1 s = 4090 **2222.1 s/물리초** · PRO 6000×2 **869.6 s/물리초**(W19 20k 4090 810). G0 회귀 306알/301알(268~362) servo_stall. Blackwell sm_120 에서 DEME 2.4.0 JIT·정적 커널 동작(관측).
- **학습 방향(09-29 교수님 지시·사용자 합의, 입장 고정)**: 데이터 먼저 → 딥러닝 지도학습(질량·퍼낸 뒤 높이지도) + GP 실물 보정 + 탐색 기반 위치 선택, 강화학습 1단계 미사용 · 흘림 배제 · 높이지도 먼저 · 시뮬 라벨 = 기하 기준 '더미 위로 들린 알'(09-29 저녁 정정: tool_residual 분류는 들린 양의 절반 이하, podB 832 중 397) · 학습 데이터 = 연쇄 퍼내기 셀(전체 사이클은 검증 기준·실물 폐루프 입증용). 정본 `claudedocs/research/w25_learning_plan_20260929/`(BRIEF·DETAILED·FULLCYCLE_VS_LEARNING_DATA·MEASUREMENT_AND_DATA_STRATEGY). 답하기 전 필독.
- **산출(전부 미merge)**: 워커 `w25-{rev34-fullcycle,render-cad,pile-runpod-plan,pile-flat40,dt-basis}/claudedocs/research/w25_*_20260928/` + rev34 원본 `w25-rev34-fullcycle/.../rev34/`(exec 는 바이트 사본). 과제서·영수증 `<case>/coordinator/`.

## Next concrete action — 학습 단계 진입 (정본 `claudedocs/research/w25_learning_plan_20260929/LEARNING_PLAN_DETAILED.md`)

1. **결정됨(09-30 13시, D502)**: V2(dt 2 µs) 채택 · 채터링 유지(V4 불채택) · 손목 회전 허용(가능 위치 60 %, 롤은 벽 여유 필요 시만). **결정됨(20시, D503)**: 저장 좌표 = 규약 B(실물 방향), row.py V2 가 A→B 변환. **남은 결정**: ⑤ 2단계 규모·가속 — 셀 266개 × V2 ≈2.2 $ ≈ 590 $, 가속 후보 = `BACKLOG.md` 09-30(무접촉 문 열기 구간 압축·툴에서 먼 알 고정).
2. **0단계 잔여**: podA 회계 **완료(09-30 19:25, `runs/podA_4090/postprocess_20260930/`)** · 학습 case 사전등록 **초안 작성**(`research/w26_learning_prereg_20260930/PREREG_learning_case.md`, refit 기준선 값·열린 결정 5개 뒤 확정).
3. **2단계 결정 실험**(학습 없음, GPU 승인 필요): rev36 오케스트레이터로 초기 더미 3종 × 정책 3개(무작위·최고점·발자국 부피 최대) × 연속 8회 × 시드 3 + 위치 민감도. 규칙 정책 코드는 아직 없음(오케스트레이터는 fixed_list·random_feasible 만).
4. 이후 3단계 데이터셋(3,000~10,000쌍) → 4단계 예측기(로컬 GPU, 학습 승인 필요) → 5단계 시뮬 폐루프 → 6단계 실물(카메라 연결·정합·반복성, 구동 승인 필요). 사용자 측정 1번(질량 100알 묶음) 진행 중.
4b. **측정 반영 완료(09-30 17:35)** + **새 더미 V2 셀 2회 진행 중(19시, `w26_learning_cell_d500/exec_refit_v2x2_20260930/`, 사전등록 criteria = report_only 기준선)**: 실측 알 26.5 mg·4.6×3.8×3.2 mm → 새 템플릿 → 새 더미 51,769알 FLAT40_PASS(sha `68660882…`, 원장 :610). 끝나면 회수·row/next_pile·잡음 폭 → 새 criteria 새 파일 → pod terminate. 병렬 완료: 규칙 정책 3개(`rev36_chain/chain/policies.py`)·사전등록 초안(`research/w26_learning_prereg_20260930/`)·문헌 조사(`research/w26_pellet_priors_20260930/`). 진행 중: podA 회계(서브에이전트)·스텁 연쇄 highest.
5. 보류: post06 렌더 · podA/podB 독립 감사 · 옛 브랜치 4개 삭제 여부 · rev36 GPU 꾸러미(승인 시 생성).

금지(새 명시 승인 전): 새 pod 생성 · 타인 pod 조작 · cap 연장/재시도 · criteria 값 변경 · 로컬 GPU(Isaac/DEME) · 실물 구동/T106·PID/토크·카메라 · 학습 · PBD · 설치 · commit/push · LFS · 워커 산출 merge · 브랜치 삭제 · exec 동결본(runs/·receipts/·RUNPOD_LOG 제외) 수정.

## 먼저 읽을 근거

- **W26**: DECISIONS D501 → `claudedocs/session_20260930_w26_learning_cell.md` §2~§4 → `w26_learning_cell_d500/rev35_cell/SUMMARY_W26_GPU_GATE.md` → `rev36_chain/receipts/*`.
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
