# START_HERE.md

Last updated: 2026-09-28 밤 — **W25: 첫 완주 렌더의 그리퍼 "덜 그려짐" 원인 규명(표시층, 설계는 최신 S1 v1) · 시뮬 상자 방향 사실 확정(22 cm 벽이 로봇 쪽) · 실물 정렬 전체 사이클 rev34 CPU 준비 · 평평한 층 비용(10~23 h > cap) · worktree 보관 2차 · 랩미팅(9/29) 자료 정정.** 새 물리·GPU·RunPod·실물 구동·설치·commit 0. 세션 문서 [session_20260928_w25_realign_fullcycle_prep.md](claudedocs/session_20260928_w25_realign_fullcycle_prep.md), D498.

> 용어: "벽시계" 대신 **실제 경과 시간(wall-clock time)**.

## Active Case — single source of truth

- **방향(W22 유지)**: 결과물 병목 = ① DEM 1회 퍼내기 30~45분·가속 수단 막힘 → "시뮬 수천 회" 불가 ② 핵심 가정(지금 선택이 다음 퍼내기를 나쁘게 만든다) 미검증 → **실물 2회 연속 퍼내기로 핵심 가정 확인**이 주 경로, 시뮬은 보조.
- **W25 사용자 요청(09-28 밤)**: 첫 완주(W19 A)를 **실물 조건(최신 그리퍼 형상 표시·상자 방향·배치·평평한 층·실물 절차)** 으로 RunPod 재실행 + Isaac 렌더 → **rev34 CPU 준비 완료, 실행은 승인 대기**(아래 Next).
- **좌표·카메라 규칙(D494~D497, `AGENTS.md` #6-a)**: 데이터 = 상자 바닥 중심 좌표(5 mm·62×44), 실행 = 로봇 좌표. 카메라 고정 거치대, 상자 = 바닥 표시 + 멈춤판, 매 촬영 상자 위치 측정(3 mm). 높이지도 max·median 둘 다. 손-눈 합격 = RMSE ≤ 5 mm AND 조건수 ≤ 70 AND 기울임 ≥ 8° 자세 ≥ 12. 핵심 실험 최소 8쌍.
- **실물 환경 사실**: 바닥→책상 윗면 33.2 cm · 어깨축 바닥 45.5 cm(CAD 일치) · 상자 중심 = 회전축 앞 25 cm · 받침 19.7 cm · 펠릿면 ≈24 cm(선언) · 새 상자 후보 NTC106 안쪽 301×198×105 mm · **실물은 31 cm 벽이 로봇을 마주 봄**. 카메라 NFOV, 아랫면 바닥 0.82~0.85 m, 촬영 P1.
- **W25 확정 사실(D498)**:
  - 🔴 **그리퍼**: 시뮬 물리 툴 = 충돌 셸(보울 벽+캡, 204 tri/부품), 재생은 셸을 그리고 USD 의 S1 v1 **문 CAD(`gripper_link`)를 `gripper` 필터로 숨김** → "덜 그려짐". 설계는 최신 S1 v1(후속 없음). B 워커 post05 초안 = CAD 두 부품을 원자료 포즈로(16,813 sync 배치 오차 ≤ 7.2e-5 mm, 메인 재현).
  - 🔴 **시뮬 상자 = 22 cm 벽이 로봇 쪽**(W19 `t_robot (0.350, −0.008, −0.283)`), 실물과 90° 다름. 전체 사이클 툴 자세는 실물 FK 를 따름(스쿱 셀만 R_W 고정).
  - **평평한 층 비용**(C, 메인 재계산 일치): NTC106 40,247~64,725알 → 전체 사이클 10.30~22.60 h·7.6~16.7 $ — **전부 현행 cap 32,400 s 초과**. 기존 20k 재사용은 12~15 mm 층이라 불가.
  - **rev34**(A): CPU 사전검토 PASS — OFF = rev32 원자료 바이트 동일(96키), ON(규약 A: `R_robot_box=Rz(−90°)`, `t_robot (0.25,0,−0.255)`) 관절 제한 위반 0·툴–트레이 54 mm(NTC106)/**33.9 mm(현재 종이 상자 310×220·벽 230 mm)**·컵 26 mm, 절차 스위치(표면 열기·채터링 ≤3회·+80 mm 재닫기) 실물 순서 재현(스텁). **미해결·결정 대기**: C4 취점 재현 0.112~0.135 mm(rev32 `r0=hypot` 상속) · 운반 높이가 종이 상자 벽 윗단보다 22 mm 위뿐 → `travel_cm` 상향 · 규약 A 부호·상자 y=0 확인 · Isaac 재생(post05+규약 A) 패치 · 열기 토크 200 · `verify_w13_self` 크래시(rev32 유래). params: `w25-rev34-fullcycle/…/rev34/params_w25_paperbox.json`.
- **산출(전부 미merge)**: 워커 `w25-{rev34-fullcycle,render-cad,pile-runpod-plan}/claudedocs/research/w25_*_20260928/` + rev34 사본 `w25-rev34-fullcycle/claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/rev34/`. 과제서·영수증 `claudedocs/runtime_logs/grasp_track/w25_realign_fullcycle_d498/coordinator/`. 이전 W22~W24 산출 경로는 세션 문서 §W22 참조(해부도·A/B/C·배치 기하·9셀 계획).

## Next concrete action — 사용자 승인 대기 (병렬)

1. **로컬 GPU(≈8 분)**: post05 로 W19 A 재렌더(그리퍼 CAD 표시) — B REPORT §6 명령, `setsid`+watchdog. 랩미팅 그림 교체 가능.
2. **RunPod rev34 전체 사이클(현재 종이 상자 310×220 × 4 cm 층 = 61,408~74,067알, 15.0~26.4 h·11~20 $)** — 사용자 결정: (a) 부피밀도 0.456/0.55(알 수)·안쪽 치수 310×220 vs 306×216 (b) 절차 스위치(표면 열기+채터링) ON/OFF (c) **cap(현행 9 h → 새 revision 에서 사전 결정, ≥32 h 제안)** (d) 비용 상한(25 $ 제안) (e) 기동 시각 (f) C4·travel_cm 처리. → 더미 생성(로컬 GPU 0.7~1.3 h, `--seed-margin-mm 4.6`, 정착 관문) → 격자 증거 재작성 → rev34 PB0~PB5 CPU 재검토 → pod(`roarm_w25_*`, 4090 SECURE, minCudaVersion 13.0) 부트스트랩 v5 사본 → 스모크 3종 → GO → 회수·terminate. dt 사다리 {2,5} µs 는 사전등록만(사용자: GPU 보류).
3. worktree 10개 2단계 **완료**(23:36, 총 보관 28·남은 실제 worktree 15). 옛 브랜치 4개(`jaehyeond/{research-survey,w12-input-audit,w12-isaac-replay,w13-full-cycle}`, 태그로 보존) 삭제 여부만 결정.
4. **A 실물 환경**(변경 없음): 회전축→책상 앞 모서리 실측 → NTC106·받침 → 카메라 설치·잡음 → 손-눈 정합(구동 승인) → 반복성 검사. **D 실측 뒤 D499(용기·격자)**.
5. 합류: 반복성 검사 통과 + 사전 등록 확정(8쌍) → 실물 핵심 실험(목표 10/12), 시뮬 rev34/9셀과 비교.

금지(새 명시 승인 전): RunPod(우리 pod 만·`roarm_w25_*`·완료 즉시 terminate·대기 시 stop) · 로컬 GPU(Isaac/DEME) · 실물 구동/T106·PID/토크·카메라 · 학습 · PBD · 설치 · commit/push · LFS · 워커 산출 merge · 브랜치 삭제.

## 먼저 읽을 근거

- AGENTS.md(#6-a) → DECISIONS_ACTIVE.md(D493~D498) → 이번 세션 문서 → 워커 REPORT.md 5건(A/B/C/D/E) + `coordinator/RUNPOD_PLAN_W25.md` → 필요 시 W22 세션 §8~§9.
- 관측은 JSON/NPZ 정본, 그림·Rerun·Isaac 은 검사층(표시층 ≠ 물리 기하).

## 이 프로젝트에서 되풀이하지 말 것 (최근 교훈)

- 🔴 **표시층 ≠ 물리 기하** — 툴을 무엇으로 그렸는지 프레임에 명시, prim 을 이름 부분 문자열로 숨기지 않기, 숨긴 목록은 매니페스트로 실측(W25).
- 🔴 **좌표 방향은 실행값(`t_robot`)으로 확정**, 도크스트링·파일명·그림 인상 금지. 기준 좌표 명시(W22·W25).
- 🔴 **cap 을 넘길 실행은 기동하지 않는다** — 새 revision 에서 결과 전에 cap 을 정한다(D492·W25).
- 검증 결과가 다음 결정을 바꾸지 못하면 돌리지 않는다(W22) · 워커 과제문에 기대 답 금지, 원자료로 재계산(W22~W25) · 손-눈은 RMSE 만으로 합격 금지(D497).
- (W21 유지) step 경계 미도달 시 소프트 상한 미발동 → 바깥 watchdog · 축소 플래그는 fail-closed 검사 우회 안 함.

## 과거 완료 결과 — 유지

- W11 dt 1 µs 517알/10.4731 g · W16 run_02 기준선 11,021.63 s · W19 A 물리 24.807 s 완주(러너 20,977.836 s), 배출 272~327알(정착 미증명) · W21 P2 n=5,000 `CELL_UNRUNNABLE`(D493) · 재닫기 문 각도 2.880/3.054/3.549°(W16/W19A/W13) · 10 µs 폭주 = 감쇠 적분 불안정(W24).

## 실물 종료 정본 / 유지할 관찰

- `claudedocs/session_20260911_hardware_closeout_next_sim.md`; HOME [0,0,90,0,0,0]. S1 v1 5/5 사이클 93 s(D481), 배출 19.95 g 1건(원장 :585). 9/18 사용자 관찰: 운반 중 유출 ≈0~1알. 실물 카메라 **미연결**.

## 신뢰하지 않을 과거 상태 / 환경

- CONTINUE_2026091x·20260923 계열 GO/COMMANDS, HANDOFF.md/TASKS.md. RunPod(09-28 23:3x 읽기 전용 확인): **우리 pod 0**, 랩 계정 타인 pod 7개(RUNNING 1 — 조작 금지), 최근 14일 랩 과금 295.57 $.
- `w13_kinematics.py:10-16` "(0,+R)" 좌표 도크스트링(낡음) · post04 주석 "순정 그리퍼 숨김"(실제는 S1 문 CAD) · 9/7 "베이스판 38 cm"(현재 33.2) · 8/31 `roarm_base` 격자(D496 개정, 코드 미반영).
- isaaclab numpy 1.26.0·psutil 5.9.8·Rerun 0.34.1·DEME 2.4.0 유지. Codex 0.158.0(= npm 최신, 업데이트 안내 없음). Codex 모델 = `gpt-5.5`/`gpt-5.6-sol`/`gpt-6-astra`.
