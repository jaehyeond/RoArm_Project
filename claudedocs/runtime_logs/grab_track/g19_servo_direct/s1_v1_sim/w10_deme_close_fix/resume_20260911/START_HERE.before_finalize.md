# START_HERE.md

Last updated: 2026-09-11 — 80th~81st(09-09~11): **sim↔실물 정합 트랙 + 펠릿 실측 + 서보 PID 조사 + 문헌·산업 조사**. 79th(S1 그랩 실물 확립, D481 `:30140`) 위에 얹은 세션이며 **원장 등재는 아직 안 했다(§원장 상태)**.

🟢 **재부팅 후 GPU 복구(09-11 13:13 KST).** 호스트 `nvidia-smi`·커널 모듈 모두 580.178.04, torch CUDA 연산 성공. 샌드박스 내부 GPU 접근은 실패하므로 호스트 실행 필요. **W10 ③' dt-only 실행 중**, 결과 대기. 최신 세션: `claudedocs/session_20260911_w10_reboot_resume.md`.

🟢 이번 세션이 세운 것: 실물 형상 v1 USD(가짜 질량 제거) · 실물 환경 Isaac 재생 · 펠릿 3축 실측 → 렌즈형 입자 모델 · DEME 로 실물 절차 퍼내기 · **Isaac 에서 로봇+DEME 입자 퍼내기 영상**.
🔴 미해결: 렌즈 더미에서 문을 **토크 정지까지** 닫으면 DEME 발산(W10, 재개 지시서 작성됨).

## 🔴 지금 상태

정본 형상 = `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1/`(79th 불변).
**신규 자산** = `local_assets/roarm_m3/{urdf/roarm_m3_s1_v1.urdf, usd_s1_v1/}` — 실물 v1 형상 + `hand_tcp` 가짜 1 kg 제거(임포터 기본값). 어깨 중력 모멘트 11~14배 과대 → 정상화, 처짐 1.5°→0.11°, **팔 토크 1.96 N·m 로 구 파지 성공**(D478 "8.0 필요" 는 가짜 질량 탓).
**펠릿 실측**(사용자, 09-10) = 긴쪽 4.5 · 폭 3.8 · 두께 2.5 mm(20알 줄·10알 포개기), 편평비 **b/c 1.52 = 렌즈형**, 알 부피 22.4 mm³ · 질량 0.020 g(ρ 905 가정). 기록 `claudedocs/runtime_logs/pellet_model/pellet_measured_20260910.json`.
**입자 엔진 확정** = **DEME 단독**(PhysX PBD 는 D457·D469 로 기각 — 안식각이 dt 의 함수). Isaac 은 로봇·환경·렌더 담당. 연결 = DEME 물리 → Isaac 재생(W9 검증 완료).

## 🔴 다음 세션이 할 일 (순서 고정)

```
0. GPU        ✅ 재부팅 후 호스트 NVML·CUDA 연산 확인(580.178.04 일치)
1. W10        렌즈 더미 "토크 정지까지 닫기" 발산 해결 — 지시서 한 장에 전부:
              claudedocs/runtime_logs/grab_track/.../w10_deme_close_fix/RESUME_W10_20260911.md
              실행 중 = ③' dt 2e-6 만(E 유지). `run_w10b.sh DE_dt2e6_c 14400` (13:13:04 KST 시작)
2. 로봇       체크리스트(사용자 합의 09-10): 펌웨어·tG 확인 → PID 행동실험(P 8↔48) →
              닫힘 상한 900 vs 790 되열림 → `weigh 5` 회당 질량. auto-memory 체크리스트 참조
3. 실물 각도  부은 각(판 ≥30×30) · 렛지 각(칸막이 뽑기) · 한 입 뒤 절단면 각(구덩이 사진)
              → 차이 ≤ 3° 면 부은 각만으로 충분(R1 E1). 이 값으로 μ·Crr 피팅 → 더미 재정착
4. 결정층     베이스라인을 산업 규칙(최고점 + 층·열 진행)으로 교체, 실패 정의 산업 목록 채택
5. 원장       W10 결론 후 Dxxx·LEDGER·session_*.md 몰아서 등재
```

## 🔬 80th~81st 가 번 것 (원장 미등재 — 등재 시 Dxxx 로)

- **Isaac URDF 임포터는 inertial 없는 링크에 1.0 kg 을 준다** — D478 의 비물리 토크 8.0 의 진짜 원인. v1 경로는 미소 inertial 주입으로 해결.
- **DEME 는 실행 간 비결정**(같은 입력·무수정 코드로 포획 273/301/315) → 회귀 게이트를 "개수 동일" 로 두면 안 된다.
- **부은 더미 안식각 단독 보정은 답이 여러 개**(2026년 2편) → 렛지 각·절단면 각 병행. 우리 예측 대상은 안식각이 아니라 **절단면 각**.
- **산업 초기 상태 = 벽까지 가득 채워 평평**, 결정 규칙 = 최고점 + 층·열. ETH+Liebherr 2025: 최고점 휴리스틱은 **잡음 없어도** RL 보다 못하다(충진율 57.7 vs 62.7 %). → 우리 베이스라인은 임의 greedy 가 아니라 이 산업 휴리스틱.
- **T:107 은 SRAM 48번(휘발)**, 부팅 후 그리퍼 상한은 1000 이 아니라 **300**(소스 0.84). 서보 과부하 보호(80 % 초과 2 s → 20 %)가 되열림 후보 원인.
- **렌즈 클럼프는 좁은 틈에서 수치 불안정**(구–구 진동 자기증폭, `c·dt/I ≈ 1.5`). 강성↑ 은 역효과, dt↓ 가 손잡이.

## Active Case — single source of truth
- **Active: `scoop_v0` — sim↔실물 정합(`s1_v1_sim/`, `pellet_model/`)**. 신규 변수: 렌즈형 입자 모델(b/c 1.52)·실측 치수·문 폐합 수치 안정.
- **현재 실행**: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/cell_DE_dt2e6_c/`, 관측·완결 기록 `.../w10_deme_close_fix/resume_20260911/`. 이번 재개 신규 변수는 dt만. Codex가 원장 기록 소유.
- 확정 불변: S1 형상(입 58, 보울 폭 36.4, 서보 0~30°) · PP 펠릿 · **DEME 단독** · 배출 위치 고정 · 로봇 동작은 정해진 경로(학습 대상 아님).
- 학습 대상 = **어디를 퍼는가(높이맵 → 양·남는 형상 예측 → 선택)** 뿐. 관절·그리퍼 제어는 학습 안 함.
- 완료·동결: g18 전체, s1_v0, `y3_d455`, grasp track, W1~W9.

## Current verified truth
```
자산     usd_s1_v1(실물 v1·질량 정상) · 환경 재생 G1~G4 PASS(상자 최소거리 25.6 mm, 립 오차 0.64 mm)
입자     렌즈 클럼프 lens6(구 7개, 실측 4.5·3.8·2.5) · 안식각 표 μ×Crr 9셀(21.6~39.0°) · 더미 20,000알 정착 800 s
퍼내기   구 더미(4.16 mm): 완주, 회당 11.8 g · 렌즈 더미: 토크 정지까지 닫으면 4/4 발산, 문 하한 5° 로만 완주(3.12 g, 유출값)
렌더     W9 Isaac+DEME 53프레임 mp4·strip, 게이트 전부 PASS(립 오차 평균 0.64 mm, 포획 154/154 보울 안)
조사     R1 문헌 40건 · R2 산업 40건 → claudedocs/research/survey_20260910/
도구     hw_s1_manual.py(+weigh/mass) · s1_cycle.sh · run_w10b.sh · sim_isaac_render_deme_scoop.py
```

## Open risks / do-not-repeat
- 🔴 GPU 작업은 호스트 실행 필요(샌드박스 NVML/CUDA 접근 실패). 🔴 원장은 배타 소유 — 현재 Codex, W10 결론 전까지 다른 도구가 쓰지 말 것. 🔴 커밋은 사용자 요청 시만(미커밋 다수).
- 🔴 높이 추정 금지(줄자 실측) · 손목 90 초과 금지 · 맨 `{"T":106}` 금지 · 그리퍼 관절에 I 켜지 말 것.
- ⚠️ Orca 바인딩이 조용히 풀려 워커 질문을 놓칠 수 있다 → 확인 전 `run-current`. ⚠️ 근접 RTX 검정면(D480) 미해결.

## Next concrete action
**1** 실행 중인 W10 dt-only 셀 감시 → 완주 시 원본 결과·비교표·Rerun 검수, 발산 시에만 사전 지정 강성 셀. **2·3** 로봇 체크리스트·실물 각도는 별도 단계.
원장: `DECISIONS.md` 30150줄(D481 `:30140`, 그 뒤 append 없음) · `EXPERIMENT_LEDGER.md` 표 끝 `:544`.

## Must read first
1. `AGENTS.md` · 2. **`claudedocs/relay/from_claude.md` §2**(이번 인계 전문) · 3. **`.../w10_deme_close_fix/RESUME_W10_20260911.md`**(W10 재개) · 4. `claudedocs/DECISIONS_ACTIVE.md` §5·§7 · 5. `docs/reference/servo_pid_st3215.md`(서보 PID 만지기 전) · 6. `claudedocs/research/survey_20260910/R{1,2}*.md`(연구 설계 근거).

## Do not trust as current
- `HANDOFF.md`, `TASKS.md`. 🔴 **`local_assets/roarm_m3/usd_s1/`(v0 형상 + `hand_tcp` 1 kg)** — v1 로 대체. 🔴 **D478 "팔 토크 8.0 N·m 필요"** — 가짜 질량 탓, 실물 1.96 으로 파지됨. 🔴 **"T:107 이 EPROM" · "부팅 후 그리퍼 상한 1000"** — 정정됨(SRAM 48, 부팅 300). 🔴 **펠릿 = 쌀알형·긴쪽 3.8** — 실측은 렌즈형 4.5×3.8×2.5. 🔴 **W8 의 "유령 접촉" 가설** — W10 진단이 반증(구–구 진동). 🔴 s1_v0·g18 이하 동결.
