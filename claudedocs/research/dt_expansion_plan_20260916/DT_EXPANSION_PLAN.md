# dt 확대 비교(1 µs → 1 ms) 별도 실행안 — 승인 대기, 미실행

작성: 2026-09-16 (W14 raw repair 세션). 이번 case의 신규 변수: [] — **이 문서는 제안서이며 새 물리·GPU·렌더를 실행하지 않았다.** 아래 표의 시간값은 기존 원자료 재읽기와 CPU 산술이다. 실행은 §7의 승인 항목을 사용자가 명시 승인한 뒤 별도 case(가칭 W15)로 시작한다.

## 1. 질문과 두 갈래

교수님 제안 = "시간 간격(dt)을 10 µs, 100 µs, 1 ms로 키우면 안정성·정확도·계산비용이 어떻게 되나". 이는 두 질문으로 갈라야 한다.

| 갈래 | dt | 질문 | 제어/진단 간격 |
|---|---|---|---|
| **A. 동일 프로토콜 민감도** | 1 / 2 / 10 µs | 같은 제어·보호선·진단 간격에서 결과(포획량·정지 원인·최대속도)가 dt에 얼마나 민감한가 | 요청 4 ms / 1 ms / 0.1 ms 그대로, 실제 지속시간은 +1 step 이내 |
| **B. 호환성 한계** | 100 µs / 1 ms | 현재 프로토콜(0.1 ms 세밀 sync)이 유지되지 않는 dt에서 무엇이 먼저 깨지는가 | **유지 불가**: 0.1 ms 요청이 0.2 ms / 1.0 ms가 됨(§3) |

A는 "dt 효과"이고 B는 "제어 간격까지 함께 바뀌는 한계 시험"이다. 둘을 한 표에 섞어 "dt만의 효과"라고 말하지 않는다.

## 2. 기존 증거 대조 — 같은 프로토콜의 세 셀

세 셀은 **같은 코드·같은 초기 NPZ·같은 제어/보호선**이고 params 차이는 `timestep_s`와 렌더 경로뿐임을 다시 확인했다(`PARAMETER_AUDIT.json` `historic_10us_to_2us_config_diff`, `w10_w11_effective_param_diff`; 이번 세션 `params_w10_DE_dt2e6_c.json` vs `params_w11_dt1e6.json` diff 2줄).

| 셀 | dt | 결과 | 물리시간 | 물리 벽시계 | 저장 최대속도 | 정지 원인 | 포획 |
|---|---:|---|---:|---:|---:|---|---:|
| W10 `cell_DE_c` | 10 µs | **rc 134, 닫힘 중 발산** (엔진 예외 22,291.97 m/s > 10,000) | 마지막 저장 2.56118 s | 2,130 s (중단까지) | 13.0303 m/s (저장 sync) | 없음(중단) | 없음 |
| W10 `cell_DE_dt2e6_c` | 2 µs | 완주 rc 0 | 3.144948 s / 2,774 sync | 1,870.9 s | 5.3291 m/s, >5 m/s 1회 | close 3.307° servo_stall / reclose 3.069° servo_stall | 541알 / 10.9592 g |
| W11 `cell_dt1e6_seed460` | 1 µs | 완주 rc 0 | 3.141871 s / 2,771 sync | 2,720.74 s | 2.7255 m/s, >5 m/s 0회 | close 3.494° servo_stall / reclose 3.076° **pinch_guard** | 517알 / 10.4731 g |

근거: `w10_deme_close_fix/run.log:23`(rc 134, wall 2130), `cell_DE_c/timeline_seed460.json`(2,158행, 마지막 sim_t 2.56118, v_particle_max 최대 13.0303), `cell_DE_c/stderr.txt`(엔진 예외), `w11_dt_sensitivity_20260911/comparison.json`(baseline/new/delta).

읽는 법:
- 10 µs는 "처음 시도하는 조건"이 아니라 **이미 같은 프로토콜에서 실패한 조건**이다. 재시험의 목적은 "실패 재현성"(DEME는 실행 간 비결정 — relay §4.2, 포획 273/301/315 사례)이지 성공 기대가 아니다.
- 2 µs→1 µs는 각 1회라 24알(4.44 %) 차이를 dt 효과로 단정할 수 없다. 재닫기 정지 원인이 servo_stall→pinch_guard로 바뀐 것도 1회 관측이다.
- **10 µs 셀은 발산 전까지도 벽시계가 빠르지 않았다**: 2,130 s / 2.56 s ≈ 832 s/s (2 µs: 595 s/s, 1 µs: 866 s/s). 큰 dt가 곧 시간 단축이라는 예측은 이 데이터로 뒷받침되지 않는다(고속 입자가 접촉 탐색 마진을 키운 탓일 가능성; 확정 아님).
- W13 전체 사이클(24.49 s 물리에 31,203 s 벽시계, 0.1 ms sync 7,505회·1 ms 3,824회·4 ms 4,974회)은 domain·유한벽·전체 경로가 달라 이 비교군이 아니다. W13 원자료의 sync당 벽시계 중앙값은 0.1 ms→0.174 s, 1 ms→1.24 s, 4 ms→4.88 s로 내부 step 수에 거의 비례한다(step당 약 1.2 ms). 즉 이 규모(20,000알·7구)에서 sync 고정비는 작고 step 수가 비용을 지배한다 — dt를 키우면 **안정하기만 하면** step 수는 준다.

## 3. 100 µs / 1 ms에서 실제 제어·진단 간격이 어떻게 바뀌나 (소스 기반 산술, 새 실측 아님)

설치 DEME 2.4.0은 요청 D를 정수 나눗셈하지 않고 **float32 dt를 float64로 반복 누산해 D 이상이 되는 순간**까지 돈다(감사 `DEME_SOURCE_BOUND.md`, dT.cpp.o 디스어셈블리). `compute_sync_table.py`로 재현한 표(`sync_duration_table.json`):

| 요청 sync | dt 1 µs | 2 µs | 10 µs | **100 µs** | **1 ms** |
|---|---:|---:|---:|---:|---:|
| 4 ms (하강·이송 등) | 4.001 ms / 4001 step | 4.002 / 2001 | 4.010 / 401 | 4.100 / 41 | 4.000 / 4 |
| 1 ms (닫힘) | 1.001 / 1001 | 1.002 / 501 | 1.010 / 101 | **1.100 / 11** | 1.000 / 1 |
| 0.1 ms (문 6° 미만 세밀 구간) | 0.101 / 101 | 0.102 / 51 | 0.110 / 11 | **0.200 / 2** | **1.000 / 1** |
| 세밀 구간 한 sync당 문 이동(22.5 °/s) | 0.00227° | 0.00229° | 0.00247° | **0.00450°** | **0.02250°** |

- float32(1e-4) = 9.9999997e-5 < 1e-4 이므로 100 µs에서는 0.1 ms 요청이 **2 step = 0.2 ms**가 된다(요청 대비 2배). float32(1e-3) = 1.0000000475e-3 ≥ 1e-3 이라 1 ms에서는 1 step.
- 1 ms에서는 0.1 ms 뒤 상태를 볼 수 없다. 물림 가드(3 N)·서보 정지(토크 상한)·pop 경고(5 m/s)·pop-stop(20 m/s)의 **확인 간격이 10배 성글어지고** 확인 사이에 문이 0.0225° 움직인다. "명목 params가 같으니 같은 제어"라고 말할 수 없다.
- **접촉 탐색 간격도 함께 커진다**: `SetCDUpdateFreq(20)`은 "kT 접촉쌍 갱신을 기다리기 전 dT step 수"다(설치 `include/DEM/API.h:106-109`, SHA `67be492d…`). step 수 고정이면 탐색 간격은 dt 1 µs의 20 µs에서 1 ms의 **20 ms**가 된다. `SetMaxVelocity(5)`는 마진 두께 계산에 쓰는 속도 상한이다(`API.h:207-210`). 단, 설치본에는 갱신 빈도 자동 조정(`SetCDMaxUpdateFreq`, `API.h:299-303` "when it is adjusted automatically")이 있어 **실제 런타임 빈도는 고정값이라고 단정하지 않는다** — 실험에서는 매 sync `GetUpdateFreq()`/`GetExpandFactor()`(`API.h:104-105, 313-314`)를 기록해 실제값을 남긴다. 이 항목을 사전 예측치로 적지 않는다.
- 따라서 B 갈래의 가능한 결과는 셋이다: ① 수치 폭주(20 m/s python 중단 또는 엔진 예외) ② 관통/끼임으로 보호선이 늦게 걸림(정지각·접촉력 변화) ③ 완주하되 결과가 다름. 셋 모두 "결과"이며 어느 것도 사후 합격선으로 바꾸지 않는다. `cd_update_freq`를 dt에 맞춰 줄이는 것은 **두 번째 변수**이므로 이 실행안에 넣지 않는다(§8 후보).

## 4. 고정하는 것 (한 번에 바꾸지 않는다)

| 항목 | 값 | 출처 |
|---|---|---|
| 초기 상태 | `pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz` SHA `659d6b0b…8812`, seed 460, 20,000알, 7구 템플릿 질량 20.257 mg | W10/W11/W13 공통 |
| 물성 | E 5 MPa, ν 0.30, CoR 0.30, μ 0.45, Crr 0.06, ρ 905, mesh E 3 GPa | `params_w11_dt1e6.json` |
| 공구/문 형상·경로 | S1 v1 STL, plunge 25 mm @ 25 mm/s, 닫힘 22.5 °/s, lift, reclose | 동일 |
| 제어 요청 간격 | 4 ms / 닫힘 1 ms / 6° 미만 0.1 ms (**명목값 고정**, 실제값은 §3처럼 dt가 결정) | 동일 |
| 보호선 | pinch 3 N, servo stall(1.96 N·m×0.9), pop 경고 5 m/s(>), python pop-stop 20 m/s(>), `error_out_vel` 1e4, `SetMaxVelocity` 5, `cd_update_freq` 20 | 동일, 사후 완화 금지 |
| 코드 | 메인 `sim_deme_scoop_s1.py` SHA `2e40f7ed…8933`(W11 실행본과 동일), 입력 6파일 해시 = W11 `preflight.json` | 실행 전 재대조 |
| **유일 변수** | `timestep_s` ∈ {1e-5, 1e-4, 1e-3} (+ 선택 {1e-6, 2e-6} 반복) | 셀당 params 파일 1개, diff는 timestep·출력경로 2줄이어야 함 |

## 5. 셀·구간·지표·중단조건

**구간**: W10/W11 스쿱 프로토콜(settle→descend→close→lift→reclose, 물리 약 3.14 s). 닫힘·재닫기(dt에 가장 민감한 구간)를 포함하고 W13처럼 9시간이 들지 않는다. 같은 **물리 구간**을 비교하며, 같은 벽시계를 돌려 다른 진행도를 비교하지 않는다.

**셀 (순차 실행, 동시 실행 금지 — run.log 00:06:27 교훈)**:

| 순서 | 셀 | dt | 목적 | 벽시계 상한 | 비고 |
|---|---|---:|---|---:|---|
| 1 | `cell_dt1e5_seed460` | 10 µs | 실패 재현성 (A) | 5,400 s | 실패해도 마지막 유효 시각·이유·이벤트 덤프 보존 |
| 2 | `cell_dt1e4_seed460` | 100 µs | 호환성 한계 (B) | 3,600 s | 0.1 ms 요청→0.2 ms 실제 |
| 3 | `cell_dt1e3_seed460` | 1 ms | 호환성 한계 (B) | 3,600 s | 0.1 ms 요청→1.0 ms 실제 |
| (선택 4·5) | `cell_dt1e6_rep2`, `cell_dt2e6_rep2` | 1 µs, 2 µs | 실행 간 변동 폭(A) | 4,000 / 3,000 s | 별도 승인; 기존 1회씩과 합쳐 각 2회 |

**셀당 기록(원자료 JSON/NPZ, 기존 `sim_deme_scoop_s1.py` 출력 그대로)**: rc·완주 여부·`diverged`·중단 사유와 마지막 유효 sim_t; sync별 sim_t(→ 실제 sync 지속시간 재계산)·phase·문 각도·최대속도·접촉 수/힘; >5 m/s sync 수; 정지 이벤트(phase, q, 원인, 토크, 최대 단일 접촉력); 포획 알 수/질량(공구 공동 기하식, 저장 좌표 기준); 립 간격; 물리 루프 벽시계·단계별 step 수; `GetUpdateFreq()`/`GetExpandFactor()`는 현재 스크립트가 기록하지 않으므로 **로그 추가가 필요하면 별도 승인**(코드 변경 = 새 변수로 취급, 물리 불변이어도 명시).

**지표 비교표(사후 합격선 없음)**: 완주/중단, 최대속도, 정지각·원인, 포획 수/질량, 벽시계, step 수, 실제 sync 간격 통계(요청 대비 비율), 사후 높이맵 MAE(기존 `analysis_w11.py` 방식). DEME 비결정성 때문에 "개수 동일"을 게이트로 쓰지 않는다(±15 % 관찰 밴드는 서술용).

**중단조건(기존 그대로)**: python pop-stop 20 m/s → pop_event NPZ 저장 후 RuntimeError; 엔진 `error_out_vel` 1e4 초과 → rc 134(core dump 가능, stderr 보존); 셀 벽시계 상한 → TERM(정리 유예 포함); 무진행 300 s(stdout·timeline mtime) → kill 후 partial 보존. 어떤 경우도 자동 재시도 없음.

**시각 검수(D324/D341)**: 완주 셀은 W11과 같이 Rerun RRD/RBL·검증 JSON·결정 PNG·실제 육안 검수 기록을 남긴다(`w11_rerun_export.py`, `verify_rrd_coverage.py` 재사용, 셀당 ≤ 600 s 예산). 중단 셀은 이벤트 덤프 + 마지막 timeline만 남기고 RRD 생략 사유를 기록한다.

## 6. 예산·환경·출력 경로

- 사전 점검: `nvidia-smi` 정상, GPU free ≥ 3,072 MiB, 다른 GPU 작업 0, 디스크 여유(셀당 ≤ 1 GB, W11 셀 약 220 MB + RRD 146 MB), isaaclab env 무수정(numpy 1.26.0/psutil 5.9.8), 설치 0.
- 벽시계 총상한(1~3번 필수 셀): 물리 5,400+3,600+3,600 = 12,600 s + Rerun/분석 3×600 s + 정리 1,200 s = **≤ 15,600 s(4시간 20분)**. 선택 4·5 추가 시 +7,000 s + 1,200 s.
- 출력: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w15_dt_ladder_<YYYYMMDD>/` (W10/W11과 같은 계보). 구성: `PREREG.md`(이 §4·§5 확정본), `params_w15_dt<val>.json`, `launch_dt<val>.json`, `cell_dt<val>_seed460/`, `comparison.json`, `REPORT_w15.md`, `w15_workflow.py`(W11 `w11_workflow.py`의 prepare/run/analyze 골격 — 기존 폴더가 있으면 거부, 자동 재시도 없음). **`run_w10b.sh` 재사용 금지**(덮어쓰기 위험, START_HERE).
- 실행 명령(승인 후 사용할 형식, W11 `launch.json`과 동일 골격):
  `/home/cgxr/miniconda3/envs/roarm/bin/python -u /home/cgxr/Documents/Robotics/RoArm_Project/sim_deme_scoop_s1.py --params <params_w15_dt<val>.json> --pile <pile NPZ> --out <cell dir> --seed 460`
  (`timeout`/무진행 감시는 workflow가 감싼다.)

## 7. 승인이 필요한 항목 (이 문서로는 아무것도 실행되지 않는다)

1. 셀 구성: 필수 1~3(10 µs / 100 µs / 1 ms) 실행 여부, 선택 4·5(1 µs·2 µs 반복) 포함 여부.
2. 벽시계 상한과 총예산(§6) 및 실행 시간대(GPU 독점 4시간 20분).
3. 출력 경로 `w15_dt_ladder_<YYYYMMDD>/` 신설.
4. `GetUpdateFreq()`/`GetExpandFactor()` 로그 추가 여부(코드 변경, 물리 불변이지만 별도 명시).
5. 완주 셀의 Rerun 렌더(GPU 아님·CPU) 실행 여부와 예산.

승인 없이 하지 않는 것: 새 DEME/Isaac GPU 실행, 재렌더, `cd_update_freq`·E·형상·알 수·경로·문 속도·보호선 변경, 학습, PBD 하이브리드, A/B/C, 실물 조회/구동, 설치, commit/push.

## 8. 이 실행안에 넣지 않은 후속 후보 (BACKLOG 성격)

- `cd_update_freq`를 dt에 맞춰 스케일(탐색 간격 20 µs 유지)한 B' 셀 — 두 번째 변수라 분리.
- 세밀 sync 요청을 dt에 맞춘 프로토콜 변형(1 ms에서 제어 간격 명시 확대) — 질문이 바뀜.
- 7구 형상 수렴/실물 물성 보정 — dt와 분리(교수 질의 보고서 §3.2).
- W13 전체 사이클에서의 dt 비교 — 비용(9 h/사이클)과 domain 차이로 이번 범위 밖.

## 근거 파일

- 과거 10 µs: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/{params_w10_DE_c.json,run.log,cell_DE_c/timeline_seed460.json,cell_DE_c/stderr.txt}`
- 2 µs/1 µs: `.../w10_deme_close_fix/cell_DE_dt2e6_c/`, `.../w11_dt_sensitivity_20260911/{comparison.json,REPORT_w11.md,launch.json,params_w11_dt1e6.json,w11_workflow.py}`
- 누산 규칙: `/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/DEME_SOURCE_BOUND.md`; 이 폴더 `compute_sync_table.py` → `sync_duration_table.json`
- 교수 질의 보고서: `claudedocs/research/professor_review_20260916/{REPORT.md,PARAMETER_AUDIT.json}` (`source_derived_sync_durations`, `historic_10us`)
- W13 sync 비용: run_01 NPZ `sync_wall_elapsed_s`/`sync_requested_duration_s`/`sync_derived_internal_steps` (이번 세션 CPU 집계, `claudedocs/session_20260916_w14_raw_repair_dt_plan.md`)
- DEME API: `/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/include/DEM/API.h` SHA `67be492d974a5adad577e53f078c426b79981abe50c1525cb6dce632b1e10ef0` (설치 2.4.0; 공식 main과 버전 불일치 가능)
