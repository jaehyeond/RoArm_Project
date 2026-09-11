# REPORT — W8: 렌즈형 실측 펠릿 더미(W7, 20,000알)에 S1 그랩 실물 절차 퍼내기 — 회당 포획 질량 + 구덩이 형상·절단면 각

실행 2026-09-10 · W8 워커 · 메인 워크트리(`~/Documents/Robotics/RoArm_Project`, HEAD `d41c258`, **커밋 없음**) · 로봇 미접촉 · 상태 원장 무수정 · 새 의존성 0
산출 폴더 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens/` · 사전 등록 게이트 `GATES_w8.md` · 판정 원문 `gates_w8.json`(`gates_w8_writer.py` 가 결과 JSON 만 읽어 씀)
코드 변경 = `sim_deme_scoop_s1.py` 1개(백업 `sim_deme_scoop_s1.py.bak_20260910_pre_w8`, diff 원문 `sim_deme_scoop_s1_w8.diff`). `sim_deme_scoop.py`(보호)·기존 산출·기존 npz 무수정.

**이번 case 의 신규 변수: [더미 1개 — 구 4.16 mm 능선 더미(18,796구) → W7 실측 렌즈 7구 클럼프 언덕 더미(20,000알 = 구 140,000행)].**
절차(D481)·서보 정지 모델·궤적·적분 설정은 W3b 그대로. 물성(mu 0.45·Crr 0.06·ρ 905)은 새 변수가 아니라 W7 더미 metadata 의 **임시값(MEASURE)** 을 그대로 옮긴 것이다.

> **용어 (처음 나올 때 한 줄씩)**
> *S1 그랩* = 고정 반쪽 보울 + 서보 직결 문(D480/D481). *립(lip)* = 보울 입 가장자리, 시뮬에서 툴 원점.
> *잠김(plunge)* = 립을 펠릿면 아래로 밀어 넣는 깊이(실물 2.5 cm). *서보 정지 모델* = 힌지 저항 모멘트 ≥ 1.96×0.9 = 1.764 N·m 이면 문이 그 각에서 멈추는 규칙.
> *클럼프(clump)* = 구 여러 개를 한 강체로 붙인 알갱이. *lens6* = 중심 1 + 링 6 = 7구로 실측 렌즈(4.5·3.8·2.5 mm)를 흉내 낸 W7 템플릿.
> *heightmap* = 5 mm 격자 셀마다 "셀 발자국 안 가장 높은 표면점"(`roarm_rl.heightmap` 경로 A, 구별 반경 정확식).
> *절단면 각(cut-face angle)* = 한 입 퍼낸 뒤 남은 구덩이 옆면의 경사각(R1 §5). 이 보고서에서는 코드 `crater_angles()` 정의로 잰 값(§4 정의).
> *충전율* = 포획 질량 ÷ (보울 공동 45.7 cm³ × 벌크 밀도 0.55 g/cm³). *DEME 비결정* = 같은 입력이라도 GPU 스레드 타이밍 때문에 run-to-run 결과가 bit 단위로 같지 않은 엔진 특성(W7 §7-7).

---
## 0. 한 줄 결론

> **사전 등록 설정으로는 렌즈 클럼프 더미를 퍼내지 못했다.** 실물 절차(잠김 25 mm, 문 45°/s, 서보 토크 정지) 그대로 돌리면 문이 마지막 몇 도(관절 2.3~3.6°, 립 틈 4.6~7.2 mm)에 이를 때 DEME 가 입자 속도 81~3664 m/s 로 강제 종료된다 — 셀 c·xp50, 그리고 원인 후보를 하나씩 바꾼 옵션 A(폐합 sync 1 ms)·옵션 E(+물림 가드 3 N) 까지 **4/4 발산**(§3-B).
> 기작은 서서히 눌리는 물림이 아니라 **한 sync 안에 갑자기 켜지는 유령 접촉**이다(발산 직전 10 ms 까지 단일 접촉력 0.5~1.4 N, 힌지 모멘트 0.6~1.1 N·m 로 정지 문턱 1.764 미달). 렌즈 구 반지름(1.16 mm)이 립 두께(4.6 mm)의 1/4 이라, 구 중심이 립 면 뒤 r 안에 들어간 채 발자국 안으로 들어오는 순간 DEME 양면 삼각형 접촉이 관입 r+|h| 로 잡는 W3b 와 같은 커널 성질이 4.16 mm 구보다 훨씬 쉽게 걸린다.
> **문 하한각 5.0°(립 틈 9.9 mm)를 둔 옵션 F 로만 완주했다**(§3-C, 사전 등록 외 설정): 중앙 셀 포획 **154알 = 3.12 g**(충전율 0.124, 구 더미 W3b 11.8 g 의 26 %) — 문이 토크 정지가 아니라 하한에서 강제로 멈춰 렌즈 길이 2.2배 틈으로 상승 중 유출된 값이라 **실물 회당 질량의 추정치로 쓰면 안 된다**. 퍼낸 뒤 구덩이(제거 36.5 cm³, 최대 깊이 14.7 mm)와 절단면 각(+x 16.3° / +y 38.1° / −x 13.2° / −y 가파름·격자 미해상, 시컨트 30°)은 보고값이다.
> 구 더미 회귀는 사전 등록 "포획 개수 동일" 기준 **FAIL(273 vs W3b 315)** 이나 원인은 코드가 아니라 DEME run-to-run 비결정이다(무수정 백업판 재실행 301, 타임라인이 재안착 2행부터 갈라짐, §3-A). D341 Rerun 1셀·D470 sha·렌더 타임라인(코디네이터 추가 요청)은 완료.

---

## 1. 무엇을 · 왜

W7 이 실측 렌즈 알(4.5·3.8·2.5 mm, 7구 클럼프, 20.26 mg) 20,000 알로 정착시킨 더미를, W3b 가 구 더미에서 완주시킨 S1 그랩 퍼내기 시뮬(`sim_deme_scoop_s1.py`)에 처음 넣어 본다. 목표는 ① 회당 포획 질량(구 더미 11.8 g 과 비교 가능한 값), ② 퍼낸 뒤 남는 구덩이 형상과 옆면 각(R1 §5·§7 의 "한 입 뒤 절단면 각" 을 시뮬에서 재는 방법 확립) 이다. 프로포절 과제 ①("한 위치를 퍼냈을 때 양 + 남을 형상 예측") 의 시뮬 쪽 데이터가 여기서 나와야 한다.

## 2. 절차 (실행 순서대로, 관찰 가능한 단계)

| 단계 | 한 일 | 확인한 것 | 원문 |
|---|---|---|---|
| ① 조사 | W3b 보고 §0~§3, W7 npz 계약(§4), R1 §5·§7, `sim_deme_scoop_s1.py` 전체, `roarm_rl.heightmap` API, isaaclab rerun 0.34.1 핀 | 메인 워크트리 `sim_pellet_model.py` 는 lens 필드가 없어 import 불가 → npz 의 `clump_template_json` 만으로 재구성해야 함 | — |
| ② 사전 등록 | `GATES_w8.md` (G1~G5·R·S, 셀 위치, 설정, 절단면 각 정의) | 실행 전 고정 | `GATES_w8.md` |
| ③ 코드 | `particle_shape: "npz_template"` 경로, 구 전개(`GetOwnerOriQ`), 전/후 heightmap, `crater_angles()`, 렌더 타임라인, (발산 대응) `dt_sync_close_s`·`door_pinch_guard_N`·`door_min_q_deg`·진단 flush — 전부 params 게이트, 기본값이면 옛 경로 | `py_compile`, 단위 검사: 전개식 vs npz 구 행 max 2.8e-17 m, 합성 35° 원뿔 → 36.0° | `sim_deme_scoop_s1_w8.diff` |
| ④ 구 회귀 | 구 더미 seed 460 + W3b params 그대로 (패치판) · 무수정 백업판 재실행 | 273 / 301 vs W3b 315 → DEME 비결정 (§3-A) | `regression_sphere/`, `regression_sphere_prew8_script/` |
| ⑤ 렌즈 셀 (사전 등록) | c(0,0) → xp50(+50,0) | 둘 다 폐합 끝 발산 (81 / 203 m/s) | `cell_c/`, `cell_xp50/` |
| ⑥ 원인 격리 | 옵션 A: 폐합 sync 4→1 ms · 옵션 E: +단일 접촉 3 N 가드 | 둘 다 발산 (226 / 3664 m/s), 가드 미발동 → 순간 유령 접촉 (§3-B) | `cell_A_c/`, `cell_E_c/` |
| ⑦ 완주 변형 | 옵션 F: +문 하한 5.0°, q<6° 매 sync 진단 | c 완주 1895 s, xp50·xm50 순차 | `cell_F_*/` |
| ⑧ 후처리 | 쉬는 입자만의 heightmap 으로 절단면 각 재계산(문 틈 유출 중 공중 알 63구 제외), 시컨트 보조각 | `crater_rest_seed460.json` | `crater_rest_writer.py` |
| ⑨ D341 Rerun | F_c → RRD/RBL/스크린샷/계약 검증 → 육안 검수(1차 w8 는 기본 커서에서 더미 미표시 → 정적 사본 추가한 w8v2 재검수) | `pass: true`, 검수 JSON | `cell_F_c/scoop_s1_seed460_w8v2_*` |
| ⑩ 판정·보고 | `gates_w8_writer.py` → `gates_w8.json`, 이 보고서 | — | — |

코디네이터 추가 요청(msg_73da043c4819, 렌더용 프레임별 궤적): 중앙 셀에서 0.05 s 마다 클럼프 pos/quat·툴/문 포즈·문 각·포획 id 를 `render_timeline_cell1.npz`(53 프레임, 27.6 MB, 프레임 규약은 `metadata_json`) 로 저장했다. 이 요청 때문에 사전 등록 c 1차 실행(10 분 진행)을 중단·재시작했다(`cell_c_try1_killed_for_render_timeline/`).
## 3. 수치 (출처 경로 포함)

### 3-A. 구 더미 회귀 (G2) — 엄격 판정 FAIL, 귀속 = DEME 비결정

같은 입력(구 더미 `pile_practical_fast_d4p16_n18796_seed460.npz` sha16 `e26b214f336a4e97`, W3b `params_fixnorm_plunge25.json`, STL·design.json 전부 sha 동일), seed 460.

| 실행 | 스크립트 | 포획 개 / g | 문 정지 close → reclose (관절°) | 립 틈 mm | 립 등가 피크 N | 벽시계 s | 원문 |
|---|---|---|---|---|---|---|---|
| W3b 원본 (09-10 10:3x) | W3b 판 (sha16 `db6e2112…`) | **315 / 11.28** | 3.595 → 2.696 (servo_stall ×2) | 5.40 | 22.7 | 91.7 | `b_diverge/scoop_fixnorm/scoop_s1_seed460.json` |
| W8 패치판 | npz_template·crater 추가판 (sha16 `38b38c6b…`) | **273 / 9.78** | 3.775 → 3.235 (servo_stall ×2) | 6.49 | 16.2 | 88.0 | `regression_sphere/scoop_s1_seed460.json` |
| **무수정 백업판 재실행** | `sim_deme_scoop_s1.py.bak_20260910_pre_w8` = W3b 판 바이트 동일 | **301 / 10.78** | 3.595 → 2.696 (servo_stall ×2) | 5.40 | 19.4 | 92.6 | `regression_sphere_prew8_script/scoop_s1_seed460.json` |
| W8 최종판 (렌더·옵션 A/E/F 게이트 추가 뒤) | sha16 `0e1d4dc032cf9aaf` | **300 / 10.74** | 3.595 → 3.235 (servo_stall ×2) | 6.49 | 22.2 | 87.4 | `regression_sphere_final_script/scoop_s1_seed460.json` (try1 은 상승 120 스텝에서 DEME stall → kill, `…_try1_stall/`) |

읽는 법: 사전 등록은 "포획 개수 동일" 이므로 **G2 = FAIL** 로 적는다. 그러나 W3b 와 바이트 동일한 스크립트를 다시 돌려도 315 가 아니라 301 이 나왔고, 패치판과 W3b 의 타임라인을 행별로 비교하면 **재안착 2행째(v_particle_max 0.0358 vs 0.0359)** 부터 갈라진다 — 그 구간은 `DoDynamicsThenSync` + 읽기만 있고 W8 코드가 개입하지 않는다. 즉 차이는 DEME GPU 의 run-to-run 비결정(W7 §7-7 에 기록된 엔진 한계, kT/dT 비동기)이며, 같은 입력의 포획 개수 산포가 최소 273~315(±7 %) 임을 이번에 처음 정량화했다. 구 경로 코드 diff(`sim_deme_scoop_s1_w8.diff`) 에서 지워진 20행은 전부 반환값에 `None` 추가·heightmap 블록 이동·print 에 벽시계 추가뿐이며 물리 호출 순서는 같다. **다음부터 회귀 게이트는 "seeded 입력 bit-동일 + 결과 허용 범위" 로 등록해야 한다**(W7 이 heightmap 5 mm 허용오차로 한 것과 같은 방식).

### 3-B. 렌즈 더미 — 사전 등록 설정 및 원인 격리 (G1 = FAIL)

모두 seed 460, 잠김 25 mm 도달(팔 정지 0), 하강 팔 힘 피크 ~1.5 N 로 하강은 정상. 죽는 자리는 전부 **폐합 끝**.

| 시도 | 바꾼 것 | 마지막 flush 행 (관절 q · 힌지 M · 단일 접촉 최대 · 입자 v_max) | DEME error-out 속도 | 벽시계 | 원문 |
|---|---|---|---|---|---|
| 사전 등록 c (0,0) | — (폐합 sync 4 ms, 토크 정지만) | q 3.42° · 0.88 N·m · 0.61 N · 0.26 m/s | **81 m/s** | 1495 s | `cell_c/timeline_seed460.json`, `stderr.txt` |
| 사전 등록 xp50 (+50,0) | — | q 2.34° · 1.12 N·m · 1.39 N · 0.58 m/s | **203 m/s** | 1543 s | `cell_xp50/…` |
| 옵션 A c | 폐합·재폐합 sync 1 ms (정지 판정 4배 촘촘) | q 3.64° · 0.56 N·m · 0.52 N · 0.45 m/s | **226 m/s** | 1554 s | `cell_A_c/…` |
| 옵션 E c | + 단일 접촉 ≥ 3 N 이면 정지(물림 가드) | q 3.19° · 0.75 N·m · 0.60 N · 0.43 m/s (가드 미발동) | **3664 m/s** | 1583 s | `cell_E_c/…` |

읽는 법(초보자용):
- 서보 정지 모델은 힌지 저항 모멘트가 1.764 N·m(립 힘 20.5 N) 에 닿아야 문을 멈춘다. 구 더미(W3b)에서는 q 3.59° 에서 모멘트가 한 sync 에 1.6→2.6 N·m 로 뛰어 멈췄다. 렌즈 더미에서는 같은 각에서 모멘트가 0.5~1.1 N·m 에 그쳐 문이 계속 닫히다가, 어느 sync 에서 갑자기 한 알이 수백 m/s 로 튄다.
- "서서히 눌리는 물림" 가설(옵션 A·E 의 전제)은 기각됐다: sync 를 4배 촘촘히 해도, 단일 접촉 3 N 에서 멈추게 해도 죽었고, 죽기 직전 10 ms 까지 단일 접촉력은 0.5~1.4 N 이었다(Hertz 로 립 면을 넘는 데 필요한 ~10 N 의 1/10). 즉 힘이 쌓여 뚫린 게 아니라 **한 sync(≤1 ms) 안에서 접촉이 켜지며 큰 관입으로 시작**된 것이다.
- 남는 기작 = W3b §1 이 커널 원문으로 확인한 DEME 양면 삼각형 접촉: 구 중심이 면 뒤(h<0)에 있어도 |h|<r 이고 투영이 삼각형 안이면 관입 r+|h| 로 잡는다. 립·파팅면·캡 모서리 근처에서 구 중심이 면 평면 뒤 r(1.16 mm) 안에 있다가 문이 회전해 발자국 안으로 들어오면 관입이 0 → 최대 2r(2.3 mm) 로 점프한다(≈ 27 N, 20 mg 알 → 1.4e6 m/s²). 4.16 mm 구(r 2.08)는 립 두께 4.6 mm 의 절반이라 이 조건이 드물고 무거워 pop 이 1.9~6.6 m/s 에 그쳤다(W3b). **이 기작은 정황 추정이다** — 죽는 순간의 owner·접촉점은 C++ terminate 때문에 남지 않았고, 옵션 F 에 넣은 q<6° 매 sync 진단(최대속도 owner 위치·최대 접촉점)은 F 가 하한에서 멈춰 발산 직전 행을 만들지 못했다.
- 4건 모두 q 2.3~3.6°, 립 틈(= R_lip 115 mm × q) 4.6~7.2 mm = 렌즈 길이 4.5 mm 의 1.0~1.6배 — 알 하나가 두 립 사이에 걸치는 바로 그 틈이다.

### 3-C. 렌즈 더미 — 옵션 F 완주 (사전 등록 외: 문 하한 5.0° + 폐합 sync 1 ms + 물림 가드 3 N(미발동)) · 3셀 · seed 460

설정 = 사전 등록(§GATES) + `door_min_q_deg 5.0`·`dt_sync_close_s 0.001`·`door_pinch_guard_N 3.0`. 원문 `cell_F_*/scoop_s1_seed460.json`, 옆면 각·제거 부피는 쉬는 입자만의 후처리 `cell_F_*/crater_rest_seed460.json`(원 JSON 값도 그 안에 병기). 실행 이력·rc·벽시계 = `run.log`.

| 셀 | 위치 mm | 펠릿면 mm | 잠김 도달 mm / 팔 정지 | 문 정지 (close → reclose) | 서보 환산° | 립 틈 mm | 물림 | 립 등가 피크 / 하강 팔 힘 피크 N | 포획 개 / g | 충전율 | z 딸림 | pop 스텝(v_max m/s) | 제거 부피 cm³ / 최대 깊이 mm | 옆면 각 ° (+x / +y / −x / −y, rest) | 벽시계 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c | (0, 0) | 41.1 | 25.0 / 0 | 4.99°(door_floor) → 4.94°(door_floor) | 7.44 | 9.9 | 8 | 1.98 / 2.55 | **154 / 3.12** | 0.124 | 210 | 0 (1.07) | 36.5 / 14.7 | +x 16.3 / +y 38.1 / -x 13.2 / -y —(시컨트 30.0) | 1895 |
| xp50 | (50, 0) | 40.0 | 25.0 / 0 | 4.99°(door_floor) → 4.94°(door_floor) | 7.44 | 9.9 | 2 | 1.24 / 1.95 | **106 / 2.15** | 0.085 | 131 | 0 (1.08) | 30.0 / 13.1 | +x —(시컨트 34.5) / +y 40.4 / -x 25.9 / -y 28.4 | 1924 |
| xm50 | (-50, 0) | 40.9 | 25.0 / 0 | 4.99°(door_floor) → 4.94°(door_floor) | 7.44 | 9.9 | 1 | 1.73 / 1.63 | **84 / 1.70** | 0.068 | 123 | 0 (1.08) | 29.1 / 16.4 | +x 15.4 / +y 34.4 / -x 11.3 / -y 34.8 | 1927 |

평균 2.32 g · COV 31.2 % (n=3)

읽는 법(초보자용):
- **하강·잠김은 세 셀 모두 정상**(잠김 25.0 mm 도달, 팔 힘 정지 0회, 팔 힘 피크 1.6~2.6 N ≪ 상한 6 N). 렌즈 더미가 구 더미보다 무르게 뚫린다.
- **문은 토크 정지가 아니라 하한(관절 4.99°, 재폐합 4.94°)에서 멈췄다.** 그때 힌지 저항 0.11~0.23 N·m(정지 1.764 의 6~13 %), 립 등가 힘 1.2~2.0 N(정지 20.5 N 의 10 %). 립 틈 9.9 mm = 렌즈 길이 4.5 mm 의 2.2배, 두께 2.5 mm 의 4배 — 상승 중 알이 흘러나온다(Rerun 검수에서 직접 보임, §6). **따라서 포획 3.12 / 2.15 / 1.70 g 은 "문을 5° 에서 멈췄을 때의 잔량"이지 실물 회당 질량 추정이 아니다.** 구 더미 W3b(11.8 g, 토크 정지 3.6°, 틈 5.4 mm)와의 26/18/14 % 는 설정이 다른 두 시뮬의 비교다.
- **셀 위치 효과**: 중앙(펠릿면 41.1 mm) > +x 50(40.0) > −x 50(40.9). 옆 셀은 언덕 비탈이라 보울이 비탈 아래쪽으로 열려(문은 y 로 닫히지만 보울 축은 x 를 가로지름) 담긴 양이 준다 — 다만 세 셀은 각 1회이고 DEME 비결정 산포(구 더미 ±7 %)가 있어 순서 이상의 주장은 못 한다. COV 31 % 는 위치 효과+비결정 합산이다.
- **구덩이**: 제거 부피 29~37 cm³(보울 공동 45.7 cm³ 의 64~80 %)에 최대 깊이 13~16 mm. 퍼낸 양(3.1 g ≈ 5.7 cm³ 벌크)보다 훨씬 큰 부피가 빠진 이유는 잠김·폐합으로 밀려난 알이 옆·아래로 재배치되고(차분 그림의 파란 퇴적) 유출 알이 비탈 아래로 굴러 내려갔기 때문이다 — "제거 부피 = 포획 부피" 가 아니다.
- **절단면 각(옆면 각, dh 기준)**: 문이 닫히는 ±y 방향 34~40°(xp50 +y 40.4 / xm50 ±y 34.4·34.8 / c +y 38.1), 보울 폭 방향 ±x 는 11~26° 로 완만(구덩이가 y 로 길고 x 로 얕음: 보울 폭 36 mm 안에서만 깊음). 가파른 벽 2건(c −y, xp50 +x)은 5 mm 격자 밴드에 고리 1개뿐이라 정의상 None, 시컨트 보조각 30°/34.5°. 절대 표면 경사각(h_post 기준, `angle_surface_deg`)은 언덕 경사가 겹쳐 ±x 에서 7~9°, +y 에서 34~39° 로 따로 JSON 에 있다. 비교용 구 더미 회귀 셀(토크 정지, 잠김 25): 25~30° 4방위 고름.
- **pop 0**(입자 최대속도 1.07~1.08 m/s, 전부 유출 알의 낙하) — 하한 5° 에서는 폐합 발산 기작이 켜지지 않았다.
- 벽시계 1895~1927 s(상한 2700 s 안; 사전 등록 2400 s 도 안). 구 더미 92 s 의 21배 = 구 수 7.4배 × 폐합 sync 4배 등.


### 3-D. 게이트 판정 (`gates_w8.json`, 사전 등록 `GATES_w8.md`)

| 게이트 | 판정 | 근거 |
|---|---|---|
| G1 발산 0 | **사전 등록 설정: FAIL** (c·xp50 발산, 옵션 A·E 도 발산) / 옵션 F: PASS 3/3 | `G1_no_divergence.diverged_attempts`, `pre_registered_setting_verdict` |
| G2 구 회귀 (개수 동일) | **FAIL** — 273 ≠ 315; 무수정 백업판 301·최종판 300 → DEME 비결정 (§3-A) | `G2_sphere_regression` |
| G3 3셀 완주 | PASS (옵션 F, rc 0, 1895/1924/1927 s ≤ 2700) | `G3_three_cells_complete` |
| G4 포획 질량 > 0 | PASS (3.12 / 2.15 / 1.70 g) — 하한 5° 캐베어트 | `G4_capture_mass_positive` |
| G5 절단면 각 보고 | 보고 완료 (3셀 × 4방위, None 2건은 사유·시컨트 병기) | `G5_crater_angle_reported.rest_only_postprocess` |
| R D341 Rerun | 완결 (`w8v2`: 계약 PASS + 육안 검수 JSON) | `R_rerun_d341` |
| S D470 sha | 기록 (스크립트·백업·보호 파일·heightmap.py·params·더미 npz·run 별 inputs_sha16) | `S_source_sha256` |
| all_pass | **false** (G2) · `all_pass_pre_registered_settings` false (G1) | |

## 4. 절단면 각 정의 (코드가 정본: `sim_deme_scoop_s1.py` `crater_angles()`)

- 입력: 퍼내기 전 heightmap h_pre(재안착 25스텝 뒤·하강 전, 구 전개·구별 반경, 5 mm 셀) 와 퍼낸 뒤 h_post(최종 프레임의 남은 입자). dh = h_pre − h_post (> 0 = 깎여 나간 깊이).
- 구덩이 중심 = 퍼내기 위치 반경 80 mm 안에서 dh ≥ 2 mm 인 셀의 dh 가중 무게중심.
- 4 방위(세계 +x·+y·−x·−y) 마다 ±22.5° 쐐기 안 셀을 5 mm 고리로 묶어 고리 평균 dh(r)·h_post(r) 를 만든다. r_peak = dh 최대 고리. 그 바깥으로 dh 가 0.2·d_max 아래로 처음 떨어지기 전까지의 고리 중 0.2~0.8·d_max 구간 고리에 최소제곱 직선 → **옆면 각 = atan|기울기|**(수평 기준). 같은 고리에 h_post 직선을 맞춘 각이 "절대 표면 경사각"(실물 Kinect 가 보는 것) — 언덕 위 구덩이라 두 값이 다르다.
- 밴드 고리가 2개 미만이면 None(가파른 벽이 5 mm 격자에서 미해상). 후처리 `crater_rest_writer.py` 가 보조로 **시컨트 각**(r_peak 고리 ↔ 첫 0.2·d_max 미만 고리) 을 낸다 — 정의 밖, 참고용.
- **쉬는 입자만** (`crater_rest_seed460.json`): 최종 프레임에 문 틈으로 흘러내리는 중인 공중 알과 툴에 얹힌 알(F_c: 63구 = 9알, z 최대 80 mm)이 h_post 에 80 mm 스파이크로 들어가므로, "중심이 h_pre(xy)+10 mm 아래인 입자" 만으로 h_post 를 다시 만들어 같은 정의를 돌린 값을 **인용값**으로 쓴다(원 결과 JSON 값도 병기).
- 검증: 합성 35° 원뿔 구덩이 → 36.0°(격자 이산화 +1°). 구 더미 회귀 셀의 실제 구덩이 → 25~30° (`regression_sphere/crater_profiles_seed460.png`).

## 5. 코드 변경 요약 (`sim_deme_scoop_s1.py`, +~330줄, diff = `sim_deme_scoop_s1_w8.diff`; 전부 params 게이트, 기본값이면 W3b 경로)

| 항목 | 내용 |
|---|---|
| `particle_shape: "npz_template"` | npz `clump_template_json` 의 `sphere_radii_m`·`offsets_m`·`mass_kg`·`moi_kg_m2`·`union_volume_m3` 로 `LoadClumpType` 재구성 + `clump_positions_m`/`clump_quaternions_xyzw` 배치. 자체 검사 = 내 전개식(p + R(q)·offset) 이 npz 구 행을 재현(2.8e-17 m)·`clump_ids` 클럼프-major. `sim_pellet_model` import 없음. 구 경로에 클럼프 npz 를 주면 SystemExit |
| 구 전개 | `expand_spheres()` (scipy Rotation, xyzw) 로 펠릿면·heightmap·스냅샷·npz `sphere_positions_m` |
| heightmap 전/후/차분 | `heightmap_pre_m`(재안착 뒤)·`heightmap_m`(최종)·`heightmap_npz_m`(파일 그대로) + PNG 3면 + 방위 프로파일 PNG |
| `crater_angles()` | §4 |
| 렌더 타임라인 | `render_timeline_dt_s`/`render_timeline_path` — 코디네이터 요청 형식(`t_s, clump_pos_m[T,N,3], clump_quat_xyzw[T,N,4], tool_pos_m, tool_quat_xyzw, door_deg, captured_ids` + door_pos/quat, phase, metadata_json) |
| 발산 대응 (사전 등록 외) | `dt_sync_close_s`(옵션 A) · `door_pinch_guard_N`(옵션 E, reason `pinch_guard`) · `door_min_q_deg`(옵션 F, reason `door_floor`) · `diag_flush_below_q_deg`(q<6° 매 sync flush + 최대속도 owner 위치·최대 접촉점) |
| 그 외 | 진행 print 에 벽시계, 폐합 flush 를 10 sync 마다(원래 매 스텝), `stops[]` 에 `max_single_contact_N` |

보조 스크립트(산출 폴더): `gates_w8_writer.py`(판정), `crater_rest_writer.py`(후처리), `w8_rerun_export.py`(D341), `write_inspection_w8.py`, `report_table.py`, 실행기 `run_w8.sh`/`run_w8A.sh`(rc 로깅 버그 — `$?` 가 `$(date)` 에 덮임, A_c rc=0 오기록)/`run_w8E.sh`/`run_w8F.sh`(수정판).

## 6. D341 Rerun (옵션 F 중앙 셀, `cell_F_c/scoop_s1_seed460_w8v2.*`)

- SDK/CLI 0.34.1 핀 일치 · footer `rrd verify` PASS · 엔티티 25/타임라인 5(`blueprint, log_time, frame, phase, sim_time_s`)/컴포넌트 계약 PASS · 고정 blueprint `.rbl` 검증 PASS · 헤드리스 스크린샷 6400×3600 — `_w8v2_rerun_validation.json` `pass: true`. 1차 `w8` 판도 계약 PASS 였으나 기본 시간 커서(+0.9 s)에서 최종 더미가 안 보여(spheres_post 가 최종 프레임에만 로그됨) 정적 사본(`geometry/pile/spheres_final_static`, `geometry/tool/final_static`) 을 추가한 `w8v2` 를 검수본으로 쓴다(둘 다 보존).
- 🔴 실제로 열어 본 것(`_w8v2_inspection.json` 관찰 7건·한계 5건 요약): 왼쪽 위 = 회색 로제트(7구 렌즈) 알이 넓은 언덕을 이루고, 그 위 80 mm 에 들린 툴(초록 두 링) 안에 **주황 포획 알 154개 뭉치**, 문 쪽 입 부근에 **하늘색 z-딸림 알 뭉치**, 그리고 입 아래로 **알이 줄지어 떨어지는 중**(9.9 mm 틈 유출의 시각 증거 = 충전율 0.124 의 이유). 시간 커서 +0.9 s 의 툴(하강 중)도 겹쳐 그려져 립 접촉점(빨강)이 더미 정수리에 띠로 보인다. 오른쪽 위 = post heightmap 점 격자(파랑→빨강) 위 4방위 흰 화살표에 각 라벨, 적합 고리 빨간 점. 아래 = z_lip(51→16→97 mm)·q(27.5→5.0 유지)·lipF(≤2 N) 곡선, 결정 이벤트 4행(door_floor ×2, CAPTURE, CRATER_ANGLE).
- 한계: 구덩이 자체는 이 카메라 각에서 옅은 패치로만 보임(matplotlib `heightmap_rest_seed460.png` 가 명확), 라벨 겹침, −y "nandeg", 커서 툴과 정적 툴이 함께 그려짐, RRD 의 절단면 각 라벨은 후처리 전 값.
- 렌더 타임라인은 Rerun 과 별개(코디네이터용 npz).
## 7. 🔴 비주장 (non-claims)

1. **옵션 F 의 포획 질량은 실물 회당 질량의 추정치가 아니다.** 문이 토크 정지가 아니라 하한 5°(립 틈 9.9 mm = 렌즈 길이 2.2배) 에서 강제로 멈췄고, 상승 중 그 틈으로 알이 흘러내리는 것이 Rerun 검수에서 직접 보인다. 실물(D481)은 서보 2.8~3.5° = 관절 0.3~1.0° 까지 닫혀 틈이 훨씬 좁다. 구 더미 11.8 g 과의 비교(26 %)는 "설정이 다른 두 시뮬" 비교다.
2. **발산 기작(§3-B 유령 접촉)은 정황 추정이다.** 죽는 순간의 알·접촉점을 확보하지 못했다. 옵션 F 의 진단 행(q<6° 매 sync) 은 하한에서 멈춰 발산 직전을 담지 못했다.
3. **물성 전부 MEASURE**(mu 0.45·Crr 0.06·ρ 905 = W6 앵커 임시값, E 5e6 = 수치 안정 placeholder, 실제 PP 는 ~1.5 GPa). 힘·정지각·포획량·절단면 각 절대값 인용 금지. 특히 E 5e6 은 립–알 접촉을 300배 무르게 만들어 이번 발산의 배경 조건이다.
4. **더미는 E 1e7 로 정착 → 본 실행 E 5e6** 으로 t=0 미세 재안착(heightmap 최대 |Δ| 2.9 mm, `heightmap.pre_vs_npz_max_abs_diff_m`). 퍼내기 전 기준은 재안착 뒤 값이다.
5. **절단면 각은 5 mm 격자·고리 평균·직선 적합의 산물**로, 가파른 벽(−y)은 미해상(None). 실물 Kinect 절단면 각(R1 E1) 과 정의를 맞추기 전에는 수치 비교 금지. 언덕 위 구덩이라 "옆면 각(dh 기준)" 과 "절대 표면 경사(h_post 기준)" 가 다르며 둘 다 보고했다.
6. **DEME run-to-run 비결정**: 같은 입력의 포획 개수 산포 273~315(구 더미). 렌즈 셀은 각 1회 실행이라 셀 간 차이(xp50·xm50 vs c)의 얼마가 위치 효과이고 얼마가 비결정인지 가를 수 없다.
7. **셀 위치 ±50 mm 는 x 축**(문이 y 로 닫히므로 쓸어 담는 방향과 직교) 으로 내가 정했다 — 지시서는 축을 지정하지 않았다. 언덕이 대칭이라 ±x 는 거울상이다.
8. **폐합 sync 1 ms·문 하한 5°·물림 가드 3 N** 은 사전 등록 외 설정이다(옵션 F 에 셋 다 들어 있다; 가드는 F 에서 발동하지 않았다). 사전 등록 설정 자체의 G1 은 FAIL 로 남긴다.
9. 렌더 타임라인 `door_deg` 는 sample 뒤 q 갱신이라 1 sync(폐합 1 ms) 지연. `tool_quat` 은 단위(규정 병진만). 프레임 규약은 npz `metadata_json`.
10. 상태 원장·커밋·새 의존성 0. 로봇 미접촉. 기존 산출·기존 npz·`sim_deme_scoop.py`(보호) 무수정. 메인 `sim_pellet_model.py` 무수정.

## 8. 판정 (일상 언어) · 다음 승인 경계

- **되는 것**: 실측 렌즈 알 더미를 스쿱 시뮬이 읽고(템플릿 재구성 검증 완료), 퍼내기 전후 heightmap·구덩이·절단면 각을 코드 정의로 뽑는 길이 열렸다. 렌더용 궤적도 나온다.
- **안 되는 것**: 실물 절차 그대로(문을 토크 정지까지 닫기)는 지금 접촉 모델(E 5e6 + DEME 양면 삼각형 셸)로는 **재현 불가** — 4/4 발산. 문 하한 5° 로만 완주하며 그 포획량은 실물 추정에 못 쓴다.
- **아직 모르는 것**: 발산 순간의 정확한 기하(어느 면·어느 구), 그리고 E 를 올리면(실제 PP 쪽으로) 발산이 사라지는지.
- 다음 승인 경계(코디네이터/사용자 결정):
  1. **강성 사다리 1셀**: E 1e7 → 1e8 (dt 2e-6, 셀당 ~2.5 h) 로 사전 등록 설정(토크 정지) 완주 여부 확인. 통과하면 이것이 렌즈 더미의 기본 설정.
  2. **셸 모델 변경**: 립·파팅면·캡을 별도 볼록 조각(closed convex) 으로 나눠 DEME 가 "안/밖" 을 판정할 수 있게 하거나, 립 모서리를 둥글려 발자국 점프를 줄이기. W3b 의 d_* 셀 교훈(열린 셸 금지) 과 함께 설계 필요.
  3. **발산 순간 포착**: 옵션 F 의 진단(q<6° 매 sync) 을 하한 없이 돌려 죽기 직전 행(최대속도 owner 위치·최대 접촉점·면 그룹) 확보(`sim_deme_s1_diverge_min.py` 식 절편 재현 포함). 셀당 ~30 min.
  4. 회귀 게이트 재등록: "seeded 입력 bit-동일 + 포획 개수 허용 범위(예 ±15 %)".
  5. 문 하한 5° 결과를 프로포절 형상 예측 데이터로 쓰려면 "문 정지각 고정" 을 절차로 채택하는 결정이 먼저 필요(실물도 문을 5° 에서 멈추는 실험이 있어야 비교 가능).

## 9. 산출물 · 규칙 준수

```
claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w8_deme_scoop_lens/
  REPORT_w8.md (이 파일)   GATES_w8.md (사전 등록)   gates_w8.json (판정, gates_w8_writer.py)   run.log (실행 이력·rc·벽시계)
  params_w8_cell_{c,xp50,xm50}.json (사전 등록)  params_w8{A,E,F}_cell_*.json (변형)  run_w8{,A,E,F}.sh  sim_deme_scoop_s1_w8.diff
  regression_sphere/ (패치판 구 회귀)  regression_sphere_prew8_script/ (무수정 백업판 재실행)
  cell_c/ cell_xp50/ (사전 등록, 발산: timeline·stderr·_obj)  cell_c_try1_killed_for_render_timeline/  cell_xm50_killed_switch_to_A/
  cell_A_c/ (발산)  cell_A_xp50_killed_after_settle/  cell_E_c/ (발산)  cell_E_xp50_killed_after_settle/
  cell_F_{c,xp50,xm50}/ (완주): scoop_s1_seed460.{json,npz}, timeline_seed460.json, heightmap_seed460.png, crater_profiles_seed460.png, snapshot_seed460.png,
      crater_rest_seed460.json, heightmap_rest_seed460.{png,npz}, crater_profiles_rest_seed460.png, _obj/
  cell_F_c/scoop_s1_seed460_w8v2.{rrd,rbl} + _w8v2_inspection.png + _w8v2_rerun_validation.json + _w8v2_inspection.json (+ 1차 w8 4종, observations_w8v2.json)
  render_timeline_cell1.npz (코디네이터 요청, 53 프레임)   crater_rest_writer.py  w8_rerun_export.py  write_inspection_w8.py  report_table.py  gates_w8_writer.py
sim_deme_scoop_s1.py (수정)   sim_deme_scoop_s1.py.bak_20260910_pre_w8 (백업, W3b 판 바이트 동일)
```
