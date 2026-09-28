# 교수 질의 검토 — 시간 간격·실제 펠릿·접촉식·Isaac 렌더

작성: 2026-09-16. 이번 case의 신규 변수: []. 새 시뮬레이션·렌더·학습·실물 조회 없음. 원본 코드/자료를 읽고 CPU로 파라미터·형상 산술과 파일 보존만 확인했다. 이 보고는 원래 실험 판정을 변경하지 않는다.

## 1. 답부터: 원래 첫 작업과 교수님 제안은 다르다

**원래 첫 작업은 W13의 원자료 판정 오류 2개를 새 revision에서 고치고 CPU 단위/회귀 테스트를 하는 것**이다. 물리를 다시 돌려서 분류 버그를 고치는 계획이 아니다.

1. phase(큰 동작 단계)가 바뀐 곳만 모아야 하는데 첫 행·subphase(세부 단계)까지 섞인 문제: 규약 기대 11개 대 기존 기록 25개.
2. 펠릿 전체가 source 상자 안에 있는지 검사할 때 바닥 쪽 구체의 최하단 대신 최상단을 비교한 문제. 원자료의 위치·ID는 보존하고 파생 분류만 새로 만든다.
3. 기존 버그 FAIL → 수정본 PASS, 별도 수식으로 교차 대조, 원본 해시 보존이 첫 완료 조건이다.
4. 그다음 재생 3결함 수정 → 운반 도중 144개 ID의 보유 판정이 사라진 원인 분리 → 동일 조건의 짧은 성능 계측 순서였다. 각각 다른 작업이며 GPU 실행은 별도 승인이다.

정본: [이전 재개 문서 §4](../../CONTINUE_20260914_W13_REPAIR_PERFORMANCE.md). W13은 약 9시간 벽시계 동안 물리 24.4868초까지만 진행한 **TIMEOUT 부분 실행**이고 원자료 FAIL2/재생 FAIL3이다. 전체 성공·배출량 검증 완료가 아니다.

교수님 제안은 별도의 **시간 간격 확대에 따른 안정성·정확도·계산비용 비교**다. 의미가 있다. 단, 기존 원자료 판정 오류를 먼저 분리하고 과거 10µs 실패도 비교표에 포함해야 한다. 이번 대화는 질의 검토·보관·새 세션 준비까지이며 실제 dt 실험은 실행하지 않았다.

## 2. 1µs·2µs·10µs·1ms를 어떻게 비교해야 하나

### 2.1 단위와 목적

`dt`는 솔버가 접촉력과 운동 상태를 갱신하는 **물리 시간 간격**이다. GPT 응답 시간, 카메라 fps, 제어 명령 간격과 다르다.

| dt | 초 | 물리 1초당 기본 스텝 수 | 1µs 대비 기본 적분 횟수 |
|---|---:|---:|---:|
| 1µs | 0.000001 | 1,000,000 | 1 |
| 2µs | 0.000002 | 500,000 | 1/2 |
| 10µs | 0.000010 | 100,000 | 1/10 |
| 100µs = 0.1ms | 0.000100 | 10,000 | 1/100 |
| 1ms | 0.001000 | 1,000 | 1/1,000 |

이는 `N≈T/dt` 산술이지 실행시간 보장값이 아니다. GPU 초기화·충돌 후보 탐색·동기화·CPU 전송·진단·압축 저장이 남는다. dt를 1,000배 키운다고 벽시계 시간이 1,000배 줄어든다고 예측하면 안 된다.

충돌의 짧은 변화보다 dt가 너무 크면 큰 겹침을 뒤늦게 발견하고 큰 반력을 한 번에 적용해 속도 폭증·침투·통과·부정확한 정지각을 만들 수 있다. 작은 dt는 시간 이산화 오차를 줄이는 방향이지만 **잘못된 형상·마찰·관성 모델까지 실물과 같게 만들어 주지는 않는다.** 실제 안정 한계는 접촉 수, 형상, 질량/관성, 강성/감쇠, 속도, 이동 벽, 적분법에 의존한다. 특정 dt를 보편적으로 안전하다고 하지 않는다.

### 2.2 10µs는 이미 실패 기록이 있다

다음은 이번에 실제 파일을 다시 읽어 확인한 값이다.

- 직전 W10 `cell_DE_c`는 **E=5MPa, dt=10µs, 문 닫힘 22.5°/s**였다.
- 저장 timeline은 닫힘 단계 `t=2.56118s`까지이며 저장 최대속도는 **13.0303m/s**다.
- 그 뒤 C++ 오류 메시지는 시스템 최대속도 **22,291.97m/s**, 엔진 예외 기준 **10,000m/s** 초과를 보고했다. 실제 PP 속도가 아니라 수치 폭주 보고값이다. 실행기 기록은 **rc=134**, 완결 result JSON은 없다.
- 이벤트 파일의 5.7008m/s, 마지막 저장값 13.0303m/s, 엔진 종료 보고값 22,291.97m/s는 **서로 다른 관측 시점·경로**다. 같은 최대값을 세 번 다르게 적은 것이 아니다.
- 그래서 10µs → 2µs로 줄였고, 2µs 결과에도 5m/s 경고가 남아 1µs를 추가한 것이다.

원문: [10µs 설정](../../runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/params_w10_DE_c.json), [마지막 저장 timeline](../../runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/cell_DE_c/timeline_seed460.json), [엔진 종료 오류](../../runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/cell_DE_c/stderr.txt), [실행기 rc 기록](../../runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/run.log:21).

| 조건 | 기존 관찰 | 해석 |
|---|---|---|
| 직전 10µs | 닫힘 중 수치 폭주, rc134 | 유효한 포획량 비교 결과가 아님 |
| W10 2µs | 포획 541알 / 10.9592437g, 저장 최대속도 5.3291m/s | 한 셀의 완료 결과, 속도 경고 남음 |
| W11 1µs | 포획 517알 / 10.4730666g, 저장 최대속도 2.7255m/s | 한 셀에서 고속 경고 감소, 수렴 증명 아님 |
| 100µs·1ms | 이 닫힘 프로토콜의 새 비교 미실행 | 성공/실패 및 시간 단축 정도 미확정 |

541→517은 24알, 약 4.44% 차이다. 단 한 번씩이며 접촉의 민감성·병렬 연산 순서 영향도 있어 통계적 우월성이나 수렴 차수로 해석하지 않는다. 원자료 effective params 차이는 dt와 출력경로뿐임을 재확인했다. 10µs 재시험도 코드·초기 상태·물성·제어·보호 조건을 고정한 새 경로에서 해야 한다. W8은 문 하한 등도 달라 깨끗한 dt 비교군으로 섞지 않는다.

**정지 더미 생성에서 dt=20µs가 통과했다는 기록도 있다.** 이것은 더미 정착 단계의 다른 조건이다. 문이 입자를 압박하는 닫힘 단계에서도 20µs가 안정하다는 증거가 아니다.

### 2.3 1ms는 현재 0.1ms 제어/진단 간격과 충돌한다

현재 하강 sync 요청은 4ms, 닫힘은 1ms, 문 각도가 6°보다 작은 세밀 구간은 **0.1ms**다. `DoDynamicsThenSync(D)`는 그 시간만큼 내부 스텝을 실행한 다음 상태를 읽는 호출이다.

설치 DEME의 실제 누적 시간은 float32 dt를 float64로 반복 더해 요청 D 이상이 되는 시점이다. 이상적인 정수 나눗셈과 다를 수 있다. 독립 [설치 바이너리 시간 경계 감사](/home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/DEME_SOURCE_BOUND.md)와 이번 CPU 산술을 대조했다. 예를 들어 nominal 1µs에서 4ms 요청은 4,001스텝·약 4.001ms다.

같은0.1ms 요청을 계산한 **소스 기반 예상값(새 GPU 측정 아님)**은 dt1/2/10/100/1000µs에서 각각 약101/102/110/200/1000µs다. 특히100µs는 float32 표현이 요청0.1ms보다 조금 작아 두 스텝이 되는 경계가 있다. 실제 실험에서는 requested/actual 시간을 모두 기록해야 한다.

**dt=1ms일 때 0.1ms 뒤 상태를 검사할 수는 없다. 최소 한 스텝부터 이미 약 1ms다.** 따라서 명목 sync 설정을 놔두더라도 실제 보호 확인 간격과 문 이동량이 달라진다. 이를 숨기고 순수 dt 효과라고 주장하면 안 된다. 다음 실험은 다음을 구분해야 한다.

1. 작은 dt 사이의 동일 제어 프로토콜 민감도 비교.
2. 1ms처럼 현재 제어 간격보다 큰 dt의 **호환성/안정성 한계 시험**. 동일 제어 유지 불가능 여부 자체도 결과다.

1ms에서 제어 간격을 몰래 늘리거나 dt를 내부적으로 줄이는 substep을 켜면 질문이 달라진다. 후자는 표시/환경 스텝만 1ms이고 실제 접촉 dt는 여전히 작을 수 있다. 별도 case와 승인 없이 이 변경을 함께 구현하지 않는다.

### 2.4 다음 dt 비교의 사전 정의 권고 — 아직 실행/새 합격선 적용 안 함

- 초기 NPZ/seed460/20,000알/7구 형상/질량/MOI/물성/공구·문 형상/속도/제어/보호선 고정.
- 같은 **물리 구간** 비교. 같은 벽시계만 돌려 서로 다른 동작 진행도를 비교하지 않는다. 실패 셀은 마지막 유효 시각과 이유를 보고한다.
- 기존 >5m/s 경고, >20m/s Python 중단, 3N 물림 보호를 사후 완화하지 않는다. coarse dt에서는 내부 폭주가 Python 확인보다 먼저 일어날 수 있음도 기록한다.
- 기록: 실제 누적 물리 시간·sync 간격·스텝 수·벽시계 시간, 종료 사유, 최대속도, 정지각/접촉력, 포획량. 겹침·에너지·동일시각 위치 오차는 관측 가능성과 계측 비용을 먼저 검토한다.
- 1→2→10µs를 우선 비교하고 100µs/1ms는 교수님 질의에 대응하는 별도 한계 셀로 설계할 수 있다. 새 명령/시간상한/정리 예산/출력 경로를 제시한 뒤 승인받는다.
- 기존 NPZ는 완전 solver checkpoint가 아니다. 접촉 이력·각속도 등을 보존하지 않은 상태를 ‘정확한 중간 재시작’으로 쓰지 않는다.

## 3. 7개 구체가 정확히 무엇인가

**실제 펠릿 한 알 ↔ 시뮬레이션 강체 한 개 ↔ 그 강체의 접촉 형상을 이루는 겹친 구 7개**다. 서로 다른 실제 펠릿 7알을 접착한 모델이 아니다.

- 중앙 큰 구 1개, XY 평면에서 60° 간격의 주변 구 6개.
- 7개 내부 상대 위치는 고정이다. 한 알의 중심 위치, 회전 자세, 선속도, 각속도, 질량, 관성을 공유한다.
- 서로 다른 **20,000알은 각각 독립적으로 움직인다.** 접촉 계산용 구성 구의 행은 140,000개다. NPZ의 clump ID를 다시 세어 모든 ID가 정확히 7행씩 갖는 것을 확인했다.
- 같은 owner의 구들은 서로 충돌시키지 않는다. 설치 `DEMContactKernels_SphereSphere.cu:171`의 동일 owner 제외가 이를 확인한다. 다른 펠릿/공구와의 여러 구성 구 접촉은 한 몸체의 합력·합토크로 모인다.

### 3.1 실제 입력된 형상 수치

| 항목 | 정확한 실행 템플릿 값 |
|---|---:|
| 목표 장축×폭×두께 | 4.5 × 3.8 × 2.5mm |
| 중앙 구 반경 | 1.250195464mm |
| 주변 6구 각각의 반경 | 1.156384312mm |
| 주변 +X 구 중심 | (1.093967523, 0, 0)mm |
| 주변 +X,+Y 구 중심 | (0.546983762, 0.644247377, 0)mm |
| 나머지 주변 중심 | 위 값의 XY 대칭 배치 |
| 실제 구 합집합의 축방향 외곽 | 4.500703671 × 3.601263379 × 2.500390928mm |
| 폭 오차 | 목표 3.8mm보다 **5.229911% 작음** |
| 템플릿 합집합 체적 | 22.383847657mm³ |
| 한 알 질량 | 20.257382129mg |
| Ixx, Iyy, Izz | (2.062768517, 2.757022245, 3.484048836) × 10⁻¹¹ kg·m² |

질량은 `m=ρV`, 목표 타원체 `V=πabc/6`, ρ=905kg/m³로 구했다. 7구 체적을 단순히 합산하면 겹침을 중복 계산하므로 그렇게 하지 않았다. 생성기는 구 합집합에 대한 Monte Carlo 적분과 체적 맞춤 스케일을 썼다. 이번에는 기존 적분값·스케일로 최종 질량/MOI를 별도로 재계산했다. **Monte Carlo 자체를 새로 돌려 수렴성을 다시 측정한 것은 아니다.**

관성은 `Ixx=ρ∫(y²+z²)dV` 등으로 구한 합집합 대각 성분을 스케일의 5제곱으로 보정했다. `LoadClumpType(mass, moi, radii, offsets, material)`에 그 3개 대각값을 넘긴다. 전체 비대각 텐서를 솔버에 넘기거나 여기서 실물 관성을 계측한 것이 아니다.

치수 a/b는 사진 판독 ±0.2mm, c는 10알 적층 ±0.05mm라는 기존 관찰에 기초한다. 정밀 3D 스캔이 아니며 사진/적층 관찰과 7구 근사 오차는 별개다. 905kg/m³·탄성·마찰·반발 등은 실물 보정이 안 됐다.

### 3.2 왜 이렇게 만들었나 — 렌더 제한 때문이 아니다

구 하나만 쓰면 렌즈형 한 알의 납작함과 방향별 접촉을 잃는다. 여러 겹친 구를 한 강체로 묶으면 구–구/구–삼각형 접촉 계산을 사용하면서 비구형 형상을 근사할 수 있다. **DEME의 접촉 표현 선택이지 ‘PhysX가 개별 펠릿을 그리지 못해서’ 한 선택이 아니다.**

이미 Isaac 화면에서도 각 펠릿 20,000개를 독립 instance로 그린다. 같은 모형을 재사용한다는 뜻이지 위치·회전까지 같은 덩어리라는 뜻이 아니다. 7구를 7개 독립 몸체로 풀어 놓으면 한 알이 분해되는 다른 물리 모델이 된다.

동시에 7구 근사가 실물보다 표면이 울퉁불퉁하고 다중 접촉을 많이 만들 수 있다는 교수님의 문제 제기는 타당하다. 폭 under-fill, 표면 요철, 마찰·구름저항과의 중복 영향이 정지 더미·끼임·포획량에 영향을 줄 가능성이 있다. **현재 문제가 7구 때문이라고 확정한 것은 아니며**, 형상 수렴/실물 보정은 dt와 분리한 후속 비교다. 표시용 모델만 매끈한 렌즈로 바꿔도 접촉 물리는 바뀌지 않는다.

정본: [canonical NPZ](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pellet_model/pile_lens_20260910/pile_lens6_a4p5_b3p8_c2p5_slab_n20000_seed460.npz), SHA256 `659d6b0bc771678a0c7209d91f550edc933d03e41922245ea0adb64eeb818812`. [생성기](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/sim_pellet_model.py:518), [실제 솔버 주입 코드](../../../sim_deme_scoop_s1.py:234), [이번 재계산 JSON](PARAMETER_AUDIT.json).

## 4. 어떤 파라미터가 어디에 들어가나

**물리 물성 / 형상·초기조건 / 제어·수치해석 / 측정·분류 / 렌더**를 분리해야 한다. 전수 필드 대조는 [PARAMETERS_ALL.md](PARAMETERS_ALL.md), 원값과 코드 참조는 [PARAMETER_AUDIT.json](PARAMETER_AUDIT.json)에 보존했다.

### 4.1 물성·형상·초기조건

| 파라미터 | 실제 값 | 물리에서 하는 일·근거 수준 |
|---|---|---|
| pellet E | 5×10⁶Pa = 5MPa | 겹침에 대한 탄성 반력 크기. 실물 PP Young률 보정값 아님 |
| tool mesh E | 3×10⁹Pa = 3GPa | 공구–입자 접촉의 유효 강성 계산에 사용 |
| domain/tray/bin 재료 E | 5MPa | W13의 mat_w는 입자와 같은 mp. 모든 벽이 tool E=3GPa인 것이 아님 |
| ν | 0.30 | 유효 Young률·전단계수 계산 |
| CoR, e | 0.30 | 반발계수. 속도에 따른 접촉 감쇠 계수 계산, 실물 낙하 보정 아님 |
| μ | 0.45 | 접선력의 Coulomb 상한. 실물 마찰시험값 아님 |
| Crr | 0.06 | 구름저항 torque-only 항. 실물 구름시험값 아님 |
| ρ | 905kg/m³ | 템플릿 제작 당시 질량/MOI에 반영한 임시 밀도 |
| 중력 | (0,0,-9.81)m/s² | 아래로 가속 |
| 몸체 수/분포 | 20,000알, 동일 템플릿 | 크기·질량 분포 다양성은 모델하지 않음 |
| 초기 위치/회전 | 동일 canonical NPZ, seed460 | 이미 정착된 위치와 quaternion을 읽음 |
| 공구 형상 | 고정 jaw/문 별도 삼각형 메시 | lip·hinge 좌표와 두께 보정 포함 |

**파라미터 이름만 보고 오해하기 쉬운 곳:**

- `particle_shape=npz_template` 경로는 NPZ의 질량/MOI/반경/offset을 직접 쓴다. JSON의 `particle_density_kg_m3`만 바꿔도 기존 NPZ 질량이 자동으로 바뀌는 구조가 아니다. `pellet_dia_mm=4.16`, `pellet_len_mm=4.16`, `clump_aspect`도 이 분기에서 실제 렌즈 치수를 정하는 값이 아니다.
- `bulk_density_g_cm3=0.55`는 입자 고체 밀도 0.905g/cm³와 다르다. 알 사이 공극을 포함한 벌크밀도 가정으로 초기 envelope/충전률 보고에 쓰이며 Hertz 힘 공식에 직접 들어가지 않는다.
- `target_repose_angle_deg=28`은 물리적으로 각도를 강제하는 힘 항이 아니다. 이번 초기 더미는 slab/FCC seed 후 중력 정착이며 자유낙하로 쌓아 28°를 측정한 결과가 아니다.
- 더미 생성은 E=10MPa·dt20µs였고, scoop에서는 E=5MPa로 시작한다. 이를 전 과정 동일 물성의 자유낙하/안식각 실험이라고 설명하면 안 된다.
- 알 모양의 기하 표면 요철과 렌더 재질 `roughness=0.6`은 다른 값이다. 후자는 빛 반사 표현이며 마찰계수로 사용하지 않는다.

### 4.2 공구·문·수치·보호

| 항목 | 값 / 의미 |
|---|---|
| DEME dt | W10 2µs, W11/W13 1µs |
| 적분법 | 설치 기본 `EXTENDED_TAYLOR`; 프로젝트 `SetIntegrator` override 없음 |
| 충돌 탐색 설정 | `cd_update_freq=20` 전달. 매 Python sync 간격과 다른 내부 설정; 이것만으로 런타임 탐색 빈도가 항상 고정이었다고 단정하지 않음 |
| 하강 | 25mm/s, 목표 침투 25mm, 접근 간격 10mm |
| 상승 | 150mm/s, W10/W11 목표 상승 80mm |
| 문 닫힘 | 22.5°/s 명목, 실제 명령은 ceil/스텝 수 계산 영향 있음 |
| 문 열림 기준 | 서보 30° − 영점 offset 2.5° = 기구 관절 27.5° |
| 문 닫힘 목표 | 기구 관절 0°, 강제 5° 하한 없음 |
| 서보 정지 모델 | 1.96N·m × 0.9 = **1.764N·m** 저항 문턱 |
| 물림 보호 | 최대 단일 접촉력 3N 조건 |
| 팔 반력 제한 | 6N |
| 속도 경고/중단 | 5m/s 경고 / Python >20m/s 중단 / 엔진 예외 10,000m/s |
| `SetMaxVelocity(5)` | 충돌 탐색 마진 추정의 기대속도 상한/anomaly 설정. 속도를 5m/s로 잘라 주는 제어기가 아님 |
| 공구 질량 | 고정 0.01827kg, 문 0.02416kg; mesh 접촉 유효질량에도 관련 |
| 공구 MOI | 고정 각축1e-5, 문 각축4e-5 kg·m²로 코드 지정. 실측 다물체 로봇 관성 모델 아님 |
| 공구 좌표 | link5 기준 lip (8.1,0,169.6)mm; hinge (0,18.821,52.035)mm |
| 보울 분류 기하 | 내부 r20mm, cheek 반폭18.2mm. 분류식과 실제 충돌 메시를 동일시하지 않음 |
| 충돌 두께 | wall1.6mm, cap2.0mm, 추가 collision wall3mm. 실물 표면과 똑같다고 단정 금지 |

공구·문은 지정 선속도/각속도로 움직이는 경계이며, 접촉 반력에 따라 Python 제어가 정지 여부를 판단한다. 로봇 전체의 모터 전류·감속기·마찰·유연성·관절 토크 동역학을 모두 푼 모델이 아니다. 반력 기반 정지 기능이 있다는 것과 실물 ST3215 서보를 정확히 재현했다는 것은 다르다.

W13 추가 배치/운반: base 회전90°, travel45cm, 운반150mm/s, source 벽5mm, bin 내경80mm/높이70mm/벽3mm/둘레48분할, 배출 높이 여유20mm·대기1.5s, domain pad60mm다. base38cm/pellet26cm는 과거 CLI 예시를 채택한 **declared_not_measured** 값이다. W13은 domain·유한 벽·전체 경로가 달라 W10/W11과 dt만 다른 비교가 아니다.

### 4.3 저장·판정도 별도 파라미터다

- W10/W11 입자 위치·회전 표시 프레임 목표 간격 0.05s. DEME 내부 스텝마다 저장한 것이 아니다.
- W13 입자 프레임 기본 0.1s + 특정 이벤트. 원자료 full sync16,304개에 대해 입자 프레임283개다.
- W13 `settlement_window_s=.25`, 속도 `.005m/s`, 이동 `.001m` 등은 정착 판정을 위한 관측 조건이지 접촉력 계수가 아니다.
- `settlement_frame_dt_s=.05`가 설정에 있어도 W13 생산 record cadence는 `particle_frame_dt_s=.1`을 읽는다. 이번 정적 검사에서 `.05` 값으로 생산 저장 주기를 바꾸는 읽기 경로를 찾지 못했다. 최종 창이 0.05s cadence를 만족하지 않았다는 기존 감사와 일치한다. **설정 파일에 써 있는 것 ≠ 실제 시행됨**의 사례다.
- 이번 전수표의 코드 참조는 정적 검색이다. 상속된 미사용 분기도 포함하므로 ‘참조가 있다’만으로 실행 적용을 증명하지 않는다.

## 5. 실제 쓰는 물리 계산식

설치 배포판은 **DEME 2.4.0**, `UseFrictionalHertzianModel()`을 사용한다. 이는 접촉 중 법선 Hertz 탄성·속도 감쇠와 접선 접촉 이력·Coulomb 제한, 구름저항을 계산하는 기본 모델이다. 아래는 일반론만 옮긴 식이 아니라 **설치 커널과 바이트 동일한 공식 소스**에 맞춘 표기다. [8개 공식/설치 소스 SHA 교차 검증](OFFICIAL_SOURCE_CROSSCHECK.json).

### 5.1 한 구성 구가 어디에 있는가

펠릿 중심 x, 회전행렬 Q, 고정 국소 offset oᵢ일 때:

`xᵢ = x + Q oᵢ`

따라서 펠릿이 회전하면 구성 구 전체가 같이 움직여 접촉 위치와 토크가 달라진다. 입자마다 quaternion과 각속도가 있다.

### 5.2 접촉 기하와 상대속도

두 구성 구의 반경 RA,RB, 중심거리 d일 때 법선 겹침은 `δ=RA+RB−d`. 커널은 `δ>0`일 때 접촉력을 계산한다. 구–메시는 구와 삼각형의 접촉점/법선으로 겹침을 계산한다. 메시 쪽 곡률 반경에는 코드상 큰 값이 들어가 평면 극한에 대응한다.

접촉점 상대속도:

`vrel = (vA + ωA×rA) − (vB + ωB×rB)`

여기서 r는 각 **몸체 중심→접촉점** 벡터이고, 각속도/벡터는 같은 좌표계로 변환한다. 법선 n(B→A)에 대해:

`vn = vrel·n`, `vt = vrel − vn n`.

**입사 방향은 바로 이 법선/접선 분해로 영향을 준다.** 단일 ‘입사각=몇 도’ 상수를 넣은 것은 아니다. 접촉마다 n·vrel이 달라진다. 필요하면 `atan2(|vt|,|vn|)` 같은 각도를 정의할 수 있지만 접근/이탈 구분과 모든 접촉 데이터 저장이 필요하며 이번에 그 전 구간 분포를 측정하지 않았다. 더미 표면 경사인 안식각과도 다르다.

### 5.3 유효 물성

`1/E* = (1−νA²)/EA + (1−νB²)/EB`

`1/G* = 2(2−νA)(1+νA)/EA + 2(2−νB)(1+νB)/EB`

`R* = RA RB/(RA+RB)`

`m* = mA mB/(mA+mB)`

여기서 mA,mB는 구 부품별 1/7 질량이 아니라 **구가 속한 owner 강체 전체의 질량**이다. 구의 반경과 owner 질량/MOI를 함께 쓰는 구현임을 명시해야 한다. 서로 같은 입자 재료면 E*=2.7472527MPa, G*=0.5656109MPa, 입자–공구(E3GPa) 접촉 E*=5.4853632MPa다. μ/e/Crr는 material-pair 표에서 읽으며 현재 로드한 재료들의 값은 동일하다.

### 5.4 법선 탄성 + 감쇠

`β = ln(e)/sqrt(ln(e)²+π²)`  (e=.3이면 β<0)

`Sn = 2 E* sqrt(R*δ)`

`kn = (2/3) Sn`

`γn = 2 sqrt(5/6) β sqrt(Sn m*)`

`Fn = (kn δ + γn vn) n`

탄성 부분만 쓰면 익숙한 Hertz 식 `Fn,elastic = (4/3)E*sqrt(R*)δ^(3/2)`이다. 감쇠항은 상대속도와 반발계수에 의존한다. β가 음수라는 부호 규약까지 포함해야 한다. 구현은 δ>0 분기와 위 식을 쓰며 별도의 접착력 모델은 연결하지 않았다. 접촉 변형량을 작은 겹침으로 표현하는 **soft-contact**이지, PP 입자 형상 자체를 유한요소로 찌그러뜨리는 모델은 아니다.

### 5.5 접선 마찰 + 이력

접촉이 지속되는 동안 접선 변위 이력 ξ를 저장한다:

`ξ ← ξ + dt vt`, 이어 `ξ ← ξ − (ξ·n)n`

`kt = 8 G* sqrt(R*δ)`

`gt = −2 sqrt(5/6) β sqrt(m*kt)`

`Ft,trial = −kt ξ − gt vt`

`|Ft| ≤ μ |Fn|`가 되도록 크기를 제한하고, 미끄러진 경우 ξ도 그 제한된 힘과 일치하도록 되계산한다. 접촉이 사라지면 이력과 접촉 경과시간을 0으로 한다. 즉 지금 위치만이 아니라 **이전에 얼마나 접선 방향으로 움직였는지**가 다음 힘에 영향을 준다. 저장 위치만으로 임의 재시작하면 이 이력이 빠질 수 있다.

### 5.6 구름저항

설치 커널은 접촉 유지시간에 따른 활성화 조건을 거친 뒤 회전 상대속도의 방향으로 `Crr·|Fn|` 크기의 **torque-only equivalent force**를 만든다. 이 값은 병진 합력에 더하지 않고 접촉점의 지렛팔과 외적하여 회전 저항에만 사용한다. 일반적인 설명용 `τ≈μr R Fn`을 설치 구현의 정확한 코드라고 대체하지 않는다.

활성화 식의 정확한 원문은 `FullHertzianForceModel.cu:69`부터다. 특히 그 내부 `R_eff` 변수는 `sqrt(RA*RB/(RA+RB))`로 정의되어, 위 법선 식의 일반적인 R*와 이름/차원을 혼동하기 쉽다. 이번 보고는 이 코드를 임의 수정하거나 이 식만으로 안정 dt를 도출하지 않았다. 회전 저항 warm-up 구현을 따로 검증할 후속 항목으로 남긴다.

### 5.7 합력·토크·관성·시간 적분

병진은 `a = g + Σ(Fn+Ft)/m`이다. 설치 force 수집 커널은 힘을 owner별로 합하고, 힘을 몸체 국소좌표로 바꾼 뒤:

`αcontact = Σ[r_local × (F_local + Frolling,torque-only_local)] / (Ixx,Iyy,Izz)`

를 성분별로 계산한다. 따라서 **회전 관성이 실제로 들어가며, 없다는 주장은 틀리다.** 다만 ‘일반 강체 Euler 방정식의 모든 항을 다 검증했다’고 확대하면 안 된다. 확인한 수집/적분 기본 경로는 이 τ/I 및 각속도 갱신이며, 일반식 `I ωdot + ω×(Iω)=τ`의 자이로 항을 별도로 확인하지 못했다. 비구형 자유회전의 정확도는 별도 시험으로 검증해야 한다. 지금 결과의 원인이 그 항이라고 단정하지 않는다.

설치 기본은 `EXTENDED_TAYLOR`이고 이번 프로젝트 소스에 변경 호출이 없다. 기본 속도 전달 정책은 중간값을 사용한다:

`v_new = v_old + a dt`

`x_new = x_old + (v_old + 0.5 a dt) dt`

`ω_new = ω_old + α dt`, `ωmid = ω_old + 0.5 α dt`

quaternion(xyzw 표기)에는 `[0.5dt·ωmid, 1]` 회전 증분을 오른쪽으로 곱한 뒤 정규화한다. 위치 저장은 voxel/subvoxel 표현을 거치며 h·힘·속도 등의 내부 계산에는 float32가 사용된다. API 배열이나 결과 JSON이 float64라고 전체 솔버가 float64인 것은 아니다. 이 의미에서 ‘그냥 전진 Euler’ 또는 ‘모두 double precision’이라는 설명도 부정확하다.

### 5.8 이번 모델에 없는 것 / 아직 검증 안 한 것

- 별도 점착·정전기·유체장·공기 drag/lift·부력 모델 없음.
- PP의 실제 점탄성 constitutive model, 소성변형, 열/온도 의존성, 파손·마모, 펠릿끼리의 접착/분리 모델 없음.
- 모든 알은 동일 템플릿·질량. 실물 입도·형상·물성 분포 보정 없음.
- W10/W11 전 알의 선/각속도·가속도 전 스텝 이력을 모두 보존하지 않았다. 내부에 운동 계산이 있다는 것과 그 전체 이력이 파일에 있다는 것은 다르다.
- W13 저장 프레임은 추가 속도/각속도를 포함하지만 내부 백만Hz 전체 기록은 아니다. 저장 최대값을 내부 매 스텝 최대값이라고 바꾸어 말하지 않는다.

## 6. Isaac 이미지·영상은 정확히 어떻게 그렸나

### 6.1 역할 분리

`DEME의 접촉/운동 계산 → 저장된 NPZ 상태 → Python 좌표/관절 변환 → Isaac의 USD 장면·카메라 RGB → PNG/MP4`

반대 방향 힘 전달은 없다. 따라서 **DEME+PhysX PBD 양방향 하이브리드가 아니라 DEME 물리의 Isaac 상태 재생**이다. PhysX는 물리 계산 엔진이고, 화면 자체는 Isaac/Omniverse 렌더러가 그린다.

W9는 W8의 154알 포획 결과를 재생한 것이다. W12가 W10의 541알과 W11의 517알 결과를 각각 동일 장면으로 비교했다. W13 영상은 다른 전체 사이클을 시도한 부분 실패 기록이다.

### 6.2 펠릿 표시 모델

W12는 실제 템플릿의 각 구를 `icosphere(subdivisions=1)` 메시로 만들어 동일 offset에 놓고 연결한다. 7개의 겹친 메시를 concatenate한 **294정점/560삼각면 prototype**이다. 물리의 해석 구와 달리 낮은 다각형의 표시 표면이다. Boolean union으로 내부 삼각면을 제거한 단일 매끈 표면이라고 하면 안 된다.

W12는 색 그룹별 PointInstancer 3개가 각 펠릿의 위치/회전을 저장 프레임으로 갱신한다. W13은 PointInstancer 1개 안의 색 prototype6개를 사용하고 각 instance의 protoIndices를 바꾼다. 두 방식 모두 20,000알의 독립 위치/회전이다.

위치는 float32, USD orientation은 `Gf.Quath(w,x,y,z)`로 넘긴다. 원자료 xyzw를 wxyz 인자로 순서를 바꾼 것이며 Quath는 half-precision 표현이다. 표시 정밀도를 과학 판정의 정본으로 역산하지 않는다. 추가 instance scale은 쓰지 않는다.

### 6.3 발표 이미지에서 실제 사용한 표시 설정

| 항목 | W12: W10/W11 비교 | W13 부분 영상 |
|---|---|---|
| 카메라별 해상도 | 1024×640 | 1600×900 |
| 카메라 | 사선/상부 2대 | 사선/상부 2대 |
| 초점거리 authoring 수치 | 각40 (기존 key `focal_mm`, 단위명 검증 필요) | 각20 (같은 주의) |
| 수평 aperture authoring 수치 | 설치 기본20.955 | 설치 기본20.955 |
| clipping | 0.05–6m | 0.05–8m |
| 카메라 위치 | 아래 좌표, 거리1.05m | 전체 표시 경계 계산으로 결정, manifest에 좌표 저장 |
| dome / distant light | 1500 / 2500 (USD/Isaac 설정 수치) | 동일 수치 |
| 펠릿 PreviewSurface roughness | 0.6 | 0.6 |
| 색 | 나머지 베이지 / carried 보라 / captured 주황 | source/bin/tool/spill/in-flight/ambiguous 6분류 |
| 동영상 저장 | 10fps, 각64프레임=6.4s | 10fps,283프레임=28.3s |
| 실제 원자료 시간 | 약3.14s | 24.4868s |

W12 카메라 side `(0.875,−0.525,0.950462)m`, top `(0.35,0,1.258)m`, target `(0.35,0,0.208)m`. 표시 좌표는 `x_display=x_DEME+(0.35,0,0.163)m`로 옮겼다. W13은 origin `(0.350013257,−0.0081,0.218819228)m`이며 같은 원점을 무조건 재사용하지 않는다. W13 정확한 두 카메라 위치/목표는 이번 JSON의 `render.w13.cameras`에 보존했다.

조명값은 실측 lux 보정값으로 제시하지 않는다. tone mapping·노출·RTX/DLSS의 전체 런타임 설정 덤프는 이번 manifest에 없으므로 모든 픽셀 설정을 완전히 복구했다고 하지 않는다. 코드가 명시한 설정, 설치 기본값, 실제 runtime 기록을 구분한다. 간단한 pinhole 관계 `u=fx X/Z+cx`, `v=fy Y/Z+cy`는 카메라 투영 설명이며 Hertz 힘 계산식이 아니다.

**이번 교차 검토의 단위 정정:** 기존 프로젝트의 `focal_mm` 이름만으로 ‘실제40mm 렌즈’라고 하면 안 된다. 설치 spawner `sensors.py:133`은 cfg 수치를 USD 속성에 변환 없이 넘긴다. 설치 `camera.h:85` 및 [OpenUSD Camera Units](https://openusd.org/release/api/class_usd_geom_camera.html)는 focal/aperture를 scene unit의1/10로 규정한다. Isaac Lab2.3.0 문서에는 cm/mm 설명이 혼재하고 `simulation_context.py:280`은 stage_units_in_meters=1.0을 전달한다. 따라서 이번 보고는 원 authoring 수치와 `fx=W*f/aperture` 비율을 정본으로 설명한다. W12 센서에 실제 기록된 fx=fy=**1954.6650390625px**, cx512/cy320도 그 비율과 일치한다. 두 길이의 단위를 같은 비율로 잘못 표기해도 pinhole 화각은 같을 수 있으므로, 이 단위명 문제가 기존 영상의 시야가 틀렸다는 판정은 아니다. 물리 렌즈 치수·심도 보정은 별도다. 설치 기본 f_stop=0으로 초점 흐림은 꺼지고 focus_distance=400 설정은 실측 초점거리가 아니다.

W12의 주황/보라는 **최종 captured/carried ID를 처음부터 강조한 색**이다. 그 순간의 힘/온도/접촉 압력 색이 아니다. 초기 프레임 주황색을 ‘이미 포획됨’이라고 해석하면 틀린다. W13은 저장 재고 라벨을 표시하므로 원자료 분류 결함도 같이 해석해야 한다.

### 6.4 로봇·PhysX가 어디까지 관여했나

- W12는 저장 tool lip 목표에서 Python IK로 팔 자세를 구하고, 문은 저장 quaternion에서 복원해 관절 상태를 직접 기록한다. Isaac scene `step(render=True)` 자체는 실행하지만 **펠릿은 plain PointInstancer이며 PhysX particle/rigidbody 접촉을 다시 풀지 않는다.**
- W13 post03은 초기 warm-up에서 scene step을 쓰고, 실제 표시 프레임 루프는 `sim.render()`만 사용하도록 수정되어 있다. W12와 똑같은 scene-step 절차라고 설명하면 안 된다.
- 표시용 팔 actuator stiffness/damping(팔2000/100, 문500/30) 등은 Isaac 표시 장면 구성값이지 DEME의 PP 물성이나 실제 서보 PID가 아니다.
- W13 코드의 ‘실측 툴 포즈’라는 주석은 **DEME에서 읽은 실제 상태**를 뜻한다. 실물 로봇 측정자료를 영상에 사용했다는 의미가 아니다.

정확한 구현: [W12 renderer](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/sim_isaac_replay_w12.py:266), [W13 post03 renderer](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/post_03_rev/src/isaac_replay_w13.py:877), [W12 runtime gate](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/w10/gates_w12_w10.json), [W13 runtime manifest](/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/partial_post_03/isaac/render_manifest.json).

## 7. PBD를 안 쓴 이유를 다시 정확히 정리

‘개별 펠릿이 렌더링 안 된다’가 아니었다. 당시 검증한 PhysX PBD 설정에서 더미가 관측창 동안 정착하지 않았고 관측시간을 늘리자 표면각도도 바뀌었다. 같은960Hz의20초/40초 관찰 결과20.759768°→16.655137°가 그 근거다. PBD에 질량이 없다는 뜻도, 모든 PBD 설정이 불가능하다는 뜻도 아니다. 원자료와 버전 대조는 [기존 물리 감사](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/research/labmeeting_20260915/physics/REPORT.md)에 있다.

NVIDIA 공식 문서도 granular PBD와 입자 질량, Points/PointInstancer 표현을 설명한다. 따라서 **Isaac에 입자를 잘 그렸다는 사실만으로 PBD 더미 정착 문제가 해결된 것은 아니다.** DEME로 정착하고 PBD로 바꾸거나 두 엔진을 함께 물리 계산에 쓰려면 입자 표현·질량·속도/회전·접촉 이력·힘 전달·중복 계산·경계 연속성을 새로 검증해야 한다. 현재 구현도 이번 작업 범위도 아니다.

## 8. 설치 버전과 공식 문서·로컬 증거

| 공식 자료 | 적용 버전·URL | 로컬 대조 |
|---|---|---|
| Project Chrono Full Hertzian Force Model | [고정 commit 12f13cb의 접촉식](https://github.com/projectchrono/DEM-Engine/blob/12f13cb15805d891eddc7b5b545d1f6823f523d7/src/kernel/DEMCustomizablePolicies/FullHertzianForceModel.cu) | 설치 deme2.4.0의 `share/DEME/kernel/DEMCustomizablePolicies/FullHertzianForceModel.cu:5` 이하. 공식8파일 SHA 일치; 이 commit을 v2.4.0 tag라 부르지 않음 |
| DEM-Engine API / integration policy | [API](https://github.com/projectchrono/DEM-Engine/blob/12f13cb15805d891eddc7b5b545d1f6823f523d7/src/DEM/API.h), [적분 정책](https://github.com/projectchrono/DEM-Engine/blob/12f13cb15805d891eddc7b5b545d1f6823f523d7/src/kernel/DEMCustomizablePolicies/IntegrationVelPassOnExtendedTaylor.cu) | 설치 `include/DEM/API.h:1558`, `Models.h:194`, `DEMIntegrationKernels.cu:155`, `DEMCollectForceKernels_Compact.cu:35` |
| Particles — Omni Physics | [107.3 공식 문서](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/particles/particles.html) | 설치 `omni.physx-107.3.26.../config/extension.toml:5`; 문서는107.3 계열, patch26의 개별 실험 결과는 로컬 자료가 근거 |
| UsdGeomPointInstancer — USDRT | [7.6.1 공식 API](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_usd_geom_point_instancer.html) | 설치 `usdrt.scenegraph-7.6.1.../include/usdrt/scenegraph/usd/usdGeom/pointInstancer.h:55`·`:111`·`:400`. 렌더 코드는 pxr.UsdGeom을 쓰며 이 문서는 동일 instancing 구조의 보조 대조. 과거 보고의7.5.1 mismatch는 이번에7.6.1 페이지로 보완 |
| Isaac Lab Camera/Simulation source | [Camera v2.3.0](https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/sim/spawners/sensors/sensors_cfg.html) | 설치 `isaaclab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py:63`, `simulation_cfg.py:220`. W12 runtime은 Sim5.1.0-rc.19+release.26219.9c81211b.gl, pip5.1.0.0과 구분 |

DEME 설치 경로 prefix: `/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/`. Isaac 설치 prefix: `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/`. 링크/해시 전수는 PARAMETER_AUDIT/OFFICIAL_SOURCE_CROSSCHECK에서 다시 확인할 수 있다. NVIDIA UI authoring 범위나 SDK hard limit을 본 프로젝트의 임의 파라미터와 혼동하지 않았다.

## 9. Git 보류와 다음 세션

앞선 게시 명단에서 `lfs:true`인 **1,016개 경로, 중복 제거982개 객체,3,316,053,800bytes(3.0883GiB)**만 보류했다. 5개 worktree의 `.gitignore`에 exact path를 추가하고 `git rm --cached`로 index에서만 제외했다. 모든 실제 파일 해시가 그대로이고 각 branch HEAD도 그대로다. 작업복사본 삭제·이동·과거 커밋 수정·commit·push는 하지 않았다.

전체 명단: [LFS_DEFERRED_20260916.md](../../LFS_DEFERRED_20260916.md). 사전/사후 증거 [LFS_BEFORE.json](LFS_BEFORE.json), [LFS_AFTER.json](LFS_AFTER.json). 현재 staged `D` 표시는 **추적 제외**이지 디스크 원본 삭제가 아니다. 코드/텍스트/기존 JSON은 포괄 ignore하지 않았다.

**과거 커밋에는 LFS 포인터가 남아 있어 그대로 push하면 과거 객체 업로드가 여전히 필요할 수 있다.** .gitignore는 기존 이력을 지우지 않는다. 공개 범위/계정 용량 확인과 게시 이력 전략 승인 전 push 보류다. 일부 과거 전송 객체 존재 여부·계정 잔여 용량은 아직 미확인이다. 기존 force-add 게시 스크립트를 재실행하지 않는다. [Git ignore 공식 의미](https://git-scm.com/docs/gitignore), [index만 제외하는 git rm --cached](https://git-scm.com/docs/git-rm), [GitHub LFS 제거와 이력/저장 한계](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage).

새 세션: [CONTINUE_20260916_PHYSICS_AUDIT_DT.md](../../CONTINUE_20260916_PHYSICS_AUDIT_DT.md). 원래 첫 case와 교수 dt 한계 비교를 섞지 않고 단계적으로 이어간다. 아직 새 물리/형상/PBD 하이브리드 실험을 실행하지 않았다.

## 10. 이번 교차 검토의 완료 범위

1. 이전 continuation에서 원래 순서를 확인했다.
2. 원 NPZ 템플릿/ID를 직접 읽어20,000×7, 외곽치수, 질량/MOI를 재계산하고 W10/W11/W13 result mirror와 대조했다.
3. W10/W11 effective params와 과거10µs 설정/로그/엔진 예외를 대조했다. 10µs 실패가 이미 존재함을 확인했다.
4. 공식 고정 commit8파일을 설치본과 바이트 대조하고 Hertz·이력 마찰·관성·적분 코드 경로를 읽었다. 물리 모델 개선/자유회전 실험은 하지 않았다.
5. W12/W13 실제 renderer와 runtime manifest를 비교해 형상·색·카메라·시간·scene step 차이를 확인했다. 새 영상 시청/렌더/공간 판정을 수행했다고 주장하지 않는다.
6. LFS 명단 원본 전수 해시·ignore·index·HEAD를 사후 검사했다. 문서/CPU 감사라 이번 연구 세션의 새 실험·RRD는 생략하며 사용자 ‘새 세션에서 실제 작업’ 요청이 사유다.

최종 요약: **원래는 판정 버그부터 수리한다. 교수님 dt 확대 제안은 의미 있지만10µs는 이미 실패했고1ms는 제어 cadence도 충돌한다. 펠릿은 이미 한 알씩 독립 강체이며7구는 한 알의 근사 형상이다. DEME가 물리를 계산하고 Isaac은 그 결과를 그린다. 실물 보정과 형상/시간 간격 수렴은 아직 남아 있다.**
