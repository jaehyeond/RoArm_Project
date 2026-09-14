# 9월 15일 랩미팅 — DEME 물리와 Isaac 재생의 역할, 펠릿 모델과 비교 결과

이번 case의 신규 변수: []. 원본 PPT를 고치는 작업이 아니라 기존 증거를 검토하여 발표 문안을 제공하는 작업이다.

## 1. 무엇을 설명하려는 발표인가

**핵심 문장:** “입자의 접촉·이동·회전은 DEME에서 계산하고, 저장된 위치와 자세를 Isaac Sim의 로봇 장면에서 재생했다. 실물 치수를 참고한 7구 렌즈형 펠릿을 사용했지만 재료 물성 보정과 전체 운반·배출 성공 검증은 아직 끝나지 않았다.”

초안은 6장이고 표지가 **9월 7일**이다. 본문의 **PBD 구형 8,000알·명목 194.4 g·포획 250알/6.075 g·18초 슬라이드쇼**는 현재 W10/W11의 결과가 아니다. 초안 notes가 가리키는 원자료는 Windows `posco-pilot` 경로로, 이번 검토에서 확보하지 못했다. 이 수치가 틀렸다고 판정한 것은 아니다. “이전 PBD 시험”으로 부록에 보관하고, 9/15 본문은 아래의 검증된 DEME/Isaac 자료로 구성하는 것이 적절하다. 원본 PPT/미디어를 삭제하거나 이동하지 않았다.

## 2. 실제 검토한 순서

1. 메인 상태 정본·재개/종료 세션·원자료를 읽어 과거 실물 자세와 현재 시뮬레이션 상태를 분리했다.
2. 실제 Orca worktree 세 곳에 분담했다. `research-survey`의 Claude Opus 5는 초안과 발표 구성, `pellet-model`의 Codex gpt-5.6-sol/high는 물리·모델, `w12-isaac-replay`의 같은 Codex는 렌더링·영상 출처를 검토했다.
3. 메인은 PPT XML, PBD 원 JSON, canonical 펠릿 NPZ, W10/W11 포획 mask와 질량을 독립 재계산했다. 설치된 DEME 접촉·관성·적분 코드와 W12 재생 코드도 직접 읽었다.
4. 기존 W12 lift/first_close/reclose PNG를 직접 열고 저장시점·주석·표시 한계를 확인했다. MP4는 SHA와 ffprobe로 확인했으며 이번 검토에서 영상 전체를 연속 시청한 것은 아니다.
5. 발표 워커 초안의 “현재 구 1개·치수 미측정” 오류와 평균높이만으로 한 물리 모순 추론을 차단했다. 과거 기본 코드 주석 대신 **실제 실행 NPZ·9/10 측정 JSON**을 근거로 정정 요청했다. 자세한 독립 검토는 [ROOT_CROSS_REVIEW.md](ROOT_CROSS_REVIEW.md).

## 3. 발표에 바로 쓸 8장 구성

### 1장 — 특별주제 연구, 9월 15일

본문:

- 펠릿 퍼내기 시뮬레이션: 물리 모델과 관찰 화면의 역할 분리
- PhysX PBD 시험에서 DEME 입자 물리 + Isaac Sim 재생으로 전환
- 이번 보고 범위: 펠릿 모델, 시간간격 비교, 결과 영상, 남은 검증

발표 노트: “더미를 퍼낸 양과 남은 모양을 예측하려면, 입자 모델과 계산 설정이 결과에 미치는 영향을 먼저 구분해야 합니다. 이번에는 그 기반과 관찰 경로를 확인했습니다.”

### 2장 — 왜 PBD 대신 DEME를 사용했나

본문:

- 시험한 PBD 설정에서는 관측 시간 안에 정지한 더미를 확보하지 못했다.
- 같은 960 Hz 설정에서도 관측창을 길게 하면 형상 경사가 **20.76° → 16.66°**로 변했고, 두 실행 모두 정착 판정에 미달했다.
- 따라서 이번 연구의 입자 물리를 DEME로 옮겼다. **PBD 전체가 불가능하거나 질량이 없다는 뜻은 아니다.**

발표 노트: “시간간격과 관측시간이 함께 달라진 실험도 있어 시간간격만의 효과라고 단정하지 않았습니다. 현재 기록은 안정된 실물 안식각을 얻었다는 증거가 아닙니다. PBD 상태를 읽을 수 없어서 포기했다는 과거 설명도 철회됐습니다.”

근거: [D467/D469 정본](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/DECISIONS.md:29189), [960 Hz 짧은 관측](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pbd_probe/cell_dt_960hz.json:285), [960 Hz 긴 관측](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/claudedocs/runtime_logs/pbd_probe/cell_ls_960hz_damp0.json:355). 이 프로젝트 probe의 10,000알과 초안의 Windows PBD 8,000알을 같은 실행으로 섞지 않는다.

### 3장 — 펠릿 한 알과 초기 더미는 어떻게 만들었나

본문:

- 실제 펠릿을 늘어놓은 사진과 10알 적층 관찰에서 **약 4.5×3.8×2.5 mm의 납작한 렌즈/알약형**으로 모델링했다.
- 한 알은 **중앙 구 1개 + 주변 구 6개를 고정한 하나의 강체**다. 7알이 따로 움직이는 것이 아니다.
- 이 강체 **20,000알**을 층 형태로 배치하고 자세를 무작위화한 뒤 중력으로 안정화했다. 접촉 형상은 총 140,000개의 구다.

발표 노트: “구끼리 겹치는 부피는 중복으로 더하지 않고 합집합으로 계산했습니다. 모형의 실제 외곽은 약 4.501×3.601×2.500 mm이므로 목표 치수와 완전히 같지는 않습니다. 정밀 스캔 CAD가 아니라 실물 치수를 참고한 근사 모델입니다.”

더미 생성은 `target_fcc/slab`, seed460의 격자 후보 선택·작은 위치 흔들림·무작위 회전 뒤 정착이다. 깔때기로 자유 낙하시켜 안식각을 측정한 더미라고 말하지 않는다. 저장 설정의 28°는 일반 생성기의 크기 산정용 항목이며 **이번 slab 분기에서는 사용하지 않는다**.

근거: [9/10 측정 기록](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/pellet_model/pellet_measured_20260910.json:25), [형상 생성기](/home/cgxr/orca/workspaces/RoArm_Project/pellet-model/sim_pellet_model.py:518), [실제 실행 template](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/cell_DE_dt2e6_c/scoop_s1_seed460.json:101).

### 4장 — 관성·입사각은 없는 것인가

본문:

- **질량과 회전 관성은 있다.** 접촉 힘·토크에 따라 병진 운동과 회전 자세를 계산한다.
- 중력, 탄성 접촉, 감쇠, 미끄럼 마찰, 반발, 구름저항을 포함했다.
- **입사 방향도 계산에 반영된다.** 다만 하나의 입사각 숫자를 고정하거나 모든 충돌의 각도를 저장·검증한 실험은 아니다.

발표 노트: “입사각은 상대 운동이 접촉면 법선과 이루는 각도입니다. 코드에서는 회전까지 포함한 접촉점 상대속도를 법선 방향과 접선 방향으로 나눕니다. 따라서 비스듬히 부딪히는 효과를 무시하지 않습니다. 관성도 코드에 넣어 놓기만 한 값이 아니라 토크에서 각가속도를 계산하고 자세를 갱신하는 데 사용됩니다.”

**안식각은 다른 말이다.** 안식각은 더미가 안정된 뒤의 경사다. 이 펠릿의 실물 안식각 보정은 미완료이며, 퍼낸 구덩이 벽의 경사를 안식각으로 바꾸어 부르지 않는다.

**포함과 검증은 다르다:** 밀도·마찰·반발·강성은 아직 실물 시험으로 맞춘 값이 아니다. 공기저항·유체 부력·점착·파손·소성변형은 이번 실행에 모델링하지 않았다. 강체 형상에 Hertz 접촉 겹침으로 힘을 계산하는 것이며 입자 표면의 실제 변형을 그린 것이 아니다.

근거: [접촉의 방향 분해](/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/share/DEME/kernel/DEMCustomizablePolicies/FullHertzianForceModel.cu:20), [질량·관성으로 가속도 계산](/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/share/DEME/kernel/DEMCollectForceKernels_Compact.cu:35), [각속도·회전 자세 갱신](/home/cgxr/miniconda3/envs/roarm/lib/python3.11/site-packages/share/DEME/kernel/DEMIntegrationKernels.cu:173).

### 5장 — DEME 계산 결과를 Isaac에서 어떻게 렌더링했나

본문:

**DEME 접촉·운동 계산 → 위치·회전 자세 저장 → Isaac 장면에 같은 형상 배치 → 카메라 화면 저장·영상 인코딩**

- 물리 모델과 같은 7개 구의 중심·반지름으로 삼각형 메시 원형을 만든다.
- 이를 복제 표시하는 `PointInstancer`로 20,000알을 배치하고, 매 저장 시점의 위치와 회전 자세를 입력한다.
- 로봇 팔은 저장된 도구 위치에 맞는 관절 자세로 표시한다. **실시간 양방향 물리 결합은 아니다.**

발표 노트: “Isaac을 안 쓴 것이 아니라 역할을 나눴습니다. 입자가 어디로 움직였는지의 원자료는 DEME이고, Isaac은 그 결과를 로봇 장면에서 보여 줍니다. W12에는 Isaac 장면 갱신 step도 있지만 이 입자들을 PhysX PBD로 다시 풀거나 계산한 힘을 DEME로 되돌리지는 않습니다. 팔이 화면에 움직인다는 사실만으로 실제 서보가 그 운동을 실행할 수 있다고 검증한 것도 아닙니다.”

메시 한 알은 294정점/560삼각형이다. 매끈한 실제 펠릿 표면과 정확히 같지 않으며, 접촉용 해석적 구와 표시용 삼각형 표면도 서로 다른 표현이다. 색은 **최종 포획 ID를 전 프레임에 고정해 강조한 것**이지 매 순간의 포획 판정 색이 아니다.

근거: [W12 prototype·입자 상태 입력](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/sim_isaac_replay_w12.py:320), [관절 상태·프레임 갱신](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/sim_isaac_replay_w12.py:364).

### 6장 — W10/W11: 시간간격만 줄이면 결과가 같나

| 항목 | W10 | W11 |
|---|---:|---:|
| 물리 시간간격 | 2 μs | 1 μs |
| 최종 도구 내부 포획 | 541알 | 517알 |
| 설정 질량으로 환산 | 10.9592 g | 10.4731 g |
| 저장 sync의 최대 입자 속도 | 5.3291 m/s | 2.7255 m/s |
| 5 m/s 초과 기록 | 1회 | 0회 |
| 재닫기 정지 사유 | `servo_stall` | `pinch_guard` |

발표 노트: “같은 초기 더미·형상·물성·제어를 사용하고 시간간격만 바꿨습니다. 저장된 과도 속도 경고는 줄었지만 포획 수와 정지 사유도 달라졌습니다. 각 조건을 한 번씩만 실행했으므로 시간간격 수렴이나 실물 정합을 증명했다고 말하지 않습니다. 표의 그램은 저울값이 아니고 도구 내부 포획이지 컵 배출량도 아닙니다.”

`servo_stall`은 이번 모형의 서보 정지 조건, `pinch_guard`는 물림 보호 조건이다. 실제 서보에서 재측정한 정지 사건이 아니다. 5 m/s는 기존 기록의 진단 경계이며 이번 발표에서 새 물리 수락 기준으로 만들지 않았다. 기록 간격보다 짧은 내부 step의 최대속도까지 입증한 것은 아니다.

근거: [W11 실제 비교](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/comparison.json:338), [메인 원자료 재계산](ROOT_NUMERIC_CROSSCHECK_01.json).

### 7장 — 같은 Isaac 장면의 비교 영상

사용할 주 영상: [W12 W10/W11 나란히 비교 MP4](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/paired_w10_w11_sourcetime_annotated.mp4).

붙일 캡션:

> 위: W10(dt=2 μs), 아래: W11(dt=1 μs). DEME가 저장한 위치·회전 자세를 같은 Isaac 장면에서 재생. 64개 최근접 시간쌍, 보간 없음. 색은 최종 포획/운반 ID의 고정 강조 표시.

2048×1630, 64프레임, 10 fps, 재생길이6.4초. 영상의 재생 시간은 물리 시간이나 계산 소요시간과 다르다. 두 기록의 대응 시각 차이는 최대3.077 ms다.

정지화면 권장: [상승 시점](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/montage_lift_annotated.png), [최종 상태](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/montage_final_annotated.png). 진단용 영문이 작으므로 본문 설명은 위 캡션으로 짧게 한다. 원본을 새로 편집·자르지는 않았다.

주의: `first_close` 그림은 각 실행에서 정지에 가장 가까운 프레임이라 W10은 정지 뒤, W11은 정지 전이다. `reclose`와 `final`은 각 실행의 같은 마지막 PF63이며, **재닫기 도중의 연속 입자 프레임은 저장되지 않았다**. 재닫기 전체를 복원했다고 발표하지 않는다.

### 8장 — 현재 확인한 것과 남은 것

본문:

- 확인: 실제 7구 모델과 질량·관성 사용, W10/W11 수치 차이, Isaac 저장 상태 재생 경로.
- 미확인: 실제 PP 물성 보정, dt 수렴, 실제 로봇 동역학 정합, 운반·컵 배출까지의 전체 성공.
- W13 전체 사이클은 **시간 초과 부분 종료**이며 성공 영상으로 사용하지 않는다. 다음은 데이터·재생 결함 수정 범위와 운반 보유 실패 원인 조사 범위를 먼저 정해야 한다.

발표 노트: “포획된 양을 컵에 안정적으로 옮기는 것은 별도 문제입니다. W13은 약8시간40분 실행했지만 전체 HOME/정착을 완료하지 못했고 원자료 규약2건·재생3건 결함이 남았습니다. 지금 결과를 학습이나 A/B/C 전략 비교로 바로 확대하지 않습니다.”

장기 순서(방향 제안, 이번 실행 승인 아님): 동일 도구·경계의 전체 동작을 검증 → 계량 기준과 물성·높이맵 보정 → 고정 기준선의 포획량/잔여형상 평가 → 그 이후 취점 선택 학습. 실물 재개·새 장시간 실행·학습·A/B/C는 별도 승인이다.

6장으로 줄일 때: 표지+목적을 짧게 하고, 6장 수치와 7장 영상을 한 장으로 합친다. 사용자 질문의 중심인 **펠릿 모델·관성/각도·Isaac 방식은 유지**한다. W13 상세와 과거 PBD 초안은 부록으로 보낸다.

## 4. 질문 대비 숫자와 근거 파일

### 펠릿 물성 — W10/W11 실제 입력

| 항목 | 사용한 값 | 지위 |
|---|---|---|
| 목표축 a/b/c | 4.5/3.8/2.5 mm | 사진 기반 추정 ±0.2/±0.2 mm, 10알 적층 관찰 ±0.05 mm; 입자별 분산 실측 아님 |
| 실제 합집합 외곽 | 4.500704/3.601263/2.500391 mm | 생성기/NPZ에서 계산한 근사형상 |
| 합집합 부피 | 22.383848 mm³ | 고정 seed Monte Carlo 적분 후 목표 타원체 부피에 맞춤; 7구 중복 합산 아님 |
| 밀도 | 905 kg/m³ | 가정값; 실물 등급·밀도 미보정 |
| 알당 질량 | 0.020257382 g | 부피×가정 밀도, 저울 측정 아님 |
| 대각 관성 Ixx/Iyy/Izz | 2.06277 / 2.75702 / 3.48405 ×10⁻¹¹ kg·m² | 합집합 기하에서 계산해 `LoadClumpType`으로 전달 |
| E / ν | 5 MPa / 0.30 | scoop 실행 설정, 실물 탄성 보정 아님 |
| μ / Crr / CoR | 0.45 / 0.06 / 0.30 | 마찰/구름저항/반발 임시 설정 |
| 중력 | (0,0,−9.81) m/s² | 포함 |

초기 더미 생성 단계는 **dt20 μs·E10 MPa**이고, W10/W11 퍼내기 단계의 **dt2/1 μs·E5 MPa**와 다르다. “처음부터 모든 단계가 같은 dt·강성”이라고 설명하지 않는다.

회전은 물리에 포함되지만 보존된 scoop 재생 자료는 전 알 위치·quaternion 중심이다. 전 입자 속도벡터·각속도·가속도 전체 이력은 저장하지 않았다. 없어진 기록을 영상에서 정밀 복원했다고 주장하지 않는다. 초기 더미의 속도 요약 두 필드 차이는 `ROOT_CROSS_REVIEW.md §3`에 보존하며 발표 숫자로 쓰지 않는다.

### 번호와 자산의 정확한 계보

| 번호 | 물리/표시의 역할 | 섞으면 안 되는 것 |
|---|---|---|
| W8 → W9 | W8 DEME 퍼내기 결과 154알을 W9에서 Isaac 재생 | W9를 W10의 영상이라고 부르지 않음 |
| W10 | dt2 μs의 DEME 실행, 포획541알 | W8과는 문속도·하한·코드도 달라 단일변수 비교 아님 |
| W11 | W10 대비 dt1 μs만 바꾼 실행, 포획517알 | 단일 dt 민감도 쌍, 수렴 아님 |
| W10/W11 → W12 | 두 결과를 같은 Isaac 장면에서 비교 재생 | 새 입자 물리 실험 아님 |
| W13 | 별도 전체 운반·배출 경로 시도, TIMEOUT 부분 결과 | 완전 사이클 성공·정착 배출량으로 승격 금지 |

재사용 전체 자산의 절대경로·해시·시점·캡션: [render/assets.json](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/research/labmeeting_20260915/render/assets.json). W9 영상은 계보 설명용, W12 paired 영상은 이번 비교의 주 자산이다. W13 파일명 `w13_full_cycle.mp4`는 성공 여부를 보증하지 않는다.

### 공식 문서와 설치판 연결

- **Particles — Omni Physics 107.3**: <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/particles/particles.html>. 설치 PhysX Core107.3.26의 같은107.3 계열 문서다. PBD granular 및 질량 지원을 확인했고 [설치 extension](/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/config/extension.toml:5)·actual probe와 대조했다. patch 자체의 별도 문서는 아니다.
- **DEM-Engine API / kernels**, Project Chrono: [고정 commit의 Hertz 접촉 코드](https://github.com/projectchrono/DEM-Engine/blob/12f13cb15805d891eddc7b5b545d1f6823f523d7/src/kernel/DEMCustomizablePolicies/FullHertzianForceModel.cu), [회전 적분 코드](https://github.com/projectchrono/DEM-Engine/blob/12f13cb15805d891eddc7b5b545d1f6823f523d7/src/kernel/DEMIntegrationKernels.cu). 물리 워커가 설치 DEME2.4.0의 API/관련 kernel4개와 byte-identical SHA를 확인했다. 이 commit을 v2.4.0 태그라고 부르지 않는다. 메인도 설치 kernel을 직접 읽었다.
- **Isaac Lab 2.3.0 Articulation source**: <https://isaac-sim.github.io/IsaacLab/v2.3.0/_modules/isaaclab/assets/articulation/articulation.html>. 설치 Lab2.3.0과 일치하며 관절 state writer를 [설치 source](/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py:517)와 대조했다. Isaac Sim metapackage5.1.0.0, 실제 runtime 문자열은5.1.0-rc.19+release.26219.9c81211b.gl이다.
- **UsdGeomPointInstancer — USDRT7.5.1**: <https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.5.1/api/classusdrt_1_1_usd_geom_point_instancer.html>. 설치 header7.6.1과 minor mismatch가 있어 일반 instance 변환 설명에만 사용하고 [설치 header](/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache/usdrt.scenegraph-7.6.1+69cbf6ad.lx64.r.cp311/include/usdrt/scenegraph/usd/usdGeom/pointInstancer.h:52) 및 실제 W12 코드가 실행 의미의 정본이다.

## 5. 판정과 다음 승인 경계

**관성이 없는 가짜 입자 영상이 아니다.** 가정한 물성을 가진 7구 강체를 DEME로 계산한 뒤 그 기록을 Isaac으로 표시한 것이다. 하지만 보기 좋은 렌더가 실물 물성·시간간격 수렴·전체 작업 성공을 증명하지도 않는다.

이 문안과 자산 목록을 PPT에 옮길 수 있다. 이번 작업은 새 PPTX·영상·물리 결과를 제작하지 않았고 원본과 기존 worktree를 보존했다. 다른 worktree의 보고서는 메인 checkout의 Git push에 자동 포함되지 않는다. 보고 인수와 상태/relay 종료 기록은 [세션 문서](/home/cgxr/Documents/Robotics/RoArm_Project/claudedocs/session_20260914_labmeeting_20260915.md)를 따른다.
