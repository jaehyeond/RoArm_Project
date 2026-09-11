# W10 reboot continuation — 2026-09-11 Codex

이번 case의 신규 변수: [`timestep_s`: 1e-5 → 2e-6 s]. E=5e6 Pa 유지. 단계 ③'가 발산한 경우에만 기존 지시서의 단계 ③(E=1e8 Pa, dt=2e-6 s) 실행.

## 범위와 원장 소유

사용자: “재부팅 완료했어. nvidia-smi부터 확인하고 w10이어가”. Codex가 Claude의 종료 relay를 읽고 이번 원장 기록을 이어받는다고 사용자에게 알렸다. W10과 관측/보고만 진행한다. 로봇 제어·형상 변경·새 의존성·학습·커밋은 실행하지 않는다.

## 재개 절차와 증거

1. 첫 명령 `nvidia-smi`: 정상 표, 드라이버 580.178.04. `/proc/driver/nvidia/version`도 580.178.04로 일치.
2. 샌드박스에서는 NVML 통신 오류 및 torch CUDA unavailable. 호스트 권한의 동일 검사에서 NVML 정상, torch 2.7.1+cu126 / compiled CUDA 12.6 / CUDA available=True, GPU tensor 2×2=4.0 확인. 재부팅 전 버전 불일치와 구별한다.
3. 입력 렌즈 더미 SHA256 앞 16자리 `659d6b0bc771678a` 확인. `params_w10_DE_c.json` 대 `params_w10_DE_dt2e6_c.json` 차이는 dt, 설명, 렌더 출력 경로뿐이다.
4. 샌드박스의 GPU 대기 실행기를 Ctrl-C로 중단(exit 130, 시뮬레이션 시작 전). 13:13:04 KST 호스트에서 `run_w10b.sh DE_dt2e6_c 14400` 재개. 가용 GPU 메모리 14244 MiB, unified exec session 47227.
5. `cell_DE_dt2e6_c/stdout.txt`에서 20,000 클럼프 × 7구 로드 및 `Initialize OK` 확인. 물리 실행 진행 중이며 완주/발산 판정은 아직 없다.

실행 폴더: `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/`.
실행 시작 증거와 변경 전 SHA256: `resume_20260911/launch_evidence.json`.
기존 과학 판정 조건: `GATES_w10.md`(수정하지 않음). 이번 실행 완결 추적: `resume_20260911/GATES.md`.

## 검증 및 관측 계획

이 세션은 dt 변화가 발산 여부를 바꿀 수 있는 실제 DEME 평가를 실행한다. 성공을 전제하지 않는다. 완료 후 원본 JSON/NPZ, 프로세스 종료 코드와 시간 상한을 대조하고, W8 옵션 F와 비교한다. 공간·접촉·시간 경로 판정이므로 Rerun 생략 사유가 없으며 D341 계약 및 실제 스크린샷 육안 검수가 필요하다.

기존 후처리에서 `gates_w10_writer.py`가 새 dt-only 셀을 열거하지 않고 `pass=` 구문 오류를 포함함을 확인했다. 과학 판정 조건을 유지하면서 후처리 실행을 수리할 예정이다. 변경 전 사본은 `resume_20260911/*.before`에 보존했다.

## 후처리 정비(실행 중, 물리 코드 변경 없음)

- `gates_w10_writer.py`: 문법 오류 수정, 단계 목록에 `DE_dt2e6_c` 추가. 사전 조건의 시간 상한·문 하한 미사용·정지 기록 존재도 실제로 검사한다. 첫 동작 확인에서 과거 구 회귀·발산 진단은 읽혔고 새 셀 결과는 미완으로 남았다.
- 기존 구 회귀 JSON의 시뮬레이션 source sha16=`c99fd9c3eab8579e`인데 현재는 `2e40f7ed279dad42`. 과거 수치가 조건 안이어도 현재 코드의 회귀 PASS로 재사용할 수 없으므로 G0에 일치 여부를 표시한다. 렌즈 셀 이후 GPU 순차 실행으로 `regression_sphere_resume_20260911/`에 재확인할 계획. 과거 회귀 실측 시간은 86.02 s(JSON).
- `w10_rerun_export.py`: 전체 sync의 툴·접촉(접촉 0도 포함) 기록, 기존 0.05 s 입자 타임라인의 7구 전개 재생, 정지 시점의 명목 q 대 실제 메시 프레임, 결정 스냅샷을 추가. Python 구문 검사는 통과했으나 최종 데이터로 export·D341 검증은 아직 실행 전이다.
- Rerun 실행 환경(읽기 확인): isaaclab `rerun-sdk=0.34.1`, `numpy=1.26.0`, `psutil=5.9.8`. 의존성 설치 없음. API는 설치본 `rerun.blueprint.TimePanel` 서명과 [공식 Blueprint API](https://ref.rerun.io/docs/python/0.32.1/blueprint/)(웹 문서는 0.32.1, 설치본은 0.34.1)·[공식 timelines](https://rerun.io/docs/concepts/logging-and-ingestion/timelines)를 참고했다. 기존 API가 제공하지 않는 시간 커서 강제 설정을 가정하지 않고 결정 장면을 별도 static view로 분리했다.


## 최종 실행 결과와 판정

- 렌즈 셀은 13:13:04→13:44:34 KST, rc=0, 실행기1890s/물리 루프1870.90s, 스톨0. 첫 닫힘 q3.307°·1.783022N·m, 재닫기 q3.069°·1.765356N·m, 모두 `servo_stall`. 조건부 강성 셀은 발산이 없어 미실행.
- 포획541개·10.9592g, 높이 기준 동반 상승612개. 템플릿 알 질량 × NPZ의 보울 내부 mask 합계로 독립 재계산해 JSON 반올림 오차 내 일치. 최종 립 간격6.152mm·립 물림0개. 서보 환산5.569°와 명목 관절3.069°를 구분한다. 실제 메시 관절2.993°(서보5.493°).
- 최대속도5.329m/s, 5m/s 초과1sync. 새로운 이벤트도 `sphere_sphere`, ghost_prepop=False. 속도가 큰 순간이 남았으며, 중단 없이 완주한 사실을 실물 물성 검증으로 승격하지 않는다.
- 구 회귀 13:45:00→13:46:45, rc0/105s, 287개·허용268~362, 두 닫힘servo_stall. source sha16 현재와 일치. G0~G3 4/4 PASS.
- W8 옵션 F154개/3.1196g 대 W10 541개/10.9592g. 두 조건은 dt 외에도 속도·진단이 다르므로 질량 차이 전체를 dt 하나에 귀속시키지 않는다. dt-only의 직전 조건은 단계② `cell_DE_c`다.
- -x 단면각은 원 계산 24.59° / 높이 필터 후 3.88°. 후처리 단면 곡선의 비단조성이 실제 이미지에서 보였고 대표성이 약한 각도다. 속도로 정지 입자를 골라낸 값은 아니다.

## D341 실제 관측 완료

SDK/CLI=0.34.1. `scoop_s1_seed460_w10.rrd` 149493626 bytes, `.rbl` 63619 bytes. footer-enabled verify·엔티티·시간축·필수 컴포넌트·고정 blueprint·스크린샷 검증 모두 PASS. 추가 `verify_rrd_coverage.py`가 저장된 RRD를 다시 읽어 툴/접촉/스칼라2774 sync·입자64프레임의 인덱스/배치 길이/시각/스칼라를 원자료와 대조했다. 이벤트의 12-sync 근방 입자와 구–구 접촉 덤프도 포함했다. 기존 W8 RRD를 대조 입력으로 넣으면 빠진 툴/접촉 시점과 입자 재생을 거부했고, 기존 스칼라 전량9종은 통과했다.

**실제 view_image 검사한 경로** (아래는 모두 `claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/cell_DE_dt2e6_c/` 아래):

1. `scoop_s1_seed460_w10_inspection.png`: 중앙 최종 장면의 들어 올려진 포획 입자·파인 더미·떨어진 입자 일부, 오른쪽 첫 토크 정지의 툴/접촉점, 아래 두 토크 정지와 포획 로그를 확인.
2. `scoop_s1_seed460_w10_decision_frames.png`: 명목/실제 문 프레임 원점·축이 거의 겹쳐 보임. 글자 라벨은 겹친다.
3. `crater_profiles_rest_seed460.png`: -x 비단조 곡선과 낮은 직선 적합각을 확인.
4. `snapshot_seed460.png`: 최종 군집이 툴 윤곽 안에 있고 입자 일부가 아래로 떨어져 있음. 원 스냅샷의 한글 폰트는 네모로 표시됨.

Rerun 기본 커서는 시작 시각. 중앙/오른쪽은 명시적으로 최종/첫 정지 static 사본이다. 오른쪽 패널 힌지가 상단에 일부 잘려 전체 프레임은 별도 PNG에서 보았다. RRD metadata의 Float64 설명은 원본 전체의 dtype을 뜻하지 않는다: 원본 NPZ 입자 중심은 float64, 툴 노드와 접촉은 float32. 원본 JSON/NPZ가 정본이며 Rerun 값을 과학 gate에 되먹이지 않았다. 상세/PNG hash는 `scoop_s1_seed460_w10_inspection.json`, machine 결과는 `_rerun_validation.json`·`_coverage.json`.

## 출처 확인과 환경

NVIDIA 공식 문서: [nvidia-smi — NVIDIA System Management Interface program](https://docs.nvidia.com/deploy/nvidia-smi/index.html). 문서는 고정 release URL이 아닌 현행 공통 CLI 문서다. 이번 적용 버전은 로컬 드라이버/커널580.178.04이며 `resume_20260911/launch_evidence.json:16` 및 원 명령 출력을 근거로 확인했다. CUDA 빌드12.6/호스트 연산 근거는 같은 JSON:18~21. 문서 버전과 설치 버전의 동일성을 가정하지 않았다. 샌드박스 접근 문제라는 구분은 호스트/샌드박스 실측 대조 결과다.

## 문서/소스 보존과 다음 범위

- D482=`claudedocs/DECISIONS.md:30153`, LEDGER 새6열 행=`claudedocs/EXPERIMENT_LEDGER.md:582`. 두 원장의 기존 전체 바이트 prefix SHA256 불변은 `resume_20260911/append_integrity_{before,after}.json`으로 확인했다.
- `sim_deme_scoop_s1.py`·보호된 `sim_deme_scoop.py`·기존 params·더미·S1 형상·`GATES_w10.md`는 변경하지 않았다. 수정은 W10 후처리, 새 관측/검증 스크립트와 상태 문서뿐이다.
- 기존 W1~W9, 사용자 실물 로그, `.claude`/펌웨어/환경 패키지는 보존. 로봇 제어·lerobot-train·커밋/푸시 없음. Codex는 Claude 전용 auto-memory를 수정하지 않았으며 연속 작업 정보는 session/relay로 남긴다.
- 80th~81st W10 외 소급 원장 등재는 이번에 하지 않았다. 해당 항목의 증거를 재검증하지 않았기 때문이다. 원래 인계와 폴더들은 유지했다.
- 다음 후보는 `BACKLOG.md` 09-11에만 기록: dt 수렴/다중 시드, 실물 각도·질량 대조, -x 각도 대표성. 한 case에 모두 구현하지 않고 신규 변수 제한을 유지한다. 로봇 및 새 물리 변수/형상은 다음 범위 결정.
- 최종 상세: `REPORT_w10.md`, `resume_20260911/verification.json`. 판정은 **W10_DT2E6_TORQUE_STOP_COMPLETE__TRANSIENT_POP_WARNING__SIM_REAL_UNCALIBRATED**.
