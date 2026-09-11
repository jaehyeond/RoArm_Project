# 2026-09-11 — 영상 워커 확인, W9/W10 산출물 재브리핑

이번 case의 신규 변수: [] (기존 기록·산출물 검토만).
사용자 요청: `worker-task_d96c65a252ec`의 영상 브리핑을 확인하고 W9/W10 산출물, 의미, 다음 행동을 채팅에 순서대로 설명.
새 실험 미실행 사유: 사용자가 기존 기록의 해석을 요청했다. 직전 같은 작업 흐름에서 W10 실제 dt 섭동 실험과 구 회귀를 완료했다. 이번 검토는 새로운 과학 판정을 만들지 않으며, 추가 모델·물리 실행은 다음 case의 변수다. 로봇 제어·학습·렌더 재실행·패키지 설치·커밋 없음.

## 1. 워커 식별과 영상 원문

- `orca-ide skills get orchestration --json`, `skills get orca-cli --json` 후 읽기 전용 명령 사용. 샌드박스의 runtime_unavailable은 호스트 읽기에서 해소. Orca 재시작·메시지 발송·새 워커·작업 바인딩·lifecycle 변경 없음.
- `dispatch-show --task task_d96c65a252ec`의 `assignee_pane_key` = `9e03e722-cac0-40b5-aff5-2fd945a73355:91150816-f211-42b4-8542-390a952d648b`.
- 현재 `Watch the YouTube video | RoArm_Project` 터미널 `term_af67a24c-dbd4-44ca-b726-adc5d4c9197f`의 tabId:leafId가 정확히 일치. PTY도 동일 `@@0ff6d557`. 과거 task 자체는 08-31 완료이며, 이후 보존된 패널의 사용자 대화가 영상 브리핑이다. 옛 dispatch 완료를 이번 영상 연구 완료 시각으로 해석하지 않는다.
- [패널 연결 근거](research/video_w9_w10_review_20260911/worker_identity.json), [기존 브리핑 두 답변](research/video_w9_w10_review_20260911/worker_briefing.txt). 원 터미널을 변경하지 않고 사본 보존.
- 영상: [VLA는 왜 World Action Model로 진화하고 있을까? | 최신 Physical AI 기술 분석](https://www.youtube.com/watch?v=wfB-jcj1mtE), 엥지유니버스 | 로봇 엔지니어, 2026-09-08, 1611초. 기존 `/tmp/watch-4xs1wud4/download/video.info.json` 확인.
- watch 스킬 기존 자료 재사용: `video.en.vtt`의 03:10~04:00, 10:40~13:15, 16:00~17:10, 24:10~25:40 구간을 파싱했다. `/tmp/watch-p7oh4k8e/frames/cue_0000.jpg`, `cue_0007.jpg`, `cue_0013.jpg` 실제 열람. 각각 RISE의 동역학/가치모델 분리, τ0-WM의 후보 비교, Masked Visual Actions의 순·역방향 도식을 확인. 영상 전체를 이번에 새로 시청했다고 주장하지 않는다.

## 2. 영상 브리핑의 적용 판단

- τ0-WM은 후보를 만든 뒤 결과를 평가하고 낮은 품질일 때 추가 계산·수정을 한다. RISE는 세계모델 내부 시행착오로 정책을 개선한다. Masked Visual Actions는 로봇의 움직이는 외형을 영상으로 제공한다. 논문/공식 프로젝트 원문 확인.
- 현재 고정 S1·고정 경로·취점 선택이라는 제약에서 적용 우선순위는 **τ0-WM의 후보 평가 구조 → RISE의 실패 상태 확장 → 다른 형상/경로가 필요한 경우 Masked Visual Actions**라는 해석을 유지한다. 원 논문 모델을 실제 도입·학습했다는 뜻이 아니다.
- 기존 브리핑의 “불확실하면 DEME를 즉석 추가”는 현재 계산 시간으로 수정해야 한다. W10은 단일 스쿱 1890초였다. 현 단계 DEME는 미리 계산하는 학습·검증용이며, 실행 중 후보 비교에는 학습된 작은 예측모델 또는 별도 지연시간 검증이 필요하다. 계산 시간은 진단 포함 현재 실행 관측이며 DEME의 원리적 속도 한계로 일반화하지 않는다.
- “자동화를 분명히 쉽게 만든다”는 검증된 성과가 아니라 가설이다. 최고점+층/열 규칙과 실제 학습모델을 비교해야 한다. 영상의 성공률이나 예전 다른 물리/형상의 오라클 결과를 현재 S1 성능으로 옮기지 않는다.
- 무인 자동화라는 목표는 유지. iSAM 공식 무인 그랩 하역 사례가 있으므로 자동화 자체만을 신규성으로 주장하지 않는다. 연구 기여 후보는 취득량/사후 형상/불확실성 예측이 실제 취점 결정에 필요한 조건을 실험으로 밝히는 것이다. 새 “최초/없음” 주장 없음.

원문: [τ0-WM v2](https://arxiv.org/abs/2606.01027v2), [RISE v2](https://arxiv.org/abs/2602.11075v2), [Masked Visual Actions 공식 프로젝트](https://masked-visual-actions.github.io/), [iSAM Autonomous grab ship unloaders](https://www.isam-ag.com/services/autonomous-ship-loaders-and-advanced-collision-protection/autonomous-grab-ship-unloaders/).

## 3. W9 — W8 물리 결과의 로봇 장면 재생

정본 `runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w9_isaac_render_deme/`.
`gates_w9.json`의 입력 해시를 실제 W8 `render_timeline_cell1.npz`와 대조해 일치 확인(sha16 a6f7ca76a6774e7c). 타임라인은 53개, captured_ids는 154개. metadata는 W8 옵션 F params를 가리킨다. W10 결과를 렌더한 것이 아니다.

- ffprobe: `render_w9.mp4` H264, 2048×640, 10 fps, 53프레임, 5.3초. 0.05초 간격 타임라인을 10 fps로 보여 주므로 약 0.5배속.
- 원본 frames 53행 재계산: 립 위치 오차 최대 1.847mm, 평균 0.637283mm. 마지막 프레임 cavity 판정 154/154.
- `keyframe_strip_w9.png` 실제 열람: 하강→닫힘→상승이 두 시점으로 배치됨. 상자가 얕게 채워져 있고 로봇 툴이 중앙 더미에 접근한다.
- `xray_run7_opacity045/frames/f_00052.png` 실제 열람: 그랩 외피가 숨겨진 상태에서 주황 포획 군집과 청록 공동 윤곽, 아래쪽 보라 입자를 확인. 상부 일부는 팔 구조에 가려져 있어 모든 입자를 눈으로 개별 계수한 것은 아니다. 154/154는 JSON 수치 근거다.
- 팔 관절 상태를 직접 기록하는 재생이고 환경 충돌도 비활성이다. 기구학적 자세 재현 근거이며 실물 토크/충돌/퍼내기 성공 검증으로 확대하지 않는다.
- mapping: 더미 표면 world z 약0.1999m, 보고서의 실물 기준0.26m. 약60mm 차이가 있으므로 실물 질량과 직접 정합했다는 결론 불가. 현재 실물 높이를 다시 실측하기 전 과거 기준을 현재값으로 단정하지 않는다.

NVIDIA 근거: **Isaac Lab Documentation — isaaclab.assets, v2.3.0**, https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_state_to_sim . 설치 metadata IsaacLab2.3.0, IsaacSim5.1.0.0, 로컬 VERSION `5.1.0-rc.19+release.26219.9c81211b.gl`; numpy1.26.0/psutil5.9.8. 설치 소스 `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py:517`의 write_joint_state_to_sim → 위치/속도 직접 기록. 프로젝트 호출 `sim_isaac_render_deme_scoop.py:259`, 입자 시각 생성 `:220`. IsaacLab 문서 버전 일치. 재생의 한계는 이 프로젝트 설정에서 내린 해석이며 NVIDIA 엔진 하드 한계 주장이 아니다.

## 4. W10 — 시간 간격 축소 후 토크 정지 완주

정본 `runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/REPORT_w10.md`, `cell_DE_dt2e6_c/scoop_s1_seed460.json`, `resume_20260911/verification.json` 재확인.

- 10μs→2μs, E5e6 유지. 첫 닫힘/재닫기 servo_stall, 문 하한각 없음. 단일 중앙 셀/seed460.
- 포획541개/10.9592g, 서보 환산 최종5.569°(명목 관절3.069°), 실제 메시 서보 환산5.493°(관절2.993°). 립 등가 간격6.152mm, 최종 립 물림0.
- 한 sync 5m/s 초과, 최대5.329m/s 잔존. “끝까지 계산됨”과 “실물 일치/수치오차 없음”은 별개.
- 현재 소스 구 회귀287개, 허용268~362. 과학 게이트4/4는 원 조건 통과이며 실물 보정 완료가 아니다.
- RRD 툴/접촉/스칼라2774sync, 입자64프레임; 원자료와의 읽기 대조 및 검수는 직전 세션 완료. 이번에 입력 render timeline의 64프레임·541 captured_ids 확인.
- `cell_DE_dt2e6_c/scoop_s1_seed460_w10_inspection.png` 재열람: 중앙은 들어 올린 주황/파란 군집, 오른쪽은 첫 닫힘 접촉과 툴 윤곽, 아래는 궤적/높이맵/이벤트. 왼쪽은 시작 시각이며 주황색은 최종 포획 여부를 미리 색칠한 것. 모든 패널이 같은 시점이 아니다.
- W10 산출은 RRD/RBL/PNG/JSON/NPZ. W10 Isaac MP4는 아직 없다. W9 영상을 W10 영상처럼 제시하지 않는다.
- W8F의154개/3.1196g 대비 증가는 속도·정지 조건 등도 달라 전부 dt 효과로 주장하지 않는다. dt-only의 직접 직전 대조는 실패한 cell_DE_c.

## 5. 다음 행동 — 제안이며 신규 case 실행 승인으로 간주하지 않음

1. 다음 실물 세션은 기존 부팅 체크리스트부터 이어간다. 펌웨어/피드백, P 적용 여부, 닫힌 문 되열림을 확인해 계측 조건을 고정한 뒤 회당 질량5회를 확보한다. 현재 로봇을 구동하지 않았다.
2. 저울값만 모으지 말고 회차별 초기 표면/취점/깊이·속도, 닫힘/상승/재닫힘 각, 질량, 전후 높이맵을 대응시킨다. 같은 초기 상태 반복이면 매회 초기화하고, 연속 스쿱이면 매회 달라진 전상태를 기록한다. 실물·시뮬의 더미 높이·깊이부터 맞춘다.
3. `weigh 5`는 `hw_s1_manual.py:251`에서 실제 scoop/place를 실행하므로 단순 저울 읽기 명령이 아니다. 회당 영점을 맞추거나 누적 저울 차이를 입력한다. `:173`의 평균은 누적 mass_log 전체이므로 이번5회는 별도 집계한다. 코드 변경 없음.
4. 시뮬 후속 후보는 같은 더미/물성을 두고 dt2→1μs만 바꾸는 좁은 수치오차 검사다. 질량/정지각/사후 높이맵/순간 속도 민감도를 확인한 뒤 물성 보정 case로 넘어간다. 한 셀 추가만으로 엄밀한 수렴/다중 시드 안정성을 확정하지 않는다. 아직 설정 작성·실행하지 않았다.
5. 믿을 수 있는 한 스쿱 전이를 확보한 뒤 최고점+층/열 규칙 → 질량 예측 → 질량+사후 형상 예측 비교. 불확실성 활용과 실패 상태 확장은 그 뒤에 각각 검증한다. 현재 고정 경로/형상은 유지.

이번 검토는 기존 공간 산출물을 해석했으며 새 공간 판정이나 새 probe를 실행하지 않았다. W9 원본과 W10의 기존 완결 RRD/검수 화면을 재사용했다. 신규 과학 결과가 없어 DECISIONS/EXPERIMENT_LEDGER 추가 없음.
