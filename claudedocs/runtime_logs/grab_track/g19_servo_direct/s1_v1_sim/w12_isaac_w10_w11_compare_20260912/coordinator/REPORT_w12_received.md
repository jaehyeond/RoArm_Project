# W12 — W10/W11 Isaac 재생 결과 수신·독립 검수

이번 case의 신규 변수: []. 기존 W10 dt2μs/W11 dt1μs 결과의 표시·비교만 수행했다.
재생 본체와 보조진단 정정 검수 완료. 워커 종료·상태 인계 기록은 세션/게이트 원장에 둔다.

## 1. 무엇을, 왜

W9는 W8의154알 결과를 Isaac에서 재생한 것이었다. 이번에는 실제 W10/W11 저장 자료를 같은 S1 로봇·상자 장면에 표시했다.
입자 힘·접촉 계산을 새로 돌린 것이 아니다. 새 물리 실행, 실물 조회/구동/카메라, 학습, A/B/C, 설치, commit/push는0건이다.

사용자 지정 모델 그대로 별도 Orca worktree를 사용했다.

- Claude **claude-opus-5**: `/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay` — Isaac 재생·영상·Rerun.
- Codex **gpt-5.6-sol / high**: `/home/cgxr/orca/workspaces/RoArm_Project/w12-input-audit` — CPU 원자료 독립 감사.
- 메인은 상태 원장과 최종 검수만 소유했다. 워커 실행 모델 확인이며 메인 세션 모델 변경을 뜻하지 않는다.

## 2. 관찰 가능한 절차

1. 결과 JSON이 지정한 실제 NPZ 경로를 찾고210개 기존 파일/HEAD 기준선을 저장했다. W10 NPZ는 셀 안이 아니라 W10 루트에 있다.
2. W10을 먼저, W11을 다음으로 Isaac에서 각64프레임 재생했다. 카메라·USD·표시원점·팔 역기구학 조건은 동일하게 유지했다.
3. 문은 명목 명령각 대신 저장된 실제 자세를 사용하고, 원자료/표시 문 노드와 립·힌지 표식을 함께 기록했다.
4. 각 run의 원자료 초를 유지해 최근접64쌍을 만들었다. 시간 정규화·입자 보간은 하지 않았다.
5. 메인이 원시 질량/정지 기록,128프레임 숫자, RRD 전체 배열, MP4 디코드 프레임 수, 주석본 장면 픽셀을 다시 검사했다. 결정 PNG를 실제로 열어 확인했다.
6. 잘못된 좌표·힘·시간 짝이 검사에서 거부되는 대조 시험을 수행했다. 원자료는 수정하지 않았다.

## 3. 수치

| 항목 | W10 dt2μs | W11 dt1μs |
|---|---:|---:|
| Isaac 재생 프레임 | 64/64 | 64/64 |
| 표시 립 최대 오차 | 1.873mm | 1.867mm |
| 원자료 최종 포획 | 541개 | 517개 |
| 원자료 포획 질량 | 10.959243732g | 10.473066561g |
| 첫 닫힘 정지 | servo_stall | servo_stall |
| 재닫기 정지 | servo_stall | pinch_guard |
| 재닫기 단일접촉 최대 | 2.599655974N | 3.008901850N |

`servo_stall`은 기존 저항 토크선에 도달한 정지, `pinch_guard`는 기존 단일접촉3N 보호선에 도달한 정지다.
포획 질량은 공구 안의 양이지 컵 배출량이 아니다. W11은24개·0.486177171g 적으며 각dt1회라 원인을 dt만으로 분리하지 못한다.

- 같은 시간 짝의 최대 차이: **3.076999992ms**. 원본/주석 MP4 각3편은 모두64프레임·10fps·6.4초이며 화면의 원자료 초가 실제 계산 시간이다.
- 사건별 스틸은 다른 기준이다. 첫 닫힘: W10 frame52는 정지후16.008ms, W11 frame51은 정지전24.947ms. 두 장을 정확히 같은 사건 시점으로 읽으면 안 된다.
- 두 결과 모두 재닫기 중의 저장 입자 프레임이 없다. frame63은 정지후 약12ms의 최종 `lift` 태그 프레임이다. 연속 재닫기 움직임을 복원한 영상이 아니다.
- 색은 **최종 포획/딸려감 ID 소속**을 시간 전체에 고정한 것이다. 영상의 공동내 개수는 최종 포획ID 부분집합 중 현재 공동 안에 있는 수이며 순간 전체 포획량/전수 분류가 아니다.
- Rerun0.34.1: 전체 재생 RRD224.7MB와 결정용7.0MB. RRD/RBL footer·검증시 해시·정확 객체/시간축/성분 계약,30개 entity의 각64프레임 배열 대조와 정적 결정 배열 대조 PASS.
- 대형 RRD 헤드리스 PNG는 일부 패널이 로드되기 전에 찍히는 한계가 있다. 실제 결정 검수는 완전히 표시된 소형 결정 RRD PNG로 수행했다. mm 오차 그래프는 별도 PNG에서 확인했다.
- 보고서 초안의 미사용 보조값7.7822mm/°는 좌표계 혼용 오류로 무효다. 축에 수직인 반경을 쓰면2.057714892mm/°이며,0.0229°의 환산 이동은 약0.0471mm다. `replay/diagnostic_correction_01.json`과 이 폴더 `diagnostic_root_reverification_01.json`이 정정 근거다. 원래 renderer/gates 메타데이터는 오류 이력으로 보존했고 실제 재생/판정은 바뀌지 않았다.

## 4. 근거 파일 — 사용자 열람 우선순위

아래 재생 경로의 공통 루트는 `/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/`다.

1. [주석 포함 동시점 비교 영상](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/paired_w10_w11_sourcetime_annotated.mp4)
2. [W10 개별 영상](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/replay_w10_annotated.mp4), [W11 개별 영상](/home/cgxr/orca/workspaces/RoArm_Project/w12-isaac-replay/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/replay/annotated/replay_w11_annotated.mp4)
3. `paired/pairing.csv`, `annotated/montage_first_close_annotated.png`, `montage_reclose_annotated.png`, `paired/mapping_error_per_frame.png`.
4. `rerun/trial2/w12_replay_mapping.{rrd,rbl}`, `w12_decision.{rrd,rbl}`, `w12_decision_inspection.png` 및 해당 validation/coverage. `rerun/trial1_defective/`와 원래 trial1 경로는 오류 보존용이며 사용할 수정본은trial2다.
5. 이 폴더의 `rrd_root_reverification_01.json`, `media_root_reverification_01.json`, `preservation_final.json`, `inspection_root.json`, 검증기4개 및 `input_baseline.json`.
6. 독립 감사 정본: `/home/cgxr/orca/workspaces/RoArm_Project/w12-input-audit/claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w12_isaac_w10_w11_compare_20260912/audit/repair_01/{audit.json,REPORT_w12_audit_repair_01.md,verify_w12_audit.py}`. 원래 audit 시도는 mutable gate 해시 의존이 있어 repair_01을 사용한다.
7. 과학 정본: 메인 `.../s1_v1_sim/w10_deme_close_fix/cell_DE_dt2e6_c/` 및 `w11_dt_sensitivity_20260911/cell_dt1e6_seed460/`의 JSON/NPZ. W11 `comparison.json`과 기존 RRD는 전체 결정 sync/접촉 증거다. W12 RRD의 Float32 사본을 과학 정본으로 사용하지 않는다.

## 5. 판정과 승인 경계

**같은 Isaac 장면에서 두 저장 결과를 비교할 수 있다. 그러나 두 물리 결과가 같거나 dt가 수렴했다고 판정할 수는 없다.**
팔의 구동 가능성·실물 정합·배출 효율을 이번 재생으로 입증하지 않았다. 표시원점 `(0.35,0,0.163)m`과5mm 표시 기준은 프로젝트 설정이지 NVIDIA의 엔진 한계나 실측 보정값이 아니다.

API 근거: **IsaacLab2.3.0, Articulation.write_joint_state_to_sim** [공식 API 문서](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_state_to_sim).
설치본 `/home/cgxr/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab/assets/articulation/articulation.py:517`, 실제 호출 `replay/sim_isaac_replay_w12.py:368`.
IsaacSim 설치 버전은 `5.1.0-rc.19+release.26219.9c81211b.gl`이고 위 문서는 일치하는 IsaacLab2.3.0 API다. 함수가 관절 상태를 직접 쓰며, 이 사용 방식이 구동기 실현 가능성의 증거가 아니라는 결론은 코드에 근거한 해석이다.

추가 dt/반복/물성, A/B/C 배출 비교, 학습, 실물 조회/구동/카메라는 새 승인 전 실행하지 않는다. 산출은 두 worktree에 보존했으며 메인의 Git push만으로 자동 포함되지 않는다.
