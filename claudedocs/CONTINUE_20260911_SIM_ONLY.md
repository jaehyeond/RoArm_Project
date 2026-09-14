작업 repo: /home/cgxr/Documents/Robotics/RoArm_Project
2026-09-11 실물 작업 종료 세션에서 이어서, 시뮬레이션 연구만 진행해.

먼저 AGENTS.md → START_HERE.md → claudedocs/DECISIONS_ACTIVE.md → claudedocs/LEDGER_RECENT.md 순서로 읽어.
relay/from_claude.md는 부팅 규약대로 읽되, 최신 인계는 claudedocs/relay/from_codex.md도 확인해. 상태가 충돌하면 START_HERE.md가 우선이야.
다음 기록으로 이 세션의 관찰·실패·연구 방향을 복원해:
- claudedocs/session_20260911_hardware_closeout_next_sim.md
- claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_real/boot_check_20260911/scoop_tilt_cycle_01/closeout_01/BRIEFING.md 및 관찰·시간분석·검증 JSON
- claudedocs/session_20260911_full_scoop_outlet_repeat.md
- claudedocs/session_20260911_video_w9_w10_review.md
- claudedocs/research/video_w9_w10_review_20260911/worker_briefing.txt
- claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w10_deme_close_fix/REPORT_w10.md 및 resume_20260911/verification.json
숫자는 연결된 원자료까지 확인하고 git status --short로 미커밋 변경을 확인해. 기존 변경을 되돌리지 마.

실물은 오늘 종료했어. 새 명시 승인 없이 serial/T105 조회, 로봇 구동, PID/토크 변경, 카메라 수집을 하지 마.
Git push는 내가 할 거야. commit/push도 실행하지 마.
마지막 실물 관측은 HOME 복귀야. 한 번의 새 scoop를 5개 실행 구간과 4번의 중간 정지/복구를 거쳐 마쳤어. 무중단 성공으로 쓰지 마.
최신 계량은 컵 9.66g, 보고 무게 24g, 고정 jaw 잔류 약 2알이야. 24g의 컵 포함 여부는 미확정으로 기록됐어. 컵 포함이면 PP 14.34g이고, PP만이면 24g이야. 후속 정정이 있으면 반영해.
마지막 잔류 처리부터 HOME까지 83.54초였어. 이전과 전송 속도·가속도 필드는 같았고 작은 단계별 이동/정착 확인과 방향 복원이 길었어.
내 가설은 잔류를 조금 허용하는 연속 작업, 또는 처음부터 기울여 한 번에 개방·배출하는 방식이 더 빠를 수 있다는 거야. 판정은 같은 HOME→scoop→배출→HOME 경계의 컵 도착 g/s와 흘림·잔류로 해.

이번에 승인하는 첫 작업은 W11: W10의 계산 간격(dt) 민감도 비교야. 새 세션의 Active Case를 이 범위로 갱신하고 실제 실행까지 진행해.
이번 case의 신규 변수는 [dt] 하나. 기존 dt=2e-6s 결과와 신규 dt=1e-6s를 비교하고 seed460, 더미, S1 형상, E5e6, 마찰, 서보, 경로는 유지해.
새 출력 경로: claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1_sim/w11_dt_sensitivity_20260911/
경로가 이미 있으면 기존 파일을 덮지 말고 새 attempt 하위 폴더를 만들어.
먼저 nvidia-smi, 설치 버전, W10 입력 해시를 확인해. closeout_01/simulation_readiness_audit.json의 6개 입력과 비교해.
DEME는 roarm 환경의 2.4.0 기준이야. 기존 환경을 사용하고 임의 설치하지 마. isaaclab의 numpy 1.26.0/psutil 5.9.8 핀을 보존해.
기존 run_w10b.sh를 그대로 실행하지 마. 새 params/out와 render_timeline_path를 포함해 모든 출력을 새 폴더로 지정해. 출력 위치 외 물리 변경은 dt뿐이야.
실행 전에 완주/정지각/포획 개수·질량/사후 높이맵/최대 속도·경고의 비교 기준을 기록하고, 실패할 수 있는 실제 시뮬레이션 한 셀을 실행해.
원자료와 Rerun 0.34.1 RRD/RBL, footer 검증, 엔티티·타임라인/원자료 대조, 결정 시점 스크린샷과 실제 육안 검수를 완료해. 불필요한 전체 영상은 만들지 마.
W10은 포획 541알/10.9592g이고 최대 입자 속도 5.329m/s 경고가 남아 있어. dt 영향부터 확인하되, 두 간격 비교만으로 엄밀한 수렴이나 실물 일치를 선언하지 마.
W9는 W8의 154알 결과를 재생한 자료이고 W10과 다른 산출물이야. W10 포획 질량과 실물 컵 도착 질량도 같은 지표가 아니야.

후속 별도 case 후보는 A 기존 배출/잔류 허용, B 배출 후 잔류 제거, C 처음부터 기울여 한 번에 배출 비교야. 이번 dt case에 함께 구현하지 마.
현재 sim_deme_scoop_s1.py는 close/lift/reclose까지이며 배출 단계가 없어. W10 NPZ는 전체 속도·접촉 내부 이력이 없는 진단 파일이므로 정확한 물리 재시작 상태라고 가정하지 마.
고정 자세/충돌 형상의 회전 적합성과 공통 초기 상태를 먼저 검증한 뒤 배출 비교를 설계해. 연속 회차의 잔류 이월은 매회 새 손실로 중복 계산하지 마.
장기 연구는 고정 S1·고정 배출 위치에서 취점 선택: 단순 높이/층·열 규칙 기준선 → 포획·도착 질량 및 사후 높이맵 예측 → 후보 선택이야. 지금 학습으로 범위를 넓히지 마.
확인한 현재 상태와 첫 실행을 한국어로 짧게 브리핑한 뒤 진행해. 관찰·절차·수치·근거 경로를 순서대로 보고하고 종료 시 START/세션/원장/relay를 갱신해.
