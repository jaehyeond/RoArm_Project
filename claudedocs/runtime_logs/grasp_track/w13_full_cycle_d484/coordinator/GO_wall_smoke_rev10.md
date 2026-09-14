# W13 단일 통 벽 시험 실행 승인 — rev10

2026-09-12 16:46 KST. 승인 주체: 메인 coordinator. 사용자 W13 시뮬레이션 배정 범위 안의 제한 시험이며 실물 승인 아님.

독립 감사 `msg_979e2469fd86`: rev10 고정 사본20/20, 외부13/13, implementation10/10 해시 일치 및 runner/판정기 변경분 검토 PASS. root는 init/sample/판정기/Rerun exporter와 rev9→10 runner 변경분 및 PLAN04를 직접 읽었다.

## 승인된 단 하나의 실행

생산 worktree `/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle`의 `claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/implementation`를 cwd로 사용한다.

`preflight_01/rev10_frozen/COMMANDS.json["runner"]`만 한 번 실행한다. `COMMANDS.json` SHA256 `b00a98bb97e522aa75cff79e938ce3537e9c8704721f6bd9985015bcdb903388`.

- runner `140b38d091e5d1a22849738c319c6f75899718848444d6d20bed96d3fd079928`
- sim `30c821a3bb0e134ec96d384481526824669c7f0d0b0db2c1b17acad60ca82c95`
- params `95101f8b799518f83a0d8a0004057e50ea01120b66681334a9911a7e5fa7e661`
- 비교기 `6cc7c34be3f6c8e6a44edc648d5531c81c30c8c0de94a5f8c1bca74c682857f4`

20,000 clump·seed460·dt1e-6 고정. genuine t=0 HOME 기록 후 settle25 sync, 목표t=.100025s. 출력은 `attempts/smoke_wall_regression_01/`만 사용한다. 실행기 자체 출력은 attempt 밖 새 전용 로그로 저장해 실행 전 빈 폴더 검사와 충돌하지 않게 한다.

실행기는 시작 전 해시/기존 산출 검사를 시행하고, 단계별1800초 제한·분리 stdout/stderr·실제 rc 영수증을 강제한다. 실패 시 보존 후 중단, 자동 재시도0. 성공적으로 원시 생성 후 동결 비교기/Rerun exporter 후처리까지 이 승인에 포함한다. 후처리 오류도 기존 산출을 덮지 않고 보고하며 물리 재실행 금지.

## 승인하지 않은 것

운반/배출 전체 production, W10/W11 재실행, dt/물성/조건 변경, A/B/C, 학습, 실물 조회/구동/카메라/PID/토크, commit/push. 이 정지 시험의 PASS는 전체 경로나 실물 가능성 PASS가 아니다. 전체 생산 경로의 접촉 후 연결 구간 검토는 별도로 계속한다.
