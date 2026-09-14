# 기존 경계 진단 RRD 화면 1장 — 제한적 GO

발행: 2026-09-13 06:46 UTC. 수신: `ctx_e5023d7f09ed`만.

- 합성 경계 진단의 기존 RRD/RBL을 읽는 헤드리스 화면 1장만 허용한다. DEME, Isaac, 물리/장면 생성, 생산 실행, 재시도 승인이 아니다. 생산 워커 GPU HOLD는 유지한다.
- 기존 `whole_clump_boundary_03/whole_clump_boundary.rrd` SHA256 `8d2554201f90f39cec0db014491f5cd8bea45a2c76f12eaa71b41d447f58bee7`, RBL `87381ba99e0c0fa0c5994678e16a6014954442e8936f4f1d773bf8d7a9f26a35`를 실행 전 대조한다.
- Rerun CLI 0.34.1의 로컬 help에서 headless/window-size/screenshot/bind/port를 확인했다. exact RRD 검사는 21개 엔티티/필수 구성과 blueprint+log_time 타임라인, footer/RBL을 통과했다. 기존 CPU PNG를 Rerun 화면으로 취급하지 않는다.
- 다음 명령 1회. 115초 뒤 TERM, 추가 5초 뒤 KILL의 총 120초 예산. 루프백에만 바인딩한다. 출력 PNG와 새 로그/영수증이 이미 있으면 실행하지 않는다. 자동 재실행 금지.

```sh
timeout --signal=TERM --kill-after=5s 115s /home/cgxr/miniconda3/envs/isaaclab/bin/rerun --headless --bind 127.0.0.1 --port auto --window-size 2400x1400 --screenshot-to /home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/whole_clump_boundary_03/rerun_headless_inspection_01.png /home/cgxr/orca/workspaces/RoArm_Project/w13-cycle-audit/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/audit/whole_clump_boundary_03/whole_clump_boundary.rrd
```

동일 진단 폴더의 새 `headless_01.stdout.log`, `headless_01.stderr.log`, `HEADLESS_RECEIPT_01.json`에 정확 argv/입력해시/시작·종료·monotonic 소요/rc/timeout/PNG 해시를 보존한다. 종료 및 자식 정리를 확인한 뒤 PNG를 실제 열어 관찰을 새 inspection addendum에 남긴다. 생성·load 성공만으로 시각 검수 PASS를 선언하지 않는다. 실패는 보존하고 root에 보고, 재시도하지 않는다.
