# rev20 actual10 — root 최종 인수 기록

2026-09-13, 신규 물리 변수 []. 실제 Isaac 준비표시 실패의 종료·해시·검사 기록이다.
이 문서는 기존 ROOT_READINESS_INSPECTION_03.json을 덮지 않는다.

## 관찰 순서와 판정

1. GO_isaac_readiness_rev20_01.md에 따른 기존 raw2행+합성6자세의 단1회 실행.
   raw행0/1은 둘 다 sync0/t0이며 서로 다른 두 시간대가 아니다. 합성6은 입자동역학이 아니다.
2. root는 원본8PNG를 전부 실제로 열었다. 팔 잘림·문 조립 표시는 개선됐으나
   source support와4벽면의 식별은 미충족. 라벨 존재와 실제 면 식별은 다르다.
   ROOT_READINESS_INSPECTION_03.json SHA256:
   f1e7a74cbf6fa83cbba751ae1d22347fe5b5d93063fc21c6a272089634ee37cf.
3. 최종 EXECUTION_RECEIPT의 단계 시작08:35:15Z, TERM08:44:30Z,
   단계 끝08:44:30Z, 정리포함555.552708초≤600초. producer가 보고한08:44:43Z는
   최종 보고/외부 관측 시각이며 단계 종료 정본과 구분한다. launcher_elapsed_s=555.601.
   자식rc0이지만 timeout=true, runner/launcher124, success=false. KILL불필요.
4. root 호스트 ps에서 정확 launcher2302696/runner2302728/renderer2302731 및
   PGID2302731 멤버가 없음을 재확인했다. 전체 ps를 출력한 첫 점검은 도구 출력이
   잘릴 수 있어 부재 증거로 쓰지 않고, 호스트에서 정확 PID/PGID를 먼저 필터한
   두 번째 명령의 헤더만 출력/rc0을 채택했다. 수동 신호·새 실행0.
5. 렌더 작업은 startup11.214/render4.093/ffmpeg0.302, close전15.61초.
   close는60초 alarm이 Kit콜백에서 발생한 뒤에도 반환하지 않아 외부 TERM으로 끝났다.
   close_timed_out=null/close_s누락은 미측정이며 false가 아니다. 최종 readiness.ok=false.
6. root가 읽기 전용 verify_resume_v2.py --check-preflight --revision rev20에
   실제10_actual/자기INSPECTION03 경로를 명시해 실행했다. rc1,63/66,
   W13R_PREFLIGHT_V2_FAIL. 실패3개: 실제 시각 인수/close 계측/launcher ok.

판정: READINESS_FAILED. production GO0/새DEME0. 독립감사10/10은 실패 상태를
정확히 검증한 검사 수이지 준비 PASS10/10이 아니다. 사용자 최대9시간 비용승인은 유지한다.

## 최종 원자료와 root 재해시

실제 출력 root:
/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/readiness/renderer_readiness_10_actual/

- READINESS_RECEIPT.json: 4d48143a8e4c5d1f5ce3bf8f1df9609a25fb9ed59cd869b91e2d558836ef09b3
- attempt/EXECUTION_RECEIPT.json: 7d02b48fb52c595548746f28e58c6b9c45c7addd420ed549a2ecbf25d3e8a0da
- attempt/RUN_STATUS.json: 808401c977af16b67977b3db03bc233f314e3a7282f0f3b2f40995a21748feb5
- render/render_manifest.json: 16a7a7065f520436c36a0227a18b3de1c0f415eb57610b2050b1709af558d362
- render/visual_mapping.json: e9ec15fbff7b48a7dec058abbb577bd8278dc43f22e201ee02688ecc0f1c4416
- 독립 REV20_READINESS_ACTUAL_AUDIT_01.json: f59bcf5f39ae1c1aca64d9bf8811ac2b0724136cf6831318796e7076f4a94e91
- 독립 REV20_CLOSE_VISUAL_DIAGNOSTIC_01.md: 7e50ac91aa29e5d45092ea737395a9bb3fa23ffa5f8cb76db8a298e2eb11c0aa
- 최종 kit_20260913_173515.log: 1397b67757dcb00c88b2912e2ca487b2effc02ab9012c514333c2d61831914d5
  (진단문서의 cabec0fc…는 프로세스 종료 전 해시다. 전체 종료 로그의 해시로 인용하지 않는다.)

## 다음 좁은 수정 경계

08:58:13Z msg_56ff30cd8eeb로 생산에 새rev21 CPU 후보만 허용했다.
기존 바닥높이의 얇은 불투명 경계/벽 하단 표시선과 글자 대비, 공식 Replicator
get_status/capture-off/stop 각각의 durable before-after 진단. 기존 cleanup 보존,
runner/numeric/physics/criteria 불변, 다음 GPU는 별도 정확GO 전 HOLD.
같은600초를 반복하지 않고 ≤180초 한정 진단 계획을 먼저 검수한다.

소스의 속성 쓰기 코드만으로 실제 prim readback 성공을 확정하지 않는다. opacity와
가림은 유력 후보지만 원인 확정이 아니다. 설치 문서의 interactive rendering을
GUI 유무와 동일시해 headless 비지원이라고 단정하지 않는다.
wait_for_replicator=False는 실제로 해결하지 못했다. 정확 native 차단 프레임은 미확정.
기존 renderer 주석의803~805에서 멈췄다는 단정도 철회해야 한다.
