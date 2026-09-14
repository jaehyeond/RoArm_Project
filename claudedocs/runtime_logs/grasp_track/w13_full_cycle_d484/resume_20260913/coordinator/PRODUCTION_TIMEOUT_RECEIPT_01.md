# W13 단일 본 실행 — 시간 한도 부분 종료 수신

작성: 2026-09-14 06:18 KST. 신규 물리 변수 []·새 물리 실행은 기존 GO의1회뿐이다.
이 문서는 종료 관측 영수증이며 전체cycle·분류·시각 계약 PASS가 아니다.

## 절차와 원본 확인

1. 원래 GO_production_rev28_01.md에 따라09-13 21:25:01KST 시작.
2. 09-14 06:00:59 root가 정확PID2558081/2558084/2558095의 호스트 생존을 확인.
3. 예정 TERM 경계06:05:01에 실행기가 child 그룹2558095에 SIGTERM15 발행.
4. child가 sync16303/t24.486803s 경계에서 SIGNAL_STOP을 기록하고 부분 원자료 저장.
5. 06:05:20 종료. root가 EXECUTION_RECEIPT/RUN_STATUS 전문·stdout·최종JSON을 읽고 정확3PID 호스트 부재 확인.
6. 생산msg_cdf29ac87e78 및msg_b2776f90c24f로 원래 launch 최종exit124와 해시를 수신했다.
7. 자동 Rerun/Isaac 후처리는0. 기존 총5400초씩의 미사용 재생 범위에서 새경로/부분표시 CPU 준비만 허용했고 아직 실행 GO는 없다.

## 실행 결과

- 정리 포함 단계 시간31218.753855초(약8시간40분18.754초)≤32400.
- child returncode0은 확인됐지만 timed_out=true/outcome=timeout/success=false/runner exit124.
- gracefully_stopped=true, killed_after_grace=false, stderr0바이트, 잔여 그룹/후손0.
- runner signals_received=[]와 child signal_received=15/signal_count=1은 서로 다른 의미다.
- 실제 child의 SIGNAL_STOP 문구가 먼저, 그 뒤 return_home_end 표식이 저장됐다. 생산 첫 메시지의 역순 설명은 원본으로 정정 요청했다.

## 원자료의 관측값 — 독립 배열/기하 검수 대기

- JSON 물리시간24.486802938s는9자리 직렬화 값이며, 저장16304sync/283입자프레임.
- abort_class=SIGNAL_STOP, diverged=false. return_home_end 표식만으로 완료 판정 금지.
- HOME 립 시작[47.8277,0,504.9088]mm/끝[44.2021,0,475.557]mm, 목표 대비29.57491mm.
- 기록된 최종 분류: source19712/receiving_bin0/tool_residual0/spill73/in_flight0/ambiguous215.
- 용기 확정분류0/가능상한11개0.2228g. 정확한 정착 배출량은 미확정.
- 정착창0.25초에는3프레임(t24.302756939/24.402781938/24.486802938), 최대간격0.100025초, cadence_ok=false/exact_single_value_allowed=false.
- bridge 사전 인증227구간 worst slack0.06566128927329115m, 실제 precheck227/227 worst0.07268225246260104m. certify 때 물리step수7336→7336. 전체 경로나 실물 실현성으로 확대하지 않는다.

## 독립 감사의 규약 불일치 보고 — FAIL 소급 변경 금지

감사msg_95f55851779d는 phase-only transition 기대값과 source 하단 판정의 차이를 보고했다. 처음에는 oracle 결함으로 해석했지만 root는 원문에서 RAW_SCHEMA_REQUIRED.md:52와:76-77이 바로 그 엄격한 규약을 요구함을 확인했다. inventory_geometry.py:21/36의 설명과 실제:107~110 source 하단식도 다르다. 실제 선행 승인 erratum이 확인되지 않으면 규약 불일치로 유지한다. 생산식 그대로의 재현 진단을 독립 acceptance PASS의 대체로 사용하지 않는다. 원자료/criteria/옛 실패를 수정하지 않는다.

## root 직접 확인한 SHA256

원본 폴더:
`/home/cgxr/orca/workspaces/RoArm_Project/w13-full-cycle/claudedocs/runtime_logs/grasp_track/w13_full_cycle_d484/resume_20260913/implementation/run_01/`

| 파일 | SHA256 |
|---|---|
| w13_cycle_seed460.json | e482b939a4cf42e5284b9b15e36a9f3613e989161ec538475e2c05f4260d9340 |
| w13_cycle_seed460.npz | 529f422e962730b17b520c3c9af2261aae7c431ec5e078cd2c368dfab0c46b0f |
| EXECUTION_RECEIPT.json | 1f71a56e1a386a6cb14e16f7a284c00026c1ea6c17b740662fc124c961fdc982 |
| RUN_STATUS.json | 1f78a918195d3f3eebf99074442bbfc1481d4a57395bcb6bd48fa84537312d70 |
| HASH_VERIFICATION_RECEIPT.json | 60e605f66a4fc8c1b4c7f79ccf3ace87f473c74fe707cc77d9488cd3c413a11f |
| timeline_seed460.json | 725dd7e3a962e3f2d9ed9083d018311bf37332ac40d45f812175f9ca8d592355 |

NPZ338283382바이트/JSON80301바이트. stdout/stderr 해시는 원본 실행 영수증과 생산 보고에 있으며 최종 보존 검수에서 함께 확인한다. 실물/설치/학습/A·B·C/새 물리/commit/push 없음.
