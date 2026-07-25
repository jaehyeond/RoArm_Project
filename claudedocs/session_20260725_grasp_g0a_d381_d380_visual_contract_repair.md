# D381 — D380 시각자료 계약 수리

Date: 2026-07-25 KST

Case: `D381 [d380_visual_contract_repair]`

Frozen attempt:
`claudedocs/runtime_logs/grasp_track/g0a_d381/attempt1_d380_visual_contract_repair/`

이번 case의 신규 변수:

1. `d380_board_pixel_layout_repair_v1`
2. `d380_rerun_notification_buffer_layout_v1`

## 1. 무엇을 왜 확인했는가

D380의 수치 감사는
`D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`로 완료됐지만,
발표용 보드에서는 제목이 겹치고 왼쪽 막대 라벨이 잘렸으며 Rerun 시작 알림이
오른쪽 요약을 가렸다. D381은 이 표현 문제만 수리하도록 승인됐다.

D381은 D380의 immutable evidence/CSV/RRD/RBL/PNG/검증 기록 11개만 읽고,
D379 재독해나 거리·부피·꼭짓점 재계산을 하지 않도록 했다. Isaac, Kit, PhysX,
USD, collider, cylinder, physics, q5, contact, target/IK/path 및 물리 설정은 모두
금지했다.

이 세션의 failure-capable 항목은 자동 글자 경계 검사, RRD/RBL strict 계약,
실제 Viewer 한 번, 원본 해상도 육안검사였다. 따라서 연구 세션 진행 규칙을
만족하며, 새로운 과학 실험이나 물리 재실행은 승인 범위가 아니었다.

## 2. 사전 점검과 사전등록

실행 전에 다음을 다시 확인했다.

- `HEAD == origin/master ==
  2acb5b99567946d343e95e61087357193da0826c`, subject `D377(376case)`.
- 기존 사용자 소유 D378-D380와 상태 문서 변경을 보존했다.
- D380 입력 11개의 SHA-256이 모두 등록값과 일치했다.
- script AST/compile PASS, 금지 NVIDIA/runtime import 없음.
- `numpy==1.26.0`, `psutil==5.9.8`, `rerun-sdk==0.34.1`,
  `matplotlib==3.10.3`, `Pillow==11.3.0`.
- 최초 초안은 `pyarrow==21.0.0`을 잘못 기대했지만 설치본은 `23.0.1`이었다.
  실제 worker 전에 기대값을 설치본과 맞추고 다시 정적 감사했다. 이는 패키지
  설치나 변경이 아니며, 사전등록 전에 완료했다.

사전등록 결과:

- checks `15/15` PASS
- failure-capable negative controls `10/10` PASS
- preregistration SHA-256
  `fdffbe48ebd6af275ca534acf197952d3f8430287a20bbdf06af7d596512cc69`
- worker/retry 등록 `1/0`
- Viewer/retry 등록 `1/0`
- worker watchdog `300s`, Viewer timeout `240s`

Source:
`d381_preregistration.json`.

## 3. 실제 실행 순서

1. Supervisor가 exact interpreter로 offline presentation worker를 한 번 시작했다.
2. Worker는 preregistration, source hash, D380 input hash, dirty baseline을 다시
   확인했다.
3. D380 저장 보드에서 link5와 gripper의 동결 형상 패널을 잘라 재사용하고,
   D380 CSV에 이미 저장된 막대 값과 D380 evidence의 저장 사실을 새
   `1920x1080` 보드에 배치했다.
4. 보드 PNG는 정상 저장됐다.
5. 이어서 Matplotlib가 측정한 글자 경계상자를 layout-validation JSON으로
   저장하려는 순간 worker가 `TypeError`로 종료됐다.
6. `_render_board()`가 반환하지 못했으므로 다음 program-order 단계인
   recording-only RRD 추출, 새 RBL, presentation merge, Rerun validation,
   Viewer capture에는 도달하지 않았다.
7. 자동 재시도는 하지 않았고 attempt1 경로를 동결했다.

## 4. 정량 결과

Supervisor:

- actual worker/retry `1/0`
- return code `1`
- elapsed `0.7154044299386442s`
- timeout `false`
- SIGTERM/SIGKILL `false/false`
- 종료 후 process group residue `false`
- source/input/dirty-baseline integrity `true/true/true`

Source:
`d381_offline_worker_supervisor.json`, SHA-256
`e2930719620aba6d67fd929c74f150d48ab4c7562ad0e27c65964721c399298a`.

첫 실패:

- exception:
  `TypeError: Object of type bool_ is not JSON serializable`
- failure phase: `board_layout_validation_json_write`
- board 저장 뒤, `d381_board_layout_validation.json` 기록 중 중단
- 잘린 JSON 크기 `144B`; authority로 사용 금지
- actual Rerun Viewer invocation `0`
- actual recording-only projection/RBL/presentation merge `0/0/0`

Source:
`d381_offline_worker_stderr.log`, SHA-256
`2218c2837dee983d451f0b38e6e0c1b398bf3679daf66051918bd5cb3dcbfeec`;
`d381_fail_stop_attestation.json`, SHA-256
`e62f6aba1340bfe54d87638564d103e18248f4e61d033cc5711026ba66939b0b`.

생성된 부분 보드:

- exact `1920x1080`
- `230110B`
- SHA-256
  `19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d`
- 제목/부제 겹침과 왼쪽 라벨 잘림은 육안상 해소됐다.
- frozen geometry crop 상단에 작은 dash 형태 잔여 픽셀이 보이지만 형상을
  가리지는 않는다.
- 자동 layout JSON과 Rerun 계약이 미완료이므로 이 보드를 D381 completion
  PASS 자료로 승격하지 않는다.

Source:
`d381_d380_visual_contract_repaired_1920x1080.png`;
`d381_partial_board_visual_inspection.json`, SHA-256
`58639f2cf3a76bc40ad6921479b32d1645abcf8498fdaba636bc1b13995ba0e1`.

## 5. 원인 교차검증

`record_bbox()`의 좌표는 명시적으로 `float(...)`로 바꿨지만,
`inside_canvas_with_6px_margin`의 네 비교 결과는 변환하지 않았다.
Matplotlib `Bbox` 좌표형은 `numpy.float64`이므로 비교 결과와 `and` 식의
최종값은 `numpy.bool_`이다. Python 표준 JSON encoder는 이를 자동으로
직렬화하지 않는다.

독립 최소 재현에서도 다음이 일치했다.

- Bbox coordinate: `float64`
- comparison: `bool_`
- boolean `and` result: `bool_`
- `json.dumps`: 같은 `TypeError`
- built-in `bool(...)` 변환 뒤에는 JSON 직렬화 가능

따라서 이번 중단은 Isaac, GPU, PhysX, Rerun 또는 D380 형상 데이터 문제가
아니다. D381 자체의 layout-validation 직렬화 경계 결함이다.

## 6. 판정과 보존 경계

Operational verdict:

`D381_BOARD_VALIDATION_JSON_SERIALIZATION_FAIL_STOP`

이 판정의 뜻:

- D381 presentation completion: FAIL_STOP
- D380 numeric verdict: 그대로 보존
- P34 authored-to-cooked identity: `false`
- `g0a_pass=false`
- actual OPEN jaw clearance/contact, cylinder physics/tipping, q5 closure,
  grasp feasibility, target/IK/path justification: 모두 `null`

금지 항목의 actual count는 모두 `0`이다:
D379 read, numeric/geometry audit, asset/USD read/write, collider materialize/
regenerate, automatic decomposition, Isaac/Kit/PhysX, cylinder, physics/public
forward, q5/contact, target/IK/path/pose 및 material/mass/actuator/physics
setting change.

D381 attempt1의 script, 잘린 JSON, PNG, stdout/stderr, supervisor를 수정하거나
같은 경로에서 재실행하지 않는다. Commit/push도 수행하지 않았다.

## 7. 다음 승인 경계

다음 최소 후보는 아직 미승인:

`D382 [d381_layout_validation_native_scalar_serialization_repair]`

권장 신규 변수:

1. NumPy/Matplotlib scalar를 Python JSON-native `bool`/`float`로 재귀 정규화
2. JSON 전체를 먼저 메모리에서 직렬화한 뒤 exclusive-create하여 잘린 JSON 방지

D382도 새 forward-only 경로의 observability-only case여야 한다. Immutable
D380 입력과 D381 표시 계약을 그대로 상속하고 worker `1`, retry `0`, Viewer
최대 `1`, retry `0`을 유지한다. P34 representation repair, live identity,
29x50 target rebase, Isaac/PhysX, physics/q5/contact는 별도 승인 전까지 금지한다.

