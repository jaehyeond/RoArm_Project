# D382 — D381 layout-validation JSON-native scalar/serialize-before-create 수리

Date: 2026-07-25 KST

## 1. 무엇을 왜 확인했는가

D381은 발표용 정적 보드를 만든 뒤 자동 배치검사 JSON을 기록하는 과정에서
`numpy.bool_`를 Python 표준 JSON writer가 처리하지 못해 멈췄다. 그 결과 144바이트의
잘린 JSON만 남았고, Rerun 배치와 Viewer 검사는 시작되지 않았다.

D382의 목적은 이 실패만 새 forward-only 경로에서 수리하는 것이었다.

- `JSON-native scalar normalization`: NumPy가 만든 `bool_`, 숫자 scalar를 Python의
  표준 `bool`, `int`, `float`로 재귀 변환한다.
- `serialize-before-create`: JSON 전체 문자열을 메모리에서 정상 생성·재파싱한 뒤에만
  새 파일을 exclusive-create한다. 직렬화 오류 때문에 내용이 잘린 파일이 생기는 일을
  막는다.

이번 case의 신규 변수:

1. `json_native_recursive_scalar_normalization_v1`
2. `serialize_before_exclusive_create_v1`

Isaac/Kit/PhysX/USD/collider/원통/physics/q5/contact/target·IK·path와 물리 설정은
모두 범위 밖으로 동결했다.

## 2. 부팅·Git·동결 입력 확인

- 실제 `HEAD == origin/master ==
  b880bc8f28c269f56f05a757dc725619d88c77b1`.
- 시작 시 worktree는 clean이었고, D382 승인 뒤 `START_HERE.md`와 새 D382 wrapper만
  작성했다.
- D380 입력 11개와 D381 입력 10개, 합계 21개 해시가 모두 등록값과 일치했다.
- 동결 D381 script SHA-256:
  `b58a60ad8f0d8873973c9171f03cb7bd75401c205db074a49a9a498d98adbd2d`.
- D382 script SHA-256:
  `ed437ec6b09c53476457cb62e62427028f34c840ed51d8c5db2798106bbccaaf`.
- 동결 D381 보드 SHA-256:
  `19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d`.
- exact interpreter와 패키지는 Python 3.11.14, Matplotlib 3.10.3,
  NumPy 1.26.0, Pillow 11.3.0, psutil 5.9.8, PyArrow 23.0.1,
  Rerun SDK/CLI 0.34.1이었다.

동결 D381 구현은 수정하지 않고 wrapper로 읽었다. D381의 출력 경로 22개를 모두
D382 새 폴더로 돌렸고, JSON writer·canonical hash·Viewer 호출 guard만 승인 범위에
맞게 감쌌다. 독립 실행 전 감사와 델타 재감사 모두 blocker 없음으로 판정했다.

## 3. 사전등록

출력:

`claudedocs/runtime_logs/grasp_track/g0a_d382/attempt1_d381_layout_validation_native_scalar_serialization_repair/`

사전등록 결과:

- checks `25/25` PASS
- failure-capable negative controls `15/15` PASS
- 입력 해시 `21/21` exact
- worker/retry `1/0`
- Rerun Viewer 최대/재시도 `1/0`
- worker watchdog `300s`, Viewer timeout `240s`
- preregistration SHA-256:
  `926cc4dc3ffe474a097ac96b8792ab7cfb9e3926a39b062610bdd8eea6b11ab8`

음성대조군은 raw `numpy.bool_`가 표준 JSON encoder에서 실패하는지, 정규화 뒤 Python
`bool`이 되는지, 지원 범위 밖 NumPy array·비문자열 key·NaN·임의 object가 파일 생성
전에 거부되는지, 두 번째 worker/Viewer 요청이 계약에서 거부되는지를 확인했다.
두 negative output path는 실제로 생성되지 않았다.

원 증거:

- `d382_preregistration.json`
- `d382_phase_markers.jsonl`

## 4. 실제 실행 순서와 결과

### 4.1 worker 시작

승인된 worker를 정확히 한 번 실행했다.

- actual worker/retry `1/0`
- return code `1`
- elapsed `1.4710988369770348s`
- timeout/TERM/KILL/process-group residue: 전부 false
- supervisor SHA-256:
  `007052bcc2fae6193f3f4fe0069bc82786022659bc45f5f023ecac2af738b316`

### 4.2 JSON 직렬화 수리 — PASS sub-result

이번 case의 직접 수리 대상은 통과했다.

- 배치검사 JSON은 완전한 `10877B` JSON으로 기록됐다.
- 자동 layout checks `9/9` PASS.
- 저장된 `inside_canvas_with_6px_margin` 값들은 전부 Python/JSON `bool`이다.
- artifact: `D382_BOARD_LAYOUT_VALIDATION_V1`
- SHA-256:
  `7b961cdf8bd606c05438e120728fe243653262de81aa45947a1db0b1c03ab79c`

따라서 D381의 `numpy.bool_` 오류와 144바이트 부분 JSON 문제는 D382에서 재발하지
않았다.

### 4.3 정적 1920×1080 보드 — PASS sub-result

- 크기 `1920×1080`
- 파일 크기 `230110B`
- SHA-256:
  `19bd70781403eb11c4eaefb6adb60ab91a5e6ca9f67f2929548f8afff0b7f06d`
- D381 보드와 byte-for-byte exact
- 원본 해상도 육안검사에서 제목·두 geometry panel·요약·두 chart·범례·footnote가
  보였고, 의사결정을 가리는 겹침이나 등록 label clipping은 보이지 않았다.

이는 D380의 숫자나 geometry를 새로 계산한 결과가 아니라, 동결 D381 그림이 그대로
재생성됐다는 뜻이다.

### 4.4 RRD/RBL 생성 — PASS sub-result

`RBL`은 Rerun Viewer의 화면 배치를 저장한 파일이고, `RRD`는 표시할 기록 데이터다.

- D380 source RRD bit-exact copy: `294449B`,
  SHA `7ae91348bc6cc64b583c1e92ff2ea8776647a660042471a075d9216b9fadcaff`
- recording-only RRD: `234179B`
- recording equivalence JSON: PASS,
  SHA `ba811049c3e21fc13300d2228445cbb06225fb09fde38e0e9ad61ebaf575b618`
- 새 RBL: `69301B`,
  SHA `979ddd6b4a32bfc97e13d75dfb99625af0d0bed90fc1fd9588347667f284b28c`
- presentation RRD: `305981B`,
  SHA `6c4ad99428f8da0ef842031b161e69db906971084da6d3444c1b76c8c27a7d9a`
- 사후 read-only `rrd verify`: 두 파일 모두 오류 없이 검증됨.

### 4.5 첫 실패 — RBL 내용이 아니라 검사 명령의 verbosity 불일치

동결 D381 validator는 다음 순서였다.

1. `rerun rrd print -v <RBL>` 실행
2. 그 출력 문자열에서 notification query, summary query, 네 spatial-view 이름 검색
3. 모두 없다고 판단해 `RuntimeError` 발생
4. loopback preflight와 Viewer 호출 전 종료

원 stderr의 실패 checks:

- `notification_buffer_query_serialized=false`
- `summary_query_serialized=false`
- `four_spatial_view_names_serialized=false`
- 반면 CLI return, 빈 notification entity, D380 input exact는 true

설치된 Rerun CLI 0.34.1의 local `rrd print --help`는 다음을 명시한다.

- `-v`: fully-qualified name을 보여주는 summary
- `-vv`: chunk metadata header, data hidden
- `-vvv`: chunk metadata와 실제 data

read-only 교차검사 결과:

- 같은 RBL을 `-v`로 읽으면 등록 marker `0/6`
- 같은 RBL을 `-vvv`로 읽으면 등록 marker `6/6`
- 두 명령 모두 return `0`

즉 RBL에서 화면 이름과 query가 빠진 것이 아니다. 값이 숨겨지는 `-v` 출력에서 값
문자열을 찾은 validator가 정상 RBL을 FAIL로 판정한 false negative다.

근거:

- `d382_offline_worker_stderr.log`
- `d382_rbl_print_verbosity_provenance_audit.json`
- frozen source
  `sim_scripts/cyl34_top_view_d381_d380_visual_contract_repair.py:1058-1089`

## 5. Viewer와 육안검사 경계

실패는 loopback preflight와 Viewer command보다 앞에서 발생했다.

- actual Rerun Viewer invocation `0`
- automatic Viewer retry `0`
- Viewer receipt 없음
- Rerun screenshot 없음
- full Rerun validation JSON 없음
- manual inspection template/완료본 없음
- worker claim과 completion summary 없음

따라서 정적 보드는 부분 육안검사 PASS지만, D382 전체 visualization completion은
PASS가 아니다. RRD/RBL 파일 존재나 `rrd verify`만으로 Viewer 화면을 봤다고 보고하지
않는다.

부분 검사:

- `d382_partial_visual_inspection.json`
- SHA
  `f9bace3484e5fa350f4e3350bc7832e069530b234e0dacb0a1145cdc32122168`

## 6. 최종 판정

Operational verdict:

`D382_LAYOUT_JSON_SERIALIZATION_REPAIR_PASS_RBL_PRINT_VERBOSITY_FALSE_NEGATIVE_FAIL_STOP`

뜻:

- D381의 JSON-native scalar와 부분 파일 문제는 수리됐다.
- 정적 보드와 RRD/RBL 생성도 성공했다.
- 그러나 동결 validator의 `-v`/`-vvv` 의미 혼동 때문에 Viewer 전에 멈췄다.
- 실제 Viewer 화면을 검사하지 못했으므로 D382 전체는 FAIL_STOP이다.
- 같은 D382 경로를 재실행하거나 덮어쓰지 않는다.

Fail-stop attestation:

- `d382_fail_stop_attestation.json`
- SHA-256:
  `73f51b45a748cf1e6f5ad5fec6bca5396f5f2348535741ac60d4132f54a6ebe5`

## 7. 과학·물리 상태는 변하지 않았다

이번 실패는 Isaac Sim, GPU, PhysX, collider geometry 또는 원통 파지 실패가 아니다.
그 구성요소를 실행하지 않았다.

- D380 numeric verdict:
  `D380_FAILED_PART_PROVENANCE_AUDIT_PASS_REPAIR_REQUIRED`
- P34 authored-to-cooked identity: false
- P34 representation repair/live identity: null
- actual OPEN jaw clearance: null
- 29×50 원통 contact/tipping/q5 closure/grasp: null
- target/IK/path justification: null
- `g0a_pass=false`

## 8. 다음 승인 경계

다음 최소 후보는 아직 미승인 D383 observability-only case다.

`D383 [d382_rbl_data_value_verbosity_validation_resume]`

- immutable D382 board/layout/RRD/RBL만 읽는다.
- 신규 변수 1개: RBL data-value 검사 권위를 `rrd print -v`에서
  `rrd print -vvv` 또는 동등한 structured exact reader로 변경한다.
- D382 파일을 다시 만들거나 같은 경로에서 worker를 재시도하지 않는다.
- 새 forward-only 경로에서 strict validation과 Viewer 최대 1회/no retry,
  원본 해상도 육안검사만 재개한다.
- Isaac/PhysX/collider/원통/physics/q5/contact/target·IK·path는 계속 `0`.

D383가 별도 승인·PASS하기 전에는 D381/D382 presentation completion을 PASS라고 하지
않는다. P34 representation/live identity와 29×50 물리시험은 그 뒤에도 각각 별도
승인이 필요하다.
