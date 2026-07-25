# 2026-07-24 — Grasp G0a D378 ephemeral identifier provenance and workload authority repair

## 1. 무엇을 왜 확인했는가

D377은 D375와 같은 P34 acquisition workload를 실행한 뒤
`UsdUtils.StageCache.Get().Erase(stage)`를 한 번 호출했고 정상 종료했다. 그러나 당시
사전등록 비교기는 두 workload hash가 다르다고 판정했으므로 D377의 공식 artifact는
`D377_STAGECACHE_ERASE_BEFORE_CLOSE_LOCALIZATION_FAIL_STOP`으로 동결되었다.

D378의 목적은 그 차이가 실제 geometry/property workload 차이인지, 실행마다 달라지는
진단 문자열 때문인지 immutable D375/D377 원증거만 읽어 판정하는 것이었다. D377 artifact를
소급 수정하지 않고 새 canonical authority를 forward-only로 등록했다.

이번 case의 신규 변수:

1. `ephemeral_identifier_exclusion_and_normalized_witness_authority_v1`
2. `ascii_only_corrected_observability_projection_v1`

Isaac/PhysX launch, USD/collider write, q5, physics step, contact query, cylinder, target/IK/path,
material/mass/actuator/physics setting change는 모두 금지했다.

## 2. Git 및 입력 동결

- 시작 시 `HEAD == origin/master ==
  2acb5b99567946d343e95e61087357193da0826c`, subject `D377(376case)`.
- 승인된 D378 변경 전 worktree는 clean이었다.
- 입력은 D375 attempt2, D377 단일 attempt, D334 collision-table sidecar와 그 callback/
  property JSON으로 제한했다.
- D375/D377/D334 sidecar, 완료된 D378 attempt 경로는 inventory와 SHA-256으로 동결했다.
- commit/push는 수행하지 않았다.

## 3. 실행 순서와 forward-only 실패 보존

### Attempt1 — 사전등록 실패, 감사 0회

`attempt1_ephemeral_identifier_provenance_and_workload_authority_repair`는 실제 감사 전에
Git dirty-path preflight에서 멈췄다. 새 output root가 허용 목록에서 빠졌고 porcelain status
첫 열 처리도 잘못되었다. offline authority audit, Rerun, board는 모두 `0`이다. 경로는
동결하고 덮어쓰지 않았다.

### Attempt2 — 권위 감사 1회 PASS, 설명판 수동검사 FAIL

`attempt2_preregistration_status_order_repair`에서 preflight를 고친 뒤 D378의 유일한
offline authority audit를 한 번 실행했다.

- evidence checks `23/23` PASS
- failure-capable negative controls `11/11` PASS
- offline audit/retry `1/0`
- Isaac/PhysX/q5/physics/contact/cylinder/USD/collider work 모두 `0`

수치 권위는 PASS했지만 1920x1080 설명판의 마지막 outcome 줄이 파란 상자 아래로 넘어가
manual visual inspection이 FAIL했다. attempt2 completion은 정확히 manual 관련 2개 check만
false로 기록하고 동결했다.

### Attempt3 — 경로 등록 문자열 preflight 실패, board 0장

START_HERE의 attempt3 경로를 두 code span으로 줄바꿈했는데 preflight는 한 줄의 exact path를
요구했다. 나머지 preregistration check는 모두 true였고, 실제 authority audit와 board
generation은 `0`이다. phase/preregistration/exception 3개 파일을 exact hash로 동결했다.

### Attempt4 — board 1장, subpixel containment gate FAIL

경로 등록 문자열만 고쳐 exact 1920x1080 board를 한 번 생성했다. 자동 검사는 정확히
`outcome_body_inside` 한 항목만 false였다.

- 실제 아래 inset: `6.3999999999999915px` (`6.40px`)
- 등록 minimum: `0.006 * 1080 = 6.48px`
- shortfall: `0.08000000000000895px`
- 다른 containment, box overlap, footer separation은 모두 PASS

육안으로는 잘리지 않았지만 등록 gate를 사후 완화하지 않고 attempt4를 FAIL_STOP으로
동결했다.

### Attempt5 — 측정된 위치 보정과 최종 완료

등록 minimum `6.48px`은 유지하고 outcome body 한 블록의 Y만 `+0.002` normalized
(`+2.16px`) 이동했다. authority audit와 Rerun Viewer는 재실행하지 않았다.

- preregistration `15/15` PASS
- target text match `1`
- 측정 아래 inset `8.559999999999963px`
- containment `12/12`
- box-row/footer overlap `4/4`
- 같은 box의 title/body separation `6/6`
- visual repair evidence `14/14`
- original-resolution manual checks `6/6`
- final completion `21/21`

최종 verdict:
`D378_EPHEMERAL_IDENTIFIER_PROVENANCE_AND_WORKLOAD_AUTHORITY_REPAIR_PASS`.

## 4. 실제 authority 결과

### 4.1 왜 기존 V1 hash가 달랐는가

D375와 D377의 V1 selected digest는 각각 다음이었다.

- D375:
  `ec930163ac2a9cdbf7342630dccd34d5467fa3618dfd0d6213066fbaa12b0b7b`
- D377:
  `758504733115b8740a972fe99ea63f9303d5759505d03a29e1e9c9570fa13c81`

selected difference `68`개는 정확히 다음 두 종류였다.

- callback witness file SHA `34`: 각 JSON의 `request_return_repr` 안 runtime object memory
  address만 실행별로 달랐다.
- `prototype_path_diagnostic` `34`: generated `__Prototype_N` ordinal만 달랐다.

원 witness와 raw SHA는 provenance로 보존했다. canonical 비교에서 제외한 것은 위의
run-dependent diagnostic projection뿐이다.

### 4.2 corrected workload는 같았는가

두 run의 corrected authoritative workload SHA-256은 모두 다음과 같았다.

`28aadb5ff26270039df58f7cd06080bf7afcdec001402e886a6edf1483fdfe31`

corrected selected diff는 `0`이었다. 실제 callback payload도 `34/34` exact이며 합계는:

- vertices `314`
- indices `1016`
- original polygons `262`

normalized witness aggregate SHA-256도 양쪽 모두
`0a56d7900470f6f75d5f63ac415d7d0f4cca5c5d941951280387ae2378abfe8c`였다.

### 4.3 property 차이는 무엇이었는가

raw property difference `40`개는:

- opaque runtime `path_id` `38`
- elapsed time `2`

뿐이었다. 이를 정확한 manifest로 제외한 normalized property diff는 `0`, 양쪽 digest는
`4710c18232e2d2259c569d01b6326bbea20b36507e5aeb9a85fbe15ca94f7c1f`였다.
mass, COM, inertia, axes, volume, AABB, local pose, semantic path, result와 part counts는
authority에 남겼다.

### 4.4 음성 대조군

총 `11/11`이 PASS했다.

- memory address/prototype ordinal/path_id/elapsed perturbation은 normalized authority를
  바꾸지 않아야 했다.
- vertex, semantic path, property volume perturbation은 authority를 반드시 바꿔야 했다.
- witness SHA 또는 prototype diagnostic 한 종류만 제외하면 여전히 mismatch여야 했다.
- 허용 범위를 넓힌 exclusion manifest는 거부되어야 했다.
- raw V1 comparator는 원 mismatch를 계속 감지해야 했다.

## 5. 종료 현상에 대한 제한된 해석

동일한 corrected workload 아래 관측된 pair는:

- D375: explicit Erase `0`, timeout true, elapsed `920.3908159369603s`, return `-9`
- D377: explicit Erase `1`, Erase contract PASS, elapsed `6.733121555997059s`, return `0`

따라서 이 pair에서는 StageCache retention이 non-exit의 conditional trigger였다는 support가
생겼다. 그러나 다음은 여전히 증명되지 않았다.

- StageCache Erase가 모든 경우의 필요조건인지: `null`
- exact native root cause: `null`
- stage object destruction: 증명 안 됨
- NVIDIA bug 5948099와의 exact identity: 증명 안 됨

D373은 Erase 없이도 return `0`이므로 universal necessity 주장은 금지한다. D377의 동결된
FAIL_STOP artifact도 수정하지 않았다.

## 6. 시각화 결과

- 최종 exact board: `1920x1080`, SHA-256
  `e788c54e34b9f3f3adbb0e2ed7322432f38d3587cd079761a36c15fb30b96b32`
- reused save-only RRD: `63,972B`, SHA-256
  `6e605b48e88e6aa0dce4b264f2db193b3dbfd9ff702895ea401a57e6763344e8`
- reused RBL: `43,735B`, SHA-256
  `08e72cbccf3eacc304fb941cec98838ab7de5660232f94378593714d26406e49`
- Rerun `0.34.1` strict footer/entity/timeline/component validation PASS
- reused Viewer screenshot: HiDPI physical `3840x2160`; audit 5 rows와 static boundary 3 rows를
  원본 해상도로 직접 검사했다.

Rerun은 관찰 자료이고, equality authority는 원 JSON/callback payload/canonical SHA다.

## 7. 무엇을 증명하지 않았는가

- full P34 live identity: `null`
- A64↔P34 physics equivalence/speed: `null`
- 29x50mm 실제 대상의 contact/tipping: `null`
- q5 closure와 grasp feasibility: `null`
- target/IK/path repair justification: `null`
- `g0a_pass=false`

역사적 D362의 34x90mm cylinder 물리결과는 그대로이며, 사용자 확정 실제 대상
29x50mm에 전이하지 않는다. 실제 mass는 물체 도착 후 측정한다.

## 8. 핵심 산출물

- authority evidence:
  `claudedocs/runtime_logs/grasp_track/g0a_d378/attempt2_preregistration_status_order_repair/d378_workload_authority_repair_evidence.json`
  SHA-256 `e9c3d1cadf9cc9516d0d08792a44b6d824fea7ac8cd0849dffc9a25f3bafda88`
- final board:
  `claudedocs/runtime_logs/grasp_track/g0a_d378/attempt5_measured_outcome_inset_repair/d378_corrected_workload_authority_repaired_1920x1080.png`
- attempt5 layout:
  `claudedocs/runtime_logs/grasp_track/g0a_d378/attempt5_measured_outcome_inset_repair/d378_attempt5_layout_validation.json`
  SHA-256 `d69be57cb7a88407d29e3c78b14bdc1df7105440789b97aee1eef9193a055cd7`
- attempt5 manual inspection:
  `claudedocs/runtime_logs/grasp_track/g0a_d378/attempt5_measured_outcome_inset_repair/d378_attempt5_manual_visual_inspection.json`
  SHA-256 `fd93103fb5094de1d2ac6e13d1a70898795de569f8d5399ff36fb0fa1636f499`
- final completion:
  `claudedocs/runtime_logs/grasp_track/g0a_d378/attempt5_measured_outcome_inset_repair/d378_final_completion_summary.json`
  SHA-256 `90a59cf01db98b9c4d54229611da5d41202b6967348138df14fdbca071da22dd`

## 9. 다음 승인 경계

D378은 종료하고 모든 attempt를 동결한다. 다음 권장 후보는 별도 승인
`D379 [p34_full_live_identity_classifier_resume]`이다. D377의 clean-exit lifecycle과 D378의
corrected workload authority를 상속하되, P34 full authored↔live callback
surface/bounds/topology-volume/property identity만 확인한다.

29x50mm target geometry rebase, 중앙 높이/반경 pose, 질량 계약, A64/P34 physics,
q5/contact/settle/hold/lift, target/IK/path는 각각 그 뒤 별도 승인이다.
