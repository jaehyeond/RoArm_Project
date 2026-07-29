# D407 harness 직접 검증 + adversarial 리뷰 발사 — 사용량 한도 비상 인계

Date: 2026-07-29. 이번 case의 신규 변수: 0 (이 세션은 harness 검증/정적 준비만;
D407 case 자체의 신규 변수 1건은 설계 확정 doc 참조). 유저가 Fable5 주간 한도
임박을 통지 → 세션 중단·인계 (95% 프로토콜 준용). **다음 세션은 다른 모델일 수
있음 — 이 문서만으로 이어받을 수 있게 전부 명시.**

Session progress rule 정당화: 이 세션의 실패 가능 실험 = science 14함수
byte-identity 직접 재검증 + 자산 7+7 재실측 (FAIL이면 harness 기각 강제) —
실제 수행, 전부 PASS. runtime 실행 없음 (승인 범위 밖 — 변동 없음).

## 1. 이 세션이 직접 검증 완료한 것 (전부 본 세션 실측 — agent 자기보고 아님)

1. **부트 사실 확인**: dirty 85 == START_HERE 주장 실측 일치. 구세션 scratchpad
   스펙 3파일 생존 (`/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/4d3cddb8-6495-4150-aa6b-41bc28b3d6f3/scratchpad/`
   {d407_worker_spec.md, d407_controller_spec.md, d407_builder_runner_spec.md} +
   map_remaining.txt; tool-results dir는 휘발 — map 보고 2건 소실, 재구성 불요).
   worker sha `1a0c8313fbaaf68f9e97c7c082036dbb1a22d34ccd6ec4fef067993adbbdbf9c`
   == 설계 doc §5의 pre-injection 보고와 bit-일치. controller sha
   `91b4647ef88daed343691092892969d256f3c05b43b3787f1bb531762a4a4f6e`.
   양쪽 ast.parse PASS (worker 3,985 / controller 1,284라인).
2. **science 14함수 byte-identity 직접 재검증 PASS (14/14)**: d362:889-939
   메커니즘(inspect.getsource + D362→DXXX/D407→DXXX 치환)의 정적 복제
   (ast.get_source_segment 대칭 추출)로 14함수 전부 normalized_source_equal.
   스크립트: 이 세션 scratchpad `verify_science14_byte_identity.py`
   (`/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/02957381-6c1f-4b7d-ab7c-11323634937c/scratchpad/`).
   worker 자체 런타임 게이트도 실재: worker:716
   `_frozen_d362_science_source_contract` (d362 모듈 import + 동일 치환).
3. **자산 7+7 재실측 완전 일치**: 설계 doc §3.4 표와 파일 수(각 정확 7)/
   sha256 prefix/바이트 크기 전부 일치. 6 non-root A↔B bit-동일, root
   `roarm_m3.usd`만 상이 (A a4be58e8 1,457B / B c02808ab 3,177B).
4. **env pin 실측 전부 일치**: isaaclab python 3.11.14, numpy 1.26.0, psutil
   5.9.8, rerun-sdk 0.34.1, rerun CLI 0.34.1
   (`/home/cgxr/miniconda3/envs/isaaclab/bin/rerun`),
   `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc` 존재 (worker:2960
   FONT_PATH와 일치). 호스트 GPU 정상 (D402-R1 호스트 경계 실측): RTX 4090
   Laptop GPU, CC 8.9, free 14,265MiB ≥ 8192 게이트.
5. **harness 직접 판독 결과 (파일:라인 인용)**:
   - 편차 ④ (final pause) **비오염 확정**: worker:1468 `_pause_timeline`은
     d362:1841과 동일 메커니즘 + `counter_unchanged/clock_unchanged/
     state_bits_unchanged` 자기검증 내장 — 물리 step 추가 없음. worker:3762
     final pause 실패 시 3764 RuntimeError. 500(200+300) step 계약 무변.
   - worker:3396-3435 admission 체크 25종. 3398 status 리터럴
     `== "PREREGISTERED_NOT_EXECUTED"`; 3397 prereg sha vs
     EXPECTED_PREREG_SHA256 (worker:59 = 64-zeros placeholder; controller:51
     동일 placeholder — 주입 대상 2곳).
   - controller B2 분기 정상: 1205-1232 leg 루프 A→B, A 실패 시 즉시
     `_fail_stop` return → B 미발사; inter-leg settle 실패 fail-closed(1231-1232);
     1236-1239 post-run 자산 rehash = §3.4 게이트 ⑤ (completion 전);
     1247-1251 sheet 실패 → manual_inspection 분류; 1253-1263 검수.
   - leg rebind 정상: worker:196 `_configure_leg`(전 leg-전역 재바인드) +
     controller:552 호출; leg_paths 캡처(controller:1211-1217)는 반환 직후라
     leg별 정확.
   - 세션doc 리터럴 체크 분담: worker:3429-3430 (BASE_GIT count≥1 + 헤딩
     `## 3. D407 확정 설계`); controller:448-455 (BASE_GIT ≥1 + controller
     sha 정확 1회 + worker sha 정확 1회 + 헤딩).
   - **prereg 소비 키 전수 (직접 grep)**: worker = leg_asset_pins(394),
     runtime_overlay_contract.allowed_dirty_paths(3394), status(3398),
     case/case_name(3399-3400), new_variables(3401), legs 멤버십(3403),
     git_baseline.head(3416), frozen_input_hashes(3419,3836),
     d334_sidecar_before(3420,3837). controller = 동일 + run_nonce 존재
     (441-442; 560에서 invocation marker에 전사) + legs set=={a,b}(437-438).
     legs payload 심층 소비 없음 (문서화 블록 — builder는 worker LEG_* 표에서
     도출해 기록만).
   - controller 상수: REGISTERED_STATIC_NEGATIVE_IDS 12개(70-85),
     TUPLE_FIELDS 4(64-69), EXPECTED_ZERO_STAGE_COUNTERS 10(86-97),
     REQUIRED_STATIC_TRUE 5(98-104), GPU exact 모델/CC(114-115),
     검수 300s/0.25s(108-109), settle 5s/180s(106-107).
   - worker `_input_paths`(347-365) = 자산 14 + D348/D334/D354×2/URDF/
     D361×2/D362×4/d351.py/d333.py/d361.py = 26 pin.
   - worker 물리 상수 블록(52-121) D362 계약 verbatim 확인: spawn
     [0.30000001192092896, 0, 0.03288299962878227], q5 OPEN
     1.5413000583648682, seed 33201, dt 0.005, 200+300, actuator 80/4/2.5/
     3.14, 임계 0.1N/0.5mm/1.0°/연속2, 원통 r0.017/h0.090/0.72kg/1.5/1.2/0.0.
6. **신규 발견 (중대 — builder 계약 요건, harness 수리 아님)**:
   worker:3418 `git_dirty_subset_of_allowlist`가 leg별 fresh process
   preflight마다 실행되고 `_status_paths`(330-334)는 `--untracked-files=all`
   무필터 전수 열거 → **leg B preflight 시점에 leg A 출력물 전부가 dirty로
   신규 등장한다. 정적 allowlist가 미래 runtime 출력 경로 전수를 포함하지
   않으면 leg B admission에서 attempt 소실.** 설계 doc §3.9의
   "allowed_dirty_paths = 실측 dirty ∪ planned 7"은 **과소**. 해법
   (프로그램적 도출): builder가 `controller._runtime_output_paths()`
   (controller:377; admission `no_preexisting_runtime_outputs`(411,457)와
   동일 소스)를 호출해 attempt-root + 양 leg dir 전 출력 경로를 rel-path로
   전사, allowed_dirty_paths = live dirty ∪ planned 정적 ∪ 미래 출력 전수.
   controller freeze manifest(488-537)는 발사 전 1회 검사뿐이라 추가 영향
   없음. 이 요건은 2-seq-worker 신구조(설계 §3.10 ④)의 정적으로 잡힌 첫
   결함이다.
7. **builder 스펙 정정 1건**: 스펙 A.2의 status 추출 regex `!=` 표기는 오기
   — 실코드 worker:3398/controller:431-432 모두 `==` (regex는
   `prereg\.get\("status"\)\s*==\s*"([^"]+)"` + re.S, 파일당 정확 1회 assert).
8. M2 검증 JSON(구 scratchpad `d407_m2_nvidia_verification.json`) 미생존 —
   prereg `installed_nvidia_primary_sources`/official_sources 전사의 원본은
   설계 확정 doc §4 (durable).

## 2. 진행 중 (백그라운드 — 한도 도달 시 중단될 수 있음)

- **adversarial 리뷰 워크플로우 `wf_64f9c071-806`** (6-lens find → dedupe →
  blocker 3-표/warning 1-표 반박 검증). 이 doc 작성 시점 finished **0/6**.
- journal (완료 agent 결과가 완료 즉시 영구 기록됨):
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/02957381-6c1f-4b7d-ab7c-11323634937c/subagents/workflows/wf_64f9c071-806/journal.jsonl`
  (+ 같은 dir의 agent-*.jsonl = 개별 agent 전문 transcript).
- 워크플로우 스크립트 (재발사용 영구 저장):
  `~/.claude/projects/-home-cgxr-Documents-Robotics-RoArm-Project/02957381-6c1f-4b7d-ab7c-11323634937c/workflows/scripts/d407-harness-adversarial-review-wf_64f9c071-806.js`
- 6 lens 구성: frozen-contract / lessons(D400-P2·D401·D402-R1·D403~D406) /
  ops(supervisor·파일규율) / science(§3.7 판정·delta·관측성) / deviations
  (편차 7건 개별 판정) / builder-contract (소비 전수 열거를
  `/tmp/claude-1000/-home-cgxr-Documents-Robotics-RoArm-Project/02957381-6c1f-4b7d-ab7c-11323634937c/scratchpad/d407_prereg_consumer_enumeration.md`
  로 기록하는 계약 포함).
- **workflow resume은 same-session 한정** → 새 세션은 journal에서 완료분
  회수 + 미완 lens만 재발사 (스크립트 파일 참조; 검증 phase 포함 재구성).

## 3. 다음 세션 필수 절차 (설계 doc §6 + 이 세션 발견 반영)

1. journal 회수 → 리뷰 완결 (미완 lens 재실행) → 채택/수리 결정.
   편차 ④는 §1.5로 이미 비오염 직접 확인 — 리뷰 결과와 교차확인만.
   편차 ①(d361 rebind)·②(invocation marker)·⑤(phase명)·⑥(소비 열거)·⑦
   (controller 규율)은 리뷰 판정 대기.
2. builder 저작: 스펙 §A + 이 doc §1.6(allowlist 미래 출력 전수 합집합) +
   §1.7(regex `==` 정정) + §1.8(M2는 설계 doc §4에서 전사).
3. runner 저작: 스펙 §B (stage A~M) — 도구 2종은 scratchpad에만, repo 산출은
   attempt dir 4파일 + 이 세션 doc 계열뿐.
4. 순서 (l1, 불변): harness 동결 → EXPECTED_PREREG_SHA256 주입(worker:59 +
   controller:51) → 세션 doc 최종화(**post-injection** controller/worker sha
   각 정확 1회 — controller:449-455가 검사) → prereg → stage K/M-late →
   attestation → 4-sha tuple → **정지**. runtime = tuple sha 인용 새 명시
   승인 (아직 없음).

## 4. 경고 (불변)

- D400~D406 전 attempt + D362 33파일 동결. 물리/q5/contact/cylinder 변경 =
  별도 승인. Isaac/GPU는 호스트 경계 (D402-R1). isaaclab env 불변 (D326).
- allowlist 밖 repo 새 파일 금지 (이 세션 신규 repo 파일 = 이 doc 1개뿐;
  도구/검증 산출물은 전부 scratchpad).
- **runtime 전 commit/push 금지** (HEAD == origin/master == a69a96d 유지).
- HANDOFF.md/half-clone 금지 (HARD RULE #7/#11).
