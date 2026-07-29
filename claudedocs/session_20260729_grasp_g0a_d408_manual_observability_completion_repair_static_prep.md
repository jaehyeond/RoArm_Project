# Session — 2026-07-29 — grasp G0a D408 manual observability completion repair

## 1. 권한·목적·불변 경계

사용자는 D408 `[d407_manual_observability_completion_repair]`의 **설계와
정적 준비**를 승인했다. 이 단계의 목적은 D407의 물리를 다시 계산하는 것이
아니라, 이미 동결된 trace/RRD를 읽기 전용으로 재생해 사람이 가려짐 없이
판독할 수 있는 화면과 실패도 정직하게 전달하는 수동 판정 경로를 준비하는
것이다.

이번 case의 신규 과학 변수: `[]` (0개).

운영·관측성 변수:

1. `d407_clean_view_capture_and_bounded_force_arrow_repair_v1`
2. `prearmed_atomic_manual_writer_pid_phase_handshake_v1`

예정 runtime root:
`claudedocs/runtime_logs/grasp_track/g0a_d408/attempt1_d407_manual_observability_completion_repair/`

이 세션에서 금지한다:

- Isaac/Kit/PhysX import·launch, physics step, q5 target/state write,
  contact query, cylinder spawn, USD/asset mutation
- D407 controller/worker spawn 또는 D407 44개 파일에 대한 쓰기
- D407 manual JSON의 사후 생성, D407 verdict의 소급 PASS
- 실제 D408 replay/capture 산출물 생성
- `isaaclab` package/pin 변경, commit/push, HANDOFF.md/TASKS.md

허용 범위는 D408 production controller/manual-writer 작성, `/tmp`의
software-only Rerun fixture, preregistration, A~M static runner,
reviewed attestation, 승인용 SHA tuple까지다.

## 2. 부팅·직접 검증

- `AGENTS.md` 전체, `START_HERE.md`, DECISIONS D407/D407-R1,
  EXPERIMENT_LEDGER tail, D407 설계·static tuple·actual-runtime 세션을
  로컬에서 다시 읽었다.
- `HEAD == origin/master ==
  a69a96d36219268e4bc5e25065cc234da9d99674`.
- D408 승인 직전 `git status --short --untracked-files=all`은 124개로
  당시 `START_HERE.md` 기대와 일치했고 예상 밖 경로는 0개였다.
- D407 attempt root는 regular file 44개, symlink 0개, manual JSON 0개다.
- D407 final은
  `D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP`,
  `classification=manual_inspection`, `pass=false`, `g0a_pass=false`로
  동결한다.

핵심 불변 입력:

| leg | artifact | SHA-256 |
|---|---|---|
| A | trace JSON | `d879e2a1a3f82e2e83f6c6130f1486b51b71b456db43895a347212adf45d1640` |
| A | RRD | `6bca0b5e065d84ebfc823b3ece9722b4aa17bc49c453447eb183652f8b7f65a5` |
| A | RBL | `85131ed01e576400a78a17368bbe063092dc6d40361b636002f3ead763603033` |
| B | trace JSON | `adecffb7a5699f3b48c2942d1fd507cbe29ad05e05df2d6cdd410ff44cc5d357` |
| B | RRD | `284c3c6d85bfb3f036a0eb4be0fe53d8b411ee464b12801a5ca50d32e35e498b` |
| B | RBL | `c7c9de3b55d4735f59d88a680d2d5e9c2d05bfdb0cc6f831079625562ed22f40` |

양 trace는 각 500 finite row다. D408의 보고 카운터는
`historical_trace_rows_read=1000`과 `new_controlled_physics_steps=0`을
분리한다.

## 3. 3-agent 적대 검토와 직접 반박 검증

세 agent는 파일 수정 권한 없이 worker/화면, controller/writer,
builder/prereg 관점으로 독립 검토했다. 메인 agent가 각 finding을 로컬
source·artifact로 다시 확인했다.

### 3.1 Blocker — 채택

1. D407 정적 writer와 실제 writer가 서로 다른 구현이었다. D408은 static
   fixture와 actual execution이 **동일 production writer SHA**를 사용한다.
2. D407 writer는 모든 항목이 true일 때만 게시할 수 있었다. D408은 각
   boolean을 명시적으로 받고 false도 게시하며, writer가 전체 `pass`를
   계산한다. 따라서 `received=true/pass=false`와 delivery timeout을
   구분한다.
3. PID만으로는 재사용 공격을 막지 못한다. controller/writer PID와
   `/proc/<pid>/stat` start ticks, 양 script SHA, 승인 tuple SHA, nonce
   hash, phase-log dev/inode/sequence/prev-row SHA, D407 manifest SHA,
   screenshot manifest SHA, 공통 monotonic deadline을 한 handshake로 묶는다.
4. manual 파일을 `exists()`와 여러 번의 path-open으로 읽으면 TOCTOU가
   남는다. D408 controller는 dirfd + `O_NOFOLLOW`, regular/nlink/size,
   동일 fd bytes+SHA+JSON, read 전후 stat 불변을 한 번만 검사한다.
5. D407 RRD의 embedded blueprint, 기본 visible plot legend, 즉시 whole-app
   capture, raw force×`0.005m/N`가 화면 가림을 만들었다. D408은 recording
   store만 새 RRD로 투영하고 source blueprint store를 제외하며, 기존
   `force_display_scale` 3개 entity를 presentation RRD에서 제거한다.
6. prereg dirty allowlist는 고정 목록이 아니라
   `live dirty + planned static outputs + consumer-derived future outputs`
   합집합이어야 한다. status는 실제 `==` 비교 source에서 도출한다.

### 3.2 Warning — 설계에 명시

- `rerun.experimental.RrdReader`는 설치 0.34.1의 experimental API다.
  version pin, recording 1/blueprint 0, footer verify, exact removed-set,
  retained inventory를 모두 fail-capable gate로 둔다.
- whole-app screenshot 자체에는 시작 notification이 남을 수 있다. 따라서
  그것은 진단용이며 manual 대상이 아니다. 단일 Spatial3D view의 고정
  viewport를 잘라 만든 clean crop만 manual 대상이고, 원본 D407 PNG를
  negative fixture로 사용해 notification detector가 실제 FAIL하는지
  검증한다.
- 표시 화살표 cap은 과학 수치 변환이 아니다. raw force vector/norm N은
  canonical trace 그대로 숫자로 표시하고, PIL glyph 길이만 viewport
  안에서 제한하며 `display_capped`를 함께 쓴다.
- 600초 수동 검수 deadline도 사람·도구 지연 위험을 완전히 제거하지
  못한다. writer를 replay 전에 pre-arm하고 publication deadline을
  controller보다 5초 이르게 둔다.

### 3.3 False-positive — 기각

- D407 RRD의 `physics_step`, q5, contact 값을 **읽는 것**은 새 physics
  step/query가 아니다.
- lavapipe/llvmpipe의 CPU software Rerun render는 Isaac/PhysX 또는 hardware
  GPU physics runtime이 아니다.
- D407 RRD/RBL footer·구조 PASS는 유효하다. 그것이 시각 판독 PASS를
  뜻하지 않을 뿐이다.
- D408 presentation PASS가 D407 overall verdict, `root_artifact_integrity`
  null, `g0a_pass=false`를 바꾸지 않는다.

## 4. D408 확정 설계 v3

### 4.1 Source admission과 불변성

builder가 D407 root의 44개 파일을 `os.scandir` 기반으로 전수 열거해 다음을
prereg에 고정한다:

- root 상대경로 exact set
- root 아래 directory exact set(빈 directory 추가도 drift)
- SHA-256, byte size, mode
- regular file, non-symlink, `st_nlink==1`

controller는 admission, leg A capture 후, leg B capture 후, manual
publication 후, completion 직전의 5개 checkpoint에서 같은 manifest를
다시 계산한다. 누락·추가·symlink·mode/link/hash drift는 첫 발견 즉시
FAIL-STOP이다. 필요한 trace/RRD/RBL은 D408 leg 폴더에 bit-exact copy를
남기되 D407 source에는 read만 수행한다.

### 4.2 Rerun clean spatial capture

leg별 순서는 다음과 같다.

1. source RRD SHA와 store inventory를 검증한다.
2. 설치 Rerun 0.34.1 `RrdReader`에서 recording store 정확히 1개를 선택한다.
3. source blueprint store는 전달하지 않는다. 동시에 exact entity
   `/contacts/{support_table,link5,gripper_link}/force_display_scale`를
   presentation stream에서 제거하고, contact point 위치·반경은 보존하되
   화면을 가리는 `Points3D:labels` component만 제거한다.
4. 새 application/recording ID로 presentation RRD를 쓰고 footer verify,
   `recordings==1`, `blueprints==0`, removed-set 정확성, retained
   entity/component/timeline을 검사한다. `/__properties` 같은 system
   metadata entity도 제외하지 않고 RowId별 Arrow field descriptor,
   timeline, component value를 canonicalize한 semantic digest로 비교한다.
   source 쪽은 위 3개 force-display entity와 `Points3D:labels`만
   의도적으로 제외하며, 나머지 system/non-system canonical row/value가
   output과 exact-identical해야 한다. source와 output의
   application/recording ID도 leg별 고정 문자열과 각각 exact-match한다.
5. single `Spatial3DView` corrected RBL을 생성한다. 원본 geometry,
   transforms, contact points는 표시하되 force vector entity는 없다.
6. 절대경로 Rerun CLI를 `shell=False`, retry 0으로 한 번 호출한다.
   `/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`을 고정해 hardware GPU가
   아니라 llvmpipe CPU software renderer임을 로그로 확인한다.
7. whole-app 1920×1080 screenshot은 진단용으로 보존한다. manual 대상은
   single-view 영역의 고정 crop `(0,104,1120,1004)`로 만든
   1120×900 clean spatial PNG다. notification detector, jaw/gripper color
   witness, cylinder color witness, nonblank/finite-dimension gate를 통과해야
   한다.

`/tmp` proof에서 recording-only projection은 source
`recordings=1/blueprints=1`을 output `recordings=1/blueprints=0`으로
바꾸었고 `rrd verify`를 통과했다. corrected RBL도 실제 단일 Spatial3D
layout을 활성화했다. persistent ViewerClient는 설치 software renderer가
요구 feature를 지원하지 않아 panic했으므로 production 경로에서 제외한다.
대신 UI가 없는 deterministic spatial crop을 최종 manual 피검체로 삼는다.

### 4.3 Bounded force glyph와 canonical decision sheet

force의 권위는 immutable trace의 Float64 vector/norm이다. leg별
1920×1080 decision sheet에는 다음을 그린다.

- clean Rerun spatial crop
- final row raw `Fx/Fy/Fz/norm N`
- 전체 500-row traversal의 body별 peak raw N과 row
- q5 actual/target, link4/link5/gripper force norm, object XY/tilt/z
  시계열
- final force 방향을 보이는 display-only glyph

glyph는 direction을 보존한 고정 2D oblique projection을 사용한다. 우선
`min(96, 0.40*stored_norm_N)`을 적용하고, 방향상 inset 경계를 넘는 경우
8px margin까지 한 번 더 clamp한다. endpoint, line, 네 text bbox는 inset
안에 있어야 하고 각 label에 immutable trace의 stored raw N,
stored-vs-recomputed norm 오차, `display_capped=true|false`를 명시한다.
cap/clamp된 길이는 과학 비교·hash authority로 사용하지 않는다.

필수 fixture는 B peak 약 464.003N, B final gripper/link5
415.498/165.253N, 양 leg 500행×4 filter=4,000개 vector/norm과 이에
대응하는 16,000개 text bbox, zero, NaN, off-screen, synthetic `1e9N`을
포함한다. 실제 finite 값은 모두 inset 안이어야 하고 NaN은 명시적으로
reject한다. 시계열은 숫자 y축과 row 1/250/500 x축을 가지며, mm인
XY/z와 deg인 tilt를 별도 panel로 분리한다.

A/B comparison은 수정된 두 1920×1080 decision sheet만 소비해
3840×1080으로 만든다.

### 4.4 Pre-armed writer와 manual publication

controller는 capture 전에 production writer를 정확히 1회 spawn한다.
retry는 0이다. 양쪽은 inherited Unix socketpair로 통신하며 raw 256-bit
nonce와 manual 판독은 argv·환경변수·repo 파일에 넣지 않는다.

handshake binding:

- controller/writer PID + proc start ticks
- controller/writer post-injection SHA
- 승인 tuple-file SHA
- nonce SHA-256
- phase log dev/inode, exact sequence, previous/current row SHA
- D407 immutable manifest SHA
- screenshot manifest SHA
- exact manual basename
- D408 root의 inherited dirfd + dev/inode
- replay pre-arm hard deadline
- prompt 시작 monotonic 시각, prompt+600초 manual deadline, 그보다
  5초 이른 writer deadline

writer는 authenticated `READY`를 반환하고 controller는 READY의 initial
phase sequence/SHA를 exact 검증한 뒤 `writer_armed` row를 fsync해야만
replay를 시작한다. leg A 후, leg B 후, prompt 직전에 controller가
`poll()+PID/start-ticks+HMAC ping/pong`을 모두 확인한다. capture가 끝나면
controller가 11개 boolean과 notes를 한 JSON line으로 stdin에서 받아
socket으로 writer에 전달한다. boolean은 true/false 모두 유효하고 전체
`pass`는 writer가 계산한다.

writer publication:

1. inherited·inode-bound D408 dirfd에서 고정
   `.d408_manual_visual_inspection.json.pending`을
   `O_CREAT|O_EXCL|O_NOFOLLOW`로 생성
2. canonical JSON write + file fsync
3. Linux `renameat2(RENAME_NOREPLACE)`; fallback 금지
4. D408 directory fsync
5. `PUBLISHED_FSYNCED` ack

controller publication read:

- publish 명령 전 또는 승인 writer의 rename 전 final/pending이 선존재하면
  FAIL. 정상 rename 뒤 ack 전 final 존재는 정상이다.
- 동일 inherited dirfd 상대 `O_RDONLY|O_NOFOLLOW|O_CLOEXEC`
- regular, `nlink==1`, size ≤64KiB
- 같은 fd의 bytes/SHA/strict JSON
- duplicate key, NaN/Infinity, extra/missing key, non-bool reject
- read 전후 dev/ino/size/mtime 불변
- 첫 candidate/ack가 malformed면 수정본을 기다리지 않고 terminal FAIL

manual deadline은 prompt 시작시각부터 정확히 600초, writer publication
deadline은 그보다 5초 이르다. `published_before_writer_deadline`은 manual
문서에 미리 쓰지 않고, directory fsync 완료시각을 담은 authenticated ack와
controller receipt에서 사후 계산한다. rename 뒤 ack 전 crash도 attempt
FAIL로 동결하며 resume/retry하지 않는다.

screenshot manifest SHA만으로는 부족하므로 clean A/B, decision A/B,
A/B comparison의 5개 PNG를 nofollow/nlink/size/dimensions/SHA로
`pre_prompt`, `before_writer_send`, writer publish 직전,
`post_publication`, `pre_completion`에 재검증한다.

### 4.5 Phase·completion·판정

phase log는 bound D408 dirfd에 exclusive-create하고 directory fsync 후
열린 fd를 유지한다. 각 행은 canonical JSON 한 번의 완전한 `os.write`,
`fsync`, 증가 sequence,
`prev_row_sha256`, current row SHA chain을 가진다.

D408에서 PASS 가능한 것은 다음뿐이다:

- source immutability
- clean spatial visibility
- raw force 수치 보존과 bounded display
- timeseries legibility
- manual result의 authenticated atomic publication

completion에는 다음을 항상 고정한다:

- `d407_final_verdict =
  D407_SDF_PHYSICS_AB_TIPPING_REMEASURE_FAIL_STOP`
- `d407_retroactive_pass = false`
- `scientific_verdict = null`
- `g0a_pass = false`
- `new_controlled_physics_steps = 0`
- 아래 9개 `scientific_null_claims` exact matrix

force closure, stable grasp, cap/rim/barrel order, exact face/manifold,
grasp feasibility, 29×50 cylinder 전이, SDF 일반 우월성, per-prim cooked SDF
identity, collider-count tipping causality는 모두 null을 유지한다.

정상 completion은 `completion_ready` phase를 마지막으로 fsync·close한
뒤 공용 고정 pending `.d408_terminal_summary.json.pending`에서 공용
`d408_terminal_summary.json`로 `renameat2(RENAME_NOREPLACE)`하여
**마지막 repo write**로 게시한다. 그 뒤 phase row나 별도 failure 파일을
쓰지 않는다. 예외 경로도 `fail_stop` phase를 seal한 뒤 같은 공용
pending/final basename에 failure schema를 게시한다. 따라서 완료와 실패는
파일시스템상 하나의 원자적 terminal endpoint를 공유하며 공존할 수 없다.

## 5. Production 파일·tuple 계약

두 production 파일만 둔다:

- `sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_controller.py`
- `sim_scripts/cyl34_top_view_d408_d407_manual_observability_completion_repair_manual_writer.py`

4-SHA tuple은 `{preregistration, reviewed static attestation, controller,
manual writer}`다. renderer는 controller 내부 기능이므로 별도 executable
SHA가 없다.

두 script의 `--print-contract-json`은 import/write side effect 없이
runtime leaf, temp grammar, subprocess argv, status literal, manual schema를
출력한다. builder는 여기서 미래 runtime output을 프로그램적으로 도출한다.

순서:

1. 설계·production script 동결
2. prereg bytes를 scratch에서 확정하고 그 SHA를 양 script에 주입
3. post-injection script SHA 계산
4. 이 session doc에 양 script SHA를 각각 정확히 1회 기록
5. prereg exclusive-create
6. static A~M과 M-late
7. reviewed attestation
8. 4-SHA tuple atomic no-clobber
9. tuple-file SHA 계산 후 실제 실행 없이 정지

## 6. Static runner A~M

- A: `-B`, HEAD/upstream, pycache delta 0, production 파일 AST
- B: prereg schema, 변수 0+2, D408 root/status literal
- C: D407 44-file exact manifest, manual absent, frozen FAIL 불변
- D: A/B trace 각 500 finite row, 6개 core SHA, RRD/RBL footer
- E: forbidden import/dynamic import/eval/exec/runpy와 D407 harness spawn 0
- F: `--print-contract-json`에서 future outputs/subprocess/manual schema 도출
- G: live dirty + planned static + future runtime 합집합과 mutation reject
- G는 Git-visible dirty exact admission/allowlist와, ignored PNG/RRD까지
  포함하는 D408 physical file/directory tree gate를 분리해 검증
- H: `/tmp` recording-only projection, blueprint 0, corrected RBL,
  llvmpipe screenshot/crop proof
- I: actual 1000 rows + B 464N/final + synthetic force cap/clip/raw-N proof
- J: 동일 production writer의 READY, true publication, false publication,
  strict single-read 및 filesystem/handshake negative fixtures
- K: Isaac/Kit/PhysX/GPU physics/q5/contact/cylinder counter exact 0
- L: exact phase chain, prearm/prompt deadline, retry 0, authenticated
  leg별 liveness, 5회 source rehash, 4회 screenshot-byte checkpoint,
  first-failure/retro-pass reject
- M: tuple validator accept/reject, static-results overall PASS를 묶는
  attestation schema, normalized pre/post-injection contract, 4-SHA binding
- M-late: 최종 실물 파일 SHA, session-doc occurrence, dirty exact set,
  tuple no-clobber 재검증

중간 projection/RRD/RBL/PNG와 fixture 결과는 `/tmp` scratch에만 만든다.
repo에는 prereg, static result, attestation, tuple 및 state/session 문서 외
정적 중간물을 만들지 않는다.

## 7. 설치 source 근거와 권위 구분

Rerun 0.34.1 설치 source:

- `rerun/experimental/_rrd_reader.py`: recording/blueprint store 분리와
  지정 store stream
- `rerun/experimental/_lazy_chunk_stream.py`: entity-path drop과 명시
  app/recording ID `write_rrd`
- `rerun/blueprint/api.py`: `.rbl` 저장과 active/default semantics
- `rerun/experimental/_viewer_client.py`: view-specific capture는 experimental

CLI 직접 proof:

- `rerun-cli 0.34.1`
- `rrd filter`, `rrd verify`, `rrd stats`
- recording-only output footer PASS, blueprint store 0
- llvmpipe `device_type: Cpu`

NVIDIA/PhysX의 새 동작·한계 주장은 이 case에서 만들지 않는다. D407의
물리 의미는 동결된 D407 session §4와 원본 JSON의 권위 그대로이며,
D408은 해당 값의 read-only presentation만 수행한다. 따라서
`physics_step` timeline을 읽는 것과 새 engine step을 실행하는 것을
명시적으로 분리한다.

## 8. 정적 구현 결과

3-agent 재검토와 메인 agent의 직접 source·fixture 반박 검증 뒤 첫
production 후보는 blocker 0으로 수렴했다. 그러나 첫 A~M 시도에서 호스트
경계 fixture가 새 blocker를 실제로 드러냈다. 기본 sandbox는 Unix
socketpair 송신을 `EPERM`으로 막고, 승격 실행은 Rerun backend 자동 선택이
필요 compute feature가 없는 장치를 골라 screenshot 전에 중단됐다.
동일 RRD/RBL에 `WGPU_BACKEND=vulkan`을 명시한 승격 probe는 CPU
llvmpipe Vulkan으로 정상 종료했다.

따라서 controller의 screenshot subprocess 환경과 contract에
`WGPU_BACKEND=vulkan` exact override를 추가한다. 이 변경은 화면 backend
선택만 고정하며 D407 trace, 물리, scientific verdict는 바꾸지 않는다.
첫 prereg와 post-injection SHA는 최종 정적 PASS 전에 생성된 폐기 후보로
취급하고, builder부터 다시 실행해 최종 SHA들을 아래에 한 번만 기록한다.

host blocker 최소 수리 후 builder를 다시 실행한 최종 prereg SHA는
`0c0f1c03d10210e205d5be0b25fd84c7d94c109fb26387f77fa22f6b984c8d0d`다.

| post-injection production file | SHA-256 |
|---|---|
| controller | `00f4317cd12fedec16e23599080e97ec6462c0e54abec872e3444ee6b8603fce` |
| manual writer | `f69d4221f79f0a6cd96a20f81714e4970b0ecc0bd99567037664dbf8a468edf7` |

두 production SHA는 이 문서에 각각 정확히 한 번만 기록한다. 이후 결과
문단은 이 표를 반복하지 않고 “위 post-injection 두 파일”로 참조한다.

### 8.1 최종 builder·static runner

scratch 전용 도구는 repo 밖
`/tmp/d408_static_prep_20260729.jwYX3B/`에서만 실행했다.

| 도구 | 행 수 | SHA-256 |
|---|---:|---|
| `d408_prereg_builder.py` | 639 | `607acd3d9c0a21a10451bc544b1c209bdc8234fdb10d6e027b6805b2c9b4f023` |
| `d408_static_runner.py` | 3,224 | `a54780319f70fc46538d4db6f144ec90d59d02f1161576427ab48c8f008ee27e` |

builder가 실제 소비자 source에서 도출한 최종 수치는 다음과 같다.

- builder 시작 live dirty 127개
- runtime 직전 기대 dirty 131개
- 허용 dirty 합집합 161개
- 미래 runtime regular file 30개, directory 2개
- D407 동결 입력 regular file 44개, directory 2개
- planned static path 11개
- D408 runtime 직전 physical static file 4개, directory 0개
- prereg 소비 field 15개와 `ast.Eq` status literal 4개 exact

첫 full A~M은 기본 sandbox가 Unix socketpair 송신을 `EPERM`으로 거부해
J에 도달하지 못했고, 두 번째 승격 실행은 자동 renderer 선택이 부적합해
H에서 멈췄다. 세 번째 실행은 화면·writer fixture를 통과했지만 K가
고정 Rerun 실행파일 경로의 conda 환경명 `isaaclab`을 실행 인자상의
Isaac launch로 오인했다. K는 executable `argv[0]`의 exact allowlist와
실제 명령 인자 `argv[1:4]`의 금지 token 검사를 분리하도록 최소 수리했다.
이 세 시도는 최종 PASS 증거로 사용하지 않는다.

최종 권위 실행은
`/tmp/d408_static_prep_20260729.jwYX3B/static_run_final_v4/`이며,
repo에 바이트 동일하게 게시한
`d408_static_fixture_results.json` SHA는
`bfb0f05784f9c01a9a8dccc5126fb8ed12d3b0748eed51e882e89ce5c9962dab`다.

- stage A~M: 13/13 PASS
- checks: 73/73 PASS
- 고의 변조 negative fixture: 26/26 reject
- production AST: 2/2 PASS
- D407 trace: A/B 각 500 finite row, 합계 1,000행
- D407 RRD/RBL footer: 4/4 PASS
- recording-only semantic canonical:
  A 12,995 rows/57,929 value cells,
  B 13,118 rows/58,783 value cells
- raw force: leg별 2,000개, 합계 4,000개
- 검증 text bbox: leg별 8,000개, 합계 16,000개
- A/B comparison: 3,840×1,080
- software Rerun viewer 2회, historical RRD read 2개
- Isaac/Kit/PhysX launch, hardware GPU job, D407 harness spawn,
  contact query, cylinder spawn, q5 read/write, USD/asset write,
  새 controlled physics step: 모두 0
- repo `__pycache__` delta: 0

화면은 양 leg 모두 Rerun 0.34.1, Vulkan CPU
`llvmpipe (LLVM 15.0.7, 256 bits)`로 생성됐다. static fixture의 다섯
manual 대상 PNG를 사람이 직접 열어 다음을 확인했다. 이는 미래 actual
D408 manual JSON을 대신하지 않는다.

1. A/B clean crop 모두 jaw·gripper와 노란 cylinder가 분명하며
   notification/text overlay가 없다.
2. A decision sheet의 final link4/link5/gripper force는 모두 0N이고,
   peak link5는 23.227865N(row 256), peak gripper는
   43.858340N(row 255)로 읽힌다.
3. B final link5는 `[76.437,-63.770,-131.906]N`,
   norm 165.253139N이고 final gripper는
   `[-151.901,190.140,336.766]N`, norm 415.498019N이다.
   gripper glyph는 96px cap이 명시돼 있다.
4. B peak link5는 357.175438N(row 358), peak gripper는
   464.002511N(row 239)로 읽힌다.
5. q5·robot force·displacement·tilt panel과 숫자 축이 판독 가능하고,
   A/B 비교판은 leg별 y축이 독립이므로 높이가 아니라 눈금을 비교하라는
   경고를 명시한다.

### 8.2 Attestation·4-SHA tuple과 정지선

최종 static authority는 다음과 같다.

| 파일 | SHA-256 |
|---|---|
| preregistration | `0c0f1c03d10210e205d5be0b25fd84c7d94c109fb26387f77fa22f6b984c8d0d` |
| static fixture results | `bfb0f05784f9c01a9a8dccc5126fb8ed12d3b0748eed51e882e89ce5c9962dab` |
| reviewed script attestation | `fa5a3cf2f1a2bb0a4899e89d26eeb41d7d83b34e2be799056498e7d7fd9d50dd` |
| proposed runtime tuple file | `97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff` |

4-SHA tuple 내부는 attestation, 위 post-injection 두 production 파일,
preregistration을 exact binding하며
`execution_status=PROPOSED_NOT_EXECUTED`다. runtime 직전 repo dirty는
정확히 131개이고 prereg의 기대 집합과 일치한다. D408 root의 physical
tree는 위 static file 4개, directory/symlink/special file 0개다.

이 정적 세션은 별도 승인 없이는 실제 replay를 실행할 수 없으므로
과학 perturbation이나 새 physics experiment를 실행하지 않았다. 대신
결정을 바꿀 수 있는 26개 변조 fixture와 production writer true/false
publication fixture를 실행했고 실제 실패 세 건을 관측·수리했다.
`runtime_executed=false`, `new_controlled_physics_steps=0`,
`scientific_verdict=null`, `g0a_pass=false`를 유지한다.

남은 위험은 다음과 같다.

- 실제 runtime의 socket inheritance, 사람의 600초 내 응답, atomic
  publication 전체 경계는 아직 한 번도 end-to-end 실행되지 않았다.
- static J는 production writer 함수와 실제 schema/HMAC를 검증하지만,
  실제 `Popen(pass_fds=...)` 프로세스 경계 자체는 미래 runtime에서만
  검증된다.
- `RrdReader`는 experimental API이고 detector는 휴리스틱이므로,
  실제 11개 boolean의 사람 판정은 여전히 필수다.
- B는 step 500에서 비정착 상태다. D408이 성공해도 D407 FAIL을 소급
  PASS하거나 SDF 안정성·force closure·grasp feasibility를 주장할 수 없다.

실제 실행은 다음 문장을 새 사용자 메시지로 명시 승인받은 뒤에만 가능하다:

`D408 tuple-file SHA-256 97c7ca51f8116053fcdc59aa9572669231d4abeb66022ed4e59c9e61af28e1ff를 사용한 read-only manual observability replay 1회(controller 1회, software Rerun viewer leg A/B 각 1회, manual writer 1회, retry 0)를 승인합니다.`
