# Session 2026-08-04 — D409 적대 리뷰 처분 + harness 수리 + 정적 fixture

이번 case의 신규 변수: [없음 — D409 정적 준비 단계 지속. 신규 변수는 3rd
세션 선언분 1개(실물 원통 기하) 불변]

과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`, 과학 verdict 없음.
attempt1 산출물 생성 0 (본 문서 작성 시점). 로봇 HW·Isaac runtime·
lerobot-train 실행 0.

## 0. 승인 범위 / 규칙 준수

- 사용자 "설계 착수" 승인 지속 (설계+정적 준비+tuple까지; attempt1은
  tuple SHA 인용 별도 승인 — retry 0). D399 사용 금지 (D398-F1 예약).
- 처분 원칙: 리뷰 무비판 수용 금지 — §1의 모든 처분은 본 세션 독립
  재검증(§1.0) 후 확정. 수리는 처분 확정분만 (§2).

## 1. 적대 리뷰 `wf_f6ae07c8-819` 처분 (전문 판독 + 독립 재검증 후)

판독 대상 = `g0a_d409/design_inputs/adversarial_review_wf_f6ae07c8-819/
adversarial_review_result_verbatim.json` **전문** (sha256 `d6077799…a13832`
본 세션 재계산 bit-일치; journal `a310b825…` 동일). manifest headline이
아니라 verbatim 파일 전문(4 lens × blockers/warnings/adjudications/notes)
을 근거로 처분했다.

### 1.0 독립 재검증 결과 (처분 전 수행 — 전부 리뷰 주장 재현, 반증 0)

| # | 재검증 항목 | 방법 | 결과 |
|---|---|---|---|
| 1 | 3파일 as-authored sha | sha256sum | worker `bd835c6f…`/controller `fb34238d…`/writer `5675341a…` bit-일치 (무변조) |
| 2 | OPS-B2/WOBS-B1 flush API | isaaclab env 라이브 probe | `RecordingStream.flush` 시그니처 `(self, *, timeout_sec: 'float' = 1e+38)`; `flush(blocking=True)` → TypeError 재현; `flush(timeout_sec=30.0)` OK. 호출부 controller:2824 + docstring:138 실재, `flush(blocking` grep은 이 파일 2곳뿐, D407 전례 worker:2655 = `flush(timeout_sec=30.0)` |
| 3 | OPS-B1 registered_metrics | grep + D371 prereg 파싱 | 신규 3파일 grep 0건; D371 prereg top-level에 `registered_metrics` 7항 실재; controller `_build_prereg`(:1017~) top-level에 부재 |
| 4 | SCI-B1 수치 | worker 함수 직접 호출 probe (읽기전용) | pose (7000,11000): overlap idx **19**, 쌍(17,18) q5=(0.7225,0.6743) d=(**3.031**,**0.707**)mm ≤ bound **3.244**mm 미배제, transient=1, q5*=**0.6596315690985648**/part_042 비인증 게시, b_checks 3항 True 소비 확인. pose (0,6500): 쌍(18,19) d=(2.833,0.724), overlap idx 20 — 리뷰 수치와 bit 단위 일치. (7000,11000) admission PASS·d_fix 6.77mm A-band FAIL·barrel_strict FAIL도 리뷰 스모크와 일치 |
| 5 | SCI-B1 코드 구조 | worker 1206-1336 판독 | 미배제 쌍 카운트만(1206-1215), anchor 스캔(1218-1232), crossing dict 인증 필드 전무(1265-1293), b_checks 소비(1318-1336) 확인 |
| 6 | D351 전례 | cyl34_..._d351_...py 판독 | `_certify_first_contact`(:1152-1273) 재귀 이분 + `certified_clear_intervals`/`unresolved_intervals`/`contact_order_certified`(:1272) + verdict 게이트(:3267-3268, :3316-3322) 전부 실재 |
| 7 | G3 root 집합 | AST 파싱 | worker 20(isaacgym 누락)/controller 21(완전)/writer 16(cupy·isaac·kit·physxcooking·torch 누락); 매칭 정규화 불일치(worker만 대소문자 구분, controller/writer는 lower()) 확인 |
| 8 | G1 fixture 부재 | controller fixture 목록 grep | 정적 fixture = N1~N5 + A1~A4 + equivalence_d335 + positive만; w_ops3 9항 표면별 변조 fixture 부재. writer:1119-1157에 `w_ops3_reject_surfaces` 9항 열거 실재 |
| 9 | G2 편차 실체 | prereg/docstring 판독 | controller C3 docstring + prereg `a64_authority.per_part_integrity` + N2 텍스트에 대체 기준(D409-canonical hash + 파일 sha) 공개 실재; 4th 세션 S1이 이미 "D409-canonical hash = runtime 재검증 기준"으로 pin |
| 10 | warning 12건 코드 주장 | 각 인용 행 판독 | 전부 실재 확인: 16,500s 오기(:136 vs 코드 17,100s), 하드코드 True 2건(:3392·:3395), static results sha 재결박 부재(:3057-3086), N1 지역함수(:1386-1391)+worker 정적 pin에 radius 부재(:1708-1726), manual_contract 6키(:1264-1271), tuple gate 후순위(:3171-3173), writer rrd_report 형식검사만(:722-723 vs 이미지 재해시 :755-768), tick label scale 1(:2460·2463), A11 docstring 과잉약속(직렬화 dict :1397-1460에 per-part 배열 전무), d371 crosscheck 무시(worker :1690-1694 태그 4종만), env 검증 numpy/hppfcl만(:1745-1751), 대소문자(worker :178) |
| 11 | C2 문서화 주장 | grep | `PRECLOSE_PASS`는 worker:2054 dict-리터럴 유일 — module 상수 부재, docstring "cross-checked against the imported worker module constants" 과잉 서술 확인 |

### 1.1 확정 blocker 3건 — 전건 수용 (반박 성공 0)

- **SCI-B1 수용 → 수리 R1**: 첫 교차 직전 chord-bound 미배제 clear-clear
  쌍을 진단 카운트만 하고 q5*/first_contact를 비인증 게시. 수리 =
  **D351식 ordered certification traversal(옵션 a) + 결정적 평가 cap +
  cap 소진/미해결 시 fail-closed `order_certified=false`(옵션 b 결합)**:
  - OPEN→CLOSED 순서로 인접 anchor 구간을 순회: 양끝 clear ∧
    min(d_i,d_j) > chord bound → 구간 인증; 폭 ≤ 1e-6 rad → bracket
    (clear/overlap endpoint 계약 유지) 또는 unresolved; 그 외 중점 분할
    (평가 memo화 — anchor 33점 재사용, 신규 q만 64-part 평가).
  - **예산 불변** (W-OPS4 등록치 3,600/pose·4.5M/run 유지): base 64+33×64
    = 2,176 + traversal 신규 평가 cap **22회**×64 = 1,408 → 최악 3,584 ≤
    3,600; run 최악 128+1,239×3,584 = 4,440,704 ≤ 4.5M. cap 소진 →
    unresolved(사유 기록) → order_certified=false fail-closed.
  - b_checks에 게이트 `first_crossing_order_certified` 추가; evidence
    first_crossing에 인증 필드(certified/unresolved/평가 사용량), CSV에
    컬럼 3개 추가. prereg에 인증 규칙 등재 (리뷰 지시 "Record the chosen
    rule in prereg").
  - 옵션 (b) 단독 기각 사유: probe 2/4 pose에서 미배제 쌍이 crossing 인접
    쌍에 발생 — (b) 단독이면 인증 가능한 crossing까지 광범위 spurious
    (B) FAIL. (a)+cap+(b)는 두 제안의 합집합 내.
- **OPS-B1 수용 → 수리 R2**: prereg에 D371 계보 필수 필드
  `registered_metrics` 추가 — D405 소비자-도출 원칙 유지 (worker
  CSV_COLUMNS verbatim + counts 키 + region entry 필드). `_audit_registered`
  에 존재+도출 일치 검사 추가.
- **OPS-B2≡WOBS-B1 수용 → 수리 R3**: controller:2824
  `flush(blocking=True)` → `flush(timeout_sec=30.0)` (D407 전례
  worker:2655 형태) + docstring :138/C12 문구 동기 정정. 수리 후
  scratchpad RRD로 관측성 호출 시퀀스 1회 스모크 (리뷰 요구).

### 1.2 내장 검토 gap G1~G3 처분

- **G1 수용 → 수리 R4**: w_ops3_reject_surfaces **9항(writer 계약 열거 =
  권위 목록, OPS lens 지시)** 별 정적 fixture 추가. 행동 실증 가능 표면
  (identity/nonce-HMAC/tuple sha binding/11-field 스키마/deadline 산술/
  manifest 정합)은 in-process 변조 실증; 파일시스템 의미론 표면
  (exclusive-create+no-replace+fsync, traversal 예산, 런 중 수리 금지)은
  AST-presence 검증 — fixture 결과에 실증 방법(behavioral/ast) 구분 공개.
  OPS-W5(R13)와 연동.
- **G2 수용 (편차 승인, 코드 무변경)**: §4.1 P1(i) 문언 "payload_sha256
  재계산"은 preimage opaque로 문자 구현 불능 (저작자 probe + FROZEN
  CHECK5 + WOBS 교차 실증). 대체 기준(D409-canonical per-part hash S1 pin
  + d348 파일 sha)은 **4th 세션 S1이 이미 "runtime 재검증 기준"으로 pin**
  했고, 판별력 실증(S1 128/128 재현 + 1-bit/1-byte 변조 FAIL 발동 —
  리뷰 2 lens 교차) + prereg/docstring 공개 완비. 편차를 설계 확정본
  δ-보완으로 승인 기록.
- **G3 수용 → 수리 R5**: 3파일 FORBIDDEN_IMPORT_ROOTS를 21-root 합집합
  (worker에 isaacgym, writer에 cupy/isaac/kit/physxcooking/torch 추가)
  으로 통일 + worker 매칭 `.lower()` 정규화 통일 (OPS lens 노트) +
  `_audit_registered` 필수 root 검사 목록 확장.

### 1.3 warning 12건 처분

| ID | 처분 | 수리번호 |
|---|---|---|
| SCI-W1 (=A11 repair) | 수용 → A11 docstring을 구현 사실(min/argmin 요약 직렬화)로 정정. 직렬화 확장 기각: §2.13 evidence 계약은 요약으로 충족, per-part 배열은 docstring 과잉 약속이었음 | R6 |
| SCI-W2 (=A6 document) | 수용(문서화) → worker A6 docstring에 권위 분담 명기 (d371 sha 강제 = controller 측 3중 교차; worker 기준값 권위 = d349 repr-equality fail-stop). d371 상수 추가는 선택사항이라 미채택(최소 변경) | R7 |
| FRZ-W1 | 수용 → out-dir 검사를 repo-anchored `Path.is_relative_to`로 교체 (substring 우회 봉쇄) | R8 |
| FRZ-W2 | 수용 → worker `_verify_environment`에 interpreter realpath + python_version 강제 추가 | R9 |
| OPS-W1 (=C11 repair) | 수용 → docstring 합계 16,500s → **17,100s** 정정 (공식이 권위) | R10 |
| OPS-W2 | 수용 → 하드코드 True 2건을 실검증으로 교체: prerun inventory 재독해(run dir 부재 + 4 static artifact 구조) + phase 계수(run1/run2_started 각 1회) + manual 게시 1회 계수 | R11 |
| OPS-W3 | 수용 → `_validate_approval_tuple`에 attestation `static_fixture_results_sha256` == 현재 파일 sha 재결박 추가 | R12 |
| OPS-W4 | 수용(강) → runtime admission에서 임포트 worker 모듈 CYL_RADIUS_M/CYL_HEIGHT_M repr vs prereg.geometry 재대조 + N1 결과 JSON에 runtime 강제 경로 명시 | R12 |
| OPS-W5 | 수용 → prereg manual_contract에 `w_ops3_reject_surfaces` verbatim 등재 (소비자-도출) | R13 |
| WOBS-W1 | 수용 → run_runtime에서 `_validate_approval_tuple`을 interface 도출(worker import/writer subprocess) **앞**으로 재배열 (미검증 바이트 실행 전 승인 게이트) | R14 |
| WOBS-W2 | 수용 → writer가 rrd/rbl/validation 3파일을 `_secure_read_relative`+sha256 재해시해 manifest 값과 대조 후 게시 | R15 |
| WOBS-W3 | 수용 → decision sheet 축 tick label + legend를 scale 2로 상향 (여백 검산 리뷰 완료) | R16 |

### 1.4 모호성 adjudication + notes 처분

- repair_required 4건: A11→R6, A12→R1, C11→R10, C12→R3. 전부 위 표에
  포섭.
- document_only 4건: A6→R7, C3→G2 승인 기록(무변경), controller-C2 →
  **R17** (docstring "imported worker module constants 교차" 과잉 서술을
  실제 구현(유일 dict-리터럴 강제 + 존재 확인)으로 정정), writer-6 →
  **R18** (writer docstring에 PNG 검증 깊이 축소 공개: signature/IHDR/
  치수/IEND+byte-sha, IDAT 디코드/CRC 없음; 보완 통제 = controller 전체
  디코드 non-blank + 사람 육안).
- 나머지 adjudication 전건 accept — 무변경.
- FROZEN 경미 노트 중 region 탐색 주석(BFS 표기, 실제 stack DFS) →
  **R19** 주석 1행 정정 (처분 명기로 근거 확보). 기타 경미 노트
  (_write_json_x 디렉토리 fsync 부재, env gate python_version 간접 보장,
  screenshot CLI stderr 병합 의도성)는 무변경 — 본 문서 기록으로 갈음.
- SCIENCE 노트 "top-rim 전 격자 예상": **수리 대상 아님 (기하 진실)** —
  4 probe pose 전부 첫 교차 witness가 top rim (독립 3중 검증: hppfcl
  Cylinder full-height 규약 + part_042 삼각형 조밀 샘플링 min 0.0156mm
  @rim + part_042 world z-extent [0.0359,0.0523] vs 원통 top 0.0379).
  z=z_center 확정 설계의 정당한 과학 결과 후보이며 tolerance 수리 금지
  (D354). attempt1 승인 전 사용자 브리핑 의무 → 본 세션 브리핑 + §5.
- OPS 노트 NOTE4: 정적 fixture 실패 = attempt1 네임스페이스 영구 소모
  (prereg 선기록 + 'already exists' fail-closed). 재시도는 명시적 사용자
  결정 필요 — 문서화 의무 이행 (본 문서 §3).
- SCIENCE 노트 "수리 시 재리뷰": 수리본 변경 영역 재검증 수행 (§2.2).

## 2. 수리 (R1~R19) 실행 기록

### 2.1 수리 diff 요약 (파일별)

**worker** (`sim_scripts/cyld29h50_..._worker.py`):
- R1: "First crossing + bisection" 블록 전면 교체 → ordered chord-bound
  certification traversal (D351 `_certify_first_contact` traverse 의미론,
  평가 memo화, 신규 평가 cap 48회). b_checks에
  `first_crossing_order_certified` 게이트 신설, evidence에
  `order_certification` 섹션(rule/certified/intervals/sub_resolution
  accepted·rejected/neighborhood/resolution/new_evals), CSV +3 컬럼
  (`first_crossing_order_certified`/`cert_unresolved_intervals`/
  `cert_new_evals`), counts에 `order_certified` 추가. 예산
  3,600/4.5M → **5,400/7.0M** (처분-기록 수정; S3 실측 기반 7.0M ≈
  53~61s ≪ 7,200s).
- R5: FORBIDDEN_IMPORT_ROOTS에 isaacgym 추가(21-root 합집합) +
  find_spec/preloaded 검사 `.lower()` 정규화.
- R6/R7: A11(요약 직렬화 사실 정정)/A6(d371 권위 분담) docstring 정정.
- R8: out-dir 검사를 repo-anchored `is_relative_to`로 교체.
- R9: `_verify_environment`에 interpreter realpath + python 3.11.14 강제.
- R19: region 탐색 주석 BFS → stack DFS 사실 정정.

**R1 수리 중 발견·해결한 설계 결함 2건 (중대 — 브리핑 §5 필수)**:
1. **D351의 min() 인증 기준은 이 기하에서 종결 불능**: crossing 접근
   기울기(실측 15~48mm/rad)가 bound 계수 2·Rmax(=67.35mm/rad)보다 작으면
   min(d_hi,d_lo)>bound는 어떤 유한 분할로도 성립 불가(1차 스모크에서
   cap 소진 실증). **선언된 A12 배제 기준의 sharp 형태인
   max(d_hi,d_lo)>bound로 정정** — soundness 논증: d(q) ≥ d_endpoint −
   bound(width) (1-Lipschitz), 접촉은 양끝 모두 bound 이하일 때만 가능.
   D351 문언과의 의도적 편차로 기록.
2. **터미널 해상도(1e-6 rad)의 clear-clear 미인증 구간**: sub-bound
   기울기에서는 crossing 직상부 구간이 원리적으로 인증 불능(2차
   스모크에서 3/6 pose 실증). **해상도-이웃 규칙 신설**: 폭 ≤1e-6 rad ∧
   유효 clear-clear ∧ bracket 상부 6.4e-5 rad 이내 ∧ per-part 배제로
   bracket part 외 접촉 불능 → 수용(q5* 해상도 공개 기록); 그 외
   fail-closed. 수용 시 chord 변위 상한 4.3µm — 게이트 최소 단위
   (CLEAR_GATE 0.1mm)의 1/23 이하.

**controller** (`..._controller.py`):
- R2: `_worker_registered_metric_names()`(AST 소비자-도출) 신설 + prereg
  `registered_metrics` (csv 43/counts 12/region 12) + `_audit_registered`
  존재·형태 검사.
- R3: `flush(blocking=True)` → `flush(timeout_sec=30.0)` (:2824) +
  docstring C11/C12 정정 (R10 포함: 16,500→17,100s).
- R4: `_fixture_w_ops3_reject_surfaces` 신설 — 9표면 전수
  (behavioral 4 + ast-presence 5, 방법 공개) + coverage 검사 +
  **fixture 실패 시 STATIC_FAIL_STATUS 기록 후 attestation/tuple 저작 전
  raise** (NOTE4 fail-closed 정합 — 기존 코드는 실패에도 PASS status
  무조건 기록이었음, 처분-기록 보강).
- R11: 완료 감사 하드코드 True 2건 → prerun inventory 재독해 +
  phase-row 계수 + manual 파일 실사.
- R12: `_validate_approval_tuple`에 attestation
  `static_fixture_results_sha256` 재결박 + run_runtime에서 worker 모듈
  geometry repr vs prereg 재대조 + N1 결과에 runtime 강제 경로 명시.
- R13: prereg manual_contract에 `w_ops3_reject_surfaces` verbatim 등재.
- R14: run_runtime에서 tuple gate를 interface 도출 앞으로 재배열.
- R16: decision sheet tick label/legend scale 1→2 (여백 검산 완료).
- R17: C2 docstring dict-literal 경로 서술 정정.
- R1 연동: WORKER_INTERFACE_CONSTANT_NAMES에 CERT 상수 2종+CSV_COLUMNS
  추가; prereg gates.b_checks_core에 order 게이트, gates.order_certification
  규칙 등재, registered_budget.r1_certification_amendment(7.0M 파생 산술).
- R5 연동: `_audit_registered` 필수 root 목록 14종으로 확장.

**manual_writer** (`..._manual_writer.py`):
- R5: FORBIDDEN_IMPORT_ROOTS 16→21 (cupy/isaac/kit/physxcooking/torch).
- R15: `_verify_screenshot_manifest`에서 rrd/rbl/validation 3파일
  `_secure_read_relative`+sha256 재해시 → manifest 값 대조 후 게시.
- R18: PNG 검증 깊이 축소(IDAT/CRC 없음) + 보완 통제 docstring 공개.

### 2.2 수리 검증 (본 세션 실측)

| 검증 | 결과 |
|---|---|
| py_compile 3/3 | PASS |
| SCI-B1 재현 probe (수리 전, as-authored) | (7000,11000)/(0,6500) 리뷰 수치 bit-일치 재현 |
| 수리 후 8-pose 스모크 | 8/8 crossing 인증 성공(`certified_traversal_bracket`), hard unresolved 0, 최대 3,840 질의 ≤ 5,400 |
| **회귀 불변식**: 인증 성공 pose의 q5*/part = as-authored 이분법 값 | (7000,11000) q5*=0.6596315690985648/part_042, (0,6500) q5*=0.6097505786983675 — **bit-일치** |
| 프로세스-간 결정성 (3 pose row canonical bytes) | 2 독립 프로세스 sha 3/3 bit-일치 |
| 정적 파이프라인 전체 in-memory dry-run | env gate/interface(csv 43·cap 48)/writer contract(9표면)/prereg(R1·R2·R13 필드)/audit(+변조 검출)/N1~N5/A1~A4/P1~P2/equivalence/w_ops3 10항 — **전항 PASS** |
| R3 관측성 시퀀스 스모크 (scratch RRD) | save(default_blueprint)→log→`flush(timeout_sec=30.0)`→disconnect→RrdReader(1+1)→`rrd verify --check-footers true` rc 0 "2 files verified without error." |
| 변경영역 3-lens 적대 재검증 | `wf_bc577f9b-dfa` — 결과는 §2.3 |

### 2.3 변경영역 적대 재검증 결과 (`wf_bc577f9b-dfa`, 3/3 완주·오류 0)

verbatim 보존: `g0a_d409/design_inputs/repair_reverify_wf_bc577f9b-dfa_{result_verbatim.json,journal.jsonl,manifest.json}`
(result sha `6d458f76…dbce5c` / journal `3a9a3f09…aebf569` / manifest
`9c76aa36…ace1813`).

- **blocker 0 / 3 lens** — 수리 성립. 핵심 확인(전문은 verbatim):
  - max() 인증 기준 soundness 확인 (조인트 축 (0,0,1) child-frame 검증 +
    정점 최대 변위가 bound에 정확 도달·초과 0 실측 + 39-sample 조밀 스캔
    위반 0).
  - 합성 transient 시나리오: **구 이분법이라면 놓쳤을 anchor-사이
    transient crossing을 traversal이 첫 교차로 정확 검출** (수리의
    정확도 개선 실증).
  - sub-resolution 수용 규칙 fail-closed 합성 검증 (경쟁 part 주입 →
    거부, 이웃 밖 → 거부).
  - fallback fail-closed (cap 소진 합성 → 게이트 전부 false, 5,248 질의
    ≤ 5,400).
  - 회귀 불변식 8/8 (구 이분법 재구현과 q5*/part float-equality).
  - 결정성: PYTHONHASHSEED 상이 2 프로세스 canonical bytes bit-일치.
  - 정적 실행 안전 lens: argparse/쓰기 경로/w_ops3 순서/env gate 현재
    통과 전부 확인, ATTEMPT_ROOT 현재 부재 확인.
- **warning 9건 (전부 완화책 실재, 채택-차단 아님)** — 다음 세션 처분:
  - SCI-R1-W1: 인증 여유 < GJK tol(1e-6mm)일 때 이론적 비건전 —
    실측 여유는 자릿수 단위로 큼. → rule 텍스트 1문장 공개(소수리).
  - SCI-R1-W2: uncertified fallback의 coarse 진단(pinch/feature/margin)에
    provenance 마커 부재 — 게이트 누출 0 확인, CSV 인접 컬럼 join으로
    필터 가능. → 수용-기록(또는 진단 컬럼 1개 추가).
  - OPS2-W1: R4 'ast' 표면의 needle이 substring이라 죽은코드/계약 문자열
    에도 매치 (deadline 표면은 강제 블록 삭제 후에도 통과 실증) —
    완화 = writer sha가 tuple에 결박. → **정적 fixture 전 소수리 권장**
    (실제 AST 노드 검사로 교체).
  - OPS2-W2: 일부 fixture는 pass=False 반환 대신 raise → STATIC_FAIL
    기록 없이 소모(여전히 fail-closed이나 증거 손실). → try-wrap 소수리.
  - OPS2-W3: R12 runtime repr 재대조가 radius/height만 — x/table_z/
    z_center 미포함. → 3행 소수리.
  - L3-W1: `_write_json_x`에 allow_nan=False 부재. → 1행 소수리.
  - L3-W2: **정적 fixture 전 어떤 파일이든 수정하면 본 재검증의
    byte-결박이 실효** — 소수리 후 재검증 필요 범위를 diff로 한정.
  - L3-W3: A4 fixture의 의도적 omni import가 controller 전역
    _SCOPE_GUARD_VIOLATIONS에 잔류 가능 — prereg는 fixture 전에 기록되어
    무영향 확인. → note-only.
  - L3-W4: 정적 선존재 검사가 4경로만 — ATTEMPT_ROOT 현재 부재로 moot.

### 2.4 as-repaired sha256 (신규 기록 — as-authored sha는 5th doc §3 보존, 변조 아님·수리임)

| 파일 (sim_scripts/) | 줄수 | as-repaired sha256 (본 세션 종료 시점) |
|---|---|---|
| cyld29h50_..._worker.py | 2,390 | `fbf204179c6a3789aa0148220244448340867747008c30ca49f8c37dfa1170bb` |
| cyld29h50_..._controller.py | 3,847 | `83e32acdc3a9515646122030c8ee00a5c83878e882390687c3ca4d9d222b8549` |
| cyld29h50_..._manual_writer.py | 1,540 | `7f04458f3e44042be297fefcdd71638f9e62f2ef35cf0114df59ea5c76ea7f01` |

주의: §2.3 warning 소수리(OPS2-W1/W2/W3, L3-W1 + SCI-R1-W1 문구)를 다음
세션에서 적용하면 sha가 다시 변한다 — **정적 fixture는 최종 수리본
기준으로 1회만** (prereg가 실행 시점 bytes를 자체 결박하므로 순서만
지키면 됨).

## 3. 정적 fixture (미실행 — 다음 세션 1순위)

context 95% 비상 프로토콜 발동(213%)으로 본 세션에서는 **실행하지 않음**.
재검증 warning 소수리 → py_compile+스모크 → 최종 sha 기록 → 그 후에만:

```
/home/cgxr/miniconda3/envs/isaaclab/bin/python -B \
  sim_scripts/cyld29h50_d409_zero_step_dual_jaw_contact_region_enumeration_controller.py \
  --mode static-prep
```

**1회만.** prereg 선기록 구조 — fixture 실패 = attempt1 네임스페이스 영구
소모 fail-closed (재시도는 명시적 사용자 결정 필요; OPS NOTE4 문서화
의무 이행). R4 수리로 fixture 실패 시 STATIC_FAIL_STATUS 기록 후
attestation/tuple 저작 전에 정지하도록 보강됨.

## 4. Attestation / Tuple (미도달)

정적 fixture 성공 시 run_static_prep이 attestation + tuple을 자동 저작
후 정지 메시지 출력. attempt1 runtime은 **tuple SHA 인용 사용자 별도
승인** 후에만 (retry 0).

## 5. 다음 단계 + 사용자 브리핑 필수 항목

1. 재검증 warning 소수리 (§2.3 목록) → py_compile → 해당 스모크 재확인
   → 최종 as-repaired sha 기록.
2. 정적 fixture 1회 (§3) → attestation/tuple 확인 후 **정지**.
3. **attempt1 승인 전 사용자 브리핑 의무 (SCIENCE lens 판정)**: 4 probe
   pose 전부에서 첫 교차 witness가 top rim (기하 진실 — witness 결함
   아님, 독립 3중 검증). 실물 D29×H50은 원통 top(37.9mm)이 TCP
   z=z_center(12.9mm)보다 25mm 위라 조가 rim에 먼저 닿음 → **전 격자
   에서 (B) fail = A∧B 셀 0개가 확정 설계의 정당한 과학 결과일 수
   있음**. 이는 tolerance로 "수리"하면 안 되며(D354), 결과가 그렇게
   나오면 z 설계 재고는 별도 case·별도 승인 사안.
4. 기울임/전도힘 손측정(2nd doc §2)은 사용자 병행 — 보고 시 기록만.

과학 상태 불변: D407 FAIL-STOP, `g0a_pass=false`, 과학 verdict 없음.
attempt1 산출물 생성 0 (ATTEMPT_ROOT 미생성 확인). 로봇 HW·Isaac
runtime·lerobot-train 실행 0. D399 금지. 동결 침범 0. git commit/push 0.

## 6. Session progress rule 충족

실패 가능 검증 다수 실행·실제 반증 발생: (i) 1차 R1 구현이 스모크에서
cap 소진으로 **실패** → min→max 기준 결함 발견·정정, (ii) 2차 스모크에서
3/6 pose 인증 실패 → sub-resolution 구조 결함 발견·해상도-이웃 규칙
신설, (iii) 3-lens 적대 재검증이 warning 9건 실발견 (blocker 0),
(iv) audit 변조 검사·8-pose 스모크·결정성 2-프로세스 검사 각각 FAIL
가능 게이트였음.

## 3. 정적 fixture (수리 완결 후 1회)

(작성 중)

## 4. Attestation / Tuple (정지 지점)

(작성 중)

## 5. 다음 단계 / 사용자 결정 대기

(작성 중)
