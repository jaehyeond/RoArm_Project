# D406 static prep — prereg status 리터럴 계약 준수 수정 · 정적 검증 (runtime 승인 = 유저 명시 지시)

Date: 2026-07-28 밤 KST. 이번 case의 신규 변수: **없음 (0개)** — 계약 준수 수정
(D405 prereg 저작 결함의 문서 수정). D405의 bundled 변수
`observability_first_live_render_repair_v1`은 라이브 미판정(R2/R3)으로 그대로
이월되며, R1은 D405 attempt1에서 이미 라이브 실증됨.

**Runtime 승인 근거**: 유저 2026-07-28 명시 지시 "D405 attempt1로 소진 — D406은
정적 준비부터 새 명시 승인. step-by-step으로 순차적으로 사고하면서 진행해" —
D405 순차 지시 전례와 동일하게 D406 rung을 정적 준비부터 end-to-end(1회 실행
포함)로 해석. 발사 전 tuple sha를 브리핑에 명시. D406 attempt1 1회로 소진되며
retry/후속 case로 연장 불가 (prereg `authority`에 명기).

## 1. 무엇을 왜

D405 actual run은 인프라·선행 게이트 전부 PASS + 수리 R1 라이브 실증 후, 동결
worker.py:2517의 prereg admission(입장 검사)에서 정지했다: D405 prereg가 status를
동결 리터럴 `"PREREGISTERED_NOT_EXECUTED"` 대신 "더 서술적인"
`"STATIC_PREP_PREREGISTERED_RUNTIME_PENDING"`으로 저작한 것 (DECISIONS D405).
D406은 그 문서 결함만 수정한다: **코드 수리 0건, 신규 변수 0개** — prereg를
동결 소스에서 도출한 리터럴로 재저작하고, 같은 실수의 재발을 막는 동결
admission replay fixture를 정적 runner에 추가한다.

## 2. 수리 내용 (전부 동결 소스 실측에서 도출)

| # | 결함 (관측 근거) | 수리 |
|---|---|---|
| 1 | D405 prereg `status` ≠ 동결 worker.py:2517 리터럴 → RuntimeError("D400 preregistration status is not frozen"), derivative 전 정지 (**D405 라이브 관측**, raw summary exception) | D406 prereg의 status를 빌더가 동결 worker 소스에서 **프로그램적으로 추출**해 저작 (하드코딩 0; 추출값 == "PREREGISTERED_NOT_EXECUTED" assert). wrapper 2종에 새 prereg sha 재내장 |
| guard | 순수 admission 체크(sha+status)가 라이브에서 첫 실행되는 것 자체가 결함 (D405 lesson) | 정적 runner **신규 stage K**: 동결 D400 worker 모듈(import-inert) 로드 후 admission 소스 5행(:2514-2518)을 내용 매칭으로 추출·**verbatim exec** — D406 prereg accept / 변조 status reject / hash drift reject / **실물 D405 prereg가 라이브 실패 메시지 bit-exact 재현** |

### 전수 소비자 감사 (D405 durable lesson 이행 — 이 세션 grep 실측)

동결 체인이 top-level prereg에서 읽는 **완전 집합**:

| # | 소비자 | 읽는 것 |
|---|---|---|
| ① | wrapper 2종 EXPECTED_PREREG_SHA256 + worker.py:2514 + preflight.py:493/585 | 파일 sha256 |
| ② | worker.py:2517 (체인 유일 status 읽기) | `status` 리터럴 |
| ③ | D401 controller merge (:182-204) | `git_baseline`, `runtime_overlay_contract.{allowed_dirty_paths, additional_frozen_repo_inputs}`, `inherited_science_contract`, `new_variables`, `git_snapshot_contract` |
| ④ | **D402 controller merge wrap (:244)** | `installed_nvidia_primary_sources` (비어있거나 malformed면 RuntimeError) |

⚠️ ④는 **DECISIONS D405의 "6키" 열거에 누락**되어 있던 7번째 필수 키 — 이
세션 grep이 복구 (D405 prereg는 상속 패턴 덕에 우연히 포함해 통과했었음).
D406 prereg는 ①~④ 전부 명시 충족 + `consumer_literal_audit` 섹션에 기록.

## 3. 실행 순서 (감사 가능 step-by-step)

1. 부트 검증 중 START_HERE.md:78-79 오기 발견·정정 (D405 attempt "16 +
   collision_asset" → 실측 13, collision_asset 미생성; 세션문서 8→7종).
2. **prereg 빌더** (scratchpad, sha 하단 표): D405 prereg sha 검증 → 상속 pin
   **69개 disk 재계산·assert** → status 리터럴 동결 소스 추출 → git 실측
   dirty 57 ∪ 계획 7 = allowlist 64 → `d406_preregistration.json` 생성,
   sha256 `c49801577f44590774927ca2b74a23be233a536025db2d784d042faf01c4c7de`.
   frozen inputs 47(=D405 37 + D405 attempt 증거 10), inherited 28 records,
   installed sources 9 (전부 재검증).
3. **wrapper 2종** (순수 rebind, 수리 0): controller 25키/worker 9키를 동결
   D405 wrapper(hash pin eb54b290/938af5dc) 위에 rebind.
   controller sha `1273753cbe5c784ede1d1a327d1e7fa7145868bdaf196764e043e15c05f9ad0b`,
   worker sha `42b141f9a0c00de58a24b49c8daec08c1b40f3270582efa2814b8512f1ad9807`.
4. **정적 runner** (D405 runner 스테이지 A~J verbatim 계승 + 신규 K):
   isaaclab python -B 실행 → **체크 47/47 + accept 26/26 + reject 56/56 전부
   PASS** (`d406_static_fixture_results.json`, sha
   `339248e990d3f7708bb7acf4601db476a876b1c0c5f1e0a24964b71bc3aaa1bd`).
5. **replay 스크린샷 실물 육안 검수** (이 세션 직접 수행): 수리본
   (scratchpad `d406_observability_replay/repaired/...png`, sha
   `35450c5b4884ffc18da3c6d5cf29e1e4b9b2cff377afc8a24f5d387e9a6855b3`,
   정확 1920×1080) — 3D 뷰에 source(파랑)/live SDF input(주황)/link5
   A64(초록) 전부 가시, status 4패널+phase 3패널 실제 JSON 렌더, 토스트는 3D
   뷰 빈 하늘에만. 미수리본(3840×2160) — 양 텍스트 뷰에 "Can only show one
   text document..." 오류 배너 재현.
6. 4-lens 적대 리뷰 (§5d) → attestation/tuple → 동결 승인 게이트 오프라인
   복제 → 실행 직전 점검 → 호스트 경계 1회 실행.

## 4. 정적 검증 수치 (source: `g0a_d406/attempt1_*/d406_static_fixture_results.json`)

- **체크 47/47 + 양성 26/26 + 음성 56/56 전부 PASS** (`all_pass: true`).
- 체인 pin 12/12 exact (D400~D405 controller/worker 전부 — D405 wrapper 2종
  pin 신규 추가).
- **stage K (신규, 동결 admission replay)**: admission 블록을 내용 매칭으로
  :2514에서 발견(하드코딩 라인 아님, 발견 위치 check), 리터럴 도출 =
  `PREREGISTERED_NOT_EXECUTED`; ① D406 prereg + 실제 sha → **accept**;
  ② status 변조 사본(D405의 실제 오류 문자열) → "status is not frozen"
  reject; ③ EXPECTED sha "0"*64 → "hash drift" reject; ④ **실물 동결 D405
  prereg + 실제 sha → 라이브 실패 예외 메시지와 문자열 bit-exact 재현**
  ("D400 preregistration status is not frozen"); ⑤ D403/D404 prereg status ==
  도출 리터럴 (체인 회귀 증거); ⑥ **rebind 전파 fixture**: d406c→d405
  25키 직접+전파(dummy) 전부 일치·초과 키 0, d406w→d405 9키 동일.
- stage B 신규: D406 prereg의 소비자 키 7종 shape 검증(allowlist 64,
  frozen inputs 47, inherited records 28, new_variables [], nvidia 9 records
  전부 {path,sha256} str) + **현재 dirty ⊆ allowlist 라이브 검증**.
- D405-layer 전 fixture 재실행 PASS: D403-derivative 실물 replay 재현(동결
  실패 체크 2/attr 2, 65 mismatch → 수리 후 0), 관측성 import 수리
  재현/수리(subprocess), probe 실측 {rerun 0.34.1, numpy 1.26.0, cli
  exists}, 크기 wrapper 리터럴 번역/통과/idempotent, truncated RRD footer
  reject, 육안검수 게이트 accept/2 reject.
- stage J 실물 replay: 수리본 RRD 1,213,978B footer verify PASS, 검증
  계약(엔티티/타임라인/컴포넌트/footer) 전부 PASS, 스크린샷 정확 1920×1080,
  `pass_before_manual=true`, headless_viewer_invocations=1; 미수리본
  3840×2160 + `pass_before_manual=false` 재현.
- `__pycache__` 신규 생성 0 (sim_scripts + roarm_rl 감시).

## 5. 도구 sha256 (scratchpad; 원본 보존)

| 도구 | sha256 |
|---|---|
| `d406_prereg_builder.py` | `afe29920a34231116443e0becb26cff218468cc83449de8ec6b97853628d0974` |
| `d406_static_runner.py` | `865dd1120e457bbed0fc4761c800d7468d20dd53dcfb2754160a8baed1067b48` |
| `d406_gate_replication.py` | `560ef8f855982bd5b73b96cdc41ce91279bc7cbd2a481d9e4e3198dc9264aa11` |
| `d406_attestation_builder.py` | `ab8187cbfa9f581407ebd639293c9f70c5cd1f7aef4edbf1db49690a79711ff5` |
| `d406_inspection_writer.py` (원자적 검수 작성기 — §5d blocker 해소) | `566e2aeefffda35344e6b2377f8415d615c3691f7f306c151719bd1820493900` |

runner 스테이지 A~J는 D405 runner(sha `75006700...`, D405 static prep doc
부록 A + D404 doc 부록 A)의 verbatim 계승이므로, 부록 A에는 신규 stage
K·변경 헤더만 수록한다. wrapper sha는 attestation/tuple 작성 시점 값이 최종
권위 — tuple이 pin.

## 5c. 라이브 육안 검수 운영 정의 (D405 §5c 상속, 실행 전 고정)

- `text_overlap_or_clipping_observed` 판정 대상 = 패널 텍스트 내용 위의
  겹침(토스트가 패널 텍스트를 덮음, 패널끼리 겹침) 및 내용 잘림. 뷰 헤더
  제목 말줄임과 스크롤 가능한 JSON 내용이 패널 하단 경계에서 끊기는 것은
  표준 뷰어 동작으로 겹침/잘림이 아니다. 단, 실제 화면이 이 정의로도
  겹침이면 **정직하게 true로 보고**하고 run은 FAIL로 받아들인다.
- `subjects_visible` 5키도 실제 보이는 것만 true — 사전 확약 없음.
- 검수 JSON은 OUT_DIR 안 임시명으로 쓰고 같은 파일시스템 rename(원자적).
  receipt 출현 전 어떤 런타임 경로도 선작성 금지.

## 5d. 4-lens 적대 리뷰 결과 (wf_fc718cbc-23d; journal ~/.claude/.../9273a397-.../subagents/workflows/wf_fc718cbc-23d/journal.jsonl; 503,644 tokens, 4 agent 전원 완료)

렌즈: repair correctness / chain freeze integrity / approval contract schema /
runtime interaction. **Blocker 1, warnings 18** — 수리 자체는 4개 렌즈 전부
"refutation failed" (각 렌즈가 독립 grep·해시 재계산으로 7-key 소비자 집합
완전성, 체인 pin 12개, 상속 pin 47+28+9, rebind 25/9키 집합 동일성을 재검증).

1. **[B, runtime]** 300s 육안검수 상호작용 = 체인 사상 첫 라이브 경로,
   zero-retry, 0.25s 폴링 first-read-wins (preflight.py:3085-3127) — 비원자적
   쓰기는 mid-write에 읽혀 attempt 소진; 백그라운드 exit 대기는 timeout 확정.
   → **해소**: 원자적 검수 작성기 사전 준비(임시명 OUT_DIR 내 생성 + 같은 fs
   os.rename; argv는 /tmp 경로만; 주제 5키 각각 명시 assert 강제 — 정직성
   가드) 후 scratchpad dry-run으로 원자 쓰기/정직 오버라이드/덮어쓰기 거부/
   잔존 temp 0 전부 검증. 발사 직후부터 ≤30s 능동 폴링(캡처된 stdout 파일
   in-process 읽기, 프로세스 스폰 0). 코드/prereg 변경 0.
2. [W 발췌 — 전체 18건 중 실행 절차 구속력 있는 것] ① attestation/tuple은
   **실물 파일**로 게이트 복제 재실행 후에만 발사 (→ 이행: 10/10 PASS);
   ② hash-identity 8행 근거를 D406 전용으로 재서술 (→ attestation
   fixture_evidence_basis에 이행); ③ tuple 저작 후 4개 바인딩 파일(wrapper
   2·prereg·attestation) 일체 수정 금지; ④ 발사 전 commit/push/stash 절대
   금지 (HEAD 변경 = attempt 소진 경로); ⑤ 감시 argv에 controller/worker
   파일명·OUT_DIR 문자열 금지 — 환경 게이트는 발사 수 초 내 발화하며 실패 시
   attempt 소진; ⑥ 검수 JSON의 screenshot_path는 프롬프트 required_fields의
   repo-상대 문자열 verbatim (작성기가 verbatim 복사); ⑦ roarm_rl 2파일은
   라이브 소비 frozen input — 발사까지 불변 유지; ⑧ 발사 직전 git
   status/HEAD/ps 3-문자열/VRAM 재감사 의무; ⑨ stage K rebind 전파는
   D406→D405 hop만 정적 검증 (하위 hop은 D405 라이브 실증 + 구조 확인으로
   충분 — 증거 한계 명시); ⑩ 프롬프트 출현 예상 t+1~2.5분, 30s 감지 여유
   ≥270s — 능동 폴링 유지 시에만 충분.

## 5e. Attestation·tuple·게이트 복제 (실물)

- attestation sha256 `e65ac4d4dfa77f215a45f34c77f5f154d1789d0e4f0d95802b4e2dbac996ac5b`
  (negative 64행 = 실행 56 + hash-identity 8, 등록 18 id superset).
- **tuple sha256 `bc54e7c51c0ca5ef367595c53dcb5e06b7a9afbb2907a607642a65ebf9adf435`**
  (prereg c4980157/attestation e65ac4d4/controller 1273753c/worker 42b141f9).
- 동결 `_validate_approval_tuple` 오프라인 복제 (isaaclab python -B, 실물
  파일): **10/10 PASS** (accept 1 + malformed sha/wrong sha/missing
  attestation/registered id 삭제/30행 미만/counter 비영/binding 불일치/tuple
  순서 교란/tuple 해시 불일치 reject 9).

## 6. 다음 단계 (남은 절차)

4-lens 리뷰 blocker 0 확인 → attestation + tuple 작성 → 동결
`_validate_approval_tuple`(preflight.py:462-595) 오프라인 복제 10/10 PASS →
실행 직전 점검(호스트 경계·allowlist 64 전량 대조·HEAD==a69a96d·잔존
Isaac/rerun 0·ppp 재확인 렌더 1회·roarm_rl 2파일 sha 대조) → controller 단독
백그라운드 발사(tuple sha 브리핑 명시) → ≥3s 후 task 출력만 참조하는 감시
(OUT_DIR 문자열 금지) → 프롬프트 감지 즉시 PNG 실물 육안 검수 → 300s 내 검수
JSON 원자적 rename → completion 수집 → step-by-step 보고.

## 7. 경고

- D400~D405 전 attempt 동결 유지. D406 attempt1은 이 세션 1회 실행으로 소진
  예정 (유저 명시 지시).
- allowlist(64) 밖 repo 파일 생성 절대 금지 — 실행 직전 전량 대조 필수.
- 이 세션의 실패 가능 요소: 정적 fixture 56 negative + 동결 admission replay
  + 실물 rerun replay + 라이브 1회 실행 (session progress rule 충족).

## 부록 A — d406_static_runner.py 신규/변경분 전문 (전체 sha `865dd1120e457bbed0fc4761c800d7468d20dd53dcfb2754160a8baed1067b48`)

> 스테이지 A~J는 D405 runner 부록 A(및 그 안에서 참조되는 D404 doc 부록 A)의
> verbatim 계승. 여기는 신규 stage K와 헤더 변경분만 수록 (계승 무결성은 위
> runner 전체 sha로 pin).

```python
# 헤더 변경분: EXPECTED에 D405 pin 2행 추가
#   D405_CONTROLLER: eb54b29025270363d18cbcc42ed7f248304bbd543e2741df95c1b5fa3b8d6365
#   D405_WORKER:     938af5dc2981da26e3e2a5b60b92df7f5ba99ce52f78d2512f99715277743912
# 신규 상수: D406_CONTROLLER/D406_WORKER/D406_PREREG, ADMISSION_PROBE_DIR,
#   CONTROLLER_REBIND_KEYS(25)/WORKER_REBIND_KEYS(9).
# stage_b 추가: d406 wrapper AST/import scan, d406 prereg sha 내장 검증,
#   d405 pin 검증, D406 prereg 소비자 키 7종 shape 검증, dirty⊆allowlist.
# stage_c: -B 거부 4건 (d405 2 + d406 2).

ADMISSION_HEAD = "if _sha(prereg_path) != EXPECTED_PREREG_SHA256:"


def extract_admission() -> tuple[str, int, str]:
    """Locate and extract the frozen admission statements by content match."""

    source_lines = D400_WORKER.read_text(encoding="utf-8").splitlines()
    starts = [
        index for index, line in enumerate(source_lines)
        if line.strip() == ADMISSION_HEAD
    ]
    assert len(starts) == 1, f"expected one admission head, got {starts}"
    start = starts[0]
    block = source_lines[start:start + 5]
    snippet = textwrap.dedent("\n".join(block))
    match = re.search(r'prereg\.get\("status"\)\s*!=\s*"([^"]+)"', snippet)
    assert match, snippet
    return snippet, start + 1, match.group(1)


def run_admission(frozen_worker, snippet: str, prereg_path: Path,
                  expected_sha: str) -> dict:
    """Execute the frozen admission statements verbatim."""

    namespace = {
        "_sha": frozen_worker._sha,
        "_read_json": frozen_worker._read_json,
        "EXPECTED_PREREG_SHA256": expected_sha,
        "prereg_path": prereg_path,
    }
    exec(  # noqa: S102 — verbatim replay of hash-pinned frozen source
        compile(snippet, "<frozen_d400_worker_admission>", "exec"), namespace
    )
    return namespace["prereg"]


def stage_k(frozen_worker, d406c, d406w) -> None:
    snippet, start_line, literal = extract_admission()
    observations["admission_replay"] = {
        "start_line": start_line,
        "derived_status_literal": literal,
    }
    check("admission_block_found_at_frozen_line_2514", start_line == 2514,
          start_line)
    check("admission_status_literal_derived_from_frozen_source",
          literal == "PREREGISTERED_NOT_EXECUTED", literal)

    d406_sha = observations["d406_prereg_sha256"]
    try:
        prereg = run_admission(frozen_worker, snippet, D406_PREREG, d406_sha)
        admitted = prereg.get("status") == literal
        detail = {"status": prereg.get("status")}
    except RuntimeError as error:
        admitted = False
        detail = str(error)
    fixture("frozen_prereg_admission_accepts_d406_prereg", "accept",
            admitted, detail)

    if ADMISSION_PROBE_DIR.exists():
        shutil.rmtree(ADMISSION_PROBE_DIR)
    ADMISSION_PROBE_DIR.mkdir(parents=True)
    tampered_doc = json.loads(D406_PREREG.read_text(encoding="utf-8"))
    tampered_doc["status"] = "STATIC_PREP_PREREGISTERED_RUNTIME_PENDING"
    tampered_path = ADMISSION_PROBE_DIR / "d406_prereg_tampered_status.json"
    tampered_path.write_text(
        json.dumps(tampered_doc, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        run_admission(frozen_worker, snippet, tampered_path,
                      sha(tampered_path))
        rejected, detail = False, None
    except RuntimeError as error:
        rejected = "status is not frozen" in str(error)
        detail = str(error)
    fixture("frozen_prereg_admission_tampered_status_rejected", "reject",
            rejected, detail)

    try:
        run_admission(frozen_worker, snippet, D406_PREREG, "0" * 64)
        rejected, detail = False, None
    except RuntimeError as error:
        rejected = "hash drift" in str(error)
        detail = str(error)
    fixture("frozen_prereg_admission_hash_drift_rejected", "reject",
            rejected, detail)

    try:
        run_admission(frozen_worker, snippet, D405_PREREG,
                      observations["d405_prereg_sha256"])
        rejected, detail = False, None
    except RuntimeError as error:
        rejected = str(error) == "D400 preregistration status is not frozen"
        detail = str(error)
    fixture("frozen_prereg_admission_reproduces_d405_live_failure", "reject",
            rejected, detail)

    d403_status = json.loads(
        (D403_ATT / "d403_preregistration.json").read_text(encoding="utf-8")
    )["status"]
    d404_status = json.loads(
        (D404_ATT / "d404_preregistration.json").read_text(encoding="utf-8")
    )["status"]
    fixture(
        "d403_d404_preregs_carry_frozen_admission_literal", "accept",
        d403_status == literal and d404_status == literal,
        {"d403": d403_status, "d404": d404_status},
    )

    c405 = d406c._load_frozen_d405_controller()
    d406c._configure_d405_paths(c405)
    direct_mismatch = [
        key for key in CONTROLLER_REBIND_KEYS
        if getattr(c405, key) != getattr(d406c, key)
    ]
    dummy = SimpleNamespace()
    c405._configure_d404_paths(dummy)
    dummy_attrs = dict(vars(dummy))
    propagated_mismatch = [
        key for key in CONTROLLER_REBIND_KEYS
        if dummy_attrs.get(key) != getattr(d406c, key)
    ]
    fixture(
        "d406_controller_rebind_propagates_all_25_keys", "accept",
        not direct_mismatch and not propagated_mismatch
        and set(dummy_attrs) == set(CONTROLLER_REBIND_KEYS),
        {"direct_mismatch": direct_mismatch,
         "propagated_mismatch": propagated_mismatch,
         "extra_keys": sorted(set(dummy_attrs)
                              - set(CONTROLLER_REBIND_KEYS))},
    )

    w405 = d406w._load_frozen_d405_worker()
    d406w._configure_d405_paths(w405)
    dummy = SimpleNamespace()
    w405._configure_d404_paths(dummy)
    dummy_attrs = dict(vars(dummy))
    propagated_mismatch = [
        key for key in WORKER_REBIND_KEYS
        if dummy_attrs.get(key) != getattr(d406w, key)
    ]
    fixture(
        "d406_worker_rebind_propagates_all_9_keys", "accept",
        not propagated_mismatch
        and set(dummy_attrs) == set(WORKER_REBIND_KEYS),
        {"propagated_mismatch": propagated_mismatch,
         "extra_keys": sorted(set(dummy_attrs) - set(WORKER_REBIND_KEYS))},
    )
```

## 부록 B — d406_prereg_builder.py (sha `afe29920a34231116443e0becb26cff218468cc83449de8ec6b97853628d0974`)

> 빌더는 (1) D405 prereg sha 검증, (2) 상속 pin 69개 disk 재계산·assert,
> (3) **status 리터럴을 동결 worker 소스에서 정규식 추출** (유일 출현 assert +
> 기대값 assert), (4) git HEAD==origin/master==a69a96d assert, (5) 현 dirty
> 57 + 계획 7경로 합집합으로 allowed_dirty_paths 64 구성 (중복 0 assert),
> (6) D405 attempt 증거 10을 frozen inputs에 추가, (7) d405 layer 5 records를
> inherited contract에 추가, (8) `consumer_literal_audit` 섹션에 전수 감사
> 결과 기록 후 prereg를 1회성(존재 시 거부)으로 기록. 전문은 scratchpad에
> 있으며 위 sha로 pin.
