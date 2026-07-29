# D405 static prep — 관측성 first-live-render 수리 3건 구현·정적 검증 (runtime 승인 = 유저 순차 지시)

Date: 2026-07-28 밤 KST. 이번 case의 신규 변수:
`[observability_first_live_render_repair_v1]` (정확히 1개 — D404가 결함 4건을 변수
1개로 일괄한 전례와 동일하게, 관측성 층의 첫 라이브/replay 실행에서 관측된 결함
3건에 대한 최소 reactive 수리 일괄).

**Runtime 승인 근거**: 유저 2026-07-28 순차 지시 "그럼 다음 최소 승인할테니
step-by-step으로 순차적으로 사고하면서 진행해" — D405 rung을 end-to-end(1회
실행 포함)로 승인. D405 attempt1 1회로 소진되며 retry/후속 case로 연장 불가
(prereg `authority`에 명기).

## 1. 무엇을 왜

D404 actual run은 기술 체인 전체를 최초로 통과한 뒤(technical_pass=true) 관측성
분기에서 FAIL_STOP했다 (DECISIONS D404). D405는 그 분기의 결함들을 controller
프로세스 한정으로 수리한다. **의무화된 정적 replay(D404 durable lesson)가 라이브
전에 잠재 결함 2건을 추가로 적발**했다 — D403 교훈("첫 라이브 실행 전에는 검증된
것이 아니다")의 재실증.

## 2. 수리 3건 (전부 설치 도구 실측 의미론에서 도출)

| # | 결함 (관측 근거) | 수리 (D405 controller wrapper) |
|---|---|---|
| 1 | script-path 실행의 `sys.path[0]=sim_scripts`, repo 루트 미포함 → preflight.py:2779 `roarm_rl` ModuleNotFoundError (**D404 라이브 관측**) | repo 루트를 sys.path **끝에** append (stdlib/site-packages 그림자 없음; worker는 roarm_rl 미사용) |
| 2 | 동결 `_write_rerun`은 `screenshot_window_size='1920x1080'`로 **물리** 1920×1080을 의도(_png_info exact 게이트·파일명·manual [1920,1080])하나, 설치 rerun 0.34.1의 `--window-size`는 **logical points**이고 headless 하니스 ppp가 고정 2.0 → 3840×2160 (**D405 replay 관측**; DISPLAY=:1/무DISPLAY/Xvfb 전부 동일 — T1/T2/T3 매트릭스) | `roarm_rl.rerun_contract.validate_rerun_artifact` 함수 객체를 wrap — 동결 리터럴 '1920x1080'만 logical '960x540'으로 번역 (1920/2.0 × 1080/2.0). 다른 크기·인자 전부 통과. 물리 게이트는 실제 1920×1080 PNG를 평가 |
| 3 | 동결 `_build_blueprint`(preflight.py:2457-2466)가 TextDocument 4개/3개를 뷰 1개에 묶음 → rerun 0.34.1이 "Can only show one text document at a time; was given 3" 오류를 내용 대신 렌더 + 토스트가 우측 텍스트 열 위에 겹침 (**D405 replay 스크린샷 관측**) | 로드된 동결 preflight 모듈의 `_build_blueprint` 함수 객체 교체 (동결 D402 controller가 이미 쓰는 loader-seam 패턴: `d404._load_frozen_d403_controller`→…→`d401._load_frozen_d400_controller` wrap). **동일 9개 엔티티**: 동결 Spatial3DView verbatim을 상단 전폭(share 0.55 — 토스트는 빈 하늘에만 겹침), 그 아래 텍스트 엔티티당 뷰 1개(status 4 + phase 3). 엔티티/타임라인/컴포넌트/로깅 변경 0 |

+ **fail-closed 사전 probe** (수리 아닌 guard): 위임·첫 쓰기 전에 contract 해석
경로/rerun SDK 0.34.1 pin/numpy 1.26.0(D326)/RERUN_CLI 존재를 검증 — 실패 시
쓰기 0에서 거부 (D404처럼 기술 완주 후 소진되는 것을 방지).

## 3. 실행 순서 (감사 가능 step-by-step)

1. **상태문서 갱신**: D404 actual run의 ledger row + DECISIONS D404 + 세션 doc +
   START_HERE overwrite.
2. **prereg 작성** (빌더, 상속 pin 48개 disk 재계산·assert 후 생성):
   `d405_preregistration.json` sha256
   `f63e6c69953926697cbb87202fbbb24bd751c897d2dca370373157dd1f4195b2`.
   allowed_dirty_paths 47, additional_frozen_repo_inputs 37(=D404의 23 + D404
   attempt 증거 12 + roarm_rl 2), inherited_science_contract 23(=18 + d404 layer
   5), installed sources 9(+rerun CLI 바이너리·SDK __init__ pin).
3. **wrapper 2종 작성**: controller(수리 3건 + probe + 4단 seam + rebind 25키 —
   D404 layer와 집합 동일), worker(순수 rebind 9키, 수리 0).
4. **정적 runner 실행** (scratchpad, isaaclab python -B): D404 runner 스테이지
   A~H verbatim 계승 + 신규 I(import/크기/seam fixture) + J(관측성 실물 replay).
   1차 실행에서 크기 결함 적발(fixture FAIL) → 수리 2·3 설계 → 재실행.
5. **관측성 실물 replay** (동결 D404 evidence 10.6MB + raw summary, 산출물은
   scratchpad에만): 미수리 재현 + 수리본 전체 + truncated RRD + 육안검수 게이트.
6. (진행 중) 4-lens 적대 리뷰 → attestation/tuple → 승인 게이트 오프라인 복제 →
   실행 직전 점검 → 호스트 경계 1회 실행 (+ 라이브 육안 검수 300s 내).

## 4. 정적 검증 수치 (source: `g0a_d405/attempt1_*/d405_static_fixture_results.json`, sha256 `a87e452338b4a29ba0f31e32d74524642437681bc7283609a890171430f412df`)

- **체크 34/34 + 양성 22/22 + 음성 51/51 전부 PASS** (`all_pass: true`).
- 체인 pin 10/10 exact (D400~D404 controller/worker 전부). D404-layer 전
  fixture(counter 11·Item 8·readback·normalizer·pxr replay) 재실행 PASS —
  D403-derivative 실물 replay 재현: 동결 실패 체크 2/attr 2, 65 mismatch →
  수리 후 0 (D404와 bit-일치).
- **수리 1 (import)**: script-path 모사 subprocess에서 미수리 import가 D404
  실패를 재현(reject), 수리 후 contract가 정확히 repo 파일로 해석(accept);
  probe 실측 `{rerun 0.34.1, numpy 1.26.0, cli exists}`; wrong-version/
  missing-cli/resolution-mismatch probe 전부 reject; site-packages 그림자 없음.
- **수리 2 (크기)**: wrapper가 정확히 동결 리터럴만 '960x540'으로 번역(accept),
  타 크기 통과(accept), 설치 idempotent(accept). **미수리 실물 재현**: 동결
  원본 함수로 replay → screenshot [3840,2160], `pass_before_manual=false`
  (reject fixture). 실험 매트릭스: DISPLAY=:1(세션 실측)/무DISPLAY/Xvfb 전부
  3840×2160; logical 960x540 → 정확히 1920×1080.
- **수리 3 (blueprint) + 전체 replay**: 4단 seam이 로드된 base에
  `_build_single_document_blueprint`를 설치함을 실측(accept). 수리본
  `_write_rerun` 전체 replay → RRD 1,213,969B 저장·footer verify PASS, 검증
  계약(엔티티 73 exact/타임라인 exact/컴포넌트/footer) 전부 PASS, 헤드리스
  스크린샷 **정확히 1920×1080** (sha `00e8350a...8432`),
  `pass_before_manual=true`, headless_viewer_invocations=1.
- **truncated_rrd_footer_rejected** (등록 id): 꼬리 4KB 절단 RRD → footer verify
  FAIL (reject).
- **육안검수 게이트 replay**: 유효 JSON accept / 잘못된 스크린샷 sha reject /
  subject 1개 false reject — 라이브 상호작용 절차 사전 검증 완료.
- **스크린샷 실물 육안 검수 (이 세션에서 직접 수행)**: 수리본
  (`d405_observability_replay/repaired/d400_rerun_viewer_1920x1080.png`,
  sha `00e8350a41ba05580e8b2c00c7c5cdf703799cb65d697f479f27324edf488432`) —
  3D 뷰에 source(파랑)/live SDF input(주황)/link5 A64(초록) 전부 가시,
  status 4패널(API schemas·owner inventory·cook queue·mass/counters)과 phase
  3패널이 **실제 JSON 내용을 렌더** (오류 배너 소멸), 토스트 3개는 3D 뷰 빈
  하늘 영역에만 겹침(텍스트 패널 겹침 0). 미수리본
  (`unrepaired/...png` 3840×2160, sha `4840f359...28b2`)은 양 텍스트 뷰에
  "Can only show one text document at a time; was given 3" 오류 배너 —
  결함 3의 시각 증거.
- `__pycache__` 신규 생성 0 (sim_scripts + roarm_rl 감시).

## 5. 도구 sha256 (scratchpad; 소스는 부록 A/B에 전문 보존)

| 도구 | sha256 |
|---|---|
| `d405_static_runner.py` | `7500670037aa373458f53dded81804ccc5a5a7a3d557b090dd3539c2e109f39c` |
| `d405_prereg_builder.py` | `445b09c132016f783221b3216e25a2d6a5f745c7c204c3cef7e677bbdb553d12` |
| controller wrapper (repo) | `eb54b29025270363d18cbcc42ed7f248304bbd543e2741df95c1b5fa3b8d6365` |
| worker wrapper (repo) | `938af5dc2981da26e3e2a5b60b92df7f5ba99ce52f78d2512f99715277743912` |

(wrapper sha는 attestation/tuple 작성 시점의 값이 최종 권위 — tuple이 pin.)

## 5b. 수리 2 실험 매트릭스 원본 증거 (prereg repair_2가 인용하는 "T1/T2/T3")

scratchpad `d405_observability_replay/`에서 동결 replay RRD에 대해 설치 rerun
0.34.1 CLI로 실측 (`--headless --port auto --window-size <W> --screenshot-to`):

| 시험 | 환경 | --window-size | 물리 출력 |
|---|---|---|---|
| T1 | `env -u DISPLAY` + scale 변수 4종=1 | 1920x1080 | **3840×2160** (`t1.png`) |
| T2 | `xvfb-run -a -s "-screen 0 1920x1080x24"` + scale 4종=1 | 1920x1080 | **3840×2160** (`t2.png`) |
| T3 | `env -u DISPLAY` | 960x540 | **1920×1080** (`t3.png`) |
| (stage J) | 세션 기본 DISPLAY=:1 | 1920x1080(동결)/960x540(수리) | 3840×2160 / 1920×1080 |

- `rerun --help` 원문: "`--window-size` Set the screen resolution (in logical
  points)"; "`--headless` ... driven by an offscreen `egui_kittest` harness".
- 숨은 ppp 플래그 없음(`--pixels-per-point` → rc=2 unknown), RERUN_* 환경변수에
  스케일 항목 없음(바이너리 strings 실측).
- 리뷰 독립 재현: reviewer probe 3종(DISPLAY=:1/미설정/:99) 전부 960x540 →
  정확히 1920×1080 (`probe_t1_display1.png`/`probe_t2_nodisplay.png`/
  `probe_t3_bogus.png`).
- 감사 노트: prereg의 "T1/T2/T3 매트릭스" 인용은 최초에 이 문서에 미기재였음
  (리뷰 warning) — 본 절이 그 원본 증거를 고정한다. prereg 자체는 sha-pin이라
  미수정.

## 5c. 라이브 육안 검수 운영 정의 (실행 전 고정 — 리뷰 권고)

- `text_overlap_or_clipping_observed`의 판정 대상 = **패널 텍스트 내용 위의
  겹침**(토스트가 패널 텍스트를 덮음, 패널끼리 겹침) 및 내용 잘림. 뷰 헤더
  제목의 말줄임(ellipsis)과 스크롤 가능한 JSON 내용이 패널 하단 경계에서
  끊기는 것은 표준 뷰어 동작으로 **겹침/잘림이 아니다**. 단, 실제 화면이 이
  정의로도 겹침이면 **정직하게 true로 보고**하고 run은 FAIL로 받아들인다.
- `subjects_visible` 5키도 실제 보이는 것만 true — 사전 확약 없음.

## 5d. 4-lens 적대 리뷰 결과 (wf_42e116e0-614; journal은 repo 밖 ~/.claude/.../subagents/workflows/wf_42e116e0-614/journal.jsonl; 691k tokens)

렌즈: repair correctness / chain freeze integrity / approval contract schema /
runtime interaction. **Blocker 3, warnings 19** — 전부 코드/prereg 변경 없이
attestation·절차로 해소:

1. **[B, approval]** attestation negative 집합은 등록 18 id의 superset 필요 —
   D405 실행분 51 + **hash-identity 재승인 8행**(source_mesh_hash/
   source_stream_bit_or_order/one_gripper_a64_still_active/
   link5_active_count_not_64/property_query_contract/cook_zero_to_zero/
   mass_com_or_inertia/worker_internal_fail_return_zero — 전부 D405가 건드리지
   않는 worker 서브시스템, 체인 10 pin bit-동일 + D404 라이브 통과 증거)로
   충족. 게이트는 pre-write라 실패해도 attempt 미소소진(단 발사 정지).
2. **[B, runtime]** 300s 육안검수 창: 백그라운드 exit 통지 대기 = 창 만료 =
   attempt 소진. → **발사 직후부터 ≤30s 간격 능동 폴링** (receipt 파일 출현
   감지 → PNG 검수 → JSON 작성), exit 대기 금지.
3. **[B, runtime]** 환경 게이트(t+0.2s)가 `ps -eo args`에서 controller/worker
   파일명 또는 OUT_DIR 문자열을 포함한 **비조상 프로세스** 발견 시 fail-stop
   (attempt 소진). → controller **단독 발사** 후 ≥3s 지나서만 감시/폴링 시작;
   발사와 같은 메시지에 OUT_DIR 명명 병렬 명령 금지.
4. [W 발췌] 검수 JSON은 **OUT_DIR 안 임시명으로 쓰고 같은 파일시스템
   rename**(원자적) — /tmp에서 cross-fs mv 금지; receipt 출현 전 어떤 런타임
   경로도 선작성 금지; zero-counter 정의문은 D404 문장 재사용 금지(D405
   replay가 rerun SDK/CLI를 scratchpad에서 실행했음을 명시); 25키 정정(반영됨);
   발사 직전 roarm_rl 2파일 sha를 prereg pin과 대조(probe는 경로/버전만 확인);
   rerun 청취 프로세스 부재 확인; 발사 직전 scratchpad 재렌더 1회로 ppp=2.0
   재확인; tuple sha를 발사 브리핑에 명시(유저 가시화); actual-run 세션 doc은
   run 이후에만 작성(allowlist 밖).

## 6. 다음 단계 (남은 절차)

4-lens 적대 리뷰(wf_42e116e0-614) blocker 0 확인 → attestation + tuple 작성 →
동결 `_validate_approval_tuple`(preflight.py:462-595) 오프라인 복제 PASS →
실행 직전 점검(호스트 경계·allowlist 47 전량 대조·HEAD==a69a96d·잔존 Isaac 0) →
호스트 경계 1회 실행. **라이브 관측성 완주 절차**: receipt/프롬프트 감지 →
스크린샷 PNG 실물 육안 검수 → 300s 내
`d400_manual_visual_inspection.json` 작성 (정직 보고 — 안 보이면 안 보인다고
쓴다) → completion 수집 → step-by-step 보고.

## 7. 경고

- D400~D404 전 attempt 동결 유지. D405 attempt1은 이 세션의 1회 실행으로 소진
  예정 (유저 순차 지시).
- allowlist(47) 밖 repo 파일 생성 절대 금지 — 실행 직전 전량 대조 필수.
- 이 세션의 실패 가능 요소: 정적 fixture 51 negative + 실물 replay + 라이브 1회
  실행 (session progress rule 충족).

## 부록 A — d405_static_runner.py 소스 전문 (sha256 `7500670037aa373458f53dded81804ccc5a5a7a3d557b090dd3539c2e109f39c`)

> 소스 전문은 scratchpad 세션 종료 후에도 감사가 가능하도록 보존한다. D404
> static prep 문서 부록 A와 동일한 관행. 스테이지 A~H는 D404 runner(그 문서
> 부록 A, sha `a47f13ce...745c7b`)의 verbatim 계승이므로 여기서는 **신규/변경
> 스테이지(I·J)와 헤더·main만** 전문 수록하고, A~H 본문은 D404 문서를 참조한다
> (계승 무결성은 위 runner 전체 sha로 pin).

```python
# 헤더/경로/EXPECTED(D404 pin 2행 추가)/allowed imports: D404 부록 A와 동일 구조.
# EXPECTED 추가분:
#   D404_CONTROLLER: 75070713db433ade735b2b227a1c642c6355fef352d82d12a3069c69b7642cef
#   D404_WORKER:     baa1e889ef324307bab695188ef3e163a7427a3f28f97150a4392c4f58ef3e82
# 신규 상수: D405_CONTROLLER/D405_WORKER/D405_PREREG, D404_ATT(동결 D404 attempt),
#   REPLAY_DIR(scratchpad), ISAACLAB_PY.
# stage_b 추가 체크: worker/controller가 실제 d405 prereg sha 내장,
#   d404 pin 일치, EXPECTED_RERUN_SDK_VERSION == frozen preflight.RERUN_VERSION
#   == "0.34.1", EXPECTED_RERUN_CLI == frozen RERUN_CLI, numpy pin 1.26.0.

IMPORT_PROBE_TEMPLATE = """
import sys
sys.path[:] = [p for p in sys.path if p not in ("", {repo!r})]
sys.path.insert(0, {simdir!r})
{repair}
try:
    import roarm_rl.rerun_contract as contract
except ModuleNotFoundError as error:
    print("IMPORT_FAIL", error)
    raise SystemExit(3)
from pathlib import Path
resolved = Path(contract.__file__).resolve()
expected = Path({repo!r}) / "roarm_rl" / "rerun_contract.py"
assert resolved == expected.resolve(), f"resolved outside repo: {{resolved}}"
assert callable(contract.validate_rerun_artifact)
print("IMPORT_OK", contract.RERUN_CONTRACT_VERSION)
"""


def stage_i(d405c) -> None:
    unrepaired = subprocess.run(
        [ISAACLAB_PY, "-B", "-c", IMPORT_PROBE_TEMPLATE.format(
            repo=str(REPO), simdir=str(SIM), repair="")],
        capture_output=True, text=True, timeout=120, cwd="/",
    )
    fixture(
        "observability_import_unrepaired_script_path_launch_rejected",
        "reject",
        unrepaired.returncode == 3
        and "IMPORT_FAIL" in unrepaired.stdout
        and "roarm_rl" in unrepaired.stdout,
        {"rc": unrepaired.returncode, "stdout": unrepaired.stdout.strip()},
    )
    repaired = subprocess.run(
        [ISAACLAB_PY, "-B", "-c", IMPORT_PROBE_TEMPLATE.format(
            repo=str(REPO), simdir=str(SIM),
            repair=f"sys.path.append({str(REPO)!r})")],
        capture_output=True, text=True, timeout=120, cwd="/",
    )
    fixture(
        "observability_import_repaired_launch_accepted", "accept",
        repaired.returncode == 0
        and "IMPORT_OK 0.34.1" in repaired.stdout,
        {"rc": repaired.returncode, "stdout": repaired.stdout.strip()},
    )

    d405c._repair_observability_import_path()
    probe = d405c._observability_preflight_probe()
    observations["d405_probe"] = probe
    fixture(
        "d405_controller_probe_accepts_on_host", "accept",
        probe["rerun_sdk_version"] == "0.34.1"
        and probe["numpy_version"] == "1.26.0"
        and probe["rerun_cli_exists"] is True
        and probe["roarm_rl_rerun_contract_file"]
        == str((REPO / "roarm_rl" / "rerun_contract.py").resolve()),
        probe,
    )

    saved_version = d405c.EXPECTED_RERUN_SDK_VERSION
    try:
        d405c.EXPECTED_RERUN_SDK_VERSION = "0.0.0"
        try:
            d405c._observability_preflight_probe()
            rejected = False
        except RuntimeError:
            rejected = True
    finally:
        d405c.EXPECTED_RERUN_SDK_VERSION = saved_version
    fixture("d405_probe_wrong_sdk_version_rejected", "reject", rejected)

    saved_cli = d405c.EXPECTED_RERUN_CLI
    try:
        d405c.EXPECTED_RERUN_CLI = Path("/dev/null/definitely_missing_rerun")
        try:
            d405c._observability_preflight_probe()
            rejected = False
        except RuntimeError:
            rejected = True
    finally:
        d405c.EXPECTED_RERUN_CLI = saved_cli
    fixture("d405_probe_missing_cli_rejected", "reject", rejected)

    saved_contract = d405c.ROARM_RL_CONTRACT_FILE
    try:
        d405c.ROARM_RL_CONTRACT_FILE = REPO / "roarm_rl" / "viz_debug.py"
        try:
            d405c._observability_preflight_probe()
            rejected = False
        except RuntimeError:
            rejected = True
    finally:
        d405c.ROARM_RL_CONTRACT_FILE = saved_contract
    fixture(
        "d405_probe_contract_resolution_mismatch_rejected", "reject", rejected
    )

    import numpy
    fixture(
        "site_packages_not_shadowed_by_repo_append", "accept",
        "site-packages" in str(Path(numpy.__file__).resolve())
        and str(REPO) in sys.path
        and sys.path[-1] != str(SIM),
        {"numpy": numpy.__file__},
    )

    # Repair 2 wrapper: exact-literal translation, passthrough, idempotence.
    d405c._install_screenshot_logical_size_repair()
    contract_mod = importlib.import_module("roarm_rl.rerun_contract")
    wrapped = contract_mod.validate_rerun_artifact
    missing = Path("/nonexistent/definitely_missing_d405_probe.rrd")
    report = wrapped(missing, screenshot_window_size="1920x1080")
    fixture(
        "screenshot_size_wrapper_translates_frozen_literal", "accept",
        report["screenshot_window_size"] == "960x540"
        and report["pass"] is False,
        {"window_size": report["screenshot_window_size"]},
    )
    report = wrapped(missing, screenshot_window_size="800x600")
    fixture(
        "screenshot_size_wrapper_passes_through_other_sizes", "accept",
        report["screenshot_window_size"] == "800x600",
    )
    d405c._install_screenshot_logical_size_repair()
    fixture(
        "screenshot_size_wrapper_install_idempotent", "accept",
        contract_mod.validate_rerun_artifact is wrapped,
    )

    # Repair 3 seam: the loaded frozen preflight carries the new blueprint.
    d404c = load_module(D404_CONTROLLER, "_d404_controller_for_seam_fixture")
    d405c._install_chain_render_repair(d404c)
    d403m = d404c._load_frozen_d403_controller()
    d402m = d403m._load_frozen_d402_controller()
    d401m = d402m._load_frozen_d401_controller()
    basem = d401m._load_frozen_d400_controller()
    fixture(
        "chain_seam_installs_render_repairs_on_loaded_base", "accept",
        basem._build_blueprint is d405c._build_single_document_blueprint,
    )


def stage_j(frozen_preflight, d405c) -> None:
    if REPLAY_DIR.exists():
        shutil.rmtree(REPLAY_DIR)
    dir_unrepaired = REPLAY_DIR / "unrepaired"
    dir_repaired = REPLAY_DIR / "repaired"
    dir_unrepaired.mkdir(parents=True)
    dir_repaired.mkdir(parents=True)

    base = frozen_preflight

    def bind(dirpath: Path) -> None:
        base.RRD_PATH = dirpath / "d400_sdf_preflight.rrd"
        base.RBL_PATH = dirpath / "d400_sdf_preflight.rbl"
        base.RERUN_VALIDATION_PATH = dirpath / "d400_rerun_validation.json"
        base.RERUN_SCREENSHOT_PATH = (
            dirpath / "d400_rerun_viewer_1920x1080.png"
        )
        base.RERUN_RECEIPT_PATH = dirpath / "d400_rerun_render_receipt.json"
        base.MANUAL_INSPECTION_PATH = (
            dirpath / "d400_manual_visual_inspection.json"
        )

    evidence = json.loads(
        (D404_ATT / "d400_live_configuration_owner_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    raw = json.loads(
        (D404_ATT / "d400_worker_raw_summary.json").read_text(encoding="utf-8")
    )

    # 1) Unrepaired defect reproduction: frozen blueprint + frozen physical
    #    size request under the installed rerun 0.34.1 headless semantics.
    #    Stage I already installed the size wrapper on the shared module, so
    #    temporarily restore the stashed frozen original for this call.
    contract_mod = importlib.import_module("roarm_rl.rerun_contract")
    frozen_validate = contract_mod._d405_frozen_validate
    wrapped_validate = contract_mod.validate_rerun_artifact
    bind(dir_unrepaired)
    contract_mod.validate_rerun_artifact = frozen_validate
    try:
        receipt_unrepaired = base._write_rerun(evidence, raw)
    finally:
        contract_mod.validate_rerun_artifact = wrapped_validate
    observations["replay_unrepaired_receipt"] = {
        "pass_before_manual": receipt_unrepaired["pass_before_manual"],
        "screenshot_size": list(receipt_unrepaired["screenshot"]["size"]),
        "screenshot_sha256": receipt_unrepaired["screenshot"].get("sha256"),
    }
    fixture(
        "replay_unrepaired_screenshot_scale_reproduces_defect", "reject",
        receipt_unrepaired["pass_before_manual"] is False
        and tuple(receipt_unrepaired["screenshot"]["size"]) == (3840, 2160)
        and receipt_unrepaired["screenshot"]["exact_1920x1080"] is False,
        observations["replay_unrepaired_receipt"],
    )

    # 2) Repaired full replay: install repairs 2-3 exactly as the live
    #    controller will, then replay the identical frozen branch.
    d405c._install_observability_render_repairs(base)
    bind(dir_repaired)
    receipt = base._write_rerun(evidence, raw)
    observations["replay_rerun_receipt"] = {
        "pass_before_manual": receipt["pass_before_manual"],
        "rrd_bytes": receipt["rrd"]["bytes"],
        "screenshot": receipt["screenshot"],
        "headless_viewer_invocations": receipt["headless_viewer_invocations"],
    }
    fixture(
        "observability_replay_write_rerun_receipt_pass_before_manual",
        "accept",
        receipt["pass_before_manual"] is True
        and receipt["screenshot"]["exact_1920x1080"] is True
        and receipt["headless_viewer_invocations"] == 1
        and base.RRD_PATH.is_file()
        and base.RBL_PATH.is_file()
        and base.RERUN_SCREENSHOT_PATH.is_file(),
        observations["replay_rerun_receipt"],
    )
    validation = json.loads(
        base.RERUN_VALIDATION_PATH.read_text(encoding="utf-8")
    )
    fixture(
        "observability_replay_validation_contract_pass", "accept",
        validation["pass"] is True
        and validation["entity_path_contract"]["pass"] is True
        and validation["timeline_contract"]["pass"] is True
        and validation["component_contract"]["pass"] is True
        and validation["footer_manifest_present"] is True
        and validation["errors"] == [],
        {"errors": validation["errors"]},
    )

    sys.path.append(str(REPO))
    from roarm_rl.rerun_contract import validate_rerun_artifact

    truncated = REPLAY_DIR / "truncated.rrd"
    blob = base.RRD_PATH.read_bytes()
    truncated.write_bytes(blob[: max(1024, len(blob) - 4096)])
    tval = validate_rerun_artifact(
        truncated,
        cli_path=base.RERUN_CLI,
        expected_version=base.RERUN_VERSION,
        timeout_s=120.0,
    )
    fixture(
        "truncated_rrd_footer_rejected", "reject",
        tval["pass"] is False,
        {"errors": tval.get("errors")},
    )

    screenshot_sha = base._sha(base.RERUN_SCREENSHOT_PATH)
    valid_manual = {
        "artifact": "D400_MANUAL_ORIGINAL_RESOLUTION_INSPECTION_V1",
        "inspection_completed": True,
        "screenshot_path": base._rel(base.RERUN_SCREENSHOT_PATH),
        "screenshot_sha256": screenshot_sha,
        "original_resolution": [1920, 1080],
        "subjects_visible": {
            "source_gripper_mesh": True,
            "live_sdf_input_mesh": True,
            "link5_a64": True,
            "api_token_attributes": True,
            "cook_queue_and_owner_status": True,
        },
        "text_overlap_or_clipping_observed": False,
        "observations": [
            "replay fixture inspection record (schema-conformance check "
            "only; the live run requires an actual visual inspection)"
        ],
    }
    saved_wait = base.MANUAL_INSPECTION_WAIT_S
    try:
        base.MANUAL_INSPECTION_WAIT_S = 5.0
        base.MANUAL_INSPECTION_PATH.write_text(
            json.dumps(valid_manual, indent=2), encoding="utf-8"
        )
        manual = base._wait_for_manual_inspection()
        fixture(
            "manual_inspection_gate_accepts_valid_inspection", "accept",
            manual["pass"] is True and all(manual["checks"].values()),
            manual["checks"],
        )

        wrong_sha = dict(valid_manual, screenshot_sha256="0" * 64)
        base.MANUAL_INSPECTION_PATH.write_text(
            json.dumps(wrong_sha, indent=2), encoding="utf-8"
        )
        manual = base._wait_for_manual_inspection()
        fixture(
            "manual_inspection_wrong_screenshot_sha_rejected", "reject",
            manual["pass"] is False
            and manual["checks"]["screenshot_sha_exact"] is False,
        )

        missing_subject = copy.deepcopy(valid_manual)
        missing_subject["subjects_visible"]["link5_a64"] = False
        base.MANUAL_INSPECTION_PATH.write_text(
            json.dumps(missing_subject, indent=2), encoding="utf-8"
        )
        manual = base._wait_for_manual_inspection()
        fixture(
            "manual_inspection_missing_subject_rejected", "reject",
            manual["pass"] is False
            and manual["checks"]["all_subjects_visible"] is False,
        )
    finally:
        base.MANUAL_INSPECTION_WAIT_S = saved_wait


# main(): D404 부록 A와 동일 골격 + frozen_preflight 로드, stage_i(d405c),
# stage_j(frozen_preflight, d405c), __pycache__ 감시를 sim_scripts+roarm_rl로
# 확장, summary artifact "D405_STATIC_FIXTURE_RESULTS_V1".
```

## 부록 B — d405_prereg_builder.py (sha256 `445b09c132016f783221b3216e25a2d6a5f745c7c204c3cef7e677bbdb553d12`)

> 빌더는 (1) D404 prereg sha 검증, (2) 상속 pin 48개 disk 재계산·assert,
> (3) git HEAD==origin/master==a69a96d assert, (4) 현 dirty + 계획 7경로의
> 합집합으로 allowed_dirty_paths 47 구성, (5) D404 attempt 증거 12 + roarm_rl
> 2를 frozen inputs에 추가, (6) d404 layer 5 레코드를 inherited contract에
> 추가, (7) rerun CLI/SDK pin 2건을 installed sources에 추가 후 prereg를
> 1회성(x-mode 아님, 존재 시 거부)으로 기록한다. 전문은 scratchpad에 있으며
> 위 sha로 pin — 구조는 본문 §2~§3과 prereg 자체(`d405_preregistration.json`)에
> 완전히 반영되어 있다.
