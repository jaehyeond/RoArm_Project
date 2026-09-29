"""준비(readiness) 렌더 **경계 런처** — 600 s 총 한도 안에서 startup·render·ffmpeg·close 를 계측한다.

usage:
  python readiness_launcher.py --run <raw_dir> --out <new_dir> [--budget-s 600]
                               [--max-frames 8] [--synthetic-phases 2] [--plan-only]
                               [--runner <path>] [--renderer-argv-override ...]

설계 원칙: 신호·마감 로직을 **또 한 벌 쓰지 않는다**
------------------------------------------------
`run_production.py` 가 이미 감사 재현 6건 + 소유권 mock + 공유 조기 차단선을 통과한 시간 계약을
들고 있다. 그래서 이 런처는 **준비용 mini revision 을 정상 동결 구조로 만들고 그 실행기를 부른다**.
런처 안에 `killpg`·`setitimer`·`SIGKILL` 은 없다.

`ROOT_READINESS_PREFLIGHT_02.md` 가 잡은 연결 결함 4건을 닫는다
-------------------------------------------------------------
1. **해시 계약 미충족** — 예전 판은 임시 manifest 에 `planned_outputs` 만 넣고
   `frozen_copies_sha256={}` 로 뒀으며, **부르는 실행기 파일이 임시 revision 바깥**이었다.
   실행기는 `manifest["external_frozen_inputs_sha256"]` 를 **직접 읽고**(없으면 KeyError),
   `runner_self` 가 revision 안에서 pin 돼 있지 않으면 `want=None` → 불일치 → **rc 3 중단**이다.
   즉 예전 런처는 렌더에 도달조차 못 했다. → 해시 검사를 **약화하지 않고**, mini revision 안에
   실행기·렌더러·그 import 모듈을 **복사해 넣고 실제 해시로 pin** 하며, 외부 입력 해시와
   criteria 해시를 manifest 에 **실제로** 채운다. 그리고 **그 복사본 실행기를 부른다**.
2. **출력 경로 충돌** — 예전 판은 `--out` 안에 계획/attempt 를 만든 뒤 **같은 경로**를 렌더러에
   줬고, 렌더러는 비어 있지 않은 출력 폴더를 거절한다(배타 소유). → 계획/영수증은 `<out>/plan`,
   실행기 attempt 는 `<out>/attempt`, **렌더 출력은 `<out>/render`** 로 분리한다.
   `<out>/render` 는 런처가 **만들지 않는다** — 렌더러가 배타적으로 만든다.
3. **합성 자세 미전달** — `--synthetic-phases` 를 넘기지 않아 raw 6 프레임만 재생되고
   운반·배출 위치·HOME 표시가 확인되지 않았다. → 승인 범위 ≤8 **안에서** 명시 합성 자세를 포함한다.
   합성 프레임은 라벨이 붙고 **물리 결과로 세지 않는다**.
4. **close timeout 이 성공으로 통과** — 렌더러가 `os._exit(rc_)` 로 나가 렌더 ok 면 rc 0 이었고,
   런처는 `close_timed_out` 을 기록만 하고 판정에 쓰지 않았다. → 렌더러는 close 시간초과에
   **전용 비성공 코드**로 나가고, 런처는 그것을 `honest_gaps` 에 넣어 `ok` 를 주장하지 않는다.
   **SIGALRM 합성 대조는 실제 Isaac C++ 종료의 무조건 시간 보장이 아니다** — 바깥 실행기의
   그룹 TERM→KILL 제한이 **여전히 필수**이며 그것이 최종 경계다.

주장하지 않는 것
--------------
이 런처는 **실행 계획과 계측 계약**이다. 렌더 성공·육안 검수 통과의 증거가 아니다.
`--plan-only` 는 GPU 를 건드리지 않는다. GPU/렌더 HOLD 중에는 그것만 쓴다.
합성 자세는 표시 점검용이며 물리 결과가 아니다.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ISAACLAB_PY = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/python")
ROARM_PY = Path("/home/cgxr/miniconda3/envs/roarm/bin/python")
USD_DEFAULT = Path("/home/cgxr/Documents/Robotics/RoArm_Project/local_assets/roarm_m3/"
                   "usd_s1_v1/roarm_m3_s1_v1.usd")
MAX_BUDGET_S = 600.0              # 계약 경계. 이 값을 넘기는 계획은 만들지 않는다.
MAX_FRAMES = 8                    # 승인 범위. 합성 포함 총합이다.
# 렌더러가 import 하는 지역 모듈까지 mini revision 에 넣어야 해시 계약이 의미를 갖는다.
# ⚠️ root `msg_fade6d8ef8c1`: rev19 에서 신규 `arm_link_bounds.py` 가 이 목록에 빠져 있었고
#    렌더러가 그것을 즉시 import 하므로 **mini 실행 사본이 import 실패로 죽었다**.
#    원후보 pin(26)·외부 핀(15)은 원본 폴더를 가리킬 뿐 **실행 사본의 완결성을 보장하지 않는다**.
#    이 목록은 렌더러·실행기의 **local import closure** 와 항상 일치해야 한다
#    (시험 `test_rev20_mini_import_closure.py` 가 AST 로 그 일치를 강제한다).
REVISION_SRC = ("run_production.py", "isaac_replay_w13.py", "arm_link_bounds.py",
                "close_diagnostics.py", "camera_framing.py", "raw_row_identity.py",
                "w13_fk.py", "w13_kinematics.py")
CLOSE_TIMEOUT_EXIT = 97           # 렌더러가 close 시간초과에 쓰는 **전용 비성공 코드**
# root msg_bcb387d139fc (3): 차단 지점 **계측 미충족** 전용 코드. 시간초과(97)와 사유가 다르다.
CLOSE_MEASURE_EXIT = 96


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="W13 원시가 있는 폴더")
ap.add_argument("--out", required=True, help="준비 렌더 산출 상위 폴더(새 경로)")
ap.add_argument("--budget-s", type=float, default=MAX_BUDGET_S, help="총 한도(초). ≤600")
ap.add_argument("--max-frames", type=int, default=MAX_FRAMES, help="총 프레임(합성 포함). 1..8")
ap.add_argument("--synthetic-phases", type=int, default=6,
                help=("운반·배출·HOME 표시 확인용 명시 합성 자세 수. 총 프레임 안에 포함된다. "
                      "**기본 6** — root msg_a2d7b00afa21: 선언된 6 웨이포인트 전부를 써야 "
                      "place_target/HOME 이 빠지지 않는다(기본 2 는 앞 2개만 집어 빠뜨렸다)"))
ap.add_argument("--res", default="1600x900")
ap.add_argument("--fps", type=int, default=10)
ap.add_argument("--usd", default=str(USD_DEFAULT))
ap.add_argument("--criteria", default=None, help="criteria.json 경로(해시를 manifest 에 넣는다)")
ap.add_argument("--runner", default=None,
                help="원본 실행기 경로(기본: 이 파일과 같은 폴더). mini revision 으로 복사된다")
ap.add_argument("--renderer", default=None,
                help="원본 렌더러 경로(기본: 이 파일과 같은 폴더). 시험에서 가짜 렌더로 바꾼다")
ap.add_argument("--render-python", default=str(ISAACLAB_PY),
                help="렌더 단계 인터프리터. 시험에서는 GPU 없는 인터프리터를 준다")
ap.add_argument("--plan-only", action="store_true",
                help="계획/동결만 하고 실행기를 부르지 않는다(GPU 미접촉). HOLD 중에는 이것만 쓴다")
A = ap.parse_args()

run_dir = Path(A.run).resolve()
out_dir = Path(A.out).resolve()
budget = float(A.budget_s)
n_total = int(A.max_frames)
n_synth = int(A.synthetic_phases)
if not (0.0 < budget <= MAX_BUDGET_S):
    raise SystemExit(f"총 한도는 0 < b ≤ {MAX_BUDGET_S} 여야 한다(계약 경계): {budget}")
if not (1 <= n_total <= MAX_FRAMES):
    raise SystemExit(f"준비 시험은 프레임 1..{MAX_FRAMES} 이어야 한다(계약 경계): {n_total}")
if not (0 <= n_synth < n_total):
    raise SystemExit(f"합성 자세 수는 0 ≤ s < 총 프레임 이어야 한다: {n_synth} / {n_total}")
if out_dir.exists() and any(out_dir.iterdir()):
    raise SystemExit(f"기존 산출이 있다(덮어쓰지 않는다): {out_dir}")

src_runner = Path(A.runner).resolve() if A.runner else (HERE / "run_production.py")
src_render = Path(A.renderer).resolve() if A.renderer else (HERE / "isaac_replay_w13.py")
for p in (src_runner, src_render, run_dir):
    if not p.exists():
        raise SystemExit(f"필수 입력이 없다: {p}")

# ── 경로 분리 (root 지적 ②) ────────────────────────────────────────────────
plan_dir = out_dir / "plan"                 # 계획 + mini revision
rev_dir = plan_dir / "revision"
att_dir = out_dir / "attempt"               # 실행기 영수증
render_dir = out_dir / "render"             # **렌더러 전용**. 여기서 만들지 않는다.
(rev_dir / "src").mkdir(parents=True, exist_ok=True)
att_dir.mkdir(parents=True, exist_ok=True)

# ── mini revision: 실행기·렌더러·import 모듈을 **복사하고 실제 해시로 pin** (지적 ①) ──
copied = {}
for name in REVISION_SRC:
    s_ = (src_runner if name == "run_production.py"
          else src_render if name == "isaac_replay_w13.py"
          else HERE / name)
    if not s_.exists():
        raise SystemExit(f"mini revision 에 넣을 모듈이 없다: {s_}")
    shutil.copy2(s_, rev_dir / "src" / name)
    copied[name] = str(s_)

# ── 예산 분배: 전부 **하나의 총 한도 안에서** 쪼갠다. 새 시간을 만들지 않는다 ──
close_budget = min(60.0, budget * 0.15)
grace = min(45.0, budget * 0.10)
child_budget = max(1.0, budget - close_budget - grace)

renderer_argv = [str(A.render_python), str(rev_dir / "src" / "isaac_replay_w13.py"),
                 "--run", str(run_dir), "--out", str(render_dir),
                 "--readiness", "--max-frames", str(n_total),
                 "--synthetic-phases", str(n_synth),          # 지적 ③
                 "--time-budget-s", f"{child_budget:.3f}",
                 "--close-budget-s", f"{close_budget:.3f}",
                 "--res", A.res, "--fps", str(int(A.fps)), "--usd", str(Path(A.usd)),
                 "--headless", "--enable_cameras"]            # readiness_01 실패 사유

commands = {
    "artifact": "W13R_READINESS_COMMANDS_V2",
    "attempt_dir": str(att_dir),
    "prospective_manifest": str(rev_dir / "MANIFEST_prospective.json"),
    "step_order": ["readiness_render"],
    "step_caps_s": {"readiness_render": budget},     # cap 은 종료처리를 **포함한** 총 한도다
    "graceful_grace_s": grace,
    "auto_retry": False,
    "preexisting_attempt_entries_allowed": [],
    "env": {"PYTHONDONTWRITEBYTECODE": "1"},
    "readiness_render": renderer_argv,
}
(rev_dir / "COMMANDS.json").write_text(json.dumps(commands, ensure_ascii=False, indent=1) + "\n")

# 외부 동결 입력 해시를 **실제로** 채운다(실행기가 직접 읽는 키 — 없으면 KeyError).
# root `msg_9a3a29c2edfc`: **실제로 읽는 원본**을 전부 pin 한다 — 원시 JSON/NPZ,
# `_obj` 표면 4개(렌더러가 실제 S1 셸/문 위상을 여기서 읽는다), 원시가 선언한 pile npz, USD.
# 가짜 run_01 을 만들거나 입력을 재생성하지 않는다. 원본을 그 자리에서 읽는다.
ext = {}
for p in sorted(run_dir.glob("w13_cycle_*.json")) + sorted(run_dir.glob("w13_cycle_*.npz")):
    ext[str(p)] = sha(p)
obj_dir = run_dir / "_obj"
n_obj = 0
if obj_dir.is_dir():
    for p in sorted(obj_dir.glob("*.obj")):
        ext[str(p)] = sha(p)
        n_obj += 1
# 렌더러는 원시 JSON 의 inputs_sha256 에서 pile npz 경로를 골라 clump 템플릿을 읽는다.
pile_read = None
_rj = sorted(run_dir.glob("w13_cycle_*.json"))
if _rj:
    try:
        _inp = json.loads(_rj[0].read_text()).get("inputs_sha256") or {}
        _c = [k for k in _inp if k.endswith(".npz")]
        if _c and Path(_c[0]).exists():
            pile_read = _c[0]
            ext[pile_read] = sha(pile_read)
    except Exception:                                                 # noqa: BLE001
        pile_read = None
usd_p = Path(A.usd)
if usd_p.exists():
    ext[str(usd_p)] = sha(usd_p)
# root `msg_f1bc52ebbed4`: 표시 팔 경계에 **실제로 쓰는** URDF/STL 도 전체 SHA 로 pin 해야
# 사용 파일이 추적된다. 렌더러가 읽는 바로 그 파일들을 같은 유도 경로로 얻는다.
try:
    sys.path.insert(0, str(HERE))
    import arm_link_bounds as _ALB                                    # noqa: E402
    for _p, _h in _ALB.display_asset_hashes().items():
        ext[str(_p)] = _h
except Exception as _exc:                                             # noqa: BLE001
    raise SystemExit(f"표시 팔 경계 자산(URDF/STL) 해시를 pin 하지 못했다: {_exc}")
if not ext:
    raise SystemExit(f"외부 동결 입력을 찾지 못했다(해시 계약 미충족): {run_dir}")
if obj_dir.is_dir() and n_obj == 0:
    raise SystemExit(f"_obj 폴더가 있는데 .obj 가 없다 — 실제 S1 위상을 읽을 수 없다: {obj_dir}")
crit = {}
if A.criteria and Path(A.criteria).exists():
    crit[str(Path(A.criteria).resolve())] = sha(Path(A.criteria).resolve())
(rev_dir / "MANIFEST_prospective.json").write_text(json.dumps({
    "artifact": "W13R_READINESS_PROSPECTIVE_V2",
    "written_before_execution": True,
    "revision_dir": str(rev_dir), "attempt_dir": str(att_dir),
    "external_frozen_inputs_sha256": ext,
    "criteria_sha256": crit,
    "planned_outputs": [str(render_dir)],
    "note": "준비 렌더용 mini revision. 해시 계약을 약화하지 않고 정상 동결 구조를 쓴다.",
}, ensure_ascii=False, indent=1) + "\n")

# REVISION_PIN: revision 폴더 안 **모든** 파일의 실제 full SHA256 (빈 dict 금지)
pin_files = {}
for p in sorted(rev_dir.rglob("*")):
    if p.is_file() and p.name != "REVISION_PIN.json" and "__pycache__" not in p.parts:
        pin_files[str(p.relative_to(rev_dir))] = sha(p)
if not pin_files:
    raise SystemExit("REVISION_PIN 이 비었다 — 해시 계약 미충족")
(rev_dir / "REVISION_PIN.json").write_text(json.dumps({
    "artifact": "W13R_READINESS_REVISION_PIN_V2", "revision": "readiness_mini",
    "revision_dir": str(rev_dir), "hash_algorithm": "sha256", "digest_length": 64,
    "n_files": len(pin_files), "frozen_copies_sha256": pin_files,
    "copied_from": copied,
    "rule": "실행 직전 실행기가 이 표를 재계산해 대조한다. runner_self 도 이 안에 있어야 한다.",
}, ensure_ascii=False, indent=1) + "\n")

mini_runner = rev_dir / "src" / "run_production.py"      # **revision 안의 복사본을 부른다**
plan = {
    "artifact": "W13R_READINESS_PLAN_V2",
    "total_budget_s": budget, "contract_max_budget_s": MAX_BUDGET_S,
    "split_inside_total": {"child_render_budget_s": round(child_budget, 3),
                           "renderer_close_budget_s": round(close_budget, 3),
                           "runner_grace_s": round(grace, 3)},
    "split_sums_to_total": bool(abs((child_budget + close_budget + grace) - budget) < 1e-9),
    # root msg_bcb387d139fc (5): 예전 보고는 "child 예산 뒤 외부 TERM" 이라고 적어 **두 시각을
    # 혼동**했다. 자식 `--time-budget-s` 는 렌더러가 **스스로** 렌더를 멈추는 시각이고, 외부
    # 실행기 TERM 은 `cap - grace` 다. 여기에 둘을 따로 적어 다시 섞이지 않게 한다. **예산 상향 없음.**
    "derived_outer_bounds": {
        "child_self_stop_at_s": round(child_budget, 3),
        "outer_runner_term_at_s": round(max(0.0, budget - grace), 3),
        "outer_runner_kill_at_s": round(max(max(0.0, budget - grace),
                                            budget - min(0.5, grace / 2.0)), 3),
        "hard_cap_s": budget,
        "term_formula": "cap - graceful_grace_s (run_production.py term_at)",
        "kill_formula": "max(term_at, cap - min(REAP_OBSERVE_S=0.5, grace/2)) (kill_at)",
        "note": ("child_self_stop_at_s 는 **TERM 시각이 아니다.** 두 값을 같은 것으로 적지 않는다."),
    },
    "inputs_actually_read": {"raw_dir": str(run_dir), "n_obj_surfaces": n_obj,
                            "arm_display_assets_pinned": True,
                            "pile_npz_from_raw_inputs": pile_read, "usd": str(usd_p),
                            "n_external_pinned": len(ext),
                            "note": ("root msg_9a3a29c2edfc: 기존 정본 raw 를 그 자리에서 읽는다. "
                                     "가짜 run_01 생성·복사·입력 재생성 0.")},
    "frames": {"total": n_total, "synthetic": n_synth, "real_raw": n_total - n_synth,
               "synthetic_note": ("명시 라벨된 자세 fixture 다. 운반·배출 위치·HOME 표시를 "
                                  "확인하기 위한 것이며 **물리 결과로 세지 않는다**.")},
    "paths": {"plan_dir": str(plan_dir), "mini_revision": str(rev_dir),
              "attempt_dir": str(att_dir), "render_out": str(render_dir),
              "render_out_created_by": "renderer (배타 소유). 런처는 만들지 않는다."},
    "hash_contract": {"revision_pin": str(rev_dir / "REVISION_PIN.json"),
                      "n_pinned": len(pin_files),
                      "runner_invoked_inside_revision": str(mini_runner),
                      "runner_self_is_pinned": "src/run_production.py" in pin_files,
                      "n_external_inputs_hashed": len(ext),
                      "n_criteria_hashed": len(crit)},
    # root msg_bcb387d139fc (3): 진단 시간과 실제 close 시간을 분리하고 정리 합계도 남긴다.
    "phases_measured": ["startup", "render", "ffmpeg", "close_diagnostic", "close",
                        "cleanup_total"],
    "close_timeout_is_non_success": True,
    "close_timeout_exit_code": CLOSE_TIMEOUT_EXIT,
    "close_measurement_unmet_is_non_success": True,
    "close_measurement_unmet_exit_code": CLOSE_MEASURE_EXIT,
    "outer_bound_is_mandatory": ("SIGALRM 합성 대조는 실제 Isaac C++ 종료의 무조건 보장이 아니다. "
                                 "바깥 실행기의 그룹 TERM→KILL 제한이 최종 경계다."),
    "signal_logic_owner": "run_production.py (런처는 신호 로직을 다시 쓰지 않는다)",
    "argv": [str(ROARM_PY), "-B", str(mini_runner), str(rev_dir)],
    "renderer_argv": renderer_argv,
    "non_claims": ["이 계획은 실행 계약이다. 렌더 성공·육안 검수 통과의 증거가 아니다.",
                   "GPU/렌더 HOLD 중에는 --plan-only 만 쓴다.",
                   "합성 프레임은 표시 점검용이며 물리 결과가 아니다."],
}
plan_path = plan_dir / "readiness_plan.json"
receipt_path = out_dir / "READINESS_RECEIPT.json"
for p in (plan_path, receipt_path):
    if p.exists():
        raise SystemExit(f"기존 영수증이 있다(덮어쓰지 않는다): {p}")
plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")

if A.plan_only:
    print(f"W13R_READINESS_PLAN_ONLY budget={budget}s "
          f"(child {child_budget:.1f} + close {close_budget:.1f} + grace {grace:.1f}) "
          f"frames={n_total}({n_synth} synthetic) pinned={len(pin_files)} ext={len(ext)} "
          f"-> {plan_path}")
    raise SystemExit(0)

# ── 실제 실행: **revision 안의** 실행기에 위임한다(신호·마감 재구현 0) ─────
t0 = time.monotonic()
proc = subprocess.run([str(ROARM_PY), "-B", str(mini_runner), str(rev_dir)], capture_output=False)
elapsed = time.monotonic() - t0

man = render_dir / "render_manifest.json"
rec = att_dir / "EXECUTION_RECEIPT.json"
st = att_dir / "RUN_STATUS.json"
phase, r_manifest = None, None
if man.exists():
    try:
        r_manifest = json.loads(man.read_text())
        phase = r_manifest.get("phase_seconds")
    except Exception:                                                 # noqa: BLE001
        phase = {"unreadable_manifest": True}
step = None
if rec.exists():
    try:
        step = (json.loads(rec.read_text()).get("steps") or [None])[0]
    except Exception:                                                 # noqa: BLE001
        step = None

# root msg_a0752618dd6c ⑤: 필드가 **없으면 false 가 아니라 null(미측정)** 이다.
# 07_actual 에서 렌더러가 close 에 매달려 이 필드를 쓰지 못했고, 그때 내가 false 로 보고한 것은
# 잘못이었다. 누락은 미측정이며 어떤 경우에도 성공을 허용하지 않는다.
close_to = (None if (not phase or "close_timed_out" not in phase)
            else bool(phase.get("close_timed_out")))
receipt = {
    "artifact": "W13R_READINESS_RECEIPT_V2",
    "plan": plan, "runner_returncode": int(proc.returncode),
    "launcher_elapsed_s": round(elapsed, 3),
    "launcher_elapsed_note": ("파이썬 기동을 포함한 바깥 측정값이다. 단계 cap 주장에 쓰지 않는다 — "
                              "단계 경계는 실행기 영수증의 stage_total_s_including_cleanup 이다."),
    "stage_total_s_including_cleanup": (step or {}).get("stage_total_s_including_cleanup"),
    "stage_within_cap_including_cleanup": (step or {}).get("stage_within_cap_including_cleanup"),
    "whole_group_absence_confirmed": (step or {}).get("whole_group_absence_confirmed"),
    "renderer_phase_seconds": phase,
    # 독립 감사 msg_30bb28c44ae5 (유일 결함): 계획은 **6 구간**을 선언하는데 이 검사는 4개만
    # 봐서 `close_diagnostic_s`·`cleanup_total_s` 누락이 조용히 통과했다. 6개 전부 검사한다.
    "phases_all_measured": bool(phase and all(k in phase for k in
                                              ("startup_s", "render_s", "ffmpeg_s",
                                               "close_diagnostic_s", "close_s",
                                               "cleanup_total_s"))),
    "close_timed_out": close_to,
    "close_timed_out_semantics": "true=시간초과 · false=예산 안 종료 확인 · null=미측정(누락). null 을 false 로 읽지 않는다.",
    "renderer_ok": (None if r_manifest is None else r_manifest.get("ok")),
    "renderer_failure_reasons": (None if r_manifest is None else r_manifest.get("failure_reasons")),
    "frames_total_planned": n_total, "frames_synthetic_planned": n_synth,
    "synthetic_is_not_physics": True,
    "honest_gaps": [],
    "non_claims": ["rc 0 은 장면 육안 검수 통과의 증거가 아니다(계약).",
                   "이 영수증은 프로세스 경계·단계 계측·표시 대응만 주장한다.",
                   "합성 프레임은 물리 결과가 아니다."],
}
if phase is None:
    receipt["honest_gaps"].append("renderer manifest 가 없어 단계 계측을 얻지 못했다 — unverified")
if not receipt["phases_all_measured"]:
    receipt["honest_gaps"].append(
        "startup/render/ffmpeg/close_diagnostic/close/cleanup_total 6 구간 중 일부가 "
        "기록되지 않았다 — unverified")
# 지적 ④: close 시간초과는 **전체 워크플로 비성공**이다(화면 품질 판정과 분리).
if close_to is True:
    receipt["honest_gaps"].append(
        "renderer close() 가 예산 안에 끝나지 않았다(close_timed_out=true) — 전체 워크플로 "
        "**비성공**으로 남긴다. 화면 품질과는 별개 문제다.")
elif close_to is None:
    receipt["honest_gaps"].append(
        "renderer close_timed_out 이 **미측정(null)** 이다 — 렌더러가 close 구간에서 기록을 "
        "남기지 못했다는 뜻이며 '시간초과 없음'의 증거가 아니다. 성공으로 취급하지 않는다.")
if int(proc.returncode) == CLOSE_TIMEOUT_EXIT:
    receipt["honest_gaps"].append(
        f"실행기가 close 시간초과 전용 코드 {CLOSE_TIMEOUT_EXIT} 를 보고했다 — 비성공")
if int(proc.returncode) == CLOSE_MEASURE_EXIT:
    receipt["honest_gaps"].append(
        # REFERENCE_RELEASE_CONTRACT_REV2 §6: rc96 을 **probe 미측정으로 단정하지 않는다**.
        # 이 코드는 참조 해제·clear_instance·close 프로브 **어느 계측이든** 미충족이면 난다.
        # 실제 사유는 영수증에서 확인해야 한다. 성공 boolean·runner·timeout 처리는 그대로다.
        f"실행기가 계측 미충족 전용 코드 {CLOSE_MEASURE_EXIT} 를 보고했다 — 비성공. "
        "사유는 하나로 단정하지 않는다: reference_release_receipt.json(필수 대상 생존·"
        "release 반환 mapping 분류) · clear_instance_receipt.json · "
        "close_localization_receipt.json 을 확인할 것. "
        "어느 경우든 '문제가 없었다'는 증거가 아니다.")
# 계측 미충족은 rc 와 **독립적으로도** 잡는다(렌더러가 코드를 못 내보낸 경우 대비).
_cl = (r_manifest or {}).get("close_localization")
if _cl is not None and _cl.get("measurement_satisfied") is not True:
    receipt["honest_gaps"].append(
        f"close 차단 지점 계측 미충족: {_cl.get('measurement_unmet_reasons')} — 비성공")
if r_manifest is not None and _cl is None:
    receipt["honest_gaps"].append(
        "renderer manifest 에 close_localization 이 없다 — **미측정(null)** 이며 "
        "'차단 없음'의 증거가 아니다. 성공으로 취급하지 않는다.")
if (step or {}).get("stage_within_cap_including_cleanup") is False:
    receipt["honest_gaps"].append("단계 전체(정리 포함)가 cap 을 넘었다 — 그 사실을 그대로 남긴다")
if elapsed > budget:
    receipt["honest_gaps"].append(
        f"바깥 전체 {elapsed:.3f}s 가 총 한도 {budget}s 를 넘었다 — 넘은 사실을 그대로 남긴다")
if r_manifest is not None and r_manifest.get("ok") is not True:
    receipt["honest_gaps"].append(
        f"renderer 가 ok 가 아니다: {r_manifest.get('failure_reasons')}")
receipt["ok"] = bool(proc.returncode == 0 and not receipt["honest_gaps"])
receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print(("W13R_READINESS_OK" if receipt["ok"] else "W13R_READINESS_FAIL"),
      f"rc={proc.returncode} elapsed={elapsed:.1f}s close_timed_out={close_to} "
      f"phases={phase} gaps={receipt['honest_gaps']}")
raise SystemExit(0 if receipt["ok"] else 1)
