"""W13 전체 사이클 Isaac **재생 전용** 렌더러 (물리 0 · W12 `sim_isaac_replay_w12.py` 패턴 이식).

무엇을 하는가
    저장된 W13 원시(npz/json)만 읽어 Isaac 장면에서 **표시**한다. 새 입자 해석·물리 적분·접촉 계산 0.
    입자는 PointInstancer 6 프로토타입(재고 6분류 색)으로 프레임마다 protoIndices 를 바꿔 그린다.
    로봇은 저장된 **실측 툴 포즈**에서 오프라인 IK 로 관절을 풀어 `write_joint_state_to_sim` 으로 놓고,
    문은 저장된 **실제 문 쿼터니언**에서 얻은 상대각으로 구동한다. 카메라 RGB → PNG → ffmpeg mp4.

프레임 계약
    저장된 입자 프레임 1개 ↔ 표시 프레임 1개 (1:1, 중간 입자 동역학을 만들지 않는다).
    각 프레임의 원자료 시각·phase·sync index·결정 태그를 오버레이와 매니페스트에 같이 적는다.
    전 phase(initial_home…return_home)와 전이·close/reclose/carry/discharge/return 를 모두 포함한다.

모드
    `--run <dir>`          본 실행 산출 전체 재생(프레임 수 = 원자료 입자 프레임 수).
    `--readiness`          **경계 시험**: 기존 W13 원시 프레임 ≤N + 명시적 합성 phase/pose fixture.
                           산출에 `synthetic=true` 와 프레임별 출처 라벨을 **눈에 띄게** 적는다.
                           전체 사이클 결과로 세지 않는다. 새 입자 해석 0.

주장하지 않는 것
    **규정 로봇 표시는 구동 검증이 아니다**(계약). IK 해가 있음을 보인 표시층이며 서보 토크·실현성 증거가 아니다.
    Isaac 장면의 상자/용기는 원시 기하를 옮겨 그린 **표시용**이고 접촉을 계산하지 않는다.
    BasicWriter·무제한 덤프를 쓰지 않는다.

post05 (W25-B, 2026-09-28) — 공구 표시를 **실제 S1 v1 CAD** 로 바꾼다
    · `/World/s1_cad/fixed`·`/World/s1_cad/door` = S1 v1 STL(`fixed_ALL.stl`·`door_ALL.stl`, link5 mm)을
      매 sync 의 **원시 owner 포즈**(tool_pos/quat = 고정, door_pos/quat = 문)로 놓는다(`cad_pose_math`).
      같은 변환을 충돌 셸에 적용하면 저장된 nodes_F_m/nodes_D_m 를 전 16,813 sync 에서
      ≤ 7.3e-5 mm 로 재현한다(`cad_pose_check.json`). CAD 는 DEME 물리 기하가 아니다(표시 전용).
    · 충돌 셸(`/World/s1/*`)은 `--show-collision-shell` 일 때만 반투명 오버레이로 그린다(기본 off).
    · 트레이·용기는 post04 그대로 원시 기록에서 그린다. 여기에 결과 JSON `fixtures` 와의 **일치 게이트**를 더했다.

post06 (W25-A, 2026-09-29) — DEME(상자) 좌표 → 로봇 좌표 **회전**을 반영한다
    · rev34 는 상자를 로봇 앞에 **돌려** 놓는다(규약 A: x_box = 로봇 −y, y_box = 로봇 +x).
      그 변환을 원자료가 직접 기록한다 — `metadata_json.w25_frame` =
      {R_robot_box, t_robot_m, box_frame_convention, box_anchor} (`sim_w13_full_cycle.py:1324-1327`,
      "p_robot = R_robot_box @ p_box + t_robot (소비자는 추측 금지)").
    · post05 의 표시 변환은 `p + origin_disp` **평행이동뿐**이라 규약 A 원자료를 그리면 상자·더미·공구가
      로봇 대비 90° 틀린다(rev34 `COMMANDS_w25_template.json` step3_isaac_replay.reason).
      post06 은 **표시 변환만** 고친다: 위치 `p_disp = R_robot_box·p_box + origin_disp`,
      자세 `R_disp = R_robot_box·R_box`(입자 쿼터니언은 `q_R ⊗ q_box`).
    · `w25_frame` 키가 **없으면**(W19 A 등 rev32/rev33 원자료) 회전 분기를 타지 않고 post05 와 **같은 식**을
      그대로 쓴다(`ROT is None` 분기). 즉 옛 원자료 재생 결과는 변하지 않는다.
    · 바꾸지 않은 것: 로봇 관절(FK/IK)·문 각도·프레임 1:1 계약·게이트·판정·물리. 회전은 표시층에만 든다.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# post05: 형제 모듈(arm_link_bounds·close_diagnostics·camera_framing·raw_row_identity·w13_fk·w13_kinematics)은
#   post04 동결 rev/src 를 **그대로** 읽는다(사본·수정 0). 새 모듈 cad_pose_math 만 HERE 에 있다.
#   다른 worktree 에 __pycache__ 를 쓰지 않도록 바이트코드 기록을 끈다.
sys.dont_write_bytecode = True
POST04_SRC = Path("/home/cgxr/orca/workspaces/RoArm_Project/w19-replay/claudedocs/runtime_logs/grasp_track/"
                  "w19_runpod_d487/A_full_cycle/replay_20260918/rev/src")
sys.path.insert(0, str(POST04_SRC))
sys.path.insert(0, str(HERE))
import cad_pose_math as CPM                                           # noqa: E402
import arm_link_bounds as ALB                                        # noqa: E402
import close_diagnostics as CD                                        # noqa: E402
import camera_framing as CF                                            # noqa: E402
import raw_row_identity as RRI                                        # noqa: E402

MAIN = Path("/home/cgxr/Documents/Robotics/RoArm_Project")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MAIN))

import w13_fk as FK                                                    # noqa: E402
import w13_kinematics as K                                             # noqa: E402

USD_DEFAULT = MAIN / "local_assets/roarm_m3/usd_s1_v1/roarm_m3_s1_v1.usd"
S1_CAD_DEFAULT = MAIN / "claudedocs/runtime_logs/grab_track/g19_servo_direct/s1_v1"
TOOL_OVERLAY_TEXT = "tool = S1 v1 CAD (posed from raw owner poses) + collision shell (optional)"
ARM_JOINTS = ["base_link_to_link1", "link1_to_link2", "link2_to_link3", "link3_to_link4", "link4_to_link5"]
DOOR_JOINT = "link5_to_gripper_link"
# root msg_a0752618dd6c ③: **좌표·비충돌성은 그대로**, 읽을 수 있는 색/면/라벨만 바꾼다.
SRC_FLOOR_COLOR = (0.62, 0.36, 0.16)   # 주변 회색 바닥과 구분되는 전용 색(표시 전용)
TRAY_FACE_COLOR = (0.25, 0.55, 0.85)   # 실제 벽 면(반투명 유지 — 입자를 가리지 않는다)
TRAY_FACE_OPACITY = 0.30
# root msg_56ff30cd8eeb (1): 바닥 높이 **불투명 갈색 4 경계 바** + 벽 **하단 식별색 경계**.
# 기존 상단 rim·수직 기둥과 함께 벽 면 **범위**를 판별 가능하게 하는 **표시선**이다.
# ⚠️ 이 바들은 **표시선이지 실제 면이 아니다.** 라벨에서 실제 면과 구분해 적는다.
# 벽을 전부 불투명으로 막지 않는다(입자 가림 금지).
EDGE_BAR_T = 0.004                     # 표시선 두께(기존 rim 바 RT 와 같은 규모)
SRC_FLOOR_EDGE_COLOR = (0.62, 0.36, 0.16)   # 바닥 평면 경계선 = source support 와 같은 갈색
WALL_BOTTOM_EDGE_COLOR = (0.10, 0.45, 0.95)  # 벽 하단 경계선 = 식별색(파랑, 면보다 진하게)
SRC_FLOOR_VIS_T = 0.002              # source support 표시 판 두께(표시 전용, 충돌 없음)
CLOSE_TIMEOUT_EXIT = 97              # close 시간초과 전용 비성공 코드(렌더 품질과 분리)
# root msg_bcb387d139fc (3): 프로브 예외·미완료·STOPPED 미관측 = **계측 미충족**.
# 시간초과(97)와 **다른 사유**이므로 코드를 섞지 않는다.
CLOSE_MEASURE_EXIT = 96              # close 차단 지점 계측 미충족 전용 비성공 코드
PLATE_Z = 0.38                       # 로봇 베이스판 z (W9/W12 와 같은 표시 환경)
INV_COLOR = [(0.62, 0.58, 0.50), (0.92, 0.55, 0.16), (0.24, 0.67, 0.35),
             (0.78, 0.24, 0.24), (0.47, 0.78, 0.94), (0.74, 0.74, 0.74)]


def wrap_text_lines(lines, max_w, measure):
    """긴 줄을 **실제 폰트 측정값**으로 재서 감싼다 (root `msg_bcb387d139fc` (1)).

    `measure(text) -> 픽셀 폭`. 가정된 글자폭·문자수 휴리스틱을 쓰지 않는다.
    단어 하나가 `max_w` 보다 길면 글자 단위로 쪼갠다 — **잘라 버리지 않는다**.
    반환: `[(원본 줄 인덱스, 조각 문자열), ...]` — 조각이 어느 줄에서 왔는지 보존하므로
    호출자가 원본 줄 기준으로 색을 정할 수 있다.
    """
    out = []
    for i, ln in enumerate(lines):
        if measure(ln) <= max_w:
            out.append((i, ln))
            continue
        cur = ""
        for word in ln.split(" "):
            cand = word if not cur else cur + " " + word
            if measure(cand) <= max_w:
                cur = cand
                continue
            if cur:
                out.append((i, cur))
                cur = ""
            if measure(word) <= max_w:
                cur = word
                continue
            piece = ""                       # 단어 자체가 한 줄보다 길다 → 글자 단위
            for ch in word:
                if piece and measure(piece + ch) > max_w:
                    out.append((i, piece))
                    piece = ch
                else:
                    piece += ch
            cur = piece
        if cur:
            out.append((i, cur))
    return out


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True, help="W13 원시가 있는 폴더(w13_cycle_seed460.{json,npz})")
parser.add_argument("--out", required=True, help="표시 산출 폴더(새 경로). 기존 산출이 있으면 거부한다")
parser.add_argument("--usd", default=str(USD_DEFAULT))
parser.add_argument("--res", default="1600x900")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--warmup", type=int, default=8)
parser.add_argument("--readiness", action="store_true",
                    help="경계 준비 시험: 기존 원시 프레임 ≤--max-frames + 명시 합성 fixture. 전체 결과 아님")
parser.add_argument("--max-frames", type=int, default=0, help="0 = 전체. readiness 에서는 ≤8 이어야 한다")
parser.add_argument("--synthetic-phases", type=int, default=0,
                    help="readiness 전용: FK 웨이포인트에서 만든 명시 합성 phase/pose 프레임 수")
parser.add_argument("--time-budget-s", type=float, default=0.0, help="0 = 무제한. readiness 는 ≤600")
parser.add_argument("--ik-only", action="store_true", help="Isaac 없이 IK/매핑만 계산하고 끝낸다")
parser.add_argument("--close-budget-s", type=float, default=60.0,
                    help="simulation_app.close() 상한(초). 남은 총 예산이 더 작으면 그쪽이 이긴다. "
                         "초과하면 사실을 영수증에 적고 os._exit 한다 — close() 행으로 매달리지 않는다")
# post05: 공구 표시 = S1 v1 CAD. 충돌 셸은 선택 오버레이(기본 off).
parser.add_argument("--s1-cad-dir", default=str(S1_CAD_DEFAULT),
                    help="S1 v1 CAD 폴더(fixed_ALL.stl·door_ALL.stl, link5 mm)")
parser.add_argument("--show-collision-shell", action="store_true",
                    help="DEME 충돌 셸(_obj 위상 + nodes_F_m/nodes_D_m)을 반투명 오버레이로 함께 그린다")
parser.add_argument("--shell-opacity", type=float, default=0.35, help="충돌 셸 오버레이 불투명도(0..1)")


def tray_display_boxes(box_bounds_m, tray_wall_t_m):
    """post04 와 **같은 식**의 트레이 표시 4 상자(DEME m): 바닥 없는 네 벽, 축정렬."""
    b = np.asarray(box_bounds_m, float)
    tw = float(tray_wall_t_m)
    x0, x1, y0, y1, z0, z1 = b[0, 0], b[0, 1], b[1, 0], b[1, 1], b[2, 0], b[2, 1]
    return {"tray_xn": ((x0 - tw, y0 - tw, z0), (x0, y1 + tw, z1)),
            "tray_xp": ((x1, y0 - tw, z0), (x1 + tw, y1 + tw, z1)),
            "tray_yn": ((x0, y0 - tw, z0), (x1, y0, z1)),
            "tray_yp": ((x0, y1, z0), (x1, y1 + tw, z1))}


def fixtures_display_consistency(fixtures, box_bounds_m, tray_wall_t_m, tray_vertices_m, tray_faces,
                                 bin_vertices_m, bin_pos_npz_m, tol_m=1e-6):
    """post05: 트레이·용기 **표시**가 결과 JSON `fixtures` 가 기록한 기하와 같은지 본다(fail-closed).

    post04 표시 규칙(변경 0): 트레이 = box_bounds_m + tray_wall_t_mm 의 축정렬 4 상자,
    용기 = NPZ bin_vertices_m/bin_faces(세계 좌표 메시) + 목표 마커 = fixtures.bin.pos_m + rim_z_m.
    어느 것도 하드코딩이 아니다. 다만 새 revision 이 트레이를 **축정렬이 아닌** 자세로 기록하면 4 상자 규칙이
    조용히 틀린다 → 기록된 트레이 메시 정점 집합이 4 상자 모서리 32 개와 **같은 집합**인지 확인한다.
    반환: {"checks": {이름: bool}, "values": {...}, "pass": bool, "failures": [이름...]}.
    """
    ft, fb = fixtures.get("tray", {}), fixtures.get("bin", {})
    b = np.asarray(box_bounds_m, float)
    checks, vals = {}, {}
    if tray_vertices_m is None:
        checks["tray_mesh_recorded"] = False
    else:
        tv = np.asarray(tray_vertices_m, float)
        corners = np.concatenate([np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                            for z in (lo[2], hi[2])], float)
                                  for lo, hi in tray_display_boxes(b, tray_wall_t_m).values()], 0)
        d_mesh_to_boxes = np.linalg.norm(tv[:, None, :] - corners[None, :, :], axis=2).min(1).max()
        d_boxes_to_mesh = np.linalg.norm(corners[:, None, :] - tv[None, :, :], axis=2).min(1).max()
        vals["tray_vertex_set_hausdorff_m"] = float(max(d_mesh_to_boxes, d_boxes_to_mesh))
        checks["tray_mesh_recorded"] = True
        checks["tray_mesh_vertex_set_eq_display_box_corners"] = bool(vals["tray_vertex_set_hausdorff_m"] <= tol_m)
        if tray_faces is not None and "n_tri" in ft:
            checks["tray_n_tri_eq_fixtures"] = bool(int(len(tray_faces)) == int(ft["n_tri"]))
    if "top_z_m" in ft:
        vals["tray_top_z_diff_m"] = float(abs(b[2, 1] - float(ft["top_z_m"])))
        checks["tray_top_z_eq_fixtures"] = bool(vals["tray_top_z_diff_m"] <= tol_m)
    else:
        checks["tray_top_z_eq_fixtures"] = False
    if "wall_t_m" in ft:
        vals["tray_wall_t_diff_m"] = float(abs(float(tray_wall_t_m) - float(ft["wall_t_m"])))
        checks["tray_wall_t_eq_fixtures"] = bool(vals["tray_wall_t_diff_m"] <= tol_m)
    else:
        checks["tray_wall_t_eq_fixtures"] = False
    if "pos_m" in fb:
        bp = np.asarray(fb["pos_m"], float)
        bv = np.asarray(bin_vertices_m, float)
        c = (bv.min(0) + bv.max(0)) / 2.0
        vals["bin_mesh_xy_center_vs_fixtures_m"] = float(np.linalg.norm(c[:2] - bp[:2]))
        checks["bin_mesh_xy_center_eq_fixtures_pos"] = bool(vals["bin_mesh_xy_center_vs_fixtures_m"] <= 1e-5)
        if bin_pos_npz_m is not None:
            vals["bin_pos_npz_vs_fixtures_m"] = float(np.abs(np.asarray(bin_pos_npz_m, float) - bp).max())
            checks["bin_pos_npz_eq_fixtures"] = bool(vals["bin_pos_npz_vs_fixtures_m"] <= tol_m)
    else:
        checks["bin_mesh_xy_center_eq_fixtures_pos"] = False
    fails = [k for k, v in checks.items() if not v]
    return {"checks": checks, "values": vals, "pass": not fails, "failures": fails,
            "tol_m": tol_m, "bin_center_tol_m": 1e-5,
            "display_rule": ("tray = post04 4 boxes from box_bounds_m + tray_wall_t_mm; bin = NPZ bin_vertices_m; "
                             "target marker = fixtures.bin.pos_m + rim_z_m (unchanged from post04)")}


def bounded_close(close_fn, budget_s, on_timeout):
    """`close_fn()` 을 **절대 상한** 안에서 부른다. 초과하면 `on_timeout()` 을 부르고 그 결과를 돌려준다.

    알려진 함정(D477 "close() 행"): Isaac 앱 `close()` 가 걸리면 프로세스가 그대로 매달린다.
    SIGALRM(ITIMER_REAL) 로 상한을 걸고, 시간이 지나면 **사실을 기록하고 빠져나온다**.
    실제 탈출(`os._exit`)은 호출자가 `on_timeout` 안에서 한다 — 그래야 이 함수를 시험할 수 있다.

    반환: `{"timed_out": bool, "measured_s": float, "budget_s": float}`.
    """
    budget_s = float(budget_s)
    if budget_s <= 0.0:
        raise ValueError(f"close 예산은 양수여야 한다: {budget_s}")
    box = {"timed_out": False}
    t0 = time.monotonic()

    def _fire(_sig, _frm):
        box["timed_out"] = True
        raise TimeoutError(f"close did not finish within {budget_s}s")

    prev = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    try:
        close_fn()
    except TimeoutError:
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev)
    out = {"timed_out": bool(box["timed_out"]),
           "measured_s": round(time.monotonic() - t0, 3), "budget_s": budget_s}
    if out["timed_out"]:
        on_timeout(out)
    return out


def main():
    # 예산은 **단조 시계**로 잰다(러너 D-runner-3 과 같은 이유 — 벽시계 점프에 흔들리지 않게).
    # 벽시계 UTC 는 보고용으로만 따로 적는다.
    t_start = time.monotonic()
    utc_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    PHASE = {"t0_monotonic": t_start, "started_utc": utc_start}
    from isaaclab.app import AppLauncher                                # noqa: E402
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    out = Path(args.out).resolve()
    W, H = (int(v) for v in args.res.split("x"))
    if args.readiness:
        if not (1 <= int(args.max_frames) <= 8):
            raise SystemExit("readiness 는 --max-frames 1..8 이어야 한다(계약 경계)")
        if not (0.0 < float(args.time_budget_s) <= 600.0):
            raise SystemExit("readiness 는 --time-budget-s 0<b≤600 이어야 한다(계약 경계)")
    man_path = out / "render_manifest.json"
    vid_path = out / "w13_full_cycle.mp4"
    # msg_37ee827485ef ③: manifest/video 만 거부하면 frames/render_plan 은 덮인다.
    # **출력 폴더 전체가 새 것**이어야 한다(배타 소유). ffmpeg 도 -y 로 기존을 덮지 않는다.
    if out.exists():
        existing = sorted(p.name for p in out.iterdir())
        if existing:
            raise SystemExit(f"출력 폴더가 비어 있지 않다(배타 소유 위반, 덮어쓰지 않는다): "
                             f"{out} · 기존 항목 {existing[:12]}")
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir()
    plan_path = out / "render_plan.json"
    for p in (man_path, vid_path, plan_path):
        if p.exists():
            raise SystemExit(f"기존 산출이 있다(덮어쓰지 않는다): {p}")

    res = json.load(open(run / "w13_cycle_seed460.json"))
    z = np.load(run / "w13_cycle_seed460.npz", allow_pickle=True)
    meta = json.loads(str(z["metadata_json"]))
    PHASES = meta["phase_order"]
    INV = [str(v) for v in np.asarray(z["inventory_labels"])]
    P = res["params"]
    ad_info = res["frames"]["adapter"]
    t_robot = np.asarray(ad_info["t_robot_m"], float)
    lip_owner = list(P["lip_l5_mm"])
    box = np.asarray(z["box_bounds_m"], float)
    pf_t = np.asarray(z["particle_frame_t_s"], float)
    pf_s = np.asarray(z["particle_frame_sync_index"], int)
    sync_t = np.asarray(z["sync_t_s"], float)
    ph = np.asarray(z["sync_phase_code"], int)
    sub = [str(v) for v in np.asarray(z["sync_subphase"])]
    tool_p = np.asarray(z["tool_pos_m"], float)
    tool_q = np.asarray(z["tool_quat_xyzw"], float)
    # ── post05: S1 v1 CAD(link5 mm) → owner 로컬(한 번). 배치 상수는 결과 JSON 에서 읽는다 ──
    CAD_FP = CPM.frame_params(res)
    CAD = CPM.load_s1_cad(args.s1_cad_dir)
    cad_local = {"fixed": CPM.owner_local_fixed(CAD["fixed"]["vertices_l5_mm"], CAD_FP["lip_owner_l5_mm"],
                                                CAD_FP["R_W"]),
                 "door": CPM.owner_local_door(CAD["door"]["vertices_l5_mm"], CAD_FP["hinge_l5_mm"],
                                              CAD_FP["q_open_deg"], CAD_FP["R_W"])}
    # ── post05: 트레이·용기 표시 ↔ 결과 JSON fixtures 일치 게이트(표시 기하는 post04 그대로) ──
    FIXTURE_CHECK = fixtures_display_consistency(
        res["fixtures"], z["box_bounds_m"], float(P["tray_wall_t_mm"]) / 1000.0,
        z["tray_vertices_m"] if "tray_vertices_m" in z.files else None,
        z["tray_faces"] if "tray_faces" in z.files else None,
        z["bin_vertices_m"], z["bin_pos_m"] if "bin_pos_m" in z.files else None)
    door_act = np.asarray(z["door_actual_deg"], float)
    nodes_D = np.asarray(z["nodes_D_m"], float)
    d_tags = [str(v) for v in np.asarray(z["decision_tags"])]
    d_fi = np.asarray(z["decision_particle_frame_index"], int)
    d_si = np.asarray(z["decision_sync_index"], int)
    pile_path = [k for k in res["inputs_sha256"] if k.endswith(".npz")][0]
    tpl = json.loads(str(np.load(pile_path, allow_pickle=True)["clump_template_json"]))
    offs = np.asarray(tpl["offsets_m"], float)
    radii = np.asarray(tpl["sphere_radii_m"], float)

    # ── post06: DEME(상자) → 로봇 좌표 회전. 원자료가 적은 값만 쓰고 추측하지 않는다 ──
    # 출처: `metadata_json.w25_frame` (rev34 `sim_w13_full_cycle.py:1324-1327`)
    #       = {"R_robot_box", "t_robot_m", "box_frame_convention", "box_anchor"},
    #       규약 정의는 `w13_fk.py:279-284` BOX_FRAME_CONVENTIONS (A = Rz(−90°), B = Rz(+90°)).
    # 키가 없으면 ROT is None → 아래 모든 헬퍼가 post05 와 **같은 식**을 탄다(옛 원자료 회귀).
    W25F = meta.get("w25_frame")
    ROT = None
    if W25F is not None:
        ROT = np.asarray(W25F["R_robot_box"], float)
        if ROT.shape != (3, 3):
            raise SystemExit(f"w25_frame.R_robot_box 는 3x3 이어야 한다: {ROT.shape}")
        if float(np.abs(ROT @ ROT.T - np.eye(3)).max()) > 1e-12 or abs(float(np.linalg.det(ROT)) - 1.0) > 1e-12:
            raise SystemExit("w25_frame.R_robot_box 가 회전행렬이 아니다(직교·det=+1 위반)")
        # 표시 z 오프셋(PLATE_Z + 어깨 높이)은 **회전 뒤** 더한다. z 축 회전이어야 그 오프셋이 유효하다.
        if (abs(float(ROT[2, 2]) - 1.0) > 1e-12 or float(np.abs(ROT[2, :2]).max()) > 1e-12
                or float(np.abs(ROT[:2, 2]).max()) > 1e-12):
            raise SystemExit("post06 은 z 축 회전(규약 A/B)만 지원한다 — R_robot_box 의 z 행/열이 단위가 아니다")
        # 축정렬 표시 상자(트레이·테두리·경계선)를 축정렬로 유지하려면 R 이 **부호 있는 축 치환**이어야 한다.
        _perm_ok = (np.abs(np.abs(ROT) - np.eye(3)[np.argmax(np.abs(ROT), axis=1)]).max() <= 1e-12
                    and sorted(np.argmax(np.abs(ROT), axis=1).tolist()) == [0, 1, 2])
        if not _perm_ok:
            raise SystemExit("post06 의 축정렬 표시 상자 규칙은 부호 있는 축 치환 R 에만 유효하다")
        # 같은 값을 두 곳이 적는다(결과 JSON frames.adapter.t_robot_m ↔ metadata w25_frame.t_robot_m).
        # 어긋나면 어느 쪽이 맞는지 추측하지 않고 멈춘다.
        _tr_meta = np.asarray(W25F["t_robot_m"], float)
        if float(np.abs(_tr_meta - t_robot).max()) > 0.0:
            raise SystemExit(f"t_robot 이 결과 JSON({t_robot.tolist()}) 과 metadata w25_frame"
                             f"({_tr_meta.tolist()}) 에서 다르다")
        if float(np.abs(ROT - np.eye(3)).max()) == 0.0:
            ROT = None                    # 기록이 단위행렬이면 회전 분기를 타지 않는다(rev32 경로와 동치)

    def _rot_quat_xyzw(R):
        """회전행렬 → 쿼터니언 (x,y,z,w). Shepperd 분기 + 왕복 검증(fail-closed)."""
        R = np.asarray(R, float)
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            t = math.sqrt(tr + 1.0) * 2.0
            q = np.array([(R[2, 1] - R[1, 2]) / t, (R[0, 2] - R[2, 0]) / t,
                          (R[1, 0] - R[0, 1]) / t, 0.25 * t], float)
        else:
            i = int(np.argmax(np.diag(R)))
            j, k = (i + 1) % 3, (i + 2) % 3
            t = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
            q = np.zeros(4, float)
            q[3] = (R[k, j] - R[j, k]) / t
            q[i] = 0.25 * t
            q[j] = (R[j, i] + R[i, j]) / t
            q[k] = (R[k, i] + R[i, k]) / t
        x, y, zq, w = q
        Rb = np.array([[1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
                       [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
                       [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)]], float)
        if float(np.abs(Rb - R).max()) > 1e-12:
            raise SystemExit("R_robot_box → 쿼터니언 왕복 검증 실패")
        return q

    Q_ROT = None if ROT is None else _rot_quat_xyzw(ROT)

    # ── 표시 원점: DEME 세계 → 로봇 세계 + 베이스판 높이 ──
    # 로봇 세계 = URDF 베이스판 기준. 표시용 Isaac world z 는 PLATE_Z 를 더해 바닥 위에 올린다.
    # ⚠️ z 이중 적용 없음: 회전은 z 축 회전이라 (R·p)[2] == p[2] 이고, t_robot[2] 는 **origin_disp 안에서만**
    #    한 번 더해진다. 아래 어떤 헬퍼도 t_robot[2] 를 다시 더하지 않는다(위 z 행/열 단위 검사가 이를 강제).
    origin_disp = np.array([t_robot[0], t_robot[1], t_robot[2] + PLATE_Z + FK.SHOULDER_ABOVE_PLATE], float)

    def deme_to_disp(pw):
        """상자(DEME) 좌표 → 표시 좌표. 끝 축이 3 이면 어떤 모양이든 된다((3,)·(N,3)·(K,N,3))."""
        p = np.asarray(pw, float)
        if ROT is None:
            return p + origin_disp                      # post05 와 **같은 식**
        return p @ ROT.T + origin_disp

    def disp_box(lo, hi):
        """축정렬 표시 상자의 두 모서리를 옮긴다. 회전이 축을 바꾸면 lo/hi 가 뒤집히므로 성분별 min/max 로 되돌린다
        (R 이 부호 있는 축 치환이라 옮긴 상자도 축정렬이다 — 위 `_perm_ok` 가 그 전제를 강제한다)."""
        a, b = deme_to_disp(lo), deme_to_disp(hi)
        return np.minimum(a, b), np.maximum(a, b)

    def disp_quat(q):
        """자세(xyzw) → 표시 좌표: R_disp = R_robot_box · R_box  ⇔  q_disp = q_R ⊗ q_box (Hamilton)."""
        q = np.asarray(q, float)
        if Q_ROT is None:
            return q                                    # post05 와 **같은 배열**
        x1, y1, z1, w1 = (float(v) for v in Q_ROT)
        x2, y2, z2, w2 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                         w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], axis=-1)

    def box_to_robot(pb):
        """상자 좌표 → **로봇** 좌표(표시 z 오프셋 없음). IK/FK 입력은 언제나 로봇 좌표다."""
        p = np.asarray(pb, float)
        if ROT is None:
            return p + t_robot                          # post05 와 **같은 식**
        return p @ ROT.T + t_robot

    def robot_to_box(pr):
        """로봇 좌표 → 상자 좌표(box_to_robot 의 역)."""
        p = np.asarray(pr, float)
        if ROT is None:
            return p - t_robot                          # post05 와 **같은 식**
        return (p - t_robot) @ ROT

    def robot_to_box_rot(Rr):
        """로봇 좌표 자세 → 상자 좌표 자세: R_box = R_box_robot · R_robot = Rᵀ · R_robot
        (rev34 `w13_fk.py:build_adapter_w25` 의 `R_scoop_owner_box = R_box_robot` 과 같은 규칙)."""
        R = np.asarray(Rr, float)
        return R if ROT is None else ROT.T @ R

    # ── 프레임 선택 ────────────────────────────────────────────────────────
    n_raw = int(len(pf_t))
    if args.readiness:
        n_real = max(0, int(args.max_frames) - int(args.synthetic_phases))
        sel = list(range(min(n_real, n_raw)))
    else:
        sel = list(range(n_raw))
        if args.max_frames:
            raise SystemExit("본 실행 재생은 프레임을 자르지 않는다(원자료 입자 프레임과 1:1)")

    # ── 합성 phase/pose fixture (readiness 전용, 명시 라벨) ────────────────
    synth = []
    if args.readiness and args.synthetic_phases:
        r0 = float(res["frames"]["r0_robot_m"])
        wp = {w["name"]: w for w in res["frames"]["waypoints"]}
        names = ["post_lift_travel", "place_retract_base90", "place_target", "place_up_travel",
                 "return_retract_base0", "return_home"][:int(args.synthetic_phases)]
        base_fi = sel[-1] if sel else 0
        for nm in names:
            q5 = wp[nm]["q5"]
            if q5 is None:
                continue
            p_l, R_l = FK.lip_pose(q5, lip_owner)
            synth.append({"name": nm, "q5_deg": [float(v) for v in q5],
                          "lip_robot_m": p_l.tolist(), "R_link5_robot": R_l.tolist(),
                          "door_deg": 0.0, "copy_particles_from_raw_frame": int(base_fi),
                          "SYNTHETIC": True,
                          "label": f"SYNTHETIC POSE FIXTURE ({nm}) — NOT A PHYSICS RESULT"})

    # ── 프레임별 표시 레코드 + 관절 결정 ────────────────────────────────────
    # 명령 모드가 "fk" 였던 sync 는 저장된 q5 가 **그 시점의 실제 관절 상태**다 → IK 를 풀지 않는다.
    # (HOME 처럼 툴이 수직이 아닌 자세는 수직 제약 IK 로 재현할 수 없다 — 억지로 풀면 수백 mm 오차가 난다.)
    # "w11"/"abs" 직교 명령 구간만 립 위치에서 IK 를 푼다(그 구간은 W11 수직 자세라 수직 분기가 맞다).
    cmd_mode = ([str(v) for v in np.asarray(z["sync_cmd_mode"])]
                if "sync_cmd_mode" in z.files else None)
    CART_SUB = {"align_to_w11_scoop_pose", "align_from_w11_scoop_pose", "door_open_at_approach",
                "plunge", "close", "lift", "reclose"}

    def mode_of(si):
        if cmd_mode is not None:
            return cmd_mode[si]
        return "w11" if sub[si] in CART_SUB else "fk"      # 구 원시 호환(모드 배열이 없을 때)

    def joints_for(si):
        """반환 (ik_record, source). 실패하면 (None, reason)."""
        # post06: 저장된 `tool_pos_m` 은 **상자(DEME) 좌표**다 → 로봇 좌표로 옮겨야 IK/FK 가 맞는다.
        lip_robot = box_to_robot(tool_p[si])
        if mode_of(si) == "fk":
            q = [float(v) for v in np.asarray(z["sync_joint_deg"][si], float)]
            p_fk, _ = FK.lip_pose(q, lip_owner)
            return {"q5_deg": q, "branch": "stored_fk_command_joints",
                    "ik_err_mm": round(float(np.linalg.norm(np.asarray(p_fk) - lip_robot)) * 1000, 4),
                    "tilt_deg": None, "limit_violations": FK.in_limits(q),
                    "joint_source": "stored sync_joint_deg (cmd_mode=fk, no IK solved)"}, "stored"
        x, y, zz = [float(v) for v in lip_robot]
        base = math.degrees(math.atan2(y, x))
        sol = FK.solve_fast(float(math.hypot(x, y)), zz, lip_owner)
        if sol is None:
            return None, "ik_unreachable"
        q = list(sol["q5"]); q[0] = base
        p_fk, _ = FK.lip_pose(q, lip_owner)
        return {"q5_deg": [float(v) for v in q], "branch": sol["branch"],
                "ik_err_mm": round(float(np.linalg.norm(np.asarray(p_fk) - lip_robot)) * 1000, 4),
                "tilt_deg": round(float(sol["tilt_deg"]), 4), "limit_violations": FK.in_limits(q),
                "joint_source": f"offline IK from stored actual lip pose (cmd_mode={mode_of(si)})"}, "ik"

    records, ik_fail = [], []
    for k, fi in enumerate(sel):
        si = int(pf_s[fi]) if pf_s[fi] >= 0 else 0
        lip_robot = box_to_robot(tool_p[si])            # post06: 상자 → 로봇
        ik, how = joints_for(si)
        if ik is None:
            ik_fail.append({"frame": int(fi), "lip_robot_m": lip_robot.tolist(), "reason": how})
            continue
        records.append({"display_index": len(records), "source": "raw_particle_frame",
                        "raw_particle_frame_index": int(fi), "source_time_s": float(pf_t[fi]),
                        "sync_index": si, "sync_time_s": float(sync_t[si]),
                        "phase": PHASES[int(ph[si])], "subphase": sub[si],
                        "cmd_mode": mode_of(si), "joint_source": how,
                        "door_actual_deg": float(door_act[si]),
                        "decision_tag": next((d_tags[j] for j in range(len(d_tags)) if int(d_fi[j]) == int(fi)), None),
                        "ik": ik, "SYNTHETIC": False})
    for sy in synth:
        # 합성 fixture 는 **선언된 관절각 그대로** 쓴다(IK 재해석 금지 — 명시 합성임을 흐리지 않는다).
        q = [float(v) for v in sy["q5_deg"]]
        p_fk, _ = FK.lip_pose(q, lip_owner)
        ik = {"q5_deg": q, "branch": "declared_synthetic_waypoint_joints",
              "ik_err_mm": round(float(np.linalg.norm(np.asarray(p_fk)
                                                      - np.asarray(sy["lip_robot_m"], float))) * 1000, 4),
              "tilt_deg": None, "limit_violations": FK.in_limits(q),
              "joint_source": "declared synthetic FK waypoint joints (no IK solved)"}
        records.append({"display_index": len(records), "source": "synthetic_pose_fixture",
                        "raw_particle_frame_index": int(sy["copy_particles_from_raw_frame"]),
                        "source_time_s": None, "sync_index": None, "sync_time_s": None,
                        "phase": sy["name"], "subphase": "SYNTHETIC", "door_actual_deg": sy["door_deg"],
                        "decision_tag": None, "ik": ik, "SYNTHETIC": True, "label": sy["label"],
                        "q5_deg_declared": sy["q5_deg"]})

    # ── 문 각도 → 선형 변위 환산(정정값 계산. 옛 7.7822 mm/° 는 좌표계 혼용 오류로 무효) ──
    si0 = int(pf_s[sel[0]]) if sel else 0
    # post06: 아래 door_r_max 는 **상자 좌표 안의 거리**라 회전 불변이다 → 변환하지 않는다(변경 0).
    R_f0 = K.quat_xyzw_to_mat(tool_q[si0])
    hinge_world = np.asarray(tool_p[si0], float) + R_f0 @ np.asarray(
        res["frames"]["door_hinge_offset_tool_owner_m"], float)
    door_r_max = float(np.linalg.norm(np.asarray(nodes_D[si0], float) - hinge_world, axis=1).max())
    mm_per_deg = 1000.0 * door_r_max * math.pi / 180.0

    manifest = {
        "artifact": "W13R_ISAAC_FULL_CYCLE_RENDER_MANIFEST_V1",
        "readiness_test": bool(args.readiness),
        "synthetic": bool(any(r["SYNTHETIC"] for r in records)),
        "SYNTHETIC_WARNING": ("이 산출에는 명시적 합성 pose fixture 프레임이 있다. 전체 사이클 결과로 세지 않는다."
                              if any(r["SYNTHETIC"] for r in records) else None),
        "run_dir": str(run), "out_dir": str(out),
        "inputs_sha256": {str(run / "w13_cycle_seed460.json"): sha256(run / "w13_cycle_seed460.json"),
                          str(run / "w13_cycle_seed460.npz"): sha256(run / "w13_cycle_seed460.npz"),
                          str(Path(args.usd).resolve()): sha256(Path(args.usd).resolve())
                          if Path(args.usd).exists() else None,
                          str(Path(__file__).resolve()): sha256(Path(__file__).resolve())},
        "n_raw_particle_frames": n_raw, "n_frames": len(records),
        "frames_one_to_one_with_raw": bool(not args.readiness and len(records) == n_raw),
        "source_time_s": [r["source_time_s"] for r in records],
        "phases_rendered": sorted({r["phase"] for r in records}),
        "phase_order_declared": PHASES, "inventory_labels": INV,
        "decision_tags_in_raw": d_tags,
        "decision_frames_covered": [int(v) for v in d_fi if int(v) in set(sel)],
        "overlays": {"source_time_s": True, "phase": True, "fixed_bin": True, "tool": True,
                     "source_marker": True, "target_marker": True, "subphase": True,
                     "door_actual_deg": True, "inventory_counts": True, "decision_tag": True,
                     "synthetic_label": bool(any(r["SYNTHETIC"] for r in records))},
        # root msg_bcb387d139fc (1) — caption 을 scene 밖 별도 여백으로 옮긴 결과를 **명시**한다.
        "caption_band": {
            "drawn_over_scene": False,
            "separate_margin_area_below_camera_tiles": True,
            "occludes_arm_or_any_scene_pixel": False,
            "camera_image_pixel_content_unchanged": True,
            "camera_image_size_unchanged": True,
            "total_output_image_height_increased": True,
            "note": ("검정 caption 패널은 카메라 타일 **아래의 별도 여백 영역**에만 그린다. "
                     "카메라 영상의 픽셀 내용과 크기는 불변이고, 그 대신 **출력 전체 높이가 "
                     "caption 높이만큼 늘어난다**. 프레임별 실측치는 rendered[].caption_geometry."),
            "long_lines_wrapped_by": "PIL 실제 폰트 측정(ImageDraw.textlength) — 가정 폭 아님",
        },
        "transition_support": {"close": "close" in PHASES, "reclose": "reclose" in PHASES,
                              "carry": "transport" in PHASES, "discharge": "discharge" in PHASES,
                              "return": "return_home" in PHASES,
                              "phase_transition_sync_indices": [int(v) for v in
                                                                np.asarray(z["transition_sync_index"], int)]},
        "door_angle_to_displacement": {
            "door_node_max_radius_from_hinge_m": round(door_r_max, 9),
            "mm_per_deg_at_far_node": round(mm_per_deg, 9),
            "formula": "1000 * r_max_from_hinge_m * pi/180 (문 평면 내 far-node 호 길이)",
            "w12_corrected_reference_mm_per_deg": 2.057714892,
            "void_value_not_used": {"mm_per_deg": 7.7822,
                                    "why": "좌표계 혼용 오류 — 인용 금지(W12 diagnostic_correction_01)"}},
        "mapping": {"deme_to_display_origin_m": origin_disp.tolist(),
                    "axes": ("DEME xyz == 로봇 세계 xyz (동결 R_W/FK 결론, 재보정 아님)" if ROT is None else
                             "DEME(상자) xyz ≠ 로봇 xyz — metadata_json.w25_frame.R_robot_box 로 회전해 그린다"),
                    # post06: 회전 영수증. 값은 **원자료 기록**이고 이 스크립트가 고른 것이 아니다.
                    "w25_frame_present": bool(W25F is not None),
                    "w25_frame_from_raw_metadata": W25F,
                    "R_robot_box_applied": (None if ROT is None else ROT.tolist()),
                    "rotation_applied_to": ("none (post05 와 같은 평행이동)" if ROT is None else
                                            "particles(pos+quat), tool/door owner poses & CAD, tray/rim/edge "
                                            "boxes, bin mesh, markers, camera bound set"),
                    "robot_joint_path_unrotated": ("관절은 로봇 좌표량이다. 저장된 상자 좌표 툴 원점만 "
                                                   "box_to_robot 으로 옮겨 IK/FK 에 넣는다."),
                    "z_offset_applied_once": ("R 은 z 축 회전이라 (R·p)[2] == p[2] 이고, t_robot[2] 는 "
                                              "origin_disp 안에서만 한 번 더해진다(이중 적용 없음)."),
                    "plate_z_m": PLATE_Z, "shoulder_above_plate_m": FK.SHOULDER_ABOVE_PLATE,
                    "lip_l5_owner_mm": lip_owner},
        "robot_display_contract": {
            "joints_from": ("cmd_mode=fk sync → 저장된 sync_joint_deg 그대로(IK 안 풂). "
                            "cmd_mode=w11/abs 직교 명령 sync → 저장된 실측 tool_pos_m 에서 오프라인 IK. "
                            "합성 fixture → 선언된 웨이포인트 관절각."),
            "cmd_mode_array_present": bool(cmd_mode is not None),
            "cmd_mode_fallback_rule": (None if cmd_mode is not None else
                                       f"구 원시 호환: subphase in {sorted(CART_SUB)} 이면 직교로 본다"),
            "door_from": "저장된 실제 door_actual_deg (문 쿼터니언에서 얻은 상대각)",
            "NOT_ACTUATION_VERIFICATION": "규정 표시다. 서보 토크·실현성 증거가 아니다.",
            # ── post04 결함 ③ 수정: 관절 출처 집계가 **이중 계수**였다 ────────────────────
            # 옛 식은 `joint_source.startswith(s[:18])` 로 **앞 18 자**만 비교했다.
            #   "offline IK from stored actual lip pose (cmd_mode=abs)"[:18] == "offline IK from st"
            #   "offline IK from stored actual lip pose (cmd_mode=w11)"[:18] == "offline IK from st"
            # 두 범주가 같은 접두사라 서로의 행까지 세서 abs=55·w11=55 로 보고했다(실제 2·53).
            # 합 338 ≠ 283 프레임(AUDIT_02 `isaac_joint_source_counts_exactly_account_for_283_frames`
            # FAIL). 정정: **정확 문자열 일치**로 세고, 합과 레코드 수의 일치를 함께 기록한다.
            "joint_source_counts": {s: sum(1 for r in records if r["ik"]["joint_source"] == s)
                                    for s in sorted({r["ik"]["joint_source"] for r in records})},
            "joint_source_counts_sum": sum(1 for _ in records),
            "joint_source_counts_are_mutually_exclusive_exact_strings": True,
            "n_display_records": len(records),
            "max_lip_reprojection_err_mm": round(max([r["ik"]["ik_err_mm"] for r in records] or [0.0]), 4),
            "max_lip_reprojection_err_mm_meaning": ("표시 관절로 FK 한 립 위치와 저장된 립 위치의 차. "
                                                    "stored-joint 프레임에서는 두꺼워진 owner 립 정의 차이만 남는다."),
            "n_frames_with_limit_violations": sum(1 for r in records if r["ik"]["limit_violations"]),
            "ik_failures": ik_fail},
        "no_basic_writer": True,
        # 🔴 이 값은 **측정 결과로만** 채운다. 아래 finalize 단계에서 프레임별 시계 대조로 확정한다.
        "no_physics_during_displayed_frames": None,
        # 미완 물리는 **반드시** 눈에 보이게 표시한다(계약 D). 조기종료/발산/계획된 중단/합성 전부 포함.
        "incomplete_physics_label": (
            None if (res.get("abort_class") is None and not res.get("diverged")
                     and res.get("stopped_early_after_phase") is None
                     and not res.get("smoke") and res.get("smoke_max_particles") is None
                     and not args.readiness)
            else ("INCOMPLETE PHYSICS / NOT A FULL CYCLE: "
                  f"readiness={bool(args.readiness)} abort_class={res.get('abort_class')} "
                  f"diverged={res.get('diverged')} "
                  f"stopped_early={res.get('stopped_early_after_phase')} "
                  f"smoke={res.get('smoke')} max_particles={res.get('smoke_max_particles')}")),
        "frames": records,
        "non_claims": [
            "규정 로봇 표시는 구동 검증이 아니다.",
            "표시용 상자/용기는 원시 기하를 옮겨 그린 것이며 Isaac 에서 접촉을 계산하지 않는다.",
            "저장된 입자 프레임 사이의 중간 동역학을 만들지 않는다(1:1 표시).",
        ],
        "fixtures_display_consistency": FIXTURE_CHECK,          # post05 게이트(아래 fail_reasons 에 반영)
        "cmd": " ".join(sys.argv),
    }
    if ik_fail:
        manifest["ik_failure_note"] = "IK 실패 프레임은 표시에서 빠졌다 — 프레임 1:1 계약이 깨졌음을 그대로 보고한다."
    json.dump(manifest, open(plan_path, "w"), ensure_ascii=False, indent=2)
    print(f"[w13-isaac] plan: {len(records)} frames (raw {n_raw}, synthetic {len(synth)}) "
          f"max lip reprojection err "
          f"{manifest['robot_display_contract']['max_lip_reprojection_err_mm']} mm", flush=True)
    if args.ik_only:
        json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
        print("W13R_ISAAC_IK_ONLY_OK")
        return 0

    # ── Isaac ──────────────────────────────────────────────────────────────
    # rev25 유일한 변경 (root `msg_2c5384e998df`): 공개 실행 옵션 `fast_shutdown=False`.
    # 설치본 근거 — `simulation_app.py:93` 기본값이 `"fast_shutdown": True` 이고 `:126` 문서가
    # "True to exit process immediately, false to shutdown each extension" 이라, 기본 경로는
    # close 뒤 **파이썬 기록이 실행되기 전에 프로세스를 즉시 종료**한다(14_actual 에서
    # close_s/close_timed_out/cleanup_total_s 가 부재한 이유). False 로 두면 확장별 정상 정리를
    # 요청하므로 **기존** post-close 회계가 돌 수 있다.
    # 경로: `app_launcher.py:59/107` 이 kwargs 를 launcher_args 에 병합 → `:415`
    # `_SIM_APP_CFG_TYPES["fast_shutdown"]=[bool]` → `:497-499` `_sim_app_config` → `:811`
    # `SimulationApp(...)`. `add_app_launcher_args` 는 `fast_shutdown` CLI 인자를 만들지 않으므로
    # `:99-106` 중복 ValueError 는 발생하지 않는다(빈 파서로 실측 확인).
    # ⚠️ 실제 close **반환**은 이것으로 증명되지 않는다 — 실행 시험이 필요하다.
    app_launcher = AppLauncher(args, fast_shutdown=False)
    simulation_app = app_launcher.app
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import Camera, CameraCfg                      # noqa: F401
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sim.spawners.materials import spawn_preview_surface
    from isaaclab.sim.utils import bind_visual_material
    from isaaclab.utils import configclass
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom, Vt
    from scipy.spatial.transform import Rotation
    import omni.usd
    import isaacsim

    vfile = Path(isaacsim.__file__).parent / "VERSION"
    manifest["versions"] = {"isaacsim": vfile.read_text().strip() if vfile.exists() else "?",
                           "torch": torch.__version__, "numpy": np.__version__,
                           "python": sys.version.split()[0]}

    def look_at_quat(pos, tgt):
        f = np.asarray(tgt, float) - np.asarray(pos, float)
        f /= np.linalg.norm(f)
        up = np.array([0, 0, 1.0]) if abs(f[2]) < 0.99 else np.array([1.0, 0, 0])
        y = np.cross(up, f); y /= np.linalg.norm(y); zz = np.cross(f, y)
        x, y_, z_, w = Rotation.from_matrix(np.stack([f, y, zz], 1)).as_quat()
        return (float(w), float(x), float(y_), float(z_))

    src_xy = deme_to_disp([0.0, 0.0, float(np.nanmax(z["heightmap_pre_m"]))])
    bin_pos = deme_to_disp(np.asarray(res["fixtures"]["bin"]["pos_m"], float))
    FOCAL = {"side": 20.0, "top": 20.0}
    APERTURE_H = 20.955
    NEAR_CLIP, FAR_CLIP = 0.05, 8.0
    # 실제 CAM_SIDE/CAM_TOP 은 트레이·테두리 경계가 정의된 뒤 **경계 계산으로** 푼다(아래).

    def cam_cfg(path, pos_tgt, focal):
        return CameraCfg(prim_path=path, update_period=0.0, height=H, width=W, data_types=["rgb"],
                         spawn=sim_utils.PinholeCameraCfg(focal_length=focal, clipping_range=(NEAR_CLIP, FAR_CLIP)),
                         offset=CameraCfg.OffsetCfg(pos=pos_tgt[0], rot=look_at_quat(*pos_tgt),
                                                    convention="world"))

    def static_box(path, lo, hi, color, opacity=1.0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c, s = (lo + hi) / 2, hi - lo
        return AssetBaseCfg(prim_path=path, spawn=sim_utils.CuboidCfg(
            size=tuple(float(v) for v in s), collision_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(float(v) for v in c)))

    tw = float(P["tray_wall_t_mm"]) / 1000.0
    x0, x1, y0, y1, z0, z1 = (float(box[0, 0]), float(box[0, 1]), float(box[1, 0]),
                              float(box[1, 1]), float(box[2, 0]), float(box[2, 1]))
    # post05: 같은 식을 모듈 함수로 옮겼다 — fixtures 일치 게이트가 **그리는 식 그대로**를 검사하게.
    TRAY = tray_display_boxes(box, tw)
    # ⚠️ 표시 전용 테두리 강조. 실제 트레이 벽은 5 mm 라 1 m 넘는 카메라 거리에서 거의 보이지 않았다
    #    (readiness_02 육안 검수 결과). **기하를 바꾸지 않고** 윗단 모서리에 얇은 강조 막대만 덧그린다.
    RT = 0.004
    RIM = {"rim_xn": ((x0 - tw, y0 - tw, z1 - RT), (x0 + RT, y1 + tw, z1)),
           "rim_xp": ((x1 - RT, y0 - tw, z1 - RT), (x1 + tw, y1 + tw, z1)),
           "rim_yn": ((x0, y0 - tw, z1 - RT), (x1, y0 + RT, z1)),
           "rim_yp": ((x0, y1 - RT, z1 - RT), (x1, y1 + tw, z1)),
           "post_a": ((x1 - RT, y0 - tw, z0), (x1 + tw, y0 + RT, z1)),
           "post_b": ((x1 - RT, y1 - RT, z0), (x1 + tw, y1 + tw, z1)),
           "post_c": ((x0 - tw, y0 - tw, z0), (x0 + RT, y0 + RT, z1)),
           "post_d": ((x0 - tw, y1 - RT, z0), (x0 + RT, y1 + tw, z1))}
    # ── 바닥 평면 / 벽 하단 **경계 표시선** (root msg_56ff30cd8eeb (1)) ──────
    # 기존 좌표를 바꾸지 않고, 이미 화면에서 읽히는 것이 확인된 "얇은 불투명 바" 기법만 재사용한다.
    # FLOOR_EDGE = source support 평면(원시 box_bounds 발자국) 둘레, 높이 z0.
    # WALL_BOT   = 트레이 벽 **하단** 둘레(벽 바깥면 tw 만큼 바깥), 높이 z0.
    # 둘은 같은 높이지만 발자국이 tw 만큼 다르므로 나란히 구분된다. 벽 면은 반투명 그대로 둔다.
    EB = EDGE_BAR_T
    FLOOR_EDGE = {"fe_xn": ((x0, y0, z0), (x0 + EB, y1, z0 + EB)),
                  "fe_xp": ((x1 - EB, y0, z0), (x1, y1, z0 + EB)),
                  "fe_yn": ((x0, y0, z0), (x1, y0 + EB, z0 + EB)),
                  "fe_yp": ((x0, y1 - EB, z0), (x1, y1, z0 + EB))}
    WALL_BOT = {"wb_xn": ((x0 - tw, y0 - tw, z0), (x0, y1 + tw, z0 + EB)),
                "wb_xp": ((x1, y0 - tw, z0), (x1 + tw, y1 + tw, z0 + EB)),
                "wb_yn": ((x0, y0 - tw, z0), (x1, y0, z0 + EB)),
                "wb_yp": ((x0, y1, z0), (x1, y1 + tw, z0 + EB))}
    # ── 카메라 프레이밍: **경계 계산**으로 담아야 할 것을 전부 넣는다 ──────
    # 예전 판은 `CAM_DIST = max(1.25, 1.55 * span)` 처럼 두 지점 간격에 곱한 **휴리스틱**이라
    # 무엇이 화면에 들어오는지 보장하지 못했고, readiness_02 육안 검수에서 트레이 벽이 안 보이고
    # 윗화면이 더미를 잘라먹었다. 상수를 키우는 것은 증명이 아니다 →
    # 담을 점집합을 모아 핀홀 FOV 에서 **필요 최소 거리를 닫힌 형태로 풀고 되투영해 확인**한다.
    _fp = []                                    # 표시 좌표계(display) 점들
    for _d in (TRAY, RIM):                      # 트레이 벽 + 표시 테두리/기둥(바닥 z0 포함)
        for _lo, _hi in _d.values():
            # post06: `aabb_corners` 는 두 점의 8 조합을 만들므로 회전으로 lo/hi 가 뒤집혀도 같은 집합이다.
            _fp.append(CF.aabb_corners(deme_to_disp(_lo), deme_to_disp(_hi)))
    # ── source support / DEME domain floor (감사 SOURCE_SUPPORT_DISPLAY_AUDIT_01) ──────
    # 정본 트레이 헬퍼는 트레이를 **바닥 없는 네 벽**으로 정의한다(`w13_kinematics.py`).
    # 따라서 원천 지지면은 **DEME 도메인 경계면 z = box_bounds[2,0]** 이다.
    # 예전 판은 표시 z=0(일반 GroundPlane)을 "지지면"으로 적어 넣었는데 **그것은 로봇 장면의
    # 바닥이고 source floor 가 아니다** — 정본 smoke 기준 DEME z=0 은 표시 z≈0.2188192277 m 에 맞는다.
    # 그래서 원시 x/y 모서리를 DEME z=box_bounds[2,0] 에서 잡아 `deme_to_disp` 로 통과시킨 쿼드를 쓴다.
    SRC_FLOOR_Z_DEME = float(box[2, 0])
    SRC_FLOOR_QUAD_DEME = np.array([[float(box[0, 0]), float(box[1, 0]), SRC_FLOOR_Z_DEME],
                                    [float(box[0, 1]), float(box[1, 0]), SRC_FLOOR_Z_DEME],
                                    [float(box[0, 1]), float(box[1, 1]), SRC_FLOOR_Z_DEME],
                                    [float(box[0, 0]), float(box[1, 1]), SRC_FLOOR_Z_DEME]], float)
    SRC_FLOOR_QUAD_DISP = deme_to_disp(SRC_FLOOR_QUAD_DEME)
    SRC_FLOOR_Z_DISP = float(SRC_FLOOR_QUAD_DISP[0, 2])
    _fp.append(SRC_FLOOR_QUAD_DISP)
    _fp.append(CF.aabb_corners((-0.05, -0.05, 0.0), (0.05, 0.05, PLATE_Z)))   # 받침대 + 베이스판
    _fp.append(np.array([[0.0, 0.0, PLATE_Z + FK.SHOULDER_ABOVE_PLATE]], float))   # 어깨(팔 원점)
    _fp.append(CF.aabb_corners(deme_to_disp(np.asarray(z["bin_vertices_m"], float).min(0)),
                               deme_to_disp(np.asarray(z["bin_vertices_m"], float).max(0))))
    # 팔·도구 실측 궤적: 저장된 모든 sync 의 도구 원점과 S1 노드(고정부/문) 전부
    _fp.append(deme_to_disp(tool_p))
    # `ROOT_READINESS_PREFLIGHT_02` ⑤: 예전엔 S1 노드 AABB 를 **대각 2점**만 넣었다 →
    # 그 상자의 나머지 6 모서리는 담긴다는 보장이 없었다. **8 모서리**를 넣는다.
    for _k in ("nodes_F_m", "nodes_D_m"):
        _n = np.asarray(z[_k], float).reshape(-1, 3)
        _fp.append(CF.aabb_corners(deme_to_disp(_n.min(0)), deme_to_disp(_n.max(0))))
    # post05: S1 v1 CAD 는 충돌 셸보다 크다(스파인·판·암·뺨). 부품별 owner 로컬 AABB 8 모서리를
    # 저장된 **모든 sync** 의 원시 owner 포즈로 옮겨 담는다(강체 변환된 AABB 는 메시를 포함 → 보수적).
    _fp.append(deme_to_disp(CPM.place(CPM.aabb_corners(cad_local["fixed"]), tool_p, tool_q).reshape(-1, 3)))
    _fp.append(deme_to_disp(CPM.place(CPM.aabb_corners(cad_local["door"]), np.asarray(z["door_pos_m"], float),
                                      np.asarray(z["door_quat_xyzw"], float)).reshape(-1, 3)))
    # ── 팔 링크의 **실제 외곽** (root msg_a0752618dd6c ②) ──────────────────
    # 07_actual 육안 검수에서 frame3 측면 상단·frames3/4/5 상면 하단의 팔이 잘렸다.
    # 그때 경계 집합은 어깨→립 **직선 5점 샘플**뿐이라 실제 링크를 포함하지 않았다 —
    # 그래서 당시 `fits=true` 로 그 관측을 반박할 수 없었다(내 claim_scope 가 이미 밝힌 한계).
    # 이제 **임의 반경 상수를 더하지 않고** 설치 URDF 의 visual mesh STL 정점에서 유도한
    # 링크별 local AABB 를 FK 로 옮겨 담는다. 출처(경로·SHA256·삼각형 수)는 영수증에 남긴다.
    _link_bounds = ALB.load_link_local_bounds()
    _chain_prov = ALB.verify_chain_matches_urdf(FK.CHAIN)
    # 감사 `msg_8a86483ae5bb` / root `msg_f1bc52ebbed4`: `FK.CHAIN` 은 반올림 상수(0.05196, PI/2)를
    # 쓰고 원 URDF 는 0.051959 · 1.5708 이다. **표시 경계는 exact XML** 로 만든다.
    # 물리/FK 경로는 그대로 둔다(여기서 FK.CHAIN 은 일치 확인 용도로만 남긴다).
    _disp_chain = ALB.urdf_display_chain()
    _asset_pins = ALB.display_asset_hashes(link_bounds=_link_bounds)
    _arm_pts, _n_arm_poses = [], 0
    for _r in records:
        _q5 = (_r.get("ik") or {}).get("q5_deg")
        if _q5 is None:
            continue
        _w, _ = ALB.link_world_corners_exact_urdf(_q5, _link_bounds, FK.SHOULDER_ABOVE_PLATE,
                                                  _disp_chain)
        if len(_w):
            _arm_pts.append(np.asarray(_w, float) + np.array([0.0, 0.0, PLATE_Z
                                                              + FK.SHOULDER_ABOVE_PLATE]))
            _n_arm_poses += 1
    if _arm_pts:
        _fp.append(np.concatenate(_arm_pts, 0))
    _n_arm_sampled = int(sum(len(a) for a in _arm_pts))
    _synth_in_bounds = [str(_r.get("phase")) for _r in records
                        if _r.get("SYNTHETIC") and (_r.get("ik") or {}).get("q5_deg") is not None]
    _n_synth_in_bounds = len(_synth_in_bounds)
    _synth_declared = [str(_r.get("phase")) for _r in records if _r.get("SYNTHETIC")]
    _synth_missing_from_bounds = [nm for nm in _synth_declared if nm not in _synth_in_bounds]
    # 입자: **표시할 행들**의 구체 바깥면까지. 중심만 보면 더미 윗면이 잘린다.
    _pp = np.asarray(z["particle_pos_m"], float)[sel] if sel else np.asarray(z["particle_pos_m"], float)[:1]
    _rmax = float(np.max(np.linalg.norm(offs, axis=1) + radii))
    _plo = _pp.reshape(-1, 3).min(0) - _rmax
    _phi = _pp.reshape(-1, 3).max(0) + _rmax
    _fp.append(CF.aabb_corners(deme_to_disp(_plo), deme_to_disp(_phi)))
    FRAME_PTS = np.concatenate(_fp, 0)

    _side_dir = np.array([-0.62, 0.62, -0.48], float)       # 눈 → 표적 방향(비스듬 측면)
    _top_dir = np.array([0.0, 1e-6, -1.0], float)           # 거의 수직 하향(up 은 모듈이 보조축으로 교체)
    FRAMING = {
        "side": CF.solve_framing(FRAME_PTS, forward=_side_dir, focal_mm=FOCAL["side"],
                                 aperture_h_mm=APERTURE_H, width_px=W, height_px=H,
                                 margin_frac=0.08, near_clip_m=NEAR_CLIP),
        "top": CF.solve_framing(FRAME_PTS, forward=_top_dir, focal_mm=FOCAL["top"],
                                aperture_h_mm=APERTURE_H, width_px=W, height_px=H,
                                margin_frac=0.08, near_clip_m=NEAR_CLIP),
    }
    CAM_SIDE = (tuple(FRAMING["side"]["eye_m"]), tuple(FRAMING["side"]["target_m"]))
    CAM_TOP = (tuple(FRAMING["top"]["eye_m"]), tuple(FRAMING["top"]["target_m"]))
    # 먼 클리핑이 해답 거리를 못 덮으면 **조용히 자르지 않고** 실패 사유로 남긴다(아래 판정에서).
    FRAMING_FAR_OK = {k: bool(v["distance_m"] + float(np.linalg.norm(
        np.asarray(v["aabb_hi_m"]) - np.asarray(v["aabb_lo_m"]))) <= FAR_CLIP)
        for k, v in FRAMING.items()}

    q0 = records[0]["ik"]["q5_deg"]
    ROBOT = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(args.usd).resolve()),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True,
                                                                      enabled_self_collisions=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, PLATE_Z),
            joint_pos={**{n: math.radians(q0[j]) for j, n in enumerate(ARM_JOINTS)},
                       DOOR_JOINT: math.radians(float(records[0]["door_actual_deg"]))}),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS, stiffness=2000.0,
                                              damping=100.0, effort_limit_sim=50.0),
                   "door": ImplicitActuatorCfg(joint_names_expr=[DOOR_JOINT], stiffness=500.0,
                                               damping=30.0, effort_limit_sim=10.0)})
    TRAYC, BINC, RIMC = (0.78, 0.56, 0.28), (0.85, 0.35, 0.35), (0.45, 0.25, 0.10)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        # 일반 바닥은 **로봇 장면용**으로 남긴다. 이것을 source support 로 식별하지 않는다.
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        # **source support / DEME domain floor** — 표시 전용, 충돌 없음(collision_props=None).
        # 원시 box_bounds x/y × z=box_bounds[2,0] 을 deme_to_disp 로 옮긴 얇은 판이다.
        # DEME 기하·물리·동역학을 바꾸지 않는다(감사 지적대로 표시 누락만 닫는다).
        source_support = static_box(
            "/World/source_support_deme_floor",
            (float(SRC_FLOOR_QUAD_DISP[:, 0].min()), float(SRC_FLOOR_QUAD_DISP[:, 1].min()),
             SRC_FLOOR_Z_DISP - SRC_FLOOR_VIS_T),
            (float(SRC_FLOOR_QUAD_DISP[:, 0].max()), float(SRC_FLOOR_QUAD_DISP[:, 1].max()),
             SRC_FLOOR_Z_DISP),
            SRC_FLOOR_COLOR, 0.95)
        dome = AssetBaseCfg(prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(intensity=1500.0))
        key = AssetBaseCfg(prim_path="/World/key", spawn=sim_utils.DistantLightCfg(intensity=2500.0),
                           init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.3827, 0.0, 0.0)))
        robot: ArticulationCfg = ROBOT
        tray_xn = static_box("/World/tray/xn", *disp_box(TRAY["tray_xn"][0],
                             TRAY["tray_xn"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
        tray_xp = static_box("/World/tray/xp", *disp_box(TRAY["tray_xp"][0],
                             TRAY["tray_xp"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
        tray_yn = static_box("/World/tray/yn", *disp_box(TRAY["tray_yn"][0],
                             TRAY["tray_yn"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
        tray_yp = static_box("/World/tray/yp", *disp_box(TRAY["tray_yp"][0],
                             TRAY["tray_yp"][1]), TRAY_FACE_COLOR, TRAY_FACE_OPACITY)
        rim_xn = static_box("/World/vis/rim_xn", *disp_box(RIM["rim_xn"][0],
                            RIM["rim_xn"][1]), RIMC, 1.0)
        rim_xp = static_box("/World/vis/rim_xp", *disp_box(RIM["rim_xp"][0],
                            RIM["rim_xp"][1]), RIMC, 1.0)
        rim_yn = static_box("/World/vis/rim_yn", *disp_box(RIM["rim_yn"][0],
                            RIM["rim_yn"][1]), RIMC, 1.0)
        rim_yp = static_box("/World/vis/rim_yp", *disp_box(RIM["rim_yp"][0],
                            RIM["rim_yp"][1]), RIMC, 1.0)
        post_a = static_box("/World/vis/post_a", *disp_box(RIM["post_a"][0],
                            RIM["post_a"][1]), RIMC, 1.0)
        post_b = static_box("/World/vis/post_b", *disp_box(RIM["post_b"][0],
                            RIM["post_b"][1]), RIMC, 1.0)
        post_c = static_box("/World/vis/post_c", *disp_box(RIM["post_c"][0],
                            RIM["post_c"][1]), RIMC, 1.0)
        post_d = static_box("/World/vis/post_d", *disp_box(RIM["post_d"][0],
                            RIM["post_d"][1]), RIMC, 1.0)
        # 바닥 평면 경계 표시선 — 불투명 갈색. 더미를 가리지 않는 얇은 바 4개.
        fe_xn = static_box("/World/vis/floor_edge_xn", *disp_box(FLOOR_EDGE["fe_xn"][0],
                           FLOOR_EDGE["fe_xn"][1]), SRC_FLOOR_EDGE_COLOR, 1.0)
        fe_xp = static_box("/World/vis/floor_edge_xp", *disp_box(FLOOR_EDGE["fe_xp"][0],
                           FLOOR_EDGE["fe_xp"][1]), SRC_FLOOR_EDGE_COLOR, 1.0)
        fe_yn = static_box("/World/vis/floor_edge_yn", *disp_box(FLOOR_EDGE["fe_yn"][0],
                           FLOOR_EDGE["fe_yn"][1]), SRC_FLOOR_EDGE_COLOR, 1.0)
        fe_yp = static_box("/World/vis/floor_edge_yp", *disp_box(FLOOR_EDGE["fe_yp"][0],
                           FLOOR_EDGE["fe_yp"][1]), SRC_FLOOR_EDGE_COLOR, 1.0)
        # 벽 하단 식별색 경계 표시선 — 상단 rim·수직 기둥과 합쳐 벽 면 **범위**를 판별하게 한다.
        wb_xn = static_box("/World/vis/wall_bottom_xn", *disp_box(WALL_BOT["wb_xn"][0],
                           WALL_BOT["wb_xn"][1]), WALL_BOTTOM_EDGE_COLOR, 1.0)
        wb_xp = static_box("/World/vis/wall_bottom_xp", *disp_box(WALL_BOT["wb_xp"][0],
                           WALL_BOT["wb_xp"][1]), WALL_BOTTOM_EDGE_COLOR, 1.0)
        wb_yn = static_box("/World/vis/wall_bottom_yn", *disp_box(WALL_BOT["wb_yn"][0],
                           WALL_BOT["wb_yn"][1]), WALL_BOTTOM_EDGE_COLOR, 1.0)
        wb_yp = static_box("/World/vis/wall_bottom_yp", *disp_box(WALL_BOT["wb_yp"][0],
                           WALL_BOT["wb_yp"][1]), WALL_BOTTOM_EDGE_COLOR, 1.0)
        pedestal = static_box("/World/vis/pedestal", (-0.05, -0.05, 0.0), (0.05, 0.05, PLATE_Z),
                              (0.40, 0.40, 0.42), 1.0)
        cam_side: CameraCfg = cam_cfg("/World/CamSide", CAM_SIDE, FOCAL["side"])
        cam_top: CameraCfg = cam_cfg("/World/CamTop", CAM_TOP, FOCAL["top"])

    sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
    stage = omni.usd.get_context().get_stage()

    # 수신 용기 = 원시 mesh 정점/면을 그대로 옮긴 표시 메시(접촉 계산 없음)
    def raw_mesh(path, verts, faces, color):
        v = deme_to_disp(np.asarray(verts, float))
        f = np.asarray(faces, int)
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v, dtype=np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(f))))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(x) for x in f.reshape(-1)]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(x) for x in v.min(0)]),
                                          Gf.Vec3f(*[float(x) for x in v.max(0)])]))
        mat = path + "_mat"
        spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6))
        bind_visual_material(path, mat)
        return {"n_vertices": int(len(v)), "n_faces": int(len(f))}

    # ── 실제 S1 고정 셸 / 문 (readiness_02·03 FAIL 사유였다) ─────────────────
    #    표시용 대체 금지: 원시 `_obj/*.obj` 의 **실제 위상**을 올리고 정점은 매 프레임
    #    저장된 `nodes_F_m`/`nodes_D_m`(엔진이 보고한 실제 메시 노드 세계 좌표)로 갱신한다.
    def _obj_faces(pattern):
        import trimesh
        cands = sorted((run / "_obj").glob(pattern))
        if not cands:
            return None, None
        m = trimesh.load(cands[0], process=False)
        return np.asarray(m.faces, int), str(cands[0])

    def tool_mesh(path, faces, n_pts, color, opacity=1.0):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.zeros((n_pts, 3), np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(faces))))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(x) for x in np.asarray(faces).reshape(-1)]))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)]))
        mat = path + "_mat"
        spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45,
                                                               opacity=float(opacity)))
        bind_visual_material(path, mat)
        return m

    nF_all = np.asarray(z["nodes_F_m"], float)
    nD_all = np.asarray(z["nodes_D_m"], float)
    f_faces, f_obj = _obj_faces("fixed_*.obj")
    d_faces, d_obj = _obj_faces("door_*.obj")
    # post05: 충돌 셸은 **선택 오버레이**(기본 off, 반투명). 공구 표시의 주체는 아래 S1 v1 CAD 다.
    S1 = {}
    if args.show_collision_shell and f_faces is not None and d_faces is not None:
        S1["fixed"] = tool_mesh("/World/s1/fixed", f_faces, nF_all.shape[1], (0.20, 0.72, 0.42),
                                args.shell_opacity)
        S1["door"] = tool_mesh("/World/s1/door", d_faces, nD_all.shape[1], (0.25, 0.50, 0.92),
                               args.shell_opacity)
    # post05: 실제 S1 v1 CAD(표시 전용 · 물리 기하 아님). 정점은 매 프레임 원시 owner 포즈로 놓는다.
    CADP = {"fixed": tool_mesh("/World/s1_cad/fixed", CAD["fixed"]["faces"], len(cad_local["fixed"]),
                               (0.86, 0.86, 0.82)),
            "door": tool_mesh("/World/s1_cad/door", CAD["door"]["faces"], len(cad_local["door"]),
                              (0.95, 0.58, 0.16))}

    door_p_all = np.asarray(z["door_pos_m"], float)
    door_q_all = np.asarray(z["door_quat_xyzw"], float)

    def set_tool(si, synth_q5=None):
        """S1 v1 CAD(필수)와 충돌 셸(선택) 정점을 놓는다.

        · post05 CAD: 그 sync 의 **원시 owner 포즈**(tool_pos/quat · door_pos/quat)로 link5 mm STL 을 놓는다
          (`cad_pose_math.place`). 같은 식이 충돌 셸에서 저장 노드를 ≤ 7.3e-5 mm 로 재현한다.
        · 원시 프레임 셸: 그 sync 의 저장된 노드 좌표를 **그대로** 쓴다(합성·보간 0).
        · 합성 fixture(msg_37ee827485ef ②): 로봇만 움직이고 공구가 직전 sync 에 남는 것을 막기 위해,
          **정준 실제 S1 기하**를 선언된 합성 포즈로 강체 변환한다. 이것은 물리 결과가 아니며
          프레임 오버레이·매니페스트에 합성으로 표시한다.
        """
        if not CADP:
            return False, "no_s1_cad_mesh"
        cad_src = {"fixed": CPM.place(cad_local["fixed"], tool_p[si], tool_q[si]),
                   "door": CPM.place(cad_local["door"], door_p_all[si], door_q_all[si])}
        src = {"fixed": (np.asarray(tool_p[si], float), K.quat_xyzw_to_mat(tool_q[si])),
               "door": (np.asarray(door_p_all[si], float), K.quat_xyzw_to_mat(door_q_all[si]))}
        targets = {k: [(CADP[k], cad_src[k])] + ([(S1[k], (nF_all if k == "fixed" else nD_all)[si])]
                                                  if S1 else [])
                   for k in ("fixed", "door")}
        mode = "cad_posed_from_raw_owner_pose" + ("+shell_raw_nodes" if S1 else "")
        xf = None
        if synth_q5 is not None:
            # 선언된 합성 관절각 → owner 포즈(순수 FK).
            p_l, R_l5 = FK.lip_pose(synth_q5, lip_owner)
            # post06: FK 결과는 **로봇 좌표**다. 아래 조립은 상자 좌표에서 이뤄지고 마지막에 deme_to_disp 를
            #         타므로, 위치·자세 둘 다 로봇 → 상자로 되돌린다(회전 없으면 post05 와 같은 식).
            p_syn = robot_to_box(p_l)
            R_syn = robot_to_box_rot(np.asarray(R_l5, float) @ np.asarray(
                res["frames"]["R_W_frozen_columns_are_link5_axes"], float).T)
            # ⚠️ 감사 `REV17_HOME_DOOR_TRANSFORM_AUDIT_01.json` 이 정량화한 결함 수정.
            #    예전 판은 문 정점을 **저장된 문 프레임**으로 국소화한 뒤 그 프레임을 `R_syn` 으로
            #    겨냥했다. 그런데 정준 문 프레임은 `door_actual_deg≈0` 에서도 툴 대비
            #    **27.499999634°** 회전해 있어서 그 기준 자세가 버려졌다
            #    (문 RMS 46.0890205 mm / max 56.045562 mm, 고정 셸은 3.545e-5 mm 로 정상).
            #    raw 의 door≈0 은 **상대 라벨**이지 `R_door = R_tool` 이 아니다.
            #    → 고정·문 **양쪽**을 **툴 소스 프레임**으로 국소화해 조립 전체의 강체 변환을
            #      보존한다: world = R_syn @ R_tool_src.T @ (nodes_src − p_tool_src) + p_syn.
            #      이 8장은 문이 닫힌 고정 자세라 표시 수정만이며 DEME·제어 변경은 없다.
            #      (문을 실제로 구동하는 미래 경우에는 이 보존된 p_rel/R_rel 위에 의도한
            #       구동각을 얹어야 하며, 지금 그것을 구현하지 않는다.)
            xf = {"fixed": (p_syn, R_syn), "door": (p_syn, R_syn)}
            mode = ("synthetic_declared_pose_rigid_transform_of_canonical_actual_assembly"
                    "_via_tool_source_frame")
        for key in ("fixed", "door"):
            p_ref, R_ref = src[key]
            if xf is not None:
                # 합성: **두 노드 모두 툴 소스 프레임** 기준으로 국소화한다(조립 상대자세 보존).
                p_ref = np.asarray(tool_p[si], float)
                R_ref = K.quat_xyzw_to_mat(tool_q[si])
            for prim, nodes in targets[key]:                 # post05: CAD 와 (선택) 셸에 같은 규칙
                if xf is not None:
                    local = (np.asarray(R_ref, float).T @ (np.asarray(nodes, float) - p_ref).T).T
                    p_t, R_t = xf[key]
                    world = (np.asarray(R_t, float) @ local.T).T + np.asarray(p_t, float)
                else:
                    world = np.asarray(nodes, float)
                v = deme_to_disp(world).astype(np.float32)
                prim.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v)))
                prim.GetExtentAttr().Set(Vt.Vec3fArray(
                    [Gf.Vec3f(*[float(x) for x in v.min(0) - 0.002]),
                     Gf.Vec3f(*[float(x) for x in v.max(0) + 0.002])]))
        return True, mode

    # post05 정정: 이 USD(roarm_m3_s1_v1, UrdfConverter 2026-09-09)의 `gripper_link` 비주얼은 순정 조가 아니라
    #   **S1 v1 문 CAD**(s1_v1_door.stl)이고, `grab_fixed`(link5 고정 결합) 비주얼은 **S1 v1 고정부 CAD**
    #   (s1_v1_fixed.stl = fixed_ALL.stl 동일 sha)다(W25-B usd_prims.md: URDF 원문 + USDC 토큰 해독).
    #   post04 는 'gripper' 필터로 문 CAD 만 숨기고 고정부 CAD 는 남겼다. post05 는 둘 다 숨기고
    #   /World/s1_cad/* 를 **원시 owner 포즈**로 그린다 — 표시 관절(IK)로 놓인 USD 비주얼과 원시 포즈 CAD 는
    #   IK 프레임에서 최대 8.66 mm(고정)/8.76 mm(문) 어긋나 이중으로 보이기 때문이다(checks/usd_visual_vs_raw_pose.json).
    HIDE_TOKENS = ("gripper", "grab_fixed")
    hidden = []
    for prim in stage.Traverse():
        pth = str(prim.GetPath())
        if any(t in pth.lower() for t in HIDE_TOKENS) and UsdGeom.Imageable(prim):
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden.append(pth)
            except Exception:                                            # noqa: BLE001
                pass

    manifest["display_meshes"] = {
        "s1_fixed_shell": {"usd_path": "/World/s1/fixed", "faces_from": f_obj,
                           "n_faces": None if f_faces is None else int(len(f_faces)),
                           "n_points": int(nF_all.shape[1]),
                           "points_from": "nodes_F_m[sync] — 엔진이 보고한 실제 메시 노드 세계 좌표",
                           "present": bool(S1), "requested": bool(args.show_collision_shell),
                           "opacity": float(args.shell_opacity) if S1 else None},
        "s1_door": {"usd_path": "/World/s1/door", "faces_from": d_obj,
                    "n_faces": None if d_faces is None else int(len(d_faces)),
                    "n_points": int(nD_all.shape[1]),
                    "points_from": "nodes_D_m[sync]", "present": bool(S1),
                    "requested": bool(args.show_collision_shell),
                    "opacity": float(args.shell_opacity) if S1 else None},
        # post05: 공구 표시의 주체 = 실제 S1 v1 CAD(표시 전용, DEME 물리 기하 아님)
        "s1_v1_cad": {
            k: {"usd_path": f"/World/s1_cad/{k}", "stl": CAD[k]["path"], "stl_sha256": CAD[k]["sha256"],
                "n_faces": int(len(CAD[k]["faces"])), "n_points": int(len(cad_local[k])),
                "stl_frame": "link5 mm" + (" (door closed, q=0)" if k == "door" else ""),
                "posed_from": ("tool_pos_m/tool_quat_xyzw[sync] (fixed owner)" if k == "fixed"
                               else "door_pos_m/door_quat_xyzw[sync] (door owner)"),
                "owner_local_formula": (
                    "R_W @ (v_l5 - lip_collision_owner_l5_mm)/1000" if k == "fixed" else
                    "R_W @ roty(q_open_joint_deg) @ (v_l5 - hinge_l5_mm)/1000"),
                "present": bool(CADP)} for k in ("fixed", "door")},
        "s1_v1_cad_frame_params_from_result_json": {
            "R_W_frozen_columns_are_link5_axes": CAD_FP["R_W"].tolist(),
            "lip_collision_owner_l5_mm": CAD_FP["lip_owner_l5_mm"].tolist(),
            "hinge_l5_mm": CAD_FP["hinge_l5_mm"].tolist(), "q_open_joint_deg": CAD_FP["q_open_deg"]},
        "s1_v1_cad_pose_check": ("W25-B cad_pose_check.json: 같은 변환을 충돌 셸(half_bowl 재생성)에 적용 → "
                                 "W19 A 16,813 sync 전부에서 nodes_F_m/nodes_D_m 와 최대 7.3e-5 mm"),
        "vendor_gripper_prims_hidden": hidden,
        "vendor_gripper_note": ("post05 정정: 이 USD 의 gripper_link 비주얼 = S1 v1 문 CAD(s1_v1_door.stl), "
                                "grab_fixed 비주얼 = S1 v1 고정부 CAD(s1_v1_fixed.stl). 표시 관절(IK) 자세라 원시 포즈와 "
                                "IK 프레임에서 최대 ~8.7 mm 어긋나므로 둘 다 숨기고 /World/s1_cad/* 를 원시 owner 포즈로 "
                                "그린다. 물리는 언제나 DEME 셸이다."),
        "hide_tokens": list(HIDE_TOKENS),
        "bin": raw_mesh("/World/bin/mesh", z["bin_vertices_m"], z["bin_faces"], BINC),
        "tray": {"drawn_as": "4 axis-aligned boxes from box_bounds_m + tray_wall_t_mm (same spec as tray_mesh)",
                 "wall_t_m": tw,
                 "display_only_rim_highlight": {
                     "reason": ("실제 벽 두께 5 mm 는 1 m 넘는 카메라 거리에서 사실상 보이지 않았다"
                                "(readiness_02 육안 검수). 기하를 바꾸지 않고 윗단 모서리·기둥만 덧그렸다."),
                     "bar_thickness_m": RT, "n_boxes": len(RIM),
                     "NOT_PHYSICS": "표시 전용. DEME 접촉 기하가 아니다."}}}

    # 마커: 취점(source) · 배출(target)
    def marker(path, p, color, r=0.012):
        sp = UsdGeom.Sphere.Define(stage, path)
        sp.CreateRadiusAttr(float(r))
        UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in p]))
        mat = path + "_mat"
        spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.3))
        bind_visual_material(path, mat)

    marker("/World/markers/source", src_xy, (1.0, 0.95, 0.1))
    marker("/World/markers/target", bin_pos + np.array([0.0, 0.0, float(res["fixtures"]["bin"]["rim_z_m"])]),
           (1.0, 0.15, 0.9))

    # 입자: 6 프로토타입(재고 분류 색) PointInstancer
    def proto(path, color):
        import trimesh
        parts = [trimesh.creation.icosphere(subdivisions=1, radius=float(r)).apply_translation(o)
                 for o, r in zip(offs, radii)]
        m = trimesh.util.concatenate(parts)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m.vertices, dtype=np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * int(len(m.faces))))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(v) for v in m.faces.reshape(-1)]))
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(m.vertex_normals,
                                                                           dtype=np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        b = m.bounds
        mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in b[0]]),
                                             Gf.Vec3f(*[float(v) for v in b[1]])]))
        mat = path + "_mat"
        spawn_preview_surface(mat, sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6))
        bind_visual_material(path, mat)
        return int(len(m.vertices))

    pi_path = "/World/pellets/inv"
    pi = UsdGeom.PointInstancer.Define(stage, pi_path)
    proto_paths = []
    for j, nm in enumerate(INV):
        pp = f"{pi_path}/proto_{j}_{nm}"
        proto(pp, INV_COLOR[j % len(INV_COLOR)])
        proto_paths.append(Sdf.Path(pp))
    pi.CreatePrototypesRel().SetTargets(proto_paths)
    n_p = int(np.asarray(z["particle_pos_m"]).shape[1])
    pi.CreateProtoIndicesAttr(Vt.IntArray([0] * n_p))
    pi.CreatePositionsAttr(); pi.CreateOrientationsAttr(); pi.CreateExtentAttr()
    manifest["instancer"] = {"path": pi_path, "n_instances": n_p,
                            "n_prototypes": len(proto_paths), "proto_semantics": INV}

    def set_particles(fi):
        pos = deme_to_disp(np.asarray(z["particle_pos_m"][fi], float))
        # post06: 알은 길쭉한 클럼프라 자세도 함께 돌려야 한다(위치만 돌리면 방향이 어긋난다).
        qt = disp_quat(np.asarray(z["particle_quat_xyzw"][fi], float))
        code = np.asarray(z["inventory_code"][fi], int)
        pi.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(pos, dtype=np.float32)))
        pi.GetOrientationsAttr().Set(Vt.QuathArray([Gf.Quath(float(q[3]), float(q[0]), float(q[1]),
                                                             float(q[2])) for q in qt]))
        pi.GetProtoIndicesAttr().Set(Vt.IntArray([int(c) for c in code]))
        lo, hi = pos.min(0) - 0.01, pos.max(0) + 0.01
        pi.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in lo]),
                                              Gf.Vec3f(*[float(v) for v in hi])]))
        return np.bincount(code, minlength=len(INV)).tolist()

    set_particles(records[0]["raw_particle_frame_index"])
    sim.reset()
    robot: Articulation = scene["robot"]
    cams = {"side": scene["cam_side"], "top": scene["cam_top"]}
    jn = list(robot.joint_names)
    ARM = [jn.index(n) for n in ARM_JOINTS]
    iD = jn.index(DOOR_JOINT)
    manifest["joint_names"] = jn
    manifest["physx_joint_limits_deg"] = {
        n: [round(math.degrees(float(lo)), 3), round(math.degrees(float(hi)), 3)]
        for n, (lo, hi) in zip(jn, robot.data.joint_pos_limits[0].cpu().numpy())}
    manifest["cameras"] = {"side": {"pos": CAM_SIDE[0], "target": CAM_SIDE[1], "focal_mm": FOCAL["side"]},
                          "top": {"pos": CAM_TOP[0], "target": CAM_TOP[1], "focal_mm": FOCAL["top"]},
                          "res": [W, H], "clipping_range_m": [NEAR_CLIP, FAR_CLIP]}
    # 프레이밍은 **계산 영수증**으로 남긴다. 어떤 점이 제약을 묶었는지까지 적는다.
    manifest["camera_framing"] = {
        "method": FRAMING["side"]["method"],
        "n_bound_points": int(len(FRAME_PTS)),
        "n_arm_link_corner_points": int(_n_arm_sampled),
        "n_poses_with_arm_links": int(_n_arm_poses),
        "arm_link_bounds_source": {
            "urdf": _chain_prov["urdf"], "urdf_sha256": _chain_prov["urdf_sha256"],
            "display_transform": ("exact URDF XML: T_parent @ T_origin(xyz,rpy) @ R_axis(q). "
                                  "FK.CHAIN 의 반올림 상수를 쓰지 않는다(물리/FK 경로 불변)."),
            "display_chain": [[n, c] for n, c, _x, _r2, _a in _disp_chain],
            "fk_chain_matches_urdf_reference_only": _chain_prov["all_match"],
            "fk_chain_max_abs_diff_m": max(r["max_abs_diff_m"] for r in _chain_prov["rows"]),
            "asset_sha256_pins": _asset_pins,
            "links": {k: {"mesh": v["mesh"], "mesh_sha256": v["mesh_sha256"],
                          "n_tris": v["n_tris"], "lo_m": v["lo_m"], "hi_m": v["hi_m"]}
                      for k, v in _link_bounds.items()},
            "derivation": ALB.PROVENANCE_NOTE,
            "no_arbitrary_radius_constant": True},
        "n_synthetic_records_in_bound_set": int(_n_synth_in_bounds),
        "synthetic_poses_in_bound_set": list(_synth_in_bounds),
        "synthetic_poses_declared": list(_synth_declared),
        "synthetic_poses_missing_from_bound_set": list(_synth_missing_from_bounds),
        "aabb_uses_eight_corners": True,
        "claim_scope": ("담는다고 주장하는 것은 **이 점집합**이다: 트레이/테두리/용기 AABB 8 모서리, "
                        "**매핑된 source support 쿼드**, 받침대·베이스판, 저장된 모든 sync 의 도구 "
                        "원점, S1 고정부/문 노드 AABB 8 모서리, 표시 행 입자 구체 바깥면, 그리고 "
                        "표시되는 모든 레코드(합성 6 포함)의 **URDF visual mesh 에서 유도한 링크별 "
                        "local AABB 8 모서리**(base_link·link1~link5)를 FK 로 옮긴 점. "
                        "AABB 는 mesh 를 포함하므로 보수적이지만 **mesh 자체는 아니다** — "
                        "'축정렬 상자를 담았다'로만 주장하고 '실제 mesh 외곽을 정확히 담았다'로 "
                        "주장하지 않는다. S1 툴은 raw 노드 경로가 따로 담는다. "
                        "누락은 실제 8프레임 육안 검수에서 검출한다."),
        "includes": ["tray walls (four walls, NO floor — canonical definition)",
                     "display rim/posts",
                     "source support / DEME domain floor quad (mapped, NOT display z=0)",
                     "pedestal + base plate", "arm shoulder origin", "bin mesh AABB",
                     "tool origin over every saved sync", "S1 fixed/door node AABB",
                     "post05: S1 v1 CAD owner-local AABB corners posed at every saved sync",
                     "pile spheres (outer surface) over rendered rows"],
        "far_clip_covers_solution": FRAMING_FAR_OK,
        "source_support": {
            "definition": ("정본 트레이는 바닥 없는 네 벽이므로 원천 지지면은 DEME 도메인 경계면 "
                           "z = box_bounds_m[2,0] 이다(감사 SOURCE_SUPPORT_DISPLAY_AUDIT_01)."),
            "deme_z_m": SRC_FLOOR_Z_DEME,
            "display_z_m": SRC_FLOOR_Z_DISP,
            "quad_corners_deme_m": SRC_FLOOR_QUAD_DEME.tolist(),
            "quad_corners_display_m": SRC_FLOOR_QUAD_DISP.tolist(),
            "prim_path": "/World/source_support_deme_floor",
            "display_only_non_colliding": True,
            "generic_ground_plane_display_z_m": 0.0,
            "generic_ground_is_not_source_support": True,
            "in_framing_bound_set": True,
            "note": ("표시 전용 쿼드다. DEME 기하·물리·동역학 변경 0, 새 물리 요구 0. "
                     "일반 GroundPlane(표시 z=0)은 로봇 장면용이며 source support 가 아니다."),
        },
        "side": FRAMING["side"], "top": FRAMING["top"],
        "non_claims": ["프레이밍 계산이 '들어온다'고 해도 실제 렌더 이미지를 육안 검수한 것은 아니다."]}
    dt = sim.get_physics_dt()

    def set_pose(rec):
        q = torch.zeros(1, robot.num_joints, device=sim.device)
        for j, idx in enumerate(ARM):
            q[0, idx] = math.radians(rec["ik"]["q5_deg"][j])
        q[0, iD] = math.radians(float(rec["door_actual_deg"]))
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()

    try:
        FONT = ImageFont.load_default(size=18)
    except TypeError:
        FONT = ImageFont.load_default()
    # 캔버스를 만들기 **전에** 폭을 재야 caption 높이를 정할 수 있다 → 1x1 측정용 Draw.
    _measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # ── 초기화와 **표시 프레임 적분**을 명확히 분리한다 ────────────────────
    # 🔴 정정: `sim.step(render=True)` 는 **물리를 전진시킨다**(isaaclab 2.3.0
    #    SimulationContext.step 507ff; render 551ff 는 forward 만 하고 playSimulations 를 잠시 끈다).
    #    이전 판이 표시 루프에서 step 을 부르면서 manifest 에 no_physics_during_replay=true 라고
    #    적은 것은 **거짓 주장**이었다(코디네이터 msg_37ee827485ef ①). 표시 프레임에서는 render 만 쓴다.
    set_pose(records[0])
    init_steps = int(args.warmup)
    for _ in range(init_steps):
        sim.step(render=True)          # ← **초기화 구간 전용**. 표시 프레임 밖이며 별도로 기록한다.
        scene.update(dt)

    def sim_clock():
        """표시 프레임에서 물리/시간이 전진하지 않음을 확인할 관측값."""
        out = {}
        for name in ("current_time", "get_physics_dt"):
            try:
                v = getattr(sim, name)
                out[name] = float(v() if callable(v) else v)
            except Exception:                                            # noqa: BLE001
                out[name] = None
        return out

    manifest["initialization_vs_display"] = {
        "initialization_physics_steps": init_steps,
        "initialization_note": ("sim.reset() 과 warmup 은 표시 루프 **밖**에서 한 번만 돈다. "
                                "이 구간은 물리를 전진시킨다는 사실을 숨기지 않는다."),
        "display_frame_call": "sim.render() only — sim.step() 을 부르지 않는다",
        "clock_before_display": sim_clock(),
    }

    budget = float(args.time_budget_s) or None
    PHASE["startup_s"] = round(time.monotonic() - t_start, 3)    # import + 앱 기동 + 장면 + reset/warmup
    rendered = []
    frame_clock_checks = []
    for rec in records:
        if budget is not None and (time.monotonic() - t_start) > budget:
            manifest["time_budget_stop"] = {"after_display_index": rec["display_index"],
                                           "budget_s": budget,
                                           "note": "경계 시험 시간 상한에서 멈췄다(연장하지 않는다)"}
            break
        counts = set_particles(rec["raw_particle_frame_index"])
        # 실제 S1 셸/문. 합성 프레임은 **선언된 합성 포즈로 정준 실제 기하를 변환**한다(공구가 뒤에 남지 않게).
        si_tool = rec["sync_index"] if rec["sync_index"] is not None else int(
            pf_s[rec["raw_particle_frame_index"]])
        tool_drawn, tool_mode = set_tool(
            int(max(0, si_tool)),
            synth_q5=(rec.get("q5_deg_declared") if rec["SYNTHETIC"] else None))
        set_pose(rec)
        # 표시 프레임: **물리 step 금지**. 운동학 상태만 밀어 넣고 렌더만 돌린다.
        clk0 = sim_clock()
        scene.write_data_to_sim()
        sim.render()
        scene.update(dt)               # 센서(카메라) 버퍼 갱신 — 물리 적분이 아니다
        sim.render()
        clk1 = sim_clock()
        if (clk0.get("current_time") is not None
                and clk1.get("current_time") is not None
                and clk1["current_time"] != clk0["current_time"]):
            raise SystemExit(f"표시 프레임에서 물리 시간이 전진했다(계약 위반): "
                             f"{clk0['current_time']} → {clk1['current_time']}")
        frame_clock_checks.append({"display_index": rec["display_index"],
                                   "t_before": clk0.get("current_time"),
                                   "t_after": clk1.get("current_time"),
                                   "advanced": bool(clk0.get("current_time") is not None
                                                    and clk1.get("current_time") is not None
                                                    and clk1["current_time"] != clk0["current_time"])})
        tiles = []
        for nm in ("side", "top"):
            rgb = cams[nm].data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            tiles.append(Image.fromarray(rgb))
        # 라벨을 **먼저** 확정해야 caption 영역 높이를 정할 수 있다(캔버스는 그 뒤에 만든다).
        lines = [f"W13 full cycle | frame {rec['display_index'] + 1}/{len(records)}",
                 f"phase {rec['phase']} / {rec['subphase']}",
                 ("source t = SYNTHETIC (no raw time)" if rec["source_time_s"] is None
                  else f"source t = {rec['source_time_s']:.6f} s  (raw particle frame "
                       f"{rec['raw_particle_frame_index']}, sync {rec['sync_index']})"),
                 f"door actual {rec['door_actual_deg']:.4f} deg  |  ik err {rec['ik']['ik_err_mm']} mm",
                 "inv " + " ".join(f"{INV[j]}={counts[j]}" for j in range(len(INV))),
                 "yellow marker = scoop source   magenta marker = fixed discharge bin"]
        # root msg_56ff30cd8eeb (1): **실제 면**과 **표시선**을 라벨에서 구분한다.
        lines.append(f"SURFACES (actual geometry): brown slab = source support / DEME domain floor "
                     f"(DEME z={SRC_FLOOR_Z_DEME:.3f} -> display z={SRC_FLOOR_Z_DISP:.5f} m, "
                     f"mostly hidden under the pile); translucent blue = tray 4 wall faces "
                     f"(kept see-through so particles are not occluded)")
        lines.append("DISPLAY LINES (not surfaces): brown bars = source-support plane outline; "
                     "blue bars = wall BOTTOM edge; gold bars/posts = wall TOP rim + verticals "
                     "-> together they bound the wall face extent. grey ground z=0 is NOT the "
                     "source support.")
        lines.append((f"{TOOL_OVERLAY_TEXT} [shell {'ON translucent' if S1 else 'off'}; {tool_mode}; "
                      f"vendor gripper hidden]"
                      if tool_drawn else ">>> WARNING: S1 v1 CAD TOOL MESH NOT DRAWN <<<"))
        if rec["decision_tag"]:
            lines.append(f"DECISION {rec['decision_tag']}")
        if rec["SYNTHETIC"]:
            lines.append(">>> SYNTHETIC POSE FIXTURE — NOT A PHYSICS RESULT <<<")
        if manifest["incomplete_physics_label"]:
            lines.append(">>> " + manifest["incomplete_physics_label"] + " <<<")
        # root msg_bcb387d139fc (1): 검정 패널을 **기존 scene 위에 그리지 않는다.**
        #   · 긴 줄은 **PIL 실제 폰트 측정값**으로 재서 wrap 한다(가정 폭 금지).
        #   · caption 은 카메라 타일 **아래의 별도 여백 영역**에만 그린다 → 팔 상단을 가리지 않는다.
        #   · 카메라 영상의 **픽셀 내용·크기는 불변**이다(타일을 그대로 paste, 변형·축소 없음).
        #   · 대신 **출력 전체 높이가 caption 만큼 늘어난다** — manifest 에 명시한다.
        _pad, _lh, _mx = 8, 22, 14
        _sw = tiles[0].width
        _sh = tiles[0].height + tiles[1].height          # scene 영역(= 이전 판의 전체 높이)
        wrapped = wrap_text_lines(lines, _sw - 2 * _mx,
                                  lambda s2: _measure.textlength(s2, font=FONT))
        _cap_h = _lh * len(wrapped) + 2 * _pad
        img = Image.new("RGB", (_sw, _sh + _cap_h))
        img.paste(tiles[0], (0, 0))
        img.paste(tiles[1], (0, tiles[0].height))        # ← 타일 픽셀 원본 그대로
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, _sh, _sw - 1, _sh + _cap_h - 1], fill=(0, 0, 0))   # caption 여백만 검정
        for k2, (_src, s2) in enumerate(wrapped):
            _hl = (">>>" in lines[_src]) or ("DECISION" in lines[_src])     # 색은 **원본 줄** 기준
            dr.text((_mx, _sh + _pad + _lh * k2), s2,
                    fill=(255, 255, 0) if _hl else (255, 255, 255), font=FONT)
        caption_geom = {"scene_w": int(_sw), "scene_h": int(_sh), "caption_h": int(_cap_h),
                        "output_h": int(_sh + _cap_h), "wrapped_lines": len(wrapped),
                        "source_lines": len(lines)}
        fp = frames_dir / f"f_{rec['display_index']:05d}.png"
        img.save(fp)
        rendered.append({"display_index": rec["display_index"], "png": str(fp),
                        "inventory_counts": counts, "s1_tool_drawn": bool(tool_drawn),
                        "s1_tool_mode": tool_mode,
                        "s1_tool_from_sync_index": int(max(0, si_tool)),
                        "caption_geometry": caption_geom})
        if rec["display_index"] % 10 == 0:
            print(f"[w13-isaac] frame {rec['display_index'] + 1}/{len(records)} "
                  f"w={time.monotonic()-t_start:.0f}s", flush=True)

    manifest["rendered"] = rendered
    manifest["n_frames_rendered"] = len(rendered)
    manifest["wall_seconds"] = round(time.monotonic() - t_start, 2)
    manifest["s1_tool_drawn_every_frame"] = bool(CADP) and all(
        r.get("s1_tool_drawn") for r in rendered) if rendered else False
    # 표시 프레임 물리 비전진은 **주장이 아니라 측정**이다.
    manifest["initialization_vs_display"]["clock_after_display"] = sim_clock()
    manifest["initialization_vs_display"]["per_frame_clock_checks"] = frame_clock_checks
    manifest["no_physics_during_displayed_frames"] = bool(
        frame_clock_checks and not any(c["advanced"] for c in frame_clock_checks)
        and all(c["t_before"] is not None for c in frame_clock_checks))
    manifest["no_physics_evidence"] = (
        "표시 루프에서 sim.step() 을 호출하지 않고, 프레임마다 sim.current_time 을 전후 비교해 "
        "전진 0 을 확인했다. 초기화(sim.reset + warmup)는 표시 루프 밖이며 물리를 전진시킨다는 사실을 "
        "initialization_vs_display 에 따로 적는다. isaaclab 2.3.0 SimulationContext.step(507ff) 이 "
        "물리를 전진시킨다는 코디네이터 지적(msg_37ee827485ef ①)에 따른 정정이다.")

    # ── 성공 판정은 **필수 조건을 전부** 본다(msg_37ee827485ef ④) ───────────
    required_frames = len(records)
    fail_reasons = []
    if not CADP:
        fail_reasons.append("S1 v1 CAD fixed/door meshes were not created")
    if args.show_collision_shell and not S1:
        fail_reasons.append("collision shell overlay requested but run _obj topology is missing")
    if not manifest["s1_tool_drawn_every_frame"]:
        fail_reasons.append("S1 v1 CAD tool was not drawn on every rendered frame")
    # post05: USD 의 S1 v1 비주얼(표시 관절 자세)이 남으면 원시 포즈 CAD 와 이중으로 보인다 → 숨김 확인을 게이트로.
    _hd = manifest["display_meshes"]["vendor_gripper_prims_hidden"]
    for _tok in ("gripper_link", "grab_fixed"):
        if not any(_tok in h for h in _hd):
            fail_reasons.append(f"USD S1 visual prim '{_tok}' was not hidden (would double-draw with /World/s1_cad)")
    if not FIXTURE_CHECK["pass"]:
        fail_reasons.append(f"tray/bin display disagrees with result JSON fixtures: {FIXTURE_CHECK['failures']}")
    if len(rendered) != required_frames:
        fail_reasons.append(f"rendered {len(rendered)} of {required_frames} planned frames")
    if ik_fail:
        fail_reasons.append(f"{len(ik_fail)} frame(s) had no joint solution")
    if manifest.get("time_budget_stop"):
        fail_reasons.append("stopped at the time budget before finishing the planned frames")
    if manifest["no_physics_during_displayed_frames"] is not True:
        fail_reasons.append("displayed-frame physics non-advance was not demonstrated")
    # 프레이밍은 **게이트다**. 담아야 할 것이 화면 밖이면 장면 준비 실패로 본다(육안 검수 지적 반영).
    if _synth_missing_from_bounds:
        fail_reasons.append(f"synthetic display poses missing from the camera bound set: "
                            f"{_synth_missing_from_bounds} — cannot claim they are framed")
    for _cn, _cv in FRAMING.items():
        if not _cv["fits"]:
            fail_reasons.append(f"camera '{_cn}' framing does not contain the bound set "
                                f"(max screen frac u={_cv['max_screen_frac_u']:.3f} "
                                f"v={_cv['max_screen_frac_v']:.3f})")
        if not FRAMING_FAR_OK[_cn]:
            fail_reasons.append(f"camera '{_cn}' far clip {FAR_CLIP} m does not cover the solved "
                                f"distance {_cv['distance_m']:.3f} m plus the scene diagonal")
    if not args.readiness and not manifest.get("frames_one_to_one_with_raw"):
        fail_reasons.append("production replay is not 1:1 with saved particle frames")
    manifest["failure_reasons"] = fail_reasons
    manifest["ok"] = not fail_reasons

    # ── visual_mapping.json (RAW_SCHEMA_REQUIRED + ERRATUM_01) ──────────────
    #    본 실행: **저장된 모든 입자 프레임이 정확히 한 줄씩**. 8-프레임 상한은 preflight fixture 전용.
    real = [r for r in records[:len(rendered)] if not r["SYNTHETIC"]]
    vm_rows = [{"row": i, "particle_frame_row": int(r["raw_particle_frame_index"]),
                "source_sync_index": int(r["sync_index"]),
                "source_time_s": float(r["source_time_s"]),
                "source_phase_code": int(ph[int(r["sync_index"])]),
                "display_index": int(r["display_index"])}
               for i, r in enumerate(real)]
    vm = {
        "artifact": "W13R_VISUAL_MAPPING_V1",
        "readiness_test": bool(args.readiness),
        "rule": ("production: 저장된 모든 입자 프레임이 정확히 한 줄. 중복·누락·지어낸 시간 금지. "
                 "≤8 상한은 preflight readiness fixture 에만 적용된다(RAW_SCHEMA_REQUIRED_ERRATUM_01)."),
        "n_raw_particle_frames": int(n_raw), "n_mapped_rows": len(vm_rows),
        "covers_every_saved_particle_frame": bool(len(vm_rows) == n_raw and not args.readiness),
        "particle_frame_row": [r["particle_frame_row"] for r in vm_rows],
        "source_sync_index": [r["source_sync_index"] for r in vm_rows],
        "source_time_s": [r["source_time_s"] for r in vm_rows],
        "source_phase_code": [r["source_phase_code"] for r in vm_rows],
        "equals_raw_arrays": {
            "source_sync_index == particle_frame_sync_index": bool(
                len(vm_rows) == n_raw and all(vm_rows[i]["source_sync_index"] == int(pf_s[i])
                                              for i in range(len(vm_rows)))),
            "source_time_s == particle_frame_t_s": bool(
                len(vm_rows) == n_raw and all(vm_rows[i]["source_time_s"] == float(pf_t[i])
                                              for i in range(len(vm_rows)))),
            "source_phase_code == sync_phase_code[particle_frame_sync_index]": bool(
                all(vm_rows[i]["source_phase_code"] == int(ph[int(pf_s[i])])
                    for i in range(len(vm_rows))))},
        # ERRATUM_03 행 정체성 — 판정은 공유 정본 `raw_row_identity` 가 한다(사본 금지).
        # readiness fixture 는 ≤8 상한이라 전수 대응만 면제되고, 순서·중복 위반은 그대로 거절된다.
        "row_identity": RRI.row_identity_report(
            particle_frame_row=[r["particle_frame_row"] for r in vm_rows],
            particle_frame_sync_index=pf_s, particle_frame_t_s=pf_t, sync_phase_code=ph,
            source_sync_index=[r["source_sync_index"] for r in vm_rows],
            source_time_s=[r["source_time_s"] for r in vm_rows],
            source_phase_code=[r["source_phase_code"] for r in vm_rows],
            decision_particle_frame_index=np.asarray(z["decision_particle_frame_index"], int),
            n_raw=int(n_raw), require_full_coverage=not bool(args.readiness)),
        "phases_covered": sorted({PHASES[r["source_phase_code"]] for r in vm_rows}),
        "all_phases_covered": bool({PHASES[r["source_phase_code"]] for r in vm_rows} == set(PHASES)),
        "transition_sync_indices": [int(v) for v in np.asarray(z["transition_sync_index"], int)],
        "scene_entities": ["source", "robot", "tool", "bin", "target", "actual"],
        "scene_entity_paths": {"source": "/World/pellets/inv + /World/markers/source",
                               "robot": "/World/envs/env_0/Robot",
                               "tool": ("/World/s1_cad/fixed + /World/s1_cad/door (S1 v1 CAD posed from raw owner "
                                        "poses, not vendor gripper); collision shell /World/s1/* only with "
                                        "--show-collision-shell"),
                               "bin": "/World/bin/mesh", "target": "/World/markers/target",
                               "actual": "저장된 실측 포즈/노드로만 그린다(합성 보간 없음)"},
        "synthetic_rows_excluded": int(len(rendered) - len(vm_rows)),
        "rows": vm_rows,
        "non_claims": ["표시 매핑은 기록 대응이며 배출 성공·구동 가능성의 증거가 아니다."],
    }
    # ERRATUM_03: 행 정체성은 **게이트다**. readiness fixture 는 ≤8 상한 때문에 전수 대응이
    # 아니므로 arange 전수 조건은 본 실행에만 적용하고, 순서·중복 위반은 **양쪽 모두** 거절한다.
    _ri = vm["row_identity"]
    fail_reasons.extend(f"visual_mapping row identity: {m}" for m in _ri["failures"])
    manifest["failure_reasons"] = fail_reasons          # 위 1차 판정에 행 정체성 결과를 합친다
    manifest["ok"] = not fail_reasons
    json.dump(vm, open(out / "visual_mapping.json", "w"), ensure_ascii=False, indent=2)
    manifest["visual_mapping"] = str(out / "visual_mapping.json")
    manifest["row_identity_gate"] = {k: _ri[k] for k in
                                     ("all_ok", "failures", "is_exact_arange", "is_strictly_increasing",
                                      "duplicated_rows", "missing_rows", "sync_index_nondecreasing",
                                      "rows_sharing_sync_with_previous", "n_same_sync_rows_preserved",
                                      "rows_deduplicated", "full_coverage_required")}
    # root msg_dc9b5fbc2940: caption 높이를 프레임마다 `len(wrapped)` 로 정하면 raw/합성/DECISION
    # 줄 수 차이로 **PNG 높이가 달라진다**. MP4 는 전 프레임 동일 치수를 요구하므로, 인코딩
    # **전에** 선택된 전 프레임의 **최대 측정 높이**를 공통 caption 으로 예약한다.
    #   · 텍스트는 **삭제·절단하지 않는다**(모자라는 프레임만 아래쪽에 검정 여백을 더한다).
    #   · 원 scene 픽셀은 건드리지 않는다 — 추가분은 전부 caption 여백 아래쪽이다.
    _caps = [r["caption_geometry"] for r in rendered]
    _common_h = max((c["output_h"] for c in _caps), default=0)
    _padded = []
    for r in rendered:
        c = r["caption_geometry"]
        c["common_output_h"] = int(_common_h)
        c["pad_added_px"] = int(_common_h - c["output_h"])
        if c["pad_added_px"] <= 0:
            continue
        _im = Image.open(r["png"])
        _nw = Image.new("RGB", (_im.width, _common_h))      # 기본 검정 = caption 여백과 같은 색
        _nw.paste(_im, (0, 0))
        _im.close()
        _nw.save(r["png"])
        _padded.append({"png": r["png"], "pad_added_px": c["pad_added_px"]})
    manifest["caption_common_height"] = {
        "common_output_h_px": int(_common_h),
        "scene_h_px": (_caps[0]["scene_h"] if _caps else None),
        "distinct_output_h_before_pad": sorted({c["output_h"] for c in _caps}),
        "n_frames_padded": len(_padded),
        "padded_frames": _padded,
        "all_frames_same_height_after_pad": bool(
            _caps and all(c["output_h"] + c["pad_added_px"] == _common_h for c in _caps)),
        "text_truncated": False,
        "scene_pixels_modified": False,
        "why": ("MP4 는 전 프레임 동일 치수를 요구한다. 프레임별 라벨 줄 수가 달라도 "
                "**공통 caption 높이**를 예약해 치수를 맞춘다. 텍스트는 자르지 않는다."),
    }
    if _caps and not manifest["caption_common_height"]["all_frames_same_height_after_pad"]:
        fail_reasons.append(
            "caption 공통 높이 패딩 후에도 프레임 치수가 일치하지 않는다 — MP4 인코딩 전제 위반")
        manifest["ok"] = False
    PHASE["render_s"] = round(time.monotonic() - t_start - PHASE["startup_s"], 3)
    _t_ff = time.monotonic()
    if rendered:
        subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", str(frames_dir / "f_%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "20", str(vid_path)], check=True)
        manifest["video"] = {"path": str(vid_path), "bytes": vid_path.stat().st_size, "fps": args.fps,
                            "n_frames": len(rendered)}
        manifest["n_frames"] = len(rendered)
        manifest["source_time_s"] = [r["source_time_s"] for r in records[:len(rendered)]]
    PHASE["ffmpeg_s"] = round(time.monotonic() - _t_ff, 3)
    # ── 종료를 **무한 대기로 두지 않는다** ──────────────────────────────────
    # 알려진 함정: Isaac 앱 `close()` 가 걸리면 프로세스가 그대로 매달린다(D477 "close() 행").
    # manifest 는 close **전에** 쓰므로 close 가 멈춰도 영수증은 남는다. 그 위에 SIGALRM 으로
    # 절대 상한을 걸고, 시간이 지나면 사실을 적고 `os._exit` 로 빠져나온다(조용히 매달리지 않는다).
    _remain = (float(args.time_budget_s) - (time.monotonic() - t_start)) if args.time_budget_s else None
    close_budget = float(args.close_budget_s)
    if _remain is not None:
        close_budget = max(1.0, min(close_budget, _remain))
    PHASE["close_budget_s"] = round(close_budget, 3)
    PHASE["total_before_close_s"] = round(time.monotonic() - t_start, 3)
    manifest["phase_seconds"] = dict(PHASE)
    manifest["budget_clock"] = "time.monotonic (벽시계 점프 비의존). started_utc 는 보고용."
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
    tag_ = "READINESS" if args.readiness else "FULL"
    if manifest["ok"]:
        print(f"W13R_ISAAC_RENDER_{tag_}_OK frames={len(rendered)} "
              f"wall={manifest['wall_seconds']}s -> {out}")
    else:
        print(f"W13R_ISAAC_RENDER_{tag_}_FAIL frames={len(rendered)} "
              f"wall={manifest['wall_seconds']}s reasons={fail_reasons} -> {out}")
    rc_ = 0 if manifest["ok"] else 1

    def _on_close_timeout(info):
        PHASE["close_s"] = info["measured_s"]
        PHASE["close_timed_out"] = True
        # root msg_bcb387d139fc (3) + msg_dc9b5fbc2940: 시간초과로 나가는 경로에서도 정리 합계를
        # 남긴다 — `close_s` 하나만 보고해 진단 구간을 누락시키지 않는다. 합계는 **구간 전체 실측**
        # (`t_end - t_cleanup0`)이며 부분합 반올림이 아니다.
        PHASE["cleanup_total_s"] = round(time.monotonic() - _t_cleanup0, 3)
        PHASE["total_s"] = round(time.monotonic() - t_start, 3)
        manifest["phase_seconds"] = dict(PHASE)
        manifest["ok"] = False
        manifest.setdefault("failure_reasons", []).append(
            "simulation_app.close() exceeded its budget — whole workflow is NOT a success "
            "(separate from image-quality judgement)")
        manifest["close"] = dict(info, exit_code=CLOSE_TIMEOUT_EXIT, note=(
            "simulation_app.close() 가 예산 안에 끝나지 않았다. 행으로 두지 않고 나간다 — "
            "이 사실을 영수증에 남긴다. 바깥 러너의 같은 cap 이 이중 경계로 남아 있다."))
        try:
            json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
        except Exception:                                             # noqa: BLE001
            pass
        print(f"W13R_ISAAC_CLOSE_TIMEOUT after {info['budget_s']:.1f}s "
              f"(measured {info['measured_s']:.1f}s) — exiting without hanging", flush=True)
        # `ROOT_READINESS_PREFLIGHT_02` ④: 예전엔 `os._exit(rc_)` 라 렌더가 ok 면 **rc 0** 이 됐고
        # 시간초과가 준비완료로 통과할 수 있었다. 시간초과는 **전체 워크플로 비성공**이다 —
        # 화면 품질 판정과 분리해 전용 코드로 나간다. 바깥 실행기 제한이 최종 경계로 남는다.
        os._exit(CLOSE_TIMEOUT_EXIT)

    # ⚠️ **철회** (root `msg_bcb387d139fc` (4)): 이전 판 주석은 "07_actual 에서 close 가 설치본
    #    simulation_app.py 803–805 의 Replicator 대기에서 멈췄다" 고 **단정**했다. 그 단정은
    #    **rev20 실측 반례로 철회한다** — rev20 은 `wait_for_replicator=False` 로 돌아서 그
    #    대기 분기가 **실행되지 않았는데도** close 가 걸렸다. 따라서 차단 지점은 **미확정**이고
    #    `wait_for_replicator=False` 는 고쳐진 것이 아니라 **한 후보를 배제한 음성 결과**다.
    #    `skip_cleanup=True`·강제 exit 0 은 쓰지 않는다. 최종 경계는 바깥 실행기 프로세스그룹이다.
    # root msg_56ff30cd8eeb (2): **close 를 부르기 전에** 공식 호출을 하나씩 재현해 차단 지점을
    # 국소화한다. 각 단계 before/after 를 durable receipt 로 flush 하므로 TERM 되어도 흔적이 남는다.
    # ⚠️ 이 프로브는 `set_capture_on_play(False)` 를 `stop()` 보다 먼저 부른다 → **순수 관측이
    #    아니라 진단 절차 변경**이다(root `msg_bcb387d139fc` (4)).
    _cd_path = out / "close_localization_receipt.json"
    # 정리 구간 전체를 감싸는 실측 시작점(root msg_dc9b5fbc2940: 부분합 반올림이 아니라 t_end-t_start).
    # ⚠️ rev24: 이 시작점은 **clear_instance 앞**이다 — 그 호출 시간도 기존 정리 예산 안에 든다.
    _t_cleanup0 = time.monotonic()

    # ── rev24 유일한 수명주기 개입 (root msg_d1e4f0b148ed) ──────────────────
    # 공개 `SimulationContext.clear_instance()` 를 **close 프로브보다, 그리고 stage-close/STOP 을
    # 유발할 수 있는 어떤 update 보다 먼저** 부른다. 설치본은 이 호출에서
    # `_app_control_on_stop_handle` 구독을 해제한다(isaaclab sim/simulation_context.py:639-646).
    # 이 revision 에서 하지 않는 것: 명시 sim.stop/timeline stop · 카메라 GC · render product
    # 재조회/파기 · 전역 destroy_hydra_textures/SyntheticData.reset · private 필드 접근.
    # ⚠️ native 인과는 **미입증**이다. 이것은 한 수명주기 변경을 먼저 시험하려는 것이다.
    _clr_path = out / "clear_instance_receipt.json"
    _t_clr0 = time.monotonic()
    try:
        _clr_rec, _clr_ent = CD.record_single_call(
            _clr_path, "SimulationContext.clear_instance", SimulationContext.clear_instance,
            note=("close 프로브·update·close 보다 먼저 부른다. 설치본 clear_instance 는 "
                  "STOP 콜백 구독을 해제한다. native 인과는 미입증."))
        manifest["clear_instance"] = {
            "receipt": str(_clr_path), "called_before_close_probe": True,
            "returned": _clr_ent["returned"], "raised": _clr_ent["raised"],
            "elapsed_s": _clr_ent["elapsed_s"],
            "measurement_satisfied": _clr_rec.get("measurement_satisfied"),
            "unmet_reasons": _clr_rec.get("measurement_unmet_reasons"),
            "native_causality_unproven": True}
    except (SystemExit, KeyboardInterrupt) as _exc:
        manifest["clear_instance"] = {
            "receipt": str(_clr_path), "called_before_close_probe": True,
            "measurement_satisfied": False, "propagated": True,
            "error": f"{type(_exc).__name__}: {_exc}"[:300]}
        PHASE["clear_instance_s"] = round(time.monotonic() - _t_clr0, 3)
        manifest["phase_seconds"] = dict(PHASE)
        manifest["ok"] = False
        try:
            json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
        except Exception:                                             # noqa: BLE001
            pass
        raise
    PHASE["clear_instance_s"] = round(time.monotonic() - _t_clr0, 3)
    # 예외는 **성공으로 바뀌지 않는다** — 렌더 품질과 무관하게 비성공으로 표시한다.
    if manifest["clear_instance"].get("measurement_satisfied") is not True:
        manifest.setdefault("honest_gaps", []).append(
            "SimulationContext.clear_instance 가 반환하지 않았거나 예외였다 — "
            "clear_instance_receipt.json 참조. 성공으로 취급하지 않는다.")
        rc_ = rc_ or CLOSE_MEASURE_EXIT

    # ── rev26 유일한 개입 (root `msg_2b0a2544945f`) ─────────────────────────
    # `clear_instance` 가 **반환한 뒤**, 그리고 기존 probe/update/close **앞에서**, 렌더러가
    # 보유한 씬 참조(cams/scene/robot/sim)를 놓는다. 표시·매니페스트의 마지막 사용은 모두
    # 끝난 뒤다(최종 참조 L1083/L1067/L1005/L1068, 최종 호출 set_pose L1062·sim_clock L1151).
    # `SimulationContext` **클래스**와 `simulation_app` 은 그대로 유지한다.
    #
    # 왜 지금인가: 프레임워크가 닫힌 **뒤** 인터프리터 정리에서 소멸자가 돌면 이미 해제된
    # native 객체를 만진다. 15_actual 의 SIGSEGV 는 커널 기록상
    # `libomni.physx.tensors.plugin.so` 에서 났고(감사), 이는 **robot/scene/sim 텐서 수명**을
    # 가리킨다 — 카메라 소멸자 인과로 좁혀진 것이 **아니다**(root `msg_111aacf8adfd`).
    # `set_pose`/`sim_clock` 이 같은 셀을 공유하지만 None 대입이 그 셀 내용을 바꾸므로
    # 별도 콜러블 삭제는 필요 없다(root CPU 반례 확인). 별칭은 가정하지 않고 실제로 조사했다:
    # `jn`(문자열 리스트) · `dt`(float) · `q`(tensor) · `rgb`(ndarray) 는 이 객체들을 붙들지 않는다.
    # ⚠️ **native 효과는 미입증**이다. 참조가 언제 풀렸는지만 기록한다.
    def _release_owned_refs():
        nonlocal cams, scene, robot, sim
        cams = None
        scene = None
        robot = None
        sim = None
        # REFERENCE_RELEASE_CONTRACT_REV2 §2: **실제 셀 값**에서 계산한 네 `is None` 과
        # 공개 `SimulationContext.instance() is None` 을 그대로 돌려준다.
        # 상수 True·repr 문자열·truthy 로 대신하지 않는다 — helper 가 exact key/bool 로 분류한다.
        return {"cams_is_none": cams is None,
                "scene_is_none": scene is None,
                "robot_is_none": robot is None,
                "sim_is_none": sim is None,
                "public_instance_is_none": SimulationContext.instance() is None}

    _rel_path = out / "reference_release_receipt.json"
    _t_rel0 = time.monotonic()
    try:
        # root `msg_d23c68b2ac6e`: `cams` 는 dict 라 weakref 가 안 되므로 그것만 넣으면
        # **카메라 수명이 하나도 관측되지 않는다**. 실제 Camera 객체 2개를 이름으로 함께 넣는다
        # (dict 자체는 weakref 미지원으로 명시해 그대로 둔다).
        _rel_rec, _rel_ent = CD.record_reference_release(
            _rel_path, {"cams": cams, "camera_side": cams["side"], "camera_top": cams["top"],
                        "scene": scene, "robot": robot, "sim": sim},
            _release_owned_refs,
            note=("clear_instance 반환 뒤 · close probe/update/close 앞. SimulationContext 클래스와 "
                  "simulation_app 은 유지. __del__ 직접 호출·전역 gc·render product 삭제 없음."))
        manifest["reference_release"] = {
            "receipt": str(_rel_path), "released_after_clear_instance": True,
            "released_before_close_probe": True,
            "returned": _rel_ent["returned"], "raised": _rel_ent["raised"],
            "elapsed_s": _rel_ent["elapsed_s"],
            # 계약 3·4: 필수와 정보성을 **분리**해 적는다(혼동 금지).
            "required_survivors": _rel_rec.get("required_survivors"),
            "informational_survivors": _rel_rec.get("informational_survivors"),
            "release_mapping_classification": _rel_rec.get("release_mapping_classification"),
            "n_weakref_observed": _rel_rec.get("n_weakref_observed"),
            "measurement_satisfied": _rel_rec.get("measurement_satisfied"),
            "unmet_reasons": _rel_rec.get("measurement_unmet_reasons"),
            "native_effect_unproven": True}
    except (SystemExit, KeyboardInterrupt) as _exc:
        manifest["reference_release"] = {
            "receipt": str(_rel_path), "measurement_satisfied": False, "propagated": True,
            "error": f"{type(_exc).__name__}: {_exc}"[:300]}
        PHASE["reference_release_s"] = round(time.monotonic() - _t_rel0, 3)
        manifest["phase_seconds"] = dict(PHASE)
        manifest["ok"] = False
        try:
            json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
        except Exception:                                             # noqa: BLE001
            pass
        raise
    PHASE["reference_release_s"] = round(time.monotonic() - _t_rel0, 3)
    if manifest["reference_release"].get("measurement_satisfied") is not True:
        manifest.setdefault("honest_gaps", []).append(
            "보유 참조 해제 계측 미충족 — 호출 미반환/예외 · **필수** 대상(카메라 2·scene·robot) "
            "생존 · release 반환 mapping 의 exact key/bool True 불만족 중 하나다. "
            "reference_release_receipt.json 참조. 정보성 sim 생존만으로는 실패가 아니다. "
            "성공으로 취급하지 않는다.")
        rc_ = rc_ or CLOSE_MEASURE_EXIT

    _t_diag0 = time.monotonic()          # 진단 구간은 clear_instance **뒤**에서 따로 잰다
    try:
        _cd = CD.localize_replicator_close(_cd_path, app_update=simulation_app.update,
                                           bounded_update_budget_s=5.0)
        manifest["close_localization"] = {
            "receipt": str(_cd_path), "summary": _cd.get("summary"),
            "completed_all_steps": _cd.get("completed_all_steps"),
            # root msg_bcb387d139fc (3): 계측 미충족을 **조용히 PASS 시키지 않는다**.
            "measurement_satisfied": _cd.get("measurement_satisfied"),
            "measurement_unmet_reasons": _cd.get("measurement_unmet_reasons"),
            "is_pure_observation": False,
            "procedure_change": _cd.get("procedure_change_note")}
    except (SystemExit, KeyboardInterrupt) as _exc:
        # root msg_dc9b5fbc2940: 렌더러 경계도 제어 예외를 **삼키지 않는다** — 기록 후 전파.
        manifest["close_localization"] = {
            "receipt": str(_cd_path), "measurement_satisfied": False,
            "measurement_unmet_reasons": [f"{type(_exc).__name__} 재전파 — 계측 미완"],
            "probe_error": f"{type(_exc).__name__}: {_exc}"[:300], "propagated": True}
        PHASE["close_diagnostic_s"] = round(time.monotonic() - _t_diag0, 3)
        manifest["phase_seconds"] = dict(PHASE)
        manifest["ok"] = False
        try:
            json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
        except Exception:                                             # noqa: BLE001
            pass
        raise
    except BaseException as _exc:                                     # noqa: BLE001
        manifest["close_localization"] = {
            "receipt": str(_cd_path), "measurement_satisfied": False,
            "measurement_unmet_reasons": ["프로브가 예외로 끝났다"],
            "probe_error": f"{type(_exc).__name__}: {_exc}"[:300], "propagated": False}
    # 진단 시간과 실제 close 시간을 **분리**해 남긴다(root msg_bcb387d139fc (3)).
    PHASE["close_diagnostic_s"] = round(time.monotonic() - _t_diag0, 3)
    # root msg_dc9b5fbc2940: close 는 **기존 정리 할당에서 남은 예산**으로 부른다. 진단이 쓴
    # 시간을 무시하고 close 에 full budget 을 또 주면 정리 할당을 초과한다. 상향 없음.
    # rev24(root msg_d1e4f0b148ed): 남은 예산은 `clear_instance` 를 **포함한** 정리 구간 전체
    # 실측(`t - _t_cleanup0`)에서 뺀다 — 그 호출도 기존 예산 안에 들어가야 한다.
    _cleanup_used_s = time.monotonic() - _t_cleanup0
    PHASE["cleanup_used_before_close_s"] = round(_cleanup_used_s, 3)
    _close_remaining = max(0.0, close_budget - _cleanup_used_s)
    PHASE["close_budget_remaining_s"] = round(_close_remaining, 3)
    if _close_remaining <= 0.0:
        manifest.setdefault("honest_gaps", []).append(
            f"진단이 정리 할당 {close_budget:.1f}s 를 전부 써서 close 에 남은 예산이 없다 — "
            "close 를 부르지 않고 시간초과 경로로 나간다. 예산을 늘리지 않는다.")
    # 계측이 충족되지 않았으면 렌더 품질과 무관하게 **비성공**으로 표시한다.
    if manifest["close_localization"].get("measurement_satisfied") is not True:
        manifest["close_localization_non_success"] = True
        manifest.setdefault("honest_gaps", []).append(
            "close 차단 지점 계측 미충족 — close_localization.measurement_unmet_reasons 참조. "
            "렌더 산출물이 있어도 이 항목은 PASS 가 아니다.")
        rc_ = rc_ or CLOSE_MEASURE_EXIT
    # root msg_dc9b5fbc2940: `manifest["ok"]` 의 의미를 **명시**한다 — 그림 품질 전용이고
    # 워크플로 성공이 아니다. 둘을 같은 값으로 읽으면 계측 미충족이 PASS 로 새어나간다.
    manifest["ok_semantics"] = {
        "manifest_ok_means": "renderer_quality_only — 프레임/IK/카메라/행정체성 등 **그림** 판정",
        "workflow_non_success": bool(rc_ != 0),
        "workflow_non_success_sources": (
            [k for k, v in (("renderer_quality", manifest["ok"] is not True),
                            ("close_measurement_unmet",
                             manifest.get("close_localization_non_success") is True)) if v]),
        "note": ("manifest.ok=True 라도 rc 가 0 이 아니면 **워크플로는 비성공**이다. "
                 f"계측 미충족 = rc {CLOSE_MEASURE_EXIT}, close 시간초과 = rc {CLOSE_TIMEOUT_EXIT}."),
    }
    # root msg_d1e4f0b148ed: **close 직전 dump 바로 앞에서** PHASE 를 manifest 로 복사한다.
    # rev23 은 이 재대입이 없어 이미 측정한 `close_diagnostic_s` 가 매니페스트에 저장되지 않았다
    # (close 가 반환하지 않으면 그 값이 영영 유실됐다). `cleanup_total_s` 는 아직 **실측 전**이라
    # PHASE 에 없고, 따라서 여기서도 null(부재) 로 남는다 — 값을 지어내지 않는다.
    manifest["phase_seconds"] = dict(PHASE)
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
    if _close_remaining <= 0.0:
        # `bounded_close` 는 0 예산을 거부한다(올바른 계약). 예산이 없으면 close 를 **부르지 않고**
        # 같은 비성공 경로로 나간다 — 바깥 러너의 그룹 TERM→KILL 이 최종 경계로 남는다.
        _on_close_timeout({"timed_out": True, "measured_s": 0.0, "budget_s": 0.0,
                           "close_not_called": True,
                           "reason": "진단이 정리 할당을 전부 소진 — close 호출 예산 없음"})
    _ci = bounded_close(lambda: simulation_app.close(wait_for_replicator=False),
                        _close_remaining, _on_close_timeout)
    PHASE["close_s"] = _ci["measured_s"]
    PHASE["close_timed_out"] = False
    # 정리 합계는 **구간 전체 실측**이다(부분합 반올림 아님).
    PHASE["cleanup_total_s"] = round(time.monotonic() - _t_cleanup0, 3)
    PHASE["total_s"] = round(time.monotonic() - t_start, 3)
    manifest["phase_seconds"] = dict(PHASE)
    manifest["close"] = dict(_ci)
    json.dump(manifest, open(man_path, "w"), ensure_ascii=False, indent=2)
    print(f"W13R_ISAAC_PHASES startup={PHASE['startup_s']}s render={PHASE['render_s']}s "
          f"ffmpeg={PHASE['ffmpeg_s']}s close={PHASE['close_s']}s total={PHASE['total_s']}s",
          flush=True)
    return rc_


if __name__ == "__main__":
    sys.exit(main())
