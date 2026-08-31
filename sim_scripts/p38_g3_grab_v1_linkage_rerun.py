"""p38 / g3 — 그랩 v1 **4절 링크 구동** 서보 0~89도 전 구간 Rerun 기록 (D463, D341 계약).

무엇을 남기나
    D341 은 "판정이 기하·자세·좌표계·궤적에 의존하면 재생 가능한 RRD 가 필수" 라고 한다.
    본 판정(링크가 서보 89도를 셸 44.5도로 바꾸고 그 사이 아무것도 안 닿는다)은 전적으로
    기하·궤적 판정이므로 RRD 를 만든다.

    - 타임라인 `servo_step` : 서보 0 -> 89도 전 91 스텝. 최종프레임만 남기지 않는다.
    - 엔티티 분리 : 셸 L / 셸 R / 서보 크랭크 / 로드 / 셸 크랭크 / 순정 가동 조 /
      브래킷 / link5 / 세 회전축 / 핀 궤적 — **판정 대상마다 별도 엔티티**.
    - 결정 스칼라 : shell_deg, mouth_mm, 전달각(입력·출력), dshell_dservo,
      링크-팔 최소여유.
    - 권위는 `design.json` / `g3_results.json` 의 수치이고 Rerun 은 **검수 증거**다
      (D341: Rerun 은 bit-exact authority 가 아니다).

버전 핀 : rerun-sdk / rerun-cli 0.34.1 (D326 IsaacLab 환경). numpy 1.26.0 / psutil 5.9.8 유지.
사용    : /home/cgxr/miniconda3/envs/isaaclab/bin/python sim_scripts/p38_g3_grab_v1_linkage_rerun.py [출력디렉터리]
"""
import sys
import json
import math
import hashlib
from pathlib import Path
import importlib.util

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else REPO / "claudedocs/runtime_logs/grab_track/g3_rerun")

RERUN_VERSION = "0.34.1"
RERUN_CLI = "/home/cgxr/miniconda3/envs/isaaclab/bin/rerun"
APP_ID = "roarm_grab_v1_d463_linkage"

_spec = importlib.util.spec_from_file_location(
    "p37", REPO / "sim_scripts/p37_g2_grab_v1_attach_probe.py")
p37 = importlib.util.module_from_spec(_spec)
_argv, sys.argv = sys.argv, ["x"]
_spec.loader.exec_module(p37)
sys.argv = _argv
G, P, K = p37.G, p37.G.P, p37.G.kin(p37.G.P)

COL = {"link5": [130, 135, 145], "jaw": [200, 120, 60],
       "shell_L": [70, 160, 235], "shell_R": [70, 200, 160],
       "bracket": [180, 180, 190], "servo_crank": [235, 90, 90],
       "rod": [245, 200, 60], "shell_crank": [170, 110, 235]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mesh_of(parts, T=None):
    m = trimesh.util.concatenate([p.copy() for p in parts])
    if T is not None:
        m.apply_transform(T)
    return m


def axis_strip(point, direction, half=45.0):
    p = np.asarray(point, float)
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    return [[(p - half * d).tolist(), (p + half * d).tolist()]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import rerun as rr
    if rr.__version__ != RERUN_VERSION:
        raise RuntimeError(f"rerun {rr.__version__} != {RERUN_VERSION} (D326 핀)")
    import rerun.blueprint as rrb
    from roarm_rl.rerun_contract import validate_rerun_artifact

    files = {"rrd": OUT / "g3_linkage.rrd", "rbl": OUT / "g3_linkage.rbl",
             "png": OUT / "g3_inspection.png",
             "validation": OUT / "rerun_validation.json",
             "results": OUT / "g3_rerun_results.json"}
    for k in ("rrd", "rbl", "png"):
        if files[k].exists():
            files[k].unlink()

    T = p37.placement()
    link5 = p37.load_mm("link5.stl")
    jaw0 = p37.jaw_in_link5()
    sL, nL = G.build_shell(P, -1)
    sR, nR = G.build_shell(P, +1)
    br, nB = G.build_bracket(P)
    lkp, nK, lk = G.build_linkage(P)
    rows = lk["rows"]

    grp = {g: [(m, n) for m, n in zip(lkp, nK) if G.linkage_group(n) == g]
           for g in ("servocrank", "rod", "shellcrank")}

    # 링크-팔 최소여유 (결정 스칼라). 전 스텝은 비싸므로 5스텝 간격.
    l5_cloud = p37.surface_cloud(link5)
    tree, pitch = p37.link5_occupancy(link5, 1.0)
    l5_lo, l5_hi = link5.bounds
    clearance = {}
    for i in range(0, len(rows), 5):
        Ts, Tr, Tk = G.linkage_pose(P, lk, i)
        parts = []
        for g_, Tg in (("servocrank", Ts), ("rod", Tr), ("shellcrank", Tk)):
            for m0, n0 in grp[g_]:
                m = m0.copy()
                m.apply_transform(T @ Tg)
                parts.append((m, n0))
        v, _w = p37.clearance_to(parts, tree, pitch, l5_lo, l5_hi, l5_cloud)
        clearance[i] = float(v)

    hinge_world = T[:3, :3] @ np.array([0.0, 0.0, 1.0])
    piv_w = {s: T[:3, :3] @ np.array([s * K["g"] / 2.0, 0.0, 0.0]) + T[:3, 3]
             for s in (-1, +1)}

    with rr.RecordingStream(APP_ID, recording_id="grab_v1_d463_linkage",
                            make_default=False, send_properties=True) as rec:
        rec.save(str(files["rrd"]), write_footer=True)

        # ── 정적: 팔 · 브래킷 · 회전축 3개 ────────────────────────────────
        for ent, m, c in (("arm/link5", link5, COL["link5"]),
                          ("grab/bracket", mesh_of(br, T), COL["bracket"])):
            rec.log(ent, rr.Mesh3D(vertex_positions=np.asarray(m.vertices, np.float32),
                                   triangle_indices=np.asarray(m.faces, np.uint32),
                                   albedo_factor=c), static=True)
        rec.log("axes/servo", rr.LineStrips3D(
            axis_strip(p37.GRIPPER_ORIGIN, p37.GRIPPER_AXIS, 40.0),
            colors=[[235, 90, 90]], radii=0.6), static=True)
        rec.log("axes/pivot_L", rr.LineStrips3D(
            axis_strip(piv_w[-1], hinge_world, 40.0),
            colors=[[70, 160, 235]], radii=0.6), static=True)
        rec.log("axes/pivot_R", rr.LineStrips3D(
            axis_strip(piv_w[+1], hinge_world, 40.0),
            colors=[[70, 200, 160]], radii=0.6), static=True)

        # ── 시간축: 서보 0 -> 89도 ────────────────────────────────────────
        pin1_tr, pin2_tr = [], []
        for i, row in enumerate(rows):
            rec.reset_time()
            rec.set_time("servo_step", sequence=i)
            Ts, Tr, Tk = G.linkage_pose(P, lk, i)

            Tj = p37.rot_about(p37.GRIPPER_ORIGIN, p37.GRIPPER_AXIS,
                               math.radians(row["servo_deg"]))
            jw = jaw0.copy()
            jw.apply_transform(Tj)
            bodies = [("arm/gripper_jaw", jw, COL["jaw"])]
            for parts, side, ent, c in ((sL, -1, "grab/shell_L", COL["shell_L"]),
                                        (sR, +1, "grab/shell_R", COL["shell_R"])):
                Rj = p37.rot_about(piv_w[side], hinge_world,
                                   math.radians(side * row["shell_deg"]))
                bodies.append((ent, mesh_of(parts, Rj @ T), c))
            for g_, Tg, ent, c in (("servocrank", Ts, "linkage/servo_crank", COL["servo_crank"]),
                                   ("rod", Tr, "linkage/rod", COL["rod"]),
                                   ("shellcrank", Tk, "linkage/shell_crank", COL["shell_crank"])):
                bodies.append((ent, mesh_of([m for m, _ in grp[g_]], T @ Tg), c))
            for ent, m, c in bodies:
                rec.log(ent, rr.Mesh3D(
                    vertex_positions=np.asarray(m.vertices, np.float32),
                    triangle_indices=np.asarray(m.faces, np.uint32),
                    albedo_factor=c))

            p1 = T[:3, :3] @ np.array([row["pin1_xy"][0], row["pin1_xy"][1],
                                       P["rod_plane_z_mm"]]) + T[:3, 3]
            p2 = T[:3, :3] @ np.array([row["pin2_xy"][0], row["pin2_xy"][1],
                                       P["rod_plane_z_mm"]]) + T[:3, 3]
            pin1_tr.append(p1.tolist())
            pin2_tr.append(p2.tolist())
            rec.log("linkage/pins", rr.Points3D(np.array([p1, p2]),
                                                colors=[[255, 255, 255], [255, 255, 255]],
                                                radii=1.4))
            rec.log("linkage/pin_paths", rr.LineStrips3D(
                [pin1_tr[:], pin2_tr[:]],
                colors=[[235, 90, 90], [170, 110, 235]], radii=0.35))

            rec.log("plots/shell_deg", rr.Scalars(float(row["shell_deg"])))
            rec.log("plots/mouth_mm", rr.Scalars(float(row["mouth_mm"])))
            rec.log("plots/trans_angle_out_deg",
                    rr.Scalars(float(row["trans_angle_deg"])))
            rec.log("plots/trans_angle_in_deg",
                    rr.Scalars(float(row["trans_angle_in_deg"])))
            rec.log("plots/dshell_dservo", rr.Scalars(float(row["dshell_dservo"])))
            if i in clearance:
                rec.log("plots/linkage_clearance_mm", rr.Scalars(clearance[i]))
            if i % 10 == 0 or i == len(rows) - 1:
                rec.log("events/phase", rr.TextLog(
                    f"servo {row['servo_deg']:6.2f}deg -> shell {row['shell_deg']:6.2f}deg "
                    f"mouth {row['mouth_mm']:6.2f}mm | mu_out {row['trans_angle_deg']:6.2f} "
                    f"mu_in {row['trans_angle_in_deg']:6.2f} | "
                    f"dshell/dservo {row['dshell_dservo']:.4f}",
                    level=rr.TextLogLevel.INFO))

        # ── 게이트 판정 ──────────────────────────────────────────────────
        design = json.load(open(REPO / "claudedocs/runtime_logs/grab_track"
                                       "/g3_linkage/design.json"))
        probe = json.load(open(REPO / "claudedocs/runtime_logs/grab_track"
                                      "/g3_attach/g2_results.json"))
        rec.reset_time()
        rec.set_time("servo_step", sequence=len(rows) - 1)
        for k, v in design["gates"].items():
            rec.log("events/verdict", rr.TextLog(
                f"design::{k} = {'PASS' if v['pass'] else 'FAIL'}",
                level=rr.TextLogLevel.INFO if v["pass"] else rr.TextLogLevel.WARN))
        for k, v in probe["gates"].items():
            rec.log("events/verdict", rr.TextLog(
                f"p37::{k} = {'PASS' if v['pass'] else 'FAIL'}",
                level=rr.TextLogLevel.INFO if v["pass"] else rr.TextLogLevel.WARN))
        n_fail = sum(1 for v in design["gates"].values() if not v["pass"])
        rec.log("events/verdict", rr.TextLog(
            f"VERDICT design gates {len(design['gates']) - n_fail}/{len(design['gates'])}"
            f" PASS (the one FAIL is self_load_ratio, deliberately held open until the"
            f" pellet bulk density is measured) | p37 "
            f"{sum(1 for v in probe['gates'].values() if v['pass'])}/"
            f"{len(probe['gates'])} PASS -> {probe['verdict']}",
            level=rr.TextLogLevel.INFO))

        # ⚠️ Rerun 내장 폰트에 한글 글리프가 없다. 뷰 제목·문서·로그를 한글로 쓰면
        #    네모(두부)로 렌더돼 **육안 검수가 불가능해진다** (D384 attempt6 전례).
        #    따라서 Rerun 에 보이는 문자열은 전부 ASCII 영문으로 쓴다.
        md = (
            f"# Grab v1 - four-bar linkage drive (D463)\n\n"
            f"**servo 0..89 deg -> shell 0..44.5 deg -> mouth 0..58.0 mm**\n\n"
            f"| linkage dimension | mm |\n|---|---|\n"
            f"| ground (servo axis <-> pivot L) | {lk['ground_len_mm']} |\n"
            f"| servo crank r2 | {lk['crank_servo_r_mm']} |\n"
            f"| connecting rod r3 | {lk['rod_len_mm']} |\n"
            f"| shell crank r4 | {lk['crank_shell_r_mm']} |\n\n"
            f"transmission angle: output "
            f"{lk['trans_angle_out_min_deg']}..{lk['trans_angle_out_max_deg']} deg, input "
            f"{lk['trans_angle_in_min_deg']}..{lk['trans_angle_in_max_deg']} deg "
            f"(band 40..140, worst margin "
            f"{design['gates']['linkage_no_dead_center']['worst_margin_deg']} deg)\n\n"
            f"A direct spur-gear drive needed a 19.5 mm centre distance but the real "
            f"servo-axis to pivot-L distance is {lk['ground_len_mm']} mm, i.e. 58.3 mm "
            f"short (D463). The linkage spans that distance directly.\n\n"
            f"Colours: link5 grey / stock jaw orange / shell L blue / shell R green / "
            f"bracket light grey / servo crank red / rod yellow / shell crank purple.\n\n"
            f"Authority = `design.json` + `g2_results.json`; this RRD is inspection "
            f"evidence only (D341).")
        rec.log("metadata/run", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN),
                static=True)

        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.TextDocumentView(origin="/metadata/run", contents="/metadata/run",
                                         name="1 | linkage dimensions + verdict"),
                    rrb.Spatial3DView(origin="/", contents=["/arm/**", "/grab/**",
                                                            "/linkage/**", "/axes/**"],
                                      name="2 | servo 0-89 deg sweep"),
                    rrb.TextLogView(origin="/events", contents="/events/**",
                                    name="3 | step log + gates"),
                    column_shares=[0.18, 0.60, 0.22]),
                rrb.Horizontal(
                    # 각도·개구(0~135)와 순간비·여유(0~3)는 스케일이 두 자릿수 달라
                    # 한 축에 겹치면 작은 쪽이 바닥에 눌린다. 뷰를 나눈다.
                    rrb.TimeSeriesView(
                        origin="/plots", name="4 | shell deg / mouth mm / transmission angle",
                        contents=["/plots/shell_deg", "/plots/mouth_mm",
                                  "/plots/trans_angle_out_deg", "/plots/trans_angle_in_deg"]),
                    rrb.TimeSeriesView(
                        origin="/plots", name="5 | dshell/dservo + linkage clearance to arm (mm)",
                        contents=["/plots/dshell_dservo", "/plots/linkage_clearance_mm"]),
                    column_shares=[0.55, 0.45]),
                row_shares=[0.80, 0.20]),
            auto_layout=False, auto_views=False, collapse_panels=True)
        rec.send_blueprint(blueprint, make_active=True, make_default=True)
        rec.flush(timeout_sec=120.0)
    blueprint.save(APP_ID, str(files["rbl"]))

    entities = ["metadata/run", "arm/link5", "arm/gripper_jaw", "grab/shell_L",
                "grab/shell_R", "grab/bracket", "linkage/servo_crank", "linkage/rod",
                "linkage/shell_crank", "linkage/pins", "linkage/pin_paths",
                "axes/servo", "axes/pivot_L", "axes/pivot_R",
                "plots/shell_deg", "plots/mouth_mm", "plots/trans_angle_out_deg",
                "plots/trans_angle_in_deg", "plots/dshell_dservo",
                "plots/linkage_clearance_mm", "events/phase", "events/verdict"]
    mesh_c = ["Mesh3D:vertex_positions", "Mesh3D:triangle_indices"]
    line_c = ["LineStrips3D:strips", "LineStrips3D:colors", "LineStrips3D:radii"]
    comps = {e: mesh_c for e in ("arm/link5", "arm/gripper_jaw", "grab/shell_L",
                                 "grab/shell_R", "grab/bracket", "linkage/servo_crank",
                                 "linkage/rod", "linkage/shell_crank")}
    comps.update({e: line_c for e in ("axes/servo", "axes/pivot_L", "axes/pivot_R",
                                      "linkage/pin_paths")})
    comps["linkage/pins"] = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    comps["metadata/run"] = ["TextDocument:text"]
    for e in entities:
        if e.startswith("plots/"):
            comps[e] = ["Scalars:scalars"]
        if e.startswith("events/"):
            comps[e] = ["TextLog:text", "TextLog:level"]

    validation = validate_rerun_artifact(
        files["rrd"],
        expected_entity_paths=entities,
        exact_entity_paths=entities,
        exact_timeline_names=["blueprint", "log_time", "servo_step"],
        expected_entity_components=comps,
        blueprint_path=files["rbl"],
        screenshot_path=files["png"],
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=600.0)
    files["validation"].write_text(json.dumps(validation, indent=2, default=str) + "\n")

    out = {"probe": "p38_g3_grab_v1_linkage_rerun",
           "rerun_version": RERUN_VERSION, "app_id": APP_ID,
           "timeline": "servo_step", "n_steps": len(rows),
           "linkage": {k: v for k, v in lk.items() if k != "rows"},
           "clearance_to_link5_mm": {str(k): round(v, 4) for k, v in clearance.items()},
           "rerun_validation_pass": bool(validation.get("pass")),
           "rerun_validation_errors": validation.get("errors"),
           "artifacts": {p.name: {"sha256_16": sha256(p)[:16], "bytes": p.stat().st_size}
                         for p in files.values() if p.exists()},
           "visual_inspection": "see visual_inspection.json in this folder (D341: 실제 육안 검수 기록 — 생성 성공은 검수가 아니다)"}
    files["results"].write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"rerun_validation pass={validation.get('pass')} errors={validation.get('errors')}")
    print(f"entities observed = {validation['entity_path_contract']['observed_non_system']}")
    print(f"timelines observed = {validation['timeline_contract']['observed']}")
    print(f"-> {OUT}")
    return 0 if validation.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
