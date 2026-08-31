"""스쿱 폐합 RRD 후처리 (D341 관측 계층).

왜 후처리인가
    DEME 는 `roarm` env 에만 있고 (rerun 0.26.2), D341 이 요구하는 rerun **0.34.1** 은
    `isaaclab` env 에만 있다. 그래서 시뮬은 `roarm` 에서 돌려 원자료를 npz 로 남기고,
    RRD 는 이 스크립트를 **isaaclab python** 으로 돌려 굽는다. 설치는 하지 않으므로
    D326 핀(numpy 1.26.0 / psutil 5.9.8)에 영향이 없다.

    ~/miniconda3/envs/isaaclab/bin/python sim_deme_scoop_rerun.py <출력폴더> <tag>

⚠️ 이것은 **재생·검수용 관측 아티팩트**다. 수치 정본은 npz/JSON 이며 Rerun 의
   Float32 사본을 과학적 게이트에 되먹이지 않는다 (D341).
⚠️ D341 **완결** 계약(rrd verify · 고정 블루프린트 + .rbl · 헤드리스 스크린샷 ·
   육안 검수 기록)은 본 스크립트가 **수행하지 않는다**. RRD 생성까지만이다.
"""
import sys, json
from pathlib import Path
import numpy as np
import rerun as rr

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "claudedocs/runtime_logs/scoop_track/s2_closure_rot")
TAG = sys.argv[2] if len(sys.argv) > 2 else "rep1"

z = np.load(OUT / f"scoop_{TAG}.npz")
tl = json.load(open(OUT / f"scoop_timeline_{TAG}.json"))
rows = tl["rows"] if isinstance(tl, dict) else tl
res = json.load(open(OUT / f"scoop_closure_{TAG}.json"))

t = z["frame_t_s"]
nL, nR = z["nodes_L_m"], z["nodes_R_m"]
cp, cf = z["contact_point_m"], z["contact_force_N"]
cframe, cside = z["contact_frame"], z["contact_side"]
rad = float(z["radii_m"][0])
n_frames = len(t)
print(f"프레임 {n_frames} · 셸 노드 {nL.shape[1]} · 접촉 샘플 {len(cp)}", flush=True)

rrd = OUT / f"scoop_{TAG}.rrd"
# 파일 싱크를 **첫 로그 전에** 붙인다 (D341). 컨텍스트 종료가 finalize 다.
with rr.RecordingStream("roarm_scoop_closure", recording_id=f"scoop_{TAG}") as stream:
    stream.save(str(rrd))
    rr.set_time("sim_time_s", duration=0.0, recording=stream)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=stream)
    rr.log("world/pile_final",
           rr.Points3D(z["positions_m"].astype(np.float32),
                       radii=np.full(len(z["positions_m"]), rad, np.float32),
                       colors=[180, 170, 140]),
           static=True, recording=stream)

    for i in range(n_frames):
        rr.set_time("sim_time_s", duration=float(t[i]), recording=stream)
        r = rows[i] if i < len(rows) else rows[-1]
        rr.log("world/shell_L", rr.Points3D(nL[i], radii=0.0008,
                                            colors=[70, 130, 200]), recording=stream)
        rr.log("world/shell_R", rr.Points3D(nR[i], radii=0.0008,
                                            colors=[200, 110, 70]), recording=stream)
        m = cframe == i
        if m.any():
            p, f = cp[m], cf[m]
            rr.log("world/contact_points",
                   rr.Points3D(p, radii=0.0010, colors=[220, 60, 60]), recording=stream)
            # 힘 화살표는 1 N = 5 mm 로 축소해 그린다 (표시용 배율 — 수치 아님)
            rr.log("world/contact_forces",
                   rr.Arrows3D(origins=p, vectors=f * 0.005, colors=[220, 60, 60]),
                   recording=stream)
        else:
            rr.log("world/contact_points", rr.Clear(recursive=False), recording=stream)
            rr.log("world/contact_forces", rr.Clear(recursive=False), recording=stream)
        rr.log("metrics/n_contacts", rr.Scalars(float(r["n_total"])), recording=stream)
        rr.log("metrics/F_total_N", rr.Scalars(float(r["F_total_N"])), recording=stream)
        rr.log("metrics/lipF_total_N", rr.Scalars(float(r["lipF_total_N"])),
               recording=stream)
        rr.log("metrics/max_single_contact_N",
               rr.Scalars(float(r.get("max_single_contact_N", 0.0))), recording=stream)
        if "phi_deg" in r:
            rr.log("metrics/phi_deg", rr.Scalars(float(r["phi_deg"])), recording=stream)

print(f"RRD -> {rrd}  ({rrd.stat().st_size/1e6:.2f} MB)")
print("⚠️ D341 완결 계약(rrd verify · .rbl · 스크린샷 · 육안 검수) 미수행 — "
      "본 스크립트는 RRD 생성까지다.")
print(f"   검수: {Path(sys.executable).parent}/rerun {rrd}")
