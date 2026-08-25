#!/usr/bin/env python
"""p26 — y1_d453 야드 테스트베드 레이아웃 설계 (물리 0, Isaac 0).

목적: source 트레이(선창 축소판) + destination bin(높이 상한 H_max)을
RoArm top-down 파지 가능 영역(annulus, D440/t3w) 안에 배치하고,
o1 manifest 실측 부피로 bin 용량 tightness(ρ)와 표준 에피소드 N_ep를 결정.
출력 = y1_design.json + y1_layout.png (D324 단일 프레임 진단).

게이트:
  G-reach-src / G-reach-bin : 내부 셀 중심 전부 annulus [R_MIN+m, R_MAX-m] 내
  G-capacity               : ρ = (A_bin·H_max·φ) / V_solid(N_ep) ∈ [1.15, 1.60]
  G-payload                : 선정 subset 최대 질량(15% infill 추정) ≤ 30 g
  G-grasp-open             : subset 최대 파지 폭 ≤ 35 mm (개구 40~45 mm 예산)
주의: φ(패킹률)·질량은 선언된 가정(실측 아님) — 결과 JSON assumptions에 명기.
"""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "sim_assets" / "posco_rocks_o1" / "manifest.json"
OUT_DIR = ROOT / "claudedocs" / "runtime_logs" / "yard_track" / "y1_d453"

# ---- 고정 상수 (근거는 각 주석) ----
CELL = 0.010            # heightmap 셀 10 mm (최소 물체 폭 22 mm의 절반 이하)
R_MIN, R_MAX = 0.150, 0.325  # D440/t3w top-down 가능 annulus [m]
MARGIN = 0.005          # 도달 여유 [m] -> 유효 [0.155, 0.320]
AZ_SRC_DEG, AZ_BIN_DEG = +25.0, -25.0  # 로봇 정면(x축) 기준 대칭 배치
WALL_T = 0.005          # 벽 두께 [m]
PHI = 0.55              # 각진 강체 random loose packing 가정 (감도 0.50/0.60)
PHI_RANGE = (0.50, 0.60)
# tightness 기준: 완주 가능성 안전(성긴 패킹에서도 rho>1.10) + 공간 압박 유지(과잉 여유 금지)
RHO_MIN_AT_PHI_LO = 1.10  # rho(phi=0.50) >= 1.10 — 미달 시 에피소드 완주 불가 위험
RHO_MAX_AT_PHI_NOM = 1.60  # rho(phi=0.55) <= 1.60 — 초과 시 place 선택 압박 소실
H_MAX_CAND = [0.050, 0.060, 0.070, 0.080, 0.090]
N_EP_CAND = [32, 28, 24, 20, 16, 12]  # 클래스 균형(4의 배수), 큰 쪽 선호
PAYLOAD_CAP_G = 30.0    # 63rd doc §7 페이로드 예산
GRASP_OPEN_CAP_MM = 35.0


def cell_centers(cx, cy, n):
    """n x n 셀 중심 좌표 (world-axis-aligned)."""
    half = n * CELL / 2.0
    xs = cx - half + CELL * (np.arange(n) + 0.5)
    ys = cy - half + CELL * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def region_search(az_deg):
    """annulus 제약을 만족하는 최대 내부 변 길이 L(=n셀)과 반경 중심 r_c 탐색."""
    az = np.deg2rad(az_deg)
    best = None
    for n in range(14, 8, -1):  # 셀 수 14->9, 큰 쪽 우선
        for r_c in np.arange(0.180, 0.2951, 0.0025):
            cx, cy = r_c * np.cos(az), r_c * np.sin(az)
            gx, gy = cell_centers(cx, cy, n)
            rr = np.hypot(gx, gy)
            lo, hi = rr.min(), rr.max()
            if lo >= R_MIN + MARGIN and hi <= R_MAX - MARGIN:
                slack = min(lo - (R_MIN + MARGIN), (R_MAX - MARGIN) - hi)
                if best is None or n > best["n_cells"] or (
                        n == best["n_cells"] and slack > best["slack"]):
                    best = dict(n_cells=n, r_center=float(r_c),
                                center=[float(cx), float(cy)],
                                L_int=float(n * CELL),
                                r_cell_min=float(lo), r_cell_max=float(hi),
                                slack=float(slack))
        if best is not None and best["n_cells"] == n:
            break  # 이 n에서 해 존재 -> 더 작은 n 불필요
    return best


def main():
    m = json.loads(MANIFEST.read_text())
    man_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()[:16]
    objs = m["objects"]
    classes = sorted({o["class_mm"] for o in objs})
    by_class = {c: sorted([o for o in objs if o["class_mm"] == c],
                          key=lambda o: o["index"]) for c in classes}

    src = region_search(AZ_SRC_DEG)
    binr = region_search(AZ_BIN_DEG)
    g_reach_src = src is not None
    g_reach_bin = binr is not None
    if not (g_reach_src and g_reach_bin):
        print("FATAL: reach 탐색 실패", src, binr)
        sys.exit(2)

    # ---- 용량: (N_ep, H_max) 결정 ----
    a_bin = binr["L_int"] ** 2
    pick = None
    for n_ep in N_EP_CAND:
        k = n_ep // len(classes)
        subset = [o for c in classes for o in by_class[c][:k]]
        v_solid = sum(o["volume_cm3"] for o in subset) * 1e-6  # m^3
        for h_max in H_MAX_CAND:
            rho = a_bin * h_max * PHI / v_solid
            rho_lo = a_bin * h_max * PHI_RANGE[0] / v_solid
            rho_hi = a_bin * h_max * PHI_RANGE[1] / v_solid
            if rho_lo >= RHO_MIN_AT_PHI_LO and rho <= RHO_MAX_AT_PHI_NOM:
                pick = dict(n_ep=n_ep, per_class=k, h_max=h_max,
                            v_solid_cm3=v_solid * 1e6, rho=rho,
                            rho_phi050=rho_lo, rho_phi060=rho_hi,
                            subset_names=[o["name"] for o in subset],
                            subset_sha16=[o["stl_sha256_16"] for o in subset],
                            mass_max_g=max(o["mass_est_15infill_g"] for o in subset),
                            mass_sum_g=sum(o["mass_est_15infill_g"] for o in subset),
                            grasp_w_max_mm=max(o["grasp_min_width_mm"] for o in subset),
                            extent_max_mm=max(o["max_extent_mm"] for o in subset))
                break
        if pick:
            break
    g_capacity = pick is not None
    if not g_capacity:
        print("FATAL: (N_ep, H_max) 조합 실패")
        sys.exit(2)

    # ---- source 더미 높이 추정 (기술 참고치, 게이트 아님) ----
    a_src = src["L_int"] ** 2
    v_bulk = pick["v_solid_cm3"] * 1e-6 / PHI
    pile_h_avg = v_bulk / a_src
    # 벽 = 기슭 산란 억제용 (피크 봉쇄 아님). 이탈 실측·수정은 yt1 물리 몫.
    src_wall_h = float(np.clip(np.ceil((pile_h_avg + 0.02) * 100) / 100, 0.05, 0.08))
    g_payload = pick["mass_max_g"] <= PAYLOAD_CAP_G
    g_grasp = pick["grasp_w_max_mm"] <= GRASP_OPEN_CAP_MM

    # ---- 트레이 비겹침 (외벽 포함, y=0 대칭 배치이므로 y-간격으로 판정) ----
    src_outer_ymin = src["center"][1] - src["L_int"] / 2 - WALL_T
    bin_outer_ymax = binr["center"][1] + binr["L_int"] / 2 + WALL_T
    tray_gap_m = float(src_outer_ymin - bin_outer_ymax)
    g_no_overlap = tray_gap_m >= 0.02  # 최소 20 mm 통로

    design = dict(
        tool="p26_y1_testbed_design_author",
        case="y1_d453",
        manifest_sha16=man_sha,
        constants=dict(cell_m=CELL, annulus_m=[R_MIN, R_MAX], margin_m=MARGIN,
                       az_src_deg=AZ_SRC_DEG, az_bin_deg=AZ_BIN_DEG,
                       wall_t_m=WALL_T, phi=PHI, phi_range=list(PHI_RANGE),
                       rho_min_at_phi_lo=RHO_MIN_AT_PHI_LO,
                       rho_max_at_phi_nom=RHO_MAX_AT_PHI_NOM),
        source=dict(**{k: v for k, v in src.items() if k != "slack"},
                    wall_h_m=src_wall_h, role="선창 축소판 — 초기 더미 영역"),
        bin=dict(**{k: v for k, v in binr.items() if k != "slack"},
                 wall_h_m=pick["h_max"],
                 h_max_m=pick["h_max"],
                 role="목적지 bin — H_max 상한이 place 선택을 유효화"),
        episode=pick,
        pile_estimate=dict(v_bulk_l=v_bulk * 1e3, pile_h_avg_m=pile_h_avg,
                           note="phi 가정 기반 참고치 — 게이트 아님, yt1 실측이 권위"),
        tray_gap_m=tray_gap_m,
        gates=dict(g_reach_src=g_reach_src, g_reach_bin=g_reach_bin,
                   g_capacity=g_capacity, g_payload=g_payload,
                   g_grasp_open=g_grasp, g_no_overlap=g_no_overlap),
        assumptions=[
            "phi=0.55는 가정(실측 아님) — rho는 phi 0.50/0.60 감도 병기",
            "질량은 manifest 15% infill 추정치 — 실측 전 sim 질량 주장 금지(o1 규약)",
            "annulus는 D440/t3w sim 결과 — 실기 재검증 전 실기 주장 금지",
        ],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "y1_design.json"
    out_json.write_text(json.dumps(design, indent=2, ensure_ascii=False))

    # ---- D324 단일 프레임 레이아웃 진단 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, ax = plt.subplots(figsize=(8, 8))
    for r, ls in [(R_MIN, "--"), (R_MAX, "--")]:
        ax.add_patch(Circle((0, 0), r, fill=False, ls=ls, color="tab:gray"))
    for reg, color, label in [(src, "tab:blue", "source tray"),
                              (binr, "tab:orange", "bin (H_max)")]:
        cx, cy = reg["center"]
        L = reg["L_int"]
        ax.add_patch(Rectangle((cx - L / 2 - WALL_T, cy - L / 2 - WALL_T),
                               L + 2 * WALL_T, L + 2 * WALL_T,
                               fill=False, color=color, lw=2.5))
        n = reg["n_cells"]
        for i in range(n + 1):
            o = -L / 2 + i * CELL
            ax.plot([cx + o, cx + o], [cy - L / 2, cy + L / 2], color=color, lw=0.4)
            ax.plot([cx - L / 2, cx + L / 2], [cy + o, cy + o], color=color, lw=0.4)
        ax.annotate(f"{label}\n{n}x{n} cells, L={L*1000:.0f}mm",
                    (cx, cy - L / 2 - 0.018), ha="center", fontsize=9, color=color)
    ax.plot(0, 0, "k^", ms=12)
    ax.annotate("RoArm base", (0, -0.02), ha="center", fontsize=9)
    ax.set_xlim(-0.05, 0.40)
    ax.set_ylim(-0.28, 0.28)
    ax.set_aspect("equal")
    ax.set_title(f"y1_d453 testbed layout — N_ep={pick['n_ep']}, "
                 f"H_max={pick['h_max']*1000:.0f}mm, rho={pick['rho']:.2f}")
    ax.grid(alpha=0.2)
    fig.savefig(OUT_DIR / "y1_layout.png", dpi=150, bbox_inches="tight")

    print(json.dumps(dict(gates=design["gates"], source=design["source"],
                          bin=design["bin"], tray_gap_m=tray_gap_m,
                          episode={k: v for k, v in pick.items()
                                   if not k.startswith("subset")},
                          pile=design["pile_estimate"]),
                     indent=2, ensure_ascii=False))
    print("design_json_sha16:",
          hashlib.sha256(out_json.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
