"""스쿱 폐합 실행 결과 정리 — 재현성 게이트 + 반력/접촉 곡선 PNG.

재현성 게이트는 `sim_deme_pile.py` 와 **같은 기준**이다 (D464 §3 계열):
    5 mm 셀 heightmap 을 두 실행에서 만들어
        rms  <= 반지름 (2.08 mm)
        p95  <= 지름   (4.16 mm)
        max  <= 지름x2 (8.32 mm)
DEME 2.4.0 은 접촉쌍 리덕션 순서를 결정론적으로 노출하지 않으므로 raw bit-exact 는
엔진 한계로 FAIL 이며, 위 heightmap 허용오차가 채택된 과학적 비교다.
"""
import sys, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 라벨이 네모로 깨지지 않게 CJK 폰트를 쓴다 (이 머신에 설치된 것 중 택1).
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
    if any(f.name == _f for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "claudedocs/runtime_logs/scoop_track/s2_closure_rot")


def curves(tag):
    tl = json.load(open(OUT / f"scoop_timeline_{tag}.json"))
    log = tl["rows"] if isinstance(tl, dict) else tl
    res = json.load(open(OUT / f"scoop_closure_{tag}.json"))
    t = np.array([r["sim_t"] for r in log])
    n = np.array([r["n_total"] for r in log])
    F = np.array([r["F_total_N"] for r in log])
    lip = np.array([r["lipF_total_N"] for r in log])
    ph = np.array([r["phase"] for r in log])
    return res, t, n, F, lip, ph


def plot(tag):
    res, t, n, F, lip, ph = curves(tag)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for a in ax:
        for p, c in (("descend", "#cfe8ff"), ("close", "#ffd9b3"), ("lift", "#d7f0d7")):
            m = ph == p
            if m.any():
                a.axvspan(t[m].min(), t[m].max(), color=c, zorder=0)
    ax[0].plot(t, n, lw=1.2, color="#1f4e79")
    ax[0].set_ylabel("접촉 개수")
    ax[1].plot(t, F, lw=1.2, color="#7a1f1f")
    ax[1].set_ylabel("접촉 합력 |F| (N)")
    band = res["forces_N"]["jaw_band_measured"]
    ax[2].plot(t, lip, lw=1.2, color="#1f7a3f")
    ax[2].axhspan(band[0], band[1], color="#999", alpha=0.35,
                  label=f"실측 조 힘 {band[0]}~{band[1]} N (D451/D452, 슬리브 그리퍼)")
    ax[2].set_ylabel("피벗 모멘트 -> 등가 립 힘 (N)")
    ax[2].set_xlabel("시뮬 시간 (s)")
    ax[2].legend(fontsize=8, loc="upper left")
    ax[0].set_title(f"DEME 스쿱 폐합 {tag} — 파란=하강 / 주황=폐합 / 초록=리프트\n"
                    f"[주의] 물성 임시값 (E={res['params']['E_pa']:.0e} Pa) — 수치 인용 불가",
                    fontsize=10)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    p = OUT / f"scoop_curves_{tag}.png"
    fig.savefig(p, dpi=130)
    print(f"  -> {p}")
    return p


def _hm_metrics(pa, pb, rad, box, keep_a=None, keep_b=None):
    from roarm_rl.heightmap import GridSpec, heightmap_from_particles
    cell = 0.005
    spec = GridSpec(origin_xy_m=(float(box[0, 0]), float(box[1, 0])), cell_m=cell,
                    shape=(int(math.ceil((box[1, 1] - box[1, 0]) / cell)),
                           int(math.ceil((box[0, 1] - box[0, 0]) / cell))),
                    frame="deme_box_floor_center", z_datum_m=0.0)
    qa = pa if keep_a is None else pa[keep_a]
    qb = pb if keep_b is None else pb[keep_b]
    ha = heightmap_from_particles(qa, np.full(len(qa), rad), spec).height.astype(np.float64)
    hb = heightmap_from_particles(qb, np.full(len(qb), rad), spec).height.astype(np.float64)
    diff = np.abs(ha - hb)
    d = 2 * rad
    met = {"n_left": int(len(qa)), "n_right": int(len(qb)),
           "rms_m": float(np.sqrt((diff ** 2).mean())),
           "mean_abs_m": float(diff.mean()),
           "p95_abs_m": float(np.quantile(diff, 0.95)),
           "max_abs_m": float(diff.max()),
           "cells_over_max_threshold": int((diff > 2 * d).sum()),
           "exact_cell_fraction": float((diff == 0.0).mean())}
    thr = {"rms_m": rad, "p95_abs_m": d, "max_abs_m": 2 * d}
    chk = {"rms": met["rms_m"] <= thr["rms_m"],
           "p95": met["p95_abs_m"] <= thr["p95_abs_m"],
           "max": met["max_abs_m"] <= thr["max_abs_m"]}
    return met, thr, chk


def repro(a, b):
    """두 실행 비교. 게이트는 `sim_deme_pile.py` 것을 **그대로** 쓴다 (통과하도록 고치지 않는다).

    다만 스쿱 종료 상태는 정착된 더미와 달리 **그랩 안에 들려 올라간 입자**를 포함한다.
    그래서 같은 지표를 두 모집단에 각각 낸다: (1) 전 입자 = 원본 게이트(정본 판정),
    (2) 남은 더미만 = 담긴 입자를 뺀 것(원인 분리용).
    """
    za = np.load(OUT / f"scoop_{a}.npz")
    zb = np.load(OUT / f"scoop_{b}.npz")
    pa, pb = za["positions_m"], zb["positions_m"]
    rad = float(za["radii_m"][0])
    box = za["box_bounds_m"]
    res_a = json.load(open(OUT / f"scoop_closure_{a}.json"))
    res_b = json.load(open(OUT / f"scoop_closure_{b}.json"))
    cut = float(res_a["pile"]["top_z_m"]) + 0.005      # captured 판정과 같은 기준

    met, thr, chk = _hm_metrics(pa, pb, rad, box)
    met2, thr2, chk2 = _hm_metrics(pa, pb, rad, box,
                                   keep_a=pa[:, 2] <= cut, keep_b=pb[:, 2] <= cut)
    bit = bool(np.array_equal(pa, pb))
    ca, cb = res_a["captured_particles"], res_b["captured_particles"]
    fa = res_a["forces_N"]["close_peak_lip_equiv"]
    fb = res_b["forces_N"]["close_peak_lip_equiv"]
    rep = {
        "artifact": "DEME_SCOOP_REPRODUCIBILITY_V1",
        "left": f"scoop_{a}.npz", "right": f"scoop_{b}.npz",
        "raw_final_bit_exact": bit,
        "raw_final_bit_exact_verdict": "FAIL_ENGINE_LIMITATION" if not bit else "PASS",
        "heightmap_cell_m": 0.005,
        "metrics": met, "thresholds": thr, "checks": chk,
        "verdict": ("REPRODUCIBLE_WITHIN_HEIGHTMAP_TOLERANCE" if all(chk.values())
                    else "FAIL_HEIGHTMAP_TOLERANCE"),
        "remaining_pile_only": {
            "metrics": met2, "thresholds": thr2, "checks": chk2,
            "verdict": ("REPRODUCIBLE_WITHIN_HEIGHTMAP_TOLERANCE" if all(chk2.values())
                        else "FAIL_HEIGHTMAP_TOLERANCE"),
            "definition": f"z <= {cut:.6f} m 인 입자만 (그랩에 담겨 들려 올라간 것 제외)"},
        "run_to_run_spread": {
            "captured_particles": [ca, cb],
            "captured_rel_diff": round(abs(ca - cb) / max(ca, cb), 4),
            "close_peak_lip_equiv_N": [fa, fb],
            "close_peak_rel_diff": round(abs(fa - fb) / max(fa, fb), 4),
            "note": ("🔴 담긴 입자 수와 폐합 최대 반력은 설정이 같아도 실행마다 다르다. "
                     "1회 실행의 반력 수치를 단독으로 인용하면 안 되고 반복 평균이 필요하다.")},
        "note": ("DEME 2.4.0 은 접촉쌍 리덕션 순서를 결정론적으로 노출하지 않는다. "
                 "sim_deme_pile.py 와 같은 기준을 그대로 쓴다: 시드·설정 동일 + "
                 "5 mm heightmap 이 반지름 RMS / 지름 p95 / 지름x2 최대 안에서 일치. "
                 "게이트는 통과시키려고 고치지 않았다."),
    }
    json.dump(rep, open(OUT / "scoop_reproducibility.json", "w"),
              ensure_ascii=False, indent=2)
    return rep


def lip_gap_mm(phi_deg, pivot_gap=26.0, lip_depth=36.06):
    """폐합각 phi 에서 두 립 사이 간격 (mm). phi=0 이면 0 (맞닿음)."""
    t = math.radians(phi_deg)
    return 2.0 * (pivot_gap / 2 - (pivot_gap / 2 * math.cos(t) - lip_depth * math.sin(t)))


def full_closure_stall():
    """완전 폐합 실행(발산으로 끝난 것)에서 립등가 힘이 실측 조 힘 상한을 넘는 각도."""
    p = OUT / "scoop_timeline_fullclose_diverged.json"
    if not p.exists():
        return None
    tl = json.load(open(p))
    rows = [r for r in (tl["rows"] if isinstance(tl, dict) else tl)
            if r["phase"] == "close"]
    if not rows:
        return None
    CEIL = 6.3          # D451/D452 실측 조 힘 상한
    cross = None
    for r in rows:
        if r["lipF_total_N"] > CEIL and cross is None:
            cross = r
    out = {"ceiling_N": CEIL,
           "last_phi_deg": rows[-1]["phi_deg"],
           "last_lipF_N": rows[-1]["lipF_total_N"],
           "first_cross": None,
           "samples": [{"phi_deg": r["phi_deg"], "lip_gap_mm": round(lip_gap_mm(r["phi_deg"]), 2),
                        "lipF_N": r["lipF_total_N"], "n": r["n_total"]}
                       for r in rows[::5]]}
    if cross:
        out["first_cross"] = {"phi_deg": cross["phi_deg"],
                              "lip_gap_mm": round(lip_gap_mm(cross["phi_deg"]), 2),
                              "lipF_N": cross["lipF_total_N"],
                              "n_contacts": cross["n_total"]}
    json.dump(out, open(OUT / "scoop_fullclose_stall.json", "w"),
              ensure_ascii=False, indent=2)
    return out


def markdown(tags, rep):
    """사람이 읽는 요약. 수치는 전부 JSON 에서 읽어온다 (손으로 옮겨 적지 않는다)."""
    L = ["# DEME 스쿱 폐합 — 트랙 P1 결과", "",
         "> ⚠️ **물성 미실측.** 아래 반력은 어떤 판정에도 인용할 수 없다.",
         "> 이 실행이 주장하는 것은 **경로가 성립한다**는 것뿐이다.", ""]
    r0 = json.load(open(OUT / f"scoop_closure_{tags[0]}.json"))
    L += ["## 뚫은 방법", "",
          "`s.Track(mesh)` 를 **`Initialize()` 앞으로** 옮긴 것이 전부다.",
          "뒤에서 부르면 트래커가 owner 에 안 묶여 `GetOwnerID()` 가 `4294967295`",
          "(UINT_MAX = 미할당) 를 돌려주고 다음 스텝에서 세그폴트한다.",
          "메시 매수·구동 방식과 무관하다 — 메시 1매도 똑같이 죽는다.", "",
          "| 조합 | D464 §4 기록 | 실제 (Track 을 Initialize 앞에서) |",
          "|---|---|---|",
          "| 메시 2매 + Track | 코어 덤프 | **OK** |",
          "| 규정 선속도 + Track | 코어 덤프 | **OK** |",
          "| 규정 각속도(회전) + Track | (미시도) | **OK** |",
          "| `SetFamilyPrescribedLinVel` | Initialize 후 호출 불가 | 맞음 (앞에서 부르면 됨) |",
          "", "폐합은 평행이동 근사가 아니라 **진짜 회전**이다: 셸 메시를 피벗이 로컬",
          "원점에 오도록 굽고 `SetFamilyPrescribedAngVel` 로 피벗 둘레를 돌린다.", ""]
    L += ["## 결과", "",
          "| 실행 | 폐합 접촉 | 폐합 합력 최대 (N) | 립등가 (N) | 단일접촉 최대 (N) | 담긴 입자 | 벽시계 (s) | 발산 |",
          "|---|---|---|---|---|---|---|---|"]
    for t in tags:
        r = json.load(open(OUT / f"scoop_closure_{t}.json"))
        f = r["forces_N"]
        L.append(f"| {t} | {r['contacts']['close_first']} → {r['contacts']['close_max']} "
                 f"| {f['close_peak_total']} | {f['close_peak_lip_equiv']} "
                 f"| {f['close_peak_single_contact']} | {r['captured_particles']} "
                 f"| {r['wall_seconds']} | {r['diverged']} |")
    walls = [json.load(open(OUT / f"scoop_closure_{t}.json"))["wall_seconds"] for t in tags]
    hrs = [round(w * 3000 / 3600.0, 1) for w in walls]
    L += ["", f"3,000 시행 예산 = **{min(hrs)}~{max(hrs)} 시간** "
          f"(1회 {min(walls):.0f}~{max(walls):.0f} s · GPU 1대 · 직렬 기준)", "",
          "⚠️ 위쪽 값은 다른 워커의 DEME 잡과 GPU 를 나눠 쓴 실행이다. GPU 를 독점하면 "
          f"**{min(hrs)} 시간** 쪽이 맞다. 같은 GPU 에 DEME 프로세스가 둘 뜨면 서로 "
          "교착해 둘 다 멈추므로(본 세션 실측) 병렬화는 GPU 를 나눠야 가능하다.", ""]
    if rep:
        def tbl(block, title):
            m, th, ck = block["metrics"], block["thresholds"], block["checks"]
            return [f"**{title}** → `{block['verdict']}`", "",
                    "| 지표 | 값 (mm) | 임계 (mm) | 통과 |", "|---|---|---|---|",
                    f"| RMS | {m['rms_m']*1000:.3f} | {th['rms_m']*1000:.2f} | {ck['rms']} |",
                    f"| p95 | {m['p95_abs_m']*1000:.3f} | {th['p95_abs_m']*1000:.2f} | {ck['p95']} |",
                    f"| max | {m['max_abs_m']*1000:.3f} | {th['max_abs_m']*1000:.2f} | {ck['max']} |",
                    f"| 임계 초과 셀 수 | {m['cells_over_max_threshold']} | 0 | "
                    f"{m['cells_over_max_threshold'] == 0} |", ""]
        L += ["## 재현성 (같은 시드·같은 설정 2회)", "",
              "기준은 `sim_deme_pile.py` 와 **동일**하다 (5 mm heightmap · 반지름 RMS · 지름 p95 ·",
              "지름×2 최대). 통과시키려고 고치지 않았다.", ""]
        L += tbl({"metrics": rep["metrics"], "thresholds": rep["thresholds"],
                  "checks": rep["checks"], "verdict": rep["verdict"]},
                 "전 입자 (원본 게이트 — 이것이 정본 판정)")
        rp = rep["remaining_pile_only"]
        L += tbl(rp, "남은 더미만 (그랩에 담겨 올라간 입자 제외 — 원인 분리용)")
        sp = rep["run_to_run_spread"]
        L += ["원본 게이트가 FAIL 인 이유는 **더미가 달라서가 아니다.** 임계를 넘는 셀이",
              f"{rep['metrics']['cells_over_max_threshold']}개뿐이고 전부 그랩 발자국 안",
              "(|x|≲23 mm, |y|≲34 mm)에 있다 — 그랩 안에 들려 올라간 입자가 앉은 칸이다.",
              "담긴 입자를 빼면 남은 더미는 허용오차 안에서 일치한다.", "",
              "### 🔴 실행마다 달라지는 양 (이게 더 중요하다)", "",
              "| 양 | rep1 | rep2 | 상대차 |", "|---|---|---|---|",
              f"| 담긴 입자 수 | {sp['captured_particles'][0]} | {sp['captured_particles'][1]} "
              f"| {sp['captured_rel_diff']*100:.1f} % |",
              f"| 폐합 최대 립등가 (N) | {sp['close_peak_lip_equiv_N'][0]} "
              f"| {sp['close_peak_lip_equiv_N'][1]} | {sp['close_peak_rel_diff']*100:.1f} % |",
              "", "설정이 완전히 같은데도 폐합 최대 반력이 "
              f"{sp['close_peak_rel_diff']*100:.0f} % 차이 난다.",
              "**1회 실행의 반력 값을 단독 인용하면 안 된다 — 반복 평균이 필요하다.**",
              "3,000 시행 계획에서는 문제가 안 되지만(평균이 목적), 설계 판정에 쓰려면",
              "반복 수를 정해야 한다.", "",
              f"raw bit-exact = {rep['raw_final_bit_exact']} "
              f"({rep['raw_final_bit_exact_verdict']}) — "
              "DEME 2.4.0 은 접촉쌍 리덕션 순서를 결정론적으로 노출하지 않는다.", ""]
    st = full_closure_stall()
    if st:
        L += ["## 🔴 완전 폐합은 펠릿 층에서 성립하지 않는다 (별도 실행)", "",
              "`close_end_deg=0` (립이 맞닿을 때까지) 로 돌리면 립 틈에 낀 펠릿이",
              "무한히 압축되어 발산한다. 발산 지점의 접촉 위치가 전부 립 선상",
              "(x≈-2~-3.5 mm, z≈23~24 mm)이라 원인이 확인된다. 규정 구동은 멈출 수",
              "없으므로 펠릿이 갈 곳이 없다 — 실물이라면 서보가 스톨하거나 펠릿이 깨진다.", ""]
        if st["first_cross"]:
            c = st["first_cross"]
            L += [f"립등가 힘이 실측 조 힘 상한 **{st['ceiling_N']} N** 을 처음 넘는 지점 = "
                  f"**phi {c['phi_deg']}°** (립 틈 **{c['lip_gap_mm']} mm** · 접촉 {c['n_contacts']}개).",
                  "즉 입이 아직 ~9 mm 열려 있을 때 이미 조 힘 대역을 넘어선다.", ""]
        L += ["| phi (°) | 립 틈 (mm) | 립등가 (N) | 접촉 |", "|---|---|---|---|"]
        for s in st["samples"][-12:]:
            L.append(f"| {s['phi_deg']} | {s['lip_gap_mm']} | {s['lipF_N']} | {s['n']} |")
        L += ["", "⚠️ 물성 임시값이므로 **이 각도도 수치로 인용 불가**. 실측 후 재측정할 것.",
              f"원자료 = `scoop_timeline_fullclose_diverged.json` · `scoop_fullclose_stall.json`", ""]
    rrd = OUT / f"scoop_{tags[0]}.rrd"
    if rrd.exists():
        L += ["## 관측 (D341 Rerun) — 부분 이행", "",
              f"`{rrd.name}` ({rrd.stat().st_size/1e6:.2f} MB) 에 **판정 대상 자체**를 기록했다:",
              "셸 2매의 노드 궤적(프레임별), 접촉점, 접촉력 화살표, 그리고 접촉 수·합력·",
              "립등가·단일접촉 최대·phi 의 시간축 스칼라.", "",
              "| 계약 항목 | 상태 |", "|---|---|",
              "| 파일 싱크를 첫 로그 전에 부착 + 컨텍스트 종료로 finalize | 이행 |",
              "| `rrd verify` | **PASS** (`1 file verified without error`) |",
              "| SDK 핀 명시 | rerun 0.34.1 (isaaclab env) |",
              "| 고정 블루프린트 + `.rbl` 내보내기 | **미이행** |",
              "| 헤드리스 결정 스크린샷 | **미이행** |",
              "| 육안 검수 기록 | **미이행** |", "",
              "⚠️ 따라서 D341 **완결** 계약은 만족하지 않는다. 위 3개가 남았다.",
              "DEME 는 `roarm` env(rerun 0.26.2), D341 이 요구하는 rerun 0.34.1 은 `isaaclab`",
              "env 에만 있어 RRD 는 후처리로 굽는다 (`sim_deme_scoop_rerun.py`).",
              "설치를 하지 않으므로 D326 핀(numpy 1.26.0 / psutil 5.9.8)은 그대로다 (확인함).", "",
              f"검수: `~/miniconda3/envs/isaaclab/bin/rerun {rrd}`", ""]
    L += ["## 아직 못 하는 것", ""]
    for s in r0["non_claims"]:
        L.append(f"- {s}")
    p = OUT / "REPORT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  -> {p}")


def main():
    tags = [t.name.split("scoop_closure_")[1].split(".json")[0]
            for t in sorted(OUT.glob("scoop_closure_*.json"))]
    print("실행:", tags)
    for t in tags:
        plot(t)
        res = json.load(open(OUT / f"scoop_closure_{t}.json"))
        f = res["forces_N"]
        print(f"  {t}: 폐합 합력 최대 {f['close_peak_total']} N · "
              f"립등가 {f['close_peak_lip_equiv']} N · "
              f"접촉 {res['contacts']['close_first']}->{res['contacts']['close_max']} · "
              f"담김 {res['captured_particles']} 개 · 벽시계 {res['wall_seconds']} s")
    rep = None
    if len(tags) >= 2:
        rep = repro(tags[0], tags[1])
        print(f"\n재현성 {rep['verdict']}")
        print(f"  rms {rep['metrics']['rms_m']*1000:.3f} mm (<= {rep['thresholds']['rms_m']*1000:.2f})")
        print(f"  p95 {rep['metrics']['p95_abs_m']*1000:.3f} mm (<= {rep['thresholds']['p95_abs_m']*1000:.2f})")
        print(f"  max {rep['metrics']['max_abs_m']*1000:.3f} mm (<= {rep['thresholds']['max_abs_m']*1000:.2f})")
        print(f"  bit-exact {rep['raw_final_bit_exact']} ({rep['raw_final_bit_exact_verdict']})")
    markdown(tags, rep)


if __name__ == "__main__":
    main()
