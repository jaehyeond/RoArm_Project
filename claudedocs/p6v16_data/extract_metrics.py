#!/usr/bin/env python3
"""Parse P6v16 train log -> CSV + side-by-side compare with P6v14c."""
import re, csv, sys
from pathlib import Path

THIS = Path(__file__).parent
LOG = THIS / "train_p6v16.out"
CSV_OUT = THIS / "p6v16_metrics.csv"
P6V14C_CSV = THIS.parent / "p6v14c_data" / "p6v14c_metrics.csv"

ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
METRIC_RE = re.compile(r"Mean episode (\w+):\s*([\d.]+)")
ITERTIME_RE = re.compile(r"Iteration time:\s*([\d.]+)s")

current_iter = None
rows = {}
target_metrics = ["grasped_frac", "gripper_open_rate", "upright_rate",
                  "jackpot_fire_rate", "stage2_grasp_frac", "stage4_success_frac"]

with LOG.open() as f:
    for line in f:
        m = ITER_RE.search(line)
        if m:
            current_iter = int(m.group(1))
            rows.setdefault(current_iter, {"iter": current_iter})
            continue
        if current_iter is None:
            continue
        m = METRIC_RE.search(line)
        if m:
            k, v = m.group(1), float(m.group(2))
            if k in target_metrics:
                rows[current_iter][k] = v
            continue
        m = ITERTIME_RE.search(line)
        if m:
            rows[current_iter]["iter_time_s"] = float(m.group(1))

iters = sorted(rows.keys())
fields = ["iter", "iter_time_s"] + target_metrics
with CSV_OUT.open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for i in iters:
        w.writerow({k: rows[i].get(k, "") for k in fields})

print(f"Parsed {len(iters)} iters -> {CSV_OUT}")

# Load P6v14c for comparison
p14c = {}
if P6V14C_CSV.exists():
    with P6V14C_CSV.open() as f:
        for r in csv.DictReader(f):
            p14c[int(r["iter"])] = r

print(f"\n=== P6v16 vs P6v14c (key iters) ===")
print(f"{'iter':>5}  {'stg4_succ':>20}  {'gripopen':>20}  {'grasped':>20}  {'jackpot':>20}")
print(f"{'':>5}  {'v15 / v14c':>20}  {'v15 / v14c':>20}  {'v15 / v14c':>20}  {'v15 / v14c':>20}")
critical = [0, 1, 2, 5, 10, 20, 50, 100, 200, 499]
for it in critical:
    if it not in rows:
        continue
    r = rows[it]
    rc = p14c.get(it, {})
    def fmt(k, fmt_str="{:.4f}"):
        v15 = r.get(k, "")
        v14 = rc.get(k, "")
        s15 = fmt_str.format(v15) if isinstance(v15, (int, float)) and v15 != "" else "  --  "
        s14 = fmt_str.format(float(v14)) if v14 not in ("", None) else "  --  "
        return f"{s15} / {s14}"
    print(f"{it:>5}  {fmt('stage4_success_frac'):>20}  "
          f"{fmt('gripper_open_rate'):>20}  "
          f"{fmt('grasped_frac'):>20}  "
          f"{fmt('jackpot_fire_rate'):>20}")

print(f"\n=== Hypothesis verdict ===")
def get(d, k, default=0.0):
    v = d.get(k, default)
    if isinstance(v, str):
        try: return float(v)
        except: return default
    return v

if 0 in rows and 1 in rows:
    g0_v15 = get(rows[0], 'gripper_open_rate')
    g1_v15 = get(rows[1], 'gripper_open_rate')
    g0_v14c = get(p14c.get(0, {}), 'gripper_open_rate')
    g1_v14c = get(p14c.get(1, {}), 'gripper_open_rate')
    print(f"gripper_open iter0→1:  v15 {g0_v15:.4f}→{g1_v15:.4f} ({100*(g1_v15-g0_v15)/max(g0_v15,1e-9):+.1f}%)")
    print(f"                       v14c {g0_v14c:.4f}→{g1_v14c:.4f} ({100*(g1_v14c-g0_v14c)/max(g0_v14c,1e-9):+.1f}%)")

if 0 in rows and 10 in rows:
    s0 = get(rows[0], 'stage4_success_frac')
    s10 = get(rows[10], 'stage4_success_frac')
    s0_14c = get(p14c.get(0, {}), 'stage4_success_frac')
    s10_14c = get(p14c.get(10, {}), 'stage4_success_frac')
    print(f"stage4_success iter0→10: v15 {s0:.4f}→{s10:.4f}")
    print(f"                         v14c {s0_14c:.4f}→{s10_14c:.4f}")

if 499 in rows:
    s499 = get(rows[499], 'stage4_success_frac')
    g499 = get(rows[499], 'stage2_grasp_frac')
    s499_14c = get(p14c.get(499, {}), 'stage4_success_frac')
    g499_14c = get(p14c.get(499, {}), 'stage2_grasp_frac')
    print(f"final iter499:  v15 stage4={s499:.4f} grasp={g499:.4f}")
    print(f"                v14c stage4={s499_14c:.4f} grasp={g499_14c:.4f}")
