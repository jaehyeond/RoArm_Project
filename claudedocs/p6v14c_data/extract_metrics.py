#!/usr/bin/env python3
"""Parse P6v14c train log -> CSV with iter x metric trajectory."""
import re, csv, sys
from pathlib import Path

LOG = Path(__file__).parent / "train_p6v14c.out"
CSV_OUT = Path(__file__).parent / "p6v14c_metrics.csv"

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
print(f"\n=== Critical iters ===")
critical = [0, 1, 5, 10, 20, 50, 100, 200, 300, 400, 499]
print(f"{'iter':>5} {'stg4_succ':>10} {'stg2_grasp':>11} {'jackpot':>9} {'grasped':>8} {'gripopen':>9} {'upright':>8}")
for it in critical:
    if it in rows:
        r = rows[it]
        print(f"{it:>5} {r.get('stage4_success_frac', 0):>10.4f} {r.get('stage2_grasp_frac', 0):>11.4f} "
              f"{r.get('jackpot_fire_rate', 0):>9.4f} {r.get('grasped_frac', 0):>8.4f} "
              f"{r.get('gripper_open_rate', 0):>9.4f} {r.get('upright_rate', 0):>8.4f}")

# Compute collapse rate
print(f"\n=== Collapse analysis ===")
if 0 in rows and 5 in rows:
    s0 = rows[0].get('stage4_success_frac', 0)
    s5 = rows[5].get('stage4_success_frac', 0)
    print(f"stage4 iter0={s0:.4f} -> iter5={s5:.4f} (delta={s5-s0:+.4f}, {100*(s5-s0)/max(s0,1e-9):+.1f}%)")
if 0 in rows and 50 in rows:
    s0 = rows[0].get('stage4_success_frac', 0)
    s50 = rows[50].get('stage4_success_frac', 0)
    print(f"stage4 iter0={s0:.4f} -> iter50={s50:.4f} (delta={s50-s0:+.4f}, {100*(s50-s0)/max(s0,1e-9):+.1f}%)")
if 0 in rows and 499 in rows:
    s0_g = rows[0].get('stage2_grasp_frac', 0)
    s499_g = rows[499].get('stage2_grasp_frac', 0)
    print(f"stage2_grasp iter0={s0_g:.4f} -> iter499={s499_g:.4f} (delta={s499_g-s0_g:+.4f})")
