#!/usr/bin/env python3
"""Parse P6v16b (RPL alpha=0.10) train log -> CSV + side-by-side compare with
P6v14c (no RPL baseline) and P6v16 (alpha=0.3)."""
import re, csv
from pathlib import Path

THIS = Path(__file__).parent
LOG = THIS / "train_p6v16c.out"
CSV_OUT = THIS / "p6v16c_metrics.csv"
P6V14C_CSV = THIS.parent / "p6v14c_data" / "p6v14c_metrics.csv"
P6V16_CSV = THIS.parent / "p6v16_data" / "p6v16_metrics.csv"

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


def load_csv(p):
    d = {}
    if not p.exists():
        return d
    with p.open() as f:
        for r in csv.DictReader(f):
            d[int(r["iter"])] = r
    return d

p14c = load_csv(P6V14C_CSV)
p16 = load_csv(P6V16_CSV)


def gf(d, k, default=0.0):
    v = d.get(k, default)
    if isinstance(v, str):
        try:
            return float(v)
        except Exception:
            return default
    return v


print("\n=== stage4_success_frac comparison (KEY hypothesis test) ===")
print(f"{'iter':>5}  {'P6v14c (no RPL)':>17}  {'P6v16 (α=0.30)':>17}  {'P6v16b (α=0.10)':>17}")
for it in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 499]:
    a = gf(p14c.get(it, {}), 'stage4_success_frac')
    b = gf(p16.get(it, {}), 'stage4_success_frac')
    c = gf(rows.get(it, {}), 'stage4_success_frac')
    print(f"{it:>5}  {a:>17.4f}  {b:>17.4f}  {c:>17.4f}")

print("\n=== gripper_open_rate (forgetting indicator) ===")
print(f"{'iter':>5}  {'P6v14c':>17}  {'P6v16 (α=0.30)':>17}  {'P6v16b (α=0.10)':>17}")
for it in [0, 1, 2, 5, 10, 50, 100, 499]:
    a = gf(p14c.get(it, {}), 'gripper_open_rate')
    b = gf(p16.get(it, {}), 'gripper_open_rate')
    c = gf(rows.get(it, {}), 'gripper_open_rate')
    print(f"{it:>5}  {a:>17.4f}  {b:>17.4f}  {c:>17.4f}")

print("\n=== Verdict (α=0.10 hypothesis test) ===")
if 10 in rows:
    s10 = gf(rows[10], 'stage4_success_frac')
    s10_14c = gf(p14c.get(10, {}), 'stage4_success_frac')
    s10_16 = gf(p16.get(10, {}), 'stage4_success_frac')
    print(f"iter 10 stage4: P6v14c={s10_14c:.4f}  P6v16(α0.3)={s10_16:.4f}  P6v16b(α0.10)={s10:.4f}")
    if s10 >= 0.20:
        print(f"VERDICT: PASS  (stage4 iter10={s10:.4f} >= 0.20 gate)")
    elif s10 >= 0.05:
        print(f"VERDICT: MARGINAL  (stage4 iter10={s10:.4f}, between 0.05 and 0.20)")
    else:
        print(f"VERDICT: FAIL  (stage4 iter10={s10:.4f} < 0.05)")

if 499 in rows:
    s = gf(rows[499], 'stage4_success_frac')
    s_14c = gf(p14c.get(499, {}), 'stage4_success_frac')
    s_16 = gf(p16.get(499, {}), 'stage4_success_frac')
    print(f"iter 499 stage4: P6v14c={s_14c:.4f}  P6v16(α0.3)={s_16:.4f}  P6v16b(α0.10)={s:.4f}")
