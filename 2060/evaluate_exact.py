"""Compare every implementation with float64 halfspace intersections on a shared set of pairs.

Pairs: 300 random pairs per case, plus every pair where any two implementations
disagree in IoU by more than 1e-4. The metric is the absolute IoU error against
the exact IoU, which is scale free and well defined for touching boxes.
"""
import csv, sys
import numpy as np, torch
from scipy.spatial import ConvexHull
from exact_volume import exact

cases = torch.load(sys.argv[1])
outs = {k: torch.load(p) for k, p in (a.split("=") for a in sys.argv[3:])}
impls = {"main_cpu": ("base", "cpu"), "main_cuda": ("base", "cuda"), "pr2059_cuda": ("opt", "cuda"),
         "new_cpu": ("polycpu", "cpu"), "new_cuda": ("polycpu", "cuda")}
g = torch.Generator().manual_seed(0); summary, rows = [], []
for name, (b1, b2) in cases.items():
    ious = {k: outs[b][name][d][1] for k, (b, d) in impls.items()}
    stack = torch.stack(list(ious.values()))
    pick = set(map(tuple, torch.stack([torch.randint(0, b1.shape[0], (300,), generator=g),
                                       torch.randint(0, b2.shape[0], (300,), generator=g)], 1).tolist()))
    pick |= set(map(tuple, ((stack.max(0).values - stack.min(0).values) > 1e-4).nonzero().tolist()))
    hull1 = [ConvexHull(b.double().numpy()).volume for b in b1]
    hull2 = [ConvexHull(b.double().numpy()).volume for b in b2]
    errs = {k: [] for k in impls}
    for n, m in sorted(pick):
        ve = exact(b1[n].double().numpy(), b2[m].double().numpy())
        if ve != ve:
            continue
        ie = ve / (hull1[n] + hull2[m] - ve)
        row = {"case": name, "n": n, "m": m, "exact_volume": ve, "exact_iou": ie}
        for k in impls:
            row[k + "_iou"] = float(ious[k][n, m]); errs[k].append(abs(row[k + "_iou"] - ie))
        rows.append(row)
    s = {"case": name, "pairs_checked": len(errs["new_cpu"])}
    for k in impls:
        e = np.array(errs[k]); s[f"{k}_max_iou_err"] = float(e.max()); s[f"{k}_pairs_over_1e-3"] = int((e > 1e-3).sum())
    summary.append(s)
    print(f"{name:32s} n={s['pairs_checked']:5d} " + " ".join(f"{k}={s[k + '_max_iou_err']:.1e}/{s[k + '_pairs_over_1e-3']}" for k in impls))
with open(sys.argv[2] + "/exact-comparison.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
with open(sys.argv[2] + "/exact-pairs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
