"""Build CSV summaries and the comparison figure from the captured JSON files."""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = Path(__file__).resolve().parent
load = lambda name: json.loads((here / name).read_text())

# --- issue #1805 results -------------------------------------------------
issue = {"baseline": load("issue1805-baseline.json"), "fixed": load("issue1805-fixed.json")}
rows = []
for label, data in issue.items():
    for order, direction in zip(("(a, b)", "(b, a)"), data["directions"]):
        (cv, ci), (gv, gi) = direction["cpu"], direction["cuda"]
        rows.append({"build": label, "order": order, "cpu_volume": cv, "cpu_iou": ci,
                     "cuda_volume": gv, "cuda_iou": gi,
                     "abs_volume_error": abs(gv - cv), "abs_iou_error": abs(gi - ci)})
with open(here / "issue1805-results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# --- clipping trace ----------------------------------------------------------
cpu = load("native-cpu-counts.json")
gpu = next(r for r in load("native-gpu-counts.json") if r["capacity"] == 768)
with open(here / "clip-counts.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["planes_applied", "cpu_side1", "cpu_side2", "cuda_side1", "cuda_side2"])
    for p in range(7):
        w.writerow([p, cpu["trace"][0][p], cpu["trace"][1][p], gpu["trace"][0][p], gpu["trace"][1][p]])

# --- benchmarks ----------------------------------------------------------
bench = {k: {c["case"]: c for c in load(f"benchmark-{k}.json")["cases"]}
         for k in ("baseline", "first-commit", "fixed")}
mib = 2 ** 20
brows = []
for case, fixed in bench["fixed"].items():
    base = bench["baseline"].get(case)
    first = bench["first-commit"][case]
    brows.append({
        "case": case, "pairs": fixed["pairs"],
        "baseline_ms": round(base["cuda_event_median_ms"], 4) if base else "",
        "first_commit_ms": round(first["cuda_event_median_ms"], 4),
        "fixed_ms": round(fixed["cuda_event_median_ms"], 4),
        "fixed_over_baseline": round(fixed["cuda_event_median_ms"] / base["cuda_event_median_ms"], 3) if base else "",
        "baseline_first_call_device_mib": round(base["device_usage_first_delta"] / mib) if base else "",
        "fixed_first_call_device_mib": round(fixed["device_usage_first_delta"] / mib),
        "fixed_torch_peak_allocated_mib": round(fixed["torch_allocated_peak"] / mib, 1),
        "fixed_matches_cpu_1e-5": all(fixed["cpu_cuda_allclose_1e_5"]),
    })
with open(here / "benchmark-results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(brows[0])); w.writeheader(); w.writerows(brows)

# --- figure ----------------------------------------------------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BEFORE, AFTER, REF = "#eb6834", "#2a78d6", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURF,
                         gridspec_kw={"width_ratios": [1, 1.6, 1]})
for ax in axes:
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
width, gap = 0.36, 0.02

def bars(ax, xs, before, after):
    for x, b, a in zip(xs, before, after):
        if b is not None:
            ax.bar(x - width / 2 - gap / 2, b, width, color=BEFORE, edgecolor=SURF, linewidth=1)
        ax.bar(x + width / 2 + gap / 2, a, width, color=AFTER, edgecolor=SURF, linewidth=1)

# (a) IoU on the issue input
ax = axes[0]
before = [r["cuda_iou"] for r in rows if r["build"] == "baseline"]
after = [r["cuda_iou"] for r in rows if r["build"] == "fixed"]
bars(ax, [0, 1], before, after)
ax.axhline(rows[0]["cpu_iou"], color=REF, linestyle="--", linewidth=1.2)
ax.text(-0.45, 0.48, f"Dashed line: CPU reference ({rows[0]['cpu_iou']:.4f})", va="top", ha="left", color=INK2, fontsize=9)
for x, b, a in zip([0, 1], before, after):
    for xx, v in ((x - width / 2, b), (x + width / 2, a)):
        near_ref = abs(v - rows[0]["cpu_iou"]) < 0.03
        ax.text(xx, v / 2 if near_ref else v, f"{v:.3f}", ha="center",
                va="center" if near_ref else "bottom", fontsize=9,
                color="#ffffff" if near_ref else INK)
ax.set_xticks([0, 1], ["iou_box3d(a, b)", "iou_box3d(b, a)"])
ax.set_ylabel("IoU"); ax.set_ylim(0, 0.5)
ax.set_title("#1805 input: CUDA IoU vs CPU", loc="left", fontsize=11, color=INK)

# (b) latency
cases = ["normal_1x1", "normal_16x16", "normal_30x100", "normal_100x100", "high_overlap_100x100"]
labels = ["1x1", "16x16", "30x100", "100x100", "100x100\nhigh overlap"]
ax = axes[1]
b = [bench["baseline"][c]["cuda_event_median_ms"] for c in cases]
a = [bench["fixed"][c]["cuda_event_median_ms"] for c in cases]
bars(ax, range(len(cases)), b, a)
for x, (bv, av) in enumerate(zip(b, a)):
    ax.text(x + width / 2, av, f"{av / bv:.2f}x", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(len(cases)), labels)
ax.set_ylabel("Median latency (ms)")
ax.set_title("Latency (labels: fixed / baseline)", loc="left", fontsize=11, color=INK)

# (c) memory
ax = axes[2]
mcases = ["normal_1x1", "normal_100x100"]
b = [bench["baseline"][c]["device_usage_first_delta"] / mib for c in mcases]
a = [bench["fixed"][c]["device_usage_first_delta"] / mib for c in mcases]
bars(ax, [0, 1], b, a)
for x, (bv, av) in enumerate(zip(b, a)):
    ax.text(x - width / 2, bv, f"{bv:.0f}", ha="center", va="bottom", fontsize=9)
    ax.text(x + width / 2, av, f"{av:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks([0, 1], ["1x1", "100x100"])
ax.set_ylabel("Device memory taken by first call (MiB)")
ax.set_title("First-call device memory", loc="left", fontsize=11, color=INK)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BEFORE, label="Baseline CUDA (main 88e182f)"),
                    Patch(color=AFTER, label="Fixed CUDA (this PR)")],
           loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 1.0))
fig.suptitle("CUDA box3d_overlap: capacity fix on NVIDIA B200", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "svg"):
    fig.savefig(here / f"iou-capacity-comparison.{ext}", dpi=160, facecolor=SURF)
