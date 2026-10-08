"""Build the benchmark CSVs and the comparison figure from the captured JSON/CSV files."""
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

here = Path(__file__).resolve().parent
load = lambda name: json.loads((here / name).read_text())
gpu = {k: {c["case"]: c for c in load(f"gpu-benchmark-{k}.json")["cases"]} for k in ("main", "pr2059", "new")}
cpu = {k: load(f"cpu-benchmark-{k}.json") for k in ("main", "new")}
exact = list(csv.DictReader(open(here / "exact-comparison.csv")))
mib = 2 ** 20

with open(here / "gpu-benchmark.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "pairs", "main_ms", "pr2059_ms", "new_ms", "new_over_main", "new_over_pr2059",
                "main_first_call_mib", "pr2059_first_call_mib", "new_first_call_mib", "new_matches_cpu_1e-5"])
    for c, r in gpu["new"].items():
        m = gpu["main"].get(c)
        w.writerow([c, r["pairs"], f"{m['cuda_event_median_ms']:.4f}" if m else "", f"{gpu['pr2059'][c]['cuda_event_median_ms']:.4f}",
                    f"{r['cuda_event_median_ms']:.4f}", f"{r['cuda_event_median_ms'] / m['cuda_event_median_ms']:.3f}" if m else "",
                    f"{r['cuda_event_median_ms'] / gpu['pr2059'][c]['cuda_event_median_ms']:.3f}",
                    round(m["device_usage_first_delta"] / mib) if m else "", round(gpu["pr2059"][c]["device_usage_first_delta"] / mib),
                    round(r["device_usage_first_delta"] / mib), all(r["cpu_cuda_allclose_1e_5"])])
with open(here / "cpu-benchmark.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "main_ms", "new_ms", "speedup"])
    for c in cpu["new"]:
        w.writerow([c, f"{cpu['main'][c]:.3f}", f"{cpu['new'][c]:.3f}", f"{cpu['main'][c] / cpu['new'][c]:.1f}"])

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NEW, MAIN, PR = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURF, gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
for ax in axes:
    ax.set_facecolor(SURF); ax.grid(axis="y", color=GRID, linewidth=0.8, which="major"); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

def bars(ax, xs, series, width):
    k = len(series)
    for j, (vals, color) in enumerate(series):
        off = (j - (k - 1) / 2) * (width + 0.02)
        for x, v in zip(xs, vals):
            if v is not None:
                ax.bar(x + off, v, width, color=color, edgecolor=SURF, linewidth=1)
    return lambda j: (j - (k - 1) / 2) * (width + 0.02)

# (a) accuracy
groups = [("Rotated\n(3 scales)", ["rotated_scale_0.01", "rotated_scale_1", "rotated_scale_100"]),
          ("Random,\nincl. identical", ["self_identical"]),
          ("Identical\n+1e-4 noise", ["self_perturbed_1e-4"]),
          ("Shared planes,\nsize ~0.01", ["grid_axis_aligned_scale_0.01", "grid_common_rotation_scale_0.01"])]
row = {r["case"]: r for r in exact}
def worst(key, cases):
    return max(float(row[c][f"{key}_max_iou_err"]) for c in cases)
CAP = 1.0
old = [worst("main_cpu", cs) for _, cs in groups]; new = [worst("new_cuda", cs) for _, cs in groups]
ax = axes[0]; off = bars(ax, range(len(groups)), [([min(v, CAP) for v in old], MAIN), (new, NEW)], 0.36)
ax.set_yscale("log"); ax.set_ylim(1e-8, 5)
for x, (o, n) in enumerate(zip(old, new)):
    ax.text(x + off(0), min(o, CAP) * 1.3, "inf" if not math.isfinite(o) else f"{o:.0e}", ha="center", va="bottom", fontsize=8.5)
    ax.text(x + off(1), n * 1.3, f"{n:.0e}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(range(len(groups)), [g for g, _ in groups], fontsize=9)
ax.set_ylabel("Max |IoU - exact IoU|")
ax.set_title("Accuracy vs float64 halfspace intersection", loc="left", fontsize=11)

# (b) GPU latency
cases = ["normal_1x1", "normal_100x100", "high_overlap_100x100", "issue1805_100x100"]
labels = ["1x1", "100x100", "100x100\nhigh overlap", "#1805 input\n100x100"]
ax = axes[1]
series = [([gpu["main"][c]["cuda_event_median_ms"] if c in gpu["main"] else None for c in cases], MAIN),
          ([gpu["pr2059"][c]["cuda_event_median_ms"] for c in cases], PR),
          ([gpu["new"][c]["cuda_event_median_ms"] for c in cases], NEW)]
off = bars(ax, range(len(cases)), series, 0.25)
ax.set_yscale("log"); ax.set_ylim(0.05, 100)
for j, (vals, _) in enumerate(series):
    for x, v in enumerate(vals):
        ax.text(x + off(j), v * 1.15 if v else 0.07, f"{v:.2f}" if v else "unsafe", ha="center", va="bottom", fontsize=8, rotation=0 if v else 90)
ax.set_xticks(range(len(cases)), labels, fontsize=9); ax.set_ylabel("Median CUDA latency (ms)")
ax.set_title("GPU latency (B200)", loc="left", fontsize=11)

# (c) CPU latency
names = ["rotated_120x127", "normal_100x100", "high_overlap_100x100", "issue1805_1x1"]
labels = ["rotated\n120x127", "100x100", "100x100\nhigh overlap", "#1805\n1x1"]
ax = axes[2]
series = [([cpu["main"][c] for c in names], MAIN), ([cpu["new"][c] for c in names], NEW)]
off = bars(ax, range(len(names)), series, 0.36)
ax.set_yscale("log"); ax.set_ylim(0.01, 5000)
for x, c in enumerate(names):
    ax.text(x + off(1), cpu["new"][c] * 1.15, f"{cpu['main'][c] / cpu['new'][c]:.1f}x\nfaster", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(len(names)), labels, fontsize=9); ax.set_ylabel("CPU latency, 1 thread (ms)")
ax.set_title("CPU latency", loc="left", fontsize=11)

fig.legend(handles=[Patch(color=MAIN, label="main (88e182f)"), Patch(color=PR, label="#2059 (CUDA only)"),
                    Patch(color=NEW, label="This PR")], loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 1.0))
fig.suptitle("box3d_overlap: polygon clipping", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
for ext in ("png", "svg"):
    fig.savefig(here / f"polygon-clipping-comparison.{ext}", dpi=160, facecolor=SURF)
