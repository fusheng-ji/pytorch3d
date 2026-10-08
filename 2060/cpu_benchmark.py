"""Single-threaded CPU latency of the native extension (median of 5 runs after 1 warmup)."""
import importlib.util, json, statistics, sys, time, torch
from pathlib import Path
here = Path(__file__).resolve().parent; sys.path.insert(0, str(here))
from benchmark_native import inputs
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location("_C", sys.argv[1]); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
cases = torch.load(here / "cases.pt")
sets = {"rotated_120x127": cases["rotated_scale_1"], "normal_100x100": inputs("normal_100x100", None, torch),
        "high_overlap_100x100": inputs("high_overlap_100x100", None, torch),
        "issue1805_1x1": inputs("issue1805_1x1", str(here / "fixture1805.json"), torch)}
out = {}
for name, (b1, b2) in sets.items():
    ext.iou_box3d(b1, b2); ts = []
    for _ in range(5):
        t0 = time.perf_counter(); ext.iou_box3d(b1, b2); ts.append((time.perf_counter() - t0) * 1e3)
    out[name] = statistics.median(ts)
print(json.dumps(out))
