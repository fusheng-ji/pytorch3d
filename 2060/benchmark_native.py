"""Fresh-process IoU CUDA resource/latency measurements; development artifact."""

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


NORMAL_CASES = [
    *[f"normal_{n}x{m}" for n in (30, 100) for m in (5, 10, 100)],
    *[f"normal_{n}x{n}" for n in (1, 4, 8, 16)],
    "high_overlap_100x100",
]
ALL_CASES = [*NORMAL_CASES, "issue1805_1x1", "issue1805_100x100"]


def inputs(case, fixture, torch):
    dimensions = case.rsplit("_", 1)[1]
    n, m = map(int, dimensions.split("x"))
    if case.startswith("issue1805_"):
        payload = json.loads(Path(fixture).read_text())
        first = torch.tensor(payload["boxes1"], dtype=torch.float32).reshape(1, 8, 3)
        second = torch.tensor(payload["boxes2"], dtype=torch.float32).reshape(1, 8, 3)
        return first.repeat(n, 1, 1), second.repeat(m, 1, 1)
    box = torch.tensor(
        [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
          [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(20261007 + n * 1000 + m)
    scale = 0.05 if case.startswith("high_overlap_") else 1.0
    first = box + torch.randn((n, 1, 3), generator=generator) * scale
    second = box + torch.randn((m, 1, 3), generator=generator) * scale
    return first, second


def measure(args):
    import torch

    torch.cuda.set_device(args.device)
    spec = importlib.util.spec_from_file_location(args.module_name, args.extension)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    boxes1, boxes2 = inputs(args.case, args.fixture, torch)
    reference = extension.iou_box3d(boxes1, boxes2)
    boxes1 = boxes1.to(f"cuda:{args.device}")
    boxes2 = boxes2.to(f"cuda:{args.device}")
    torch.cuda.synchronize()
    before_free, total = torch.cuda.mem_get_info()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = extension.iou_box3d(boxes1, boxes2)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000
    after_free, _ = torch.cuda.mem_get_info()
    after_allocated = torch.cuda.memory_allocated()
    after_reserved = torch.cuda.memory_reserved()
    max_errors = [float((actual.cpu() - expected).abs().max())
                  for actual, expected in zip(output, reference)]
    all_close = [bool(torch.allclose(actual.cpu(), expected, atol=1e-5, rtol=1e-5))
                 for actual, expected in zip(output, reference)]
    for _ in range(5):
        output = extension.iou_box3d(boxes1, boxes2)
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = extension.iou_box3d(boxes1, boxes2)
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    final_free, _ = torch.cuda.mem_get_info()
    props = torch.cuda.get_device_properties(args.device)
    return {
        "case": args.case,
        "pairs": boxes1.shape[0] * boxes2.shape[0],
        "shape": [boxes1.shape[0], boxes2.shape[0]],
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension": str(Path(args.extension).resolve()),
        "device": args.device,
        "first_call_ms": first_call_ms,
        "cuda_event_median_ms": statistics.median(samples),
        "cuda_event_min_ms": min(samples),
        "cuda_event_max_ms": max(samples),
        "samples_ms": samples,
        "device_memory_total": total,
        "device_free_before": before_free,
        "device_free_after_first": after_free,
        "device_free_final": final_free,
        "device_usage_first_delta": before_free - after_free,
        "device_usage_final_delta": before_free - final_free,
        "torch_allocated_before": before_allocated,
        "torch_allocated_after_first": after_allocated,
        "torch_reserved_before": before_reserved,
        "torch_reserved_after_first": after_reserved,
        "torch_allocated_peak": torch.cuda.max_memory_allocated(),
        "torch_reserved_peak": torch.cuda.max_memory_reserved(),
        "cpu_cuda_max_absolute_error": max_errors,
        "cpu_cuda_allclose_1e_5": all_close,
        "results_finite": [bool(torch.isfinite(t).all()) for t in output],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True)
    parser.add_argument("--module-name", default="_C")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--fixture", default=str(Path(__file__).with_name("fixture1805.json")))
    parser.add_argument("--case", choices=ALL_CASES)
    parser.add_argument("--include-issue", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.case:
        result = measure(args)
        print(json.dumps(result))
        return
    results = []
    cases = ALL_CASES if args.include_issue else NORMAL_CASES
    for case in cases:
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--extension", args.extension, "--module-name", args.module_name,
                   "--device", str(args.device), "--samples", str(args.samples),
                   "--fixture", args.fixture, "--case", case]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)
        results.append(result)
        print(f"{case}: {result['cuda_event_median_ms']:.6f} ms; "
              f"first device delta={result['device_usage_first_delta'] / 2**20:.1f} MiB; "
              f"peak allocated={result['torch_allocated_peak'] / 2**20:.1f} MiB; "
              f"CPU close={result['cpu_cuda_allclose_1e_5']}", flush=True)
        if args.output:
            Path(args.output).write_text(json.dumps({"cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
