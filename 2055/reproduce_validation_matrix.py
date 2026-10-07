#!/usr/bin/env python
"""Reproduce the supplementary validation matrix for PyTorch3D PR #2055.

Use the same Python/PyTorch/CUDA environment to build the full native extension
at baseline 88e182f989c80836f4bd744e0d9cb1852762ce01 and patched commit
45f9132748ee1cac04e87bc7783b570b8022ae5c. Preserve the baseline extension file,
then build the patched extension in the checkout supplied with --repo-root.

Example, using relative paths to independently built checkouts:
  python reproduce_validation_matrix.py \
    --repo-root pytorch3d-patched --output-dir matrix-output \
    --baseline-extension pytorch3d-baseline/pytorch3d/_C.cpython-312-x86_64-linux-gnu.so

The original 165-row matrix uses CPU and two CUDA devices. This script defaults
to that matrix and requires both GPUs. Pass --devices cpu, or cpu cuda:0, for an
explicit smaller matrix (55 rows per device); devices are never silently skipped.
The original issue CSV is downloaded from its public attachment unless supplied
with --original-csv. Permanent regressions remain in the repository test module.
"""

import argparse
import csv
import importlib.util
import json
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

ORIGINAL_URL = (
    "https://github.com/facebookresearch/pytorch3d/files/11039116/point_cloud1.csv"
)
BASELINE_REVISION = "88e182f989c80836f4bd744e0d9cb1852762ce01"
COW_SOURCE = "docs/tutorials/data/cow_mesh/cow.obj"
POINT_COUNTS = (
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2048,
)


def environment(implementation, revision):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.system(),
        "extension": "pytorch3d._C (" + implementation + ")",
        "revision": revision,
    }


def original_records(fps, original_csv, devices, revision, baseline):
    coordinates = np.loadtxt(original_csv, delimiter=",", dtype=np.float32)
    if coordinates.shape != (2048, 3):
        raise ValueError(
            f"Expected the original 2048 x 3 CSV, got {coordinates.shape}."
        )
    records = []
    for device in devices:
        points = torch.from_numpy(coordinates).unsqueeze(0).to(device)
        points.requires_grad_()
        sampled, indices = fps(points, K=2048)
        sampled.sum().backward()
        selected = indices[0].cpu()
        unique_count = selected.unique().numel()
        zero_count = selected.eq(0).sum().item()
        all_gradients_one = bool(points.grad.eq(1).all().item())
        if baseline:
            assert unique_count == 1971, unique_count
            assert zero_count == 78, zero_count
            assert not all_gradients_one
            status = "REPRODUCED"
            checks = "known_duplicate_index_defect|gradient_overcount_reproduced"
        else:
            assert unique_count == 2048, unique_count
            assert zero_count == 1, zero_count
            assert all_gradients_one
            torch.testing.assert_close(selected.sort().values, torch.arange(2048))
            status = "PASS"
            checks = "unique_indices|full_permutation|gather_gradient_ones"
        implementation = "baseline" if baseline else "patched"
        record = {
            "scenario": "original_issue_csv",
            "implementation": implementation + "_native",
            "device": device,
            "device_name": (
                "CPU" if device == "cpu" else torch.cuda.get_device_name(device)
            ),
            "batch": 1,
            "P": 2048,
            "lengths": "2048",
            "K": "2048",
            "source": ORIGINAL_URL,
            "checks": checks,
            "unique_indices": unique_count,
            "index_0_count": zero_count,
            "gradient_all_ones": all_gradients_one,
            "status": status,
        }
        record.update(environment(implementation, revision))
        records.append(record)
    return records


def check_forward_backward(fps, points, lengths, requested_counts):
    sampled, indices = fps(points, lengths=lengths, K=requested_counts)
    batch, physical_count, dimensions = points.shape
    max_requested = int(requested_counts.max().item())
    assert indices.shape == (batch, max_requested)
    assert sampled.shape == (batch, max_requested, dimensions)
    selected_cpu = indices.cpu()
    sampled_cpu = sampled.detach().cpu()
    points_cpu = points.detach().cpu()
    lengths_cpu = lengths.cpu().tolist()
    requested_cpu = requested_counts.cpu().tolist()
    for n, (valid_count, requested) in enumerate(zip(lengths_cpu, requested_cpu)):
        count = min(valid_count, requested)
        selected = selected_cpu[n, :count]
        assert selected.unique().numel() == count
        assert selected.ge(0).all()
        assert selected.lt(valid_count).all()
        assert selected[0].item() == 0
        if count == valid_count:
            torch.testing.assert_close(
                selected.sort().values, torch.arange(valid_count)
            )
        assert selected_cpu[n, count:].eq(-1).all()
        assert sampled_cpu[n, count:].eq(0).all()
        torch.testing.assert_close(sampled_cpu[n, :count], points_cpu[n, selected])

    weights_cpu = (
        torch.arange(batch * max_requested * dimensions, dtype=torch.float32)
        .reshape(batch, max_requested, dimensions)
        .remainder(17)
        .add(1)
    )
    (sampled * weights_cpu.to(points.device)).sum().backward()
    expected_gradient = torch.zeros((batch, physical_count, dimensions))
    for n in range(batch):
        for output_index, input_index in enumerate(selected_cpu[n].tolist()):
            if input_index >= 0:
                expected_gradient[n, input_index] += weights_cpu[n, output_index]
    torch.testing.assert_close(points.grad.cpu(), expected_gradient, rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-extension", type=Path, required=True)
    parser.add_argument("--baseline-revision", default=BASELINE_REVISION)
    parser.add_argument("--original-csv", type=Path)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0", "cuda:1"])
    parser.add_argument(
        "--baseline-csv-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.baseline_extension = args.baseline_extension.resolve()
    if not (args.repo_root / "pytorch3d").is_dir():
        parser.error("--repo-root must identify a PyTorch3D source checkout.")
    if not args.baseline_extension.is_file():
        parser.error(
            "--baseline-extension must identify the preserved baseline native library."
        )
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices must not contain duplicates.")
    for device in args.devices:
        parsed_device = torch.device(device)
        if parsed_device.type == "cpu" and device == "cpu":
            continue
        if parsed_device.type != "cuda" or parsed_device.index is None:
            parser.error(
                "--devices supports cpu and explicit CUDA indices such as cuda:0."
            )
        if parsed_device.index >= torch.cuda.device_count():
            parser.error(
                f"Requested device {device} is unavailable; use an explicit smaller --devices list."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.original_csv = (
        args.original_csv.resolve()
        if args.original_csv
        else args.output_dir / "point_cloud1.csv"
    )
    sys.path.insert(0, str(args.repo_root))

    if args.baseline_csv_child:
        import pytorch3d

        spec = importlib.util.spec_from_file_location(
            "pytorch3d._C", args.baseline_extension
        )
        baseline_module = importlib.util.module_from_spec(spec)
        sys.modules["pytorch3d._C"] = baseline_module
        spec.loader.exec_module(baseline_module)
        pytorch3d._C = baseline_module
        from pytorch3d.ops import sample_farthest_points

        print(
            json.dumps(
                original_records(
                    sample_farthest_points,
                    args.original_csv,
                    args.devices,
                    args.baseline_revision,
                    baseline=True,
                )
            )
        )
        return

    from pytorch3d import _C
    from pytorch3d.io import load_obj
    from pytorch3d.ops.sample_farthest_points import (
        sample_farthest_points,
        sample_farthest_points_naive,
    )

    if not Path(_C.__file__).resolve().is_relative_to(args.repo_root):
        raise RuntimeError(
            "The patched native extension must be built inside --repo-root."
        )
    patched_revision = subprocess.run(
        ["git", "-C", str(args.repo_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if not args.original_csv.exists():
        urllib.request.urlretrieve(ORIGINAL_URL, args.original_csv)
    baseline_run = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--baseline-csv-child",
            "--repo-root",
            str(args.repo_root),
            "--output-dir",
            str(args.output_dir),
            "--baseline-extension",
            str(args.baseline_extension),
            "--baseline-revision",
            args.baseline_revision,
            "--original-csv",
            str(args.original_csv),
            "--devices",
            *args.devices,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    records = json.loads(baseline_run.stdout)
    records.extend(
        original_records(
            sample_farthest_points,
            args.original_csv,
            args.devices,
            patched_revision,
            baseline=False,
        )
    )
    metadata = environment("patched", patched_revision)
    print("ENVIRONMENT " + json.dumps(metadata), flush=True)

    def record_case(
        scenario,
        implementation,
        device,
        batch,
        point_count,
        lengths,
        requested_counts,
        source,
        checks,
    ):
        record = {
            "scenario": scenario,
            "implementation": implementation,
            "device": device,
            "device_name": (
                "CPU" if device == "cpu" else torch.cuda.get_device_name(device)
            ),
            "batch": batch,
            "P": point_count,
            "lengths": "|".join(str(count) for count in lengths),
            "K": "|".join(str(count) for count in requested_counts),
            "source": source,
            "checks": checks,
            "status": "PASS",
        }
        record.update(metadata)
        records.append(record)
        print(json.dumps(record), flush=True)

    for device in args.devices:
        for pattern in ("all_coincident", "partial_duplicates"):
            for point_count in POINT_COUNTS:
                lengths_list = [point_count, point_count // 2 + 1]
                requested_list = [point_count + 3, point_count + 1]
                if pattern == "all_coincident":
                    base = torch.ones((2, point_count, 3), dtype=torch.float32)
                else:
                    groups = torch.arange(point_count, dtype=torch.float32).div(
                        4, rounding_mode="floor"
                    )
                    coordinates = (
                        torch.stack(
                            (groups, groups.remainder(7), groups.remainder(11)), dim=1
                        )
                        / 64
                    )
                    base = coordinates.unsqueeze(0).repeat(2, 1, 1)
                base[1, lengths_list[1] :] = 100
                points = base.to(device).requires_grad_()
                lengths = torch.tensor(lengths_list, device=device)
                requested_counts = torch.tensor(requested_list, device=device)
                check_forward_backward(
                    sample_farthest_points, points, lengths, requested_counts
                )
                record_case(
                    pattern,
                    "patched_native",
                    device,
                    2,
                    point_count,
                    lengths_list,
                    requested_list,
                    "deterministic_synthetic",
                    "unique_indices|valid_range|full_permutation|padding|gather|weighted_gradient",
                )

    vertices, _, _ = load_obj(args.repo_root / COW_SOURCE, load_textures=False)
    if vertices.shape != (2930, 3):
        raise ValueError(
            f"Expected the repository cow OBJ with 2930 vertices, got {vertices.shape}."
        )
    base = torch.cat((vertices, vertices[:17], torch.zeros((9, 3)))).unsqueeze(0)
    for device in args.devices:
        points = base.to(device, copy=True).requires_grad_()
        lengths = torch.tensor([2947], device=device)
        requested_counts = torch.tensor([2950], device=device)
        check_forward_backward(
            sample_farthest_points, points, lengths, requested_counts
        )
        record_case(
            "real_cow_obj_e2e",
            "patched_native",
            device,
            1,
            2956,
            [2947],
            [2950],
            COW_SOURCE,
            "disk_obj_load|unique_indices|valid_range|full_permutation|padding|gather|weighted_gradient",
        )

    for fps in (sample_farthest_points, sample_farthest_points_naive):
        for device in args.devices:
            points = torch.ones((2, 8, 3), device=device)
            wrong_batch = torch.tensor([3, 3, 3], device=device)
            for argument, message in (
                ("K", "K and points must have"),
                ("lengths", "points and lengths must have"),
            ):
                keywords = {argument: wrong_batch}
                try:
                    fps(points, **keywords)
                except ValueError as error:
                    assert message in str(error), str(error)
                else:
                    raise AssertionError(
                        f"{fps.__name__} accepted incorrect {argument} batch on {device}"
                    )
                record_case(
                    "invalid_" + argument + "_batch",
                    fps.__name__,
                    device,
                    2,
                    8,
                    [3, 3, 3] if argument == "lengths" else [8, 8],
                    [3, 3, 3] if argument == "K" else [50, 50],
                    "deterministic_synthetic",
                    "ValueError|message",
                )

    device_count = len(args.devices)
    assert len(records) == 55 * device_count, len(records)
    fields = (
        "scenario",
        "implementation",
        "device",
        "device_name",
        "batch",
        "P",
        "lengths",
        "K",
        "source",
        "checks",
        "unique_indices",
        "index_0_count",
        "gradient_all_ones",
        "status",
        "python",
        "torch",
        "cuda",
        "platform",
        "extension",
        "revision",
    )
    matrix_path = args.output_dir / "validation-matrix.csv"
    with matrix_path.open("w", newline="") as matrix_file:
        writer = csv.DictWriter(matrix_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    for record in records[: 2 * device_count]:
        print("ORIGINAL " + json.dumps(record), flush=True)
    print(
        json.dumps(
            {
                "matrix": matrix_path.name,
                "total_rows": len(records),
                "devices": args.devices,
                "PASS": sum(record["status"] == "PASS" for record in records),
                "REPRODUCED": sum(
                    record["status"] == "REPRODUCED" for record in records
                ),
                "synthetic_forward_backward": 48 * device_count,
                "real_obj_e2e": device_count,
                "api_errors": 4 * device_count,
                "original_csv_baseline_and_patched": 2 * device_count,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
