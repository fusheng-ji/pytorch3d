#!/usr/bin/env python3
"""Capture actual PyTorch3D camera sampling, regression results and comparison plots.

Use --checkout to select an independently built baseline or patched checkout.
Use the patched regression file for --test-file on both runs. No implementation
is copied or replaced by this script. Figures use the two captured CSV files.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
import warnings


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capture(args):
    checkout = Path(args.checkout).resolve()
    sys.path.insert(0, str(checkout))
    import torch
    from pytorch3d.implicitron.models.view_pooler.view_sampler import (
        project_points_and_sample,
    )
    from pytorch3d.renderer import PerspectiveCameras

    rows = []
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    for device in devices:
        pts = torch.tensor(
            [
                [
                    [0.13, -0.09, 2.0],
                    [0.12, 0.17, -2.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1e-4],
                ]
            ],
            device=device,
            requires_grad=True,
        )
        R = torch.eye(3, device=device)[None].requires_grad_()
        T = torch.zeros(1, 3, device=device, requires_grad=True)
        feat_map = (
            torch.linspace(0.1, 2.0, 24, device=device)
            .reshape(1, 1, 4, 6)
            .requires_grad_()
        )
        mask_map = (
            torch.linspace(0.2, 0.9, 24, device=device)
            .reshape(1, 1, 4, 6)
            .requires_grad_()
        )
        camera = PerspectiveCameras(R=R, T=T, device=device)
        sampled, masks = project_points_and_sample(
            pts, {"features": feat_map}, camera, mask_map
        )
        for point, label in enumerate(
            ("front", "behind", "camera_plane", "tiny_positive")
        ):
            loss = sampled["features"][0, 0, point].sum() + masks[0, 0, point].sum()
            gradients = torch.autograd.grad(
                loss, (pts, R, T, feat_map, mask_map), retain_graph=True
            )
            row = {
                "variant": args.variant,
                "device": device,
                "point": label,
                "camera_depth": pts[0, point, 2].item(),
                "sampled_feature": sampled["features"][0, 0, point, 0].item(),
                "sampled_mask": masks[0, 0, point, 0].item(),
                "point_gradient_l1": gradients[0][0, point].abs().sum().item(),
                "camera_R_gradient_l1": gradients[1].abs().sum().item(),
                "camera_T_gradient_l1": gradients[2].abs().sum().item(),
                "feature_map_gradient_l1": gradients[3].abs().sum().item(),
                "mask_map_gradient_l1": gradients[4].abs().sum().item(),
                "gradients_finite": all(
                    torch.isfinite(gradient).all().item() for gradient in gradients
                ),
            }
            rows.append(row)
    write_csv(args.output / f"{args.variant}-numerical-results.csv", rows)

    test_path = Path(args.test_file).resolve()
    spec = importlib.util.spec_from_file_location(
        "camera_regression_candidate", test_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    regression_rows = []

    class MatrixResult(unittest.TextTestResult):
        def addSubTest(self, test, subtest, err):
            details = dict(subtest.params)
            regression_rows.append(
                {
                    "variant": args.variant,
                    "test": test._testMethodName,
                    "device": details.get("device", "cpu"),
                    "camera": details.get("camera", ""),
                    "mode": details.get("mode", "bilinear"),
                    "masked": details.get("masked", ""),
                    "aggregator": details.get("aggregator", ""),
                    "status": "PASS" if err is None else "FAIL",
                    "error": (
                        ""
                        if err is None
                        else f"{err[0].__name__}: {str(err[1]).splitlines()[0]}"
                    ),
                }
            )
            super().addSubTest(test, subtest, err)

        def addSuccess(self, test):
            if test._testMethodName == "test_depth_gradcheck":
                regression_rows.append(
                    {
                        "variant": args.variant,
                        "test": test._testMethodName,
                        "device": "cpu",
                        "camera": "PerspectiveCameras",
                        "mode": "bilinear",
                        "masked": True,
                        "aggregator": "",
                        "status": "PASS",
                        "error": "",
                    }
                )
            super().addSuccess(test)

    names = [
        "test_nonpositive_camera_depth",
        "test_transformed_cameras_and_point_grid",
        "test_depth_and_sequence_masks",
        "test_depth_gradients",
        "test_depth_gradcheck",
        "test_sampler_aggregator_backward",
    ]
    regression_log = args.output / f"{args.variant}-regressions.log"
    with regression_log.open("w") as stream:
        suite = unittest.TestSuite(module.TestViewsampling(name) for name in names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = unittest.TextTestRunner(
                stream=stream, verbosity=2, resultclass=MatrixResult
            ).run(suite)
    log_text = regression_log.read_text().replace(
        str(test_path), "tests/implicitron/test_viewsampling.py"
    )
    log_text = log_text.replace(str(checkout) + "/", "")
    log_text = log_text.replace(sys.prefix + "/", "<python-env>/")
    regression_log.write_text(log_text)
    write_csv(args.output / f"{args.variant}-validation-matrix.csv", regression_rows)

    # Record rather than work around the current public camera float32 limitation.
    try:
        double_camera = PerspectiveCameras(
            R=torch.eye(3, dtype=torch.float64)[None],
            T=torch.zeros(1, 3, dtype=torch.float64),
            focal_length=torch.ones(1, 2, dtype=torch.float64),
            principal_point=torch.zeros(1, 2, dtype=torch.float64),
        )
        project_points_and_sample(
            torch.tensor([[[0.1, 0.2, 2.0]]], dtype=torch.float64),
            {"features": torch.ones(1, 1, 4, 6, dtype=torch.float64)},
            double_camera,
            None,
        )
        double_status = "Supported"
    except RuntimeError as error:
        double_status = str(error)
    meta = {
        "variant": args.variant,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        "source_file_sha256": hashlib.sha256(
            (
                checkout / "pytorch3d/implicitron/models/view_pooler/view_sampler.py"
            ).read_bytes()
        ).hexdigest(),
        "regression_test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "modified_source_files": subprocess.check_output(
            ["git", "diff", "--name-only"], cwd=checkout, text=True
        ).splitlines(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": {
            device: ("CPU" if device == "cpu" else torch.cuda.get_device_name(device))
            for device in devices
        },
        "regression_methods": result.testsRun,
        "regression_cases": len(regression_rows),
        "passed_cases": sum(row["status"] == "PASS" for row in regression_rows),
        "failed_cases": sum(row["status"] == "FAIL" for row in regression_rows),
        "float64_public_path": double_status,
        "gradcheck": {"dtype": "float32", "eps": 1e-3, "atol": 1e-3, "rtol": 1e-2},
        "limitations": [
            "One Linux/software/GPU-model stack",
            "No downstream training",
            "No comprehensive performance benchmark",
        ],
    }
    (args.output / f"{args.variant}-environment.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2))
    if args.variant == "patched" and not result.wasSuccessful():
        raise SystemExit(1)


def plot(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    captured = {}
    for variant in ("baseline", "patched"):
        with (output / f"{variant}-numerical-results.csv").open() as stream:
            captured[variant] = [
                row for row in csv.DictReader(stream) if row["device"] == "cpu"
            ]
    labels = [
        "Front\nz = 2",
        "Behind\nz = -2",
        "Camera plane\nz = 0",
        "Tiny positive\nz = 0.0001",
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    metrics = [
        ("sampled_feature", "Sampled feature"),
        ("sampled_mask", "Sampled mask"),
        ("feature_map_gradient_l1", "Feature-map gradient L1"),
    ]
    for axis, (key, title) in zip(axes, metrics):
        for offset, variant, color in (
            (-0.18, "baseline", "#dc704a"),
            (0.18, "patched", "#28829b"),
        ):
            values = [float(row[key]) for row in captured[variant]]
            bars = axis.bar(
                x + offset, values, width=0.36, color=color, label=variant.capitalize()
            )
            precision = ".2g" if key == "sampled_mask" else ".3g"
            axis.bar_label(
                bars,
                labels=[format(value, precision) for value in values],
                padding=3,
                fontsize=9,
            )
        axis.set_xticks(x, labels, fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, axis.get_ylim()[1] * 1.18)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Camera-space depth filtering: unchanged front samples, zero invalid samples and gradients",
        fontsize=13,
    )
    fig.savefig(output / "camera-depth-comparison.png", dpi=180)
    fig.savefig(output / "camera-depth-comparison.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--variant", choices=("baseline", "patched"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plot:
        plot(args.output)
    else:
        if not all((args.checkout, args.test_file, args.variant)):
            parser.error("capture needs --checkout, --test-file and --variant")
        capture(args)


if __name__ == "__main__":
    main()
