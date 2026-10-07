#!/usr/bin/env python3
"""Capture and plot native FPS evidence for the original issue #1486 fixture.

Run ``capture`` in separate processes for the baseline and patched builds. Each
capture uses the public API with K=P and L=sampled.sum(), and writes measured
indices and gathered gradients. ``plot`` compares the saved captures without
requiring PyTorch or CUDA. The original CSV and native binaries are not copied.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_URL = (
    "https://github.com/facebookresearch/pytorch3d/files/11039116/point_cloud1.csv"
)
COLORS = {"before": "#c6642a", "after": "#176f9f"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_root(explicit_root):
    if explicit_root is not None:
        return explicit_root.resolve()
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pytorch3d" / "__init__.py").is_file():
            return candidate
    return None


def cumulative_unique(indices):
    seen = set()
    result = np.empty(len(indices), dtype=np.int32)
    for position, input_index in enumerate(indices):
        seen.add(int(input_index))
        result[position] = len(seen)
    return result


def capture(args):
    import torch

    root = repo_root(args.repo_root)
    if root is not None:
        sys.path.insert(0, str(root))
    import pytorch3d

    # Import a preserved baseline in this fresh process before ops imports _C.
    if args.native_extension is not None:
        spec = importlib.util.spec_from_file_location(
            "pytorch3d._C", args.native_extension.resolve()
        )
        extension = importlib.util.module_from_spec(spec)
        sys.modules["pytorch3d._C"] = extension
        spec.loader.exec_module(extension)
        pytorch3d._C = extension

    from pytorch3d import _C
    from pytorch3d.ops import sample_farthest_points

    coordinates = np.loadtxt(args.input, delimiter=",", dtype=np.float32)
    if coordinates.shape != (2048, 3) or not np.isfinite(coordinates).all():
        raise ValueError("Expected the finite (2048, 3) original issue CSV.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extension_hash = sha256(Path(_C.__file__))

    for device in args.devices:
        points = torch.from_numpy(coordinates).unsqueeze(0).to(device)
        points.requires_grad_()
        sampled, selected = sample_farthest_points(
            points, K=len(coordinates), random_start_point=False
        )
        sampled.sum().backward()
        indices = selected[0].detach().cpu().numpy().astype(np.int32)
        gradients = points.grad[0].detach().cpu().numpy()
        if not ((indices >= 0) & (indices < len(coordinates))).all():
            raise AssertionError("Selected indices must be valid input indices.")
        counts = np.bincount(indices, minlength=len(coordinates)).astype(np.int32)
        np.testing.assert_array_equal(
            sampled[0].detach().cpu().numpy(), coordinates[indices]
        )
        np.testing.assert_array_equal(
            gradients, np.repeat(counts[:, None], coordinates.shape[1], axis=1)
        )
        errors = gradients.astype(np.float64) - 1.0
        unique_count = int(np.count_nonzero(counts))
        metrics = {
            "unique_indices": unique_count,
            "repeated_selections": int(len(indices) - unique_count),
            "omitted_input_indices": int(np.count_nonzero(counts == 0)),
            "indices_selected_once": int(np.count_nonzero(counts == 1)),
            "index_0_count": int(counts[0]),
            "max_selection_count": int(counts.max()),
            "unique_selected_coordinates": int(
                len(np.unique(coordinates[indices], axis=0))
            ),
            "gradient_min": float(gradients.min()),
            "gradient_max": float(gradients.max()),
            "gradient_max_abs_error_vs_one": float(np.abs(errors).max()),
            "gradient_component_rmse_vs_one": float(np.sqrt(np.square(errors).mean())),
        }
        gpu_name = (
            torch.cuda.get_device_name(torch.device(device))
            if torch.device(device).type == "cuda"
            else "CPU"
        )
        stem = f"capture-{args.label}-{device.replace(':', '-')}"
        output = args.output_dir / f"{stem}.npz"
        np.savez_compressed(
            output,
            indices=indices,
            selection_counts=counts,
            gradients=gradients,
            cumulative_unique_indices=cumulative_unique(indices),
        )
        metadata = {
            "schema_version": 1,
            "label": args.label,
            "device": device,
            "device_name": gpu_name,
            "implementation": "native_public_sample_farthest_points",
            "source_revision": args.source_revision,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_source_url": SOURCE_URL,
            "input_sha256": sha256(args.input),
            "input_shape": list(coordinates.shape),
            "input_unique_coordinates": int(len(np.unique(coordinates, axis=0))),
            "input_dtype": "float32",
            "K": len(coordinates),
            "random_start_point": False,
            "loss": "sampled.sum()",
            "gradient_reference": "one for every input coordinate when K=P",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "operating_system": platform.system(),
            "architecture": platform.machine(),
            "native_extension_sha256": extension_hash,
            "capture_file": output.name,
            "capture_sha256": sha256(output),
            "metrics": metrics,
        }
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"label": args.label, "device": device, **metrics}))


def load_captures(directory):
    captures = {}
    for metadata_file in sorted(directory.glob("capture-*.json")):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        key = (metadata["label"], metadata["device"])
        if key in captures:
            raise ValueError(f"Duplicate capture: {key}")
        filename = metadata["capture_file"]
        if Path(filename).name != filename:
            raise ValueError("Capture filename must not contain a directory.")
        capture_file = directory / filename
        if sha256(capture_file) != metadata["capture_sha256"]:
            raise ValueError(f"Capture checksum mismatch: {filename}")
        with np.load(capture_file, allow_pickle=False) as arrays:
            data = {name: arrays[name].copy() for name in arrays.files}
        indices = data["indices"]
        np.testing.assert_array_equal(
            data["selection_counts"],
            np.bincount(indices, minlength=metadata["input_shape"][0]),
        )
        np.testing.assert_array_equal(
            data["cumulative_unique_indices"], cumulative_unique(indices)
        )
        captures[key] = (metadata, data)
    devices = sorted({device for label, device in captures})
    if not devices or any(
        (label, device) not in captures for device in devices for label in COLORS
    ):
        raise ValueError("Need matching before/after captures for every device.")
    input_hashes = {metadata["input_sha256"] for metadata, data in captures.values()}
    if len(input_hashes) != 1:
        raise ValueError("All captures must use the same input CSV.")
    for label in COLORS:
        representative_metadata, representative_data = captures[(label, devices[0])]
        for device in devices[1:]:
            metadata, data = captures[(label, device)]
            if metadata["metrics"] != representative_metadata["metrics"]:
                raise ValueError("Devices disagree; do not combine the measurements.")
            np.testing.assert_array_equal(
                data["selection_counts"], representative_data["selection_counts"]
            )
            np.testing.assert_array_equal(
                data["cumulative_unique_indices"],
                representative_data["cumulative_unique_indices"],
            )
    return devices, captures


def write_tables(output_dir, devices, captures):
    first_metadata = next(iter(captures.values()))[0]
    metric_fields = list(first_metadata["metrics"])
    fields = ["label", "device", "device_name", *metric_fields]
    with (output_dir / "numerical-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label in COLORS:
            for device in devices:
                metadata, data = captures[(label, device)]
                writer.writerow(
                    {
                        "label": label,
                        "device": device,
                        "device_name": metadata["device_name"],
                        **metadata["metrics"],
                    }
                )
    fields = [
        "label",
        "device",
        "input_index",
        "selection_count",
        "gradient_x",
        "gradient_y",
        "gradient_z",
        "max_gradient_error_vs_one",
    ]
    with (output_dir / "per-input-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label in COLORS:
            for device in devices:
                metadata, data = captures[(label, device)]
                for index, (count, gradient) in enumerate(
                    zip(data["selection_counts"], data["gradients"])
                ):
                    writer.writerow(
                        {
                            "label": label,
                            "device": device,
                            "input_index": index,
                            "selection_count": int(count),
                            "gradient_x": float(gradient[0]),
                            "gradient_y": float(gradient[1]),
                            "gradient_z": float(gradient[2]),
                            "max_gradient_error_vs_one": float(
                                np.abs(gradient.astype(np.float64) - 1.0).max()
                            ),
                        }
                    )


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import ScalarFormatter

    devices, captures = load_captures(args.capture_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, devices, captures)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#87919c",
            "text.color": "#172838",
            "axes.labelcolor": "#172838",
            "xtick.color": "#344757",
            "ytick.color": "#344757",
            "svg.hashsalt": "fps-issue-1486-evidence",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.76, bottom=0.31, wspace=0.37)
    fig.text(
        0.035,
        0.94,
        "Farthest point sampling: original issue #1486",
        fontsize=18,
        fontweight="bold",
    )
    first_metadata = captures[("before", devices[0])][0]
    count = first_metadata["input_shape"][0]
    names = list(
        dict.fromkeys(captures[("before", d)][0]["device_name"] for d in devices)
    )
    subtitle = (
        f"{count:,} input indices  |  K = {count:,}  |  start index = 0  |  "
        f"{len(devices)} devices ({', '.join(names)})"
    )
    fig.text(0.035, 0.885, subtitle, fontsize=10.5, color="#425666")
    fig.legend(
        handles=[
            Patch(color=COLORS[label], label=label.capitalize()) for label in COLORS
        ],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.917),
        ncol=2,
        frameon=False,
    )

    axis = axes[0]
    for label in COLORS:
        metadata, data = captures[(label, devices[0])]
        x = np.arange(1, count + 1)
        axis.plot(
            x, data["cumulative_unique_indices"], color=COLORS[label], linewidth=2.8
        )
        end = int(data["cumulative_unique_indices"][-1])
        axis.annotate(
            f"{end:,}",
            (count, end),
            xytext=(-5, 7 if label == "after" else -17),
            textcoords="offset points",
            ha="right",
            color=COLORS[label],
            fontweight="bold",
        )
    axis.set_title("Distinct selected input indices", loc="left", pad=12)
    axis.set_xlim(count - 149, count + 3)
    axis.set_ylim(count - 155, count + 10)
    axis.set_xlabel("Selection step (last 150)")
    axis.set_ylabel("Cumulative distinct indices")
    axis.grid(axis="both", alpha=0.18)

    axis = axes[1]
    multiplicities = sorted(
        {
            int(value)
            for label in COLORS
            for value in captures[(label, devices[0])][1]["selection_counts"]
        }
    )
    positions = np.arange(len(multiplicities))
    width = 0.34
    for offset, label in zip((-width / 2, width / 2), COLORS):
        counts = captures[(label, devices[0])][1]["selection_counts"]
        heights = [int(np.count_nonzero(counts == m)) for m in multiplicities]
        bars = axis.bar(positions + offset, heights, width=width, color=COLORS[label])
        for bar, height in zip(bars, heights):
            axis.annotate(
                f"{height:,}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha=(
                    ("right" if label == "before" else "left")
                    if height >= 1000
                    else "center"
                ),
                color=COLORS[label],
                fontsize=9,
                fontweight="bold",
            )
    axis.set_title("Input selection multiplicity", loc="left", pad=12)
    axis.set_xticks(positions, [str(m) for m in multiplicities])
    axis.set_xlabel("Times an input index is selected")
    axis.set_ylabel("Number of indices (symlog scale)")
    axis.set_yscale("symlog", linthresh=2)
    axis.set_ylim(0, count * 2.6)
    axis.set_yticks([0, 1, 10, 100, 1000])
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)

    axis = axes[2]
    positions = np.arange(len(devices))
    for offset, label in zip((-width / 2, width / 2), COLORS):
        heights = [
            captures[(label, device)][0]["metrics"]["gradient_max_abs_error_vs_one"]
            for device in devices
        ]
        bars = axis.bar(positions + offset, heights, width=width, color=COLORS[label])
        for bar, height in zip(bars, heights):
            axis.annotate(
                f"{height:g}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                color=COLORS[label],
                fontweight="bold",
            )
    axis.set_title("Gather gradient error", loc="left", pad=12)
    axis.set_xticks(positions, ["CPU" if d == "cpu" else d.upper() for d in devices])
    axis.set_xlabel("Native execution device")
    axis.set_ylabel(r"$\max\,|\partial L / \partial p - 1|$")
    max_error = max(
        metadata["metrics"]["gradient_max_abs_error_vs_one"]
        for metadata, data in captures.values()
    )
    axis.set_ylim(0, max(1, max_error) * 1.18)
    axis.grid(axis="y", alpha=0.18)
    axis.set_axisbelow(True)

    unique_locations = first_metadata["input_unique_coordinates"]
    fig.text(
        0.035,
        0.195,
        "Data: issue #1486 attachment point_cloud1.csv. "
        "L = sum(sampled coordinates); expected input gradient = 1 when K = P.",
        fontsize=10,
    )
    fig.text(
        0.035,
        0.145,
        f"All {len(devices)} devices agree. Unique coordinate locations remain "
        f"{unique_locations:,} before and after. No timing or model-accuracy measurement.",
        fontsize=10,
        color="#425666",
    )
    fig.text(
        0.035,
        0.095,
        f"Stack: {first_metadata['operating_system']}; Python {first_metadata['python']}; "
        f"PyTorch {first_metadata['pytorch']}; CUDA {first_metadata['cuda_runtime']}.",
        fontsize=9.5,
        color="#607381",
    )
    for extension in ("png", "svg"):
        metadata = {
            "Title": "Native FPS before/after evidence for issue #1486",
            "Description": "Measured selection indices and gathered-point gradients; no timing benchmark.",
        }
        if extension == "png":
            metadata = {
                "Title": metadata["Title"],
                "Description": metadata["Description"],
            }
        fig.savefig(
            args.output_dir / f"fps-comparison.{extension}",
            dpi=200,
            facecolor="white",
            metadata=metadata,
        )
        if extension == "svg":
            svg = args.output_dir / "fps-comparison.svg"
            svg.write_text(
                "\n".join(
                    line.rstrip()
                    for line in svg.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)
    summary = {
        "input_source_url": SOURCE_URL,
        "input_sha256": first_metadata["input_sha256"],
        "devices": devices,
        "capture_count": len(captures),
        "measurements": [
            {
                "label": label,
                "device": device,
                **captures[(label, device)][0]["metrics"],
            }
            for label in COLORS
            for device in devices
        ],
        "limitations": [
            "One original fixture and one Linux/Python/PyTorch/CUDA software stack.",
            "The two CUDA devices are both NVIDIA B200 GPUs.",
            "The gradient metric checks gathered-point gradients for L=sampled.sum().",
            "Distinct coordinate coverage is unchanged; distinct input-index coverage changes.",
            "No performance benchmark, downstream model training or accuracy claim.",
        ],
    }
    (args.output_dir / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"captures": len(captures), "devices": devices, "status": "verified"}
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "capture", help="Capture one build in a fresh process."
    )
    command.add_argument(
        "--input", type=Path, required=True, help="Original point_cloud1.csv."
    )
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--label", choices=COLORS, required=True)
    command.add_argument("--devices", nargs="+", default=["cpu"])
    command.add_argument(
        "--repo-root", type=Path, help="Source checkout of the built pytorch3d."
    )
    command.add_argument(
        "--native-extension", type=Path, help="Optional preserved native _C binary."
    )
    command.add_argument(
        "--source-revision",
        required=True,
        help="Revision used to build the native extension.",
    )
    command.set_defaults(run=capture)
    command = commands.add_parser(
        "plot", help="Render saved matching before/after captures."
    )
    command.add_argument("--capture-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.set_defaults(run=plot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
