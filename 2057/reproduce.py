"""Capture actual IoU validation before/after and plot measured face residuals.

Run this script in a process with the desired PyTorch3D checkout first on
PYTHONPATH, then combine the two captures using --plot-before/--plot-after.
The native extension must be built for that checkout; no kernel is stubbed.
"""

import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
import pytorch3d
from pytorch3d import _C
from pytorch3d.ops.iou_box3d import _box_planes, _check_coplanar, box3d_overlap

UNIT_BOX = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]
ISSUE_BOX = [
    [-115.3384, 31.8733, 0.2705],
    [-115.3384, 31.8733, 4.5234],
    [-109.9031, 23.1563, 4.5234],
    [-109.9031, 23.1563, 0.2705],
    [-118.6852, 29.7865, 0.2705],
    [-118.6852, 29.7865, 4.5234],
    [-113.2499, 21.0695, 4.5234],
    [-113.2499, 21.0695, 0.2705],
]


def residuals(boxes):
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    vertices = boxes.index_select(1, faces.flatten()).reshape(-1, 6, 4, 3)
    v0, v1, v2, v3 = vertices.unbind(2)
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)
    signed = ((v3 - v0) * normal).sum(-1)
    aggregate = (v3 - v0).reshape(-1, 1, 18).bmm(normal.reshape(-1, 18, 1))
    return signed, aggregate.abs().flatten()


def capture(label, output):
    output.mkdir(parents=True, exist_ok=True)
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    rows, face_rows = [], []

    def check(case, boxes1, boxes2, expected, eps=1e-4, helper=False, known=None):
        signed, aggregate = residuals(boxes1)
        signed2, aggregate2 = residuals(boxes2)
        native_called, accepted, error = False, False, ""
        volume, iou = None, None
        if helper:
            try:
                _check_coplanar(boxes1, eps=eps)
                accepted = True
            except ValueError as exception:
                error = str(exception)
        else:
            with patch(
                "pytorch3d.ops.iou_box3d._C.iou_box3d", wraps=_C.iou_box3d
            ) as native:
                try:
                    volume, iou = box3d_overlap(boxes1, boxes2, eps=eps)
                    accepted = True
                except ValueError as exception:
                    error = str(exception)
                native_called = native.called
        passed = accepted == expected
        if known is not None and accepted:
            expected_volume, expected_iou = known
            passed = passed and torch.allclose(
                volume, expected_volume, atol=1e-5, rtol=1e-5
            )
            passed = passed and torch.allclose(iou, expected_iou, atol=1e-5, rtol=1e-5)
        baseline_bug = label == "before" and (
            case.startswith("cancel/") or case == "threshold-multi/0.5"
        )
        if not passed and not baseline_bug:
            raise AssertionError(f"{label} {boxes1.device} {case}: {error}")
        rows.append(
            {
                "label": label,
                "device": str(boxes1.device),
                "case": case,
                "batch1": len(boxes1),
                "batch2": len(boxes2),
                "eps": eps,
                "accepted": accepted,
                "expected_accepted": expected,
                "native_called": native_called,
                "status": "REPRODUCED" if baseline_bug and not passed else "PASS",
                "aggregate_abs_max": max(
                    aggregate.max().item(), aggregate2.max().item()
                ),
                "face_residual_abs_max": max(
                    signed.abs().max().item(), signed2.abs().max().item()
                ),
                "volume_0_0": "" if volume is None else volume[0, 0].item(),
                "iou_0_0": "" if iou is None else iou[0, 0].item(),
                "error": error,
            }
        )
        for box_input, face_values in (("boxes1", signed), ("boxes2", signed2)):
            for batch_index in range(len(face_values)):
                for face in range(6):
                    face_rows.append(
                        {
                            "label": label,
                            "device": str(boxes1.device),
                            "case": case,
                            "box_input": box_input,
                            "batch_index": batch_index,
                            "face": face,
                            "signed_residual": face_values[batch_index, face].item(),
                            "eps": eps,
                        }
                    )

    for device in devices:
        unit = torch.tensor(UNIT_BOX, dtype=torch.float32, device=device)
        for cancel in (False, True):
            invalid = unit.clone()
            invalid[3, 2] = 0.25
            if cancel:
                invalid[7, 2] = 0.75
            for batch_size in (1, 3):
                boxes = unit.repeat(batch_size, 1, 1)
                boxes[batch_size // 2] = invalid
                for side in ("first", "second"):
                    name = (
                        f"{'cancel' if cancel else 'single'}/batch{batch_size}/{side}"
                    )
                    if side == "first":
                        check(name, boxes, unit[None], False)
                    else:
                        check(name, unit[None], boxes, False)
        eps = 2.0**-10
        for factor in (0.5, 1.0, 2.0):
            for same_sign in (False, True):
                box = unit.clone()
                box[3, 2] = factor * eps
                if same_sign:
                    box[7, 2] += factor * eps
                prefix = "threshold-multi" if same_sign else "threshold"
                check(
                    f"{prefix}/{factor}",
                    box[None],
                    unit[None],
                    factor < 1,
                    eps=eps,
                    helper=True,
                )
        transforms = [
            ("identity", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]),
            ("translation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [2, -3, 4]),
            ("rotation", [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 0]),
            ("scale", [[2, 0, 0], [0, 0.5, 0], [0, 0, 3]], [0, 0, 0]),
            ("shear", [[1, 0.25, 0], [0, 1, 0.5], [0, 0, 1]], [0, 0, 0]),
        ]
        for name, matrix, translation in transforms:
            matrix = unit.new_tensor(matrix)
            translation = unit.new_tensor(translation)
            box1 = unit @ matrix + translation
            box2 = (unit + unit.new_tensor([0.5, 0, 0])) @ matrix + translation
            boxes = torch.stack((box1, box2))
            known_volume = unit.new_tensor([[1, 0.5], [0.5, 1]]) * matrix.det().abs()
            known_iou = unit.new_tensor([[1, 1 / 3], [1 / 3, 1]])
            check(f"affine/{name}", boxes, boxes, True, known=(known_volume, known_iou))
        zero = torch.zeros_like(unit)[None]
        check("zero/first", zero, unit[None], False)
        check("zero/second", unit[None], zero, False)
        issue = unit.new_tensor(ISSUE_BOX)
        for count in (1, 2):
            boxes = issue.repeat(count, 1, 1)
            check(f"issue1894/batch{count}", boxes, boxes, True)
    for filename, records in (
        (f"{label}-matrix.csv", rows),
        (f"{label}-face-residuals.csv", face_rows),
    ):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    metadata = {
        "label": label,
        "python_precision": "float32 boxes",
        "python": platform.python_version(),
        "platform": platform.system(),
        "source_commit": subprocess.check_output(
            [
                "git",
                "-C",
                str(Path(pytorch3d.__file__).resolve().parent.parent),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": devices,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
        "native_extension_available": True,
        "matrix_rows": len(rows),
        "pass_rows": sum(row["status"] == "PASS" for row in rows),
        "baseline_reproductions": sum(row["status"] == "REPRODUCED" for row in rows),
    }
    (output / f"{label}-environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata))


def plot(before, after, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    output.mkdir(parents=True, exist_ok=True)

    def read(path):
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    before_rows = read(before / "before-matrix.csv")
    after_rows = read(after / "after-matrix.csv")
    faces = read(after / "after-face-residuals.csv")
    selected = [
        row
        for row in faces
        if row["device"] == "cpu"
        and row["case"] == "cancel/batch1/first"
        and row["box_input"] == "boxes1"
    ]
    signed = [float(row["signed_residual"]) for row in selected]
    old = next(
        row
        for row in before_rows
        if row["device"] == "cpu" and row["case"] == "cancel/batch1/first"
    )
    new = next(
        row
        for row in after_rows
        if row["device"] == "cpu" and row["case"] == "cancel/batch1/first"
    )
    fig = plt.figure(figsize=(11, 4.5), layout="constrained")
    axis3d = fig.add_subplot(1, 2, 1, projection="3d")
    unit = torch.tensor(UNIT_BOX, dtype=torch.float64)
    invalid = unit.clone()
    invalid[3, 2], invalid[7, 2] = 0.25, 0.75
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for vertices, color, style in ((unit, "#aaaaaa", "--"), (invalid, "#ce4266", "-")):
        axis3d.add_collection3d(
            Line3DCollection(
                [vertices[[i, j]].numpy() for i, j in edges],
                colors=color,
                linestyles=style,
                linewidths=2,
            )
        )
    axis3d.scatter(*invalid.T.numpy(), color="#ce4266", s=25)
    for index in (3, 7):
        point = invalid[index].tolist()
        axis3d.text(*point, f"  v{index}: z={point[2]}")
    axis3d.set(
        xlim=(-0.1, 1.1),
        ylim=(-0.1, 1.1),
        zlim=(-0.1, 1.1),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="Two non-planar faces",
    )
    axis3d.view_init(elev=22, azim=-55)
    axis = fig.add_subplot(1, 2, 2)
    axis.bar(
        range(6),
        signed,
        color=["#ce4266" if abs(value) > 1e-4 else "#6aaeb1" for value in signed],
    )
    axis.axhline(0, color="#444444", linewidth=0.8)
    for sign in (-1, 1):
        axis.axhline(sign * 1e-4, color="#e49628", linestyle="--", linewidth=1)
    axis.set(
        xticks=range(6),
        xlabel="Face index",
        ylabel="Signed fourth-vertex residual",
        ylim=(-0.32, 0.32),
        title="Face errors cancel in the old sum",
    )
    axis.text(
        0.29,
        0.98,
        f"Old |sum|: {float(old['aggregate_abs_max']):.3g}  →  accepted\nFixed max |face|: {float(new['face_residual_abs_max']):.3g}  →  rejected\nTolerance: 1e-4",
        transform=axis.transAxes,
        va="top",
        fontsize=10,
    )
    fig.suptitle(
        "IoU3D coplanarity: measured counterexample (CPU; both B200 GPUs agree)",
        fontsize=13,
    )
    for suffix in ("png", "svg"):
        fig.savefig(output / f"coplanarity-comparison.{suffix}", dpi=220)
    with (output / "validation-matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(before_rows[0]))
        writer.writeheader()
        writer.writerows(before_rows + after_rows)
    summary = []
    for old_row in before_rows:
        if old_row["case"] not in (
            "cancel/batch1/first",
            "affine/identity",
            "issue1894/batch1",
            "issue1894/batch2",
        ):
            continue
        new_row = next(
            row
            for row in after_rows
            if (row["device"], row["case"]) == (old_row["device"], old_row["case"])
        )
        summary.append(
            {
                "device": old_row["device"],
                "case": old_row["case"],
                "before_accepted": old_row["accepted"],
                "after_accepted": new_row["accepted"],
                "before_native_called": old_row["native_called"],
                "after_native_called": new_row["native_called"],
                "old_aggregate_abs": old_row["aggregate_abs_max"],
                "per_face_abs_max": new_row["face_residual_abs_max"],
                "before_iou_0_0": old_row["iou_0_0"],
                "after_iou_0_0": new_row["iou_0_0"],
            }
        )
    with (output / "comparison-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", choices=("before", "after"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-before", type=Path)
    parser.add_argument("--plot-after", type=Path)
    args = parser.parse_args()
    if args.label:
        capture(args.label, args.output)
    if args.plot_before and args.plot_after:
        plot(args.plot_before, args.plot_after, args.output)


if __name__ == "__main__":
    main()
