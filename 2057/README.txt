IoU3D per-face coplanarity validation
====================================

Related to https://github.com/facebookresearch/pytorch3d/issues/1894.
The original RTX 3090 hardware-dependent discrepancy was NOT reproduced
on this environment. This contribution fixes a separate, deterministic
cross-face cancellation defect in the same validation function.

Measured counterexample
----------------------
Starting with the documented unit box, set vertex 3 z=0.25 and vertex 7
z=0.75. The six signed face residuals are [0.25, 0, 0, 0, 0, -0.25].
Their aggregate is zero although two faces exceed the default 1e-4
tolerance by a factor of 2500. The baseline accepts the illegal box and
calls the native IoU kernel; the repaired public API raises
ValueError("Plane vertices are not coplanar") before entering the kernel.
This behavior agrees on CPU and both NVIDIA B200 GPUs.

Files
-----
- comparison-summary.csv: compact actual before/after measurements.
- validation-matrix.csv: 138 actual case records, 69 per checkout;
  after has 69 PASS, before has 54 PASS and 15 REPRODUCED baseline bugs.
- before/after-matrix.csv: original per-checkout records.
- before/after-face-residuals.csv: signed residuals for every face, batch
  element, and both public API inputs.
- before/after-environment.json: version and hardware metadata.
- coplanarity-comparison.png/.svg: plot generated from the captures.
- reproduce.py: captures real public API and native extension behavior;
  patch(wraps=...) is a call observer, not a native kernel replacement.

Coverage
--------
Each checkout runs CPU/CUDA:0/CUDA:1 over cancelled and single-face
violations, batches of one/three, both API input positions, power-of-two
tolerance boundaries, five valid affine transforms, unchanged zero-area
errors, and the original issue fixture in batches of one/two.
Valid affine overlap cases use two equally transformed boxes shifted by
half a unit before transformation: expected IoU is 1/3 off diagonal and
1 on diagonal, with intersection volume scaled by |det(transform)|.
The invalid box's baseline native numeric outputs are diagnostic records,
not reference geometric volumes or claims of mathematically valid IoU.

Reproduce
---------
Build the actual native extension in both the baseline and fixed checkout.
In commands below, BASELINE, FIXED, SCRIPT and OUT are paths chosen by you;
SCRIPT is this directory's reproduce.py and OUT is a shared output folder.

  cd "$BASELINE"
  PYTHONPATH="$BASELINE" python "$SCRIPT" --label before --output "$OUT"
  cd "$FIXED"
  PYTHONPATH="$FIXED" python "$SCRIPT" --label after --output "$OUT"
  PYTHONPATH="$FIXED" python "$SCRIPT" --plot-before "$OUT" --plot-after "$OUT" --output "$OUT"

Python dependencies are PyTorch, a built PyTorch3D checkout and matplotlib.
Captures use float32 boxes; tolerance boundaries use exactly representable
epsilon=2^-10 and offsets epsilon/2, epsilon, and 2*epsilon. The strict
comparison remains distance < epsilon, so equality is rejected.
Two faces each at epsilon/2 with the same residual sign are accepted by
the repair and incorrectly rejected by the baseline's aggregate check.

Validation limits
-----------------
Linux, Python 3.12, PyTorch 2.11.0+cu130, CUDA 13.0, two B200 GPUs.
No RTX 3090, downstream training, performance benchmark, or unsupported
box3d_overlap backward is claimed.

Four added permanent test methods pass; ten regression subcases fail
when the new test module is loaded with the untouched baseline package.
Complete IoU module passes with Python/NumPy/PyTorch seed zero. With
Python random seed six, an existing CPU overlap tolerance assertion fails
on both checkouts with exactly the same maximum difference
5.728006362915039e-05 (relative 0.0001507101987954229).
The native kernels and comparison tolerances are unchanged.

Captured source commits:
- Baseline: 88e182f989c80836f4bd744e0d9cb1852762ce01
- Repair: 4b2cd2e86dfbd7ab137007907f4c6bcc2f2b5580
