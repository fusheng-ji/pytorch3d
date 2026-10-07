Evidence for camera-space depth filtering (related issue: #1359)
================================================================

What is measured
----------------
The numerical CSV files capture the actual project_points_and_sample function
from separate baseline and patched source checkouts, using the same complete
native extension built at the main baseline. The fixed
input is four points: front (z=2), behind (z=-2), on the camera plane (z=0), and
small positive depth (z=0.0001, below the existing eps=0.01). A ramp feature map
and soft mask are bilinearly sampled through a real PerspectiveCameras instance.
Each point's sampled feature plus mask is differentiated with respect to points,
camera R/T, the feature map, and the soft mask. No implementation is replaced.

CPU numerical results (CUDA:0 and CUDA:1 agree within floating-point tolerance):

Point           Feature before -> after  Mask before -> after  Point gradient L1
front           1.083870 -> 1.083870      0.562478 -> 0.562478   0.814478 -> 0.814478
behind          1.144174 -> 0             0.584696 -> 0          0.855739 -> 0
camera plane    1.050000 -> 0             0.550000 -> 0          158.260864 -> 0
tiny positive   1.050000 -> 1.050000      0.550000 -> 0.550000   158.260864 -> 158.260864

Invalid points' camera, feature-map and soft-mask gradients are also exactly
zero after the fix. Front samples and gradients retain the original values.
The feature-map gradient L1 is 1 before and 0 after for each invalid point.
The plot is generated from the captured CPU CSV rows; raw values for all three
devices are preserved in the CSV files.

Validation matrix
-----------------
The submitted regression test module is loaded against each checkout. Six new
test methods produce 205 cases in this environment:
- 72: CPU/CUDA:0/CUDA:1 x four camera classes x three interpolation modes x
  image masking enabled/disabled; front, behind, camera plane and tiny-positive.
- 24: rotated and translated mixed views, two point batches, multidimensional
  point grids, multiple feature maps, different channels and image aspect ratios.
- 24: public ViewSampler depth filtering plus mismatched sequence IDs.
- 72: exact-zero finite invalid gradients and independent front sample/gradient
  agreement, across the same device/camera/interpolation/image-mask matrix.
- 12: actual ViewSampler -> Identity/AVG feature aggregator -> weighted loss
  -> backward, with target-view exclusion disabled.
- 1: CPU float32 finite-difference gradcheck with eps=1e-3, atol=1e-3, rtol=1e-2.

All 204 behavioral cases fail against the baseline and pass with the fix.
The derivative-consistency gradcheck passes both versions. Baseline failure
logs are expected evidence of the reported defect, not additional checkouts
to be fixed. The patched validation matrix contains 205 PASS records.

Reproduction
------------
Use the existing project installation instructions to build native extensions
in both checkouts, then use the Python environment with the required implicitron
dependencies (including hydra-core and omegaconf). Select each checkout explicitly
so that the imported modules and native extension come from that checkout:

python capture_camera_evidence.py --checkout baseline_checkout \
  --test-file patched_checkout/tests/implicitron/test_viewsampling.py \
  --variant baseline --output camera_evidence
python capture_camera_evidence.py --checkout patched_checkout \
  --test-file patched_checkout/tests/implicitron/test_viewsampling.py \
  --variant patched --output camera_evidence
python capture_camera_evidence.py --output camera_evidence --plot

The baseline is main commit 88e182f989c80836f4bd744e0d9cb1852762ce01.
Source/test hashes and any uncommitted source paths at capture time are recorded
in environment JSON files. The CSVs preserve full float values, not just the
rounded table above. Matplotlib is needed only to render the PNG/SVG.

Limits
------
Validation uses Linux, Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0 and two
NVIDIA B200s (one GPU model). No downstream model training or comprehensive
performance benchmark is claimed.

The unmodified public camera transforms cast Rotate/Translate to float32.
Therefore the requested CPU double-precision public-path gradcheck is not
supported by the current baseline camera implementation. Both actual versions'
float64 failures are captured in the environment JSON. The finite-difference
check uses float32 with an appropriately larger step instead; unrelated camera
dtype behavior is not changed by this PR.
