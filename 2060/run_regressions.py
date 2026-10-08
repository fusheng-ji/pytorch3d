"""Run the IoU regression tests added for the capacity and polygon clipping fixes."""
import random
import sys
import unittest

import torch
from tests.test_iou_box3d import TestIoU3D

random.seed(0)
names = sorted(
    name
    for name in dir(TestIoU3D)
    if name.startswith("test_iou_")
    and any(key in name for key in ("pair_counts", "batched_pairs", "repeated", "nondefault_stream",
                                    "device_guard", "gh1805", "empty_inputs", "coplanar_faces", "rotated_exact"))
)
suite = unittest.TestSuite(TestIoU3D(name) for name in names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
for device in range(torch.cuda.device_count()):
    torch.cuda.synchronize(device)
sys.exit(not result.wasSuccessful())
