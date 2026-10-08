import random
import sys
import unittest
import torch
from tests.test_iou_box3d import TestIoU3D

random.seed(0)
names = [
    'test_iou_batched_worker_reuse_cpu',
    'test_iou_batched_worker_reuse_cuda',
    'test_iou_cuda_device_guard',
    'test_iou_empty_inputs_cpu',
    'test_iou_empty_inputs_cuda',
    'test_iou_gh1805_cpu',
    'test_iou_gh1805_cuda',
    'test_iou_nondefault_stream_cuda',
    'test_iou_repeated_invocations_cpu',
    'test_iou_repeated_invocations_cuda',
    'test_iou_worker_counts_cpu',
    'test_iou_worker_counts_cuda',
]
suite = unittest.TestSuite(TestIoU3D(name) for name in names)
result = unittest.TextTestRunner(verbosity=2).run(suite)
for device in range(torch.cuda.device_count()):
    torch.cuda.synchronize(device)
sys.exit(not result.wasSuccessful())
