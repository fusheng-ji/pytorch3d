import json
import math
from pathlib import Path
import statistics
import torch
from pytorch3d import _C
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import axis_angle_to_matrix

# Deterministic oriented boxes with well separated plane positions.
base = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=torch.float32) - 0.5
generator = torch.Generator().manual_seed(17771805)
rotations = axis_angle_to_matrix(torch.randn(27,3,generator=generator) * 0.6)
lengths = 0.8 + torch.rand(27,1,3,generator=generator) * 0.9
centers = torch.randn(27,1,3,generator=generator) * 0.1
boxes = (base[None] * lengths) @ rotations.transpose(1,2) + centers
first, second = boxes[:12], boxes[12:]
reference = box3d_overlap(first, second)
first, second = first.cuda(), second.cuda()
actual = box3d_overlap(first, second)
reverse = box3d_overlap(second, first)
torch.cuda.synchronize()
for gpu,cpu,swapped in zip(actual,reference,reverse):
    torch.testing.assert_close(gpu.cpu(),cpu,rtol=1e-4,atol=1e-6)
    torch.testing.assert_close(gpu,swapped.t(),rtol=1e-4,atol=1e-6)
    assert torch.isfinite(gpu).all()
assert actual[0].min() >= 0
assert actual[1].min() >= 0 and actual[1].max() <= 1
samples=[]
for _ in range(5): _C.iou_box3d(first,second)
for _ in range(31):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    _C.iou_box3d(first,second)
    end.record()
    end.synchronize()
    samples.append(start.elapsed_time(end))
report={'seed':17771805,'shape':[12,15],'distribution':'independently rotated, heavily overlapping boxes','cpu_cuda_max_abs_error':[float((g.cpu()-c).abs().max()) for g,c in zip(actual,reference)],'symmetry_max_abs_error':[float((g-s.t()).abs().max()) for g,s in zip(actual,reverse)],'median_cuda_ms':statistics.median(samples),'samples_ms':samples}
Path('build/iou-capacity-validation/rotated-stress.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
