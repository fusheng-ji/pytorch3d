"""Generate the box3d_overlap evaluation scenarios and save them to sys.argv[1]."""
import sys
import torch
from pytorch3d.transforms import random_rotations

UNIT = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=torch.float32)
torch.manual_seed(1234)
g = torch.Generator().manual_seed(1234)

def aabb(lo, size):
    return UNIT[None] * size[:, None] + lo[:, None]

def rot(boxes, R, t=None):
    out = boxes @ R.transpose(-1, -2)
    return out if t is None else out + t

cases = {}
# 1. random rotated boxes at several scales
for scale in (1e-2, 1.0, 1e2):
    n = 120
    def rb(n):
        c = (UNIT - 0.5)[None] * (0.4 + torch.rand(n, 1, 3, generator=g)) @ random_rotations(n).transpose(1, 2)
        return (c + 0.25 * torch.randn(n, 1, 3, generator=g)) * scale
    cases[f"rotated_scale_{scale:g}"] = (rb(n), rb(n + 7))
# 2. grid-snapped axis-aligned boxes: shared planes, touching, identical, nested
def grid(n):
    lo = torch.randint(0, 4, (n, 3), generator=g).float() * 0.25
    size = torch.randint(1, 5, (n, 3), generator=g).float() * 0.25
    return aabb(lo, size)
a, b = grid(150), grid(160)
cases["grid_axis_aligned"] = (a, b)
# 3. same grid boxes under a common rotation + translation (coplanarity in general position)
R = random_rotations(1)[0]; t = torch.tensor([0.3, -1.2, 2.0])
cases["grid_common_rotation"] = (rot(a, R, t), rot(b, R, t))
# 4. identical rotated boxes (self IoU = 1) and slightly perturbed copies
r = random_rotations(80)
base = (UNIT - 0.5)[None] * (0.3 + torch.rand(80, 1, 3, generator=g)) @ r.transpose(1, 2)
cases["self_identical"] = (base, base.clone())
cases["self_perturbed_1e-4"] = (base, base + 1e-4 * torch.randn(base.shape, generator=g))
# 5. nested boxes sharing one corner (three coplanar faces each)
lo = torch.zeros(60, 3); big = aabb(lo, torch.full((60, 3), 1.0)); small = aabb(lo, 0.1 + 0.8 * torch.rand(60, 3, generator=g))
Rn = random_rotations(1)[0]
cases["nested_shared_corner_rotated"] = (rot(big, Rn), rot(small, Rn))

for k in ("grid_axis_aligned", "grid_common_rotation"):
    a, b = cases[k]
    cases[k + "_scale_0.01"] = (a * 0.01, b * 0.01)
torch.save(cases, sys.argv[1])
