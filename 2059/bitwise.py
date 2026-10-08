import importlib.util, json, sys, torch
from pytorch3d.transforms import random_rotations
def load(p):
    s = importlib.util.spec_from_file_location('_C', p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
which, so, fx = sys.argv[1], sys.argv[2], sys.argv[3]
ext = load(so)
torch.manual_seed(0)
unit = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=torch.float32) - 0.5
def rand(n):
    return (unit[None] * (0.5 + torch.rand(n, 1, 3))) @ random_rotations(n).transpose(1, 2) + 0.3 * torch.randn(n, 1, 3)
p = json.load(open(fx))
b1 = torch.cat([rand(200), torch.tensor(p['boxes1']).reshape(1, 8, 3), unit[None]])
b2 = torch.cat([rand(150), torch.tensor(p['boxes2']).reshape(1, 8, 3), unit[None]])
v, i = ext.iou_box3d(b1.cuda(), b2.cuda())
torch.save((v.cpu(), i.cpu()), f'{sys.argv[4]}')
