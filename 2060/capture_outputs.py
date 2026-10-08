"""Run one native build on every saved case; save CPU and CUDA volumes and IoUs."""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("_C", sys.argv[1]); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
cases = torch.load(sys.argv[2]); out = {}
for name, (b1, b2) in cases.items():
    out[name] = {"cpu": ext.iou_box3d(b1, b2), "cuda": tuple(t.cpu() for t in ext.iou_box3d(b1.cuda(), b2.cuda()))}
torch.save(out, sys.argv[3])
