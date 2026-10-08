import importlib.util
import json
from pathlib import Path
import sys
import torch

spec = importlib.util.spec_from_file_location('_C', sys.argv[1])
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
fixture = json.loads(Path(__file__).with_name('fixture1805.json').read_text())
first, second = [torch.tensor(fixture[key],dtype=torch.float32).reshape(1,8,3) for key in ('boxes1','boxes2')]
report = {'extension':str(Path(sys.argv[1]).resolve()),'directions':[]}
for a,b in ((first,second),(second,first)):
    cpu = extension.iou_box3d(a,b)
    gpu = extension.iou_box3d(a.cuda(),b.cuda())
    torch.cuda.synchronize()
    report['directions'].append({'cpu':[value.item() for value in cpu], 'cuda':[value.item() for value in gpu]})
print(json.dumps(report))
