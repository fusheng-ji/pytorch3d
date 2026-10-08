"""Compile only iou_box3d.cu from <src_dir>, link with cached objects into <out_so>."""
import json, pathlib, shlex, subprocess, sys
D = pathlib.Path(__file__).resolve().parents[1]
m = json.loads((D / 'native/build-manifest.json').read_text())
src_dir, out_dir, extra = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
out_dir.mkdir(parents=True, exist_ok=True)
cmd = next(c for n, c in m['recompiled'] if n == 'iou_box3d/iou_box3d.cu')
cmd = list(cmd)
obj = out_dir / 'iou_box3d.o'
i = cmd.index('-c'); cmd[i + 1] = str(src_dir / 'iou_box3d/iou_box3d.cu')
cmd[cmd.index('-o') + 1] = str(obj)
cmd.insert(1, f'-I{src_dir}')  # variant headers win over the shared csrc tree
cmd += extra
r = subprocess.run(cmd, text=True, capture_output=True)
(out_dir / 'ptxas.log').write_text(r.stdout + r.stderr)
r.check_returncode()
link = [str(obj) if a.endswith('/native/iou_box3d/iou_box3d.o') else a for a in m['link']]
link[link.index('-o') + 1] = str(out_dir / '_C.so')
subprocess.run(link, check=True)
print(out_dir / '_C.so')
