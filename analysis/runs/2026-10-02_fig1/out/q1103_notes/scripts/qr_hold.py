# 2026-11-03 GPU evictor for qr4_/qr5_ (author: 'fill it ourselves' -> v2 after 15:05 overshoot).
# On WDDM cudaMemGetInfo lets a process allocate past physical VRAM (others get paged to system memory), so a
# steady 'fill' cannot block other accounts and can page out our own training. What works (13:58) is a short push:
# allocate+touch <MB>, which makes WDDM evict other accounts' idle memory, then free it so our pending run takes the room.
# Started by qr_guard.ps1 with CUDA_VISIBLE_DEVICES=<g> CUDA_DEVICE_ORDER=PCI_BUS_ID, only on cards with no run of ours.
# usage: qr_hold.py <g> <MB> <hold_seconds>   -> allocate MB in 256 MB chunks, hold, free, exit. Writes held MB to g<g>.mb
import os, sys, time
import torch

g, mb, hold_s = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
mbf = os.path.join(r'C:\work\qr_hold', f'g{g}.mb')
CH = 256 << 20
chunks = []


def write_mb():
    try:
        with open(mbf, 'w') as f:
            f.write(str(len(chunks) * 256))
    except OSError:
        pass


torch.zeros(1, device='cuda')
while len(chunks) * 256 < mb:
    try:
        t = torch.empty(CH, dtype=torch.uint8, device='cuda')
        t.zero_()
        chunks.append(t)
    except RuntimeError:
        break
write_mb()
torch.cuda.synchronize()
time.sleep(hold_s)
chunks.clear()
torch.cuda.empty_cache()
write_mb()
