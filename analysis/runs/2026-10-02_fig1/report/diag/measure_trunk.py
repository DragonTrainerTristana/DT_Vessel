# vessel_gym_train.py(trunk 설정) 그대로 + 15초마다 VRAM 찍는 스레드. 11 GB 넘으면 스스로 중단 (코드 수정 없음, 저장은 스크래치)
import os, runpy, sys, threading, time, torch
tag, outdir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.getcwd())
sys.argv = ['vessel_gym_train.py', '--arm', 'ON', '--resume', 'C:/Users/OSH/VESSEL_checkpoints/comm_intent/f_dagger_d6_s47.pt', '--resume_at', '0',
            '--steps', '327680', '--comm_on_at', '0', '--envs', '128', '--vessels', '16', '--rollout', '32', '--ring', '1.0', '--crossing', '0',
            '--max_partners', '8', '--seed', '47', '--ckpt_every', '2', '--save', f'{outdir}/{tag}_trunk.pt', '--csv', f'{outdir}/{tag}_trunk.csv']
t0 = time.time(); gb = 1 / 2**30


def mon():
    while True:
        time.sleep(15)
        if torch.cuda.is_initialized():
            r = torch.cuda.memory_reserved() * gb
            print(f"[vram] t={time.time() - t0:5.0f}s alloc={torch.cuda.memory_allocated() * gb:.2f} reserved={r:.2f} "
                  f"peak_alloc={torch.cuda.max_memory_allocated() * gb:.2f} peak_reserved={torch.cuda.max_memory_reserved() * gb:.2f} GB", flush=True)
            if r > 11.0:
                print(f"[vram] ABORT reserved={r:.2f} GB > 11 GB at t={time.time() - t0:.0f}s", flush=True); os._exit(3)


threading.Thread(target=mon, daemon=True).start()
try:
    runpy.run_path('vessel_gym_train.py', run_name='__main__')
finally:
    print(f"[vram] END peak_alloc={torch.cuda.max_memory_allocated() * gb:.2f}GB peak_reserved={torch.cuda.max_memory_reserved() * gb:.2f}GB wall={time.time() - t0:.0f}s", flush=True)
