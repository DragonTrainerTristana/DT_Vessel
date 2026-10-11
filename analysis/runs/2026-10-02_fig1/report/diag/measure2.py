# dagger_init.py 그대로 + 15초마다 VRAM(allocated/reserved/최대) 찍는 스레드 (코드 수정 없음)
import os, runpy, sys, threading, time, torch
out, extra = sys.argv[1], sys.argv[2:]
sys.argv = ['imitation/dagger_init.py', '--init', 'C:/Users/OSH/VESSEL_checkpoints/comm_intent/f_init_d6_s47.pt', '--seed', '47',
            '--teacher', 'vo56h150', '--save', out] + extra
t0 = time.time(); gb = 1 / 2**30; stop = False


def mon():
    while not stop:
        time.sleep(15)
        if torch.cuda.is_initialized() and torch.cuda.memory_reserved() * gb > 11.0:
            print(f"[vram] ABORT reserved={torch.cuda.memory_reserved() * gb:.2f} GB > 11 GB at t={time.time() - t0:.0f}s", flush=True); os._exit(3)
        if torch.cuda.is_initialized():
            print(f"[vram] t={time.time() - t0:5.0f}s alloc={torch.cuda.memory_allocated() * gb:.2f} reserved={torch.cuda.memory_reserved() * gb:.2f} "
                  f"peak_alloc={torch.cuda.max_memory_allocated() * gb:.2f} peak_reserved={torch.cuda.max_memory_reserved() * gb:.2f} GB", flush=True)


threading.Thread(target=mon, daemon=True).start()
try:
    runpy.run_path('imitation/dagger_init.py', run_name='__main__')
finally:
    stop = True
    print(f"[vram] END peak_alloc={torch.cuda.max_memory_allocated() * gb:.2f}GB peak_reserved={torch.cuda.max_memory_reserved() * gb:.2f}GB wall={time.time() - t0:.0f}s", flush=True)
