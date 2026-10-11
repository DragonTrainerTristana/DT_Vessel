# P0 진단(측정만, 학습·저장 없음): DAgger 학생 체크포인트와 선생님 라벨을 같은 상태에서 비교.
#   driver=teacher|student 로 배를 몰면서 (선생님 라벨, 학생 평균 행동)을 모음 → 라벨 분포·조건부 일치율.
import os, sys, json
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), 'imitation'))
import torch
import config as cfg
import vessel_gym as vg
from vessel_gym_train import parse_obs, FrameStack, make_others_msg, comm_gather
from ckpt_io import restore_policy, make_env_from_snapshot
from vo_teacher import Teacher

ckpt, driver, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
BURN, T, TOL, CLIP = int(sys.argv[4]) if len(sys.argv) > 4 else 300, int(sys.argv[5]) if len(sys.argv) > 5 else 300, 0.2, 0.995
dev = 'cuda'
torch.manual_seed(seed)
r = restore_policy(ckpt, dev, tag='[diag]')
policy = r.policy; policy.eval()
E, N = 128, 16
env = make_env_from_snapshot(r.snap, device=dev, num_envs=E, seed=seed, n_vessels=N, tag='[diag]')
teacher = Teacher('vo56h150', env)
obs = env.reset(); radar, goal, self_s, sit = parse_obs(obs)
fs = FrameStack(E, N, dev); fs.reset_all(radar)
om0 = make_others_msg(env, 'OFF', E, N, dev)
on = (r.arm == 'ON'); K = int(r.max_partners or cfg.MAX_COMM_PARTNERS)
lab, stu, smp, seen = [], [], [], []
oc = torch.zeros(5, device=dev)
for t in range(BURN + T):
    x = fs.get()
    with torch.no_grad():
        om = comm_gather(policy, env, x, goal, self_s, sit, K)[0] if on else om0
        act_s, _, mean_s, _ = policy.ctr_actor(x, goal, self_s, om, sit)
        a_star, choice = teacher.act(env)
        d = torch.linalg.norm(env.pos.unsqueeze(1) - env.pos.unsqueeze(2), dim=-1) + torch.eye(N, device=dev) * 1e9
    if t >= BURN:
        lab.append(a_star[..., 0].clamp(-CLIP, CLIP).flatten()); stu.append(torch.tanh(mean_s[..., 0]).flatten())
        smp.append(act_s[..., 0].flatten()); seen.append((d.min(-1).values <= 56.0).flatten())
    a_exec = a_star if driver == "teacher" else (torch.tanh(mean_s) if driver == "student_mean" else act_s)
    obs, _, done, outcome = env.step(a_exec)
    if t >= BURN:
        oc += torch.bincount(outcome.reshape(-1), minlength=5)[:5].float()
    teacher.after_step(env, a_exec, done)
    radar, goal, self_s, sit = parse_obs(obs); fs.push(radar, done)
lab, stu, smp, seen = torch.cat(lab), torch.cat(stu), torch.cat(smp), torch.cat(seen)
nz = lab.abs() > 1e-6
agree = (stu - lab).abs() < TOL
pc = lambda m: 100.0 * float(m.float().mean()) if m.numel() else float('nan')
n = float(oc[1:].sum())
out = dict(ckpt=os.path.basename(ckpt), driver=driver, seed=seed, samples=int(lab.numel()),
           label_nonzero_pct=pc(nz), ship_within_56m_pct=pc(seen),
           agree_all_pct=pc(agree), agree_when_label_zero_pct=pc(agree[~nz]), agree_when_label_nonzero_pct=pc(agree[nz]),
           mean_abs_label_nonzero=float(lab[nz].abs().mean()), mean_abs_student_when_label_nonzero=float(stu[nz].abs().mean()),
           mean_abs_student_when_label_zero=float(stu[~nz].abs().mean()),
           sign_match_when_label_nonzero_pct=pc((torch.sign(stu[nz]) == torch.sign(lab[nz]))),
           sampled_std_a0=float((smp - stu).std()),      # 실행 행동(act_s[0]) − tanh(평균): 샘플 잡음 크기
           eps=n, goal_pct=100.0 * float(oc[1]) / n if n else None, vColl_pct=100.0 * float(oc[2]) / n if n else None)
print('[diag] ' + json.dumps(out, ensure_ascii=False), flush=True)
