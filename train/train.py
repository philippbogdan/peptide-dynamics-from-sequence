#!/usr/bin/env python3
"""Train the network on a prepared set (train/prepare.py).

    python train/train.py PREPARED.pkl OUT_DIR STEPS [PEPTIDES_PER_STEP] [RUNS_PER_PEPTIDE]

Holds out 5% of the peptides to report the validation negative log-likelihood per frame. Writes OUT_DIR/net.pt and
OUT_DIR/train_log.txt. Options by environment: GBOUND (bound on the prefactor correction, 4 for the published
model), SEED, LR, WD. The published model: 6,000 steps (results/training/train_log.txt).
"""
import os
import pickle
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network as N        # noqa: E402


def main():
    prep, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    per_step = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    runs = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
    os.makedirs(out, exist_ok=True)
    lf = open(os.path.join(out, "train_log.txt"), "a")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    d = pickle.load(open(prep, "rb"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}; {len(d['peptides'])} peptides")
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    net = N.KineticNet(d["vocab"], W=2, d=64, h=192).to(device)
    net.init_baseline(d["F0"], d["G0"], g_bound=float(os.environ.get("GBOUND", 4.0)))
    peps = d["peptides"]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(peps))
    rng = np.random.default_rng(seed)
    n_val = max(1, len(peps) // 20)
    val, trn = [peps[i] for i in order[:n_val]], [peps[i] for i in order[n_val:]]
    for p in peps:
        p["obj"] = N.to_device(types.SimpleNamespace(**p["idx"]), device)
    lr = float(os.environ.get("LR", 2e-3))
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=lr, weight_decay=float(os.environ.get("WD", 1e-4)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    dt = d["dt_ps"]

    def nll_per_frame(p, k=None):
        """Negative log-likelihood per frame of one peptide, from all its runs or from k drawn at random."""
        cfg, rl, nx, has = p["seg"]
        n = len(rl)
        idx, scale = (np.arange(n), 1.0) if k is None or k >= n else (rng.integers(0, n, size=k), n / k)
        tot = 0.0
        for s in range(0, len(idx), 8192):         # chunks keep memory bounded
            sl = idx[s:s + 8192]
            dev = lambda a: torch.as_tensor(np.asarray(a[sl]).astype(np.int64) if a.dtype != bool else np.asarray(a[sl]), device=device)
            tot = tot + N.nll(net, p["obj"], dev(cfg), dev(rl), dev(nx), dev(has), dt).sum()
        return tot * scale / p["n_frames"]

    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for i in rng.choice(len(trn), size=min(per_step, len(trn)), replace=False, p=np.full(len(trn), 1.0 / len(trn))):
            loss = nll_per_frame(trn[i], runs)
            (loss / per_step).backward()
            tot += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                v = np.mean([float(nll_per_frame(p, 20000)) for p in val])
            log(f"step {step} train {tot / per_step:.5f} val {v:.5f} t {time.time() - t0:.0f}s")
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    log(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
