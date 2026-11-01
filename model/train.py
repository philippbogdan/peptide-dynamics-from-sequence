"""Fit the kinetic network to compressed training paths, preferably on one GPU.

Usage: python model/train.py PREP.pkl OUT_DIR STEPS [PEPTIDES_PER_STEP] [RUNS_PER_PEPTIDE]

The last two arguments default to 16 and 2048. A fixed 5% split of the prepared
peptides supplies validation negative log-likelihoods per frame. Outputs are
OUT_DIR/net.pt and an appended OUT_DIR/train_log.txt.

Environment settings: W=2, D=64, H=192 for network size; LR=0.002, WD=0.0001,
SEED=0 and GBOUND=2. Use GBOUND=4 for the supplied model configuration. CUDA is
selected when available; otherwise training runs on the CPU.
"""
import os, pickle, sys, time, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import ctmc as CT


def _load_prep(path):
    import gzip
    return pickle.load(gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb"))


def pep_obj(idx, device):
    """Restore a compact peptide index and its tensors on the selected device."""
    p = types.SimpleNamespace(**idx)
    return CT.to_device(p, device)


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

    d = _load_prep(prep)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}; {len(d['peptides'])} peptides")
    W, Dm, H = int(os.environ.get("W", 2)), int(os.environ.get("D", 64)), int(os.environ.get("H", 192))
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    net = CT.KineticNet(d["vocab"], W=W, d=Dm, h=H).to(device)
    net.init_baseline(d["F0"], d["G0"], g_bound=float(os.environ.get("GBOUND", 2.0)))
    peps = d["peptides"]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(peps))
    rng = np.random.default_rng(int(os.environ.get("SEED", 0)))
    n_val = max(1, len(peps) // 20)
    val, trn = [peps[i] for i in order[:n_val]], [peps[i] for i in order[n_val:]]
    for p in peps:
        p["obj"] = pep_obj(p["idx"], device)
        # Keep compact paths in CPU memory; transfer only the sampled run batches.
        p["t"] = p["seg"]
    opt = torch.optim.AdamW([q for q in net.parameters() if q.requires_grad], lr=float(os.environ.get("LR", 2e-3)),
                            weight_decay=float(os.environ.get("WD", 1e-4)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(os.environ.get("LR", 2e-3)), total_steps=steps,
                                                pct_start=0.05)
    dt = d["dt_ps"]

    def nll_per_frame(p, k=None):
        """Estimate loss per frame from all runs or a uniformly sampled subset."""
        cfg, rl, nx, has = p["t"]
        n = len(rl)
        if k is None or k >= n:
            idx = np.arange(n)
            scale = 1.0
        else:
            idx = rng.integers(0, n, size=k)
            scale = n / k
        dev = lambda a: torch.as_tensor(np.asarray(a[idx]).astype(np.int64) if a.dtype != bool else np.asarray(a[idx]),
                                        device=device)
        tot = 0.0
        # Limit temporary tensors when validation uses many residence runs.
        for s in range(0, len(idx), 8192):
            sl = idx[s:s + 8192]
            d2 = lambda a: torch.as_tensor(np.asarray(a[sl]).astype(np.int64) if a.dtype != bool else np.asarray(a[sl]),
                                           device=device)
            lo = CT.nll(net, p["obj"], d2(cfg), d2(rl), d2(nx), d2(has), dt)
            tot = tot + lo.sum()
        return tot * scale / p["n_frames"]

    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        tot = 0.0
        for i in rng.choice(len(trn), size=min(per_step, len(trn)), replace=False):
            l = nll_per_frame(trn[i], runs)
            (l / per_step).backward()
            tot += float(l.detach())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            with torch.no_grad():
                v = np.mean([float(nll_per_frame(p, 20000)) for p in val])
            log(f"step {step} train {tot / per_step:.5f} val {v:.5f} t {time.time() - t0:.0f}s")
    torch.save(net.state_dict(), os.path.join(out, "net.pt"))
    log(f"done {time.time() - t0:.0f} s; W {W} D {Dm} H {H}")


if __name__ == "__main__":
    main()
