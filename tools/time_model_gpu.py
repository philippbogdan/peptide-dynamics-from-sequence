#!/usr/bin/env python3
"""GPU time of the model on one card, to set beside the MD's.

    python tools/time_model_gpu.py PEPTIDE ...      -> results/cost/model_t4.json (named after the card it ran on)

The jump process runs on the GPU in PyTorch (train/network.py, the implementation the weights were trained in;
the steps between two frames are captured as one CUDA graph), from Metropolis starts on the GPU; torsions and
atoms are then made on the CPU by pepdyn. As for the MD, the whole wall time is charged to the card. Two amounts
per peptide: 1 microsecond as 100 trajectories of 10 ns, and 2.49 microseconds as 200 trajectories of 12.45 ns.
Needs torch with CUDA.
"""
import json
import os
import sys
import time
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "train"))
import network as N                                       # noqa: E402
import pepdyn                                             # noqa: E402
from pepdyn import model as M                             # noqa: E402


def load_network(mod):
    """The published weights in the PyTorch network."""
    meta = mod.meta
    vocab = types.SimpleNamespace(types=meta["residue_types"], n_types=5 + len(meta["residue_types"]),
                                  n_states=meta["n_states"], max_v=meta["max_v"], state_id={})
    for (rn, kind), n, first in zip(meta["site_keys"], meta["site_n"], meta["site_first_id"]):
        for s in range(n):
            vocab.state_id[((rn, kind), s)] = first + s
    net = N.KineticNet(vocab, W=meta["W"], d=meta["d"], h=meta["h"])
    z = np.load(os.path.join(HERE, "models", mod.name + ".npz"))
    names = {n.replace(".", "_"): n for n in net.state_dict()}
    net.load_state_dict({names[k[2:]]: torch.as_tensor(z[k]) for k in z.files if k.startswith("w_")})
    net.g_bound = float(meta["g_bound"])
    return net.cuda().eval(), vocab


@torch.no_grad()
def starts(net, pep, n, sweeps=200, seed=0):
    """Metropolis sampling of exp(-E) with single-site proposals, on the GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.zeros(n, pep.S, dtype=torch.long, device="cuda")
    E = net.energy(pep, x)
    for _ in range(sweeps):
        for c in range(pep.S):
            if int(pep.t["site_n"][c]) < 2:
                continue
            prop = x.clone()
            prop[:, c] = torch.randint(0, int(pep.t["site_n"][c]), (n,), generator=g, device="cuda")
            Ep = net.energy(pep, prop)
            acc = torch.rand(n, generator=g, device="cuda") < torch.exp(-(Ep - E)).clamp(max=1.0)
            x, E = torch.where(acc[:, None], prop, x), torch.where(acc, Ep, E)
    return x


def main():
    mod = pepdyn.load_model()
    net, vocab = load_network(mod)
    rows = []
    for name in sys.argv[1:]:
        p = mod.peptide(name)
        row = dict(name=name, sites=int(p.S))
        for label, n_traj, ns in (("1000 ns", 100, 10.0), ("2490 ns", 200, 12.45)):
            torch.cuda.synchronize()
            t0 = time.time()
            pep = N.to_device(N.PeptideIndex(vocab, p.residues, p.sites, [tuple(k) for k in p.keys],
                                             temperature=mod.meta["temperature_K"]), "cuda")
            frames = N.simulate(net, pep, starts(net, pep, n_traj), ns * 1000.0, 10.0, mod.meta["dt_ps"], "cuda")
            torch.cuda.synchronize()
            site = frames.cpu().numpy().astype(np.int64)
            t1 = time.time()
            X = pepdyn.frames_from_states(mod, p, site.reshape(-1, p.S), np.random.default_rng(0))[2].astype(np.float32)
            t2 = time.time()
            row[label] = dict(dynamics_s=t1 - t0, atoms_s=t2 - t1, total_s=t2 - t0, gpu_s_per_ns=(t2 - t0) / (n_traj * ns))
        # the chain on the card is the chain of the numpy code: largest difference between their site populations
        table = M.rate_table(mod, p)
        if table is not None:
            rng = np.random.default_rng(1)
            ref = M.simulate(mod, p, M.equilibrium_starts(mod, p, 200, rng, table), 12.45, rng, table=table)
            pop = lambda a, c: np.bincount(a[..., c].ravel(), minlength=p.site_n[c]) / a[..., c].size
            row["largest_population_difference_from_numpy_chain"] = float(max(np.abs(pop(site, c) - pop(ref, c)).max() for c in range(p.S)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    card = torch.cuda.get_device_name(0)
    out = os.path.join(HERE, "results", "cost", "model_" + card.split()[-1].lower() + ".json")
    json.dump(dict(card=card, torch=torch.__version__, peptides=rows), open(out, "w"), indent=1)
    print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
