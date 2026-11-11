#!/usr/bin/env python3
"""Score the model against the reference MD of the held-out peptides, from the files in this repository.

    python evaluate.py                    every held-out peptide whose reference defines states -> results/held_out.json
    python evaluate.py GCDE CVNR          only these (printed, not saved)
    python evaluate.py --trade            error against simulated length, model and MD -> results/trade.json
    python evaluate.py --sites            per-residue states of all 40 held-out peptides -> results/sites.json
    python evaluate.py --sites --octapeptides   the same for 30 octapeptides -> results/sites_octapeptides.json
    python evaluate.py --references data/reference_seed1 ...   on the states another k-means seed gives

For each peptide the model generates 5 microseconds (200 trajectories of 25 ns, each started from exp(-E)/Z), the
torsions are measured on the heavy atoms it wrote, each frame is assigned to the reference's states
(data/reference/<PEPTIDE>.npz, which also holds the state of every MD frame), and the same Markov-model
estimators give populations, free energies, mean first-passage times and relaxation times for the model, for the
whole reference and for each of its two halves. Needs numpy only; about a minute per peptide on a laptop.
"""
import argparse
import glob
import json
import os
import time

import numpy as np

import pepdyn
from pepdyn import build, compare, emit, states
from pepdyn import model as M

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("mfpt_worst_factor", "f_rms_kT", "tv", "slowest_rel_err", "others_worst_rel_err")
TRADE_NS = (10, 20, 30, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 2490)
MAIN_SITE_POPULATION = 0.01
REFERENCES = os.path.join(HERE, "data", "reference")      # the state definitions used (--references)


def reference(name):
    path = os.path.join(REFERENCES, name + ".npz")
    z = np.load(path, allow_pickle=False)
    if not bool(z["states_ok"]):
        return None
    d = z["md_states"].astype(np.int64)
    h = len(d) // 2
    return dict(name=name, split=str(z["split"]), sm=states.StateModel.load(path), main=z["main"], A=d[:h], B=d[h:])


def model_states(mod, pep, sm, total_ns, n_traj, seed, table, block=20):
    """State labels of n_traj model trajectories totalling total_ns, read off the heavy atoms the model writes."""
    rng = np.random.default_rng(seed)
    x0 = M.equilibrium_starts(mod, pep, n_traj, rng, table)
    site = M.simulate(mod, pep, x0, total_ns / n_traj, rng, table=table)
    frames = site.shape[1]
    labels, again, left = [], 0.0, 0.0
    for s in range(0, n_traj, block):                       # a block of trajectories at a time keeps memory small
        part = site[s:s + block]
        _, top, X, o = pepdyn.frames_from_states(mod, pep, part.reshape(-1, pep.S).astype(np.int64), rng)
        feats = emit.features(pep, build.measure(pep, top, X.astype(np.float32)))
        labels += [sm.assign(feats[k * frames:(k + 1) * frames]) for k in range(len(part))]
        again += o["drawn_again"] * len(part) / n_traj
        left += o["left_overlapping"] * len(part) / n_traj
    return labels, dict(drawn_again=again, left_overlapping=left)


def summary(c):
    return {k: c[k] for k in KEYS + ("pass", "pass_all", "mfpt_pairs", "relaxation", "dF")}


def judge(mod, name, seed=0, total_ns=5000.0, n_traj=200, table=None):
    ref = reference(name)
    sm, main = ref["sm"], ref["main"]
    pep = mod.peptide(name)
    t0 = time.perf_counter()
    table = M.rate_table(mod, pep, pepdyn.TABLE_LIMIT) if table is None else table
    dm, overlaps = model_states(mod, pep, sm, total_ns, n_traj, seed, table)
    seconds = time.perf_counter() - t0
    md, mdA, mdB, mo = sm.msm([ref["A"], ref["B"]]), sm.msm([ref["A"]]), sm.msm([ref["B"]]), sm.msm(dm)
    return dict(name=name, split=ref["split"], n_states=sm.n_states, main_states=main.tolist(), lag_ns=sm.lag_ns,
                joint_configurations=pep.n_configs, seconds=seconds, overlaps=overlaps,
                model=summary(compare.compare(mo, md, main)), halves=summary(compare.compare(mdA, mdB, main)),
                shuffled_fails=not compare.compare(sm.msm(compare.shuffled([ref["A"], ref["B"]])), md, main)["pass_all"],
                doubled_clock_fails=not compare.compare(compare.DoubledClock(md), md, main)["pass_all"],
                populations=dict(md=md.pi.tolist(), half_a=mdA.pi.tolist(), half_b=mdB.pi.tolist(), model=mo.pi.tolist()),
                relaxation_ns=dict(md=md.ts[:5].tolist(), half_a=mdA.ts[:5].tolist(), half_b=mdB.ts[:5].tolist(),
                                   model=mo.ts[:5].tolist()))


def trade(mod, name, seed=0, table=None):
    """Errors against half B of the reference: the first l ns of half A, and l ns of the model."""
    ref = reference(name)
    sm, main = ref["sm"], ref["main"]
    pep = mod.peptide(name)
    table = M.rate_table(mod, pep, pepdyn.TABLE_LIMIT) if table is None else table
    mdB = sm.msm([ref["B"]])
    row = dict(name=name, split=ref["split"], md={}, model={})
    for ns in TRADE_NS:
        c = compare.compare(sm.msm([ref["A"][:int(round(ns / sm.dt_ns))]]), mdB, main)
        row["md"][ns] = {k: c[k] for k in KEYS}
        dm, _ = model_states(mod, pep, sm, ns, max(1, min(200, ns // 10)), seed + ns, table)
        c = compare.compare(sm.msm(dm), mdB, main)
        row["model"][ns] = {k: c[k] for k in KEYS}
    return row


def site_error(ref, cand, keep):
    """Root-mean-square of ln(candidate / reference population) over the kept site states, in kT."""
    with np.errstate(divide="ignore"):
        d = np.log(np.maximum(cand[keep], 1e-6) / ref[keep])
    return float(np.sqrt(np.mean(d ** 2)))


def sites(mod, name, seed=0, total_ns=20000.0, n_traj=800, table=None, folder="reference"):
    """Without any state definition of the whole peptide: the population of every state of every site (one
    backbone basin or rotamer of one residue) and how often each site jumps, for 20 microseconds of the model
    (2 for an octapeptide), for the whole reference and for its two halves."""
    z = np.load(os.path.join(HERE, "data", folder, name + ".npz"), allow_pickle=False)
    pep = mod.peptide(name)
    counts, jumps, h = z["site_counts"].astype(float), z["site_jumps"].astype(float), int(z["frames_per_half"])
    V = counts.shape[2]
    pA, pB, p = counts[0] / h, counts[1] / h, counts.sum(0) / (2 * h)
    keep = p >= MAIN_SITE_POPULATION
    rng = np.random.default_rng(seed)
    site = M.simulate(mod, pep, M.equilibrium_starts(mod, pep, n_traj, rng, table), total_ns / n_traj, rng, table=table)
    q = np.stack([np.bincount(site[:, :, c].ravel(), minlength=V) for c in range(pep.S)]) / site[:, :, 0].size
    rate_md = jumps.sum(0) / (2 * h * 0.01)                                   # jumps per ns
    rate_a, rate_b = jumps[0] / (h * 0.01), jumps[1] / (h * 0.01)
    rate_model = (site[:, 1:] != site[:, :-1]).sum((0, 1)) / (n_traj * (site.shape[1] - 1) * 0.01)
    ok = (jumps.min(0) >= 10) & (rate_model > 0)                               # sites that jump often enough to rate
    log_error = lambda a, b: float(np.median(np.abs(np.log(a[ok] / b[ok])))) if ok.any() else None
    return dict(name=name, split=str(z["split"]), joint_configurations=pep.n_configs, n_site_states=int(keep.sum()),
                n_sites_rated=int(ok.sum()), model_f_rms_kT=site_error(p, q, keep), halves_f_rms_kT=site_error(pB, pA, keep),
                model_rate_log_error=log_error(rate_model, rate_md), halves_rate_log_error=log_error(rate_a, rate_b),
                populations=dict(md=p.tolist(), half_a=pA.tolist(), half_b=pB.tolist(), model=q.tolist()),
                jumps_per_ns=dict(md=rate_md.tolist(), half_a=rate_a.tolist(), half_b=rate_b.tolist(), model=rate_model.tolist()),
                sites=[str(x) for x in z["site_names"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("peptides", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="repeat with this many seeds (seed, seed + 1, ...)")
    ap.add_argument("--trade", action="store_true")
    ap.add_argument("--sites", action="store_true")
    ap.add_argument("--octapeptides", action="store_true", help="with --sites: the 30 test octapeptides instead")
    ap.add_argument("--out", default=None, help="write the results to this file")
    ap.add_argument("--references", default=None, help="folder of state definitions, instead of data/reference")
    a = ap.parse_args()
    global REFERENCES
    REFERENCES = os.path.abspath(a.references) if a.references else REFERENCES
    folder = "octapeptides" if a.octapeptides else "reference"
    listed = os.path.join(HERE, "data", folder) if a.sites else REFERENCES
    names = a.peptides or sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(listed, "*.npz")))
    names = [n for n in names if a.sites or reference(n) is not None]
    mod = pepdyn.load_model("implicit-310K")
    runs = [dict(seed=a.seed + k, peptides=[]) for k in range(a.seeds)]
    for name in names:
        table = M.rate_table(mod, mod.peptide(name), pepdyn.TABLE_LIMIT)
        for run in runs:
            if a.sites:
                r = (sites(mod, name, run["seed"], 2000.0, 80, table, folder) if a.octapeptides
                     else sites(mod, name, run["seed"], table=table))
                run["peptides"].append(r)
                print(f"{name} ({r['split']}): site free energies, model {r['model_f_rms_kT']:.3f} kT, MD halves "
                      f"{r['halves_f_rms_kT']:.3f} kT; jump rates off by a factor, model "
                      f"{np.exp(r['model_rate_log_error']):.3f}, MD halves {np.exp(r['halves_rate_log_error']):.3f}", flush=True)
                continue
            r = trade(mod, name, run["seed"], table) if a.trade else judge(mod, name, run["seed"], table=table)
            run["peptides"].append(r)
            if a.trade:
                print(name, run["seed"], {ns: round(v["mfpt_worst_factor"], 2) for ns, v in r["model"].items()}, flush=True)
            else:
                print(f"{name} ({r['split']}, seed {run['seed']}): model " + " ".join(f"{r['model'][k]:.3f}" for k in KEYS)
                      + " | MD halves " + " ".join(f"{r['halves'][k]:.3f}" for k in KEYS) + f" | {r['seconds']:.0f} s", flush=True)
    if not a.peptides or a.out:
        out = a.out or os.path.join(HERE, "results", "trade.json" if a.trade else "sites_octapeptides.json" if a.octapeptides
                                    else "sites.json" if a.sites else "held_out.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(dict(model="implicit-310K", runs=runs), open(out, "w"), indent=1)
        print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
