#!/usr/bin/env python3
"""Rebuild a file of data/reference from the public corpus.

    python tools/make_reference.py CVNR validation        downloads the trajectory (about 0.8 GB) and its PDB
    python tools/make_reference.py CVNR validation --corpus DIR   reads DIR/CVNR.npz and DIR/CVNR.pdb instead
    python tools/make_reference.py ANKSMIEA test --sites-only --out data/octapeptides

The corpus is ManyPeptidesMD (https://huggingface.co/datasets/transferable-samplers/many-peptides-md, MIT):
5 microseconds of implicit-solvent MD per held-out tetrapeptide, a frame every 10 ps. The script measures the
torsions of every frame, drops the first 10 ns, splits the rest into two halves, finds the peptide's metastable
states from both halves (pepdyn/states.py: discover), and stores the state definition, the state of every frame,
the main states and, for each site of the model's alphabet, its populations and jumps in each half.

Needs numpy and deeptime (k-means and PCCA+ in the state rule).
"""
import argparse
import io
import os
import sys
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pepdyn                                             # noqa: E402
from pepdyn import build, compare, emit, states           # noqa: E402

BASE = "https://huggingface.co/datasets/transferable-samplers/many-peptides-md/resolve/main"
FOLDER = dict(test="test", validation="val")
EQUILIBRATION_FRAMES = 1000                               # 10 ns


class Atoms:
    """Where each heavy atom sits in the corpus's atom order, in the form build.measure expects."""

    def __init__(self, pdb_text):
        self.index, k = {}, 0
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                self.index[(int(line[22:26]), line[12:16].strip())] = k
                k += 1


def fetch(peptide, split, corpus):
    if corpus:
        return np.load(os.path.join(corpus, peptide + ".npz"))["positions"], open(os.path.join(corpus, peptide + ".pdb")).read()
    folder = FOLDER[split]
    with urllib.request.urlopen(f"{BASE}/trajectories/{folder}/{len(peptide)}AA/{peptide}.npz") as r:
        positions = np.load(io.BytesIO(r.read()))["positions"]
    with urllib.request.urlopen(f"{BASE}/pdb_tarfiles/{folder}.tar") as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()))
        member = [m for m in tar.getmembers() if m.name.endswith("/" + peptide + ".pdb")][0]
        return positions, tar.extractfile(member).read().decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("peptide")
    ap.add_argument("split", choices=sorted(FOLDER))
    ap.add_argument("--corpus", help="folder holding PEPTIDE.npz and PEPTIDE.pdb, instead of downloading")
    ap.add_argument("--sites-only", action="store_true", help="skip the state rule: only the per-site statistics")
    ap.add_argument("--seed", type=int, default=0, help="seed of the k-means in the state rule (default 0, as stored)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "reference"))
    a = ap.parse_args()
    mod = pepdyn.load_model()
    pep = mod.peptide(a.peptide)
    positions, pdb = fetch(a.peptide, a.split, a.corpus)
    T = np.concatenate([build.measure(pep, Atoms(pdb), positions[s:s + 50000] * 10.0)       # nm to angstrom
                        for s in range(0, len(positions), 50000)]).astype(np.float32)[EQUILIBRATION_FRAMES:]
    h = len(T) // 2
    sm, out = None, dict(peptide=a.peptide, split=a.split, states_ok=False)
    if not a.sites_only:
        X = emit.features(pep, T)
        sm, report = states.discover([X[:h], X[h:2 * h]], seed=a.seed)
        out.update(states_ok=sm is not None, best_timescales_ns=np.array(report["best_timescales_ns"][:8]))
    if sm is not None:
        d = [sm.assign(X[:h]), sm.assign(X[h:2 * h])]                # each half is assigned on its own
        out.update(mean=sm.mean, vecs=sm.vecs, centers=sm.centers, core_of_cluster=sm.core_of_cluster.astype(np.int16),
                   state_of_cluster=sm.state_of_cluster.astype(np.int16), n_states=sm.n_states, lag_ns=sm.lag_ns,
                   dt_ns=sm.dt_ns, md_states=np.concatenate(d).astype(np.int8), setting=np.array(report["setting"], float),
                   main=np.where(sm.msm(d).pi >= compare.MAIN_POPULATION)[0].astype(np.int16))
    V = int(pep.site_n.max())
    counts, jumps = np.zeros((2, pep.S, V), np.int64), np.zeros((2, pep.S), np.int64)
    for k in range(2):
        L = emit.site_states(mod, pep, T[k * h:(k + 1) * h].astype(np.float64))
        for c in range(pep.S):
            counts[k, c] = np.bincount(L[:, c], minlength=V)
            jumps[k, c] = (L[1:, c] != L[:-1, c]).sum()
    out.update(site_counts=counts, site_jumps=jumps, frames_per_half=h,
               site_names=np.array([f"{pep.residues[i]}{i + 1}:{kind}" for i, kind in pep.sites]))
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, a.peptide + ".npz"), **out)
    if a.sites_only:
        print(a.peptide, pep.S, "sites")
    else:
        print(a.peptide, "states:", sm.n_states if sm is not None else "none (no setting reproduces its relaxation times)",
              "| slowest relaxation times (ns):", np.round(report["best_timescales_ns"][:4], 2))


if __name__ == "__main__":
    main()
