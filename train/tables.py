#!/usr/bin/env python3
"""Torsion tables of corpus trajectories, the input of train/prepare.py.

    python train/tables.py CORPUS_DIR OUT_DIR STRIDE NAME ...

CORPUS_DIR holds NAME.npz (positions in nm, as in ManyPeptidesMD) and NAME.pdb. Training trajectories have a frame
every 1 ps, so STRIDE 10 gives the 10 ps frames the published model was fitted to. Writes OUT_DIR/NAME.npz with
res, tors (frames, residues, 7) and dt_ps.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from pepdyn import build                                   # noqa: E402
from pepdyn.model import ONE_TO_THREE                      # noqa: E402


class Peptide:
    def __init__(self, name):
        self.residues = [ONE_TO_THREE[c] for c in name]


class Atoms:
    def __init__(self, path):
        lines = [l for l in open(path) if l.startswith(("ATOM", "HETATM"))]
        self.index = {(int(l[22:26]), l[12:16].strip()): k for k, l in enumerate(lines)}


def main():
    corpus, out, stride = sys.argv[1], sys.argv[2], int(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[4:]:
        X = np.load(os.path.join(corpus, name + ".npz"))["positions"][::stride]
        pep, atoms = Peptide(name), Atoms(os.path.join(corpus, name + ".pdb"))
        T = np.concatenate([build.measure(pep, atoms, X[s:s + 50000] * 10.0) for s in range(0, len(X), 50000)])
        np.savez_compressed(os.path.join(out, name + ".npz"), res=np.array(pep.residues), tors=T.astype(np.float16),
                            dt_ps=1.0 * stride)
        print(name, T.shape, flush=True)


if __name__ == "__main__":
    main()
