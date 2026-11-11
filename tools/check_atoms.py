#!/usr/bin/env python3
"""How well do torsions and fixed bond geometry give back MD's atoms?

    python tools/check_atoms.py CORPUS_DIR GCDE WNMA ...   -> results/atoms.json

For each peptide, CORPUS_DIR holds PEPTIDE.npz (positions in nm, as in ManyPeptidesMD) and PEPTIDE.pdb. Every
100th frame of the MD is taken, its torsions are measured, the heavy atoms are rebuilt from those torsions with
the model's builder (same bond lengths and angles for every frame), and the rebuilt frame is compared with the MD
frame: root-mean-square deviation after superposition, and the closest approach of two atoms more than three
bonds apart, which MD never lets come within 2.2 angstrom.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pepdyn                                             # noqa: E402
from pepdyn import build                                  # noqa: E402
from pepdyn.pipeline import MIN_DISTANCE                  # noqa: E402


def rmsd(A, B):
    """Per frame, after the best superposition (Kabsch)."""
    A, B = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(np.einsum("fai,faj->fij", A, B))
    S[:, -1] *= np.sign(np.linalg.det(U @ Vt))
    return np.sqrt(np.maximum((A ** 2).sum((1, 2)) + (B ** 2).sum((1, 2)) - 2 * S.sum(1), 0) / A.shape[1])


def main():
    corpus, names = sys.argv[1], sys.argv[2:]
    stride = int(os.environ.get("STRIDE", 100))
    mod = pepdyn.load_model()
    rows = []
    for name in names:
        pep = mod.peptide(name)
        top = build.Topology(pep)
        atoms = [(int(l[22:26]), l[12:16].strip()) for l in open(os.path.join(corpus, name + ".pdb")) if l.startswith("ATOM")]
        order = [atoms.index((num, a)) for a, _, num, _ in top.atoms]
        X = np.load(os.path.join(corpus, name + ".npz"))["positions"][::stride][:, order].astype(float) * 10.0
        T = build.measure(pep, top, X)
        at = lambda num, a: X[:, top.index[(num, a)]]
        omega = np.stack([build.dihedral(at(i, "CA"), at(i, "C"), at(i + 1, "N"), at(i + 1, "CA"))
                          for i in range(1, len(pep.residues))], 1)
        _, R = build.coordinates(pep, T, None, omega=omega)
        pairs = build.far_pairs(top)
        d = rmsd(R, X)
        close_md, close_rebuilt = build.closest(X, pairs), build.closest(R, pairs)
        rows.append(dict(name=name, frames=len(X), rmsd_median=float(np.median(d)), rmsd_95=float(np.percentile(d, 95)),
                         md_closest_median=float(np.median(close_md)), md_overlapping=float(np.mean(close_md < MIN_DISTANCE)),
                         rebuilt_overlapping=float(np.mean(close_rebuilt < MIN_DISTANCE))))
        print(name, {k: round(v, 3) for k, v in rows[-1].items() if k != "name"}, flush=True)
    out = os.path.join(HERE, "results", "atoms.json")
    json.dump(dict(min_distance=MIN_DISTANCE, peptides=rows), open(out, "w"), indent=1)
    print("wrote", os.path.relpath(out, HERE))


if __name__ == "__main__":
    main()
