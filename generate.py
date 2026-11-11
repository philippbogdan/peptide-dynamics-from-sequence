#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy>=2"]
# ///
"""Generate the dynamics of a peptide from its sequence.

    python generate.py GCDE                      one trajectory of 1 microsecond, written to GCDE.pdb and GCDE.dcd
    python generate.py GCDE --ns 10 --traj 100   a hundred trajectories of 10 ns each (GCDE_000.dcd, ...)

Every heavy atom is written every 10 ps. The .pdb file holds the first frame and the bonds; VMD, PyMOL, MDTraj
and MDAnalysis read the .dcd file with it. Only numpy is needed.
"""
import argparse
import os

import numpy as np

import pepdyn
from pepdyn import build


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", help="one-letter amino-acid sequence, for example GCDE")
    ap.add_argument("--ns", type=float, default=1000.0, help="length of each trajectory in nanoseconds (default 1000)")
    ap.add_argument("--traj", type=int, default=1, help="number of trajectories, each started from equilibrium (default 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".", help="output folder (default: here)")
    ap.add_argument("--torsions", action="store_true", help="also save the site states and torsion angles (.npz)")
    a = ap.parse_args()

    out = pepdyn.generate(a.sequence, ns=a.ns, n_traj=a.traj, seed=a.seed)
    pep, top, X = out["peptide"], out["topology"], out["xyz"]
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, pep.sequence)
    build.write_pdb(stem + ".pdb", top, X[0, 0])
    for k in range(a.traj):
        build.write_dcd(stem + (".dcd" if a.traj == 1 else f"_{k:03d}.dcd"), X[k])
    if a.torsions:
        np.savez_compressed(stem + ".npz", states=out["states"], torsions=out["torsions"].astype(np.float32),
                            sites=np.array([f"{pep.residues[i]}{i + 1}:{kind}" for i, kind in pep.sites]))
    s, o = out["seconds"], out["overlaps"]
    print(f"{pep.sequence}: {a.traj} x {a.ns:g} ns, {X.shape[1] * a.traj} frames of {top.n_atoms} heavy atoms, "
          f"in {s['total']:.1f} s (rates {s['rates_and_starts']:.1f}, dynamics {s['dynamics']:.1f}, atoms {s['atoms']:.1f})")
    print(f"frames drawn again for an overlap: {100 * o['drawn_again']:.1f}%, left with one: {100 * o['left_overlapping']:.2f}%")
    print(f"wrote {stem}.pdb and {a.traj} .dcd file{'s' if a.traj > 1 else ''}")


if __name__ == "__main__":
    main()
