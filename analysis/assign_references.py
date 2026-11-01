"""Save the reference halves as state-label trajectories for lightweight reanalysis.

Usage: python analysis/assign_references.py

For each saved state model, write results/dtrajs/SEQUENCE.reference.npz with
half_a and half_b arrays. Raw torsion tables must be available.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np


def main():
    os.makedirs(P.DTRAJS, exist_ok=True)
    for name in P.judged_peptides():
        _, _, dA, dB = P.reference_halves(name)
        np.savez_compressed(os.path.join(P.DTRAJS, name + ".reference.npz"), half_a=dA.astype(np.int8),
                            half_b=dB.astype(np.int8))
        print(name, len(dA), len(dB), flush=True)


if __name__ == "__main__":
    main()
