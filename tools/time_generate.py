#!/usr/bin/env python3
"""Time the generation of one microsecond for every held-out peptide on one core of this machine's CPU.

    python tools/time_generate.py "Apple M4"     -> results/timing.json

One trajectory of 1 microsecond (100,000 frames) per peptide, from the sequence to the heavy atoms in memory, in
one process whose linear algebra is held to one thread.
"""
import glob
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"                                  # before numpy is imported: one thread, one core

import numpy as np                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pepdyn                                             # noqa: E402


def main():
    machine = sys.argv[1]
    mod = pepdyn.load_model()
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "reference", "*.npz"))):
        name = os.path.basename(f)[:-4]
        out = pepdyn.generate(name, ns=1000.0, n_traj=1, seed=0, model=mod)
        s = out["seconds"]
        rows.append(dict(name=name, joint_configurations=out["peptide"].n_configs, heavy_atoms=out["topology"].n_atoms,
                         tabulated=out["peptide"].n_configs <= pepdyn.TABLE_LIMIT, **{k: round(v, 3) for k, v in s.items()}))
        print(name, rows[-1], flush=True)
    total = np.array([r["total"] for r in rows])
    json.dump(dict(machine=machine, numpy=np.__version__, threads=1,
                   what="one trajectory of 1 microsecond, 100,000 frames of heavy atoms",
                   median_seconds=float(np.median(total)), fastest_seconds=float(total.min()), slowest_seconds=float(total.max()),
                   peptides=rows), open(os.path.join(HERE, "results", "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
