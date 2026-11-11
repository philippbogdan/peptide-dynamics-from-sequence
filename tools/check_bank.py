#!/usr/bin/env python3
"""How often does a generated frame ask the torsion bank for a combination of site states it does not hold?

    python tools/check_bank.py     -> results/bank.json

Such a residue borrows the torsions of another combination with the same backbone state (pepdyn/emit.py), so
its atoms then show other rotamers than its states say. Counted on 1 microsecond of each held-out peptide
(100 ns of the three whose rates are not tabulated).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import pepdyn                                             # noqa: E402
from pepdyn import model as M                             # noqa: E402


def main():
    mod = pepdyn.load_model()
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "reference", "*.npz"))):
        name = os.path.basename(f)[:-4]
        pep = mod.peptide(name)
        rng = np.random.default_rng(0)
        table = M.rate_table(mod, pep, pepdyn.TABLE_LIMIT)
        ns = 1000.0 if table is not None else 100.0
        x = M.simulate(mod, pep, M.equilibrium_starts(mod, pep, 10, rng, table), ns / 10, rng, table=table)
        x = x.reshape(-1, pep.S).astype(np.int64)
        by_residue = {}
        for c, (i, _) in enumerate(pep.sites):
            by_residue.setdefault(i, []).append(c)
        borrows = np.zeros(len(x), bool)
        for i, cols in by_residue.items():
            uniq, inv = np.unique(x[:, cols], axis=0, return_inverse=True)
            absent = np.array([(pep.residues[i], pep.bb_kinds[i], tuple(int(v) for v in row)) not in mod.bank_index for row in uniq])
            borrows |= absent[inv.ravel()]
        rows.append(dict(name=name, frames=len(x), frames_that_borrow=int(borrows.sum())))
        print(name, rows[-1], flush=True)
    frames, borrow = sum(r["frames"] for r in rows), sum(r["frames_that_borrow"] for r in rows)
    json.dump(dict(combinations_in_the_bank=len(mod.bank_keys), share_of_frames_that_borrow=borrow / frames,
                   largest_share_for_one_peptide=max(r["frames_that_borrow"] / r["frames"] for r in rows), peptides=rows),
              open(os.path.join(HERE, "results", "bank.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
