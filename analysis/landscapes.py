"""Export matched reference and model projections for landscape visualisation.

Usage: python analysis/landscapes.py [BUNDLE_DIR]

The default bundle is model/bundle. Write results/landscapes/SEQUENCE.npz
with the leading two reference TICA coordinates, state labels and backbone
torsions. Model sampling uses seed 100 and 200 paths totalling 5 us. Every
25th output frame is kept, giving 0.25 ns spacing within each path; model
paths are concatenated in the saved arrays.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np
import data as D
import ctmc as CT
import sitepaths as SP
import states as ST
import bundle as BU

N_TRAJ = 200
TOTAL_NS = 5000.0
EVERY = 25
SEED = 100


def tica(sm, X):
    """Project features onto the leading two stored, scaled TICA components."""
    return ((X - sm.mean) @ sm.tica_vecs)[:, :2]


def main():
    out_dir = os.path.join(P.RESULTS, "landscapes")
    os.makedirs(out_dir, exist_ok=True)
    b = BU.load(sys.argv[1] if len(sys.argv) > 1 else BU.DEFAULT)
    b["bank"].rng = np.random.default_rng(SEED)
    for name in P.judged_peptides():
        sm = ST.load(os.path.join(P.STATES, name + ".npz"))
        tb = P.test_table(name)
        p = SP.prepare([(name, tb)], b["sitedefs"], b["vocab"])
        if not p:
            continue
        p = p[0]
        table = CT.rate_table(b["net"], CT.to_device(p["pep"], "cpu"), "cpu", max_configs=150000, chunk=4096)
        if table is None:
            continue
        T_md = tb["tors"][P.EQUIL_FRAMES:]
        X_md = D.features_from_tors(tb["res"], T_md)
        x0 = CT.table_starts(table, N_TRAJ, seed=SEED)
        fr = CT.simulate_table(table, x0, TOTAL_NS / N_TRAJ * 1000.0, 10.0, b["dt_ps"], seed=SEED).astype(np.int64)
        bbks = [D.bb_kind(tb["tors"][0], i) for i in range(len(tb["res"]))]
        T_mod = np.concatenate([SP.emit(b["bank"], tb["res"], bbks, p["sl"], fr[k]) for k in range(N_TRAJ)])
        X_mod = D.features_from_tors(tb["res"], T_mod)
        np.savez_compressed(os.path.join(out_dir, name + ".npz"),
                            md_tica=tica(sm, X_md)[::EVERY].astype(np.float32),
                            model_tica=tica(sm, X_mod)[::EVERY].astype(np.float32),
                            md_states=sm.assign(X_md)[::EVERY].astype(np.int8),
                            model_states=np.concatenate([sm.assign(X_mod[k * len(fr[0]):(k + 1) * len(fr[0])])
                                                         for k in range(N_TRAJ)])[::EVERY].astype(np.int8),
                            md_phi_psi=T_md[::EVERY, :, :2].astype(np.float16),
                            model_phi_psi=T_mod[::EVERY, :, :2].astype(np.float16),
                            residues=np.array(tb["res"]), frame_ns=P.FRAME_NS * EVERY)
        print(name, len(X_md), len(X_mod), flush=True)


if __name__ == "__main__":
    main()
