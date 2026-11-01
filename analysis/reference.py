"""Discover metastable states and compare the two halves of each MD reference.

Usage: python analysis/reference.py [SEQUENCE ...]

With no sequence arguments, process every available test torsion table.
NPROC sets the number of concurrent processes (default 4). Results go to
results/references/; accepted state models go to results/states/. Shuffled
frames and a doubled clock provide controls. The two reference halves are
segments of one simulation, not independent runs.
"""

import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np


def one(name):
    """Apply the state rule to one peptide, then save reference estimates and controls."""
    import data as D
    import states as ST
    import check as CK
    out = os.path.join(P.REFERENCES, name + ".json")
    tb = P.test_table(name)
    step = int(round(P.FRAME_NS * 1000.0 / tb["dt_ps"]))
    X = D.features_from_tors(tb["res"], tb["tors"][::step])[P.EQUIL_FRAMES:]
    h = len(X) // 2
    XA, XB = X[:h], X[h:2 * h]
    sm, rep = ST.discover([XA, XB], P.FRAME_NS)
    r = dict(peptide=name, ns_per_replica=h * P.FRAME_NS, state_rule=dict(best=rep["best_timescales_ns"][:8],
             n_macro=rep["n_macro"], chosen=rep.get("chosen")), states_ok=sm is not None)
    if sm is not None:
        dA, dB = sm.assign(XA), sm.assign(XB)
        ref = CK.estimates(sm, [dA, dB])
        main = np.where(ref["pi"] >= CK.MAIN_MIN_POP)[0]
        r.update(pi=ref["pi"].tolist(), ts=ref["ts"][:6].tolist(), main=main.tolist(),
                 mfpt=np.where(np.isfinite(ref["mfpt"]), ref["mfpt"], -1).tolist(),
                 equal_footing=CK.compare(sm, CK.estimates(sm, [dA]), CK.estimates(sm, [dB]), main=main),
                 control_shuffled=CK.compare(sm, CK.estimates(sm, CK.shuffled([dA, dB])), ref, main=main),
                 control_doubled_clock=CK.compare(sm, dict(pi=ref["pi"], mfpt=2 * ref["mfpt"], ts=2 * ref["ts"]),
                                                  ref, main=main))
        ST.save(sm, os.path.join(P.STATES, name + ".npz"))
    json.dump(r, open(out, "w"), indent=1, default=float)
    return out


def main():
    os.makedirs(P.STATES, exist_ok=True)
    os.makedirs(P.REFERENCES, exist_ok=True)
    names = sys.argv[1:] or sorted(f[:-4] for f in os.listdir(os.path.join(P.TORSIONS, "test")) if f.endswith(".npz"))
    with Pool(int(os.environ.get("NPROC", 4))) as p:
        for r in p.imap_unordered(one, names):
            print(r, flush=True)


if __name__ == "__main__":
    main()
