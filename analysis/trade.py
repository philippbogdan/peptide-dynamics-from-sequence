"""Measure error as a function of available MD or model trajectory length.

Usage: python analysis/trade.py [OUT.json [BUNDLE_DIR]]

Defaults are results/trade.json and model/bundle. At each total length from
10 to 2490 ns, compare the beginning of reference half A and a new model
sample separately with half B. Save model state paths in results/dtrajs/.
All comparisons use the same reference-defined main states and tolerances.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np
import data as D
import ctmc as CT
import sitepaths as SP
import check as CK
import states as ST
import bundle as BU

LENGTHS_NS = (10, 30, 100, 300, 1000, 2490)
KEYS = ("mfpt_worst_factor", "f_rms_kT", "tv", "slowest_rel_err")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P.RESULTS, "trade.json")
    bdir = sys.argv[2] if len(sys.argv) > 2 else BU.DEFAULT
    b = BU.load(bdir)
    net, dt = b["net"], b["dt_ps"]
    res = []
    for name in P.judged_peptides():
        main_ = np.array(json.load(open(os.path.join(P.REFERENCES, name + ".json")))["main"], int)
        sm, tb, dA, dB = P.reference_halves(name)
        refB = CK.estimates(sm, [dB])
        p = SP.prepare([(name, tb)], b["sitedefs"], b["vocab"])
        if not p:
            continue
        p = p[0]
        table = CT.rate_table(net, CT.to_device(p["pep"], "cpu"), "cpu", max_configs=150000, chunk=4096)
        if table is None:
            continue
        bbks = [D.bb_kind(tb["tors"][0], i) for i in range(len(tb["res"]))]
        row = dict(name=name, md={}, model={})
        keep = {}
        for L in LENGTHS_NS:
            k = int(round(L / P.FRAME_NS))
            if k <= len(dA):
                row["md"][L] = {kk: CK.compare(sm, CK.estimates(sm, [dA[:k]]), refB, main=main_)[kk] for kk in KEYS}
            n_traj = max(1, min(200, L // 10))
            x0 = CT.table_starts(table, n_traj, seed=L)
            fr = CT.simulate_table(table, x0, L / n_traj * 1000.0, 10.0, dt, seed=L).astype(np.int64)
            dm = [sm.assign(D.features_from_tors(tb["res"], SP.emit(b["bank"], tb["res"], bbks, p["sl"], fr[j])))
                  for j in range(n_traj)]
            keep[f"model_{L}ns"] = np.array(dm, dtype=np.int8)
            row["model"][L] = {kk: CK.compare(sm, CK.estimates(sm, dm), refB, main=main_)[kk] for kk in KEYS}
        np.savez_compressed(os.path.join(P.DTRAJS, name + ".trade.npz"), **keep)
        res.append(row)
        print(json.dumps(dict(name=name, md={L: round(v["f_rms_kT"], 3) for L, v in row["md"].items()},
                              model={L: round(v["f_rms_kT"], 3) for L, v in row["model"].items()})), flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
