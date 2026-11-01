"""Compare sequence-conditioned samples with the reference metastable-state kinetics.

Usage: python analysis/evaluate.py [SEED [BUNDLE_DIR]]

The defaults are seed 0 and model/bundle. For each supported peptide with a
saved state model and at most 150,000 joint configurations, generate 200
paths of 25 ns, totalling 5 us. Initial configurations follow the continuous-
time generator's Boltzmann weights; subsequent updates use its finite step.
Save state paths in results/dtrajs/ and metrics in evaluation_seedSEED.json.

The reference supplies the evaluation states and measured comparison values.
Its coordinate frames do not initialise or condition generated paths.
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

N_TRAJ = 200
TOTAL_NS = 5000.0


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out = os.path.join(P.RESULTS, f"evaluation_seed{seed}.json")
    b = BU.load(sys.argv[2] if len(sys.argv) > 2 else BU.DEFAULT)
    if seed:
        b["bank"].rng = np.random.default_rng(seed)
    sd, vocab, bank, dt, net = b["sitedefs"], b["vocab"], b["bank"], b["dt_ps"], b["net"]
    names = sorted(f[:-4] for f in os.listdir(os.path.join(P.TORSIONS, "test")) if f.endswith(".npz"))
    os.makedirs(P.DTRAJS, exist_ok=True)
    results = []
    for name in names:
        sfile = os.path.join(P.STATES, name + ".npz")
        if not os.path.exists(sfile):
            continue
        tb = P.test_table(name)
        p = SP.prepare([(name, tb)], sd, vocab)
        if not p:
            continue
        p = p[0]
        info = json.load(open(os.path.join(P.REFERENCES, name + ".json")))
        sm, _, dA, dB = P.reference_halves(name, ST.load(sfile))
        ref = CK.estimates(sm, [dA, dB])
        main_ = np.array(info["main"], int)
        table = CT.rate_table(net, CT.to_device(p["pep"], "cpu"), "cpu", max_configs=150000, chunk=4096)
        if table is None:
            continue
        x0 = CT.table_starts(table, N_TRAJ, seed=seed)
        fr = CT.simulate_table(table, x0, TOTAL_NS / N_TRAJ * 1000.0, 10.0, dt, seed=seed).astype(np.int64)
        bbks = [D.bb_kind(tb["tors"][0], i) for i in range(len(tb["res"]))]
        dm = [sm.assign(D.features_from_tors(tb["res"], SP.emit(bank, tb["res"], bbks, p["sl"], fr[k])))
              for k in range(N_TRAJ)]
        mod = CK.estimates(sm, dm)
        r = dict(name=name, model=CK.compare(sm, mod, ref, main=main_),
                 equal_footing=CK.compare(sm, CK.estimates(sm, [dA]), CK.estimates(sm, [dB]), main=main_),
                 ts_ref=ref["ts"][:5].tolist(), ts_model=mod["ts"][:5].tolist(),
                 pi_ref=ref["pi"].tolist(), pi_model=mod["pi"].tolist())
        r.update(SP.site_metrics(p["L"].astype(np.int64), 0.01, fr.reshape(-1, fr.shape[-1]), 0.01, p["keys"]))
        results.append(r)
        np.savez_compressed(os.path.join(P.DTRAJS, name + f".model_seed{seed}.npz"), model=np.array(dm, dtype=np.int8))
        print(json.dumps(dict(name=name, mfpt=round(r["model"]["mfpt_worst_factor"], 3),
                              f_rms=round(r["model"]["f_rms_kT"], 3), tv=round(r["model"]["tv"], 3),
                              slowest=round(r["model"]["slowest_rel_err"], 3))), flush=True)
        json.dump(results, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
