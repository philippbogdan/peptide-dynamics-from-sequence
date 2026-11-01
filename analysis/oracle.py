"""Test rate errors after replacing model energies with MD configuration frequencies.

Usage: python analysis/oracle.py [OUT.json [BUNDLE_DIR]]

Defaults are results/oracle.json and model/bundle. The corrected rate table
retains every learned symmetric prefactor. Each comparison samples 5 us of
model dynamics and uses the same reference states. This is a diagnostic that
uses MD information, not sequence-only prediction.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np
import data as D
import ctmc as CT
import sitepaths as SB
import check as CK
import states as ST
import bundle as BU

KEYS = ("mfpt_worst_factor", "f_rms_kT", "tv", "slowest_rel_err", "others_worst_rel_err")


def corrected(table, L):
    """Replace observed joint-state energies with -ln(MD frequency), preserving prefactors.

    For configurations absent from MD, retain a shifted model energy subject to
    a floor corresponding to half an observation. Rescale each forward and
    reverse rate by opposite half-energy differences."""
    sizes, k, E = table
    n, S, V = k.shape
    codes = np.ravel_multi_index(L.T.astype(np.int64), sizes)
    cnt = np.bincount(codes, minlength=n).astype(float)
    N = cnt.sum()
    vis = cnt > 0
    Ft = np.where(vis, -np.log(np.maximum(cnt, 1) / N), 0.0)
    c = np.sum(cnt[vis] * (Ft[vis] - E[vis])) / N
    # Unvisited configurations keep a conservative finite energy instead of being forbidden.
    Enew = np.where(vis, Ft, np.maximum(E + c, -np.log(0.5 / N)))
    dlt = Enew - E
    strides = np.cumprod(np.concatenate([[1], sizes[::-1][:-1]]))[::-1]
    cfg = np.stack(np.unravel_index(np.arange(n), sizes), 1)
    k2 = k.copy()
    for s_ in range(S):
        for v in range(int(sizes[s_])):
            tgt = np.arange(n) + (v - cfg[:, s_]) * strides[s_]
            k2[:, s_, v] = k[:, s_, v] * np.exp(-(dlt[tgt] - dlt) / 2)
    return sizes, k2.astype(np.float32), Enew


def judge(table, p, tb, sm, ref, main_, bank, dt, total_ns=5000.0):
    """Sample 200 paths and compare their reference-state estimates at a fixed total length in ns."""
    x0 = CT.table_starts(table, 200)
    fr = CT.simulate_table(table, x0, total_ns / 200 * 1000.0, 10.0, dt).astype(np.int64)
    bbks = [D.bb_kind(tb["tors"][0], i) for i in range(len(tb["res"]))]
    dm = [sm.assign(D.features_from_tors(tb["res"], SB.emit(bank, tb["res"], bbks, p["sl"], fr[b]))) for b in range(200)]
    mod = CK.estimates(sm, dm)
    c = CK.compare(sm, mod, ref, main=main_)
    return {k: c[k] for k in KEYS} | dict(pass_all=c["pass_all"], ts=mod["ts"][:3].tolist())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P.RESULTS, "oracle.json")
    b = BU.load(sys.argv[2] if len(sys.argv) > 2 else BU.DEFAULT)
    sd, vocab, bank, dt, net = b["sitedefs"], b["vocab"], b["bank"], b["dt_ps"], b["net"]
    rows = []
    for name in P.judged_peptides():
        info = json.load(open(os.path.join(P.REFERENCES, name + ".json")))
        sm, tb, dA, dB = P.reference_halves(name)
        ref = CK.estimates(sm, [dA, dB])
        main_ = np.array(info["main"], int)
        p = SB.prepare([(name, tb)], sd, vocab)
        if not p:
            continue
        p = p[0]
        table = CT.rate_table(net, CT.to_device(p["pep"], "cpu"), "cpu", max_configs=150000, chunk=4096)
        if table is None:
            continue
        a = judge(table, p, tb, sm, ref, main_, bank, dt)
        o = judge(corrected(table, p["L"]), p, tb, sm, ref, main_, bank, dt)
        rows.append(dict(name=name, model=a, thermo_oracle=o, ts_ref=ref["ts"][:3].tolist(),
                         replicas={k: info["equal_footing"][k] for k in KEYS}))
        print(name, json.dumps({k: round(o[k], 3) for k in KEYS}), flush=True)
        json.dump(rows, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
