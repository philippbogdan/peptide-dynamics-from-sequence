"""Compare sequence-context effects on symmetric local transition rates.

Usage: python analysis/barriers.py [OUT.json [BUNDLE_DIR]]

Defaults are results/barriers.json and model/bundle. All 30 test torsion
tables are needed for the shared transition-type baseline. Rows record
G = 0.5 * ln(k_ab * k_ba) with rates in inverse ns. The headline summary
uses rows with at least 20 MD transitions and subtracts that shared baseline.

SIM_PS_PER_PATH sets each path to 25,000 ps: 200 paths total 5 us.
Transition counts and residence times are computed separately for each path,
then summed, so independent paths do not introduce boundary transitions.
The JSON rows do not embed run metadata; preserve the sampling protocol
alongside the result when changing path duration or transition counting.
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths as P
import numpy as np
import ctmc as CT
import sitepaths as SP
import bundle as BU

SIM_PS_PER_PATH = 25000.0


def site_rates(L, c, n, frame_ns):
    """Count transitions and residence times for one site of a single path.

    Residence times use frame_ns. Passing concatenated independent paths also
    counts their joins as transitions."""
    x = L[:, c].astype(int)
    occ = np.bincount(x, minlength=n).astype(float) * frame_ns
    ch = x[1:] != x[:-1]
    cnt = np.zeros((n, n))
    np.add.at(cnt, (x[:-1][ch], x[1:][ch]), 1.0)
    return cnt, occ


def pooled(names, sd, vocab):
    """Aggregate MD counts and residence times by residue/site type across the named sequences."""
    cnt, occ = {}, {}
    for name in names:
        p = SP.prepare([(name, P.test_table(name))], sd, vocab)
        if not p:
            continue
        L = p[0]["L"].astype(int)
        for c, key in enumerate(p[0]["keys"]):
            x = L[:, c]
            n = int(max(x.max() + 1, 1))
            o = occ.setdefault(key, np.zeros(16))
            o[:n] += np.bincount(x, minlength=n) * P.FRAME_NS
            C = cnt.setdefault(key, np.zeros((16, 16)))
            ch = x[1:] != x[:-1]
            np.add.at(C, (x[:-1][ch], x[1:][ch]), 1.0)
    return cnt, occ


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(P.RESULTS, "barriers.json")
    b = BU.load(sys.argv[2] if len(sys.argv) > 2 else BU.DEFAULT)
    sd, vocab, dt, net = b["sitedefs"], b["vocab"], b["dt_ps"], b["net"]
    gb = net.g_bound
    PB = net.pair_bias.detach().numpy()
    all_test = sorted(f[:-4] for f in os.listdir(os.path.join(P.TORSIONS, "test")) if f.endswith(".npz"))
    pc, po = pooled(all_test, sd, vocab)
    rows = []
    for name in P.judged_peptides():
        tb = P.test_table(name)
        p = SP.prepare([(name, tb)], sd, vocab)
        if not p:
            continue
        p = p[0]
        pep = CT.to_device(p["pep"], "cpu")
        table = CT.rate_table(net, pep, "cpu", max_configs=150000, chunk=4096)
        if table is None:
            continue
        sizes, k, E = table
        n = len(E)
        pi = np.exp(-(E - E.min()))
        pi /= pi.sum()
        x0 = CT.table_starts(table, 200)
        fr = CT.simulate_table(table, x0, SIM_PS_PER_PATH, 10.0, dt)
        L = p["L"]
        strides = np.cumprod(np.concatenate([[1], sizes[::-1][:-1]]))[::-1]
        cfg = np.stack(np.unravel_index(np.arange(n), sizes), 1)
        sid = np.asarray(pep.site_ids)
        with np.errstate(divide="ignore"):
            lk = np.log(k)
        for c, key in enumerate(p["keys"]):
            m = int(sizes[c])
            cm, om = site_rates(L, c, m, P.FRAME_NS)
            per_path = [site_rates(f_, c, m, P.FRAME_NS) for f_ in fr]
            cq, oq = sum(r_[0] for r_ in per_path), sum(r_[1] for r_ in per_path)
            C, O = pc[key], po[key]
            for a in range(m):
                for bb in range(a + 1, m):
                    if cm[a, bb] + cm[bb, a] < 4:
                        continue
                    gmd = 0.5 * np.log(cm[a, bb] / om[a] * cm[bb, a] / om[bb]) if cm[a, bb] and cm[bb, a] else None
                    gmo = 0.5 * np.log(cq[a, bb] / oq[a] * cq[bb, a] / oq[bb]) if cq[a, bb] and cq[bb, a] else None
                    gp = (0.5 * np.log(C[a, bb] / O[a] * C[bb, a] / O[bb])
                          if min(C[a, bb], C[bb, a]) >= 10 else None)
                    u = np.nonzero(cfg[:, c] == a)[0]
                    w = u + (bb - a) * strides[c]
                    gc = 0.5 * (lk[u, c, bb] + lk[w, c, a]) - PB[sid[c, a], sid[c, bb]]
                    wt = pi[u] + pi[w]
                    wt = wt / wt.sum()
                    rows.append(dict(name=name, site=list(key), a=a, b=bb, md_transitions=int(cm[a, bb] + cm[bb, a]),
                                     G_md=gmd, G_model=gmo, G_pooled_test=gp,
                                     g_corr_mean=float((wt * gc).sum()),
                                     frac_saturated=float(wt[np.abs(gc) > gb - 0.2].sum())))
        print(name, len(rows), flush=True)
        json.dump(rows, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
