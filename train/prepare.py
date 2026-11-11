#!/usr/bin/env python3
"""Prepare a training set: site definitions, vocabulary, torsion bank, empirical baseline and, for each training
peptide, its static index and its run-length-encoded site trajectory.

    python train/prepare.py OUT.pkl DT_PS TEMPERATURE TABLE.npz ...

TABLE.npz are torsion tables (train/tables.py makes them from the corpus): res, tors (frames, residues, 7), dt_ps.
DT_PS must be a whole multiple of their frame interval; frames are thinned to it. The published model:
DT_PS 10, TEMPERATURE 310, the 1,500 training tetrapeptides of ManyPeptidesMD in alphabetical order.
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network as N        # noqa: E402
import sites as S          # noqa: E402


def load(path):
    z = np.load(path)
    return dict(res=[str(r) for r in z["res"]], tors=z["tors"].astype(np.float32), dt_ps=float(z["dt_ps"]))


def index(tables, sd, vocab, bank=None):
    """Site trajectories of the peptides whose sites the alphabet holds; fills the bank if one is given."""
    out = []
    for name, tb in tables:
        try:
            L, sl, keys, Cf = sd.site_traj(tb)
        except KeyError:
            continue
        if any((k, 0) not in vocab.state_id for k in keys):
            continue
        pep = N.PeptideIndex(vocab, tb["res"], sl, keys, temperature=tb.get("temperature", N.T_REF))
        if bank is not None:
            bank.add(tb, L, sl, keys, Cf)
        out.append(dict(name=name, tb=tb, L=L, sl=sl, keys=keys, pep=pep, seg=N.segments(L)))
    return out


def main():
    out, dt, temp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tabs = []
    for f in sys.argv[4:]:
        tb = load(f)
        m = int(round(dt / tb["dt_ps"]))
        assert m >= 1 and abs(m * tb["dt_ps"] - dt) < 1e-9, f
        if m > 1:
            tb["tors"], tb["dt_ps"] = tb["tors"][::m], dt
        tb["temperature"] = temp
        tabs.append((os.path.basename(f)[:-4], tb))
    sd = S.SiteDefs().fit([t for _, t in tabs])
    vocab = N.Vocab(sd)
    bank = S.Bank()
    peps = index(tabs, sd, vocab, bank)
    F0, G0 = N.empirical_baseline(vocab, peps, dt)
    slim = []
    for p in peps:
        pi = p["pep"]
        cfg, rl, nx, has = p["seg"]
        seg = (cfg.astype(np.int8), rl.astype(np.int32), nx.astype(np.int8), has)
        slim.append(dict(name=p["name"], n_frames=len(p["L"]), seg=seg, keys=p["keys"], sl=p["sl"],
                         idx=dict(R=pi.R, types=pi.types, site_res=pi.site_res, site_kind=pi.site_kind,
                                  site_n=pi.site_n, site_ids=pi.site_ids, S=pi.S, temperature=pi.temperature)))
    pickle.dump(dict(sitedefs=sd, vocab=vocab, bank=bank, F0=F0, G0=G0, dt_ps=dt, peptides=slim), open(out, "wb"), protocol=4)
    print(out, len(slim), "peptides", sum(p["n_frames"] for p in slim) * dt / 1000, "ns")


if __name__ == "__main__":
    main()
