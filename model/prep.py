"""Build the training bundle and compressed site-state paths from torsion tables.

Usage: python model/prep.py OUT.pkl DT_PS TEMPERATURE TABLE.npz ...

DT_PS is a whole multiple of each input frame interval. The tables are
subsampled to that interval before fitting state definitions, the vocabulary,
the emission bank and baseline energies and rates. An optional @TEMPERATURE
argument changes the temperature assigned to subsequent tables. Temperatures
are in kelvin. OUT.pkl contains all quantities needed by model/train.py.
"""
import os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import data as D
import ctmc as CT
import sitepaths as SB


def main():
    out, dt, temp = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    tabs = []
    for f in sys.argv[4:]:
        if f.startswith("@"):
            temp = float(f[1:])
            continue
        tb = SB.load(f)
        m = int(round(dt / tb["dt_ps"]))
        assert m >= 1 and abs(m * tb["dt_ps"] - dt) < 1e-9, f
        if m > 1:
            tb["tors"], tb["dt_ps"] = tb["tors"][::m], dt
        tb["temperature"] = temp
        tabs.append((os.path.basename(f)[:-4], tb))
    sd = D.SiteDefs().fit([t for _, t in tabs])
    vocab = CT.Vocab(sd)
    bank = D.Bank()
    peps = SB.prepare(tabs, sd, vocab, bank)
    F0, G0 = CT.empirical_baseline(vocab, peps, dt)
    slim = []
    for p in peps:
        pi = p["pep"]
        cfg, rl, nx, has = p["seg"]
        seg = (cfg.astype(np.int8), rl.astype(np.int32), nx.astype(np.int8), has)
        slim.append(dict(name=p["name"], n_frames=len(p["L"]), seg=seg, keys=p["keys"], sl=p["sl"],
                         idx=dict(R=pi.R, types=pi.types, site_res=pi.site_res, site_kind=pi.site_kind,
                                  site_n=pi.site_n, site_ids=pi.site_ids, S=pi.S, temperature=pi.temperature)))
    pickle.dump(dict(sitedefs=sd, vocab=vocab, bank=bank, F0=F0, G0=G0, dt_ps=dt, peptides=slim),
                open(out, "wb"), protocol=4)
    print(out, len(slim), "peptides", sum(p["n_frames"] for p in slim) * dt / 1000, "ns")


if __name__ == "__main__":
    main()
