"""Prepare site-state trajectories and emit torsions from the training bank.

These local states belong to the kinetic model. Evaluation uses a separate
set of metastable states fitted to the molecular-dynamics reference.
"""

import numpy as np

import ctmc as CT


def load(path):
    """Read a torsion NPZ into residue names, float32 torsions and the frame interval in ps."""
    z = np.load(path)
    return dict(res=[str(r) for r in z["res"]], tors=z["tors"].astype(np.float32), dt_ps=float(z["dt_ps"]))


def prepare(tables, sd, vocab, bank=None):
    """Build peptide indices and compressed state paths for supported torsion tables.

    Tables with missing state definitions or vocabulary entries are skipped.
    If bank is supplied, add core-frame torsions to it."""
    out = []
    for name, tb in tables:
        try:
            L, sl, keys, Cf = sd.site_traj(tb, return_core=True)
        except KeyError:
            continue
        if any((k, 0) not in vocab.state_id for k in keys):
            continue
        nt, ct = CT.termini(tb["tors"][0])
        pep = CT.PeptideIndex(vocab, tb["res"], sl, keys, nterm=nt, cterm=ct, temperature=tb.get("temperature", CT.T_REF))
        if bank is not None:
            bank.add(tb, L, sl, keys, Cf)
        out.append(dict(name=name, tb=tb, L=L, sl=sl, keys=keys, pep=pep, seg=CT.segments(L)))
    return out


def emit(bank, tb_res, bbks, sl, frames):
    """Draw torsions independently for each residue and frame, conditional on local states.

    Return (frames, residues, 7) radians. For an unseen combination, prefer
    stored torsions with the same backbone state, then any matching residue type
    and terminal geometry."""
    n = len(frames)
    R = len(tb_res)
    T = np.full((n, R, 7), np.nan, np.float32)
    by_res = {}
    for c, (i, kind) in enumerate(sl):
        by_res.setdefault(i, []).append(c)
    for i, cols in by_res.items():
        st = frames[:, cols]
        uniq, inv = np.unique(st, axis=0, return_inverse=True)
        for u, row in enumerate(uniq):
            sel = np.where(inv.ravel() == u)[0]
            arr = bank.sample(tb_res[i], bbks[i], row, len(sel))
            if arr is None:
                cands = [k for k in bank.bank if k[0] == tb_res[i] and k[1] == bbks[i] and k[2][0] == row[0]]
                if not cands:
                    cands = [k for k in bank.bank if k[0] == tb_res[i] and k[1] == bbks[i]]
                k = cands[bank.rng.integers(len(cands))]
                arr = bank.bank[k][bank.rng.integers(0, len(bank.bank[k]), size=len(sel))]
            T[sel, i] = arr
    return T


def site_metrics(L_md, dt_md, L_mod, dt_mod, keys, min_pop=0.01):
    """Compare local-state free energies and total switching rates per site.

    Both inputs must be single paths with no artificial joins between independent
    trajectories. The dt arguments use the same time unit; rate ratios are unitless."""
    dF, dlogk = [], []
    for c, key in enumerate(keys):
        n = int(max(L_md[:, c].max(), L_mod[:, c].max())) + 1
        pm = np.bincount(L_md[:, c], minlength=n) / len(L_md)
        pq = np.bincount(L_mod[:, c], minlength=n) / len(L_mod)
        for s_ in range(n):
            if pm[s_] >= min_pop:
                dF.append(-np.log(max(pq[s_], 1e-6) / pm[s_]))
        km = (L_md[1:, c] != L_md[:-1, c]).sum() / (len(L_md) * dt_md)
        kq = (L_mod[1:, c] != L_mod[:-1, c]).sum() / (len(L_mod) * dt_mod)
        if km > 0 and kq > 0:
            dlogk.append(np.log(kq / km))
    dF = np.array(dF)
    return dict(site_f_rms=float(np.sqrt(np.mean((dF - dF.mean()) ** 2))) if len(dF) else None,
                site_f_max=float(np.abs(dF).max()) if len(dF) else None,
                site_logk_median_abs=float(np.median(np.abs(dlogk))) if dlogk else None,
                site_logk_max_abs=float(np.max(np.abs(dlogk))) if dlogk else None)
