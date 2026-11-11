"""From site states to torsion angles.

The torsion bank holds, for every residue type in every combination of its site states, torsions of that residue
seen in the training MD (frames in which every site sat in the core of its basin). A frame of the model draws
one bank row per residue, so a residue's own torsions are mutually consistent, while two frames of one basin and
two residues of one frame are drawn independently.
"""
import numpy as np

COLUMNS = ("phi", "psi", "chi1", "chi2", "chi3", "chi4", "pair")


def torsions(model, pep, frames, rng):
    """Site states (n, S) -> torsions (n, residues, 7) in radians, NaN where a residue has no such torsion.
    Columns: phi, psi, chi1 to chi4, and the second methyl of Val (on chi1) or Leu (on chi2)."""
    frames = np.asarray(frames, np.int64)
    n = len(frames)
    T = np.full((n, len(pep.residues), 7), np.nan)
    by_res = {}
    for c, (i, _) in enumerate(pep.sites):
        by_res.setdefault(i, []).append(c)
    for i, cols in by_res.items():
        rn, bbk = pep.residues[i], pep.bb_kinds[i]
        uniq, inv = np.unique(frames[:, cols], axis=0, return_inverse=True)
        inv = inv.ravel()
        for u, row in enumerate(uniq):
            sel = np.where(inv == u)[0]
            k = model.bank_index.get((rn, bbk, tuple(int(v) for v in row)))
            if k is None:      # a combination never seen in training: same backbone state, any rotamers
                cands = [j for j, key in enumerate(model.bank_keys) if key[:2] == (rn, bbk) and key[2][0] == row[0]]
                if not cands:
                    cands = [j for j, key in enumerate(model.bank_keys) if key[:2] == (rn, bbk)]
                k = cands[rng.integers(len(cands))]
            rows = model.bank_rows(k)
            T[sel, i] = rows[rng.integers(0, len(rows), size=len(sel))]
    return T


def features(pep, T):
    """The torsion features the state definitions of the comparison use: (cos, sin) of every torsion, with
    two-fold symmetric ends doubled (Phe and Tyr chi2, Asp chi2, Glu chi3) and the two equivalent methyl torsions
    of Val and Leu summed as unit vectors, so that the labels of equivalent atoms do not matter."""
    doubled = {("PHE", 2), ("TYR", 2), ("ASP", 2), ("GLU", 3)}
    paired = {("VAL", 1), ("LEU", 2)}
    cols = []
    for i, rn in enumerate(pep.residues):
        for c in (0, 1):
            if np.isfinite(T[0, i, c]):
                cols += [np.cos(T[:, i, c]), np.sin(T[:, i, c])]
        for k in range(1, 5):
            a = T[:, i, 1 + k]
            if not np.isfinite(T[0, i, 1 + k]):
                continue
            if (rn, k) in paired:
                b = T[:, i, 6]
                cols += [np.cos(a) + np.cos(b), np.sin(a) + np.sin(b)]
            elif (rn, k) in doubled:
                cols += [np.cos(2 * a), np.sin(2 * a)]
            else:
                cols += [np.cos(a), np.sin(a)]
    return np.stack(cols, 1).astype(np.float32)


def site_states(model, pep, T):
    """Torsions (n, residues, 7) of one continuous trajectory -> the model's site states (n, S): each frame takes
    the basin of the last core it visited, so a recrossing of a basin boundary is not a jump."""
    out = np.zeros((len(T), pep.S), np.int16)
    for c, ((i, kind), key) in enumerate(zip(pep.sites, pep.keys)):
        g = model.grid[key]
        if kind == "bb" and pep.bb_kinds[i] == "2d":
            a = (np.mod(T[:, i, 0], 2 * np.pi) / (2 * np.pi) * 36).astype(int) % 36
            b = (np.mod(T[:, i, 1], 2 * np.pi) / (2 * np.pi) * 36).astype(int) % 36
            lab, core = g["lab"][a, b], g["core"][a, b]
        else:
            col = (1 if pep.bb_kinds[i] == "psi" else 0) if kind == "bb" else 1 + int(kind[-1])
            a = (np.mod(T[:, i, col], g["period"]) / g["period"] * 72).astype(int) % 72
            lab, core = g["lab"][a], g["core"][a]
        idx = np.where(core, np.arange(len(lab)), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = np.argmax(core) if core.any() else 0
        out[:, c] = lab[idx]
    return out
