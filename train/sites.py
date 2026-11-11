"""Local conformational states ("sites") of a peptide: the model's discrete alphabet, found from training MD.

Each residue has a backbone site (the basin of its (phi, psi) map, or of the one angle a terminal residue has) and
one site per side-chain torsion (its rotamer basin). Basins are found per residue type: a histogram of the
torsion(s) on the circle or torus gives a free energy; steepest descent splits it into basins; shallow basins
(barrier below MERGE_KT above their minimum) and rare ones (below MIN_POP) are merged into their neighbour. A
basin's core is where the free energy is within CORE_KT of its minimum. Frames are assigned to the last core
visited (transition-based assignment).

A torsion table is a dict: res (residue names), tors (frames, residues, 7) in radians with the columns phi, psi,
chi1 to chi4 and the partner torsion of an equivalent methyl pair (Val CG2 on chi1, Leu CD2 on chi2), NaN where
absent, and dt_ps.
"""
import numpy as np

NB2, NB1 = 36, 72          # bins for (phi, psi) and for a single torsion
MERGE_KT, MIN_POP, CORE_KT = 1.0, 0.002, 2.5
SYMMETRIC = {("PHE", 2), ("TYR", 2), ("ASP", 2), ("GLU", 3)}   # two-fold symmetric chi: the angle modulo pi


def _smooth_periodic(H, sigma_bins=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(H, sigma_bins, mode="wrap")


def _basins(F):
    """Steepest-descent basins on a periodic grid (1D or 2D): labels (same shape), minima, neighbour index maps."""
    shape = F.shape
    idx = np.arange(F.size).reshape(shape)
    if F.ndim == 1:
        nbrs = [np.roll(idx, s) for s in (-1, 1)]
    else:
        nbrs = [np.roll(np.roll(idx, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    Ff = F.ravel()
    best = idx.ravel().copy()
    for nb in nbrs:
        nb = nb.ravel()
        better = Ff[nb] < Ff[best]
        best = np.where(better, nb, best)
    tgt = best.copy()
    for _ in range(F.size):                      # follow the pointers to their fixed points
        nt = best[tgt]
        if np.all(nt == tgt):
            break
        tgt = nt
    minima, lab = np.unique(tgt, return_inverse=True)
    return lab.reshape(shape), minima, nbrs


def _merge(F, P, lab, nbrs):
    """Merge shallow or rare basins into the neighbour across their lowest saddle."""
    Ff, labf = F.ravel(), lab.ravel().copy()
    while True:
        ids = np.unique(labf)
        if len(ids) == 1:
            break
        fmin = {i: Ff[labf == i].min() for i in ids}
        pop = {i: P.ravel()[labf == i].sum() for i in ids}
        sad = {}                                 # saddle between two basins: lowest ridge over adjacent bins
        for nb in nbrs:
            nb = nb.ravel()
            a, b = labf, labf[nb]
            m = a != b
            h = np.maximum(Ff, Ff[nb])[m]
            for x, y, v in zip(a[m], b[m], h):
                k = (min(x, y), max(x, y))
                if v < sad.get(k, np.inf):
                    sad[k] = v
        worst, target, depth_w = None, None, np.inf
        for i in ids:
            cands = [(v, (k[1] if k[0] == i else k[0])) for k, v in sad.items() if i in k]
            if not cands:
                continue
            v, j = min(cands)
            depth = v - fmin[i]
            key = depth if pop[i] >= MIN_POP else -1.0
            if (depth < MERGE_KT or pop[i] < MIN_POP) and key < depth_w:
                worst, target, depth_w = i, j, key
        if worst is None:
            break
        labf[labf == worst] = target
    _, labf = np.unique(labf, return_inverse=True)
    return labf.reshape(F.shape)


def fit_1d(angles, period=2 * np.pi):
    a = np.mod(angles, period)
    H, _ = np.histogram(a, bins=NB1, range=(0, period))
    P = H / H.sum()
    F = -np.log(_smooth_periodic(P + 1e-7, 1.0))
    lab, _, nbrs = _basins(F)
    lab = _merge(F, P, lab, nbrs)
    core = np.zeros(NB1, bool)
    for i in np.unique(lab):
        m = lab == i
        core |= m & (F <= F[m].min() + CORE_KT)
    return dict(kind="1d", period=period, lab=lab, core=core, F=F, n=int(lab.max() + 1))


def fit_2d(phi, psi):
    H, _, _ = np.histogram2d(np.mod(phi, 2 * np.pi), np.mod(psi, 2 * np.pi), bins=NB2,
                             range=[[0, 2 * np.pi], [0, 2 * np.pi]])
    P = H / H.sum()
    F = -np.log(_smooth_periodic(P + 1e-7, 1.0))
    lab, _, nbrs = _basins(F)
    lab = _merge(F, P, lab, nbrs)
    core = np.zeros_like(lab, bool)
    for i in np.unique(lab):
        m = lab == i
        core |= m & (F <= F[m].min() + CORE_KT)
    return dict(kind="2d", lab=lab, core=core, F=F, n=int(lab.max() + 1))


def raw_labels(defn, a, b=None):
    """Basin label and core flag of every frame."""
    if defn["kind"] == "1d":
        k = (np.mod(a, defn["period"]) / defn["period"] * NB1).astype(int) % NB1
        return defn["lab"][k], defn["core"][k]
    i = (np.mod(a, 2 * np.pi) / (2 * np.pi) * NB2).astype(int) % NB2
    j = (np.mod(b, 2 * np.pi) / (2 * np.pi) * NB2).astype(int) % NB2
    return defn["lab"][i, j], defn["core"][i, j]


def core_fill(lab, core):
    """Transition-based assignment: each frame takes the label of the last core visited."""
    idx = np.where(core, np.arange(len(lab)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(core) if core.any() else 0
    idx[idx < 0] = first
    return lab[idx]


def site_list(resnames, tors_sample):
    """Sites of a peptide: (residue position, kind) with kind in bb, chi1..chi4, from which torsions exist."""
    out = []
    for i, rn in enumerate(resnames):
        out.append((i, "bb"))
        for k in range(1, 5):
            if np.isfinite(tors_sample[i, 1 + k]):
                out.append((i, f"chi{k}"))
    return out


def bb_kind(tors_sample, i):
    phi_ok, psi_ok = np.isfinite(tors_sample[i, 0]), np.isfinite(tors_sample[i, 1])
    return "2d" if (phi_ok and psi_ok) else ("psi" if psi_ok else "phi")


def site_key(rn, kind, bbk):
    return (rn, kind if kind != "bb" else "bb-" + bbk)


class SiteDefs:
    """Basin definitions per (residue name, site kind)."""

    def __init__(self):
        self.defs = {}

    def fit(self, tables, max_frames=400000, seed=0):
        rng = np.random.default_rng(seed)
        pool = {}
        for tb in tables:
            T = tb["tors"]
            sub = T[rng.choice(len(T), size=min(len(T), 20000), replace=False)]
            for i, rn in enumerate(tb["res"]):
                bbk = bb_kind(T[0], i)
                key = site_key(rn, "bb", bbk)
                pool.setdefault(key, []).append(sub[:, i, :2])
                for k in range(1, 5):
                    if np.isfinite(T[0, i, 1 + k]):
                        pool.setdefault(site_key(rn, f"chi{k}", bbk), []).append(sub[:, i, 1 + k:2 + k])
        for key, arrs in pool.items():
            A = np.concatenate(arrs)[:max_frames]
            rn, kind = key
            if kind == "bb-2d":
                self.defs[key] = fit_2d(A[:, 0], A[:, 1])
            elif kind == "bb-psi":
                self.defs[key] = fit_1d(A[:, 1])
            elif kind == "bb-phi":
                self.defs[key] = fit_1d(A[:, 0])
            else:
                per = np.pi if (rn, int(kind[3])) in SYMMETRIC else 2 * np.pi
                self.defs[key] = fit_1d(A[:, 0], per)
        return self

    def site_traj(self, tb):
        """(frames, sites) labels (transition-based), the site list, its keys, and per frame and site whether
        the frame sits in the core of its own basin."""
        T = tb["tors"]
        sl = site_list(tb["res"], T[0])
        L = np.zeros((len(T), len(sl)), np.int16)
        Cf = np.zeros((len(T), len(sl)), bool)
        keys = []
        for c, (i, kind) in enumerate(sl):
            rn = tb["res"][i]
            bbk = bb_kind(T[0], i)
            key = site_key(rn, kind, bbk)
            keys.append(key)
            d = self.defs[key]
            if kind == "bb":
                if bbk == "2d":
                    lab, core = raw_labels(d, T[:, i, 0], T[:, i, 1])
                elif bbk == "psi":
                    lab, core = raw_labels(d, T[:, i, 1])
                else:
                    lab, core = raw_labels(d, T[:, i, 0])
            else:
                lab, core = raw_labels(d, T[:, i, 1 + int(kind[3])])
            L[:, c] = core_fill(lab, core)
            Cf[:, c] = core & (lab == L[:, c])
        return L, sl, keys, Cf


class Bank:
    """Observed torsions of each residue type in each combination of its site states (core frames only)."""

    def __init__(self, max_per=400, seed=0):
        self.max_per, self.rng = max_per, np.random.default_rng(seed)
        self.bank = {}

    def add(self, tb, L, sl, keys, Cf):
        T = tb["tors"]
        by_res = {}
        for c, (i, kind) in enumerate(sl):
            by_res.setdefault(i, []).append(c)
        for i, cols in by_res.items():
            allcore = Cf[:, cols].all(1)
            if not allcore.any():
                continue
            idx = np.where(allcore)[0]
            lab = L[idx][:, cols]
            rn = tb["res"][i]
            bbk = bb_kind(T[0], i)
            uniq, inv = np.unique(lab, axis=0, return_inverse=True)
            for u, row in enumerate(uniq):
                key = (rn, bbk, tuple(int(v) for v in row))
                sel = idx[inv.ravel() == u]
                if len(sel) > self.max_per:
                    sel = self.rng.choice(sel, self.max_per, replace=False)
                prev = self.bank.get(key)
                arr = T[sel, i, :]
                self.bank[key] = arr if prev is None else np.concatenate([prev, arr])[-4 * self.max_per:]
