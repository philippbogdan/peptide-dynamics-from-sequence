"""Learn local conformational states from periodic torsion histograms.

Backbone phi/psi angles define one site; each available chi angle defines
another. Smoothed histograms give free energies in kT. Steepest descent
assigns bins to minima, then low barriers and small populations are merged.
A core lies within CORE_KT of its basin minimum. Between cores, a trajectory
keeps the state of the most recently visited core.
"""
import numpy as np

NB2, NB1 = 36, 72
MERGE_KT, MIN_POP, CORE_KT = 1.0, 0.002, 2.5
SYMMETRIC = {("PHE", 2), ("TYR", 2), ("ASP", 2), ("GLU", 3)}


def _smooth_periodic(H, sigma_bins=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(H, sigma_bins, mode="wrap")


def _basins(F):
    """Return basin labels, minimum-bin indices and neighbours on a periodic grid."""
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

    # Repeated downhill links terminate at a local minimum.
    tgt = best.copy()
    for _ in range(F.size):
        nt = best[tgt]
        if np.all(nt == tgt):
            break
        tgt = nt
    minima, lab = np.unique(tgt, return_inverse=True)
    return lab.reshape(shape), minima, nbrs


def _merge(F, P, lab, nbrs):
    """Join shallow or sparsely occupied basins across their lowest boundary saddle."""
    Ff, labf = F.ravel(), lab.ravel().copy()
    while True:
        ids = np.unique(labf)
        if len(ids) == 1:
            break
        fmin = {i: Ff[labf == i].min() for i in ids}
        pop = {i: P.ravel()[labf == i].sum() for i in ids}

        # A boundary saddle is the lowest maximum energy across adjacent bins.
        sad = {}
        for nb in nbrs:
            nb = nb.ravel()
            a, b = labf, labf[nb]
            m = a != b
            h = np.maximum(Ff, Ff[nb])[m]
            for x, y, v in zip(a[m], b[m], h):
                k = (min(x, y), max(x, y))
                if v < sad.get(k, np.inf):
                    sad[k] = v

        # Underpopulated basins take priority over merely shallow ones.
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
    """Fit periodic basins and their cores for one torsion angle, in radians."""
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
    """Fit backbone basins and cores on the periodic phi/psi grid, in radians."""
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
    """Look up the basin label and core membership of every supplied angle."""
    if defn["kind"] == "1d":
        k = (np.mod(a, defn["period"]) / defn["period"] * NB1).astype(int) % NB1
        return defn["lab"][k], defn["core"][k]
    i = (np.mod(a, 2 * np.pi) / (2 * np.pi) * NB2).astype(int) % NB2
    j = (np.mod(b, 2 * np.pi) / (2 * np.pi) * NB2).astype(int) % NB2
    return defn["lab"][i, j], defn["core"][i, j]


def core_fill(lab, core):
    """Carry each core label forward until a different core is visited.

    Frames preceding any core visit use the earliest core; if none is visited,
    all frames receive the initial basin label."""
    idx = np.where(core, np.arange(len(lab)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.argmax(core) if core.any() else 0
    idx[idx < 0] = first
    return lab[idx]
