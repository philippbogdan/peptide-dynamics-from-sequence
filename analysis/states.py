"""Define metastable evaluation states using only the MD reference.

Symmetry-aware torsion features enter TICA at 0.1 ns. Positive components
above 0.01 autocorrelation are eigenvalue-scaled, retaining at most 20.
K-means supplies 50 to 200 microstates, then a second fit at twice that size.
For each relaxation mode, the target is the largest implied timescale among
microstate models whose lag is shorter than that timescale.

PCCA+ memberships, or spectral clustering above 12 sets, define candidate
metastable states. Crisp assignments and two core thresholds are tried in
order of increasing state count. Accept the earliest setting whose Markov
model reproduces every target above 1 ns within 10%; when none exceeds 1 ns,
check the slowest target instead. Failure returns no state model. StateModel
maps torsion-feature matrices, not raw coordinates, to the resulting labels.
"""
import numpy as np
from dataclasses import dataclass, field

import kinetics as K

TICA_LAG_NS = 0.1
TICA_MAXDIM = 20
MICRO_LAGS_NS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
RELAX_MIN_NS = 1.0
STATE_TOL = 0.10
FRAMES_PER_MICRO = 2000
MICRO_MIN, MICRO_MAX = 50, 200


def search_order(n_frames, m0):
    """List candidate (microstates, macrostates, PCCA lag, core threshold, macro lag).

    Lags are in ns. A zero core threshold gives crisp assignment. Candidate
    counts take precedence when ordering the settings."""
    n1 = int(np.clip(n_frames // FRAMES_PER_MICRO, MICRO_MIN, MICRO_MAX))
    return [(nm, m, pl, ct, ml) for m in range(m0, m0 + EXTRA_MACRO + 1) for nm in (n1, 2 * n1)
            for pl in (1.0, 0.5) for ct in (0.0, 0.5, 0.7) for ml in (1.0, 2.0, 0.5, 0.2, 0.1)]


EXTRA_MACRO = 6


@dataclass
class StateModel:
    """Store the reference projection, cluster centres, assignments and Markov-model lag."""
    mean: np.ndarray
    tica_vecs: np.ndarray
    centers: np.ndarray
    core_of_micro: np.ndarray
    macro_of_micro: np.ndarray
    n_states: int
    macro_lag_ns: float
    dt_ns: float
    info: dict = field(default_factory=dict)

    def micro(self, X):
        """Assign features to their nearest cluster centre in the scaled TICA coordinates."""
        Y = (X - self.mean) @ self.tica_vecs

        out = np.empty(len(Y), dtype=np.int32)
        c2 = (self.centers ** 2).sum(1)
        for s in range(0, len(Y), 200000):
            y = Y[s:s + 200000]
            d = c2[None, :] - 2 * y @ self.centers.T
            out[s:s + 200000] = np.argmin(d, axis=1)
        return out

    def assign(self, X):
        """Map one feature path to metastable labels, carrying the last core label forward."""
        m = self.micro(X)
        return _core_fill(self.core_of_micro[m], self.macro_of_micro[m])

    def msm(self, dtrajs):
        """Fit a reversible Markov model at this state definition's saved lag."""
        return K.MSM(dtrajs, self.n_states, int(round(self.macro_lag_ns / self.dt_ns)), self.dt_ns)


def tica(Xs, lag):
    """Fit a reversible time-lagged covariance projection; lag is measured in frames."""
    mean = np.concatenate(Xs).mean(0)
    C0 = np.zeros((Xs[0].shape[1],) * 2)
    Ct = np.zeros_like(C0)
    n0 = 0
    for X in Xs:
        Z = X - mean
        if len(Z) <= lag:
            continue
        a, b = Z[:-lag], Z[lag:]
        C0 += a.T @ a + b.T @ b
        Ct += a.T @ b + b.T @ a
        n0 += 2 * len(a)
    C0 /= n0
    Ct /= n0

    # Discard nearly singular covariance directions before whitening.
    s, U = np.linalg.eigh(C0)
    keep = s > 1e-8 * s.max()
    W = U[:, keep] / np.sqrt(s[keep])
    lam, V = np.linalg.eigh(W.T @ Ct @ W)
    o = np.argsort(lam)[::-1]
    lam, V = lam[o], V[:, o]
    vecs = W @ V


    # Retain weak but slow signals instead of imposing a cumulative-variance cutoff.
    dim = int(min(TICA_MAXDIM, max(1, (lam > 0.01).sum())))

    return mean, vecs[:, :dim] * lam[:dim], lam


def kmeans(Y, k, seed=0, stride=None):
    """Fit reproducible k-means centres to a thinned sample of projected frames."""
    from deeptime.clustering import KMeans
    stride = stride or max(1, len(Y) // 200000)
    est = KMeans(n_clusters=k, init_strategy="kmeans++", max_iter=100, fixed_seed=seed, n_jobs=8)
    return est.fit(np.ascontiguousarray(Y[::stride])).fetch_model().cluster_centers


def micro_models(dtrajs, n, dt_ns, lags=MICRO_LAGS_NS):
    """Fit a reversible microstate model at each requested lag in ns."""
    out = {}
    for L in lags:
        lf = int(round(L / dt_ns))
        m = K.MSM(dtrajs, n, lf, dt_ns)
        out[L] = m
    return out


def best_timescales(models, kmax=30):
    """Take each mode's largest timescale among models with a shorter lag."""
    best = np.zeros(kmax)
    src = [None] * kmax
    for key, m in models.items():
        ts = m.ts[:kmax]
        for k, t in enumerate(ts):
            if t > m.lag_ns and t > best[k]:
                best[k], src[k] = t, key
    return best, src


PCCA_MAX = 12


def pcca_memberships(T, pi, m):
    """Return m-set memberships from PCCA+ or a spectral partition.

    For m above PCCA_MAX, k-means on scaled right eigenvectors produces crisp,
    one-hot memberships instead of a fuzzy PCCA+ decomposition."""
    if m <= PCCA_MAX:
        from deeptime.markov import pcca
        return pcca(T, m, pi).memberships
    s = np.sqrt(pi)
    S = s[:, None] * T / s[None, :]
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    o = np.argsort(lam)[::-1][:m]
    R = (U[:, o] / s[:, None]) * lam[o]
    from deeptime.clustering import KMeans
    km = KMeans(n_clusters=m, init_strategy="kmeans++", max_iter=300, fixed_seed=1, n_jobs=1)
    lab = km.fit(np.ascontiguousarray(R)).fetch_model().transform(np.ascontiguousarray(R))
    chi = np.zeros((len(T), m))
    chi[np.arange(len(T)), lab] = 1.0
    return chi


def discover(Xs, dt_ns, verbose=False):
    """Fit states from a list of separate feature paths at dt_ns frame spacing.

    Return (StateModel, report) on success or (None, report) if every candidate
    fails the timescale rule. Path boundaries are preserved in lagged counts."""
    lag = int(round(TICA_LAG_NS / dt_ns))
    mean, vecs, lam = tica(Xs, lag)
    Ys = [(X - mean) @ vecs for X in Xs]
    Yall = np.concatenate(Ys)
    report = dict(tica_dim=int(vecs.shape[1]), tries=[])
    n1 = int(np.clip(len(Yall) // FRAMES_PER_MICRO, MICRO_MIN, MICRO_MAX))
    cache, all_models = {}, {}
    for nm in (n1, 2 * n1):
        centers = kmeans(Yall, nm)
        sm0 = StateModel(mean, vecs, centers, np.zeros(nm, int), np.zeros(nm, int), 1, 0.1, dt_ns)
        dtr = [sm0.micro(X) for X in Xs]
        models = micro_models(dtr, nm, dt_ns)
        cache[nm] = (centers, dtr, models)
        for key, mo in models.items():
            all_models[(nm, key)] = mo
    best, src = best_timescales(all_models)
    target = best[best > RELAX_MIN_NS]
    m0 = 1 + max(1, len(target))
    report["best_timescales_ns"] = [float(x) for x in best[:max(10, len(target) + 3)]]
    report["best_source"] = [str(s) for s in src[:max(10, len(target) + 3)]]
    report["n_macro_min"] = m0
    pcca_cache = {}
    tg_all = target if len(target) else best[:1]
    for (nm, m, pl, ct, ml) in search_order(len(Yall), m0):
        # Require a model lag shorter than half every target relaxation time.
        if ml > 0.5 * tg_all.min():
            continue
        centers, dtr, models = cache[nm]
        mo = models[pl]
        act = mo.active
        if len(act) < m + 1:
            continue
        if (nm, m, pl) not in pcca_cache:
            try:
                pcca_cache[(nm, m, pl)] = pcca_memberships(mo.T_active, mo.pi[act], m)
            except Exception as e:
                pcca_cache[(nm, m, pl)] = e
        chi = pcca_cache[(nm, m, pl)]
        if isinstance(chi, Exception):
            continue
        memb = np.zeros((nm, m))
        memb[act] = chi
        macro = np.argmax(memb, 1)
        if len(set(macro)) < m:
            continue
        core = np.where(memb.max(1) >= ct, macro, -1)
        if len(set(core[core >= 0])) < m:
            continue
        sm = StateModel(mean, vecs, centers, core, macro, m, ml, dt_ns)
        dm = [_core_fill(core[d], macro[d]) for d in dtr]
        M = sm.msm(dm)
        if len(M.active) < m:
            continue
        k = len(target) if len(target) else 1
        ts = M.ts[:k]
        tg = target if len(target) else best[:1]
        ok = len(ts) >= k and np.all(np.abs(ts / tg - 1) <= STATE_TOL)
        report["tries"].append(dict(setting=(nm, m, pl, ct, ml), macro_ts=[float(x) for x in M.ts[:k + 1]], ok=bool(ok)))
        if verbose:
            print(report["tries"][-1], flush=True)
        if ok:
            sm.info = dict(setting=(nm, m, pl, ct, ml), target_ns=[float(x) for x in tg],
                           macro_ts=[float(x) for x in M.ts[:k + 2]], tica_dim=int(vecs.shape[1]))
            report["chosen"] = (nm, m, pl, ct, ml)
            report["n_macro"] = m
            return sm, report
    report["n_macro"] = None
    report["tries"] = report["tries"][-20:]
    return None, report


def _core_fill(c, macro):
    """Propagate the latest nonnegative core label along a path.

    Use the earliest core for preceding frames. If the path never enters a
    core, use its crisp macrostate labels instead."""
    c = np.asarray(c)
    if not (c >= 0).any():
        return np.asarray(macro).astype(np.int32)
    idx = np.where(c >= 0, np.arange(len(c)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = np.argmax(c >= 0)
    return c[idx].astype(np.int32)


def save(sm, path):
    """Write projection, cluster centres, state assignments and fit metadata to NPZ."""
    import json
    np.savez_compressed(path, mean=sm.mean, tica_vecs=sm.tica_vecs, centers=sm.centers, core_of_micro=sm.core_of_micro,
                        macro_of_micro=sm.macro_of_micro, n_states=sm.n_states, macro_lag_ns=sm.macro_lag_ns,
                        dt_ns=sm.dt_ns, info=json.dumps(sm.info, default=float))


def load(path):
    """Restore a StateModel from its NPZ representation without pickle objects."""
    import json
    z = np.load(path, allow_pickle=False)
    return StateModel(mean=z["mean"], tica_vecs=z["tica_vecs"], centers=z["centers"], core_of_micro=z["core_of_micro"],
                      macro_of_micro=z["macro_of_micro"], n_states=int(z["n_states"]),
                      macro_lag_ns=float(z["macro_lag_ns"]), dt_ns=float(z["dt_ns"]), info=json.loads(str(z["info"])))
