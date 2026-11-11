"""Metastable states of a peptide, found from its reference MD alone and then applied unchanged to any candidate.

The rule (discover):
1. Torsion features (pepdyn/emit.py: features) of every reference frame.
2. Time-lagged independent component analysis at lag 0.1 ns, kinetic-map scaled, keeping every component with
   autocorrelation above 0.01 (at most 20).
3. k-means clusters in those coordinates, one per about 2,000 frames (50 to 200, and twice that).
4. Markov models on the clusters at lags 0.05 to 2 ns give the reference's relaxation times, each taken as the
   largest value any of these models gives (by the variational principle an estimate falls short of the truth).
5. The clusters are lumped into m metastable sets by PCCA+, m starting at 1 + the number of relaxation times
   above 1 ns and growing by up to 6.
6. Frames are assigned crisply, or to the last core visited (clusters whose membership exceeds 0.5 or 0.7), so
   that a recrossing of a boundary is not a transition.
7. The first setting, in a fixed order with the fewest states first, whose Markov model on the sets reproduces
   every reference relaxation time above 1 ns within 10% defines the states (the slowest one, if none is above
   1 ns). The lag of that model is at most half the shortest relaxation time it has to reproduce. If no setting
   does, the peptide has no states and cannot be judged this way.

Applying a StateModel needs only numpy. Finding one (discover) also needs deeptime.
"""
import numpy as np

from .markov import MSM

TICA_LAG_NS, TICA_MAXDIM = 0.1, 20
CLUSTER_LAGS_NS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
SLOW_NS, TOLERANCE = 1.0, 0.10
FRAMES_PER_CLUSTER, CLUSTERS_MIN, CLUSTERS_MAX = 2000, 50, 200
EXTRA_STATES, PCCA_MAX = 6, 12


class StateModel:
    """Maps torsion features to state labels: projection, nearest cluster centre, state of the last core visited."""

    def __init__(self, mean, vecs, centers, core_of_cluster, state_of_cluster, n_states, lag_ns, dt_ns=0.01):
        self.mean, self.vecs, self.centers = mean, vecs, centers
        self.core_of_cluster, self.state_of_cluster = np.asarray(core_of_cluster), np.asarray(state_of_cluster)
        self.n_states, self.lag_ns, self.dt_ns = int(n_states), float(lag_ns), float(dt_ns)

    def cluster(self, X):
        Y = (X - self.mean) @ self.vecs
        out = np.empty(len(Y), np.int32)
        c2 = (self.centers ** 2).sum(1)
        for s in range(0, len(Y), 200000):
            y = Y[s:s + 200000]
            out[s:s + 200000] = np.argmin(c2[None, :] - 2 * y @ self.centers.T, axis=1)
        return out

    def assign(self, X):
        """State labels of one continuous trajectory's feature matrix."""
        m = self.cluster(X)
        return core_fill(self.core_of_cluster[m], self.state_of_cluster[m])

    def msm(self, dtrajs):
        return MSM(dtrajs, self.n_states, int(round(self.lag_ns / self.dt_ns)), self.dt_ns)

    def save(self, path, **extra):
        np.savez_compressed(path, mean=self.mean, vecs=self.vecs, centers=self.centers,
                            core_of_cluster=self.core_of_cluster, state_of_cluster=self.state_of_cluster,
                            n_states=self.n_states, lag_ns=self.lag_ns, dt_ns=self.dt_ns, **extra)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["mean"], z["vecs"], z["centers"], z["core_of_cluster"], z["state_of_cluster"],
                   int(z["n_states"]), float(z["lag_ns"]), float(z["dt_ns"]))


def core_fill(core, state):
    """Each frame takes the state of the last core visited (before the first core: the first core's)."""
    core = np.asarray(core)
    if not (core >= 0).any():
        return np.asarray(state).astype(np.int32)
    idx = np.where(core >= 0, np.arange(len(core)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = np.argmax(core >= 0)
    return core[idx].astype(np.int32)


def tica(Xs, lag):
    mean = np.concatenate(Xs).mean(0)
    C0 = np.zeros((Xs[0].shape[1],) * 2)
    Ct = np.zeros_like(C0)
    n = 0
    for X in Xs:
        Z = X - mean
        if len(Z) > lag:
            a, b = Z[:-lag], Z[lag:]
            C0 += a.T @ a + b.T @ b
            Ct += a.T @ b + b.T @ a
            n += 2 * len(a)
    C0, Ct = C0 / n, Ct / n
    s, U = np.linalg.eigh(C0)
    keep = s > 1e-8 * s.max()
    W = U[:, keep] / np.sqrt(s[keep])
    lam, V = np.linalg.eigh(W.T @ Ct @ W)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    dim = int(min(TICA_MAXDIM, max(1, (lam > 0.01).sum())))
    return mean, (W @ V)[:, :dim] * lam[:dim]


def _kmeans(Y, k, seed=0):
    from deeptime.clustering import KMeans
    est = KMeans(n_clusters=k, init_strategy="kmeans++", max_iter=100, fixed_seed=seed, n_jobs=8)
    return est.fit(np.ascontiguousarray(Y[::max(1, len(Y) // 200000)])).fetch_model().cluster_centers


def _memberships(T, pi, m):
    """Memberships of m metastable sets: PCCA+, or above PCCA_MAX sets k-means on the leading eigenvectors."""
    if m <= PCCA_MAX:
        from deeptime.markov import pcca
        return pcca(T, m, pi).memberships
    from deeptime.clustering import KMeans
    s = np.sqrt(pi)
    S = s[:, None] * T / s[None, :]
    lam, U = np.linalg.eigh(0.5 * (S + S.T))
    o = np.argsort(lam)[::-1][:m]
    R = np.ascontiguousarray((U[:, o] / s[:, None]) * lam[o])
    lab = KMeans(n_clusters=m, init_strategy="kmeans++", max_iter=300, fixed_seed=1, n_jobs=1).fit(R).fetch_model().transform(R)
    chi = np.zeros((len(T), m))
    chi[np.arange(len(T)), lab] = 1.0
    return chi


def discover(Xs, dt_ns=0.01, seed=0):
    """Feature matrices of the reference replicas -> (StateModel or None, report). seed: of the k-means."""
    mean, vecs = tica(Xs, int(round(TICA_LAG_NS / dt_ns)))
    Y = np.concatenate([(X - mean) @ vecs for X in Xs])
    n1 = int(np.clip(len(Y) // FRAMES_PER_CLUSTER, CLUSTERS_MIN, CLUSTERS_MAX))
    cache, best = {}, np.zeros(30)
    for nc in (n1, 2 * n1):
        centers = _kmeans(Y, nc, seed)
        probe = StateModel(mean, vecs, centers, np.zeros(nc, int), np.zeros(nc, int), 1, 0.1, dt_ns)
        dtr = [probe.cluster(X) for X in Xs]
        models = {L: MSM(dtr, nc, int(round(L / dt_ns)), dt_ns) for L in CLUSTER_LAGS_NS}
        cache[nc] = (centers, dtr, models)
        for m in models.values():
            for k, t in enumerate(m.ts[:30]):
                if t > m.lag_ns:
                    best[k] = max(best[k], t)
    target = best[best > SLOW_NS]
    goal = target if len(target) else best[:1]
    m0 = 1 + max(1, len(target))
    report = dict(best_timescales_ns=[float(x) for x in best[:max(10, len(target) + 3)]], n_states=None)
    memberships = {}
    for m in range(m0, m0 + EXTRA_STATES + 1):
        for nc in (n1, 2 * n1):
            for pcca_lag in (1.0, 0.5):
                for core in (0.0, 0.5, 0.7):
                    for lag in (1.0, 2.0, 0.5, 0.2, 0.1):
                        if lag > 0.5 * goal.min():
                            continue
                        centers, dtr, models = cache[nc]
                        mo = models[pcca_lag]
                        if len(mo.active) < m + 1:
                            continue
                        key = (nc, m, pcca_lag)
                        if key not in memberships:
                            try:
                                memberships[key] = _memberships(mo.T, mo.pi[mo.active], m)
                            except Exception as e:      # PCCA+ can fail on a nearly decomposable matrix
                                memberships[key] = e
                        if isinstance(memberships[key], Exception):
                            continue
                        memb = np.zeros((nc, m))
                        memb[mo.active] = memberships[key]
                        state = np.argmax(memb, 1)
                        cores = np.where(memb.max(1) >= core, state, -1)
                        if len(set(state)) < m or len(set(cores[cores >= 0])) < m:
                            continue
                        sm = StateModel(mean, vecs, centers, cores, state, m, lag, dt_ns)
                        M = sm.msm([core_fill(cores[d], state[d]) for d in dtr])
                        k = len(goal)
                        if len(M.active) >= m and len(M.ts) >= k and np.all(np.abs(M.ts[:k] / goal - 1) <= TOLERANCE):
                            report.update(n_states=m, setting=[nc, m, pcca_lag, core, lag])
                            return sm, report
    return None, report
