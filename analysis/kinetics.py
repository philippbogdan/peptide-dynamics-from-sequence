"""Estimate populations, passage times and relaxation from discrete state paths.

Every time returned by this module is in ns. Input paths contain integer
labels at dt_ns spacing. Keep independent trajectories as separate arrays
so lagged transition counts never cross their boundaries.
"""
import numpy as np


def count_matrix(dtrajs, n, lag):
    """Count all lagged state pairs within each path, ignoring negative labels."""
    C = np.zeros((n, n))
    for d in dtrajs:
        d = np.asarray(d)
        if len(d) <= lag:
            continue
        a, b = d[:-lag], d[lag:]
        ok = (a >= 0) & (b >= 0)
        np.add.at(C, (a[ok], b[ok]), 1.0)
    return C


def largest_connected(C):
    """Choose the strongly connected component with the greatest outgoing count mass."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    nc, lab = connected_components(csr_matrix(C > 0), directed=True, connection="strong")
    sizes = np.bincount(lab, weights=C.sum(1))
    return np.where(lab == np.argmax(sizes))[0]


def reversible_mle(C):
    """Fit a reversible transition matrix and normalised stationary populations on connected counts."""
    from deeptime.markov.tools.estimation import transition_matrix
    from deeptime.markov.tools.analysis import stationary_distribution
    C = np.asarray(C, float)
    if C.shape[0] == 1:
        return np.ones((1, 1)), np.ones(1)
    T = np.asarray(transition_matrix(C, reversible=True, maxiter=200000, maxerr=1e-8))
    pi = stationary_distribution(T)
    return T, pi / pi.sum()


def timescales(T, lag_ns, k=None, pi=None):
    """Convert the nonstationary eigenvalues of T to relaxation times in ns.

    With pi, diagonal similarity produces a symmetric matrix for eigensolving.
    Eigenvalues are clipped to positive values below one before taking logs."""
    if T.shape[0] < 2:
        return np.zeros(0)
    if pi is not None:
        s = np.sqrt(pi)
        S = s[:, None] * T / s[None, :]
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
    else:
        ev = np.real(np.linalg.eigvals(T))
    ev = np.sort(ev)[::-1][1:]
    ev = np.clip(ev, 1e-300, 1 - 1e-15)
    ts = np.where(ev > 0, -lag_ns / np.log(np.abs(ev)), 0.0)
    ts = np.sort(ts)[::-1]
    return ts[:k] if k else ts


def mfpt_matrix(T, lag_ns):
    """Solve mean passage times in ns for each ordered pair; diagonal entries are zero."""
    n = T.shape[0]
    M = np.zeros((n, n))
    for j in range(n):
        idx = [i for i in range(n) if i != j]
        A = np.eye(n - 1) - T[np.ix_(idx, idx)]
        m = np.linalg.solve(A, np.ones(n - 1))
        M[idx, j] = m * lag_ns
    return M


class MSM:
    """Fit a reversible Markov model on the best-supported connected component.

    Other states have zero population and NaN mean passage times. lag_frames is
    a count of frames; dt_ns converts it to the physical model lag."""

    def __init__(self, dtrajs, n, lag_frames, dt_ns):
        self.n, self.lag, self.dt = n, lag_frames, dt_ns
        C = count_matrix(dtrajs, n, lag_frames)
        self.C = C
        act = largest_connected(C)
        self.active = act
        T, pi_a = reversible_mle(C[np.ix_(act, act)])
        self.T_active = T
        self.pi = np.zeros(n)
        self.pi[act] = pi_a
        lag_ns = lag_frames * dt_ns
        self.lag_ns = lag_ns
        self.ts = timescales(T, lag_ns, pi=pi_a)
        self._mfpt = None

    @property
    def mfpt(self):
        """Compute and cache the full mean-passage-time matrix, with NaN outside the active set."""
        if self._mfpt is None:
            act = self.active
            self._mfpt = np.full((self.n, self.n), np.nan)
            self._mfpt[np.ix_(act, act)] = mfpt_matrix(self.T_active, self.lag_ns)
        return self._mfpt

    def free_energy(self):
        """Return dimensionless state free energies, -ln(pi), with infinity for zero population."""
        with np.errstate(divide="ignore"):
            return -np.log(self.pi)


def direct_mfpt(dtrajs, n, dt_ns):
    """Estimate entry-to-entry passage times directly from labelled paths.

    Return mean times in ns and the number of completed passages. Initial
    residence is not an entry, and passages unfinished at the end of a path
    are omitted; finite trajectories can therefore censor long passages."""
    S = np.zeros((n, n))
    N = np.zeros((n, n))
    for d in dtrajs:
        d = np.asarray(d)
        if len(d) < 2:
            continue
        change = np.where(d[1:] != d[:-1])[0] + 1
        starts = np.concatenate([[0], change])
        labs = d[starts]
        times = starts * dt_ns

        entries = {j: times[labs == j] for j in range(n)}
        for i in range(n):
            ti = times[labs == i]
            if starts[0] == 0 and labs[0] == i:
                ti = ti[1:]
            for j in range(n):
                if i == j or len(ti) == 0:
                    continue
                tj = entries[j]
                pos = np.searchsorted(tj, ti, side="right")
                ok = pos < len(tj)
                S[i, j] += (tj[pos[ok]] - ti[ok]).sum()
                N[i, j] += ok.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / N, N
