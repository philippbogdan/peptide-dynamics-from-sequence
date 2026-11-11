"""Markov-model estimators, the same for the reference MD, its halves and the model.

A discrete trajectory is an integer array of state labels, one per frame, frames dt_ns apart. All times are in
nanoseconds.
"""
import numpy as np


def count_matrix(dtrajs, n, lag):
    """Transitions counted between frames `lag` frames apart."""
    C = np.zeros((n, n))
    for d in dtrajs:
        d = np.asarray(d)
        if len(d) > lag:
            np.add.at(C, (d[:-lag], d[lag:]), 1.0)
    return C


def largest_connected(C):
    """States of the largest strongly connected set of the count graph (by counts)."""
    n = len(C)
    reach = (C > 0) | np.eye(n, dtype=bool)
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        reach = reach | ((reach.astype(int) @ reach.astype(int)) > 0)
    both = reach & reach.T
    labels = np.array([np.argmax(both[i]) for i in range(n)])
    weight = {l: C[labels == l].sum() for l in set(labels)}
    return np.where(labels == max(weight, key=weight.get))[0]


def reversible_mle(C, tol=1e-12, max_iter=1000000):
    """Maximum-likelihood transition matrix under detailed balance, and its stationary distribution, for counts
    C on a connected set (the fixed-point iteration of Bowman et al., J. Chem. Phys. 131, 124101, 2009)."""
    C = np.asarray(C, float)
    n = len(C)
    if n == 1:
        return np.ones((1, 1)), np.ones(1)
    sym, ci = C + C.T, C.sum(1)
    X = sym / sym.sum()
    for _ in range(max_iter):
        xi = X.sum(1)
        q = ci / xi
        Xn = sym / (q[:, None] + q[None, :])
        Xn /= Xn.sum()
        done = np.abs(Xn.sum(1) - xi).max() < tol
        X = Xn
        if done:
            break
    xi = X.sum(1)
    return X / xi[:, None], xi / xi.sum()


def timescales(T, pi, lag_ns):
    """Relaxation times -lag / ln(eigenvalue), slowest first."""
    if len(T) < 2:
        return np.zeros(0)
    s = np.sqrt(pi)
    S = s[:, None] * T / s[None, :]
    ev = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))[::-1][1:]
    ev = np.clip(ev, 1e-300, 1 - 1e-15)
    return np.sort(np.where(ev > 0, -lag_ns / np.log(np.abs(ev)), 0.0))[::-1]


def mfpt_matrix(T, lag_ns):
    """M[i, j]: mean time to first reach state j from state i."""
    n = len(T)
    M = np.zeros((n, n))
    for j in range(n):
        rest = [i for i in range(n) if i != j]
        M[rest, j] = np.linalg.solve(np.eye(n - 1) - T[np.ix_(rest, rest)], np.ones(n - 1)) * lag_ns
    return M


class MSM:
    """Reversible maximum-likelihood Markov model on n states at a lag. States outside the connected set have
    population 0 and undefined passage times."""

    def __init__(self, dtrajs, n, lag_frames, dt_ns):
        self.n, self.lag_ns = n, lag_frames * dt_ns
        self.C = count_matrix(dtrajs, n, lag_frames)
        self.active = largest_connected(self.C)
        self.T, pi = reversible_mle(self.C[np.ix_(self.active, self.active)])
        self.pi = np.zeros(n)
        self.pi[self.active] = pi
        self.ts = timescales(self.T, pi, self.lag_ns)
        self.mfpt = np.full((n, n), np.nan)
        self.mfpt[np.ix_(self.active, self.active)] = mfpt_matrix(self.T, self.lag_ns)
