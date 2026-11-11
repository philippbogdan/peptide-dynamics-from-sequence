"""The estimators and the comparison."""
import glob
import os

import numpy as np

from pepdyn import compare, markov, states

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chain(T, n, seed=0):
    rng = np.random.default_rng(seed)
    c = np.cumsum(T, 1)
    x = np.zeros(n, int)
    u = rng.random(n)
    for t in range(1, n):
        x[t] = min(np.searchsorted(c[x[t - 1]], u[t]), len(T) - 1)
    return x


def test_estimators_recover_a_known_chain():
    T = np.array([[0.98, 0.015, 0.005], [0.03, 0.96, 0.01], [0.02, 0.02, 0.96]])
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmax(np.real(w))])
    pi /= pi.sum()
    m = markov.MSM([chain(T, 400000)], 3, 1, 0.01)
    assert np.abs(m.pi - pi).max() < 0.01
    assert np.abs(m.T - T).max() < 0.005
    exact = markov.mfpt_matrix(T, 0.01)
    off = ~np.eye(3, dtype=bool)
    assert np.abs(m.mfpt[off] / exact[off] - 1).max() < 0.1
    assert np.allclose(m.T * m.pi[:, None], (m.T * m.pi[:, None]).T, atol=1e-9)      # detailed balance of the estimate


def test_two_state_passage_time_is_one_over_the_rate():
    T = np.array([[0.9, 0.1], [0.2, 0.8]])
    M = markov.mfpt_matrix(T, 1.0)
    assert np.isclose(M[0, 1], 10.0) and np.isclose(M[1, 0], 5.0)
    assert np.isclose(markov.timescales(T, np.array([2 / 3, 1 / 3]), 1.0)[0], -1 / np.log(0.7))


def test_only_the_connected_states_are_estimated():
    d = np.array([0, 1, 0, 1, 1, 0, 0, 1, 2, 2, 2, 2])           # state 2 is entered and never left
    m = markov.MSM([d], 3, 1, 1.0)
    assert set(m.active) == {0, 1} and m.pi[2] == 0 and np.isnan(m.mfpt[0, 2])


def test_the_comparison_fails_what_it_must():
    """On every stored reference: the same frames shuffled in time, and the same reference with its clock doubled."""
    files = sorted(glob.glob(os.path.join(HERE, "data", "reference", "*.npz")))
    judged = 0
    for f in files:
        z = np.load(f)
        if not bool(z["states_ok"]):
            continue
        sm = states.StateModel.load(f)
        d = z["md_states"].astype(np.int64)
        halves = [d[:len(d) // 2], d[len(d) // 2:]]
        ref = sm.msm(halves)
        assert compare.compare(ref, ref, z["main"])["pass_all"]
        assert not compare.compare(sm.msm(compare.shuffled(halves)), ref, z["main"])["pass_all"]
        assert not compare.compare(compare.DoubledClock(ref), ref, z["main"])["pass_all"]
        judged += 1
    assert judged == 15


def test_the_stored_states_obey_the_rule():
    """Every stored state definition gives its reference's relaxation times above 1 ns within 10%, from a Markov
    model whose lag is at most half the shortest of them."""
    for f in sorted(glob.glob(os.path.join(HERE, "data", "reference*", "*.npz"))):
        z = np.load(f)
        if not bool(z["states_ok"]):
            continue
        sm = states.StateModel.load(f)
        d = z["md_states"].astype(np.int64)
        best = z["best_timescales_ns"]
        goal = best[best > states.SLOW_NS] if (best > states.SLOW_NS).any() else best[:1]
        assert sm.lag_ns <= 0.5 * goal.min() + 1e-9, f
        ts = sm.msm([d[:len(d) // 2], d[len(d) // 2:]]).ts[:len(goal)]
        assert len(ts) == len(goal) and np.all(np.abs(ts / goal - 1) <= states.TOLERANCE + 1e-6), f
