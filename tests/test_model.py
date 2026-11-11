"""The network, its rate table and its dynamics."""
import os

import numpy as np
import pytest

import pepdyn
from pepdyn import model as M

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def mod():
    return pepdyn.load_model()


def test_matches_the_training_implementation(mod):
    """Log-rates and energies equal those of the PyTorch network the weights were trained in (stored vectors)."""
    ref = np.load(os.path.join(HERE, "torch_reference.npz"))
    for seq in sorted({k.split("_")[0] for k in ref.files}):
        pep = mod.peptide(seq)
        cur = ref[seq + "_cur"].astype(np.int64)
        logk, E = M.log_rates(mod, pep, cur)
        want = ref[seq + "_logk"]
        fin = np.isfinite(want)
        assert np.array_equal(fin, np.isfinite(logk))
        assert np.abs(logk[fin] - want[fin]).max() < 2e-5
        assert np.abs(E - ref[seq + "_E"]).max() < 5e-5
        assert np.abs(M.energy(mod, pep, cur) - E).max() < 1e-9


@pytest.mark.parametrize("seq", ["GCDE", "AAIT", "PYHF"])
def test_table_and_fast_rates_equal_the_reference_rates(mod, seq):
    pep = mod.peptide(seq)
    table = M.rate_table(mod, pep)
    idx = np.random.default_rng(0).choice(pep.n_configs, size=min(pep.n_configs, 300), replace=False)
    cfg = table.configs(idx)
    want = np.exp(M.log_rates(mod, pep, cfg)[0])
    for got in (table.k[idx], M.move_rates(mod, pep, cfg)):
        assert np.abs(got - want).max() <= 2e-5 * want.max()
        assert np.array_equal(got == 0, want == 0)


@pytest.mark.parametrize("seq", ["GCDE", "WNMA"])
def test_detailed_balance(mod, seq):
    """k(x -> x') exp(-E(x)) = k(x' -> x) exp(-E(x')) for every move of every configuration."""
    pep = mod.peptide(seq)
    t = M.rate_table(mod, pep)
    idx = np.arange(pep.n_configs)
    cfg = t.configs(idx)
    worst = 0.0
    for c, v in pep.moves:
        ok = cfg[:, c] != v
        if not ok.any():                 # a site with a single state has no move
            continue
        moved = idx + (v - cfg[:, c]) * t.strides[c]
        forward = np.log(t.k[idx[ok], c, v].astype(float)) - t.E[idx[ok]]
        back = np.log(t.k[moved[ok], c, cfg[ok, c]].astype(float)) - t.E[moved[ok]]
        worst = max(worst, np.abs(forward - back).max())
    assert worst < 1e-5


def site_populations(states, pep):
    return [np.bincount(states[..., c].ravel(), minlength=pep.site_n[c]) / states[..., c].size for c in range(pep.S)]


def test_starts_are_drawn_from_the_equilibrium_of_the_rates(mod):
    """Both ways of drawing starting configurations sample exp(-E) / Z."""
    pep = mod.peptide("AAIT")
    table = M.rate_table(mod, pep)
    p = table.equilibrium().reshape(table.sizes)
    exact = [p.sum(tuple(a for a in range(pep.S) if a != c)) for c in range(pep.S)]
    rng = np.random.default_rng(1)
    for x in (M.equilibrium_starts(mod, pep, 4000, rng, table), M.equilibrium_starts(mod, pep, 4000, rng, None, sweeps=25)):
        for got, want in zip(site_populations(x, pep), exact):
            assert np.abs(got - want).max() < 0.035


def test_both_paths_of_the_dynamics_are_one_chain(mod):
    """With and without a rate table, trajectories have the same site populations and jump rates."""
    pep = mod.peptide("GCDE")
    table = M.rate_table(mod, pep)
    rng = np.random.default_rng(2)
    x0 = M.equilibrium_starts(mod, pep, 300, rng, table)
    a = M.simulate(mod, pep, x0, 3.0, rng, table=table)
    b = M.simulate(mod, pep, x0, 3.0, rng)
    for pa, pb in zip(site_populations(a, pep), site_populations(b, pep)):
        assert np.abs(pa - pb).max() < 0.03
    ja, jb = (a[:, 1:] != a[:, :-1]).mean((0, 1)), (b[:, 1:] != b[:, :-1]).mean((0, 1))
    assert np.abs(ja - jb).max() < 0.01 + 0.1 * ja.max()


def test_any_sequence_runs(mod):
    out = pepdyn.generate("ACDEFGHIKLMNPQRSTVWY", ns=0.1, n_traj=2, seed=0, model=mod)
    assert out["xyz"].shape[:2] == (2, 10) and np.isfinite(out["xyz"]).all()
    with pytest.raises(ValueError):
        mod.peptide("GXDE")


def test_the_chain_settles_near_the_equilibrium_of_the_rates(mod):
    """The 10 ps chain is not the jump process sampled every 10 ps, so its own equilibrium is near exp(-E) / Z
    without being it: for GCDE, about 4% away in total variation."""
    pep = mod.peptide("GCDE")
    table = M.rate_table(mod, pep)
    n = len(table.E.ravel())
    k = np.asarray(table.k, float).reshape(n, pep.S, -1)                # rates per ps: configuration, site, state
    K = k.sum(-1)
    move = -np.expm1(-mod.meta["dt_ps"] * K)                            # each site leaves its state with this probability
    cfg = table.configs(np.arange(n))
    P = np.ones((n, n))
    for c in range(pep.S):
        to = move[:, c, None] * k[:, c] / np.maximum(K[:, c, None], 1e-300)
        to[np.arange(n), cfg[:, c]] += 1 - move[:, c]
        P *= to[:, cfg[:, c]]
    assert np.allclose(P.sum(1), 1)
    energy = table.equilibrium().ravel()
    chain = energy.copy()
    for _ in range(100000):
        new = chain @ P
        if np.abs(new - chain).sum() < 1e-12:
            break
        chain = new
    assert 0.03 < 0.5 * np.abs(chain - energy).sum() < 0.06
