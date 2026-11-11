"""Sequence in, heavy-atom trajectory out."""
import os
import time

import numpy as np

from . import build, emit
from . import model as M

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
TABLE_LIMIT = 1000000      # joint configurations up to which every rate is tabulated once (about 1 GB)
MIN_DISTANCE = 2.2         # angstrom: closer than this, two atoms more than three bonds apart overlap
DRAWS = 100                # torsion draws tried for a frame before the least overlapping one is kept


def load_model(name="implicit-310K"):
    """A model by name (a file in models/) or by path."""
    path = name if os.path.exists(name) else os.path.join(MODELS, name + ".npz")
    return M.Model(path)


def frames_from_states(mod, pep, states, rng, redraw=True):
    """Site states (n, S) -> torsions (n, residues, 7), topology, coordinates (n, atoms, 3) and the share of
    frames drawn more than once and of frames left with an overlap.

    Each residue's torsions are drawn on their own, and atoms are built with fixed bond lengths and angles, so a
    draw can put two atoms on top of each other. Such a frame is drawn again, up to DRAWS times, and the draw
    with the widest closest contact is kept. The site states, which carry the dynamics, are never changed."""
    T = emit.torsions(mod, pep, states, rng)
    top, X = build.coordinates(pep, T, rng)
    pairs = build.far_pairs(top)
    d = build.closest(X, pairs)
    again = d < MIN_DISTANCE
    first = again.mean()
    for _ in range(DRAWS - 1 if redraw else 0):
        idx = np.where(again)[0]
        if not len(idx):
            break
        T2 = emit.torsions(mod, pep, states[idx], rng)
        _, X2 = build.coordinates(pep, T2, rng)
        d2 = build.closest(X2, pairs)
        better = d2 > d[idx]
        T[idx[better]], X[idx[better]], d[idx[better]] = T2[better], X2[better], d2[better]
        again = d < MIN_DISTANCE
    return T, top, X, dict(drawn_again=float(first), left_overlapping=float(again.mean()))


def generate(sequence, ns=100.0, n_traj=1, seed=0, model="implicit-310K", frame_ps=10.0, redraw=True):
    """Generate n_traj trajectories of ns nanoseconds each for a peptide sequence.

    Returns a dict with: states (n_traj, frames, sites), the model's site states; torsions (n_traj, frames,
    residues, 7) in radians; xyz (n_traj, frames, atoms, 3) in angstrom; topology; peptide; overlaps; and
    seconds, the wall time of each stage. Every trajectory starts from the model's own equilibrium."""
    mod = load_model(model) if isinstance(model, str) else model
    rng = np.random.default_rng(seed)
    pep = mod.peptide(sequence)
    t0 = time.perf_counter()
    table = M.rate_table(mod, pep, TABLE_LIMIT)
    x0 = M.equilibrium_starts(mod, pep, n_traj, rng, table)
    t1 = time.perf_counter()
    states = M.simulate(mod, pep, x0, ns, rng, frame_ps, table)
    t2 = time.perf_counter()
    flat = states.reshape(-1, pep.S).astype(np.int64)
    T, top, X, overlaps = frames_from_states(mod, pep, flat, rng, redraw)
    t3 = time.perf_counter()
    return dict(peptide=pep, model=mod, states=states, topology=top, overlaps=overlaps,
                torsions=T.reshape(n_traj, -1, *T.shape[1:]),
                xyz=X.reshape(n_traj, -1, top.n_atoms, 3).astype(np.float32),
                seconds=dict(rates_and_starts=t1 - t0, dynamics=t2 - t1, atoms=t3 - t2, total=t3 - t0))
