"""The comparison: rates, free energies and relaxation times of a candidate against a reference.

States come from the reference alone (pepdyn/states.py). On those states a Markov model gives, for any set of
trajectories, populations, free energies F = -ln(population) in kT, mean first-passage times between states and
relaxation times. Main states hold at least 1% of the reference population.
"""
import numpy as np

MAIN_POPULATION = 0.01
TOLERANCE = dict(mfpt_factor=1.5, f_rms_kT=0.1, tv=0.02, slowest=0.10, others=0.20)


def compare(cand, ref, main=None):
    """cand, ref: MSM objects on the same states. Returns the measured values and whether each is in tolerance."""
    main = np.where(ref.pi >= MAIN_POPULATION)[0] if main is None else np.asarray(main, int)
    out = dict(main_states=main.tolist(), n_states=int(ref.n))
    pairs, worst = [], 1.0
    for i in main:
        for j in main:
            if i != j:
                a, b = cand.mfpt[i, j], ref.mfpt[i, j]
                f = max(a / b, b / a) if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 else np.inf
                pairs.append((int(i), int(j), float(a), float(b), float(f)))
                worst = max(worst, f)
    out.update(mfpt_pairs=pairs, mfpt_worst_factor=float(worst))
    with np.errstate(divide="ignore"):
        dF = -np.log(cand.pi[main]) + np.log(ref.pi[main])
    out["dF"] = [float(x) for x in dF]
    out["f_rms_kT"] = float(np.sqrt(((dF - dF.mean()) ** 2).mean())) if np.all(np.isfinite(dF)) else float("inf")
    out["tv"] = float(0.5 * np.abs(cand.pi - ref.pi).sum())
    rel = []
    for k in range(len(ref.ts)):
        if k > 0 and ref.ts[k] <= 1.0:
            break
        c = cand.ts[k] if k < len(cand.ts) else 0.0
        rel.append((k, float(c), float(ref.ts[k]), float(abs(c / ref.ts[k] - 1))))
    out["relaxation"] = rel
    out["slowest_rel_err"] = rel[0][3] if rel else float("inf")
    out["others_worst_rel_err"] = max([r[3] for r in rel[1:]], default=0.0)
    t = TOLERANCE
    out["pass"] = dict(rates=bool(worst <= t["mfpt_factor"]), free_energy=bool(out["f_rms_kT"] <= t["f_rms_kT"]),
                       populations=bool(out["tv"] <= t["tv"]),
                       relaxation=bool(out["slowest_rel_err"] <= t["slowest"] and out["others_worst_rel_err"] <= t["others"]))
    out["pass_all"] = all(out["pass"].values())
    return out


def shuffled(dtrajs, seed=0):
    """The same frames in random order: populations kept, all kinetics destroyed. A comparison must fail it."""
    rng = np.random.default_rng(seed)
    flat = np.concatenate(dtrajs)
    rng.shuffle(flat)
    cuts = np.cumsum([len(d) for d in dtrajs])[:-1]
    return np.split(flat, cuts)


class DoubledClock:
    """A reference with every time doubled. A comparison must fail it."""

    def __init__(self, ref):
        self.n, self.pi, self.mfpt, self.ts = ref.n, ref.pi, 2 * ref.mfpt, 2 * ref.ts
