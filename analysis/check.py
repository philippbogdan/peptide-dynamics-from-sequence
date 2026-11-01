"""Compare candidates and references on one common set of metastable states.

Both use reversible Markov estimates at the reference-selected lag. Agreement
requires all of: every mean passage time between reference states with at
least 1% population within a factor 1.5; centred free-energy RMS at most
0.1 kT on those states; total variation at most 0.02; slowest relaxation
within 10%; and each remaining reference relaxation above 1 ns within 20%.
The identical comparison is applied to the two halves of the MD reference.
"""
import numpy as np

MAIN_MIN_POP = 0.01
TOL = dict(mfpt_factor=1.5, f_rms_kT=0.1, tv=0.02, slowest=0.10, others=0.20)


def estimates(sm, dtrajs):
    """Fit the saved state model to paths and expose its populations, MFPTs and timescales."""
    M = sm.msm(dtrajs)
    return dict(pi=M.pi, mfpt=M.mfpt, ts=M.ts, active=M.active, msm=M)


def compare(sm, cand, ref, main=None):
    """Return measured discrepancies and Boolean tolerance checks.

    cand and ref are outputs of estimates. Optional main fixes the reference
    states used for free energies and passage times. Missing or invalid passage
    times give an infinite factor; missing candidate modes count as zero time."""
    pi_r, pi_c = ref["pi"], cand["pi"]
    main = np.where(pi_r >= MAIN_MIN_POP)[0] if main is None else main
    out = dict(main_states=main.tolist(), n_states=int(sm.n_states))

    # Missing passage times must fail rather than disappearing from the comparison.
    worst, pairs = 1.0, []
    for i in main:
        for j in main:
            if i == j:
                continue
            a, b = cand["mfpt"][i, j], ref["mfpt"][i, j]
            f = np.inf if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else max(a / b, b / a)
            pairs.append((int(i), int(j), float(a), float(b), float(f)))
            worst = max(worst, f)
    out["mfpt_pairs"] = pairs
    out["mfpt_worst_factor"] = float(worst)
    out["pass_rates"] = bool(worst <= TOL["mfpt_factor"])

    # Free energies have an arbitrary additive constant; remove the mean difference.
    with np.errstate(divide="ignore"):
        dF = -np.log(pi_c[main]) + np.log(pi_r[main])
    if np.all(np.isfinite(dF)):
        dF = dF - dF.mean()
        rms = float(np.sqrt((dF ** 2).mean()))
    else:
        rms = float("inf")
    tv = float(0.5 * np.abs(pi_c - pi_r).sum())
    out.update(f_rms_kT=rms, tv=tv, pass_free_energy=bool(rms <= TOL["f_rms_kT"]), pass_tv=bool(tv <= TOL["tv"]))

    tr, tc = ref["ts"], cand["ts"]
    rel = []
    ok_slow = len(tr) > 0 and len(tc) > 0 and abs(tc[0] / tr[0] - 1) <= TOL["slowest"]
    ok_oth = True
    for k in range(len(tr)):
        if k > 0 and tr[k] <= 1.0:
            break
        c = tc[k] if k < len(tc) else 0.0
        e = abs(c / tr[k] - 1)
        rel.append((k, float(c), float(tr[k]), float(e)))
        if k > 0 and e > TOL["others"]:
            ok_oth = False
    out["relaxation"] = rel
    out["slowest_rel_err"] = rel[0][3] if rel else float("inf")
    out["others_worst_rel_err"] = max([r[3] for r in rel[1:]], default=0.0)
    out["pass_relaxation"] = bool(ok_slow and ok_oth)
    out["pass_all"] = bool(out["pass_rates"] and out["pass_free_energy"] and out["pass_tv"] and out["pass_relaxation"])
    return out


def shuffled(dtrajs, seed=0):
    """Randomly permute all frame labels, then restore the original path lengths."""
    rng = np.random.default_rng(seed)
    allf = np.concatenate(dtrajs)
    rng.shuffle(allf)
    out, s = [], 0
    for d in dtrajs:
        out.append(allf[s:s + len(d)])
        s += len(d)
    return out
