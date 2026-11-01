"""Recompute headline statistics from the small files supplied under results/.

Usage: python reproduce.py

Refit Markov estimates from saved state-label paths, summarise four model
samples, compare equal-length MD and model data, and estimate the reference
length needed for a precise rate comparison. Timing, energy-replacement and
local-rate diagnostics are summarised from their saved records. No raw MD,
model sampling, training or GPU is required. Write results/headline.json and
print the numerical report. These are development results: the public test
peptides also informed model selection.
"""

import json, os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import numpy as np
import paths as P
import check as CK
import states as ST

BINS_NS = [(3, 10), (10, 30), (30, 100), (100, 300), (300, 1000), (1000, np.inf)]
TRADE_KEYS = ("mfpt_worst_factor", "f_rms_kT", "tv", "slowest_rel_err")


def load_npz(path):
    """Read saved discrete labels as int32 arrays."""
    z = np.load(path)
    return {k: z[k].astype(np.int32) for k in z.files}


def median(xs):
    return float(np.median(np.asarray(xs, float)))


def seeds():
    """Find every model sampling seed represented by saved state paths."""
    return sorted({int(f.split("_seed")[1][:-4]) for f in os.listdir(P.DTRAJS) if ".model_seed" in f})


def held_out(seed):
    """Re-estimate one model sample and both reference halves on their saved states."""
    rows = []
    for name in P.judged_peptides():
        f = os.path.join(P.DTRAJS, name + f".model_seed{seed}.npz")
        if not os.path.exists(f):
            continue
        sm = ST.load(os.path.join(P.STATES, name + ".npz"))
        main_ = np.array(json.load(open(os.path.join(P.REFERENCES, name + ".json")))["main"], int)
        r = load_npz(os.path.join(P.DTRAJS, name + ".reference.npz"))
        dA, dB = r["half_a"], r["half_b"]
        ref = CK.estimates(sm, [dA, dB])
        mod = CK.estimates(sm, list(load_npz(f)["model"]))
        rows.append(dict(name=name, model=CK.compare(sm, mod, ref, main=main_),
                         replicas=CK.compare(sm, CK.estimates(sm, [dA]), CK.estimates(sm, [dB]), main=main_),
                         slowest_ref_ns=float(ref["ts"][0]), slowest_model_ns=float(mod["ts"][0]),
                         main=main_.tolist(), pi_ref=ref["pi"].tolist(), pi_model=mod["pi"].tolist(),
                         ts_ref=ref["ts"][:5].tolist(), ts_model=mod["ts"][:5].tolist()))
    return rows


def mfpt_bins(rows, who):
    """Group ordered state pairs by reference passage time and report factor-1.5 agreement."""
    pairs = [(b, f) for r in rows for (_, _, _, b, f) in r[who]["mfpt_pairs"]]
    out = []
    for lo, hi in BINS_NS:
        fs = [f for b, f in pairs if lo <= b < hi]
        out.append(dict(ref_mfpt_ns=[lo, None if hi == np.inf else hi], pairs=len(fs),
                        within_1p5=float(np.mean(np.array(fs) <= 1.5)) if fs else None))
    return out


def lstar():
    """Estimate required reference length from the slowest mode and its two population masses.

    Use the sign of the slow nonstationary eigenvector to divide the states.
    The returned interval is an order-of-magnitude counting estimate in us,
    not a confidence interval or a measured convergence length."""
    rows = []
    for name in P.judged_peptides():
        sm = ST.load(os.path.join(P.STATES, name + ".npz"))
        r = load_npz(os.path.join(P.DTRAJS, name + ".reference.npz"))
        dA, dB = r["half_a"], r["half_b"]
        M = CK.estimates(sm, [dA, dB])["msm"]
        T, pi = M.T_active, M.pi[M.active]
        w, v = np.linalg.eig(T.T)
        phi = np.real(v[:, np.argsort(-np.real(w))[1]])
        pa = float(pi[phi / pi > 0].sum())
        tau = float(M.ts[0])
        main_ = np.array(json.load(open(os.path.join(P.REFERENCES, name + ".json")))["main"], int)
        ef = CK.compare(sm, CK.estimates(sm, [dA]), CK.estimates(sm, [dB]), main=main_)
        rows.append(dict(name=name, slowest_ns=tau, p_minor=min(pa, 1 - pa),
                         lstar_us=[1.0 * tau / (pa * (1 - pa)), 2.0 * tau / (pa * (1 - pa))],
                         halves_pass_all=bool(ef["pass_all"]), halves_slowest_rel_err=ef["slowest_rel_err"],
                         halves_mfpt_worst_factor=ef["mfpt_worst_factor"]))
    return rows


def trade(cost):
    """Refit length-dependent errors from saved paths and attach measured timing factors."""
    per = []
    for name in P.judged_peptides():
        f = os.path.join(P.DTRAJS, name + ".trade.npz")
        if not os.path.exists(f):
            continue
        sm = ST.load(os.path.join(P.STATES, name + ".npz"))
        main_ = np.array(json.load(open(os.path.join(P.REFERENCES, name + ".json")))["main"], int)
        r = load_npz(os.path.join(P.DTRAJS, name + ".reference.npz"))
        dA, dB = r["half_a"], r["half_b"]
        refB = CK.estimates(sm, [dB])
        z = load_npz(f)
        row = dict(name=name, md={}, model={})
        for key in z:
            L = int(key.split("_")[1][:-2])
            row["model"][L] = CK.compare(sm, CK.estimates(sm, list(z[key])), refB, main=main_)
            k = int(round(L / P.FRAME_NS))
            if k <= len(dA):
                row["md"][L] = CK.compare(sm, CK.estimates(sm, [dA[:k]]), refB, main=main_)
        per.append(row)
    lengths = sorted(per[0]["model"])
    table = []
    for L in lengths:
        e = dict(length_ns=L, md_gpu_s=L * cost["md_gpu_s_per_ns"], model_gpu_s=L * cost["model_gpu_s_per_ns"])
        for who in ("md", "model"):
            e[who] = {k: median([r[who][L][k] for r in per if L in r[who]]) for k in TRADE_KEYS}
        table.append(e)
    return table


def costs():
    """Read the model timing and the greatest aggregate MD throughput in the supplied records."""
    m = json.load(open(os.path.join(P.RESULTS, "cost", "model_t4.json")))
    b = [json.loads(l) for l in open(os.path.join(P.RESULTS, "cost", "md_implicit_t4.jsonl"))]
    best = max(x["agg"] for x in b)
    return dict(model_gpu_s_per_ns=m["model_gpu_s_per_ns"], md_ns_per_day_best=best, md_gpu_s_per_ns=86400.0 / best)


def oracle():
    """Summarise the saved MD-energy replacement experiment without sampling new paths."""
    rs = json.load(open(os.path.join(P.RESULTS, "oracle.json")))
    out = {}
    for who in ("model", "thermo_oracle", "replicas"):
        out[who] = {k: median([r[who][k] for r in rs]) for k in TRADE_KEYS}
    out["oracle_pass_all"] = sum(bool(r["thermo_oracle"]["pass_all"]) for r in rs)
    out["peptides"] = len(rs)
    return out


def context_skill(min_events=20):
    """Compare MD and model log-rate deviations after filtering by observed event count."""
    rows = json.load(open(os.path.join(P.RESULTS, "barriers.json")))
    x, y, g = [], [], []
    for r in rows:
        if r["md_transitions"] < min_events or None in (r["G_md"], r["G_model"], r["G_pooled_test"]):
            continue
        x.append(r["G_md"] - r["G_pooled_test"])
        y.append(r["G_model"] - r["G_pooled_test"])
        g.append(r["G_md"])
    x, y, g = map(np.array, (x, y, g))
    out = {}
    for lab, s in (("all", np.ones(len(x), bool)), ("slow", g < -3)):
        out[lab] = dict(n=int(s.sum()), corr=float(np.corrcoef(x[s], y[s])[0, 1]),
                        slope=float(np.polyfit(x[s], y[s], 1)[0]), md_sd=float(x[s].std()),
                        rms_model_error=float(np.sqrt(np.mean((y[s] - x[s]) ** 2))),
                        rms_pooled_error=float(np.sqrt(np.mean(x[s] ** 2))))
    return out


def summary(rows):
    """Summarise one model sample while retaining per-peptide populations, times and errors."""
    keys = TRADE_KEYS + ("others_worst_rel_err",)
    return dict(
        median_model={k: median([r["model"][k] for r in rows]) for k in keys},
        median_halves={k: median([r["replicas"][k] for r in rows]) for k in keys},
        model_pass_all=sum(bool(r["model"]["pass_all"]) for r in rows),
        f_rms_at_or_below_halves=sum(r["model"]["f_rms_kT"] <= r["replicas"]["f_rms_kT"] for r in rows),
        mfpt_by_timescale=dict(model=mfpt_bins(rows, "model"), halves=mfpt_bins(rows, "replicas")),
        slowest_relaxation=[dict(name=r["name"], reference_ns=r["slowest_ref_ns"], model_ns=r["slowest_model_ns"],
                                 too_fast_factor=r["slowest_ref_ns"] / r["slowest_model_ns"]) for r in rows],
        per_peptide=[dict(name=r["name"], main_states=r["main"], pi_reference=r["pi_ref"], pi_model=r["pi_model"],
                          relaxation_ns_reference=r["ts_ref"], relaxation_ns_model=r["ts_model"],
                          model={k: r["model"][k] for k in keys + ("pass_all", "mfpt_pairs")},
                          halves={k: r["replicas"][k] for k in keys + ("pass_all", "mfpt_pairs")}) for r in rows])


def main():
    refs = [json.load(open(os.path.join(P.REFERENCES, f))) for f in sorted(os.listdir(P.REFERENCES))]
    c = costs()
    rows = {sd: held_out(sd) for sd in seeds()}
    samples = {sd: summary(r) for sd, r in rows.items()}
    pooled_rows = [r for sd in rows for r in rows[sd]]
    ls = lstar()
    from scipy.stats import spearmanr
    rho = spearmanr([r["lstar_us"][0] for r in ls], [r["halves_slowest_rel_err"] for r in ls])
    rho2 = spearmanr([r["lstar_us"][0] for r in ls], [r["halves_mfpt_worst_factor"] for r in ls])
    H = dict(test_peptides=len(refs), peptides_with_states=sum(bool(r["states_ok"]) for r in refs),
             held_out_judged=len(next(iter(samples.values()))["per_peptide"]), model_samples=samples,
             mfpt_by_timescale_all_samples=dict(model=mfpt_bins(pooled_rows, "model"),
                                                halves=mfpt_bins(rows[seeds()[0]], "replicas")),
             lstar=ls, lstar_spearman=dict(slowest=[rho.statistic, rho.pvalue], mfpt=[rho2.statistic, rho2.pvalue]),
             halves_pass_all=sum(r["halves_pass_all"] for r in ls),
             cost=c, trade=trade(c), oracle=oracle(), context_skill=context_skill())
    json.dump(H, open(os.path.join(P.RESULTS, "headline.json"), "w"), indent=1, default=float)
    print(f"held-out peptides judged: {H['held_out_judged']} (test peptides {H['test_peptides']}, "
          f"with states {H['peptides_with_states']}); model samples: {len(samples)} x 5 us")
    for sd, S in samples.items():
        m, e = S["median_model"], S["median_halves"]
        print(f"sample {sd}: median F RMS {m['f_rms_kT']:.2f} kT [halves {e['f_rms_kT']:.2f}], "
              f"worst MFPT {m['mfpt_worst_factor']:.2f} [{e['mfpt_worst_factor']:.2f}], TV {m['tv']:.3f} [{e['tv']:.3f}], "
              f"slowest {m['slowest_rel_err']:.0%} [{e['slowest_rel_err']:.0%}], others {m['others_worst_rel_err']:.0%} "
              f"[{e['others_worst_rel_err']:.0%}]; F at or below halves on {S['f_rms_at_or_below_halves']}/"
              f"{len(S['per_peptide'])}; every tolerance {S['model_pass_all']}/{len(S['per_peptide'])}")
    print("first-passage times within 1.5x, by reference MFPT, all samples pooled: model [halves]")
    for a, b in zip(H["mfpt_by_timescale_all_samples"]["model"], H["mfpt_by_timescale_all_samples"]["halves"]):
        lo, hi = a["ref_mfpt_ns"]
        lab = f"{lo}-{hi} ns" if hi else f"> {lo / 1000:g} us"
        print(f"  {lab:12s} {a['pairs']:3d} pairs  {a['within_1p5']:.0%} [{b['within_1p5']:.0%} of {b['pairs']}]")
    print("first-passage times within 1.5x per sample (3-10, 10-30, 30-100, 100-300, 300-1000, >1000 ns):")
    for sd, S in samples.items():
        print(f"  sample {sd}: " + ", ".join(f"{x['within_1p5']:.0%}" for x in S["mfpt_by_timescale"]["model"]))
    print("slowest relaxation, reference ns, then model ns per sample:")
    names = [r["name"] for r in next(iter(samples.values()))["slowest_relaxation"]]
    for k, n in enumerate(names):
        rs = [S["slowest_relaxation"][k] for S in samples.values()]
        print(f"  {n}  {rs[0]['reference_ns']:6.2f}  " + "  ".join(f"{r['model_ns']:6.2f} (x{r['too_fast_factor']:.2f})" for r in rs))
    for sd, S in samples.items():
        fast = [r["name"] for r in S["slowest_relaxation"] if r["too_fast_factor"] >= 2]
        print(f"  sample {sd}: at least 2x too fast: {len(fast)} ({', '.join(fast)})")
    print(f"reference halves meeting every tolerance: {H['halves_pass_all']}/{len(ls)}")
    lo = min(ls, key=lambda r: r["lstar_us"][0])
    hi = max(ls, key=lambda r: r["lstar_us"][1])
    print(f"reference length a fair rate test needs: {lo['lstar_us'][0]:.0f} us ({lo['name']}) to "
          f"{hi['lstar_us'][1] / 1000:.1f} ms ({hi['name']}); halves here 2.5 us; Spearman rho with halves' slowest "
          f"error {rho.statistic:.2f} (p {rho.pvalue:.2f}), with worst MFPT {rho2.statistic:.2f} (p {rho2.pvalue:.2f})")
    print(f"cost on one T4: MD {c['md_gpu_s_per_ns']:.1f} GPU-s/ns, model {c['model_gpu_s_per_ns']:.4f} GPU-s/ns")
    print("error vs GPU time (median; worst MFPT factor / F RMS kT / TV / slowest error):")
    for t in H["trade"]:
        f = lambda d: " / ".join(f"{d[k]:.3g}" for k in TRADE_KEYS)
        print(f"  {t['length_ns']:5d} ns  MD {t['md_gpu_s']:8.0f} GPU-s: {f(t['md'])}   "
              f"model {t['model_gpu_s']:6.1f} GPU-s: {f(t['model'])}")
    o = H["oracle"]
    print(f"MD free energies imposed on the model ({o['peptides']} peptides): F {o['model']['f_rms_kT']:.2f} -> "
          f"{o['thermo_oracle']['f_rms_kT']:.2f} kT, TV {o['model']['tv']:.3f} -> {o['thermo_oracle']['tv']:.3f}, "
          f"slowest {o['model']['slowest_rel_err']:.0%} -> {o['thermo_oracle']['slowest_rel_err']:.0%}, MFPT "
          f"{o['model']['mfpt_worst_factor']:.2f} -> {o['thermo_oracle']['mfpt_worst_factor']:.2f}; every tolerance "
          f"{o['oracle_pass_all']}/{o['peptides']}")
    for lab, s_ in H["context_skill"].items():
        print(f"context skill ({lab} transitions, n={s_['n']}): corr {s_['corr']:.2f}, slope {s_['slope']:.2f}, "
              f"RMS error model {s_['rms_model_error']:.2f} vs pooled rate {s_['rms_pooled_error']:.2f}")


if __name__ == "__main__":
    main()
