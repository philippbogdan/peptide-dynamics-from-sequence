#!/usr/bin/env python3
"""Every number quoted in the README and in docs/evidence.md, recomputed from the files in results/.

    python tools/recompute.py              each number, with the file it comes from
    python tools/recompute.py --markdown   the tables of docs/evidence.md

tests/test_numbers.py fails if the README or docs/evidence.md quote anything else.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the 13 test peptides on whose MD variants of the model were compared: where they are scored, optimistically
DEVELOPMENT = ["ARIP", "CCVH", "CIPQ", "DMTL", "FESD", "GCDE", "GGRS", "HEAV", "NCFG", "SVND", "TAPF", "VPFY", "WNMA"]
BETWEEN_STATES = ("mfpt_worst_factor", "f_rms_kT")       # errors that need at least two main states
RECORDING = "DMTL"                                        # the peptide drawn in figures/recording.png
KEYS = ("mfpt_worst_factor", "f_rms_kT", "tv", "slowest_rel_err", "others_worst_rel_err")
SETS = {"all": ("development", "other test", "validation"), "compared on": ("development",), "validation": ("validation",)}
NAMES = dict(f_rms_kT="free-energy error", mfpt_worst_factor="worst passage-time factor", slowest_rel_err="slowest relaxation error")


RECORD = ""                                               # "seed1/": the results on the states another k-means seed gives


def load(name):
    name = RECORD + name if name in ("held_out.json", "trade.json") else name
    return json.load(open(os.path.join(HERE, "results", name)))


def by_peptide(runs):
    out = {}
    for run in runs:
        for r in run["peptides"]:
            out.setdefault(r["name"], []).append(r)
    return out


def group(name, split):
    return "development" if name in DEVELOPMENT else "other test" if split == "test" else "validation"


def counts(r, key):
    """A peptide with a single main state has no passage time and no free-energy difference to be wrong about."""
    return key not in BETWEEN_STATES or len(r["main_states"]) >= 2


def typical(key, groups, who="model"):
    """The median over peptides within one evaluation; for the model, its median, minimum and maximum over the
    seeds (the reference has one realisation, so its halves give one value)."""
    runs = load("held_out.json")["runs"]
    vals = [float(np.median([r[who][key] for r in run["peptides"] if group(r["name"], r["split"]) in groups and counts(r, key)]))
            for run in runs]
    return (float(np.median(vals)), min(vals), max(vals)) if who == "model" else vals[0]


def per_peptide():
    """Each scored peptide: the median over seeds of the model's errors, the errors between the MD's halves."""
    rows = {}
    for name, rs in by_peptide(load("held_out.json")["runs"]).items():
        rows[name] = dict(group=group(name, rs[0]["split"]), main_states=len(rs[0]["main_states"]),
                          model={k: float(np.median([r["model"][k] for r in rs])) for k in KEYS},
                          halves={k: rs[0]["halves"][k] for k in KEYS},
                          slowest_md=rs[0]["relaxation_ns"]["md"][0],
                          slowest_model=float(np.median([r["relaxation_ns"]["model"][0] for r in rs])),
                          drawn_again=float(np.median([r["overlaps"]["drawn_again"] for r in rs])),
                          left_overlapping=float(np.median([r["overlaps"]["left_overlapping"] for r in rs])),
                          halves_pass=rs[0]["halves"]["pass_all"], controls_fail=rs[0]["shuffled_fails"] and rs[0]["doubled_clock_fails"])
    return rows


def passage_share(groups, lo, hi):
    """Share of passage times with a reference value in [lo, hi) ns that are within a factor 1.5: for the model
    (each passage time the median over seeds) and between the MD's halves."""
    model, halves = [], []
    for name, rs in by_peptide(load("held_out.json")["runs"]).items():
        if group(name, rs[0]["split"]) not in groups:
            continue
        pairs = np.array([[p[2:4] for p in r["model"]["mfpt_pairs"]] for r in rs])
        if pairs.size:
            ref, mo = pairs[0, :, 1], np.median(pairs[:, :, 0], 0)
            sel = (ref >= lo) & (ref < hi)
            model += list(np.maximum(mo[sel] / ref[sel], ref[sel] / mo[sel]) <= 1.5)
            h = np.array([p[2:4] for p in rs[0]["halves"]["mfpt_pairs"]])
            sel = (h[:, 1] >= lo) & (h[:, 1] < hi)
            halves += list(np.maximum(h[sel, 0] / h[sel, 1], h[sel, 1] / h[sel, 0]) <= 1.5)
    return float(np.mean(model)), float(np.mean(halves)), len(model)


def slow():
    """Peptides whose slowest relaxation takes more than 5 ns in MD: how many the model makes faster, by what factor."""
    rows = [r for r in per_peptide().values() if r["slowest_md"] > 5.0]
    faster = [r["slowest_md"] / r["slowest_model"] for r in rows if r["slowest_model"] < r["slowest_md"]]
    return len(rows), len(faster), float(np.median(faster))


HALF_NS = 2495.0                                          # one half of a 5 microsecond reference, after the first 10 ns
SLOW_SITE_PER_NS = 0.1                                    # a site that jumps less than once in 10 ns


def sites(name="sites.json"):
    """Every site that jumps at least ten times in each half of the MD: the factor between its jump rate in the
    model and in the MD, and between the two halves of the MD. Per peptide: the root-mean-square error of the
    free energies of its sites' states."""
    rows = [v[0] for v in by_peptide(load(name)["runs"]).values()]
    md, model, halves, split, faster = [], [], [], [], []
    for r in rows:
        j = r["jumps_per_ns"]
        m, a, b, mo = (np.array(j[k]) for k in ("md", "half_a", "half_b", "model"))
        ok = (np.minimum(a, b) * HALF_NS >= 10 - 1e-6) & (mo > 0)
        md += list(m[ok])
        model += list(np.abs(np.log(mo[ok] / m[ok])))
        halves += list(np.abs(np.log(a[ok] / b[ok])))
        faster += list(mo[ok] > m[ok])
        split += [r["split"]] * int(ok.sum())
    md, model, halves, split, faster = np.array(md), np.array(model), np.array(halves), np.array(split), np.array(faster)
    off = lambda x: float(np.exp(np.median(x))) - 1       # the median factor between two rates, as a fraction off
    out = dict(peptides=len(rows), sites=len(md), slowest_per_ns=float(md.min()), fastest_per_ns=float(md.max()),
               rate_model=off(model), rate_halves=off(halves),
               within_1p5_model=float(np.mean(model <= np.log(1.5))), within_1p5_halves=float(np.mean(halves <= np.log(1.5))),
               free_energy_model=float(np.median([r["model_f_rms_kT"] for r in rows])),
               free_energy_halves=float(np.median([r["halves_f_rms_kT"] for r in rows])))
    slow = md < SLOW_SITE_PER_NS
    if slow.any():
        out.update(slow_sites=int(slow.sum()), slow_rate_model=off(model[slow]), slow_rate_halves=off(halves[slow]),
                   slow_faster=int(faster[slow].sum()))
    for name_ in ("test", "validation"):
        sel = split == name_
        if sel.any():
            out[name_] = dict(sites=int(sel.sum()), rate_model=off(model[sel]), rate_halves=off(halves[sel]),
                              free_energy_model=float(np.median([r["model_f_rms_kT"] for r in rows if r["split"] == name_])),
                              free_energy_halves=float(np.median([r["halves_f_rms_kT"] for r in rows if r["split"] == name_])))
    return out


def trade(groups):
    """Against half B of the reference, by simulated length: the median over peptides of each error, for the first
    part of half A and for the model (each peptide's median over seeds)."""
    runs = load("trade.json")["runs"]
    lengths = sorted(int(k) for k in runs[0]["peptides"][0]["md"])
    several = {name for name, r in per_peptide().items() if r["main_states"] >= 2}
    out = dict(lengths=lengths, md={}, model={})
    for key in KEYS:
        names = [r["name"] for r in runs[0]["peptides"] if group(r["name"], r["split"]) in groups
                 and (key not in BETWEEN_STATES or r["name"] in several)]
        out["md"][key] = [float(np.median([r["md"][str(L)][key] for r in runs[0]["peptides"] if r["name"] in names])) for L in lengths]
        per = {}
        for run in runs:
            for r in run["peptides"]:
                if r["name"] in names:
                    per.setdefault(r["name"], []).append([r["model"][str(L)][key] for L in lengths])
        out["model"][key] = [float(x) for x in np.median([np.median(v, 0) for v in per.values()], 0)]
    return out


def equal_length(groups, ns=2490):
    """The like-for-like comparison: ns of the model, and the first ns of one half of the MD, each against the
    other half. Medians over peptides (for the model, of each peptide's median over seeds)."""
    t = trade(groups)
    i = t["lengths"].index(ns)
    return {key: (t["model"][key][i], t["md"][key][i]) for key in KEYS}


def cost():
    """GPU seconds on one Tesla T4: MD per nanosecond at its best (eight simulations sharing the card), with the
    corpus's 1 fs steps and with 4 fs steps, and the model for 1 microsecond (median over the scored peptides)."""
    best = {}
    for r in load("cost/md_t4.json")["runs"]:
        b = best.setdefault(r["protocol"], {})
        b[r["peptide"]] = max(b.get(r["peptide"], 0), r["aggregate_ns_per_day"])
    md = {p: 86400.0 / float(np.mean(list(v.values()))) for p, v in best.items()}
    m = load("cost/model_t4.json")["peptides"]
    total = np.array([p["1000 ns"]["total_s"] for p in m])
    return dict(md_s_per_ns=md["corpus"], md_4fs_s_per_ns=md["hmr"], speed_up_4fs=md["corpus"] / md["hmr"],
                model_s_per_us=float(np.median(total)), model_s_per_us_min=float(total.min()), model_s_per_us_max=float(total.max()),
                model_card_s_per_us=float(np.median([p["1000 ns"]["dynamics_s"] for p in m])))


def matched(groups):
    """The estimate of GPU time at matched passage-time error: the error the model reaches with 1 microsecond of
    trajectories, the length of MD whose error first falls to it (interpolated between the lengths scored, on
    logarithmic axes), and the ratio of the two GPU times on one Tesla T4."""
    t, c = trade(groups), cost()
    L = np.array(t["lengths"], float)
    md, model = np.array(t["md"]["mfpt_worst_factor"]), np.array(t["model"]["mfpt_worst_factor"])
    target = float(model[list(L).index(1000.0)])
    k = int(np.argmax(md <= target))                      # first scored length at which MD is at least as good
    frac = (np.log(md[k - 1]) - np.log(target)) / (np.log(md[k - 1]) - np.log(md[k]))
    md_ns = float(np.exp(np.log(L[k - 1]) + frac * (np.log(L[k]) - np.log(L[k - 1]))))
    md_s = md_ns * c["md_s_per_ns"]
    return dict(model_error=target, md_ns=md_ns, md_gpu_hours=md_s / 3600.0, model_gpu_s=c["model_s_per_us"],
                ratio=md_s / c["model_s_per_us"], ratio_4fs=md_s / c["speed_up_4fs"] / c["model_s_per_us"])


def other_seed():
    """The state rule run with another seed for its k-means (data/reference_seed1): how the verdicts move, and
    the comparison at equal length and the estimate of cost on those states (results/seed1/)."""
    global RECORD
    first, second = ({os.path.basename(f)[:-4]: np.load(f) for f in sorted(glob.glob(os.path.join(HERE, "data", d, "*.npz")))}
                     for d in ("reference", "reference_seed1"))
    has = lambda z: bool(z["states_ok"])
    keep = [n for n in first if has(first[n]) and has(second[n])]
    out = dict(with_states=sum(has(z) for z in second.values()),
               gain=sum(has(second[n]) and not has(first[n]) for n in first),
               lose=sum(has(first[n]) and not has(second[n]) for n in first), keep=len(keep),
               other_number=sum(int(first[n]["n_states"]) != int(second[n]["n_states"]) for n in keep))
    RECORD = "seed1/"
    try:
        out.update(equal=equal_length(SETS["all"]), matched=matched(SETS["all"]))
    finally:
        RECORD = ""
    return out


def summary():
    """name -> (value, formatted as quoted, source file)"""
    pct = lambda x: f"{100 * x:.0f}%"
    out = {}
    rows = per_peptide()
    for label, groups in SETS.items():
        n = sum(r["group"] in groups for r in rows.values())
        out[f"{label}: peptides"] = (n, str(n), "held_out.json")
        for key, unit in (("f_rms_kT", " kT"), ("mfpt_worst_factor", ""), ("slowest_rel_err", "%")):
            mid, lo, hi = typical(key, groups)
            h = typical(key, groups, "halves")
            f = (lambda x: pct(x)) if unit == "%" else (lambda x: f"{x:.2f}{unit}")
            out[f"{label}: {NAMES[key]}, model"] = (mid, f(mid), "held_out.json")
            out[f"{label}: {NAMES[key]}, model, least over seeds"] = (lo, f(lo), "held_out.json")
            out[f"{label}: {NAMES[key]}, model, most over seeds"] = (hi, f(hi), "held_out.json")
            out[f"{label}: {NAMES[key]}, MD halves"] = (h, f(h), "held_out.json")
        for key, (mo, md) in equal_length(groups).items():
            if key in NAMES:
                f = (lambda x: pct(x)) if key == "slowest_rel_err" else (lambda x, key=key: f"{x:.2f}" + (" kT" if key == "f_rms_kT" else ""))
                out[f"{label}: at equal length, {NAMES[key]}, model"] = (mo, f(mo), "trade.json")
                out[f"{label}: at equal length, {NAMES[key]}, MD"] = (md, f(md), "trade.json")
        for name, lo, hi in (("up to 100 ns", 0.0, 100.0), ("above 100 ns", 100.0, 1e9)):
            m, h, n_pairs = passage_share(groups, lo, hi)
            out[f"{label}: passage times {name}, number"] = (n_pairs, str(n_pairs), "held_out.json")
            out[f"{label}: passage times {name} within 1.5, model"] = (m, pct(m), "held_out.json")
            out[f"{label}: passage times {name} within 1.5, MD halves"] = (h, pct(h), "held_out.json")
    several = {k: r for k, r in rows.items() if r["main_states"] >= 2}
    out["all: peptides with at least two main states"] = (len(several), str(len(several)), "held_out.json")
    for key in BETWEEN_STATES:                             # the peptide of the README's first figure
        place = 1 + sorted(several, key=lambda k: several[k]["model"][key]).index(RECORDING)
        out[f"recording: {RECORDING}, place by {NAMES[key]}, least first"] = (place, str(place), "held_out.json")
    n, faster, factor = slow()
    out["slow: peptides slower than 5 ns in MD"] = (n, str(n), "held_out.json")
    out["slow: of which the model is faster"] = (faster, str(faster), "held_out.json")
    out["slow: by a median factor"] = (factor, f"{factor:.1f}", "held_out.json")
    too_fast = sum(r["slowest_model"] < r["slowest_md"] for r in rows.values())
    out["slow: peptides whose slowest relaxation is faster in the model"] = (too_fast, str(too_fast), "held_out.json")
    passed = sum(r["halves_pass"] for r in rows.values())
    failed = sum(r["controls_fail"] for r in rows.values())
    out["reference: peptides whose halves meet every tolerance"] = (passed, str(passed), "held_out.json")
    out["reference: peptides on which both controls fail"] = (failed, str(failed), "held_out.json")
    left = [r["left_overlapping"] for r in rows.values()]
    again = [r["drawn_again"] for r in rows.values()]
    out["atoms: frames drawn again, least"] = (min(again), pct(min(again)), "held_out.json")
    out["atoms: frames drawn again, most"] = (max(again), pct(max(again)), "held_out.json")
    out["atoms: frames left overlapping, most"] = (max(left), f"{100 * max(left):.1f}%", "held_out.json")
    a = load("atoms.json")["peptides"]
    out["atoms: rebuilt MD frames, RMSD from"] = (min(p["rmsd_median"] for p in a), f"{min(p['rmsd_median'] for p in a):.2f}", "atoms.json")
    out["atoms: rebuilt MD frames, RMSD to"] = (max(p["rmsd_median"] for p in a), f"{max(p['rmsd_median'] for p in a):.2f}", "atoms.json")
    b = load("bank.json")
    out["atoms: frames that borrow another combination's torsions"] = (b["share_of_frames_that_borrow"], f"{100 * b['share_of_frames_that_borrow']:.2f}%", "bank.json")
    out["atoms: the same, most for one peptide"] = (b["largest_share_for_one_peptide"], f"{100 * b['largest_share_for_one_peptide']:.1f}%", "bank.json")
    for name, label in (("sites.json", "sites"), ("sites_octapeptides.json", "octapeptides")):
        if not os.path.exists(os.path.join(HERE, "results", name)):
            continue
        s = sites(name)
        kT = lambda x: f"{x:.2f} kT"
        out[f"{label}: peptides"] = (s["peptides"], str(s["peptides"]), name)
        out[f"{label}: sites"] = (s["sites"], str(s["sites"]), name)
        out[f"{label}: jump rate, model"] = (s["rate_model"], pct(s["rate_model"]), name)
        out[f"{label}: jump rate, MD halves"] = (s["rate_halves"], pct(s["rate_halves"]), name)
        out[f"{label}: jump rates within 1.5, model"] = (s["within_1p5_model"], pct(s["within_1p5_model"]), name)
        out[f"{label}: jump rates within 1.5, MD halves"] = (s["within_1p5_halves"], pct(s["within_1p5_halves"]), name)
        out[f"{label}: site free energy, model"] = (s["free_energy_model"], kT(s["free_energy_model"]), name)
        out[f"{label}: site free energy, MD halves"] = (s["free_energy_halves"], kT(s["free_energy_halves"]), name)
        if label == "octapeptides":
            out["octapeptides: sites that jump less than once in 10 ns"] = (s["slow_sites"], str(s["slow_sites"]), name)
            out["octapeptides: their jump rate, model, a factor"] = (s["slow_rate_model"] + 1, f"{s['slow_rate_model'] + 1:.0f}", name)
            out["octapeptides: their jump rate, MD halves"] = (s["slow_rate_halves"], pct(s["slow_rate_halves"]), name)
        if label == "sites":
            out["sites: slowest site, jumps per ns"] = (s["slowest_per_ns"], f"{s['slowest_per_ns']:.3f}", name)
            out["sites: fastest site, jumps per ns"] = (s["fastest_per_ns"], f"{s['fastest_per_ns']:.0f}", name)
            out["sites: sites that jump less than once in 10 ns"] = (s["slow_sites"], str(s["slow_sites"]), name)
            out["sites: their jump rate, model"] = (s["slow_rate_model"], pct(s["slow_rate_model"]), name)
            out["sites: their jump rate, MD halves"] = (s["slow_rate_halves"], pct(s["slow_rate_halves"]), name)
            out["sites: of them, faster in the model than in the MD"] = (s["slow_faster"], str(s["slow_faster"]), name)
            v = s["validation"]
            out["sites: validation, jump rate, model"] = (v["rate_model"], pct(v["rate_model"]), name)
            out["sites: validation, jump rate, MD halves"] = (v["rate_halves"], pct(v["rate_halves"]), name)
            out["sites: validation, site free energy, model"] = (v["free_energy_model"], kT(v["free_energy_model"]), name)
            out["sites: validation, site free energy, MD halves"] = (v["free_energy_halves"], kT(v["free_energy_halves"]), name)
    c = cost()
    out["cost: MD, GPU seconds per ns"] = (c["md_s_per_ns"], f"{c['md_s_per_ns']:.0f}", "cost/md_t4.json")
    out["cost: MD, GPU hours per microsecond"] = (c["md_s_per_ns"] / 3.6, f"{c['md_s_per_ns'] / 3.6:.1f}", "cost/md_t4.json")
    out["cost: MD with 4 fs steps, times cheaper"] = (c["speed_up_4fs"], f"{c['speed_up_4fs']:.1f}", "cost/md_t4.json")
    out["cost: model, GPU seconds per microsecond"] = (c["model_s_per_us"], f"{c['model_s_per_us']:.0f}", "cost/model_t4.json")
    out["cost: model, GPU seconds per microsecond, least"] = (c["model_s_per_us_min"], f"{c['model_s_per_us_min']:.0f}", "cost/model_t4.json")
    out["cost: model, GPU seconds per microsecond, most"] = (c["model_s_per_us_max"], f"{c['model_s_per_us_max']:.0f}", "cost/model_t4.json")
    for label in ("all", "compared on"):
        m = matched(SETS[label])
        out[f"cost, {label}: model's passage-time error with 1 microsecond"] = (m["model_error"], f"{m['model_error']:.1f}", "trade.json")
        out[f"cost, {label}: MD needed to match, microseconds"] = (m["md_ns"] / 1000, f"{m['md_ns'] / 1000:.1f}", "trade.json")
        out[f"cost, {label}: MD needed to match, GPU hours"] = (m["md_gpu_hours"], f"{m['md_gpu_hours']:.1f}", "trade.json, cost/")
        out[f"cost, {label}: ratio of GPU times"] = (m["ratio"], f"{round(m['ratio'], -2):,.0f}", "trade.json, cost/")
    o = other_seed()
    for name, key in (("peptides with states", "with_states"), ("peptides that gain states", "gain"), ("peptides that lose them", "lose"),
                      ("peptides that keep them", "keep"), ("of those, with another number of states", "other_number")):
        out[f"other seed: {name}"] = (o[key], str(o[key]), "../data/reference_seed1/")
    for key, (mo, md) in o["equal"].items():
        if key in NAMES:
            f = (lambda x: pct(x)) if key == "slowest_rel_err" else (lambda x, key=key: f"{x:.2f}" + (" kT" if key == "f_rms_kT" else ""))
            out[f"other seed: at equal length, {NAMES[key]}, model"] = (mo, f(mo), "seed1/trade.json")
            out[f"other seed: at equal length, {NAMES[key]}, MD"] = (md, f(md), "seed1/trade.json")
    m = o["matched"]
    out["other seed: MD needed to match, microseconds"] = (m["md_ns"] / 1000, f"{m['md_ns'] / 1000:.1f}", "seed1/trade.json")
    out["other seed: ratio of GPU times"] = (m["ratio"], f"{round(m['ratio'], -2):,.0f}", "seed1/trade.json, cost/")
    t = load("timing.json")
    out["timing: seconds per microsecond on one CPU core, median"] = (t["median_seconds"], f"{t['median_seconds']:.0f}", "timing.json")
    out["timing: seconds per microsecond, fastest"] = (t["fastest_seconds"], f"{t['fastest_seconds']:.1f}", "timing.json")
    out["timing: seconds per microsecond, slowest"] = (t["slowest_seconds"], f"{t['slowest_seconds']:.0f}", "timing.json")
    return out


def peptide_table():
    label = {"development": "test, compared on", "other test": "test", "validation": "validation"}
    rows = ["| peptide | set | worst passage time | free energy (kT) | populations | slowest relaxation | other relaxations | slowest relaxation, model / MD (ns) |",
            "|---|---|---|---|---|---|---|---|"]
    two = lambda m, h: f"{m:.2f} / {h:.2f}"
    between = lambda r, k: two(r["model"][k], r["halves"][k]) if r["main_states"] >= 2 else "one main state"
    pc = lambda m, h: f"{100 * m:.0f}% / {100 * h:.0f}%"
    order = ["development", "other test", "validation"]
    for name, r in sorted(per_peptide().items(), key=lambda kv: (order.index(kv[1]["group"]), kv[0])):
        m, h = r["model"], r["halves"]
        rows.append(f"| {name} | {label[r['group']]} | {between(r, 'mfpt_worst_factor')} | {between(r, 'f_rms_kT')} | "
                    f"{two(m['tv'], h['tv'])} | {pc(m['slowest_rel_err'], h['slowest_rel_err'])} | "
                    f"{pc(m['others_worst_rel_err'], h['others_worst_rel_err'])} | {r['slowest_model']:.1f} / {r['slowest_md']:.1f} |")
    return "\n".join(rows)


def number_table():
    rows = ["| number | value | from `results/` |", "|---|---|---|"]
    rows += [f"| {name} | {text} | `{source}` |" for name, (_, text, source) in summary().items()]
    return "\n".join(rows)


if __name__ == "__main__":
    if "--markdown" in sys.argv:
        print(number_table() + "\n\n" + peptide_table())
    else:
        for name, (value, text, source) in summary().items():
            print(f"{name:75s} {text:>10s}   results/{source}")
