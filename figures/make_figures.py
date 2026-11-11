#!/usr/bin/env python3
"""Draw the figures of the README from the files in this repository.

    python figures/make_figures.py            all figures
    python figures/make_figures.py recording  one of: recording, evidence, cost, method

Needs numpy and matplotlib.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pepdyn                                   # noqa: E402

INK, MODEL, QUIET, FRESH = "#1f2328", "#0550ae", "#b7bdc5", "#d9480f"
BASINS = ["#2f6fd0", "#e2703a", "#3a9e5f", "#8a63d2", "#c69026"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200, "savefig.dpi": 200})
WIDTH = 8.38                                    # inches: 1676 px, twice the width GitHub shows a README image at
NAMES = {"phi": "φ", "psi": "ψ", "chi1": "χ1", "chi2": "χ2", "chi3": "χ3", "chi4": "χ4"}
ONE = {v: k for k, v in pepdyn.model.ONE_TO_THREE.items()}


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def recording(peptide="DMTL", ns=200.0):
    """Every torsion angle of one held-out peptide against time: MD beside the model."""
    z = np.load(os.path.join(ROOT, "data", "traces", peptide + ".npz"))
    md = z["torsions"].astype(float)
    md[md == -32768] = np.nan
    n = int(ns * 100)
    md = md[:n]
    out = pepdyn.generate(peptide, ns=ns, seed=0)
    pep = out["peptide"]
    mo = np.degrees(out["torsions"][0])
    chans = [(i, c) for i in range(len(pep.residues)) for c in range(6) if np.isfinite(md[0, i, c])]
    fig = plt.figure(figsize=(WIDTH, 0.42 * len(chans) + 0.75))
    left, right, top, bottom, gap = 0.135, 0.012, 0.05, 0.078, 0.018
    w = (1 - left - right - gap) / 2
    h = (1 - top - bottom) / len(chans)
    for r, (i, c) in enumerate(chans):
        # cut the circle where MD is rarest, so that no basin straddles the edge of the panel
        hist = np.histogram(np.mod(md[:, i, c], 360), bins=72, range=(0, 360))[0]
        smooth = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(7), "valid")
        cut = 5.0 * np.argmin(smooth) + 2.5
        for col, (a, colour) in enumerate(((md, INK), (mo, MODEL))):
            ax = fig.add_axes([left + col * (w + gap), 1 - top - (r + 1) * h + 0.0035, w, h - 0.007])
            y = np.mod(a[:, i, c] - cut, 360)
            img = np.histogram2d(np.arange(n) * 0.01, y, bins=(int(n / 27), 60), range=((0, ns), (0, 360)))[0].T
            rgb = np.array(matplotlib.colors.to_rgb(colour))
            alpha = np.clip(img / np.percentile(img[img > 0], 92), 0, 1) ** 0.6
            ax.imshow(1 - alpha[..., None] * (1 - rgb), origin="lower", extent=(0, ns, 0, 360), aspect="auto",
                      interpolation="bilinear")
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_facecolor("#f6f8fa")
            if r == len(chans) - 1:
                ax.set_xticks(np.arange(0, ns + 1, 50))
                ax.set_xlabel("time (ns)")
                ax.tick_params(length=2.5, pad=2)
                labels = ax.get_xticklabels()
                labels[0].set_ha("left")
                labels[-1].set_ha("right")
            else:
                ax.set_xticks([])
            if r == 0:
                ax.set_title("Molecular dynamics" if col == 0 else f"The model, from the letters {' '.join(peptide)}",
                             fontsize=10, fontweight="bold", color=colour, pad=6)
        fig.text(left - 0.012, 1 - top - (r + 0.5) * h, NAMES[("phi", "psi", "chi1", "chi2", "chi3", "chi4")[c]],
                 ha="right", va="center", fontsize=9)
    # the four letters down the left edge, each beside its own torsions
    for i, rn in enumerate(pep.residues):
        rows = [r for r, (j, _) in enumerate(chans) if j == i]
        y0, y1 = 1 - top - (rows[-1] + 1) * h + 0.006, 1 - top - rows[0] * h - 0.006
        fig.add_artist(plt.Line2D([0.058, 0.058], [y0, y1], color=INK, lw=1.2))
        fig.text(0.03, (y0 + y1) / 2, ONE[rn], ha="center", va="center", fontsize=15, fontweight="bold")
    save(fig, "recording.png")


def load(name):
    return json.load(open(os.path.join(ROOT, "results", name)))


def by_peptide(runs):
    """{peptide: [its record in each seed]}"""
    out = {}
    for run in runs:
        for r in run["peptides"]:
            out.setdefault(r["name"], []).append(r)
    return out


def loglog(ax, lo, hi, label_x, label_y, band=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, zorder=1)
    if band:
        ax.fill_between([lo, hi], [lo / band, hi / band], [lo * band, hi * band], color="#eaeef2", zorder=0, lw=0)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect("equal")
    ax.tick_params(length=2.5, pad=2)
    plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)


def evidence():
    """From one residue's jumps to the peptide's slowest process: the model against MD, and MD against itself."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import recompute
    sites = by_peptide(load("sites.json")["runs"])
    held = by_peptide(load("held_out.json")["runs"])
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.72))
    fig.subplots_adjust(left=0.062, right=0.995, top=0.87, bottom=0.265, wspace=0.4)
    dot = dict(s=7, lw=0, zorder=3)
    quiet = dict(s=7, lw=0, color=QUIET, zorder=2)
    colour = lambda r: FRESH if r["split"] == "validation" else MODEL
    title = dict(fontsize=9, fontweight="bold", loc="left", pad=5)

    ax = axes[0]                                              # every site's jump rate
    loglog(ax, 2e-3, 2e2, "MD (jumps per ns)", "model (jumps per ns)")
    ax.set_xticks([0.01, 1, 100])
    ax.set_yticks([0.01, 1, 100])
    for name, rs in sorted(sites.items(), key=lambda kv: kv[1][0]["split"]):
        j = rs[0]["jumps_per_ns"]
        md, a, b, mo = (np.array(j[k]) for k in ("md", "half_a", "half_b", "model"))
        ok = (np.minimum(a, b) * recompute.HALF_NS >= 10 - 1e-6) & (mo > 0)      # at least ten jumps in each half of the MD
        ax.scatter(a[ok], b[ok], **quiet)
        ax.scatter(md[ok], mo[ok], color=colour(rs[0]), **dot)
    ax.set_title("each residue's jumps", **title)

    ax = axes[1]                                              # free energy of every main state
    ax.plot([0, 5], [0, 5], color=INK, lw=0.6, zorder=1)
    for name, rs in sorted(held.items(), key=lambda kv: kv[1][0]["split"]):
        main = rs[0]["main_states"]
        f = lambda key: -np.log(np.maximum(np.median([np.array(r["populations"][key])[main] for r in rs], 0), 1e-6))
        ax.scatter(f("half_a"), f("half_b"), **quiet)
        ax.scatter(f("md"), f("model"), color=colour(rs[0]), **dot)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_aspect("equal")
    ax.set_xlabel("MD (kT)")
    ax.set_ylabel("model (kT)")
    ax.tick_params(length=2.5, pad=2)
    ax.set_title("each state's free energy", **title)

    ax = axes[2]                                              # every passage time
    loglog(ax, 1, 3e3, "MD (ns)", "model (ns)", band=1.5)
    for name, rs in sorted(held.items(), key=lambda kv: kv[1][0]["split"]):
        pairs = np.array([[p[2:4] for p in r["model"]["mfpt_pairs"]] for r in rs])          # seeds, pairs, (model, MD)
        if pairs.size:
            h = np.array([p[2:4] for p in rs[0]["halves"]["mfpt_pairs"]])
            ax.scatter(h[:, 1], h[:, 0], **quiet)
            ax.scatter(pairs[0, :, 1], np.median(pairs[:, :, 0], 0), color=colour(rs[0]), **dot)
    ax.set_title("each passage time", **title)

    ax = axes[3]                                              # the slowest relaxation of each peptide
    loglog(ax, 0.4, 60, "MD (ns)", "model (ns)")
    for name, rs in sorted(held.items(), key=lambda kv: kv[1][0]["split"]):
        t = lambda key: np.median([r["relaxation_ns"][key][0] for r in rs])
        ax.scatter(t("half_a"), t("half_b"), **dict(quiet, s=13))
        ax.scatter(t("md"), t("model"), color=colour(rs[0]), **dict(dot, s=13))
    ax.set_title("each slowest process", **title)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=c) for c in (MODEL, FRESH, QUIET)]
    fig.legend(handles, ["model against MD, test peptides", "model against MD, validation peptides",
                         "one half of the MD against the other"], loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, handletextpad=0.2, columnspacing=2.2, bbox_to_anchor=(0.5, -0.02))
    save(fig, "evidence.png")


def cost():
    """Passage-time error against GPU time on one Tesla T4, for MD and for the model: the median over the scored
    peptides of the worst passage-time factor against one half of the MD."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import recompute
    groups = recompute.SETS["all"]
    t, c, m = recompute.trade(groups), recompute.cost(), recompute.matched(groups)
    L = np.array(t["lengths"], float)
    timed = load("cost/model_t4.json")["peptides"]
    t1, t2 = (float(np.median([p[k]["total_s"] for p in timed])) for k in ("1000 ns", "2490 ns"))
    seconds = {"md": L * c["md_s_per_ns"], "model": t1 + (L - 1000.0) * (t2 - t1) / 1490.0}      # the model: a start-up cost, then linear
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.17)
    for who, colour, label in (("md", INK, "MD"), ("model", MODEL, "the model")):
        y = np.array(t[who]["mfpt_worst_factor"])
        ok = np.isfinite(y) & (y < 6)
        ax.plot(seconds[who][ok], y[ok], "o-", color=colour, ms=3.5, lw=1.3)
        ax.text(seconds[who][ok][0], y[ok][0] * 1.08, label, color=colour, fontweight="bold", ha="center", va="bottom", fontsize=9.5)
    x0, x1, y = c["model_s_per_us"], m["md_ns"] * c["md_s_per_ns"], m["model_error"]
    ax.annotate("", xy=(x1 * 0.93, y), xytext=(x0 * 1.12, y), arrowprops=dict(arrowstyle="<->", color=QUIET, lw=1.1))
    ax.text(np.sqrt(x0 * x1), y * 1.035, f"{round(m['ratio'], -2):,.0f} times less GPU time for the same error", ha="center",
            va="bottom", fontsize=8.5, color="#57606a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(4, 9e4)
    ax.set_ylim(1.4, 6.5)
    ax.set_yticks([1.5, 2, 3, 5])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["10 s", "100 s", "1,000 s", "10,000 s"])
    ax.set_xlabel("GPU time per peptide on one Tesla T4")
    ax.set_ylabel("worst passage-time error (factor)")
    ax.tick_params(length=2.5, pad=2)
    save(fig, "cost.png")


def _basin_image(g):
    """A site's basins as an image: each basin in its colour, strong where the training MD is, white where it is not."""
    lab, F = g["lab"], g["F"]
    p = np.exp(-(F - F.min()))
    alpha = np.clip(p / np.percentile(p, 97), 0, 1) ** 0.45
    rgb = np.array([matplotlib.colors.to_rgb(c) for c in BASINS])[lab]
    return 1 - alpha[..., None] * (1 - rgb)


def method(peptide="DMTL", ns=12.0):
    """How the model works, drawn with its own states and one of its own trajectories."""
    sys.path.insert(0, HERE)
    import draw_molecule as dm
    mod = pepdyn.load_model()
    out = pepdyn.generate(peptide, ns=ns, seed=1, model=mod)
    pep, top = out["peptide"], out["topology"]
    H = 3.05
    fig = plt.figure(figsize=(WIDTH, H))
    head = dict(fontsize=10.5, fontweight="bold", va="bottom")
    grey = "#57606a"
    kinds = ["bb", "chi1", "chi2", "chi3", "chi4"]
    kinds = [k for k in kinds if any(kind == k for _, kind in pep.sites)]
    # 1: the states of each residue, a column per residue and a row per kind of site
    x0, col_w, top1 = 0.085, 0.076, 0.74
    fig.text(0.012, 0.915, "1  Each residue has a few states", **head)
    row_h = [0.25] + [0.145] * (len(kinds) - 1)
    for r, kind in enumerate(kinds):
        y_mid = top1 - sum(row_h[:r]) - row_h[r] / 2
        fig.text(x0 - 0.008, y_mid, "backbone" if kind == "bb" else NAMES[kind], ha="right", va="center", fontsize=8.5, color=grey)
        for i, rn in enumerate(pep.residues):
            key = [k for (j, kd), k in zip(pep.sites, pep.keys) if j == i and kd == kind]
            if not key:
                continue
            g = mod.grid[key[0]]
            size = 0.066 if kind == "bb" else 0.04                 # as a fraction of the figure width
            ax = fig.add_axes([x0 + (i + 0.5) * col_w - size / 2, y_mid - size * WIDTH / H / 2, size, size * WIDTH / H])
            if g["lab"].ndim == 2:                                 # (phi, psi) map: phi across, psi up, both from -180
                img = np.roll(np.roll(_basin_image(g), 18, 0), 18, 1).transpose(1, 0, 2)
                ax.imshow(img, origin="lower", extent=(-180, 180, -180, 180), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(0.5)
                    sp.set_color(QUIET)
            else:                                                  # one torsion: a ring
                img = _basin_image(g)
                for b in range(72):
                    ax.add_patch(matplotlib.patches.Wedge((0, 0), 1.0, b * 5, b * 5 + 5.3, width=0.45, color=img[b], lw=0))
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
                ax.set_aspect("equal")
                ax.axis("off")
    for i, rn in enumerate(pep.residues):
        fig.text(x0 + (i + 0.5) * col_w, 0.815, ONE[rn], ha="center", va="center", fontsize=16, fontweight="bold")
    # 2: the jumps
    x1, w2, top2, h2 = 0.465, 0.3, 0.74, 0.585
    fig.text(x1 - 0.045, 0.915, "2  A network sets how often each jumps", **head)
    fig.text(x1 - 0.045, 0.815, "rate of a jump = exp( g − ΔE / 2 )", fontsize=9.5, va="center", color=grey)
    states = out["states"][0]
    row = h2 / pep.S
    cmap = matplotlib.colors.ListedColormap(BASINS)
    for c, (i, kind) in enumerate(pep.sites):
        ax = fig.add_axes([x1, top2 - (c + 1) * row + 0.005, w2, row - 0.01])
        ax.imshow(states[None, :, c], aspect="auto", cmap=cmap, vmin=0, vmax=len(BASINS) - 1, interpolation="nearest",
                  extent=(0, ns, 0, 1))
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if c == pep.S - 1:
            ax.set_xticks([0, ns / 2, ns])
            ax.set_xticklabels(["0", f"{ns / 2:g}", f"{ns:g} ns"])
            ax.tick_params(length=2, pad=1.5, labelsize=8.5)
        else:
            ax.set_xticks([])
    for i, rn in enumerate(pep.residues):
        rows = [c for c, (j, _) in enumerate(pep.sites) if j == i]
        y0, y1 = top2 - (rows[-1] + 1) * row + 0.008, top2 - rows[0] * row - 0.008
        fig.add_artist(plt.Line2D([x1 - 0.012, x1 - 0.012], [y0, y1], color=INK, lw=1.0))
        fig.text(x1 - 0.03, (y0 + y1) / 2, ONE[rn], ha="center", va="center", fontsize=11, fontweight="bold")
    # 3: the atoms
    x2, w3, h3 = 0.815, 0.18, 0.62
    fig.text(x2, 0.915, "3  Atoms are built", **head)
    fig.text(x2, 0.815, "from MD's torsions", fontsize=9.5, va="center", color=grey)
    X = out["xyz"][0].astype(float)
    frame = X[np.argmax(((X - X.mean(1, keepdims=True)) ** 2).sum((1, 2)))]          # the most extended frame
    axes3 = dm.view_axes(frame[None])
    proj = (frame - frame.mean(0)) @ axes3.T
    px = int(w3 * WIDTH * 200), int(h3 * H * 200)
    scale = 0.84 * min(px[0] / 2 / np.abs(proj[:, 0]).max(), px[1] / 2 / np.abs(proj[:, 1]).max())
    img = dm.draw(frame, [a[3] for a in top.atoms], top.bonds(), axes3, frame.mean(0), scale, px, fog=0.3)
    ax = fig.add_axes([x2 - 0.005, 0.12, w3, h3])
    ax.imshow(img, interpolation="lanczos")
    ax.axis("off")
    for xa in (x1 - 0.078, x2 - 0.03):                             # arrows between the three parts
        fig.add_artist(matplotlib.patches.FancyArrowPatch((xa, 0.45), (xa + 0.02, 0.45), transform=fig.transFigure,
                                                          arrowstyle="-|>", mutation_scale=9, color=INK, lw=1))
    save(fig, "method.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["recording", "evidence", "cost", "method"]
    for name in which:
        globals()[name]()
