# Peptide dynamics from sequence

Generate motion for a four-residue peptide from its sequence using an **estimated 2,900 times less GPU time per nanosecond** than molecular dynamics (MD): 0.0074 versus 21.6 GPU-seconds on one T4. No simulation frames or starting structure of that peptide are needed.

[![GCDE heavy atoms from MD and the sequence model at the same scale, with Cys phi traces over 50 ns; the model briefly occupies positive phi.](figures/gcde-motion.png)](figures/gcde-motion.mp4)

Independent 50 ns paths of GCDE, shown at the same scale and playback rate; click to play. In a separate measurement, generating a full microsecond took **1.8 seconds on a desktop CPU**, including coordinates and file writing.

![Four errors against estimated GPU time: free energy and worst passage time above, population and slowest relaxation below, with infinite errors in separate strips above the top panels.](figures/error-cost.png)

A 2.49 µs model sample costs an estimated **18.5 GPU-seconds per peptide**. MD needs about 21,600 GPU-seconds to match its median worst passage-time error on the sampled lengths, an estimated **1,200-fold cost reduction**.

![Free-energy errors for all 13 peptides and four model samples above; passage-time accuracy by timescale below, with individual and pooled model results and MD-half comparisons.](figures/accuracy.png)

Across four model samples, median free-energy error is **0.26 to 0.39 kT**, against **0.35 kT** between the MD halves. For passages taking 3 to 10 ns in MD, **83%** of model estimates are within a factor of 1.5, falling to **28%** for 300 to 1,000 ns; the MD halves score 100% and 14%, respectively.

![All 13 evaluated peptides on their own reference-defined TICA axes, with model probability per bin in colour and MD contours targeting 50%, 80% and 95% of sampled probability.](figures/conformation-atlas.png)

Colour shows model probability per bin; solid, dashed and dotted MD contours target 50%, 80% and 95% of sampled probability for each peptide. Overlapping conformations alone do not establish correct timing.

[Evidence, definitions and limitations](docs/evidence.md) · [Result files](results/README.md) · [Reproduce the numbers](REPRODUCE.md). MD data: [ManyPeptidesMD](data/README.md).

## What this does not show

- Independent validation: these 13 public test tetrapeptides also chose the model variant. The evidence covers implicit solvent at 310 K only.
- Exact kinetics: the slowest relaxation is at least twice too fast in 5 to 6 peptides per sample. The four samples each contain 5 µs; their spread and the noisy MD halves limit the comparison.
- A universal speed-up: the 1,200-fold comparison matches one error measure. GPU timing uses a same-size sibling bundle, excludes analysis and scales with generated length; the CPU result is one run for one peptide.

## How to use it

The trained bundle is included and loads automatically. From this directory:

```bash
pip install -r requirements.txt
python model/generate.py GCDE output --ns 1000 --trajectories 1
```

Load `output/heavy.pdb` with `output/traj_000.xtc` in a molecular viewer, or read `output/dynamics.npz` for states and torsions. Change `GCDE` to another tetrapeptide; [model/](model/README.md) explains the limits and [REPRODUCE.md](REPRODUCE.md) covers retraining for other data. Free to use: [CC0](LICENSE) for original work, with [MIT terms](NOTICE.md) retained for dataset-derived material.
