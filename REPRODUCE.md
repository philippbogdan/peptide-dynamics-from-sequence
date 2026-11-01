# Reproduce the numbers

From this directory, install the dependencies and run:

```bash
pip install -r requirements.txt
python reproduce.py
```

This takes minutes on a laptop CPU. It reads scientific inputs only from `results/`, refits Markov estimates from saved state paths, prints the headline numbers and writes `results/headline.json`. No raw trajectories, GPU, training or model sampling are required. It recalculates the four model samples, reference-half comparisons, passage-time bins, length-dependent errors and reference-length estimates. The energy-replacement, local-rate and timing summaries use their saved JSON records; those experiments are not rerun by this command.

The cost ratios in `results/value_summary.json` are arithmetic on this output, not additional fits: divide `cost.md_gpu_s_per_ns` by `cost.model_gpu_s_per_ns` for the per-ns ratio. For each final model error, find the shortest row of `trade` whose MD error is no greater, then divide its `md_gpu_s` by the final row's `model_gpu_s`. A missing match stays null. `reproduce.py` refreshes the inputs to this arithmetic but does not overwrite that snapshot. [The evidence page](docs/evidence.md) explains the matched-error claims and every retained limitation.

## Run from public inputs

Run all commands below from this directory in the same environment. Full reanalysis writes into `results/` and replaces matching files. The supplied bundle is sufficient; training again is optional. Data conversion, state fitting, evaluation and generation run on a CPU. Use one GPU for training and for a comparable MD timing measurement.

### Download and extract torsions

All 30 test trajectories occupy about 24.4 GB before torsion extraction. The 1,500 training trajectories occupy about 455 GB. Allow additional space for PDB archives, torsion tables and temporary files. Training data are needed only to fit a new bundle.

```bash
python data/download.py test
python data/cache_torsions.py test
```

The default locations are `data/raw/` and `data/torsions/`. Downloads accept `--names HEAV CIPQ` to select peptides and `--out DIR` to change the destination. Conversion accepts `--raw DIR`, `--out DIR`, `--stride N` and `--procs N` (default 4). It adds a `test/` or `train/` subdirectory to the output parent. Existing nonempty downloads and existing torsion tables are reused; changing conversion settings does not invalidate a cache.

Keep the test stride at 1, giving the 10 ps spacing expected by the analysis. For an alternative torsion location, set `PEPDYN_TORSIONS` to the parent containing `test/` and `train/`.

### Define states, sample and evaluate

```bash
python analysis/reference.py
python analysis/assign_references.py
python analysis/evaluate.py 0
python analysis/evaluate.py 1
python analysis/evaluate.py 2
python analysis/evaluate.py 3
python analysis/trade.py
python analysis/barriers.py
python analysis/oracle.py
python analysis/landscapes.py
python reproduce.py
```

`reference.py` removes the initial 10 ns and divides each remaining 4.99 us trajectory into equal halves. It fits states using both halves and saves the reference comparisons and controls. Optional sequence arguments restrict this step; `NPROC` sets concurrent processes, default 4. `assign_references.py` saves the discrete reference paths used by the lightweight command.

Each evaluation seed generates 200 paths of 25 ns per peptide, totalling 5 us, at 10 ps intervals. Starts are sampled from the continuous-time generator's Boltzmann weights. Subsequent simultaneous-site updates use the trained finite step and need not preserve that distribution exactly. The generated torsions are assigned to reference states with the same mapping used for MD.

There are 16 accepted reference-state models but 13 evaluated peptides. Evaluation skips absent state models, unsupported site definitions or vocabulary entries, and joint configuration spaces exceeding 150,000 states. KKAP, QAKR and SHKS have 472,392, 708,588 and 236,196 joint configurations, respectively, exceeding that enumeration limit. These omissions must remain visible when interpreting the 13-peptide medians.

`trade.py` compares partial reference half A and new model samples with half B, for total lengths from 10 ns to 2.49 us. Longer model runs were not evaluated. Infinite errors retain cases with missing main states. `barriers.py` needs all 30 test tables for its aggregate transition-type baseline. `oracle.py` replaces observed joint-state energies with MD estimates while retaining learned prefactors; unobserved configurations retain bounded, shifted model energies. `landscapes.py` generates a separate seed-100 sample for matched projections.

To select a different bundle, use these actual positional interfaces:

```text
python analysis/evaluate.py SEED BUNDLE_DIR
python analysis/trade.py OUT.json BUNDLE_DIR
python analysis/barriers.py OUT.json BUNDLE_DIR
python analysis/oracle.py OUT.json BUNDLE_DIR
python analysis/landscapes.py BUNDLE_DIR
```

### Fit a new bundle

```bash
python data/download.py train
python data/cache_torsions.py train --stride 2
python model/prep.py OUT.pkl 10 310 data/torsions/train/*.npz
GBOUND=4 python model/train.py OUT.pkl OUTDIR 6000
python model/bundle.py OUT.pkl OUTDIR/net.pt model/bundle 4
```

The last command replaces the supplied bundle files. Preparation uses all 1,500 training tables and thins them to 10 ps. Training reserves 5% of the peptides for validation, so optimiser updates use the remaining 95%. Its optional final arguments are `PEPTIDES_PER_STEP` and `RUNS_PER_PEPTIDE`, defaulting to 16 and 2048. The network defaults are `W=2`, `D=64` and `H=192`; packaging assumes those dimensions. `SEED`, `LR` and `WD` also configure training. `GBOUND=4` must be supplied explicitly because the script default is 2.

The supplied model's recorded 6,000-step training took 12,924 s on one GPU. Its per-step batch settings were not recorded, so the commands above use the script defaults. They describe how to refit the model, not a guarantee of identical weights.

### Generate from a sequence

```bash
python model/generate.py HEAV generated/HEAV --ns 1000 --trajectories 100 --seed 0
```

This writes 100 paths of 10 ns each: `--ns` is the total across paths. Outputs are `dynamics.npz` with site labels and torsions, `heavy.pdb`, one `traj_*.xtc` per path, and `cost.json`. `--no-atoms` omits the XTC files but still writes the PDB template. `--bundle DIR` selects another bundle. Coordinates use template bond geometry, without minimisation or ring-closure enforcement. For more than 150,000 configurations, generation starts every site in state zero without equilibration. The demonstrated accuracy is limited to tetrapeptides at the stated conditions.

### Time the MD protocol

OpenMM is optional and is not in `requirements.txt`:

```bash
pip install openmm
python analysis/bench_md.py data/raw/pdb/test/HEAV.pdb 60
```

The command minimises the structure, runs 2,000 warm-up steps and times one simulation for approximately the requested seconds. It tries CUDA, OpenCL and then CPU, and reports the selected platform. Use a GPU to compare with the supplied GPU cost measurements.

The saved MD rates are 3,959 and 4,009 ns/day aggregate for HEAV and CIPQ, using eight concurrent simulations on one NVIDIA Tesla T4. A single invocation above does not reproduce that concurrency. The saved model cost, 0.0074 GPU-s/ns, came from 1 us of ADKH generation including setup, dynamics, torsion emission and heavy-atom reconstruction. It used a sibling bundle with the same network size and time step but trained under different conditions, not the supplied bundle. The full card was charged to each timing. State assignment and Markov analysis are excluded from both costs; a fresh CPU `cost.json` is not a replacement for this GPU measurement.

## Interpretation and retained limitations

These are development results on public data. The same 13 peptides informed model selection, with no separate untouched test set. Four independent model samples quantify sampling variation at 5 us each; they do not remove selection effects or the uncertainty of medians over 13 peptides. The reference halves are correlated segments of one simulation, and none of the 16 pairs meets every tolerance. The estimated reference length needed for a precise rate test is an order-of-magnitude counting argument, not a measured convergence time.

The saved oracle result uses an earlier model sample than the four evaluation seeds. The saved timing records are measurements, not rerun by `reproduce.py`. The supplied `results/barriers.json` uses within-path transition counts: `barriers.py` uses 25,000 ps per path for 200 paths, totalling 5 us, and sums counts without adding transitions between paths. The supplied `headline.json` contains the final context-skill summary from this record; its values are reported on the [evidence page](docs/evidence.md). If the diagnostic record changes, rerun `reproduce.py` to refresh the summary. `evaluate.py`'s supplementary local-site metrics still flatten independent paths and count their joins as transitions. The main metastable-state comparisons preserve path boundaries. Regenerating the headline report uses the saved diagnostic records without new sampling.
