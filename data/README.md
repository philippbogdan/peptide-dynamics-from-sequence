# Public trajectories and compact torsion tables

The inputs are [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md), released with *Amortized Sampling with Transferable Normalizing Flows* (NeurIPS 2025). Its simulations use uncapped zwitterionic peptides, amber14, OBC1 implicit solvent, 310 K, Langevin friction 0.3/ps, a 1 fs step and no constraints. The dataset's MIT terms are retained in [NOTICE.md](../NOTICE.md).

| File | Contents |
| --- | --- |
| [train_tetrapeptides.txt](train_tetrapeptides.txt) | 1,500 training sequences, 200 ns each: 300 us in total, about 455 GB of raw trajectories. |
| [test_tetrapeptides.txt](test_tetrapeptides.txt) | 30 distinct test sequences, 5 us each: about 24.4 GB of raw trajectories. None appears in the training list. |
| [download.py](download.py) | Fetches the selected split's PDB archive and trajectories, optionally restricted to named sequences. |
| [cache_torsions.py](cache_torsions.py) | Converts coordinates to per-residue torsions, retaining the frame interval in each NPZ. |

Raw coordinates are downloaded separately. A torsion table contains `res`, `dt_ps` and `tors`, shaped `(frames, residues, 7)` in radians: phi, psi, chi1 to chi4 and an equivalent-atom partner angle for Val or Leu. Missing angles are NaN. Training tables use float16; test tables use float32.

Use stride 2 for the training cache; `model/prep.py` then thins it to 10 ps. Test tables retain their native 10 ps spacing. Existing cache files are reused, so changing stride requires a different output directory or deliberate replacement of those files. The analysis expects 10 ps test tables. Commands and storage requirements are in [REPRODUCE.md](../REPRODUCE.md).
