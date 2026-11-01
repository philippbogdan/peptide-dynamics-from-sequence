# Sequence to peptide dynamics

The supplied bundle generates local conformational paths and heavy-atom trajectories from an amino-acid sequence. No MD frames or starting structure of that peptide are needed. The demonstrated setting is uncapped tetrapeptides at 310 K in OBC1 implicit solvent.

```text
sequence -> local states -> transition rates -> state paths -> torsions -> heavy atoms
```

Each residue has a backbone state and one state per available side-chain torsion. A network learns energies and symmetric transition prefactors from neighbouring residues. Its continuous-time rates obey detailed balance. Generation uses simultaneous finite-step updates at 10 ps; this approximation does not enforce the generator's exact equilibrium distribution.

| File | Role |
| --- | --- |
| [generate.py](generate.py) | Sequence-to-trajectory command. Writes site labels, torsions, a template PDB and optional XTC coordinates. `--ns` is the total duration across all paths. |
| [bundle/](bundle/) and [bundle.py](bundle.py) | Trained weights, state definitions, emission bank and baseline rates; loading and packaging functions. |
| [ctmc.py](ctmc.py) | Network, likelihood, rate enumeration and continuous-time or finite-step samplers. Enumeration permits exact Boltzmann starts for small configuration spaces. |
| [sites.py](sites.py) | Backbone and rotamer basins fitted to periodic torsion histograms. |
| [data.py](data.py) | Torsion extraction, symmetry-aware features and training-derived torsion emission. |
| [sitepaths.py](sitepaths.py) | Path preparation and independent torsion draws conditional on local states. |
| [zmatrix.py](zmatrix.py) | Geometric heavy-atom reconstruction; no energy minimisation or ring-closure constraint. |
| [prep.py](prep.py), [train.py](train.py) | Training-data preparation and maximum-likelihood fitting. |

Preparation uses all 1,500 training sequences. The optimiser reserves 5% for validation. The supplied network uses a residue-window radius of 2, embedding width 64, hidden width 192 and a log-prefactor correction bounded by 4. It was fitted for 6,000 steps.

The public test sequences also informed model selection, so the reported comparisons are development results. Larger configuration spaces fall back to state-zero starts without equilibration; longer peptides and other physical conditions have not been validated. See [REPRODUCE.md](../REPRODUCE.md) for commands and [NOTICE.md](../NOTICE.md) for the bundle's data terms.
