# Measure thermodynamics and kinetics on common states

The reference defines the metastable states. MD and generated paths are then compared through the same reversible Markov estimator: populations, free energies, mean passage times and relaxation times. Comparing the two reference halves exposes the uncertainty in the yardstick itself.

| File | Question it answers |
| --- | --- |
| [states.py](states.py) | Can a compact state model retain every reference relaxation above 1 ns within 10%? Fits torsion-feature TICA, microstates and metastable assignments. |
| [kinetics.py](kinetics.py) | What populations and times follow from these discrete paths? Preserves trajectory boundaries in transition counts. |
| [check.py](check.py) | Do all passage-time, free-energy, population and relaxation tolerances hold? |
| [reference.py](reference.py) | Which peptides admit the state definition, how well do their reference halves agree, and do shuffled-time and doubled-clock controls fail? |
| [assign_references.py](assign_references.py) | Saves the reference state paths used by the lightweight reproduction command. |
| [evaluate.py](evaluate.py) | How does a 5 us model sample compare with the full reference? Uses 200 paths per peptide and accepts a sampling seed. |
| [trade.py](trade.py) | How do errors change with total trajectory length, up to 2.49 us, against reference half B? |
| [oracle.py](oracle.py) | What changes when MD configuration energies replace model energies while learned prefactors stay fixed? |
| [barriers.py](barriers.py) | How well are sequence-dependent log-rate deviations reproduced relative to an aggregate transition-type baseline? Requires all 30 test peptides. |
| [landscapes.py](landscapes.py) | Exports matched TICA projections, states and backbone torsions for visual comparison. |
| [bench_md.py](bench_md.py) | Times one OpenMM simulation under the reference protocol. |
| [paths.py](paths.py) | Defines shared locations, frame spacing and the equilibration cut. |

The 30 test peptides yield 16 accepted state models and 13 model comparisons. Reference “replicas” are two 2.495 us halves of one trajectory after discarding 10 ns, not independent simulations. None of the 16 pairs meets every tolerance. The four model samples measure sampling variation; they do not constitute an untouched test set.

[REPRODUCE.md](../REPRODUCE.md) gives the full execution order, skip conditions and limitations of the diagnostic and timing records. [../reproduce.py](../reproduce.py) recalculates the headline comparisons from saved state paths without raw MD or new model samples.
