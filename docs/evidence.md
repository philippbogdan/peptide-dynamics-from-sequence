# What the numbers mean

The model takes a sequence and generates local conformational paths, torsions and reconstructed heavy atoms. It receives no trajectory or starting structure of the peptide being generated. MD for that peptide supplies the states and the yardstick used to assess the output.

## Cost and accuracy are separate comparisons

The headline cost ratio is 21.5515 / 0.0074271, about 2,900, in GPU-seconds per generated nanosecond. The inputs are in [headline.json](../results/headline.json), under `cost`; [value_summary.json](../results/value_summary.json) records this arithmetic and the error-matched comparisons. This is an estimate for generation throughput, not a claim of equal accuracy.

MD timing uses the corpus protocol: OpenMM, amber14, OBC1 implicit solvent, a 1 fs step and eight concurrent simulations on one NVIDIA Tesla T4. Aggregate throughput was 3,959 ns/day for HEAV and 4,009 ns/day for CIPQ; the faster record gives the MD cost per nanosecond. Model timing generated 1 µs of ADKH across 100 paths, including setup, dynamics, torsion emission and heavy-atom building. It used a sibling bundle with the same network size and 10 ps step, trained under different conditions. Applying its timing to this bundle and other tetrapeptides is an assumption; length and configuration-space size can affect cost. The whole card is charged in both timings. Neither includes state assignment or Markov analysis. These are generation costs after training, with no training cost amortised into them.

The error-cost curves compare the beginning of MD half A with half B, and separate model samples with half B. Each point is a median over 13 peptides. Lengths are 10, 30, 100, 300, 1,000 and 2,490 ns; missing main states retain infinite free-energy or passage-time errors. The top panels show these two errors, with infinite values in separate strips; the bottom panels show population and slowest-relaxation errors, which are finite at every plotted length. MD uses blue squares and dashed lines; the model uses orange circles and solid lines. GPU time and the finite passage-time factors use logarithmic axes. Multiplying the sampled lengths by the measured per-ns costs estimates GPU time, including linear scaling of the setup cost. These are not separately timed runs at each length.

At 2,490 ns, the model's median worst passage-time error is a factor of 2.54 at an estimated 18.5 GPU-seconds. The shortest sampled MD length at least as accurate on this measure is 1,000 ns: a factor of 2.35 at about 21,600 GPU-seconds. The resulting cost ratio is about 1,200. There is no interpolation between lengths. Population error alone is matched at 300 ns of MD, giving about 350 times the cost; matching both population and passage-time errors requires 2,490 ns of MD. No sampled MD length matches the model's final free-energy or slowest-relaxation error. This does not establish superiority to a converged MD estimate: half B is itself noisy, and the errors need not decrease monotonically. Model lengths beyond 2.49 µs were not judged in this comparison.

The [CPU record](../results/cost/model_cpu_gcde.json) is a separate measurement: 1 µs of GCDE in 1.785 s on an Apple M4 CPU, including bundle loading, rate enumeration, sampling, torsion emission, coordinates and XTC writing. It is one run for one peptide, not an error-matched CPU versus GPU benchmark.

## Samples and the yardstick

The public ManyPeptidesMD test split has 30 tetrapeptides, separate from the 1,500 training sequences. The same 13 test peptides reported here were used during development to compare training-set sizes, time steps and prefactor bounds and choose this model. There is no separate untouched test set. These medians describe a small development testbed and can move with a few peptides.

Each reference trajectory contains 5 µs. The initial 10 ns is removed, leaving two halves of 2.495 µs from the same simulation, not independent replicas. Both halves define the common reference states. A state model must reproduce every reference relaxation time above 1 ns within 10%; 16 peptides meet this rule. The remaining 14 cannot be assessed by this state-based comparison with the available reference length.

Three accepted state models are omitted from model evaluation: KKAP, QAKR and SHKS have 472,392, 708,588 and 236,196 joint configurations, exceeding the 150,000-configuration rate-table limit. Generation can fall back to a slower sampler, but those peptides were not judged. The resulting comparison has 13 peptides, not all 30.

Each of four independent model samples contains 200 paths of 25 ns per peptide, or 5 µs, with equilibrium starts from the model's continuous-time generator. The sample medians of free-energy error are 0.33, 0.39, 0.36 and 0.26 kT; the MD-half median is 0.35 kT. CIPQ has the largest free-energy error in every sample; NCFG varies from 0.05 to 0.39 kT. In the accuracy chart's upper panel, small orange circles show each sample and thin vertical lines span the four samples for each peptide; open diamonds show the MD-half comparison. The ranges show sampling spread, not confidence intervals or generalisation uncertainty. Being near the halves' disagreement does not demonstrate equivalence to MD.

The slowest relaxation is at least twice too fast in 5 to 6 peptides per sample.

No reference-half pair meets every tolerance among the 16 accepted state models. The four model samples meet every tolerance for 0, 0, 1 and 0 of the 13 peptides. Shuffling the reference frames or doubling its clock also fails the tolerances. The yardstick detects temporal distortion but does not supply precise ground truth at every timescale.

## Definitions

Torsion features include backbone phi and psi and side-chain chi1 to chi4 as sine and cosine, with symmetric end groups made label-blind. Reference-only TICA at 0.1 ns lag, k-means microstates and PCCA+ give crisp or core metastable assignments. A reversible maximum-likelihood Markov model at a per-peptide lag of 0.2 to 2 ns supplies populations, free energies, mean first-passage times and relaxation times. Main states have reference population at least 1%.

Free energy is `F = -ln(p)` in kT. Its error is the root-mean-square difference over main states after subtracting the mean difference. With only one main state, this subtraction makes the free-energy error zero by definition; it does not prove accurate populations. Population error is total variation, `0.5 * sum(abs(p_model - p_reference))`, over all states. A passage-time error is `max(t_model / t_reference, t_reference / t_model)`; the worst pair is taken within each peptide before taking the median over peptides. Relaxation error is relative to the reference.

All tolerances must hold together: every main-state passage time within a factor of 1.5; free-energy error at most 0.1 kT; population error at most 0.02; slowest-relaxation error at most 10%; and error at most 20% for every other reference relaxation above 1 ns.

Passage-time bins use the reference time and include their lower edge but exclude their upper edge; the last starts at 1,000 ns. The pooled model percentages count state-pair estimates across four samples, not independent peptides. The six model denominators are 112, 192, 44, 72, 68 and 12; the MD-half denominators are 29, 47, 10, 13, 22 and 3. Model bins use the full reference, whereas half comparisons use half B, so a pair can fall in a different bin. In the 3 to 10 ns bin, individual model samples score 79% to 89%; in the 300 to 1,000 ns bin, 18% to 47%. All model estimates at or above 1,000 ns miss the factor-1.5 tolerance, but this is only three ordered pairs per sample. The accuracy chart's lower panel shows individual samples as small orange circles, pooled percentages as larger circles, sample ranges as thin vertical lines and MD-half percentages as open diamonds.

## What the other diagnostics establish

The reference-length estimate `L* = (1,000 to 2,000) * tau / (p_a * p_b)` is a counting argument, with consistent time units. Here `tau` is the slowest relaxation and the populations lie on either side of that process. Relative rate noise scales approximately as the inverse square root of the event count; resolving a 10% relaxation difference between two replicas at roughly two to three standard deviations needs about 800 to 1,800 transitions per replica. Estimated lengths range from 11 µs to 5.0 ms. These are order-of-magnitude estimates, not measured convergence times. Half disagreement rises with this estimate: Spearman rho is 0.56 for both slowest-relaxation and worst passage-time error, with p values 0.03 and 0.02.

The [energy-replacement diagnostic](../results/oracle.json) inserts MD-derived joint-configuration energies while retaining learned transition prefactors. Unobserved configurations keep bounded, shifted model energies. Median free-energy error changes from 0.26 to 0.10 kT, but worst passage-time error changes from a factor of 1.87 to 2.11; no peptide meets every tolerance. Better populations alone do not repair the kinetics. Its original model sample differs from the four headline samples.

The [local-rate diagnostic](../results/barriers.json) compares `G = 0.5 * ln(k_ab * k_ba)`, with rates in inverse ns, after subtracting the pooled transition-type value over all 30 test peptides. Its summary uses transitions with at least 20 MD events and finite rates; the slow subset has `G_md < -3`. Model counts are accumulated within each of 200 separate 25 ns paths, totalling 5 µs, without counting joins between paths. The final `context_skill` summary in [headline.json](../results/headline.json) gives:

| Transitions | Count | Correlation | Slope | Model RMS error | Pooled-rate RMS error |
| --- | --- | --- | --- | --- | --- |
| All eligible | 264 | 0.61 | 0.60 | 0.39 | 0.45 |
| Slow subset | 14 | 0.24 | 0.19 | 0.61 | 0.48 |

Correlation and fitted slope compare model deviations with MD deviations; RMS errors are in natural-log rate units. The model improves on the pooled baseline across all eligible transitions but has greater error in the slow subset. The supplementary local-site metrics in `evaluation_seed*.json` still count joins after flattening paths and are not used for the headline state comparisons.

## Physical scope and the visuals

The evidence covers uncapped zwitterionic tetrapeptides in amber14 and OBC1 implicit solvent at 310 K, with Langevin friction 0.3/ps, a 1 fs MD step and no constraints. It does not establish accuracy in explicit water, at other temperatures or for longer peptides.

The model's continuous-time rates obey detailed balance with stationary weights `exp(-E)`. The 10 ps simultaneous-site update used for sampling preserves this only approximately. Above the rate-table limit, generation starts from all-zero site states without equilibration. Emitted torsions are drawn from training-derived conditional banks; coordinates use ideal geometry without minimisation or enforced ring closure. Heavy-atom movies illustrate those outputs, not a new atomistic MD integrator.

The GCDE movie displays independent 50 ns excerpts at the same spatial scale and playback rate. Each frame is rigidly aligned on the N, CA and C atoms of residues 2 and 3, with one shared camera orientation. Displayed frames are 0.1 ns apart; faint sticks show up to ten earlier displayed backbone conformations. No conformations or intermediate frames are interpolated. The lower strips show cysteine phi as thin solid traces, with a square MD marker and a circular model marker. The model stays near +60 degrees for about 10 ns in this excerpt, while MD stays at negative phi. The movie is not the 1 µs CPU timing run or a quantitative accuracy test.

The atlas shows all 13 evaluated peptides, with each peptide code above its panel. It uses all supplied points from the reference and a separate seed-100 model sample, at 0.25 ns spacing. The reference covers 4.99 µs after equilibration removal; the model covers 200 independent 25 ns paths. Each peptide has its own reference-defined TICA axes, so coordinate values are comparable between MD and model within a panel only.

Within each panel, both sources use the same 96 by 96 grid. Model colour shows the fraction of model points in each bin, with a shared logarithmic colour scale across panels; it is probability per bin, not density. Solid, dashed and dotted MD contours outline bins at or above thresholds targeting 50%, 80% and 95% of MD probability, respectively. Including all tied bins makes the actual enclosed masses 50.0% to 52.2%, 80.0% to 83.6% and 95.3% to 97.2% across peptides. Empty model bins are white, with no smoothing or pseudocounts. Projection discards information and does not measure kinetics.
