# The comparison, on one page

**What is compared.** For one held-out peptide: a reference, 5 µs of MD from the public corpus read every 10 ps
(first 10 ns dropped, the rest cut into two halves of 2.495 µs), and a candidate, 5 µs of model trajectories
(200 of 25 ns, each started as the model starts any trajectory, from exp(−E)/Z: [the model](model.md)). Both
are reduced to torsion angles; the candidate's are measured on the heavy atoms it wrote.

**1. Describe each frame by its torsions.** φ and ψ of every residue that has them and every side-chain torsion
χ1 to χ4, each as (cos, sin). Where the end atoms are chemically equivalent the description is blind to their
labels: the angle is doubled for Phe and Tyr χ2, Asp χ2 and Glu χ3, and the two equivalent methyl torsions of
Val χ1 and Leu χ2 are summed as unit vectors.

**2. Find the metastable states from the reference alone** (`pepdyn/states.py`). Slow coordinates by time-lagged
independent component analysis (lag 0.1 ns, every component with autocorrelation above 0.01, at most 20);
clusters in those coordinates, one per 2,000 frames (50 to 200, and twice that). Markov models on the clusters,
at six lags from 0.05 to 2 ns for each of the two numbers of clusters, give the reference's relaxation times:
the k-th is taken as the largest k-th value among the twelve models. The clusters are lumped into states by
PCCA+, starting from one more state than there are relaxation times above 1 ns and adding up to six. A frame
takes the state of its cluster or, in the stricter settings, of the last *core* it visited (clusters whose
membership of a state exceeds 0.5 or 0.7), so that a recrossing of a boundary is not a transition. The first
setting, in a fixed order, whose Markov model on the states reproduces every reference relaxation time above
1 ns within 10% (the slowest one, if none is above 1 ns), at a lag of at most half the shortest of them, defines
the states; four of the fifteen definitions are of the first kind, without cores. **If no setting does, the
peptide has no states and cannot be judged this way**: that is the fate of 19 of the 30 test peptides and 6 of
the 10 validation peptides. The verdict leans on the clustering: with another seed for k-means
(`tools/make_reference.py --seed 1`) three peptides gain states, one loses them, and seven of the other fourteen
change their number of states. On those states (`data/reference_seed1`) the comparison at equal length reads
0.29 kT against 0.14 kT in free energy and a factor 2.18 against 1.35 in the worst passage time, where the
stored ones give 0.36 against 0.20 and 2.28 against 1.53.

**3. Measure everything the same way** (`pepdyn/markov.py`). A reversible maximum-likelihood Markov model on the
states, at the lag fixed in step 2, gives populations π, free energies F = −ln π (in kT), the mean first-passage
time between every ordered pair of states (the expected time to first reach one from the other) and the
relaxation times. The same estimator is applied to the candidate, to the whole reference and to each of its
halves. Main states are those with at least 1% of the reference population; a peptide with a single main state
(TAPF) has no passage times or free-energy differences, and counts for populations and relaxation times only.

**4. Errors** (`pepdyn/compare.py`).

| quantity | error | tolerance |
|---|---|---|
| passage times | the worst factor between candidate and reference over all ordered pairs of main states | 1.5 |
| free energies | root mean square of F(candidate) − F(reference) over main states, mean difference removed | 0.1 kT |
| populations | total variation, half the sum of the absolute differences of π over all states | 0.02 |
| slowest relaxation | relative error of the slowest relaxation time | 10% |
| other relaxations | worst relative error of the other relaxation times above 1 ns, paired in order of size | 20% |

**5. Is the reference good enough to judge at these tolerances?** The same errors are measured between its two
halves. No peptide's halves meet all five tolerances, so no candidate can be certified at them: the errors of
the model are reported beside the errors of the reference against itself. Two halves hold half the data on each
side, so their disagreement overstates the noise in the comparison of 5 µs with 5 µs. The like-for-like
comparison is in `evaluate.py --trade`: 2.49 µs of the model, and one half of the reference, each against the
other half.

**6. Does the comparison catch what it must?** Two candidates must fail on every peptide, and do
(`tests/test_compare.py`): the reference's own frames shuffled in time (populations kept, all kinetics
destroyed) and the reference with its clock doubled (every time twice too long).

**Without states.** All 40 peptides, the 25 without states among them, are compared site by site
(`evaluate.py --sites`): for every backbone and side-chain site of the model's alphabet, the population of each
of its states and the number of jumps per nanosecond, in 20 µs of model trajectories, in the whole reference and
in each half. Here the model's states are the ones it generated, not read back from its atoms; the reference's
are read from its torsions with the same basins.
