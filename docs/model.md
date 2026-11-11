# The model in full

A peptide's motion, seen every 10 ps, is mostly jumps. Each residue's backbone sits in one basin of its (φ, ψ) map
and now and then hops to another; each side-chain torsion sits in one rotamer and now and then flips. The model
keeps only that: one small integer per site, and a rate for every possible jump.

## States

A **site** is one backbone or one side-chain torsion χ1 to χ4 of one residue. The backbone site of an inner
residue is the basin of its (φ, ψ) map; the first residue has no φ and the last no ψ (the termini are charged),
so their backbone site is the basin of the one angle they have.

The basins of each kind of site are found once, from 4 µs of training MD that show it (`train/sites.py`: its
first twenty occurrences among the training peptides in alphabetical order): a histogram of the angle or of the
(φ, ψ) map gives a free energy; steepest descent splits it into basins; a basin shallower than 1 kT or holding
less than 0.2% is merged into its neighbour. The **core** of a basin is where the free energy is within 2.5 kT of
its minimum, and a frame belongs to the last core its trajectory visited, so a brief recrossing of a ridge is not
a jump. The 20 amino acids give 99 kinds of site with 247 states in all (figure in the README, part 1;
`models/implicit-310K.npz` holds the grids).

A **configuration** $x$ is the state of every site of the peptide: 4 to 20 integers for a tetrapeptide.

## Rates

A jump changes one site. The model is an energy $E(x)$, in units of kT, and a prefactor $g$:

$$k(x \to x') = \exp\left(g - \tfrac{1}{2}\,[E(x') - E(x)]\right) \quad \text{per ps}.$$

- $E(x)$ is a sum, over the residues and the two termini, of a small network applied to a window of five tokens.
  A token is the sum of an embedding of the residue type and embeddings of the current states of its sites.
- $g$ comes from a second network that reads the window of the jumping residue *without the jumping site's own
  state*, and a code of the pair (state left, state entered) that is symmetric in the two. So $g$ is the same
  for a jump and its reverse.

Hence $k(x \to x') / k(x' \to x) = \exp(-[E(x') - E(x)])$: detailed balance holds exactly, whatever the networks
learn, and the jump process has the equilibrium $\exp(-E)/Z$ (`tests/test_model.py` checks every jump of every
configuration of two peptides).

Both networks correct a baseline that needs no network: the free energy of each state, added to $E$, and the
symmetric part of each jump's rate, added to $g$, both pooled over all training peptides by residue type. The
free energies are then trained with the networks. The baseline of the rates stays fixed and the network's
correction to it is bounded, four log units either way, so that beyond that the model can make a jump rarer only
by raising an energy. Networks and baseline hold 314,556 numbers, of which 253,052 are trained
(`pepdyn/model.py`, under 400 lines of numpy; `train/network.py` is the same in PyTorch).

## Training

Maximum likelihood on the public MD of 1,500 tetrapeptides, 200 ns each, read every 10 ps (ManyPeptidesMD,
amber14 with OBC1 implicit solvent at 310 K). 75 of the peptides are left out of the gradient steps, to watch the
likelihood; the basins, the baseline and the torsion bank below use all 1,500. What is fitted is the chain that is
later run: in one 10 ps step, every site independently leaves its state with probability $1 - \exp(-K\,\Delta t)$,
where $K$ is the sum of its jump rates in the current configuration, for a state drawn in proportion to those
rates. 6,000 steps of AdamW took 3.6 hours on one GPU (`results/training/train_log.txt`).

## Generation

1. **Start.** Configurations are drawn from $\exp(-E)/Z$: exactly when the peptide's joint space can be listed,
   by Metropolis sampling otherwise.
2. **Jump.** The 10 ps chain above is run, many trajectories at once. For 94% of tetrapeptides the joint space
   has under a million configurations and every rate is tabulated once, after which a step is a table lookup.
   The chain is not the jump process sampled every 10 ps: within a step a site moves at most once, and the sites
   move independently. So its own equilibrium is near $\exp(-E)/Z$ but not equal to it (4% apart in total
   variation for GCDE, `tests/test_model.py`), and a trajectory drifts from the one to the other over the
   peptide's relaxation times. It is the chain, started this way, that was fitted to MD and that is scored.
3. **Torsions.** For every residue in every combination of its site states, the model keeps up to 1,600 sets of
   torsions from training frames in which every site of the residue sat in the core of its basin: 400 from each
   of the last four training peptides that show the combination (the bank, 1.46 million rows). Each frame draws
   one row per residue. Two frames in the same state are therefore independent draws, and so are two residues
   of one frame: motion faster than a jump, and any coupling between residues beyond their states, is not
   modelled. A combination the bank does not hold borrows the torsions of another with the same backbone
   state: 0.02% of the frames generated for the 40 held-out peptides, 0.3% at most for one (`results/bank.json`).
4. **Atoms.** Heavy atoms are placed from the torsions with fixed bond lengths and angles
   (`pepdyn/residues.json`, measured on the corpus's starting structures), proline's ring closed exactly, peptide
   bonds trans with the spread they have in the training MD (0.18 rad), and the C-terminal carboxylate, which
   has no state in the model, held at its commonest orientation there (130°). MD frames rebuilt this way from
   their own torsions, peptide bonds included, lie 0.51 to 0.57 Å from the originals (`results/atoms.json`).
   Fixed geometry and independent draws can put two atoms on top of each other, which MD never does: a frame in
   which two atoms more than three bonds apart come within 2.2 Å is drawn again, up to 99 times, with its states
   unchanged, and the draw with the widest closest contact is kept (at most 0.2% of a peptide's frames keep
   such a contact).

Hydrogens are not written.
