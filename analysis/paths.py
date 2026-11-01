"""Shared paths, frame spacing and reference preparation for the analysis scripts.

PEPDYN_TORSIONS overrides the torsion-table directory; results always live
under the repository's results directory. Test trajectories are treated as
310 K data with a 10 ps frame interval and a 10 ns equilibration cut.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))

TORSIONS = os.environ.get("PEPDYN_TORSIONS", os.path.join(ROOT, "data", "torsions"))
RESULTS = os.path.join(ROOT, "results")
STATES = os.path.join(RESULTS, "states")
REFERENCES = os.path.join(RESULTS, "references")
DTRAJS = os.path.join(RESULTS, "dtrajs")
TEMPERATURE_K = 310.0
FRAME_NS = 0.01
EQUIL_FRAMES = 1000


def test_table(name):
    """Load one test torsion table and attach the reference temperature."""
    import sitepaths as SP
    tb = SP.load(os.path.join(TORSIONS, "test", name + ".npz"))
    tb["temperature"] = TEMPERATURE_K
    return tb


def judged_peptides():
    """List sequences with a saved reference-state model.

    This includes sequences later skipped by model evaluation, for example when
    their configuration space exceeds its enumeration limit."""
    return sorted(f[:-4] for f in os.listdir(STATES) if f.endswith(".npz"))


def reference_halves(name, sm=None):
    """Assign two equal halves after removing the initial equilibration frames.

    This helper expects torsion tables already spaced at 10 ps. Both halves
    come from one MD trajectory and are not independent simulations."""
    import data as D
    import states as ST
    sm = sm or ST.load(os.path.join(STATES, name + ".npz"))
    tb = test_table(name)
    X = D.features_from_tors(tb["res"], tb["tors"])[EQUIL_FRAMES:]
    h = len(X) // 2
    return sm, tb, sm.assign(X[:h]), sm.assign(X[h:2 * h])
