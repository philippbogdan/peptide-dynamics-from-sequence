"""Atoms from torsions: geometry, round trip and files."""
import numpy as np
import pytest

import pepdyn
from pepdyn import build


@pytest.fixture(scope="module")
def out():
    return pepdyn.generate("ACDEFGHIKLMNPQRSTVWY", ns=3.0, n_traj=1, seed=3)


def test_atoms_realise_the_torsions(out):
    """Torsions measured on the single-precision coordinates equal the torsions drawn (proline's chi2, which the
    ring closure may shift, to 0.2 degrees)."""
    pep, top = out["peptide"], out["topology"]
    T = out["torsions"][0]
    got = build.measure(pep, top, out["xyz"][0])
    assert np.array_equal(np.isfinite(T), np.isfinite(got))
    err = np.abs(np.angle(np.exp(1j * (got - T))))
    assert np.nanmax(err) < np.radians(0.2)
    pro = pep.residues.index("PRO")
    err[:, pro, 3] = 0.0
    assert np.nanmax(err) < 1e-4


def test_bond_lengths_are_the_library_values(out):
    top, X = out["topology"], out["xyz"][0].astype(float)
    for a, b in top.bonds():
        d = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
        assert d.max() - d.min() < 2e-3, (top.atoms[a], top.atoms[b])
        assert 1.2 < d.mean() < 1.85


def test_residues_are_l_amino_acids(out):
    """The improper torsion N, C, CA, CB has the sign of an L residue (glycine has no CB)."""
    top, X = out["topology"], out["xyz"][0].astype(float)
    for (num, name), k in top.index.items():
        if name == "CB":
            at = lambda a: X[:, top.index[(num, a)]]
            improper = np.degrees(build.dihedral(at("N"), at("C"), at("CA"), at("CB")))
            assert np.all((improper > 100) & (improper < 140)), (num, improper.min(), improper.max())


def test_overlaps_are_redrawn():
    """Drawing a frame again keeps its site states and removes most overlaps."""
    mod = pepdyn.load_model()
    states = pepdyn.generate("WNMA", ns=40.0, seed=0, model=mod)["states"][0].astype(np.int64)
    pep = mod.peptide("WNMA")
    kept = pepdyn.frames_from_states(mod, pep, states, np.random.default_rng(0), redraw=False)[3]
    T, top, X, again = pepdyn.frames_from_states(mod, pep, states, np.random.default_rng(0), redraw=True)
    assert again["drawn_again"] == kept["drawn_again"] == kept["left_overlapping"] > 0.1
    assert again["left_overlapping"] < 0.25 * kept["left_overlapping"]
    assert np.isclose(np.mean(build.closest(X, build.far_pairs(top)) < pepdyn.pipeline.MIN_DISTANCE), again["left_overlapping"])


def test_files_round_trip(out, tmp_path):
    top, X = out["topology"], out["xyz"][0]
    build.write_pdb(tmp_path / "p.pdb", top, X[0])
    build.write_dcd(tmp_path / "p.dcd", X)
    assert np.array_equal(build.read_dcd(tmp_path / "p.dcd"), X)
    lines = [l for l in open(tmp_path / "p.pdb") if l.startswith("ATOM")]
    assert len(lines) == top.n_atoms
    xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])
    assert np.abs(xyz - X[0]).max() < 6e-4
    assert [l[12:16].strip() for l in lines] == [a[0] for a in top.atoms]
