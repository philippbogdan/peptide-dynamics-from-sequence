"""From torsion angles to heavy atoms, and trajectory files.

Bond lengths, bond angles and the fixed dihedrals come from pepdyn/residues.json (template geometry of the
training corpus's force field); the torsions drive the rest. Each atom is placed from three atoms already placed
(NeRF). Peptide bonds are trans, with omega drawn around 180 degrees. Coordinates are in angstrom.
"""
import json
import os
import struct

import numpy as np

LIBRARY = json.load(open(os.path.join(os.path.dirname(__file__), "residues.json")))
COLUMN = dict(phi=0, psi=1, chi1=2, chi2=3, chi3=4, chi4=5, pair=6)
OMEGA_SD = 0.18                     # radians: the spread of the peptide-bond torsion in the training MD
C_TERMINAL_PSI = np.radians(130.0)  # the last residue has no psi in the model: its carboxylate is held at the
#                                     orientation (N, CA, C, O) it most often has in the training MD


def place(a, b, c, bond, angle, dihedral):
    """The point d with |cd| = bond, angle (b, c, d) = angle and dihedral (a, b, c, d) = dihedral (radians)."""
    bc = c - b
    bc = bc / np.linalg.norm(bc, axis=-1, keepdims=True)
    n = np.cross(b - a, bc)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    m = np.cross(n, bc)
    dihedral = np.broadcast_to(dihedral, a.shape[:-1])
    x = -bond * np.cos(angle)
    y = bond * np.sin(angle) * np.cos(dihedral)
    z = bond * np.sin(angle) * np.sin(dihedral)
    return c + x * bc + y[..., None] * m + z[..., None] * n


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.arctan2((np.cross(b1, v) * w).sum(-1), (v * w).sum(-1))


class Topology:
    """Heavy atoms of a peptide with charged termini, in the order they are written: per residue N, CA, C, O,
    then the side chain; the last residue also has OXT."""

    def __init__(self, pep):
        self.atoms = []            # (atom name, residue name, residue number, element)
        n = len(pep.residues)
        for i, rn in enumerate(pep.residues):
            names = list(LIBRARY[rn]["atoms"])
            if i == n - 1:
                names.insert(4, "OXT")
            self.atoms += [(a, rn, i + 1, a[0]) for a in names]
        self.index = {(num, name): k for k, (name, _, num, _) in enumerate(self.atoms)}
        self.n_atoms = len(self.atoms)

    def bonds(self):
        """Pairs of bonded atoms (indices), from the residue library and the chain."""
        out = []
        for (num, name), k in self.index.items():
            out += [(self.index[(num, a)], self.index[(num, b)]) for a, b in LIBRARY[self.atoms[k][1]]["bonds"] if a == name]
            if name == "OXT":
                out.append((self.index[(num, "C")], k))
            if name == "N" and num > 1:
                out.append((self.index[(num - 1, "C")], k))
        return sorted(set(tuple(sorted(b)) for b in out))


def coordinates(pep, T, rng, omega=None):
    """Torsions (frames, residues, 7) -> (Topology, coordinates (frames, atoms, 3) in angstrom).
    omega (frames, residues - 1), the torsion of each peptide bond, is drawn around 180 degrees unless given."""
    top = Topology(pep)
    F, n = len(T), len(pep.residues)
    X = np.zeros((F, top.n_atoms, 3))
    rad = np.radians
    lib = [LIBRARY[r] for r in pep.residues]
    bb = [r["backbone"] for r in lib]
    if omega is None:
        omega = np.pi + rng.normal(0.0, OMEGA_SD, size=(F, n - 1))
    at = lambda num, name: top.index[(num, name)]
    X[:, at(1, "CA")] = [bb[0]["N_CA"], 0.0, 0.0]                    # the first residue's N sits at the origin
    t = rad(bb[0]["N_CA_C"])
    X[:, at(1, "C")] = X[:, at(1, "CA")] + bb[0]["CA_C"] * np.array([-np.cos(t), np.sin(t), 0.0])
    for i in range(n):
        num, b = i + 1, bb[i]
        N, CA, C = X[:, at(num, "N")], X[:, at(num, "CA")], X[:, at(num, "C")]
        if i == n - 1:
            ct = LIBRARY["C_TERMINUS"]
            X[:, at(num, "O")] = place(N, CA, C, ct["C_O"], rad(ct["CA_C_O"]), C_TERMINAL_PSI)
            X[:, at(num, "OXT")] = place(N, CA, C, ct["C_OXT"], rad(ct["CA_C_OXT"]), C_TERMINAL_PSI + rad(ct["OXT_from_O"]))
        else:
            psi, nb = T[:, i, 1], bb[i + 1]
            X[:, at(num, "O")] = place(N, CA, C, b["C_O"], rad(b["CA_C_O"]), psi + np.pi)
            X[:, at(num + 1, "N")] = place(N, CA, C, nb["C_N"], rad(nb["CA_C_N"]), psi)
            X[:, at(num + 1, "CA")] = place(CA, C, X[:, at(num + 1, "N")], nb["N_CA"], rad(nb["C_N_CA"]), omega[:, i])
            X[:, at(num + 1, "C")] = place(C, X[:, at(num + 1, "N")], X[:, at(num + 1, "CA")], nb["CA_C"], rad(nb["N_CA_C"]),
                                           T[:, i + 1, 0])
        for a in lib[i]["side"]:
            p, g, gg = (X[:, at(num, r)] for r in a["refs"])
            d = rad(a["offset"]) if a["driver"] is None else T[:, i, COLUMN[a["driver"]]] + rad(a["offset"])
            if "closes" in a:
                X[:, at(num, a["name"])] = close_ring(gg, g, p, X[:, at(num, a["closes"][0])], a["bond"], a["closes"][1], d)
            else:
                X[:, at(num, a["name"])] = place(gg, g, p, a["bond"], rad(a["angle"]), d)
    return top, X


def close_ring(a, b, c, e, bond_c, bond_e, target, chunk=20000):
    """The point d at bond_c from c and bond_e from e whose dihedral (a, b, c, d) is nearest to target: proline's
    CD, which must bond both CG and N. With fixed bond angles the two torsions of the ring cannot both be met
    exactly, so the ring is closed and chi2 follows as closely as the closure allows."""
    out = np.zeros_like(c)
    target = np.broadcast_to(target, c.shape[:-1])
    coarse = np.linspace(0.0, 2 * np.pi, 120, endpoint=False)[None]             # 3 degrees apart, then 0.1
    fine = np.radians(np.linspace(-3.0, 3.0, 61))[None]
    for s in range(0, len(c), chunk):
        sl = slice(s, s + chunk)
        axis = e[sl] - c[sl]
        dist = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = axis / dist
        along = (dist ** 2 + bond_c ** 2 - bond_e ** 2) / (2 * dist)
        radius = np.sqrt(np.maximum(bond_c ** 2 - along ** 2, 0.0))
        u = np.cross(axis, b[sl] - c[sl])
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        v = np.cross(axis, u)
        centre = (c[sl] + along * axis)[:, None]
        t = np.broadcast_to(coarse, (len(centre), coarse.shape[1]))
        for _ in range(2):
            ring = centre + radius[:, None] * (np.cos(t)[..., None] * u[:, None] + np.sin(t)[..., None] * v[:, None])
            d = dihedral(a[sl][:, None], b[sl][:, None], c[sl][:, None], ring)
            best = np.argmin(np.abs(np.angle(np.exp(1j * (d - target[sl][:, None])))), axis=1)
            rows = np.arange(len(best))
            out[sl] = ring[rows, best]
            t = t[rows, best][:, None] + fine
    return out


PAIR_ATOMS = {"VAL": ("N", "CA", "CB", "CG2"), "LEU": ("CA", "CB", "CG", "CD2")}


def measure(pep, top, X):
    """Coordinates (frames, atoms, 3) -> the torsions (frames, residues, 7) they realise: the inverse of
    coordinates(), with the same columns as the torsion bank and NaN where a residue has no such torsion."""
    X = np.asarray(X, np.float64)
    n = len(pep.residues)
    T = np.full((len(X), n, 7), np.nan)
    at = lambda num, name: X[:, top.index[(num, name)]]
    for i, rn in enumerate(pep.residues):
        num = i + 1
        if i > 0:
            T[:, i, 0] = dihedral(at(num - 1, "C"), at(num, "N"), at(num, "CA"), at(num, "C"))
        if i < n - 1:
            T[:, i, 1] = dihedral(at(num, "N"), at(num, "CA"), at(num, "C"), at(num + 1, "N"))
        for k, quad in LIBRARY[rn]["chi"].items():
            T[:, i, COLUMN[k]] = dihedral(*[at(num, a) for a in quad])
        if rn in PAIR_ATOMS:
            T[:, i, 6] = dihedral(*[at(num, a) for a in PAIR_ATOMS[rn]])
    return T


def far_pairs(top):
    """Index arrays (i, j) of the heavy-atom pairs more than three bonds apart."""
    n = top.n_atoms
    near = np.eye(n, dtype=int)
    bonded = np.zeros((n, n), int)
    for a, b in top.bonds():
        bonded[a, b] = bonded[b, a] = 1
    for _ in range(3):
        near = ((near + near @ bonded) > 0).astype(int)
    i, j = np.where(np.triu(near == 0, 1))
    return i, j


def closest(X, pairs, chunk=20000):
    """Per frame, the shortest distance between two atoms more than three bonds apart."""
    i, j = pairs
    out = np.zeros(len(X))
    for s in range(0, len(X), chunk):
        x = X[s:s + chunk].astype(np.float32)
        out[s:s + chunk] = np.sqrt(((x[:, i] - x[:, j]) ** 2).sum(-1).min(1))
    return out


def write_pdb(path, top, x):
    """One frame (atoms, 3) as a PDB file with CONECT records."""
    with open(path, "w") as f:
        for k, ((name, res, num, el), p) in enumerate(zip(top.atoms, x)):
            label = f" {name:<3s}" if len(name) < 4 else name
            f.write(f"ATOM  {k + 1:5d} {label} {res:3s} A{num:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {el:>2s}\n")
        for a, b in top.bonds():
            f.write(f"CONECT{a + 1:5d}{b + 1:5d}\n")
        f.write("END\n")


def write_dcd(path, X, frame_ps=10.0):
    """Frames (frames, atoms, 3) in angstrom as a CHARMM/NAMD DCD file, which VMD, PyMOL, MDTraj and MDAnalysis
    read with the PDB file as topology."""
    X = np.asarray(X, np.float32)
    F, n = X.shape[:2]
    with open(path, "wb") as f:
        head = struct.pack("<4s9if10i", b"CORD", F, 1, 1, F, 0, 0, 0, 0, 0, frame_ps / 0.04888821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24)
        f.write(struct.pack("<i", len(head)) + head + struct.pack("<i", len(head)))
        title = struct.pack("<i80s", 1, b"peptide-dynamics-from-sequence".ljust(80))
        f.write(struct.pack("<i", len(title)) + title + struct.pack("<i", len(title)))
        f.write(struct.pack("<iii", 4, n, 4))
        for frame in X:
            for axis in range(3):
                f.write(struct.pack("<i", 4 * n) + frame[:, axis].tobytes() + struct.pack("<i", 4 * n))


def read_dcd(path):
    """Coordinates (frames, atoms, 3) of a DCD file written by write_dcd."""
    raw = open(path, "rb").read()
    F = struct.unpack_from("<i", raw, 8)[0]
    pos = 4 + 84 + 4
    pos += 4 + struct.unpack_from("<i", raw, pos)[0] + 4
    n = struct.unpack_from("<i", raw, pos + 4)[0]
    pos += 12
    X = np.zeros((F, n, 3), np.float32)
    for k in range(F):
        for axis in range(3):
            X[k, :, axis] = np.frombuffer(raw, "<f4", n, pos + 4)
            pos += 8 + 4 * n
    return X
