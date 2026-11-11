"""Write pepdyn/residues.json: the internal coordinates of the heavy atoms of the 20 residues.

They are measured on the corpus's own starting structures (ManyPeptidesMD, pdb_tarfiles/test.tar: AMBER template
geometry). Every side-chain atom is stored as (name, parent, grandparent, great-grandparent, bond length, bond
angle, driver, offset): it sits at that bond length from its parent, at that angle to the grandparent, and at
dihedral (great-grandparent, grandparent, parent, atom) = driver torsion + offset, or = offset where no named
torsion turns the bond grandparent-parent. The named torsions follow mdtraj's definitions of chi1 to chi4. The
second methyl of an equivalent pair (Val CG2, Leu CD2) is driven by its own torsion, which the torsion bank holds.
Proline's CD also records the length of the bond that closes its ring.

Usage: python tools/make_residue_library.py PDB_DIR [OUT.json]
"""
import glob
import json
import os
import sys

import numpy as np

CHI = {  # the four atoms that define chi1 to chi4, first match wins
    1: [("N", "CA", "CB", "CG"), ("N", "CA", "CB", "CG1"), ("N", "CA", "CB", "SG"), ("N", "CA", "CB", "OG"),
        ("N", "CA", "CB", "OG1")],
    2: [("CA", "CB", "CG", "CD"), ("CA", "CB", "CG", "CD1"), ("CA", "CB", "CG1", "CD1"), ("CA", "CB", "CG", "OD1"),
        ("CA", "CB", "CG", "ND1"), ("CA", "CB", "CG", "SD")],
    3: [("CB", "CG", "CD", "NE"), ("CB", "CG", "CD", "CE"), ("CB", "CG", "CD", "OE1"), ("CB", "CG", "SD", "CE")],
    4: [("CG", "CD", "NE", "CZ"), ("CG", "CD", "CE", "NZ")],
}
PAIR = {"VAL": "CG2", "LEU": "CD2"}
ORDER = ["N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2", "ND1", "ND2", "OD1",
         "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1", "OE2", "CZ", "CZ2", "CZ3", "NZ", "CH2",
         "NH1", "NH2", "OH"]


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1n) * b1n, b2 - np.dot(b2, b1n) * b1n
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))))


def angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))))


def wrap(x):
    return float((x + 180.0) % 360.0 - 180.0)


def read_pdb(path):
    """Residues in order: list of (name, {atom: xyz}) with heavy atoms only."""
    out = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            name, res, num = line[12:16].strip(), line[17:20], int(line[22:26])
            if line[76:78].strip() == "H" or name.lstrip("0123456789").startswith("H"):
                continue
            if not out or out[-1][2] != num:
                out.append(("HIS" if res in ("HIE", "HID", "HIP") else res, {}, num))
            out[-1][1][name] = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return [(r, a) for r, a, _ in out]


def side_chain(name, X):
    side = [n for n in X if n not in ("N", "CA", "C", "O", "OXT")]
    parent, order, seen, queue = {"CA": "N", "N": "C"}, [], {"N", "CA", "C", "O"}, ["CA"]
    while queue:                                   # breadth-first from CA over bonds
        u = queue.pop(0)
        for v in sorted(side, key=ORDER.index):
            if v not in seen and np.linalg.norm(X[v] - X[u]) < 1.95 and not (name == "PRO" and {u, v} == {"N", "CD"}):
                seen.add(v)
                parent[v] = u
                order.append(v)
                queue.append(v)
    assert len(order) == len(side), (name, order, side)
    bond_chi, chi_val, chi_atoms = {}, {}, {}
    for k, quads in CHI.items():
        for q in quads:
            if all(a in X for a in q) and k not in chi_val:
                bond_chi[(q[1], q[2])] = k
                chi_val[k] = dihedral(*[X[a] for a in q])
                chi_atoms[f"chi{k}"] = list(q)
    atoms = []
    for a in order:
        p = parent[a]
        g = parent[p]
        gg = parent[g]
        d = dihedral(X[gg], X[g], X[p], X[a])
        k = bond_chi.get((g, p))
        entry = dict(name=a, refs=[p, g, gg], bond=float(np.linalg.norm(X[a] - X[p])), angle=angle(X[g], X[p], X[a]))
        if PAIR.get(name) == a:
            entry.update(driver="pair", offset=0.0)
        elif k is not None:
            entry.update(driver=f"chi{k}", offset=wrap(d - chi_val[k]))
        else:
            entry.update(driver=None, offset=wrap(d))
        if name == "PRO" and a == "CD":            # the ring closes on N: the builder keeps this bond length too
            entry["closes"] = ["N", float(np.linalg.norm(X["CD"] - X["N"]))]
        atoms.append(entry)
    return atoms, chi_atoms


def main():
    pdb_dir = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "pepdyn", "residues.json")
    lib, bb, cterm = {}, {}, []
    for f in sorted(glob.glob(os.path.join(pdb_dir, "*.pdb"))):
        res = read_pdb(f)
        if len(res) != 4:
            continue
        for i, (name, X) in enumerate(res):
            if name not in lib:
                side, chi = side_chain(name, X)
                heavy = [a for a in X if a != "OXT"]
                bonds = sorted(sorted([a, b], key=ORDER.index) for k, a in enumerate(heavy) for b in heavy[k + 1:]
                               if np.linalg.norm(X[a] - X[b]) < 1.95)
                lib[name] = dict(side=side, chi=chi, bonds=bonds)
            b = bb.setdefault(name, {})
            b.setdefault("N_CA", []).append(np.linalg.norm(X["CA"] - X["N"]))
            b.setdefault("CA_C", []).append(np.linalg.norm(X["C"] - X["CA"]))
            b.setdefault("C_O", []).append(np.linalg.norm(X["O"] - X["C"]))
            b.setdefault("N_CA_C", []).append(angle(X["N"], X["CA"], X["C"]))
            b.setdefault("CA_C_O", []).append(angle(X["CA"], X["C"], X["O"]))
            if i > 0:
                P = res[i - 1][1]
                b.setdefault("C_N", []).append(np.linalg.norm(X["N"] - P["C"]))
                b.setdefault("CA_C_N", []).append(angle(P["CA"], P["C"], X["N"]))
                b.setdefault("C_N_CA", []).append(angle(P["C"], X["N"], X["CA"]))
            if "OXT" in X:
                cterm.append([np.linalg.norm(X["OXT"] - X["C"]), angle(X["CA"], X["C"], X["OXT"]),
                              np.linalg.norm(X["O"] - X["C"]), angle(X["CA"], X["C"], X["O"]),
                              wrap(dihedral(X["N"], X["CA"], X["C"], X["OXT"]) - dihedral(X["N"], X["CA"], X["C"], X["O"]))])
    assert len(lib) == 20, sorted(lib)
    for name in lib:
        lib[name]["backbone"] = {k: float(np.mean(v)) for k, v in bb[name].items()}
        lib[name]["atoms"] = ["N", "CA", "C", "O"] + sorted([a["name"] for a in lib[name]["side"]], key=ORDER.index)
    c = np.mean(cterm, 0)
    lib["C_TERMINUS"] = dict(C_OXT=float(c[0]), CA_C_OXT=float(c[1]), C_O=float(c[2]), CA_C_O=float(c[3]),
                             OXT_from_O=float(np.median([r[4] for r in cterm])))
    json.dump(dict(sorted(lib.items())), open(out, "w"), indent=1)
    for name, r in sorted(lib.items()):
        if name == "C_TERMINUS":
            print(name, {k: round(v, 3) for k, v in r.items()})
            continue
        print(name, {k: round(v, 3) for k, v in r["backbone"].items()})
        for a in r["side"]:
            print("   ", a["name"], a["refs"], round(a["bond"], 3), round(a["angle"], 1), a["driver"], round(a["offset"], 1))


if __name__ == "__main__":
    main()
