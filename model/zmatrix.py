"""Rebuild heavy-atom coordinates from a template and named torsion angles.

The template fixes bond lengths, bond angles and offsets between dependent
dihedrals. Natural extension reference frame (NeRF) placement reconstructs
atoms from three already placed neighbours, in NumPy batches. Coordinates
use nm and torsions use radians. This is a geometric reconstruction with
no energy minimisation or explicit ring-closure constraint.
"""
import io
import numpy as np

CHI_BONDS = {
    "ARG": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE")],
    "ASN": [("CA", "CB"), ("CB", "CG")], "ASP": [("CA", "CB"), ("CB", "CG")],
    "CYS": [("CA", "CB")], "GLN": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")],
    "GLU": [("CA", "CB"), ("CB", "CG"), ("CG", "CD")], "HIS": [("CA", "CB"), ("CB", "CG")],
    "ILE": [("CA", "CB"), ("CB", "CG1")], "LEU": [("CA", "CB"), ("CB", "CG")],
    "LYS": [("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE")], "MET": [("CA", "CB"), ("CB", "CG"), ("CG", "SD")],
    "PHE": [("CA", "CB"), ("CB", "CG")], "PRO": [("CA", "CB"), ("CB", "CG")], "SER": [("CA", "CB")],
    "THR": [("CA", "CB")], "TRP": [("CA", "CB"), ("CB", "CG")], "TYR": [("CA", "CB"), ("CB", "CG")],
    "VAL": [("CA", "CB")],
}


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return np.arctan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))


class ZMatrix:
    """Define atom placement from an MDTraj topology and template coordinates (atoms, 3) in nm."""

    def __init__(self, top, xyz):
        self.top = top
        n = top.n_atoms
        nbr = [[] for _ in range(n)]
        for a, b in top.bonds:

            # Removing this traversal edge lets chi2 place CD; ring closure is not enforced.
            if a.residue.name == "PRO" and b.residue == a.residue and {a.name, b.name} == {"N", "CD"}:
                continue
            nbr[a.index].append(b.index)
            nbr[b.index].append(a.index)

        self.bond_label = {}
        res = list(top.residues)
        for k, r in enumerate(res):
            if r.name in ("ACE", "NME"):
                continue
            at = {a.name: a.index for a in r.atoms}
            if k > 0:
                self.bond_label[(at["N"], at["CA"])] = f"phi{r.index}"
            if k < len(res) - 1:
                self.bond_label[(at["CA"], at["C"])] = f"psi{r.index}"
            for c, (a1, a2) in enumerate(CHI_BONDS.get(r.name, []), start=1):
                if a1 in at and a2 in at:
                    self.bond_label[(at[a1], at[a2])] = f"chi{c}_{r.index}"
        for k in range(len(res) - 1):
            ra, rb = res[k], res[k + 1]
            ca = {a.name: a.index for a in ra.atoms}
            cb = {a.name: a.index for a in rb.atoms}
            self.bond_label[(ca["C"], cb["N"])] = f"omega{ra.index}"

        # Traverse from the initial residue using a stack to keep each branch contiguous.
        order, parent = [], {}
        start = [a.index for a in res[0].atoms]
        ch3 = [a.index for a in res[0].atoms if a.name in ("CH3", "CA")][0]
        seen = {ch3}
        queue = [ch3]
        while queue:
            u = queue.pop()
            order.append(u)
            for v in sorted(nbr[u], reverse=True):
                if v not in seen:
                    seen.add(v)
                    parent[v] = u
                    queue.append(v)
        assert len(order) == n, "disconnected topology"
        pos_in = {a: i for i, a in enumerate(order)}
        self.order = np.array(order)
        refs = np.full((n, 3), -1)
        for a in order[1:]:
            p = parent[a]
            g = parent.get(p)
            if g is None:
                cands = [x for x in nbr[p] if pos_in[x] < pos_in[a] and x != a]
                g = cands[0] if cands else None
            if g is None:
                refs[a] = [p, -1, -1]
                continue
            gg = parent.get(g)
            if gg is None or gg == a or gg == p:
                cands = [x for x in nbr[g] if x != p and pos_in[x] < pos_in[a]]
                if not cands:
                    cands = [x for x in nbr[p] if x != g and pos_in[x] < pos_in[a]]
                gg = cands[0] if cands else -1
            refs[a] = [p, g, gg]
        self.refs = refs
        X = np.asarray(xyz, float)
        self.bond = np.zeros(n)
        self.angle = np.zeros(n)
        self.dih = np.zeros(n)
        self.driver = [None] * n
        for a in order[1:]:
            p, g, gg = refs[a]
            self.bond[a] = np.linalg.norm(X[a] - X[p])
            if g >= 0:
                u1, u2 = X[a] - X[p], X[g] - X[p]
                self.angle[a] = np.arccos(np.clip(np.dot(u1, u2) / np.linalg.norm(u1) / np.linalg.norm(u2), -1, 1))
            if g >= 0 and gg >= 0:
                self.dih[a] = _dihedral(X[a], X[p], X[g], X[gg])
                lab = self.bond_label.get((g, p)) or self.bond_label.get((p, g))
                self.driver[a] = lab

        self.template = X
        self.labels = sorted({d for d in self.driver if d})
        # Match placement dihedrals to the named torsions measured on the template.
        tmpl_t = self.named_torsions(X[None])[0]
        self.offset = np.zeros(n)
        for a in order[1:]:
            if self.driver[a]:
                self.offset[a] = self.dih[a] - tmpl_t[self.labels.index(self.driver[a])]
        self.first3 = order[:3]

    def torsion_quads(self):
        """Map torsion labels to atom quadruplets using backbone and side-chain connectivity."""
        q = {}
        res = list(self.top.residues)
        for k, r in enumerate(res):
            at = {a.name: a.index for a in r.atoms}
            if r.name in ("ACE", "NME"):
                continue
            if k > 0:
                prev = {a.name: a.index for a in res[k - 1].atoms}
                q[f"phi{r.index}"] = (prev["C"], at["N"], at["CA"], at["C"])
            if k < len(res) - 1:
                nxt = {a.name: a.index for a in res[k + 1].atoms}
                q[f"psi{r.index}"] = (at["N"], at["CA"], at["C"], nxt["N"])
            chain = ["N", "CA"] + [b for _, b in CHI_BONDS.get(r.name, [])]
            for c in range(1, len(CHI_BONDS.get(r.name, [])) + 1):
                a0, a1, a2 = chain[c - 1], chain[c], chain[c + 1]

                nb = [b for b in self._nbrs(at[a2]) if b not in (at[a1],)]
                if nb:
                    q[f"chi{c}_{r.index}"] = (at[a0], at[a1], at[a2], min(nb))
        for k in range(len(res) - 1):
            ra, rb = res[k], res[k + 1]
            ca = {a.name: a.index for a in ra.atoms}
            cb = {a.name: a.index for a in rb.atoms}
            a0 = ca.get("CA", ca.get("CH3"))
            b0 = cb.get("CA", cb.get("C"))
            q[f"omega{ra.index}"] = (a0, ca["C"], cb["N"], b0)
        return q

    def _nbrs(self, i):
        return [b.index if a.index == i else a.index for a, b in self.top.bonds if i in (a.index, b.index)]

    def named_torsions(self, X):
        """Extract radians from coordinate frames in self.labels order."""
        q = self.torsion_quads()
        out = np.zeros((len(X), len(self.labels)))
        for j, lab in enumerate(self.labels):
            a, b, c, d = q[lab]
            out[:, j] = _batch_dihedral(X[:, a], X[:, b], X[:, c], X[:, d])
        return out

    def build(self, tors):
        """Map torsions (frames, labels) in radians to coordinates (frames, atoms, 3) in nm."""
        tors = np.atleast_2d(tors)
        F, n = len(tors), self.top.n_atoms
        X = np.zeros((F, n, 3))
        o = self.order
        X[:, o[0]] = self.template[o[0]]
        X[:, o[1]] = self.template[o[1]]
        X[:, o[2]] = self.template[o[2]]
        lab_idx = {l: i for i, l in enumerate(self.labels)}
        for a in o[3:]:
            p, g, gg = self.refs[a]
            if gg < 0:
                X[:, a] = self.template[a] - self.template[p] + X[:, p]
                continue
            d = self.dih[a] if self.driver[a] is None else tors[:, lab_idx[self.driver[a]]] + self.offset[a]
            X[:, a] = _nerf(X[:, gg], X[:, g], X[:, p], self.bond[a], self.angle[a], d)
        return X


def _batch_dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    return np.arctan2(y, x)


def _nerf(a, b, c, bond, angle, dih):
    """Place an atom from three neighbours, a bond length, a bond angle and a dihedral."""
    bc = c - b
    bc /= np.linalg.norm(bc, axis=-1, keepdims=True)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    m = np.cross(n, bc)
    dih = np.broadcast_to(dih, a.shape[:-1])
    d2 = np.stack([-bond * np.cos(angle) * np.ones_like(dih), bond * np.sin(angle) * np.cos(dih),
                   bond * np.sin(angle) * np.sin(dih)], axis=-1)
    return c + d2[..., 0:1] * bc + d2[..., 1:2] * m + d2[..., 2:3] * n
