"""Convert molecular trajectories into local states and torsion features.

A torsion table contains residue names in ``res`` and an array ``tors`` of
shape (frames, residues, 7), in radians. Its columns are phi, psi, chi1 to
chi4, then an equivalent-atom partner torsion for Val chi1 or Leu chi2.
Missing angles are NaN. ACE and NME caps are omitted from the residue list.

SiteDefs learns backbone and rotamer basins from training tables. Bank
stores residue torsions conditional on the resulting local states;
features_from_tors supplies the symmetry-aware features used for evaluation.
"""
import numpy as np
import mdtraj as md

import sites as S

PAIR = {("VAL", 1): ("CG1", "CG2"), ("LEU", 2): ("CD1", "CD2")}
CHI_FN = [md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_chi4]


def torsion_table(traj):
    """Extract backbone, side-chain and equivalent-atom torsions with MDTraj."""
    top = traj.topology
    res = [r for r in top.residues if r.name not in ("ACE", "NME")]
    rpos = {r.index: k for k, r in enumerate(res)}
    T = np.full((traj.n_frames, len(res), 7), np.nan, dtype=np.float32)
    for col, fn in ((0, md.compute_phi), (1, md.compute_psi)):
        idx, ang = fn(traj)
        for q, a in zip(idx, ang.T):
            r = top.atom(int(q[1])).residue.index
            if r in rpos:
                T[:, rpos[r], col] = a
    for k, fn in enumerate(CHI_FN, start=1):
        idx, ang = fn(traj)
        for q, a in zip(idx, ang.T):
            r = top.atom(int(q[1])).residue
            if r.index not in rpos:
                continue
            T[:, rpos[r.index], 1 + k] = a
            if (r.name, k) in PAIR:
                a1, a2 = PAIR[(r.name, k)]
                last = top.atom(int(q[3])).name
                other = a2 if last == a1 else a1
                o = [x.index for x in r.atoms if x.name == other]
                # Include the equivalent atom so evaluation cannot depend on its label.
                q2 = np.array([[q[0], q[1], q[2], o[0]]])
                T[:, rpos[r.index], 6] = md.compute_dihedrals(traj, q2)[:, 0]
    return dict(res=[r.name for r in res], tors=T)


def site_list(resnames, tors_sample):
    """List (residue index, site kind) pairs for the torsions present in a frame."""
    out = []
    for i, rn in enumerate(resnames):
        out.append((i, "bb"))
        for k in range(1, 5):
            if np.isfinite(tors_sample[i, 1 + k]):
                out.append((i, f"chi{k}"))
    return out


def bb_kind(tors_sample, i):
    """Identify whether a residue has both backbone angles, phi only or psi only."""
    phi_ok, psi_ok = np.isfinite(tors_sample[i, 0]), np.isfinite(tors_sample[i, 1])
    return "2d" if (phi_ok and psi_ok) else ("psi" if psi_ok else "phi")


def site_key(rn, kind, bbk):
    """Key a local state definition by residue type, torsion and terminal geometry."""
    return (rn, kind if kind != "bb" else "bb-" + bbk)


class SiteDefs:
    """Backbone and rotamer basin definitions shared by each residue type."""

    def __init__(self):
        self.defs = {}

    def fit(self, tables, max_frames=400000, seed=0):
        """Fit basins to at most 20,000 sampled frames per table and max_frames per key."""
        rng = np.random.default_rng(seed)
        pool = {}
        for tb in tables:
            T = tb["tors"]
            sub = T[rng.choice(len(T), size=min(len(T), 20000), replace=False)]
            for i, rn in enumerate(tb["res"]):
                bbk = bb_kind(T[0], i)
                key = site_key(rn, "bb", bbk)
                pool.setdefault(key, []).append(sub[:, i, :2])
                for k in range(1, 5):
                    if np.isfinite(T[0, i, 1 + k]):
                        pool.setdefault(site_key(rn, f"chi{k}", bbk), []).append(sub[:, i, 1 + k:2 + k])
        for key, arrs in pool.items():
            A = np.concatenate(arrs)[:max_frames]
            rn, kind = key
            if kind == "bb-2d":
                self.defs[key] = S.fit_2d(A[:, 0], A[:, 1])
            elif kind == "bb-psi":
                self.defs[key] = S.fit_1d(A[:, 1])
            elif kind == "bb-phi":
                self.defs[key] = S.fit_1d(A[:, 0])
            else:
                per = np.pi if (rn, int(kind[3])) in S.SYMMETRIC else 2 * np.pi
                self.defs[key] = S.fit_1d(A[:, 0], per)
        return self

    def n_states(self, key):
        return self.defs[key]["n"]

    def site_traj(self, tb, return_core=False):
        """Assign a torsion table to local states using the most recently visited core.

        Return labels (frames, sites), the site list and definition keys. With
        return_core=True, also return a Boolean mask of visits to the assigned core."""
        T = tb["tors"]
        sl = site_list(tb["res"], T[0])
        L = np.zeros((len(T), len(sl)), np.int16)
        Cf = np.zeros((len(T), len(sl)), bool)
        keys = []
        for c, (i, kind) in enumerate(sl):
            rn = tb["res"][i]
            bbk = bb_kind(T[0], i)
            key = site_key(rn, kind, bbk)
            keys.append(key)
            d = self.defs[key]
            if kind == "bb":
                if bbk == "2d":
                    lab, core = S.raw_labels(d, T[:, i, 0], T[:, i, 1])
                elif bbk == "psi":
                    lab, core = S.raw_labels(d, T[:, i, 1])
                else:
                    lab, core = S.raw_labels(d, T[:, i, 0])
            else:
                lab, core = S.raw_labels(d, T[:, i, 1 + int(kind[3])])
            L[:, c] = S.core_fill(lab, core)
            Cf[:, c] = core & (lab == L[:, c])
        if return_core:
            return L, sl, keys, Cf
        return L, sl, keys


def features_from_tors(resnames, T):
    """Return cosine/sine feature pairs in residue and torsion order.

    Equivalent Val and Leu atoms contribute summed trigonometric features;
    twofold symmetric side chains use twice the angle. Shape: (frames, features)."""
    cols = []
    for i, rn in enumerate(resnames):
        for c in (0, 1):
            if np.isfinite(T[0, i, c]):
                cols += [np.cos(T[:, i, c]), np.sin(T[:, i, c])]
        for k in range(1, 5):
            a = T[:, i, 1 + k]
            if not np.isfinite(T[0, i, 1 + k]):
                continue
            if (rn, k) in PAIR:
                b = T[:, i, 6]
                cols += [np.cos(a) + np.cos(b), np.sin(a) + np.sin(b)]
            elif (rn, k) in S.SYMMETRIC:
                cols += [np.cos(2 * a), np.sin(2 * a)]
            else:
                cols += [np.cos(a), np.sin(a)]
    return np.stack(cols, 1).astype(np.float32)


class Bank:
    """Training torsions indexed by residue type, backbone geometry and local states."""

    def __init__(self, max_per=400, seed=0):
        self.max_per, self.rng = max_per, np.random.default_rng(seed)
        self.bank = {}

    def add(self, tb, L, sl, keys, Cf):
        """Collect torsions only where every site of the residue occupies its core."""
        T = tb["tors"]
        by_res = {}
        for c, (i, kind) in enumerate(sl):
            by_res.setdefault(i, []).append(c)
        for i, cols in by_res.items():
            allcore = Cf[:, cols].all(1)
            if not allcore.any():
                continue
            idx = np.where(allcore)[0]
            lab = L[idx][:, cols]
            rn = tb["res"][i]
            bbk = bb_kind(T[0], i)
            uniq, inv = np.unique(lab, axis=0, return_inverse=True)
            for u, row in enumerate(uniq):
                key = (rn, bbk, tuple(int(v) for v in row))
                sel = idx[inv.ravel() == u]
                if len(sel) > self.max_per:
                    sel = self.rng.choice(sel, self.max_per, replace=False)
                prev = self.bank.get(key)
                arr = T[sel, i, :]
                self.bank[key] = arr if prev is None else np.concatenate([prev, arr])[-4 * self.max_per:]

    def sample(self, rn, bbk, states, n):
        """Draw n torsion rows, or return None when this state combination is absent."""
        key = (rn, bbk, tuple(int(v) for v in states))
        arr = self.bank.get(key)
        if arr is None:
            return None
        return arr[self.rng.integers(0, len(arr), size=n)]


def feature_labels(resnames, T0):
    """Name each cosine/sine feature pair in features_from_tors order."""
    out = []
    for i, rn in enumerate(resnames):
        for c, nm in ((0, "phi"), (1, "psi")):
            if np.isfinite(T0[i, c]):
                out.append(f"{nm}{i}")
        for k in range(1, 5):
            if np.isfinite(T0[i, 1 + k]):
                out.append(f"chi{k}_{i}")
    return out


class SnippetBank:
    """Store contiguous training segments during which a residue keeps its local states.

    This optional emission method retains within-segment temporal correlations.
    The released generator uses independent torsion draws from Bank instead."""

    def __init__(self, stride=1, max_frames=200000, seed=0):
        self.stride, self.max_frames = stride, max_frames
        self.rng = np.random.default_rng(seed)
        self.seg = {}
        self.count = {}

    def add(self, tb, L, sl, keys):
        T = tb["tors"]
        by_res = {}
        for c, (i, kind) in enumerate(sl):
            by_res.setdefault(i, []).append(c)
        for i, cols in by_res.items():
            lab = L[:, cols]
            change = np.any(lab[1:] != lab[:-1], axis=1)
            starts = np.concatenate([[0], np.where(change)[0] + 1])
            ends = np.concatenate([starts[1:], [len(lab)]])
            rn, bbk = tb["res"][i], bb_kind(T[0], i)
            for s, e in zip(starts, ends):
                key = (rn, bbk, tuple(int(v) for v in lab[s]))
                if self.count.get(key, 0) >= self.max_frames:
                    continue
                piece = T[s:e:self.stride, i, :]
                self.seg.setdefault(key, []).append(piece.astype(np.float32))
                self.count[key] = self.count.get(key, 0) + len(piece)

    def draw(self, key, n):
        """Join randomly selected stored segments until n frames are available."""
        pieces = self.seg.get(key)
        if not pieces:
            return None
        out, got = [], 0
        while got < n:
            p = pieces[self.rng.integers(len(pieces))]
            if len(p) > n - got:
                s = self.rng.integers(0, len(p) - (n - got) + 1)
                p = p[s:s + n - got]
            out.append(p)
            got += len(p)
        return np.concatenate(out)[:n]


def emit_snippets(sbank, fallback, resnames, bbks, sl, frames, every):
    """Emit (frames, residues, 7) torsions from constant-state segments.

    The spacing multiplier every converts stored segments to the requested
    output interval. Missing segments fall back to independent Bank draws."""
    n = len(frames)
    T = np.full((n, len(resnames), 7), np.nan, np.float32)
    by_res = {}
    for c, (i, kind) in enumerate(sl):
        by_res.setdefault(i, []).append(c)
    for i, cols in by_res.items():
        st = frames[:, cols]
        change = np.any(st[1:] != st[:-1], axis=1)
        starts = np.concatenate([[0], np.where(change)[0] + 1])
        ends = np.concatenate([starts[1:], [n]])
        for s, e in zip(starts, ends):
            key = (resnames[i], bbks[i], tuple(int(v) for v in st[s]))
            seg = sbank.draw(key, (e - s) * every)
            if seg is not None:
                T[s:e, i] = seg[::every][:e - s]
            else:
                arr = fallback.sample(resnames[i], bbks[i], st[s], e - s)
                if arr is None:
                    cands = [k for k in fallback.bank if k[0] == resnames[i] and k[1] == bbks[i]]
                    k = cands[fallback.rng.integers(len(cands))]
                    arr = fallback.bank[k][fallback.rng.integers(0, len(fallback.bank[k]), size=e - s)]
                T[s:e, i] = arr
    return T
