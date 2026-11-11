"""The model: a sequence-conditioned Markov jump process over local conformational states.

A peptide is described by one small integer per site. Every residue has a backbone site (the basin of its
(phi, psi) map) and one site per side-chain torsion (its rotamer). A configuration x holds the state of every
site, and a move changes one site.

    energy   E(x) = sum over residues of e(window of five residue tokens around it)          (in kT)
    rate     k(x -> x') = exp( g - (E(x') - E(x)) / 2 )

g is computed from the window with the moving site's own state removed, and from a code that is symmetric in the
pair (state left, state entered). So g is the same for a move and its reverse, k(x -> x') / k(x' -> x) equals
exp(-(E(x') - E(x))), detailed balance holds exactly and the equilibrium distribution is exp(-E) / Z.

A residue token is the sum of an embedding of the residue type and the embeddings of the current states of its
sites. Termini are tokens too. Only numpy is needed.
"""
import functools
import json

import numpy as np

PAD, N_TERMINUS, C_TERMINUS = 0, 1, 2            # special tokens; residue types follow from 5
ONE_TO_THREE = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE",
                    L="LEU", K="LYS", M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _quiet(f):
    """Run f without floating-point warnings: some numpy builds raise spurious ones in matrix products, and the
    rate of a move that does not exist is meant to be exp(-inf) = 0."""
    @functools.wraps(f)
    def g(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return f(*args, **kwargs)
    return g


class Model:
    """Weights, state alphabet and torsion bank of one trained model (a .npz written by train/export.py)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.w = {k[2:]: z[k].astype(np.float64) for k in z.files if k.startswith("w_")}
        self.w32 = {k: v.astype(np.float32) for k, v in self.w.items()}
        m = self.meta
        self.name = m["name"]
        self.W, self.g_bound, self.dt_ps = int(m["W"]), float(m["g_bound"]), float(m["dt_ps"])
        self.beta = float(m["t_ref_K"]) / float(m["temperature_K"])
        self.types = m["residue_types"]
        self.max_v = int(m["max_v"])
        self.site_n = {tuple(k): n for k, n in zip(m["site_keys"], m["site_n"])}
        self.site_first = {tuple(k): i for k, i in zip(m["site_keys"], m["site_first_id"])}
        # basin grids of every site: label and core flag per bin of the torsion (or of the (phi, psi) map)
        self.grid = {}
        for kind in ("grid2", "grid1"):
            for j, i in enumerate(z[kind + "_key"]):
                self.grid[tuple(m["site_keys"][int(i)])] = dict(
                    lab=z[kind + "_lab"][j], core=z[kind + "_core"][j], F=z[kind + "_F"][j],
                    period=float(z["grid1_period"][j]) if kind == "grid1" else None)
        # torsion bank: for every (residue, backbone kind, states of its sites), torsions seen in training MD
        self.bank_keys = [(r, b, tuple(s)) for r, b, s in m["bank_keys"]]
        self.bank_index = {k: i for i, k in enumerate(self.bank_keys)}
        self._bank, self._bank_offsets = z["bank"], z["bank_offsets"]

    def bank_rows(self, i):
        """Torsions (radians; NaN where the residue has none) of bank entry i: columns phi, psi, chi1 to chi4 and
        the partner of an equivalent methyl pair (Val CG2 on chi1, Leu CD2 on chi2)."""
        q = self._bank[self._bank_offsets[i]:self._bank_offsets[i + 1]]
        return np.where(q == -32768, np.nan, q.astype(np.float64) * (np.pi / 32767.0))

    def peptide(self, sequence):
        return Peptide(self, sequence)


class Peptide:
    """One sequence indexed for the network: its sites, their sizes and the global ids of their states."""

    def __init__(self, model, sequence):
        seq = sequence.upper()
        bad = sorted(set(seq) - set(ONE_TO_THREE))
        if bad or not seq:
            raise ValueError(f"sequence must use the 20 standard one-letter codes, got {sequence!r}")
        if len(seq) < 2:
            raise ValueError("a peptide needs at least two residues")
        self.sequence = seq
        self.residues = [ONE_TO_THREE[c] for c in seq]
        n = len(seq)
        # the termini are charged: the first residue has no phi and the last no psi, so their backbone site is
        # the basin of one torsion instead of the (phi, psi) map
        self.bb_kinds = ["2d" if 0 < i < n - 1 else ("psi" if i == 0 else "phi") for i in range(n)]
        self.sites, keys = [], []
        for i, rn in enumerate(self.residues):
            self.sites.append((i, "bb"))
            keys.append((rn, "bb-" + self.bb_kinds[i]))
            for c in range(1, 5):
                if (rn, f"chi{c}") in model.site_n:
                    self.sites.append((i, f"chi{c}"))
                    keys.append((rn, f"chi{c}"))
        self.keys = keys
        self.S, self.R = len(keys), n + 2
        self.site_n = np.array([model.site_n[k] for k in keys])
        self.site_res = np.array([i + 1 for i, _ in self.sites])                 # residue slot, N-terminus at 0
        self.site_kind = np.array([0 if k == "bb" else int(k[3]) for _, k in self.sites])
        self.site_ids = np.zeros((self.S, model.max_v), np.int64)
        for c, k in enumerate(keys):
            self.site_ids[c, :model.site_n[k]] = model.site_first[k] + np.arange(model.site_n[k])
        self.types = np.array([N_TERMINUS] + [model.types[r] for r in self.residues] + [C_TERMINUS])
        self.moves = np.array([(c, v) for c in range(self.S) for v in range(self.site_n[c])])   # (site, state entered)
        self.n_configs = int(np.prod(self.site_n.astype(object)))


def _slots(tok, Wt):
    """Every residue token through the first-layer weights of every window slot: (B, R, d), (L, d, h) -> (B, R, L, h)."""
    return np.tensordot(tok, Wt, axes=([2], [1]))


def _window_sum(P, W):
    """pre[b, j] = sum over slots l of P[b, j + l - W, l] (nothing beyond the ends)."""
    B, R, L, h = P.shape
    out = np.zeros((B, R, h), P.dtype)
    for l in range(L):
        off = l - W
        lo, hi = max(0, -off), min(R, R - off)
        out[:, lo:hi] += P[:, lo + off:hi + off, l]
    return out


def _pad_proj(pad, Wt, R, W):
    """What the padding tokens beyond both ends add to each residue's window."""
    out = np.zeros((R, Wt.shape[2]), Wt.dtype)
    for l in range(2 * W + 1):
        off = l - W
        for j in range(R):
            if not 0 <= j + off < R:
                out[j] += pad @ Wt[l]
    return out


def _head(x, w1, b1, w2, b2):
    return _silu(_silu(x) @ w1.T + b1) @ w2.T + b2


def _energy_head(model, pre, w=None):
    """(enthalpy-like, entropy-like) -> E / kT = h * T_ref / T - s."""
    w = model.w if w is None else w
    o = _head(pre, w["E2_1_weight"], w["E2_1_bias"], w["E2_3_weight"], w["E2_3_bias"])
    return o[..., 0] * model.beta - o[..., 1]


def _prefactor_head(model, pre, w=None):
    """Log-prefactor a - b * T_ref / T."""
    w = model.w if w is None else w
    o = _head(pre, w["G2_1_weight"], w["G2_1_bias"], w["G2_3_weight"], w["G2_3_bias"])
    return o[..., 0] - o[..., 1] * model.beta


def _tokens(model, pep, cur, w=None):
    w = model.w if w is None else w
    B = cur.shape[0]
    cur_ids = pep.site_ids[np.arange(pep.S), cur]                                # (B, S)
    sid = np.zeros((B, pep.R, 5), np.int64)
    sid[:, pep.site_res, pep.site_kind] = cur_ids
    tok = w["type_emb_weight"][pep.types][None] + w["state_emb_weight"][sid].sum(-2)
    return tok, cur_ids


@_quiet
def energy(model, pep, cur, chunk=2048):
    """E(x) / kT of configurations cur (B, S)."""
    w, W = model.w, model.W
    cur = np.asarray(cur, np.int64)
    out = np.zeros(len(cur))
    pad = _pad_proj(w["type_emb_weight"][PAD], w["E1"], pep.R, W) + w["E1b"]
    for s in range(0, len(cur), chunk):
        tok, cur_ids = _tokens(model, pep, cur[s:s + chunk])
        pre = _window_sum(_slots(tok, w["E1"]), W) + pad
        out[s:s + chunk] = _energy_head(model, pre).sum(1) + w["state_bias"][cur_ids].sum(1)
    return out


@_quiet
def log_rates(model, pep, cur):
    """Log rate (per ps) of every single-site move out of configurations cur (B, S), and their energies.

    Returns logk (B, S, V), -inf where site c has no state v or already is in it, and E (B,)."""
    w, W, R, S = model.w, model.W, pep.R, pep.S
    cur = np.asarray(cur, np.int64)
    V = pep.site_ids.shape[1]
    tok, cur_ids = _tokens(model, pep, cur)
    pad = w["type_emb_weight"][PAD]
    pre = _window_sum(_slots(tok, w["E1"]), W) + _pad_proj(pad, w["E1"], R, W) + w["E1b"]
    e0 = _energy_head(model, pre)                                                # (B, R)
    res = pep.site_res
    cur_emb = w["state_emb_weight"][cur_ids]                                     # (B, S, d)
    tgt_emb = w["state_emb_weight"][pep.site_ids]                                # (S, V, d)
    diff = tgt_emb[None] - cur_emb[:, :, None]                                   # (B, S, V, d)
    dE = np.zeros((cur.shape[0], S, V))
    for delta in range(-W, W + 1):               # a move changes the windows of the residues within W of it
        j = res + delta
        valid = (j >= 0) & (j < R)
        jj = np.clip(j, 0, R - 1)
        dpre = diff @ w["E1"][W - delta]                                         # (B, S, V, h)
        e_new = _energy_head(model, pre[:, jj][:, :, None] + dpre)
        dE += np.where(valid[None, :, None], e_new - e0[:, jj][:, :, None], 0.0)
    # prefactor: the window of the moving residue without the moving site's state, and a symmetric pair code
    preG = _window_sum(_slots(tok, w["G1"]), W) + _pad_proj(pad, w["G1"], R, W)
    g_win = preG[:, res] - cur_emb @ w["G1"][W]                                  # (B, S, h)
    kind = np.broadcast_to(w["kind_emb_weight"][pep.site_kind][None, :, None], diff.shape)
    pair = np.concatenate([cur_emb[:, :, None] + tgt_emb[None], cur_emb[:, :, None] * tgt_emb[None], kind], -1)
    g = _prefactor_head(model, g_win[:, :, None] + pair @ w["G1p_weight"].T + w["G1p_bias"])
    if model.g_bound:                            # a bounded correction: a state never entered must cost energy
        g = model.g_bound * np.tanh(g / model.g_bound)
    dE += w["state_bias"][pep.site_ids][None] - w["state_bias"][cur_ids][:, :, None]
    g += w["pair_bias"][cur_ids[:, :, None], pep.site_ids[None]]
    logk = g - 0.5 * dE
    ar = np.arange(V)
    ok = (ar[None, None] < pep.site_n[None, :, None]) & (ar[None, None] != cur[..., None])
    E = e0.sum(1) + w["state_bias"][cur_ids].sum(1)
    return np.where(ok, logk, -np.inf), E


class RateTable:
    """Every move rate of every joint configuration of a peptide whose joint space is small enough to list."""

    def __init__(self, sizes, k, E):
        self.sizes, self.k, self.E = sizes, k, E
        self.strides = np.cumprod(np.concatenate([[1], sizes[::-1][:-1]]))[::-1]
        self._chain = {}

    def index(self, x):
        return (np.asarray(x, np.int64) * self.strides).sum(-1)

    def configs(self, idx):
        return np.stack(np.unravel_index(idx, self.sizes), -1)

    def equilibrium(self):
        p = np.exp(-(self.E - self.E.min()))
        return p / p.sum()

    @_quiet
    def chain(self, dt):
        """For a step dt: the probability that each site moves, and the cumulative distribution of where to."""
        if dt not in self._chain:
            K = self.k.sum(-1)
            P = np.cumsum(self.k / np.maximum(K[..., None], 1e-300), -1, dtype=np.float32)
            self._chain = {dt: (-np.expm1(-dt * K), P)}
        return self._chain[dt]


def _prefactors(model, pep, cur):
    """Log-prefactor g of every move (pep.moves) out of configurations cur (B, S): (B, moves), in single precision."""
    w, W = model.w32, model.W
    c, v = pep.moves[:, 0], pep.moves[:, 1]
    tok, cur_ids = _tokens(model, pep, cur, w)
    preG = _window_sum(_slots(tok, w["G1"]), W) + _pad_proj(w["type_emb_weight"][PAD], w["G1"], pep.R, W)
    cur_emb = w["state_emb_weight"][cur_ids]                                     # (B, S, d)
    g_win = preG[:, pep.site_res] - cur_emb @ w["G1"][W]                         # (B, S, h)
    a, b = cur_emb[:, c], w["state_emb_weight"][pep.site_ids[c, v]][None]        # (B, P, d), (1, P, d)
    kind = np.broadcast_to(w["kind_emb_weight"][pep.site_kind[c]][None], a.shape)
    pair = np.concatenate([a + b, a * b, kind], -1)
    g = _prefactor_head(model, g_win[:, c] + pair @ w["G1p_weight"].T + w["G1p_bias"], w)
    if model.g_bound:
        g = model.g_bound * np.tanh(g / model.g_bound)
    return g + w["pair_bias"][cur_ids[:, c], pep.site_ids[c, v][None]]


@_quiet
def move_rates(model, pep, cur):
    """Rates (per ps) of every single-site move out of configurations cur (B, S), as (B, S, V) with zeros where
    site c has no state v or already is in it. Equal to exp(log_rates), in single precision and without the
    work on states a site does not have: what the dynamics uses when the joint space is too large to tabulate."""
    w, W, R = model.w32, model.W, pep.R
    cur = np.asarray(cur, np.int64)
    c, v = pep.moves[:, 0], pep.moves[:, 1]
    tok, cur_ids = _tokens(model, pep, cur, w)
    pre = (_window_sum(_slots(tok, w["E1"]), W)
           + _pad_proj(w["type_emb_weight"][PAD], w["E1"], R, W) + w["E1b"])
    e0 = _energy_head(model, pre, w)
    diff = w["state_emb_weight"][pep.site_ids[c, v]][None] - w["state_emb_weight"][cur_ids][:, c]     # (B, P, d)
    dE = w["state_bias"][pep.site_ids[c, v]][None] - w["state_bias"][cur_ids[:, c]]
    res = pep.site_res[c]
    for delta in range(-W, W + 1):
        j = res + delta
        ok = (j >= 0) & (j < R)
        if ok.any():
            e_new = _energy_head(model, pre[:, j[ok]] + diff[:, ok] @ w["E1"][W - delta], w)
            dE[:, ok] += e_new - e0[:, j[ok]]
    rate = np.exp(_prefactors(model, pep, cur) - 0.5 * dE)
    rate[cur[:, c] == v[None]] = 0.0
    k = np.zeros((len(cur), pep.S, pep.site_ids.shape[1]), np.float32)
    k[:, c, v] = rate
    return k


@_quiet
def rate_table(model, pep, max_configs=400000, chunk=2048):
    """The RateTable of a peptide, or None if its joint space holds more than max_configs configurations.

    Equal to log_rates on every configuration, and faster: the energy of each configuration is computed once and
    the energy change of a move is read off as a difference of two entries."""
    if pep.n_configs > max_configs:
        return None
    n, sizes = pep.n_configs, pep.site_n.astype(np.int64)
    strides = np.cumprod(np.concatenate([[1], sizes[::-1][:-1]]))[::-1]
    E = np.zeros(n)
    for s in range(0, n, 64 * chunk):
        idx = np.arange(s, min(n, s + 64 * chunk))
        E[idx] = energy(model, pep, np.stack(np.unravel_index(idx, sizes), 1))
    pairs = pep.moves
    k = np.zeros((n, pep.S, pep.site_ids.shape[1]), np.float32)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        cfg = np.stack(np.unravel_index(idx, sizes), 1)
        g = _prefactors(model, pep, cfg)
        moved = idx[:, None] + (pairs[None, :, 1] - cfg[:, pairs[:, 0]]) * strides[pairs[:, 0]][None]
        rate = np.exp(g - 0.5 * (E[moved] - E[idx][:, None]))
        rate[cfg[:, pairs[:, 0]] == pairs[None, :, 1]] = 0.0                     # staying is not a move
        k[idx[:, None], pairs[None, :, 0], pairs[None, :, 1]] = rate
    return RateTable(sizes, k, E)


def equilibrium_starts(model, pep, n, rng, table=None, sweeps=200):
    """n configurations drawn from exp(-E) / Z, the equilibrium of the jump rates: exactly from a table, else
    by Metropolis sampling with single-site proposals."""
    if table is not None:
        return table.configs(rng.choice(len(table.E), size=n, p=table.equilibrium()))
    x = np.zeros((n, pep.S), np.int64)
    E = energy(model, pep, x)
    for _ in range(sweeps):
        for c in range(pep.S):
            if pep.site_n[c] < 2:
                continue
            prop = x.copy()
            prop[:, c] = rng.integers(0, pep.site_n[c], size=n)
            Ep = energy(model, pep, prop)
            acc = rng.random(n) < np.exp(np.minimum(0.0, -(Ep - E)))
            x[acc], E[acc] = prop[acc], Ep[acc]
    return x


@_quiet
def simulate(model, pep, x0, ns, rng, frame_ps=10.0, table=None):
    """The model's dynamics: a chain with step dt (10 ps, the frame spacing it was fitted at) in which every site
    leaves its state with probability 1 - exp(-dt K), K its exit rate in the current configuration, for a state
    drawn in proportion to the rates. This is the chain whose likelihood was maximised in training. Its
    equilibrium is close to exp(-E) / Z, not equal: several sites can move in one step.
    Rows of x0 (B, S) advance in parallel; returns the site states every frame_ps, (B, frames, S) int16."""
    dt = model.dt_ps
    every, n_frames = int(round(frame_ps / dt)), int(round(ns * 1000.0 / frame_ps))
    if every < 1 or abs(every * dt - frame_ps) > 1e-9:
        raise ValueError(f"frames must be a whole number of {dt:g} ps steps apart")
    x = np.array(x0, np.int64)
    B, S = x.shape
    out = np.zeros((B, n_frames, S), np.int16)
    if table is not None:
        pmove, P = table.chain(dt)
        idx = table.index(x)
        for f in range(n_frames):
            for _ in range(every):
                mv = rng.random((B, S)) < pmove[idx]
                if mv.any():
                    b, c = np.nonzero(mv)
                    v = (P[idx[b], c] < rng.random(len(b))[:, None]).sum(1)
                    v = np.minimum(v, table.sizes[c] - 1)          # guards the rounding of the last cumulative sum
                    np.add.at(idx, b, (v - x[b, c]) * table.strides[c])
                    x[b, c] = v
            out[:, f] = x
        return out
    for f in range(n_frames):
        for _ in range(every):
            k = move_rates(model, pep, x)
            K = k.sum(-1)
            mv = rng.random((B, S)) < -np.expm1(-dt * K)
            if mv.any():
                b, c = np.nonzero(mv)
                P = np.cumsum(k[b, c] / K[b, c][:, None], -1)
                x[b, c] = np.minimum((P < rng.random(len(b))[:, None]).sum(1), pep.site_n[c] - 1)
        out[:, f] = x
    return out
