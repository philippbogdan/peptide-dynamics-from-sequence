"""Sequence-conditioned rates for changes in local peptide conformations.

A configuration contains one backbone or rotamer state per site. For a move
that changes one site, k(x -> y) = exp(g(x, y) - (E(y) - E(x))/2), with rates
in inverse picoseconds and dimensionless energy E. Residue windows determine
E; the prefactor uses the surrounding configuration and an unordered pair of
site states. Thus g(x, y) = g(y, x), and the continuous-time generator obeys
detailed balance with Boltzmann weights exp(-E).

Training uses a per-site finite-step likelihood. The table and tau samplers
use simultaneous site updates at that step; they approximate the continuous-
time process and do not enforce its exact stationary distribution. The
Gillespie sampler instead follows continuous-time single-site jumps.
"""
import numpy as np
import torch
import torch.nn as nn

PAD, NTERM, CTERM, ACE, NME = 0, 1, 2, 3, 4
T_REF = 300.0


class Vocab:
    """Assign shared integer identifiers to residue types and local conformational states."""

    def __init__(self, sitedefs):
        self.types = {}
        self.state_id = {}
        self.max_sites = 5
        n = 1
        for key in sorted(sitedefs.defs):
            for s in range(sitedefs.defs[key]["n"]):
                self.state_id[(key, s)] = n
                n += 1
        self.n_states = n
        for rn in sorted({k[0] for k in sitedefs.defs}):
            self.types[rn] = 5 + len(self.types)
        self.n_types = 5 + len(self.types)
        self.max_v = max(d["n"] for d in sitedefs.defs.values())


class PeptideIndex:
    """Store a sequence, its terminal tokens, local sites and allowed state identifiers."""

    def __init__(self, vocab, resnames, site_list, keys, nterm=NTERM, cterm=CTERM, temperature=T_REF):
        self.R = len(resnames) + 2
        self.temperature = float(temperature)
        self.types = np.array([nterm] + [vocab.types[r] for r in resnames] + [cterm])
        S = len(site_list)
        self.site_res = np.array([i + 1 for i, _ in site_list])
        self.site_kind = np.array([0 if k == "bb" else int(k[3]) for _, k in site_list])
        self.site_n = np.array([vocab_n(vocab, key) for key in keys])
        self.site_ids = np.zeros((S, vocab.max_v), np.int64)
        for c, key in enumerate(keys):
            for s in range(self.site_n[c]):
                self.site_ids[c, s] = vocab.state_id[(key, s)]
        self.S = S


def termini(tors0):
    """Infer cap tokens from the presence of terminal phi and psi torsions."""
    nt = ACE if np.isfinite(tors0[0, 0]) else NTERM
    ct = NME if np.isfinite(tors0[-1, 1]) else CTERM
    return nt, ct


def vocab_n(vocab, key):
    """Count the consecutive state labels defined for one residue/site key."""
    n = 0
    while (key, n) in vocab.state_id:
        n += 1
    return n


class KineticNet(nn.Module):
    """Learn local energy terms and symmetric corrections to baseline log prefactors."""
    def __init__(self, vocab, W=2, d=64, h=192):
        super().__init__()
        self.W, self.d, self.h = W, d, h
        L = 2 * W + 1
        self.type_emb = nn.Embedding(vocab.n_types, d)
        self.state_emb = nn.Embedding(vocab.n_states, d, padding_idx=0)
        self.kind_emb = nn.Embedding(5, d)

        # Position-specific projections share a nonlinear head across residue windows.
        self.E1 = nn.Parameter(torch.randn(L, d, h) / np.sqrt(L * d))
        self.E1b = nn.Parameter(torch.zeros(h))
        self.E2 = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

        # The move head combines surrounding residues with an unordered state pair.
        self.G1 = nn.Parameter(torch.randn(L, d, h) / np.sqrt(L * d))
        self.G1p = nn.Linear(3 * d, h)
        self.G2 = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        nn.init.zeros_(self.E2[-1].weight)
        nn.init.zeros_(self.E2[-1].bias)
        nn.init.zeros_(self.G2[-1].weight)
        with torch.no_grad():
            self.G2[-1].bias.copy_(torch.tensor([-3.0, 0.0]))


        self.state_bias = nn.Parameter(torch.zeros(vocab.n_states))
        self.pair_bias = nn.Parameter(torch.zeros(vocab.n_states, vocab.n_states))
        self.g_bound = 0.0
        self.beta = 1.0

    @torch.no_grad()
    def init_baseline(self, F, G, g_bound=2.0):
        """Initialise state energies F and symmetric log prefactors G.

        F has shape (global states,) and G has shape (global states, global states).
        The prefactor baseline is fixed. A nonzero g_bound limits its learned
        correction to plus or minus that value; state energies remain trainable."""
        self.state_bias.copy_(torch.as_tensor(F, dtype=self.state_bias.dtype))
        self.pair_bias.copy_(torch.as_tensor(G, dtype=self.pair_bias.dtype))
        self.pair_bias.requires_grad_(False)
        self.G2[-1].bias.copy_(torch.tensor([0.0, 0.0]))
        self.g_bound = float(g_bound)

    def _E(self, pre):
        """Combine the two energy outputs as h * T_REF / T - s."""
        o = self.E2(pre)
        return o[..., 0] * self.beta - o[..., 1]

    def _G(self, pre):
        """Combine the two log-prefactor outputs as a - b * T_REF / T."""
        o = self.G2(pre)
        return o[..., 0] - o[..., 1] * self.beta

    def tokens(self, types, state_ids):
        """Add residue and site embeddings: (B, R) and (B, R, 5) become (B, R, d)."""
        return self.type_emb(types) + self.state_emb(state_ids).sum(-2)

    def _slot_proj(self, tok, Wt):
        """Project each residue token for every window position: (B, R, L, h)."""
        return torch.einsum("brd,ldh->brlh", tok, Wt)

    def _window_sum(self, P):
        """Sum position-specific projections around each residue, omitting out-of-range slots."""
        B, R, L, h = P.shape
        W = self.W
        out = torch.zeros(B, R, h, device=P.device, dtype=P.dtype)
        for l in range(L):
            off = l - W
            lo, hi = max(0, -off), min(R, R - off)
            out[:, lo:hi] += P[:, lo + off:hi + off, l]
        return out

    def _pad_proj(self, Wt, R, device):
        """Supply the learned padding contribution for windows extending beyond the sequence."""
        W = self.W
        pad = self.type_emb.weight[PAD]
        out = torch.zeros(R, Wt.shape[2], device=device)
        for l in range(2 * W + 1):
            off = l - W
            for j in range(R):
                if not (0 <= j + off < R):
                    out[j] += pad @ Wt[l]
        return out

    def energy_terms(self, tok):
        """Evaluate each residue-window contribution to the configuration energy."""
        B, R, _ = tok.shape
        pre = self._window_sum(self._slot_proj(tok, self.E1)) + self._pad_proj(self.E1, R, tok.device) + self.E1b
        return self._E(pre)

    def energy(self, pep, cur):
        """Return dimensionless energies for configurations cur of shape (batch, sites)."""
        t = pep.t
        B, S, V = cur.shape[0], pep.S, t["site_ids"].shape[1]
        self.beta = T_REF / getattr(pep, "temperature", T_REF)
        cur_ids = torch.gather(t["site_ids"].expand(B, S, V), 2, cur[..., None]).squeeze(-1)
        sid = torch.zeros(B, pep.R, 5, dtype=torch.long, device=cur.device)
        sid[:, t["site_res"], t["site_kind"]] = cur_ids
        tok = self.tokens(t["types"].expand(B, pep.R), sid)
        return self.energy_terms(tok).sum(1) + self.state_bias[cur_ids].sum(1)

    def log_rates(self, pep, cur):
        """Return log rates (B, S, V) and energies (B,) for configurations cur (B, S).

        pep.t holds the peptide tensors. S is the number of sites and V the maximum
        number of states per site. Invalid targets and unchanged states have log
        rate minus infinity. Rates use ps as the time unit."""
        t = pep.t
        B, S, V = cur.shape[0], pep.S, t["site_ids"].shape[1]
        R, W = pep.R, self.W
        L = 2 * W + 1
        dev = cur.device
        self.beta = T_REF / getattr(pep, "temperature", T_REF)
        cur_ids = torch.gather(t["site_ids"].expand(B, S, V), 2, cur[..., None]).squeeze(-1)
        sid = torch.zeros(B, R, 5, dtype=torch.long, device=dev)
        sid[:, t["site_res"], t["site_kind"]] = cur_ids
        tok = self.tokens(t["types"].expand(B, R), sid)
        PE = self._slot_proj(tok, self.E1)
        pre = self._window_sum(PE) + self._pad_proj(self.E1, R, dev) + self.E1b
        e0 = self._E(pre)
        res = t["site_res"]
        cur_emb = self.state_emb(cur_ids)
        tgt_emb = self.state_emb(t["site_ids"])
        diff = tgt_emb[None] - cur_emb[:, :, None]
        dE = torch.zeros(B, S, V, device=dev)
        # A changed site alters only energy windows containing its residue.
        for delta in range(-W, W + 1):
            j = res + delta
            valid = (j >= 0) & (j < R)
            jj = j.clamp(0, R - 1)
            slot = W - delta
            dpre = torch.einsum("bsvd,dh->bsvh", diff, self.E1[slot])
            pre_j = pre[:, jj]
            e_new = self._E(pre_j[:, :, None] + dpre)
            e_old = e0[:, jj][:, :, None]
            dE = dE + torch.where(valid[None, :, None], e_new - e_old, torch.zeros_like(e_new))

        # Remove the changing site from context to keep the prefactor reversible.
        PG = self._slot_proj(tok, self.G1)
        preG = self._window_sum(PG) + self._pad_proj(self.G1, R, dev)
        g_win = preG[:, res] - torch.einsum("bsd,dh->bsh", cur_emb, self.G1[W])
        kind = self.kind_emb(t["site_kind"])
        pair = torch.cat([cur_emb[:, :, None] + tgt_emb[None], cur_emb[:, :, None] * tgt_emb[None],
                          kind[None, :, None].expand(B, S, V, -1)], -1)
        g = self._G(g_win[:, :, None] + self.G1p(pair))
        # Bound only the learned correction, retaining the empirical baseline.
        if self.g_bound:
            g = self.g_bound * torch.tanh(g / self.g_bound)
        dE = dE + self.state_bias[t["site_ids"]][None] - self.state_bias[cur_ids][:, :, None]
        g = g + self.pair_bias[cur_ids[:, :, None], t["site_ids"][None]]
        logk = g - 0.5 * dE
        ar = torch.arange(V, device=dev)
        ok = (ar[None, None] < t["site_n"][None, :, None]) & (ar[None, None] != cur[..., None])
        E = e0.sum(1) + self.state_bias[cur_ids].sum(1)
        return torch.where(ok, logk, torch.full_like(logk, -np.inf)), E


def to_device(pep, device):
    """Attach tensor copies of a peptide index to pep.t on the requested device."""
    pep.t = dict(types=torch.as_tensor(pep.types, device=device)[None],
                 site_res=torch.as_tensor(pep.site_res, device=device),
                 site_kind=torch.as_tensor(pep.site_kind, device=device),
                 site_n=torch.as_tensor(pep.site_n, device=device),
                 site_ids=torch.as_tensor(pep.site_ids, device=device))
    return pep


def segments(L):
    """Compress a path into configurations, residence lengths and subsequent configurations.

    Input shape is (frames, sites). Outputs have one row per constant run;
    has_next is false for the final run because its exit is unobserved."""
    change = np.any(L[1:] != L[:-1], axis=1)
    starts = np.concatenate([[0], np.where(change)[0] + 1])
    ends = np.concatenate([starts[1:], [len(L)]])
    cfg = L[starts]
    nxt = np.concatenate([L[starts[1:]], L[-1:]])
    has = np.concatenate([np.ones(len(starts) - 1, bool), [False]])
    return cfg.astype(np.int64), (ends - starts).astype(np.int64), nxt.astype(np.int64), has


def nll(net, pep, cfg, runlen, nxt, has, dt):
    """Return the finite-step negative log-likelihood of each constant-state run.

    At step dt in ps, each site stays with probability exp(-dt * exit_rate).
    Conditional on moving, its destination is proportional to the outgoing
    rates. Exits of the final run are not included in the likelihood."""
    logk, _ = net.log_rates(pep, cfg)
    k = torch.exp(logk)
    K = k.sum(-1)
    moved = (nxt != cfg) & has[:, None]
    stay_steps = (runlen - 1).float()[:, None] + (~moved).float() * has[:, None].float()
    loss = (stay_steps * dt * K).sum(1)
    lk_move = torch.gather(logk, 2, nxt[..., None]).squeeze(-1)
    lp = torch.log((-torch.expm1(-dt * K)).clamp_min(1e-30)) + lk_move - torch.log(K.clamp_min(1e-30))
    loss = loss - torch.where(moved, lp, torch.zeros_like(lp)).sum(1)
    return loss


@torch.no_grad()
def simulate(net, pep, x0, t_end, frame_dt, device, max_events=None):
    """Sample single-site jumps with Gillespie waiting times and record regular frames.

    Times t_end and frame_dt are in ps. x0 has shape (batch, sites); the returned
    int16 array has shape (batch, frames, sites). max_events is currently unused."""
    B = x0.shape[0]
    n_frames = int(round(t_end / frame_dt))
    out = torch.zeros(B, n_frames, pep.S, dtype=torch.int16, device=device)
    x = x0.clone().to(device)
    t = torch.zeros(B, device=device)
    nxt_frame = torch.zeros(B, dtype=torch.long, device=device)
    ar = torch.arange(B, device=device)
    while True:
        logk, _ = net.log_rates(pep, x)
        k = torch.exp(logk).reshape(B, -1)
        R = k.sum(1)
        tau = -torch.log(torch.rand(B, device=device)) / R
        t_new = t + tau

        # Emit the current configuration until the next sampled jump time.
        while True:
            ft = (nxt_frame + 1).float() * frame_dt
            rec = (ft < t_new) & (nxt_frame < n_frames)
            if not rec.any():
                break
            out[ar[rec], nxt_frame[rec]] = x[rec].to(torch.int16)
            nxt_frame = nxt_frame + rec.long()
        if bool((nxt_frame >= n_frames).all()):
            break
        move = torch.multinomial(k / R[:, None], 1).squeeze(1)
        V = logk.shape[2]
        c, v = move // V, move % V
        x[ar, c] = v
        t = t_new
    return out


@torch.no_grad()
def simulate_tau(net, pep, x0, t_end, frame_dt, dt, device, chunk=None):
    """Sample simultaneous per-site updates at interval dt, with all times in ps.

    A site moves with probability 1 - exp(-dt * exit_rate); its target is drawn
    from the rates before any site changes. Return (batch, frames, sites) int16
    labels at frame_dt spacing. frame_dt should be an integer multiple of dt;
    chunk is currently unused."""
    B, S = x0.shape
    every = int(round(frame_dt / dt))
    n_frames = int(round(t_end / frame_dt))
    out = torch.zeros(B, n_frames, S, dtype=torch.int16, device=device)
    x = x0.clone().to(device)
    for f in range(n_frames):
        for _ in range(every):
            logk, _ = net.log_rates(pep, x)
            k = torch.exp(logk)
            K = k.sum(-1)
            move = torch.rand(B, S, device=device) < -torch.expm1(-dt * K)
            if bool(move.any()):
                b, c = torch.nonzero(move, as_tuple=True)
                p = k[b, c]
                v = torch.multinomial(p / p.sum(-1, keepdim=True), 1).squeeze(1)
                x[b, c] = v
        out[:, f] = x.to(torch.int16)
    return out


def empirical_baseline(vocab, peps, dt, pseudo=0.5):
    """Aggregate state populations and transition counts by residue/site type.

    Return F = -ln(population) and G = 0.5 * ln(k_uv * k_vu), with rates in
    inverse ps when dt is in ps. peps contains state paths L and their keys;
    small pseudocounts keep unobserved states and transitions finite."""
    n = vocab.n_states
    cnt = np.full(n, pseudo)
    trans = np.zeros((n, n))
    dwell = np.full(n, pseudo)
    for p in peps:
        L, keys = p["L"], p["keys"]
        for c, key in enumerate(keys):
            ids = np.array([vocab.state_id[(key, k)] for k in range(int(L[:, c].max()) + 1)])
            col = ids[L[:, c].astype(int)]
            np.add.at(cnt, col, 1.0)
            ch = col[1:] != col[:-1]
            np.add.at(trans, (col[:-1][ch], col[1:][ch]), 1.0)
    dwell = cnt * dt

    F = np.zeros(n)
    by_key = {}
    for (key, k), i in vocab.state_id.items():
        by_key.setdefault(key, []).append(i)
    for key, ids in by_key.items():
        ids = np.array(ids)
        F[ids] = -np.log(cnt[ids] / cnt[ids].sum())
    k = (trans + 0.1) / dwell[:, None]
    G = 0.5 * (np.log(k) + np.log(k).T)
    return F, G


@torch.no_grad()
def rate_table(net, pep, device, max_configs=400000, chunk=8192):
    """Enumerate all configurations and return (site_sizes, rates, energies).

    Rates have shape (configurations, sites, maximum states per site), in inverse
    ps. Return None if the product of site sizes exceeds max_configs. This
    explicit enumeration is practical only for small configuration spaces."""
    sizes = np.asarray(pep.site_n, dtype=np.int64)
    n = int(np.prod(sizes))
    if n > max_configs:
        return None
    V = int(pep.t["site_ids"].shape[1])
    out = np.zeros((n, pep.S, V), np.float32)
    E = np.zeros(n)
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        cfg = np.stack(np.unravel_index(idx, sizes), 1)
        logk, e = net.log_rates(pep, torch.as_tensor(cfg, device=device))
        out[s:s + len(idx)] = torch.exp(logk).cpu().numpy()
        E[s:s + len(idx)] = e.cpu().numpy()
    return sizes, out, E


def table_starts(table, n, seed=0):
    """Draw n configurations from the normalised exp(-E) weights in a rate table.

    These are equilibrium starts for the continuous-time generator, not a
    stationarity guarantee for the finite-step sampler."""
    sizes, _, E = table
    p = np.exp(-(E - E.min()))
    p /= p.sum()
    idx = np.random.default_rng(seed).choice(len(p), size=n, p=p)
    return np.stack(np.unravel_index(idx, sizes), 1)


def simulate_table(table, x0, t_end, frame_dt, dt, seed=0):
    """Run the simultaneous-site finite-step chain using precomputed NumPy rates.

    x0 has shape (batch, sites); return (batch, frames, sites) int16 labels.
    t_end, frame_dt and dt are in ps. frame_dt should be an integer multiple
    of dt. Independent paths are stored on separate rows."""
    sizes, k = table[0], table[1]
    rng = np.random.default_rng(seed)
    B, S = x0.shape
    every = int(round(frame_dt / dt))
    n_frames = int(round(t_end / frame_dt))
    strides = np.cumprod(np.concatenate([[1], sizes[::-1][:-1]]))[::-1]
    x = np.array(x0, dtype=np.int64)
    idx = (x * strides).sum(1)
    K = k.sum(-1)
    P = np.cumsum(k / np.maximum(K[..., None], 1e-300), -1)
    pmove = -np.expm1(-dt * K)
    out = np.zeros((B, n_frames, S), np.int16)
    for f in range(n_frames):
        for _ in range(every):
            mv = rng.random((B, S)) < pmove[idx]
            if mv.any():
                b, c = np.nonzero(mv)
                u = rng.random(len(b))
                v = (P[idx[b], c] < u[:, None]).sum(1)
                # Repeated row indices require accumulated updates when several sites move.
                np.add.at(idx, b, (v - x[b, c]) * strides[c])
                x[b, c] = v

        out[:, f] = x
    return out


def _tau_step(net, pep, x, dt):
    """Draw one simultaneous-site step using Gumbel-max targets and fixed tensor shapes."""
    logk, _ = net.log_rates(pep, x)
    K = torch.exp(logk).sum(-1)
    move = torch.rand_like(K) < -torch.expm1(-dt * K)
    g = -torch.log(-torch.log(torch.rand_like(logk).clamp_min(1e-20)))
    v = torch.argmax(logk + g, dim=-1)
    return torch.where(move, v, x)


@torch.no_grad()
def simulate_tau_fast(net, pep, x0, t_end, frame_dt, dt, device):
    """Run finite-step sampling with regular output frames, using times in ps.

    On CUDA, capture all updates between output frames in one graph. The CPU
    path uses the same update kernel without graph capture."""
    B, S = x0.shape
    every = int(round(frame_dt / dt))
    n_frames = int(round(t_end / frame_dt))
    out = torch.zeros(B, n_frames, S, dtype=torch.int16, device=device)
    x = x0.clone().to(device)
    if device != "cuda":
        for f in range(n_frames):
            for _ in range(every):
                x = _tau_step(net, pep, x, dt)
            out[:, f] = x.to(torch.int16)
        return out
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = static_x
        for _ in range(3):
            y = _tau_step(net, pep, y, dt)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = static_x
        for _ in range(every):
            y = _tau_step(net, pep, y, dt)
        static_out = y
    for f in range(n_frames):
        graph.replay()
        static_x.copy_(static_out)
        out[:, f] = static_out.to(torch.int16)
    return out
