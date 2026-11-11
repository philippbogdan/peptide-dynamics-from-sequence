"""The network and its likelihood, in PyTorch: what the published weights were trained with.

A configuration x holds one state per site (backbone basin, rotamer basins); a move changes one site.
  energy   E(x) = sum over residues of e(window of residue tokens around it)                    (kT)
  rate     k(x -> x') = exp( g(window without the moving site's state, {state left, state entered}) - (E(x') - E(x)) / 2 )
g is symmetric in the pair and blind to the moving site's own state, so k(x -> x') / k(x' -> x) = exp(-(E(x') - E(x))):
detailed balance holds exactly. A residue token is the sum of a residue-type embedding and the embeddings of its
sites' current states; the termini are tokens too. The model is fitted by maximum likelihood to site-state
trajectories read every dt: in one step each site leaves its state with probability 1 - exp(-dt K), K its exit
rate in the current configuration, for a state drawn in proportion to the rates.

pepdyn/model.py is the same network in numpy; tests/torch_reference.npz holds this one's outputs for comparison.
"""
import numpy as np
import torch
import torch.nn as nn

PAD, NTERM, CTERM = 0, 1, 2        # special tokens (charged termini); 3 and 4 are unused; residue types start at 5
T_REF = 300.0


class Vocab:
    """Global ids for residue types and for (site key, state)."""

    def __init__(self, sitedefs):
        self.types = {}
        self.state_id = {}
        self.max_sites = 5      # backbone and chi1..chi4
        n = 1                   # 0 = no site
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
    """Static description of one peptide for the network: residue tokens, sites, allowed moves."""

    def __init__(self, vocab, resnames, site_list, keys, nterm=NTERM, cterm=CTERM, temperature=T_REF):
        self.R = len(resnames) + 2
        self.temperature = float(temperature)
        self.types = np.array([nterm] + [vocab.types[r] for r in resnames] + [cterm])
        S = len(site_list)
        self.site_res = np.array([i + 1 for i, _ in site_list])          # residue slot (N-terminus at 0)
        self.site_kind = np.array([0 if k == "bb" else int(k[3]) for _, k in site_list])
        self.site_n = np.array([vocab_n(vocab, key) for key in keys])
        self.site_ids = np.zeros((S, vocab.max_v), np.int64)               # global id of state s of site c
        for c, key in enumerate(keys):
            for s in range(self.site_n[c]):
                self.site_ids[c, s] = vocab.state_id[(key, s)]
        self.S = S


def vocab_n(vocab, key):
    n = 0
    while (key, n) in vocab.state_id:
        n += 1
    return n


class KineticNet(nn.Module):
    def __init__(self, vocab, W=2, d=64, h=192):
        super().__init__()
        self.W, self.d, self.h = W, d, h
        L = 2 * W + 1
        self.type_emb = nn.Embedding(vocab.n_types, d)
        self.state_emb = nn.Embedding(vocab.n_states, d, padding_idx=0)
        self.kind_emb = nn.Embedding(5, d)
        # energy MLP; its first layer acts slot by slot on the window of residue tokens
        self.E1 = nn.Parameter(torch.randn(L, d, h) / np.sqrt(L * d))
        self.E1b = nn.Parameter(torch.zeros(h))
        self.E2 = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        # prefactor MLP: window (moving site removed) + symmetric pair code
        self.G1 = nn.Parameter(torch.randn(L, d, h) / np.sqrt(L * d))
        self.G1p = nn.Linear(3 * d, h)
        self.G2 = nn.Sequential(nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
        nn.init.zeros_(self.E2[-1].weight)
        nn.init.zeros_(self.E2[-1].bias)
        nn.init.zeros_(self.G2[-1].weight)
        with torch.no_grad():
            self.G2[-1].bias.copy_(torch.tensor([-3.0, 0.0]))
        # empirical baseline (set by init_baseline): free energy of each state and symmetric log-prefactor of
        # each transition; the networks above learn corrections to it from the sequence context
        self.state_bias = nn.Parameter(torch.zeros(vocab.n_states))
        self.pair_bias = nn.Parameter(torch.zeros(vocab.n_states, vocab.n_states))
        self.g_bound = 0.0
        self.beta = 1.0

    @torch.no_grad()
    def init_baseline(self, F, G, g_bound=2.0):
        """F: (n_states,) free energy per global state id (kT); G: (n_states, n_states) symmetric log prefactor.
        The prefactor baseline is kept fixed and the network's correction to it bounded by +-g_bound."""
        self.state_bias.copy_(torch.as_tensor(F, dtype=self.state_bias.dtype))
        self.pair_bias.copy_(torch.as_tensor(G, dtype=self.pair_bias.dtype))
        self.pair_bias.requires_grad_(False)
        self.G2[-1].bias.copy_(torch.tensor([0.0, 0.0]))
        self.g_bound = float(g_bound)

    def _E(self, pre):
        """Energy head: (enthalpy-like, entropy-like) -> E/kT = h * T_REF/T - s."""
        o = self.E2(pre)
        return o[..., 0] * self.beta - o[..., 1]

    def _G(self, pre):
        """Log-prefactor head: a - b * T_REF/T (Arrhenius-like)."""
        o = self.G2(pre)
        return o[..., 0] - o[..., 1] * self.beta

    def tokens(self, types, state_ids):
        """types: (B, R) residue type ids; state_ids: (B, R, 5) global state ids (0 = none) -> (B, R, d)."""
        return self.type_emb(types) + self.state_emb(state_ids).sum(-2)

    def _slot_proj(self, tok, Wt):
        """Per-slot projections P[b, r, l, :] = tok[b, r] @ Wt[l]  -> (B, R, L, h)."""
        return torch.einsum("brd,ldh->brlh", tok, Wt)

    def _window_sum(self, P):
        """pre[b, j] = sum_l P[b, j + l - W, l] (zero beyond the ends, where the PAD token sits) -> (B, R, h)."""
        B, R, L, h = P.shape
        W = self.W
        out = torch.zeros(B, R, h, device=P.device, dtype=P.dtype)
        for l in range(L):
            off = l - W
            lo, hi = max(0, -off), min(R, R - off)
            out[:, lo:hi] += P[:, lo + off:hi + off, l]
        return out

    def _pad_proj(self, Wt, R, device):
        """Contribution of PAD tokens beyond the ends to each residue's window (R, h)."""
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
        B, R, _ = tok.shape
        pre = self._window_sum(self._slot_proj(tok, self.E1)) + self._pad_proj(self.E1, R, tok.device) + self.E1b
        return self._E(pre)

    def energy(self, pep, cur):
        """E(x)/kT of configurations cur (B, S) (no rates): for Boltzmann sampling of starting configurations."""
        t = pep.t
        B, S, V = cur.shape[0], pep.S, t["site_ids"].shape[1]
        self.beta = T_REF / getattr(pep, "temperature", T_REF)
        cur_ids = torch.gather(t["site_ids"].expand(B, S, V), 2, cur[..., None]).squeeze(-1)
        sid = torch.zeros(B, pep.R, 5, dtype=torch.long, device=cur.device)
        sid[:, t["site_res"], t["site_kind"]] = cur_ids
        tok = self.tokens(t["types"].expand(B, pep.R), sid)
        return self.energy_terms(tok).sum(1) + self.state_bias[cur_ids].sum(1)

    def log_rates(self, pep, cur):
        """pep: PeptideIndex (tensors in pep.t); cur: (B, S) current state index of each site.
        Returns logk (B, S, V): log rate of moving site c to state v (-inf where invalid or current), and E (B,)."""
        t = pep.t
        B, S, V = cur.shape[0], pep.S, t["site_ids"].shape[1]
        R, W = pep.R, self.W
        L = 2 * W + 1
        dev = cur.device
        self.beta = T_REF / getattr(pep, "temperature", T_REF)
        cur_ids = torch.gather(t["site_ids"].expand(B, S, V), 2, cur[..., None]).squeeze(-1)    # (B, S)
        sid = torch.zeros(B, R, 5, dtype=torch.long, device=dev)
        sid[:, t["site_res"], t["site_kind"]] = cur_ids
        tok = self.tokens(t["types"].expand(B, R), sid)                                 # (B, R, d)
        PE = self._slot_proj(tok, self.E1)                                              # (B, R, L, h)
        pre = self._window_sum(PE) + self._pad_proj(self.E1, R, dev) + self.E1b         # (B, R, h)
        e0 = self._E(pre)                                                               # (B, R)
        res = t["site_res"]
        cur_emb = self.state_emb(cur_ids)                                               # (B, S, d)
        tgt_emb = self.state_emb(t["site_ids"])                                         # (S, V, d)
        diff = tgt_emb[None] - cur_emb[:, :, None]                                      # (B, S, V, d)
        dE = torch.zeros(B, S, V, device=dev)
        for delta in range(-W, W + 1):
            j = res + delta
            valid = (j >= 0) & (j < R)
            jj = j.clamp(0, R - 1)
            slot = W - delta
            dpre = torch.einsum("bsvd,dh->bsvh", diff, self.E1[slot])
            pre_j = pre[:, jj]                                                          # (B, S, h)
            e_new = self._E(pre_j[:, :, None] + dpre)                                   # (B, S, V)
            e_old = e0[:, jj][:, :, None]
            dE = dE + torch.where(valid[None, :, None], e_new - e_old, torch.zeros_like(e_new))
        # prefactor: window of the moving residue with the moving site's state removed
        PG = self._slot_proj(tok, self.G1)                                              # (B, R, L, h)
        preG = self._window_sum(PG) + self._pad_proj(self.G1, R, dev)                   # (B, R, h)
        g_win = preG[:, res] - torch.einsum("bsd,dh->bsh", cur_emb, self.G1[W])         # (B, S, h)
        kind = self.kind_emb(t["site_kind"])                                            # (S, d)
        pair = torch.cat([cur_emb[:, :, None] + tgt_emb[None], cur_emb[:, :, None] * tgt_emb[None],
                          kind[None, :, None].expand(B, S, V, -1)], -1)                 # (B, S, V, 3d)
        g = self._G(g_win[:, :, None] + self.G1p(pair))                                 # (B, S, V)
        if self.g_bound:   # bounded prefactor correction: absent transitions must be explained by free energy
            g = self.g_bound * torch.tanh(g / self.g_bound)
        dE = dE + self.state_bias[t["site_ids"]][None] - self.state_bias[cur_ids][:, :, None]
        g = g + self.pair_bias[cur_ids[:, :, None], t["site_ids"][None]]
        logk = g - 0.5 * dE
        ar = torch.arange(V, device=dev)
        ok = (ar[None, None] < t["site_n"][None, :, None]) & (ar[None, None] != cur[..., None])
        E = e0.sum(1) + self.state_bias[cur_ids].sum(1)
        return torch.where(ok, logk, torch.full_like(logk, -np.inf)), E


def to_device(pep, device):
    pep.t = dict(types=torch.as_tensor(pep.types, device=device)[None],
                 site_res=torch.as_tensor(pep.site_res, device=device),
                 site_kind=torch.as_tensor(pep.site_kind, device=device),
                 site_n=torch.as_tensor(pep.site_n, device=device),
                 site_ids=torch.as_tensor(pep.site_ids, device=device))
    return pep


def segments(L):
    """Compress a site trajectory (frames, S) into runs: (configs (n,S), run lengths (n,), next configs (n,S),
    has_next (n,)). A run's config holds for `len` frames; then the trajectory moves to `next`."""
    change = np.any(L[1:] != L[:-1], axis=1)
    starts = np.concatenate([[0], np.where(change)[0] + 1])
    ends = np.concatenate([starts[1:], [len(L)]])
    cfg = L[starts]
    nxt = np.concatenate([L[starts[1:]], L[-1:]])
    has = np.concatenate([np.ones(len(starts) - 1, bool), [False]])
    return cfg.astype(np.int64), (ends - starts).astype(np.int64), nxt.astype(np.int64), has


def nll(net, pep, cfg, runlen, nxt, has, dt):
    """Negative log-likelihood (per-site tau-leap at step dt) of a batch of runs of one peptide."""
    logk, _ = net.log_rates(pep, cfg)                        # (B, S, V)
    k = torch.exp(logk)
    K = k.sum(-1)                                            # (B, S) exit rate of each site
    moved = (nxt != cfg) & has[:, None]                      # (B, S)
    stay_steps = (runlen - 1).float()[:, None] + (~moved).float() * has[:, None].float()
    loss = (stay_steps * dt * K).sum(1)
    lk_move = torch.gather(logk, 2, nxt[..., None]).squeeze(-1)
    lp = torch.log((-torch.expm1(-dt * K)).clamp_min(1e-30)) + lk_move - torch.log(K.clamp_min(1e-30))
    loss = loss - torch.where(moved, lp, torch.zeros_like(lp)).sum(1)
    return loss


def empirical_baseline(vocab, peps, dt, pseudo=0.5):
    """Per-state free energies -ln p and symmetric log prefactors 0.5 ln(k_uv k_vu) from site trajectories, pooled
    by site key (residue type, site kind). peps: dicts with L (frames, S), keys."""
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
    # normalise populations within each site key
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


def _tau_step(net, pep, x, dt):
    """One tau-leap step with static shapes and no host synchronisation (CUDA-graph friendly): every site leaves with
    probability 1 - exp(-dt K); its target is drawn in proportion to the rates by the Gumbel-max trick."""
    logk, _ = net.log_rates(pep, x)
    K = torch.exp(logk).sum(-1)                                     # (B, S)
    move = torch.rand_like(K) < -torch.expm1(-dt * K)
    g = -torch.log(-torch.log(torch.rand_like(logk).clamp_min(1e-20)))
    v = torch.argmax(logk + g, dim=-1)                              # (B, S)
    return torch.where(move, v, x)


@torch.no_grad()
def simulate(net, pep, x0, t_end, frame_dt, dt, device):
    """The chain the likelihood describes, B trajectories (rows of x0) in parallel; returns the site states every
    frame_dt as (B, frames, S). On a GPU the steps between two frames run as one captured CUDA graph."""
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
    with torch.cuda.stream(s):                                      # warm-up outside the graph
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
