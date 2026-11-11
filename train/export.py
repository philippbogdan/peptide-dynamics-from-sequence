#!/usr/bin/env python3
"""Write a trained model as the single file pepdyn loads: plain arrays, no pickle.

    python train/export.py PREPARED.pkl NET.pt OUT.npz NAME TEMPERATURE G_BOUND DESCRIPTION [VECTORS.npz]

The file holds the weights, the basin grids of every site, the torsion bank (angles as 16-bit integers) and, as a
JSON string, the vocabulary and settings. With VECTORS.npz it also writes the network's log-rates and energies for
random configurations of a few sequences, the reference for testing another implementation (tests/).
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network as N        # noqa: E402

ONE = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE", L="LEU", K="LYS",
           M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")


def main():
    src, netf, out, name, temperature, g_bound, desc = sys.argv[1:8]
    temperature, g_bound = float(temperature), float(g_bound)
    d = pickle.load(open(src, "rb"))
    sd, vocab, bank = d["sitedefs"], d["vocab"], d["bank"]
    weights = torch.load(netf, map_location="cpu")
    arrays = {"w_" + k.replace(".", "_"): v.numpy().astype(np.float32) for k, v in weights.items()}

    keys = sorted(sd.defs)
    site_keys, site_n, first_id = [], [], []
    g2, g1 = dict(lab=[], core=[], F=[], idx=[]), dict(lab=[], core=[], F=[], idx=[], period=[])
    for i, k in enumerate(keys):
        df = sd.defs[k]
        site_keys.append([k[0], k[1]])
        site_n.append(int(df["n"]))
        first_id.append(int(vocab.state_id[(k, 0)]))
        g = g2 if df["kind"] == "2d" else g1
        g["lab"].append(df["lab"].astype(np.int8))
        g["core"].append(df["core"].astype(bool))
        g["F"].append(df["F"].astype(np.float32))
        g["idx"].append(i)
        if df["kind"] == "1d":
            g["period"].append(float(df["period"]))
    arrays.update(grid2_lab=np.stack(g2["lab"]), grid2_core=np.stack(g2["core"]), grid2_F=np.stack(g2["F"]),
                  grid2_key=np.array(g2["idx"], np.int16),
                  grid1_lab=np.stack(g1["lab"]), grid1_core=np.stack(g1["core"]), grid1_F=np.stack(g1["F"]),
                  grid1_key=np.array(g1["idx"], np.int16), grid1_period=np.array(g1["period"], np.float32))

    bank_keys, offsets, rows = [], [0], []
    for k, arr in bank.bank.items():           # the order is kept: a draw for an unseen combination indexes into it
        bank_keys.append([k[0], k[1], [int(v) for v in k[2]]])
        a = np.asarray(arr, np.float64)
        rows.append(np.where(np.isfinite(a), np.round(a / np.pi * 32767.0), -32768).astype(np.int16))
        offsets.append(offsets[-1] + len(a))
    arrays.update(bank=np.concatenate(rows), bank_offsets=np.array(offsets, np.int64))

    meta = dict(name=name, description=desc, termini="charged", temperature_K=temperature, t_ref_K=N.T_REF,
                dt_ps=float(d["dt_ps"]), W=2, d=64, h=192, g_bound=g_bound, residue_types=vocab.types,
                n_states=int(vocab.n_states), max_v=int(vocab.max_v), site_keys=site_keys, site_n=site_n,
                site_first_id=first_id, bank_keys=bank_keys, parameters=int(sum(v.numel() for v in weights.values())))
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(out, **arrays)
    print(out, round(os.path.getsize(out) / 1e6, 1), "MB;", len(keys), "kinds of site,", offsets[-1], "bank rows")

    if len(sys.argv) > 8:
        net = N.KineticNet(vocab, W=2, d=64, h=192)
        net.init_baseline(d["F0"], d["G0"], g_bound=g_bound)
        net.load_state_dict(weights)
        net.eval()
        rng, vec = np.random.default_rng(0), {}
        for seq in ("GCDE", "KRWP", "AAIT", "WNMA", "PYHF", "LVQS", "MTNG", "ACDEFGHIK"):
            res = [ONE[c] for c in seq]
            sl, kk = [], []
            for i, rn in enumerate(res):
                bbk = "2d" if 0 < i < len(res) - 1 else ("psi" if i == 0 else "phi")
                sl.append((i, "bb"))
                kk.append((rn, "bb-" + bbk))
                for c in range(1, 5):
                    if (rn, f"chi{c}") in sd.defs:
                        sl.append((i, f"chi{c}"))
                        kk.append((rn, f"chi{c}"))
            if any(k not in sd.defs for k in kk):            # a small training set may lack a kind of site
                continue
            pep = N.to_device(N.PeptideIndex(vocab, res, sl, kk, temperature=temperature), "cpu")
            cur = np.stack([rng.integers(0, int(m), size=48) for m in pep.site_n], 1)
            with torch.no_grad():
                logk, E = net.log_rates(pep, torch.as_tensor(cur))
            vec[seq + "_cur"], vec[seq + "_site_n"] = cur.astype(np.int16), np.asarray(pep.site_n, np.int16)
            vec[seq + "_logk"], vec[seq + "_E"] = logk.numpy().astype(np.float32), E.numpy().astype(np.float32)
        np.savez_compressed(sys.argv[8], **vec)
        print("reference vectors:", sys.argv[8])


if __name__ == "__main__":
    main()
