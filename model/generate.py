"""Generate peptide site paths, torsions and optional heavy-atom coordinates on a CPU.

Usage: python model/generate.py SEQUENCE OUT_DIR [options]

SEQUENCE uses one-letter amino-acid codes. --ns is the total duration across
all trajectories, saved at 10 ps intervals. Rates are evaluated at 310 K.
For at most 150,000 configurations, starts are drawn from the generator's
Boltzmann distribution. Larger spaces start with every site in state zero
and have no equilibration period. Dynamics use simultaneous finite-step
updates in both cases.
"""

import argparse, io, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import data as D
import ctmc as CT
import sitepaths as SP
import zmatrix as ZM
import bundle as BU

FRAME_PS = 10.0
ONE2THREE = dict(A="ALA", R="ARG", N="ASN", D="ASP", C="CYS", Q="GLN", E="GLU", G="GLY", H="HIS", I="ILE", L="LEU",
                 K="LYS", M="MET", F="PHE", P="PRO", S="SER", T="THR", W="TRP", Y="TYR", V="VAL")
COLUMN = dict(phi=0, psi=1, chi1=2, chi2=3, chi3=4, chi4=5)


def heavy_template(seq):
    """Construct an uncapped ideal heavy-atom PDB from the amino-acid sequence."""
    import PeptideBuilder
    from PeptideBuilder import Geometry
    from Bio.PDB import PDBIO
    geos = []
    for aa in seq:
        g = Geometry.geometry(aa)
        g.phi, g.psi_im1 = -75.0, 145.0
        geos.append(g)
    st = PeptideBuilder.initialize_res(geos[0])
    for g in geos[1:]:
        st = PeptideBuilder.add_residue(st, g)
    PeptideBuilder.add_terminal_OXT(st)
    buf = io.StringIO()
    pio = PDBIO()
    pio.set_structure(st)
    pio.save(buf)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description="Generate peptide dynamics from a sequence at 310 K.")
    ap.add_argument("sequence", help="sequence in one-letter amino-acid codes")
    ap.add_argument("out", help="output directory for state paths, torsions and coordinates")
    ap.add_argument("--ns", type=float, default=1000.0, help="total ns across all trajectories (default: 1000)")
    ap.add_argument("--trajectories", type=int, default=10, help="number of equal-length paths (default: 10)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    ap.add_argument("--bundle", default=BU.DEFAULT, help="directory containing bundle.pkl and net.pt")
    ap.add_argument("--no-atoms", action="store_true", help="omit XTC coordinates; still write the template PDB and torsions")
    a = ap.parse_args()
    seq = a.sequence.upper()
    assert all(c in ONE2THREE for c in seq), "one-letter codes of the 20 standard amino acids"
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    b = BU.load(a.bundle)
    import mdtraj as md
    open(os.path.join(a.out, "heavy.pdb"), "w").write(heavy_template(seq))
    top_t = md.load(os.path.join(a.out, "heavy.pdb"))
    tb0 = D.torsion_table(top_t)
    sl = D.site_list(tb0["res"], tb0["tors"][0])
    keys = [D.site_key(tb0["res"][i], kind, D.bb_kind(tb0["tors"][0], i)) for i, kind in sl]
    missing = [k for k in keys if (k, 0) not in b["vocab"].state_id]
    assert not missing, f"sites the trained model has never seen: {missing}"
    nt, ct = CT.termini(tb0["tors"][0])
    pep = CT.to_device(CT.PeptideIndex(b["vocab"], tb0["res"], sl, keys, nterm=nt, cterm=ct, temperature=310.0), "cpu")
    per_ps = a.ns / a.trajectories * 1000.0
    table = CT.rate_table(b["net"], pep, "cpu", max_configs=150000, chunk=4096)
    if table is not None:
        x0 = CT.table_starts(table, a.trajectories, seed=a.seed)
        frames = CT.simulate_table(table, x0, per_ps, FRAME_PS, b["dt_ps"], seed=a.seed).astype(np.int64)
    else:
        torch.manual_seed(a.seed)
        x0 = torch.zeros(a.trajectories, pep.S, dtype=torch.long)
        frames = CT.simulate_tau(b["net"], pep, x0, per_ps, FRAME_PS, b["dt_ps"], "cpu").numpy().astype(np.int64)
    t_dyn = time.time() - t0
    rng = np.random.default_rng(a.seed)
    b["bank"].rng = rng
    bbks = [D.bb_kind(tb0["tors"][0], i) for i in range(len(tb0["res"]))]
    T = np.stack([SP.emit(b["bank"], tb0["res"], bbks, sl, frames[k]) for k in range(a.trajectories)])
    np.savez_compressed(os.path.join(a.out, "dynamics.npz"), sites=frames.astype(np.int8), torsions=T.astype(np.float32),
                        site_keys=np.array([f"{tb0['res'][i]}{i + 1}:{kind}" for i, kind in sl]),
                        residues=np.array(tb0["res"]), frame_ps=FRAME_PS)
    if not a.no_atoms:
        from mdtraj.formats import XTCTrajectoryFile
        Z = ZM.ZMatrix(top_t.topology, top_t.xyz[0])
        res_index = [r.index for r in top_t.topology.residues]
        for k in range(a.trajectories):
            tors = np.zeros((T.shape[1], len(Z.labels)))
            for j, lab in enumerate(Z.labels):
                if lab.startswith("omega"):
                    tors[:, j] = np.pi + rng.normal(0, 0.1, size=len(tors))
                    continue
                name, r = (lab.split("_")[0], int(lab.split("_")[1])) if "_" in lab else (lab[:3], int(lab[3:]))
                tors[:, j] = T[k, :, res_index.index(r), COLUMN[name]]
            X = Z.build(tors).astype(np.float32)
            with XTCTrajectoryFile(os.path.join(a.out, f"traj_{k:03d}.xtc"), "w") as f:
                f.write(X, time=np.arange(1, len(X) + 1) * FRAME_PS)
    cost = dict(sequence=seq, total_ns=a.ns, trajectories=a.trajectories, dynamics_s=t_dyn, total_s=time.time() - t0)
    json.dump(cost, open(os.path.join(a.out, "cost.json"), "w"), indent=1)
    print(json.dumps(cost))


if __name__ == "__main__":
    main()
