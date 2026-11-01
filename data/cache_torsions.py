"""Convert downloaded coordinates to compact torsion tables.

Usage: python data/cache_torsions.py {test,train} [options]

Read raw/{split}/*.npz and the corresponding PDB files, then write one NPZ
per sequence under torsions/{split}. Each output contains res, tors and
dt_ps. Training torsions use float16; test torsions use float32. Existing
outputs are reused without checking whether the requested stride changed.
"""

import argparse, glob, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "model"))


def one(args):
    """Extract a subsampled trajectory as a torsion table and atomically replace its output."""
    split, stride, raw, out_dir, f = args
    import numpy as np, mdtraj as md
    import data as D
    seq = os.path.basename(f)[:-4]
    out = os.path.join(out_dir, seq + ".npz")
    if os.path.exists(out):
        return out
    z = np.load(f, allow_pickle=False)
    top = md.load(os.path.join(raw, "pdb", split, seq + ".pdb")).topology
    tb = D.torsion_table(md.Trajectory(z["positions"][::stride], top))
    dt_ps = (1.0 if split == "train" else 10.0) * stride
    dtype = np.float16 if split == "train" else np.float32
    np.savez_compressed(out + ".tmp.npz", res=np.array(tb["res"]), tors=tb["tors"].astype(dtype), dt_ps=dt_ps)
    os.replace(out + ".tmp.npz", out)
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract torsion tables from downloaded peptide coordinates.")
    ap.add_argument("split", choices=("test", "train"), help="dataset split to convert")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth raw frame (default: 1; training preparation: 2)")
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"), help="input directory (default: data/raw)")
    ap.add_argument("--out", default=os.path.join(HERE, "torsions"), help="output parent; a split subdirectory is added (default: data/torsions)")
    ap.add_argument("--procs", type=int, default=4, help="concurrent conversion processes (default: 4)")
    a = ap.parse_args()
    out_dir = os.path.join(a.out, a.split)
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.raw, a.split, "*.npz")))
    with Pool(a.procs) as p:
        for r in p.imap_unordered(one, [(a.split, a.stride, a.raw, out_dir, f) for f in files]):
            print(r, flush=True)


if __name__ == "__main__":
    main()
