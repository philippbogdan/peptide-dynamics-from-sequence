"""Fetch ManyPeptidesMD tetrapeptide coordinates and PDB topologies.

Usage: python data/download.py {test,train} [--names SEQUENCE ...] [--out DIR]

Sequence lists default to this folder's test_tetrapeptides.txt or
train_tetrapeptides.txt. Raw files remain outside the published repository.
The dataset and its derived material retain the terms in NOTICE.md.
"""

import argparse, os, shutil, tarfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://huggingface.co/datasets/transferable-samplers/many-peptides-md/resolve/main/"


def fetch(path, dest):
    """Download through a temporary file; an existing nonempty destination is reused."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(BASE + path) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser(description="Download public ManyPeptidesMD tetrapeptide files.")
    ap.add_argument("split", choices=("test", "train"), help="dataset split to download")
    ap.add_argument("--names", nargs="*", help="sequence names; default: every name in the split list")
    ap.add_argument("--out", default=os.path.join(HERE, "raw"), help="raw-data directory (default: data/raw)")
    a = ap.parse_args()
    names = a.names or open(os.path.join(HERE, f"{a.split}_tetrapeptides.txt")).read().split()
    tar = os.path.join(a.out, "pdb", f"{a.split}.tar")
    fetch(f"pdb_tarfiles/{a.split}.tar", tar)
    with tarfile.open(tar) as t:
        t.extractall(os.path.join(a.out, "pdb"), filter="data")
    for i, n in enumerate(names):
        sub = f"{n[0]}/" if a.split == "train" else ""
        fetch(f"trajectories/{a.split}/4AA/{sub}{n}.npz", os.path.join(a.out, a.split, n + ".npz"))
        print(f"{i + 1}/{len(names)} {n}", flush=True)


if __name__ == "__main__":
    main()
