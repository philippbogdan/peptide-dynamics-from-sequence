"""Save or load the network and the training-derived objects needed for generation.

Usage: python model/bundle.py PREP.pkl NET.pt OUT_DIR G_BOUND

The command writes bundle.pkl and net.pt under OUT_DIR. The network defaults
are a residue-window radius of 2, embedding width 64 and hidden width 192;
these must match the checkpoint. G_BOUND must match the training setting.
"""

import gzip, os, pickle, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import ctmc as CT

DEFAULT = os.path.join(HERE, "bundle")


def load(bdir=DEFAULT, device="cpu"):
    """Load the state definitions, emission bank, baselines and network in evaluation mode."""
    b = pickle.load(open(os.path.join(bdir, "bundle.pkl"), "rb"))
    net = CT.KineticNet(b["vocab"], W=b["W"], d=b["d"], h=b["h"])
    net.init_baseline(b["F0"], b["G0"], g_bound=b["g_bound"])
    net.load_state_dict(torch.load(os.path.join(bdir, "net.pt"), map_location=device))
    b["net"] = net.to(device).eval()
    return b


def make(prep, netf, out, g_bound, W=2, d=64, h=192):
    """Extract generation inputs from a prepared training set and copy the checkpoint."""
    p = pickle.load(gzip.open(prep, "rb") if prep.endswith(".gz") else open(prep, "rb"))
    os.makedirs(out, exist_ok=True)
    pickle.dump(dict(sitedefs=p["sitedefs"], vocab=p["vocab"], bank=p["bank"], F0=p["F0"], G0=p["G0"],
                     dt_ps=p["dt_ps"], W=W, d=d, h=h, g_bound=float(g_bound)),
                open(os.path.join(out, "bundle.pkl"), "wb"), protocol=4)
    shutil.copy(netf, os.path.join(out, "net.pt"))


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
    print("bundle", sys.argv[3])
