"""The README and docs/evidence.md quote the numbers that the files in results/ give, and no others."""
import importlib.util
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("recompute", os.path.join(HERE, "tools", "recompute.py"))
recompute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recompute)


def text(path):
    return open(os.path.join(HERE, path), encoding="utf-8").read()


def test_evidence_tables_are_the_recomputed_ones():
    doc = text("docs/evidence.md")
    block = re.search(r"<!-- numbers -->\n(.*?)\n<!-- /numbers -->", doc, re.S).group(1)
    assert block.strip() == (recompute.number_table() + "\n\n" + recompute.peptide_table()).strip()


def test_readme_quotes_the_recomputed_numbers():
    s = {k: v[1] for k, v in recompute.summary().items()}
    readme = " ".join(text("README.md").split())
    eq = "all: at equal length, "
    val = "validation: at equal length, "
    quoted = [
        # the first lines
        f"a microsecond in {s['timing: seconds per microsecond on one CPU core, median']} seconds on one CPU core",
        f"median of {s['sites: peptides']} peptides; {s['timing: seconds per microsecond, slowest']} at most",
        # the table
        f"within {s['sites: jump rate, model']} | within {s['sites: jump rate, MD halves']}",
        f"{s[eq + 'free-energy error, model']} off | {s[eq + 'free-energy error, MD']} off",
        f"off by a factor {s[eq + 'worst passage-time factor, model']} | off by a factor {s[eq + 'worst passage-time factor, MD']}",
        f"{s[eq + 'slowest relaxation error, model']} off | {s[eq + 'slowest relaxation error, MD']} off",
        f"| 1 | {s['other seed: ratio of GPU times']} to {s['cost, all: ratio of GPU times']} |",
        # each residue
        f"The {s['sites: peptides']} peptides have {s['sites: sites']} backbone and side-chain sites",
        f"to {s['sites: fastest site, jumps per ns']} times per ns",
        f"within {s['sites: jump rate, model']} of MD's (median; {s['sites: validation, jump rate, model']} on the ten validation",
        f"within a factor 1.5 for {s['sites: jump rates within 1.5, model']} of the sites",
        f"the {s['sites: sites that jump less than once in 10 ns']} sites that jump less than once in 10 ns are off by "
        f"{s['sites: their jump rate, model']}, where the MD's halves differ by {s['sites: their jump rate, MD halves']}, and "
        f"{s['sites: of them, faster in the model than in the MD']} of them are too fast",
        # states
        f"For {s['all: peptides']} of the {s['sites: peptides']} peptides the MD defines metastable states",
        f"wrong by {s[eq + 'free-energy error, model']} and its worst passage time by a factor {s[eq + 'worst passage-time factor, model']}",
        f"by {s[eq + 'free-energy error, MD']} and a factor {s[eq + 'worst passage-time factor, MD']}",
        f"it is {s[val + 'free-energy error, model']} against {s[val + 'free-energy error, MD']} and a factor "
        f"{s[val + 'worst passage-time factor, model']} against {s[val + 'worst passage-time factor, MD']}",
        f"with another seed {s['other seed: peptides with states']} peptides have states, and it is "
        f"{s['other seed: at equal length, free-energy error, model']} against {s['other seed: at equal length, free-energy error, MD']} and a "
        f"factor {s['other seed: at equal length, worst passage-time factor, model']} against "
        f"{s['other seed: at equal length, worst passage-time factor, MD']}",
        # slow transitions
        f"faster than MD's in {s['slow: peptides whose slowest relaxation is faster in the model']} of the {s['all: peptides']} peptides",
        f"in {s['slow: of which the model is faster']} of the {s['slow: peptides slower than 5 ns in MD']} where it takes more than 5 ns in MD, "
        f"by a factor {s['slow: by a median factor']} (median)",
        # the reference
        f"slowest relaxation time by {s[eq + 'slowest relaxation error, MD']}",
        f"on only {s['all: passage times above 100 ns within 1.5, MD halves']} of passage times above 100 ns",
        f"for {int(s['sites: peptides']) - int(s['all: peptides'])} of the {s['sites: peptides']} peptides "
        f"({int(s['sites: peptides']) - int(s['other seed: peptides with states'])} with the other seed) no set of states",
        # cost
        f"MD needs {s['cost: MD, GPU seconds per ns']} GPU-seconds per nanosecond",
        f"needs {s['cost: model, GPU seconds per microsecond']} seconds for a microsecond",
        f"MD needs {s['cost, all: MD needed to match, microseconds']} µs: an estimated {s['cost, all: ratio of GPU times']} times more "
        f"GPU time, or {s['other seed: ratio of GPU times']} on the states the other seed gives",
        f"would be {s['cost: MD with 4 fs steps, times cheaper']} times cheaper",
        # how it works, scope
        f"about {s['atoms: rebuilt MD frames, RMSD from'][:3]} Å",
        f"on {s['octapeptides: peptides']} octapeptides its jump rates are off by {s['octapeptides: jump rate, model']} (the MD's halves "
        f"by {s['octapeptides: jump rate, MD halves']}), and the {s['octapeptides: sites that jump less than once in 10 ns']} rarest by a "
        f"factor {s['octapeptides: their jump rate, model, a factor']}",
        f"on all {s['all: peptides']} with states",
    ]
    missing = [q for q in quoted if q not in readme]
    assert not missing, missing
    evidence = " ".join(text("docs/evidence.md").split())
    beside_the_recording = [
        f"Of the {s['all: peptides with at least two main states']} peptides scored on passage times and free energies, DMTL is "
        f"the {s['recording: DMTL, place by worst passage-time factor, least first']}th by either error",
        f"the {s['recording: DMTL, place by free-energy error, least first']}th by either error",
    ]
    assert not [q for q in beside_the_recording if q not in evidence]
    assert round(recompute.summary()["cost: MD, GPU hours per microsecond"][0]) == 6          # "six GPU-hours"
    assert round(1 / recompute.summary()["sites: slowest site, jumps per ns"][0], -1) == 150  # "once in 150 ns"
    assert s["reference: peptides whose halves meet every tolerance"] == "0"                 # "No peptide's halves"


def test_no_dashes_that_are_not_hyphens():
    for base, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for f in files:
            if f.endswith((".md", ".py", ".txt", ".json")) and f != "LICENSE":
                body = open(os.path.join(base, f), encoding="utf-8").read()
                assert chr(0x2014) not in body and chr(0x2013) not in body, os.path.join(base, f)
