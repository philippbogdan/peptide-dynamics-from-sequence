"""Time one OpenMM simulation using the ManyPeptidesMD implicit-solvent protocol.

Usage: python analysis/bench_md.py PDB [SECONDS]

SECONDS defaults to 60 and measures integration after minimisation and
2,000 warm-up steps. Settings are amber14, OBC1, 310 K, 0.3/ps friction,
a 1 fs step and no constraints. Platform selection tries CUDA, OpenCL,
then CPU. Output is a JSON record with platform, atom count and ns/day.
This command times one simulation; the supplied T4 cost record used eight
concurrent simulations, so reproducing its aggregate rate needs that
concurrency and the same hardware.
"""

import json, sys, time
import openmm as mm, openmm.app as app, openmm.unit as u
pdb = app.PDBFile(sys.argv[1])
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60
ff = app.ForceField("amber14-all.xml", "implicit/obc1.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=2.0 * u.nanometer,
                         constraints=None)
integ = mm.LangevinMiddleIntegrator(310 * u.kelvin, 0.3 / u.picosecond, 1.0 * u.femtosecond)
ctx = None
for name in ("CUDA", "OpenCL", "CPU"):
    try:
        plat = mm.Platform.getPlatformByName(name)
        ctx = mm.Context(system, integ, plat)
        break
    except Exception:
        continue
ctx.setPositions(pdb.positions)
mm.LocalEnergyMinimizer.minimize(ctx)
integ.step(2000)
n, t0 = 0, time.time()
while time.time() - t0 < secs:
    integ.step(5000)
    n += 5000
ctx.getState(getEnergy=True)
dt = time.time() - t0
print(json.dumps(dict(pdb=sys.argv[1], platform=plat.getName(), atoms=pdb.topology.getNumAtoms(),
                      ns_per_day=n * 1e-6 / dt * 86400)))
