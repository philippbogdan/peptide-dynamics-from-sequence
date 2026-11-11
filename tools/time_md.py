#!/usr/bin/env python3
"""Throughput of implicit-solvent MD of one corpus peptide on one GPU (OpenMM, CUDA), for three protocols.

    python tools/time_md.py PEPTIDE.pdb PROTOCOL [SECONDS]     prints one JSON line with ns_per_day

    corpus  amber14 + OBC1, no constraints, LangevinMiddle at 310 K, friction 0.3 per ps, 1 fs steps: the protocol
            of ManyPeptidesMD, the corpus the model was trained and scored on
    hbonds  the same with bonds to hydrogen constrained and 2 fs steps
    hmr     the same with bonds to hydrogen constrained, hydrogen masses of 4 amu and 4 fs steps

results/cost/md_t4.json holds the aggregate of 1, 4, 8, 12 and 16 copies run at once on one Tesla T4 under the
NVIDIA multi-process service (without it, copies sharing a card slow each other down), 40 s each after
minimisation and 5,000 steps: eight copies give the most. The passage times of the two faster protocols were not
checked against the corpus protocol.
"""
import json
import sys
import time

import openmm as mm
import openmm.app as app
import openmm.unit as u


def main():
    pdb = app.PDBFile(sys.argv[1])
    protocol = sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 40
    ff = app.ForceField("amber14-all.xml", "implicit/obc1.xml")
    kw = dict(nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=2.0 * u.nanometer)
    if protocol == "corpus":
        system, dt = ff.createSystem(pdb.topology, constraints=None, **kw), 1.0
    elif protocol == "hbonds":
        system, dt = ff.createSystem(pdb.topology, constraints=app.HBonds, **kw), 2.0
    elif protocol == "hmr":
        system, dt = ff.createSystem(pdb.topology, constraints=app.HBonds, hydrogenMass=4.0 * u.amu, **kw), 4.0
    else:
        raise SystemExit("protocol: corpus, hbonds or hmr")
    integrator = mm.LangevinMiddleIntegrator(310 * u.kelvin, 0.3 / u.picosecond, dt * u.femtosecond)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
    context.setPositions(pdb.positions)
    mm.LocalEnergyMinimizer.minimize(context)
    integrator.step(5000)
    steps, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        integrator.step(5000)
        steps += 5000
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    wall = time.time() - t0
    print(json.dumps(dict(pdb=sys.argv[1].split("/")[-1], protocol=protocol, step_fs=dt, atoms=pdb.topology.getNumAtoms(),
                          ns_per_day=steps * dt * 1e-6 / wall * 86400, finite_energy=bool(energy == energy))))


if __name__ == "__main__":
    main()
