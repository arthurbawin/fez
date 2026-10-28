"""Check the quadratic-pressure gauge with a moving, uniform pure phase.

This small BDF2 case has no interface or gravity. It preserves the user's
diagnostic outputs by writing to the CTest work directory. No bubble is run.
"""

import argparse
import math
import os
import sys
from pathlib import Path

# Import shared checks without leaving bytecode in the source tree.
sys.dont_write_bytecode = True
from validate_quasi_incompressible import diagnostics, run


def check(directory):
    rows = diagnostics(directory)
    assert len(rows) == 6 and abs(rows[-1]["time"] - 0.02) < 1e-14
    phase_error = max(abs(row[key] + 1) for row in rows
                      for key in ("phi_min_q", "phi_max_q"))
    mass_drift = max(abs(row["mass_affine"] - rows[0]["mass_affine"])
                     for row in rows)
    assert phase_error < 1e-8, f"Uniform phase was not preserved: {phase_error}"
    assert mass_drift < 1e-9, f"Mass drift: {mass_drift}"
    assert min(row["density_affine_min_q"] for row in rows) > 0.99999
    assert rows[-1]["velocity_l2"] > 1e-3, "The pure-phase test must be moving"
    assert max(row["diffusive_flux_l2"] for row in rows) < 1e-9
    assert all(math.isfinite(row["total_energy"]) for row in rows)
    assert max(abs(row["temporal_energy_balance_residual"])
               for row in rows) < 1e-8
    print(f"{directory.name}: pure-phase error={phase_error:.3e}, "
          f"mass drift={mass_drift:.3e}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    solver = args.solver.resolve()
    workdir = args.workdir.resolve()
    text = (Path(__file__).parent /
            "pure_moving_quasi_incompressible.prm").read_text()
    serial = workdir / "p3_p2_serial"
    run(solver, serial, text)
    reference = check(serial)
    if args.mpiexec:
        parallel = workdir / "p3_p2_mpi2"
        run(solver, parallel, text,
            [str(args.mpiexec.resolve()), "-n", "2"])
        rows = check(parallel)
        for a, b in zip(reference, rows):
            for key in ("time", "phi_min_q", "phi_max_q", "mass_affine",
                        "velocity_l2", "total_energy"):
                assert abs(a[key] - b[key]) < 1e-9, \
                    f"Serial/MPI mismatch in {key}: {a[key]} != {b[key]}"
        print("Serial/two-rank agreement OK", flush=True)


if __name__ == "__main__":
    main()
