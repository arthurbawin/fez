"""Check per-phase energies, independent output options, MPI and restart.

Uses small analytic fields (applied exactly, without solving) and an
interface relaxation for the quasi-incompressible CHNS model.
"""

import argparse
import math
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
from validate_chns_postprocessing import close, compare_tables, configuration, table
from validate_quasi_incompressible import bdf2_fixture, check_rejected_configuration, run


PARTS = ("kinetic_energy", "bulk_energy", "gradient_energy",
         "interfacial_energy", "gravitational_energy", "total_energy")
SIGMA = 3 / (2 * math.sqrt(2))


def exact_expression(text, name, expression):
    pattern = rf"(subsection exact {name}\s+set Function expression = )[^\n]+"
    text, count = re.subn(pattern, lambda m: m[1] + expression, text)
    assert count == 1, f"Missing exact field {name}"
    return text


def analytic_fixture(fixtures, *, phase="2*x-1", dim=2):
    """Affine phase (by default) and uniform velocity on the unit square/cube."""
    text = (fixtures / "analytic_quasi_incompressible.prm").read_text()
    text = exact_expression(text, "cahn hilliard tracer", phase)
    if dim == 3:
        text = text.replace("set dimension = 2", "set dimension = 3")
        text = text.replace("2, 2 : 0., 0. : 1., 1. : true",
                            "2, 1, 1 : 0., 0., 0. : 1., 1., 1. : true")
        text = text.replace("set body force = 0.2, -0.3",
                            "set body force = 0.2, -0.3, 0.1")
        text = exact_expression(text, "velocity", "0.2; -0.1; 0.05")
        text = text.replace("set number = 4", "set number = 6")
        for boundary_type in ("velocity_mms", "dirichlet_mms"):
            marker = f"    set type = {boundary_type}\n  end\nend"
            additions = "".join(
                f"  subsection boundary {i}\n    set id = {i}\n"
                f"    set name = z_{side}\n    set type = {boundary_type}\n  end\n"
                for i, side in ((4, "min"), (5, "max")))
            assert marker in text
            text = text.replace(marker,
                                f"    set type = {boundary_type}\n  end\n" + additions + "end", 1)
    return text


def check_affine(rows, dim=2, *, sigma=SIGMA, epsilon=1.):
    """Exact energies of phi = 2x-1, rho = 1+19x, |v|^2 = 0.05 (+0.0025 in 3D)."""
    speed_squared = .05 + (.0025 if dim == 3 else 0.)
    for row in rows:
        for i, lo, hi in ((0, .5, 1.), (1, 0., .5)):
            mass = hi-lo + 19/2 * (hi*hi-lo*lo)
            x_moment = (hi*hi-lo*lo)/2 + 19/3 * (hi**3-lo**3)
            suffix = f"_phase_{i}"
            close(row["kinetic_energy" + suffix], .5*speed_squared*mass, "affine kinetic")
            close(row["bulk_energy" + suffix], sigma/(15*epsilon), "affine bulk")
            close(row["gradient_energy" + suffix], sigma*epsilon, "affine gradient")
            close(row["gravitational_energy" + suffix],
                  -.2*x_moment + (.1 if dim == 3 else .15)*mass, "affine gravity")


def checkpoint_text(restart=False):
    return ("\nsubsection Checkpoint Restart\n"
            f"  set enable checkpoint = {str(not restart).lower()}\n"
            "  set checkpoint frequency = 1\n"
            f"  set restart = {str(restart).lower()}\nend\n")


def monitoring(text, *, phase=True, global_energy=True, conservation=False,
               write=True, frequency=1, prefix="chns_phase_energy", verbose=False):
    text = configuration(text, energy=global_energy, conservation=conservation)
    return text + ("\nsubsection Postprocessing\n  subsection chns phase energy\n"
                   f"    set enable = {str(phase).lower()}\n"
                   f"    set write results = {str(write).lower()}\n"
                   f"    set output frequency = {frequency}\n"
                   f"    set output prefix = {prefix}\n"
                   "    set precision = 16\n"
                   f"    set verbosity = {'verbose' if verbose else 'quiet'}\n"
                   "  end\nend\n")


def check_rows(directory, *, prefix="chns_phase_energy", global_energy=True):
    rows = table(directory, prefix)
    expected = {"time", "iteration"} | {
        f"{part}_phase_{i}" for part in PARTS for i in (0, 1)}
    for row in rows:
        assert row.keys() == expected, f"Unexpected phase-energy columns: {row.keys() ^ expected}"
        for i in (0, 1):
            suffix = f"_phase_{i}"
            close(row["bulk_energy" + suffix] + row["gradient_energy" + suffix],
                  row["interfacial_energy" + suffix], "interfacial sum")
            close(sum(row[part + suffix] for part in
                      ("kinetic_energy", "interfacial_energy", "gravitational_energy")),
                  row["total_energy" + suffix], "total sum")
    if global_energy:
        totals = {row["time"]: row for row in table(directory, "chns_energy")}
        for row in rows:
            total = totals[row["time"]]
            for part in PARTS:
                close(row[f"{part}_phase_0"] + row[f"{part}_phase_1"],
                      total[part], part)
    else:
        assert not list(directory.glob("**/chns_energy.txt"))
    assert not list(directory.glob("**/*.vtu")), "Phase energy must work without VTU"
    assert not list(directory.glob("**/quasi_incompressible_diagnostics.csv"))
    return rows


def restart_suite(solver, workdir, text, reference, launch=()):
    first = text.replace("set t_end            = 0.008", "set t_end            = 0.004")
    directory = workdir / "restart"
    # Save in-memory history even when no phase-energy text file was written.
    run(solver, directory, monitoring(first, write=False) + checkpoint_text(), launch)
    assert not list(directory.glob("**/chns_phase_energy.txt"))
    sidecar = next(directory.glob("**/checkpoint.postprocessing"))
    assert sidecar.read_bytes().startswith(b"FEZ_POSTPROCESSING_V1\n")
    run(solver, directory, monitoring(text) + checkpoint_text(restart=True), launch)
    compare_tables(check_rows(directory), reference)

    for name in ("enable_on_restart", "no_sidecar"):
        directory = workdir / name
        run(solver, directory, monitoring(first, phase=False) + checkpoint_text(), launch)
        before = table(directory, "chns_energy")
        if name == "no_sidecar":
            sidecar = next(directory.glob("**/checkpoint.postprocessing"))
            sidecar.rename(sidecar.with_suffix(".postprocessing.backup"))
        run(solver, directory, monitoring(text) + checkpoint_text(restart=True), launch)
        rows = check_rows(directory)
        compare_tables(rows, [row for row in reference if row["time"] >= .004-1e-12])
        if name == "enable_on_restart":
            # The existing energy table must retain its original rows exactly.
            compare_tables(table(directory, "chns_energy")[:len(before)], before)

    # A table can also be the only enabled monitor before and after a restart.
    directory = workdir / "phase_only_restart"
    run(solver, directory, monitoring(first, global_energy=False, write=False)
        + checkpoint_text(), launch)
    run(solver, directory, monitoring(text, global_energy=False)
        + checkpoint_text(restart=True), launch)
    compare_tables(check_rows(directory, global_energy=False), reference)
    print("Restart: full in-memory history, late activation, phase-only and missing sidecar")


def serial_suite(solver, workdir, fixtures):
    for dim in (2, 3):
        directory = workdir / f"analytic_{dim}d"
        run(solver, directory, monitoring(analytic_fixture(fixtures, dim=dim)))
        rows = check_rows(directory)
        assert len(rows) == 5
        check_affine(rows, dim=dim)
    print("Analytic split: all energy components in 2D/3D")

    for phase, index, density in ((1, 0, 20.), (-1, 1, 1.), (0, 0, 10.5)):
        directory = workdir / f"uniform_{phase}"
        run(solver, directory, monitoring(analytic_fixture(fixtures, phase=str(phase))))
        for row in check_rows(directory):
            for part in PARTS:
                close(row[f"{part}_phase_{1-index}"], 0., "absent phase")
            close(row[f"kinetic_energy_phase_{index}"], .025*density)
            close(row[f"gravitational_energy_phase_{index}"], .05*density)
    print("Uniform phases: absent-phase energies vanish; phi=0 belongs to phase 0")

    text = analytic_fixture(fixtures)
    for name, options in (
        ("phase_only", dict(global_energy=False)),
        ("disabled", dict(phase=False)),
        ("no_write", dict(write=False)),
        ("custom_frequency", dict(prefix="custom_phase_energy", frequency=3)),
        ("verbose", dict(verbose=True)),
    ):
        directory = workdir / name
        log = run(solver, directory, monitoring(text, **options))
        prefix = options.get("prefix", "chns_phase_energy")
        exists = bool(list(directory.glob(f"**/{prefix}.txt")))
        assert exists == (options.get("phase", True) and options.get("write", True))
        if exists:
            assert len(check_rows(directory, prefix=prefix,
                                  global_energy=options.get("global_energy", True))) == 5
        assert ("CHNS phase energy at t=" in log) == options.get("verbose", False)
    print("Independent enable/write, prefix, verbosity and final output between frequency multiples")

    unsupported = text.replace("set CHNS model          = quasi_incompressible",
                               "set CHNS model          = abels")
    unsupported = unsupported.replace("set mobility model      = degenerate_absolute",
                                      "set mobility model      = constant")
    check_rejected_configuration(solver, workdir / "unsupported", monitoring(
        unsupported, global_energy=False),
        "postprocessings require CHNS model = quasi_incompressible")

    interface = bdf2_fixture((fixtures / "interface_quasi_incompressible.prm").read_text())
    directory = workdir / "interface"
    run(solver, directory, monitoring(interface))
    rows = check_rows(directory)
    assert len(rows) >= 5
    restart_suite(solver, workdir, interface, rows)


def mpi_suite(solver, workdir, fixtures, mpiexec):
    launch = (str(mpiexec), "-n", "2")
    text = analytic_fixture(fixtures)
    run(solver, workdir / "analytic_serial", monitoring(text))
    run(solver, workdir / "analytic_mpi", monitoring(text), launch)
    compare_tables(check_rows(workdir / "analytic_mpi"),
                   check_rows(workdir / "analytic_serial"))
    interface = bdf2_fixture((fixtures / "interface_quasi_incompressible.prm").read_text())
    run(solver, workdir / "interface_serial", monitoring(interface))
    run(solver, workdir / "interface_mpi", monitoring(interface), launch)
    rows = check_rows(workdir / "interface_mpi")
    compare_tables(rows, check_rows(workdir / "interface_serial"))
    restart_suite(solver, workdir, interface, rows, launch)
    print("One/two MPI ranks: analytic energies, solved BDF2 interface and checkpoint history agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir.resolve()))
    print(f"Validation logs: {workdir}", flush=True)
    if args.mpiexec:
        mpi_suite(args.solver.resolve(), workdir, fixtures, args.mpiexec)
    else:
        serial_suite(args.solver.resolve(), workdir, fixtures)


if __name__ == "__main__":
    main()
