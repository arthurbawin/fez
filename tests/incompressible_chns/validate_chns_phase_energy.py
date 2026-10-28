"""Check per-phase energies, independent output options, MPI and restart.

Uses small analytic fields and an interface relaxation; no rising-bubble run.
An optional --legacy-solver checks a real V1 checkpoint from the old executable.
"""

import argparse
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
from validate_chns_postprocessing import close, compare_tables, configuration, table
from validate_quasi_incompressible import bdf2_fixture, check_rejected_configuration, run
from validate_stepien_postprocessing import PARTS, check_affine, checkpoint_text, fixture


def monitoring(text, *, phase=True, global_energy=True, conservation=False,
               write=True, frequency=1, prefix="chns_phase_energy", verbose=False):
    text = configuration(text, energy=global_energy, conservation=conservation)
    if phase is None:  # No new parameter section when running the old executable.
        return text
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
    assert not list(directory.glob("**/stepien_diagnostics.csv"))
    return rows


def analytic_fixture(fixtures, *, model="quasi_incompressible", phase="2*x-1", dim=2):
    text = fixture(fixtures, phase=phase, dim=dim)
    text = text.replace("set CHNS model = stepien", f"set CHNS model = {model}")
    text = text.replace("set write stepien diagnostics = true",
                        "set write stepien diagnostics = false")
    if model != "stepien":
        for option in ("stepien conservative transport", "stepien skew symmetric inertia"):
            text = text.replace(f"set {option} = true", f"set {option} = false")
    return text


def restart_suite(solver, workdir, text, reference, launch=(), legacy_solver=None):
    first = text.replace("set t_end            = 0.008", "set t_end            = 0.004")
    directory = workdir / "restart"
    # Save in-memory history even when no phase-energy text file was written.
    run(solver, directory, monitoring(first, write=False) + checkpoint_text(), launch)
    assert not list(directory.glob("**/chns_phase_energy.txt"))
    sidecar = next(directory.glob("**/checkpoint.postprocessing"))
    assert sidecar.read_bytes().startswith(b"FEZ_POSTPROCESSING_V2\n")
    run(solver, directory, monitoring(text) + checkpoint_text(restart=True), launch)
    compare_tables(check_rows(directory), reference)

    for name, initial_phase in (("enable_on_restart", False), ("no_sidecar", False)):
        directory = workdir / name
        run(solver, directory, monitoring(first, phase=initial_phase) + checkpoint_text(), launch)
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

    if legacy_solver is not None:
        directory = workdir / "legacy_v1"
        run(legacy_solver, directory, monitoring(first, phase=None) + checkpoint_text(), launch)
        sidecar = next(directory.glob("**/checkpoint.postprocessing"))
        assert sidecar.read_bytes().startswith(b"FEZ_POSTPROCESSING_V1\n")
        before = table(directory, "chns_energy")
        run(solver, directory, monitoring(text) + checkpoint_text(restart=True), launch)
        compare_tables(check_rows(directory),
                       [row for row in reference if row["time"] >= .004-1e-12])
        compare_tables(table(directory, "chns_energy")[:len(before)], before)
    print("Restart: full in-memory history, late activation, phase-only and old checkpoints")


def serial_suite(solver, workdir, fixtures, legacy_solver):
    for model in ("quasi_incompressible", "stepien"):
        for dim in (2, 3):
            directory = workdir / f"{model}_{dim}d"
            run(solver, directory, monitoring(analytic_fixture(fixtures, model=model, dim=dim)))
            rows = check_rows(directory)
            assert len(rows) == 5
            # Only check_affine's dissipation check needs a global-table column.
            check_affine([dict(row, viscous_dissipation=0.) for row in rows], dim=dim)
    print("Analytic split: all energy components in 2D/3D for both CHNS models")

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

    unsupported = text.replace("set CHNS model = quasi_incompressible", "set CHNS model = abels")
    unsupported = unsupported.replace("set mobility model      = degenerate_absolute",
                                      "set mobility model      = constant")
    check_rejected_configuration(solver, workdir / "unsupported", monitoring(
        unsupported, global_energy=False), "CHNS phase energy postprocessing requires")

    interface = bdf2_fixture((fixtures / "interface_quasi_incompressible.prm").read_text())
    directory = workdir / "interface"
    run(solver, directory, monitoring(interface))
    rows = check_rows(directory)
    assert len(rows) >= 5
    restart_suite(solver, workdir, interface, rows, legacy_solver=legacy_solver)


def mpi_suite(solver, workdir, fixtures, mpiexec):
    launch = (str(mpiexec), "-n", "2")
    for model in ("quasi_incompressible", "stepien"):
        text = analytic_fixture(fixtures, model=model)
        run(solver, workdir / f"{model}_serial", monitoring(text))
        run(solver, workdir / f"{model}_mpi", monitoring(text), launch)
        compare_tables(check_rows(workdir / f"{model}_mpi"),
                       check_rows(workdir / f"{model}_serial"))
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
    parser.add_argument("--legacy-solver", type=Path)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    args.workdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir.resolve()))
    print(f"Validation logs: {workdir}", flush=True)
    if args.mpiexec:
        mpi_suite(args.solver.resolve(), workdir, fixtures, args.mpiexec)
    else:
        serial_suite(args.solver.resolve(), workdir, fixtures,
                     args.legacy_solver.resolve() if args.legacy_solver else None)


if __name__ == "__main__":
    main()
