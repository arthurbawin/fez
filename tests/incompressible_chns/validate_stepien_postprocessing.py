"""Verify Stepien energy tables with analytic fields, MPI and restart.

Manufactured fields isolate the quadrature and phase partition from solver
errors. A separate small nonlinear solve compares the standard table with the
Stepien CSV. Uses only Python's standard library and retains all solver logs.
"""

import argparse
import csv
import math
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
from validate_quasi_incompressible import run
from validate_chns_postprocessing import close, compare_tables, table


PARTS = ("kinetic_energy", "bulk_energy", "gradient_energy",
         "interfacial_energy", "gravitational_energy", "total_energy")
SIGMA = 3 / (2 * math.sqrt(2))


def configuration(text, *, enabled=True, write=True, frequency=1,
                  prefix="chns_energy"):
    return text + ("\nsubsection Postprocessing\n"
                   "  subsection chns energy\n"
                   f"    set enable = {str(enabled).lower()}\n"
                   f"    set write results = {str(write).lower()}\n"
                   f"    set output frequency = {frequency}\n"
                   f"    set output prefix = {prefix}\n"
                   "    set precision = 16\n"
                   "    set verbosity = quiet\n"
                   "  end\nend\n")


def exact_expression(text, name, expression):
    pattern = rf"(subsection exact {name}\s+set Function expression = )[^\n]+"
    text, count = re.subn(pattern, lambda m: m[1] + expression, text)
    assert count == 1, f"Missing exact field {name}"
    return text


def fixture(fixtures, *, phase="2*x-1", dim=2, limiter=False, end="0.04"):
    text = (fixtures / "mms_stepien_article.prm").read_text()
    text = text.replace("set scheme           = BDF2", "set scheme           = BDF1")
    text = text.replace("set t_end            = 0.02", f"set t_end            = {end}")
    text = text.replace("set compare jacobian matrix with finite differences = true",
                        "set compare jacobian matrix with finite differences = false")
    text = text.replace("set apply exact solution                   = false",
                        "set apply exact solution                   = true")
    text = text.replace("set body force = 0, 0", "set body force = 0.2, -0.3")
    text = text.replace("set surface tension     = 1",
                        f"set enable tracer limiter = {str(limiter).lower()}\n"
                        "  set surface tension     = 1")
    for name, expression in (("velocity", "0.2; -0.1"),
                             ("pressure", "0.1*(x+y-1)"),
                             ("cahn hilliard tracer", phase),
                             ("cahn hilliard potential", "0.1+0.05*x")):
        text = exact_expression(text, name, expression)
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


def check_rows(directory, prefix="chns_energy"):
    rows = table(directory, prefix)
    expected = {"time", "iteration", "reference_time", *PARTS,
                "energy_change_from_reference", "viscous_dissipation",
                "diffusive_dissipation"}
    expected |= {f"{part}_phase_{i}" for part in PARTS for i in (0, 1)}
    for row in rows:
        assert row.keys() == expected, f"Unexpected Stepien columns: {row.keys() ^ expected}"
        assert all(math.isfinite(value) for value in row.values())
        for part in PARTS:
            close(row[f"{part}_phase_0"] + row[f"{part}_phase_1"], row[part], part)
        for suffix in ("", "_phase_0", "_phase_1"):
            close(row["bulk_energy" + suffix] + row["gradient_energy" + suffix],
                  row["interfacial_energy" + suffix], "interfacial sum")
            close(sum(row[part + suffix] for part in
                      ("kinetic_energy", "interfacial_energy", "gravitational_energy")),
                  row["total_energy" + suffix], "total sum")
        close(row["energy_change_from_reference"],
              row["total_energy"] - rows[0]["total_energy"], "reference energy")
        assert row["viscous_dissipation"] >= -1e-13
        assert row["diffusive_dissipation"] >= -1e-13
    assert not list(directory.glob("**/chns_conservation.txt")), "Conservation was not enabled"
    assert not list(directory.glob("**/*.vtu")), "Energy must work without VTU"
    return rows


def compare_csv(directory):
    rows = check_rows(directory)
    paths = list(directory.glob("**/stepien_diagnostics.csv"))
    assert len(paths) == 1
    with paths[0].open() as stream:
        diagnostics = [{key: float(value) for key, value in row.items()}
                       for row in csv.DictReader(stream)]
    assert len(rows) == len(diagnostics)
    for row, diagnostic in zip(rows, diagnostics):
        for key in ("time", "iteration", "kinetic_energy", "interfacial_energy",
                    "gravitational_energy", "total_energy", "viscous_dissipation",
                    "diffusive_dissipation"):
            close(row[key], diagnostic[key], f"CSV {key}")


def check_affine(rows, dim=2, *, sigma=SIGMA, epsilon=1.):
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
        close(row["viscous_dissipation"], 0.)


def checkpoint_text(restart=False):
    return ("\nsubsection Checkpoint Restart\n"
            f"  set enable checkpoint = {str(not restart).lower()}\n"
            "  set checkpoint frequency = 1\n"
            f"  set restart = {str(restart).lower()}\nend\n")


def restart_suite(solver, workdir, text, reference, launch=()):
    directory = workdir / "restart"
    first = text.replace("set t_end            = 0.04", "set t_end            = 0.02")
    # Save the complete in-memory table without relying on a previous text file.
    run(solver, directory, configuration(first, write=False) + checkpoint_text(), launch)
    assert not list(directory.glob("**/chns_energy.txt"))
    assert list(directory.glob("**/checkpoint.postprocessing"))
    (directory / "solver.log").rename(directory / "before_restart.log")
    run(solver, directory, configuration(text) + checkpoint_text(restart=True), launch)
    rows = check_rows(directory)
    compare_tables(rows, reference)
    assert len(rows) == 5

    directory = workdir / "restart_enable_energy"
    run(solver, directory, configuration(first, enabled=False) + checkpoint_text(), launch)
    assert not list(directory.glob("**/chns_energy.txt"))
    (directory / "solver.log").rename(directory / "before_restart.log")
    run(solver, directory, configuration(text) + checkpoint_text(restart=True), launch)
    rows = check_rows(directory)
    assert len(rows) == 3, "New monitoring must include the restart snapshot and subsequent steps"
    close(rows[0]["time"], .02)
    for row in rows:
        close(row["reference_time"], .02, "new monitoring reference")
    expected = [dict(row, reference_time=.02,
                     energy_change_from_reference=row["total_energy"]-reference[2]["total_energy"])
                for row in reference[2:]]
    compare_tables(rows, expected)


def serial_suite(solver, workdir, fixtures):
    text = fixture(fixtures)
    directory = workdir / "affine"
    run(solver, directory, configuration(text))
    rows = check_rows(directory)
    assert len(rows) == 5
    check_affine(rows)
    compare_csv(directory)
    print("Affine two-phase field: analytic kinetic, bulk, gradient and gravity in each phase")

    directory = workdir / "scaled_interface"
    scaled = text.replace("set surface tension     = 1", "set surface tension     = 2")
    scaled = scaled.replace("set interface thickness = 1", "set interface thickness = 0.25")
    run(solver, directory, configuration(scaled))
    check_affine(check_rows(directory), sigma=2*SIGMA, epsilon=.25)
    compare_csv(directory)
    print("Surface tension and interface thickness have the expected independent prefactors")

    for phase, density, i in ((1, 20., 0), (-1, 1., 1)):
        directory = workdir / f"pure_{i}"
        run(solver, directory, configuration(fixture(fixtures, phase=str(phase))))
        for row in check_rows(directory):
            close(row["kinetic_energy"], .025*density)
            close(row["interfacial_energy"], 0.)
            close(row["gravitational_energy"], .05*density)
            for part in PARTS:
                close(row[f"{part}_phase_{1-i}"], 0., "absent phase")
        compare_csv(directory)
    print("Uniform translation in either pure phase has the expected energy and an empty other phase")

    directory = workdir / "zero_phase"
    run(solver, directory, configuration(fixture(fixtures, phase="0")))
    for row in check_rows(directory):
        close(row["bulk_energy"], SIGMA/4)
        close(row["gradient_energy"], 0.)
        for part in PARTS:
            close(row[part + "_phase_1"], 0., "zero phase belongs to phase 0")

    for limiter in (False, True):
        directory = workdir / f"overshoot_{limiter}"
        run(solver, directory, configuration(fixture(fixtures, phase="-1.2", limiter=limiter)))
        for row in check_rows(directory):
            close(row["kinetic_energy"], .025*(1. if limiter else -.9), "limited kinetic density")
            close(row["gravitational_energy"], -.045, "affine gravity density")
            close(row["bulk_energy"], SIGMA/4*(1.2**2-1)**2, "unlimited double well")
            close(row["gradient_energy"], 0.)
        compare_csv(directory)
    print("Property limiter affects kinetic density; gravity and free energy retain the raw phase")

    directory = workdir / "affine_3d"
    run(solver, directory, configuration(fixture(fixtures, dim=3, end="0.01")))
    check_affine(check_rows(directory), dim=3)
    compare_csv(directory)
    print("Three-dimensional affine energy and phase split agree with analytic integrals")

    for name, options in (("frequency", dict(frequency=3, prefix="custom_energy")),
                          ("disabled", dict(enabled=False)),
                          ("no_write", dict(write=False))):
        directory = workdir / name
        run(solver, directory, configuration(text, **options))
        paths = list(directory.glob("**/*energy.txt"))
        if name == "frequency":
            compare_tables(check_rows(directory, "custom_energy"), rows)
        else:
            assert not paths, f"Unexpected output for {name}"
    print("Enable/write, custom prefix and output frequency are independent of VTU and retain every step")

    directory = workdir / "no_csv"
    standard = text.replace("set write stepien diagnostics = true",
                            "set write stepien diagnostics = false")
    standard = standard.replace("set stepien conservative transport = true",
                                "set stepien conservative transport = false")
    standard = standard.replace("set stepien skew symmetric inertia = true",
                                "set stepien skew symmetric inertia = false")
    run(solver, directory, configuration(standard))
    compare_tables(check_rows(directory), rows)
    assert not list(directory.glob("**/stepien_diagnostics.csv"))
    print("The default Stepien forms produce the same energy table without detailed diagnostics")

    solved = (fixtures / "mms_stepien_article.prm").read_text().replace(
        "set compare jacobian matrix with finite differences = true",
        "set compare jacobian matrix with finite differences = false")
    directory = workdir / "solved_bdf2"
    run(solver, directory, configuration(solved))
    compare_csv(directory)
    print("An actual BDF2 nonlinear solve agrees with the detailed Stepien CSV")
    restart_suite(solver, workdir, text, rows)
    print("Checkpoint restart restores tables without duplicates; newly enabled monitoring starts at the restart time")


def mpi_suite(solver, workdir, fixtures, mpiexec):
    text = fixture(fixtures)
    launch = (str(mpiexec), "-n", "2")
    for name, command in (("serial", ()), ("parallel", launch)):
        directory = workdir / name
        run(solver, directory, configuration(text), command)
        check_affine(check_rows(directory))
        compare_csv(directory)
    reference = check_rows(workdir / "serial")
    compare_tables(check_rows(workdir / "parallel"), reference)
    restart_suite(solver, workdir, text, reference, launch)
    print("One/two-rank energies, phase sums and checkpoint restart agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path)
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    # Keep earlier logs while making repeated CTest runs independent of old outputs.
    workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir.resolve()))
    print(f"Stepien postprocessing logs: {workdir}", flush=True)
    fixtures = Path(__file__).resolve().parent
    if args.mpiexec:
        mpi_suite(args.solver.resolve(), workdir, fixtures, args.mpiexec)
    else:
        serial_suite(args.solver.resolve(), workdir, fixtures)


if __name__ == "__main__":
    main()
