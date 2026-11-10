"""Small CHNS postprocessing checks, independent of VTU and mesh quality.

No rising-bubble simulation is launched. Uses only Python's standard library.
"""

import argparse
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from validate_quasi_incompressible import run, bdf2_fixture, diagnostics


def configuration(text, *, detailed=False, conservation=True, energy=True,
                  write=True, frequency=1, custom=False):
    text = text.replace("set write quasi incompressible diagnostics = true",
                        f"set write quasi incompressible diagnostics = {str(detailed).lower()}")
    text += "\nsubsection Postprocessing\n"
    for section, enabled in (("conservation", conservation), ("energy", energy)):
        text += (f"  subsection chns {section}\n"
                 f"    set enable = {str(enabled).lower()}\n"
                 f"    set write results = {str(write).lower()}\n"
                 f"    set output frequency = {frequency}\n"
                 "    set precision = 16\n"
                 "    set verbosity = quiet\n")
        if custom:
            text += f"    set output prefix = custom_{section}\n"
        text += "  end\n"
    return text + "end\n"


def table(directory, name):
    paths = list(directory.glob(f"**/{name}.txt"))
    assert len(paths) == 1, f"Expected exactly one {name} table in {directory}"
    lines = [line.split() for line in paths[0].read_text().splitlines() if line.strip()]
    header = lines[0]
    assert len(header) == len(set(header)), "Repeated table columns"
    assert all(len(row) == len(header) for row in lines[1:])
    rows = [dict(zip(header, map(float, row))) for row in lines[1:]]
    assert rows and all(b["time"] > a["time"] for a, b in zip(rows, rows[1:])), \
        "Postprocessing contains duplicate times"
    return rows


def close(a, b, name="quantity", tol=2e-10):
    if math.isnan(b):
        assert math.isnan(a), f"Expected unavailable {name}"
    else:
        assert math.isfinite(a) and abs(a-b) < tol * max(1., abs(b)), \
            f"Wrong {name}: {a} vs {b}"


def compare_tables(a, b):
    assert len(a) == len(b), f"Different table lengths: {len(a)} vs {len(b)}"
    for actual, expected in zip(a, b):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            close(actual[key], value, key)


def compare_diagnostics(directory):
    c, e, old = table(directory, "chns_conservation"), table(directory, "chns_energy"), diagnostics(directory)
    assert len(c) == len(e) == len(old)
    for conservation, energy, diagnostic in zip(c, e, old):
        for key in ("time", "iteration", "domain_volume", "phase_integral", "mass_affine", "mass_positive"):
            close(conservation[key], diagnostic[key], key)
        for key in energy.keys() & diagnostic.keys():
            close(energy[key], diagnostic[key], key)
        close(energy["bulk_energy"] + energy["gradient_energy"], energy["interfacial_energy"])
        close(energy["kinetic_energy"] + energy["interfacial_energy"] + energy["gravitational_energy"], energy["total_energy"])
        close(conservation["mass_change_from_reference"], conservation["mass_affine"] - c[0]["mass_affine"])
        close(conservation["mass_relative_change"], (conservation["mass_affine"] - c[0]["mass_affine"]) / c[0]["mass_affine"])
        close(conservation["phase_change_from_reference"], conservation["phase_integral"] - c[0]["phase_integral"])
        close(conservation["phase_volume_relative_change"], (conservation["phase_integral"] - c[0]["phase_integral"]) / c[0]["domain_volume"])
        close(energy["energy_change_from_reference"], energy["total_energy"] - e[0]["total_energy"])
    assert not list(directory.glob("**/*.vtu")), "Test unexpectedly depends on VTU"


def analytic_initial_state(solver, directory, fixtures):
    text = (fixtures / "mms_quasi_incompressible_bdf2.prm").read_text()
    text = text.replace("set t_end            = 0.2", "set t_end            = 0.02")
    run(solver, directory, configuration(text))
    c, e = table(directory, "chns_conservation")[0], table(directory, "chns_energy")[0]
    a, b, d = .2, .05, .03
    # Exact integrals of (a+b*x+d*y)^k over the unit square.
    def moment(k):
        return sum(math.factorial(k) / (math.factorial(i)*math.factorial(j)*math.factorial(k-i-j)) *
                   a**i * b**j * d**(k-i-j) / ((j+1)*(k-i-j+1))
                   for i in range(k+1) for j in range(k-i+1))
    close(c["phase_integral"], .24, "affine phase integral")
    close(c["mass_affine"], 10.5+9.5*.24, "affine mass integral")
    close(e["bulk_energy"], 3/(8*math.sqrt(2))*(moment(4)-2*moment(2)+1), "bulk energy")
    close(e["gradient_energy"], 3/(4*math.sqrt(2))*(b*b+d*d), "gradient energy")
    rho = 10.5+9.5*a
    kinetic = .5*.02**2 * (rho*7/3 + 9.5*b*7/6 + 9.5*d*17/12)
    gravity = -.2*(rho/2+9.5*b/3+9.5*d/4) + .3*(rho/2+9.5*b/4+9.5*d/3)
    close(e["kinetic_energy"], kinetic, "kinetic energy")
    close(e["gravitational_energy"], gravity, "gravitational energy")
    assert not list(directory.glob("**/quasi_incompressible_diagnostics.csv"))
    print("Analytic phase, mass and all four energy components agree")


def serial_suite(solver, workdir, fixtures):
    pure = (fixtures / "pure_quasi_incompressible.prm").read_text()
    zero = pure.replace("set Function expression = 1 # pure phase", "set Function expression = 0 # pure phase")
    zero = zero.replace("set body force = 0, -1", "set body force = 0, 0")
    directory = workdir / "zero_phase"
    text = configuration(zero)
    text += ("\nsubsection Postprocessing\n  subsection chns phase volume\n"
             "    set enable = true\n    set verbosity = quiet\n"
             "    set precision = 16\n    set output prefix = geometric_volumes\n  end\nend\n")
    run(solver, directory, text)
    c, e = table(directory, "chns_conservation"), table(directory, "chns_energy")
    assert len(c) == len(e) == 5
    for row in c:
        close(row["mass_affine"], 500.5)
        close(row["phase_integral"], 0.)
        close(row["phase_volume_relative_change"], 0.)
    for row in e:
        close(row["bulk_energy"], 3/(8*math.sqrt(2)*.1))
        close(row["gradient_energy"], 0.)
    assert len(table(directory, "geometric_volumes")) == 5
    assert not list(directory.glob("**/*.vtu"))
    assert not list(directory.glob("**/quasi_incompressible_diagnostics.csv"))
    print("Zero initial phase: finite conservation errors, analytic energy, old indicators independent of mesh quality")

    for name, options in (("conservation_only", dict(energy=False)),
                          ("energy_only", dict(conservation=False)),
                          ("custom_frequency", dict(custom=True, frequency=2)),
                          ("disabled", dict(conservation=False, energy=False)),
                          ("no_write", dict(write=False))):
        directory = workdir / name
        run(solver, directory, configuration(zero, **options))
        prefix = "custom_" if options.get("custom") else "chns_"
        for quantity in ("conservation", "energy"):
            exists = bool(list(directory.glob(f"**/{prefix}{quantity}.txt")))
            expected = options.get(quantity, True) and options.get("write", True)
            assert exists == expected, f"Wrong enable/write behavior for {name}/{quantity}"
            if exists:
                assert len(table(directory, prefix+quantity)) == 5
    print("Independent enable/write options, prefixes and output frequency verified")

    extension = zero.replace("set Function expression = 0 # pure phase", "set Function expression = -1.02 # pure phase")
    directory = workdir / "density_extension"
    run(solver, directory, configuration(extension))
    for row in table(directory, "chns_conservation"):
        close(row["mass_affine"], -8.99)
        close(row["mass_positive"], 1.)
    print("Conserved affine density stays distinct from the positive kinetic extension")

    interface = (fixtures / "interface_quasi_incompressible.prm").read_text()
    for name, text in (("bdf1", interface), ("bdf2", bdf2_fixture(interface))):
        directory = workdir / name
        run(solver, directory, configuration(text, detailed=True))
        compare_diagnostics(directory)
    print("BDF1/BDF2 standard tables agree with the shared detailed diagnostics")
    analytic_initial_state(solver, workdir / "analytic", fixtures)
    restart_suite(solver, workdir, configuration(bdf2_fixture(interface)))


def restart_suite(solver, workdir, text):
    checkpoint = ("\nsubsection Checkpoint Restart\n  set enable checkpoint = true\n"
                  "  set checkpoint frequency = 1\nend\n")
    directory = workdir / "restart"
    for path in directory.glob("**/chns_*.txt"):
        path.rename(path.with_suffix(".previous_run"))
    first = text.replace("set t_end            = 0.008", "set t_end            = 0.004")
    # Monitor in memory, save tables in checkpoint, and enable output only later.
    run(solver, directory, first.replace("set write results = true", "set write results = false") + checkpoint)
    assert not list(directory.glob("**/chns_*.txt"))
    assert list(directory.glob("**/checkpoint.postprocessing"))
    (directory / "solver.log").rename(directory / "before_restart.log")
    resumed = checkpoint.replace("set enable checkpoint = true", "set enable checkpoint = false\n  set restart = true")
    run(solver, directory, text + resumed)
    compare_tables(table(directory, "chns_conservation"), table(workdir / "bdf2", "chns_conservation"))
    compare_tables(table(directory, "chns_energy"), table(workdir / "bdf2", "chns_energy"))
    print("Checkpoint restores original references and complete tables without relying on output files")

    directory = workdir / "old_checkpoint"
    run(solver, directory, first + checkpoint)
    path = next(directory.glob("**/checkpoint.postprocessing"))
    path.rename(path.with_suffix(".postprocessing.backup"))
    run(solver, directory, text + resumed)
    c = table(directory, "chns_conservation")
    assert len(c) == 3
    close(c[0]["time"], .004)
    assert all(abs(row["reference_time"]-.004) < 1e-14 for row in c)
    print("Old checkpoint without sidecar starts an explicitly dated monitoring reference")


def mpi_suite(solver, workdir, fixtures, mpiexec):
    text = configuration(bdf2_fixture((fixtures / "interface_quasi_incompressible.prm").read_text()))
    run(solver, workdir / "serial", text)
    run(solver, workdir / "parallel", text, (str(mpiexec), "-n", "2"))
    for name in ("chns_conservation", "chns_energy"):
        compare_tables(table(workdir / "parallel", name), table(workdir / "serial", name))
    print("Conservation and energy tables agree on one and two MPI ranks")
    checkpoint = ("\nsubsection Checkpoint Restart\n  set enable checkpoint = true\n"
                  "  set checkpoint frequency = 1\nend\n")
    directory = workdir / "restart_parallel"
    first = text.replace("set t_end            = 0.008", "set t_end            = 0.004")
    launch = (str(mpiexec), "-n", "2")
    run(solver, directory, first + checkpoint, launch)
    (directory / "solver.log").rename(directory / "before_restart.log")
    resumed = checkpoint.replace("set enable checkpoint = true", "set enable checkpoint = false\n  set restart = true")
    run(solver, directory, text + resumed, launch)
    for name in ("chns_conservation", "chns_energy"):
        compare_tables(table(directory, name), table(workdir / "parallel", name))
    print("Two-rank restart restores references and all table rows without duplicate times")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path)
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    if args.mpiexec:
        mpi_suite(args.solver.resolve(), args.workdir.resolve(), fixtures, args.mpiexec)
    else:
        serial_suite(args.solver.resolve(), args.workdir.resolve(), fixtures)


if __name__ == "__main__":
    main()
