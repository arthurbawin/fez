"""Verify and prepare quasi-incompressible interfacial profile correction cases.

The core and mms suites launch small verification cases. The physics suite
prepares a lambda sweep by default; running and summarizing are separate modes.
Only the Python standard library and the existing validation helpers are used.
"""

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from validate_quasi_incompressible import (
    bdf2_fixture,
    check_jacobian,
    check_rejected_configuration,
    check_space_convergence,
    diagnostics,
    run,
)


CORRECTION_COLUMNS = (
    "profile_correction_work",
    "profile_correction_flux_l2",
    "energy_balance_residual_with_correction",
    "temporal_energy_balance_residual_with_correction",
)
DEFAULT_LAMBDAS = "0,0.3,0.6,1.2,2.5,5"


def replace_parameter(text, name, value, required=True):
    """Replace an existing parameter without depending on column alignment."""
    pattern = rf"(?m)^(\s*set {re.escape(name)}\s*=\s*)[^\n#]*(.*)$"
    text, count = re.subn(pattern, lambda match: match[1] + str(value) + match[2], text)
    if required:
        assert count == 1, f"Expected one parameter {name!r}, found {count}"
    return text


def profile_fixture(text, strength=0.6, enabled=True, reference=None):
    """Insert correction parameters immediately after the selected model."""
    if reference is None:
        match = re.search(r"(?m)^\s*set profile correction reference mobility\s*=\s*([^#\n]+)", text)
        reference = match[1].strip() if match else "0"
    for name in ("enable profile correction", "profile correction strength",
                 "profile correction reference mobility"):
        text = re.sub(rf"(?m)^\s*set {re.escape(name)}\s*=[^\n]*\n", "", text)
    insertion = (
        f"\n  set enable profile correction = {str(enabled).lower()}"
        f"\n  set profile correction strength = {strength:.17g}"
        f"\n  set profile correction reference mobility = {reference}"
    )
    text, count = re.subn(
        r"(?m)^(\s*set CHNS model\s*=\s*quasi_incompressible\s*)(?=\n|$)",
        lambda match: match[1].rstrip() + insertion, text,
    )
    assert count == 1, "Expected one set CHNS model = quasi_incompressible"
    return text


def set_density(text, fluid, value):
    """Modify only the density in the requested Physical properties/Fluid block."""
    lines = text.splitlines(keepends=True)
    sections = []
    count = 0
    for index, line in enumerate(lines):
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("subsection "):
            sections.append(stripped[len("subsection "):].strip())
        elif stripped == "end":
            sections.pop()
        elif sections[-2:] == ["Physical properties", f"Fluid {fluid}"]:
            if re.match(r"set density\s*=", stripped):
                lines[index] = re.sub(r"(set density\s*=\s*)[^#\n]*",
                                      lambda match: match[1] + str(value), line)
                count += 1
    assert count == 1, f"Expected one density for fluid {fluid}"
    return "".join(lines)


def equal_density_fixture(text):
    sections = []
    density = None
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("subsection "):
            sections.append(stripped[len("subsection "):].strip())
        elif stripped == "end":
            sections.pop()
        elif sections[-2:] == ["Physical properties", "Fluid 0"]:
            match = re.match(r"set density\s*=\s*(.+)", stripped)
            if match:
                density = match[1]
    assert density is not None, "The base parameter file must specify fluid 0 density"
    return set_density(text, 1, density)


def close(a, b, tolerance=2e-13):
    return (math.isnan(a) and math.isnan(b)) or (
        math.isfinite(a) and math.isfinite(b)
        and abs(a - b) <= tolerance * max(1., abs(a), abs(b))
    )


def check_correction_columns(rows):
    assert rows, "Expected diagnostic rows"
    for row in rows:
        assert all(name in row for name in CORRECTION_COLUMNS)
        work = row["profile_correction_work"]
        norm = row["profile_correction_flux_l2"]
        assert math.isfinite(work) and math.isfinite(norm) and norm >= 0.
        assert close(row["energy_balance_residual_with_correction"],
                     row["energy_balance_residual"] + row["dt"] * work)
        assert close(row["temporal_energy_balance_residual_with_correction"],
                     row["temporal_energy_balance_residual"] + work)
        if row["effective_bdf_order"] == 0:
            assert work == 0., "Initial correction work must be zero"


def check_closed_case(directory, require_work=True):
    rows = diagnostics(directory)
    check_correction_columns(rows)
    initial = rows[0]
    assert len(rows) >= 3
    scale = max(1., abs(initial["total_energy"]))
    for row in rows:
        assert abs(row["phase_integral"] - initial["phase_integral"]) < 2e-9
        assert abs(row["mass_affine"] - initial["mass_affine"]) < 2e-8
        if row["effective_bdf_order"] == 2:
            # The temporal balance is a power; use dt to compare energy units.
            balance = row["dt"] * row["temporal_energy_balance_residual_with_correction"]
        else:
            balance = row["energy_balance_residual_with_correction"]
        assert math.isfinite(balance) and abs(balance) < 2e-9 * scale, \
            f"Corrected energy identity failed at t={row['time']}: {balance}"
    if require_work:
        assert any(abs(row["profile_correction_work"]) > 1e-14 for row in rows[1:]), \
            "The interface case must exercise nonzero correction work"
    print(f"{directory.name}: conserved phase/mass and corrected energy identity", flush=True)
    return rows


def check_zero_strength(reference_directory, zero_directory):
    """Compare the literal CSV values in every column that predates correction."""
    tables = []
    for directory in (reference_directory, zero_directory):
        paths = list(directory.glob("**/quasi_incompressible_diagnostics.csv"))
        assert len(paths) == 1
        with paths[0].open() as stream:
            tables.append(list(csv.DictReader(stream)))
    assert len(tables[0]) == len(tables[1])
    for reference, zero in zip(*tables):
        assert reference.keys() == zero.keys()
        for name in reference:
            if name not in CORRECTION_COLUMNS:
                assert reference[name] == zero[name], \
                    f"Zero-strength regression changed {name}: {reference[name]} != {zero[name]}"
    for directory in (reference_directory, zero_directory):
        rows = diagnostics(directory)
        check_correction_columns(rows)
        assert all(row["profile_correction_work"] == 0.
                   and row["profile_correction_flux_l2"] == 0. for row in rows)
    print("Disabled correction and lambda=0 have identical existing CSV columns", flush=True)


def core_suite(solver, workdir, fixtures):
    manufactured = (fixtures / "mms_quasi_incompressible_2d.prm").read_text()
    interface = (fixtures / "interface_quasi_incompressible.prm").read_text()
    for density in (20, 1000):
        for name, base in (("mms", manufactured), ("interface", interface)):
            text = profile_fixture(set_density(base, 0, density))
            text = replace_parameter(text, "enable_line_search", "true")
            text = replace_parameter(text, "compare jacobian matrix with finite differences", "true")
            directory = workdir / f"jacobian_{name}_rho{density}"
            error = check_jacobian(run(solver, directory, text))
            print(f"{directory.name}: maximum assembled Jacobian error {error:.3e}", flush=True)
            if name == "interface":
                check_closed_case(directory)

    for scheme, base in (("bdf1", interface), ("bdf2", bdf2_fixture(interface))):
        reference = workdir / f"{scheme}_disabled"
        zero = workdir / f"{scheme}_lambda0"
        run(solver, reference, profile_fixture(base, enabled=False))
        run(solver, zero, profile_fixture(base, strength=0.))
        check_zero_strength(reference, zero)
        if scheme == "bdf2":
            directory = workdir / "closed_bdf2"
            text = replace_parameter(profile_fixture(base), "enable_line_search", "true")
            run(solver, directory, text)
            check_closed_case(directory)

    # The phase space represents an affine inverse profile on this mesh.
    planar = profile_fixture(interface)
    planar = planar.replace("tanh(2*cos(pi*x))", "tanh((x-0.5)/(sqrt(2)*0.1))")
    planar = replace_parameter(planar, "t_end", "0.002")
    planar = replace_parameter(planar, "enable_line_search", "true")
    directory = workdir / "planar_initial"
    run(solver, directory, planar)
    rows = diagnostics(directory)
    check_correction_columns(rows)
    assert rows[0]["time"] == 0.
    assert rows[0]["profile_correction_flux_l2"] < 1e-13, \
        f"Planar reconstructed equilibrium has nonzero initial flux: {rows[0]}"
    print("Planar tanh profile: initial reconstructed correction flux vanishes", flush=True)

    corrected = profile_fixture(interface)
    invalid = (
        ("model", replace_parameter(corrected, "CHNS model", "abels"),
         "Profile correction is only implemented for CHNS model = quasi_incompressible."),
        ("reference", replace_parameter(corrected, "mobility model", "degenerate"),
         "Profile correction with mobility model = degenerate requires a positive profile correction reference mobility."),
        ("compensation", replace_parameter(corrected, "enable density mobility compensation", "true"),
         "Profile correction is not implemented with density mobility compensation."),
        ("thickness", replace_parameter(corrected, "interface thickness", "0"),
         "Profile correction requires positive interface thickness and surface tension."),
        ("surface_tension", replace_parameter(corrected, "surface tension", "0"),
         "Profile correction requires positive interface thickness and surface tension."),
    )
    for name, text, message in invalid:
        check_rejected_configuration(solver, workdir / f"invalid_{name}", text, message)
        print(f"Invalid correction configuration rejected: {name}", flush=True)


def mms_suite(solver, workdir, fixtures):
    base = (fixtures / "mms_quasi_incompressible_2d.prm").read_text()
    base = replace_parameter(base, "convergence steps", "3")
    # The eight-triangle mesh is pre-asymptotic for velocity at lambda=2.5.
    base = replace_parameter(base, "dealii mesh parameters",
                             "4, 4 : 0., 0. : 1., 1. : true")
    base = replace_parameter(base, "dt", "0.005")
    base = replace_parameter(base, "t_end", "0.01")
    base = replace_parameter(base, "compare jacobian matrix with finite differences", "false")
    base = replace_parameter(base, "enable_line_search", "true")
    # Remove time dependence to isolate the reconstruction's spatial error.
    base = base.replace("*(1+t)", "").replace("*cos(t)", "")
    for strength in (0.6, 2.5):
        directory = workdir / lambda_name(strength)
        orders = check_space_convergence(run(solver, directory, profile_fixture(base, strength)))
        print(f"lambda={strength:g}: spatial phase orders "
              + ", ".join(f"{order:.3f}" for order in orders), flush=True)


def lambda_name(strength):
    return "lambda_" + f"{strength:.12g}".replace(".", "p").replace("-", "m").replace("+", "p")


def parse_lambdas(text):
    values = [float(value.strip()) for value in text.split(",")]
    assert values and all(math.isfinite(value) and value >= 0. for value in values)
    assert len({lambda_name(value) for value in values}) == len(values), \
        "Lambda values must produce distinct case names"
    return values


def prepare_physics(args, workdir):
    base_path = args.base_prm.resolve()
    base = base_path.read_text()
    if args.equal_density:
        base = equal_density_fixture(base)
    base = replace_parameter(base, "write quasi incompressible diagnostics", "true", required=False)
    if not re.search(r"(?m)^\s*set write quasi incompressible diagnostics\s*=", base):
        base = re.sub(r"(?m)^(\s*set CHNS model\s*=\s*quasi_incompressible[^\n]*)$",
                      r"\1\n  set write quasi incompressible diagnostics = true", base)
    # Keep generated cases independent and resolve the usual external mesh input.
    base = replace_parameter(base, "output directory", "./results_profile_correction/")
    base = re.sub(r"(?m)^(\s*set mesh file\s*=\s*)([^#\n]+)",
                  lambda match: match[1] + str((base_path.parent / match[2].strip()).resolve()), base)
    workdir.mkdir(parents=True, exist_ok=True)
    cases = []
    for strength in parse_lambdas(args.lambdas):
        directory = workdir / lambda_name(strength)
        directory.mkdir(parents=True, exist_ok=True)
        parameter_file = directory / "input.prm"
        parameter_file.write_text(profile_fixture(base, strength, reference=args.reference_mobility))
        cases.append({"lambda": strength, "directory": str(directory),
                      "parameter_file": str(parameter_file),
                      "equal_density": int(args.equal_density), "base_prm": str(base_path)})
    with (workdir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)
    if args.solver:
        commands = []
        for case in cases:
            command = [*shlex.split(args.launch), str(args.solver.resolve()), case["parameter_file"]]
            commands.append(f"(cd {shlex.quote(case['directory'])} && {shlex.join(command)} > solver.log 2>&1)")
        (workdir / "commands.sh").write_text("#!/bin/sh\nset -eu\n" + "\n".join(commands) + "\n")
    print(f"Prepared {len(cases)} cases in {workdir / 'cases.csv'}", flush=True)
    return cases


def read_cases(workdir):
    with (workdir / "cases.csv").open() as stream:
        cases = list(csv.DictReader(stream))
    assert cases, "The physics manifest is empty"
    return cases


def accepted_step(row):
    return row["dt"] > 0. and row["effective_bdf_order"] > 0


def physics_rows(directory):
    """Merge restart histories, counting each accepted physical time once."""
    paths = sorted(directory.glob("**/quasi_incompressible_diagnostics.csv"),
                   key=lambda path: (path.stat().st_mtime_ns, str(path)))
    assert paths, f"No diagnostics found in {directory}"
    merged = {}
    for path in paths:
        with path.open() as stream:
            for raw in csv.DictReader(stream):
                row = {name: float(value) for name, value in raw.items()}
                assert math.isfinite(row["time"])
                previous = merged.get(row["time"])
                # Preserve an accepted step's dt and work against a restart
                # initialization snapshot at the same physical time. Among
                # accepted duplicates, retain the latest available record.
                if (previous is None or accepted_step(row)
                        or not accepted_step(previous)):
                    merged[row["time"]] = row
    rows = [merged[time] for time in sorted(merged)]
    check_correction_columns(rows)
    return rows


def finite_max(values):
    # Optional BDF energy diagnostics can be NaN; infinity remains visible.
    values = [value for value in values if not math.isnan(value)]
    return max(values) if values else math.nan


def summarize_physics(workdir):
    summaries = []
    for case in read_cases(workdir):
        rows = physics_rows(Path(case["directory"]))
        first, last = rows[0], rows[-1]
        accepted = [row for row in rows if accepted_step(row)]
        second_order = any(row["effective_bdf_order"] == 2 for row in accepted)
        balance_name = ("temporal_energy_balance_residual_with_correction" if second_order
                        else "energy_balance_residual_with_correction")
        summaries.append({
            "lambda": float(case["lambda"]),
            "equal_density": case["equal_density"],
            "initial_time": first["time"], "final_time": last["time"],
            "accepted_diagnostic_steps": len(accepted),
            "phase_drift": last["phase_integral"] - first["phase_integral"],
            "max_phase_drift": max(abs(row["phase_integral"] - first["phase_integral"]) for row in rows),
            "mass_drift": last["mass_affine"] - first["mass_affine"],
            "max_mass_drift": max(abs(row["mass_affine"] - first["mass_affine"]) for row in rows),
            "phi_min": min(row["phi_min_q"] for row in rows),
            "phi_max": max(row["phi_max_q"] for row in rows),
            "area_change": last["bubble_area"] - first["bubble_area"],
            "relative_area_change": ((last["bubble_area"] - first["bubble_area"]) / first["bubble_area"]
                                     if first["bubble_area"] != 0. else math.nan),
            "center_y": last["center_y"], "rise_velocity": last["rise_velocity"],
            "integrated_profile_correction_work": sum(row["dt"] * row["profile_correction_work"] for row in accepted),
            "integrated_diffusive_dissipation": sum(row["dt"] * row["diffusive_dissipation"] for row in accepted),
            "max_velocity_divergence_l2": max(row["velocity_divergence_l2"] for row in rows),
            "max_corrected_balance_residual": finite_max(abs(row[balance_name]) for row in accepted),
            "corrected_balance_units": "power" if second_order else "energy",
            "max_corrected_energy_balance_residual": finite_max(abs(row["energy_balance_residual_with_correction"]) for row in accepted),
            "max_corrected_temporal_energy_balance_residual": finite_max(abs(row["temporal_energy_balance_residual_with_correction"]) for row in accepted),
        })
    with (workdir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {workdir / 'summary.csv'}", flush=True)


def physics_suite(args, solver, workdir):
    if args.mode == "prepare":
        prepare_physics(args, workdir)
        return
    if args.mode == "summarize":
        summarize_physics(workdir)
        return
    cases = (prepare_physics(args, workdir) if args.base_prm else read_cases(workdir))
    # Physics jobs may outlast the shared small-case helper's 150-second limit.
    for case in cases:
        directory = Path(case["directory"])
        command = [*shlex.split(args.launch), str(solver), case["parameter_file"]]
        print(f"Running lambda={case['lambda']}: {shlex.join(command)}", flush=True)
        with (directory / "solver.log").open("w") as log:
            result = subprocess.run(command, cwd=directory, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
        assert result.returncode == 0, f"Solver failed; see {directory / 'solver.log'}"
    summarize_physics(workdir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "mms", "physics"), default="core")
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run", "summarize"), default="prepare",
                        help="Physics sweep operation (default: prepare)")
    parser.add_argument("--base-prm", type=Path, help="Quasi-incompressible physics case")
    parser.add_argument("--lambdas", default=DEFAULT_LAMBDAS)
    parser.add_argument("--equal-density", action="store_true")
    parser.add_argument("--reference-mobility", type=float,
                        help="Reference mobility for a parsed degenerate law")
    parser.add_argument("--launch", default="", help="Physics launcher, e.g. 'mpiexec -n 4'")
    parser.add_argument("--timeout", type=float,
                        help="Optional timeout in seconds per physics case")
    args = parser.parse_args()
    needs_solver = args.suite != "physics" or args.mode == "run"
    if needs_solver and args.solver is None:
        parser.error("--solver is required for core, mms, and physics --mode run")
    if args.suite == "physics" and args.mode == "prepare" and args.base_prm is None:
        parser.error("--base-prm is required for physics --mode prepare")
    if args.reference_mobility is not None and (
            not math.isfinite(args.reference_mobility) or args.reference_mobility <= 0.):
        parser.error("--reference-mobility must be finite and positive")
    if args.timeout is not None and args.timeout <= 0.:
        parser.error("--timeout must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    solver = args.solver.resolve() if args.solver else None
    workdir = args.workdir.resolve()
    fixtures = Path(__file__).resolve().parent
    if args.suite == "core":
        core_suite(solver, workdir, fixtures)
    elif args.suite == "mms":
        mms_suite(solver, workdir, fixtures)
    else:
        physics_suite(args, solver, workdir)


if __name__ == "__main__":
    main()
