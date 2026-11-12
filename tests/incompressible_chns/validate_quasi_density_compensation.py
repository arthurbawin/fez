"""Verify capped density compensation and compare short rising-bubble runs.

The default core suite runs small Jacobian, conservation and energy checks.
Physical bubble runs require --suite bubbles or --suite all. Each invocation
keeps inputs, logs, diagnostics and a summary in its own run directory.
Only the Python standard library is required.
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
import tempfile
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from validate_quasi_incompressible import bdf2_fixture, check_jacobian, diagnostics
from validate_quasi_profile_correction import check_closed_case

FIXTURES = Path(__file__).resolve().parent
EXAMPLES = FIXTURES.parents[1] / "examples/incompressible_chns/quasi_incompressible"
ENABLE = "enable quasi incompressible density mobility compensation"
MAXIMUM = "quasi incompressible density mobility compensation maximum"


def entry(text, path, value):
    """Replace or insert one parameter using its complete subsection path."""
    lines, stack, output = text.splitlines(), [], []
    found = False
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped.startswith("subsection "):
            stack.append(stripped[len("subsection "):].strip())
        elif stripped == "end":
            if tuple(stack) == path[:-1] and not found:
                output.append("  " * len(stack) + f"set {path[-1]} = {value}")
                found = True
            stack.pop()
        elif tuple(stack) == path[:-1] and re.match(
                r"set\s+" + re.escape(path[-1]) + r"\s*=", stripped):
            if found:
                raise ValueError(f"Duplicate parameter: {path}")
            line = "  " * len(stack) + f"set {path[-1]} = {value}"
            found = True
        output.append(line)
    if not found:
        raise ValueError(f"Missing subsection for {path}")
    return "\n".join(output) + "\n"


def configure(text, maximum=None, bdf2=False, correction=False):
    if bdf2:
        text = bdf2_fixture(text)
    text = entry(text, ("Cahn Hilliard", "enable profile correction"), str(correction).lower())
    text = entry(text, ("Cahn Hilliard", ENABLE), str(maximum is not None).lower())
    if maximum is not None:
        text = entry(text, ("Cahn Hilliard", MAXIMUM), maximum)
    text = entry(text, ("Nonlinear solver", "enable_line_search"), "true")
    return text


def execute(solver, folder, text, timeout, prepare_only=False):
    folder.mkdir()
    parameters = folder / "input.prm"
    parameters.write_text(text)
    if prepare_only:
        return "", 0.
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = time.monotonic()
    with (folder / "solver.log").open("w") as log:
        try:
            result = subprocess.run([str(solver), str(parameters)], cwd=folder,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Timeout; inspect {folder / 'solver.log'}")
    output = (folder / "solver.log").read_text()
    if result.returncode:
        raise RuntimeError(f"Solver failed in {folder}:\n{output[-4000:]}")
    return output, time.monotonic() - started


def metrics(folder):
    rows = diagnostics(folder)
    first, last = rows[0], rows[-1]
    if len(rows) < 2:
        raise AssertionError("No accepted time step")
    for row in rows:
        for key in ("mass_affine", "phase_integral", "total_energy",
                    "phi_min_q", "phi_max_q", "temporal_energy_balance_residual"):
            if not math.isfinite(row[key]):
                raise AssertionError(f"Non-finite {key} in {folder}")
    accepted = rows[1:]
    return {
        "final_time": last["time"], "accepted_steps": len(accepted),
        "mass_relative_drift": max(abs(r["mass_affine"] - first["mass_affine"])
                                   / abs(first["mass_affine"]) for r in rows),
        "phase_volume_drift": max(abs(r["phase_integral"] - first["phase_integral"])
                                  / first["domain_volume"] for r in rows),
        "energy_balance_max": max(abs(r["temporal_energy_balance_residual"])
                                  for r in accepted),
        "bdf1_energy_identity_max": max(
            (abs(r["energy_balance_residual"]) for r in accepted
             if r["effective_bdf_order"] == 1), default=0.),
        "energy_increase_max": max(b["total_energy"] - a["total_energy"]
                                   for a, b in zip(rows, rows[1:])),
        "phi_min": min(r["phi_min_q"] for r in rows),
        "phi_max": max(r["phi_max_q"] for r in rows),
        "density_affine_min": min(r["density_affine_min_q"] for r in rows),
        "negative_density_volume_max": max(r["negative_density_volume_q"] for r in rows),
        "bubble_area_final": last["bubble_area"],
        "bubble_area_relative_change": (last["bubble_area"] - first["bubble_area"])
                                      / first["bubble_area"] if first["bubble_area"] else None,
        "center_y_final": last["center_y"],
        "rise_velocity_final": last["rise_velocity"],
        "divergence_l2_max": max(r["velocity_divergence_l2"] for r in accepted),
        "diffusive_dissipation_final": last["diffusive_dissipation"],
    }


def core(args, report):
    energy = (FIXTURES / "energy_quasi_incompressible.prm").read_text()
    for bdf2 in (False, True):
        order = 2 if bdf2 else 1
        if args.baseline_solver:
            original = bdf2_fixture(energy) if bdf2 else energy
            old = args.workdir / f"regression_bdf{order}_before"
            new = args.workdir / f"regression_bdf{order}_disabled"
            execute(args.baseline_solver, old, original, args.timeout)
            execute(args.solver, new, original, args.timeout)
            a = next(old.rglob("quasi_incompressible_diagnostics.csv")).read_bytes()
            b = next(new.rglob("quasi_incompressible_diagnostics.csv")).read_bytes()
            assert a == b, "Disabled-option regression changed numerical diagnostics"
            report[f"regression_bdf{order}"] = "diagnostics byte identical"
        manufactured = (FIXTURES / "mms_quasi_incompressible_2d.prm").read_text()
        # Cap=2 cuts through the MMS phase range, exercising both branches.
        manufactured = manufactured.replace(
            "0.2 + 0.1*cos(pi*x)*cos(pi*y)*cos(t)",
            "-0.3 + 0.1*cos(pi*x)*cos(pi*y)*cos(t)")
        for maximum in (2, 1000):
            name = f"jacobian_bdf{order}_cap{maximum}"
            folder = args.workdir / name
            log, duration = execute(args.solver, folder,
                configure(manufactured, maximum, bdf2), args.timeout)
            report[name] = {"matrix_absolute_error": check_jacobian(log),
                            "wall_seconds": duration}
            print(name, report[name], flush=True)
        interface = (FIXTURES / "interface_quasi_incompressible.prm").read_text()
        for light_density in (100, 1):
            for maximum in (100, 1000):
                name = f"closed_bdf{order}_1000_{light_density}_cap{maximum}"
                text = configure(interface, maximum, bdf2)
                text = entry(text, ("Physical properties", "Fluid 0", "density"), 1000)
                text = entry(text, ("Physical properties", "Fluid 1", "density"), light_density)
                folder = args.workdir / name
                _, duration = execute(args.solver, folder, text, args.timeout)
                result = metrics(folder)
                assert result["mass_relative_drift"] < 2e-11, (name, result)
                assert result["phase_volume_drift"] < 2e-11, (name, result)
                assert result["energy_balance_max"] < 2e-7, (name, result)
                if not bdf2:
                    assert result["energy_increase_max"] < 2e-9, (name, result)
                result["wall_seconds"] = duration
                report[name] = result
                print(name, result, flush=True)
                save(args.workdir, report)

        # Exercise the reconstructed correction together with the compensated
        # mobility, using both cap branches in the assembled Jacobian checks.
        for maximum in (2, 1000):
            name = f"jacobian_corrected_bdf{order}_cap{maximum}"
            folder = args.workdir / name
            log, duration = execute(args.solver, folder,
                configure(manufactured, maximum, bdf2, correction=True), args.timeout)
            report[name] = {"matrix_absolute_error": check_jacobian(log),
                            "wall_seconds": duration}
            print(name, report[name], flush=True)
        for light_density in (100, 1):
            name = f"closed_corrected_bdf{order}_1000_{light_density}_cap1000"
            text = configure(interface, 1000, bdf2, correction=True)
            text = entry(text, ("Physical properties", "Fluid 0", "density"), 1000)
            text = entry(text, ("Physical properties", "Fluid 1", "density"), light_density)
            folder = args.workdir / name
            _, duration = execute(args.solver, folder, text, args.timeout)
            rows = check_closed_case(folder)
            report[name] = {"final_time": rows[-1]["time"],
                            "wall_seconds": duration,
                            "corrected_energy_identity": "passed"}
            save(args.workdir, report)


def bubbles(args, report):
    for case in (1, 2):
        original = (EXAMPLES / f"rising_bubble_case{case}.prm").read_text()
        for maximum in (None, 100, 1000):
            name = f"bubble_case{case}_" + ("baseline" if maximum is None else f"cap{maximum}")
            text = configure(original, maximum)
            for path, value in (
                (("Output", "output directory"), str(args.workdir / name / "results") + "/"),
                (("Output", "write vtu results"), "true"),
                (("Mesh", "dealii mesh parameters"),
                 f"{args.mesh}, {2 * args.mesh} : 0., 0. : 1., 2. : true"),
                (("Time integration", "dt"), args.dt),
                (("Time integration", "t_end"), args.t_end),
                (("Nonlinear solver", "verbosity"), "verbose"),
            ):
                text = entry(text, path, value)
            folder = args.workdir / name
            log, duration = execute(args.solver, folder, text, args.timeout, args.prepare_only)
            if args.prepare_only:
                report[name] = {"parameters": str(folder / "input.prm")}
            else:
                result = metrics(folder)
                assert abs(result["final_time"] - args.t_end) < 1e-10, (name, result)
                assert result["mass_relative_drift"] < 2e-10, (name, result)
                assert result["phase_volume_drift"] < 2e-10, (name, result)
                assert result["bdf1_energy_identity_max"] < 2e-7, (name, result)
                assert result["energy_increase_max"] < 2e-8, (name, result)
                result["wall_seconds"] = duration
                result["newton_residual_evaluations"] = len(re.findall(r"Newton iter\.", log))
                report[name] = result
            print(name, report[name], flush=True)
            save(args.workdir, report)


def save(root, report):
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    bubble_results = [(k, v) for k, v in report.items()
                      if k.startswith("bubble_") and "final_time" in v]
    if bubble_results:
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", *bubble_results[0][1]])
            writer.writeheader()
            for name, result in bubble_results:
                writer.writerow({"case": name, **result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--baseline-solver", type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--suite", choices=("core", "bubbles", "all"), default="core")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mesh", type=int, default=32, help="Bubble subdivisions along x")
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--t-end", type=float, default=0.04)
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per individual run")
    args = parser.parse_args()
    if args.prepare_only and args.suite != "bubbles":
        parser.error("--prepare-only requires --suite bubbles")
    if args.mesh < 2 or not (args.dt > 0 and args.t_end > 0 and args.timeout > 0):
        parser.error("mesh >= 2 and positive dt, t-end and timeout are required")
    args.solver, args.workdir = args.solver.resolve(), args.workdir.resolve()
    if args.baseline_solver:
        args.baseline_solver = args.baseline_solver.resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.workdir = Path(tempfile.mkdtemp(prefix="run_", dir=args.workdir))
    print(f"Results: {args.workdir}", flush=True)
    report = {}
    if args.suite in ("core", "all"):
        core(args, report)
    if args.suite in ("bubbles", "all"):
        bubbles(args, report)
    save(args.workdir, report)


if __name__ == "__main__":
    main()
