#!/usr/bin/env python3
"""Generate and run controlled Stepien/BDF comparisons; standard library only."""

import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VARIANTS = {
    "baseline": ("constant", False, False),
    "article_mobility": ("degenerate_absolute", False, False),
    "conservative": ("degenerate_absolute", True, False),
    "conservative_skew": ("degenerate_absolute", True, True),
}


def replace(text, key, value):
    text, count = re.subn(
        rf"(?m)^(\s*set {re.escape(key)}\s*=).*$",
        lambda match: f"{match[1]} {value}", text)
    if count != 1:
        raise ValueError(f"Expected one entry for {key}, found {count}")
    return text


def summarize(directory, variant, completed, elapsed, target):
    filename = directory / "results" / "stepien_diagnostics.csv"
    rows = []
    if filename.exists():
        with filename.open() as stream:
            rows = [{key: float(value) for key, value in row.items()}
                    for row in csv.DictReader(stream)]
    result = {"variant": variant, "status": "failed", "elapsed_seconds": elapsed}
    if not rows:
        return result, rows
    first, last = rows[0], rows[-1]
    finite = all(math.isfinite(value) for row in rows for value in row.values())
    success = completed and finite and math.isclose(last["time"], target, abs_tol=1e-9)
    result.update(
        status="completed" if success else "incomplete",
        final_time=last["time"],
        center_y=last["center_y"], rise_velocity=last["rise_velocity"],
        weighted_center_y=last["weighted_center_y"],
        weighted_rise_velocity=last["weighted_rise_velocity"],
        phase_integral_drift=last["phase_integral"]-first["phase_integral"],
        max_phase_integral_drift=max(abs(row["phase_integral"]-first["phase_integral"])
                                     for row in rows),
        mass_affine_relative_drift=(last["mass_affine"]-first["mass_affine"])
                                   / first["mass_affine"],
        mass_used_relative_drift=(last["mass_used"]-first["mass_used"])
                                 / first["mass_used"],
        bubble_area_relative_drift=(last["bubble_area"]-first["bubble_area"])
                                   / first["bubble_area"],
        initial_energy=first["total_energy"], final_energy=last["total_energy"],
        energy_change=last["total_energy"]-first["total_energy"],
        energy_increase_steps=sum(
            later["total_energy"] > earlier["total_energy"] +
            1e-10 * max(1., abs(earlier["total_energy"]))
            for earlier, later in zip(rows, rows[1:])),
        phi_min_q=min(row["phi_min_q"] for row in rows),
        phi_max_q=max(row["phi_max_q"] for row in rows))
    return result, rows


def write_reports(output, summaries, histories, settings):
    (output / "summary.json").write_text(json.dumps(
        {"settings": settings, "results": summaries}, indent=2) + "\n")
    keys = list(dict.fromkeys(key for result in summaries for key in result))
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fields = [("center_y", "Centre de la bulle"),
              ("rise_velocity", "Vitesse de montée"),
              ("total_energy", "Énergie totale mesurée"),
              ("phase_integral", "Variation de l’intégrale de phase")]
    for axis, (field, title) in zip(axes.flat, fields):
        for variant, rows in histories.items():
            if not rows:
                continue
            values = [row[field] for row in rows]
            if field == "phase_integral":
                values = [value-values[0] for value in values]
            axis.plot([row["time"] for row in rows], values, label=variant)
        axis.set(title=title, xlabel="Temps")
        axis.grid(alpha=0.3)
    if any(histories.values()):
        axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Comparaison exploratoire BDF2 — h={settings['h']:.5g}, "
                 f"ε={settings['epsilon']:.5g}, Δt={settings['dt']:.5g}")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, default=ROOT / "buildRelease/incompressible_chns")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cells", type=int, default=32, help="cells across the unit width")
    parser.add_argument("--dt", type=positive, help="default 0.128 h")
    parser.add_argument("--epsilon", type=positive, help="default 0.64 h")
    parser.add_argument("--t-end", type=positive, default=3.)
    parser.add_argument("--baseline-mobility", type=positive, default=6.1e-6)
    parser.add_argument("--pressure-ref0", type=positive, default=1.)
    parser.add_argument("--pressure-ref1", type=positive, default=1.)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--timeout", type=positive, default=3600., help="seconds per variant")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.cells < 2 or args.ranks < 1:
        parser.error("--cells must be at least 2 and --ranks at least 1")
    h = 1. / args.cells
    dt, epsilon = args.dt or 0.128*h, args.epsilon or 0.64*h
    output = (args.output or ROOT / "tmp/incompressible_chns" /
              ("stepien_article_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    settings = {"h": h, "dt": dt, "epsilon": epsilon, "t_end": args.t_end,
                "cells": args.cells, "scheme": "BDF2", "ranks": args.ranks,
                "pressure_ref0": args.pressure_ref0, "pressure_ref1": args.pressure_ref1,
                "baseline_mobility": args.baseline_mobility,
                "solver": str(args.solver.resolve()),
                "note": "Exploratory comparison; not a benchmark validation or energy proof."}
    template = (HERE / "benchmark2.prm").read_text()
    jobs = []
    for variant in args.variants:
        mobility, conservative, skew = VARIANTS[variant]
        directory = output / variant
        (directory / "results").mkdir(parents=True)
        text = template
        changes = {
            "output directory": str(directory / "results") + "/",
            "dealii mesh parameters": f"{args.cells}, {2*args.cells} : 0., 0. : 1., 2. : true",
            "dt": format(dt, ".16g"), "t_end": format(args.t_end, ".16g"),
            "interface thickness": format(epsilon, ".16g"),
            "Function constants": f"eps={epsilon:.16g}, R0=0.25, xc=0.5, yc=0.5",
            "mobility": format(args.baseline_mobility, ".16g"),
            "mobility model": mobility,
            "stepien conservative transport": str(conservative).lower(),
            "stepien skew symmetric inertia": str(skew).lower(),
        }
        for key, value in changes.items():
            text = replace(text, key, value)
        references = iter((args.pressure_ref0, args.pressure_ref1))
        text = re.sub(r"(?m)^(\s*set pressure reference\s*=).*$",
                      lambda match: f"{match[1]} {next(references):.16g}", text)
        prm = directory / "simulation.prm"
        prm.write_text(text)
        jobs.append((variant, directory, prm))
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Prepared {len(jobs)} variants in {output}", flush=True)
    if args.generate_only:
        return 0
    solver = args.solver.resolve()
    if not solver.is_file():
        parser.error(f"Solver does not exist: {solver}")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               DEAL_II_NUM_THREADS="1", MKL_NUM_THREADS="1")
    summaries, histories = [], {}
    for variant, directory, prm in jobs:
        command = [str(solver), str(prm)]
        if args.ranks > 1:
            command = ["mpirun", "-n", str(args.ranks)] + command
        print(f"Running {variant}...", flush=True)
        start = time.monotonic()
        with (directory / "solver.log").open("w") as log:
            try:
                result = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=args.timeout)
                completed = result.returncode == 0
            except subprocess.TimeoutExpired:
                completed = False
                log.write("\nComparison runner: time limit reached.\n")
        summary, rows = summarize(directory, variant, completed,
                                  time.monotonic()-start, args.t_end)
        summaries.append(summary)
        histories[variant] = rows
        write_reports(output, summaries, histories, settings)
        print(f"{variant}: {summary['status']}; {summary['elapsed_seconds']:.1f} s", flush=True)
    return 0 if all(result["status"] == "completed" for result in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
