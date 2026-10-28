"""Verify the quasi-incompressible CHNS discretization on small cases.

Does not launch rising-bubble or phase-separation simulations.
Only Python's standard library is required. Logs and diagnostics are retained.
Use --suite physics for pure hydrostatic states and interface relaxation,
--suite core for BDF1 Jacobians/MMS and the original mass/energy cases,
--suite bdf2 for the extension, or --suite all for all three groups.
--mpiexec retains the separate two-rank suite and covers both schemes.
"""

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


def run(solver, directory, text, launch=()):
    directory.mkdir(parents=True, exist_ok=True)
    parameter_file = directory / "input.prm"
    parameter_file.write_text(text)
    result = subprocess.run(
        [*launch, str(solver), str(parameter_file)], cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=150,
    )
    (directory / "solver.log").write_text(result.stdout)
    if result.returncode:
        raise AssertionError(f"Solver failed in {directory}:\n{result.stdout[-5000:]}")
    return result.stdout


def check_rejected_configuration(solver, directory, text, message):
    directory.mkdir(parents=True, exist_ok=True)
    parameter_file = directory / "input.prm"
    parameter_file.write_text(text)
    result = subprocess.run(
        [str(solver), str(parameter_file)], cwd=directory,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30,
    )
    (directory / "solver.log").write_text(result.stdout)
    assert result.returncode != 0, f"Unsupported configuration accepted: {directory.name}"
    # deal.II wraps exception text to fit its banner; parameter names can span
    # several lines even though the diagnostic itself remains unchanged.
    assert " ".join(message.split()) in " ".join(result.stdout.split()), \
        f"Wrong rejection for {directory.name}: expected {message!r}\n{result.stdout[-3000:]}"


def configuration_suite(solver, workdir, fixtures):
    energy = (fixtures / "energy_quasi_incompressible.prm").read_text()
    periodic = (fixtures / "mms_quasi_incompressible_periodic.prm").read_text()
    pure = (fixtures / "pure_quasi_incompressible.prm").read_text()
    cases = (
        ("stationary", energy.replace("set scheme           = BDF1",
                                      "set scheme           = stationary"),
         "requires scheme = BDF1 or BDF2"),
        ("bdf2_missing_startup", bdf2_fixture(energy).replace(
            "set bdf start method = BDF1", "set bdf start method = initial condition"),
         "requires bdf start method = BDF1"),
        ("bdf2_zero_start", bdf2_fixture(energy, ratio=0),
         "requires 0 < bdf start step ratio < 1"),
        ("bdf2_zero_second_step", bdf2_fixture(energy, ratio=1),
         "requires 0 < bdf start step ratio < 1"),
        ("missing_pressure_gauge", energy.replace(
            "set enforce zero mean pressure = true",
            "set enforce zero mean pressure = false"),
         "requires exactly one pressure gauge"),
        ("both_pressure_gauges", energy.replace(
            "set fix pressure constant      = false",
            "set fix pressure constant      = true"),
         "Both fixing a pressure DoF"),
        ("open_fluid_boundary", energy.replace("set type = no_slip",
                                               "set type = outflow", 1),
         "supports no_slip/slip fluid boundaries"),
        ("prescribed_phase_boundary", energy.replace("set type = no_flux",
                                                     "set type = input_function", 1),
         "supports natural no_flux Cahn-Hilliard boundaries"),
        ("periodic_body_force", periodic.replace("set body force = 0, 0",
                                                 "set body force = 0, -1"),
         "periodic box requires zero body force"),
        ("negative_mobility_prefactor", pure.replace(
            "set degenerate absolute mobility prefactor = 0.1",
            "set degenerate absolute mobility prefactor = -0.1"),
         "degenerate absolute mobility prefactor"),
    )
    for name, text, message in cases:
        check_rejected_configuration(solver, workdir / f"invalid_{name}", text, message)
        print(f"Unsupported configuration rejected: {name}")


def check_jacobian(log):
    errors = [float(x) for x in re.findall(
        r"Max absolute error analytical vs fd matrix is ([\deE+.-]+)", log
    )]
    assert len(errors) >= 2, "Expected assembled matrix checks at both time steps"
    assert max(errors) < 2e-5, f"Assembled Jacobian error: {max(errors)}"
    assert "Entry (" not in log, "A Jacobian block exceeds the prescribed tolerances"
    return max(errors)


def diagnostics(directory):
    paths = list(directory.glob("**/quasi_incompressible_diagnostics.csv"))
    assert len(paths) == 1, f"Expected one diagnostics file in {directory}"
    with paths[0].open() as stream:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]


def check_energy(directory, require_decay=True):
    rows = diagnostics(directory)
    if any(row["effective_bdf_order"] == 2 for row in rows):
        return check_bdf2_energy(directory, require_decay)
    assert len(rows) == 5, "Expected initial state and four accepted steps"
    initial = rows[0]
    for row in rows:
        for key in ("phase_integral", "mass_affine", "total_energy",
                    "energy_balance_residual", "energy_inequality_residual",
                    "numerical_kinetic_energy_loss",
                    "numerical_interfacial_energy_loss"):
            assert math.isfinite(row[key]), f"Non-finite diagnostic {key}"
        assert abs(row["phase_integral"] - initial["phase_integral"]) < 2e-9
        assert abs(row["mass_affine"] - initial["mass_affine"]) < 2e-8
        scale = max(1., abs(initial["total_energy"]))
        assert row["energy_inequality_residual"] < 2e-9 * scale
        assert abs(row["energy_balance_residual"]) < 2e-9 * scale
        assert row["viscous_dissipation"] >= -1e-14
        assert row["diffusive_dissipation"] >= -1e-14
        assert row["numerical_kinetic_energy_loss"] >= -1e-14
        assert row["numerical_interfacial_energy_loss"] >= -1e-14
        assert abs(row["numerical_energy_loss"] -
                   row["numerical_kinetic_energy_loss"] -
                   row["numerical_interfacial_energy_loss"]) < 2e-12 * scale
    if require_decay:
        assert rows[-1]["total_energy"] < initial["total_energy"] - 1e-7, \
            "The verification case must dissipate a measurable amount of energy"
    return max(abs(row["energy_balance_residual"]) for row in rows)


def bdf2_fixture(text, ratio=0.5):
    text, count = re.subn(r"(set scheme\s*=\s*)BDF1", r"\g<1>BDF2", text)
    assert count == 1, "Expected one time integration scheme"
    if re.search(r"set bdf start step ratio\s*=", text):
        return re.sub(r"(set bdf start step ratio\s*=\s*)[^\n]+",
                      lambda m: m[1] + str(ratio), text)
    return text.replace("set bdf start method = BDF1",
                        "set bdf start method = BDF1\n"
                        f"  set bdf start step ratio = {ratio}")


def check_bdf2_coefficients(rows, first_dt=None):
    """Check actual history lengths, including startup and changed restart dt."""
    previous_dt = first_dt
    for row in rows:
        order = row["effective_bdf_order"]
        if order == 0:
            continue
        dt = row["dt"]
        if order == 1:
            expected = (1 / dt, -1 / dt, 0.)
        else:
            assert previous_dt is not None
            expected = (1 / dt + 1 / (dt + previous_dt),
                        -1 / dt - 1 / previous_dt,
                        dt / (previous_dt * (dt + previous_dt)))
        for i, value in enumerate(expected):
            assert abs(row[f"bdf_c{i}"] - value) < 2e-12 * max(1., abs(value)), \
                f"Wrong BDF coefficient {i} at t={row['time']}: {row}"
        previous_dt = dt


def check_bdf2_energy(directory, require_decay=True):
    """Check the tested equations, without assuming total-energy monotonicity."""
    rows = diagnostics(directory)
    assert len(rows) >= 4
    assert rows[0]["effective_bdf_order"] == 0
    assert rows[1]["effective_bdf_order"] == 1
    assert all(row["effective_bdf_order"] == 2 for row in rows[2:])
    check_bdf2_coefficients(rows)
    initial = rows[0]
    scale = max(1., abs(initial["total_energy"]))
    for row in rows:
        assert abs(row["phase_integral"] - initial["phase_integral"]) < 2e-9
        assert abs(row["mass_affine"] - initial["mass_affine"]) < 2e-8
        for key in ("total_energy", "physical_energy_increment_plus_dissipation",
                    "temporal_energy_balance_residual"):
            assert math.isfinite(row[key]), f"Non-finite {key}"
        assert row["viscous_dissipation"] >= -1e-14
        assert row["diffusive_dissipation"] >= -1e-14
        assert abs(row["temporal_energy_balance_residual"]) < 2e-8 * scale
        assert abs(row["phase_time_derivative_integral"]) < 2e-8
        assert abs(row["mass_time_derivative_integral"]) < 2e-6
        if row["effective_bdf_order"] == 2:
            for key in ("energy_balance_residual", "energy_inequality_residual",
                        "numerical_energy_loss", "numerical_kinetic_energy_loss",
                        "numerical_interfacial_energy_loss"):
                assert math.isnan(row[key]), f"BDF1-only diagnostic {key} used in BDF2"
            for key in ("bdf2_modified_kinetic_energy", "bdf2_modified_gradient_energy"):
                assert row[key] >= 0. and math.isfinite(row[key])
        if row["bdf2_constant_step"]:
            for part in ("kinetic", "gradient"):
                assert row[f"bdf2_{part}_remainder"] >= 0.
                assert abs(row[f"bdf2_{part}_identity_residual"]) < 2e-12 * scale
        else:
            assert math.isnan(row["bdf2_kinetic_identity_residual"])
    if require_decay:
        # Observed behavior of these fixtures, not a general BDF2 theorem.
        assert rows[-1]["total_energy"] < initial["total_energy"] - 1e-7
    return max(abs(row["temporal_energy_balance_residual"]) for row in rows)


def check_pure_state(directory, phase, density):
    """A P1 pressure can balance constant gravity exactly for either pure fluid."""
    balance = check_energy(directory, require_decay=False)
    rows = diagnostics(directory)
    initial_energy = rows[0]["total_energy"]
    for row in rows:
        for key in ("phi_min_q", "phi_max_q"):
            assert abs(row[key] - phase) < 2e-10, \
                f"Pure phase changed: {key}={row[key]} instead of {phase}"
        for key in ("density_affine_min_q", "density_affine_max_q"):
            assert abs(row[key] - density) < 2e-7
        assert row["velocity_l2"] < 2e-9, "Hydrostatic state develops velocity"
        assert row["velocity_divergence_l2"] < 2e-8, \
            "Pure hydrostatic state develops divergence"
        assert row["diffusive_flux_l2"] < 1e-10, \
            "Pressure gradient causes diffusion in a pure phase"
        assert row["diffusive_dissipation"] < 5e-8
        assert row["negative_density_volume_q"] == 0.
        assert row["density_extension_l1"] < 2e-8
        assert abs(row["total_energy"] - initial_energy) < \
            2e-9 * max(1., abs(initial_energy))
    return balance


def check_interface_relaxation(directory):
    """Exercise a nonuniform, unforced interface and its actual CH dissipation."""
    balance = check_energy(directory)
    rows = diagnostics(directory)
    assert rows[0]["phi_min_q"] < -0.5 and rows[0]["phi_max_q"] > 0.5, \
        "Relaxation fixture must contain both phases and a diffuse interface"
    diffusive_loss = sum(row["dt"] * row["diffusive_dissipation"]
                         for row in rows[1:])
    assert diffusive_loss > 1e-9, \
        "Relaxation must exercise a nonzero Cahn-Hilliard energy loss"
    assert max(row["diffusive_flux_l2"] for row in rows[1:]) > 1e-7, \
        "Relaxation must exercise the assembled diffusion flux"
    assert rows[-1]["interfacial_energy"] < rows[0]["interfacial_energy"] - 1e-8
    for row in rows:
        assert row["density_affine_min_q"] > 0., \
            "This moderate-contrast relaxation fixture must retain positive density"
        assert row["negative_density_volume_q"] == 0.
        assert row["density_extension_l1"] < 1e-10
    return balance, diffusive_loss


def check_density_extension(directory):
    """Intentionally invalid phase values must remain visible in diagnostics/VTU."""
    balance = check_energy(directory, require_decay=False)
    rows = diagnostics(directory)
    phase = -1.02
    affine_density = 0.5 * (1001. + 999. * phase)
    volume = 1.  # The verification mesh is the unit square.
    for row in rows:
        assert abs(row["phi_min_q"] - phase) < 2e-10
        assert abs(row["phi_max_q"] - phase) < 2e-10
        for key in ("density_affine_min_q", "density_affine_max_q"):
            assert abs(row[key] - affine_density) < 2e-7
        assert abs(row["mass_affine"] - affine_density * volume) < 2e-8
        assert abs(row["mass_positive"] - volume) < 2e-10
        assert abs(row["negative_density_volume_q"] - volume) < 2e-12
        assert abs(row["phase_overshoot_volume_q"] - volume) < 2e-12
        assert abs(row["density_extension_l1"] -
                   (1. - affine_density) * volume) < 2e-8
        assert row["velocity_l2"] < 2e-9
        assert row["velocity_divergence_l2"] < 2e-8
        assert row["diffusive_flux_l2"] < 1e-10
        assert row["numerical_kinetic_energy_loss"] < 1e-12
        assert row["numerical_interfacial_energy_loss"] < 1e-12
        assert abs(row["total_energy"] - rows[0]["total_energy"]) < 2e-9

    vtu_files = list(directory.glob("**/*.vtu"))
    assert vtu_files, "Density-extension regression must exercise VTU output"
    for path in vtu_files:
        # Names are in the XML header even when numerical arrays are compressed.
        header = path.read_text(errors="ignore")
        names = set(re.findall(r'<DataArray\b[^>]*\bName=[\"\']([^\"\']+)', header))
        assert {"density", "density_affine"} <= names, \
            f"Missing positive/affine density output fields in {path}"
    return balance


def physics_suite(solver, workdir, fixtures):
    pure = (fixtures / "pure_quasi_incompressible.prm").read_text()
    for inverse in (False, True):
        text = pure
        densities = (1000., 1.)
        if inverse:
            text = text.replace("set density             = 1000", "set density             = SWAP")
            text = text.replace("set density             = 1\n", "set density             = 1000\n")
            text = text.replace("set density             = SWAP", "set density             = 1")
            densities = (1., 1000.)
        for phase in (1, -1):
            name = f"pure_{'inverse_' if inverse else ''}{'plus' if phase > 0 else 'minus'}"
            directory = workdir / name
            case = text.replace("set Function expression = 1 # pure phase",
                                f"set Function expression = {phase} # pure phase")
            run(solver, directory, case)
            error = check_pure_state(directory, phase, densities[phase < 0])
            print(f"{name}: invariant hydrostatic phase, zero flux/divergence; "
                  f"balance residual {error:.3e}")

    # This deliberately nonphysical homogeneous state checks that the positive
    # inertial extension never hides a negative affine density in the output.
    extension = pure.replace("set Function expression = 1 # pure phase",
                             "set Function expression = -1.02 # pure phase")
    extension = extension.replace("set body force = 0, -1", "set body force = 0, 0")
    extension = extension.replace("set write vtu results    = false",
                                  "set write vtu results    = true")
    directory = workdir / "density_extension"
    run(solver, directory, extension)
    error = check_density_extension(directory)
    print(f"density_extension: negative affine density remains visible in CSV/VTU; "
          f"balance residual {error:.3e}")

    interface = (fixtures / "interface_quasi_incompressible.prm").read_text()
    for name, text in (("interface_absolute", interface),
                       ("interface_quadratic", quadratic_mobility_fixture(interface))):
        directory = workdir / name
        run(solver, directory, text)
        error, loss = check_interface_relaxation(directory)
        print(f"{name}: phase/mass/energy checks passed; "
              f"CH energy loss {loss:.3e}, balance residual {error:.3e}")


def quadratic_mobility_fixture(text):
    # Same peak mobility as 0.1*epsilon^2*abs(1-phi^2) for epsilon=0.1.
    # The parsed expression already includes the amplitude; 'mobility' applies
    # only to the constant law and must not be used as a prefactor here.
    return text.replace("set mobility model      = degenerate_absolute",
                        "set mobility model      = degenerate\n"
                        "  subsection degenerate mobility\n"
                        "    set Function expression = 0.001*(1-x*x)*(1-x*x)\n"
                        "  end")


def time_convergence_fixture(text):
    text = text.replace("set type                   = space",
                        "set type                   = time")
    text = text.replace("set convergence steps      = 1",
                        "set convergence steps      = 3")
    text = text.replace("set dt               = 0.01", "set dt               = 0.02")
    text = text.replace("set t_end            = 0.02", "set t_end            = 0.2")
    text = text.replace("set spatial mesh index   = 1", "set spatial mesh index   = 0")
    text = text.replace("set body force = 0.2, -0.3", "set body force = 0, 0")
    text = text.replace("set type = dirichlet_mms", "set type = no_flux")
    text = text.replace("set compare jacobian matrix with finite differences = true",
                        "set compare jacobian matrix with finite differences = false")
    for name, expression in (("velocity", "0; 0"), ("pressure", "0"),
                             ("cahn hilliard tracer", "0.2+0.1*sin(t)"),
                             ("cahn hilliard potential", "0.1*cos(t)")):
        pattern = rf"(subsection exact {name}\s+set Function expression = )[^\n]+"
        text, count = re.subn(pattern, lambda m: m[1] + expression, text)
        assert count == 1, f"Could not set manufactured {name}"
    return text


def check_time_convergence(log):
    block = log.rsplit("\nL2_norm\n", 1)[1].split("\n\n", 1)[0]
    rows = [line.split() for line in block.splitlines() if re.match(r"\s*\d", line)]
    assert len(rows) == 3, "Expected three time refinement levels"
    errors = [float(row[7]) for row in rows]
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(0.85 < order < 1.2 for order in orders), f"Time orders: {orders}"
    mu_errors = [float(row[9]) for row in rows]
    mu_orders = [math.log2(mu_errors[i] / mu_errors[i + 1]) for i in range(2)]
    assert all(0.85 < order < 1.2 for order in mu_orders), \
        f"Potential time orders: {mu_orders}"
    assert all(abs(float(row[i])) < 1e-10 for row in rows for i in (3, 5)), \
        "Uniform manufactured state must retain zero velocity and pressure"
    return orders


def check_space_convergence(log):
    block = log.rsplit("\nL2_norm\n", 1)[1].split("\n\n", 1)[0]
    rows = [line.split() for line in block.splitlines() if re.match(r"\s*\d", line)]
    assert len(rows) == 3, "Expected three spatial refinement levels"
    errors = [float(row[7]) for row in rows]
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(order > 1.5 for order in orders), f"Spatial phase orders: {orders}"
    for name, column in (("velocity", 3), ("pressure", 5), ("potential", 9)):
        errors = [float(row[column]) for row in rows]
        field_orders = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
        assert field_orders[0] > 1. and field_orders[1] > 1.5, \
            f"Spatial {name} orders: {field_orders}"
    return orders


def final_time_errors(directory):
    paths = list(directory.glob("**/time_errors.txt"))
    assert len(paths) == 1
    lines = paths[0].read_text().splitlines()
    header = next(line.split() for line in lines if line.split()[:1] == ["t"])
    rows = [line.split() for line in lines if re.match(r"\s*[+]?[0-9]", line)]
    assert rows
    return dict(zip(header, map(float, rows[-1])))


def bdf2_time_convergence(solver, workdir, fixtures):
    base = (fixtures / "mms_quasi_incompressible_bdf2.prm").read_text()
    for name, rho0, rho1 in (("equal", 1, 1), ("large", 1000, 1),
                             ("inverse", 1, 1000)):
        text = base.replace("set density             = 20", f"set density             = {rho0}")
        text = text.replace("subsection Fluid 1\n    set density             = 1",
                            f"subsection Fluid 1\n    set density             = {rho1}")
        # The pressure test is not zero at the boundary: its natural flux
        # must vanish, including for unequal densities. Keep grad(mu+alpha*p)=0.
        text = text.replace("(19/21)*0.1*(x+y-1)*cos(2*t)",
                            f"({rho0-rho1}/{rho0+rho1})*0.1*(x+y-1)*cos(2*t)")
        errors = []
        for index, dt in enumerate((0.02, 0.01, 0.005)):
            directory = workdir / f"time_{name}_{index}"
            case = text.replace("set dt               = 0.02", f"set dt               = {dt}")
            run(solver, directory, case)
            rows = diagnostics(directory)
            assert rows[1]["effective_bdf_order"] == 1
            check_bdf2_coefficients(rows)
            errors.append(final_time_errors(directory))
        for field in ("u", "p", "phi", "mu"):
            orders = [math.log2(errors[i][field] / errors[i + 1][field])
                      for i in range(2)]
            assert all(1.7 < order < 2.3 for order in orders), \
                f"BDF2 {name} {field} orders: {orders}; errors: {errors}"
            print(f"BDF2 {name} {field} final-time orders: " +
                  ", ".join(f"{order:.3f}" for order in orders))


def bdf2_restart(solver, workdir, text):
    reference = workdir / "restart_reference"
    run(solver, reference, text)
    checkpoint = ("\nsubsection Checkpoint Restart\n"
                  "  set enable checkpoint = true\n"
                  "  set checkpoint frequency = 1\n"
                  "end\n")
    directory = workdir / "restart_same_dt"
    first = text.replace("set t_end            = 0.008", "set t_end            = 0.004")
    run(solver, directory, first + checkpoint)
    (directory / "solver.log").rename(directory / "before_restart.log")
    resumed = text + checkpoint.replace("set enable checkpoint = true",
                                        "set enable checkpoint = false\n  set restart = true")
    log = run(solver, directory, resumed)
    assert "Reading checkpoint" in log
    actual, expected = diagnostics(directory), diagnostics(reference)
    assert len(actual) == len(expected), "Duplicated checkpoint diagnostic row"
    for a, b in zip(actual, expected):
        for key, value in b.items():
            if math.isnan(value):
                assert math.isnan(a[key])
            else:
                assert abs(a[key] - value) < 2e-9 * max(1., abs(value)), \
                    f"Restart mismatch in {key}: {a[key]} vs {value}"
    print("BDF2 restart: uninterrupted and resumed diagnostics agree")

    directory = workdir / "restart_changed_dt"
    run(solver, directory, first + checkpoint)
    (directory / "solver.log").rename(directory / "before_restart.log")
    run(solver, directory, resumed.replace("set dt               = 0.002",
                                           "set dt               = 0.001"))
    rows = diagnostics(directory)
    check_bdf2_coefficients(rows)
    assert any(abs(row["dt"] - 0.001) < 1e-14 and row["time"] > 0.004
               for row in rows)
    check_bdf2_energy(directory)
    print("BDF2 restart with changed dt: accepted step history and balances verified")


def bdf2_suite(solver, workdir, fixtures):
    for case in ("2d", "3d", "periodic"):
        text = bdf2_fixture((fixtures / f"mms_quasi_incompressible_{case}.prm").read_text())
        directory = workdir / f"jacobian_{case}"
        error = check_jacobian(run(solver, directory, text))
        check_bdf2_coefficients(diagnostics(directory))
        print(f"BDF2 {case} assembled Jacobian error: {error:.3e}")

    energy = (fixtures / "energy_quasi_incompressible.prm").read_text()
    for name, text in (("equal", energy.replace("set density             = 20", "set density             = 1")),
                       ("unequal", energy),
                       ("large", energy.replace("set density             = 20", "set density             = 1000"))):
        directory = workdir / f"energy_{name}"
        # Exercise unequal actual steps at startup, then constant steps.
        run(solver, directory, bdf2_fixture(text, ratio=0.1))
        error = check_bdf2_energy(directory)
        print(f"BDF2 {name}: mass and tested energy balance residual {error:.3e}")

    pure = (fixtures / "pure_quasi_incompressible.prm").read_text()
    for rho0, rho1 in ((1000, 1), (1, 1000)):
        text = pure.replace("set density             = 1000", "set density             = SWAP")
        text = text.replace("set density             = 1\n", f"set density             = {rho1}\n")
        text = text.replace("set density             = SWAP", f"set density             = {rho0}")
        for phase in (1, -1):
            directory = workdir / f"pure_{rho0}_{rho1}_{phase}"
            case = text.replace("set Function expression = 1 # pure phase",
                                f"set Function expression = {phase} # pure phase")
            run(solver, directory, bdf2_fixture(case))
            check_pure_state(directory, phase, rho0 if phase == 1 else rho1)
    print("BDF2: four pure hydrostatic states preserve zero flux and velocity")

    interface = (fixtures / "interface_quasi_incompressible.prm").read_text()
    for name, text in (("absolute", interface), ("quadratic", quadratic_mobility_fixture(interface))):
        directory = workdir / f"interface_{name}"
        run(solver, directory, bdf2_fixture(text))
        error, loss = check_interface_relaxation(directory)
        print(f"BDF2 interface {name}: CH loss {loss:.3e}; balance {error:.3e}")
    bdf2_restart(solver, workdir, bdf2_fixture(interface))
    bdf2_time_convergence(solver, workdir, fixtures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "core", "physics", "bdf2"), default="all",
                        help="Serial verification group (default: all)")
    parser.add_argument("--mpiexec", type=Path,
                        help="Run only MPI partition/periodicity checks on two ranks")
    args = parser.parse_args()
    solver = args.solver.resolve()
    workdir = args.workdir.resolve()
    fixtures = Path(__file__).resolve().parent
    if args.mpiexec:
        launch = (str(args.mpiexec), "-n", "2")
        text = (fixtures / "mms_quasi_incompressible_periodic.prm").read_text()
        text = text.replace("set compare jacobian matrix with finite differences = true",
                            "set compare jacobian matrix with finite differences = false")
        run(solver, workdir / "periodic_serial", text)
        run(solver, workdir / "periodic_mpi", text, launch)
        serial = diagnostics(workdir / "periodic_serial")
        parallel = diagnostics(workdir / "periodic_mpi")
        assert len(serial) == len(parallel)
        for a, b in zip(serial, parallel):
            for key in ("phase_integral", "mass_affine", "mass_positive",
                        "total_energy", "viscous_dissipation", "diffusive_dissipation"):
                assert abs(a[key] - b[key]) < 2e-8 * max(1., abs(a[key])), \
                    f"Periodic MPI mismatch for {key}: {a[key]} vs {b[key]}"
        print("Periodic manufactured case: serial and two-rank diagnostics agree")
        text = (fixtures / "energy_quasi_incompressible.prm").read_text()
        run(solver, workdir / "energy_mpi", text, launch)
        error = check_energy(workdir / "energy_mpi")
        print(f"Two-rank mass/energy checks passed; balance residual {error:.3e}")
        text = (fixtures / "interface_quasi_incompressible.prm").read_text()
        run(solver, workdir / "interface_mpi", text, launch)
        error, loss = check_interface_relaxation(workdir / "interface_mpi")
        print(f"Two-rank interface relaxation passed; CH energy loss {loss:.3e}, "
              f"balance residual {error:.3e}")
        text = bdf2_fixture((fixtures / "mms_quasi_incompressible_periodic.prm").read_text())
        text = text.replace("set compare jacobian matrix with finite differences = true",
                            "set compare jacobian matrix with finite differences = false")
        run(solver, workdir / "bdf2_periodic_serial", text)
        run(solver, workdir / "bdf2_periodic_mpi", text, launch)
        serial = diagnostics(workdir / "bdf2_periodic_serial")
        parallel = diagnostics(workdir / "bdf2_periodic_mpi")
        assert len(serial) == len(parallel)
        for a, b in zip(serial, parallel):
            for key in ("phase_integral", "mass_affine", "total_energy",
                        "kinetic_temporal_power", "bulk_temporal_power",
                        "gradient_temporal_power", "temporal_energy_balance_residual"):
                assert abs(a[key] - b[key]) < 2e-8 * max(1., abs(a[key])), \
                    f"BDF2 periodic MPI mismatch for {key}: {a[key]} vs {b[key]}"
        text = bdf2_fixture((fixtures / "interface_quasi_incompressible.prm").read_text())
        run(solver, workdir / "bdf2_interface_mpi", text, launch)
        error, loss = check_interface_relaxation(workdir / "bdf2_interface_mpi")
        print(f"BDF2 serial/MPI periodic agreement and two-rank interface balance: {error:.3e}")
        return

    if args.suite in ("all", "physics"):
        physics_suite(solver, workdir, fixtures)
    if args.suite == "physics":
        return
    if args.suite in ("all", "bdf2"):
        bdf2_suite(solver, workdir / "bdf2" if args.suite == "all" else workdir, fixtures)
    if args.suite == "bdf2":
        return

    for case in ("2d", "3d", "periodic"):
        text = (fixtures / f"mms_quasi_incompressible_{case}.prm").read_text()
        log = run(solver, workdir / case, text)
        error = check_jacobian(log)
        print(f"{case}: assembled Jacobian maximum absolute error {error:.3e}")

    energy = (fixtures / "energy_quasi_incompressible.prm").read_text()
    for name, text in (("energy", energy),
                       ("energy_equal_density", energy.replace(
                           "set density             = 20", "set density             = 1")),
                       ("energy_large_ratio", energy.replace(
                           "set density             = 20", "set density             = 1000"))):
        directory = workdir / name
        run(solver, directory, text)
        error = check_energy(directory)
        print(f"{name}: mass and energy checks passed; balance residual {error:.3e}")

    text = time_convergence_fixture(
        (fixtures / "mms_quasi_incompressible_2d.prm").read_text())
    orders = check_time_convergence(run(solver, workdir / "time_convergence", text))
    print("Manufactured phase time convergence orders: " +
          ", ".join(f"{order:.3f}" for order in orders))

    text = (fixtures / "mms_quasi_incompressible_2d.prm").read_text()
    text = text.replace("set convergence steps      = 1",
                        "set convergence steps      = 3")
    text = text.replace("set dt               = 0.01", "set dt               = 0.005")
    text = text.replace("set t_end            = 0.02", "set t_end            = 0.01")
    text = text.replace("*(1+t)", "").replace("*cos(t)", "")
    text = text.replace("set compare jacobian matrix with finite differences = true",
                        "set compare jacobian matrix with finite differences = false")
    orders = check_space_convergence(run(solver, workdir / "space_convergence", text))
    print("Manufactured phase spatial convergence orders: " +
          ", ".join(f"{order:.3f}" for order in orders))

    configuration_suite(solver, workdir, fixtures)


if __name__ == "__main__":
    main()
