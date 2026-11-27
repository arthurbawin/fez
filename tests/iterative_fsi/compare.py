"""Run the same FE/BDF problem with MUMPS and FGMRES; never invent baselines.

The default LU PC isolates the iterative plumbing. --pc bjacobi measures the
unvalidated ILU baseline separately. Run from CTest or see --help.
"""
import argparse
import math
from pathlib import Path
import re
import subprocess


def parameter_text(source, case, method):
    folder = source / "tests" / "monolithic_fsi"
    text = (folder / (case + ".prm")).read_text(encoding="utf-8-sig")
    # Historical tests run two directories below their copied meshes.
    text = text.replace("../../mms_meshes/", (folder / "mms_meshes").as_posix() + "/")
    text = re.sub(r"(set\s+mesh file\s*=\s*)\.\./\.\./([^\s#]+)",
                  lambda m: m[1] + (folder / m[2]).as_posix(), text)
    text = re.sub(r"(set\s+output directory\s*=).*", r"\1 results/", text)
    text = re.sub(r"(set\s+enable timer\s*=).*", r"\1 true", text)
    # Only the linear method changes. Effective linear tolerances are passed
    # through PETSc, leaving all FE, quadrature, BDF and Newton parameters intact.
    text, count = re.subn(r"(set\s+method\s*=\s*)direct_mumps", r"\g<1>" + method, text)
    if count != 1:
        raise ValueError("Expected exactly one main-physics direct_mumps setting")
    return text


def observables(output):
    values = []
    for line in output.splitlines():
        if re.match(r"^(F[xyz]|[xyz])\s*=", line) or "Rigid-body rotation angle:" in line:
            values.extend(float(x) for x in re.findall(
                r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", line))
        # MMS table rows, including both mesh/time refinements and all fields.
        elif re.match(r"^\s*\d+\s+\d+\s+\d+\.\d+\s", line):
            values.extend(float(x) for x in line.split() if x != "-")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("solver", "reference", "source", "work", "case", "mpiexec"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--pc", choices=("lu", "bjacobi"), default="lu")
    args = parser.parse_args()
    source, work = Path(args.source).resolve(), Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = {}
    for method, executable in (("direct_mumps", args.reference), ("fgmres", args.solver)):
        directory = work / method
        directory.mkdir(exist_ok=True)
        prm = directory / "case.prm"
        prm.write_text(parameter_text(source, args.case, method), encoding="utf-8")
        command = [args.mpiexec, "-n", str(args.ranks), str(Path(executable).resolve()),
                   str(prm), "-log_view"]
        if method == "fgmres":
            command += ["-fsi_pc_type", args.pc, "-fsi_ksp_rtol", "1e-11",
                        "-fsi_ksp_atol", "1e-13", "-fsi_ksp_max_it", "1000",
                        "-fsi_compare_mumps", "-fsi_comparison_tolerance", "1e-7",
                        "-fsi_mechanical_diagnostics"]
            if args.pc == "lu":
                command += ["-fsi_pc_factor_mat_solver_type", "mumps"]
            else:
                command += ["-fsi_sub_pc_type", "ilu",
                            "-fsi_sub_pc_factor_shift_type", "nonzero"]
            if args.case == "jacobian_matrix":
                command += ["-fsi_check_jacobian"]
        result = subprocess.run(command, cwd=directory, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                encoding="utf-8", errors="replace", timeout=550)
        (directory / "run.log").write_text(result.stdout, encoding="utf-8")
        (directory / "command.txt").write_text(repr(command), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{method} failed ({result.returncode}); see {directory / 'run.log'}")
        logs[method] = result.stdout
    if "FSI FGMRES iterations=" not in logs["fgmres"]:
        raise AssertionError("No FGMRES solve was exercised")
    if "FSI MUMPS correction_difference=" not in logs["fgmres"]:
        raise AssertionError("No direct/iterative correction comparison was exercised")
    if args.case == "jacobian_matrix" and "FSI global Jv error=" not in logs["fgmres"]:
        raise AssertionError("No global Jacobian check was exercised")
    before, after = (observables(logs[m]) for m in ("direct_mumps", "fgmres"))
    if len(before) != len(after):
        raise AssertionError("Different numbers of physical/MMS observations")
    if not before and args.case != "jacobian_matrix":
        raise AssertionError("No physical/MMS observations to compare")
    for a, b in zip(before, after):
        if not math.isfinite(b) or not math.isclose(a, b, rel_tol=2e-6, abs_tol=2e-8):
            raise AssertionError(f"Physical/MMS mismatch: {a} versus {b}")
    print(f"Compared {len(before)} observations; correction/residual checks passed in the solver.")


if __name__ == "__main__":
    main()
