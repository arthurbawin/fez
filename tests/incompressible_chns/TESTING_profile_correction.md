# Quasi-incompressible profile correction verification

Run these commands from the repository root. Build the implementation before
running the verification programs. No reference
`.output` is supplied for the new unit test.

## Build and unit test reference

The existing Release build uses Ninja. Build the solver and the standalone unit
test template without running either executable:

```sh
ninja -C buildRelease -j
ninja -C buildRelease -j unit_tests.quasi_incompressible_profile_correction
```

The unit test checks the regularized inverse and its derivative, the equilibrium
flux, derivatives in the interface and activation tails, divergence on an
analytic field, reconstructed affine-distance profiles on a P1 triangle,
elementary Jacobians including nodal overshoots, the shared pressure/phase flux,
the local mass cancellation, and the reference mobility amplitudes. Assertions
must pass before its log can become a reference. Its executable writes `output`
in its working directory through the existing test logging helper.

The following commands **execute** the unit test and create its reference; run
them only when ready to perform validation:

```sh
mkdir -p buildRelease/profile_correction_unit_reference
(
  cd buildRelease/profile_correction_unit_reference
  ../tests/unit_tests/unit_tests.quasi_incompressible_profile_correction
)
cat buildRelease/profile_correction_unit_reference/output
cp buildRelease/profile_correction_unit_reference/output tests/unit_tests/quasi_incompressible_profile_correction.output
cmake -S . -B buildRelease
ctest --test-dir buildRelease -R 'unit_tests/quasi_incompressible_profile_correction' --output-on-failure
```

Review the assertion-free log before the `cp`. Reconfiguration lets
`deal_ii_pickup_tests()` discover the new reference and register its comparison
test. The separate executable target exists so the template can be compiled
before this reference is available.

## Core suite

Run the suite directly:

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite core \
  --solver "$PWD/buildRelease/incompressible_chns" \
  --workdir "$PWD/buildRelease/profile_correction_core"
```

Or use its CTest registration:

```sh
ctest --test-dir buildRelease \
  -R '^incompressible_chns/quasi_profile_correction$' --output-on-failure
```

The suite derives every input from the existing fixtures, inserting correction
parameters immediately after `set CHNS model = quasi_incompressible`. It checks:

- Assembled analytical versus finite-difference Jacobians on the two-dimensional
  manufactured fixture and the closed interface fixture, at density ratios
  20 and 1000. It reuses the existing maximum absolute matrix error threshold
  `2e-5` and the per-entry diagnostic checks.
- Disabled correction versus enabled correction with `lambda = 0`, for both
  first- and second-order backward differentiation. Every pre-existing CSV
  field is compared as literal output text. Work and correction flux norms must
  be zero; the corrected balances must equal their original balances.
- Phase and affine mass conservation on the closed interface, with absolute
  tolerances `2e-9` and `2e-8`. The corrected energy identity must be below
  `2e-9 * max(1, |initial total energy|)`, and correction work must be nonzero
  after initialization. The second-order case is also checked. Its temporal
  power residual is multiplied by `dt` only for this energy-unit assertion.
- A resting planar initial profile
  `tanh((x-0.5)/(sqrt(2)*0.1))`. Its initial correction flux norm must be below
  `1e-13`, and initial correction work must be zero. Only its initial profile
  is an equilibrium check; the finite closed-domain boundary conditions affect
  subsequent evolution.
- Rejection of a different CHNS model, parsed degenerate mobility without a
  positive reference, density mobility compensation, zero interface thickness,
  and zero surface tension. The exact required exception messages are matched.

No monotone total-energy assertion is made: profile correction work has no
prescribed sign. The checks use the corrected identity instead.

The four appended CSV columns are:

```text
profile_correction_work
profile_correction_flux_l2
energy_balance_residual_with_correction
temporal_energy_balance_residual_with_correction
```

The energy balance correction is `dt * profile_correction_work`. The temporal
balance is a power and its correction is `profile_correction_work`, without an
extra `dt`. The original energy balance remains NaN on second-order steps, as
before; the temporal balance is the applicable check there.

## Manufactured spatial convergence

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite mms \
  --solver "$PWD/buildRelease/incompressible_chns" \
  --workdir "$PWD/buildRelease/profile_correction_mms"
```

This runs three spatial refinements at `lambda = 0.6` and `lambda = 2.5`, using
the existing two-dimensional manufactured fixture and convergence helper.
Time dependence is removed from the manufactured fields to isolate spatial
error. The phase orders must exceed 1.5. Velocity, pressure, and potential must
have an order above 1 on the first pair and above 1.5 on the second pair.
The source uses the exact continuous phase; reconstruction is part of the
discrete approximation and contributes spatial error.

## Physics sweep and cluster preparation

Preparation creates one directory and `input.prm` per strength, together with
`cases.csv`. If `--solver` is provided, it also writes `commands.sh`, containing
one independent subshell command per case. Preparation does not launch a solver.

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite physics --mode prepare \
  --base-prm examples/incompressible_chns/quasi_incompressible/rising_bubble_case1.prm \
  --solver "$PWD/buildRelease/incompressible_chns" \
  --workdir "$PWD/buildRelease/profile_correction_physics" \
  --lambdas 0,0.3,0.6,1.2,2.5,5
```

The default strengths are already `0,0.3,0.6,1.2,2.5,5`; omit `--lambdas` to use
them. Output is placed in each case's `results_profile_correction` directory.
A configured `mesh file` path is resolved relative to the base parameter file.
Other external inputs and restart paths should already be absolute. Preserve
the prepared parameter files and manifest when submitting independent jobs.

For comparison with the equal-density implementation, prepare another sweep:

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite physics --mode prepare --equal-density \
  --base-prm examples/incompressible_chns/quasi_incompressible/rising_bubble_case1.prm \
  --solver "$PWD/buildRelease/incompressible_chns" \
  --workdir "$PWD/buildRelease/profile_correction_equal_density"
```

`--equal-density` copies fluid 0's density into fluid 1. It preserves the other
fluid properties. For a parsed degenerate mobility, supply the strictly positive
`--reference-mobility VALUE` or put `profile correction reference mobility` in the
base file. Constant and named quadratic mobilities use `mobility` as their
reference. The absolute degenerate law uses its prefactor times `epsilon^2`.
Density mobility compensation must be disabled in the base case.

Run all prepared cases serially, retaining each solver log:

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite physics --mode run \
  --solver "$PWD/buildRelease/incompressible_chns" \
  --workdir "$PWD/buildRelease/profile_correction_physics"
```

To use a launcher, append for example `--launch 'mpiexec -n 4'`. The physics
runner has no timeout by default; `--timeout SECONDS` sets one per case. Unlike
the verification suites, it does not use the shared helper's small-case timeout.
Passing `--base-prm` in `run` mode prepares the manifest before launching it.
For separate cluster submissions, use the generated input files and commands
instead of the sequential runner. `run` also summarizes after all cases finish.

Summarize completed cluster cases without launching a solver:

```sh
python3 tests/incompressible_chns/validate_quasi_profile_correction.py \
  --suite physics --mode summarize \
  --workdir "$PWD/buildRelease/profile_correction_physics"
```

`summary.csv` records final and maximum absolute phase/mass drifts, minimum and
maximum sampled phase, absolute and relative area change, final `center_y` and
`rise_velocity`, sums of `dt * profile_correction_work` and
`dt * diffusive_dissipation`, maximum velocity divergence norm, and corrected
balance maxima. It contains separate energy and temporal-power maxima.
`max_corrected_balance_residual` selects the temporal balance for cases containing
second-order steps, otherwise the energy balance; `corrected_balance_units`
identifies the choice. The second-order NaNs in the original energy column
therefore do not hide a failed temporal identity.

Initial rows with zero `dt` are excluded from work/dissipation sums. Restart
histories are merged by physical time. An accepted step takes precedence over
an initialization snapshot at the same time, preserving its `dt * W`; among
accepted duplicates, the latest record is used. Steps are counted once.
Store diagnostics for every
accepted step to obtain complete sums. If only a restarted segment is available,
drifts are measured from its first available row and sums cover that segment.
The summary is an experimental comparison, not an assertion that every strength
improves the interface or preserves monotone energy.
