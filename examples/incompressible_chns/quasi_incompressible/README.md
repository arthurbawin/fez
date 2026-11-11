# Quasi-incompressible Cahn-Hilliard Navier-Stokes

This directory documents the `quasi_incompressible` model of the
`incompressible_chns` solver: a two-phase Cahn-Hilliard Navier-Stokes model
with mass-averaged velocity, discretized with the fully discrete,
unconditionally energy-stable Galerkin scheme of

> A. Brunk and M. F. P. ten Eikelder, *A simple, fully-discrete,
> unconditionally energy-stable method for the two-phase Navier-Stokes
> Cahn-Hilliard model with arbitrary density ratios*, J. Comput. Phys. 548
> (2026), 114558.

It is selected with

```text
subsection Cahn Hilliard
  set CHNS model = quasi_incompressible
end
```

## Model

### Notation and mixture laws

Fluid 0 corresponds to $\phi = 1$ and fluid 1 to $\phi = -1$. With the
densities $\rho_0, \rho_1$ and the dynamic viscosities
$\eta_i = \rho_i \nu_i$ (FEZ takes kinematic viscosities $\nu_i$ as input),

$$
\rho(\phi) = \frac{1+\phi}{2}\rho_0 + \frac{1-\phi}{2}\rho_1,
\qquad
\alpha = \frac{\rho_1-\rho_0}{\rho_0+\rho_1},
\qquad
\tilde\sigma = \frac{3\sigma}{2\sqrt 2}.
$$

The mass density $\rho(\phi)$ is affine and is used for the mass, the
convective mass flux and gravity. The inertia and the viscosity use positive
extensions, i.e. the same laws evaluated at the clipped phase:

$$
\tilde\rho(\phi) = \rho(\max(-1, \min(1, \phi))),
\qquad
\tilde\eta(\phi) = \eta(\max(-1, \min(1, \phi))).
$$

The phase unknown itself is never clipped. The mobility $M(\phi) \ge 0$ is

| `mobility model` | $M(\phi)$ |
|---|---|
| `constant` | `mobility` |
| `degenerate_absolute` | $m_0\,\varepsilon^2\,\lvert 1-\phi^2 \rvert$, with $m_0$ = `degenerate absolute mobility prefactor` |

### Equations

The unknowns are the velocity $\mathbf v$, the pressure $p$, the phase
$\phi$ and the chemical potential $\mu$. The pressure is the shifted pressure
$p = p^* + p_r - \phi\mu$, so that the capillary force reads
$\phi\nabla\mu$. With $\mathbf J = M(\phi)(\nabla\mu + \alpha\nabla p)$:

$$
\begin{aligned}
&\nabla\cdot\mathbf v - \alpha\,\nabla\cdot\mathbf J = 0,\\
&\tilde\rho\,\partial_t\mathbf v + \tfrac12(\partial_t\tilde\rho)\,\mathbf v
 + \rho\,(\mathbf v\cdot\nabla)\mathbf v
 + \tfrac12\,[\nabla\cdot(\rho\mathbf v)]\,\mathbf v
 - \nabla\cdot\Big[\tilde\eta\big(\nabla\mathbf v + \nabla\mathbf v^T
 - \tfrac{2}{d}(\nabla\cdot\mathbf v)\mathbf I\big)\Big]
 + \nabla p + \phi\nabla\mu = \rho\,\mathbf g,\\
&\partial_t\phi + \nabla\cdot(\phi\mathbf v) - \nabla\cdot\mathbf J = 0,\\
&\mu - \frac{\tilde\sigma}{\varepsilon}(\phi^3-\phi)
 + \tilde\sigma\varepsilon\,\Delta\phi = 0.
\end{aligned}
$$

Multiplying the phase equation by $(\rho_0-\rho_1)/2$ and the continuity
equation by $(\rho_0+\rho_1)/2$ and adding them, the diffusive fluxes cancel
and the mass of the affine density is conserved:
$\partial_t\rho + \nabla\cdot(\rho\mathbf v) = 0$. For $-1 \le \phi \le 1$,
$\tilde\rho = \rho$ and the momentum equation is equivalent to its
conservative form; outside $[-1, 1]$, the extended inertia is the
regularization used by the scheme.

### Boundary conditions

The domain must be closed:

- fluid boundaries are `no_slip` ($\mathbf v = 0$) or `slip`
  ($\mathbf v\cdot\mathbf n = 0$, zero tangential traction);
- Cahn-Hilliard boundaries are `no_flux`, which imposes naturally
  $\nabla\phi\cdot\mathbf n = 0$ and $\mathbf J\cdot\mathbf n = 0$;
- the pressure is defined up to a constant, fixed by exactly one of
  `enforce zero mean pressure` or `fix pressure constant`.

Manufactured solutions may use `velocity_mms` and `dirichlet_mms`. With
`quasi incompressible periodic box = true`, the opposite faces $2k$ and
$2k+1$ of a colorized `rectangle` are identified for every field; this
requires quadrilaterals, no boundary condition, zero body force, and a
pressure pinned at an interior `pressure reference point`.

## Discretization

### Weak form

For test functions $(\mathbf z, q, \psi, \chi)$, with $\mathbf z$ vanishing
on `no_slip` walls and $\mathbf z\cdot\mathbf n = 0$ on `slip` walls:

$$
\begin{aligned}
&-(q, \nabla\cdot\mathbf v) - \alpha\,(\mathbf J, \nabla q) = 0,\\
&(\tilde\rho\,\partial_t\mathbf v, \mathbf z)
 + \tfrac12(\partial_t\tilde\rho\,\mathbf v, \mathbf z)
 + \tfrac12(\rho\,(\mathbf v\cdot\nabla)\mathbf v, \mathbf z)
 - \tfrac12(\rho\,(\mathbf v\cdot\nabla)\mathbf z, \mathbf v)
 + (\tilde\eta\,(\nabla\mathbf v + \nabla\mathbf v^T), \nabla\mathbf z)
 - \tfrac2d(\tilde\eta\,\nabla\cdot\mathbf v, \nabla\cdot\mathbf z)\\
&\qquad - (p, \nabla\cdot\mathbf z) + (\phi\nabla\mu, \mathbf z)
 - (\rho\,\mathbf g, \mathbf z) = 0,\\
&(\partial_t\phi, \psi) - (\phi\mathbf v, \nabla\psi)
 + (\mathbf J, \nabla\psi) = 0,\\
&(\mu, \chi) - \frac{\tilde\sigma}{\varepsilon}(\phi^3-\phi, \chi)
 - \tilde\sigma\varepsilon\,(\nabla\phi, \nabla\chi) = 0.
\end{aligned}
$$

The two convective integrals cancel for $\mathbf z = \mathbf v$.

### Time discretization (BDF1)

Given $(\mathbf v^n, \phi^n)$, the four fields at $t^{n+1}$ are computed
simultaneously, with $\Delta t_n = t^{n+1} - t^n$. Every coefficient without a
superscript is evaluated at $t^{n+1}$, except:

- the first inertia term uses the previous density:
  $\tilde\rho(\phi^n)\,(\mathbf v^{n+1}-\mathbf v^n)/\Delta t_n
  + \big(\tilde\rho(\phi^{n+1})-\tilde\rho(\phi^n)\big)/(2\Delta t_n)\,\mathbf v^{n+1}$;
- the double-well potential uses the exact divided difference of
  $F(s) = (s^2-1)^2/4$:

$$
\frac{\tilde\sigma}{\varepsilon}\,
\frac{(\phi^{n+1}+\phi^n)\big((\phi^{n+1})^2+(\phi^n)^2-2\big)}{4},
\qquad
\frac{(\phi^{n+1}+\phi^n)\big((\phi^{n+1})^2+(\phi^n)^2-2\big)}{4}\,
(\phi^{n+1}-\phi^n) = F(\phi^{n+1}) - F(\phi^n),
$$

which reduces to $\tilde\sigma(\phi^3-\phi)/\varepsilon$ when
$\phi^{n+1} = \phi^n$.

### Space discretization

The velocity and pressure use a Taylor-Hood pair (e.g. $P_2/P_1$ or
$Q_2/Q_1$) and the phase and potential use continuous elements of degree
$\ge 1$. The energy identity below requires the same space for the phase and
the potential; with different degrees (e.g. a $P_2$ phase and a $P_1$
potential) the scheme still runs, but the identity no longer holds. The fully
coupled nonlinear system is solved with Newton's

method and the exact Jacobian; the fields at $t^n$ are fixed during the
solve. No stabilization is used.

### Discrete energy identity

Testing with $\mathbf z = \mathbf v^{n+1}$, $q = p^{n+1}$,
$\psi = \mu^{n+1}$ and $\chi = \phi^{n+1}-\phi^n$, and treating gravity with
$q = \psi = \mathbf g\cdot\mathbf x$ (affine, hence in the discrete spaces up
to a constant for $q$), gives

$$
E^{n+1} - E^n
+ \Delta t_n\int_\Omega M\,\lvert\nabla\mu + \alpha\nabla p\rvert^2
+ \frac{\Delta t_n}{2}\int_\Omega \tilde\eta\,
  \Big\lvert\nabla\mathbf v + \nabla\mathbf v^T
  - \tfrac2d(\nabla\cdot\mathbf v)\mathbf I\Big\rvert^2
+ \frac12\int_\Omega \tilde\rho(\phi^n)\,\lvert\mathbf v^{n+1}-\mathbf v^n\rvert^2
+ \frac{\tilde\sigma\varepsilon}{2}\int_\Omega
  \lvert\nabla\phi^{n+1}-\nabla\phi^n\rvert^2 = 0,
$$

with the total energy

$$
E = \int_\Omega \frac{\tilde\rho(\phi)}{2}\lvert\mathbf v\rvert^2
+ \frac{\tilde\sigma}{\varepsilon}F(\phi)
+ \frac{\tilde\sigma\varepsilon}{2}\lvert\nabla\phi\rvert^2
- \rho(\phi)\,\mathbf g\cdot\mathbf x .
$$

The two integrals with $\Delta t_n$ are the physical (diffusive and viscous)
dissipations; the last two are the numerical dissipation of the scheme. The
energy is therefore non-increasing for any time step.

### BDF2 variant

With `scheme = BDF2` (and `bdf start method = BDF1`), the inertia becomes
$\sqrt{\tilde\rho}\,D_t^{BDF2}(\sqrt{\tilde\rho}\,\mathbf v)$, the phase rate
uses the BDF2 derivative, and the potential uses the implicit
$\phi^3-\phi$. This variant is second order in time, but the energy estimate
above is not proved for it. The diagnostics report the corresponding BDF2
modified energies and identity residuals for constant time steps.

## Parameters

In `subsection Cahn Hilliard`:

| Parameter | Description |
|---|---|
| `CHNS model` | `abels` (default) or `quasi_incompressible` |
| `mobility model` | `constant` or `degenerate_absolute` |
| `mobility` | constant mobility |
| `degenerate absolute mobility prefactor` | $m_0$ in $m_0\varepsilon^2\lvert 1-\phi^2\rvert$ (default 0.1) |
| `surface tension`, `interface thickness` | $\sigma$ and $\varepsilon$ |
| `write quasi incompressible diagnostics` | write `quasi_incompressible_diagnostics.csv` |
| `quasi incompressible periodic box` | periodic rectangle, see above |

The model requires a fixed mesh (no mesh adaptation, no ALE), `scheme = BDF1`
or `BDF2`, `enable supg = false` and `enable tracer supg = false`, and two
fluids with positive densities. These requirements are checked at startup.

## Outputs

In addition to the solution fields, the VTU/PVD output contains the
`mobility` field $M(\phi)$, evaluated at the visualization points from the
phase as in the assembly.

The following postprocessings, in `subsection Postprocessing`, write a table
`<output prefix>.txt` in the output directory. They are computed with the
assembly quadrature and do not require VTU output.

| Subsection | Content |
|---|---|
| `chns conservation` | domain volume, $\int\rho$, $\int\phi$, $\int\tilde\rho$, and their changes from the first monitored state and from the previous step |
| `chns energy` | kinetic, bulk, gradient, interfacial, gravitational and total energies, viscous and diffusive dissipations, and the residual of the discrete energy identity (`energy_balance_residual`, BDF1) or of the BDF2 identities |
| `chns phase energy` | the energy components integrated separately over $\phi \ge 0$ (`_phase_0`) and $\phi < 0$ (`_phase_1`) |

These tables and their reference state are saved with the checkpoint in a
`<checkpoint>.postprocessing` file, so that a restart continues them.
`quasi_incompressible_diagnostics.csv` additionally reports the numerical
dissipations, phase extrema and overshoot volumes, negative affine density
volume, $\lVert\mathbf v\rVert$, $\lVert\nabla\cdot\mathbf v\rVert$,
$\lVert\mathbf J\rVert$, the bubble ($\phi < 0$) area, centroid height and
rise velocity, and the BDF coefficients.

## Examples

| File | Case |
|---|---|
| `rising_bubble_case1.prm` | rising bubble, Hysing et al. case 1 (density ratio 10) |
| `rising_bubble_case2.prm` | rising bubble, Hysing et al. case 2 (density ratio 1000) |
| `phase_separation.prm` | spinodal decomposition in a periodic box |

Run them with

```sh
mpirun -n 4 ./incompressible_chns rising_bubble_case1.prm
```

## Tests

- `unit_tests/quasi_incompressible`: Jacobian against finite differences
  (BDF1/BDF2, constant and variable steps, both mobility laws, clipped
  phases), discrete energy identity, divided difference of the potential, BDF2
  kinetic identity and BDF2 restart coefficients.
- `unit_tests/zero_mean_pressure`: pivot of the zero-mean pressure constraint
  for linear and quadratic pressures.
- `incompressible_chns/quasi_incompressible_*`: assembled Jacobian, MMS time
  and space convergence orders, mass and energy balances, hydrostatic and
  periodic states, rejected configurations, BDF2 and restarts.
- `incompressible_chns/chns_postprocessing*`, `chns_phase_energy*`,
  `quasi_pressure_gauge`: monitoring tables against analytic fields, output
  options, checkpoint/restart and serial/MPI agreement.

## Implementation

| What | Where |
|---|---|
| Residual, exact Jacobian and assembler | `include/assembly/quasi_incompressible_chns_assembler.h` |
| Mixture laws and mobility | `include/cahn_hilliard.h` |
| Parameter checks, MMS source terms, constraints, diagnostics | `src/incompressible_chns_solver.cpp` |
| Domain integrals | `PostProcessingTools::compute_chns_integrals` |
| Tables and checkpoint state | `PostProcessingHandler::process_chns_integrals` |
