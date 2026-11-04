
# Change Log
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/).
Please use the tags "Added", "Changed" or "Fixed", depending on the nature of the changes.

## [0.0.0] 2026-11-04

### Added
Added a pull request template. [#113](https://github.com/arthurbawin/fez/pull/113)

## [0.0.0] 2026-11-03

### Changed
Improved reproducibility of anisotropic mesh adaptation tests across platforms [#83](https://github.com/arthurbawin/fez/pull/83):
 - Metric tensor coefficients given to MMG are rounded to 20 significant bits (about 6 significant digits) of their binary mantissa, using exact operations. The previous rounding kept a fixed number of decimal places, i.e., more significant digits for large coefficients.
 - When testing is enabled, the bundled MMG is compiled with `-ffp-contract=off` (GCC and Clang), so that it does not use fused multiply-adds (on by default with Clang on arm64) and performs the same floating-point operations on all platforms.

### Added
Added 2D unsteady anisotropic mesh adaptation tests with the transient fixed-point method, for an MMS with the heat solver, and a rising bubble with the CHNS solver. [#83](https://github.com/arthurbawin/fez/pull/83) 

## [0.0.0] 2026-11-02

### Changed
Reworked the linear solvers into the `LinearSolvers` namespace [#105](https://github.com/arthurbawin/fez/pull/105):
 - The "tolerance" parameter is replaced by "absolute tolerance" and "relative tolerance": iterative solvers stop when the residual norm is below the absolute tolerance, or below the relative tolerance times the initial residual norm.
 - The preconditioner is now chosen with the "preconditioner" parameter, which defaults to none. Previously, GMRES always used ILU and CG always used block Jacobi.
 - `solve_gmres` is templated on the preconditioner type, and returns the number of iterations and whether GMRES converged. A non-converged GMRES solve now prints a warning and its last iterate is used as an inexact Newton update, instead of throwing.
 - The Navier-Stokes base class delegates iterative solves to the derived solvers, which choose the suitable solvers and preconditioners. The incompressible NS solver supports GMRES; the heat and linear elasticity solvers support MUMPS and CG.

Renamed the tests whose output depends on optional deal.II features to state them explicitly (e.g. `with_p4est=true`, `with_lapack=true`, PETSc version and hypre), and moved the tree-based adaptation tests to the directories of their physics. [#105](https://github.com/arthurbawin/fez/pull/105)

Shortened most tests and enabled caching previously built files with ccache, to speedup CI pipelines. [#106](https://github.com/arthurbawin/fez/pull/106)

Add the possible tests dependencies in their name, as for deal.II tests (e.g., with_pest=true). This groups the tests with their solver, and removes the "tests/tree_based_adaptation" directory. [#107](https://github.com/arthurbawin/fez/pull/107)
 
### Added
Added an AMG preconditioner for the GMRES solves of the incompressible NS solver, using hypre's BoomerAMG through PETSc, with an ILU smoother. For P1-P1 and Q1-Q1 elements, the block size of the system matrix is set to the number of components, and BoomerAMG coarsens nodes rather than individual dofs. Requires PETSc 3.22 or higher, configured with hypre. [#105](https://github.com/arthurbawin/fez/pull/105)

Added a preconditioner rebuild heuristic for the incompressible NS solver [#105](https://github.com/arthurbawin/fez/pull/105):
 - Steady problems rebuild the preconditioner at each linear solve.
 - Unsteady problems reuse it across Newton iterations and time steps, and rebuild it after a mesh or dof change, a change of the leading BDF coefficient (time step or BDF order), a non-converged GMRES solve, or when the GMRES iterations exceed both 1.5 times, and 5 more than, those of the first solve with the current preconditioner.
 - If GMRES does not converge with a reused preconditioner, the preconditioner is rebuilt and the system is solved again once.
 - The Newton solver output now prints "(P)" next to the residual when the preconditioner was rebuilt.

Added the following parameters to the "Linear solver" subsection: "preconditioner" (none, ilu, block_jacobi or amg), "absolute tolerance", "relative tolerance", "max krylov vectors" (GMRES restart length) and "right preconditioning" (right or left preconditioning for GMRES). [#105](https://github.com/arthurbawin/fez/pull/105)

## [0.0.0] 2026-10-29
 
### Changed
Modified the lgpi mesh preset to be more axis aligned. [#102](https://github.com/arthurbawin/fez/pull/102)

## [0.0.0] 2026-10-22
 
### Added
Added the "lgpi" deal.II preset mesh (2D only) for the large gas pocket impact benchmark: a rectangular box minus a quarter ellipse in the bottom left corner. [#99](https://github.com/arthurbawin/fez/pull/99)

### Fixed
Fixed bugs found while reviewing the solvers, assemblers and mesh adaptation tools [#100](https://github.com/arthurbawin/fez/pull/100):
 - Linear elasticity: the ScratchData `components` vector was sized with the number of quadrature points instead of the number of dofs per cell, causing out-of-bounds writes.
 - NS with Lagrange multiplier: the time error estimator was never attached (crash with adaptive time stepping), and the hp line identities loops now skip artificial cells.
 - Monolithic FSI: owned boundary dofs only seen as boundary dofs by other ranks (3D) were missed when exchanging boundary dofs; the Lagrange multiplier error now uses the kinematic viscosity, consistent with the solved equations; fixed an MPI operator leak and made the rigid-body displacement check two-sided and across ranks; rigid-body rotation now throws at setup for unsupported coupling schemes.
 - CHNS: the MMS source term now uses the solver's parameters (consistent interface thickness during fixed-point updates) and the same tracer limiter as the ScratchData; an `open_mms` boundary condition now throws instead of being silently ignored.
 - Heat: the initial condition is now interpolated on the final mesh after initial (pre-)refinement, and constraints are applied to the exact solution.
 - Transient fixed point: previous solutions interpolated from the exact solution at the start of a subinterval were shifted by one time step; min/max refinement levels are now enforced on all levels for tree-based adaptation; space-time complexity and mesh counts are now accumulated as 64-bit integers.
 - Metric fields: ghost metrics are synchronized before the non-deterministic gradation, and after the deterministic one.
 - Function objects (source terms, exact solutions) now update their own time in `set_time()`, and component lookups in `MMSSolution` use the component ordering.
 - Per-time-step MMS errors and the Lagrange multiplier dof counts are now written from the root process only, and the parallel metric gradation only prints its progress when verbose.

### Changed
Removed the currently unsupported Trilinos code paths from the incompressible NS, CHNS, heat and compressible NS solvers. Added FIXME comments for known issues not addressed yet (compressible NS weak-pressure Jacobian and forced source term, moving-mesh solution transfer between time subintervals). [#100](https://github.com/arthurbawin/fez/pull/100)

## [0.0.0] 2026-06-04
 
### Changed
Merged both CopyData structures into a single one. [#61](https://github.com/arthurbawin/fez/pull/61)

Refactored ScratchData to use an integer template argument containing all the desired updates during call to the reinit function. [#62](https://github.com/arthurbawin/fez/pull/62)

## [0.0.0] 2026-05-29
 
### Added
Extended the transient fixed-point framework to the Navier-Stokes solver base class,
enabling unsteady metric-based mesh adaptation for Navier-Stokes derived solvers.
Improvement to the metric field interface to allow defining multiple metrics and choosing
one as the metric for mesh adaptation (e.g., their intersection). [#60](https://github.com/arthurbawin/fez/pull/60)

This has the following practical changes:
 - the "core" data structures (triangulation, dof_handler, present and previous solutions vectors) are stored in a TransientFixedPointData, which is in charge of their ownership (through unique_ptr). The plain data (for the triangulation, dof_handler, ...) in the NS class have been replaced by non-owning raw pointers.
 - postprocess_solution() is now called on the initial condition in the NS base class, since the Riemannian metric adapted to the initial time is required for unsteady metric-based adaptation. This changes tests which output the result of e.g. forces computations at each time step (the zero-th step has to be added). 
 
## [0.0.0] 2026-05-29
 
### Added
Adding reconstructions of derivatives of order 3.
[#59](https://github.com/arthurbawin/fez/pull/59)
