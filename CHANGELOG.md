
# Change Log
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/).
Please use the tags "Added", "Changed" or "Fixed", depending on the nature of the changes.
 
## [0.0.0] 2026-10-22
 
### Added
Added the "lgpi" deal.II preset mesh (2D only) for the large gas pocket impact benchmark: a rectangular box minus a quarter ellipse in the bottom left corner. [#99](https://github.com/arthurbawin/fez/pull/99)

### Fixed
Fixed bugs found while reviewing the solvers, assemblers and mesh adaptation tools [#99](https://github.com/arthurbawin/fez/pull/99):
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
Removed the currently unsupported Trilinos code paths from the incompressible NS, CHNS, heat and compressible NS solvers. Added FIXME comments for known issues not addressed yet (compressible NS weak-pressure Jacobian and forced source term, moving-mesh solution transfer between time subintervals). [#99](https://github.com/arthurbawin/fez/pull/99)

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