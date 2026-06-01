
# Change Log
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/).
Please use the tags "Added", "Changed" or "Fixed", depending on the nature of the changes.
 
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