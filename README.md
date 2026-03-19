[![workflows/github-CI-debug](https://github.com/arthurbawin/fez/actions/workflows/debug.yml/badge.svg?branch=master)](https://github.com/arthurbawin/fez/actions/workflows/debug.yml?query=branch%3Amaster)
[![workflows/github-CI-release](https://github.com/arthurbawin/fez/actions/workflows/release.yml/badge.svg?branch=master)](https://github.com/arthurbawin/fez/actions/workflows/release.yml?query=branch%3Amaster)

fez
================

fez is a collection of finite element solvers in C++ for the resolution of single and multiphase flows and fluid-structure interaction (FSI) problems, in a continuous Galerkin setting.
The solvers use an implicit and monolithic approach, solving all the flow variables in a single system and allowing for the resolution of numerically challenging problems, such as FSI simulations at zero structure-to-fluid mass ratio or compressible navier-stokes bi-fluid wave impact simulation.

fez is built on [deal.II](https://dealii.org/), an open-source framework for parallel finite element computations, and borrows design patterns from [Lethe](https://github.com/chaos-polymtl/lethe), a collection of deal.II-based open-source solvers for computational fluid dynamics (CFD) and discrete element methods (DEM).

Current scope
------------------

At the moment, the repository contains solvers and validation cases for:

- incompressible Navier-Stokes
- incompressible Navier-Stokes with Lagrange multipliers with HP frame
- incompressible Cahn-Hilliard / Navier-Stokes (CHNS)
- ALE variants of incompressible CHNS
- linear elasticity
- heat equation
- ALE FSI incompressible Navier-Stokes with HP frame
- ALE FSI incompressible Navier-Stokes without HP frame
- compressible Navier-Stokes

The code supports 2D and 3D cases depending on the solver and parameter file, and is designed for hybrid MPI+OpenMP-parallel runs through deal.II and PETSc.

Main dependencies
------------------

An installation of deal.II is required to compile and run fez, see for instance [here](https://dealii.org/current_release/download/) for the steps.

fez currently requires or uses the following external libraries through deal.II:

- [PETSc](https://petsc.org/release/) for distributed linear algebra and solvers
- [MUMPS](https://mumps-solver.org/index.php?page=home) as the main direct sparse solver currently used by most solvers through PETSc
- [Trilinos](https://trilinos.github.io/) can also be selected as a linear algebra backend if deal.II was compiled with it
- [Gmsh](https://gmsh.info/) to read and write `.msh` meshes
- [METIS](https://github.com/KarypisLab/METIS) for mesh partitioning
- [SymEngine](https://github.com/symengine/symengine) to run the manufactured-solution-based test suite

By default, fez selects the PETSc backend when it is available in deal.II, and falls back to Trilinos otherwise.

Repository structure
------------------

The main directories are organized as follows:

- `include/`: public headers for the core library and solver classes used by sources files.
- `src/`: the sources files of the solver who implement the reusable library code
- `solvers/`: the solvers files with one executable per solver
- `tests/`: regression tests, MMS validation cases, and unit tests
- `examples/`: parameter files and mesh files to demonstrate how to use different solvers and parameter configurations.
- `data/`: additional meshes, parameter files, and working datasets
- `contrib/`: bundled third-party utilities, including Eigen headers

A typical code path is:

1. a solver executable from `solvers/` reads a `.prm` file,
2. the parameter system in `include/parameter_reader.h` and `src/parameter_reader.cpp` configures the run,
3. the corresponding solver class from `include/` and `src/` assembles and solves the problem,
4. post-processing and outputs are handled by the common library infrastructure.

Build
------------------

After your version of deal.II was compiled and installed with the required libraries, clone fez from this repository with:

    $ git clone https://github.com/arthurbawin/fez
    $ cd fez

Then configure and compile with:

    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make -j<N>

If deal.II was installed in, say, `$HOME/local` instead of your default install directory (such as `/usr/local`), simply specify the path to the installed deal.II by adding to the CMake call:

    $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HOME/local ..

You can also explicitly choose the linear algebra backend at configure time:

    $ cmake -DFEZ_FORCE_PETSC=ON ..
    $ cmake -DFEZ_FORCE_TRILINOS=ON ..

Running a solver
------------------

Each executable expects a parameter file as input, for example:

    $ mpirun -np 4 ./incompressible_ns ../tests/incompressible_ns/mms_3d_bdf2.prm

or in serial:

    $ ./heat ../tests/heat/mms_2d_bdf2_time_step_adaptative.prm

Most executables detect the problem dimension directly from the parameter file and then instantiate the appropriate 2D or 3D solver.

Testing
------------------

fez contains both solver regression tests and lower-level unit tests.

The current `tests/` tree includes:

- solver-based tests for `incompressible_ns`
- solver-based tests for `incompressible_ns_lambda`
- solver-based tests for `incompressible_chns`
- solver-based tests for `incompressible_chns_ale`
- solver-based tests for `heat`
- solver-based tests for `linear_elasticity`
- solver-based tests for `fsi`
- solver-based tests for `monolithic_fsi`
- unit tests in `tests/unit_tests`

In practice, the test suite is largely based on:

- manufactured solutions (MMS)
- regression checks against stored `.output` files
- serial and MPI runs, with several cases checked for multiple processor counts

At the moment, the repository contains:

- 41 parameterized test cases (`*.prm`)
- 5 unit test source files in `tests/unit_tests`
- reference outputs for serial and parallel configurations

To enable tests at configuration time:

    $ cmake -DENABLE_TESTING=ON ..

If deal.II was compiled with SymEngine and CMake was run with `-DENABLE_TESTING=ON`, you can run the full test suite with:

    $ ctest --output-on-failure

Development workflow
------------------

When developing, the recommended workflow is:

1. configure a debug build with testing enabled
2. compile the full project
3. run the test suite before opening a pull request
4. add a regression test for every bug fix or new feature whenever possible

A typical development build is:

    $ mkdir build-debug && cd build-debug
    $ cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TESTING=ON ..
    $ make -j<N>
    $ ctest --output-on-failure

The GitHub CI currently runs both:

- a Debug configuration
- a Release configuration

Both workflows build the code with `ENABLE_TESTING=1` and run `ctest`.
At the moment, the CI excludes the `incompressible_ns_lambda` 3D tests because of an hp-related issue observed in the current deal.II container used by the workflow.

Adding a new test
------------------

To add a new solver regression test:

- add a parameter file `my_test.prm` in the relevant subdirectory of `tests/`
- generate the corresponding reference output `my_test.output`
- add any required mesh files next to the test if needed

To add a new unit or library-level test:

- add a source file `my_test.cc` in `tests/unit_tests`
- generate the associated reference output file

Because the test discovery relies on deal.II's CMake testing macros, placing the files in the appropriate test directory is generally enough.

Implemented executables
------------------

The current `solvers/` directory builds the following executables:

- `compressible_ns`
- `fsi`
- `heat`
- `incompressible_chns`
- `incompressible_chns_ale`
- `incompressible_ns`
- `incompressible_ns_lambda`
- `linear_elasticity`
- `monolithic_fsi`

License
--------

To be added.

Authors
--------

- Arthur Bawin
- Joan Le Pouhaër
- Malo Le Scoul
- Marcin Stepien
