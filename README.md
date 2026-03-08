[![workflows/github-CI-debug](https://github.com/arthurbawin/fez/actions/workflows/debug.yml/badge.svg?branch=master)](https://github.com/arthurbawin/fez/actions/workflows/debug.yml?query=branch%3Amaster)
[![workflows/github-CI-release](https://github.com/arthurbawin/fez/actions/workflows/release.yml/badge.svg?branch=master)](https://github.com/arthurbawin/fez/actions/workflows/release.yml?query=branch%3Amaster)

fez
================

fez is a collection of finite element solvers in C++ for the resolution of single and multiphase flows and fluid-structure interaction (FSI) problems, in a continuous Galerkin setting.
The solvers use an implicit and monolithic approach, solving all the flow variables in a single system and allowing for the resolution of numerically challenging problems, such as FSI simulations at low or zero structure-to-fluid mass ratio.

fez is built on [deal.II](https://dealii.org/), an open-source framework for parallel finite element computations, and borrows design patterns from [Lethe](https://github.com/chaos-polymtl/lethe), a collection of deal.II-based open-source solvers for computational fluid dynamics (CFD) and discrete element methods (DEM).

Installation
------------------

An installation of deal.II is required to compile and run fez, see for instance [here](https://dealii.org/current_release/download/) for the steps.
In addition to the base installation, deal.II should be compiled with the following dependencies :

- [PETSc](https://petsc.org/release/#) for the parallel linear algebra structures and solvers, which should be configured with the direct solver [MUMPS](https://mumps-solver.org/index.php?page=home), as it is currently used by most solvers.
- The [Gmsh](https://gmsh.info/) software development kit (SDK) to allow reading and writing meshes in .msh format
- [METIS](https://github.com/KarypisLab/METIS) for mesh partitioning

To run the test suite, [SymEngine](https://github.com/symengine/symengine) is also required to compute symbolically the source terms required by the method of manufactured solutions.

After your version of deal.II was compiled and installed with these libraries, clone fez from this repository with:

    $ git clone https://github.com/arthurbawin/fez
    $ cd fez

Then configure and compile with:

    $ mkdir build && cd build
    $ cmake -DCMAKE_BUILD_TYPE=Release ..
    $ make -j<N>

If deal.II was installed in, say, $HOME/local instead of your default install directory (such as /usr/local), simply specify the path to the installed deal.II by adding to the CMake call:

    $ -DCMAKE_PREFIX_PATH=$HOME/local

If deal.II was compiled with SymEngine and CMake was run with -DENABLE_TESTING=1, you can optionally run the test suite with:

    $ ctest

License:
--------

To be added.

Authors:
--------

- Arthur Bawin
- Joan Le Pouhaër
- Malo Le Scoul
- Marcin Stepien
