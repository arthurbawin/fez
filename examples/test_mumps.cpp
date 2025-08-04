// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2004 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// test the PETSc SparseDirectMumps solver


#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <typeinfo>

// #include "../tests.h"
#include "dealii_testmatrix.h"

using namespace dealii;

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  {
    const unsigned int n_points = 32;
    unsigned int       size     = (n_points - 1) * (n_points - 1);

    std::cout << "Size " << n_points << " Unknowns " << size << std::endl;

    // Make matrix
    FDMatrix                    testproblem(n_points, n_points);
    PETScWrappers::SparseMatrix A(size, size, 5);
    testproblem.five_point(A);

    IndexSet indices(size);
    indices.add_range(0, size);
    PETScWrappers::MPI::Vector f(indices, MPI_COMM_WORLD);
    PETScWrappers::MPI::Vector u(indices, MPI_COMM_WORLD);
    A.compress(VectorOperation::insert);

    // A.view("matrix.mtx");

    std::cout << "Matrix rows: " << A.m() << ", nnz: " << A.n_nonzero_elements() << std::endl;

    // First test non-symmetric mode:
    {
      std::cout << "Testing non-symmetric mode" << std::endl;

      u = 0.;
      f = 1.234;

      SolverControl                    cn;
      PETScWrappers::SparseDirectMUMPS solver(cn);
      solver.solve(A, u, f);

      PETScWrappers::MPI::Vector tmp(indices, MPI_COMM_WORLD);
      std::cout << "residual = " << A.residual(tmp, u, f) << std::endl;
    }

    // // Now also test the case where the matrix can be assumed to be
    // // symmetric (which of course it is here):
    // {
    //   std::cout << "Testing symmetric mode" << std::endl;

    //   u = 0.;
    //   f = 1.23;

    //   SolverControl                    cn;
    //   PETScWrappers::SparseDirectMUMPS solver(cn);
    //   solver.set_symmetric_mode(true);
    //   solver.solve(A, u, f);

    //   u.print(std::cout);

    //   PETScWrappers::MPI::Vector tmp(indices, MPI_COMM_WORLD);
    //   std::cout << "residual = " << A.residual(tmp, u, f) << std::endl;
    // }
  }
}
