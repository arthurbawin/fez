#ifndef TYPES_H
#define TYPES_H

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// Linear algebra
namespace LA
{
#if defined(FEZ_WITH_PETSC)
  using namespace dealii::LinearAlgebraPETSc;
  using ParMatrixType = dealii::LinearAlgebraPETSc::MPI::SparseMatrix;
  using ParVectorType = dealii::LinearAlgebraPETSc::MPI::Vector;
  using ConstMatrixIterator =
    dealii::PETScWrappers::MatrixIterators::const_iterator;
#elif defined(FEZ_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
  using ParMatrixType = dealii::LinearAlgebraTrilinos::MPI::SparseMatrix;
  using ParVectorType = dealii::LinearAlgebraTrilinos::MPI::Vector;
  using ConstMatrixIterator =
    dealii::TrilinosWrappers::SparseMatrixIterators::Iterator<true>;
#else

#  error FEZ_WITH_PETSC or FEZ_WITH_TRILINOS required

#endif
} // namespace LA

#endif
