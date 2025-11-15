
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <mumps_solver.h>

// Shorthand notation for PETSc error codes.
#define AssertPETSc(code)                        \
  do                                             \
  {                                              \
    PetscErrorCode ierr = (code);                \
    AssertThrow(ierr == 0, ExcPETScError(ierr)); \
  }                                              \
  while (false)

DEAL_II_NAMESPACE_OPEN

namespace PETScWrappers
{
  SparseDirectMUMPSReuse::SparseDirectMUMPSReuse(SolverControl        &cn,
                                                 const AdditionalData &data)
    : SparseDirectMUMPS(cn, data)
  {
    initialize_ksp_with_comm(MPI_COMM_WORLD);
    AssertPETSc(KSPSetType(ksp, KSPPREONLY));
    AssertPETSc(KSPGetPC(ksp, &pc));
    AssertPETSc(PCSetType(pc, PCLU));
    AssertPETSc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  }

  void SparseDirectMUMPSReuse::solve(const MatrixBase &A,
                                     VectorBase       &x,
                                     const VectorBase &b)
  {
    AssertPETSc(KSPSetOperators(ksp, A, A));
    AssertPETSc(PCFactorSetUpMatSolverType(pc));
    AssertPETSc(PCFactorGetMatrix(pc, &factored_matrix));

    // Sequential and parallel reordering strategy
    MatMumpsSetIcntl(factored_matrix, 7, 2);
    MatMumpsSetIcntl(factored_matrix, 28, 2);
    MatMumpsSetIcntl(factored_matrix, 29, 1);

    // Override with command line options
    AssertPETSc(KSPSetFromOptions(ksp));

    // Solve
    AssertPETSc(KSPSolve(ksp, b, x));
  }

} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE