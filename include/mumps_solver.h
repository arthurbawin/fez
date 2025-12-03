
#ifndef MUMPS_SOLVER_H
#define MUMPS_SOLVER_H

#include <deal.II/lac/petsc_solver.h>

DEAL_II_NAMESPACE_OPEN

namespace PETScWrappers
{
  class SparseDirectMUMPSReuse : public SparseDirectMUMPS
  {
  public:
    /**
     * Constructor.
     */
    SparseDirectMUMPSReuse(SolverControl        &cn,
                           const AdditionalData &data = AdditionalData());

    /**
     * The method to solve the linear system.
     */
    void solve(const MatrixBase &A, VectorBase &x, const VectorBase &b);

  protected:
    bool first = true;
    PC   pc;
    Mat  factored_matrix;
  };
} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
