#ifndef LINEAR_DIRECT_SOLVER_H
#define LINEAR_DIRECT_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/affine_constraints.h>
#include <types.h>

using namespace dealii;

template <typename VectorType>
class GenericSolver;

/**
 * 
 */
void solve_linear_system_direct(GenericSolver<LA::ParVectorType> *solver,
                           LA::ParMatrixType                &system_matrix,
                           const IndexSet                   &locally_owned_dofs,
                           const AffineConstraints<double>  &zero_constraints);

#endif