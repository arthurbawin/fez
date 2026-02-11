#ifndef LINEAR_DIRECT_SOLVER_H
#define LINEAR_DIRECT_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <mumps_solver.h>
#include <parameters.h>
#include <types.h>

using namespace dealii;

template <typename VectorType>
class GenericSolver;

/**
 *
 */
void solve_linear_system_direct(
  GenericSolver<LA::ParVectorType> *solver,
  const Parameters::LinearSolver   &linear_solver_param,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints);

void solve_linear_system_direct(
  GenericSolver<LA::ParVectorType>      *solver,
  const Parameters::LinearSolver        &linear_solver_param,
  LA::ParMatrixType                     &system_matrix,
  const IndexSet                        &locally_owned_dofs,
  const AffineConstraints<double>       &zero_constraints,
  PETScWrappers::SparseDirectMUMPSReuse &direct_solver);

void solve_linear_system_iterative(
  GenericSolver<LA::ParVectorType> *solver,
  const Parameters::LinearSolver   &linear_solver_param,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints);

#endif