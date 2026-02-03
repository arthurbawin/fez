#ifndef NONLINEAR_SOLVER_H
#define NONLINEAR_SOLVER_H

/**
 * This structure is borrowed from the Lethe project.
 */

#include <deal.II/base/observer_pointer.h>
#include <parameters.h>

using namespace dealii;

template <typename VectorType>
class GenericSolver;

/**
 * Abstract base class for nonlinear solvers.
 *
 * This is required because the GenericSolver is templated,
 * so the implementation of the solve() function should be
 * in the header of the file that implements it (or explicitly instantiated for
 * all VectorTypes...)
 * FIXME: Make this clearer (-:
 */
template <typename VectorType>
class NonLinearSolver
{
protected:
  const Parameters::NonLinearSolver param;

  // The attached solver
  ObserverPointer<GenericSolver<VectorType>> solver;

public:
  /**
   * Constructor
   */
  NonLinearSolver(const Parameters::NonLinearSolver &param,
                  GenericSolver<VectorType>         *solver)
    : param(param)
    , solver(ObserverPointer<GenericSolver<VectorType>>(solver))
  {}

  /**
   * Destructor
   */
  virtual ~NonLinearSolver() {}

  /**
   * Solve the nonlinear problem (with e.g. Newton's method in the associated
   * derived class).
   */
  virtual void solve(const bool first_step) = 0;
};

#endif