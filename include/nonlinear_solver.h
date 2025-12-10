#ifndef NONLINEAR_SOLVER_H
#define NONLINEAR_SOLVER_H

/**
 * This structure is borrowed from the Lethe project.
 */

#include <parameters.h>
#ifdef __has_include
#  if __has_include(<deal.II/base/observer_pointer.h>)

     // deal.II récent (9.8.0-pre etc.) : on utilise directement le header officiel
#    include <deal.II/base/observer_pointer.h>

#  else

     // deal.II plus ancien (comme 9.6.2 sur Trillium) : on fabrique un équivalent
#    include <deal.II/base/smartpointer.h>
#    include <deal.II/base/subscriptor.h>
namespace dealii
{
  template <typename T, typename P = void>
  using ObserverPointer = SmartPointer<T, P>;

  using EnableObserverPointer = Subscriptor;
} // namespace dealii

#  endif
#else

   // Si __has_include n’existe pas : on utilise le fallback « à l’ancienne »
#  include <deal.II/base/smartpointer.h>
#  include <deal.II/base/subscriptor.h>

namespace dealii
{
  template <typename T, typename P = void>
  using ObserverPointer = SmartPointer<T, P>;

  using EnableObserverPointer = Subscriptor;
} // namespace dealii

#endif


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
  virtual ~NonLinearSolver()
  {}

  /**
   * Solve the nonlinear problem (with e.g. Newton's method in the associated
   * derived class).
   */
  virtual void solve(const bool first_step) = 0;
};

#endif