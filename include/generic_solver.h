#ifndef GENERIC_SOLVER_H
#define GENERIC_SOLVER_H

/**
 * This structure is borrowed from the Lethe project.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <newton_solver.h>
#include <nonlinear_solver.h>
#include <parameter_reader.h>
#include <types.h>

using namespace dealii;

/**
 * Abstract base class for a generic solver with the following common members:
 *
 * - simulation parameters
 * - mpi communicator
 * - nonlinear solver
 */
template <typename VectorType>
class GenericSolver : public EnableObserverPointer
{
public:
  GenericSolver(const Parameters::NonLinearSolver &nonlinear_solver_param,
                const Parameters::Timer           &timer_param)
    : mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    // Disable timer if needed
    if (!timer_param.enable_timer)
      computing_timer.disable_output();

    // Create the nonlinear solver (Newton-Raphson solver)
    nonlinear_solver =
      std::make_shared<NewtonSolver<VectorType>>(nonlinear_solver_param, this);
  }

  virtual ~GenericSolver() {}

public:
  /**
   * Solve the problem
   */
  virtual void run() = 0;

  /**
   *
   */
  virtual void assemble_matrix() = 0;

  /**
   *
   */
  virtual void assemble_rhs() = 0;

  /**
   *
   */
  virtual void
  solve_linear_system(const bool apply_inhomogeneous_constraints) = 0;

  /**
   *
   */
  void solve_nonlinear_problem(const bool first_step)
  {
    nonlinear_solver->solve(first_step);
  }

  /**
   *
   */
  virtual AffineConstraints<double> &get_nonzero_constraints() = 0;

  /**
   *
   */
  void distribute_nonzero_constraints()
  {
    const auto &nonzero_constraints = this->get_nonzero_constraints();
    nonzero_constraints.distribute(local_evaluation_point);
  }

  // Non-const getters
  VectorType &get_present_solution() { return present_solution; }
  VectorType &get_evaluation_point() { return evaluation_point; }
  VectorType &get_local_evaluation_point() { return local_evaluation_point; }
  VectorType &get_newton_update() { return newton_update; }
  VectorType &get_system_rhs() { return system_rhs; }

public:
  MPI_Comm           mpi_communicator;
  const unsigned int mpi_rank;
  const unsigned int mpi_size;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

protected:
  // If parallel vector type, these are vectors with ghost entries (read only)
  VectorType present_solution;
  VectorType evaluation_point;

  // If parallel vector type, these are vectors w/o ghost entries (owned)
  VectorType local_evaluation_point;
  VectorType newton_update;
  VectorType system_rhs;

  std::shared_ptr<NonLinearSolver<VectorType>> nonlinear_solver;

  // Friend-ness is not inherited, so each derived nonlinear solver
  // should be marked as friend individually.
  friend class NewtonSolver<VectorType>;
};

#endif