#ifndef GENERIC_SOLVER_H
#define GENERIC_SOLVER_H

/**
 * This structure is borrowed from the Lethe project.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <error_handler.h>
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
                const Parameters::Timer           &timer_param,
                const Parameters::Mesh            &mesh_param,
                const Parameters::TimeIntegration &time_param,
                const Parameters::MMS             &mms_param,
                const SolverType                   solver_type)
    : mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , solver_type(solver_type)
    , mesh_param(mesh_param)
    , time_param(time_param)
    , mms_param(mms_param)
  {
    // Disable timer if needed
    if (!timer_param.enable_timer)
      computing_timer.disable_output();

    // Create the nonlinear solver (Newton-Raphson solver)
    nonlinear_solver =
      std::make_shared<NewtonSolver<VectorType>>(nonlinear_solver_param, this);

    // Create the error handlers
    for (auto norm : mms_param.norms_to_compute)
      error_handlers[norm] =
        std::make_shared<ErrorHandler>(mms_param, time_param);
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
  template <int dim>
  void run_convergence_loop()
  {
    for (unsigned int i_conv = 0; i_conv < mms_param.n_convergence; ++i_conv)
    {
      mms_param.current_step = i_conv;
      for (auto &[norm, handler] : error_handlers)
        handler->clear_error_history();

      // If a manufactured solution test is run, bypass the given mesh file
      // and run the i_conv-th prescribed mms mesh
      // Update mesh file. Keep previous mesh file only if it is a time
      // convergence study.
      bool update_mesh = mms_param.type == Parameters::MMS::Type::space ||
                         mms_param.type == Parameters::MMS::Type::spacetime ||
                         (mms_param.type == Parameters::MMS::Type::time &&
                          mms_param.use_space_convergence_mesh && i_conv == 0);

      if (update_mesh)
      {
        // This change is accounted for in the reset() function of each
        // derived solver
        mms_param.mesh_suffix = i_conv;

        // If this is a time convergence study using an indexed mesh used
        // for space convergence studies, override the mesh_suffix with
        // the specified mesh index
        if (mms_param.type == Parameters::MMS::Type::time &&
            mms_param.use_space_convergence_mesh)
          mms_param.mesh_suffix = mms_param.spatial_mesh_index;

        if (mms_param.run_only_step >= 0)
        {
          mms_param.current_step = mms_param.run_only_step;

          // Set the mesh suffix to the only run step for space studies
          // For time studies, the "run only" only affects the time step.
          if (mms_param.type == Parameters::MMS::Type::space ||
              mms_param.type == Parameters::MMS::Type::spacetime)
            mms_param.mesh_suffix = mms_param.run_only_step;
        }

        if (!mms_param.use_deal_ii_cube_mesh)
        {
          mms_param.override_mesh_filename(mesh_param, mms_param.mesh_suffix);
          pcout << "Convergence test with manufactured solution:" << std::endl;
          pcout << "Mesh file was changed to " << mesh_param.filename
                << std::endl;
        }
      }

      // Update time step starting at second iteration
      bool update_time_step =
        (i_conv > 0) && (mms_param.type == Parameters::MMS::Type::time ||
                         mms_param.type == Parameters::MMS::Type::spacetime);

      if (update_time_step)
      {
        // This change is accounted for in the reset() function of each
        // derived solver
        time_param.dt *= mms_param.time_step_reduction_factor;
      }

      this->run();

      // If unsteady, compute the Lp time norm for this convergence step
      if (time_param.scheme != Parameters::TimeIntegration::Scheme::stationary)
        for (auto &[norm, handler] : error_handlers)
          handler->compute_temporal_error();

      if (mms_param.run_only_step >= 0)
        break;
    }

    for (auto &[norm, handler] : error_handlers)
      handler->template compute_rates<dim>();
    if (mpi_rank == 0)
      for (auto &[norm, handler] : error_handlers)
      {
        std::cout << std::endl;
        std::cout << Patterns::Tools::Convert<VectorTools::NormType>::to_string(
                       norm)
                  << std::endl;
        handler->write_rates();
      }
  }

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

  VectorType &get_present_solution() { return present_solution; }
  VectorType &get_evaluation_point() { return evaluation_point; }
  VectorType &get_local_evaluation_point() { return local_evaluation_point; }
  VectorType &get_newton_update() { return newton_update; }
  VectorType &get_system_rhs() { return system_rhs; }

  const Parameters::TimeIntegration &get_time_parameters()
  {
    return time_param;
  }

public:
  MPI_Comm           mpi_communicator;
  const unsigned int mpi_rank;
  const unsigned int mpi_size;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

protected:
  SolverType solver_type;

  // If parallel vector type, these are vectors with ghost entries (read only)
  VectorType present_solution;
  VectorType evaluation_point;

  // If parallel vector type, these are vectors w/o ghost entries (owned)
  VectorType local_evaluation_point;
  VectorType newton_update;
  VectorType system_rhs;

  std::shared_ptr<NonLinearSolver<VectorType>> nonlinear_solver;

  // Data to perform a space and/or time convergence study
  Parameters::Mesh            mesh_param;
  Parameters::TimeIntegration time_param;
  Parameters::MMS             mms_param;

  // An ErrorHandler for each error norm
  std::map<VectorTools::NormType, std::shared_ptr<ErrorHandler>> error_handlers;

  // Friend-ness is not inherited, so each derived nonlinear solver
  // should be marked as friend individually.
  friend class NewtonSolver<VectorType>;
};

#endif
