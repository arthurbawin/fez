
#include <components_ordering.h>
#include <solver_info.h>
#include <time_handler.h>
#include <timestep_adaptation.h>

#include <algorithm>

constexpr auto STAT = Parameters::TimeIntegration::Scheme::stationary;
constexpr auto BDF1 = Parameters::TimeIntegration::Scheme::BDF1;
constexpr auto BDF2 = Parameters::TimeIntegration::Scheme::BDF2;

TimeHandler::TimeHandler(const Parameters::TimeIntegration &time_parameters)
  : time_parameters(time_parameters)
  , steady_scheme(time_parameters.scheme == STAT)
  , current_time(time_parameters.t_initial)
  , current_time_iteration(0)
  , initial_time(time_parameters.t_initial)
  , final_time(time_parameters.t_end)
  , initial_dt(time_parameters.dt)
  , current_dt(time_parameters.dt)
  , scheme(time_parameters.scheme)
  , with_adaptive_timestep(time_parameters.adaptation.enable)
  , rolledback_step(false)
  , n_consecutive_rejected_steps(0)
  , n_rejected_steps(0)
  , last_nonlinear_solver_converged(true)
  , max_cfl_number(0.)
{
  switch (scheme)
  {
    case STAT:
      bdf_order            = 0;
      n_previous_solutions = 0;
      break;
    case BDF1:
      bdf_order            = 1;
      n_previous_solutions = 1;
      break;
    case BDF2:
    {
      bdf_order            = 2;
      n_previous_solutions = 2;

      // Set the initial time step if using BDF1 as starting method
      if (time_parameters.bdfstart ==
          Parameters::TimeIntegration::BDFStart::BDF1)
      {
        current_dt = initial_dt * time_parameters.bdf_starting_step_ratio;
      }
      break;
    }
  }

  simulation_times.resize(n_previous_solutions + 1, initial_time);
  all_simulation_times.push_back(initial_time);
  time_steps.resize(n_previous_solutions + 1, initial_dt);
  bdf_coefficients.resize(n_previous_solutions + 1, 0.);

  if (!is_steady() && with_adaptive_timestep)
    error_estimator =
      std::make_shared<BDFErrorEstimator>(time_parameters, *this);
}

TimeHandler::~TimeHandler() = default;

void TimeHandler::validate_parameters(const ComponentOrdering &ordering) const
{
  // Safety checks depending on the solver
  if (time_parameters.adaptation.strategy ==
      Parameters::TimeIntegration::Adaptation::AdaptationStrategy::CFL)
    AssertThrow(
      ordering.has_variable(SolverInfo::VariableType::velocity),
      ExcMessage(
        "You are trying to adapt the time step based on the maximum "
        "Courant-Fredrichs-Lewy (CFL) number, which requires a velocity field, "
        "with a solver which does not solve for the velocity variable. "
        "The alternative is to adapt based on the BDF truncation error, "
        "by setting \"set adaptation strategy = bdf truncation error\"."));
}

bool TimeHandler::is_steady() const
{
  return steady_scheme;
}

bool TimeHandler::is_starting_step() const
{
  return current_time_iteration < n_previous_solutions;
}

bool TimeHandler::last_step_was_starting_step() const
{
  return ((int)current_time_iteration - 1) < (int)n_previous_solutions &&
         ((int)current_time_iteration - 1) > 0;
}

bool TimeHandler::is_finished() const
{
  if (scheme == STAT)
  {
    // Stop after a single "time step"
    return current_time_iteration > 0;
  }
  return current_time >= final_time - 1e-10;
}

void TimeHandler::set_bdf_coefficients(
  const bool                                force_scheme,
  const Parameters::TimeIntegration::Scheme forced_scheme)
{
  const Parameters::TimeIntegration::Scheme used_scheme =
    force_scheme ? forced_scheme : scheme;
  switch (used_scheme)
  {
    case STAT:
      bdf_coefficients[0] = 0.;
      break;
    case BDF1:
    {
      const double dt     = time_steps[0];
      bdf_coefficients[0] = 1. / dt;
      bdf_coefficients[1] = -1. / dt;
      break;
    }
    case BDF2:
    {
      const double dt      = time_steps[0];
      const double prev_dt = time_steps[1];
      bdf_coefficients[0]  = 1.0 / dt + 1.0 / (dt + prev_dt);
      bdf_coefficients[1]  = -1.0 / dt - 1.0 / (prev_dt);
      bdf_coefficients[2]  = dt / prev_dt * 1. / (dt + prev_dt);
      break;
    }
  }
}

void TimeHandler::advance(const ConditionalOStream &pcout)
{
  current_time_iteration++;

  if (scheme == STAT)
    return;

  // Rotate the times and time steps
  if (!rolledback_step)
    for (unsigned int i = n_previous_solutions; i >= 1; --i)
    {
      simulation_times[i] = simulation_times[i - 1];
      time_steps[i]       = time_steps[i - 1];
    }

  // Force another BDF scheme to set the coefficients if this is a starting step
  bool force_scheme = false;
  if (scheme == BDF2 &&
      time_parameters.bdfstart == Parameters::TimeIntegration::BDFStart::BDF1)
    if (is_starting_step())
      force_scheme = true;

  // Update the tables and BDF coefficients
  current_time += current_dt;
  simulation_times[0] = current_time;
  time_steps[0]       = current_dt;
  set_bdf_coefficients(force_scheme);

  if (time_parameters.verbosity == Parameters::Verbosity::verbose)
  {
    pcout << std::endl
          << "Time step " << current_time_iteration
          << " - Advancing to t = " << current_time;
    if (with_adaptive_timestep &&
        time_parameters.adaptation.verbosity == Parameters::Verbosity::verbose)
      pcout << " with time step " << current_dt;
    pcout << '.' << std::endl;
  }

  // Advance the time tables in the error estimator
  if (with_adaptive_timestep)
    error_estimator->advance(*this);

  rolledback_step = false;
}

void TimeHandler::rotate_solutions(
  const LA::ParVectorType        &present_solution,
  std::vector<LA::ParVectorType> &previous_solutions) const
{
  if (!this->is_steady())
  {
    // Rotate the additional solution vector in the error estimator
    // Do this first, before overwriting the last solution
    if (!is_steady() && with_adaptive_timestep)
      error_estimator->rotate_additional_solution(previous_solutions.back());

    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
}

void TimeHandler::attach_data_to_error_estimator(
  const ComponentOrdering          &ordering,
  const IndexSet                   &locally_relevant_dofs,
  const std::vector<unsigned char> &dofs_to_component)
{
  if (error_estimator)
    error_estimator->attach_data(ordering,
                                 locally_relevant_dofs,
                                 dofs_to_component);
}

const LA::ParVectorType &TimeHandler::get_error_estimator_as_solution() const
{
  return error_estimator->get_error_estimator_as_solution();
}

void TimeHandler::set_last_nonlinear_solve_status(const bool flag) const
{
  last_nonlinear_solver_converged = flag;
}

bool TimeHandler::is_timestep_accepted(
  LA::ParVectorType                    &present_solution,
  const std::vector<LA::ParVectorType> &previous_solutions)
{
  using Strategy = Parameters::TimeIntegration::Adaptation::AdaptationStrategy;

  const unsigned int mpi_rank =
    Utilities::MPI::this_mpi_process(present_solution.get_mpi_communicator());

  // Nonlinear solver failed to find a solution.
  if (!last_nonlinear_solver_converged)
  {
    // First check that it threw correctly from there, if it was needed
    AssertThrow(!is_steady(), ExcInternalError());
    AssertThrow(with_adaptive_timestep, ExcInternalError());
    AssertThrow(!is_starting_step(), ExcInternalError());
    AssertThrow(!last_step_was_starting_step(), ExcInternalError());

    // Then reject the step
    if (mpi_rank == 0)
      std::cout << "Rejecting time step because nonlinear solver failed to "
                   "find a solution"
                << std::endl;
    goto reject_step;
  }

  // Always accept steady-state step or unsteady step without adaptation
  if (is_steady() || !with_adaptive_timestep)
    goto accept_step;

  // Also always accept BDF starting steps for now
  if (is_starting_step() || last_step_was_starting_step())
  {
    // Print a warning if the CFL was too high
    if (time_parameters.adaptation.strategy == Strategy::CFL)
    {
      const double cfl_ratio =
        max_cfl_number / time_parameters.adaptation.target_cfl;
      if (cfl_ratio > time_parameters.adaptation.reject_cfl_factor)
        if (mpi_rank == 0)
          std::cout
            << "\nWarning : The CFL number at this time steps exceeds the "
               "ratio for step rejection,\nbut this time step was not rejected "
               "as it is a starting step for the chosen BDF time\nintegration "
               "scheme. Consider reducing the initial time step to limit the "
               "CFL\nnumber during the initial steps.\n"
            << std::endl;
    }
    goto accept_step;
  }

  if (time_parameters.adaptation.strategy == Strategy::BDFTruncationError)
  {
    // FIXME: Clarify this.
    // We need bdf_order + 2 solutions to compute the error estimate,
    // and for methods with starting steps, we also want to start the error
    // estimation with steps which are not the initial condition.
    // Thus, start error estimation at step n_starting_steps + bdf_order + 1.
    const unsigned int n_starting_steps = bdf_order - 1;
    if (current_time_iteration <= bdf_order + n_starting_steps)
      return true;

    // Compute the error estimator for this time step
    error_estimator->compute_error_estimator(*this,
                                             present_solution,
                                             previous_solutions);

    if (time_parameters.adaptation.reject_timestep_with_large_error)
    {
      const auto &max_errors    = error_estimator->get_max_errors();
      const auto &target_errors = time_parameters.adaptation.target_error;

      for (const auto &[variable, error] : max_errors)
      {
        Assert(target_errors.count(variable) > 0, ExcInternalError());
        const double error_ratio = error / target_errors.at(variable);

        if (error_ratio > time_parameters.adaptation.reject_error_factor)
        {
          if (mpi_rank == 0)
            std::cout << "Rejecting step because error ratio for variable \"" +
                           SolverInfo::to_string(variable) + "\" is = "
                      << error_ratio << std::endl;
          goto reject_step;
        }
      }
    }
  }

  if (time_parameters.adaptation.strategy == Strategy::CFL)
  {
    //
    if (time_parameters.adaptation.reject_timestep_with_large_cfl)
    {
      const double cfl_ratio =
        max_cfl_number / time_parameters.adaptation.target_cfl;

      if (cfl_ratio > time_parameters.adaptation.reject_cfl_factor)
      {
        if (mpi_rank == 0)
          std::cout << "Rejecting time step because CFL ratio is too large : "
                    << cfl_ratio << " > "
                    << time_parameters.adaptation.target_cfl << std::endl;
        goto reject_step;
      }
    }
  }

accept_step:
  // Accepting step : reset counter and update time step
  n_consecutive_rejected_steps = 0;
  all_simulation_times.push_back(current_time);
  set_next_timestep(true);
  return true;

reject_step:
  rolledback_step = true;
  n_consecutive_rejected_steps++;
  n_rejected_steps++;

  // Exit with failure if we already rejected too many steps
  const unsigned int max_rejections = 5;
  if (n_consecutive_rejected_steps > max_rejections)
  {
    throw std::runtime_error("Could not find an acceptable time step after " +
                             std::to_string(max_rejections) +
                             " step rejections. Aborting simulation.");
  }

  // Get next time step
  const double timestep_copy = current_dt;
  set_next_timestep(false);

  // Rollback the changes from this time step
  present_solution = previous_solutions[0];
  current_time -= timestep_copy;
  current_time_iteration--;

  if (mpi_rank == 0)
    std::cout << "Trying again with time step = " << current_dt << std::endl;
  return false;
}

unsigned int TimeHandler::get_n_rejected_steps() const
{
  return n_rejected_steps;
}

void TimeHandler::set_max_cfl(const double max_cfl)
{
  max_cfl_number = max_cfl;
}

double clamp_timestep(const double                       current_timestep,
                      const double                       next_timestep,
                      const Parameters::TimeIntegration &time_parameters)
{
  double clamped_timestep = next_timestep;

  // Bound the increase/decrease ratio
  double ratio = next_timestep / current_timestep;
  ratio        = std::min(time_parameters.adaptation.max_timestep_increase,
                   std::max(time_parameters.adaptation.max_timestep_reduction,
                            ratio));
  // Bound by the BDF2 swing factor if needed
  if (time_parameters.scheme == BDF2)
    ratio = std::min(1. + sqrt(2.), ratio);
  clamped_timestep = ratio * current_timestep;

  // Bound the absolute time step
  clamped_timestep = std::min(time_parameters.adaptation.max_timestep,
                              std::max(time_parameters.adaptation.min_timestep,
                                       clamped_timestep));

  return clamped_timestep;
}

void TimeHandler::set_next_timestep(const bool step_was_accepted)
{
  // BDF starting steps are not modified, even when time step adaptation is
  // enabled. Set the next time step depending on the current iteration number.
  if (!with_adaptive_timestep || is_starting_step() ||
      last_step_was_starting_step())
  {
    if (scheme == BDF2 &&
        time_parameters.bdfstart == Parameters::TimeIntegration::BDFStart::BDF1)
    {
      if (is_starting_step())
      {
        // Current step was starting step
        // For next step, apply difference to reach dt
        current_dt =
          initial_dt * (1. - time_parameters.bdf_starting_step_ratio);
      }
      else if (last_step_was_starting_step())
      {
        // Continue simulation with the prescribed constant time step.
        // If time adaptation is enabled and this step is too large, it may
        // be rejected starting at the next time step.
        current_dt = initial_dt;
      }
    }

    // Make sure the last time step is the prescribed end time
    current_dt = std::min(current_dt, final_time - current_time);

    return;
  }

  using Strategy = Parameters::TimeIntegration::Adaptation::AdaptationStrategy;

  double next_timestep = current_dt;

  if (step_was_accepted)
  {
    if (time_parameters.adaptation.strategy == Strategy::BDFTruncationError)
    {
      const unsigned int n_starting_steps = bdf_order - 1;
      if (current_time_iteration <= bdf_order + n_starting_steps)
        return;

      // Error estimator was already computed when checking for whether or not
      // the last time step should be accepted.
      // Simply compute the next time step from the stored errors.
      next_timestep = error_estimator->get_next_timestep(current_dt);
    }
    if (time_parameters.adaptation.strategy == Strategy::CFL)
    {
      next_timestep =
        current_dt * time_parameters.adaptation.target_cfl / max_cfl_number;
    }

    // Make sure the last time step is the prescribed end time
    next_timestep = std::min(next_timestep, final_time - current_time);
  }
  else
  {
    const double reduction_factor = 0.9;

    if (!last_nonlinear_solver_converged)
    {
      // Heuristic for when the nonlinear solver failed
      // Simply halve the last time step
      next_timestep = 0.5 * current_dt;
    }
    else if (time_parameters.adaptation.strategy ==
             Strategy::BDFTruncationError)
    {
      // Reduce the time step based on the error estimate from this rejected
      // step, and apply a slight safety factor to avoid multiple rejections
      // in a row.
      next_timestep =
        reduction_factor * error_estimator->get_next_timestep(current_dt);
    }
    else if (time_parameters.adaptation.strategy == Strategy::CFL)
    {
      // Reduce the time step based on the computed CFL, and apply a slight
      // safety factor to avoid multiple rejections in a row.
      next_timestep = reduction_factor * current_dt *
                      time_parameters.adaptation.target_cfl / max_cfl_number;
    }
    else
    {
      DEAL_II_ASSERT_UNREACHABLE();
    }

    AssertThrow(next_timestep < current_dt, ExcInternalError());
  }

  // Bound progression ratio and absolute time step
  next_timestep = clamp_timestep(current_dt, next_timestep, time_parameters);

  // TODO: Do not account for required times right now, as this comes
  // with a plethora of corner cases to ensure we do not reach tiny time steps
  // if the next required time is just before/after the next predicted time.
  // The tiny time steps should be merged, and if not possible, maybe the
  // accepted time step should be halved, to obtain a time step within bounds
  // at the next step.

  current_dt = next_timestep;
}

double TimeHandler::compute_time_derivative_at_quadrature_node(
  const unsigned int                      quadrature_node_index,
  const double                            present_solution,
  const std::vector<std::vector<double>> &previous_solutions) const
{
  if (scheme == Parameters::TimeIntegration::Scheme::stationary)
    return 0.;
  if (scheme == Parameters::TimeIntegration::Scheme::BDF1 ||
      scheme == Parameters::TimeIntegration::Scheme::BDF2)
  {
    double value_dot = bdf_coefficients[0] * present_solution;
    for (unsigned int i = 1; i < bdf_coefficients.size(); ++i)
      value_dot +=
        bdf_coefficients[i] * previous_solutions[i - 1][quadrature_node_index];
    return value_dot;
  }
  DEAL_II_ASSERT_UNREACHABLE();
}

void TimeHandler::write_timestep_history(std::ostream &out) const
{
  for (unsigned int i = 0; i < all_simulation_times.size(); ++i)
  {
    const double time = all_simulation_times[i];
    if (i > 0)
      out << time << " " << time - all_simulation_times[i - 1] << std::endl;
    else
      out << time << " " << std::endl;
  }
}

void TimeHandler::save() const
{
  // TODO
  DEAL_II_NOT_IMPLEMENTED();
}

void TimeHandler::load()
{
  // TODO
  DEAL_II_NOT_IMPLEMENTED();
}

void TimeHandler::update_parameters_after_restart(
  const Parameters::TimeIntegration &new_parameters)
{
  // 'steady_scheme' was loaded from the checkpoint
  // Save it before overwriting with the new simulation's scheme.
  const bool is_checkpoint_steady = steady_scheme;
  steady_scheme                   = new_parameters.scheme == STAT;

  // Nothing to update if current solver is steady-state
  if (steady_scheme)
    return;

  // Detect a restart from a stationary checkpoint into an unsteady simulation.
  if (is_checkpoint_steady)
  {
    // Overwrite the serialized parameters that were loaded from the stationary
    // checkpoint with their new values.
    initial_time           = new_parameters.t_initial;
    final_time             = new_parameters.t_end;
    current_time           = initial_time;
    current_time_iteration = 0;
    current_dt             = initial_dt;
    with_adaptive_timestep = new_parameters.adaptation.enable;

    // Set the initial time step if using BDF1 as BDF2 starting method
    if (scheme == BDF2 &&
        new_parameters.bdfstart == Parameters::TimeIntegration::BDFStart::BDF1)
    {
      current_dt = initial_dt * new_parameters.bdf_starting_step_ratio;
    }

    simulation_times.assign(n_previous_solutions + 1, initial_time);
    time_steps.assign(n_previous_solutions + 1, initial_dt);

    set_bdf_coefficients();
    return;
  }

  // For now, both the interrupted and restarted simulation should agree
  // on whether adaptive time stepping is used.
  AssertThrow(
    with_adaptive_timestep == new_parameters.adaptation.enable,
    ExcMessage(
      "When restarting a simulation from a checkpoint, both the interrupted "
      "and the restarted simulations should agree on whether adaptive time "
      "stepping is used. The parameters used for this restarted simulation do "
      "not agree with the those from the checkpointed one (one was using "
      "adaptive time stepping and this one does not, or vice versa)."));

  // Update time step and final time, then update BDF coefficients
  final_time = new_parameters.t_end;

  // If time step adaptation is enabled, keep the predicted time step
  // stored in the checkpointed data. Otherwise, use the new dt.
  if (!with_adaptive_timestep)
  {
    current_dt    = new_parameters.dt;
    time_steps[0] = new_parameters.dt;
  }

  set_bdf_coefficients();
}
