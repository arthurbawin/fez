
#include <components_ordering.h>
#include <time_handler.h>
#include <timestep_adaptation.h>

#include <algorithm>

constexpr auto STAT = Parameters::TimeIntegration::Scheme::stationary;
constexpr auto BDF1 = Parameters::TimeIntegration::Scheme::BDF1;
constexpr auto BDF2 = Parameters::TimeIntegration::Scheme::BDF2;

TimeHandler::TimeHandler(const Parameters::TimeIntegration &time_parameters)
  : time_parameters(time_parameters)
  , current_time(time_parameters.t_initial)
  , current_time_iteration(0)
  , initial_time(time_parameters.t_initial)
  , final_time(time_parameters.t_end)
  , initial_dt(time_parameters.dt)
  , current_dt(time_parameters.dt)
  , scheme(time_parameters.scheme)
{
  switch (scheme)
  {
    case STAT:
      n_previous_solutions = 0;
      break;
    case BDF1:
      n_previous_solutions = 1;
      break;
    case BDF2:
      n_previous_solutions = 2;
      break;
  }

  simulation_times.resize(n_previous_solutions + 1, initial_time);
  time_steps.resize(n_previous_solutions + 1, initial_dt);
  bdf_coefficients.resize(n_previous_solutions + 1, 0.);

  if (!is_steady() && time_parameters.adaptation.enable)
    error_estimator =
      std::make_shared<BDFErrorEstimator>(time_parameters, *this);
}

TimeHandler::~TimeHandler() = default;

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

bool TimeHandler::is_steady() const
{
  return scheme == Parameters::TimeIntegration::Scheme::stationary;
}

bool TimeHandler::is_starting_step() const
{
  return current_time_iteration < n_previous_solutions;
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

void TimeHandler::advance(const ConditionalOStream &pcout)
{
  current_time_iteration++;

  if (scheme != STAT)
  {
    // Rotate the times and time steps
    for (unsigned int i = n_previous_solutions; i >= 1; --i)
    {
      simulation_times[i] = simulation_times[i - 1];
      time_steps[i]       = time_steps[i - 1];
    }

    if (scheme == BDF1)
    {
      // Self-starting: update values and coefficients and proceed
      current_time += current_dt;
      simulation_times[0] = current_time;
      time_steps[0]       = current_dt;
      set_bdf_coefficients();
    }

    if (scheme == BDF2)
    {
      if (time_parameters.bdfstart ==
          Parameters::TimeIntegration::BDFStart::BDF1)
      {
        // Start with BDF1
        const double starting_step_ratio = 0.1;

        if (this->is_starting_step())
        {
          current_dt = initial_dt * starting_step_ratio;
          current_time += current_dt;
          simulation_times[0] = current_time;
          time_steps[0]       = current_dt;

          // Force scheme to BDF1
          set_bdf_coefficients(true);
        }
        else if (current_time_iteration - 1 < n_previous_solutions)
        {
          // Previous step was starting step
          current_dt = initial_dt * (1. - starting_step_ratio);
          current_time += current_dt;
          simulation_times[0] = current_time;
          time_steps[0]       = current_dt;
          set_bdf_coefficients();
          current_dt = initial_dt;
        }
        else
        {
          // Continue with regular time step
          current_time += current_dt;
          simulation_times[0] = current_time;
          time_steps[0]       = current_dt;
          set_bdf_coefficients();
        }
      }
      else
      {
        current_time += current_dt;
        simulation_times[0] = current_time;
        time_steps[0]       = current_dt;
        set_bdf_coefficients();
      }
    }

    if (time_parameters.verbosity == Parameters::Verbosity::verbose)
    {
      pcout << std::endl
            << "Time step " << current_time_iteration
            << " - Advancing to t = " << current_time;
      if (time_parameters.adaptation.enable &&
          time_parameters.adaptation.verbosity ==
            Parameters::Verbosity::verbose)
        pcout << " with time step " << current_dt;
      pcout << '.' << std::endl;
    }
  }

  // Advance the time tables in the error estimator
  if (!is_steady() && time_parameters.adaptation.enable)
    error_estimator->advance(*this);
}

void TimeHandler::rotate_solutions(
  const LA::ParVectorType        &present_solution,
  std::vector<LA::ParVectorType> &previous_solutions) const
{
  if (!this->is_steady())
  {
    // Rotate the additional solution vector in the error estimator
    // Do this first, before overwriting the last solution
    if (!is_steady() && time_parameters.adaptation.enable)
      error_estimator->rotate_additional_solution(previous_solutions.back());

    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
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

void TimeHandler::set_next_timestep(
  const ComponentOrdering              &ordering,
  const LA::ParVectorType              &present_solution,
  const std::vector<LA::ParVectorType> &previous_solutions,
  const IndexSet                       &locally_relevant_dofs,
  const std::vector<unsigned char>     &dofs_to_component)
{
  const auto &adapt = time_parameters.adaptation;

  if (is_steady() || !adapt.enable)
    // Steady, or unsteady but without adaptation : nothing to do
    return;

  // Determine the next time step based on the error estimator
  const double target_time_step =
    error_estimator->compute_next_timestep_from_error_estimator(
      *this,
      ordering,
      present_solution,
      previous_solutions,
      locally_relevant_dofs,
      dofs_to_component);

  // Bound progression ratio and absolute time step
  double next_timestep =
    clamp_timestep(current_dt, target_time_step, time_parameters);

  // TODO: Do not account for required times right now, as this comes
  // with a plethora of corner cases to ensure we do not reach tiny time steps
  // if the next required time is just before/after the next predicted time.
  // The tiny time steps should be merged, and if not possible, maybe the
  // accepted time step should be halved, to obtain a time step within bounds
  // at the next step.

  // auto next_required = std::upper_bound(adapt.required_times.begin(),
  //                                       adapt.required_times.end(),
  //                                       current_time);
  // if (next_required != adapt.required_times.end())
  // {
  //   if (*next_required < current_time + next_timestep)
  //   {
  //     double timestep_to_required = *next_required - current_time;
  //   }
  //   else
  //   {
  //     // Next required time is after the next predicted time.
  //   }
  // }

  // FIXME: Same considerations when limiting the time step to reach t_end
  // Should maybe merge a small final time step, if possible.
  next_timestep = std::min(next_timestep, final_time - current_time);

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
  if (scheme == STAT)
    return;

  // Update time step and final time, then update BDF coefficients
  final_time    = new_parameters.t_end;
  current_dt    = new_parameters.dt;
  time_steps[0] = new_parameters.dt;
  set_bdf_coefficients();
}
