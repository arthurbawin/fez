
#include <time_handler.h>

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

  previous_times.resize(n_previous_solutions + 1, initial_time);
  time_steps.resize(n_previous_solutions + 1, initial_dt);
  bdf_coefficients.resize(n_previous_solutions + 1, 0.);
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
    for (unsigned int i = n_previous_solutions - 1; i >= 1; --i)
    {
      previous_times[i] = previous_times[i - 1];
      time_steps[i]     = time_steps[i - 1];
    }

    if (scheme == BDF1)
    {
      // Self-starting: update values and coefficients and proceed
      current_time += current_dt;
      previous_times[0] = current_time;
      time_steps[0]     = current_dt;
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
          previous_times[0] = current_time;
          time_steps[0]     = current_dt;

          // Force scheme to BDF1
          set_bdf_coefficients(true);
        }
        else if (current_time_iteration - 1 < n_previous_solutions)
        {
          // Previous step was starting step
          current_dt = initial_dt * (1. - starting_step_ratio);
          current_time += current_dt;
          previous_times[0] = current_time;
          time_steps[0]     = current_dt;
          set_bdf_coefficients();
          current_dt = initial_dt;
        }
        else
        {
          // Continue with regular time step
          current_time += current_dt;
          previous_times[0] = current_time;
          time_steps[0]     = current_dt;
          set_bdf_coefficients();
        }
      }
      else
      {
        current_time += current_dt;
        previous_times[0] = current_time;
        time_steps[0]     = current_dt;
        set_bdf_coefficients();
      }
    }

    if (time_parameters.verbosity == Parameters::Verbosity::verbose)
      pcout << std::endl
            << "Time step " << current_time_iteration
            << " - Advancing to t = " << current_time << '.' << std::endl;
  }
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

void TimeHandler::apply_restart_overrides(
  const Parameters::TimeIntegration &new_params)
{

  final_time = new_params.t_end;

  if (scheme == STAT)
    return;

  // 2) dt : règle d’override
  // -> il te faut une convention côté paramètres.
  // Par ex: si new_params.dt > 0 et que tu veux qu'il override toujours,
  // tu l’appliques. (Ou ajoute un flag dédié, cf. section 4.)
  const double new_dt = new_params.dt;

  // Si tu ne veux PAS override dt automatiquement, remplace par un if(flag)
  if (new_dt > 0.)
  {
    // BDF1: simple
    if (scheme == BDF1)
    {
      current_dt   = new_dt;
      time_steps[0]= new_dt;
      set_bdf_coefficients();
    }
    // BDF2: pas variable, on garde prev_dt (= time_steps[1]) et on change dt
    else if (scheme == BDF2)
    {
      current_dt = new_dt;

      // IMPORTANT: time_steps size = n_previous_solutions+1 = 3 en BDF2
      // time_steps[0] = dt prochain step
      // time_steps[1] = dt précédent (on le garde)
      time_steps[0] = new_dt;

      set_bdf_coefficients();
    }
  }
}


void TimeHandler::load()
{
  // TODO
  DEAL_II_NOT_IMPLEMENTED();
}