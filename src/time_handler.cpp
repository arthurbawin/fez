
#include <time_handler.h>
#include <algorithm>
#include <cmath>
#include <limits>


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

  // Build programmed dt schedule for MMS verification if requested.
  build_programmed_dt_schedule();
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

  if (scheme == STAT)
    return;

  // Programmed dt schedule (paper-style tests)
  if (use_programmed_dt_schedule && pending_dt_queue.empty())
  {
    const unsigned int idx = current_time_iteration - 1;
    AssertThrow(idx < programmed_dt.size(), ExcMessage("Programmed dt schedule exhausted."));
    current_dt = programmed_dt[idx];
  }


  // ------------------------------------------------------------
  // 0) Si une rampe de dt est en cours (restart), on force current_dt
  //    pour CE step, puis on consomme la queue.
  // ------------------------------------------------------------
  if (!pending_dt_queue.empty())
  {
    current_dt = pending_dt_queue.front();
    pending_dt_queue.pop_front();
  }

  // ------------------------------------------------------------
  // 1) Rotate the times and time steps
  //    IMPORTANTE: on doit décaler jusqu'à i = n_previous_solutions,
  //    sinon en BDF2, time_steps[2] n'est jamais mis à jour.
  // ------------------------------------------------------------
  for (unsigned int i = n_previous_solutions; i > 0; --i)
  {
    previous_times[i] = previous_times[i - 1];
    time_steps[i]     = time_steps[i - 1];
  }

  // ------------------------------------------------------------
  // 2) Mise à jour temps + dt + coeff BDF
  // ------------------------------------------------------------
  if (scheme == BDF1)
  {
    current_time += current_dt;
    previous_times[0] = current_time;
    time_steps[0]     = current_dt;
    set_bdf_coefficients();
  }
  else if (scheme == BDF2)
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

namespace
{
  inline void get_ratio_bounds_for_next_step(const Parameters::TimeIntegration::Scheme scheme,
                                             const bool                                next_step_is_bdf2,
                                             double                                   &ratio_min,
                                             double                                   &ratio_max)
  {
    (void) scheme;
    if (next_step_is_bdf2)
    {
      ratio_min = 0.2;
      ratio_max = 1.0 + std::sqrt(2.0);
    }
    else
    {
      ratio_min = 0.1;
      ratio_max = 5.0;
    }
  }

  inline unsigned int compute_linear_ramp_N_increase(const double r_target,
                                                     const double ratio_max)
  {
    // r_target = dt_new/dt_old > ratio_max
    // For linear ramp, max ratio between steps occurs at first step:
    // dt1/dt0 = 1 + (r_target - 1)/N <= ratio_max
    // => N >= (r_target - 1)/(ratio_max - 1)
    const double denom = (ratio_max - 1.0);
    if (denom <= 0.0)
      return 1;

    const double Nmin = (r_target - 1.0) / denom;
    return std::max(1u, static_cast<unsigned int>(std::ceil(Nmin)));
  }

  inline unsigned int compute_linear_ramp_N_decrease(const double r_target,
                                                     const double ratio_min)
  {
    // r_target = dt_new/dt_old < ratio_min
    // For linear decreasing ramp, the MIN ratio between steps occurs at LAST step:
    // dtN/dt(N-1) = 1 / (1 + (dt_old/dt_new - 1)/N) >= ratio_min
    // => N >= (dt_old/dt_new - 1)/(1/ratio_min - 1)
    if (ratio_min <= 0.0 || ratio_min >= 1.0)
      return 1;

    const double inv = (1.0 / ratio_min - 1.0);
    if (inv <= 0.0)
      return 1;

    const double r_inv = 1.0 / r_target; // dt_old/dt_new > 1
    const double Nmin  = (r_inv - 1.0) / inv;
    return std::max(1u, static_cast<unsigned int>(std::ceil(Nmin)));
  }

  inline void build_linear_ramp(std::deque<double> &queue,
                                const double        dt_old,
                                const double        dt_new,
                                const unsigned int  N)
  {
    queue.clear();
    if (N == 0)
      return;

    // dt_k = dt_old + k*(dt_new - dt_old)/N , k=1..N
    for (unsigned int k = 1; k <= N; ++k)
    {
      const double alpha = static_cast<double>(k) / static_cast<double>(N);
      const double dt_k  = dt_old + alpha * (dt_new - dt_old);
      queue.push_back(dt_k);
    }
  }
} // namespace


void TimeHandler::build_programmed_dt_schedule()
{
  use_programmed_dt_schedule = false;
  programmed_dt.clear();

  if (time_parameters.is_steady())
    return;

  // Only used when adaptative_dt is enabled but dt_control_mode != vautrin.
  if (!time_parameters.adaptative_dt)
    return;

  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::vautrin)
    return;

  const double T = final_time - initial_time;
  if (T <= 0.0)
    return;

  // We infer the intended number of steps from the "reference" dt
  // of the current run (this dt is refined by the MMS driver).
  const double dt_ref = initial_dt;
  const double N_real = T / dt_ref;
  const unsigned int N = static_cast<unsigned int>(std::llround(N_real));

  AssertThrow(N >= 2,
             ExcMessage("Programmed dt schedule requires at least 2 time steps."));

  // Ensure that dt_ref yields an (almost) integer number of steps.
  AssertThrow(std::abs(N_real - static_cast<double>(N)) < 1e-6,
             ExcMessage("For programmed dt schedules, dt must yield an integer number of steps "
                        "over [t_initial,t_end]."));

  programmed_dt.resize(N, 0.0);

  const auto clamp_dt = [&](const double dt) -> double
  {
    return std::min(std::max(dt, time_parameters.dt_min), time_parameters.dt_max);
  };

  // 0-stability safety bound often used for variable-step BDF2.
  const double ratio_limit =
    1.0 + std::sqrt(2.0) - std::max(0.0, time_parameters.dt_ratio_margin);

  // Helper: geometric schedule on [0, T] with N steps, given gamma.
  const auto build_geometric = [&](const bool increasing)
  {
    const double gamma = time_parameters.dt_schedule_gamma;

    // r = gamma^(1/(N-1)) and safety clamp.
    double r = std::pow(gamma, 1.0 / static_cast<double>(N - 1));
    if (r > ratio_limit)
      r = ratio_limit;

    // dt0 such that sum_{i=0}^{N-1} dt0 * r^i = T.
    const double rN = std::pow(r, static_cast<double>(N));
    const double dt0 =
      (std::abs(r - 1.0) < 1e-14 ? T / static_cast<double>(N)
                                 : T * (r - 1.0) / (rN - 1.0));

    for (unsigned int i = 0; i < N; ++i)
    {
      const unsigned int j = increasing ? i : (N - 1 - i);
      programmed_dt[i] = dt0 * std::pow(r, static_cast<double>(j));
    }
  };

  // ---------- increasing / decreasing ----------
  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::increasing)
  {
    build_geometric(true);
  }
  else if (time_parameters.dt_control_mode ==
           Parameters::TimeIntegration::DtControlMode::decreasing)
  {
    build_geometric(false);
  }

  // ---------- increasing then decreasing ----------
  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::inc_dec)
  {
    AssertThrow(N % 2 == 0,
               ExcMessage("inc_dec programmed schedule requires an even N."));
    const unsigned int Nh = N / 2;
    const double Th = 0.5 * T;
    const double gamma = time_parameters.dt_schedule_gamma;

    double r = std::pow(gamma, 1.0 / static_cast<double>(Nh - 1));
    if (r > ratio_limit)
      r = ratio_limit;

    const double rNh = std::pow(r, static_cast<double>(Nh));
    const double dt0h =
      (std::abs(r - 1.0) < 1e-14 ? Th / static_cast<double>(Nh)
                                 : Th * (r - 1.0) / (rNh - 1.0));

    // First half increasing.
    for (unsigned int i = 0; i < Nh; ++i)
      programmed_dt[i] = dt0h * std::pow(r, static_cast<double>(i));

    // Second half decreasing (mirror).
    for (unsigned int i = 0; i < Nh; ++i)
      programmed_dt[Nh + i] = programmed_dt[Nh - 1 - i];
  }

  // ---------- alternating ----------
  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::alternating)
  {
    AssertThrow(N % 2 == 0,
               ExcMessage("alternating programmed schedule requires an even N."));
    const double q = time_parameters.dt_alternating_ratio;

    // Pattern: dt_big, dt_small, dt_big, dt_small, ...
    // with dt_big/dt_small = q and sum = T.
    const double dt_small = T / (static_cast<double>(N) * (0.5 * (q + 1.0)));
    const double dt_big   = q * dt_small;

    for (unsigned int i = 0; i < N; ++i)
      programmed_dt[i] = (i % 2 == 0 ? dt_big : dt_small);
  }

  // Final safety: clamp and enforce exact sum by adjusting the last step.
  double sum = 0.0;
  for (unsigned int i = 0; i < N; ++i)
  {
    programmed_dt[i] = clamp_dt(programmed_dt[i]);
    sum += programmed_dt[i];
  }

  // Adjust last dt to hit final time exactly.
  programmed_dt[N - 1] += (T - sum);

  programmed_dt[N - 1] = clamp_dt(programmed_dt[N - 1]);

  use_programmed_dt_schedule = true;

  // Initialize current_dt for the first step.
  current_dt = programmed_dt[0];
}

void TimeHandler::apply_restart_overrides(
  const Parameters::TimeIntegration &new_params)
{
  // Toujours : on peut changer t_end au restart
  final_time = new_params.t_end;

  if (scheme == STAT)
    return;

  // Si l'utilisateur ne fournit pas un dt > 0, on ne change rien
  if (!(new_params.dt > 0.0))
    return;

  const double dt_old = current_dt;
  const double dt_new = new_params.dt;

  // Met à jour les paramètres internes (important si tu veux "dt imposé")
  time_parameters.dt = dt_new;
  initial_dt         = dt_new;

  // Nettoie toute ancienne rampe
  pending_dt_queue.clear();

  // Si dt_old est foireux, on force direct
  if (!(dt_old > 0.0))
  {
    current_dt = dt_new;
    return;
  }

  // Est-ce que le prochain step utilise "vraiment" BDF2 ?
  // (si on est trop tôt dans l'historique, on retombe sur BDF1)
  const bool next_step_is_bdf2 =
    (scheme == BDF2) && (current_time_iteration >= n_previous_solutions);

  double ratio_min = 0.0, ratio_max = 0.0;
  get_ratio_bounds_for_next_step(scheme, next_step_is_bdf2, ratio_min, ratio_max);

  const double r = dt_new / dt_old;

  // Cas OK : changement pas trop violent => jump direct
  if (r >= ratio_min && r <= ratio_max)
  {
    current_dt = dt_new;
    return;
  }

  // Cas violent : on construit une rampe LINÉAIRE en respectant les ratios
  unsigned int N = 1;
  if (r > ratio_max)
    N = compute_linear_ramp_N_increase(r, ratio_max);
  else if (r < ratio_min)
    N = compute_linear_ramp_N_decrease(r, ratio_min);

  build_linear_ramp(pending_dt_queue, dt_old, dt_new, N);

  // On utilise le premier dt de la rampe pour le prochain step
  if (!pending_dt_queue.empty())
  {
    current_dt = pending_dt_queue.front();
    pending_dt_queue.pop_front();
  }
  else
  {
    current_dt = dt_new;
  }
}



void TimeHandler::load()
{
  // TODO
  DEAL_II_NOT_IMPLEMENTED();
}