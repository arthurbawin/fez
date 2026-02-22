#include <time_handler.h>
#include <components_ordering.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
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
  , dt_ref_bounds(time_parameters.dt)
  , scheme(time_parameters.scheme)
  , mms_dt_reference(time_parameters.dt) // <-- capture dt_ref ONCE here
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
    return current_time_iteration > 0;

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
    AssertThrow(idx < programmed_dt.size(),
                ExcMessage("Programmed dt schedule exhausted."));
    current_dt = programmed_dt[idx];
  }

  // 0) Linear ramp dt (restart)
  if (!pending_dt_queue.empty())
  {
    current_dt = pending_dt_queue.front();
    pending_dt_queue.pop_front();
  }

  // 1) Rotate times and time steps (shift up to n_previous_solutions)
  for (unsigned int i = n_previous_solutions; i > 0; --i)
  {
    previous_times[i] = previous_times[i - 1];
    time_steps[i]     = time_steps[i - 1];
  }

  // 2) Update time + dt + BDF coefficients
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
      const double starting_step_ratio = 0.1;

      if (this->is_starting_step())
      {
        current_dt = initial_dt * starting_step_ratio;

        current_time += current_dt;
        previous_times[0] = current_time;
        time_steps[0]     = current_dt;

        set_bdf_coefficients(true); // Force BDF1
      }
      else if (current_time_iteration - 1 < n_previous_solutions)
      {
        current_dt = initial_dt * (1. - starting_step_ratio);

        current_time += current_dt;
        previous_times[0] = current_time;
        time_steps[0]     = current_dt;

        set_bdf_coefficients();
        current_dt = initial_dt;
      }
      else
      {
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
          << " - Advancing to t = "
          << std::fixed << std::setprecision(10) << current_time
          << " with dt = " << std::fixed << std::setprecision(10) << current_dt
          << '.' << std::endl;
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
      value_dot += bdf_coefficients[i] * previous_solutions[i - 1][quadrature_node_index];
    return value_dot;
  }

  DEAL_II_ASSERT_UNREACHABLE();
}

void TimeHandler::save() const
{
  DEAL_II_NOT_IMPLEMENTED();
}

namespace
{
  inline void get_ratio_bounds_for_next_step(const Parameters::TimeIntegration::Scheme /*scheme*/,
                                             const bool                                next_step_is_bdf2,
                                             double                                   &ratio_min,
                                             double                                   &ratio_max)
  {
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
    const double denom = (ratio_max - 1.0);
    if (denom <= 0.0)
      return 1;

    const double Nmin = (r_target - 1.0) / denom;
    return std::max(1u, static_cast<unsigned int>(std::ceil(Nmin)));
  }

  inline unsigned int compute_linear_ramp_N_decrease(const double r_target,
                                                     const double ratio_min)
  {
    if (ratio_min <= 0.0 || ratio_min >= 1.0)
      return 1;

    const double inv = (1.0 / ratio_min - 1.0);
    if (inv <= 0.0)
      return 1;

    const double r_inv = 1.0 / r_target;
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

  if (!time_parameters.adaptative_dt)
    return;

  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::vautrin)
    return;

  const double T = final_time - initial_time;
  if (T <= 0.0)
    return;

  const double dt_ref = initial_dt;
  const double N_real = T / dt_ref;
  const unsigned int N = static_cast<unsigned int>(std::llround(N_real));

  AssertThrow(N >= 2,
             ExcMessage("Programmed dt schedule requires at least 2 time steps."));

  AssertThrow(std::abs(N_real - static_cast<double>(N)) < 1e-6,
             ExcMessage("For programmed dt schedules, dt must yield an integer number of steps "
                        "over [t_initial,t_end]."));

  programmed_dt.resize(N, 0.0);

  const double dt_min_eff = get_effective_dt_min();
  const double dt_max_eff = get_effective_dt_max();

  const auto clamp_dt = [&](const double dt) -> double
  {
    return std::min(std::max(dt, dt_min_eff), dt_max_eff);
  };

  const double ratio_limit =
    1.0 + std::sqrt(2.0) - std::max(0.0, time_parameters.dt_ratio_margin);

  const auto build_geometric = [&](const bool increasing)
  {
    const double gamma = time_parameters.dt_schedule_gamma;

    double r = std::pow(gamma, 1.0 / static_cast<double>(N - 1));
    if (r > ratio_limit)
      r = ratio_limit;

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

    for (unsigned int i = 0; i < Nh; ++i)
      programmed_dt[i] = dt0h * std::pow(r, static_cast<double>(i));

    for (unsigned int i = 0; i < Nh; ++i)
      programmed_dt[Nh + i] = programmed_dt[Nh - 1 - i];
  }

  if (time_parameters.dt_control_mode ==
      Parameters::TimeIntegration::DtControlMode::alternating)
  {
    AssertThrow(N % 2 == 0,
               ExcMessage("alternating programmed schedule requires an even N."));
    const double q = time_parameters.dt_alternating_ratio;

    const double dt_small = T / (static_cast<double>(N) * (0.5 * (q + 1.0)));
    const double dt_big   = q * dt_small;

    for (unsigned int i = 0; i < N; ++i)
      programmed_dt[i] = (i % 2 == 0 ? dt_big : dt_small);
  }

  double sum = 0.0;
  for (unsigned int i = 0; i < N; ++i)
  {
    programmed_dt[i] = clamp_dt(programmed_dt[i]);
    sum += programmed_dt[i];
  }

  programmed_dt[N - 1] += (T - sum);
  programmed_dt[N - 1] = clamp_dt(programmed_dt[N - 1]);

  use_programmed_dt_schedule = true;
  current_dt = programmed_dt[0];
}

void TimeHandler::apply_restart_overrides(const Parameters::TimeIntegration &new_params)
{
  final_time = new_params.t_end;

  if (scheme == STAT)
    return;

  if (!(new_params.dt > 0.0))
    return;

  const double dt_old = current_dt;
  const double dt_new = new_params.dt;

  time_parameters.dt = dt_new;
  initial_dt         = dt_new;
  dt_ref_bounds      = dt_new;

  // IMPORTANT: on ne touche PAS mms_dt_reference ici.
  // C'est le dt du 1er run MMS, ça doit rester constant.

  pending_dt_queue.clear();

  if (!(dt_old > 0.0))
  {
    current_dt = dt_new;
    return;
  }

  const bool next_step_is_bdf2 =
    (scheme == BDF2) && (current_time_iteration >= n_previous_solutions);

  double ratio_min = 0.0, ratio_max = 0.0;
  get_ratio_bounds_for_next_step(scheme, next_step_is_bdf2, ratio_min, ratio_max);

  const double r = dt_new / dt_old;

  if (r >= ratio_min && r <= ratio_max)
  {
    current_dt = dt_new;
    return;
  }

  unsigned int N = 1;
  if (r > ratio_max)
    N = compute_linear_ramp_N_increase(r, ratio_max);
  else if (r < ratio_min)
    N = compute_linear_ramp_N_decrease(r, ratio_min);

  build_linear_ramp(pending_dt_queue, dt_old, dt_new, N);

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
  DEAL_II_NOT_IMPLEMENTED();
}

// =========================
// Vautrin adaptive time-step control
// =========================

void TimeHandler::compute_vautrin_error_estimate(
  LA::ParVectorType                    &e_star,
  const LA::ParVectorType              &u_np1,
  const std::vector<LA::ParVectorType> &previous_solutions,
  const unsigned int                    order) const
{
  AssertThrow(!this->is_steady(), ExcMessage("No time error estimate in steady."));
  AssertThrow(order == 1 || order == 2, ExcMessage("Only order 1 or 2 supported."));
  AssertThrow(previous_solutions.size() >= order + 1,
              ExcMessage("Not enough history in previous_solutions. Need order+1 vectors."));
  AssertThrow(bdf_coefficients.size() >= order + 1,
              ExcMessage("bdf_coefficients size inconsistent with requested order."));
  AssertThrow(time_steps.size() >= 1,
              ExcMessage("Need at least one stored time step (h0)."));

  const double h0 = time_steps[0];
  const double h1 = (time_steps.size() > 1 ? time_steps[1] : h0);
  const double h2 = (time_steps.size() > 2 ? time_steps[2] : h1);

  const unsigned int p = order;
  const unsigned int k = p + 1;

  const double factorial = (k == 2 ? 2.0 : 6.0);
  const double sign = (k % 2 == 0 ? +1.0 : -1.0);

  double sum = 0.0;
  sum += bdf_coefficients[1] * std::pow(h0, static_cast<int>(k));
  if (p == 2)
    sum += bdf_coefficients[2] * std::pow(h0 + h1, static_cast<int>(k));

  const double C_p1 = sign * (sum / factorial);

  LA::ParVectorType u_p1;
  u_p1.reinit(u_np1);
  u_p1 = 0.0;

  if (p == 1)
  {
    LA::ParVectorType dd10;
    dd10.reinit(u_np1);
    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    LA::ParVectorType dd11;
    dd11.reinit(u_np1);
    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    LA::ParVectorType dd2;
    dd2.reinit(u_np1);
    dd2 = dd10;
    dd2.add(-1.0, dd11);
    dd2 *= (1.0 / (h0 + h1));

    u_p1 = dd2;
    u_p1 *= 2.0; // 2!
  }
  else
  {
    AssertThrow(previous_solutions.size() >= 3,
                ExcMessage("Need u^{n-2} in previous_solutions[2] for BDF2 error estimate."));
    AssertThrow(time_steps.size() >= 3,
                ExcMessage("Need 3 stored time steps (h0,h1,h2) for BDF2 error estimate."));

    LA::ParVectorType dd10;
    dd10.reinit(u_np1);
    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    LA::ParVectorType dd11;
    dd11.reinit(u_np1);
    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    LA::ParVectorType dd12;
    dd12.reinit(u_np1);
    dd12 = previous_solutions[1];
    dd12.add(-1.0, previous_solutions[2]);
    dd12 *= (1.0 / h2);

    LA::ParVectorType dd20;
    dd20.reinit(u_np1);
    dd20 = dd10;
    dd20.add(-1.0, dd11);
    dd20 *= (1.0 / (h0 + h1));

    LA::ParVectorType dd21;
    dd21.reinit(u_np1);
    dd21 = dd11;
    dd21.add(-1.0, dd12);
    dd21 *= (1.0 / (h1 + h2));

    LA::ParVectorType dd3;
    dd3.reinit(u_np1);
    dd3 = dd20;
    dd3.add(-1.0, dd21);
    dd3 *= (1.0 / (h0 + h1 + h2));

    u_p1 = dd3;
    u_p1 *= 6.0; // 3!
  }

  e_star = u_p1;
  e_star *= (C_p1 * std::pow(h0, static_cast<int>(p + 1)));
}

double TimeHandler::compute_scaled_vautrin_ratio(
  const LA::ParVectorType          &e_star,
  const LA::ParVectorType          &u_star,
  const std::vector<unsigned char> &dofs_to_component,
  const unsigned int                component_q,
  const double                      epsilon_q,
  const double                      u_seuil,
  const MPI_Comm                    comm) const
{
  double local_sum_e2 = 0.0;
  double local_sum_u2 = 0.0;
  unsigned long long local_neq = 0;

  for (const auto gi : u_star.locally_owned_elements())
  {
    if (dofs_to_component[gi] != component_q)
      continue;

    const double e = e_star[gi];
    const double u = u_star[gi];

    local_sum_e2 += e * e;
    local_sum_u2 += u * u;
    local_neq++;
  }

  const double global_sum_e2 = dealii::Utilities::MPI::sum(local_sum_e2, comm);
  const double global_sum_u2 = dealii::Utilities::MPI::sum(local_sum_u2, comm);
  const unsigned long long global_neq = dealii::Utilities::MPI::sum(local_neq, comm);

  if (global_neq == 0)
    return std::numeric_limits<double>::max();

  const double e_abs  = std::sqrt(global_sum_e2 / double(global_neq));
  const double u_norm = std::sqrt(global_sum_u2 / double(global_neq));

  const double err = (u_norm < u_seuil) ? e_abs : (e_abs / u_norm);

  if (err == 0.0)
    return std::numeric_limits<double>::max();

  return epsilon_q / err;
}

double TimeHandler::propose_next_dt_vautrin(const double       dt,
                                            const double       R,
                                            const unsigned int order,
                                            const double       safety,
                                            const double       ratio_min,
                                            const double       ratio_max,
                                            const double       dt_min,
                                            const double       dt_max)
{
  double factor = ratio_max;

  if (R > 0.0)
  {
    factor = safety * std::pow(R, 1.0 / double(order + 1));
    factor = std::min(ratio_max, std::max(ratio_min, factor));
  }

  double dt_new = dt * factor;
  dt_new        = std::min(dt_max, std::max(dt_min, dt_new));
  return dt_new;
}

double TimeHandler::get_effective_dt_min() const
{
  const double f = time_parameters.dt_min_factor;

  if (!(dt_ref_bounds > 0.0) || !(f > 0.0))
    return 0.0;

  return dt_ref_bounds * f;
}

double TimeHandler::get_effective_dt_max() const
{
  const double f = time_parameters.dt_max_factor;

  if (!(dt_ref_bounds > 0.0) || !(f > 0.0))
    return std::numeric_limits<double>::infinity();

  return dt_ref_bounds * f;
}

bool TimeHandler::update_dt_after_converged_step_vautrin(
  const LA::ParVectorType                            &u_np1,
  const std::vector<LA::ParVectorType>               &previous_solutions_dt_control,
  const std::vector<unsigned char>                   &dofs_to_component,
  const std::vector<std::pair<unsigned int, double>> &component_eps,
  const double                                        u_seuil,
  const double                                        safety,
  const double                                        /*dt_min_factor*/,
  const double                                        /*dt_max_factor*/,
  const MPI_Comm                                      comm,
  const double                                        t_end,
  const bool                                          clamp_to_t_end,
  double                                             *out_R,
  unsigned int                                       *out_order,
  double                                             *out_dt_used,
  double                                             *out_dt_next,
  LA::ParVectorType                                  *out_e_star)
{
  if (this->is_steady())
    return false;

  if (!time_parameters.adaptative_dt)
    return false;

  if (time_parameters.dt_control_mode !=
      Parameters::TimeIntegration::DtControlMode::vautrin)
    return false;

  unsigned int order = 1;
  if (scheme == Parameters::TimeIntegration::Scheme::BDF2 &&
      !this->is_starting_step())
    order = 2;

  if (previous_solutions_dt_control.size() < order + 1)
    return false;

  LA::ParVectorType e_star;
  e_star.reinit(u_np1);

  this->compute_vautrin_error_estimate(e_star,
                                       u_np1,
                                       previous_solutions_dt_control,
                                       order);

  // Verbose: print max relative e* per component
  if (time_parameters.verbosity == Parameters::Verbosity::verbose)
  {
    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);

    for (const auto &ce : component_eps)
    {
      const unsigned int comp = ce.first;

      double local_max_rel = 0.0;
      unsigned long long local_count = 0;

      for (const auto gi : u_np1.locally_owned_elements())
      {
        if (dofs_to_component[gi] != comp)
          continue;

        const double e = e_star[gi];
        const double u = u_np1[gi];

        const double denom = std::max(std::abs(u), u_seuil);
        const double rel   = std::abs(e) / denom;

        local_max_rel = std::max(local_max_rel, rel);
        local_count++;
      }

      const double global_max_rel = dealii::Utilities::MPI::max(local_max_rel, comm);
      const unsigned long long global_count = dealii::Utilities::MPI::sum(local_count, comm);

      if (my_rank == 0)
      {
        std::cout
          << "[Vautrin] comp=" << comp
          << "  e*_rel_max=" << std::scientific << std::setprecision(6)
          << global_max_rel
          << "  (ndofs=" << std::dec << global_count << ")"
          << std::endl;
      }
    }
  }

  // R = min_q (epsilon_q / err_q)
  double R = std::numeric_limits<double>::infinity();
  for (const auto &ce : component_eps)
  {
    const unsigned int comp = ce.first;
    const double eps        = ce.second;

    R = std::min(R,
                 this->compute_scaled_vautrin_ratio(e_star,
                                                    u_np1,
                                                    dofs_to_component,
                                                    comp,
                                                    eps,
                                                    u_seuil,
                                                    comm));
  }

  if (!std::isfinite(R) || R <= 0.0)
    return false;

  const double ratio_min = (order == 2 ? 0.2 : 0.1);
  const double ratio_max = (order == 2 ? (1.0 + std::sqrt(2.0)) : 5.0);

  const double dt_used = current_dt;

  const double dt_min = this->get_effective_dt_min();
  const double dt_max = this->get_effective_dt_max();

  double dt_next =
    TimeHandler::propose_next_dt_vautrin(dt_used, R, order, safety,
                                         ratio_min, ratio_max, dt_min, dt_max);

  if (clamp_to_t_end && current_time < t_end)
  {
    const double remaining = t_end - current_time;
    if (dt_next > remaining)
      dt_next = remaining;
  }

  current_dt = dt_next;

  if (out_R)       *out_R       = R;
  if (out_order)   *out_order   = order;
  if (out_dt_used) *out_dt_used = dt_used;
  if (out_dt_next) *out_dt_next = dt_next;
  if (out_e_star)  *out_e_star  = e_star;

  return true;
}
