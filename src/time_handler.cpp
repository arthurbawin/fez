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
  // ------------------------------------------------------------
  // Backup state *before* we advance, so a rejected step can be
  // retried with a smaller dt on the SAME (t_n -> t_{n+1}) interval.
  // ------------------------------------------------------------
  if (!this->is_steady())
  {
    has_step_backup               = true;
    backup_current_time           = current_time;
    backup_current_time_iteration = current_time_iteration;
    backup_current_dt             = current_dt;
    backup_previous_times         = previous_times;
    backup_time_steps             = time_steps;
    backup_bdf_coefficients       = bdf_coefficients;
    backup_pending_dt_queue       = pending_dt_queue;
  }

  // Increase the step counter (this step is "attempted" now)
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
              ExcMessage("Need at least order+1 previous solutions."));
  AssertThrow(bdf_coefficients.size() >= order + 1,
              ExcMessage("bdf_coefficients inconsistent with order."));
  AssertThrow(time_steps.size() >= order,
              ExcMessage("Need enough stored time steps."));

  const double h0 = time_steps[0];
  const double h1 = (time_steps.size() > 1 ? time_steps[1] : h0);
  const double h2 = (time_steps.size() > 2 ? time_steps[2] : h1);

  const unsigned int p = order;

  // Build divided difference of order (p+1), i.e. dd_{p+1} ~ u^{(p+1)}/(p+1)!
  LA::ParVectorType dd_p1;
  dd_p1.reinit(u_np1);
  dd_p1 = 0.0;

  if (p == 1)
  {
    // dd10 = [u_{n+1}, u_n]
    LA::ParVectorType dd10;
    dd10.reinit(u_np1);
    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    // dd11 = [u_n, u_{n-1}]
    LA::ParVectorType dd11;
    dd11.reinit(u_np1);
    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    // dd2 = [u_{n+1}, u_n, u_{n-1}]
    dd_p1 = dd10;
    dd_p1.add(-1.0, dd11);
    dd_p1 *= (1.0 / (h0 + h1));
  }
  else
  {
    AssertThrow(previous_solutions.size() >= 3,
                ExcMessage("Need u^{n-2} for BDF2 error estimate."));
    AssertThrow(time_steps.size() >= 3,
                ExcMessage("Need h0,h1,h2 for BDF2 error estimate."));

    // First-order divided differences
    LA::ParVectorType dd10, dd11, dd12;
    dd10.reinit(u_np1);
    dd11.reinit(u_np1);
    dd12.reinit(u_np1);

    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    dd12 = previous_solutions[1];
    dd12.add(-1.0, previous_solutions[2]);
    dd12 *= (1.0 / h2);

    // Second-order
    LA::ParVectorType dd20, dd21;
    dd20.reinit(u_np1);
    dd21.reinit(u_np1);

    dd20 = dd10;
    dd20.add(-1.0, dd11);
    dd20 *= (1.0 / (h0 + h1));

    dd21 = dd11;
    dd21.add(-1.0, dd12);
    dd21 *= (1.0 / (h1 + h2));

    // Third-order: dd3 = [u_{n+1},u_n,u_{n-1},u_{n-2}]
    dd_p1.reinit(u_np1);
    dd_p1 = dd20;
    dd_p1.add(-1.0, dd21);
    dd_p1 *= (1.0 / (h0 + h1 + h2));
  }

  // Build coefficient K = (-1)^{p+1} * h0 * sum_{i=1..p} alpha_i * H_i^{p+1}
  // with H1 = h0, H2 = h0+h1
  const double sign = ((p + 1) % 2 == 0 ? +1.0 : -1.0);

  double sum = 0.0;
  if (p >= 1)
    sum += bdf_coefficients[1] * std::pow(h0, static_cast<int>(p + 1));
  if (p >= 2)
    sum += bdf_coefficients[2] * std::pow(h0 + h1, static_cast<int>(p + 1));

  const double K = sign * h0 * sum;

  e_star = dd_p1;
  e_star *= K;
}


double TimeHandler::compute_scaled_vautrin_ratio(
  const LA::ParVectorType          &e_star,
  const LA::ParVectorType          &u_star,
  const std::vector<unsigned char> &dofs_to_component,
  const unsigned int                component_q,
  const double                      epsilon_q,
  const double                      /*u_seuil*/,
  const MPI_Comm                    comm) const
{
  (void)u_star; // on n'en a plus besoin en absolu-only

  double local_sum_e2 = 0.0;
  unsigned long long local_neq = 0;

  for (const auto gi : e_star.locally_owned_elements())
  {
    if (dofs_to_component[gi] != component_q)
      continue;

    const double e = e_star[gi];
    local_sum_e2 += e * e;
    local_neq++;
  }

  const double global_sum_e2 = dealii::Utilities::MPI::sum(local_sum_e2, comm);
  const unsigned long long global_neq =
    dealii::Utilities::MPI::sum(local_neq, comm);

  if (global_neq == 0)
    return std::numeric_limits<double>::max();

  const double e_abs = std::sqrt(global_sum_e2 / double(global_neq));

  // Erreur absolue uniquement : err = e_abs
  const double err = e_abs;

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
  LA::ParVectorType                                  *out_e_star,
  const double                                        reject_factor,
  bool                                               *out_step_accepted,
  double                                             *out_dt_retry)

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
  // Verbose: print e* abs max + eps (always meaningful even if step is rejected)
std::vector<double> comp_estar_abs_max;
std::vector<unsigned int> comp_ids;
comp_estar_abs_max.reserve(component_eps.size());
comp_ids.reserve(component_eps.size());

if (time_parameters.verbosity == Parameters::Verbosity::verbose)
{
  const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);

  for (const auto &ce : component_eps)
  {
    const unsigned int comp = ce.first;

    double local_max_abs = 0.0;
    unsigned long long local_count = 0;

    for (const auto gi : u_np1.locally_owned_elements())
    {
      if (dofs_to_component[gi] != comp)
        continue;

      local_max_abs = std::max(local_max_abs, std::abs(e_star[gi]));
      local_count++;
    }

    const double global_max_abs =
      dealii::Utilities::MPI::max(local_max_abs, comm);
    const unsigned long long global_count =
      dealii::Utilities::MPI::sum(local_count, comm);

    if (my_rank == 0)
    {
      std::cout
        << "[Vautrin] comp=" << comp
        << "  e*_abs_max=" << std::scientific << std::setprecision(6)
        << global_max_abs
        << "  (ndofs=" << std::dec << global_count << ")"
        << std::endl;
    }

    comp_ids.push_back(comp);
    comp_estar_abs_max.push_back(global_max_abs);
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
  const double dt_used = current_dt;

  // ------------------------------------------------------------
  // Reject only when enough history is meaningful.
  // We NEVER reject step 1 or 2 (startup).
  // ------------------------------------------------------------
  const bool rejection_enabled =
    (reject_factor > 0.0) && (current_time_iteration > 2);

  const double R_accept = rejection_enabled ? (1.0 / reject_factor) : 0.0;
  const bool step_accepted = (R >= R_accept); // if R_accept=0 => always accept


  // Print eps and e* max compared to threshold (also for rejected steps)
  if (time_parameters.verbosity == Parameters::Verbosity::verbose)
  {
    const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
    if (my_rank == 0)
    {
      for (unsigned int k = 0; k < component_eps.size(); ++k)
      {
        const unsigned int comp = component_eps[k].first;
        const double eps        = component_eps[k].second;

        // Find matching e* max we computed above (same order)
        const double emax = (k < comp_estar_abs_max.size() ? comp_estar_abs_max[k] : std::numeric_limits<double>::quiet_NaN());

        std::cout
          << "[Vautrin] comp=" << comp
          << "  eps=" << std::scientific << std::setprecision(6) << eps
          << "  reject_thr=" << std::scientific << std::setprecision(6) << (reject_factor * eps)
          << "  e*_abs_max=" << std::scientific << std::setprecision(6) << emax
          << "  status=" << (step_accepted ? "ACCEPT" : "REJECT")
          << std::endl;
      }
    }
  }


  const double dt_min = this->get_effective_dt_min();
  const double dt_max = this->get_effective_dt_max();

  const double ratio_min = (order == 2 ? 0.2 : 0.1);
  const double ratio_max = (order == 2 ? (1.0 + std::sqrt(2.0)) : 5.0);

  if (rejection_enabled &&!step_accepted)
  {
    // Rejected step: we must decrease dt (cap ratio_max at 1.0)
    const double ratio_max_reject = 1.0;

    double dt_retry =
      TimeHandler::propose_next_dt_vautrin(dt_used, R, order, safety,
                                           ratio_min, ratio_max_reject,
                                           dt_min, dt_max);

    // If the Vautrin proposal does not reduce (can happen when capped),
    // fall back to a monotone interpolation between previous_dt and dt_used.
    // We use the already-computed R = min_q (eps_q / err_q).
    // On a rejected step, R < 1 and smaller R means 'too much error'.
    double previous_dt = dt_used;
    if (time_steps.size() >= 2)
      previous_dt = time_steps[1]; // dt used at the previous accepted/attempted step

    const double w = std::clamp(std::abs(R), 0.0, 1.0);
    const double dt_interp = previous_dt + (dt_used - previous_dt) * w;

    // Choose the most conservative reduction (smallest dt), while respecting dt_min/dt_max.
    dt_retry = std::min(dt_retry, dt_interp);
    dt_retry = std::max(dt_min, std::min(dt_retry, dt_max));

    // Guarantee a strict decrease on reject
    if (dt_retry >= dt_used)
      dt_retry = std::max(dt_min, std::nextafter(dt_used, 0.0));

    current_dt = dt_retry;

    if (out_step_accepted) *out_step_accepted = false;
    if (out_dt_retry)      *out_dt_retry      = dt_retry;

    if (out_R)       *out_R       = R;
    if (out_order)   *out_order   = order;
    if (out_dt_used) *out_dt_used = dt_used;
    if (out_dt_next) *out_dt_next = dt_retry;
    if (out_e_star)  *out_e_star  = e_star;

    if (time_parameters.verbosity == Parameters::Verbosity::verbose)
    {
      const unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
      if (my_rank == 0)
      {
        std::cout
          << "[REJECT] step=" << current_time_iteration
          << " t=" << std::scientific << std::setprecision(10) << current_time
          << " dt_used=" << std::scientific << std::setprecision(10) << dt_used
          << " dt_new=" << std::scientific << std::setprecision(10) << dt_retry
          << " (R=" << std::scientific << std::setprecision(6) << R << ")"
          << std::endl;
      }
    }

    return false; // rejected
  }

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

void TimeHandler::rollback_last_advance(const double dt_retry)
{
  if (!has_step_backup)
    return;

  current_time           = backup_current_time;
  current_time_iteration = backup_current_time_iteration;

  current_dt       = dt_retry;
  previous_times   = backup_previous_times;
  time_steps       = backup_time_steps;
  bdf_coefficients = backup_bdf_coefficients;
  pending_dt_queue = backup_pending_dt_queue;

  // Important: on repart sur un état cohérent, et dt_retry sera utilisé au prochain advance().
  has_step_backup = false;
}

