#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>

#include <parameters.h>
#include <types.h>

#include <deque>
#include <limits>
#include <utility>
#include <vector>

class ComponentOrdering;

using namespace dealii;

/**
 * This class takes care of the time integration-related data:
 *
 * - updating the current time and time step count
 * - update the BDF coefficients if using a BDF method
 */
class TimeHandler
{
public:
  TimeHandler(const Parameters::TimeIntegration &time_parameters);

  /**
   * Update the BDF coefficients given the current and previous time steps.
   */
  void set_bdf_coefficients(
    const bool                                force_scheme = false,
    const Parameters::TimeIntegration::Scheme forced_scheme =
      Parameters::TimeIntegration::Scheme::BDF1);

  /**
   * Returns true if the time integration scheme is "stationary"
   */
  bool is_steady() const
  {
    return scheme == Parameters::TimeIntegration::Scheme::stationary;
  }

  /**
   * For BDF methods, return true if the current time step is a "starting step"
   * (none for BDF1, first steps for BDF2).
   */
  bool is_starting_step() const
  {
    return current_time_iteration < n_previous_solutions;
  }

  /**
   * Returns true if the simulation should stop:
   * - steady: stop after one iteration
   * - unsteady: stop when t >= t_end
   */
  bool is_finished() const;

  /**
   * Rotate times/time steps and advance current time by current_dt.
   */
  void advance(const ConditionalOStream &pcout);

  /**
   * Shift the BDF solutions by one (u^{n-1} becomes u^n, etc.)
   */
  template <typename VectorType>
  void rotate_solutions(const VectorType        &present_solution,
                        std::vector<VectorType> &previous_solutions) const;

  /**
   * Compute the approximation of the time derivative at a DoF index,
   * using BDF coefficients and stored history vectors.
   */
  template <typename VectorType>
  double compute_time_derivative(const types::global_dof_index  index,
                                 const VectorType              &present_solution,
                                 const std::vector<VectorType> &previous_solutions) const;

  /**
   * Time derivative of a scalar at quadrature node, given current and previous
   * values stored in scratch.
   */
  double compute_time_derivative_at_quadrature_node(
    const unsigned int                      quadrature_node_index,
    const double                            present_solution,
    const std::vector<std::vector<double>> &previous_solutions) const;

  /**
   * Time derivative of a Tensor<1,dim> at quadrature node.
   */
  template <int dim>
  Tensor<1, dim> compute_time_derivative_at_quadrature_node(
    const unsigned int                              quadrature_node_index,
    const Tensor<1, dim>                           &present_solution,
    const std::vector<std::vector<Tensor<1, dim>>> &previous_solutions) const;

  /**
   * Save / load time integration data (not implemented here).
   */
  void save() const;
  void load();

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version);

  // ------------------------------------------------------------
  // Programmed dt schedules (paper-style variable-step tests)
  // ------------------------------------------------------------
  void build_programmed_dt_schedule();

  /**
   * Override time integration parameters at restart (e.g. changing dt).
   */
  void apply_restart_overrides(const Parameters::TimeIntegration &new_params);

  // =========================
  // Vautrin adaptive dt-control
  // =========================

  /** Compute Vautrin time error estimate e_star. */
  void compute_vautrin_error_estimate(
    LA::ParVectorType                    &e_star,
    const LA::ParVectorType              &u_np1,
    const std::vector<LA::ParVectorType> &previous_solutions,
    const unsigned int                    order) const;

  /** Compute scaled ratio R_q = eps_q / err_q for component q. */
  double compute_scaled_vautrin_ratio(
    const LA::ParVectorType          &e_star,
    const LA::ParVectorType          &u_star,
    const std::vector<unsigned char> &dofs_to_component,
    const unsigned int                component_q,
    const double                      epsilon_q,
    const double                      u_seuil,
    const MPI_Comm                    comm) const;

  /** Propose dt_{n+1} from dt_n and global ratio R. */
  static double propose_next_dt_vautrin(const double       dt,
                                       const double       R,
                                       const unsigned int order,
                                       const double       safety,
                                       const double       ratio_min,
                                       const double       ratio_max,
                                       const double       dt_min,
                                       const double       dt_max);

  double get_effective_dt_min() const;
  double get_effective_dt_max() const;

  /** Low-level Vautrin update: user provides (component, eps) list. */
  bool update_dt_after_converged_step_vautrin(
    const LA::ParVectorType                            &u_np1,
    const std::vector<LA::ParVectorType>               &previous_solutions_dt_control,
    const std::vector<unsigned char>                   &dofs_to_component,
    const std::vector<std::pair<unsigned int, double>> &component_eps,
    const double                                        u_seuil,
    const double                                        safety,
    const double                                        dt_min_factor,
    const double                                        dt_max_factor,
    const MPI_Comm                                      comm,
    const double                                        t_end,
    const bool                                          clamp_to_t_end = true,
    double                                             *out_R         = nullptr,
    unsigned int                                       *out_order     = nullptr,
    double                                             *out_dt_used   = nullptr,
    double                                             *out_dt_next   = nullptr,
    LA::ParVectorType                                  *out_e_star    = nullptr);

  /**
   * High-level overload: build component_eps from ordering + ti,
   * AND (optionally) scale eps with dt for MMS comparisons:
   *   eps <- eps * (dt_run/dt_ref)^p  (p=1 for BDF1, p=2 for BDF2)
   */
  bool update_dt_after_converged_step_vautrin(
    const LA::ParVectorType               &u_np1,
    const std::vector<LA::ParVectorType>  &previous_solutions_dt_control,
    const std::vector<unsigned char>      &dofs_to_component,
    const ComponentOrdering               &ordering,
    const Parameters::TimeIntegration     &ti,
    const MPI_Comm                         comm,
    const double                           t_end,
    const bool                             clamp_to_t_end = true,
    double                                *out_R         = nullptr,
    unsigned int                          *out_order     = nullptr,
    double                                *out_dt_used   = nullptr,
    double                                *out_dt_next   = nullptr,
    LA::ParVectorType                     *out_e_star    = nullptr);

  // ------------------------------------------------------------
  // Public state (as in your current code)
  // ------------------------------------------------------------
public:
  Parameters::TimeIntegration time_parameters;

  double       current_time;
  unsigned int current_time_iteration;

  double              initial_time;
  double              final_time;
  std::vector<double> previous_times;

  double              initial_dt;
  double              current_dt;
  std::vector<double> time_steps;

  double dt_ref_bounds;

  Parameters::TimeIntegration::Scheme scheme;

  unsigned int        n_previous_solutions;
  std::vector<double> bdf_coefficients;

  // dt ramp (restart)
  std::deque<double> pending_dt_queue;

  // programmed dt
  bool                use_programmed_dt_schedule = false;
  std::vector<double> programmed_dt;

private:
  // ============================================================
  // MMS eps scaling: dt reference captured ONCE (first run).
  // When MMS driver refines dt (dt <- dt/2), you can scale eps as:
  // eps <- eps * (dt/dt_ref)^p, which gives eps/2^p when dt is halved.
  // ============================================================
  double mms_dt_reference = -1.0;
};

/* ---------------- Template functions ----------------- */

template <typename VectorType>
void TimeHandler::rotate_solutions(const VectorType        &present_solution,
                                   std::vector<VectorType> &previous_solutions) const
{
  if (this->is_steady() || previous_solutions.empty())
    return;

  for (std::size_t j = previous_solutions.size(); j-- > 1;)
    previous_solutions[j] = previous_solutions[j - 1];

  previous_solutions[0] = present_solution;
}

template <typename VectorType>
double TimeHandler::compute_time_derivative(
  const types::global_dof_index  index,
  const VectorType              &present_solution,
  const std::vector<VectorType> &previous_solutions) const
{
  if (scheme == Parameters::TimeIntegration::Scheme::stationary)
    return 0.;

  if (scheme == Parameters::TimeIntegration::Scheme::BDF1 ||
      scheme == Parameters::TimeIntegration::Scheme::BDF2)
  {
    double value_dot = bdf_coefficients[0] * present_solution[index];
    for (unsigned int i = 1; i < bdf_coefficients.size(); ++i)
      value_dot += bdf_coefficients[i] * previous_solutions[i - 1][index];
    return value_dot;
  }

  DEAL_II_ASSERT_UNREACHABLE();
}

template <int dim>
Tensor<1, dim> TimeHandler::compute_time_derivative_at_quadrature_node(
  const unsigned int                              quadrature_node_index,
  const Tensor<1, dim>                           &present_solution,
  const std::vector<std::vector<Tensor<1, dim>>> &previous_solutions) const
{
  if (scheme == Parameters::TimeIntegration::Scheme::stationary)
    return Tensor<1, dim>();

  if (scheme == Parameters::TimeIntegration::Scheme::BDF1 ||
      scheme == Parameters::TimeIntegration::Scheme::BDF2)
  {
    Tensor<1, dim> value_dot = bdf_coefficients[0] * present_solution;
    for (unsigned int i = 1; i < bdf_coefficients.size(); ++i)
      value_dot += bdf_coefficients[i] * previous_solutions[i - 1][quadrature_node_index];
    return value_dot;
  }

  DEAL_II_ASSERT_UNREACHABLE();
}

template <class Archive>
void TimeHandler::serialize(Archive &ar, const unsigned int /*version*/)
{
  ar & initial_time;
  ar & final_time;
  ar & current_time;
  ar & current_time_iteration;
  ar & previous_times;
  ar & current_dt;
  ar & time_steps;
  ar & bdf_coefficients;

  ar & initial_dt;
  ar & dt_ref_bounds;

  ar & mms_dt_reference;
}

#endif
