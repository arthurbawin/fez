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

  double get_current_dt() const { return current_dt; }


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

  /**
   * Override time integration parameters at restart
   */
  void update_parameters_after_restart(const Parameters::TimeIntegration &new_params);


  /** Compute the BDF3 time error estimate e_star. */
  void compute_error_estimate(
    LA::ParVectorType                    &e_star,
    const LA::ParVectorType              &u_np1,
    const std::vector<LA::ParVectorType> &previous_solutions,
    const unsigned int                    order) const;

  /** Compute scaled ratio R_q = eps_q / err_q for component q. */
  double compute_scaled_ratio(
    const LA::ParVectorType          &e_star,
    const LA::ParVectorType          &u_star,
    const std::vector<unsigned char> &dofs_to_component,
    const unsigned int                component_q,
    const double                      epsilon_q,
    const MPI_Comm                    comm) const;

  /** Propose dt_{n+1} from dt_n and global ratio R. */
  static double propose_next_dt(const double       dt,
                                       const double       R,
                                       const unsigned int order,
                                       const double       safety,
                                       const double       ratio_min,
                                       const double       ratio_max,
                                       const double       dt_min,
                                       const double       dt_max);

  double get_effective_dt_min() const;
  double get_effective_dt_max() const;

  /**update the new dt when the step as converged */
  bool update_dt_after_converged_step(
    const LA::ParVectorType                            &u_np1,
    const std::vector<LA::ParVectorType>               &previous_solutions_dt_control,
    const std::vector<unsigned char>                   &dofs_to_component,
    const std::vector<std::pair<unsigned int, double>> &component_eps,
    const double                                        safety,
    const MPI_Comm                                      comm,
    const double                                        t_end,
    const bool                                          clamp_to_t_end = true,
    double                                             *out_R         = nullptr,
    unsigned int                                       *out_order     = nullptr,
    double                                             *out_dt_used   = nullptr,
    double                                             *out_dt_next   = nullptr,
    LA::ParVectorType                                  *out_e_star    = nullptr,
    const double reject_factor = 2.0,
    bool *out_step_accepted = nullptr,
    double *out_dt_retry = nullptr);

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

  // Rollback the last advance() (used when a time step is rejected)
  void rollback_last_advance(const double dt_retry);

private:

  bool                has_step_backup = false;
  double              backup_current_time = 0.0;
  unsigned int        backup_current_time_iteration = 0;
  double              backup_current_dt = 0.0;
  std::vector<double> backup_previous_times;
  std::vector<double> backup_time_steps;
  std::vector<double> backup_bdf_coefficients;
  std::deque<double>  backup_pending_dt_queue;

  // ------------------------------------------------------------
  // Historical guard for "bad" dt proposals after a rejection.
  //
  // Scalar mismatch based on limiting component:
  //   delta_n = | err_n - eps |
  // where err_n is the scalar error measure used in dt-control (here: RMS(|e*|))
  // for the component that realizes R = min_q (eps_q/err_q).
  //
  // Ratio:
  //   r_n = delta_n / delta_{n-1}
  //
  // If the next proposed dt is rejected, we freeze r_n (the ratio that led to it).
  // When the step later passes (with smaller dt), we store:
  //   history_reject_ratio = r_n
  //   history_reject_dt_limit = dt_that_passed
  //
  // Later, if r_current > trigger_factor * history_reject_ratio,
  // we cap dt_next by history_reject_dt_limit.
  // ------------------------------------------------------------
  double history_reject_ratio = std::numeric_limits<double>::infinity();
  double history_reject_dt_limit = std::numeric_limits<double>::infinity();

  double last_mismatch_delta = std::numeric_limits<double>::quiet_NaN();
  double last_mismatch_ratio = std::numeric_limits<double>::quiet_NaN();

  bool   pending_reject_history = false;
  double pending_reject_ratio = std::numeric_limits<double>::quiet_NaN();

  double history_reject_trigger_factor = 0.7;

};



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
}

#endif
