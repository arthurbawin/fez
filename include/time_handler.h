#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <components_ordering.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>
#include <parameters.h>
#include <types.h>

// Forward declaration
class BDFErrorEstimator;

using namespace dealii;

/**
 * This class takes care of the time integration-related data:
 *
 * - updating the current time and time step count
 * - update the BDF coefficients if using a BDF method
 *
 * TODO: Implement this class as a derived DiscreteTime
 */
class TimeHandler
{
public:
  /**
   * Constructor.
   */
  TimeHandler(const Parameters::TimeIntegration &time_parameters);

  /**
   * The destructor is the default destructor, but because this class stores a
   * pointer to a forward declared BDFErrorEstimator, the destructor is declared
   * here but implemented in time_handler.cpp. This is because the poiner's
   * destructor needs to know the sizeof the object it is pointing to, which is
   * not known in this file.
   */
  ~TimeHandler();

  /**
   * Update the BDF coefficients given the current and previous
   * time steps.
   */
  void
  set_bdf_coefficients(const bool force_scheme = false,
                       const Parameters::TimeIntegration::Scheme forced_scheme =
                         Parameters::TimeIntegration::Scheme::BDF1);

  /**
   * Returns true if the time integration scheme is "stationary"
   */
  bool is_steady() const;

  /**
   * For BDF methods, return true if the current time step is a
   * "starting step" (none for BDF1, first for BDF2).
   */
  bool is_starting_step() const;

  /**
   * Returns true if the simulation should stop:
   * - always true if simulation is steady
   * - if t >= t_end if unsteady
   */
  bool is_finished() const;

  /**
   * Rotate the computed time step i+1 to position i.
   */
  void advance(const ConditionalOStream &pcout);

  /**
   * Shift the BDF solutions by one (u^{n-1} becomes u^n, etc.)
   */
  void
  rotate_solutions(const LA::ParVectorType        &present_solution,
                   std::vector<LA::ParVectorType> &previous_solutions) const;

  /**
   * Attach solver data to the error estimator.
   * If adaptive time stepping is enabled, this function must be called before
   * the first call to is_timestep_accepted(), which relies on the error
   * estimates to accept or reject a step.
   */
  void attach_data_to_error_estimator(
    const ComponentOrdering          &ordering,
    const IndexSet                   &locally_relevant_dofs,
    const std::vector<unsigned char> &dofs_to_component);

  /**
   * Return the vector of the last computed error estimator at each dof.
   *
   * The goal of this function is to provide the error estimator in a format
   * compatible with the L^p error norm routines, e.g., to compute the
   * discrepancy between the error estimator and the true error. Consequently,
   * this function is only available if compute_error_on_estimator is enabled in
   * the parameter file.
   */
  const LA::ParVectorType &get_error_estimator_as_solution() const;

  /**
   * Checks if the solution obtained at this step after the nonlinear solve is
   * acceptable or not.
   *
   * If adaptive time stepping is disabled or if the simulation is steady-state,
   * this function always returns true.
   *
   * If adaptive time stepping is enabled, this compares the estimated
   * truncation error to the specified target error for each variable. If the
   * estimated error is too high, the step is rejected and we try again with a
   * smaller time step. The passed @p present_solution is set to first vector in
   * @p previous_solutions, and the current time, time step and step counter are
   * rolled back to their previous values. If the estimated error is low enough,
   * the step is accepted and the next time step is set based on the most
   * critical truncation error estimate.
   *
   * Since truncation error is not computed during the starting steps, these
   * steps are always accepted and the time step is not modified.
   *
   * If the time step is rejected too many times in a row, the simulation stops.
   *
   * FIXME: Maybe set the maximum number of rejections as a user parameter.
   * FIXME: Also allow reducing the time step if the Newton method did not
   * converge.
   */
  bool is_timestep_accepted(
    LA::ParVectorType                    &present_solution,
    const std::vector<LA::ParVectorType> &previous_solutions);

  /**
   * Compute the approximation of the time derivative of the field associated to
   * the index-th dof, e.g. the sum c_i * u^(n - i) where c_i are the BDF
   * coefficients. This only makes sense for nodal finite elements, for which
   * the dof represents an actual field value.
   */
  template <typename VectorType>
  double compute_time_derivative(
    const types::global_dof_index  index,
    const VectorType              &present_solution,
    const std::vector<VectorType> &previous_solutions) const;

  /**
   * Same as above but for the time derivative of a scalar,
   * given the current and previous vectors, at index-th quadrature node.
   *
   * This is tailored for a previous_solutions vector stored in a scratch data.
   */
  double compute_time_derivative_at_quadrature_node(
    const unsigned int                      quadrature_node_index,
    const double                            present_solution,
    const std::vector<std::vector<double>> &previous_solutions) const;

  /**
   * Same as above but for the time derivative of a vector (Tensor<1, dim>).
   */
  template <int dim>
  Tensor<1, dim> compute_time_derivative_at_quadrature_node(
    const unsigned int                              quadrature_node_index,
    const Tensor<1, dim>                           &present_solution,
    const std::vector<std::vector<Tensor<1, dim>>> &previous_solutions) const;

  /**
   * Save the time integration data to a file.
   */
  void save() const;

  /**
   * Load time integration data from existing file.
   */
  void load();

  /**
   *
   */
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version);

  /** Override the time integration parameters (e.g. for restarting a simulation
   * with different time step). This will update the time integration scheme and
   * recompute the BDF coefficients if needed.
   */
  void update_parameters_after_restart(
    const Parameters::TimeIntegration &new_parameters);

private:
  /**
   * Determine and apply the next time step.
   */
  void set_next_timestep();

public:
  Parameters::TimeIntegration time_parameters;

  double              current_time;
  unsigned int        current_time_iteration;
  double              initial_time;
  double              final_time;
  std::vector<double> simulation_times;

  double              initial_dt;
  double              current_dt;
  std::vector<double> time_steps;

  Parameters::TimeIntegration::Scheme scheme;

  unsigned int        n_previous_solutions;
  std::vector<double> bdf_coefficients;
  unsigned int        bdf_order;

  bool         rolledback_step;
  unsigned int n_consecutive_rejected_steps;
  unsigned int n_rejected_steps;

public:
  std::shared_ptr<BDFErrorEstimator> error_estimator;
};

/* ---------------- Template functions ----------------- */

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
      value_dot +=
        bdf_coefficients[i] * previous_solutions[i - 1][quadrature_node_index];
    return value_dot;
  }
  DEAL_II_ASSERT_UNREACHABLE();
}

template <class Archive>
void TimeHandler::serialize(Archive &ar, const unsigned int /*version*/)
{
  ar &initial_time;
  ar &final_time;
  ar &current_time;
  ar &current_time_iteration;
  ar &simulation_times;
  ar &current_dt;
  ar &time_steps;
  ar &bdf_coefficients;
}

#endif
