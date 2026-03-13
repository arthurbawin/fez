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
   * Determine and apply the next time step.
   */
  void
  set_next_timestep(const ComponentOrdering              &ordering,
                    const LA::ParVectorType              &present_solution,
                    const std::vector<LA::ParVectorType> &previous_solutions,
                    const IndexSet                       &locally_relevant_dofs,
                    const std::vector<unsigned char>     &dofs_to_component);

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
