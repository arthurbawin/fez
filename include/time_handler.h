#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <components_ordering.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>
#include <parameters.h>
#include <solver_info.h>

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
  bool is_steady() const
  {
    return scheme == Parameters::TimeIntegration::Scheme::stationary;
  }

  /**
   * For BDF methods, return true if the current time step is a
   * "starting step" (none for BDF1, first for BDF2).
   */
  bool is_starting_step() const
  {
    return current_time_iteration < n_previous_solutions;
  }

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
  template <typename VectorType>
  void rotate_solutions(const VectorType        &present_solution,
                        std::vector<VectorType> &previous_solutions) const;

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
   *
   */
  template <typename VectorType>
  void
  compute_error_estimator(const VectorType              &present_solution,
                          const std::vector<VectorType> &previous_solutions);

public:
  Parameters::TimeIntegration time_parameters;

  double              current_time;
  unsigned int        current_time_iteration;
  double              initial_time;
  double              final_time;
  std::vector<double> previous_times;

  double              initial_dt;
  double              current_dt;
  std::vector<double> time_steps;

  Parameters::TimeIntegration::Scheme scheme;

  unsigned int        n_previous_solutions;
  std::vector<double> bdf_coefficients;
};

/* ---------------- Template functions ----------------- */

template <typename VectorType>
void TimeHandler::rotate_solutions(
  const VectorType        &present_solution,
  std::vector<VectorType> &previous_solutions) const
{
  if (!this->is_steady())
  {
    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
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
  ar &previous_times;
  ar &current_dt;
  ar &time_steps;
  ar &bdf_coefficients;
}

template <typename VectorType>
void TimeHandler::compute_error_estimator(
  const VectorType                 &present_solution,
  const std::vector<VectorType>    &previous_solutions,
  const IndexSet                   &locally_relevant_dofs,
  const std::vector<unsigned char> &dofs_to_component)
{
  Assert(dofs_to_component.size() > 0,
         ExcMessage("dofs_to_component should be filled"));

  std::map<SolverInfo::VariableType, double> max_error;
  for (const auto var : SolverInfo::variable_types)
    max_error[var] = 0.;

  // const unsigned int local_size = present_solution.locally_owned_size();

  if (scheme == Parameters::TimeIntegration::Scheme::BDF1)
  {
    constexpr unsigned int n_sol = 3;
    AssertDimension(previous_times.size(), n_sol);

    const unsigned int p    = 1;
    const double       tnp1 = previous_times[0];
    const double       tn   = previous_times[1];
    const double       h    = tnp1 - tn;
    const double       H1   = h;

    std::array<double, n_sol> times, values;
    for (unsigned int i = 0; i < n_sol; ++i)
      times[i] = previous_times[i];

    for (const auto &dof : locally_relevant_dofs)
    {
      values[0] = present_solution[dof];
      for (unsigned int i = 0; i < n_sol - 1; ++i)
        values[i + 1] = previous_solutions[i][dof];

      // FIXME: error proportional to h^3 ???
      const double error = h * bdf_coefficients[1] * h * h *
                           divided_difference_order_2(times, values);

      // Get dof component, then variable type of this component
      const auto comp =
        dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
      const auto type    = ComponentOrdering::component_to_variable_type(comp);
      max_error.at(type) = std::max(max_error.at(type), error);
    }

    // Synchronize the max errors across ranks
    // TODO
  }
  else if (scheme == Parameters::TimeIntegration::Scheme::BDF2) {}
  else
  {
    DEAL_II_NOT_IMPLEMENTED();
  }
}
#endif
