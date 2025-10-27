#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <deal.II/base/types.h>
#include <parameters.h>

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
  void set_bdf_coefficients();

  /**
   * Rotates the computed time step i+1 to position i.
   */
  void rotate();

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

public:
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
    for (unsigned int i = 1; i < n_previous_solutions; ++i)
      value_dot += bdf_coefficients[i] * previous_solutions[i - 1][index];
    return value_dot;
  }
  DEAL_II_ASSERT_UNREACHABLE();
}

#endif