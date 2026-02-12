#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>
#include <parameters.h>
#include <deque>


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

  /** Compute the Vautrin time–integration error estimator based on the current
    * solution and a set of previous time-step solutions. This estimator can be
    * used for adaptive time-step control.
    */
  template <typename VectorType>
  void compute_vautrin_error_estimate(VectorType              &e_star,
                                    const VectorType        &u_np1,
                                    const std::vector<VectorType> &previous_solutions,
                                    const unsigned int       order) const;


  /** Override the time integration parameters (e.g. for restarting a simulation
   * with different time step). This will update the time integration scheme and
   * recompute the BDF coefficients if needed.
   */
  void apply_restart_overrides(const Parameters::TimeIntegration &new_params);

  std::deque<double> pending_dt_queue; // dts "programmés" (rampe restart)




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
  ar & initial_time;
  ar & final_time;
  ar & current_time;
  ar & current_time_iteration;
  ar & previous_times;
  ar & current_dt;
  ar & time_steps;
  ar & bdf_coefficients;
}

template <typename VectorType>
void TimeHandler::compute_vautrin_error_estimate(
  VectorType                    &e_star,
  const VectorType              &u_np1,
  const std::vector<VectorType> &previous_solutions,
  const unsigned int             order) const
{
  AssertThrow(!this->is_steady(),
              ExcMessage("No time error estimate in steady."));
  AssertThrow(order == 1 || order == 2,
              ExcMessage("Only order 1 or 2 supported."));
  AssertThrow(previous_solutions.size() >= order + 1,
              ExcMessage("Not enough history in previous_solutions. Need order+1 vectors."));
  AssertThrow(bdf_coefficients.size() >= order + 1,
              ExcMessage("bdf_coefficients size inconsistent with requested order."));
  AssertThrow(time_steps.size() >= 1,
              ExcMessage("Need at least one stored time step (h0)."));

  // time_steps[0]=h_{n+1}, time_steps[1]=h_n, time_steps[2]=h_{n-1}
  const double h0 = time_steps[0];
  const double h1 = (time_steps.size() > 1 ? time_steps[1] : h0);
  const double h2 = (time_steps.size() > 2 ? time_steps[2] : h1);

  const unsigned int p = order;   // BDF order (1 or 2)
  const unsigned int k = p + 1;   // p+1 (2 or 3)

  // factorial(k): only 2 or 6 here
  const double factorial = (k == 2 ? 2.0 : 6.0);
  // (-1)^k
  const double sign = (k % 2 == 0 ? +1.0 : -1.0);

  // H0=0, H1=h0, H2=h0+h1
  // Convention: bdf_coefficients[0] multiplies u_{n+1}, [1] -> u_n, [2] -> u_{n-1}
  // i=0 term is alpha0 * 0^k = 0 for k>=2, so skip.
  double sum = 0.0;
  sum += bdf_coefficients[1] * std::pow(h0, static_cast<int>(k));
  if (p == 2)
    sum += bdf_coefficients[2] * std::pow(h0 + h1, static_cast<int>(k));

  const double C_p1 = sign * (sum / factorial);

  // Compute u^{(p+1)} via divided differences on non-uniform grid
  VectorType u_p1;
  u_p1.reinit(u_np1);
  u_p1 = 0.0;

  if (p == 1)
  {
    // dd1_0 = (u_{n+1}-u_n)/h0
    VectorType dd10;
    dd10.reinit(u_np1);
    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    // dd1_1 = (u_n-u_{n-1})/h1
    VectorType dd11;
    dd11.reinit(u_np1);
    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    // dd2 = (dd1_0 - dd1_1)/(h0+h1)
    VectorType dd2;
    dd2.reinit(u_np1);
    dd2 = dd10;
    dd2.add(-1.0, dd11);
    dd2 *= (1.0 / (h0 + h1));

    // u'' ≈ 2! * dd2
    u_p1 = dd2;
    u_p1 *= 2.0;
  }
  else // p==2
  {
    AssertThrow(previous_solutions.size() >= 3,
                ExcMessage("Need u^{n-2} in previous_solutions[2] for BDF2 error estimate."));
    AssertThrow(time_steps.size() >= 3,
                ExcMessage("Need 3 stored time steps (h0,h1,h2) for BDF2 error estimate."));

    // dd1_0 = (u_{n+1}-u_n)/h0
    VectorType dd10;
    dd10.reinit(u_np1);
    dd10 = u_np1;
    dd10.add(-1.0, previous_solutions[0]);
    dd10 *= (1.0 / h0);

    // dd1_1 = (u_n-u_{n-1})/h1
    VectorType dd11;
    dd11.reinit(u_np1);
    dd11 = previous_solutions[0];
    dd11.add(-1.0, previous_solutions[1]);
    dd11 *= (1.0 / h1);

    // dd1_2 = (u_{n-1}-u_{n-2})/h2
    VectorType dd12;
    dd12.reinit(u_np1);
    dd12 = previous_solutions[1];
    dd12.add(-1.0, previous_solutions[2]);
    dd12 *= (1.0 / h2);

    // dd2_0 = (dd1_0 - dd1_1)/(h0+h1)
    VectorType dd20;
    dd20.reinit(u_np1);
    dd20 = dd10;
    dd20.add(-1.0, dd11);
    dd20 *= (1.0 / (h0 + h1));

    // dd2_1 = (dd1_1 - dd1_2)/(h1+h2)
    VectorType dd21;
    dd21.reinit(u_np1);
    dd21 = dd11;
    dd21.add(-1.0, dd12);
    dd21 *= (1.0 / (h1 + h2));

    // dd3 = (dd2_0 - dd2_1)/(h0+h1+h2)
    VectorType dd3;
    dd3.reinit(u_np1);
    dd3 = dd20;
    dd3.add(-1.0, dd21);
    dd3 *= (1.0 / (h0 + h1 + h2));

    // u''' ≈ 3! * dd3
    u_p1 = dd3;
    u_p1 *= 6.0;
  }

  // e_star = C_{p+1} * h_{n+1} * u^{(p+1)}
  e_star.reinit(u_np1);
  e_star = u_p1;
  e_star *= (C_p1 * h0);
}



#endif