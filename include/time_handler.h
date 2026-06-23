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
   * here but implemented in time_handler.cpp. This is because the pointer's
   * destructor needs to know the sizeof the object it is pointing to, which is
   * not known in this file.
   */
  ~TimeHandler();

  /**
   * Set the initial and final times for the @p interval_index-th time subinterval.
   */
  void set_time_interval(const unsigned int interval_index);

  /**
   * Validate the parameters depending on the derived solver using this object.
   * This prevents for instance using CFL as a time step adaptation criterion
   * with solvers that do not have the fluid velocity as a variable.
   */
  void validate_parameters(const ComponentOrdering &ordering) const;

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
   * For BDF methods, return true if the previous time step was a
   * "starting step".
   */
  bool last_step_was_starting_step() const;

  /**
   * Returns true if the simulation should stop:
   * - always true if simulation is steady
   * - if t >= t_end if unsteady
   */
  bool is_finished() const;

  /**
   * Return the time step used to advance for this time iteration.
   *
   * Note: the time step rejection mechanism works by trying nonlinear solves
   * until a solution is found, and decrease the time step otherwise. Thus, the
   * time step (current_dt) is updated to the next time step as soon as the
   * solver leaves the nonlinear solver, meaning that all evaluations of
   * time_handler.current_dt *after* calling is_timestep_accepted() will read
   * the time step for the next time iteration. Instead, this function returns
   * time_steps[0], which is updated in the advance() function. Similarly to the
   * issue with current_dt, this also means that calling this function before
   * calling advance() will return a lagged time step, so be cautions.
   * Currently, nothing is done in the time integration loops before calling the
   * advance() function, which is why this choice has been done.
   */
  double get_current_timestep() const;

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
   * Return a copy of the current BDF coefficients.
   */
  const std::vector<double> &get_bdf_coefficients() const;

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
   * Inform this object of the sucess of the last nonlinear solve.
   */
  void set_last_nonlinear_solve_status(const bool flag) const;

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
   * Return the total number of rejected time steps so far.
   */
  unsigned int get_n_rejected_steps() const;

  /**
   * Update the max CFL number.
   */
  void set_max_cfl(const double max_cfl);

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
   * Write the complete times and time steps history to the given stream.
   */
  void write_timestep_history(std::ostream &out = std::cout) const;

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
    const Parameters::TimeIntegration &new_parameters,
    const ConditionalOStream          &pcout);

private:
  /**
   * Update the BDF coefficients given the current and previous
   * time steps.
   */
  void
  set_bdf_coefficients(const bool force_scheme = false,
                       const Parameters::TimeIntegration::Scheme forced_scheme =
                         Parameters::TimeIntegration::Scheme::BDF1);

  /**
   * Determine and apply the next time step.
   */
  void set_next_timestep(const bool step_was_accepted);

  /**
   * Increment the integral of dt(t)^-1 over the current time subinterval.
   * This is actually done by simply incrementing the time steps count for that
   * interval, which is equivalent.
   */
  void increment_inverse_time_step_integral();

public:
  // A copy of the associated parameters
  Parameters::TimeIntegration time_parameters;

  // This value is true if the chosen time stepping scheme actually solves for
  // a steady-state solution, and is false otherwise.
  bool steady_scheme;

  // The number of time subintervals in the simulation.
  // If not using the transient fixed point mesh adaptation method, this value
  // is 1, i.e., the simulation time interval is split into a single subinterval
  // (itself). Otherwise, this is the prescribed number of (equal) subintervals.
  unsigned int n_time_intervals;

  // Current time subinterval
  unsigned int current_time_interval;

  // Integral of the time step on each time subinterval
  std::vector<double> inverse_time_step_integrals;

  std::vector<unsigned int> n_steps_on_each_interval;

  // Current simulation time
  double current_time;

  // Current (global) time step counter of the simulation
  unsigned int current_time_iteration;

  // Current time step counter for the current time interval
  unsigned int current_time_iteration_in_interval;

  // Time step counter at which the last restart happened
  unsigned int time_iteration_at_last_restart;

  // Initial time (for this subinterval)
  double initial_time;

  // Final time (for this subinterval)
  double final_time;

  // Initial times of all subintervals
  std::vector<double> initial_times;

  // Final times of all subintervals
  std::vector<double> final_times;

  // The simulation times for the current and last N time steps
  // FIXME: misnomer, this only includes the recent times
  std::vector<double> simulation_times;

  // Log of all simulation times
  std::vector<double> all_simulation_times;

  // Initial simulation time step
  double initial_dt;

  // Current time step
  double current_dt;

  // Log of the last N time steps, used to compute BDF coefficients
  std::vector<double> time_steps;

  // Time stepping scheme
  Parameters::TimeIntegration::Scheme scheme;

  // Number of previous solutions required to compute time derivatives.
  // For BDF schemes, this is equal to the BDF order.
  unsigned int n_previous_solutions;

  // If the time stepping scheme is a BDF method, this vector stores the
  // coefficients of the BDF expansion.
  std::vector<double> bdf_coefficients;

  // If the time stepping scheme is a BDF method, the order of the method
  unsigned int bdf_order;

  // A flag specifying whether the simulation uses adaptive time steps
  bool with_adaptive_timestep;

  // This value is true if the current step is a step that has been "rolled
  // back", that is, that is being re-tried with a smaller time step because the
  // nonlinear solver failed to find a solution during the previous attempt.
  bool rolledback_step;

  // The number of time steps that were rejected consecutively.
  // The simulation will abort if too many steps are rejected in a row, to avoid
  // progressing with very small time steps.
  unsigned int n_consecutive_rejected_steps;

  // Total number of time steps that were rejected during the simulation.
  unsigned int n_rejected_steps;

  // This value is true if the last nonlinear solve found a solution for this
  // time step, and is false otherwise (i.e., if the solver either diverged or
  // ran out of iterations without converging).
  mutable bool last_nonlinear_solver_converged;

  // The maximum CFL number computed for the current mesh and solution
  double max_cfl_number;

  // A pointer to the error estimator, used to compute BDF truncation error and
  // provide the next time step based on the error on all fields.
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
  ar &with_adaptive_timestep;
  ar &steady_scheme;
}

#endif
