#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>
#include <parameters.h>
#include <types.h>
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
  *
  * This overload is the common (non-template) implementation used by all
  * solvers in the code base (Navier-Stokes, Heat, etc.), which rely on the
  * same parallel vector type.
  */
void compute_vautrin_error_estimate(LA::ParVectorType                    &e_star,
                                    const LA::ParVectorType              &u_np1,
                                    const std::vector<LA::ParVectorType> &previous_solutions,
                                    const unsigned int                    order) const;

/** Compute the scaled Vautrin ratio for a given component q:
  *   R_q = epsilon_q / err_q
  * where err_q is computed as:
  *   - absolute RMS error if ||u||_RMS < u_seuil,
  *   - relative RMS error otherwise.
  *
  * The component is determined with dofs_to_component (global DoF -> component).
  */
double compute_scaled_vautrin_ratio(
  const LA::ParVectorType              &e_star,
  const LA::ParVectorType              &u_star,
  const std::vector<unsigned char>     &dofs_to_component,
  const unsigned int                    component_q,
  const double                          epsilon_q,
  const double                          u_seuil,
  const MPI_Comm                        comm) const;

/** Compute dt_{n+1} from dt_n and the global ratio R = min_q R_q. */
static double propose_next_dt_vautrin(const double       dt,
                                     const double       R,
                                     const unsigned int order,
                                     const double       safety,
                                     const double       ratio_min,
                                     const double       ratio_max,
                                     const double       dt_min,
  const double       dt_max);

/** High-level adaptive time-step update (Vautrin-based). This is intended to
  * be called by any solver after a *converged* time step, *before* rotating
  * the solution history vectors.
  *
  * Inputs:
  *  - u_np1: converged solution at t_{n+1}
  *  - previous_solutions_dt_control: history vectors [u_n, u_{n-1}, ...]
  *    (size must be >= order+1)
  *  - dofs_to_component: global DoF -> component index
  *  - component_eps: list of (component, epsilon) pairs used for dt control
  */
bool update_dt_after_converged_step_vautrin(
  const LA::ParVectorType                               &u_np1,
  const std::vector<LA::ParVectorType>                  &previous_solutions_dt_control,
  const std::vector<unsigned char>                      &dofs_to_component,
  const std::vector<std::pair<unsigned int, double>>    &component_eps,
  const double                                           u_seuil,
  const double                                           safety,
  const double                          dt_min_factor,
  const double                          dt_max_factor,
  const MPI_Comm                                         comm,
  const double                                           t_end,
  const bool                                             clamp_to_t_end = true,
  // optional outputs (for logging/debug)
  double                                                *out_R = nullptr,
  unsigned int                                          *out_order = nullptr,
  double                                                *out_dt_used = nullptr,
  double                                                *out_dt_next = nullptr,
  LA::ParVectorType                                     *out_e_star = nullptr) ;
/** Override the time integration parameters (e.g. for restarting a simulation
   * with different time step). This will update the time integration scheme and
   * recompute the BDF coefficients if needed.
   */
  void apply_restart_overrides(const Parameters::TimeIntegration &new_params);

  std::deque<double> pending_dt_queue; // dts "programmés" (rampe restart)

  // ------------------------------------------------------------
  // Programmed dt schedules (paper-style variable-step tests)
  // ------------------------------------------------------------
  bool                use_programmed_dt_schedule = false;
  std::vector<double> programmed_dt;

  void build_programmed_dt_schedule();




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

  double              dt_ref_bounds;

  Parameters::TimeIntegration::Scheme scheme;

  unsigned int        n_previous_solutions;
  std::vector<double> bdf_coefficients;

    
  double get_effective_dt_min() const;
  double get_effective_dt_max() const;
};

/* ---------------- Template functions ----------------- */

template <typename VectorType>
void TimeHandler::rotate_solutions(const VectorType        &present_solution,
                                  std::vector<VectorType> &previous_solutions) const
{
  if (this->is_steady() || previous_solutions.empty())
    return;

  // Décalage: [0] <- present, [1] <- old[0], [2] <- old[1], ...
  for (std::size_t j = previous_solutions.size(); j-- > 1; )
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

  ar & initial_dt;
  ar & dt_ref_bounds;
}





#endif