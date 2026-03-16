#ifndef TIMESTEP_ADAPTATION_H
#define TIMESTEP_ADAPTATION_H

#include <components_ordering.h>
// #include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
// #include <deal.II/base/types.h>
#include <parameters.h>
#include <solver_info.h>
#include <time_handler.h>
#include <types.h>
#include <utilities.h>

using namespace dealii;

/**
 * The purpose of this class is to compute the next time step of an unsteady
 * simulation based on the trunaction error of the BDF time stepping methods.
 * For a method of order p, the truncation error of order p + 1 writes :
 *
 * e \simeq (-1)^{p+1} h (\sum_{i=0}^p \alpha_i H_i^{p+1}) [u^{n+1}, \ldots,
 * u^{n+1-(p+1)}],
 *
 * where h is the current time step, the \alpha_i are the BDF coefficients,
 * H_i := t_{n+1} - t_{n+1-i} and the brackets [., ..., .] are the (forward)
 * divided differences of the solution u at times n+1 to n+1-(p+1).
 *
 * This error estimate is combined with the a priori relation e_n \propto
 * h_n^{p+1} to obtain an estimation of the next time step :
 *
 *  \frac{e_{n+1}}{e_n} \approx \left(\frac{h_{n+1}}{h_n}\right)^{p+1}.
 *
 * Letting \epsilon denote the target error e_{n+1}, this yield the time step
 *
 *  h_{n+1} \approx h_n * \left(\frac{\epsilon}{e_n}\right)^{\frac{1}{p+1}}.
 *
 * The estimation of e_n requires an additional solution vector, which is
 * stored in this class.
 */
class BDFErrorEstimator
{
public:
  /**
   * Constructor.
   */
  BDFErrorEstimator(const Parameters::TimeIntegration &time_parameters,
                    const TimeHandler                 &time_handler);

  /**
   * Attach solver data to this object.
   * This function is called by attach_data_to_error_estimator in the
   * TimeHandler, which must be called once before computing the first error
   * estimates.
   */
  void attach_data(const ComponentOrdering          &ordering,
                   const IndexSet                   &locally_relevant_dofs,
                   const std::vector<unsigned char> &dofs_to_component);

  /**
   * Rotate the stored simulation times t_{n+1} through t_{n+1-(p+1)}.
   */
  void advance(const TimeHandler &time_handler);

  /**
   * Rotate the solution vector u^{n+1-(p+1)}, that is, perform the assignment
   * u^{n+1-(p+1)} := u^{n+1-p}. @p solution should be u^{n+1-p}.
   */
  void rotate_additional_solution(const LA::ParVectorType &solution);

  /**
   * Compute the next time step based on the truncation error estimate.
   * The maximum error is computed over all dofs and stored independently for
   * each variable. The next time step is set as the most critical time step
   * according to the target error for each variable.
   *
   * This time step is clamped by the TimeHandler to respect the user bounds
   * (min/max time step, max increase/decrease between two time steps).
   */
  void compute_error_estimator(
    const TimeHandler                    &time_handler,
    const LA::ParVectorType              &present_solution,
    const std::vector<LA::ParVectorType> &previous_solutions);

  /**
   * Return the maximum error estimate for each variable.
   */
  const std::map<SolverInfo::VariableType, double> &get_max_errors() const;

  /**
   * Compute and return the next time step based on the stored error estimates.
   * This time step is:
   *
   * h_{n+1} = h_n * \left(\frac{\epsilon}{e_n}\right)^{\frac{1}{p+1}},
   *
   * where e_n is the computed truncation error and h_n is @p current_timestep.
   */
  double get_next_timestep(const double current_timestep) const;

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

private:
  const Parameters::TimeIntegration &time_parameters;

  // Non-owning pointers
  const ComponentOrdering          *ordering              = nullptr;
  const IndexSet                   *locally_relevant_dofs = nullptr;
  const std::vector<unsigned char> *dofs_to_component     = nullptr;

  // Order p of the used BDF method
  unsigned int bdf_order;

  // Store here the additional vector needed to compute the truncation error for
  // the BDF scheme of order p + 1.
  // Alternatively, we could increment n_previous_solutions by 1 in the time
  // handler, but that would mean that we can no longer use
  // previous_solutions.size() whenever time adaptation is enabled...
  LA::ParVectorType additional_solution;

  const unsigned int n_previous_solutions;

  // Simulation times at step n+1 through n+1-(p+1)
  std::vector<double> simulation_times;

  // Variables present in the solution vector
  std::vector<SolverInfo::VariableType> handled_variables;

  // For each variable, the maximum error over the dofs of this variable
  std::map<SolverInfo::VariableType, double> max_error;

  // Tracks whether a full vector of estimator should be stored
  bool save_full_error_estimator;

  // The IndexSet of locally owned dofs, needed to initialize the vectors below
  IndexSet locally_owned_elements;

  // The ghosted and fully distributed vectors of error estimator at each dof
  LA::ParVectorType error_estimator;
  LA::ParVectorType fully_distributed_error_estimator;
};

#endif
