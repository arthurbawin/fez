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
 *  h_{n+1} \approx \left(\frac{\epsilon}{e_n}\right)^{\frac{1}{p+1}}.
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
  double compute_next_timestep_from_error_estimator(
    const TimeHandler                    &time_handler,
    const ComponentOrdering              &ordering,
    const LA::ParVectorType              &present_solution,
    const std::vector<LA::ParVectorType> &previous_solutions,
    const IndexSet                       &locally_relevant_dofs,
    const std::vector<unsigned char>     &dofs_to_component);

private:
  const Parameters::TimeIntegration &time_parameters;

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
};

#endif
