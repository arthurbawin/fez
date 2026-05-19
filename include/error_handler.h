#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <deal.II/base/convergence_table.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <mesh_adaptation/transient_fixed_point.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <time_handler.h>

using namespace dealii;

/**
 * A thin wrapper to the ConvergenceTable for error computations.
 */
class ErrorHandler
{
public:
  /**
   * Constructor.
   */
  ErrorHandler(const Parameters::Mesh            &mesh_param,
               const Parameters::MMS             &mms_parameters,
               const Parameters::TimeIntegration &time_parameters);

  /**
   * Create an entry in the error table to store an error norm in space and/or
   * in time for a given field.
   */
  void create_entry(const std::string &field_name);

  /**
   * Add all required reference data (number of mesh elements, vertices, etc.)
   * for this convergence study.
   */
  template <int dim>
  void add_reference_data(
    const TimeHandler                  &time_handler,
    const TransientFixedPointData<dim> &transient_fixed_point_data,
    const Triangulation<dim>           &triangulation,
    const DoFHandler<dim>              &dof_handler);

  /**
   * Add an integer reference value (number of mesh elements or dof).
   * These values won't be printed in scientific notation.
   */
  void add_reference_data(const std::string &name, const unsigned int value);

  /**
   * Add the (constant) time step used for this step of convergence study.
   */
  void add_time_step(double time_step);

  /**
   * Add a spatial error entry.
   *
   * If the simulation is steady, this error is directly added to
   * the underlying error table to compute convergence.
   *
   * If it is unsteady, this stores the spatial error at time t. The prescribed
   * L^p norm in time is computed at the end of the simulation.
   * The error at all times are kept, e.g., to be plotted in postprocessing.
   */
  void add_error(const std::string &field_name,
                 const double       error_val,
                 const double       time = 0.);

  /**
   * Return the unsteady errors (t, e(t)) for the given field.
   */
  const std::vector<std::pair<double, double>> &
  get_unsteady_errors(const std::string &field_name) const;

  /**
   * Write the errors for all stored fields.
   * If simulation is unsteady, print the errors for each time step.
   */
  void write_errors(std::ostream      &out       = std::cout,
                    const unsigned int precision = 6) const;

  /**
   * Compute the temporal or spacetime error if needed
   */
  void compute_temporal_error();

  /**
   * Clear the t - error(t) table, e.g. in between two convergence steps.
   */
  void clear_error_history();

  /**
   * Compute the convergence rates with respect to the number of elements,
   * number of dofs, time step or number of time steps, depending on the
   * type of convergence study being run.
   */
  template <int dim>
  void compute_rates();

  /**
   * Write the convergence table to the given stream.
   * Simply forward the call to the underlying ConvergenceTable.
   */
  void write_rates(std::ostream &out = std::cout) const;

private:
  /**
   * Add an error to the underlying error table to compute convergence.
   */
  void add_steady_error(const std::string &error_name, const double error_val);

  /**
   * Store a spatial error at time t for an unsteady simulation.
   */
  void add_unsteady_error(const std::string &error_name,
                          const double       error_val,
                          const double       time);

public:
  const Parameters::Mesh &mesh_param;

  const Parameters::MMS &mms_param;

  const Parameters::TimeIntegration &time_param;

  bool is_steady;

  bool with_metric_based_adaptation;

  // For unsteady problems: keep the spatial errors at all time steps
  // For each field, a vector of (t, error(t)) pairs.
  std::map<std::string, std::vector<std::pair<double, double>>> unsteady_errors;

  ConvergenceTable error_table;

  // Use vector of keys to maintain prescribed errors order
  std::vector<std::string> ordered_field_keys;

  std::map<std::string, std::unique_ptr<double>> domain_errors;
};

/* ---------------- Template functions ----------------- */

#endif
