#ifndef METRIC_FIELD_PARAMETERS_H
#define METRIC_FIELD_PARAMETERS_H

using namespace dealii;

#include <metric_tensor.h>
#include <parameters.h>
#include <parsed_function_symengine.h>

namespace Parameters
{
  /**
   * A single metric tensor field for anisotropic mesh adaptation
   */
  template <int dim>
  class MetricField
  {
  public:
    Verbosity verbosity;

    // Min and max allowed mesh size along any principal direction
    double min_meshsize;
    double max_meshsize;

    // Associated min eigenvalue = 1. / max_meshsize^2
    //        and max eigenvalue = 1. / min_meshsize^2
    double min_eigenvalue;
    double max_eigenvalue;

    struct AnalyticalMetric
    {
      bool enable;

      // The metric components are represented with a ParsedFunction, which
      // supports automatic differentiation for the gradient.
      // The Christoffel symbols require only the first metric derivatives, so
      // this is enough for testing, but maybe we can switch to symbolic
      // derivatives at some point if needed.
      std::shared_ptr<Functions::ParsedFunction<dim>> callback;
    } analytical_metric;

    struct Gradation
    {
      bool enable;
      // Specify if gradation should be done in a deterministic way
      // TODO: Add comments
      bool deterministic;
      // Prescribed gradation value (geometric size progression)
      double gradation;
      // Max number of passes when smoothing a metric field
      unsigned int max_iterations;
      // Tolerance : if the difference between any two metrics is lower than
      // this value between two passes, stop smoothing.
      double tolerance;
      // Space in which a single metric spans a full metric field
      typename MetricTensor<dim>::SpanningSpace spanning_space;
    } gradation;

    /**
     * The analytical scalar field used to create the metric field, if required.
     *
     * A metric field can be constructed from either an anisotropic error
     * estimate of the numerical solution, or from the analytical derivatives of
     * this field (for prototyping and debug purposes mostly).
     */
    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      analytical_field;

  public:
    /**
     * Constructor
     */
    MetricField();

    /**
     * Set newtime as the current time in the callbacks
     */
    void set_time(const double newtime);

    /**
     *
     */
    void declare_parameters(ParameterHandler  &prm,
                            const unsigned int index) const;

    /**
     *
     */
    void read_parameters(ParameterHandler &prm, const unsigned int index);
  };

  /**
   *
   */
  template <int dim>
  void declare_metric_fields(ParameterHandler  &prm,
                             const unsigned int n_metric_fields);

  /**
   *
   */
  template <int dim>
  void
  read_metric_fields(ParameterHandler              &prm,
                     const unsigned int             n_metric_fields,
                     std::vector<MetricField<dim>> &metric_fields_parameters);
} // namespace Parameters

/* ---------------- template and inline functions ----------------- */

template <int dim>
void Parameters::declare_metric_fields(ParameterHandler  &prm,
                                       const unsigned int n_metric_fields)
{
  const MetricField<dim> dummy_metric_field;
  prm.enter_subsection("Metric tensor fields");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of metric tensor fields");
    for (unsigned int i = 0; i < n_metric_fields; ++i)
      dummy_metric_field.declare_parameters(prm, i);
  }
  prm.leave_subsection();
}

template <int dim>
void Parameters::read_metric_fields(
  ParameterHandler                          &prm,
  const unsigned int                         n_metric_fields,
  std::vector<Parameters::MetricField<dim>> &metric_fields_parameters)
{
  prm.enter_subsection("Metric tensor fields");
  {
    // Number of fields was already parsed
    for (unsigned int i = 0; i < n_metric_fields; ++i)
      metric_fields_parameters[i].read_parameters(prm, i);
  }
  prm.leave_subsection();
}

#endif
