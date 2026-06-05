#ifndef METRIC_FIELD_PARAMETERS_H
#define METRIC_FIELD_PARAMETERS_H

#include <metric_tensor.h>
#include <parameters.h>
#include <parsed_function_symengine.h>
#include <solver_info.h>

namespace Parameters
{
  using namespace dealii;

  /**
   * Generic parameters related to metric fields.
   */
  struct MetricFields
  {
    /**
     * Id of the metric field used to adapt the mesh(es)
     */
    unsigned int metric_for_adaptation;

    /**
     * This flags specifies if the metric fields should be computed even if mesh
     * adaptation is disabled (for debug mostly).
     */
    bool always_compute;

    void declare_parameters(ParameterHandler &prm) const;
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Parameters for a single metric field.
   */
  template <int dim>
  class MetricField
  {
  public:
    Verbosity verbosity;
    unsigned int mesh_quality_output_frequency;
    std::string  mesh_quality_output_name;

    // The unique index of this field, in [0, n_metric_fields)
    unsigned int id;

    // A flag specifying if these parameters are the ones used to adapt the mesh
    // Several metric fields can be defined, then the intersection of these can
    // be chosen as the metric for adaptation, for instance.
    bool use_for_adaptation;

    /**
     * The "type" of metric that should be computed. For now only two choices:
     *
     * - "interpolation_error" refers to the optimal multiscale metric that
     *   minimizes in W^{s,p} norm an interpolation error estimate based on
     *   Taylor remainders. This metric is currently used for most mesh
     * adaptation applications.
     *
     * - "graph" refers to the the metric induced by the graph of a function
     *   (x, f(x)) with x in R^dim. This metric is simply given by
     *
     *   [M] = I + grad(f) \otimes grad(f), with I the identity tensor.
     *
     *   This metric does not hold any information regarding an error estimate,
     *   and thus cannot be used as-is for mesh adaptation. It is currently used
     *   to measure the quality of a deformed mesh with respect to a scalar
     * field.
     */
    enum class MetricType
    {
      interpolation_error,
      graph
    } type;

    // The variable from which the underlying metric field is computed
    SolverInfo::VariableType variable;

    // Metric fields are only defined from a scalar field.
    // If variable is vector-valued, the chosen vector component of the variable
    // above.
    unsigned int component;

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

    /**
     * The analytical scalar field used to create the metric field, if required.
     *
     * A metric field can be constructed from either an anisotropic error
     * estimate of the numerical solution, or from the analytical derivatives of
     * this field (for prototyping and debug purposes mostly).
     */
    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      analytical_field;

    /**
     * Parameters for the computation of the optimal multiscale metric
     * minimizing the W^{s,p} norm of the interpolation error (e.g., L^p norm
     * for s = 0, H^1 seminorm for s = 1, p = 2). This metric is the one
     * described by F. Alauzet & A. Loseille [ref], as well as J.-M. Mirebeau
     * [ref].
     *
     * These parameters are relevant only if type is "interpolation_error".
     */
    struct MultiscaleMetric
    {
      /**
       * The Sobolev norm for which the interpolation error is minimized.
       */
      enum TargetNorm
      {
        L1_norm,
        L2_norm,
        L4_norm,
        Linfty_norm,
        H1_seminorm
      } target_norm;

      /**
       * Return a string version of the target norm.
       */
      static std::string to_string(const TargetNorm norm)
      {
        switch (norm)
        {
          case TargetNorm::L1_norm:
            return "L1";
          case TargetNorm::L2_norm:
            return "L2";
          case TargetNorm::L4_norm:
            return "L4";
          case TargetNorm::Linfty_norm:
            return "Linfty";
          case TargetNorm::H1_seminorm:
            return "H1 seminorm";
        }
        DEAL_II_ASSERT_UNREACHABLE();
        return "";
      }

      // Target s and p of the target W^{s,p} norm above (e.g., L^2 = W^{0,2}).
      unsigned int s;
      unsigned int p;

      // Target number of mesh vertices after adaptation (without gradation)
      unsigned int n_target_vertices;

      // If true, use the derivatives of "analytical_field" above to compute
      // the anisotropic measure of the interpolation error estimate.
      // If false, use reconstructed derivatives.
      bool use_analytical_derivatives;
    } multiscale;

    /**
     * Parameters controlling the gradation (size limitation or smoothing)
     * of the metric field.
     */
    struct Gradation
    {
      Verbosity verbosity;

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
     * Parameters specifying whether this field is the intersection of one or
     * more other fields.
     */
    struct Intersection
    {
      Verbosity verbosity;

      // List of metric field ids with which this metric will be intersected
      std::vector<unsigned int> intersect_with;
    } intersection;

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
     * Declare the parameters in the ParameterHandler for the index-th metric
     * field
     */
    void declare_parameters(ParameterHandler  &prm,
                            const unsigned int index) const;

    /**
     * Read the parameters from the ParameterHandler for the index-th metric
     * field
     */
    void read_parameters(ParameterHandler &prm, const unsigned int index);
  };

  /**
   * Declare all @p n_metric_fields metric fields.
   */
  template <int dim>
  void declare_metric_fields(ParameterHandler  &prm,
                             const unsigned int n_metric_fields);

  /**
   * Read all @p n_metric_fields metric fields.
   */
  template <int dim>
  void
  read_metric_fields(ParameterHandler              &prm,
                     const unsigned int             n_metric_fields,
                     std::vector<MetricField<dim>> &metric_fields_parameters,
                     unsigned int                  &metric_for_adaptation);
} // namespace Parameters

/* ---------------- template and inline functions ----------------- */

namespace Parameters
{
  template <int dim>
  void declare_metric_fields(ParameterHandler   &prm,
                             const unsigned int  n_metric_fields,
                             const MetricFields &metrics_parameters)
  {
    const MetricField<dim> dummy_metric_field;
    prm.enter_subsection("Metric tensor fields");
    {
      metrics_parameters.declare_parameters(prm);
      for (unsigned int i = 0; i < n_metric_fields; ++i)
        dummy_metric_field.declare_parameters(prm, i);
    }
    prm.leave_subsection();
  }

  template <int dim>
  void
  read_metric_fields(ParameterHandler              &prm,
                     const unsigned int             n_metric_fields,
                     MetricFields                  &metrics_parameters,
                     std::vector<MetricField<dim>> &metric_fields_parameters)
  {
    prm.enter_subsection("Metric tensor fields");
    {
      metrics_parameters.read_parameters(prm);

      if (n_metric_fields > 0)
        AssertThrow(metrics_parameters.metric_for_adaptation < n_metric_fields,
                    ExcMessage(
                      "You specified that the mesh(es) should be "
                      "adapted with the metric field with index " +
                      std::to_string(metrics_parameters.metric_for_adaptation) +
                      ", but this index is not in the half-open "
                      "interval [0, n_metric_fields)."));

      for (unsigned int i = 0; i < n_metric_fields; ++i)
      {
        metric_fields_parameters[i].id = i;
        metric_fields_parameters[i].use_for_adaptation =
          (i == metrics_parameters.metric_for_adaptation);
        metric_fields_parameters[i].read_parameters(prm, i);

        for (auto other_id :
             metric_fields_parameters[i].intersection.intersect_with)
          AssertThrow(other_id < n_metric_fields,
                      ExcMessage("Metric field " + std::to_string(i) +
                                 " should be intersected with metric field " +
                                 std::to_string(other_id) +
                                 ", but this index is not in the half-open "
                                 "interval [0, n_metric_fields)."));
      }
    }
    prm.leave_subsection();
  }
} // namespace Parameters

#endif
