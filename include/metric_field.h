#ifndef METRIC_FIELD_h
#define METRIC_FIELD_h

#include <deal.II/base/tensor_function.h>
#include <metric_tensor.h>
#include <parameter_reader.h>
#include <parameters.h>

using namespace dealii;

/**
 * A metric field is a discrete field of MetricTensors, stored at the owned mesh
 * vertices.
 *
 * The field is tied to a mesh for e.g. computing the integral of the
 * determinant, smoothing, etc.
 */
template <int dim>
class MetricField
{
public:
  /**
   * Constructor.
   */
  MetricField(const ParameterReader<dim> &param,
              const Triangulation<dim>   &triangulation);

  /**
   * Copy constructor. Deleted to avoid copying by mistake.
   */
  MetricField(const MetricField<dim> &other) = delete;

  /**
   * Set the metrics of this field to the analytical field described by @p function.
   */
  void set_metrics_from_function(const TensorFunction<2, dim> &function);

  /**
   * Compute the metric field from either error on solution
   * or given analytical derivatives.
   */
  void compute_metrics();

  /**
   * Return the vector of MetricTensors constituting this field.
   */
  const std::vector<MetricTensor<dim>> &get_metrics() const { return metrics; }

  /**
   * Apply gradation (smoothing) to the metric field.
   */
  void metric_gradation(const double       gradation,
                        const unsigned int max_iterations,
                        const double       tolerance);

  /**
   * Intersect all the metrics in this field with the metrics in @p other.
   * The number of metric tensors in both fields must match, but aside from
   * this, no other safety check is performed.
   */
  void intersect_with(const MetricField<dim> &other);

  /**
   * Compute integral on mesh of metric determinant.
   */
  double compute_integral_determinant() const;

  /**
   * Write metric field to .vtu file.
   *
   * Note: In ParaView, the Tensor Glyph filter scales the tensor field
   * with the eigenvalues. If we want to visualize the unit ellipsoids,
   * then we should export M^(-1).
   */
  void writeToVTU(const std::string &filename,
                  bool               export_inverse_metrics = true) const;

private:
  /**
   * Compute metric tensors assuming a linearly interpolated field.
   * Optimal metric is the scaled absolute Hessian matrix.
   */
  void compute_metrics_P1();

private:
  const ParameterReader<dim> &param;
  const Triangulation<dim>   &triangulation;

  // Number of mesh vertices on this partition
  unsigned int n_vertices;

  // The stored metric tensors
  std::vector<MetricTensor<dim>> metrics;
};

#endif
