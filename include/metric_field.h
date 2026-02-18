#ifndef METRIC_FIELD_h
#define METRIC_FIELD_h

#include <metric_tensor.h>

using namespace dealii;

/**
 * A metric field is a per-vertex field of MetricTensors.
 * The field is tied to a mesh for e.g. computing
 * the integral of the determinant, smoothing, etc.
 */
template <int dim>
class MetricField
{
public:
  const Triangulation<dim>      &triangulation;
  std::vector<MetricTensor<dim>> _metrics;

public:
  MetricField(const Triangulation<dim> &mesh);

  /**
   * Compute the metric field from either error on solution
   * or given analytical derivatives.
   */
  void computeMetrics();

  /**
   * Apply gradation (smoothing) to the metric field.
   */
  void metricGradation(const double       gradation,
                       const unsigned int maxIteration,
                       const double       tolerance);

  /**
   * Intersect all the metrics in this field with the metrics in otherField.
   * The number of metric tensors in both fields must match, but aside from
   * this, no other safety check is performed.
   */
  void intersectWith(const MetricField<dim> &otherField);

  /**
   * Compute integral on mesh of metric determinant.
   */
  double computeIntegralDeterminant() const;

  /**
   * Write metric field to .vtu file.
   *
   * Note: In ParaView, the Tensor Glyph filter scales the tensor field
   * with the eigenvalues. If we want to visualize the unit ellipsoids,
   * then we should export M^(-1).
   */
  void writeToVTU(const std::string &filename,
                  bool               exportInverseMetrics = true) const;

private:
  /**
   * Compute metric tensors assuming a linearly interpolated field.
   * Optimal metric is the scaled absolute Hessian matrix.
   */
  void computeMetricsP1();
};

#endif
