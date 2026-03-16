#ifndef METRIC_TENSOR_H
#define METRIC_TENSOR_H

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include <vector>

using namespace dealii;

/**
 * A metric tensor is a SymmetricTensor<2, dim, double>
 * with additional member functions.
 *
 * Positive-definiteness is enforced when assigning a
 * complete matrix to a MetricTensor instance, but not
 * when using the individual entry accessors [i][j].
 */
template <int dim>
class MetricTensor : public SymmetricTensor<2, dim, double>
{
  static_assert(dim == 2 || dim == 3,
                "MetricTensor is only defined for dim = 2, 3.");

  //
  // Maybe a metric could know to which Point<dim> it is associated?
  //

public:
  MetricTensor() = default;
  MetricTensor(const SymmetricTensor<2, dim> &other)
    : SymmetricTensor<2, dim>(other)
  {
    if (!(determinant(*this) > 0))
      throw std::runtime_error(
        "Cannot create a MetricTensor from a non SPD matrix!");
  }

  /**
   * Limit the eigenvalues to lambdaMin and lambdaMax
   */
  void boundEigenvalues(const double lambdaMin, const double lambdaMax);

  MetricTensor<dim> log() const;
  MetricTensor<dim> exp() const;

  /**
   * Compute metric intersection of this metric with other.
   * Checks first if metrics are diagonal or multiple of one another.
   * If not, compute their simultaneousReduction.
   */
  MetricTensor<dim> intersection(const MetricTensor<dim> &other) const;

  /**
   * Span a metric at distance pq with prescribed gradation.
   */
  MetricTensor<dim> spanMetric(const double          gradation,
                               const Tensor<1, dim> &pq) const;

  /**
   * Assign a vector with N*(N+1)/2 components to the metric.
   */
  constexpr MetricTensor<dim> &operator=(const Vector<double> &vec)
  {
    constexpr unsigned int n_components = dim * (dim + 1) / 2;

    AssertThrow(vec.size() == n_components,
                ExcDimensionMismatch(vec.size(), n_components));

    if constexpr (dim == 2)
    {
      (*this)[0][0] = vec[0];
      (*this)[1][1] = vec[1];
      (*this)[0][1] = vec[2];
    }
    else
    {
      // TODO: Check ordering
      (*this)[0][0] = vec[0];
      (*this)[1][1] = vec[1];
      (*this)[2][2] = vec[2];
      (*this)[0][1] = vec[3];
      (*this)[0][2] = vec[4];
      (*this)[1][2] = vec[5];
    }

    if (!(determinant(*this) > 0))
      throw std::runtime_error(
        "Cannot create a MetricTensor from a non SPD matrix!");

    return *this;
  }
};

/**
 * Take the absolute value P * |D| * P^T of the symmetric matrix stored
 * in vec with N*(N+1)/2 components.
 */
template <int dim>
MetricTensor<dim> absoluteValue(const Vector<double> &vec);

#endif
