#ifndef METRIC_TENSOR_H
#define METRIC_TENSOR_H

#include <deal.II/base/tensor.h>
#include <deal.II/grid/tria.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace dealii;

// Eigen real and complex matrix types in 2D and 3D
template <int dim>
struct EigenRealMatrix;

template <>
struct EigenRealMatrix<2>
{
  using type = Eigen::Matrix2d;
};

template <>
struct EigenRealMatrix<3>
{
  using type = Eigen::Matrix3d;
};

template <int dim>
struct EigenComplexMatrix;

template <>
struct EigenComplexMatrix<2>
{
  using type = Eigen::Matrix2cd;
};

template <>
struct EigenComplexMatrix<3>
{
  using type = Eigen::Matrix3cd;
};

DeclExceptionMsg(ExcNotSPD,
                 "You are performing an operation on a metric tensor, which "
                 "should be symmetric and positive-definite (SPD) by "
                 "definition, but the underlying tensor is not SPD.");

DeclExceptionMsg(
  ExcOperationMadeNotSPD,
  "You performed an operation on a metric tensor with an initial SPD "
  "underlying tensor which resulted in a non-SPD tensor.");

DeclExceptionMsg(
  ExcAssignFromNotSPD,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from a non-SPD SymmetricTensor.");

DeclExceptionMsg(
  ExcAssignFromNotSPDArray,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from an array that "
  "represents a non-SPD SymmetricTensor. You should maybe double-check "
  "the ordering of the entries of the array.");

#define AssertIsSPD(metric_tensor) \
  Assert(dealii::determinant(metric_tensor) > 0, ExcNotSPD())

#define AssertAssignFromSPD(symmetric_tensor) \
  Assert(dealii::determinant(symmetric_tensor) > 0, ExcAssignFromNotSPD())

#define AssertAssignFromSPDArray(metric_tensor) \
  Assert(dealii::determinant(metric_tensor) > 0, ExcAssignFromNotSPDArray())

/**
 * A metric tensor is a positive-definite SymmetricTensor<2, dim, double>, with
 * additional member functions.
 */
template <int dim>
class MetricTensor : public SymmetricTensor<2, dim>
{
  static constexpr unsigned int n_independent_components =
    SymmetricTensor<2, dim>::n_independent_components;

public:
  /**
   * Constructor
   */
  MetricTensor(
    const SymmetricTensor<2, dim> &other = unit_symmetric_tensor<dim>());

  /**
   * Constructor
   */
  MetricTensor(const double (&array)[n_independent_components]);

  /**
   * Copy constructor
   */
  MetricTensor(const MetricTensor<dim> &other);

  /**
   * Assignment operator from a MetricTensor.
   * In debug, this also checks that m is SPD.
   */
  constexpr MetricTensor<dim> &operator=(const MetricTensor<dim> &m);

  /**
   * Assignment operator from a SymmetricTensor, assumed to be SPD.
   * This operator simply checks that t is in fact SPD (in debug only), then
   * calls the assignment operator for SymmetricTensors.
   */
  constexpr MetricTensor<dim> &operator=(const SymmetricTensor<2, dim> &t);

  /**
   * Assignment operator from an array with N*(N+1)/2 components.
   * As for SymmetricTensors, this requires knowing the layout of the indices,
   * which can be obtained through the unrolled_index() function.
   */
  constexpr MetricTensor<dim> &operator=(const double (&array)[n_independent_components]);

  /**
   * Limit the eigenvalues to lambdaMin and lambdaMax
   */
  void bound_eigenvalues(const double lambdaMin, const double lambdaMax);

  /**
   *
   */
  MetricTensor<dim> log() const;

  /**
   *
   */
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
  MetricTensor<dim> span_metric(const double          gradation,
                                const Tensor<1, dim> &pq) const;

private:
  // Matrix representation of the underlying tensor in Eigen format.
  // This is *not* the matrix of the eigendecomposition of this metric tensor.
  EigenRealMatrix<dim> matrix_eigen;
};

/* ---------------- template and inline functions ----------------- */

template <int dim>
MetricTensor<dim>::MetricTensor(const SymmetricTensor<2, dim> &other)
  : SymmetricTensor<2, dim>(other)
{
  std::cout << "Assigning from St with det = " << determinant(other) << std::endl;
  AssertAssignFromSPD(other);
}

template <int dim>
MetricTensor<dim>::MetricTensor(const double (&array)[n_independent_components])
  : SymmetricTensor<2, dim>(array)
{
  AssertAssignFromSPDArray(*this);
}

template <int dim>
constexpr inline MetricTensor<dim> &
MetricTensor<dim>::operator=(const MetricTensor<dim> &m)
{
  AssertAssignFromSPD(m);
  SymmetricTensor<2, dim>::operator=(m);
  return *this;
}

template <int dim>
constexpr inline MetricTensor<dim> &
MetricTensor<dim>::operator=(const SymmetricTensor<2, dim> &t)
{
  AssertAssignFromSPD(t);
  SymmetricTensor<2, dim>::operator=(t);
  return *this;
}

template <int dim>
constexpr inline MetricTensor<dim> &
MetricTensor<dim>::operator=(const double (&array)[n_independent_components])
{
  if constexpr (dim == 2)
  {
    (*this)[0][0] = array[0];
    (*this)[1][1] = array[1];
    (*this)[0][1] = array[2];
  }
  else
  {
    (*this)[0][0] = array[0];
    (*this)[1][1] = array[1];
    (*this)[2][2] = array[2];
    (*this)[0][1] = array[3];
    (*this)[0][2] = array[4];
    (*this)[1][2] = array[5];
  }
  AssertAssignFromSPDArray(*this);
  return *this;
}

#endif
