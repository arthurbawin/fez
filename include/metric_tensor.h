#ifndef METRIC_TENSOR_H
#define METRIC_TENSOR_H

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
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
   * Constructor. Generate a MetricTensor from a SymmetricTensor. Assumes that
   * @p t is already positive-definite (it has positive determinant), and in
   * debug mode this is in fact checked.
   *
   * Unlike SymmetricTensors, there is no constructor to create a default metric
   * tensor with arbitrary or zero values, since the tensor should be SPD.
   */
  MetricTensor(const SymmetricTensor<2, dim> &t = unit_symmetric_tensor<dim>());

  /**
   * A constructor that creates a symmetric tensor from an array holding its
   * independent N*(N+1)/2 components. Identical to the one for
   * SymmetricTensors, but here it additionally checks for postiive-definiteness
   * of the input array. The input array is expected to be in the correct
   * indices order, as returned by unrolled_index().
   */
  MetricTensor(const double (&array)[n_independent_components]);

  /**
   * Copy constructor.
   */
  MetricTensor(const MetricTensor<dim> &other);

  /**
   * Assignment operator from a MetricTensor.
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
  constexpr MetricTensor<dim> &
  operator=(const double (&array)[n_independent_components]);

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
  /**
   *
   */
  void compute_eigendecomposition();

private:
  // Matrix representation of the underlying tensor in Eigen format.
  // This is *not* the matrix of the eigendecomposition of this metric tensor.
  typename EigenRealMatrix<dim>::type matrix_eigen;

  Eigen::Matrix<double, dim, 1>   eigenvalues;
  Eigen::Matrix<double, dim, dim> eigenvectors;
};

/**
 * Re-arrange the dim*(dim+1)/2 components of a ParsedFunction into a rank-2
 * TensorFunction representing a Riemannian metric. The components of the
 * ParsedFunction should match the layout of a SymmetricTensor, that is :
 *
 *  - in 2D : xx; yy; xy
 *  - in 3D : xx; yy; zz; xy; xz; yz
 *
 * These components are assumed in represent an SPD matrix, which is checked in
 * debug mode.
 */
template <int dim>
class MetricFunctionFromComponents : public TensorFunction<2, dim>
{
public:
  MetricFunctionFromComponents(
    const Functions::ParsedFunction<dim> &parsed_function)
    : parsed_function(parsed_function)
  {}

  using metric_tensor_type = SymmetricTensor<2, dim>;

  DeclException3(ExcUserFunNotSPD,
                 Point<dim>,
                 metric_tensor_type,
                 double,
                 << "You are trying to assign a metric tensor, which should be "
                    "symmetric and positive-definite (SPD) by definition, from "
                    "an analytical metric given in the parameter file, but the "
                    "parsed function returned a tensor with nonpositive "
                    "determinant.\n\nThe provided function evaluated at point ["
                 << arg1 << "] returned the components " << arg2
                 << ", with det = " << arg3 << ".\n\n"
                 << "You should maybe double-check that (i) he components "
                    "of the metric tensor do indeed yield an SPD matrix, and "
                    "(ii) the components were given in the right order, which "
                    "is :\n\n in 2D : set Function expression = xx; yy; xy\n "
                    "in 3D : set Function expression = xx; yy; zz; xy; xz; yz");

  virtual Tensor<2, dim> value(const Point<dim> &p) const override
  {
    SymmetricTensor<2, dim> res;
    if constexpr (dim == 2)
    {
      res[0][0] = parsed_function.value(p, 0);
      res[1][1] = parsed_function.value(p, 1);
      res[0][1] = parsed_function.value(p, 2);
    }
    else
    {
      res[0][0] = parsed_function.value(p, 0);
      res[1][1] = parsed_function.value(p, 1);
      res[2][2] = parsed_function.value(p, 2);
      res[0][1] = parsed_function.value(p, 3);
      res[0][2] = parsed_function.value(p, 4);
      res[1][2] = parsed_function.value(p, 5);
    }
    if constexpr (running_in_debug_mode())
    {
      const double det = determinant(res);
      Assert(det > 0, ExcUserFunNotSPD(p, res, det));
    }
    return res;
  }

private:
  const Functions::ParsedFunction<dim> &parsed_function;
};

/* ---------------- template and inline functions ----------------- */

#endif
