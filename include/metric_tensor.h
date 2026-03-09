#ifndef METRIC_TENSOR_H
#define METRIC_TENSOR_H

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>

#include <Eigen/Dense>

using namespace dealii;

// Eigen real and complex matrix types in 2D and 3D
template <int dim>
struct EigenReal;

template <>
struct EigenReal<2>
{
  using matrix_type = Eigen::Matrix2d;
  using vector_type = Eigen::Vector2d;
};

template <>
struct EigenReal<3>
{
  using matrix_type = Eigen::Matrix3d;
  using vector_type = Eigen::Vector3d;
};

template <int dim>
struct EigenComplex;

template <>
struct EigenComplex<2>
{
  using matrix_type = Eigen::Matrix2cd;
  using vector_type = Eigen::Vector2cd;
};

template <>
struct EigenComplex<3>
{
  using matrix_type = Eigen::Matrix3cd;
  using vector_type = Eigen::Vector3cd;
};

/**
 * Return true if t is positive definite using Sylvester's criterion
 */
template <int dim>
constexpr inline bool is_positive_definite(const SymmetricTensor<2, dim> &t);

template <>
constexpr inline bool is_positive_definite(const SymmetricTensor<2, 2> &t)
{
  return t[0][0] > 0. && determinant(t) > 0.;
}

template <>
constexpr inline bool is_positive_definite(const SymmetricTensor<2, 3> &t)
{
  return t[0][0] > 0. && (t[0][0] * t[1][1] - t[0][1] * t[0][1]) > 0. &&
         determinant(t) > 0.;
}

template <int dim>
constexpr inline bool is_orthonormal(const Tensor<2, dim> &t)
{
  return (t * transpose(t) - unit_symmetric_tensor<dim>()).norm() < 1e-14;
}

// DeclExceptionMsg(ExcNotSPD,
//                  "You are performing an operation on a metric tensor, which "
//                  "should be symmetric and positive-definite (SPD) by "
//                  "definition, but the underlying tensor is not SPD.");

DeclExceptionMsg(
  ExcOperationMadeNotSPD,
  "You performed an operation on a metric tensor with an initial SPD "
  "underlying tensor which resulted in a non-SPD tensor.");

DeclExceptionMsg(
  ExcAssignFromNotSPD,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from a SymmetricTensor that is not "
  "positive-definite.");

DeclExceptionMsg(
  ExcAssignFromNotSPDArray,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from an array that "
  "represents a SymmetricTensor that is not positive-definite. "
  "You should maybe double check the ordering of the entries of the array.");

DeclException1(
  ExcEigenvalueNotPositive,
  double,
  << "At leat one eigenvalue used to create a metric tensor is not positive: "
  << arg1);

DeclException1(ExcEigenvectorsNotOrthonormal,
               std::string,
               << "The set of eigenvectors used to create a metric tensor is "
                  "not orthonormal: "
               << arg1);

/**
 * #define AssertIsSPD(metric_tensor) \
 *   Assert(is_positive_definite(metric_tensor), ExcNotSPD())
 */

#define AssertAssignFromSPD(symmetric_tensor) \
  Assert(is_positive_definite(symmetric_tensor), ExcAssignFromNotSPD())

#define AssertAssignFromSPDArray(metric_tensor) \
  Assert(is_positive_definite(metric_tensor), ExcAssignFromNotSPDArray())

/**
 * A metric tensor is a positive-definite SymmetricTensor<2, dim, double>, with
 * additional member functions.
 */
template <int dim>
class MetricTensor : public SymmetricTensor<2, dim>
{
public:
  using EigenRealMatrix = typename EigenReal<dim>::matrix_type;
  using EigenRealVector = typename EigenReal<dim>::vector_type;

  static constexpr unsigned int n_independent_components =
    SymmetricTensor<2, dim>::n_independent_components;

public:
  /**
   * Constructor. Generate a MetricTensor from a SymmetricTensor. Assumes that
   * @p t is already positive-definite (it has all positive eigenvalues), and in
   * debug mode this is in fact checked.
   *
   * Unlike SymmetricTensors, there is no constructor to create a default metric
   * tensor with arbitrary or zero values, since the tensor should be SPD.
   */
  MetricTensor(const SymmetricTensor<2, dim> &t = unit_symmetric_tensor<dim>());

  /**
   * Constructor. Generate a MetricTensor from a Tensor. Assumes that @p t is
   * symmetric and positive-definite (it has all positive eigenvalues), and in
   * debug mode this is in fact checked.
   *
   * This constructor allows the creation of a metric tensor from its
   * eigendecomposition M = QDQ^T. Since the matrix Q of eigenvectors is *not*
   * symmetric in general, it is stored as a Tensor<2, dim> and not a
   * SymmetricTensor<2, dim>, and the result of QDQ^T is also a Tensor<2, dim>.
   */
  MetricTensor(const Tensor<2, dim> &t);

  /**
   * Constructor. Generate a MetricTensor from its eigendecomposition. Assumes
   * that the @p eigenvalues are all positive and that the @p eigenvectors are
   * orthonormal, which is checked in debug mode.
   *
   * This constructor is an alternative to the one above. @p eigenvectors is
   * expected to be the matrix Q in the eigendecomposition M = QDQ^T.
   */
  MetricTensor(const Tensor<2, dim> &eigenvectors,
               const Tensor<1, dim> &eigenvalues);

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
  MetricTensor<dim> &operator=(const MetricTensor<dim> &m);

  /**
   * Assignment operator from a SymmetricTensor, assumed to be SPD.
   * This operator simply checks that t is in fact SPD (in debug only), then
   * calls the assignment operator for SymmetricTensors.
   */
  MetricTensor<dim> &operator=(const SymmetricTensor<2, dim> &t);

  /**
   * Assignment operator from an array with N*(N+1)/2 components.
   * As for SymmetricTensors, this requires knowing the layout of the indices,
   * which can be obtained through the unrolled_index() function.
   */
  MetricTensor<dim> &operator=(const double (&array)[n_independent_components]);

  /**
   * Return the eigenvalues as a Tensor<1, dim>
   */
  const Tensor<1, dim> &get_eigenvalues() const;

  /**
   * Return the orthonormal eigenvectors as a Tensor<2, dim>
   */
  const Tensor<2, dim> &get_eigenvectors() const;

  /**
   * Return a const reference to the underlying Eigen matrix
   */
  const EigenRealMatrix &get_eigen_matrix() const;

  /**
   * Return the eigenvalues as a vector in Eigen format
   */
  const EigenRealVector &get_eigenvalues_as_eigen() const;

  /**
   * Return the orthonormal eigenvectors as an Eigen matrix
   */
  const EigenRealMatrix &get_eigenvectors_as_eigen() const;

  /**
   * Limit the eigenvalues to [min_eigenvalue, max_eigenvalue].
   * After calling this function, the underlying matrix is set to:
   *
   *             M = Q * diag(lambda_1', ..., lambda_d') * Q^T,
   *
   * with lambda_i' := min(max(lambda_i, min_eigenvalue), max_eigenvalue).
   */
  void bound_eigenvalues(const double min_eigenvalue,
                         const double max_eigenvalue);

  /**
   * Same as the function above, but returns a new metric with the bounded
   * eigenvalues.
   */
  MetricTensor<dim> bounded_eigenvalues(const double min_eigenvalue,
                                        const double max_eigenvalue) const;

  /**
   * Return the matrix logarithm of the underlying matrix of this metric.
   * This is logm(M) := Q * log(D) * Q^T, which is not the componentwise
   * logarithm. Computed with Eigen's log().
   */
  MetricTensor<dim> log() const;

  /**
   * Return the matrix exponential of the underlying matrix of this metric.
   * This is expm(M) := Q * exp(D) * Q^T, which is not the componentwise
   * exponential. Computed with Eigen's exp().
   */
  MetricTensor<dim> exp() const;

  /**
   * Compute metric intersection of this metric with other.
   * Checks first if metrics are diagonal or multiple of one another.
   * If not, compute their simultaneous reduction.
   */
  MetricTensor<dim> intersection(const MetricTensor<dim> &other,
                                 const double             tolerance) const;

  /**
   * Compute metric intersection of this metric with other using the routines
   * from the MMG library.
   */
  MetricTensor<dim> intersection_mmg(const MetricTensor<dim> &other) const;

  /**
   * Span a metric at distance pq with prescribed gradation.
   */
  MetricTensor<dim> span_metric(const double          gradation,
                                const Tensor<1, dim> &pq) const;

private:
  // Matrix representation of the underlying tensor in Eigen format.
  // This is *not* the matrix of the eigendecomposition of this metric tensor.
  EigenRealMatrix matrix_eigen;

  // The eigendecomposition stored in Eigen format
  EigenRealVector eigenvalues;
  EigenRealMatrix eigenvectors;

  // The eigendecomposition stored as Tensors
  Tensor<1, dim> eigenvalues_t;
  Tensor<2, dim> eigenvectors_t;
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
                    "parsed function returned a non SPD tensor.\n\n "
                    "The provided function evaluated at point ["
                 << arg1 << "] returned the components " << arg2
                 << ", with determinant = " << arg3 << ".\n\n"
                 << "You should maybe double-check that (i) the components "
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
      Assert(is_positive_definite(res), ExcUserFunNotSPD(p, res, det));
    }
    return res;
  }

private:
  const Functions::ParsedFunction<dim> &parsed_function;
};

/* ---------------- template and inline functions ----------------- */

#endif
