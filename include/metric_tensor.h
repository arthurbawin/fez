#ifndef METRIC_TENSOR_H
#define METRIC_TENSOR_H

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <parameters.h>

#include <sstream>

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

DeclExceptionMsg(
  ExcOperationMadeNotSPD,
  "You performed an operation on a metric tensor with an initial SPD "
  "underlying tensor which resulted in a non-SPD tensor.");

DeclException2(
  ExcAssignFromNotSPD,
  std::string,
  double,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from a SymmetricTensor that is not "
  "positive-definite: "
    << arg1 << "\n det = " << arg2);

DeclExceptionMsg(
  ExcAssignFromNotSPDArray,
  "You are trying to create a metric tensor, which should be symmetric and "
  "positive-definite (SPD) by definition, from an array that "
  "represents a SymmetricTensor that is not positive-definite. "
  "You should maybe double check the ordering of the entries of the array.");

DeclException2(ExcInterpolatedNotSPD,
               std::string,
               double,
               "An interpolated metric tensor is not positive-definite: "
                 << arg1 << "\n det = " << arg2);

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

DeclException1(ExcMultiplyByNonPositive,
               double,
               << "You are trying to multiply the entries of a metric tensor "
                  "by a nonpositive real number : "
               << arg1);

#define AssertAssignFromSPD(symmetric_tensor)                              \
  if constexpr (running_in_debug_mode())                                   \
  {                                                                        \
    std::ostringstream oss;                                                \
    oss << symmetric_tensor;                                               \
    Assert(is_positive_definite(symmetric_tensor),                         \
           ExcAssignFromNotSPD(oss.str(), determinant(symmetric_tensor))); \
  }

#define AssertAssignFromSPDArray(metric_tensor) \
  Assert(is_positive_definite(metric_tensor), ExcAssignFromNotSPDArray())

/**
 * A metric tensor is a positive-definite SymmetricTensor<2, dim, double>, with
 * additional member functions.
 *
 * Because most operations of metric tensors rely on their eigendecomposition,
 * a real-valued matrix in Eigen's format is stored as well, to use Eigen's
 * functions, such as their eigendecomposition, matrix logarithm, exponential,
 * etc.
 *
 * FIXME: deal.II also provides eigendecomposition routines, so it's worth
 * checking if the Eigen interface is really needed.
 */
template <int dim>
class MetricTensor : public SymmetricTensor<2, dim>
{
public:
  using EigenRealMatrix = typename EigenReal<dim>::matrix_type;
  using EigenRealVector = typename EigenReal<dim>::vector_type;

  static constexpr unsigned int n_independent_components =
    SymmetricTensor<2, dim>::n_independent_components;

  /**
   * The space in which a metric spans a complete metric field with prescribed
   * gradation. The chosen space determines if anisotropy is maintained as the
   * distance increases.
   *
   * See also F. Alauzet's paper "Size gradation control of anisotropic meshes".
   *
   * FIXME: Add comments.
   */
  enum SpanningSpace
  {
    euclidean,
    metric,
    exp_metric
  };

public:
  /**
   * Constructor. Generate a MetricTensor from a SymmetricTensor. Assumes that
   * @p t is already positive-definite (it has all positive eigenvalues), and in
   * debug mode this is in fact checked.
   *
   * Unlike SymmetricTensors, there is no constructor to create a default metric
   * tensor with arbitrary or zero values, since the tensor should be SPD.
   */
  explicit MetricTensor(
    const SymmetricTensor<2, dim> &t = unit_symmetric_tensor<dim>());

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
  explicit MetricTensor(const Tensor<2, dim> &t);

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
   * Multiply all entries of this metric tensor by @p factor.
   * This is the same operator as for SymmetricTensors, but in debug it also
   * checks that factor is a positive real number.
   */
  constexpr MetricTensor<dim> &operator*=(const double &factor);

  /**
   * Divide all entries of this metric tensor by @p factor.
   * This is the same operator as for SymmetricTensors, but in debug it also
   * checks that factor is a positive real number.
   */
  constexpr MetricTensor<dim> &operator/=(const double &factor);

  /**
   * Although the sum of two positive-definite matrices remains a
   * positive-definite matrix, metric tensors live on manifolds which are not
   * necessarily vector spaces, so it does not make sense to add them in
   * general, and the incrementation with another tensor is deleted.
   */
  constexpr MetricTensor<dim> &
  operator+=(const SymmetricTensor<2, dim> &) = delete;

  /**
   * Same as above but deletes incremention with a MetricTensor<dim>.
   */
  constexpr MetricTensor<dim> &operator+=(const MetricTensor<dim> &) = delete;

  /**
   * In addition to the comments for the incrementation operator, decrementing
   * does not preserve positive-definiteness in general, and is thus deleted.
   */
  constexpr MetricTensor<dim> &
  operator-=(const SymmetricTensor<2, dim> &) = delete;

  /**
   * Same as above but deletes decremention with a MetricTensor<dim>.
   */
  constexpr MetricTensor<dim> &operator-=(const MetricTensor<dim> &) = delete;

  /**
   * Metric negation. Deleted for obvious reasons.
   */
  constexpr MetricTensor<dim> operator-() = delete;

  /**
   * Return the eigenvalues as a Tensor<1, dim>.
   */
  const Tensor<1, dim> &get_eigenvalues() const;

  /**
   * Return the orthonormal eigenvectors as a Tensor<2, dim>.
   */
  const Tensor<2, dim> &get_eigenvectors() const;

  /**
   * Return the underlying Eigen matrix.
   */
  const EigenRealMatrix &get_eigen_matrix() const;

  /**
   * Return the eigenvalues as a vector in Eigen format.
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
   *
   * Note that the log of a metric is not a metric in general, e.g.,
   * log I = 0.
   */
  SymmetricTensor<2, dim> log() const;

  /**
   * Return the matrix exponential of the underlying matrix of this metric.
   * This is expm(M) := Q * exp(D) * Q^T, which is not the componentwise
   * exponential. Computed with Eigen's exp().
   */
  MetricTensor<dim> exp() const;

  /**
   * Return the matrix square root of the underlying matrix of this metric.
   * This is sqrtm(M) := Q * sqrt(D) * Q^T, which is not the componentwise
   * square root. Computed with Eigen's sqrt().
   */
  MetricTensor<dim> sqrt() const;

  /**
   * Return the matrix square root of the underlying matrix of this metric.
   * This is isqrtm(M) := Q * D^{-1/2} * Q^T. Computed with Eigen's pow().
   */
  MetricTensor<dim> inverse_sqrt() const;

  /**
   * Compute metric intersection of this metric with @p other, using their
   * simultaneous reduction M^{-1} * other.
   * The intersection is computed with the routines from the MMG library.
   */
  MetricTensor<dim> intersection(const MetricTensor<dim> &other) const;

  /**
   * Span a metric at prescribed @p distance vector with prescribed @p gradation
   * and in the prescribed @p spanning_space.
   *
   * TODO: Add description.
   */
  MetricTensor<dim> span_metric(const SpanningSpace   spanning_space,
                                const double          gradation,
                                const Tensor<1, dim> &distance) const;

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

template <int dim>
constexpr inline MetricTensor<dim> &
MetricTensor<dim>::operator*=(const double &d)
{
  Assert(d > 0, ExcMultiplyByNonPositive(d));
  SymmetricTensor<2, dim>::operator*=(d);
  return *this;
}

template <int dim>
constexpr inline MetricTensor<dim> &
MetricTensor<dim>::operator/=(const double &d)
{
  Assert(d > 0, ExcMultiplyByNonPositive(d));
  SymmetricTensor<2, dim>::operator/=(d);
  return *this;
}

/* ----------------- Non-member functions operating on metric tensors. ------ */

/**
 * The sum of two MetricTensors is not meaningful in general (even though it
 * preserves positive-definiteness), and is thus deleted.
 */
template <int dim>
constexpr MetricTensor<dim> operator+(const MetricTensor<dim> &left,
                                      const MetricTensor<dim> &right) = delete;

/**
 * Sum of SymmetricTensor and MetricTensor. See comment above.
 */
template <int dim>
constexpr MetricTensor<dim> operator+(const SymmetricTensor<2, dim> &left,
                                      const MetricTensor<dim> &right) = delete;

/**
 * Sum of MetricTensor and SymmetricTensor. See comment above.
 */
template <int dim>
constexpr MetricTensor<dim>
operator+(const MetricTensor<dim>       &left,
          const SymmetricTensor<2, dim> &right) = delete;

/**
 * As mentionned in the comments for operator-=, subtracting metric tensors does
 * not yield a metric tensor in general (simply take I - I = 0). However, it is
 * useful to keep a subtraction operator between metrics and returning a
 * SymmetricTensor, to measure differences between metrics, for instance. Since
 * the constructor is marked explicit, implicit conversion from a
 * SymmetricTensor to a MetricTensor is forbidden, so that if m1 and m2 are
 * MetricTensors,
 *
 * SymmetricTensor<2, dim> res = m1 - m2;
 *
 * is allowed, but
 *
 * MetricTensor<dim> res = m1 - m2;
 *
 * is not. Note that this function is not really needed, since the operator- of
 * the base class does the exact same thing already. It is simply here to
 * emphasize that the difference of two metrics does not produce a MetricTensor.
 */
template <int dim>
constexpr SymmetricTensor<2, dim> operator-(const MetricTensor<dim> &left,
                                            const MetricTensor<dim> &right)
{
  return SymmetricTensor<2, dim>(left) - SymmetricTensor<2, dim>(right);
}

/**
 * Multiplication of a metric tensor with a scalar from the right.
 * The scalar should be positive, and this is checked in debug.
 */
template <int dim>
constexpr inline MetricTensor<dim> operator*(const MetricTensor<dim> &m,
                                             const double             factor)
{
  Assert(factor > 0, ExcMultiplyByNonPositive(factor));
  MetricTensor<dim> mm(m);
  mm *= factor;
  return mm;
}

/**
 * Multiplication of a metric tensor with a scalar from the right.
 * The scalar should be positive, and this is checked in debug.
 */
template <int dim>
constexpr inline MetricTensor<dim> operator*(const double             factor,
                                             const MetricTensor<dim> &m)
{
  Assert(factor > 0, ExcMultiplyByNonPositive(factor));
  MetricTensor<dim> mm(m);
  mm *= factor;
  return mm;
}

/**
 * Division of a metric by a scalar.
 * The scalar should be positive, and this is checked in debug.
 */
template <int dim>
constexpr inline MetricTensor<dim> operator/(const MetricTensor<dim> &m,
                                             const double             factor)
{
  Assert(factor > 0, ExcMultiplyByNonPositive(factor));
  MetricTensor<dim> mm(m);
  mm /= factor;
  return mm;
}

#endif
