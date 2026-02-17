
#include <metric_tensor.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

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

template <int dim>
typename EigenRealMatrix<dim>::type metric2eigen(const Tensor<2, dim> &tensor)
{
  using RealMatrix = typename EigenRealMatrix<dim>::type;
  RealMatrix eigenMat;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      eigenMat(i, j) = tensor[i][j];
  return eigenMat;
}

template <int dim>
typename EigenRealMatrix<dim>::type
metric2eigen(const MetricTensor<dim> &metric)
{
  using RealMatrix = typename EigenRealMatrix<dim>::type;
  RealMatrix eigenMat;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      eigenMat(i, j) = metric[i][j];
  return eigenMat;
}

// Return a new MetricTensor
template <int dim>
MetricTensor<dim> eigen2metric(const Eigen::Matrix<double, dim, dim> &eigenMat)
{
  MetricTensor<dim> metric;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      metric[i][j] = eigenMat(i, j);
  return metric;
}

// Assign into existing MetricTensor
template <int dim>
void eigen2metric(const Eigen::Matrix<double, dim, dim> &eigenMat,
                  MetricTensor<dim>                     &metric)
{
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i; j < dim; ++j)
      metric[i][j] = eigenMat(i, j);
}

template <int dim>
MetricTensor<dim> MetricTensor<dim>::log() const
{
  using RealMatrix    = typename EigenRealMatrix<dim>::type;
  RealMatrix eigenMat = metric2eigen(*this);
  RealMatrix logM     = eigenMat.log();
  return eigen2metric(logM);
}

template <int dim>
MetricTensor<dim> MetricTensor<dim>::exp() const
{
  using RealMatrix    = typename EigenRealMatrix<dim>::type;
  RealMatrix eigenMat = metric2eigen(*this);
  RealMatrix expM     = eigenMat.exp();
  return eigen2metric(expM);
}

/**
 * Eigendecomposition for a symmetric matrix.
 * Real eigenvalues and orthogonal eigenvectors.
 */
template <int dim>
void eigenDecompositionSymmetric(const Eigen::Matrix<double, dim, dim> &A,
                                 Eigen::Matrix<double, dim, 1>   &eigenvalues,
                                 Eigen::Matrix<double, dim, dim> &eigenvectors)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, dim, dim>> eigensolver(
    A, Eigen::ComputeEigenvectors);

  if (eigensolver.info() != Eigen::Success)
    throw std::runtime_error("Eigen decomposition failed");

  eigenvalues = eigensolver.eigenvalues(); // Real-valued vector
  eigenvectors =
    eigensolver.eigenvectors(); // Columns are eigenvectors (orthonormal)
}

/**
 * Eigendecomposition for arbitrary matrix, possibly
 * with complex eigenvalues and eigenvectors.
 */
template <int dim>
void eigenDecomposition(
  const Eigen::Matrix<double, dim, dim>         &A,
  Eigen::Matrix<std::complex<double>, dim, 1>   &eigenvalues,
  Eigen::Matrix<std::complex<double>, dim, dim> &eigenvectors)
{
  Eigen::EigenSolver<Eigen::Matrix<double, dim, dim>> eigensolver(A, true);

  if (eigensolver.info() != Eigen::Success)
    throw std::runtime_error("Eigen decomposition failed");

  eigenvalues  = eigensolver.eigenvalues();
  eigenvectors = eigensolver.eigenvectors();
}

template <int dim>
void MetricTensor<dim>::boundEigenvalues(const double lambdaMin,
                                         const double lambdaMax)
{
  Eigen::Matrix<double, dim, 1>   eigenvalues;
  Eigen::Matrix<double, dim, dim> eigenvectors;
  eigenDecompositionSymmetric<dim>(metric2eigen(*this),
                                   eigenvalues,
                                   eigenvectors);
  for (auto &val : eigenvalues)
    val = std::min(lambdaMax, std::max(lambdaMin, val));
  Eigen::Matrix<double, dim, dim> res =
    eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
  eigen2metric(res, *this);
}

template <int dim>
MetricTensor<dim> absoluteValue(const Vector<double> &vec)
{
  constexpr unsigned int n_components = dim * (dim + 1) / 2;
  AssertThrow(vec.size() == n_components,
              ExcDimensionMismatch(vec.size(), n_components));

  Eigen::Matrix<double, dim, 1>   eigenvalues;
  Eigen::Matrix<double, dim, dim> eigenvectors;

  Eigen::Matrix<double, dim, dim> eigenMat;

  // Fill in the LOWER triangular part
  if constexpr (dim == 2)
  {
    eigenMat(0, 0) = vec[0];
    eigenMat(1, 1) = vec[1];
    eigenMat(1, 0) = vec[2];
  }
  if constexpr (dim == 3)
  {
    eigenMat(0, 0) = vec[0];
    eigenMat(1, 1) = vec[1];
    eigenMat(2, 2) = vec[2];
    eigenMat(1, 0) = vec[3]; // Check ordering
    eigenMat(2, 0) = vec[4];
    eigenMat(2, 1) = vec[5];
  }

  eigenDecompositionSymmetric<dim>(eigenMat, eigenvalues, eigenvectors);

  Eigen::Matrix<double, dim, dim> absD = eigenvalues.cwiseAbs().asDiagonal();

  Eigen::Matrix<double, dim, dim> res =
    eigenvectors * absD * eigenvectors.transpose();
  return eigen2metric(res);
}

template MetricTensor<2> absoluteValue<2>(const Vector<double> &vec);
template MetricTensor<3> absoluteValue<3>(const Vector<double> &vec);

// template <int dim>
// MetricTensor<dim>
// MetricTensor<dim>::intersection(const MetricTensor<dim> &other) const
// {
//   // using RealMatrix    = Eigen::Matrix<double, dim, dim>;
//   using RealMatrix    = typename EigenRealMatrix<dim>::type;
//   using ComplexMatrix = typename EigenComplexMatrix<dim>::type;
//   using ComplexVector = Eigen::Matrix<std::complex<double>, dim, 1>;

//   const MetricTensor<dim> &M1 = *this;
//   const MetricTensor<dim> &M2 = other;

//   RealMatrix EM1 = metric2eigen(*this);
//   RealMatrix EM2 = metric2eigen(other);

//   // Otherwise compute simultaneous reduction
//   // The matrix N = M1^{-1} * M2 is SPD but NOT symmetric in general
//   Tensor<2, dim> N =
//     contract<0, 1>(Tensor<2, dim>(invert(M1)), Tensor<2, dim>(M2));

//   RealMatrix N_eigen = metric2eigen(N);

//   Eigen::EigenSolver<RealMatrix> eigensolver(N_eigen);
//   if (eigensolver.info() != Eigen::Success)
//   {
//     throw std::runtime_error("Eigendecomposition failed");
//   }

//   ComplexMatrix P       = eigensolver.eigenvectors(); // columns = v_j
//   ComplexMatrix P_inv   = P.inverse();
//   ComplexMatrix P_inv_H = P_inv.adjoint(); // P^{-H}

//   std::array<std::array<std::complex<double>, 2>, dim> mu; // mu[j][i]

//   for (int j = 0; j < dim; ++j)
//   {
//     ComplexVector v_j = P.col(j);
//     mu[j][0]          = v_j.adjoint() * EM1 * v_j;
//     mu[j][1]          = v_j.adjoint() * EM2 * v_j;
//   }

//   // Step 3: For each eigenvector v_j, get max_i |mu_{i,j}|
//   ComplexMatrix D = ComplexMatrix::Zero();
//   for (int j = 0; j < dim; ++j)
//   {
//     D(j, j) = std::abs(mu[j][0]) >= std::abs(mu[j][1]) ? mu[j][0] : mu[j][1];
//   }

//   ComplexMatrix res     = P_inv_H * D * P_inv;
//   RealMatrix    resReal = res.real();

//   MetricTensor intersection = eigen2metric(resReal);

//   if (determinant(intersection) <= 0)
//   {
//     std::cout << intersection << std::endl;
//     std::cout << determinant(intersection) << std::endl;
//     throw std::runtime_error(
//       "Metric intersection has non-positive determinant");
//   }

//   return intersection;
// }

// Check if two values are close relative to their order of magnitude
bool isClose(const double a, const double b, const double tol)
{
  return std::fabs(a - b) <= tol * std::max({1.0, std::fabs(a), std::fabs(b)});
}

template <int dim>
bool isSymmetric(const MetricTensor<dim> &m, double tol)
{
  if constexpr (dim == 2)
  {
    return isClose(m[0][1], m[1][0], tol);
  }
  else
  {
    return isClose(m[0][1], m[1][0], tol) && isClose(m[0][2], m[2][0], tol) &&
           isClose(m[1][2], m[1][2], tol);
  }
}

template <int dim>
bool isDiagonal(const MetricTensor<dim> &m, double tol)
{
  if constexpr (dim == 2)
  {
    return isClose(m[0][1], 0.0, tol);
  }
  else
  {
    return isClose(m[0][1], 0.0, tol) && isClose(m[0][2], 0.0, tol) &&
           isClose(m[1][2], 0.0, tol);
  }
}

template <int dim>
bool areEqual(const MetricTensor<dim> &m1,
              const MetricTensor<dim> &m2,
              double                   tol)
{
  if constexpr (dim == 2)
  {
    return isClose(m1[0][0], m2[0][0], tol) &&
           isClose(m1[0][1], m2[0][1], tol) && isClose(m1[1][1], m2[1][1], tol);
  }
  else
  {
    return isClose(m1[0][0], m2[0][0], tol) &&
           isClose(m1[0][1], m2[0][1], tol) &&
           isClose(m1[1][1], m2[1][1], tol) &&
           isClose(m1[0][2], m2[0][2], tol) &&
           isClose(m1[2][2], m2[2][2], tol) && isClose(m1[1][2], m2[1][2], tol);
  }
}

template <int dim>
bool isMultiple(const MetricTensor<dim> &m1,
                const MetricTensor<dim> &m2,
                double                   tol)
{
  if constexpr (dim == 2)
  {
    // Diagonal elements must not be zero
    if (std::fabs(m1[0][0]) < tol || std::fabs(m1[1][1]) < tol)
      return false;

    double r0 = m2[0][0] / m1[0][0];
    double r2 = m2[1][1] / m1[1][1];

    if (!isClose(r0, r2, tol))
      return false;

    bool m1_offdiag_zero = std::fabs(m1[0][1]) < tol;
    bool m2_offdiag_zero = std::fabs(m2[0][1]) < tol;

    if (m1_offdiag_zero && m2_offdiag_zero)
    {
      // Both diagonal, ratios match => scalar multiple
      return true;
    }

    if (m1_offdiag_zero != m2_offdiag_zero)
    {
      // One is diagonal, other is not => not scalar multiple
      return false;
    }

    // Both off-diagonal are non-zero; compare ratio
    double r1 = m2[0][1] / m1[0][1];
    return isClose(r0, r1, tol);
  }
  else
  {
    // TODO
    AssertThrow(false, ExcNotImplemented());
  }
}

// Containment check: is A ⪯ B (i.e., unit ball of B contains unit ball of A)
// Return true if m2 is contained in m1
template <int dim>
bool isContained(const MetricTensor<dim> &m1,
                 const MetricTensor<dim> &m2,
                 double                   tol)
{
  if constexpr (dim == 2)
  {
    double minor1 = m2[0][0] - m1[0][0];
    double det    = (m2[0][0] - m1[0][0]) * (m2[1][1] - m1[1][1]) -
                 (m2[0][1] - m1[0][1]) * (m2[0][1] - m1[0][1]);
    return minor1 > -tol && det > -tol;
  }
  else
  {
    // TODO
    AssertThrow(false, ExcNotImplemented());
  }
}

// Compute the intersection metric
template <int dim>
MetricTensor<dim>
MetricTensor<dim>::intersection(const MetricTensor<dim> &m2) const
{
  const double             tolerance = 1e-12;
  const MetricTensor<dim> &m1        = *this;

  AssertThrow(isSymmetric(m1, tolerance) && isSymmetric(m2, tolerance),
              ExcMessage(
                "An operand for metric intersection is not symmetric"));

  // Case 1: Equal
  if (areEqual(m1, m2, tolerance))
  {
    return m1;
  }

  // Case 2: Diagonal
  if (isDiagonal(m1, tolerance) && isDiagonal(m2, tolerance))
  {
    MetricTensor<dim> m;
    m[0][0] = std::max(m1[0][0], m2[0][0]);
    m[0][1] = 0.;
    m[1][0] = 0.;
    m[1][1] = std::max(m1[1][1], m2[1][1]);
    if constexpr (dim == 3)
    {
      m[0][2] = 0.;
      m[1][2] = 0.;
      m[2][2] = std::max(m1[2][2], m2[2][2]);
    }
    return m;
  }

  // Case 3: Scalar multiples
  if (isMultiple(m1, m2, tolerance))
  {
    double scale = m2[0][0] / m1[0][0];
    return ((scale <= 1) ? m1 : m2);
  }

  // Case 4: Containment
  if (isContained(m2, m1, tolerance))
  {
    return m1;
  }
  if (isContained(m1, m2, tolerance))
  {
    return m2;
  }

  // Case 5: General case - simultaneous reduction N = m1^-1 * m2.
  // - Diagonalize N, denote eigenvectors by v_j and V = [v_1 v_2].
  // - Compute eigenvalues lambda_ij := v_j^T * m_i * v_j.
  // - Intersection is V^-T * diag([max_i lambda_i1, max_i lambda_i2]) * V^-1
  //
  // Use symbolic expression of V^-T computed with Matlab.

  if constexpr (dim == 2)
  {
    const double a1 = m1[0][0], b1 = m1[0][1], c1 = m1[1][1];
    const double a2 = m2[0][0], b2 = m2[0][1], c2 = m2[1][1];

    const double r   = a1 * b2 - a2 * b1;
    const double s   = a1 * c2 - a2 * c1;
    const double t   = b2 * c1 - c2 * b1;
    const double den = sqrt(s * s + 4. * r * t);

    if (std::abs(r) < tolerance)
    {
      // An eigenvector of N is aligned with the x-axis,
      // the matrix of eigenvectors of N writes:
      //
      // [ -b2/a2  1 ]
      // [      1  0 ]
      //
      // Return exact solution directly.

      const double l0 = std::max(c1 - (b1 * b2) / a2, c2 - (b2 * b2) / a2);
      const double l1 = std::max((a2 * b1) / b2, a2);

      MetricTensor<dim> res;
      res[0][0] = l1;
      res[0][1] = (b2 * l1) / a2;
      res[1][0] = res[0][1];
      res[1][1] = l0 + (l1 * b2 * b2) / (a2 * a2);
      return res;
    }
    else
    {
      if (den < tolerance)
      {
        // N has repeated eigenvalue? N is multiple of identity in some basis
        // Metrics are multiple of one another? Should not happen
        std::cout << den << std::endl;
        throw std::runtime_error("den < tolerance");
      }

      const double v00 = (den - s) / (2. * r);
      const double v01 = -(den + s) / (2. * r);

      Tensor<1, dim> eigenvector0, eigenvector1;

      eigenvector0[0] = v00;
      eigenvector0[1] = 1.;
      eigenvector1[0] = v01;
      eigenvector1[1] = 1.;

      const double l0 = std::max(eigenvector0 * m1 * eigenvector0,
                                 eigenvector0 * m2 * eigenvector0);
      const double l1 = std::max(eigenvector1 * m1 * eigenvector1,
                                 eigenvector1 * m2 * eigenvector1);

      Tensor<2, dim> eigenvectors, eigenvalues_diag;

      // Eigenvectors of V^-T
      eigenvectors[0][0] = r / den;
      eigenvectors[0][1] = -r / den;
      eigenvectors[1][0] = 0.5 + s / (2. * den);
      eigenvectors[1][1] = 0.5 - s / (2. * den);

      eigenvalues_diag[0][0] = l0;
      eigenvalues_diag[1][0] = 0.;
      eigenvalues_diag[0][1] = 0.;
      eigenvalues_diag[1][1] = l1;

      // V^-T * D * V^-1 is symmetric up to round-off error.
      // dealii wants a SymmetricTensor to be exactly symmetric,
      // so the result is symmetrized.
      return MetricTensor<dim>(
        symmetrize(eigenvectors * eigenvalues_diag * transpose(eigenvectors)));
    }
  }
  else
  {
    // TODO: simultaneous reduction in 3D
    AssertThrow(false, ExcNotImplemented());
  }
}

// Span in metric space
template <int dim>
MetricTensor<dim> MetricTensor<dim>::spanMetric(const double          gradation,
                                                const Tensor<1, dim> &pq) const
{
  const double dotProd_M = pq * (*this) * pq;
  double       eta       = 1. + std::sqrt(dotProd_M) * std::log(gradation);
  return (*this) / (eta * eta);
}

template class MetricTensor<2>;
template class MetricTensor<3>;
