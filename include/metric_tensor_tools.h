#ifndef METRIC_TENSOR_TOOLS_H
#define METRIC_TENSOR_TOOLS_H

#include <metric_intersection_mmg.h>
#include <metric_tensor.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace MetricTensorTools
{
  using namespace dealii;

  template <int dim>
  using EigenRealMatrix = typename MetricTensor<dim>::EigenRealMatrix;
  template <int dim>
  using EigenRealVector = typename MetricTensor<dim>::EigenRealVector;

  template <int dim>
  DeclException1(ExcSymDecompositionFailed,
                 typename EigenReal<dim>::matrix_type,
                 << "Eigen decomposition failed for symmetric matrix:" << arg1
                 << ".");

  template <int dim>
  DeclException1(ExcEigenMatNotSym,
                 typename EigenReal<dim>::matrix_type,
                 << "Eigen matrix is expected to be symmetric, but is not:"
                 << arg1 << ".");

  /**
   * Return a new Eigen matrix from a MetricTensor<dim>
   */
  template <int dim>
  typename EigenReal<dim>::matrix_type
  metric2eigen(const MetricTensor<dim> &metric)
  {
    typename EigenReal<dim>::matrix_type eigen_matrix;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = metric[i][j];
    return eigen_matrix;
  }

  /**
   * Fill an existing Eigen matrix from a MetricTensor<dim>
   */
  template <int dim>
  void metric2eigen(const MetricTensor<dim>              &metric,
                    typename EigenReal<dim>::matrix_type &eigen_matrix)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = metric[i][j];
  }

  /**
   * Return a new Eigen matrix from a (Symmetric)Tensor<2, dim>
   */
  template <int dim, typename TensorType>
  typename EigenReal<dim>::matrix_type tensor2eigen(const TensorType &t)
  {
    typename EigenReal<dim>::matrix_type eigen_matrix;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = t[i][j];
    return eigen_matrix;
  }

  /**
   * Fill an existing Eigen matrix from a (Symmetric)Tensor<2, dim>
   */
  template <int dim, typename TensorType>
  void tensor2eigen(const TensorType                     &t,
                    typename EigenReal<dim>::matrix_type &eigen_matrix)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = t[i][j];
  }

  /**
   * Return a new Eigen matrix from a Tensor<1, dim>
   */
  template <int dim>
  typename EigenReal<dim>::vector_type vector2eigen(const Tensor<1, dim> &t)
  {
    typename EigenReal<dim>::vector_type eigen_vector;
    for (unsigned int i = 0; i < dim; ++i)
      eigen_vector[i] = t[i];
    return eigen_vector;
  }

  /**
   * Fill an existing Eigen matrix from a Tensor<1, dim>
   */
  template <int dim>
  void vector2eigen(const Tensor<1, dim>                 &t,
                    typename EigenReal<dim>::vector_type &eigen_vector)
  {
    for (unsigned int i = 0; i < dim; ++i)
      eigen_vector(i) = t[i];
  }

  /**
   * Return a new MetricTensor from an Eigen matrix
   */
  template <int dim>
  MetricTensor<dim> eigen2metric(const EigenRealMatrix<dim> &eigen_matrix)
  {
    MetricTensor<dim> metric;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        metric[i][j] = eigen_matrix(i, j);
    return metric;
  }

  /**
   * Fill an existing MetricTensor from an Eigen matrix
   */
  template <int dim>
  void eigen2metric(const EigenRealMatrix<dim> &eigen_matrix,
                    MetricTensor<dim>          &metric)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
        metric[i][j] = eigen_matrix(i, j);
  }

  /**
   * Eigendecomposition for a symmetric matrix.
   * Real eigenvalues and orthogonal eigenvectors.
   */
  template <int dim>
  void eigen_decomposition_symmetric(const EigenRealMatrix<dim> &A,
                                     EigenRealVector<dim>       &eigenvalues,
                                     EigenRealMatrix<dim>       &eigenvectors)
  {
    Assert(A.isApprox(A.transpose()), ExcEigenMatNotSym<dim>(A));

    Eigen::SelfAdjointEigenSolver<EigenRealMatrix<dim>> eigensolver(
      A, Eigen::ComputeEigenvectors);

    Assert(eigensolver.info() == Eigen::Success,
           ExcSymDecompositionFailed<dim>(A));

    // Real-valued vector of eigenvalues
    eigenvalues = eigensolver.eigenvalues();
    // Columns are the orthonormal eigenvectors
    eigenvectors = eigensolver.eigenvectors();
  }

  /**
   * Eigendecomposition for a symmetric matrix.
   * Calls the function above and copies the Eigen data into Tensors.
   */
  template <int dim>
  void eigen_decomposition_symmetric(const EigenRealMatrix<dim> &A,
                                     EigenRealVector<dim>       &eigenvalues,
                                     EigenRealMatrix<dim>       &eigenvectors,
                                     Tensor<1, dim>             &eigenvalues_t,
                                     Tensor<2, dim>             &eigenvectors_t)
  {
    eigen_decomposition_symmetric<dim>(A, eigenvalues, eigenvectors);

    // Store as deal.II tensors
    for (unsigned int di = 0; di < dim; ++di)
    {
      eigenvalues_t[di] = eigenvalues(di);
      for (unsigned int dj = 0; dj < dim; ++dj)
        eigenvectors_t[di][dj] = eigenvectors(di, dj);
    }
  }

  /**
   * Eigendecomposition for the matrix representation of a symmetric tensor.
   * Return the eigendecomposition as deal.II tensors.
   */
  template <int dim>
  void eigen_decomposition_symmetric(const SymmetricTensor<2, dim> &t,
                                     Tensor<1, dim> &eigenvalues_t,
                                     Tensor<2, dim> &eigenvectors_t)
  {
    EigenRealVector<dim> eigenvalues;
    EigenRealMatrix<dim> eigenvectors, matrix_eigen;
    tensor2eigen<dim>(t, matrix_eigen);
    eigen_decomposition_symmetric<dim>(matrix_eigen, eigenvalues, eigenvectors);
    for (unsigned int di = 0; di < dim; ++di)
    {
      eigenvalues_t[di] = eigenvalues(di);
      for (unsigned int dj = 0; dj < dim; ++dj)
        eigenvectors_t[di][dj] = eigenvectors(di, dj);
    }
  }

  /**
   * Eigendecomposition for arbitrary matrix, possibly
   * with complex eigenvalues and eigenvectors.
   */
  template <int dim>
  void eigen_decomposition(
    const EigenRealMatrix<dim>                    &A,
    Eigen::Matrix<std::complex<double>, dim, 1>   &eigenvalues,
    Eigen::Matrix<std::complex<double>, dim, dim> &eigenvectors)
  {
    Eigen::EigenSolver<EigenRealMatrix<dim>> eigensolver(A, true);

    if (eigensolver.info() != Eigen::Success)
      throw std::runtime_error("Eigen decomposition failed");

    eigenvalues  = eigensolver.eigenvalues();
    eigenvectors = eigensolver.eigenvectors();
  }

  /**
   * Return the metric defined by M = Q*|D|*Q^T, where Q*D*Q^T is the
   * eigendecomposition of the given symmetric tensor.
   */
  template <int dim>
  MetricTensor<dim> absolute_value(const SymmetricTensor<2, dim> &t)
  {
    EigenRealVector<dim> eigenvalues;
    EigenRealMatrix<dim> eigenvectors, matrix_eigen;
    tensor2eigen<dim>(t, matrix_eigen);
    eigen_decomposition_symmetric<dim>(matrix_eigen, eigenvalues, eigenvectors);
    EigenRealMatrix<dim> res = eigenvectors *
                               eigenvalues.cwiseAbs().asDiagonal() *
                               eigenvectors.transpose();
    return eigen2metric<dim>(res);
  }

  // Check if two values are close relative to their order of magnitude
  inline bool
  is_relatively_close(const double a, const double b, const double tol)
  {
    return std::abs(a - b) <= tol * std::max({1., std::abs(a), std::abs(b)});
  }

  /**
   * Return true if the metric m is diagonal up to tolerance
   */
  template <int dim>
  bool is_diagonal(const MetricTensor<dim> &m, double tol)
  {
    if constexpr (dim == 2)
      return is_relatively_close(m[0][1], 0., tol);
    else
      return is_relatively_close(m[0][1], 0., tol) &&
             is_relatively_close(m[0][2], 0., tol) &&
             is_relatively_close(m[1][2], 0., tol);
  }

  /**
   * Return true if the metrics m1 and m2 are relatively equal componentwise
   */
  template <int dim>
  bool are_relatively_equal(const MetricTensor<dim> &m1,
                            const MetricTensor<dim> &m2,
                            double                   tol)
  {
    for (unsigned int i = 0; i < m1.n_independent_components; ++i)
    {
      const double v1 = m1[m1.unrolled_to_component_indices(i)];
      const double v2 = m2[m2.unrolled_to_component_indices(i)];
      if (!is_relatively_close(v1, v2, tol))
        return false;
    }
    return true;
  }

  template <int dim>
  bool is_multiple(const MetricTensor<dim> &m1,
                   const MetricTensor<dim> &m2,
                   double                   tol)
  {
    if constexpr (running_in_debug_mode())
    {
      // Make sure that diagonal entries are positive, which really should never
      // fail, and that the non-diagonal entries are also nonzero, which should
      // have been caught by is_diagonal
      for (unsigned int d = 0; d < dim; ++d)
        Assert(m1[d][d] > 0, ExcInternalError());
      for (unsigned int i = 0; i < m1.n_independent_components; ++i)
        Assert(!is_relatively_close(m1[m1.unrolled_to_component_indices(i)],
                                    0.,
                                    tol),
               ExcInternalError());
    }

    const double ratio = m2[0][0] / m1[0][0];
    for (unsigned int i = 0; i < m1.n_independent_components; ++i)
    {
      const double r12 = m2[m2.unrolled_to_component_indices(i)] /
                         m1[m1.unrolled_to_component_indices(i)];
      if (!is_relatively_close(ratio, r12, tol))
        return false;
    }
    return true;
  }

  /**
   * Return true if m2 is contained in m1 in the sense of their unit balls, that
   * is, if the unit ball of m2 is contained in the unit ball of m1 (ball(m2) ⪯
   * ball(m1)).
   */
  template <int dim>
  bool is_contained(const MetricTensor<dim> &m1,
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

  template <int dim>
  MetricTensor<dim> intersection(const MetricTensor<dim> &m1,
                                 const MetricTensor<dim> &m2,
                                 const double             tolerance)
  {
    // Case 1: Metrics are equal up to tolerance
    if (are_relatively_equal(m1, m2, tolerance))
      return m1;

    // Case 2: Both metrics are diagonal
    // Return the max eigenvalue (minimum size) along each axis
    if (is_diagonal(m1, tolerance) && is_diagonal(m2, tolerance))
    {
      MetricTensor<dim> m(unit_symmetric_tensor<dim>());
      for (unsigned int d = 0; d < dim; ++d)
        m[d][d] = std::max(m1[d][d], m2[d][d]);
      return m;
    }

    // Case 3: Metrics are scalar multiples of one another
    // Return the one with the greatest scaling
    if (is_multiple(m1, m2, tolerance))
    {
      const double scale = m2[0][0] / m1[0][0];
      return ((scale > 1.) ? m2 : m1);
    }

    // Case 4: Containment
    if (is_contained(m2, m1, tolerance))
    {
      return m1; // m1 is contained in m2
    }
    if (is_contained(m1, m2, tolerance))
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
        return MetricTensor<dim>(symmetrize(eigenvectors * eigenvalues_diag *
                                            transpose(eigenvectors)));
      }
    }
    else
    {
      // TODO: simultaneous reduction in 3D
      AssertThrow(false, ExcNotImplemented());
    }
  }
} // namespace MetricTensorTools

#endif
