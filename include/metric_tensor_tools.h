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
   * Return a new SymmetricTensor from an Eigen matrix
   */
  template <int dim>
  SymmetricTensor<2, dim>
  eigen2symtensor(const EigenRealMatrix<dim> &eigen_matrix)
  {
    SymmetricTensor<2, dim> t;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        t[i][j] = eigen_matrix(i, j);
    return t;
  }

  /**
   * Fill an existing SymmetricTensor from an Eigen matrix
   */
  template <int dim>
  void eigen2symtensor(const EigenRealMatrix<dim> &eigen_matrix,
                       SymmetricTensor<2, dim>    &t)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
        t[i][j] = eigen_matrix(i, j);
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
   *
   * Because the result is intended to be positive-definite, the eigenvalues are
   * also clamped in this function. For the case t = 0, for example, this avoids
   * the need for an intermediary absolute_value() function which would return a
   * SymmetricTensor<2, dim>, from which we would then bound the eigenvalues.
   */
  template <int dim>
  MetricTensor<dim> absolute_value(const SymmetricTensor<2, dim> &t,
                                   const double min_eigenvalue,
                                   const double max_eigenvalue)
  {
    Assert(min_eigenvalue < max_eigenvalue,
           ExcMessage("The prescribed minimum eigenvalue should be smaller "
                      "than the prescribed maximum eigenvalue."));

    EigenRealVector<dim> eigenvalues;
    EigenRealMatrix<dim> eigenvectors, matrix_eigen;
    tensor2eigen<dim>(t, matrix_eigen);
    eigen_decomposition_symmetric<dim>(matrix_eigen, eigenvalues, eigenvectors);

    // Clamp the absolute value of the eigenvalues
    for (unsigned int d = 0; d < dim; ++d)
      eigenvalues[d] =
        std::min(max_eigenvalue,
                 std::max(min_eigenvalue, std::abs(eigenvalues[d])));

    EigenRealMatrix<dim> res =
      eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    return eigen2metric<dim>(res);
  }

  /**
   * Apply metric gradation on the mesh edge @p p - @p q.
   */
  template <int dim>
  bool gradation_on_edge(
    const Point<dim>                               &p,
    const Point<dim>                               &q,
    const typename MetricTensor<dim>::SpanningSpace spanning_space,
    const double                                    gradation,
    const double                                    relative_tolerance,
    MetricTensor<dim>                              &Mp,
    MetricTensor<dim>                              &Mq)
  {
    bool metricChanged = false;

    // Span Mp to q, intersect and check if Mq needs to be reduced
    MetricTensor<dim> MpAtq = Mp.span_metric(spanning_space, gradation, q - p);
    MpAtq                   = Mq.intersection(MpAtq);

    const double relative_norm_q = (MpAtq - Mq).norm() / Mq.norm();

    if (relative_norm_q > relative_tolerance)
    {
      Mq            = MpAtq;
      metricChanged = true;
    };

    // FIXME: We already span Mq, to save one iteration.
    // Is it always the best choice?

    // Idem for Mq at p
    MetricTensor<dim> MqAtp = Mq.span_metric(spanning_space, gradation, p - q);
    MqAtp                   = Mp.intersection(MqAtp);

    const double relative_norm_p = (MqAtp - Mp).norm() / Mp.norm();

    if (relative_norm_p > relative_tolerance)
    {
      Mp            = MqAtp;
      metricChanged = true;
    };

    return metricChanged;
  }
} // namespace MetricTensorTools

#endif
