#ifndef METRIC_TENSOR_TOOLS_H
#define METRIC_TENSOR_TOOLS_H

// #include <dealii/lac/vector.h>

#include <metric_tensor.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace MetricTensorTools
{
  using namespace dealii;

  template <int dim>
  DeclException1(ExcSymDecompositionFailed,
                 typename EigenRealMatrix<dim>::type,
                 << "Eigen decomposition failed for symmetric matrix:" << arg1
                 << ".");

  template <int dim>
  DeclException1(ExcEigenMatNotSym,
                 typename EigenRealMatrix<dim>::type,
                 << "Eigen matrix is expected to be symmetric, but is not:"
                 << arg1 << ".");

  /**
   * Take the absolute value P * |D| * P^T of the symmetric matrix stored
   * in vec with N*(N+1)/2 components.
   */
  // template <int dim>
  // MetricTensor<dim> absolute_value(const Vector<double> &vec);

  /**
   * Return a new Eigen matrix
   */
  template <int dim>
  typename EigenRealMatrix<dim>::type
  metric2eigen(const MetricTensor<dim> &metric)
  {
    typename EigenRealMatrix<dim>::type eigen_matrix;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = metric[i][j];
    return eigen_matrix;
  }

  /**
   * Fill an existing Eigen matrix
   */
  template <int dim>
  void metric2eigen(const MetricTensor<dim>            &metric,
                    typename EigenRealMatrix<dim>::type eigen_matrix)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigen_matrix(i, j) = metric[i][j];
  }

  /**
   * Return a new MetricTensor
   */
  template <int dim>
  MetricTensor<dim>
  eigen2metric(const Eigen::Matrix<double, dim, dim> &eigen_matrix)
  {
    MetricTensor<dim> metric;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        metric[i][j] = eigen_matrix(i, j);
    return metric;
  }

  /**
   * Fill an existing MetricTensor
   */
  template <int dim>
  void eigen2metric(const Eigen::Matrix<double, dim, dim> &eigen_matrix,
                    MetricTensor<dim>                     &metric)
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
  void
  eigen_decomposition_symmetric(const Eigen::Matrix<double, dim, dim> &A,
                                Eigen::Matrix<double, dim, 1>   &eigenvalues,
                                Eigen::Matrix<double, dim, dim> &eigenvectors,
                                Tensor<1, dim>                  &eigenvalues_t,
                                Tensor<2, dim>                  &eigenvectors_t)
  {
    Assert(A.isApprox(A.transpose()), ExcEigenMatNotSym<dim>(A));

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, dim, dim>> eigensolver(
      A, Eigen::ComputeEigenvectors);

    Assert(eigensolver.info() == Eigen::Success,
           ExcSymDecompositionFailed<dim>(A));

    // Real-valued vector of eigenvalues
    eigenvalues = eigensolver.eigenvalues();
    // Columns are the orthonormal eigenvectors
    eigenvectors = eigensolver.eigenvectors();

    // Store as deal.II tensors
    for (unsigned int di = 0; di < dim; ++di)
    {
      eigenvalues_t[di] = eigenvalues(di);
      for (unsigned int dj = 0; dj < dim; ++dj)
        eigenvectors_t[di][dj] = eigenvectors(di, dj);
    }
  }
} // namespace MetricTensorTools

#endif
