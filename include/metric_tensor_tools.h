#ifndef METRIC_TENSOR_TOOLS_H
#define METRIC_TENSOR_TOOLS_H

// #include <dealii/lac/vector.h>

#include <metric_tensor.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace MetricTensorTools
{
  using namespace dealii;

  /**
   * Take the absolute value P * |D| * P^T of the symmetric matrix stored
   * in vec with N*(N+1)/2 components.
   */
  // template <int dim>
  // MetricTensor<dim> absolute_value(const Vector<double> &vec);

  /**
   *
   */
  template <int dim>
  typename EigenRealMatrix<dim>::type metric2eigen(const Tensor<2, dim> &tensor)
  {
    typename EigenRealMatrix<dim>::type eigenMat;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigenMat(i, j) = tensor[i][j];
    return eigenMat;
  }

  /**
   *
   */
  template <int dim>
  typename EigenRealMatrix<dim>::type
  metric2eigen(const MetricTensor<dim> &metric)
  {
    typename EigenRealMatrix<dim>::type eigenMat;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        eigenMat(i, j) = metric[i][j];
    return eigenMat;
  }

  /**
   * Return a new MetricTensor
   */
  template <int dim>
  MetricTensor<dim>
  eigen2metric(const Eigen::Matrix<double, dim, dim> &eigenMat)
  {
    MetricTensor<dim> metric;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        metric[i][j] = eigenMat(i, j);
    return metric;
  }

  /**
   * Assign into existing MetricTensor
   */
  template <int dim>
  void eigen2metric(const Eigen::Matrix<double, dim, dim> &eigenMat,
                    MetricTensor<dim>                     &metric)
  {
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i; j < dim; ++j)
        metric[i][j] = eigenMat(i, j);
  }
} // namespace MetricTensorTools

#endif
