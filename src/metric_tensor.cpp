
#include <deal.II/lac/vector.h>
#include <metric_intersection_mmg.h>
#include <metric_tensor.h>
#include <metric_tensor_tools.h>

#include <unsupported/Eigen/MatrixFunctions>

template <int dim>
using EigenRealMatrix = typename MetricTensor<dim>::EigenRealMatrix;
template <int dim>
using EigenRealVector = typename MetricTensor<dim>::EigenRealVector;

template <int dim>
MetricTensor<dim>::MetricTensor(const SymmetricTensor<2, dim> &t)
  : SymmetricTensor<2, dim>(t)
  , matrix_eigen(MetricTensorTools::metric2eigen(*this))
{
  AssertAssignFromSPD(t);
  MetricTensorTools::eigen_decomposition_symmetric(
    matrix_eigen, eigenvalues, eigenvectors, eigenvalues_t, eigenvectors_t);
}

template <int dim>
MetricTensor<dim>::MetricTensor(const Tensor<2, dim> &t)
  : SymmetricTensor<2, dim>(t)
  , matrix_eigen(MetricTensorTools::metric2eigen(*this))
{
  AssertAssignFromSPD(*this);
  MetricTensorTools::eigen_decomposition_symmetric(
    matrix_eigen, eigenvalues, eigenvectors, eigenvalues_t, eigenvectors_t);
}

template <int dim>
MetricTensor<dim>::MetricTensor(const Tensor<2, dim> &eigenvectors,
                                const Tensor<1, dim> &eigenvalues)
  : SymmetricTensor<2, dim>()
  , eigenvalues(MetricTensorTools::vector2eigen(eigenvalues))
  , eigenvectors(MetricTensorTools::tensor2eigen<dim>(eigenvectors))
  , eigenvalues_t(eigenvalues)
  , eigenvectors_t(eigenvectors)
{
  auto diag = unit_symmetric_tensor<dim>();
  for (int d = 0; d < dim; ++d)
  {
    Assert(eigenvalues_t[d] > 0., ExcEigenvalueNotPositive(eigenvalues_t[d]));
    diag[d][d] = eigenvalues_t[d];
  }

  if constexpr (running_in_debug_mode())
    if (!is_orthonormal(eigenvectors_t))
    {
      std::ostringstream oss;
      oss << eigenvectors_t;
      Assert(false, ExcEigenvectorsNotOrthonormal(oss.str()));
    }

  // Set this tensor to Q * D * Q^T
  SymmetricTensor<2, dim>::operator=(
    symmetrize(eigenvectors_t * diag * transpose(eigenvectors_t)));

  // Double check that the result is SPD
  AssertAssignFromSPD(*this);
  MetricTensorTools::metric2eigen(*this, matrix_eigen);
}

template <int dim>
MetricTensor<dim>::MetricTensor(const double (&array)[n_independent_components])
  : SymmetricTensor<2, dim>(array)
  , matrix_eigen(MetricTensorTools::metric2eigen(*this))
{
  AssertAssignFromSPDArray(*this);
  MetricTensorTools::eigen_decomposition_symmetric(
    matrix_eigen, eigenvalues, eigenvectors, eigenvalues_t, eigenvectors_t);
}

template <int dim>
MetricTensor<dim>::MetricTensor(const MetricTensor<dim> &m)
  : SymmetricTensor<2, dim>(m)
  , matrix_eigen(m.matrix_eigen)
  , eigenvalues(m.eigenvalues)
  , eigenvectors(m.eigenvectors)
  , eigenvalues_t(m.eigenvalues_t)
  , eigenvectors_t(m.eigenvectors_t)
{
  AssertAssignFromSPD(m);
}

template <int dim>
MetricTensor<dim> &MetricTensor<dim>::operator=(const MetricTensor<dim> &m)
{
  AssertAssignFromSPD(m);
  SymmetricTensor<2, dim>::operator=(m);
  matrix_eigen   = m.matrix_eigen;
  eigenvalues    = m.eigenvalues;
  eigenvectors   = m.eigenvectors;
  eigenvalues_t  = m.eigenvalues_t;
  eigenvectors_t = m.eigenvectors_t;
  return *this;
}

template <int dim>
MetricTensor<dim> &
MetricTensor<dim>::operator=(const SymmetricTensor<2, dim> &t)
{
  AssertAssignFromSPD(t);
  SymmetricTensor<2, dim>::operator=(t);
  MetricTensorTools::metric2eigen(*this, matrix_eigen);
  MetricTensorTools::eigen_decomposition_symmetric(
    matrix_eigen, eigenvalues, eigenvectors, eigenvalues_t, eigenvectors_t);
  return *this;
}

template <int dim>
MetricTensor<dim> &
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
  MetricTensorTools::metric2eigen(*this, matrix_eigen);
  MetricTensorTools::eigen_decomposition_symmetric(
    matrix_eigen, eigenvalues, eigenvectors, eigenvalues_t, eigenvectors_t);
  return *this;
}

template <int dim>
const Tensor<1, dim> &MetricTensor<dim>::get_eigenvalues() const
{
  return eigenvalues_t;
}

template <int dim>
const Tensor<2, dim> &MetricTensor<dim>::get_eigenvectors() const
{
  return eigenvectors_t;
}

template <int dim>
const typename MetricTensor<dim>::EigenRealMatrix &
MetricTensor<dim>::get_eigen_matrix() const
{
  return matrix_eigen;
}

template <int dim>
const typename MetricTensor<dim>::EigenRealVector &
MetricTensor<dim>::get_eigenvalues_as_eigen() const
{
  return eigenvalues;
}

template <int dim>
const typename MetricTensor<dim>::EigenRealMatrix &
MetricTensor<dim>::get_eigenvectors_as_eigen() const
{
  return eigenvectors;
}

template <int dim>
void MetricTensor<dim>::bound_eigenvalues(const double min_eigenvalue,
                                          const double max_eigenvalue)
{
  Assert(min_eigenvalue > 0., ExcMessage("min_eigenvalue must be positive"));
  Assert(max_eigenvalue > 0., ExcMessage("max_eigenvalue must be positive"));
  Assert(max_eigenvalue >= min_eigenvalue,
         ExcMessage("max_eigenvalue must be greater than min_eigenvalue"));

  // Update the eigenvalues stored as both Eigen and Tensor<1, dim>
  for (unsigned int d = 0; d < dim; ++d)
  {
    eigenvalues[d] =
      std::min(max_eigenvalue, std::max(min_eigenvalue, eigenvalues[d]));
    eigenvalues_t[d] =
      std::min(max_eigenvalue, std::max(min_eigenvalue, eigenvalues_t[d]));
  }
  matrix_eigen =
    eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
  MetricTensorTools::eigen2metric(matrix_eigen, *this);
}

template <int dim>
MetricTensor<dim>
MetricTensor<dim>::bounded_eigenvalues(const double min_eigenvalue,
                                       const double max_eigenvalue) const
{
  MetricTensor<dim> res(*this);
  res.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
  return res;
}

template <int dim>
SymmetricTensor<2, dim> MetricTensor<dim>::log() const
{
  return MetricTensorTools::eigen2symtensor<dim>(matrix_eigen.log());
}

template <int dim>
MetricTensor<dim> MetricTensor<dim>::exp() const
{
  return MetricTensorTools::eigen2metric<dim>(matrix_eigen.exp());
}

template <int dim>
MetricTensor<dim> MetricTensor<dim>::sqrt() const
{
  return MetricTensorTools::eigen2metric<dim>(matrix_eigen.sqrt());
}

template <int dim>
MetricTensor<dim> MetricTensor<dim>::inverse_sqrt() const
{
  return MetricTensorTools::eigen2metric<dim>(matrix_eigen.pow(-0.5));
}

template <int dim>
MetricTensor<dim>
MetricTensor<dim>::intersection(const MetricTensor<dim> &other) const
{
  // MMG accounts for min/max sizes directly in the intersection
  // Give dummies for now
  const double hmin = 1e-15;
  const double hmax = 1e22;

  MetricTensor<dim> res;

  // Forward the call to MMG5_intersecmet22 or 33
  if constexpr (dim == 2)
  {
    double    m[3] = {(*this)[0][0], (*this)[0][1], (*this)[1][1]};
    double    n[3] = {other[0][0], other[0][1], other[1][1]};
    double    inter[3];
    const int mmg_retval = MMG5_intersecmet22(hmin, hmax, m, n, inter);
    AssertThrow(mmg_retval == 1,
                ExcMessage("Could not compute metric intersection with MMG"));
    res[0][0] = inter[0];
    res[0][1] = inter[1];
    res[1][1] = inter[2];
  }
  else
  {
    double    m[6] = {(*this)[0][0],
                      (*this)[0][1],
                      (*this)[0][2],
                      (*this)[1][1],
                      (*this)[1][2],
                      (*this)[2][2]};
    double    n[6] = {other[0][0],
                      other[0][1],
                      other[0][2],
                      other[1][1],
                      other[1][2],
                      other[2][2]};
    double    inter[6];
    const int mmg_retval = MMG5_intersecmet33(hmin, hmax, m, n, inter);
    AssertThrow(mmg_retval == 1,
                ExcMessage("Could not compute metric intersection with MMG"));
    res[0][0] = inter[0];
    res[0][1] = inter[1];
    res[0][2] = inter[2];
    res[1][1] = inter[3];
    res[1][2] = inter[4];
    res[2][2] = inter[5];
  }

  return res;
}

template <int dim>
MetricTensor<dim>
MetricTensor<dim>::span_metric(const SpanningSpace   spanning_space,
                               const double          gradation,
                               const Tensor<1, dim> &pq) const
{
  if (spanning_space == euclidean)
  {
    // Span metric in euclidean space
    DEAL_II_NOT_IMPLEMENTED();
  }
  else if (spanning_space == metric)
  {
    // Span metric in metric space: scale all entries equally
    const double dot_prod = pq * (*this) * pq;
    double       eta      = 1. + std::sqrt(dot_prod) * std::log(gradation);
    return (*this) / (eta * eta);
  }
  else if (spanning_space == exp_metric)
  {
    DEAL_II_NOT_IMPLEMENTED();
  }
  else
    DEAL_II_NOT_IMPLEMENTED();
  return *this;
}

template class MetricTensor<2>;
template class MetricTensor<3>;
