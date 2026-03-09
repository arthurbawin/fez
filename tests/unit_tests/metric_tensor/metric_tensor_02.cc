
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

#include "../../tests.h"

#include "metric_tensor.h"
#include "metric_tensor_tools.h"

/**
 * Operations on MetricTensors
 */
void test_2d()
{
  SymmetricTensor<2, 2> t;
  t[0][0] = 3;
  t[1][1] = 4;
  t[0][1] = 2;
  MetricTensor<2> m(t);
  deallog << "Metric tensor : " << m << std::endl;

  // Matrix logarithm and exponential
  deallog << "log(M)        : " << m.log() << std::endl;
  deallog << "exp(M)        : " << m.exp() << std::endl;

  // Bound metric eigenvalues
  // The original eigenvalues are [1.438, 5.562]
  MetricTensor<2> m_copy(m);
  {
    // Bound the lower eigenvalue
    const double min_eigenvalue = 2;
    const double max_eigenvalue = 6.;
    m.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded lower eigenvalue : " << m << std::endl;
    deallog << "eigenvalues are now           : " << m.get_eigenvalues()
            << std::endl;

    // Compare with the function that returns a metric
    AssertThrow((m - m_copy.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }
  {
    // Bound the upper eigenvalue
    const double min_eigenvalue = 2;
    const double max_eigenvalue = 4.;
    m.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded upper eigenvalue : " << m << std::endl;
    deallog << "eigenvalues are now           : " << m.get_eigenvalues()
            << std::endl;
    AssertThrow((m - m_copy.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }

  {
    // Absolute value of non-SPD SymmetricTensor
    SymmetricTensor<2, 2> t;
    t[0][0] = 3;
    t[1][1] = 1;
    t[0][1] = 2;
    Tensor<1, 2> eigenvalues;
    Tensor<2, 2> eigenvectors;
    MetricTensorTools::eigen_decomposition_symmetric(t,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "T non-SPD     : " << t << std::endl;
    deallog << "eigenvalues   : " << eigenvalues << std::endl;
    const auto absT = MetricTensorTools::absolute_value(t);
    deallog << "abs(T)        : " << absT << std::endl;
    MetricTensorTools::eigen_decomposition_symmetric(absT,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "eigenvalues   : " << eigenvalues << std::endl;
  }

  deallog << "OK" << std::endl;
}

void test_3d()
{
  SymmetricTensor<2, 3> t;
  t[0][0] = 1;
  t[1][1] = 2;
  t[2][2] = 6;
  t[0][1] = 1;
  t[0][2] = -1;
  t[1][2] = 1;
  MetricTensor<3> m(t);
  deallog << "Metric tensor : " << m << std::endl;
  deallog << "log(M)        : " << m.log() << std::endl;
  deallog << "exp(M)        : " << m.exp() << std::endl;

  // Original eigenvalues are [6.076513e-02, 2.593272e+00, 6.345963e+00]
  MetricTensor<3> m_copy(m);

  {
    // Bound the lower eigenvalue
    const double min_eigenvalue = 0.1;
    const double max_eigenvalue = 50.;
    m.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded lower eigenvalue : " << m << std::endl;
    deallog << "eigenvalues are now           : " << m.get_eigenvalues()
            << std::endl;
    AssertThrow((m - m_copy.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }
  {
    // Bound the upper eigenvalue
    const double min_eigenvalue = 0.1;
    const double max_eigenvalue = 5.;
    m.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded upper eigenvalue : " << m << std::endl;
    deallog << "eigenvalues are now           : " << m.get_eigenvalues()
            << std::endl;
    AssertThrow((m - m_copy.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }

  {
    // Absolute value of non-SPD SymmetricTensor
    SymmetricTensor<2, 3> t;
    t[0][0] = 1;
    t[1][1] = 2;
    t[2][2] = 3;
    t[0][1] = 4;
    t[0][2] = 5;
    t[1][2] = 6;
    Tensor<1, 3> eigenvalues;
    Tensor<2, 3> eigenvectors;
    MetricTensorTools::eigen_decomposition_symmetric(t,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "T non-SPD     : " << t << std::endl;
    deallog << "eigenvalues   : " << eigenvalues << std::endl;
    const auto absT = MetricTensorTools::absolute_value(t);
    deallog << "abs(T)        : " << absT << std::endl;
    MetricTensorTools::eigen_decomposition_symmetric(absT,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "eigenvalues   : " << eigenvalues << std::endl;
  }

  deallog << "OK" << std::endl;
}

int main()
{
  deal_II_exceptions::disable_abort_on_exception();

  initlog();
  deallog << std::setprecision(6);
  deallog << std::scientific;

  test_2d();
  test_3d();
}
