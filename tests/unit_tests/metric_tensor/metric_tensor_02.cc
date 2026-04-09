
#include <deal.II/base/symmetric_tensor.h>

#include <iomanip>

#include "../../tests.h"

#include "metric_tensor.h"
#include "metric_tensor_tools.h"
#include "parameters.h"

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
  deallog << "Metric tensor : " << std::endl;
  print_tensor_formatted<2>(m, deallog);

  // Matrix logarithm and exponential
  deallog << "log(M) : " << std::endl;
  print_tensor_formatted<2>(m.log(), deallog);
  deallog << "exp(M) : " << std::endl;
  print_tensor_formatted<2>(m.exp(), deallog);

  // Bound metric eigenvalues
  // The original eigenvalues are [1.438, 5.562]
  MetricTensor<2> m_copy(m);
  deallog << m_copy.get_eigenvalues() << std::endl;
  {
    // Bound the lower eigenvalue
    const double min_eigenvalue = 2;
    const double max_eigenvalue = 6.;
    m_copy.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded lower eigenvalue : " << std::endl;
    print_tensor_formatted<2>(m_copy, deallog);
    deallog << "eigenvalues are now : " << std::endl;
    deallog << m_copy.get_eigenvalues() << std::endl;

    // Compare with the function that returns a metric
    AssertThrow((m_copy - m.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }
  {
    // Bound the upper eigenvalue
    const double min_eigenvalue = 2;
    const double max_eigenvalue = 4.;
    m_copy.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded upper eigenvalue : " << std::endl;
    print_tensor_formatted<2>(m_copy, deallog);
    deallog << "eigenvalues are now : " << std::endl;
    deallog << m_copy.get_eigenvalues() << std::endl;
    AssertThrow((m_copy - m.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
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
    deallog << "T non-SPD : " << std::endl;
    print_tensor_formatted<2>(t, deallog);
    deallog << "eigenvalues : " << std::endl;
    deallog << eigenvalues << std::endl;

    const auto absT = MetricTensorTools::absolute_value(t, 1e-10, 1e10);
    deallog << "abs(T) : " << std::endl;
    print_tensor_formatted<2>(absT, deallog);
    MetricTensorTools::eigen_decomposition_symmetric(absT,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "eigenvalues : " << std::endl;
    deallog << eigenvalues << std::endl;
  }

  {
    // Square root and inverse square root
    const auto sqrt_m  = m.sqrt();
    const auto isqrt_m = m.inverse_sqrt();
    deallog << "sqrt(M) : " << std::endl;
    print_tensor_formatted<2>(sqrt_m, deallog);
    deallog << "isqrt(M) : " << std::endl;
    print_tensor_formatted<2>(isqrt_m, deallog);
    deallog << "sqrt(M) * isqrt(M) : " << std::endl;
    print_tensor_formatted<2>(Tensor<2, 2>(sqrt_m) * Tensor<2, 2>(isqrt_m),
                              deallog);
    // Product should be the identity
    AssertThrow((Tensor<2, 2>(sqrt_m) * Tensor<2, 2>(isqrt_m) -
                 unit_symmetric_tensor<2>())
                    .norm() < 1e-14,
                ExcInternalError());
  }

  {
    // Spanned metric from p to p + (1,1)
    const double gradation = 1.5;
    Tensor<1, 2> distance;
    distance[0] = 1.;
    distance[1] = 1.;

    deallog << "spanned metric in metric space : " << std::endl;
    print_tensor_formatted<2>(
      m.span_metric(MetricTensor<2>::SpanningSpace::metric,
                    gradation,
                    distance),
      deallog);
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
  deallog << "Metric tensor : " << std::endl;
  print_tensor_formatted<3>(m, deallog);

  // Matrix logarithm and exponential
  deallog << "log(M) : " << std::endl;
  print_tensor_formatted<3>(m.log(), deallog);
  deallog << "exp(M) : " << std::endl;
  print_tensor_formatted<3>(m.exp(), deallog);

  // Original eigenvalues are [6.076513e-02, 2.593272e+00, 6.345963e+00]
  MetricTensor<3> m_copy(m);
  {
    // Bound the lower eigenvalue
    const double min_eigenvalue = 0.1;
    const double max_eigenvalue = 50.;
    m_copy.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded lower eigenvalue : " << std::endl;
    print_tensor_formatted<3>(m_copy, deallog);
    deallog << "eigenvalues are now : " << std::endl;
    deallog << m_copy.get_eigenvalues() << std::endl;
    AssertThrow((m_copy - m.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
                    .norm() < 1e-13,
                ExcInternalError());
  }
  {
    // Bound the upper eigenvalue
    const double min_eigenvalue = 0.1;
    const double max_eigenvalue = 5.;
    m_copy.bound_eigenvalues(min_eigenvalue, max_eigenvalue);
    deallog << "with bounded upper eigenvalue : " << std::endl;
    print_tensor_formatted<3>(m_copy, deallog);
    deallog << "eigenvalues are now : " << std::endl;
    deallog << m_copy.get_eigenvalues() << std::endl;
    AssertThrow((m_copy - m.bounded_eigenvalues(min_eigenvalue, max_eigenvalue))
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
    deallog << "T non-SPD : " << std::endl;
    print_tensor_formatted<3>(t, deallog);
    deallog << "eigenvalues : " << std::endl;
    deallog << eigenvalues << std::endl;

    const auto absT = MetricTensorTools::absolute_value(t, 1e-10, 1e10);
    deallog << "abs(T) : " << std::endl;
    print_tensor_formatted<3>(absT, deallog);
    MetricTensorTools::eigen_decomposition_symmetric(absT,
                                                     eigenvalues,
                                                     eigenvectors);
    deallog << "eigenvalues : " << std::endl;
    deallog << eigenvalues << std::endl;
  }

  {
    // Square root and inverse square root
    const auto sqrt_m  = m.sqrt();
    const auto isqrt_m = m.inverse_sqrt();
    deallog << "sqrt(M) : " << std::endl;
    print_tensor_formatted<3>(sqrt_m, deallog);
    deallog << "isqrt(M) : " << std::endl;
    print_tensor_formatted<3>(isqrt_m, deallog);
    deallog << "sqrt(M) * isqrt(M) : " << std::endl;
    print_tensor_formatted<3>(Tensor<2, 3>(sqrt_m) * Tensor<2, 3>(isqrt_m),
                              deallog);
    // Product should be the identity
    AssertThrow((Tensor<2, 3>(sqrt_m) * Tensor<2, 3>(isqrt_m) -
                 unit_symmetric_tensor<3>())
                    .norm() < 1e-13,
                ExcInternalError());
  }

  {
    // Spanned metric from p to p + (1,1)
    const double gradation = 1.5;
    Tensor<1, 3> distance;
    distance[0] = 1.;
    distance[1] = 1.;
    distance[2] = 1.;

    deallog << "spanned metric in metric space : " << std::endl;
    print_tensor_formatted<3>(
      m.span_metric(MetricTensor<3>::SpanningSpace::metric,
                    gradation,
                    distance),
      deallog);
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
