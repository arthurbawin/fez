
#include <deal.II/base/symmetric_tensor.h>

#include "../../tests.h"

#include "metric_tensor.h"
#include "metric_tensor_tools.h"

/**
 * Intersection of MetricTensors
 */
void test_2d()
{
  const double diff = 1e-12;
  const double tol  = 1e-11;

  {
    // Metrics are almost equal
    MetricTensor<2> m1({3., 4., 2.});
    MetricTensor<2> m2({3. + diff, 4. + diff, 2. + diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<2>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<2>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<2>(m1.intersection(m2), deallog);
  }

  {
    // Both metrics are almost diagonal
    MetricTensor<2> m1({3., 4., diff});
    MetricTensor<2> m2({5., 1., diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<2>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<2>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<2>(m1.intersection(m2), deallog);
  }

  {
    // Metrics are scalar multiples of one another
    // Metrics are not diagonals
    MetricTensor<2> m1({3., 4., 2.});
    MetricTensor<2> m2({6. + diff, 8. + diff, 4. + diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<2>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<2>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<2>(m1.intersection(m2), deallog);
  }

  {
    // One metric is completely contained in the other
    // Case m1 contained in m2
    MetricTensor<2> m1({3., 4., 2.});
    MetricTensor<2> m2({6., 10., 4.});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<2>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<2>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<2>(m1.intersection(m2), deallog);
  }

  {
    // Test from MMG (with deal.II ordering)
    MetricTensor<2> m1({508., 502., -504});
    MetricTensor<2> m2({4020., 1020., -2020});
    MetricTensor<2> sol({4500., 1500., -2500.}); /* Exact intersection */
    deallog << "m1: " << std::endl;
    print_tensor_formatted<2>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<2>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<2>(m1.intersection(m2), deallog);
    SymmetricTensor<2, 2> res =
      SymmetricTensor<2, 2>(m1.intersection(m2)) - SymmetricTensor<2, 2>(sol);
    AssertThrow(res.norm() < 1e-12, ExcInternalError());
  }
  deallog << "OK" << std::endl;
}

void test_3d()
{
  const double diff = 1e-12;
  const double tol  = 1e-11;

  {
    // Metrics are almost equal
    MetricTensor<3> m1({1., 2., 6., 1., -1., 1.});
    MetricTensor<3> m2(
      {1. + diff, 2. + diff, 6. + diff, 1. + diff, -1. + diff, 1. + diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<3>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<3>(m2, deallog);
    deallog << "intersection: " << std::endl;
    print_tensor_formatted<3>(m1.intersection(m2), deallog);
  }

  {
    // Both metrics are almost diagonal
    MetricTensor<3> m1({3., 4., 5., 0., 0., 0.});
    MetricTensor<3> m2({5., 1., 6., 0. + diff, 0. - diff, 0. + diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<3>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<3>(m2, deallog);
    deallog << "intersection: " << std::endl;
    print_tensor_formatted<3>(m1.intersection(m2), deallog);
  }

  {
    // Metrics are scalar multiples of one another (not diagonals)
    MetricTensor<3> m1({1., 2., 6., 1., -1., 1.});
    MetricTensor<3> m2(
      {3. + diff, 6. + diff, 18. - diff, 3. - diff, -3. - diff, 3. - diff});
    deallog << "m1: " << std::endl;
    print_tensor_formatted<3>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<3>(m2, deallog);
    deallog << "intersection: " << std::endl;
    print_tensor_formatted<3>(m1.intersection(m2), deallog);
  }

  {
    // Test from MMG (with deal.II ordering)
    MetricTensor<3> m1(
      {111. / 2., 111. / 2., 111. / 2., -109. / 2., 89. / 2., -91. / 2.});
    MetricTensor<3> m2(
      {409. / 2., 409. / 2., 409. / 2., -393 / 2., -407. / 2., 391. / 2.});
    MetricTensor<3> sol(
      {254., 254., 254., -246., -154., 146.}); /* Exact intersection */
    const MetricTensor<3> res = m1.intersection(m2);
    deallog << "m1: " << std::endl;
    print_tensor_formatted<3>(m1, deallog);
    deallog << "m2: " << std::endl;
    print_tensor_formatted<3>(m2, deallog);
    deallog << "intersection : " << std::endl;
    print_tensor_formatted<3>(res, deallog);
    SymmetricTensor<2, 3> diff =
      SymmetricTensor<2, 3>(res) - SymmetricTensor<2, 3>(sol);
    AssertThrow(diff.norm() < 1e-12, ExcInternalError());
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
