
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/lac/vector.h>

#include "../../tests.h"

#include "metric_tensor.h"

/**
 * Creation of MetricTensors from SymmetricTensors and arrays
 */
void test_metric_2d()
{
  /*
   * Default-initialized a MetricTensor (to the identity)
   */
  MetricTensor<2> m0;
  deallog << "Metric tensor    : " << m0 << std::endl;

  // Initialize from positive-definite SymmetricTensor
  SymmetricTensor<2, 2> t1;
  t1[0][0] = 3; // det = 3*4 - 2*2 = 8 : SPD
  t1[1][1] = 4;
  t1[0][1] = 2;
  deallog << "Symmetric tensor : " << t1 << std::endl;
  MetricTensor<2> m1(t1);
  deallog << "Metric tensor    : " << m1 << std::endl;
  AssertThrow(m1 == t1, ExcInternalError());

  // Print underlying Eigen matrix and eigendecomposition
  const auto e_val = m1.get_eigenvalues();
  const auto e_vec = m1.get_eigenvectors();
  deallog << "Eigen matrix     : " << m1.get_eigen_matrix() << std::endl;
  deallog << "Eigenvalues      : " << e_val << std::endl;
  deallog << "Eigenvectors     : " << e_vec << std::endl;
  deallog << "determinant      : " << determinant(m1) << std::endl;

  auto d  = unit_symmetric_tensor<2>();
  d[0][0] = e_val[0];
  d[1][1] = e_val[1];

  // Reassemble the metric from its eigendecomposition
  MetricTensor<2> assembled(e_vec * d * transpose(e_vec));
  AssertThrow((m1 - assembled).norm() < 1e-14, ExcInternalError());

  /*
   * Initialize from non-PD tensor, this should throw in debug
   */
  SymmetricTensor<2, 2> t2;
  t2[0][0] = 1; // det = 1*4 - 2*2 = 0
  t2[1][1] = 4;
  t2[0][1] = 2;
  deallog << "Symmetric tensor : " << t2 << std::endl;
  try
  {
    MetricTensor<2> m2(t2);
  }
  catch (ExceptionBase &e)
  {
    deallog << e.get_exc_name() << std::endl;
  }

  /*
   * Initialize from an array
   */
  // SPD
  double v1[3] = {3, 4, 2};
  deallog << "Metric tensor    : " << MetricTensor<2>(v1) << std::endl;
  // Non-SPD
  double v2[3] = {1, 4, 2};
  try
  {
    MetricTensor<2> m(v2);
  }
  catch (ExceptionBase &e)
  {
    deallog << e.get_exc_name() << std::endl;
  }
  deallog << "OK" << std::endl;
}

void test_metric_3d()
{
  MetricTensor<3> m0;
  deallog << "Metric tensor    : " << m0 << std::endl;

  // Initialize from positive-definite SymmetricTensor
  SymmetricTensor<2, 3> t1;
  t1[0][0] = 1;
  t1[1][1] = 2;
  t1[2][2] = 27;
  t1[0][1] = 1;
  t1[0][2] = 5;
  t1[1][2] = 6;
  deallog << "Symmetric tensor : " << t1 << std::endl;
  MetricTensor<3> m1(t1);
  deallog << "Metric tensor    : " << m1 << std::endl;
  AssertThrow(m1 == t1, ExcInternalError());

  // Print underlying Eigen matrix and eigendecomposition
  const auto e_val = m1.get_eigenvalues();
  const auto e_vec = m1.get_eigenvectors();
  deallog << "Eigen matrix     : " << m1.get_eigen_matrix() << std::endl;
  deallog << "Eigenvalues      : " << e_val << std::endl;
  deallog << "Eigenvectors     : " << e_vec << std::endl;
  deallog << "determinant      : " << determinant(m1) << std::endl;

  auto d  = unit_symmetric_tensor<3>();
  d[0][0] = e_val[0];
  d[1][1] = e_val[1];
  d[2][2] = e_val[2];

  // Reassemble the metric from its eigendecomposition
  MetricTensor<3> assembled(symmetrize(e_vec * d * transpose(e_vec)));
  AssertThrow((m1 - assembled).norm() < 1e-13, ExcInternalError());

  SymmetricTensor<2, 3> t2;
  t2[0][0] = 2;
  t2[1][1] = 2;
  t2[2][2] = 1;
  t2[0][1] = 3;
  t2[0][2] = 1;
  t2[1][2] = 1;
  deallog << "Symmetric tensor : " << t2 << std::endl;
  try
  {
    MetricTensor<3> m2(t2);
  }
  catch (ExceptionBase &e)
  {
    deallog << e.get_exc_name() << std::endl;
  }

  // From SPD array
  double v1[6] = {1, 2, 27, 1, 5, 6};
  deallog << "Metric tensor    : " << MetricTensor<3>(v1) << std::endl;
  // Non-SPD array
  double v2[6] = {2, 2, 1, 3, 1, 1};
  try
  {
    MetricTensor<3> m(v2);
  }
  catch (ExceptionBase &e)
  {
    deallog << e.get_exc_name() << std::endl;
  }
  deallog << "OK" << std::endl;
}

int main()
{
  deal_II_exceptions::disable_abort_on_exception();

  initlog();
  deallog << std::setprecision(4);

  test_metric_2d();
  test_metric_3d();
}
