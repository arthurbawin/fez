
#include <deal.II/base/convergence_table.h>
#include <utilities.h>

#include <cmath>
#include <vector>

#include "../tests.h"

using namespace dealii;

double u(const double t)
{
  return std::sin(t);
}

double d2u(const double t)
{
  return -std::sin(t);
}

double d3u(const double t)
{
  return -std::cos(t);
}

/**
 * Test convergence of divided differences.
 * In the time interval [0,2] for decreasing time steps, take the exact
 * function values and evaluate the divided differences for the second
 * and third-order derivatives.
 *
 * Check that the approximate derivatives are first-order accurate for all
 * derivative orders.
 */
void test_divided_differences()
{
  const double t0 = 0.;
  const double t1 = 2.;
  double       h  = 0.4; // Initial time step

  std::vector<double> times_d2(3), values_d2(3);
  std::vector<double> times_d3(4), values_d3(4);

  ConvergenceTable table;

  const unsigned int n_convergence_steps = 8;
  for (unsigned int i = 0; i < n_convergence_steps; ++i, h /= 2.)
  {
    double max_error_dd2 = 0., max_error_dd3 = 0.;

    double t = 0.;
    while (t <= t1 + 1e-12)
    {
      if (t >= 2 * h)
      {
        // Test divided difference of order 2
        // First-order accurate approximation of d2u/dt2
        times_d2         = {t, t - h, t - 2. * h};
        values_d2        = {u(times_d2[0]), u(times_d2[1]), u(times_d2[2])};
        const double dd2 = divided_difference<2>(times_d2, values_d2);

        max_error_dd2 = std::max(max_error_dd2, std::abs(2. * dd2 - d2u(t)));
      }
      if (t >= 3 * h)
      {
        // Test divided difference of order 3
        // First-order accurate approximation of d3u/dt3
        times_d3         = {t, t - h, t - 2. * h, t - 3. * h};
        values_d3        = {u(times_d3[0]),
                            u(times_d3[1]),
                            u(times_d3[2]),
                            u(times_d3[3])};
        const double dd3 = divided_difference<3>(times_d3, values_d3);

        max_error_dd3 = std::max(max_error_dd3, std::abs(6. * dd3 - d3u(t)));
      }
      t += h;
    }

    table.add_value("h", h);
    table.add_value("error_dd2", max_error_dd2);
    table.add_value("error_dd3", max_error_dd3);
  }

  table.set_scientific("error_dd2", true);
  table.set_scientific("error_dd3", true);

  table.write_text(std::cout);
  table.evaluate_convergence_rates("error_dd2",
                                   "h",
                                   ConvergenceTable::reduction_rate_log2,
                                   /*dim = */ 1);
  table.evaluate_convergence_rates("error_dd3",
                                   "h",
                                   ConvergenceTable::reduction_rate_log2,
                                   /*dim = */ 1);

  deallog << "Convergence of divided differences:" << std::endl;
  table.write_text(deallog.get_file_stream());
  deallog << "OK" << std::endl;
}

int main()
{
  initlog();
  deallog << std::setprecision(6);
  deallog << std::scientific;
  test_divided_differences();
}
