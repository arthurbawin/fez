#include <cahn_hilliard.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "../tests.h"

using namespace dealii;

void check(const double value,
           const double expected,
           const double tolerance = 1e-9)
{
  AssertThrow(std::abs(value - expected) <=
                tolerance *
                  std::max({1e-8, std::abs(value), std::abs(expected)}),
              ExcMessage("Stepien/article equivalence check failed."));
}

template <int dim>
void test()
{
  ParameterHandler              prm;
  Parameters::CahnHilliard<dim> param;
  param.declare_parameters(prm);
  prm.enter_subsection("Cahn Hilliard");
  prm.set("CHNS model", "stepien");
  prm.set("mobility model", "degenerate_absolute");
  prm.set("interface thickness", "0.02");
  prm.leave_subsection();
  param.read_parameters(prm);
  AssertThrow(!param.stepien_conservative_transport &&
                !param.stepien_skew_symmetric_inertia &&
                !param.write_stepien_diagnostics,
              ExcMessage("The experimental assembly must be opt-in."));

  for (const auto &densities : std::array<std::array<double, 2>, 4>{
         {{{1., 1.}}, {{1000., 1.}}, {{1., 1000.}}, {{20., 0.3}}}})
  {
    const double r0 = densities[0], r1 = densities[1];
    const double b = 0.5 * (r0 + r1), a = 0.5 * (r0 - r1);
    const double k = r0 * r1 / b;
    for (const double phi : {-1.2, -1., -0.7, 0., 0.4, 1., 1.2})
    {
      const auto   result = CahnHilliard::evaluate_mobility(param, phi, r0, r1);
      const double m      = 0.1 * 0.02 * 0.02 * std::abs(1. - phi * phi);
      check(result.value, k * k * m);
      if (std::abs(phi) == 1.)
      {
        check(result.first_derivative, 0.);
        continue;
      }
      const double h  = 1e-5;
      const auto plus = CahnHilliard::evaluate_mobility(param, phi + h, r0, r1);
      const auto minus =
        CahnHilliard::evaluate_mobility(param, phi - h, r0, r1);
      check(result.first_derivative,
            (plus.value - minus.value) / (2. * h),
            1e-6);
      check(result.second_derivative,
            (plus.first_derivative - minus.first_derivative) / (2. * h),
            1e-6);
      if (std::abs(phi) > 1.)
        continue;

      // Independently express the chemical driving potential in article
      // variables, retaining unequal reference pressures.
      const double mu = 0.37, p = 0.8, pr0 = 3., pr1 = 1.;
      const double rho = b + a * phi;
      const double pr  = 0.5 * ((1. + phi) * pr0 + (1. - phi) * pr1);
      const double pressure_article = p + pr - phi * mu;
      const double w                = mu - a / b * pressure_article;
      const double xi               = (rho * mu - a * (p + pr)) / (r0 * r1);
      check(w, k * xi);
      const double grad_phi = 0.23, grad_mu = -0.4, grad_p = 0.11;
      const double grad_w = grad_mu - a / b *
                                        (grad_p + 0.5 * (pr0 - pr1) * grad_phi -
                                         grad_phi * mu - phi * grad_mu);
      const double grad_xi =
        (rho * grad_mu - a * grad_p + a * (mu - 0.5 * (pr0 - pr1)) * grad_phi) /
        (r0 * r1);
      // The phase flux and the diffusive energy dissipation must agree.
      check(m * grad_w, result.value / k * grad_xi);
      check(m * grad_w * grad_w, result.value * grad_xi * grad_xi);
    }
  }
  deallog << "dim " << dim << ": article mobility and pressure conversion OK"
          << std::endl;

  // BDF2 differentiates the densities of all time levels, including a
  // property limiter in the history. Test the production helper directly.
  param.with_tracer_limiter = true;
  const std::vector<double>              coefficients{150., -200., 50.};
  const std::vector<std::vector<double>> history{{-1.2}, {0.4}};
  const double rho = CahnHilliard::linear_mixing(0.2, 1000., 1.);
  check(CahnHilliard::bdf_density_derivative(
          param, 1000., 1., rho, coefficients, history, 0),
        150. * rho - 200. + 50. * CahnHilliard::linear_mixing(0.4, 1000., 1.));
  deallog << "dim " << dim << ": BDF density history OK" << std::endl;

  prm.enter_subsection("Cahn Hilliard");
  prm.set("enable density mobility compensation", "true");
  prm.leave_subsection();
  bool rejected = false;
  try
  {
    param.read_parameters(prm);
  }
  catch (const std::exception &)
  {
    rejected = true;
  }
  AssertThrow(rejected,
              ExcMessage("Double density compensation must be rejected."));
  deallog << "dim " << dim << ": incompatible compensation rejected OK"
          << std::endl;
}

int main()
{
  initlog();
  test<2>();
  test<3>();
}
