#include <cahn_hilliard.h>

#include <algorithm>
#include <array>
#include <cmath>

#include "../tests.h"

using namespace dealii;

void check_close(const double actual,
                 const double expected,
                 const double tolerance          = 1e-9,
                 const double absolute_tolerance = 0.)
{
  const double scale = std::max({1e-12, std::abs(actual), std::abs(expected)});
  AssertThrow(std::abs(actual - expected) <=
                tolerance * scale + absolute_tolerance,
              ExcMessage("Density-compensated mobility check failed."));
}

template <int dim>
void check_derivatives(const Parameters::CahnHilliard<dim> &param,
                       const double                         phi,
                       const double                         rho0,
                       const double                         rho1)
{
  const double h        = 1e-5;
  const auto   evaluate = [&](const double x) {
    return CahnHilliard::evaluate_mobility(param, x, rho0, rho1);
  };
  const auto center = evaluate(phi);
  const auto minus2 = evaluate(phi - 2. * h);
  const auto minus1 = evaluate(phi - h);
  const auto plus1  = evaluate(phi + h);
  const auto plus2  = evaluate(phi + 2. * h);
  // Differentiate the production values and first derivatives independently.
  const double first_fd =
    (minus2.value - 8. * minus1.value + 8. * plus1.value - plus2.value) /
    (12. * h);
  const double second_fd =
    (minus2.first_derivative - 8. * minus1.first_derivative +
     8. * plus1.first_derivative - plus2.first_derivative) /
    (12. * h);
  const double roundoff_tolerance =
    1e-9 * std::max(std::abs(center.value), 1e-6);
  check_close(center.first_derivative, first_fd, 1e-6, roundoff_tolerance);
  check_close(center.second_derivative, second_fd, 1e-6, roundoff_tolerance);
}

template <int dim>
void test_mobility()
{
  ParameterHandler              prm;
  Parameters::CahnHilliard<dim> param;
  param.declare_parameters(prm);
  prm.enter_subsection("Cahn Hilliard");
  prm.set("CHNS model", "stepien");
  prm.set("mobility", "1e-6");
  prm.leave_subsection();
  param.read_parameters(prm);
  AssertThrow(!param.enable_density_mobility_compensation,
              ExcMessage("Density compensation must be disabled by default."));

  const std::array<std::array<double, 2>, 6> densities = {{{{1., 1.}},
                                                           {{2., 2.}},
                                                           {{10., 1.}},
                                                           {{1000., 1.}},
                                                           {{1., 1000.}},
                                                           {{20., 0.3}}}};
  const std::array<double, 5> phases = {{-0.99, -0.7, 0., 0.35, 0.97}};
  for (const auto &rho : densities)
    for (const double phi : phases)
    {
      const auto result =
        CahnHilliard::evaluate_mobility(param, phi, rho[0], rho[1]);
      check_close(result.value, 1e-6);
      check_close(result.first_derivative, 0.);
      check_close(result.second_derivative, 0.);
    }
  deallog << "dim " << dim << ": default constant mobility OK" << std::endl;

  // Exercise the parameter reader, rather than setting the new member directly.
  prm.enter_subsection("Cahn Hilliard");
  prm.set("enable density mobility compensation", "true");
  prm.leave_subsection();
  param.read_parameters(prm);
  for (const auto &rho : densities)
    for (const double phi : phases)
    {
      const auto result =
        CahnHilliard::evaluate_mobility(param, phi, rho[0], rho[1]);
      const double mixture = CahnHilliard::linear_mixing(phi, rho[0], rho[1]);
      const double normalized_density = mixture / (rho[0] * rho[1]);
      // The principal diffusion coefficient must be M0 for every density ratio.
      check_close(result.value * normalized_density * normalized_density, 1e-6);
      check_derivatives(param, phi, rho[0], rho[1]);
    }
  deallog << "dim " << dim << ": compensated constant mobility OK" << std::endl;

  prm.enter_subsection("Cahn Hilliard");
  prm.set("mobility model", "degenerate");
  prm.enter_subsection("degenerate mobility");
  prm.set("Function expression", "1e-6*(1-x*x)*(1-x*x)");
  prm.leave_subsection();
  prm.leave_subsection();
  param.read_parameters(prm);
  param.enable_density_mobility_compensation = false;
  for (const double phi : phases)
  {
    const auto result = CahnHilliard::evaluate_mobility(param, phi, 1000., 1.);
    check_close(result.value, 1e-6 * std::pow(1. - phi * phi, 2));
    check_derivatives(param, phi, 1000., 1.);
  }
  param.enable_density_mobility_compensation = true;
  for (const auto &rho : densities)
  {
    for (const double phi : phases)
    {
      const auto result =
        CahnHilliard::evaluate_mobility(param, phi, rho[0], rho[1]);
      const double mixture = CahnHilliard::linear_mixing(phi, rho[0], rho[1]);
      const double normalized_density = mixture / (rho[0] * rho[1]);
      check_close(result.value * normalized_density * normalized_density,
                  1e-6 * std::pow(1. - phi * phi, 2));
      check_derivatives(param, phi, rho[0], rho[1]);
    }
    for (const double phi : {-1., 1.})
    {
      const auto result =
        CahnHilliard::evaluate_mobility(param, phi, rho[0], rho[1]);
      check_close(result.value, 0.);
      check_close(result.first_derivative, 0.);
    }
  }
  deallog << "dim " << dim << ": compensated degenerate mobility OK"
          << std::endl;

  // Independently exercise the property limiter and mobility limiter outside
  // [-1,1]. Moderate densities keep the unclipped mixture positive as well.
  for (const bool property_limiter : {false, true})
    for (const bool mobility_limiter : {false, true})
    {
      param.with_tracer_limiter     = property_limiter;
      param.mobility_tracer_limiter = mobility_limiter;
      for (const double phi : {-1.2, 1.2})
        check_derivatives(param, phi, 2., 1.);
    }
  deallog << "dim " << dim << ": limiter chain rules OK" << std::endl;

  param.with_tracer_limiter     = false;
  bool invalid_density_rejected = false;
  try
  {
    CahnHilliard::evaluate_mobility(param, -1.2, 1000., 1.);
  }
  catch (const std::exception &)
  {
    invalid_density_rejected = true;
  }
  AssertThrow(invalid_density_rejected,
              ExcMessage("A non-positive mixture density must be rejected."));

  // A compensation intended for Stepien must not silently affect other models.
  prm.enter_subsection("Cahn Hilliard");
  prm.set("CHNS model", "abels");
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
              ExcMessage("Compensation must reject the Abels model."));
  deallog << "dim " << dim << ": unsupported model rejected OK" << std::endl;
}

int main()
{
  initlog();
  test_mobility<2>();
  test_mobility<3>();
}
