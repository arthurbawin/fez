#include <assembly/incompressible_chns_assemblers.h>
#include <time_handler.h>

#include <array>

#include "../tests.h"

using namespace Assembly::IncompressibleCHNS::EnergyStable;

void check(const double a, const double b, const double tolerance = 2e-8)
{
  AssertThrow(std::abs(a - b) <=
                tolerance * std::max({1., std::abs(a), std::abs(b)}),
              ExcMessage("Quasi-incompressible CHNS check failed: " +
                         std::to_string(a) + " != " + std::to_string(b)));
}

template <int dim>
Field<dim> shifted(Field<dim> s, const Field<dim> &z, const double h)
{
  s.velocity += h * z.velocity;
  s.velocity_gradient += h * z.velocity_gradient;
  s.pressure += h * z.pressure;
  s.pressure_gradient += h * z.pressure_gradient;
  s.phase += h * z.phase;
  s.phase_gradient += h * z.phase_gradient;
  s.potential += h * z.potential;
  s.potential_gradient += h * z.potential_gradient;
  return s;
}

template <int dim>
std::vector<Field<dim>> jet_basis()
{
  std::vector<Field<dim>> basis;
  for (unsigned int i = 0; i < dim; ++i)
  {
    Field<dim> z;
    z.velocity[i] = 1.;
    basis.push_back(z);
    for (unsigned int j = 0; j < dim; ++j)
    {
      z                         = {};
      z.velocity_gradient[i][j] = 1.;
      basis.push_back(z);
    }
  }
  for (unsigned int k = 0; k < 3; ++k)
  {
    Field<dim> z;
    (k == 0 ? z.pressure : (k == 1 ? z.phase : z.potential)) = 1.;
    basis.push_back(z);
    for (unsigned int j = 0; j < dim; ++j)
    {
      z                                                                = {};
      (k == 0 ? z.pressure_gradient :
                (k == 1 ? z.phase_gradient : z.potential_gradient))[j] = 1.;
      basis.push_back(z);
    }
  }
  return basis;
}

template <int dim>
void test()
{
  ParameterHandler              prm;
  Parameters::CahnHilliard<dim> param;
  param.declare_parameters(prm);
  prm.enter_subsection("Cahn Hilliard");
  prm.set("CHNS model", "quasi_incompressible");
  prm.set("interface thickness", "0.07");
  prm.set("surface tension", "0.9");
  prm.set("degenerate absolute mobility prefactor", "0.037");
  prm.set("enable mobility tracer limiter", "false");
  prm.enter_subsection("degenerate mobility");
  prm.set("Function expression", "0.013*(1-x*x)^2");
  prm.leave_subsection();
  prm.leave_subsection();
  Field<dim> s;
  s.pressure  = 0.7;
  s.potential = -0.23;
  Tensor<1, dim> old_v, body;
  for (unsigned int i = 0; i < dim; ++i)
  {
    s.velocity[i]           = 0.12 * (i + 1);
    old_v[i]                = -0.07 * (i + 1);
    body[i]                 = -0.3 * (i + 1);
    s.pressure_gradient[i]  = 0.31 - 0.1 * i;
    s.potential_gradient[i] = -0.13 + 0.2 * i;
    s.phase_gradient[i]     = 0.27 + 0.1 * i;
    for (unsigned int j = 0; j < dim; ++j)
      s.velocity_gradient[i][j] = 0.15 + 0.11 * i - 0.09 * j;
  }
  const auto basis = jet_basis<dim>();
  for (const std::string law :
       {"constant", "degenerate", "degenerate_absolute"})
  {
    prm.enter_subsection("Cahn Hilliard");
    prm.set("mobility model", law);
    prm.leave_subsection();
    param.read_parameters(prm);
    for (const auto rho : std::array<std::array<double, 2>, 3>{
           {{{1., 1.}}, {{1000., 1.}}, {{1., 1000.}}}})
      for (const double phi : {-1.3, -0.8, 0.2, 0.9, 1.4})
        for (const double old_phi : {-1.2, 0.1, 1.1})
          for (const std::vector<double> &coefficients :
               {std::vector<double>{1. / 0.017, -1. / 0.017, 0.},
                std::vector<double>{1.5 / 0.017, -2. / 0.017, 0.5 / 0.017},
                std::vector<double>{1. / 0.017 + 1. / 0.043,
                                    -1. / 0.017 - 1. / 0.026,
                                    0.017 / (0.026 * 0.043)}})
          {
            s.phase              = phi;
            const auto make_data = [&](const Field<dim> &a) {
              return make_point_data(param,
                                     rho[0],
                                     rho[1],
                                     1.3,
                                     0.07,
                                     coefficients,
                                     a.phase,
                                     old_phi,
                                     old_v,
                                     body,
                                     -0.45,
                                     0.3 * old_v);
            };
            const auto data = make_data(s);
            // The article mobility is used directly in the new model, without
            // the density conversion needed by the legacy Stepien variable.
            if (law == "degenerate_absolute")
              check(data.mobility,
                    0.037 * 0.07 * 0.07 * std::abs(1. - phi * phi));
            check(data.positive_density,
                  CahnHilliard::linear_mixing(CahnHilliard::tracer_limiter(phi),
                                              rho[0],
                                              rho[1]));
            for (const auto &t : basis)
              for (const auto &z : basis)
              {
                const auto f = [&](const double h) {
                  const auto a = shifted(s, z, h);
                  return residual(a, t, make_data(a));
                };
                const double h = 1e-3;
                const double fd =
                  (-f(2. * h) + 8. * f(h) - 8. * f(-h) + f(-2. * h)) /
                  (12. * h);
                check(jacobian(s, t, z, data), fd);
              }
          }
  }
  deallog
    << "dim " << dim
    << ": BDF1/BDF2 constant/variable-step Jacobians, three mobility laws, "
       "equal/unequal densities and extensions OK"
    << std::endl;

  // Independently verify the discrete kinetic/free-energy chain rule. The
  // combined energy test is w=v, q=-p, psi=mu, xi=-(phi-old_phi)/dt.
  // Flux, pressure and capillary terms cancel; convection is antisymmetric.
  body = {};
  for (const double phi : {-1.3, -0.4, 0.7, 1.4})
  {
    s.phase                = phi;
    const double   old_phi = -0.2;
    const double   inv_dt  = 10.;
    const auto     d       = make_point_data(param,
                                   1000.,
                                   1.,
                                   1.3,
                                   0.07,
                                             {inv_dt, -inv_dt},
                                   phi,
                                   old_phi,
                                   old_v,
                                   body);
    Tensor<1, dim> old_grad;
    old_grad[0] = -0.3;
    Field<dim> t;
    t.velocity           = s.velocity;
    t.velocity_gradient  = s.velocity_gradient;
    t.pressure           = -s.pressure;
    t.pressure_gradient  = -s.pressure_gradient;
    t.phase              = s.potential;
    t.phase_gradient     = s.potential_gradient;
    t.potential          = -inv_dt * (phi - old_phi);
    t.potential_gradient = -inv_dt * (s.phase_gradient - old_grad);
    const double delta_kinetic =
      0.5 * d.positive_density * s.velocity.norm_square() -
      0.5 * d.old_positive_density * old_v.norm_square();
    const auto F = [](const double x) {
      return 0.25 * std::pow(x * x - 1., 2);
    };
    const double delta_free =
      d.well_coefficient * (F(phi) - F(old_phi)) +
      0.5 * d.gradient_coefficient *
        (s.phase_gradient.norm_square() - old_grad.norm_square());
    const double numerical =
      0.5 * d.old_positive_density * (s.velocity - old_v).norm_square() +
      0.5 * d.gradient_coefficient *
        (s.phase_gradient - old_grad).norm_square();
    const auto   strain = symmetrize(s.velocity_gradient);
    const double dissipation =
      2. * d.viscosity * scalar_product(strain, strain) -
      2. / dim * d.viscosity * std::pow(trace(s.velocity_gradient), 2) +
      d.mobility *
        (s.potential_gradient + d.alpha * s.pressure_gradient).norm_square();
    check(residual(s, t, d),
          inv_dt * (delta_kinetic + delta_free + numerical) + dissipation,
          1e-12);
  }
  deallog << "dim " << dim << ": discrete energy identity OK" << std::endl;
  for (const double a : {-1.3, -0.7, 0., 0.5, 1.4})
    for (const double b : {a, a + 1e-12, -0.2})
      check(averaged_potential_derivative(a, b) * (a - b),
            0.25 * (std::pow(a * a - 1., 2) - std::pow(b * b - 1., 2)),
            1e-13);
  deallog << "dim " << dim << ": exact potential divided difference OK"
          << std::endl;

  // Independently test the BDF2 kinetic G identity at arbitrary density
  // changes, including clipped phases, and the full temporal energy test.
  for (const double phi : {-1.3, -0.4, 0.7, 1.4})
  {
    s.phase              = phi;
    const double dt      = 0.017;
    const double old_phi = 0.12, older_phi = -1.2;
    const auto   older_v = 0.3 * old_v;
    const auto   d       = make_point_data(param,
                                   1000.,
                                   1.,
                                   1.3,
                                   0.07,
                                           {1.5 / dt, -2. / dt, 0.5 / dt},
                                   phi,
                                   old_phi,
                                   old_v,
                                   body,
                                   older_phi,
                                   older_v);
    const auto   a       = std::sqrt(d.positive_density) * s.velocity;
    const auto   b       = std::sqrt(d.old_positive_density) * old_v;
    const auto   c       = std::sqrt(CahnHilliard::linear_mixing(
                     CahnHilliard::tracer_limiter(older_phi), 1000., 1.)) *
                   older_v;
    const double modified_change =
      0.25 * (a.norm_square() + (2. * a - b).norm_square() - b.norm_square() -
              (2. * b - c).norm_square());
    check(dt * (inertia(s.velocity, d) * s.velocity),
          modified_change + 0.25 * (a - 2. * b + c).norm_square(),
          1e-12);

    Tensor<1, dim> old_grad, older_grad;
    old_grad[0]             = -0.3;
    older_grad[0]           = 0.2;
    const double phase_rate = d.c0 * phi + d.phase_history;
    const auto   gradient_rate =
      (1.5 * s.phase_gradient - 2. * old_grad + 0.5 * older_grad) / dt;
    Field<dim> t;
    t.velocity           = s.velocity;
    t.velocity_gradient  = s.velocity_gradient;
    t.pressure           = -s.pressure;
    t.pressure_gradient  = -s.pressure_gradient;
    t.phase              = s.potential;
    t.phase_gradient     = s.potential_gradient;
    t.potential          = -phase_rate;
    t.potential_gradient = -gradient_rate;
    const auto   strain  = symmetrize(s.velocity_gradient);
    const double dissipation =
      2. * d.viscosity * scalar_product(strain, strain) -
      2. / dim * d.viscosity * std::pow(trace(s.velocity_gradient), 2) +
      d.mobility *
        (s.potential_gradient + d.alpha * s.pressure_gradient).norm_square();
    check(residual(s, t, d),
          inertia(s.velocity, d) * s.velocity +
            d.well_coefficient * phi * (phi * phi - 1.) * phase_rate +
            d.gradient_coefficient * (s.phase_gradient * gradient_rate) +
            dissipation,
          1e-12);
  }
  deallog << "dim " << dim
          << ": BDF2 kinetic identity and temporal energy test OK" << std::endl;

  // Check pure-phase invariance under a pressure gradient, density-independent
  // mobility in article variables, and the constant legacy flux conversion.
  for (const double prefactor : {0., 0.025, 0.1})
  {
    prm.enter_subsection("Cahn Hilliard");
    prm.set("mobility model", "degenerate_absolute");
    prm.set("degenerate absolute mobility prefactor",
            std::to_string(prefactor));
    prm.leave_subsection();
    param.read_parameters(prm);
    for (const double phi : {-1., -0.6, 0., 0.7, 1.})
    {
      const auto a = CahnHilliard::evaluate_mobility(param, phi, 1000., 1.);
      const auto b = CahnHilliard::evaluate_mobility(param, phi, 1., 1000.);
      const auto c = CahnHilliard::evaluate_mobility(param, phi, 1., 1.);
      check(a.value, b.value);
      check(a.value, c.value);
      if (std::abs(phi) == 1. || prefactor == 0.)
      {
        check(a.value, 0.);
        check(a.value *
                (s.potential_gradient - 1.e6 * s.pressure_gradient).norm(),
              0.);
      }
      prm.enter_subsection("Cahn Hilliard");
      prm.set("CHNS model", "stepien");
      prm.leave_subsection();
      param.read_parameters(prm);
      const auto legacy =
        CahnHilliard::evaluate_mobility(param, phi, 1000., 1.);
      const double k = 2. * 1000. / 1001.;
      check(legacy.value, k * k * a.value);
      prm.enter_subsection("Cahn Hilliard");
      prm.set("CHNS model", "quasi_incompressible");
      prm.leave_subsection();
      param.read_parameters(prm);
    }
  }
  deallog << "dim " << dim
          << ": density-independent mobility, pure phases and prefactor OK"
          << std::endl;
}

int main()
{
  initlog();
  // A restart directly after the BDF1 startup must retain its accepted dt
  // and complete the startup schedule before using the nominal BDF2 step.
  ParameterHandler            prm;
  Parameters::TimeIntegration param;
  param.declare_parameters(prm);
  prm.enter_subsection("Time integration");
  prm.set("verbosity", "quiet");
  prm.set("scheme", "BDF2");
  prm.set("bdf start method", "BDF1");
  prm.set("bdf start step ratio", "0.5");
  prm.set("dt", "0.002");
  prm.set("t_initial", "0");
  prm.set("t_end", "0.04");
  prm.leave_subsection();
  param.read_parameters(prm);
  // ParameterReader normally sets this from the mesh-adaptation settings.
  param.n_time_intervals = 1;
  TimeHandler              time(param);
  const ConditionalOStream quiet(std::cout, false);
  time.advance(quiet);
  check(time.bdf_coefficients[2], 0.);
  time.update_parameters_after_restart(param);
  check(time.current_dt, 0.001);
  check(time.time_steps[0], 0.001);
  check(time.bdf_coefficients[2], 0.);
  time.advance(quiet);
  check(time.current_time, 0.002);
  check(time.bdf_coefficients[0], 1500.);
  check(time.bdf_coefficients[1], -2000.);
  check(time.bdf_coefficients[2], 500.);
  // Changing the next step must not overwrite the accepted previous one.
  param.dt = 0.0005;
  time.update_parameters_after_restart(param);
  check(time.time_steps[0], 0.001);
  time.advance(quiet);
  check(time.current_time, 0.0025);
  check(time.bdf_coefficients[0], 1. / 0.0005 + 1. / 0.0015);
  check(time.bdf_coefficients[2], 0.0005 / (0.001 * 0.0015));
  deallog << "BDF2 startup/restart coefficients and accepted-step history OK"
          << std::endl;
  test<2>();
  test<3>();
}
