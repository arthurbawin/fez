#include <assembly/sponge_layer.h>

#include <array>

#include "../tests.h"

namespace
{
  void close(const double actual,
             const double expected,
             const double tolerance = 1e-11)
  {
    AssertThrow(
      std::isfinite(actual) &&
        std::abs(actual - expected) <= tolerance * (1. + std::abs(expected)),
      ExcMessage("Unexpected sponge-layer value: " + std::to_string(actual) +
                 " != " + std::to_string(expected)));
  }

  Parameters::SpongeLayer make_sponge()
  {
    Parameters::SpongeLayer sponge{};
    sponge.inflow  = {true, 0., 2., 2., 1., -2., 3., 4.};
    sponge.outflow = {true, 0., 2., 6., -3., 2., -1., -2.};
    return sponge;
  }

  void test_parameters()
  {
    Parameters::SpongeLayer sponge{};
    ParameterHandler        prm;
    sponge.declare_parameters(prm);
    sponge.read_parameters(prm);
    AssertThrow(!sponge.any_enabled(), ExcInternalError());
    for (const auto *band : {&sponge.inflow, &sponge.outflow})
    {
      close(band->sigma_max, .46);
      close(band->x_start, 0.);
      close(band->x_end, 1.);
      close(band->u, 1.);
      close(band->v, 0.);
      close(band->p_ref, 0.);
      close(band->T_ref, 0.);
    }
    // Exercise the actual subsection names and both independent bands.
    prm.parse_input_from_string("subsection Sponge Layer\n"
                                " subsection inflow\n"
                                "  set enable = true\n"
                                "  set x_start = -2\n"
                                "  set x_end = -1\n"
                                "  set sigma_max = 3\n"
                                "  set u = 4\n"
                                "  set v = -5\n"
                                "  set p_ref = 6\n"
                                "  set T_ref = 7\n"
                                " end\n"
                                " subsection outflow\n"
                                "  set enable = true\n"
                                "  set sigma_max = 8\n"
                                " end\n"
                                "end\n");
    sponge.read_parameters(prm);
    AssertThrow(sponge.inflow.enable && sponge.outflow.enable,
                ExcInternalError());
    close(sponge.inflow.x_start, -2.);
    close(sponge.inflow.x_end, -1.);
    close(sponge.inflow.sigma_max, 3.);
    close(sponge.inflow.u, 4.);
    close(sponge.inflow.v, -5.);
    close(sponge.inflow.p_ref, 6.);
    close(sponge.inflow.T_ref, 7.);
    close(sponge.outflow.sigma_max, 8.);

    for (const auto &name : {"inflow", "outflow"})
      for (const double end : {0., -1.})
      {
        Parameters::SpongeLayer invalid{};
        ParameterHandler        p;
        invalid.declare_parameters(p);
        p.enter_subsection("Sponge Layer");
        p.enter_subsection(name);
        p.set("enable", "true");
        p.set("x_end", std::to_string(end));
        p.leave_subsection();
        p.leave_subsection();
        bool rejected = false;
        try
        {
          invalid.read_parameters(p);
        }
        catch (const std::exception &)
        {
          rejected = true;
        }
        AssertThrow(rejected, ExcMessage("Invalid sponge width accepted"));
      }
    bool rejected = false;
    try
    {
      prm.enter_subsection("Sponge Layer");
      prm.enter_subsection("outflow");
      prm.set("sigma_max", "-1");
    }
    catch (const std::exception &)
    {
      rejected = true;
    }
    AssertThrow(rejected, ExcMessage("Negative relaxation rate accepted"));
    deallog << "parameters OK" << std::endl;
  }

  template <int dim>
  void test_profile()
  {
    auto       sponge = make_sponge();
    Point<dim> x;
    // Independent values at the plateau, endpoints, quarter and midpoint.
    const std::array<double, 7> positions{{-1., 0., .5, 1., 1.5, 2., 3.}};
    const std::array<double, 7> values{
      {0., 0., 53. / 512., .5, 459. / 512., 1., 1.}};
    for (unsigned int i = 0; i < positions.size(); ++i)
    {
      x[0] = positions[i];
      close(Assembly::sponge_profile(x, sponge.outflow, false), 6. * values[i]);
      close(Assembly::sponge_profile(x, sponge.inflow, true),
            2. * (1. - values[i]));
    }
    double previous = 0.;
    for (unsigned int i = 0; i <= 100; ++i)
    {
      x[0]               = 2. * i / 100.;
      const double value = Assembly::sponge_profile(x, sponge.outflow, false);
      AssertThrow(value >= previous && value <= 6., ExcInternalError());
      previous = value;
    }
    x[0] = 1.;
    // At the midpoint, sigma_in = 1, sigma_out = 3.
    const auto state = Assembly::compute_sponge_state(x, sponge);
    close(state.sigma_q, 4.);
    close(state.u_inf[0], -2.);
    close(state.u_inf[1], 1.);
    if constexpr (dim == 3)
      close(state.u_inf[2], 0.);
    close(state.p_ref_q, 0.);
    close(state.T_ref_q, -.5);
    sponge.inflow.enable = sponge.outflow.enable = false;
    const auto disabled = Assembly::compute_sponge_state(x, sponge);
    close(disabled.sigma_q, 0.);
    close(disabled.u_inf.norm(), 0.);
    close(disabled.p_ref_q, 0.);
    close(disabled.T_ref_q, 0.);
    deallog << "profile and overlap " << dim << "D OK" << std::endl;
  }

  template <int dim>
  struct Scratch
  {
    const unsigned int                n_q_points = 2, dofs_per_cell = dim + 2;
    std::array<unsigned int, dim + 2> components;
    std::array<Point<dim>, 2>         quadrature_points;
    std::array<double, 2>             JxW_moving{{.7, 1.3}};
    std::array<double, 2>             density, a_p, b_T;
    std::array<double, 2> present_pressure_values, present_temperature_values;
    std::array<Tensor<1, dim>, 2>                      present_velocity_values;
    std::array<std::array<Tensor<1, dim>, dim + 2>, 2> phi_u{};
    std::array<std::array<double, dim + 2>, 2>         phi_p{}, phi_T{};

    Scratch()
    {
      for (unsigned int i = 0; i < dim + 2; ++i)
        components[i] = i;
      quadrature_points[0][0] = .5;
      quadrature_points[1][0] = 1.5;
      for (unsigned int q = 0; q < 2; ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          phi_u[q][d][d] = 1. + .2 * q + .1 * d;
        phi_p[q][dim]     = .8 + .3 * q;
        phi_T[q][dim + 1] = 1.4 - .2 * q;
      }
    }

    void set_state(const Vector<double>                    &dofs,
                   const Assembly::SpongeMaterialConstants &material)
    {
      for (unsigned int q = 0; q < 2; ++q)
      {
        present_velocity_values[q] = Tensor<1, dim>();
        for (unsigned int d = 0; d < dim; ++d)
          present_velocity_values[q] += dofs[d] * phi_u[q][d];
        const double p = present_pressure_values[q] = dofs[dim] * phi_p[q][dim];
        const double T = present_temperature_values[q] =
          dofs[dim + 1] * phi_T[q][dim + 1];
        density[q] = material.rho_ref * (1. + p / material.p_ref) /
                     (1. + T / material.T_ref);
        a_p[q] = 1. / (material.p_ref + p);
        b_T[q] = 1. / (material.T_ref + T);
      }
    }
  };

  template <int dim>
  void test_assembly()
  {
    const Assembly::SpongeMaterialConstants    material{2., 10., 5., 3.};
    const ComponentOrderingCompressibleNS<dim> ordering;
    auto                                       sponge = make_sponge();
    Scratch<dim>                               scratch;
    Vector<double>                             dofs(dim + 2);
    for (unsigned int d = 0; d < dim; ++d)
      dofs[d] = .4 * (d + 1);
    dofs[dim]     = 1.2;
    dofs[dim + 1] = -.3;
    scratch.set_state(dofs, material);
    Vector<double>     rhs(dim + 2);
    FullMatrix<double> matrix(dim + 2, dim + 2);
    Assembly::sponge_layer_rhs<dim>(ordering, sponge, material, scratch, rhs);
    Assembly::sponge_layer_matrix<dim>(
      ordering, sponge, material, scratch, matrix);

    // Independent source oracle: sum each band's rate times its own mismatch,
    // rather than reusing compute_sponge_state or the weighted-reference
    // formula.
    Vector<double>              expected(dim + 2);
    const std::array<double, 2> ramp{{53. / 512., 459. / 512.}};
    for (unsigned int q = 0; q < 2; ++q)
      for (unsigned int band = 0; band < 2; ++band)
      {
        const auto  &b     = band == 0 ? sponge.inflow : sponge.outflow;
        const double sigma = b.sigma_max * (band == 0 ? 1. - ramp[q] : ramp[q]);
        for (unsigned int d = 0; d < dim; ++d)
        {
          const double target = d == 0 ? b.u : (d == 1 ? b.v : 0.);
          expected[d] -= scratch.JxW_moving[q] * scratch.density[q] * sigma *
                         (scratch.present_velocity_values[q][d] - target) *
                         scratch.phi_u[q][d][d];
        }
        expected[dim] -=
          scratch.JxW_moving[q] * sigma * scratch.phi_p[q][dim] *
          ((scratch.present_pressure_values[q] - b.p_ref) /
             (material.p_ref + scratch.present_pressure_values[q]) -
           (scratch.present_temperature_values[q] - b.T_ref) /
             (material.T_ref + scratch.present_temperature_values[q]));
        expected[dim + 1] -= scratch.JxW_moving[q] * scratch.density[q] *
                             material.cp * sigma *
                             (scratch.present_temperature_values[q] - b.T_ref) *
                             scratch.phi_T[q][dim + 1];
      }
    for (unsigned int i = 0; i < dim + 2; ++i)
      close(rhs[i], expected[i]);

    // Active relaxation must add to, rather than replace, the NS terms.
    Vector<double>     seeded_rhs(dim + 2);
    FullMatrix<double> seeded_matrix(dim + 2, dim + 2);
    for (unsigned int i = 0; i < dim + 2; ++i)
    {
      seeded_rhs[i] = 7. + i;
      for (unsigned int j = 0; j < dim + 2; ++j)
        seeded_matrix(i, j) = 3. + i + 2 * j;
    }
    Assembly::sponge_layer_rhs<dim>(
      ordering, sponge, material, scratch, seeded_rhs);
    Assembly::sponge_layer_matrix<dim>(
      ordering, sponge, material, scratch, seeded_matrix);
    for (unsigned int i = 0; i < dim + 2; ++i)
    {
      close(seeded_rhs[i], rhs[i] + 7. + i);
      for (unsigned int j = 0; j < dim + 2; ++j)
        close(seeded_matrix(i, j), matrix(i, j) + 3. + i + 2 * j);
    }

    const double step = 1e-6;
    for (unsigned int j = 0; j < dim + 2; ++j)
    {
      Vector<double> plus(dim + 2), minus(dim + 2);
      auto           perturbed = dofs;
      perturbed[j] += step;
      scratch.set_state(perturbed, material);
      Assembly::sponge_layer_rhs<dim>(
        ordering, sponge, material, scratch, plus);
      perturbed[j] -= 2. * step;
      scratch.set_state(perturbed, material);
      Assembly::sponge_layer_rhs<dim>(
        ordering, sponge, material, scratch, minus);
      for (unsigned int i = 0; i < dim + 2; ++i)
        close(matrix(i, j), -(plus[i] - minus[i]) / (2. * step), 1e-7);
    }
    // A reference state has zero source, with a nonzero relaxation Jacobian.
    for (unsigned int q = 0; q < 2; ++q)
    {
      const auto state =
        Assembly::compute_sponge_state(scratch.quadrature_points[q], sponge);
      scratch.present_velocity_values[q]    = state.u_inf;
      scratch.present_pressure_values[q]    = state.p_ref_q;
      scratch.present_temperature_values[q] = state.T_ref_q;
      scratch.density[q]                    = material.rho_ref *
                           (1. + state.p_ref_q / material.p_ref) /
                           (1. + state.T_ref_q / material.T_ref);
      scratch.a_p[q] = 1. / (material.p_ref + state.p_ref_q);
      scratch.b_T[q] = 1. / (material.T_ref + state.T_ref_q);
    }
    rhs = 0.;
    Assembly::sponge_layer_rhs<dim>(ordering, sponge, material, scratch, rhs);
    close(rhs.l2_norm(), 0.);
    matrix = 0.;
    Assembly::sponge_layer_matrix<dim>(
      ordering, sponge, material, scratch, matrix);
    AssertThrow(matrix.frobenius_norm() > 0., ExcInternalError());

    // Disabled and zero-rate bands must preserve existing element
    // contributions.
    for (const bool zero_rate : {false, true})
    {
      sponge = make_sponge();
      if (zero_rate)
        sponge.inflow.sigma_max = sponge.outflow.sigma_max = 0.;
      else
        sponge.inflow.enable = sponge.outflow.enable = false;
      for (unsigned int i = 0; i < dim + 2; ++i)
      {
        rhs[i] = 7. + i;
        for (unsigned int j = 0; j < dim + 2; ++j)
          matrix(i, j) = 3. + i + 2 * j;
      }
      Assembly::sponge_layer_rhs<dim>(ordering, sponge, material, scratch, rhs);
      Assembly::sponge_layer_matrix<dim>(
        ordering, sponge, material, scratch, matrix);
      for (unsigned int i = 0; i < dim + 2; ++i)
      {
        close(rhs[i], 7. + i);
        for (unsigned int j = 0; j < dim + 2; ++j)
          close(matrix(i, j), 3. + i + 2 * j);
      }
    }
    deallog << "residual and finite-difference Jacobian " << dim << "D OK"
            << std::endl;
  }
} // namespace

int main()
{
  deal_II_exceptions::disable_abort_on_exception();
  try
  {
    initlog();
    test_parameters();
    test_profile<2>();
    test_profile<3>();
    test_assembly<2>();
    test_assembly<3>();
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
