#include <assembly/incompressible_chns_assemblers.h>
#include <deal.II/base/quadrature_lib.h>

#include <algorithm>
#include <array>
#include <limits>

#include "../tests.h"

using namespace Assembly::IncompressibleCHNS::EnergyStable;

namespace
{
  void check_close(const double      actual,
                   const double      expected,
                   const double      relative_tolerance = 2.e-7,
                   const double      absolute_tolerance = 2.e-10,
                   const std::string &description = "Profile correction")
  {
    AssertThrow(std::isfinite(actual) && std::isfinite(expected) &&
                  std::abs(actual - expected) <=
                    absolute_tolerance + relative_tolerance *
                                           std::max(std::abs(actual),
                                                    std::abs(expected)),
                ExcMessage(description + " check failed: " +
                           std::to_string(actual) + " != " +
                           std::to_string(expected)));
  }

  void test_inverse_profile()
  {
    const double cutoff =
      std::sqrt(1. - 0.25 * CahnHilliard::profile_correction_tail);
    const double h = 1.e-11;
    const auto   center = CahnHilliard::inverse_profile(cutoff);
    for (const double direction : {-1., 1.})
    {
      const auto side = CahnHilliard::inverse_profile(cutoff + direction * h);
      check_close(side.value - direction * h * center.derivative,
                  center.value,
                  1.e-11,
                  1.e-11,
                  "Inverse value continuity");
      check_close(side.derivative,
                  center.derivative,
                  1.e-7,
                  1.e-10,
                  "Inverse derivative continuity");
    }
    for (const double x :
         {0., 0.2, 0.9, cutoff - 2.e-5, cutoff, cutoff + 2.e-5, 1., 1.03})
    {
      const auto positive = CahnHilliard::inverse_profile(x);
      const auto negative = CahnHilliard::inverse_profile(-x);
      check_close(negative.value, -positive.value, 1.e-14, 1.e-14,
                  "Inverse oddness");
      check_close(negative.derivative, positive.derivative, 1.e-14, 1.e-14,
                  "Inverse derivative evenness");
      const double step = x > 0.99 ? 1.e-9 : 1.e-6;
      const double finite_difference =
        (CahnHilliard::inverse_profile(x + step).value -
         CahnHilliard::inverse_profile(x - step).value) /
        (2. * step);
      check_close(positive.derivative, finite_difference, 2.e-6, 1.e-8,
                  "Inverse finite difference");
    }
    deallog << "Nodal inverse: continuity, oddness and derivatives OK"
            << std::endl;
  }

  template <int dim>
  void test_flux_and_derivatives()
  {
    const double epsilon = 0.13;
    Tensor<1, dim> normal;
    for (unsigned int i = 0; i < dim; ++i)
      normal[i] = i + 1.;
    normal /= normal.norm();
    for (const double r : {-0.999, -0.99, -0.9, -0.3, 0., 0.4, 0.99, 0.999})
    {
      const double Q = (1. - r * r) / (std::sqrt(2.) * epsilon);
      const auto   flux = CahnHilliard::evaluate_profile_correction<dim>(
        r, Q * normal, {}, epsilon, false);
      check_close(flux.flux.norm(), 0., 0.,
                  128. * std::numeric_limits<double>::epsilon() *
                    std::max(1., Q),
                  "Equilibrium profile flux");
    }
    const double transition =
      std::sqrt(1. - 0.375 * CahnHilliard::profile_correction_tail);
    for (const double r :
         {-1.03, -1., -transition, -0.9, -0.2, 0., 0.7, transition, 1., 1.03})
    {
      Tensor<1, dim> gradient;
      const double Q = std::abs(1. - r * r) / (std::sqrt(2.) * epsilon);
      for (unsigned int i = 0; i < dim; ++i)
        gradient[i] = Q * (0.27 + 0.19 * i) + 0.001;
      CahnHilliard::ProfileCorrectionLinearization<dim> derivative;
      CahnHilliard::evaluate_profile_correction<dim>(
        r, gradient, {}, epsilon, false, &derivative);
      const bool tail = std::abs(r) > 0.99 && std::abs(r) < 1.;
      const double phase_step = tail ? 2.e-9 : 1.e-6;
      const auto plus_phase = CahnHilliard::evaluate_profile_correction<dim>(
        r + phase_step, gradient, {}, epsilon, false);
      const auto minus_phase = CahnHilliard::evaluate_profile_correction<dim>(
        r - phase_step, gradient, {}, epsilon, false);
      for (unsigned int i = 0; i < dim; ++i)
        check_close(derivative.flux_phi[i],
                    (plus_phase.flux[i] - minus_phase.flux[i]) /
                      (2. * phase_step),
                    tail ? 5.e-5 : 2.e-7,
                    2.e-8,
                    "Flux phase derivative");
      for (unsigned int j = 0; j < dim; ++j)
      {
        const double step = 1.e-6 * std::max(1.e-3, gradient.norm());
        auto plus_gradient = gradient, minus_gradient = gradient;
        plus_gradient[j] += step;
        minus_gradient[j] -= step;
        const auto plus = CahnHilliard::evaluate_profile_correction<dim>(
          r, plus_gradient, {}, epsilon, false);
        const auto minus = CahnHilliard::evaluate_profile_correction<dim>(
          r, minus_gradient, {}, epsilon, false);
        for (unsigned int i = 0; i < dim; ++i)
          check_close(derivative.flux_gradient[i][j],
                      (plus.flux[i] - minus.flux[i]) / (2. * step),
                      tail ? 5.e-5 : 2.e-7,
                      2.e-8,
                      "Flux gradient derivative");
      }
    }
    deallog << "dim " << dim
            << ": equilibrium flux and local derivatives, including activation tails OK"
            << std::endl;
  }

  struct AnalyticPhase
  {
    double         value;
    Tensor<1, 2> gradient;
    Tensor<2, 2> hessian;
  };

  AnalyticPhase analytic_phase(const Point<2> &point)
  {
    const double x = point[0], y = point[1];
    AnalyticPhase result;
    result.value = 0.22 + 0.13 * std::sin(0.7 * x) + 0.08 * x * y +
                   0.05 * y * y;
    result.gradient[0] = 0.091 * std::cos(0.7 * x) + 0.08 * y;
    result.gradient[1] = 0.08 * x + 0.1 * y;
    result.hessian[0][0] = -0.0637 * std::sin(0.7 * x);
    result.hessian[0][1] = result.hessian[1][0] = 0.08;
    result.hessian[1][1] = 0.1;
    return result;
  }

  void test_divergence()
  {
    const double epsilon = 0.17;
    for (const auto &point :
         {Point<2>(-0.3, 0.2), Point<2>(0.1, -0.4), Point<2>(0.7, 0.8)})
    {
      const auto phase = analytic_phase(point);
      const auto correction = CahnHilliard::evaluate_profile_correction<2>(
        phase.value, phase.gradient, phase.hessian, epsilon);
      double finite_difference = 0.;
      const double step = 2.e-5;
      for (unsigned int j = 0; j < 2; ++j)
      {
        auto plus = point, minus = point;
        plus[j] += step;
        minus[j] -= step;
        const auto phase_plus = analytic_phase(plus);
        const auto phase_minus = analytic_phase(minus);
        const auto flux_plus = CahnHilliard::evaluate_profile_correction<2>(
          phase_plus.value, phase_plus.gradient, {}, epsilon, false);
        const auto flux_minus = CahnHilliard::evaluate_profile_correction<2>(
          phase_minus.value, phase_minus.gradient, {}, epsilon, false);
        finite_difference +=
          (flux_plus.flux[j] - flux_minus.flux[j]) / (2. * step);
      }
      check_close(correction.divergence, finite_difference, 2.e-7, 1.e-8,
                  "Analytic divergence");
    }
    deallog << "Analytic flux divergence matches spatial finite differences OK"
            << std::endl;
  }

  // A manually mapped affine P1 triangle. The quadrature is a genuine
  // simplex rule; the nonlinear reconstructed field is integrated at its nodes.
  struct Triangle
  {
    std::array<Point<2>, 3> vertices = {{Point<2>(-0.04, -0.03),
                                        Point<2>(0.08, -0.01),
                                        Point<2>(0.02, 0.09)}};
    std::vector<Tensor<1, 2>> gradients;
    double                    determinant;

    Triangle()
      : gradients(3)
    {
      const auto a = vertices[1] - vertices[0];
      const auto b = vertices[2] - vertices[0];
      determinant = a[0] * b[1] - a[1] * b[0];
      gradients[1][0] = b[1] / determinant;
      gradients[1][1] = -b[0] / determinant;
      gradients[2][0] = -a[1] / determinant;
      gradients[2][1] = a[0] / determinant;
      gradients[0] = -gradients[1] - gradients[2];
    }

    std::vector<double> shape(const Point<2> &point) const
    {
      return {1. - point[0] - point[1], point[0], point[1]};
    }
  };

  struct ElementCorrection
  {
    std::array<double, 3>                residual{};
    std::array<std::array<double, 3>, 3> jacobian{};
  };

  ElementCorrection element_correction(const Triangle              &triangle,
                                       const std::array<double, 3> &phase,
                                       const double                  epsilon,
                                       const double                  amplitude,
                                       const bool                    equilibrium = false)
  {
    const QGaussSimplex<2> quadrature(4);
    std::vector<CahnHilliard::ProfileInverse> inverse(3);
    for (unsigned int a = 0; a < 3; ++a)
      inverse[a] = CahnHilliard::inverse_profile(phase[a]);
    ElementCorrection result;
    for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      const auto shape = triangle.shape(quadrature.point(q));
      const auto profile = CahnHilliard::reconstruct_profile<2>(
        inverse, shape, triangle.gradients);
      CahnHilliard::ProfileCorrectionLinearization<2> derivative;
      const auto correction = CahnHilliard::evaluate_profile_correction<2>(
        profile.value, profile.gradient(), {}, epsilon, false, &derivative);
      if (equilibrium)
        check_close(correction.flux.norm(), 0., 0.,
                    256. * std::numeric_limits<double>::epsilon() *
                      std::max(1., profile.gradient().norm()),
                    "Reconstructed planar profile");
      const double weight =
        std::abs(triangle.determinant) * quadrature.weight(q);
      for (unsigned int a = 0; a < 3; ++a)
      {
        result.residual[a] +=
          weight * amplitude * (correction.flux * triangle.gradients[a]);
        for (unsigned int b = 0; b < 3; ++b)
        {
          const auto variation = profile.variation(
            inverse[b].derivative * shape[b],
            inverse[b].derivative * triangle.gradients[b], {}, false);
          const auto flux_variation =
            derivative.variation(variation.value, variation.gradient, {});
          result.jacobian[a][b] +=
            weight * amplitude * (flux_variation.flux * triangle.gradients[a]);
        }
      }
    }
    return result;
  }

  void test_triangle()
  {
    const Triangle triangle;
    const double epsilon = 0.1, amplitude = 0.07;
    std::array<double, 3> planar;
    const double cutoff =
      std::sqrt(1. - 0.25 * CahnHilliard::profile_correction_tail);
    for (unsigned int a = 0; a < 3; ++a)
    {
      const auto &x = triangle.vertices[a];
      const double distance = 0.6 * x[0] + 0.8 * x[1] - 0.01;
      planar[a] = std::tanh(distance / (std::sqrt(2.) * epsilon));
      AssertThrow(std::abs(planar[a]) < cutoff,
                  ExcMessage("Planar nodes must use the exact inverse branch."));
    }
    element_correction(triangle, planar, epsilon, amplitude, true);
    const std::array<std::array<double, 3>, 5> cases =
      {{{{-0.7, 0.1, 0.6}},
        {{-0.9996, -0.9993, -0.9989}},
        {{0.9988, 0.9994, 0.99965}},
        {{0.97, 1.00002, 0.985}},
        {{-1.00005, -0.97, -0.985}}}};
    for (unsigned int k = 0; k < cases.size(); ++k)
    {
      const auto center =
        element_correction(triangle, cases[k], epsilon, amplitude);
      const double step = k == 0 ? 1.e-6 : 1.e-8;
      for (unsigned int b = 0; b < 3; ++b)
      {
        auto plus = cases[k], minus = cases[k];
        plus[b] += step;
        minus[b] -= step;
        const auto positive =
          element_correction(triangle, plus, epsilon, amplitude);
        const auto negative =
          element_correction(triangle, minus, epsilon, amplitude);
        for (unsigned int a = 0; a < 3; ++a)
          check_close(center.jacobian[a][b],
                      (positive.residual[a] - negative.residual[a]) /
                        (2. * step),
                      k == 0 ? 2.e-6 : 5.e-5,
                      2.e-8,
                      "P1 element nodal Jacobian");
      }
    }
    deallog << "P1 triangle: exact planar reconstruction and nodal Jacobians "
               "in interface, tails and overshoots OK"
            << std::endl;
  }

  template <int dim>
  void test_coupled_rows()
  {
    ParameterHandler prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    prm.enter_subsection("Cahn Hilliard");
    prm.set("CHNS model", "quasi_incompressible");
    prm.leave_subsection();
    parameters.read_parameters(prm);
    Field<dim> state, phase_test, pressure_test, other_test, direction;
    state.phase = 0.2;
    state.potential = -0.1;
    Tensor<1, dim> added_flux, gradient;
    for (unsigned int i = 0; i < dim; ++i)
    {
      added_flux[i] = 0.17 - 0.09 * i;
      gradient[i] = -0.23 + 0.15 * i;
      state.potential_gradient[i] = 0.3 + 0.11 * i;
      state.pressure_gradient[i] = -0.07 * (i + 1);
      other_test.velocity[i] = 0.2 * (i + 1);
      other_test.potential_gradient[i] = 0.1;
    }
    phase_test.phase_gradient = gradient;
    pressure_test.pressure_gradient = gradient;
    other_test.potential = 0.3;
    direction.profile_flux_variation = added_flux;
    for (const double rho0 : {1., 20., 1000.})
    {
      const double rho1 = 1.;
      const auto before = make_point_data(parameters, rho0, rho1, 0.2, 0.1,
                                           {10., -10.}, state.phase, -0.1,
                                           Tensor<1, dim>(), Tensor<1, dim>());
      auto after = before;
      after.profile_flux = added_flux;
      const double phase_change =
        residual(state, phase_test, after) - residual(state, phase_test, before);
      const double pressure_change = residual(state, pressure_test, after) -
                                     residual(state, pressure_test, before);
      const double product = added_flux * gradient;
      check_close(phase_change, product, 1.e-13, 1.e-13,
                  "Residual phase row");
      check_close(pressure_change, -before.alpha * product, 1.e-13, 1.e-13,
                  "Residual pressure row");
      check_close(0.5 * (rho0 - rho1) * phase_change -
                    0.5 * (rho0 + rho1) * pressure_change,
                  0., 0., 2.e-11, "Residual mass cancellation");
      const double phase_derivative =
        jacobian(state, phase_test, direction, before);
      const double pressure_derivative =
        jacobian(state, pressure_test, direction, before);
      check_close(phase_derivative, product, 1.e-13, 1.e-13,
                  "Jacobian phase row");
      check_close(pressure_derivative, -before.alpha * product, 1.e-13, 1.e-13,
                  "Jacobian pressure row");
      check_close(0.5 * (rho0 - rho1) * phase_derivative -
                    0.5 * (rho0 + rho1) * pressure_derivative,
                  0., 0., 2.e-11, "Jacobian mass cancellation");
      check_close(residual(state, other_test, after),
                  residual(state, other_test, before), 0., 0.,
                  "Unchanged momentum and potential residuals");
      check_close(jacobian(state, other_test, direction, before), 0., 0., 0.,
                  "Unchanged momentum and potential Jacobians");
    }
    deallog << "dim " << dim
            << ": shared phase/pressure flux, Jacobian and local mass cancellation OK"
            << std::endl;
  }

  void test_amplitude()
  {
    ParameterHandler prm;
    Parameters::CahnHilliard<2> parameters;
    parameters.declare_parameters(prm);
    prm.enter_subsection("Cahn Hilliard");
    prm.set("CHNS model", "quasi_incompressible");
    prm.set("interface thickness", "0.07");
    prm.set("surface tension", "0.9");
    prm.set("mobility", "0.013");
    prm.set("degenerate absolute mobility prefactor", "0.037");
    prm.set("profile correction strength", "0.6");
    prm.set("profile correction reference mobility", "0.023");
    prm.leave_subsection();
    parameters.read_parameters(prm);
    check_close(CahnHilliard::profile_correction_amplitude(parameters),
                0., 0., 0., "Disabled amplitude");
    for (const std::string law :
         {"constant", "degenerate_absolute", "degenerate"})
    {
      prm.enter_subsection("Cahn Hilliard");
      prm.set("enable profile correction", "true");
      prm.set("mobility model", law);
      prm.leave_subsection();
      parameters.read_parameters(prm);
      const double reference = law == "degenerate_absolute" ?
                                 0.037 * 0.07 * 0.07 :
                                 (law == "degenerate" ? 0.023 : 0.013);
      const double expected =
        2. * 0.6 * (3. * 0.9 / (2. * std::sqrt(2.))) / 0.07 * reference;
      check_close(CahnHilliard::profile_correction_amplitude(parameters),
                  expected, 1.e-14, 1.e-14, "Constant reference amplitude");
    }
    parameters.profile_correction_strength = 0.;
    check_close(CahnHilliard::profile_correction_amplitude(parameters),
                0., 0., 0., "Zero-strength amplitude");
    deallog << "Constant amplitude for all three mobility laws OK" << std::endl;
  }
} // namespace

int main()
{
  initlog();
  test_inverse_profile();
  test_flux_and_derivatives<2>();
  test_flux_and_derivatives<3>();
  test_divergence();
  test_triangle();
  test_coupled_rows<2>();
  test_coupled_rows<3>();
  test_amplitude();
}
