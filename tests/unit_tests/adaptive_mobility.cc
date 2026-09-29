#include <cahn_hilliard.h>

#include <array>
#include <cmath>
#include <string>
#include <utility>

#include "../tests.h"

namespace
{
  template <int dim>
  void check_velocity_derivatives()
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    parameters.read_parameters(prm);
    parameters.epsilon_interface = 0.4;

    Tensor<1, dim> velocity, gradient;
    velocity[0] = 0.4;
    velocity[1] = -0.2;
    gradient[0] = 0.7;
    gradient[1] = 0.3;
    if constexpr (dim == 3)
    {
      velocity[2] = 0.6;
      gradient[2] = -0.2;
    }

    // Start with m > 0 and u.grad(phi) != 0: using the full mobility in
    // the adaptive derivative's denominator fails this finite difference.
    for (const double m : {0.4, 0.})
      for (const double coefficient : {1.3, 0.02})
        for (const double delta : {0.2, 1e-4})
          for (const double speed :
               {1., 0., delta / (coefficient * gradient.norm())})
          {
            parameters.adaptive_mobility_m = m;
            const auto u                   = speed * velocity;
            const auto evaluate            = [&](const Tensor<1, dim> &v) {
              return CahnHilliard::evaluate_adaptative_mobility(
                parameters, 0.7, 1., 0., v, gradient, coefficient, delta);
            };
            const auto   evaluation = evaluate(u);
            const double velocity_scale =
              std::max(u.norm(), delta / (coefficient * gradient.norm()));
            const double   step = 1e-5 * velocity_scale;
            Tensor<1, dim> derivative_fd;
            for (unsigned int d = 0; d < dim; ++d)
            {
              Tensor<1, dim> perturbation;
              perturbation[d]  = step;
              derivative_fd[d] = (evaluate(u + perturbation).value -
                                  evaluate(u - perturbation).value) /
                                 (2. * step);
            }
            const double scaled_error =
              (derivative_fd - evaluation.derivative_wrt_velocity).norm() /
              (coefficient * gradient.norm());
            AssertThrow(scaled_error < 2e-7,
                        ExcMessage(
                          "Adaptive mobility velocity derivative "
                          "differs from FD: dim=" +
                          std::to_string(dim) + ", m=" + std::to_string(m) +
                          ", scaled error=" + std::to_string(scaled_error)));

            const auto reversed = evaluate(-u);
            AssertThrow(evaluation.value == reversed.value &&
                          (evaluation.derivative_wrt_velocity +
                           reversed.derivative_wrt_velocity)
                              .norm() == 0.,
                        ExcMessage("Adaptive mobility velocity reversal "
                                   "parity failed"));
            AssertThrow(evaluation.derivative_wrt_tracer == 0. &&
                          evaluation.second_derivative_wrt_tracer == 0.,
                        ExcMessage("Adaptive mobility acquired an explicit "
                                   "tracer-value dependency"));
            if (speed == 0.)
              AssertThrow(evaluation.derivative_wrt_velocity.norm() == 0.,
                          ExcMessage("Adaptive mobility velocity derivative "
                                     "must vanish at rest"));
          }
    deallog << "Adaptive mobility velocity derivatives in " << dim << "D OK"
            << std::endl;
  }

  template <int dim>
  void check_interfacial_mobility_at_rest()
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    parameters.read_parameters(prm);
    parameters.adaptive_mobility_m = 0.25;

    // For a signed-distance tanh profile, the m contribution is
    // m*(1-phi^2)^2, independently of epsilon. These values are computed
    // by hand for m=1/4 and delta=1/32, including both interface and bulk.
    constexpr std::array<double, 4> phi_values = {{0., 0.5, 0.9, 1.}};
    constexpr std::array<double, 4> expected   = {
      {0.28125, 0.171875, 0.040275, 0.03125}};
    Tensor<1, dim> normal, tangent, rest;
    normal[0]  = 0.6;
    normal[1]  = 0.8;
    tangent[0] = -0.8;
    tangent[1] = 0.6;
    if constexpr (dim == 3)
    {
      normal[0]  = 1. / 3.;
      normal[1]  = 2. / 3.;
      normal[2]  = 2. / 3.;
      tangent[0] = -2. / 3.;
      tangent[1] = 1. / 3.;
    }

    for (const double epsilon : {0.125, 0.5})
    {
      parameters.epsilon_interface = epsilon;
      for (unsigned int k = 0; k < phi_values.size(); ++k)
      {
        const double phi = phi_values[k];
        const auto   gradient =
          (1. - phi * phi) / (std::sqrt(2.) * epsilon) * normal;
        for (const auto &velocity : {rest, tangent})
        {
          const auto evaluation = CahnHilliard::evaluate_adaptative_mobility(
            parameters, phi, 1., 0., velocity, gradient, 1.3, 0.03125);
          AssertThrow(std::abs(evaluation.value - expected[k]) < 1e-14,
                      ExcMessage("The interfacial m contribution must remain "
                                 "active at rest and tangential motion"));
        }
      }
    }
    deallog << "Adaptive mobility interfacial contribution in " << dim << "D OK"
            << std::endl;
  }

  template <int dim>
  void check_gradient_derivatives_and_hessians()
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    parameters.read_parameters(prm);
    parameters.epsilon_interface = 0.4;
    using MobilityModel = typename Parameters::CahnHilliard<dim>::MobilityModel;

    Tensor<1, dim> u, g, zero, orthogonal_u, orthogonal_g;
    for (unsigned int d = 0; d < dim; ++d)
    {
      u[d] = 0.4 - 0.3 * d;
      g[d] = 0.7 - 0.2 * d;
    }
    orthogonal_u[0]                                                       = 0.4;
    orthogonal_g[1]                                                       = 0.7;
    const std::array<std::pair<Tensor<1, dim>, Tensor<1, dim>>, 5> states = {
      {{u, g},
       {zero, g},
       {u, zero},
       {zero, zero},
       {orthogonal_u, orthogonal_g}}};

    for (const auto model :
         {MobilityModel::adaptive, MobilityModel::adaptive_mobility_3})
      for (const double m : {0., 0.4})
        for (const double delta : {0.2, 0.003})
          for (const auto &state : states)
          {
            parameters.mobility_model      = model;
            parameters.adaptive_mobility_m = m;
            constexpr double coefficient   = 1.3;
            const auto       evaluator =
              CahnHilliard::get_mobility_evaluation_function(parameters);
            const auto evaluate = [&](const Tensor<1, dim> &velocity,
                                      const Tensor<1, dim> &gradient) {
              return evaluator(parameters,
                               0.2,
                               1.,
                               0.,
                               velocity,
                               gradient,
                               coefficient,
                               delta);
            };
            const auto &velocity = state.first;
            const auto &gradient = state.second;
            const auto  value    = evaluate(velocity, gradient);
            const auto  hessians =
              CahnHilliard::evaluate_adaptive_mobility_hessians(
                parameters, velocity, gradient, coefficient, delta);
            const double step_u =
              1e-5 *
              std::min(1.,
                       delta / (coefficient * std::max(1., gradient.norm())));
            const double step_g =
              1e-5 *
              std::min(1.,
                       delta / (coefficient * std::max(1., velocity.norm())));
            const std::string context =
              " dim=" + std::to_string(dim) + " m=" + std::to_string(m) +
              " model=" + std::to_string(static_cast<int>(model));
            auto check_vector = [&](const Tensor<1, dim> &actual,
                                    const Tensor<1, dim> &expected,
                                    const std::string    &quantity) {
              AssertThrow((actual - expected).norm() <
                            2e-6 * std::max(1., expected.norm()),
                          ExcMessage(quantity + " differs from FD" + context));
            };

            for (unsigned int d = 0; d < dim; ++d)
            {
              Tensor<1, dim> direction;
              direction[d] = 1.;
              const auto u_plus =
                evaluate(velocity + step_u * direction, gradient);
              const auto u_minus =
                evaluate(velocity - step_u * direction, gradient);
              const auto g_plus =
                evaluate(velocity, gradient + step_g * direction);
              const auto g_minus =
                evaluate(velocity, gradient - step_g * direction);
              const double gradient_fd =
                (g_plus.value - g_minus.value) / (2. * step_g);
              AssertThrow(std::abs(gradient_fd -
                                   value.derivative_wrt_tracer_gradient[d]) <
                            2e-6 * std::max(1., std::abs(gradient_fd)),
                          ExcMessage("Tracer-gradient partial differs from FD" +
                                     context));
              check_vector(hessians.velocity_velocity * direction,
                           (u_plus.derivative_wrt_velocity -
                            u_minus.derivative_wrt_velocity) /
                             (2. * step_u),
                           "Velocity-velocity Hessian");
              check_vector(hessians.velocity_gradient * direction,
                           (g_plus.derivative_wrt_velocity -
                            g_minus.derivative_wrt_velocity) /
                             (2. * step_g),
                           "Velocity-gradient Hessian");
              check_vector(transpose(hessians.velocity_gradient) * direction,
                           (u_plus.derivative_wrt_tracer_gradient -
                            u_minus.derivative_wrt_tracer_gradient) /
                             (2. * step_u),
                           "Gradient-velocity Hessian");
              check_vector(hessians.gradient_gradient * direction,
                           (g_plus.derivative_wrt_tracer_gradient -
                            g_minus.derivative_wrt_tracer_gradient) /
                             (2. * step_g),
                           "Gradient-gradient Hessian");
            }
          }
    deallog << "Adaptive mobility gradient partials and Hessian blocks in "
            << dim << "D OK" << std::endl;
  }

  template <int dim>
  void check_spatial_gradient()
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    parameters.read_parameters(prm);
    parameters.epsilon_interface = 0.4;
    using MobilityModel = typename Parameters::CahnHilliard<dim>::MobilityModel;

    // Independent polynomial fields: u has a nonsymmetric gradient; phi has
    // nonzero diagonal and mixed Hessian entries. Spatial FD evaluates only M.
    const auto fields = [](const Point<dim> &x) {
      Tensor<1, dim> velocity, gradient;
      for (unsigned int i = 0; i < dim; ++i)
      {
        velocity[i] = 0.25 + 0.1 * i + 0.05 * (i + 1) * x[i] * x[i];
        gradient[i] = 0.3 - 0.15 * i + 0.1 * (i + 1) * x[i];
        for (unsigned int j = 0; j < dim; ++j)
          velocity[i] += (0.07 * (i + 1) - 0.04 * (j + 1)) * x[j];
      }
      gradient[0] += 0.12 * x[1];
      gradient[1] += 0.12 * x[0];
      return std::make_pair(velocity, gradient);
    };
    Point<dim>     point;
    Tensor<2, dim> velocity_gradient, tracer_hessian;
    for (unsigned int i = 0; i < dim; ++i)
    {
      point[i]             = 0.2 + 0.1 * i;
      tracer_hessian[i][i] = 0.1 * (i + 1);
      for (unsigned int j = 0; j < dim; ++j)
        velocity_gradient[i][j] = 0.07 * (i + 1) - 0.04 * (j + 1);
      velocity_gradient[i][i] += 0.1 * (i + 1) * point[i];
    }
    tracer_hessian[0][1] = tracer_hessian[1][0] = 0.12;

    for (const auto model :
         {MobilityModel::adaptive, MobilityModel::adaptive_mobility_3})
      for (const double m : {0., 0.4})
      {
        parameters.mobility_model      = model;
        parameters.adaptive_mobility_m = m;
        const auto evaluator =
          CahnHilliard::get_mobility_evaluation_function(parameters);
        const auto evaluate_at = [&](const Point<dim> &x) {
          const auto state = fields(x);
          return evaluator(
            parameters, 0.2, 1., 0., state.first, state.second, 1.3, 0.2);
        };
        const auto value = evaluate_at(point);
        const auto analytical =
          transpose(velocity_gradient) * value.derivative_wrt_velocity +
          tracer_hessian * value.derivative_wrt_tracer_gradient;
        Tensor<1, dim>   numerical;
        constexpr double step = 1e-5;
        for (unsigned int d = 0; d < dim; ++d)
        {
          auto plus = point, minus = point;
          plus[d] += step;
          minus[d] -= step;
          numerical[d] =
            (evaluate_at(plus).value - evaluate_at(minus).value) / (2. * step);
        }
        AssertThrow((analytical - numerical).norm() <
                      1e-8 * std::max(1., numerical.norm()),
                    ExcMessage("Mobility spatial gradient differs from FD"));
      }
    deallog << "Adaptive mobility spatial chain rule in " << dim << "D OK"
            << std::endl;
  }

  void check_negative_m_rejected()
  {
    ParameterHandler            prm;
    Parameters::CahnHilliard<2> parameters;
    parameters.declare_parameters(prm);
    bool rejected = false;
    try
    {
      prm.enter_subsection("Cahn Hilliard");
      prm.set("adaptive mobility m", "-0.1");
      prm.leave_subsection();
      parameters.read_parameters(prm);
    }
    catch (const ExceptionBase &)
    {
      rejected = true;
    }
    AssertThrow(rejected,
                ExcMessage("Negative m permits negative interfacial mobility "
                           "and must be rejected"));
    deallog << "Negative adaptive mobility m is rejected" << std::endl;
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  check_velocity_derivatives<2>();
  check_velocity_derivatives<3>();
  check_interfacial_mobility_at_rest<2>();
  check_interfacial_mobility_at_rest<3>();
  check_negative_m_rejected();
  check_gradient_derivatives_and_hessians<2>();
  check_gradient_derivatives_and_hessians<3>();
  check_spatial_gradient<2>();
  check_spatial_gradient<3>();
}
