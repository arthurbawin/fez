#include <cahn_hilliard.h>

#include "../tests.h"

namespace
{
  template <int dim>
  void check_derivatives()
  {
    Parameters::CahnHilliard<dim> parameters;
    Tensor<1, dim>                velocity, gradient, direction;
    for (unsigned int d = 0; d < dim; ++d)
    {
      velocity[d]  = 0.3 + 0.2 * d;
      gradient[d]  = d == 0 ? -0.7 : 0.4;
      direction[d] = d == 0 ? 0.2 : -0.6;
    }
    direction /= direction.norm();

    for (const double coefficient : {1.7, 2e-7})
      for (const double delta : {0.2, 1e-8})
        for (const double speed : {0., 1e-9 * delta, delta, 1.})
        {
          const auto u        = speed * velocity;
          auto       evaluate = [&](const Tensor<1, dim> &v) {
            return CahnHilliard::evaluate_adaptative_mobility_3(
              parameters, 0.7, 1., 0., v, gradient, coefficient, delta);
          };
          const auto evaluation = evaluate(u);
          const auto reversed   = evaluate(-u);
          const auto hessian =
            CahnHilliard::adaptive_mobility_3_velocity_hessian<dim>(u,
                                                                    coefficient,
                                                                    delta);
          const auto reversed_hessian =
            CahnHilliard::adaptive_mobility_3_velocity_hessian<dim>(-u,
                                                                    coefficient,
                                                                    delta);
          const double radius = std::hypot(u.norm(), delta);
          const double step   = 2e-5 * radius;

          // Differences of the public value and first-derivative outputs
          // independently check both derivative orders.
          const auto   plus     = evaluate(u + step * direction);
          const auto   minus    = evaluate(u - step * direction);
          const double first_fd = (plus.value - minus.value) / (2. * step);
          const auto   second_fd =
            (plus.derivative_wrt_velocity - minus.derivative_wrt_velocity) /
            (2. * step);
          AssertThrow(std::abs(first_fd - evaluation.derivative_wrt_velocity *
                                            direction) < 2e-7 * coefficient,
                      ExcMessage(
                        "Mobility 3 velocity derivative differs from FD"));
          AssertThrow((second_fd - hessian * direction).norm() <
                        2e-7 * coefficient / radius,
                      ExcMessage(
                        "Mobility 3 velocity Hessian differs from FD"));
          AssertThrow(evaluation.value == reversed.value &&
                        (evaluation.derivative_wrt_velocity +
                         reversed.derivative_wrt_velocity)
                            .norm() == 0. &&
                        (hessian - reversed_hessian).norm() == 0.,
                      ExcMessage("Mobility 3 velocity reversal parity failed"));
          AssertThrow(evaluation.derivative_wrt_tracer == 0. &&
                        evaluation.second_derivative_wrt_tracer == 0. &&
                        evaluation.adaptive_sensitivity == 0.,
                      ExcMessage("Mobility 3 acquired a tracer dependency"));
          const auto changed_tracer =
            CahnHilliard::evaluate_adaptative_mobility_3(parameters,
                                                         -0.9,
                                                         0.3,
                                                         0.4,
                                                         u,
                                                         -3. * gradient,
                                                         coefficient,
                                                         delta);
          AssertThrow(changed_tracer.value == evaluation.value &&
                        (changed_tracer.derivative_wrt_velocity -
                         evaluation.derivative_wrt_velocity)
                            .norm() == 0.,
                      ExcMessage(
                        "Mobility 3 depends on tracer or its gradient"));
          if (speed == 0.)
          {
            AssertThrow(evaluation.derivative_wrt_velocity.norm() == 0.,
                        ExcMessage("Regularized mobility derivative at rest"));
            for (unsigned int i = 0; i < dim; ++i)
              for (unsigned int j = 0; j < dim; ++j)
                AssertThrow(std::abs(hessian[i][j] -
                                     (i == j ? coefficient / delta : 0.)) <
                              1e-13 * coefficient / delta,
                            ExcMessage("Regularized mobility Hessian at rest"));
          }
        }
    deallog << "Adaptive mobility 3 velocity derivatives and Hessian in " << dim
            << "D OK" << std::endl;
  }

  template <int dim>
  void check_other_models()
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    parameters.read_parameters(prm);
    parameters.adaptive_mobility_m = 0.;
    Tensor<1, dim> velocity, gradient, direction;
    velocity[0]  = 0.4;
    velocity[1]  = -0.2;
    gradient[0]  = 0.3;
    gradient[1]  = 0.7;
    direction[0] = -0.1;
    direction[1] = 0.5;
    if constexpr (dim == 3)
    {
      velocity[2]  = 0.6;
      gradient[2]  = -0.2;
      direction[2] = 0.3;
    }
    for (const auto evaluator :
         {&CahnHilliard::evaluate_constant_mobility<dim>,
          &CahnHilliard::evaluate_degenerate_mobility<dim>})
      AssertThrow(
        evaluator(parameters, 0.7, 1., 0., velocity, gradient, 1.3, 0.2)
            .derivative_wrt_velocity.norm() == 0.,
        ExcMessage("Velocity-independent mobility derivative is nonzero"));
    for (const auto evaluator :
         {&CahnHilliard::evaluate_adaptative_mobility<dim>,
          &CahnHilliard::evaluate_adaptative_mobility_2<dim>})
    {
      const auto evaluation =
        evaluator(parameters, 0.7, 1., 0., velocity, gradient, 1.3, 0.2);
      const double h          = 1e-6;
      const double difference = (evaluator(parameters,
                                           0.7,
                                           1.,
                                           0.,
                                           velocity + h * direction,
                                           gradient,
                                           1.3,
                                           0.2)
                                   .value -
                                 evaluator(parameters,
                                           0.7,
                                           1.,
                                           0.,
                                           velocity - h * direction,
                                           gradient,
                                           1.3,
                                           0.2)
                                   .value) /
                                (2. * h);
      AssertThrow(std::abs(difference - evaluation.derivative_wrt_velocity *
                                          direction) < 1e-9,
                  ExcMessage(
                    "Gradient-dependent mobility velocity derivative"));
    }
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  check_derivatives<2>();
  check_derivatives<3>();
  check_other_models<2>();
  check_other_models<3>();
  deallog << "Existing mobility models retain their velocity derivatives"
          << std::endl;
}
