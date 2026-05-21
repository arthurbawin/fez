#ifndef H_TARGET_TOOLS_H
#define H_TARGET_TOOLS_H

#include <deal.II/base/tensor.h>

#include <algorithm>
#include <cmath>

using namespace dealii;

namespace HTargetTools
{
  template <int dim>
  inline double
  reference_size_from_cell_measure(const double measure)
  {
    return std::pow(std::max(measure, 1e-30), 1.0 / dim);
  }

  inline double
  smooth_time_ramp(const double time,
                   const double ramp_time)
  {
    if (ramp_time <= 0.)
      return 1.;

    const double s = std::max(0., std::min(time / ramp_time, 1.));
    return s * s * (3. - 2. * s);
  }

  inline double
  target_size_from_unbounded_variable(const double eta,
                                      const double h_min)
  {
    return h_min + std::exp(eta);
  }

  inline double
  target_size_derivative_from_unbounded_variable(const double eta)
  {
    return std::exp(eta);
  }

  inline double
  target_size_second_derivative_from_unbounded_variable(const double eta)
  {
    return std::exp(eta);
  }

  inline double
  unbounded_variable_from_target_size(const double h,
                                      const double h_min)
  {
    const double eps_h = 1e-14;
    return std::log(std::max(h - h_min, eps_h));
  }

  inline double
  clamp_value(const double value,
              const double lower,
              const double upper)
  {
    return std::max(lower, std::min(value, upper));
  }

  inline double
  smooth_step_tanh(const double g,
                   const double g_min,
                   const double g_max)
  {
    if (g_max <= g_min)
      return g >= g_max ? 1.0 : 0.0;

    const double g0 =
      0.5 * (g_min + g_max);

    const double delta_g =
      (g_max - g_min) / (2.0 * std::atanh(0.9));

    return 0.5 * (1.0 + std::tanh((g - g0) / delta_g));
  }

  inline double
  smooth_step_tanh_derivative(const double g,
                              const double g_min,
                              const double g_max)
  {
    if (g_max <= g_min)
      return 0.0;

    const double g0 =
      0.5 * (g_min + g_max);

    const double delta_g =
      (g_max - g_min) / (2.0 * std::atanh(0.9));

    const double argument =
      (g - g0) / delta_g;

    const double sech =
      1.0 / std::cosh(argument);

    return 0.5 * sech * sech / delta_g;
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_from_recovered_gradient(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u,
    const double          eps)
  {
    const double norm_u =
      std::sqrt(u * u + eps * eps);

    Tensor<1, dim> grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
        grad_abs_u[a] += u[c] / norm_u * grad_u[c][a];

    return grad_abs_u;
  }

  template <int dim>
  double
  gradient_abs_velocity_weight(const Tensor<1, dim> &grad_abs_u,
                               const double          g_min,
                               const double          g_max,
                               const double          eps)
  {
    (void)eps;

    return smooth_step_tanh(grad_abs_u.norm(),
                            g_min,
                            g_max);
  }

  inline double
  target_size_from_weight(const double weight,
                          const double h_background,
                          const double h_min)
  {
    const double lower =
      std::min(h_min, h_background);

    const double upper =
      std::max(h_min, h_background);

    return clamp_value(h_background + weight * (h_min - h_background),
                       lower,
                       upper);
  }

  template <int dim>
  double
  target_size_from_gradient_abs_velocity(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          g_min,
    const double          g_ref,
    const double          g_max,
    const double          exponent,
    const double          eps)
  {
    (void)g_ref;
    (void)exponent;

    const double weight =
      gradient_abs_velocity_weight<dim>(grad_abs_u,
                                        g_min,
                                        g_max,
                                        eps);

    return target_size_from_weight(weight,
                                   h_background,
                                   h_min);
  }

  template <int dim>
  Tensor<1, dim>
  target_size_from_gradient_abs_velocity_derivative(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          g_min,
    const double          g_ref,
    const double          g_max,
    const double          exponent,
    const double          eps)
  {
    (void)g_ref;
    (void)exponent;

    Tensor<1, dim> derivative;

    const double g =
      grad_abs_u.norm();

    if (g <= eps)
      return derivative;

    const double dh_dg =
      (h_min - h_background)
      * smooth_step_tanh_derivative(g,
                                    g_min,
                                    g_max);

    derivative =
      dh_dg / g * grad_abs_u;

    return derivative;
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_variation(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u,
    const Tensor<1, dim> &du,
    const Tensor<2, dim> &dgrad_u,
    const double          eps)
  {
    const double norm_u =
      std::sqrt(u * u + eps * eps);

    const double norm_u3 =
      norm_u * norm_u * norm_u;

    const double u_dot_du =
      u * du;

    Tensor<1, dim> d_grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
      {
        const double d_u_over_norm =
          du[c] / norm_u
          - u[c] * u_dot_du / norm_u3;

        d_grad_abs_u[a] +=
          d_u_over_norm * grad_u[c][a]
          + u[c] / norm_u * dgrad_u[c][a];
      }

    return d_grad_abs_u;
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_variation_wrt_position(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u,
    const Tensor<2, dim> &grad_delta_x,
    const double          eps)
  {
    const double norm_u =
      std::sqrt(u * u + eps * eps);

    const Tensor<2, dim> d_grad_u =
      -grad_u * grad_delta_x;

    Tensor<1, dim> d_grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
        d_grad_abs_u[a] +=
          u[c] / norm_u * d_grad_u[c][a];

    return d_grad_abs_u;
  }

} // namespace HTargetTools

#endif
