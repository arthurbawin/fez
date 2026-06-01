#ifndef ASSEMBLY_MESH_CONCENTRATION_TOOLS_H
#define ASSEMBLY_MESH_CONCENTRATION_TOOLS_H

#include <assembly/ale_geometry.h>
#include <components_ordering.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <cmath>
#include <string>

using namespace dealii;

namespace MeshConcentrationTools
{
  enum class Method
  {
    none,
    force,
    h_target
  };

  enum class ForceVariable
  {
    automatic,
    tracer,
    enlarged_tracer
  };

  enum class HTargetIndicator
  {
    velocity_gradient,
    velocity_hessian
  };

  struct MarkerForcingFactor
  {
    double value      = 0.;
    double derivative = 0.;
  };

  constexpr double velocity_norm_regularization_epsilon = 1e-12;
  constexpr double gradient_norm_regularization_epsilon = 1e-3;
  constexpr double hessian_norm_regularization_epsilon  = 1e-3;

  inline Method
  parse_method(const std::string &method)
  {
    if (method == "none")
      return Method::none;
    if (method == "force")
      return Method::force;
    if (method == "h target" || method == "h_target")
      return Method::h_target;

    AssertThrow(false,
                ExcMessage("Unknown mesh concentration method: " + method));
    return Method::none;
  }

  inline ForceVariable
  parse_force_variable(const std::string &variable)
  {
    if (variable == "automatic")
      return ForceVariable::automatic;
    if (variable == "tracer" || variable == "phase")
      return ForceVariable::tracer;
    if (variable == "enlarged tracer" || variable == "enlarged_tracer" ||
        variable == "psi" || variable == "phase enlarged")
      return ForceVariable::enlarged_tracer;

    AssertThrow(false,
                ExcMessage("Unknown mesh concentration force variable: " +
                           variable));
    return ForceVariable::automatic;
  }

  inline HTargetIndicator
  parse_h_target_indicator(const std::string &indicator)
  {
    if (indicator == "velocity gradient" || indicator == "grad u" ||
        indicator == "velocity_gradient")
      return HTargetIndicator::velocity_gradient;
    if (indicator == "velocity hessian" || indicator == "hessian u" ||
        indicator == "velocity_hessian")
      return HTargetIndicator::velocity_hessian;

    AssertThrow(false,
                ExcMessage("Unknown h_target mesh-concentration indicator: " +
                           indicator));
    return HTargetIndicator::velocity_gradient;
  }

  template <bool with_enlarged>
  inline ForceVariable
  active_force_variable(const ForceVariable variable)
  {
    if (variable == ForceVariable::automatic)
    {
      if constexpr (with_enlarged)
        return ForceVariable::enlarged_tracer;
      else
        return ForceVariable::tracer;
    }

    AssertThrow(variable != ForceVariable::enlarged_tracer || with_enlarged,
                ExcMessage("The enlarged tracer mesh-concentration force "
                           "variable requires the enlarged CHNS-ALE solver."));
    return variable;
  }

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

  template <int dim>
  inline double
  regularized_norm(const Tensor<1, dim> &v,
                   const double          eps)
  {
    return std::sqrt(v * v + eps * eps);
  }

  template <int dim>
  inline double
  tensor3_inner_product(const Tensor<3, dim> &a,
                        const Tensor<3, dim> &b)
  {
    double result = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          result += a[i][j][k] * b[i][j][k];
    return result;
  }

  template <int dim>
  inline double
  regularized_norm(const Tensor<3, dim> &v,
                   const double          eps)
  {
    return std::sqrt(tensor3_inner_product(v, v) + eps * eps);
  }

  inline double
  smooth_step_tanh(const double g,
                   const double g_min,
                   const double g_max,
                   const double steepness)
  {
    Assert(g_max > g_min, ExcMessage("g_max must be greater than g_min."));
    Assert(steepness > 0., ExcMessage("steepness must be positive."));

    const double s        = (g - g_min) / (g_max - g_min);
    const double argument = steepness * (s - 0.5);
    return 0.5 * (1.0 + std::tanh(argument));
  }

  inline double
  smooth_step_tanh_derivative(const double g,
                              const double g_min,
                              const double g_max,
                              const double steepness)
  {
    Assert(g_max > g_min, ExcMessage("g_max must be greater than g_min."));
    Assert(steepness > 0., ExcMessage("steepness must be positive."));

    const double s        = (g - g_min) / (g_max - g_min);
    const double argument = steepness * (s - 0.5);
    const double t        = std::tanh(argument);

    return 0.5 * steepness * (1.0 - t * t) / (g_max - g_min);
  }

  inline double
  target_size_from_weight(const double weight,
                          const double h_background,
                          const double h_min)
  {
    return h_background + weight * (h_min - h_background);
  }

  inline double
  target_size_from_indicator_norm(const double indicator_norm,
                                  const double h_background,
                                  const double h_min,
                                  const double indicator_min,
                                  const double indicator_max,
                                  const double steepness)
  {
    return target_size_from_weight(smooth_step_tanh(indicator_norm,
                                                   indicator_min,
                                                   indicator_max,
                                                   steepness),
                                   h_background,
                                   h_min);
  }

  inline double
  target_size_indicator_norm_derivative(const double indicator_norm,
                                        const double h_background,
                                        const double h_min,
                                        const double indicator_min,
                                        const double indicator_max,
                                        const double steepness)
  {
    return (h_min - h_background)
           * smooth_step_tanh_derivative(indicator_norm,
                                         indicator_min,
                                         indicator_max,
                                         steepness);
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_from_recovered_gradient(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u)
  {
    const double norm_u =
      regularized_norm(u, velocity_norm_regularization_epsilon);

    Tensor<1, dim> grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
        grad_abs_u[a] += u[c] / norm_u * grad_u[c][a];

    return grad_abs_u;
  }

  template <int dim>
  double
  target_size_from_gradient_abs_velocity(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          g_min,
    const double          g_max,
    const double          steepness)
  {
    const double g =
      regularized_norm(grad_abs_u, gradient_norm_regularization_epsilon);
    return target_size_from_indicator_norm(g,
                                           h_background,
                                           h_min,
                                           g_min,
                                           g_max,
                                           steepness);
  }

  template <int dim>
  Tensor<1, dim>
  target_size_from_gradient_abs_velocity_derivative(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          g_min,
    const double          g_max,
    const double          steepness)
  {
    Tensor<1, dim> derivative;

    const double g =
      regularized_norm(grad_abs_u, gradient_norm_regularization_epsilon);
    const double dh_dg =
      target_size_indicator_norm_derivative(g,
                                            h_background,
                                            h_min,
                                            g_min,
                                            g_max,
                                            steepness);

    derivative = dh_dg / g * grad_abs_u;
    return derivative;
  }

  template <int dim>
  double
  target_size_from_velocity_hessian(const Tensor<3, dim> &hessian_u,
                                    const double          h_background,
                                    const double          h_min,
                                    const double          hessian_min,
                                    const double          hessian_max,
                                    const double          steepness)
  {
    const double g =
      regularized_norm(hessian_u, hessian_norm_regularization_epsilon);
    return target_size_from_indicator_norm(g,
                                           h_background,
                                           h_min,
                                           hessian_min,
                                           hessian_max,
                                           steepness);
  }

  template <int dim>
  Tensor<3, dim>
  target_size_from_velocity_hessian_derivative(
    const Tensor<3, dim> &hessian_u,
    const double          h_background,
    const double          h_min,
    const double          hessian_min,
    const double          hessian_max,
    const double          steepness)
  {
    Tensor<3, dim> derivative;

    const double g =
      regularized_norm(hessian_u, hessian_norm_regularization_epsilon);
    const double dh_dg =
      target_size_indicator_norm_derivative(g,
                                            h_background,
                                            h_min,
                                            hessian_min,
                                            hessian_max,
                                            steepness);

    const double factor = dh_dg / g;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          derivative[i][j][k] = factor * hessian_u[i][j][k];

    return derivative;
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_variation(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u,
    const Tensor<1, dim> &du,
    const Tensor<2, dim> &dgrad_u)
  {
    const double norm_u =
      regularized_norm(u, velocity_norm_regularization_epsilon);
    const double norm_u3 = norm_u * norm_u * norm_u;
    const double u_dot_du = u * du;

    Tensor<1, dim> d_grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
      {
        const double d_u_over_norm =
          du[c] / norm_u - u[c] * u_dot_du / norm_u3;

        d_grad_abs_u[a] += d_u_over_norm * grad_u[c][a] +
                           u[c] / norm_u * dgrad_u[c][a];
      }

    return d_grad_abs_u;
  }

  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_variation_wrt_position(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &grad_u,
    const Tensor<2, dim> &grad_delta_x)
  {
    const double norm_u =
      regularized_norm(u, velocity_norm_regularization_epsilon);

    const Tensor<2, dim> d_grad_u = -grad_u * grad_delta_x;

    Tensor<1, dim> d_grad_abs_u;

    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
        d_grad_abs_u[a] += u[c] / norm_u * d_grad_u[c][a];

    return d_grad_abs_u;
  }

  template <int dim>
  Tensor<3, dim>
  velocity_hessian_variation_wrt_position(
    const Tensor<2, dim> &grad_u,
    const Tensor<3, dim> &hessian_u,
    const Tensor<2, dim> &grad_delta_x,
    const Tensor<3, dim> &hessian_delta_x)
  {
    Tensor<3, dim> d_hessian_u;

    for (unsigned int c = 0; c < dim; ++c)
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
          for (unsigned int a = 0; a < dim; ++a)
            d_hessian_u[c][i][j] -=
              grad_delta_x[a][i] * hessian_u[c][a][j] +
              grad_delta_x[a][j] * hessian_u[c][i][a] +
              hessian_delta_x[a][i][j] * grad_u[c][a];

    return d_hessian_u;
  }

  inline void
  mesh_forcing_support_and_jacobian(const double phase_value,
                                    const double gamma,
                                    double       &support,
                                    double       &support_jacobian)
  {
    constexpr double phi_max_user = 0.998;
    const double     phi_max_safe =
      std::min(phi_max_user, 0.98 / std::max(gamma, 1e-14));

    const double z                 = phase_value / phi_max_safe;
    const double t                 = std::tanh(z);
    const double regularized_phase = phi_max_safe * t;
    const double regularized_jac   = 1. - t * t;
    const double denominator =
      1. - gamma * gamma * regularized_phase * regularized_phase;

    support = 1. / denominator;
    support_jacobian =
      2. * gamma * gamma * regularized_phase * regularized_jac /
      (denominator * denominator);
  }

  inline void
  mesh_forcing_factor_and_jacobian(const double phase_value,
                                   const double gamma,
                                   double       &factor,
                                   double       &factor_jacobian)
  {
    double support          = 0.;
    double support_jacobian = 0.;
    mesh_forcing_support_and_jacobian(phase_value,
                                      gamma,
                                      support,
                                      support_jacobian);

    factor          = phase_value * support;
    factor_jacobian = support + phase_value * support_jacobian;
  }

  inline MarkerForcingFactor
  mesh_forcing_factor(const double phase_value,
                      const double gamma)
  {
    MarkerForcingFactor factor;
    mesh_forcing_factor_and_jacobian(phase_value,
                                     gamma,
                                     factor.value,
                                     factor.derivative);
    return factor;
  }

  inline void
  smooth_power_equalized_phase_and_jacobian(const double phase_value,
                                            const double exponent,
                                            double       &equalized_phase,
                                            double       &equalized_jacobian)
  {
    if (std::abs(exponent - 1.) < 1e-14)
    {
      equalized_phase    = phase_value;
      equalized_jacobian = 1.;
      return;
    }

    constexpr double delta = 2e-2;
    const double     a     = phase_value * phase_value + delta * delta;

    equalized_phase = phase_value * std::pow(a, 0.5 * (exponent - 1.));
    equalized_jacobian =
      std::pow(a, 0.5 * (exponent - 3.)) *
      (delta * delta + exponent * phase_value * phase_value);
  }

  template <typename CahnHilliardParameter>
  inline void
  mesh_forcing_factor_and_jacobian(
    const CahnHilliardParameter &cahn_hilliard,
    const double                 phase_value,
    double                      &factor,
    double                      &factor_jacobian)
  {
    mesh_forcing_factor_and_jacobian(
      phase_value,
      cahn_hilliard.mff_regularization_gamma,
      factor,
      factor_jacobian);
  }

  template <typename CahnHilliardParameter>
  inline MarkerForcingFactor
  mesh_forcing_factor(const CahnHilliardParameter &cahn_hilliard,
                      const double                 phase_value)
  {
    return mesh_forcing_factor(phase_value,
                               cahn_hilliard.mff_regularization_gamma);
  }

  template <typename CahnHilliardParameter>
  inline void
  enlarged_mesh_forcing_factor_and_jacobian(
    const CahnHilliardParameter &cahn_hilliard,
    const double                 phase_value,
    double                      &factor,
    double                      &factor_jacobian)
  {
    double equalized_phase    = 0.;
    double equalized_jacobian = 0.;
    smooth_power_equalized_phase_and_jacobian(
      phase_value,
      cahn_hilliard.mff_enlarged_factor_equalization_exponent,
      equalized_phase,
      equalized_jacobian);

    mesh_forcing_factor_and_jacobian(
      cahn_hilliard, equalized_phase, factor, factor_jacobian);
    factor_jacobian *= equalized_jacobian;
  }

  template <typename CahnHilliardParameter>
  inline MarkerForcingFactor
  enlarged_mesh_forcing_factor(
    const CahnHilliardParameter &cahn_hilliard,
    const double                 phase_value)
  {
    MarkerForcingFactor factor;
    enlarged_mesh_forcing_factor_and_jacobian(cahn_hilliard,
                                              phase_value,
                                              factor.value,
                                              factor.derivative);
    return factor;
  }

  template <bool with_enlarged, typename CahnHilliardParameter>
  inline double
  phase_marker_epsilon(const CahnHilliardParameter &cahn_hilliard)
  {
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    if constexpr (with_enlarged)
      if (variable == ForceVariable::enlarged_tracer)
        return cahn_hilliard.epsilon_interface_enlarged;

    return cahn_hilliard.epsilon_interface;
  }

  template <bool with_enlarged, typename CahnHilliardParameter>
  inline MarkerForcingFactor
  phase_marker_forcing_factor(
    const CahnHilliardParameter &cahn_hilliard,
    const double                 marker_value)
  {
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    if constexpr (with_enlarged)
      if (variable == ForceVariable::enlarged_tracer)
        return enlarged_mesh_forcing_factor(cahn_hilliard, marker_value);

    return mesh_forcing_factor(cahn_hilliard, marker_value);
  }

  template <bool with_enlarged, typename CahnHilliardParameter>
  inline double
  phase_marker_compression_coefficient(
    const CahnHilliardParameter &cahn_hilliard)
  {
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    if constexpr (with_enlarged)
      if (variable == ForceVariable::enlarged_tracer)
        return cahn_hilliard.mff_enlarged_compression_factor;

    return cahn_hilliard.mff_physics_compression_factor;
  }

  template <bool with_enlarged,
            typename CahnHilliardParameter,
            typename ScratchData>
  inline const auto &
  phase_marker_value(const CahnHilliardParameter &cahn_hilliard,
                     const ScratchData           &scratch,
                     const unsigned int           q)
  {
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    if constexpr (with_enlarged)
      if (variable == ForceVariable::enlarged_tracer)
        return scratch.psi_values[q];

    return scratch.tracer_values[q];
  }

  template <int dim,
            bool with_enlarged,
            typename CahnHilliardParameter,
            typename ScratchData>
  inline const Tensor<1, dim> &
  phase_marker_gradient(const CahnHilliardParameter &cahn_hilliard,
                        const ScratchData           &scratch,
                        const unsigned int           q)
  {
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    if constexpr (with_enlarged)
      if (variable == ForceVariable::enlarged_tracer)
        return scratch.psi_gradients[q];

    return scratch.tracer_gradients[q];
  }

  template <int dim>
  inline Tensor<1, dim>
  compression_forcing(const double               coefficient,
                      const double               epsilon,
                      const MarkerForcingFactor &factor,
                      const Tensor<1, dim>      &marker_gradient)
  {
    return coefficient * epsilon * factor.value * marker_gradient;
  }

  template <int dim>
  inline Tensor<1, dim>
  transport_forcing(const double          coefficient,
                    const double          epsilon,
                    const Tensor<1, dim> &convective_velocity,
                    const Tensor<1, dim> &marker_gradient)
  {
    return coefficient * epsilon * epsilon *
           ((convective_velocity * marker_gradient) * marker_gradient);
  }

  template <int  dim,
            bool with_enlarged,
            typename CahnHilliardParameter,
            typename ScratchData,
            typename VectorType>
  inline void
  assemble_chns_rhs(const ComponentOrdering     &ordering,
                    const CahnHilliardParameter &cahn_hilliard,
                    const ScratchData           &scratch,
                    VectorType                  &local_rhs)
  {
    if (cahn_hilliard.mesh_concentration_method != Method::force)
      return;

    const double marker_epsilon =
      phase_marker_epsilon<with_enlarged>(cahn_hilliard);
    const double compression_coefficient =
      phase_marker_compression_coefficient<with_enlarged>(cahn_hilliard);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      const auto &marker_value =
        phase_marker_value<with_enlarged>(cahn_hilliard, scratch, q);
      const auto &marker_gradient =
        phase_marker_gradient<dim, with_enlarged>(cahn_hilliard, scratch, q);
      const auto &u_conv = scratch.present_convective_velocity[q];

      const MarkerForcingFactor marker_factor =
        phase_marker_forcing_factor<with_enlarged>(cahn_hilliard,
                                                   marker_value);

      Tensor<1, dim> mesh_forcing;
      mesh_forcing += compression_forcing(compression_coefficient,
                                          marker_epsilon,
                                          marker_factor,
                                          marker_gradient);
      mesh_forcing += transport_forcing(cahn_hilliard.mff_transport_factor,
                                        marker_epsilon,
                                        u_conv,
                                        marker_gradient);

      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        if (ordering.is_position(scratch.components[i]))
          local_rhs(i) +=
            scratch.phi_x[q][i] * mesh_forcing * scratch.JxW_fixed[q];
    }
  }

  template <int  dim,
            bool with_enlarged,
            typename CahnHilliardParameter,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_chns_matrix(const ComponentOrdering     &ordering,
                       const CouplingTableType     &coupling_table,
                       const CahnHilliardParameter &cahn_hilliard,
                       const double                 bdf_c0,
                       const ScratchData           &scratch,
                       MatrixType                  &local_matrix)
  {
    if (cahn_hilliard.mesh_concentration_method != Method::force)
      return;

    const double marker_epsilon =
      phase_marker_epsilon<with_enlarged>(cahn_hilliard);
    const double compression_coefficient =
      phase_marker_compression_coefficient<with_enlarged>(cahn_hilliard);
    const auto variable =
      active_force_variable<with_enlarged>(
        cahn_hilliard.mesh_concentration_force_variable);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      const auto &marker_value =
        phase_marker_value<with_enlarged>(cahn_hilliard, scratch, q);
      const auto &marker_gradient =
        phase_marker_gradient<dim, with_enlarged>(cahn_hilliard, scratch, q);
      const auto &u_conv = scratch.present_convective_velocity[q];
      double      marker_u_dot_gradient = 0.;
      if constexpr (with_enlarged)
        marker_u_dot_gradient = u_conv * marker_gradient;
      else
        marker_u_dot_gradient = scratch.velocity_dot_tracer_gradient[q];

      const MarkerForcingFactor marker_factor =
        phase_marker_forcing_factor<with_enlarged>(cahn_hilliard,
                                                   marker_value);

      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        if (!ordering.is_position(scratch.components[i]))
          continue;

        const auto &phi_x_i = scratch.phi_x[q][i];

        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          if (coupling_table[scratch.components[i]][scratch.components[j]] !=
              DoFTools::always)
            continue;

          const unsigned int comp_j   = scratch.components[j];
          double             local_ij = 0.;

          if (ordering.is_velocity(comp_j))
            local_ij -=
              phi_x_i *
              (cahn_hilliard.mff_transport_factor *
               (marker_epsilon * marker_epsilon) *
               ((scratch.phi_u[q][j] * marker_gradient) * marker_gradient));

          if (ordering.is_position(comp_j))
          {
            const auto &G = scratch.grad_phi_x_moving[q][j];
            const Tensor<1, dim> mesh_velocity_delta =
              Assembly::ALE::mesh_velocity_variation(bdf_c0,
                                                     scratch.phi_x[q][j]);
            const Tensor<1, dim> transported_marker_gradient =
              Assembly::ALE::gradient_variation(marker_gradient, G);

            local_ij -=
              phi_x_i *
              (cahn_hilliard.mff_transport_factor *
                 (marker_epsilon * marker_epsilon) *
                 (mesh_velocity_delta * marker_gradient * marker_gradient +
                  (u_conv * transported_marker_gradient) * marker_gradient +
                  marker_u_dot_gradient * transported_marker_gradient));

            local_ij -=
              phi_x_i *
              (compression_coefficient * marker_epsilon *
               marker_factor.value * transported_marker_gradient);
          }

          if constexpr (with_enlarged)
            if (variable == ForceVariable::enlarged_tracer &&
                ordering.is_psi(comp_j))
              local_ij -=
                phi_x_i *
                (compression_coefficient *
                   marker_epsilon *
                   (marker_factor.derivative * scratch.shape_psi[q][j] *
                      marker_gradient +
                    marker_factor.value * scratch.grad_shape_psi[q][j]) +
                 cahn_hilliard.mff_transport_factor *
                   (marker_epsilon * marker_epsilon) *
                   ((u_conv * scratch.grad_shape_psi[q][j]) *
                      marker_gradient +
                    marker_u_dot_gradient * scratch.grad_shape_psi[q][j]));

          if (variable == ForceVariable::tracer && ordering.is_tracer(comp_j))
            local_ij -=
              phi_x_i *
              (cahn_hilliard.mff_transport_factor *
                 (marker_epsilon * marker_epsilon) *
                 ((u_conv * scratch.grad_shape_phi[q][j]) * marker_gradient +
                  marker_u_dot_gradient * scratch.grad_shape_phi[q][j]));

          if (variable == ForceVariable::tracer && ordering.is_tracer(comp_j))
            local_ij -=
              phi_x_i *
              (compression_coefficient * marker_epsilon *
               (marker_factor.derivative * scratch.shape_phi[q][j] *
                  marker_gradient +
                marker_factor.value * scratch.grad_shape_phi[q][j]));

          local_matrix(i, j) += local_ij * scratch.JxW_fixed[q];
        }
      }
    }
  }

  inline double
  positive_part_smooth(const double value)
  {
    constexpr double eps = 1e-10;
    return 0.5 * (value + std::sqrt(value * value + eps * eps));
  }

  inline double
  positive_part_smooth_derivative(const double value)
  {
    constexpr double eps = 1e-10;
    return 0.5 * (1.0 + value / std::sqrt(value * value + eps * eps));
  }

  inline double
  raw_size_pressure(const double coefficient,
                    const double current_size_weight,
                    const double h_background,
                    const double h_current,
                    const double h_target,
                    const double h_min)
  {
    const double eps_h     = 1e-14;
    const double h_safe     = std::max(h_target, h_min + eps_h);
    const double h_bg_safe  = std::max(h_background, eps_h);
    const double h_cur_safe = std::max(h_current, eps_h);

    return coefficient *
           (std::log(h_bg_safe / h_safe) +
            current_size_weight * std::log(h_cur_safe / h_safe));
  }

  inline double
  size_pressure(const double coefficient,
                const double current_size_weight,
                const double h_background,
                const double h_current,
                const double h_target,
                const double h_min)
  {
    return positive_part_smooth(raw_size_pressure(coefficient,
                                                  current_size_weight,
                                                  h_background,
                                                  h_current,
                                                  h_target,
                                                  h_min));
  }

  inline double
  size_pressure_derivative_factor(const double coefficient,
                                  const double current_size_weight,
                                  const double h_background,
                                  const double h_current,
                                  const double h_target,
                                  const double h_min)
  {
    return positive_part_smooth_derivative(
      raw_size_pressure(coefficient,
                        current_size_weight,
                        h_background,
                        h_current,
                        h_target,
                        h_min));
  }
} // namespace MeshConcentrationTools

#endif
