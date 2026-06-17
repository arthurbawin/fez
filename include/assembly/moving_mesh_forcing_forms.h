#ifndef ASSEMBLY_MOVING_MESH_FORCING_FORMS_H
#define ASSEMBLY_MOVING_MESH_FORCING_FORMS_H

#include <assembly/ale_geometry.h>
#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>
#include <parameters.h>

#include <algorithm>
#include <cmath>

using namespace dealii;

namespace Assembly::MovingMeshForcing
{
  // CHNS-ALE moving-mesh forcing. The phase marker is psi in the enlarged
  // solver and phi otherwise; the physical compression term always uses phi.
  struct MarkerForcingFactor
  {
    double value      = 0.;
    double derivative = 0.;
  };

  // Regularizing support for f(m) = m / (1 - gamma^2 center(m)^2).
  // The numerator remains the raw marker; center only keeps the denominator
  // bounded when gamma is close to one.
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

    support         = 1. / denominator;
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
  mesh_forcing_factor(const double phase_value, const double gamma)
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

    // Smooth odd approximation of sign(psi) * |psi|^q. The small delta keeps
    // the map differentiable at psi=0 for Newton.
    constexpr double delta = 2e-2;
    const double     a     = phase_value * phase_value + delta * delta;

    equalized_phase = phase_value * std::pow(a, 0.5 * (exponent - 1.));
    equalized_jacobian =
      std::pow(a, 0.5 * (exponent - 3.)) *
      (delta * delta + exponent * phase_value * phase_value);
  }

  template <int dim>
  inline void mesh_forcing_factor_and_jacobian(
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const double                         phase_value,
    double                              &factor,
    double                              &factor_jacobian)
  {
    mesh_forcing_factor_and_jacobian(
      phase_value,
      cahn_hilliard.mff_regularization_gamma,
      factor,
      factor_jacobian);
  }

  template <int dim>
  inline MarkerForcingFactor
  mesh_forcing_factor(const Parameters::CahnHilliard<dim> &cahn_hilliard,
                      const double                         phase_value)
  {
    return mesh_forcing_factor(phase_value,
                               cahn_hilliard.mff_regularization_gamma);
  }

  template <int dim>
  inline void
  enlarged_mesh_forcing_factor_and_jacobian(
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const double                         phase_value,
    double                              &factor,
    double                              &factor_jacobian)
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

  template <int dim>
  inline MarkerForcingFactor
  enlarged_mesh_forcing_factor(
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const double                         phase_value)
  {
    MarkerForcingFactor factor;
    enlarged_mesh_forcing_factor_and_jacobian(cahn_hilliard,
                                              phase_value,
                                              factor.value,
                                              factor.derivative);
    return factor;
  }

  template <bool with_enlarged, typename ScratchData>
  inline const auto &
  phase_marker_value(const ScratchData &scratch, const unsigned int q)
  {
    if constexpr (with_enlarged)
      return scratch.psi_values[q];
    else
      return scratch.tracer_values[q];
  }

  template <int dim, bool with_enlarged, typename ScratchData>
  inline const Tensor<1, dim> &
  phase_marker_gradient(const ScratchData &scratch, const unsigned int q)
  {
    if constexpr (with_enlarged)
      return scratch.psi_gradients[q];
    else
      return scratch.tracer_gradients[q];
  }

  template <int dim, bool with_enlarged>
  inline double
  phase_marker_epsilon(
    const Parameters::CahnHilliard<dim> &cahn_hilliard)
  {
    if constexpr (with_enlarged)
      return cahn_hilliard.epsilon_interface_enlarged;
    else
      return cahn_hilliard.epsilon_interface;
  }

  template <int dim, bool with_enlarged>
  inline MarkerForcingFactor
  phase_marker_forcing_factor(
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const double                         marker_value)
  {
    if constexpr (with_enlarged)
      return enlarged_mesh_forcing_factor(cahn_hilliard, marker_value);
    else
      return mesh_forcing_factor(cahn_hilliard, marker_value);
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
            typename ScratchData,
            typename VectorType>
  inline void
  assemble_chns_rhs(const ComponentOrdering             &ordering,
                    const Parameters::CahnHilliard<dim> &cahn_hilliard,
                    const ScratchData                   &scratch,
                    VectorType                          &local_rhs)
  {
    const double marker_epsilon =
      phase_marker_epsilon<dim, with_enlarged>(cahn_hilliard);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const auto &marker_value =
          phase_marker_value<with_enlarged>(scratch, q);
        const auto &marker_gradient =
          phase_marker_gradient<dim, with_enlarged>(scratch, q);
        const auto &u_conv = scratch.present_convective_velocity[q];

        const MarkerForcingFactor marker_factor =
          phase_marker_forcing_factor<dim, with_enlarged>(cahn_hilliard,
                                                          marker_value);
        const MarkerForcingFactor tracer_factor =
          mesh_forcing_factor(cahn_hilliard, scratch.tracer_values[q]);

        Tensor<1, dim> mesh_forcing;
        if constexpr (with_enlarged)
          mesh_forcing += compression_forcing(
            cahn_hilliard.mff_enlarged_compression_factor,
            marker_epsilon,
            marker_factor,
            marker_gradient);
        mesh_forcing += transport_forcing(cahn_hilliard.mff_transport_factor,
                                          marker_epsilon,
                                          u_conv,
                                          marker_gradient);
        mesh_forcing += compression_forcing(
          cahn_hilliard.mff_physics_compression_factor,
          cahn_hilliard.epsilon_interface,
          tracer_factor,
          scratch.tracer_gradients[q]);

      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        if (ordering.is_position(scratch.components[i]))
          local_rhs(i) +=
            scratch.phi_x[q][i] * mesh_forcing * scratch.JxW_fixed[q];
    }
  }

  template <int  dim,
            bool with_enlarged,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_chns_matrix(const ComponentOrdering             &ordering,
                       const CouplingTableType             &coupling_table,
                       const Parameters::CahnHilliard<dim> &cahn_hilliard,
                       const double                         bdf_c0,
                       const ScratchData                   &scratch,
                       MatrixType                          &local_matrix)
  {
    const double marker_epsilon =
      phase_marker_epsilon<dim, with_enlarged>(cahn_hilliard);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const auto &marker_value =
          phase_marker_value<with_enlarged>(scratch, q);
        const auto &marker_gradient =
          phase_marker_gradient<dim, with_enlarged>(scratch, q);
        const auto &u_conv = scratch.present_convective_velocity[q];
        double      marker_u_dot_gradient = 0.;
        if constexpr (with_enlarged)
          marker_u_dot_gradient = u_conv * marker_gradient;
        else
          marker_u_dot_gradient = scratch.velocity_dot_tracer_gradient[q];

        const MarkerForcingFactor marker_factor =
          phase_marker_forcing_factor<dim, with_enlarged>(cahn_hilliard,
                                                          marker_value);
        const MarkerForcingFactor tracer_factor =
          mesh_forcing_factor(cahn_hilliard, scratch.tracer_values[q]);

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
                     ((scratch.phi_u[q][j] * marker_gradient) *
                      marker_gradient));

                if (ordering.is_position(comp_j))
                  {
                    const auto &G = scratch.grad_phi_x_moving[q][j];
                    const Tensor<1, dim> mesh_velocity_delta =
                      Assembly::ALE::mesh_velocity_variation(bdf_c0,
                                                             scratch.phi_x[q][j]);
                    const Tensor<1, dim> transported_marker_gradient =
                      Assembly::ALE::gradient_variation(marker_gradient, G);
                    const Tensor<1, dim> transported_tracer_gradient =
                      Assembly::ALE::gradient_variation(scratch.tracer_gradients[q],
                                                        G);

                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_transport_factor *
                         (marker_epsilon * marker_epsilon) *
                         (mesh_velocity_delta * marker_gradient *
                            marker_gradient +
                          (u_conv * transported_marker_gradient) *
                            marker_gradient +
                          marker_u_dot_gradient *
                            transported_marker_gradient));

                    if constexpr (with_enlarged)
                      local_ij -=
                        phi_x_i *
                        (cahn_hilliard.mff_enlarged_compression_factor *
                         marker_epsilon * marker_factor.value *
                         transported_marker_gradient);

                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_physics_compression_factor *
                       cahn_hilliard.epsilon_interface *
                       tracer_factor.value *
                       transported_tracer_gradient);
                  }

                if constexpr (with_enlarged)
                  if (ordering.is_psi(comp_j))
                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_enlarged_compression_factor *
                         marker_epsilon *
                         (marker_factor.derivative *
                            scratch.shape_psi[q][j] *
                            marker_gradient +
                          marker_factor.value *
                            scratch.grad_shape_psi[q][j]) +
                       cahn_hilliard.mff_transport_factor *
                         (marker_epsilon * marker_epsilon) *
                         ((u_conv * scratch.grad_shape_psi[q][j]) *
                            marker_gradient +
                          marker_u_dot_gradient *
                            scratch.grad_shape_psi[q][j]));

                if (!with_enlarged && ordering.is_tracer(comp_j))
                  local_ij -=
                    phi_x_i *
                    (cahn_hilliard.mff_transport_factor *
                       (marker_epsilon * marker_epsilon) *
                       ((u_conv * scratch.grad_shape_phi[q][j]) *
                          marker_gradient +
                        marker_u_dot_gradient * scratch.grad_shape_phi[q][j]));

                if (ordering.is_tracer(comp_j))
                  local_ij -=
                    phi_x_i *
                    (cahn_hilliard.mff_physics_compression_factor *
                     cahn_hilliard.epsilon_interface *
                     (tracer_factor.derivative * scratch.shape_phi[q][j] *
                        scratch.tracer_gradients[q] +
                      tracer_factor.value * scratch.grad_shape_phi[q][j]));

                local_matrix(i, j) += local_ij * scratch.JxW_fixed[q];
              }
          }
      }
  }
} // namespace Assembly::MovingMeshForcing

#endif
