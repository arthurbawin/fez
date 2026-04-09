#ifndef ASSEMBLY_MOVING_MESH_FORCING_FORMS_H
#define ASSEMBLY_MOVING_MESH_FORCING_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>

#include <algorithm>
#include <cmath>

#include <parameters.h>

using namespace dealii;

namespace Assembly::MovingMeshForcing
{
  inline void
  simple_mesh_forcing_factor_and_jacobian(const double phase_value,
                                          double       &factor,
                                          double       &factor_jacobian)
  {
    factor          = phase_value;
    factor_jacobian = 1.;
  }

  inline void
  regularized_band_mesh_forcing_factor_and_jacobian(const double phase_value,
                                                    const double mff_band_factor,
                                                    double       &factor,
                                                    double       &factor_jacobian)
  {
    constexpr double phi_max_user = 0.998;
    const double     phi_max_safe =
      std::min(phi_max_user, 0.98 / std::max(mff_band_factor, 1e-14));

    const double z                 = phase_value / phi_max_safe;
    const double t                 = std::tanh(z);
    const double regularized_phase = phi_max_safe * t;
    const double regularized_jac   = 1. - t * t;
    const double denominator =
      1. - mff_band_factor * mff_band_factor * regularized_phase * regularized_phase;

    factor = regularized_phase / denominator;
    factor_jacobian =
      regularized_jac *
      (1. + mff_band_factor * mff_band_factor * regularized_phase *
               regularized_phase) /
      (denominator * denominator);
  }

  template <int dim, bool with_enlarged, typename ScratchData, typename VectorType>
  inline void
  assemble_chns_rhs(const ComponentOrdering             &ordering,
                    const Parameters::CahnHilliard<dim> &cahn_hilliard,
                    const ScratchData                   &scratch,
                    VectorType                          &local_rhs)
  {
    const double enlarged_epsilon =
      with_enlarged ? cahn_hilliard.epsilon_interface_enlarged :
                      cahn_hilliard.epsilon_interface;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const auto &enlarged_phase_value =
          with_enlarged ? scratch.psi_values[q] : scratch.tracer_values[q];
        const auto &enlarged_phase_gradient =
          with_enlarged ? scratch.psi_gradients[q] : scratch.tracer_gradients[q];
        const auto &u_conv = scratch.present_convective_velocity[q];

        double enlarged_factor = 0.;
        double enlarged_factor_jacobian = 0.;
        if (cahn_hilliard.mesh_forcing_law ==
            Parameters::CahnHilliard<dim>::MeshForcingLaw::simple)
          simple_mesh_forcing_factor_and_jacobian(enlarged_phase_value,
                                                  enlarged_factor,
                                                  enlarged_factor_jacobian);
        else
          regularized_band_mesh_forcing_factor_and_jacobian(
            enlarged_phase_value,
            cahn_hilliard.mff_band_factor,
            enlarged_factor,
            enlarged_factor_jacobian);

        (void)enlarged_factor_jacobian;

        Tensor<1, dim> mesh_forcing;
        mesh_forcing += cahn_hilliard.mff_enlarged_compression_factor *
                        enlarged_epsilon * enlarged_factor *
                        enlarged_phase_gradient;
        mesh_forcing +=
          cahn_hilliard.mff_transport_factor *
          (enlarged_epsilon * enlarged_epsilon) *
          ((u_conv * enlarged_phase_gradient) * enlarged_phase_gradient);
        mesh_forcing += cahn_hilliard.mff_physics_compression_factor *
                        cahn_hilliard.epsilon_interface *
                        scratch.tracer_values[q] * scratch.tracer_gradients[q];

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          if (ordering.is_position(scratch.components[i]))
            local_rhs(i) +=
              scratch.phi_x[q][i] * mesh_forcing * scratch.JxW_fixed[q];
      }
  }

  template <int dim,
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
    const double enlarged_epsilon =
      with_enlarged ? cahn_hilliard.epsilon_interface_enlarged :
                      cahn_hilliard.epsilon_interface;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const auto &enlarged_phase_value =
          with_enlarged ? scratch.psi_values[q] : scratch.tracer_values[q];
        const auto &enlarged_phase_gradient =
          with_enlarged ? scratch.psi_gradients[q] : scratch.tracer_gradients[q];
        const auto &u_conv = scratch.present_convective_velocity[q];
        const double enlarged_u_dot_grad_phase =
          with_enlarged ? u_conv * scratch.psi_gradients[q] :
                          scratch.velocity_dot_tracer_gradient[q];

        double enlarged_factor          = 0.;
        double enlarged_factor_jacobian = 0.;
        if (cahn_hilliard.mesh_forcing_law ==
            Parameters::CahnHilliard<dim>::MeshForcingLaw::simple)
          simple_mesh_forcing_factor_and_jacobian(enlarged_phase_value,
                                                  enlarged_factor,
                                                  enlarged_factor_jacobian);
        else
          regularized_band_mesh_forcing_factor_and_jacobian(
            enlarged_phase_value,
            cahn_hilliard.mff_band_factor,
            enlarged_factor,
            enlarged_factor_jacobian);

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

                const unsigned int comp_j = scratch.components[j];
                double             local_ij = 0.;

                if (ordering.is_velocity(comp_j))
                  local_ij -=
                    phi_x_i *
                    (cahn_hilliard.mff_transport_factor *
                     (enlarged_epsilon * enlarged_epsilon) *
                     ((scratch.phi_u[q][j] * enlarged_phase_gradient) *
                      enlarged_phase_gradient));

                if (ordering.is_position(comp_j))
                  {
                    const auto  &G   = scratch.grad_phi_x_moving[q][j];
                    const Tensor<1, dim> transported_enlarged_gradient =
                      -transpose(G) * enlarged_phase_gradient;
                    const Tensor<1, dim> transported_tracer_gradient =
                      -transpose(G) * scratch.tracer_gradients[q];

                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_enlarged_compression_factor *
                         enlarged_epsilon * enlarged_factor *
                         transported_enlarged_gradient +
                       cahn_hilliard.mff_transport_factor *
                         (enlarged_epsilon * enlarged_epsilon) *
                         ((-bdf_c0) * scratch.phi_x[q][j] *
                            enlarged_phase_gradient * enlarged_phase_gradient +
                          (u_conv * transported_enlarged_gradient) *
                            enlarged_phase_gradient +
                          enlarged_u_dot_grad_phase *
                            transported_enlarged_gradient));

                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_physics_compression_factor *
                       cahn_hilliard.epsilon_interface *
                       transported_tracer_gradient);
                  }

                if constexpr (with_enlarged)
                  if (ordering.is_psi(comp_j))
                    local_ij -=
                      phi_x_i *
                      (cahn_hilliard.mff_enlarged_compression_factor *
                         enlarged_epsilon *
                         (enlarged_factor_jacobian * scratch.shape_psi[q][j] *
                            enlarged_phase_gradient +
                          enlarged_factor * scratch.grad_shape_psi[q][j]) +
                       cahn_hilliard.mff_transport_factor *
                         (enlarged_epsilon * enlarged_epsilon) *
                         ((u_conv * scratch.grad_shape_psi[q][j]) *
                            enlarged_phase_gradient +
                          enlarged_u_dot_grad_phase *
                            scratch.grad_shape_psi[q][j]));

                if (!with_enlarged && ordering.is_tracer(comp_j))
                  local_ij -=
                    phi_x_i *
                    (cahn_hilliard.mff_enlarged_compression_factor *
                       enlarged_epsilon *
                       (enlarged_factor_jacobian * scratch.shape_phi[q][j] *
                          enlarged_phase_gradient +
                        enlarged_factor * scratch.grad_shape_phi[q][j]) +
                     cahn_hilliard.mff_transport_factor *
                       (enlarged_epsilon * enlarged_epsilon) *
                       ((u_conv * scratch.grad_shape_phi[q][j]) *
                          enlarged_phase_gradient +
                        enlarged_u_dot_grad_phase * scratch.grad_shape_phi[q][j]));

                if (ordering.is_tracer(comp_j))
                  local_ij -=
                    phi_x_i *
                    (cahn_hilliard.mff_physics_compression_factor *
                     cahn_hilliard.epsilon_interface *
                     scratch.grad_shape_phi[q][j]);

                local_matrix(i, j) += local_ij * scratch.JxW_fixed[q];
              }
          }
      }
  }
} // namespace Assembly::MovingMeshForcing

#endif
