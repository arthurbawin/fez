#ifndef ASSEMBLY_STABILIZATION_FORMS_H
#define ASSEMBLY_STABILIZATION_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>

using namespace dealii;

namespace Assembly
{
  template <int dim,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_ns_matrix_stabilization(const ComponentOrdering &ordering,
                                   const CouplingTableType &coupling_table,
                                   const ScratchData       &scratch,
                                   const double             nu,
                                   const double             bdf_c0,
                                   const bool               stabilization_enabled,
                                   MatrixType              &local_matrix)
  {
    if (!stabilization_enabled)
      return;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const double tau = scratch.stabilization_tau_momentum[q];
        if (tau <= 0.)
          continue;

        const auto &u_conv = scratch.present_velocity_values[q];
        const auto &R      = scratch.strong_residual_momentum[q];
        const auto &grad_u = scratch.present_velocity_gradients[q];

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          {
            const unsigned int comp_i = scratch.components[i];
            const bool         i_is_u = ordering.is_velocity(comp_i);
            const bool         i_is_p = ordering.is_pressure(comp_i);
            if (!i_is_u && !i_is_p)
              continue;

            for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
              {
                const unsigned int comp_j = scratch.components[j];
                double             local_matrix_ij = 0.;

                if (ordering.is_velocity(comp_j))
                  {
                    const Tensor<1, dim> dR_du_j =
                      bdf_c0 * scratch.phi_u[q][j] +
                      scratch.grad_phi_u[q][j] * u_conv +
                      grad_u * scratch.phi_u[q][j] -
                      nu * scratch.diffusion_phi_u[q][j];

                    if (i_is_p)
                      local_matrix_ij -=
                        tau * dR_du_j * scratch.grad_phi_p[q][i];

                    if (i_is_u)
                      {
                        const Tensor<1, dim> supg_test =
                          scratch.grad_phi_u[q][i] * u_conv;
                        local_matrix_ij +=
                          tau * (scratch.grad_phi_u[q][i] * scratch.phi_u[q][j]) *
                          R;
                        local_matrix_ij += tau * supg_test * dR_du_j;
                      }
                  }

                if (ordering.is_pressure(comp_j))
                  {
                    if (i_is_p)
                      local_matrix_ij -=
                        tau * scratch.grad_phi_p[q][j] * scratch.grad_phi_p[q][i];

                    if (i_is_u)
                      {
                        const Tensor<1, dim> supg_test =
                          scratch.grad_phi_u[q][i] * u_conv;
                        local_matrix_ij +=
                          tau * supg_test * scratch.grad_phi_p[q][j];
                      }
                  }

                local_matrix(i, j) += local_matrix_ij * scratch.JxW_moving[q];
              }
          }
      }
  }

  template <int dim, typename ScratchData, typename VectorType>
  inline void
  assemble_ns_rhs_stabilization(const ComponentOrdering &ordering,
                                const ScratchData       &scratch,
                                const bool               stabilization_enabled,
                                VectorType              &local_rhs)
  {
    if (!stabilization_enabled)
      return;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const double tau = scratch.stabilization_tau_momentum[q];
        if (tau <= 0.)
          continue;

        const auto &u_conv = scratch.present_velocity_values[q];

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          {
            const unsigned int comp_i = scratch.components[i];
            double             local_rhs_i = 0.;

            if (ordering.is_pressure(comp_i))
              local_rhs_i += tau * scratch.strong_residual_momentum[q] *
                             scratch.grad_phi_p[q][i];

            if (ordering.is_velocity(comp_i))
              local_rhs_i -=
                tau * ((scratch.grad_phi_u[q][i] * u_conv) *
                       scratch.strong_residual_momentum[q]);

            local_rhs(i) += local_rhs_i * scratch.JxW_moving[q];
          }
      }
  }

  template <int dim, typename ScratchData, typename VectorType>
  inline void
  assemble_chns_rhs_stabilization(const ComponentOrdering &ordering,
                                  const ScratchData       &scratch,
                                  const bool               stabilization_enabled,
                                  VectorType              &local_rhs)
  {
    if (!stabilization_enabled)
      return;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const double tau_mom    = scratch.stabilization_tau_momentum[q];
        const double tau_tracer = scratch.stabilization_tau_tracer[q];
        if (tau_mom <= 0. && tau_tracer <= 0.)
          continue;

        const auto &u_conv = scratch.present_convective_velocity[q];

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          {
            const unsigned int comp_i = scratch.components[i];
            double             local_rhs_i = 0.;

            if (tau_mom > 0.)
              {
                if (ordering.is_pressure(comp_i))
                  local_rhs_i += tau_mom * scratch.strong_residual_momentum[q] *
                                 scratch.grad_phi_p[q][i];

                if (ordering.is_velocity(comp_i))
                  local_rhs_i -=
                    tau_mom *
                    ((scratch.grad_phi_u[q][i] * u_conv) *
                     scratch.strong_residual_momentum[q]);
              }

            if (tau_tracer > 0. && ordering.is_tracer(comp_i))
              {
                const double supg_test = u_conv * scratch.grad_shape_phi[q][i];
                local_rhs_i -=
                  tau_tracer * supg_test * scratch.strong_residual_tracer[q];
              }

            local_rhs(i) += local_rhs_i * scratch.JxW_moving[q];
          }
      }
  }

  template <int dim,
            bool with_moving_mesh,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_chns_matrix_stabilization(const ComponentOrdering &ordering,
                                     const CouplingTableType &coupling_table,
                                     const ScratchData       &scratch,
                                     const double             bdf_c0,
                                     const bool               stabilization_enabled,
                                     MatrixType              &local_matrix)
  {
    if (!stabilization_enabled)
      return;

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const double tau_mom    = scratch.stabilization_tau_momentum[q];
        const double tau_tracer = scratch.stabilization_tau_tracer[q];
        if (tau_mom <= 0. && tau_tracer <= 0.)
          continue;

        const auto &u_conv = scratch.present_convective_velocity[q];

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          {
            const unsigned int comp_i      = scratch.components[i];
            const bool         i_is_u      = ordering.is_velocity(comp_i);
            const bool         i_is_p      = ordering.is_pressure(comp_i);
            const bool         i_is_tracer = ordering.is_tracer(comp_i);

            const Tensor<1, dim> grad_phi_p_i =
              i_is_p ? scratch.grad_phi_p[q][i] : Tensor<1, dim>();
            const Tensor<2, dim> grad_phi_u_i =
              i_is_u ? scratch.grad_phi_u[q][i] : Tensor<2, dim>();
            const Tensor<1, dim> supg_test_momentum =
              i_is_u ? grad_phi_u_i * u_conv : Tensor<1, dim>();
            const Tensor<1, dim> grad_shape_phi_i =
              i_is_tracer ? scratch.grad_shape_phi[q][i] : Tensor<1, dim>();
            const double supg_test_tracer =
              i_is_tracer ? u_conv * grad_shape_phi_i : 0.;

            for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
              {
                if (coupling_table[comp_i][scratch.components[j]] !=
                    DoFTools::always)
                  continue;

                const unsigned int comp_j = scratch.components[j];
                double             local_matrix_ij = 0.;

                if (tau_mom > 0. && (i_is_u || i_is_p))
                  {
                    if (ordering.is_velocity(comp_j))
                      {
                        const auto &phi_u_j      = scratch.phi_u[q][j];
                        const auto &grad_phi_u_j = scratch.grad_phi_u[q][j];

                        const Tensor<1, dim> dR_du_j =
                          bdf_c0 * phi_u_j + grad_phi_u_j * u_conv +
                          scratch.present_velocity_gradients[q] * phi_u_j -
                          scratch.stabilization_nu_eff[q] *
                            scratch.diffusion_phi_u[q][j];

                        if (i_is_p)
                          local_matrix_ij -= tau_mom * dR_du_j * grad_phi_p_i;

                        if (i_is_u)
                          {
                            local_matrix_ij +=
                              tau_mom * (grad_phi_u_i * phi_u_j) *
                              scratch.strong_residual_momentum[q];
                            local_matrix_ij +=
                              tau_mom * supg_test_momentum * dR_du_j;
                          }

                        const Tensor<1, dim> extra_dR =
                          scratch.stabilization_inv_rho[q] *
                            scratch.diffusive_flux_factor *
                            (grad_phi_u_j * scratch.potential_gradients[q]) -
                          2. * scratch.stabilization_inv_rho[q] *
                            scratch.derivative_dynamic_viscosity_wrt_tracer[q] *
                            Tensor<2, dim>(symmetrize(grad_phi_u_j)) *
                            scratch.tracer_gradients[q];

                        if (i_is_p)
                          local_matrix_ij -= tau_mom * extra_dR * grad_phi_p_i;
                        if (i_is_u)
                          local_matrix_ij +=
                            tau_mom * supg_test_momentum * extra_dR;
                      }

                    if (ordering.is_pressure(comp_j))
                      {
                        const Tensor<1, dim> scaled_grad_p_j =
                          scratch.stabilization_inv_rho[q] *
                          scratch.grad_phi_p[q][j];

                        if (i_is_p)
                          local_matrix_ij -=
                            tau_mom * scaled_grad_p_j * grad_phi_p_i;
                        if (i_is_u)
                          local_matrix_ij +=
                            tau_mom * supg_test_momentum * scaled_grad_p_j;
                      }

                    if (ordering.is_tracer(comp_j))
                      {
                        const double dinvrho_dphi =
                          -scratch.stabilization_inv_rho[q] *
                          scratch.stabilization_inv_rho[q] *
                          scratch.derivative_density_wrt_tracer[q];
                        const double dnueff_dphi =
                          scratch.stabilization_inv_rho[q] *
                            scratch.derivative_dynamic_viscosity_wrt_tracer[q] -
                          scratch.stabilization_nu_eff[q] *
                            scratch.stabilization_inv_rho[q] *
                            scratch.derivative_density_wrt_tracer[q];
                        const Tensor<2, dim> eps_u =
                          Tensor<2, dim>(scratch.present_velocity_sym_gradients[q]);

                        const Tensor<1, dim> dR_dphi_j =
                          scratch.shape_phi[q][j] *
                            (dinvrho_dphi *
                               (scratch.diffusive_flux_factor *
                                  (scratch.present_velocity_gradients[q] *
                                   scratch.potential_gradients[q]) +
                                scratch.tracer_values[q] *
                                  scratch.potential_gradients[q] +
                                scratch.present_pressure_gradients[q] +
                                scratch.source_term_velocity[q]) +
                             scratch.stabilization_inv_rho[q] *
                               scratch.potential_gradients[q] -
                             dnueff_dphi *
                               scratch.present_velocity_lap_plus_graddiv[q] -
                             2. * dinvrho_dphi *
                               scratch.derivative_dynamic_viscosity_wrt_tracer[q] *
                               eps_u * scratch.tracer_gradients[q]) -
                          2. * scratch.stabilization_inv_rho[q] *
                            scratch.derivative_dynamic_viscosity_wrt_tracer[q] *
                            eps_u * scratch.grad_shape_phi[q][j];

                        if (i_is_p)
                          local_matrix_ij -= tau_mom * dR_dphi_j * grad_phi_p_i;
                        if (i_is_u)
                          local_matrix_ij +=
                            tau_mom * supg_test_momentum * dR_dphi_j;
                      }

                    if (ordering.is_potential(comp_j))
                      {
                        const Tensor<1, dim> dR_dmu_j =
                          scratch.stabilization_inv_rho[q] *
                          (scratch.diffusive_flux_factor *
                             (scratch.present_velocity_gradients[q] *
                              scratch.grad_shape_mu[q][j]) +
                           scratch.tracer_values[q] * scratch.grad_shape_mu[q][j]);

                        if (i_is_p)
                          local_matrix_ij -= tau_mom * dR_dmu_j * grad_phi_p_i;
                        if (i_is_u)
                          local_matrix_ij +=
                            tau_mom * supg_test_momentum * dR_dmu_j;
                      }

                    if constexpr (with_moving_mesh)
                      if (ordering.is_position(comp_j))
                        {
                          const Tensor<2, dim> &G   =
                            scratch.grad_phi_x_moving[q][j];
                          const Tensor<2, dim>  GT  = transpose(G);
                          const double          trG = trace(G);

                          const Tensor<1, dim> du_conv_dx_j =
                            -bdf_c0 * scratch.phi_x[q][j];
                          const Tensor<2, dim> dgrad_u_dx_j =
                            -scratch.present_velocity_gradients[q] * G;
                          const Tensor<1, dim> dgrad_p_dx_j =
                            -GT * scratch.present_pressure_gradients[q];
                          const Tensor<1, dim> dgrad_mu_dx_j =
                            -GT * scratch.potential_gradients[q];

                          const Tensor<1, dim> dR_dx_j =
                            dgrad_u_dx_j * u_conv +
                            scratch.present_velocity_gradients[q] * du_conv_dx_j +
                            scratch.stabilization_inv_rho[q] * dgrad_p_dx_j +
                            scratch.stabilization_inv_rho[q] *
                              scratch.tracer_values[q] * dgrad_mu_dx_j;

                          if (i_is_p)
                            {
                              const Tensor<1, dim> dgrad_phi_p_i_dx_j =
                                -GT * grad_phi_p_i;
                              local_matrix_ij -= tau_mom * dR_dx_j * grad_phi_p_i;
                              local_matrix_ij -=
                                tau_mom * scratch.strong_residual_momentum[q] *
                                dgrad_phi_p_i_dx_j;
                              local_matrix_ij -=
                                tau_mom * scratch.strong_residual_momentum[q] *
                                grad_phi_p_i * trG;
                            }

                          if (i_is_u)
                            {
                              const Tensor<2, dim> dgrad_phi_u_i_dx_j =
                                -grad_phi_u_i * G;
                              const Tensor<1, dim> d_supg_test_dx_j =
                                dgrad_phi_u_i_dx_j * u_conv +
                                grad_phi_u_i * du_conv_dx_j;

                              local_matrix_ij +=
                                tau_mom * supg_test_momentum * dR_dx_j;
                              local_matrix_ij +=
                                tau_mom * d_supg_test_dx_j *
                                scratch.strong_residual_momentum[q];
                              local_matrix_ij +=
                                tau_mom * supg_test_momentum *
                                scratch.strong_residual_momentum[q] * trG;
                            }
                        }
                  }

                if (tau_tracer > 0. && i_is_tracer)
                  {
                    if (ordering.is_velocity(comp_j))
                      {
                        local_matrix_ij +=
                          tau_tracer *
                          (scratch.phi_u[q][j] * grad_shape_phi_i) *
                          scratch.strong_residual_tracer[q];
                        local_matrix_ij +=
                          tau_tracer * supg_test_tracer *
                          (scratch.phi_u[q][j] * scratch.tracer_gradients[q]);
                      }

                    if (ordering.is_tracer(comp_j))
                      {
                        const double dR_dphi_j =
                          bdf_c0 * scratch.shape_phi[q][j] +
                          u_conv * scratch.grad_shape_phi[q][j];
                        local_matrix_ij +=
                          tau_tracer * supg_test_tracer * dR_dphi_j;
                      }

                    if (ordering.is_potential(comp_j))
                      local_matrix_ij +=
                        tau_tracer * supg_test_tracer *
                        (-scratch.mobility * scratch.laplacian_shape_mu[q][j]);

                    if constexpr (with_moving_mesh)
                      if (ordering.is_position(comp_j))
                        {
                          const Tensor<2, dim> &G   =
                            scratch.grad_phi_x_moving[q][j];
                          const Tensor<2, dim>  GT  = transpose(G);
                          const double          trG = trace(G);

                          const Tensor<1, dim> du_conv_dx_j =
                            -bdf_c0 * scratch.phi_x[q][j];
                          const Tensor<1, dim> dgrad_tracer_dx_j =
                            -GT * scratch.tracer_gradients[q];
                          const double dR_dx_j =
                            du_conv_dx_j * scratch.tracer_gradients[q] +
                            u_conv * dgrad_tracer_dx_j;

                          const Tensor<1, dim> dgrad_shape_phi_i_dx_j =
                            -GT * grad_shape_phi_i;
                          const double d_supg_test_dx_j =
                            du_conv_dx_j * grad_shape_phi_i +
                            u_conv * dgrad_shape_phi_i_dx_j;

                          local_matrix_ij +=
                            tau_tracer * supg_test_tracer * dR_dx_j;
                          local_matrix_ij +=
                            tau_tracer * d_supg_test_dx_j *
                            scratch.strong_residual_tracer[q];
                          local_matrix_ij +=
                            tau_tracer * supg_test_tracer *
                            scratch.strong_residual_tracer[q] * trG;
                        }
                  }

                local_matrix(i, j) += local_matrix_ij * scratch.JxW_moving[q];
              }
          }
      }
  }
} // namespace Assembly

#endif
