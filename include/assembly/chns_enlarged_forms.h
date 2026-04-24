#ifndef ASSEMBLY_CHNS_ENLARGED_FORMS_H
#define ASSEMBLY_CHNS_ENLARGED_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>
#include <parameters.h>

#include <cmath>

using namespace dealii;

namespace Assembly
{
  // Calibration used to map the user-facing width factor to the enlarged
  // diffuse-interface thickness used by the psi reconstruction.
  inline constexpr double psi_interface_width_calibration_coefficient =
    1.21860379;

  inline double
  calibrated_enlarged_interface_thickness(
    const double epsilon_interface,
    const double psi_interface_width_factor)
  {
    AssertThrow(psi_interface_width_factor >= 1.,
                ExcMessage("'psi interface width factor' must be >= 1.0."));

    const double target_eps_eff = psi_interface_width_factor * epsilon_interface;
    const double delta_eps =
      std::sqrt((target_eps_eff * target_eps_eff -
                 epsilon_interface * epsilon_interface) /
                psi_interface_width_calibration_coefficient);

    return epsilon_interface + delta_eps;
  }

  inline double
  psi_mu_correction_eta(const double tracer_value)
  {
    // Smooth band weight that keeps the correction localized near the diffuse interface.
    const double band = 1. - tracer_value * tracer_value;
    return band * band;
  }

  inline double
  psi_mu_correction_eta_jacobian(const double tracer_value)
  {
    const double band = 1. - tracer_value * tracer_value;
    return -4. * tracer_value * band;
  }

  template <int dim>
  inline double
  compute_psi_mu_correction_prefactor(
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const double                         sigma_tilde,
    const double                         epsilon,
    const double                         length_scale_sq)
  {
    // Keep the correction opt-in and pay nothing when it is disabled.
    if (std::abs(cahn_hilliard.psi_mu_correction_factor) < 1e-14)
      return 0.;
    return cahn_hilliard.psi_mu_correction_factor * length_scale_sq /
           (epsilon * sigma_tilde);
  }

  template <int dim, typename ScratchDataType, typename VectorType>
  inline void
  assemble_psi_equation_rhs(const ComponentOrdering &ordering,
                            const ScratchDataType   &scratch,
                            const Parameters::CahnHilliard<dim> &cahn_hilliard,
                            const double             length_scale_sq,
                            VectorType              &local_rhs)
  {
    const double correction_prefactor = compute_psi_mu_correction_prefactor(
      cahn_hilliard,
      scratch.sigma_tilde,
      scratch.epsilon,
      length_scale_sq);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double psi_mu_correction =
            correction_prefactor *
            psi_mu_correction_eta(scratch.tracer_values[q]) *
            scratch.potential_values[q];
          const double local_rhs_i =
            scratch.shape_psi[q][i] *
              (scratch.psi_values[q] - scratch.tracer_values[q] +
               scratch.source_term_psi[q] - psi_mu_correction) +
            length_scale_sq * scalar_product(scratch.grad_shape_psi[q][i],
                                             scratch.psi_gradients[q]);

          local_rhs(i) -= local_rhs_i * scratch.JxW_moving[q];
        }
  }

  template <int dim,
            bool with_moving_mesh,
            typename ScratchDataType,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_psi_equation_matrix(const ComponentOrdering &ordering,
                               const CouplingTableType &coupling_table,
                               const ScratchDataType   &scratch,
                               const Parameters::CahnHilliard<dim> &cahn_hilliard,
                               const double             length_scale_sq,
                               MatrixType              &local_matrix)
  {
    const double correction_prefactor = compute_psi_mu_correction_prefactor(
      cahn_hilliard,
      scratch.sigma_tilde,
      scratch.epsilon,
      length_scale_sq);

    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double          phi_i  = scratch.shape_psi[q][i];
          const Tensor<1, dim> &grad_i = scratch.grad_shape_psi[q][i];
          const double eta_weight =
            psi_mu_correction_eta(scratch.tracer_values[q]);
          const double eta_weight_jacobian =
            psi_mu_correction_eta_jacobian(scratch.tracer_values[q]);
          const double psi_mu_correction =
            correction_prefactor * eta_weight * scratch.potential_values[q];

          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            {
              if (coupling_table[scratch.components[i]][scratch.components[j]] !=
                  DoFTools::always)
                continue;

              const unsigned int comp_j = scratch.components[j];
              double             local_matrix_ij = 0.;

              if (ordering.is_psi(comp_j))
                {
                  local_matrix_ij += phi_i * scratch.shape_psi[q][j];
                  local_matrix_ij +=
                    length_scale_sq *
                    scalar_product(grad_i, scratch.grad_shape_psi[q][j]);
                }

              if (ordering.is_tracer(comp_j))
                local_matrix_ij -=
                  phi_i *
                  (scratch.shape_phi[q][j] +
                   correction_prefactor * eta_weight_jacobian *
                     scratch.potential_values[q] * scratch.shape_phi[q][j]);

              if (ordering.is_potential(comp_j))
                local_matrix_ij -=
                  phi_i * correction_prefactor * eta_weight *
                  scratch.shape_mu[q][j];

              if constexpr (with_moving_mesh)
                if (ordering.is_position(comp_j))
                  {
                    const Tensor<2, dim> &G   = scratch.grad_phi_x_moving[q][j];
                    const double          trG = trace(G);

                    local_matrix_ij +=
                      phi_i *
                      (scratch.psi_values[q] - scratch.tracer_values[q] -
                       psi_mu_correction) *
                      trG;
                    local_matrix_ij +=
                      length_scale_sq *
                      (scalar_product(-transpose(G) * grad_i,
                                      scratch.psi_gradients[q]) +
                       scalar_product(grad_i,
                                      -transpose(G) * scratch.psi_gradients[q]) +
                       scalar_product(grad_i, scratch.psi_gradients[q]) * trG);
                  }

              local_matrix(i, j) += local_matrix_ij * scratch.JxW_moving[q];
            }
        }
  }
} // namespace Assembly

#endif
