#ifndef ASSEMBLY_PSEUDOSOLID_FORMS_H
#define ASSEMBLY_PSEUDOSOLID_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>
#include <assembly/mesh_concentration_tools.h>
#include <parameters.h>

#include <algorithm>
#include <cmath>

using namespace dealii;

namespace Assembly::Pseudosolid
{
  namespace MaterialLaw
  {
    template <int dim>
    inline double
    linear_matrix_contribution(const double          lame_mu,
                               const double          lame_lambda,
                               const double          div_test,
                               const Tensor<2, dim> &grad_test,
                               const double          div_trial,
                               const Tensor<2, dim> &grad_trial)
    {
      return lame_lambda * div_trial * div_test +
             lame_mu *
               scalar_product(grad_trial + transpose(grad_trial), grad_test);
    }

    template <int dim>
    inline double
    neo_hookean_matrix_contribution(const double          lame_mu,
                                    const double          lame_lambda,
                                    const Tensor<2, dim> &F_inv,
                                    const Tensor<2, dim> &F_inv_T,
                                    const double          J,
                                    const Tensor<2, dim> &grad_test,
                                    const Tensor<2, dim> &grad_trial)
    {
      const Tensor<2, dim> dP =
        lame_mu * grad_trial +
        (lame_mu - lame_lambda * std::log(J)) *
          (F_inv_T * transpose(grad_trial) * F_inv_T) +
        lame_lambda * trace(F_inv * grad_trial) * F_inv_T;

      return scalar_product(dP, grad_test);
    }

    template <int dim>
    inline double
    ogden_matrix_contribution(const double          lame_mu,
                              const double          lame_lambda,
                              const double          beta,
                              const Tensor<2, dim> &,
                              const Tensor<2, dim> &F_inv,
                              const Tensor<2, dim> &F_inv_T,
                              const double          J,
                              const Tensor<2, dim> &grad_test,
                              const Tensor<2, dim> &grad_trial)
    {
      AssertThrow(std::abs(beta) > 1e-14,
                  ExcMessage("Ogden pseudosolid law requires beta != 0."));

      const Tensor<2, dim> dF         = grad_trial;
      const Tensor<2, dim> dF_inv_T   = -F_inv_T * transpose(dF) * F_inv_T;
      const double         Jm_beta    = std::pow(J, -beta);
      const double         tr_Finv_dF = trace(F_inv * dF);
      const double         volumetric_stress =
        (1.0 / beta) * (1.0 - Jm_beta);

      const Tensor<2, dim> dP_vol =
        lame_lambda * (Jm_beta * tr_Finv_dF * F_inv_T +
                       volumetric_stress * dF_inv_T);
      const Tensor<2, dim> dP_shape = lame_mu * (dF - dF_inv_T);

      return scalar_product(dP_shape + dP_vol, grad_test);
    }

    template <int dim>
    inline double
    linear_rhs_contribution(const double          lame_mu,
                            const double          lame_lambda,
                            const double          trace_strain,
                            const Tensor<2, dim> &strain,
                            const double          div_test,
                            const Tensor<2, dim> &grad_test)
    {
      return lame_lambda * trace_strain * div_test +
             2.0 * lame_mu * scalar_product(strain, grad_test);
    }

    template <int dim>
    inline double
    neo_hookean_rhs_contribution(const double          lame_mu,
                                 const double          lame_lambda,
                                 const Tensor<2, dim> &F,
                                 const Tensor<2, dim> &F_inv_T,
                                 const double          J,
                                 const Tensor<2, dim> &grad_test)
    {
      const Tensor<2, dim> P =
        lame_mu * (F - F_inv_T) + lame_lambda * std::log(J) * F_inv_T;

      return scalar_product(P, grad_test);
    }

    template <int dim>
    inline double
    ogden_rhs_contribution(const double          lame_mu,
                           const double          lame_lambda,
                           const double          beta,
                           const Tensor<2, dim> &F,
                           const Tensor<2, dim> &F_inv_T,
                           const double          J,
                           const Tensor<2, dim> &grad_test)
    {
      AssertThrow(std::abs(beta) > 1e-14,
                  ExcMessage("Ogden pseudosolid law requires beta != 0."));

      const double Jm_beta = std::pow(J, -beta);

      const Tensor<2, dim> P_shape = lame_mu * (F - F_inv_T);
      const Tensor<2, dim> P_vol =
        lame_lambda * (1.0 / beta) * (1.0 - Jm_beta) * F_inv_T;

      return scalar_product(P_shape + P_vol, grad_test);
    }
  } // namespace MaterialLaw

  namespace MeshConcentration
  {
    template <int dim>
    inline double equivalent_size_stiffness(
      const Parameters::PseudoSolid<dim> &,
      const double                        lame_mu,
      const double                        lame_lambda)
    {
      return std::max(lame_lambda + 2.0 * lame_mu / dim, 1e-14);
    }

    inline double pressure_coefficient(const double ramp,
                                       const double size_pressure_coefficient,
                                       const double equivalent_stiffness)
    {
      return ramp * size_pressure_coefficient * equivalent_stiffness;
    }

    inline double size_pressure(const double coefficient,
                                const double current_size_weight,
                                const double h_background,
                                const double h_current,
                                const double h_target,
                                const double h_min)
    {
      return MeshConcentrationTools::size_pressure(coefficient,
                                                   current_size_weight,
                                                   h_background,
                                                   h_current,
                                                   h_target,
                                                   h_min);
    }

    inline double size_pressure_derivative_factor(
      const double coefficient,
      const double current_size_weight,
      const double h_background,
      const double h_current,
      const double h_target,
      const double h_min)
    {
      return MeshConcentrationTools::size_pressure_derivative_factor(
        coefficient,
        current_size_weight,
        h_background,
        h_current,
        h_target,
        h_min);
    }

    template <int dim>
    inline Tensor<2, dim>
    piola_stress(const double          p_size,
                 const Tensor<2, dim> &F_inv_T,
                 const double          J)
    {
      return J * p_size * F_inv_T;
    }

    template <int dim>
    inline Tensor<2, dim>
    piola_derivative_wrt_position(
      const double          coefficient,
      const double          current_size_weight,
      const double          pressure_derivative_factor,
      const double          p_size,
      const Tensor<2, dim> &F_inv_T,
      const double          J,
      const Tensor<2, dim> &grad_variation_current)
    {
      const double div_dx = trace(grad_variation_current);
      const double delta_p =
        pressure_derivative_factor * coefficient * current_size_weight / dim *
        div_dx;

      return J * ((p_size * div_dx + delta_p) * F_inv_T -
                  p_size * transpose(grad_variation_current) * F_inv_T);
    }

    template <int dim>
    inline Tensor<2, dim>
    piola_derivative_wrt_h(const double          coefficient,
                           const double          current_size_weight,
                           const double          pressure_derivative_factor,
                           const double          h_target,
                           const double          h_min,
                           const double          dh_deta,
                           const double          phi_eta,
                           const Tensor<2, dim> &F_inv_T,
                           const double          J)
    {
      const double eps_h   = 1e-14;
      const double h_safe  = std::max(h_target, h_min + eps_h);
      const double delta_h = dh_deta * phi_eta;
      const double delta_p =
        -pressure_derivative_factor * coefficient *
        (1.0 + current_size_weight) * delta_h / h_safe;

      return J * delta_p * F_inv_T;
    }
  } // namespace MeshConcentration

  template <int dim>
  inline double
  matrix_contribution(
    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
    const double          lame_mu,
    const double          lame_lambda,
    const Tensor<2, dim> &F,
    const Tensor<2, dim> &F_inv,
    const Tensor<2, dim> &F_inv_T,
    const double          J,
    const double          div_test,
    const Tensor<2, dim> &grad_test,
    const double          div_trial,
    const Tensor<2, dim> &grad_trial)
  {
    if (pseudosolid_parameters.constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
      return MaterialLaw::neo_hookean_matrix_contribution(
        lame_mu, lame_lambda, F_inv, F_inv_T, J, grad_test, grad_trial);
    if (pseudosolid_parameters.constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden)
      return MaterialLaw::ogden_matrix_contribution(
        lame_mu,
        lame_lambda,
        pseudosolid_parameters.ogden_beta,
        F,
        F_inv,
        F_inv_T,
        J,
        grad_test,
        grad_trial);

    return MaterialLaw::linear_matrix_contribution(
      lame_mu, lame_lambda, div_test, grad_test, div_trial, grad_trial);
  }

  template <int dim>
  inline double
  rhs_contribution(
    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
    const double          lame_mu,
    const double          lame_lambda,
    const double          trace_strain,
    const Tensor<2, dim> &strain,
    const Tensor<2, dim> &F,
    const Tensor<2, dim> &F_inv_T,
    const double          J,
    const double          div_test,
    const Tensor<2, dim> &grad_test)
  {
    if (pseudosolid_parameters.constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
      return MaterialLaw::neo_hookean_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);
    if (pseudosolid_parameters.constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden)
      return MaterialLaw::ogden_rhs_contribution(lame_mu,
                                                 lame_lambda,
                                                 pseudosolid_parameters.ogden_beta,
                                                 F,
                                                 F_inv_T,
                                                 J,
                                                 grad_test);

    return MaterialLaw::linear_rhs_contribution(
      lame_mu, lame_lambda, trace_strain, strain, div_test, grad_test);
  }

  template <int dim,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void assemble_chns_matrix(
    const ComponentOrdering            &ordering,
    const CouplingTableType            &coupling_table,
    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
    const ScratchData                  &scratch,
    MatrixType                         &local_matrix)
  {
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        if (!ordering.is_position(scratch.components[i]))
          continue;

        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          {
            if (coupling_table[scratch.components[i]][scratch.components[j]] !=
                  DoFTools::always ||
                !ordering.is_position(scratch.components[j]))
              continue;

            local_matrix(i, j) +=
              matrix_contribution(pseudosolid_parameters,
                                  scratch.lame_mu[q],
                                  scratch.lame_lambda[q],
                                  scratch.present_position_gradients[q],
                                  scratch.present_position_inv_gradients[q],
                                  scratch.present_position_inv_gradients_T[q],
                                  scratch.present_position_J[q],
                                  scratch.div_phi_x[q][i],
                                  scratch.grad_phi_x[q][i],
                                  scratch.div_phi_x[q][j],
                                  scratch.grad_phi_x[q][j]) *
              scratch.JxW_fixed[q];
          }
      }
  }

  template <int dim, typename ScratchData, typename VectorType>
  inline void
  assemble_chns_rhs(const ComponentOrdering            &ordering,
                    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
                    const ScratchData                  &scratch,
                    VectorType                         &local_rhs)
  {
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      const double trace_strain =
        trace(scratch.present_position_gradients[q]) - static_cast<double>(dim);
      const Tensor<2, dim> strain =
        Tensor<2, dim>(symmetrize(scratch.present_position_gradients[q]) -
                       unit_symmetric_tensor<dim>());

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          if (ordering.is_position(scratch.components[i]))
            local_rhs(i) -=
              (rhs_contribution(pseudosolid_parameters,
                                scratch.lame_mu[q],
                                scratch.lame_lambda[q],
                                trace_strain,
                                strain,
                                scratch.present_position_gradients[q],
                                scratch.present_position_inv_gradients_T[q],
                                scratch.present_position_J[q],
                                scratch.div_phi_x[q][i],
                                scratch.grad_phi_x[q][i]) +
               scratch.phi_x[q][i] * scratch.source_term_position[q]) *
              scratch.JxW_fixed[q];
      }
  }
} // namespace Assembly::Pseudosolid

#endif
