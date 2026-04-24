#ifndef ASSEMBLY_PSEUDOSOLID_FORMS_H
#define ASSEMBLY_PSEUDOSOLID_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>

#include <cmath>

#include <parameters.h>

using namespace dealii;

namespace Assembly::Pseudosolid
{
  // struct CustomPenaltyLawData
  // {
  //   double volumetric_stress  = 0.0;
  //   double volumetric_tangent = 0.0;
  // };

  // template <int dim>
  // inline CustomPenaltyLawData
  // compute_custom_penalty_law_data(
  //   const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
  //   const double                        J)
  // {
  //   AssertThrow(pseudosolid_parameters.custom_penalty_k >= 1,
  //               ExcMessage("Custom penalty law requires K >= 1."));

  //   const double a = 1.0 - pseudosolid_parameters.custom_penalty_j_min;

  //   const double shifted_J = J + a;


  //   const double inv_a_pow_k =
  //     1.0 / std::pow(a,
  //                    static_cast<double>(pseudosolid_parameters.custom_penalty_k));
  //   const double ratio = a / shifted_J;

  //   double series      = 0.0;
  //   double series_term = inv_a_pow_k * ratio;
  //   for (unsigned int m = 1; m < pseudosolid_parameters.custom_penalty_k; ++m)
  //     {
  //       series += series_term / static_cast<double>(m);
  //       series_term *= ratio;
  //     }

  //   CustomPenaltyLawData data;
  //   data.volumetric_stress =
  //     std::log(J) + inv_a_pow_k * std::log(J / shifted_J) + series -
  //     pseudosolid_parameters.custom_penalty_identity_shift;
  //   data.volumetric_tangent = 1.0 + series_term;

  //   return data;
  // }

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
           lame_mu * scalar_product(grad_trial + transpose(grad_trial), grad_test);
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
  HN_0_matrix_contribution(const double          lame_mu,
                          const double          lame_lambda,
                          const Tensor<2, dim> &F,
                          const Tensor<2, dim> &F_inv,
                          const Tensor<2, dim> &F_inv_T,
                          const double          J,
                          const Tensor<2, dim> &grad_test,
                          const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);
    const double tr_Finv_grad = trace(F_inv * grad_trial);
    // dF = grad_trial
    const Tensor<2, dim> dF = grad_trial;
    // d(F^{-T}) = -F^{-T} (dF)^T F^{-T}
    const Tensor<2, dim> dF_inv_T =
      -F_inv_T * transpose(dF) * F_inv_T;
    // dI1 = 2 F : dF
    const double dI1 = 2.0 * scalar_product(F, dF);
    // d(J^{-2/3}) = -(2/3) J^{-2/3} tr(F^{-1} dF)
    const double dJm23 =
      -(2.0 / 3.0) * Jm23 * tr_Finv_grad;

    // --- isochoric part ---
    const Tensor<2, dim> term_iso =
      lame_mu *
      (
        dJm23 * (F - (I1 / 3.0) * F_inv_T) +
        Jm23 *
          (
            dF
            - (dI1 / 3.0) * F_inv_T
            - (I1 / 3.0) * dF_inv_T
          )
      );
    // --- volumetric part ---
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> term_vol =
      lame_kappa *
      (
        tr_Finv_grad * F_inv_T +
        std::log(J) * dF_inv_T
      );

    const Tensor<2, dim> dP = term_iso + term_vol;

    return scalar_product(dP, grad_test);
  }

  template <int dim>
  inline double
  HN_1_matrix_contribution(const double          lame_mu,
                                          const double          lame_lambda,
                                          const Tensor<2, dim> &F,
                                          const Tensor<2, dim> &F_inv,
                                          const Tensor<2, dim> &F_inv_T,
                                          const double          J,
                                          const Tensor<2, dim> &grad_test,
                                          const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);
    const double tr_Finv_grad = trace(F_inv * grad_trial);
    const Tensor<2, dim> dF = grad_trial;
    const Tensor<2, dim> dF_inv_T =
      -F_inv_T * transpose(dF) * F_inv_T;
    const double dI1 = 2.0 * scalar_product(F, dF);
    const double dJm23 =
      -(2.0 / 3.0) * Jm23 * tr_Finv_grad;
    // --- isochoric part ---
    const Tensor<2, dim> term_iso =
      lame_mu *
      (dJm23 * (F - (I1 / 3.0) * F_inv_T) +
        Jm23 *
          (dF
            - (dI1 / 3.0) * F_inv_T
            - (I1 / 3.0) * dF_inv_T)
      );
    // --- volumetric part ---
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> term_vol =
      lame_kappa *
      (
        (1.0/2.0)*(J + 1.0/J) * tr_Finv_grad * F_inv_T +
        (1.0/2.0)*(J - 1.0/J) * dF_inv_T
      );

    const Tensor<2, dim> dP = term_iso + term_vol;

    return scalar_product(dP, grad_test);
  }

  template <int dim>
  inline double
  Ogden_1_matrix_contribution(const double          lame_mu,
                              const double          lame_lambda,
                              const Tensor<2, dim> &F,
                              const Tensor<2, dim> &F_inv,
                              const Tensor<2, dim> &F_inv_T,
                              const double          J,
                              const Tensor<2, dim> &grad_test,
                              const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);
    const double tr_Finv_grad = trace(F_inv * grad_trial);
    const Tensor<2, dim> dF = grad_trial;
    const Tensor<2, dim> dF_inv_T =
      -F_inv_T * transpose(dF) * F_inv_T;
    const double dI1 = 2.0 * scalar_product(F, dF);
    const double dJm23 =
      -(2.0 / 3.0) * Jm23 * tr_Finv_grad;
    // --- isochoric part ---
    const Tensor<2, dim> term_iso =
      lame_mu *
      (
        dJm23 * (F - (I1 / 3.0) * F_inv_T) +
        Jm23 *
          (
            dF
            - (dI1 / 3.0) * F_inv_T
            - (I1 / 3.0) * dF_inv_T
          )
      );
    // --- volumetric part ---
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> term_vol =
      lame_kappa *
      (
        (1.0/J) * tr_Finv_grad * F_inv_T +
        ( 1.0 - 1.0/J) * dF_inv_T
      );

    const Tensor<2, dim> dP = term_iso + term_vol;

    return scalar_product(dP, grad_test);
  }

  template <int dim>
  inline double
  Ogden_2_matrix_contribution(const double          lame_mu,
                              const double          lame_lambda,
                              const Tensor<2, dim> &F,
                              const Tensor<2, dim> &F_inv,
                              const Tensor<2, dim> &F_inv_T,
                              const double          J,
                              const Tensor<2, dim> &grad_test,
                              const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);
    const double tr_Finv_grad = trace(F_inv * grad_trial);
    const Tensor<2, dim> dF = grad_trial;
    const Tensor<2, dim> dF_inv_T =
      -F_inv_T * transpose(dF) * F_inv_T;
    const double dI1 = 2.0 * scalar_product(F, dF);
    const double dJm23 =
      -(2.0 / 3.0) * Jm23 * tr_Finv_grad;
    // --- isochoric part ---
    const Tensor<2, dim> term_iso =
      lame_mu *
      (
        dJm23 * (F - (I1 / 3.0) * F_inv_T) +
        Jm23 *
          (
            dF
            - (dI1 / 3.0) * F_inv_T
            - (I1 / 3.0) * dF_inv_T
          )
      );
    // --- volumetric part ---
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> term_vol =
      lame_kappa *
      (
        std::pow(J, -2.0) * tr_Finv_grad * F_inv_T +
        (1.0/2.0)*(1 - std::pow(J, -2.0)) * dF_inv_T
      );

    const Tensor<2, dim> dP = term_iso + term_vol;

    return scalar_product(dP, grad_test);
  }  

  template <int dim>
  inline double
  Ogden_2_classique_matrix_contribution(const double          lame_mu,
                                  const double          lame_lambda,
                                  const Tensor<2, dim> &F_inv,
                                  const Tensor<2, dim> &F_inv_T,
                                  const double          J,
                                  const Tensor<2, dim> &grad_test,
                                  const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> dP = 
          lame_mu * grad_trial
          + lame_mu * (F_inv_T * transpose(grad_trial) * F_inv_T)
          + lame_lambda *
            (
              std::pow(J, -2.0) * trace(F_inv * grad_trial) * F_inv_T 
              - (1.0/2.0)*(1 - std::pow(J, -2.0)) * (F_inv_T * transpose(grad_trial) * F_inv_T)
            );
    return scalar_product(dP, grad_test);
  }

  template <int dim>
  inline double
  quad_matrix_contribution(const double          lame_mu,
                          const double          lame_lambda,
                          const Tensor<2, dim> &F,
                          const Tensor<2, dim> &F_inv,
                          const Tensor<2, dim> &F_inv_T,
                          const double          J,
                          const Tensor<2, dim> &grad_test,
                          const Tensor<2, dim> &grad_trial)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);
    const double tr_Finv_grad = trace(F_inv * grad_trial);
    const Tensor<2, dim> dF = grad_trial;
    const Tensor<2, dim> dF_inv_T =
      -F_inv_T * transpose(dF) * F_inv_T;
    const double dI1 = 2.0 * scalar_product(F, dF);
    const double dJm23 =
      -(2.0 / 3.0) * Jm23 * tr_Finv_grad;
    // --- isochoric part ---
    const Tensor<2, dim> term_iso =
      lame_mu *
      (
        dJm23 * (F - (I1 / 3.0) * F_inv_T) +
        Jm23 *
          (
            dF
            - (dI1 / 3.0) * F_inv_T
            - (I1 / 3.0) * dF_inv_T
          )
      );
    // --- volumetric part ---
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> term_vol =
      lame_kappa *
      (
        2.0*J*(J-1) * tr_Finv_grad * F_inv_T +
        J*(J-1) * dF_inv_T
      );

    const Tensor<2, dim> dP = term_iso + term_vol;

    return scalar_product(dP, grad_test);
  }  

  template <int dim>
  inline double
  matrix_contribution(
    const typename Parameters::PseudoSolid<dim>::ConstitutiveModel
      constitutive_model,
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
    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
      return neo_hookean_matrix_contribution(
        lame_mu, lame_lambda, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::HN_0)
      return HN_0_matrix_contribution(
        lame_mu, lame_lambda, F, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::HN_1)
      return HN_1_matrix_contribution(
        lame_mu, lame_lambda, F, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_1)
      return Ogden_1_matrix_contribution(
        lame_mu, lame_lambda, F, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_2)
      return Ogden_2_matrix_contribution(
        lame_mu, lame_lambda, F, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_2_classique)
      return Ogden_2_classique_matrix_contribution(
        lame_mu, lame_lambda, F_inv, F_inv_T, J, grad_test, grad_trial);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::quad)
      return quad_matrix_contribution(
        lame_mu, lame_lambda, F, F_inv, F_inv_T, J, grad_test, grad_trial);

    return linear_matrix_contribution(
      lame_mu, lame_lambda, div_test, grad_test, div_trial, grad_trial);
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
  HN_0_rhs_contribution(const double          lame_mu,
                                      const double          lame_lambda,
                                      const Tensor<2, dim> &F,
                                      const Tensor<2, dim> &F_inv_T,
                                      const double          J,
                                      const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);

    // Isochoric part
    const Tensor<2, dim> P_iso =
      lame_mu * Jm23 * (F - (I1 / 3.0) * F_inv_T);

    // Volumetric part (log formulation)
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> P_vol =
      lame_kappa * std::log(J) * F_inv_T;

    const Tensor<2, dim> P = P_iso + P_vol;

    return scalar_product(P, grad_test);
  }

  template <int dim>
  inline double
  HN_1_rhs_contribution(const double          lame_mu,
                        const double          lame_lambda,
                        const Tensor<2, dim> &F,
                        const Tensor<2, dim> &F_inv_T,
                        const double          J,
                        const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);

    // Isochoric part
    const Tensor<2, dim> P_iso =
      lame_mu * Jm23 * (F - (I1 / 3.0) * F_inv_T);

    // Volumetric part (log formulation)
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> P_vol =
      lame_kappa * (1.0/2.0)*(J - 1.0/J) * F_inv_T;

    const Tensor<2, dim> P = P_iso + P_vol;

    return scalar_product(P, grad_test);
  }  

  template <int dim>
  inline double
  Ogden_1_rhs_contribution(const double          lame_mu,
                          const double          lame_lambda,
                          const Tensor<2, dim> &F,
                          const Tensor<2, dim> &F_inv_T,
                          const double          J,
                          const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);

    // Isochoric part
    const Tensor<2, dim> P_iso =
      lame_mu * Jm23 * (F - (I1 / 3.0) * F_inv_T);

    // Volumetric part
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> P_vol =
      lame_kappa * (1.0 - 1.0/J) * F_inv_T;

    const Tensor<2, dim> P = P_iso + P_vol;

    return scalar_product(P, grad_test);
  }  

  template <int dim>
  inline double
  Ogden_2_rhs_contribution(const double          lame_mu,
                          const double          lame_lambda,
                          const Tensor<2, dim> &F,
                          const Tensor<2, dim> &F_inv_T,
                          const double          J,
                          const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);

    // Isochoric part
    const Tensor<2, dim> P_iso =
      lame_mu * Jm23 * (F - (I1 / 3.0) * F_inv_T);

    // Volumetric part
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> P_vol =
      lame_kappa * (1.0/2.0)*(1 - std::pow(J, -2.0)) * F_inv_T;

    const Tensor<2, dim> P = P_iso + P_vol;

    return scalar_product(P, grad_test);
  }  

  template <int dim>
  inline double
  Ogden_2_classique_rhs_contribution(const double          lame_mu,
                               const double          lame_lambda,
                               const Tensor<2, dim> &F,
                               const Tensor<2, dim> &F_inv_T,
                               const double          J,
                               const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> P =
      lame_mu * (F - F_inv_T) + lame_lambda * (1.0/2.0)*(1 - std::pow(J, -2.0)) * F_inv_T;

    return scalar_product(P, grad_test);
  }

  template <int dim>
  inline double
  quad_rhs_contribution(const double          lame_mu,
                        const double          lame_lambda,
                        const Tensor<2, dim> &F,
                        const Tensor<2, dim> &F_inv_T,
                        const double          J,
                        const Tensor<2, dim> &grad_test)
  {
    const Tensor<2, dim> C = transpose(F) * F;
    const double I1 = trace(C);
    const double Jm23 = std::pow(J, -2.0 / 3.0);

    // Isochoric part
    const Tensor<2, dim> P_iso =
      lame_mu * Jm23 * (F - (I1 / 3.0) * F_inv_T);

    // Volumetric part
    const double lame_kappa = lame_lambda + (2.0/3.0) * lame_mu;
    const Tensor<2, dim> P_vol =
      lame_kappa * J*(J-1) * F_inv_T;

    const Tensor<2, dim> P = P_iso + P_vol;

    return scalar_product(P, grad_test);
  }  

  template <int dim>
  inline double
  rhs_contribution(
    const typename Parameters::PseudoSolid<dim>::ConstitutiveModel
      constitutive_model,
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
    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
      return neo_hookean_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::HN_0)
      return HN_0_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::HN_1)
      return HN_1_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_1)
      return Ogden_1_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_2)
      return Ogden_2_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::Ogden_2_classique)
      return Ogden_2_classique_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    if (constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::quad)
      return quad_rhs_contribution(
        lame_mu, lame_lambda, F, F_inv_T, J, grad_test);

    return linear_rhs_contribution(
      lame_mu, lame_lambda, trace_strain, strain, div_test, grad_test);
  }

  template <int dim,
            typename ScratchData,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_chns_matrix(
    const ComponentOrdering &ordering,
    const CouplingTableType &coupling_table,
    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
    const ScratchData &scratch,
    MatrixType        &local_matrix)
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
                matrix_contribution(pseudosolid_parameters.constitutive_model,
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
  assemble_chns_rhs(
    const ComponentOrdering &ordering,
    const Parameters::PseudoSolid<dim> &pseudosolid_parameters,
    const ScratchData &scratch,
    VectorType        &local_rhs)
  {
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      {
        const double          trace_strain =
          trace(scratch.present_position_gradients[q]) - static_cast<double>(dim);
        const Tensor<2, dim> strain =
          Tensor<2, dim>(symmetrize(scratch.present_position_gradients[q]) -
                         unit_symmetric_tensor<dim>());

        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          if (ordering.is_position(scratch.components[i]))
            local_rhs(i) -=
              (rhs_contribution(pseudosolid_parameters.constitutive_model,
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
