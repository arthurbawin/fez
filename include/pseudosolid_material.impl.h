#ifndef PSEUDOSOLID_MATERIAL_IMPL_H
#define PSEUDOSOLID_MATERIAL_IMPL_H

namespace Parameters
{
  // distance a phi=0
  template <int dim>
  double PseudoSolid<dim>::evaluate_distance_to_phi0(
    const double phi,
    const double epsilon_interface) const
  {
    constexpr double phi_clip    = 0.999999;
    const double     phi_clamped = std::max(-phi_clip, std::min(phi, phi_clip));

    return std::sqrt(2.0) * epsilon_interface *
           std::abs(std::atanh(phi_clamped));
  }

  // facteur de rigidité
  template <int dim>
  double PseudoSolid<dim>::evaluate_stiffness_factor_from_phi(
    const double phi,
    const double epsilon_interface,
    const double stiffness_min_factor) const
  {
    const double d  = evaluate_distance_to_phi0(phi, epsilon_interface);
    const double d0 = d_phi_0;

    AssertThrow(d0 > 0.0, ExcMessage("d_phi_0 must be > 0"));
    AssertThrow(stiffness_min_factor >= 0.0 && stiffness_min_factor < 1.0,
                ExcMessage("Pseudo-solid stiffness minimum factor must satisfy "
                           "0 <= factor < 1"));

    const double s_core = (d * d) / (d * d + d0 * d0);

    return stiffness_min_factor + (1.0 - stiffness_min_factor) * s_core;
  }

  // derivee du facteur de rigidité
  template <int dim>
  double PseudoSolid<dim>::evaluate_stiffness_factor_derivative_from_phi(
    const double phi,
    const double epsilon_interface,
    const double stiffness_min_factor) const
  {
    const double h = 1e-6;

    const double sp = evaluate_stiffness_factor_from_phi(phi + h,
                                                         epsilon_interface,
                                                         stiffness_min_factor);
    const double sm = evaluate_stiffness_factor_from_phi(phi - h,
                                                         epsilon_interface,
                                                         stiffness_min_factor);

    return (sp - sm) / (2.0 * h);
  }

  // lame calculation
  template <int dim>
  std::pair<double, double> PseudoSolid<dim>::evaluate_lame_from_phi_value(
    const double phi,
    const double epsilon_interface,
    const double lame_lambda_base,
    const double lame_mu_base) const
  {
    if (stiffness_model == StiffnessModel::direct_lame)
      return {lame_lambda_base, lame_mu_base};

    const double s_lambda =
      evaluate_stiffness_factor_from_phi(phi,
                                         epsilon_interface,
                                         lambda_min_factor);
    const double s_mu =
      evaluate_stiffness_factor_from_phi(phi, epsilon_interface, mu_min_factor);

    return {s_lambda * lame_lambda_base, s_mu * lame_mu_base};
  }

  // derivee des lame
  template <int dim>
  std::pair<double, double>
  PseudoSolid<dim>::evaluate_lame_derivatives_from_phi_value(
    const double phi,
    const double epsilon_interface,
    const double lame_lambda_base,
    const double lame_mu_base) const
  {
    if (stiffness_model == StiffnessModel::direct_lame)
      return {0.0, 0.0};

    const double dlambda_scale_dphi =
      evaluate_stiffness_factor_derivative_from_phi(phi,
                                                    epsilon_interface,
                                                    lambda_min_factor);
    const double dmu_scale_dphi =
      evaluate_stiffness_factor_derivative_from_phi(phi,
                                                    epsilon_interface,
                                                    mu_min_factor);

    return {dlambda_scale_dphi * lame_lambda_base,
            dmu_scale_dphi * lame_mu_base};
  }
} // namespace Parameters

#endif
