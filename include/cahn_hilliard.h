#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

#include <cmath>

namespace CahnHilliard
{
  template <int dim>
  inline bool
  is_abels_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::Abels;
  }

  template <int dim>
  inline bool
  is_ding_horriche_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::DingHorriche;
  }

  template <int dim>
  inline bool
  use_abels_diffusive_inertia(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param);
  }

  template <int dim>
  inline bool
  use_abels_capillary_phi_grad_mu(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param);
  }

  template <int dim>
  inline bool
  use_ding_horriche_capillary_mu_grad_phi(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_ding_horriche_model(param);
  }

  template <int dim>
  inline const char *
  model_name(const Parameters::CahnHilliard<dim> &param)
  {
    if (is_ding_horriche_model(param))
      return "Ding-Horriche";
    return "Abels";
  }

  template <int dim>
  inline double
  sigma_tilde_from_surface_tension(
    const Parameters::CahnHilliard<dim> &param)
  {
    return 3. / (2. * std::sqrt(2.)) * param.surface_tension;
  }

  template <int dim>
  inline double
  potential_double_well_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return 1.;
    return sigma_tilde / param.epsilon_interface;
  }

  template <int dim>
  inline double
  potential_gradient_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return param.epsilon_interface * param.epsilon_interface;
    return sigma_tilde * param.epsilon_interface;
  }

  template <int dim>
  inline double
  contact_angle_gradient_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    return potential_gradient_coefficient(param, sigma_tilde);
  }

  /*
   * Surface term coefficient for contact angle boundary condition.
   * This is the scaling applied to the contact angle surface integral in the
   * weak form. It accounts for model-dependent non-dimensionalizations.
   *
   * The contact angle BC contributes to the weak form of the potential
   * equation as an integrated boundary term with coefficient:
   *   epsilon^2 * contact_angle_surface_coefficient
   */
  template <int dim>
  inline double
  contact_angle_surface_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    // The contact angle boundary term uses the same diffusive scaling as the
    // bulk potential gradient term. The boundary contribution is
    // potential_gradient_coefficient * g(phi), where g(phi) = n·grad(phi).
    return potential_gradient_coefficient(param, sigma_tilde);
  }

  /*
   * Static wetting condition for a tanh diffuse interface:
   *
   *   n . grad(phi) = -cos(theta) / (sqrt(2) eps) * (1 - phi^2)
   *
   * theta is measured through the phi = +1 phase. Flip the sign if using the
   * opposite phase convention.
   */
  inline double contact_angle_normal_derivative(const double phi,
                                                const double epsilon,
                                                const double theta)
  {
    return -std::cos(theta) * (1. - phi * phi) /
           (std::sqrt(2.) * epsilon);
  }

  inline double contact_angle_normal_derivative_jacobian(const double phi,
                                                         const double epsilon,
                                                         const double theta)
  {
    return 2. * std::cos(theta) * phi / (std::sqrt(2.) * epsilon);
  }

  template <int dim>
  inline double
  ding_horriche_capillary_coefficient(
    const Parameters::CahnHilliard<dim> &param)
  {
    /*
     * Horriche's final CADYF system (3.19)-(3.22) uses the unscaled potential
     *
     *   mu_hat = phi^3 - phi - eps^2 Delta phi
     *
     * and the physical momentum force (gamma / eps) mu_hat grad(phi). For
     * CHNSModel::DingHorriche, FEZ's potential unknown is this unscaled
     * mu_hat, so the coefficient is gamma / eps directly.
     */
    return param.surface_tension / param.epsilon_interface;
  }

  /**
   * Simply return the passed phase marker
   */
  inline double tracer_identity(const double phase_marker)
  {
    return phase_marker;
  }

  /**
   * Apply a limiter to the phase tracer : res = max(-1, min(1, phi))
   */
  inline double tracer_limiter(const double phase_marker)
  {
    return std::max(-1., std::min(1., phase_marker));
  }

  /**
   * Return a pointer to the limiter function used for the phase field tracer
   */
  using TracerLimiterFunction = double (*)(double);

  template <int dim>
  TracerLimiterFunction
  get_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.with_tracer_limiter)
      return &tracer_limiter;
    else
      return &tracer_identity;
  }

  /**
   * Return a pointer to the tracer limiter function used for the degenerate
   * mobility calcul
   */
  using MobilityTracerLimiterFunction = double (*)(double);

  template <int dim>
  MobilityTracerLimiterFunction
  get_mobility_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.mobility_tracer_limiter)
      return &tracer_limiter;
    else
      return &tracer_identity;
  }

  /**
   * Apply linear mixing from value A (when phase marker = 1) to value B (phase
   * marker = -1).
   */
  inline double linear_mixing(const double phase_marker,
                              const double val_a,
                              const double val_b)
  {
    return 0.5 * ((val_a - val_b) * phase_marker + (val_a + val_b));
  }

  /**
   * Derivative w.r.t. the tracer of the linear mixing function
   */
  inline double linear_mixing_derivative(const double /*phase_marker*/,
                                         const double val_a,
                                         const double val_b)
  {
    return 0.5 * (val_a - val_b);
  }

  /**
   * Constant mobility
   */
  template <int dim>
  inline double constant_mobility(const Parameters::CahnHilliard<dim> &param,
                                  const double /*phase_marker*/)
  {
    return param.mobility;
  }

  /**
   * Degenerate mobility, according to the formula in the .param
   */
  template <int dim>
  inline double degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                                    const double phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->value(p);
  }

  /**
   * Choice between constant and degenerate mobility
   */
  template <int dim>
  using MobilityFunction = double (*)(const Parameters::CahnHilliard<dim> &,
                                      double);

  template <int dim>
  MobilityFunction<dim>
  get_mobility_function(const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility<dim>;
    else
      return &degenerate_mobility<dim>;
  }

  template <int dim>
  inline double
  constant_mobility_derivative(const Parameters::CahnHilliard<dim> & /*param*/,
                               const double /*phase_marker*/)
  {
    return 0.;
  }

  template <int dim>
  inline double
  degenerate_mobility_derivative(const Parameters::CahnHilliard<dim> &param,
                                 const double phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->gradient(p)[0];
  }

  template <int dim>
  MobilityFunction<dim>
  get_mobility_derivative_function(const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility_derivative<dim>;
    else
      return &degenerate_mobility_derivative<dim>;
  }

  template <int dim>
  inline double constant_mobility_second_derivative(
    const Parameters::CahnHilliard<dim> & /*param*/,
    const double /*phase_marker*/)
  {
    return 0.;
  }

  template <int dim>
  inline double degenerate_mobility_second_derivative(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->hessian(p)[0][0];
  }

  template <int dim>
  MobilityFunction<dim> get_mobility_second_derivative_function(
    const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility_second_derivative<dim>;
    else
      return &degenerate_mobility_second_derivative<dim>;
  }

} // namespace CahnHilliard

#endif
