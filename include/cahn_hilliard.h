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
  is_stepien_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::Stepien;
  }

  template <int dim>
  inline bool
  use_abels_diffusive_inertia(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param);
  }

  /**
   * Set to true to make the Stepien momentum equation carry the Abels
   * capillary force phi grad(mu) instead of its own (dpr - mu) grad(phi).
   * Used to reduce the Stepien model to Abels one term at a time: the two
   * forms differ by the pure gradient grad(phi mu) once dpr = 0, so they are
   * equivalent up to the pressure shift p -> p - phi mu.
   *
   * Note that the dpr grad(phi) contribution is dropped along with the
   * Stepien form, so this is only meaningful when p_ref,0 = p_ref,1.
   */
  inline constexpr bool use_abels_capillary_for_stepien = false;

  template <int dim>
  inline bool
  use_abels_capillary_phi_grad_mu(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param) ||
           (use_abels_capillary_for_stepien && is_stepien_model(param));
  }

  template <int dim>
  inline bool
  use_ding_horriche_capillary_mu_grad_phi(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_ding_horriche_model(param);
  }

  template <int dim>
  inline bool
  use_stepien_capillary_form(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_stepien_model(param);
  }

  /**
   * Ratio lambda / eta of the second viscosity to the dynamic viscosity in the
   * Stepien viscous stress
   *
   *   tau = eta(phi) (grad v + grad v^T) + lambda(phi) (div v) I.
   *
   * Unlike the Abels model, the Stepien velocity field is not solenoidal, so
   * the second term does contribute to the momentum equation. Stokes'
   * hypothesis (vanishing bulk viscosity kappa = lambda + 2/3 eta) gives
   * lambda = -2/3 eta; lambda therefore inherits the linear tracer dependence
   * of the dynamic viscosity, and dlambda/dphi = -2/3 deta/dphi.
   */
  inline constexpr double stokes_second_viscosity_ratio = -2. / 3.;

  template <int dim>
  inline const char *
  model_name(const Parameters::CahnHilliard<dim> &param)
  {
    if (is_abels_model(param))
      return "Abels";
    if (is_ding_horriche_model(param))
      return "Ding-Horriche";
    if (is_stepien_model(param))
      return "Stepien";
  
    DEAL_II_ASSERT_UNREACHABLE();
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

    if (is_stepien_model(param))
      return sigma_tilde / param.epsilon_interface;

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

    if (is_stepien_model(param))
      return param.epsilon_interface * sigma_tilde;

    return sigma_tilde * param.epsilon_interface;
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
     * and the physical momentum force (gamma / eps) mu_hat grad(phi). For the
     * tanh profile used by FEZ, gamma must be the normalized coefficient
     *
     *   gamma = 3 / (2 sqrt(2)) sigma
     *
     * so that the diffuse-interface energy integrates to the physical surface
     * tension sigma. Using sigma / eps directly would give an effective
     * surface tension (2 sqrt(2) / 3) sigma and under-predict the
     * Young-Laplace jump.
     */
    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;
    return sigma_tilde / param.epsilon_interface;
  }

  template <int dim>
  inline double
  stepien_capillary_coefficient(
    const Parameters::CahnHilliard<dim> & /*param*/)
  {
    // The Stepien momentum capillary force is muhat * grad(phi). Since the
    // potential unknown mu already stores muhat (same scaling as the potential
    // equation), the force prefactor is exactly 1.
    return 1.;
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
