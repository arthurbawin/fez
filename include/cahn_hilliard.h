#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

#include <cmath>

namespace CahnHilliard
{
  // --- CHNS model selection -------------------------------------------------
  // Two diffuse-interface models share the same unknowns (u, p, phi, mu) but
  // differ in the potential scaling, the capillary momentum force and the
  // presence of diffusive inertia:
  //   * abels          : mu = sigma_tilde/eps phi(phi^2-1) - sigma_tilde eps
  //                      lap(phi); capillary force phi*grad(mu); diffusive
  //                      inertia present.
  //   * ding_horriche  : mu = phi(phi^2-1) - eps^2 lap(phi) (unscaled
  //                      potential); capillary force gamma*mu*grad(phi); no
  //                      diffusive inertia.

  template <int dim>
  inline bool is_abels_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::abels;
  }

  template <int dim>
  inline bool is_ding_horriche_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::ding_horriche;
  }

  // Abels with non-linear mixing: the material properties (density, viscosity)
  // are affine in the sharpened material marker q = tanh(k phi)/tanh(k) instead
  // of phi. Everything else is the Abels model, so is_abels_model() below is
  // deliberately NOT true for it (the marker abstraction handles the
  // difference at the value level, not through a structural branch).
  template <int dim>
  inline bool is_abels_nlm_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::abels_nlm;
  }

  template <int dim>
  inline const char *model_name(const Parameters::CahnHilliard<dim> &param)
  {
    if (is_ding_horriche_model(param))
      return "Ding-Horriche";
    if (is_abels_nlm_model(param))
      return "Abels (non-linear mixing)";
    return "Abels";
  }

  /**
   * Coefficient of the double-well term phi(phi^2 - 1) in the potential
   * equation. Ding-Horriche uses the unscaled potential (coefficient 1);
   * Abels scales it by sigma_tilde / epsilon.
   */
  template <int dim>
  inline double
  potential_double_well_coefficient(const Parameters::CahnHilliard<dim> &param,
                                    const double sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return 1.;
    return sigma_tilde / param.epsilon_interface;
  }

  /**
   * Coefficient of the gradient term grad(phi) in the potential equation.
   * Ding-Horriche uses eps^2; Abels uses sigma_tilde * epsilon.
   */
  template <int dim>
  inline double
  potential_gradient_coefficient(const Parameters::CahnHilliard<dim> &param,
                                 const double sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return param.epsilon_interface * param.epsilon_interface;
    return sigma_tilde * param.epsilon_interface;
  }

  /**
   * Coefficient gamma of the Ding-Horriche capillary momentum force
   * gamma * mu * grad(phi). The normalized surface tension sigma_tilde / eps is
   * used so that the tanh diffuse-interface energy integrates to the physical
   * surface tension sigma.
   */
  template <int dim>
  inline double
  ding_horriche_capillary_coefficient(const Parameters::CahnHilliard<dim> &param)
  {
    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;
    return sigma_tilde / param.epsilon_interface;
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
   * Return a pointer to the tracer limiter used when evaluating the degenerate
   * mobility M(phi). Optional and independent from the material-property
   * limiter above.
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
   * Surface-term coefficient for the static contact-angle (wetting) condition in
   * the potential equation. It matches the bulk gradient-term coefficient
   * (potential_gradient_coefficient) so the wetting boundary term scales
   * consistently with the volume Cahn-Hilliard energy, including the
   * model-dependent scaling (sigma_tilde * epsilon for Abels, eps^2 for
   * Ding-Horriche).
   */
  template <int dim>
  inline double
  contact_angle_surface_coefficient(const Parameters::CahnHilliard<dim> &param,
                                    const double sigma_tilde)
  {
    return potential_gradient_coefficient(param, sigma_tilde);
  }

  /**
   * Static wetting condition for a tanh diffuse interface:
   *
   *   n . grad(phi) = -cos(theta) (1 - phi^2) / (sqrt(2) eps),
   *
   * with theta the equilibrium contact angle measured through the phi = +1
   * phase. Returns the prescribed normal derivative g(phi).
   */
  inline double contact_angle_normal_derivative(const double phi,
                                                const double epsilon,
                                                const double theta)
  {
    return -std::cos(theta) * (1. - phi * phi) / (std::sqrt(2.) * epsilon);
  }

  /**
   * Derivative of contact_angle_normal_derivative w.r.t. the tracer phi.
   */
  inline double contact_angle_normal_derivative_jacobian(const double phi,
                                                         const double epsilon,
                                                         const double theta)
  {
    return 2. * std::cos(theta) * phi / (std::sqrt(2.) * epsilon);
  }

  // --- Mobility M(phi) ------------------------------------------------------
  // The mobility is either a constant or a parsed function of the tracer phi
  // (its single variable x is phi). The degenerate helpers evaluate the parsed
  // function, its first and second derivative at x = phi.

  template <int dim>
  using MobilityFunction = double (*)(const Parameters::CahnHilliard<dim> &,
                                      double);

  template <int dim>
  inline double constant_mobility(const Parameters::CahnHilliard<dim> &param,
                                  const double /*phi*/)
  {
    return param.mobility;
  }

  template <int dim>
  inline double degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                                    const double phi)
  {
    dealii::Point<dim> p;
    p[0] = phi;
    return param.degenerate_mobility->value(p);
  }

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
                               const double /*phi*/)
  {
    return 0.;
  }

  template <int dim>
  inline double
  degenerate_mobility_derivative(const Parameters::CahnHilliard<dim> &param,
                                 const double                         phi)
  {
    dealii::Point<dim> p;
    p[0] = phi;
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
  inline double
  constant_mobility_second_derivative(const Parameters::CahnHilliard<dim> &,
                                      const double /*phi*/)
  {
    return 0.;
  }

  template <int dim>
  inline double
  degenerate_mobility_second_derivative(const Parameters::CahnHilliard<dim> &param,
                                        const double phi)
  {
    dealii::Point<dim> p;
    p[0] = phi;
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

  // --- Material phase marker m(phi) -----------------------------------------
  // The material properties (density, viscosity) and the transported/conserved
  // variable are affine in a material marker m(phi):
  //   * abels / ding_horriche : m = phi          (m' = 1, m'' = 0)
  //   * abels_nlm             : m = q = tanh(k phi)/tanh(k)
  // For abels_nlm this sharpens the material transition (large k) while phi
  // keeps the capillary energy. The solved unknowns stay (u, p, phi, mu):
  //   - phi is the DOF (NOT q): q = s_k^{-1} inverse is atanh, undefined at
  //     q = +/-1, so phi would blow up in the bulk on any over/undershoot.
  //   - mu is the potential conjugate to q (mu_q), NOT mu_phi: the potential
  //     equation then reads s'_k(phi) mu = mu_phi(phi), i.e. s'_k always
  //     MULTIPLIES; taking mu_phi as the DOF would need mu_q = mu_phi / s'_k in
  //     the transport flux, i.e. a DIVISION by s'_k (which vanishes far from
  //     the interface) -> blow-up. See material_phase helpers below.

  // tanh mixing from val_a (marker = 1) to val_b (marker = -1); with
  // (val_a, val_b) = (1, -1) this returns q = tanh(k phi)/tanh(k).
  inline double tanh_mixing(const double phase_marker,
                            const double val_a,
                            const double val_b,
                            const double k)
  {
    const double tanh_k   = std::tanh(k);
    const double tanh_phi = std::tanh(k * phase_marker);
    return ((tanh_k + tanh_phi) * val_a + (tanh_k - tanh_phi) * val_b) /
           (2. * tanh_k);
  }

  // Derivative w.r.t. the tracer of tanh_mixing.
  inline double tanh_mixing_derivative(const double phase_marker,
                                       const double val_a,
                                       const double val_b,
                                       const double k)
  {
    const double tanh_k    = std::tanh(k);
    const double tanh_phi  = std::tanh(k * phase_marker);
    const double sech2_phi = 1. - tanh_phi * tanh_phi;
    return 0.5 * (val_a - val_b) * k / tanh_k * sech2_phi;
  }

  // Second derivative w.r.t. the tracer of tanh_mixing.
  inline double tanh_mixing_second_derivative(const double phase_marker,
                                              const double val_a,
                                              const double val_b,
                                              const double k)
  {
    const double tanh_k    = std::tanh(k);
    const double tanh_phi  = std::tanh(k * phase_marker);
    const double sech2_phi = 1. - tanh_phi * tanh_phi;
    return -(val_a - val_b) * k * k / tanh_k * sech2_phi * tanh_phi;
  }

  // Material marker m(phi) and its first two derivatives, as branchless
  // functions selected once by the dispatchers below (as for the mobility), so
  // the hot loops carry no per-quadrature-point model branch. The identity
  // branch (m = phi, m' = 1, m'' = 0) keeps every non-nlm model byte-neutral.
  template <int dim>
  using MaterialPhaseFunction = double (*)(const Parameters::CahnHilliard<dim> &,
                                           double);

  template <int dim>
  inline double
  material_phase_identity(const Parameters::CahnHilliard<dim> & /*param*/,
                          const double phi)
  {
    return phi;
  }

  template <int dim>
  inline double
  material_phase_identity_derivative(const Parameters::CahnHilliard<dim> &,
                                     const double /*phi*/)
  {
    return 1.;
  }

  template <int dim>
  inline double
  material_phase_identity_second_derivative(const Parameters::CahnHilliard<dim> &,
                                            const double /*phi*/)
  {
    return 0.;
  }

  template <int dim>
  inline double
  material_phase_tanh(const Parameters::CahnHilliard<dim> &param,
                      const double                         phi)
  {
    return tanh_mixing(phi, 1., -1., param.tanh_mixing_steepness);
  }

  template <int dim>
  inline double
  material_phase_tanh_derivative(const Parameters::CahnHilliard<dim> &param,
                                 const double                         phi)
  {
    return tanh_mixing_derivative(phi, 1., -1., param.tanh_mixing_steepness);
  }

  template <int dim>
  inline double
  material_phase_tanh_second_derivative(const Parameters::CahnHilliard<dim> &param,
                                        const double phi)
  {
    return tanh_mixing_second_derivative(
      phi, 1., -1., param.tanh_mixing_steepness);
  }

  template <int dim>
  MaterialPhaseFunction<dim>
  get_material_phase_function(const Parameters::CahnHilliard<dim> &param)
  {
    if (is_abels_nlm_model(param))
      return &material_phase_tanh<dim>;
    return &material_phase_identity<dim>;
  }

  template <int dim>
  MaterialPhaseFunction<dim> get_material_phase_derivative_function(
    const Parameters::CahnHilliard<dim> &param)
  {
    if (is_abels_nlm_model(param))
      return &material_phase_tanh_derivative<dim>;
    return &material_phase_identity_derivative<dim>;
  }

  template <int dim>
  MaterialPhaseFunction<dim> get_material_phase_second_derivative_function(
    const Parameters::CahnHilliard<dim> &param)
  {
    if (is_abels_nlm_model(param))
      return &material_phase_tanh_second_derivative<dim>;
    return &material_phase_identity_second_derivative<dim>;
  }
} // namespace CahnHilliard

#endif
