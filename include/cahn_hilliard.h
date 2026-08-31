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
   * Return a pointer to the tracer limiter used when evaluating a
   * tracer-dependent mobility. Optional and independent from the
   * material-property limiter above.
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
  inline double degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                                    const double phi)
  {
    dealii::Point<dim> p;
    p[0] = phi;
    return param.degenerate_mobility->value(p);
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
  inline double
  degenerate_mobility_second_derivative(const Parameters::CahnHilliard<dim> &param,
                                        const double phi)
  {
    dealii::Point<dim> p;
    p[0] = phi;
    return param.degenerate_mobility->hessian(p)[0][0];
  }

  /** Mobility data at one quadrature point. The adaptive sensitivity is
   * dM_reg/d(u.grad(phi)) for gradient-dependent adaptive models and zero for
   * all other mobility models. */
  template <int dim>
  struct MobilityEvaluation
  {
    double value;
    double derivative_wrt_tracer;
    double second_derivative_wrt_tracer;
    double adaptive_sensitivity;
  };

  /** Input and chain-rule derivatives used by a mobility evaluator.
   * Degenerate mobility is a function of the material marker; adaptive
   * mobilities act on the transported tracer itself. */
  struct MobilityTracerArgument
  {
    double value;
    double first_derivative;
    double second_derivative;
  };

  template <int dim>
  inline MobilityTracerArgument
  select_mobility_tracer_argument(
    const Parameters::CahnHilliard<dim> &param,
    const double                         tracer,
    const double                         material_marker,
    const double                         material_marker_derivative,
    const double                         material_marker_second_derivative)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::degenerate)
      return {material_marker,
              material_marker_derivative,
              material_marker_second_derivative};
    return {tracer, 1., 0.};
  }

  template <int dim>
  using MobilityEvaluationFunction = MobilityEvaluation<dim> (*)
    (const Parameters::CahnHilliard<dim> &,
     double,
     double,
     double,
     const dealii::Tensor<1, dim> &,
     const dealii::Tensor<1, dim> &,
     double,
     double);

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_constant_mobility(const Parameters::CahnHilliard<dim> &param,
                             const double,
                             const double,
                             const double,
                             const dealii::Tensor<1, dim> &,
                             const dealii::Tensor<1, dim> &,
                             const double,
                             const double)
  {
    return {param.mobility, 0., 0., 0.};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                               const double                         phi,
                               const double                         phi_d,
                               const double                         phi_dd,
                               const dealii::Tensor<1, dim> &,
                               const dealii::Tensor<1, dim> &,
                               const double,
                               const double)
  {
    const double value = degenerate_mobility(param, phi);
    const double derivative = degenerate_mobility_derivative(param, phi);
    return {value,
            derivative * phi_d,
            degenerate_mobility_second_derivative(param, phi) * phi_d * phi_d +
              derivative * phi_dd,
            0.};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_adaptative_mobility(const Parameters::CahnHilliard<dim> &param,
                               const double,
                               const double,
                               const double,
                               const dealii::Tensor<1, dim> &velocity,
                               const dealii::Tensor<1, dim> &tracer_gradient,
                               const double adaptive_coefficient,
                               const double delta)
  {
    const double raw = adaptive_coefficient * (velocity * tracer_gradient);
    const double grad_phi_sq = tracer_gradient * tracer_gradient;
    const double extra_gradient_term =
      param.adaptive_mobility_m * 2. * param.epsilon_interface *
      param.epsilon_interface * grad_phi_sq;
    const double value = std::sqrt(raw * raw + delta * delta) +
                         extra_gradient_term;
    return {value, 0., 0., raw / value * adaptive_coefficient};
  }

  /** Value and first two tracer derivatives of the fixed tail weight used by
   * adaptative_mobility_2. */
  struct AdaptiveMobilityTailWeight
  {
    double value;
    double first_derivative;
    double second_derivative;
  };

  /** Extend the adaptive-mobility sensor from |phi|=0.9 to |phi|=0.999.
   *
   * In the resolved tail, W(phi)=(1-0.9^2)/(1-phi^2), which cancels the
   * decay of the gradient of an equilibrium tanh profile. Quintic smoothstep
   * blends make the unit core and bounded outer plateau C2-compatible.
   */
  inline AdaptiveMobilityTailWeight
  evaluate_adaptive_mobility_2_tail_weight(const double phi)
  {
    constexpr double phi_ref       = 0.9;
    constexpr double phi_ref_end   = 0.91;
    constexpr double phi_cap_start = 0.9989;
    constexpr double phi_cap       = 0.999;
    constexpr double r_ref         = phi_ref * phi_ref;
    constexpr double r_ref_end     = phi_ref_end * phi_ref_end;
    constexpr double r_cap_start   = phi_cap_start * phi_cap_start;
    constexpr double r_cap         = phi_cap * phi_cap;
    constexpr double q_ref         = 1. - r_ref;
    constexpr double q_cap         = 1. - r_cap;
    constexpr double weight_max    = q_ref / q_cap;

    const double r = phi * phi;
    if (r <= r_ref)
      return {1., 0., 0.};
    if (r >= r_cap)
      return {weight_max, 0., 0.};

    const double q                 = 1. - r;
    const double reciprocal        = q_ref / q;
    const double reciprocal_d_r    = q_ref / (q * q);
    const double reciprocal_dd_r2 = 2. * q_ref / (q * q * q);
    double       weight             = reciprocal;
    double       weight_d_r         = reciprocal_d_r;
    double       weight_dd_r2       = reciprocal_dd_r2;

    if (r < r_ref_end)
    {
      const double width = r_ref_end - r_ref;
      const double t     = (r - r_ref) / width;
      const double t2    = t * t;
      const double t3    = t2 * t;
      const double t4    = t3 * t;
      const double t5    = t4 * t;
      const double smoothstep = 6. * t5 - 15. * t4 + 10. * t3;
      const double smoothstep_d_t = 30. * t2 * (t - 1.) * (t - 1.);
      const double smoothstep_dd_t2 =
        60. * t * (2. * t2 - 3. * t + 1.);
      const double smoothstep_d_r = smoothstep_d_t / width;
      const double smoothstep_dd_r2 =
        smoothstep_dd_t2 / (width * width);

      weight = 1. + smoothstep * (reciprocal - 1.);
      weight_d_r = smoothstep_d_r * (reciprocal - 1.) +
                   smoothstep * reciprocal_d_r;
      weight_dd_r2 = smoothstep_dd_r2 * (reciprocal - 1.) +
                     2. * smoothstep_d_r * reciprocal_d_r +
                     smoothstep * reciprocal_dd_r2;
    }
    else if (r > r_cap_start)
    {
      const double width = r_cap - r_cap_start;
      const double t     = (r - r_cap_start) / width;
      const double t2    = t * t;
      const double t3    = t2 * t;
      const double t4    = t3 * t;
      const double t5    = t4 * t;
      const double smoothstep = 6. * t5 - 15. * t4 + 10. * t3;
      const double smoothstep_d_t = 30. * t2 * (t - 1.) * (t - 1.);
      const double smoothstep_dd_t2 =
        60. * t * (2. * t2 - 3. * t + 1.);
      const double smoothstep_d_r = smoothstep_d_t / width;
      const double smoothstep_dd_r2 =
        smoothstep_dd_t2 / (width * width);

      weight = (1. - smoothstep) * reciprocal +
               smoothstep * weight_max;
      weight_d_r = (1. - smoothstep) * reciprocal_d_r +
                   smoothstep_d_r * (weight_max - reciprocal);
      weight_dd_r2 = (1. - smoothstep) * reciprocal_dd_r2 -
                     2. * smoothstep_d_r * reciprocal_d_r +
                     smoothstep_dd_r2 * (weight_max - reciprocal);
    }

    return {weight,
            2. * phi * weight_d_r,
            2. * weight_d_r + 4. * r * weight_dd_r2};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_adaptative_mobility_2(const Parameters::CahnHilliard<dim> &,
                                 const double                         phi,
                                 const double                         phi_d,
                                 const double                         phi_dd,
                                 const dealii::Tensor<1, dim> &velocity,
                                 const dealii::Tensor<1, dim> &tracer_gradient,
                                 const double adaptive_coefficient,
                                 const double delta)
  {
    const auto weight = evaluate_adaptive_mobility_2_tail_weight(phi);
    const double weight_d = weight.first_derivative * phi_d;
    const double weight_dd = weight.second_derivative * phi_d * phi_d +
                             weight.first_derivative * phi_dd;
    const double velocity_dot_gradient = velocity * tracer_gradient;
    const double raw = adaptive_coefficient * weight.value *
                       velocity_dot_gradient;
    const double raw_d = adaptive_coefficient * weight_d *
                         velocity_dot_gradient;
    const double raw_dd = adaptive_coefficient * weight_dd *
                          velocity_dot_gradient;
    const double value = std::sqrt(raw * raw + delta * delta);
    const double derivative = raw * raw_d / value;
    const double second_derivative =
      (raw_d * raw_d + raw * raw_dd) / value -
      (raw * raw_d) * (raw * raw_d) / (value * value * value);
    const double sensitivity =
      raw / value * adaptive_coefficient * weight.value;
    return {value, derivative, second_derivative, sensitivity};
  }

  template <int dim>
  inline MobilityEvaluation<dim>
  evaluate_adaptative_mobility_3(const Parameters::CahnHilliard<dim> &,
                                 const double,
                                 const double,
                                 const double,
                                 const dealii::Tensor<1, dim> &velocity,
                                 const dealii::Tensor<1, dim> &,
                                 const double adaptive_coefficient,
                                 const double delta)
  {
    const double velocity_norm = std::sqrt(velocity * velocity + delta * delta);
    const double value = adaptive_coefficient * velocity_norm;
    return {value, 0., 0., 0.};
  }

  template <int dim>
  MobilityEvaluationFunction<dim> get_mobility_evaluation_function(
    const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &evaluate_constant_mobility<dim>;
    else if (param.mobility_model ==
             Parameters::CahnHilliard<dim>::MobilityModel::degenerate)
      return &evaluate_degenerate_mobility<dim>;
    else if (param.mobility_model ==
             Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_2)
      return &evaluate_adaptative_mobility_2<dim>;
    else if (param.mobility_model ==
             Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_3)
      return &evaluate_adaptative_mobility_3<dim>;
    else
      return &evaluate_adaptative_mobility<dim>;
  }

  struct AdaptiveMobilityScaling
  {
    double coefficient;
    double delta;
  };

  template <int dim>
  inline bool
  is_adaptive_mobility_model(const Parameters::CahnHilliard<dim> &param)
  {
    using MobilityModel =
      typename Parameters::CahnHilliard<dim>::MobilityModel;
    return param.mobility_model == MobilityModel::adaptive ||
           param.mobility_model == MobilityModel::adaptive_mobility_2 ||
           param.mobility_model == MobilityModel::adaptive_mobility_3;
  }

  template <int dim>
  inline AdaptiveMobilityScaling
  get_adaptive_mobility_scaling(const Parameters::CahnHilliard<dim> &param)
  {
    using MobilityModel =
      typename Parameters::CahnHilliard<dim>::MobilityModel;

    AssertThrow(is_adaptive_mobility_model(param),
                dealii::ExcMessage("Adaptive-mobility time adaptation requires "
                                   "an adaptive mobility model."));
    AssertThrow(param.surface_tension > 0.,
                dealii::ExcMessage("Adaptive mobility requires positive "
                                   "surface tension."));
    AssertThrow(param.epsilon_interface > 0.,
                dealii::ExcMessage("Adaptive mobility requires positive "
                                   "interface thickness."));

    const double epsilon = param.epsilon_interface;
    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;

    if (param.mobility_model == MobilityModel::adaptive)
      return {param.adaptive_mobility_n * std::sqrt(2.) * epsilon * epsilon *
                epsilon / sigma_tilde,
              param.adaptive_mobility_delta};
    if (param.mobility_model == MobilityModel::adaptive_mobility_2)
      return {param.adaptive_mobility_2_n * std::sqrt(2.) * epsilon * epsilon *
                epsilon / sigma_tilde,
              param.adaptive_mobility_2_delta};

    return {param.adaptive_mobility_3_n * epsilon * epsilon / sigma_tilde,
            param.adaptive_mobility_3_delta};
  }

  template <int dim>
  inline double compute_adaptive_mobility_number(
    const double                         timestep,
    const Parameters::CahnHilliard<dim> &param,
    const double                         max_mobility)
  {
    AssertThrow(is_adaptive_mobility_model(param),
                dealii::ExcMessage("Adaptive-mobility time adaptation requires "
                                   "an adaptive mobility model."));
    AssertThrow(std::isfinite(timestep) && timestep > 0.,
                dealii::ExcMessage("The time step must be finite and positive."));
    AssertThrow(std::isfinite(max_mobility) && max_mobility >= 0.,
                dealii::ExcMessage("The maximum mobility must be finite and "
                                   "non-negative."));
    AssertThrow(param.surface_tension > 0. && param.epsilon_interface > 0.,
                dealii::ExcMessage("Adaptive mobility requires positive "
                                   "surface tension and interface thickness."));

    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;
    const double epsilon = param.epsilon_interface;
    return timestep * sigma_tilde * max_mobility /
           (epsilon * epsilon * epsilon);
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
