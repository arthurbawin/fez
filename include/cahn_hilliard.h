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
    return param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::Abels ||
           param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::Sharp;
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
  is_sharp_material_diffuse_capillary_model(
    const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::Sharp;
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
    if (is_sharp_material_diffuse_capillary_model(param))
      return "Sharp";
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

  /*
   * Surface term coefficient for the static contact angle condition in the
   * potential equation. This follows the same model-dependent scaling as the
   * bulk gradient term.
   */
  template <int dim>
  inline double
  contact_angle_surface_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    return potential_gradient_coefficient(param, sigma_tilde);
  }

  /*
   * Static wetting condition for a tanh diffuse interface:
   *
   *   n . grad(phi) = -cos(theta) / (sqrt(2) eps) * (1 - phi^2)
   *
   * theta is measured through the phi = +1 phase.
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
   * Apply a tanh mixing from value A (when phase marker = 1) to value B (phase
   * marker = -1). This sharpens the material transition in the tails while
   * preserving the endpoint values.
   */
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

  /**
   * Derivative w.r.t. the tracer of the tanh mixing function.
   */
  inline double tanh_mixing_derivative(const double phase_marker,
                                       const double val_a,
                                       const double val_b,
                                       const double k)
  {
    const double tanh_k    = std::tanh(k);
    const double cosh_phi  = std::cosh(k * phase_marker);
    const double sech2_phi = 1. / (cosh_phi * cosh_phi);

    return 0.5 * (val_a - val_b) * k / tanh_k * sech2_phi;
  }

  /**
   * Second derivative w.r.t. the tracer of the tanh mixing function.
   */
  inline double tanh_mixing_second_derivative(const double phase_marker,
                                              const double val_a,
                                              const double val_b,
                                              const double k)
  {
    const double tanh_k    = std::tanh(k);
    const double tanh_phi  = std::tanh(k * phase_marker);
    const double cosh_phi  = std::cosh(k * phase_marker);
    const double sech2_phi = 1. / (cosh_phi * cosh_phi);

    return -(val_a - val_b) * k * k / tanh_k * sech2_phi * tanh_phi;
  }

  template <int dim>
  inline double material_phase_marker(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return tanh_mixing(
        phase_marker, 1., -1., param.tanh_mixing_steepness);

    return phase_marker;
  }

  template <int dim>
  inline double material_phase_derivative_wrt_tracer(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return tanh_mixing_derivative(
        phase_marker, 1., -1., param.tanh_mixing_steepness);

    return 1.;
  }

  template <int dim>
  inline double material_phase_second_derivative_wrt_tracer(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return tanh_mixing_second_derivative(
        phase_marker, 1., -1., param.tanh_mixing_steepness);

    return 0.;
  }

  template <int dim>
  inline double material_property_mixing(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker,
    const double                         val_a,
    const double                         val_b)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return linear_mixing(material_phase_marker(param, phase_marker),
                           val_a,
                           val_b);

    return linear_mixing(phase_marker, val_a, val_b);
  }

  template <int dim>
  inline double material_property_derivative_wrt_tracer(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker,
    const double                         val_a,
    const double                         val_b)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return linear_mixing_derivative(0., val_a, val_b) *
             material_phase_derivative_wrt_tracer(param, phase_marker);

    return linear_mixing_derivative(phase_marker, val_a, val_b);
  }

  template <int dim>
  inline double material_property_second_derivative_wrt_tracer(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker,
    const double                         val_a,
    const double                         val_b)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return linear_mixing_derivative(0., val_a, val_b) *
             material_phase_second_derivative_wrt_tracer(param, phase_marker);

    return 0.;
  }

  template <int dim>
  inline double material_property_derivative_wrt_material_phase(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker,
    const double                         val_a,
    const double                         val_b)
  {
    if (is_sharp_material_diffuse_capillary_model(param))
      return linear_mixing_derivative(0., val_a, val_b);

    return linear_mixing_derivative(phase_marker, val_a, val_b);
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
