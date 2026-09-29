#ifndef CHNS_PHASE_DIFFUSION_H
#define CHNS_PHASE_DIFFUSION_H

#include <cahn_hilliard.h>

namespace CahnHilliard
{
  /** Local data for div(K_phi), or a direction in those data. With profile
   * correction, tracer and its derivatives are reconstructed fields. Mobility
   * and its spatial gradient retain the dependencies of the raw CH fields. */
  template <int dim>
  struct PhaseDiffusionPoint
  {
    double                 tracer             = 0.;
    dealii::Tensor<1, dim> tracer_gradient    = {};
    dealii::Tensor<2, dim> tracer_hessian     = {};
    dealii::Tensor<1, dim> potential_gradient = {};
    dealii::Tensor<2, dim> potential_hessian  = {};
    double                 mobility           = 0.;
    dealii::Tensor<1, dim> mobility_gradient  = {};
  };

  namespace PhaseDiffusionInternal
  {
    // Derivatives of a scalar coefficient with respect to phi and
    // s=|grad(phi)|^2. The squared norm avoids divisions by |grad(phi)|
    // in the profile driver, which is also defined at zero gradient.
    struct RadialCoefficient
    {
      double value   = 0.;
      double phi     = 0.;
      double s       = 0.;
      double phi_phi = 0.;
      double phi_s   = 0.;
      double s_s     = 0.;
    };

    inline double quintic_transition_second_derivative(const double value,
                                                       const double lower,
                                                       const double upper)
    {
      if (value <= lower || value >= upper)
        return 0.;
      const double width = upper - lower;
      const double t     = (value - lower) / width;
      return 60. * t * (2. * t * t - 3. * t + 1.) / (width * width);
    }

    inline double phase_activation_second_derivative(const double phi)
    {
      const double tail = profile_correction_tail_scale();
      const double s    = 1. - phi * phi;
      return -2. * quintic_transition_derivative(s, 0.25 * tail, 0.5 * tail) +
             4. * phi * phi *
               quintic_transition_second_derivative(s, 0.25 * tail, 0.5 * tail);
    }

    // F_profile = f(phi,s) grad(phi), with the same q-relative
    // regularization and phase activation as profile_correction_flux_driver.
    inline RadialCoefficient
    profile_coefficient(const double phi, const double s, const double epsilon)
    {
      const double a = profile_correction_phase_activation(phi);
      if (a == 0.)
        return {};
      const double ap     = profile_correction_phase_activation_derivative(phi);
      const double app    = phase_activation_second_derivative(phi);
      const double length = std::sqrt(2.) * epsilon;
      const double q      = (1. - phi * phi) / length;
      const double qp     = -2. * phi / length;
      const double qpp    = -2. / length;
      const double beta2  = profile_correction_beta * profile_correction_beta;
      const double c      = std::sqrt(1. + beta2);
      const double inverse_r  = 1. / std::sqrt(s + beta2 * q * q);
      const double inverse_r3 = inverse_r * inverse_r * inverse_r;
      const double inverse_r5 = inverse_r3 * inverse_r * inverse_r;
      const double factor     = 1. - c * q * inverse_r;
      return {a * factor,
              ap * factor - a * c * qp * s * inverse_r3,
              0.5 * a * c * q * inverse_r3,
              app * factor - 2. * ap * c * qp * s * inverse_r3 -
                a * c * qpp * s * inverse_r3 +
                3. * a * c * beta2 * q * qp * qp * s * inverse_r5,
              0.5 * ap * c * q * inverse_r3 -
                a * c * qp * (inverse_r3 - 1.5 * s * inverse_r5),
              -0.75 * a * c * q * inverse_r5};
    }

    // The weighted normal projector is I-b(phi,s) g tensor g, where
    // b=a(phi)^2*chi(sqrt(s))^2/s. Its inactive branch is identically zero,
    // so no division by a vanishing gradient is required.
    template <int dim>
    inline RadialCoefficient
    projection_coefficient(const Parameters::CahnHilliard<dim> &param,
                           const double                         phi,
                           const double                         s)
    {
      const double a = profile_correction_phase_activation(phi);
      if (a == 0.)
        return {};
      const double l   = std::sqrt(s);
      const double chi = flux_correction_gradient_activation(param, l);
      if (chi == 0.)
        return {};
      const double ap  = profile_correction_phase_activation_derivative(phi);
      const double app = phase_activation_second_derivative(phi);
      const double chip =
        flux_correction_gradient_activation_derivative(param, l);
      const double g0   = 1. / (std::sqrt(2.) * param.epsilon_interface);
      const double tail = profile_correction_tail_scale();
      const double chipp =
        quintic_transition_second_derivative(l,
                                             g0 * 0.25 * tail,
                                             g0 * 0.5 * tail);
      const double il        = 1. / l;
      const double il2       = il * il;
      const double il3       = il2 * il;
      const double il4       = il2 * il2;
      const double il5       = il4 * il;
      const double il6       = il3 * il3;
      const double radial    = chi * chi * il2;
      const double radial_s  = chi * chip * il3 - chi * chi * il4;
      const double radial_ss = 0.5 * (chip * chip + chi * chipp) * il4 -
                               2.5 * chi * chip * il5 + 2. * chi * chi * il6;
      return {a * a * radial,
              2. * a * ap * radial,
              a * a * radial_s,
              2. * (ap * ap + a * app) * radial,
              2. * a * ap * radial_s,
              a * a * radial_ss};
    }
  } // namespace PhaseDiffusionInternal

  /** Value and cached directional linearization of div(K_phi).
   * The reference mobility, hence the profile-correction coefficient, is
   * frozen. All local dependencies enter through PhaseDiffusionPoint. */
  template <int dim>
  class PhaseDiffusionLinearization
  {
  public:
    double value = 0.;

    PhaseDiffusionLinearization() = default;

    PhaseDiffusionLinearization(const Parameters::CahnHilliard<dim> &param,
                                const PhaseDiffusionPoint<dim>      &point,
                                const double reference_mobility)
      : point(point)
    {
      const auto &g = point.tracer_gradient;
      const auto &v = point.potential_gradient;
      const auto &h = point.tracer_hessian;
      const auto &t = point.potential_hessian;
      s             = g * g;
      trace_h       = dealii::trace(h);
      trace_t       = dealii::trace(t);
      g_h_g         = g * (h * g);
      g_v           = g * v;
      g_h_v         = g * (h * v);
      g_t_g         = g * (t * g);
      grad_m_g      = point.mobility_gradient * g;

      if (has_interface_profile_correction(param))
      {
        profile =
          PhaseDiffusionInternal::profile_coefficient(point.tracer,
                                                      s,
                                                      param.epsilon_interface);
        kappa = profile_correction_coefficient(param, reference_mobility);
        if (has_interface_flux_correction(param))
          projection =
            PhaseDiffusionInternal::projection_coefficient(param,
                                                           point.tracer,
                                                           s);
      }

      projection_gradient = projection.phi * s + 2. * projection.s * g_h_g +
                            projection.value * trace_h;
      projection_divergence =
        projection_gradient * g_v + projection.value * (g_h_v + g_t_g);
      const double profile_divergence =
        profile.value * trace_h + profile.phi * s + 2. * profile.s * g_h_g;
      value = point.mobility * trace_t + point.mobility_gradient * v -
              projection.value * g_v * grad_m_g -
              point.mobility * projection_divergence +
              kappa * profile_divergence;
    }

    double variation(const PhaseDiffusionPoint<dim> &direction) const
    {
      const auto  &g        = point.tracer_gradient;
      const auto  &v        = point.potential_gradient;
      const auto  &h        = point.tracer_hessian;
      const auto  &t        = point.potential_hessian;
      const auto  &dg       = direction.tracer_gradient;
      const auto  &dv       = direction.potential_gradient;
      const auto  &dh       = direction.tracer_hessian;
      const auto  &dt       = direction.potential_hessian;
      const double dphi     = direction.tracer;
      const double ds       = 2. * g * dg;
      const double dtrace_h = dealii::trace(dh);
      const double dtrace_t = dealii::trace(dt);
      const double dg_h_g   = dg * (h * g) + g * (dh * g) + g * (h * dg);
      const double dg_v     = dg * v + g * dv;
      const double dg_h_v   = dg * (h * v) + g * (dh * v) + g * (h * dv);
      const double dg_t_g   = dg * (t * g) + g * (dt * g) + g * (t * dg);
      const double dgrad_m_g =
        direction.mobility_gradient * g + point.mobility_gradient * dg;

      const double db     = projection.phi * dphi + projection.s * ds;
      const double db_phi = projection.phi_phi * dphi + projection.phi_s * ds;
      const double db_s   = projection.phi_s * dphi + projection.s_s * ds;
      const double dprojection_gradient =
        db_phi * s + projection.phi * ds + 2. * db_s * g_h_g +
        2. * projection.s * dg_h_g + db * trace_h + projection.value * dtrace_h;
      const double dprojection_divergence =
        dprojection_gradient * g_v + projection_gradient * dg_v +
        db * (g_h_v + g_t_g) + projection.value * (dg_h_v + dg_t_g);

      const double df     = profile.phi * dphi + profile.s * ds;
      const double df_phi = profile.phi_phi * dphi + profile.phi_s * ds;
      const double df_s   = profile.phi_s * dphi + profile.s_s * ds;
      const double dprofile_divergence =
        df * trace_h + profile.value * dtrace_h + df_phi * s +
        profile.phi * ds + 2. * df_s * g_h_g + 2. * profile.s * dg_h_g;

      return direction.mobility * trace_t + point.mobility * dtrace_t +
             direction.mobility_gradient * v + point.mobility_gradient * dv -
             db * g_v * grad_m_g - projection.value * dg_v * grad_m_g -
             projection.value * g_v * dgrad_m_g -
             direction.mobility * projection_divergence -
             point.mobility * dprojection_divergence +
             kappa * dprofile_divergence;
    }

  private:
    PhaseDiffusionPoint<dim>                  point;
    PhaseDiffusionInternal::RadialCoefficient profile, projection;
    double                                    kappa                 = 0.;
    double                                    s                     = 0.;
    double                                    trace_h               = 0.;
    double                                    trace_t               = 0.;
    double                                    g_h_g                 = 0.;
    double                                    g_v                   = 0.;
    double                                    g_h_v                 = 0.;
    double                                    g_t_g                 = 0.;
    double                                    grad_m_g              = 0.;
    double                                    projection_gradient   = 0.;
    double                                    projection_divergence = 0.;
  };

  template <int dim>
  inline PhaseDiffusionLinearization<dim> evaluate_phase_diffusion_divergence(
    const Parameters::CahnHilliard<dim> &param,
    const PhaseDiffusionPoint<dim>      &point,
    const double                         reference_mobility)
  {
    return {param, point, reference_mobility};
  }
} // namespace CahnHilliard

#endif
