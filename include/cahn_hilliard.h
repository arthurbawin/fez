#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

#include <cmath>

namespace CahnHilliard
{
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
   * sigma_tilde * epsilon, so the wetting boundary term scales consistently with
   * the volume Cahn-Hilliard energy. (When model switching is added this becomes
   * model-dependent, like the bulk gradient term.)
   */
  inline double contact_angle_surface_coefficient(const double sigma_tilde,
                                                  const double epsilon)
  {
    return sigma_tilde * epsilon;
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
} // namespace CahnHilliard

#endif
