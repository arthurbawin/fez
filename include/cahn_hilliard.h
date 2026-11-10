#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

#include <cmath>

namespace CahnHilliard
{
  template <int dim>
  inline bool
  is_quasi_incompressible_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::QuasiIncompressible;
  }

  /**
   * Normalized surface tension sigma_tilde = 3 sigma / (2 sqrt(2)), such that
   * the free energy of the tanh profile integrates to sigma across the
   * interface.
   */
  template <int dim>
  inline double
  sigma_tilde_from_surface_tension(const Parameters::CahnHilliard<dim> &param)
  {
    return 3. / (2. * std::sqrt(2.)) * param.surface_tension;
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
   * Derivative of tracer_limiter if @p enabled, 1 otherwise. At the two
   * corners phi = -1 and phi = 1, the derivative from inside [-1, 1] is used.
   */
  inline double tracer_limiter_derivative(const double phase_marker,
                                          const bool   enabled)
  {
    return enabled && (phase_marker < -1. || phase_marker > 1.) ? 0. : 1.;
  }

  /**
   * Return a pointer to the limiter function used for the phase field tracer.
   * The quasi-incompressible model always evaluates its positive density and
   * viscosity extensions on the clipped tracer.
   */
  using TracerLimiterFunction = double (*)(double);

  template <int dim>
  TracerLimiterFunction
  get_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.with_tracer_limiter || is_quasi_incompressible_model(param))
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
   * Mobility M(phi) and its first two derivatives with respect to phi.
   */
  struct MobilityEvaluation
  {
    double value;
    double first_derivative;
    double second_derivative;
  };

  /**
   * Evaluate the mobility and its derivatives at @p phase_marker:
   *
   * - constant:            M = mobility,
   * - degenerate_absolute: M = m0 epsilon^2 |1 - phi^2|.
   *
   * The degenerate mobility is non-negative for every phi, including
   * overshoots outside [-1, 1]. At its two kinks phi = +-1, the derivative
   * zero is chosen from the Clarke generalized gradient.
   */
  template <int dim>
  inline MobilityEvaluation
  evaluate_mobility(const Parameters::CahnHilliard<dim> &param,
                    const double                         phase_marker)
  {
    using MobilityModel = typename Parameters::CahnHilliard<dim>::MobilityModel;

    if (param.mobility_model == MobilityModel::degenerate_absolute)
    {
      const double amplitude = param.degenerate_absolute_mobility_prefactor *
                               param.epsilon_interface *
                               param.epsilon_interface;
      const double well = 1. - phase_marker * phase_marker;
      const double sign = well > 0. ? 1. : (well < 0. ? -1. : 0.);
      return {amplitude * std::abs(well),
              -2. * amplitude * phase_marker * sign,
              -2. * amplitude * sign};
    }

    return {param.mobility, 0., 0.};
  }
} // namespace CahnHilliard

#endif
