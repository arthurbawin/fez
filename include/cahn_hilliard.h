#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

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
} // namespace CahnHilliard

#endif
