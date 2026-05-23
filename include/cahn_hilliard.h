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
