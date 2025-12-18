#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>

/**
 * Apply linear mixing from value A (when phase marker = 1) to value B (phase
 * marker = -1).
 *
 */
inline double cahn_hilliard_linear_mixing(const double phase_marker,
                                          const double val_a,
                                          const double val_b)
{
  return 0.5 * ((val_a - val_b) * phase_marker + (val_a + val_b));
}

/**
 * Derivative w.r.t. the tracer of the linear mixing function
 *
 */
inline double
cahn_hilliard_linear_mixing_derivative(const double /*phase_marker*/,
                                       const double val_a,
                                       const double val_b)
{
  return 0.5 * (val_a - val_b);
}

// inline double
// cahn_hilliard_mixing(const Parameters::Multiphase::MixingModel &mixing_model,
//                      const double                               phase_marker,
//                      const double                               val_a,
//                      const double                               val_b)
// {
//   switch (mixing_model)
//   {
//     case Parameters::Multiphase::MixingModel::linear:
//       return cahn_hilliard_linear_mixing(phase_marker, val_a, val_b);
//   }
// }

#endif