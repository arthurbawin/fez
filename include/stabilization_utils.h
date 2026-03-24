#ifndef STABILIZATION_UTILS_H
#define STABILIZATION_UTILS_H

#include <deal.II/base/tensor.h>

#include <cmath>
#include <limits>
#include <vector>

/**
 * Pre-computation utilities for SUPG/PSPG stabilization.
 * Called exclusively from ScratchData::reinit*(). Not part of the assembly
 * layer — see assembly/stabilization_forms.h for the actual contributions.
 */
namespace Stabilization
{
  using namespace dealii;

  template <int dim>
  inline double
  compute_streamline_length(const Tensor<1, dim>              &advection_velocity,
                            const std::vector<Tensor<2, dim>> &grad_phi_u,
                            const std::vector<unsigned int>   &components,
                            const unsigned int                 velocity_component,
                            const double                       fallback_length)
  {
    const double norm = advection_velocity.norm();
    if (norm <= std::numeric_limits<double>::epsilon())
      return fallback_length;

    const Tensor<1, dim> s = advection_velocity / norm;
    double denominator     = 0.;
    for (unsigned int k = 0; k < grad_phi_u.size(); ++k)
      if (components[k] == velocity_component)
        {
          const double proj = s * grad_phi_u[k][velocity_component];
          denominator += proj * proj;
        }

    return (denominator <= std::numeric_limits<double>::epsilon()) ?
             fallback_length :
             std::sqrt(2. / denominator);
  }

  inline double compute_tau(const double dt,
                            const bool   is_steady,
                            const double advection_norm,
                            const double diffusivity,
                            const double h_tau)
  {
    if (h_tau <= std::numeric_limits<double>::epsilon())
      return 0.;

    const double temporal  = (is_steady || dt <= std::numeric_limits<double>::epsilon()) ?
                               0. : 4. / (dt * dt);
    const double convective = 4. * advection_norm * advection_norm / (h_tau * h_tau);
    const double diffusive  = 144. * diffusivity * diffusivity / std::pow(h_tau, 4);
    const double denom      = std::sqrt(temporal + convective + diffusive);

    return (denom <= std::numeric_limits<double>::epsilon()) ? 0. : 1. / denom;
  }

}

#endif