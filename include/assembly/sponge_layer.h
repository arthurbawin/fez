#ifndef ASSEMBLY_SPONGE_LAYER_H
#define ASSEMBLY_SPONGE_LAYER_H

#include <components_ordering.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <parameters.h>

#include <algorithm>

using namespace dealii;

namespace Assembly
{
  /**
   * Material constants needed by the compressible sponge-layer relaxation
   * source. The sponge contribution does not depend on viscosity or thermal
   * conductivity, only on the reference state of the EOS and the heat
   * capacity that scales the energy relaxation.
   */
  struct SpongeMaterialConstants
  {
    double rho_ref;
    double p_ref;
    double T_ref;
    double cp;
  };

  /**
   * Per-quadrature-point sponge state. sigma_q sums the inflow and outflow
   * band profiles; (u_inf, p_ref_q, T_ref_q) is the sigma-weighted average
   * of the two bands' reference states. This makes the relaxation source
   *   sigma_in*(phi - phi_ref_in) + sigma_out*(phi - phi_ref_out)
   * exact both when the bands are disjoint and when they overlap.
   */
  template <int dim>
  struct SpongeQuadratureState
  {
    double         sigma_q = 0.0;
    Tensor<1, dim> u_inf;
    double         p_ref_q = 0.0;
    double         T_ref_q = 0.0;
  };

  /**
   * Evaluate the sponge relaxation profile sigma(x) for a single band at a
   * quadrature point. Uses a quintic Hermite (smootherstep) ramp on
   * [x_start, x_end], scaled by sigma_max. The is_inflow flag mirrors the
   * profile so that sigma is sigma_max at x_start and 0 at x_end (inflow
   * band) instead of the default 0 at x_start and sigma_max at x_end
   * (outflow band). Returns 0 when the band is disabled.
   */
  template <int dim>
  double sponge_profile(const Point<dim>                    &x,
                        const Parameters::SpongeLayer::Band &band,
                        const bool                           is_inflow);

  /**
   * Compute the sigma_q and sigma-weighted reference state at a quadrature
   * point.
   */
  template <int dim>
  SpongeQuadratureState<dim>
  compute_sponge_state(const Point<dim>              &qp,
                       const Parameters::SpongeLayer &sponge);

  /**
   * Add the sponge-layer contribution to the local Jacobian. No-op when
   * the sponge is disabled. Templated on ScratchData to follow the pattern
   * used elsewhere in include/assembly/.
   */
  template <int dim, typename ScratchData>
  void sponge_layer_matrix(const ComponentOrdering       &ordering,
                           const Parameters::SpongeLayer &sponge,
                           const SpongeMaterialConstants &material,
                           const ScratchData             &scratch_data,
                           FullMatrix<double>            &local_matrix);

  /**
   * Add the sponge-layer contribution to the local residual. No-op when
   * the sponge is disabled.
   */
  template <int dim, typename ScratchData>
  void sponge_layer_rhs(const ComponentOrdering       &ordering,
                        const Parameters::SpongeLayer &sponge,
                        const SpongeMaterialConstants &material,
                        const ScratchData             &scratch_data,
                        Vector<double>                &local_rhs);
} // namespace Assembly

/* ---------------- Template definitions ----------------- */

template <int dim>
double Assembly::sponge_profile(const Point<dim>                    &x,
                                const Parameters::SpongeLayer::Band &band,
                                const bool                           is_inflow)
{
  if (!band.enable)
    return 0.0;

  const double r_raw = (x[0] - band.x_start) / (band.x_end - band.x_start);
  const double r     = std::min(std::max(r_raw, 0.0), 1.0);

  // Smootherstep S(r) = 10 r^3 - 15 r^4 + 6 r^5: S(0)=0, S(1)=1, with
  // vanishing first and second derivatives at both ends.
  const double S = r * r * r * (10.0 - 15.0 * r + 6.0 * r * r);

  return band.sigma_max * (is_inflow ? (1.0 - S) : S);
}

template <int dim>
Assembly::SpongeQuadratureState<dim>
Assembly::compute_sponge_state(const Point<dim>              &qp,
                               const Parameters::SpongeLayer &sponge)
{
  SpongeQuadratureState<dim> state;

  const double sigma_in  = sponge_profile(qp, sponge.inflow, true);
  const double sigma_out = sponge_profile(qp, sponge.outflow, false);
  state.sigma_q          = sigma_in + sigma_out;
  if (state.sigma_q > 0.0)
  {
    const double w_in  = sigma_in / state.sigma_q;
    const double w_out = sigma_out / state.sigma_q;
    state.u_inf[0]     = w_in * sponge.inflow.u + w_out * sponge.outflow.u;
    if constexpr (dim > 1)
      state.u_inf[1] = w_in * sponge.inflow.v + w_out * sponge.outflow.v;
    state.p_ref_q = w_in * sponge.inflow.p_ref + w_out * sponge.outflow.p_ref;
    state.T_ref_q = w_in * sponge.inflow.T_ref + w_out * sponge.outflow.T_ref;
  }

  return state;
}

template <int dim, typename ScratchData>
void Assembly::sponge_layer_matrix(const ComponentOrdering       &ordering,
                                   const Parameters::SpongeLayer &sponge,
                                   const SpongeMaterialConstants &material,
                                   const ScratchData             &scratch_data,
                                   FullMatrix<double>            &local_matrix)
{
  if (!sponge.any_enabled())
    return;

  const double rho_ref = material.rho_ref;
  const double cp      = material.cp;
  const double alpha_r = 1.0 / material.p_ref;
  const double beta_r  = 1.0 / material.T_ref;

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const SpongeQuadratureState<dim> st =
      compute_sponge_state<dim>(scratch_data.quadrature_points[q], sponge);
    if (st.sigma_q == 0.0)
      continue;

    const double JxW = scratch_data.JxW_moving[q];

    const double rho = scratch_data.density[q];

    const auto &phi_u = scratch_data.phi_u[q];
    const auto &phi_p = scratch_data.phi_p[q];
    const auto &phi_T = scratch_data.phi_T[q];

    const auto &p_val = scratch_data.present_pressure_values[q];
    const auto &T_val = scratch_data.present_temperature_values[q];
    const auto &u_val = scratch_data.present_velocity_values[q];

    const double a_denom    = alpha_r * p_val + 1.0;
    const double b_denom    = beta_r * T_val + 1.0;
    const double b_denom_sq = b_denom * b_denom;
    const double a_denom_sq = a_denom * a_denom;

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratch_data.components[i];
      const bool         i_is_u      = ordering.is_velocity(component_i);
      const bool         i_is_p      = ordering.is_pressure(component_i);
      const bool         i_is_T      = ordering.is_temperature(component_i);

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratch_data.components[j];
        const bool         j_is_u      = ordering.is_velocity(component_j);
        const bool         j_is_p      = ordering.is_pressure(component_j);
        const bool         j_is_T      = ordering.is_temperature(component_j);

        double m        = 0.0;
        bool   assemble = false;

        if (i_is_u && j_is_u)
        {
          assemble = true;
          m += phi_u[i] * rho * st.sigma_q * phi_u[j];
        }
        else if (i_is_u && j_is_p)
        {
          assemble = true;
          m += rho_ref * alpha_r * st.sigma_q / b_denom *
               ((u_val - st.u_inf) * phi_u[i]) * phi_p[j];
        }
        else if (i_is_u && j_is_T)
        {
          assemble = true;
          m += -rho_ref * beta_r * a_denom * st.sigma_q / b_denom_sq *
               ((u_val - st.u_inf) * phi_u[i]) * phi_T[j];
        }
        else if (i_is_p && j_is_p)
        {
          assemble = true;
          m += phi_p[i] * alpha_r * st.sigma_q * (1.0 + alpha_r * st.p_ref_q) /
               a_denom_sq * phi_p[j];
        }
        else if (i_is_p && j_is_T)
        {
          assemble = true;
          m += -phi_p[i] * beta_r * st.sigma_q * (1.0 + beta_r * st.T_ref_q) /
               b_denom_sq * phi_T[j];
        }
        else if (i_is_T && j_is_p)
        {
          assemble = true;
          m += phi_T[i] * rho_ref * alpha_r * cp * st.sigma_q *
               (T_val - st.T_ref_q) / b_denom * phi_p[j];
        }
        else if (i_is_T && j_is_T)
        {
          assemble = true;
          m += phi_T[i] * rho_ref * cp * st.sigma_q * a_denom *
               (1.0 + beta_r * st.T_ref_q) / b_denom_sq * phi_T[j];
        }

        if (assemble)
          local_matrix(i, j) += m * JxW;
      }
    }
  }
}

template <int dim, typename ScratchData>
void Assembly::sponge_layer_rhs(const ComponentOrdering       &ordering,
                                const Parameters::SpongeLayer &sponge,
                                const SpongeMaterialConstants &material,
                                const ScratchData             &scratch_data,
                                Vector<double>                &local_rhs)
{
  (void)ordering;

  if (!sponge.any_enabled())
    return;

  const double cp = material.cp;

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const SpongeQuadratureState<dim> st =
      compute_sponge_state<dim>(scratch_data.quadrature_points[q], sponge);
    if (st.sigma_q == 0.0)
      continue;

    const double JxW = scratch_data.JxW_moving[q];

    const double rho = scratch_data.density[q];
    const double a_p = scratch_data.a_p[q];
    const double b_T = scratch_data.b_T[q];

    const auto &phi_u = scratch_data.phi_u[q];
    const auto &phi_p = scratch_data.phi_p[q];
    const auto &phi_T = scratch_data.phi_T[q];

    const auto &p_val = scratch_data.present_pressure_values[q];
    const auto &T_val = scratch_data.present_temperature_values[q];
    const auto &u_val = scratch_data.present_velocity_values[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const double r =
        // Continuity
        a_p * st.sigma_q * (p_val - st.p_ref_q) * phi_p[i] -
        b_T * st.sigma_q * (T_val - st.T_ref_q) * phi_p[i]
        // Momentum
        + rho * st.sigma_q * (u_val - st.u_inf) * phi_u[i]
        // Energy
        + rho * cp * st.sigma_q * (T_val - st.T_ref_q) * phi_T[i];

      local_rhs(i) -= r * JxW;
    }
  }
}

#endif
