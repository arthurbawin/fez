#ifndef ASSEMBLY_STABILIZATION_FORMS_H
#define ASSEMBLY_STABILIZATION_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

/**
 * SUPG/PSPG assembly contributions for incompressible NS and CHNS.
 *
 * All stabilization quantities (tau, strong residuals, diffusion_phi_u,
 * laplacian_shape_mu) are pre-computed by ScratchData::reinit*() and only
 * read here.
 *
 * Diffusion formulation:
 *   Laplacian form (plain NS, CHNS) : diffusion_phi_u = Δφ_u
 *   Divergence form (lambda solver) : diffusion_phi_u = Δφ_u + ∇(∇·φ_u)
 * The choice is encoded in scratch.diffusion_phi_u by ScratchData (driven by
 * enable_lagrange_multiplier); no switch parameter is needed here.
 *
 * Sign convention:  system_rhs = −G(u_h),  system_matrix = dG/du
 *   PSPG  RHS : += +τ R · ∇φ_p[i]
 *   SUPG  RHS : −= +τ (u·∇φ_i) · R
 *   Jacobian  : signs mirror RHS (PSPG −=, SUPG +=)
 *
 * τ linearisation w.r.t. u_j is omitted (quasi-Newton).
 *
 * Entry points:
 *   supg_pspg_rhs              — NS / NS_lambda / CHNS momentum RHS
 *   supg_pspg_matrix           — NS / NS_lambda momentum Jacobian
 *   supg_pspg_matrix_chns_full — CHNS momentum Jacobian
 *   supg_tracer_rhs            — CHNS tracer RHS
 *   supg_tracer_matrix_full    — CHNS tracer Jacobian
 */
namespace Assembly
{

  // ── NS / CHNS momentum : SUPG + PSPG — RHS ───────────────────────────────

  /**
   * RHS contribution. Call after standard terms, before *= JxW.
   * @param u_conv  u (Eulerian) or u − w_mesh (ALE).
   */
  template <int dim, typename ScratchData>
  inline void supg_pspg_rhs(const ComponentOrdering &ordering,
                            const unsigned int       q,
                            const unsigned int       i,
                            const double             tau,
                            const Tensor<1, dim>    &u_conv,
                            const ScratchData       &scratch,
                            double                  &local_rhs_i)
  {
    const unsigned int    comp_i = scratch.components[i];
    const Tensor<1, dim> &R      = scratch.strong_residual_momentum[q];

    if (ordering.is_pressure(comp_i))
      local_rhs_i += tau * R * scratch.grad_phi_p[q][i];

    if (ordering.is_velocity(comp_i))
      local_rhs_i -= tau * (scratch.grad_phi_u[q][i] * u_conv) * R;
  }

  // ── NS / NS_lambda momentum : SUPG + PSPG — Jacobian ─────────────────────

  /**
   * Jacobian contribution. Call after standard terms, before *= JxW.
   * Blocks: (i_p,j_u), (i_p,j_p), (i_u,j_u), (i_u,j_p).
   *
   * @param nu      ν (plain NS) or η/ρ (CHNS, passed by supg_pspg_matrix_chns_full).
   * @param u_conv  u (Eulerian) or u − w_mesh (ALE).
   * @param inv_rho 1/ρ (default 1 for single-fluid NS).
   */
  template <int dim, typename ScratchData>
  inline void supg_pspg_matrix(const ComponentOrdering &ordering,
                               const unsigned int       q,
                               const unsigned int       i,
                               const unsigned int       j,
                               const double             tau,
                               const double             nu,
                               const double             bdf_c0,
                               const Tensor<1, dim>    &u_conv,
                               const ScratchData       &scratch,
                               double                  &local_matrix_ij,
                               const double             inv_rho = 1.0)
  {
    const unsigned int    comp_i = scratch.components[i];
    const unsigned int    comp_j = scratch.components[j];
    const bool            i_is_u = ordering.is_velocity(comp_i);
    const bool            i_is_p = ordering.is_pressure(comp_i);
    const bool            j_is_u = ordering.is_velocity(comp_j);
    const bool            j_is_p = ordering.is_pressure(comp_j);
    const Tensor<1, dim> &R      = scratch.strong_residual_momentum[q];

    // dR/du_j  (diffusion_phi_u encodes Laplacian vs Divergence form)
    const auto dR_du_j = [&]() -> Tensor<1, dim> {
      return bdf_c0 * scratch.phi_u[q][j] + scratch.grad_phi_u[q][j] * u_conv +
             scratch.present_velocity_gradients[q] * scratch.phi_u[q][j] -
             nu * scratch.diffusion_phi_u[q][j];
    };

    if (i_is_p && j_is_u)
      local_matrix_ij -= tau * dR_du_j() * scratch.grad_phi_p[q][i];

    if (i_is_p && j_is_p)
      local_matrix_ij -=
        tau * (inv_rho * scratch.grad_phi_p[q][j]) * scratch.grad_phi_p[q][i];

    if (i_is_u)
    {
      const Tensor<1, dim> supg_test_i = scratch.grad_phi_u[q][i] * u_conv;

      if (j_is_u)
      {
        local_matrix_ij +=
          tau * (scratch.grad_phi_u[q][i] * scratch.phi_u[q][j]) * R;
        local_matrix_ij += tau * supg_test_i * dR_du_j();
      }
      if (j_is_p)
        local_matrix_ij +=
          tau * supg_test_i * (inv_rho * scratch.grad_phi_p[q][j]);
    }
  }

  // ── CHNS tracer : SUPG — RHS ──────────────────────────────────────────────

  /**
   * RHS contribution. Call after standard terms, before *= JxW.
   */
  template <int dim, typename ScratchData>
  inline void supg_tracer_rhs(const ComponentOrdering &ordering,
                              const unsigned int       q,
                              const unsigned int       i,
                              const double             tau_tracer,
                              const Tensor<1, dim>    &u_conv,
                              const ScratchData       &scratch,
                              double                  &local_rhs_i)
  {
    if (!ordering.is_tracer(scratch.components[i]))
      return;

    local_rhs_i -= tau_tracer * (u_conv * scratch.grad_shape_phi[q][i]) *
                   scratch.strong_residual_tracer[q];
  }

  // ── CHNS momentum : SUPG + PSPG — Jacobian, all column blocks ────────────
  //
  // Handles all column types (j_u, j_p, j_φ, j_μ, j_x) in one pass.
  // with_moving_mesh must be explicit: supg_pspg_matrix_chns_full<dim,
  // with_moving_mesh>(...).
  //
  // j_u / j_p  — plain-NS blocks (via supg_pspg_matrix) plus CHNS-specific
  //              extra dR/du_j terms:
  //                + inv_rho · M·(ρ1-ρ0)/2 · (∇φ_j · ∇μ)   [J-flux]
  //                − 2·inv_rho·(dη/dφ) · ∇φ · sym(∇φ_j)     [variable
  //                viscosity]
  //
  // j_φ  — dR/dφ_j, with linear mixing (d²η/dφ² = d²ρ/dφ² = 0):
  //           φ_j · { d(1/ρ)/dφ·[...] + (1/ρ)·∇μ − d(ν_eff)/dφ·(Δu+∇div u) −
  //           ... } − 2·(1/ρ)·(dη/dφ) · ε(u) · ∇φ_j
  //
  // j_μ  — dR/dμ_j = inv_rho · (M·(ρ1-ρ0)/2 · (∇u) + φ·Id) · ∇φ_μ_j
  //
  // j_x  — ALE geometry variation dR/dx_j (only when with_moving_mesh)
  template <int dim, bool with_moving_mesh, typename ScratchData>
  inline void supg_pspg_matrix_chns_full(const ComponentOrdering &ordering,
                                         const unsigned int       q,
                                         const unsigned int       i,
                                         const unsigned int       j,
                                         const double             tau,
                                         const double             nu_eff,
                                         const double             bdf_c0,
                                         const Tensor<1, dim>    &u_conv,
                                         const double             inv_rho,
                                         const ScratchData       &scratch,
                                         double &local_matrix_ij)
  {
    const unsigned int comp_i = scratch.components[i];
    const bool         i_is_u = ordering.is_velocity(comp_i);
    const bool         i_is_p = ordering.is_pressure(comp_i);
    if (!i_is_u && !i_is_p)
      return;

    const unsigned int comp_j   = scratch.components[j];
    const bool         j_is_u   = ordering.is_velocity(comp_j);
    const bool         j_is_phi = ordering.is_tracer(comp_j);
    const bool         j_is_mu  = ordering.is_potential(comp_j);

    // ── j_u / j_p : plain-NS blocks + CHNS extra dR/du_j ───────────────────
    if (j_is_u || ordering.is_pressure(comp_j))
    {
      // plain-NS blocks; diffusion_phi_u is divergence form for CHNS
      supg_pspg_matrix<dim>(ordering,
                            q,
                            i,
                            j,
                            tau,
                            nu_eff,
                            bdf_c0,
                            u_conv,
                            scratch,
                            local_matrix_ij,
                            inv_rho);

      if (j_is_u)
      {
        // CHNS-specific extra residual contributions from variable ρ, η, and
        // J-flux
        const Tensor<1, dim> extra_dR =
          inv_rho * scratch.diffusive_flux_factor *
            (scratch.grad_phi_u[q][j] * scratch.potential_gradients[q]) -
          2. * inv_rho * scratch.derivative_dynamic_viscosity_wrt_tracer[q] *
            Tensor<2, dim>(symmetrize(scratch.grad_phi_u[q][j])) *
            scratch.tracer_gradients[q];

        if (i_is_p)
          local_matrix_ij -= tau * extra_dR * scratch.grad_phi_p[q][i];
        if (i_is_u)
          local_matrix_ij +=
            tau * (scratch.grad_phi_u[q][i] * u_conv) * extra_dR;
      }
    }

    // ── j_φ : dR/dφ_j ───────────────────────────────────────────────────────
    if (j_is_phi)
    {
      const double drho_dphi = scratch.derivative_density_wrt_tracer[q];
      const double deta_dphi =
        scratch.derivative_dynamic_viscosity_wrt_tracer[q];
      const double phi_j        = scratch.shape_phi[q][j];
      const double dinvrho_dphi = -inv_rho * inv_rho * drho_dphi;
      const double dnueff_dphi =
        inv_rho * deta_dphi - nu_eff * inv_rho * drho_dphi;

      const Tensor<1, dim> &grad_phi = scratch.tracer_gradients[q];
      const Tensor<1, dim> &grad_mu  = scratch.potential_gradients[q];
      const Tensor<2, dim>  eps_u =
        Tensor<2, dim>(scratch.present_velocity_sym_gradients[q]);

      const Tensor<1, dim> dR_dphi_j =
        phi_j *
          (dinvrho_dphi * (scratch.diffusive_flux_factor *
                             (scratch.present_velocity_gradients[q] * grad_mu) +
                           scratch.tracer_values[q] * grad_mu +
                           scratch.present_pressure_gradients[q] +
                           scratch.source_term_velocity[q]) +
           inv_rho * grad_mu -
           dnueff_dphi * scratch.present_velocity_lap_plus_graddiv[q] -
           2. * dinvrho_dphi * deta_dphi * eps_u * grad_phi) -
        2. * inv_rho * deta_dphi * eps_u * scratch.grad_shape_phi[q][j];

      if (i_is_p)
        local_matrix_ij -= tau * dR_dphi_j * scratch.grad_phi_p[q][i];
      if (i_is_u)
        // ∂(supg_test)/∂φ_j = 0 since supg_test = ∇φ_u[i]·u_conv does not
        // depend on φ
        local_matrix_ij +=
          tau * (scratch.grad_phi_u[q][i] * u_conv) * dR_dphi_j;
    }

    // ── j_μ : dR/dμ_j ───────────────────────────────────────────────────────
    if (j_is_mu)
    {
      const Tensor<1, dim> &grad_mu_j = scratch.grad_shape_mu[q][j];
      const Tensor<1, dim>  dR_dmu_j =
        inv_rho * (scratch.diffusive_flux_factor *
                     (scratch.present_velocity_gradients[q] * grad_mu_j) +
                   scratch.tracer_values[q] * grad_mu_j);

      if (i_is_p)
        local_matrix_ij -= tau * dR_dmu_j * scratch.grad_phi_p[q][i];
      if (i_is_u)
        local_matrix_ij += tau * (scratch.grad_phi_u[q][i] * u_conv) * dR_dmu_j;
    }

    // ── j_x : ALE geometry variation dR/dx_j ────────────────────────────────
    if constexpr (with_moving_mesh)
    {
      if (ordering.is_position(comp_j))
      {
        const Tensor<2, dim> &G   = scratch.grad_phi_x_moving[q][j];
        const Tensor<2, dim>  GT  = transpose(G);
        const double          trG = trace(G);

        const Tensor<1, dim> du_conv_dx_j = -bdf_c0 * scratch.phi_x[q][j];
        const Tensor<2, dim> dgrad_u_dx_j =
          -scratch.present_velocity_gradients[q] * G;
        const Tensor<1, dim> dgrad_p_dx_j =
          -GT * scratch.present_pressure_gradients[q];
        const Tensor<1, dim> dgrad_mu_dx_j =
          -GT * scratch.potential_gradients[q];

        const Tensor<1, dim> dR_dx_j =
          dgrad_u_dx_j * u_conv +
          scratch.present_velocity_gradients[q] * du_conv_dx_j +
          inv_rho * dgrad_p_dx_j +
          inv_rho * scratch.tracer_values[q] * dgrad_mu_dx_j;

        const Tensor<1, dim> &R = scratch.strong_residual_momentum[q];

        if (i_is_p)
        {
          const Tensor<1, dim> dgrad_phi_p_i_dx_j =
            -GT * scratch.grad_phi_p[q][i];
          local_matrix_ij -= tau * dR_dx_j * scratch.grad_phi_p[q][i];
          local_matrix_ij -= tau * R * dgrad_phi_p_i_dx_j;
          local_matrix_ij -= tau * R * scratch.grad_phi_p[q][i] * trG;
        }

        if (i_is_u)
        {
          const Tensor<2, dim> dgrad_phi_u_i_dx_j =
            -scratch.grad_phi_u[q][i] * G;
          const Tensor<1, dim> d_supg_test_dx_j =
            dgrad_phi_u_i_dx_j * u_conv +
            scratch.grad_phi_u[q][i] * du_conv_dx_j;

          local_matrix_ij +=
            tau * (scratch.grad_phi_u[q][i] * u_conv) * dR_dx_j;
          local_matrix_ij += tau * d_supg_test_dx_j * R;
          local_matrix_ij +=
            tau * (scratch.grad_phi_u[q][i] * u_conv) * R * trG;
        }
      }
    }
  }

  // ── CHNS tracer : SUPG — Jacobian, all column blocks ─────────────────────
  //
  // Handles all column types in one pass:
  //   j_u  — linearisation of u_conv·∇φ w.r.t. u_j
  //   j_φ  — linearisation of ∂φ/∂t + u_conv·∇φ w.r.t. φ_j
  //   j_μ  — linearisation of −M·Δμ w.r.t. μ_j
  //   j_x  — ALE geometry variation (only when with_moving_mesh)
  template <int dim, bool with_moving_mesh, typename ScratchData>
  inline void supg_tracer_matrix_full(const ComponentOrdering &ordering,
                                      const unsigned int       q,
                                      const unsigned int       i,
                                      const unsigned int       j,
                                      const double             tau_tracer,
                                      const double             bdf_c0,
                                      const Tensor<1, dim>    &u_conv,
                                      const ScratchData       &scratch,
                                      double                  &local_matrix_ij)
  {
    if (!ordering.is_tracer(scratch.components[i]))
      return;

    const double       R_tracer    = scratch.strong_residual_tracer[q];
    const double       supg_test_i = u_conv * scratch.grad_shape_phi[q][i];
    const unsigned int comp_j      = scratch.components[j];

    // ── j_u ─────────────────────────────────────────────────────────────────
    if (ordering.is_velocity(comp_j))
    {
      local_matrix_ij += tau_tracer *
                         (scratch.phi_u[q][j] * scratch.grad_shape_phi[q][i]) *
                         R_tracer;
      local_matrix_ij += tau_tracer * supg_test_i *
                         (scratch.phi_u[q][j] * scratch.tracer_gradients[q]);
    }

    // ── j_φ ─────────────────────────────────────────────────────────────────
    if (ordering.is_tracer(comp_j))
    {
      const double dR_dphi_j = bdf_c0 * scratch.shape_phi[q][j] +
                               u_conv * scratch.grad_shape_phi[q][j];
      local_matrix_ij += tau_tracer * supg_test_i * dR_dphi_j;
    }

    // ── j_μ ─────────────────────────────────────────────────────────────────
    if (ordering.is_potential(comp_j))
      local_matrix_ij += tau_tracer * supg_test_i *
                         (-scratch.mobility * scratch.laplacian_shape_mu[q][j]);

    // ── j_x : ALE geometry variation ────────────────────────────────────────
    if constexpr (with_moving_mesh)
    {
      if (ordering.is_position(comp_j))
      {
        const Tensor<2, dim> &G   = scratch.grad_phi_x_moving[q][j];
        const Tensor<2, dim>  GT  = transpose(G);
        const double          trG = trace(G);

        const Tensor<1, dim> du_conv_dx_j   = -bdf_c0 * scratch.phi_x[q][j];
        const Tensor<1, dim> dgrad_phi_dx_j = -GT * scratch.tracer_gradients[q];

        const double dR_dx_j =
          du_conv_dx_j * scratch.tracer_gradients[q] + u_conv * dgrad_phi_dx_j;

        const Tensor<1, dim> dgrad_shape_phi_i_dx_j =
          -GT * scratch.grad_shape_phi[q][i];
        const double d_supg_test_dx_j =
          du_conv_dx_j * scratch.grad_shape_phi[q][i] +
          u_conv * dgrad_shape_phi_i_dx_j;

        local_matrix_ij += tau_tracer * supg_test_i * dR_dx_j;
        local_matrix_ij += tau_tracer * d_supg_test_dx_j * R_tracer;
        local_matrix_ij += tau_tracer * supg_test_i * R_tracer * trG;
      }
    }
  }

} // namespace Assembly

#endif
