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
 * A single supg_pspg_rhs / supg_pspg_matrix pair handles both plain NS and
 * CHNS momentum: the only difference is the strong residual stored in
 * scratch.strong_residual_momentum[q], which is set appropriately by the
 * corresponding reinit layer.
 *
 * Sign convention:  system_rhs = −G(u_h),  system_matrix = dG/du
 *   PSPG  RHS : += +τ R · ∇φ_p[i]
 *   SUPG  RHS : −= +τ (u·∇φ_i) · R
 *   Jacobian  : signs mirror RHS (PSPG −=, SUPG +=)
 *
 * τ linearisation w.r.t. u_j is omitted (quasi-Newton).
 */
namespace Assembly
{

  // ── NS / CHNS momentum : SUPG + PSPG ─────────────────────────────────────

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

  /**
   * Jacobian contribution. Call after standard terms, before *= JxW.
   * Blocks: (i_p,j_u), (i_p,j_p), (i_u,j_u), (i_u,j_p).
   *
   * @param nu      η/ρ (CHNS) or ν (plain NS).
   * @param u_conv  u (Eulerian) or u − w_mesh (ALE).
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
      return bdf_c0 * scratch.phi_u[q][j] +
             scratch.grad_phi_u[q][j] * u_conv +
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
        local_matrix_ij += tau * supg_test_i * (inv_rho * scratch.grad_phi_p[q][j]); // <-- 3. Ajout inv_rho
    }
  }

  // ── CHNS momentum : SUPG + PSPG — all (i,j) blocks ──────────────────────
  //
  // Extends supg_pspg_matrix to the full CHNS momentum residual.
  // Handles every column type (u, p, …) by delegating to supg_pspg_matrix,
  // then adds the two CHNS-specific dR/du_j terms for j_u columns:
  //   + inv_rho · diffusive_flux_factor · (∇φ_j · ∇μ)   [J-flux / ∇u term]
  //   − 2·inv_rho·(dη/dφ) · ∇φ · sym(∇φ_j)              [variable viscosity]
  //
  // @param nu_eff   η/ρ (variable, CHNS effective kinematic viscosity).
  // @param inv_rho  1/ρ (with density floor applied by the caller).
  // The CHNS-specific material data (diffusive_flux_factor, deta/dφ, ∇φ, ∇μ)
  // are read directly from scratch.
  template <int dim, typename ScratchData>
  inline void supg_pspg_matrix_chns(const ComponentOrdering &ordering,
                                    const unsigned int       q,
                                    const unsigned int       i,
                                    const unsigned int       j,
                                    const double             tau,
                                    const double             nu_eff,
                                    const double             bdf_c0,
                                    const Tensor<1, dim>    &u_conv,
                                    const double             inv_rho,
                                    const ScratchData       &scratch,
                                    double                  &local_matrix_ij)
  {
    // ── 1. Plain-NS blocks (p-p, p-u, u-u, u-p) with nu_eff. ───────────────
    // diffusion_phi_u already encodes the divergence form for CHNS.
    supg_pspg_matrix<dim>(ordering, q, i, j, tau, nu_eff, bdf_c0, u_conv,
                          scratch, local_matrix_ij, inv_rho);

    // ── 2. CHNS extra terms: only non-zero for j_u columns. ─────────────────
    if (!ordering.is_velocity(scratch.components[j]))
      return;

    const unsigned int comp_i = scratch.components[i];
    const bool         i_is_u = ordering.is_velocity(comp_i);
    const bool         i_is_p = ordering.is_pressure(comp_i);
    if (!i_is_u && !i_is_p)
      return;

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
 
  // ── CHNS momentum : SUPG + PSPG — blocks (i_u,j_φ) and (i_p,j_φ) ────────
  //
  // Full dR/dφ_j for the CHNS momentum residual R (scaled by 1/ρ).
  // φ_j = shape_phi[q][j]  (scalar, from scratch).
  //
  // With linear mixing (η and ρ linear in φ), d²η/dφ² = d²ρ/dφ² = 0, so:
  //
  //   dR/dφ_j = φ_j · {
  //     d(inv_rho)/dφ · [M·(ρ1-ρ0)/2 · (∇u)·∇μ + φ·∇μ + ∇p + f_src]
  //     + inv_rho · ∇μ
  //     − d(ν_eff)/dφ · (Δu + ∇div u)
  //     − 2 · d(inv_rho)/dφ · (dη/dφ) · ∇φ · ε(u)
  //   }
  //   − 2·inv_rho·(dη/dφ) · ε(u) · ∇φ_j          [gradient-of-φ term]
  //
  // where:
  //   d(inv_rho)/dφ  = −inv_rho² · drho_dphi
  //   d(ν_eff)/dφ    = inv_rho·deta_dphi − nu_eff·inv_rho·drho_dphi
  //
  // All material data (∇φ, ∇μ, ∇p, f_u, Δu+∇div u, ε(u), M·(ρ1-ρ0)/2)
  // are read directly from scratch.
  //
  // @param inv_rho  1/ρ (with density floor applied by the caller).
  template <int dim, typename ScratchData>
  inline void supg_pspg_matrix_chns_phi(const ComponentOrdering &ordering,
                                        const unsigned int       q,
                                        const unsigned int       i,
                                        const unsigned int       j,
                                        const double             tau,
                                        const double             nu_eff,
                                        const Tensor<1, dim>    &u_conv,
                                        const double             inv_rho,
                                        const ScratchData       &scratch,
                                        double                  &local_matrix_ij)
  {
    if (!ordering.is_tracer(scratch.components[j]))
      return;

    const unsigned int comp_i = scratch.components[i];
    const bool         i_is_u = ordering.is_velocity(comp_i);
    const bool         i_is_p = ordering.is_pressure(comp_i);
    if (!i_is_u && !i_is_p)
      return;

    const double drho_dphi    = scratch.derivative_density_wrt_tracer[q];
    const double deta_dphi    = scratch.derivative_dynamic_viscosity_wrt_tracer[q];
    const double phi_j        = scratch.shape_phi[q][j];
    const double dinvrho_dphi = -inv_rho * inv_rho * drho_dphi;
    const double dnueff_dphi  = inv_rho * deta_dphi - nu_eff * inv_rho * drho_dphi;

    const Tensor<1, dim> &grad_phi   = scratch.tracer_gradients[q];
    const Tensor<1, dim> &grad_mu    = scratch.potential_gradients[q];
    const Tensor<2, dim>  eps_u      = Tensor<2, dim>(scratch.present_velocity_sym_gradients[q]);

    // dR/dφ_j as Tensor<1,dim>:
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
      // Note: ∂(supg_test)/∂φ_j = 0 since supg_test = ∇φ_u[i]·u_conv
      // does not depend on φ — only dR/dφ_j contributes here.
      local_matrix_ij +=
        tau * (scratch.grad_phi_u[q][i] * u_conv) * dR_dphi_j;
  }
 
  // ── CHNS momentum : SUPG + PSPG — blocks (i_u,j_μ) and (i_p,j_μ) ────────
  //
  // Full dR/dμ_j for the CHNS momentum residual R (scaled by 1/ρ).
  // ∇φ_μ_j = grad_shape_mu[q][j]  (Tensor<1,dim>, from scratch).
  //
  //   dR/dμ_j = inv_rho · (diffusive_flux_factor · (∇u) + φ·Id) · ∇φ_μ_j
  //
  // Material data (diffusive_flux_factor, φ) are read from scratch.
  //
  // @param inv_rho  1/ρ (with density floor applied by the caller).
  template <int dim, typename ScratchData>
  inline void supg_pspg_matrix_chns_mu(const ComponentOrdering &ordering,
                                       const unsigned int       q,
                                       const unsigned int       i,
                                       const unsigned int       j,
                                       const double             tau,
                                       const Tensor<1, dim>    &u_conv,
                                       const double             inv_rho,
                                       const ScratchData       &scratch,
                                       double                  &local_matrix_ij)
  {
    if (!ordering.is_potential(scratch.components[j]))
      return;

    const unsigned int comp_i = scratch.components[i];
    const bool         i_is_u = ordering.is_velocity(comp_i);
    const bool         i_is_p = ordering.is_pressure(comp_i);
    if (!i_is_u && !i_is_p)
      return;

    // dR/dμ_j as Tensor<1,dim>:
    const Tensor<1, dim> &grad_mu_j = scratch.grad_shape_mu[q][j];
    const Tensor<1, dim>  dR_dmu_j  =
      inv_rho * (scratch.diffusive_flux_factor *
                   (scratch.present_velocity_gradients[q] * grad_mu_j) +
                 scratch.tracer_values[q] * grad_mu_j);

    if (i_is_p)
      local_matrix_ij -= tau * dR_dmu_j * scratch.grad_phi_p[q][i];
    if (i_is_u)
      local_matrix_ij +=
        tau * (scratch.grad_phi_u[q][i] * u_conv) * dR_dmu_j;
  }
 
  // ── CHNS tracer : SUPG ────────────────────────────────────────────────────
  //
  // Tracer equation: ∂φ/∂t + u_conv·∇φ − M·Δμ + f_φ = 0
  // Linearised blocks: (i_φ, j_u), (i_φ, j_φ), (i_φ, j_μ).
 
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
 
    local_rhs_i -= tau_tracer *
                   (u_conv * scratch.grad_shape_phi[q][i]) *
                   scratch.strong_residual_tracer[q];
  }
 
  // ── CHNS tracer : SUPG — Jacobian ─────────────────────────────────────────
  //
  // Linearised blocks (i_φ, j_u), (i_φ, j_φ), (i_φ, j_μ).
  // Mobility and ∇φ are read directly from scratch.
  template <int dim, typename ScratchData>
  inline void supg_tracer_matrix(const ComponentOrdering &ordering,
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

    if (ordering.is_velocity(comp_j))
      {
        local_matrix_ij += tau_tracer *
                           (scratch.phi_u[q][j] * scratch.grad_shape_phi[q][i]) *
                           R_tracer;
        local_matrix_ij += tau_tracer * supg_test_i *
                           (scratch.phi_u[q][j] * scratch.tracer_gradients[q]);
      }

    if (ordering.is_tracer(comp_j))
      {
        const double dR_dphi_j = bdf_c0 * scratch.shape_phi[q][j] +
                                 u_conv * scratch.grad_shape_phi[q][j];
        local_matrix_ij += tau_tracer * supg_test_i * dR_dphi_j;
      }

    if (ordering.is_potential(comp_j))
      local_matrix_ij += tau_tracer * supg_test_i *
                         (-scratch.mobility * scratch.laplacian_shape_mu[q][j]);
  }
 
} 

#endif 