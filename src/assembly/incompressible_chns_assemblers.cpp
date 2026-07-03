
#include <assembly/incompressible_chns_assemblers.h>
#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <parameter_reader.h>
#include <scratch_data.h>

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void
    VolumeAssembler<dim, ScratchData, CopyData, assembly_flags>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd = scratch_data;

      if constexpr (BaseType::with_stabilization)
        Assert(
          sd.enable_stabilization,
          ExcMessage(
            "The assemblers for the incompressible Cahn-Hilliard Navier-Stokes "
            "equations are "
            "set to assemble SUPG-PSPG stabilization terms, but computation of "
            "the required data was not enabled in the provided ScratchData."));
      if constexpr (BaseType::with_tracer_stabilization)
        Assert(
          sd.enable_tracer_stabilization,
          ExcMessage(
            "The assemblers for the incompressible Cahn-Hilliard Navier-Stokes "
            "equations are "
            "set to assemble SUPG stabilization term for the tracer equation, "
            "but computation of "
            "the required data was not enabled in the provided ScratchData."));

      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      // Potential-equation coefficients of the double-well term phi(phi^2-1)
      // and the gradient term grad(phi). The Abels model scales them by
      // sigma_tilde/eps and sigma_tilde*eps; Ding-Horriche uses the unscaled
      // potential (1 and eps^2). The capillary coefficient gamma is only used
      // by the Ding-Horriche capillary force gamma*mu*grad(phi).
      double double_well_coeff, gradient_coeff;
      if constexpr (BaseType::with_ding_horriche)
      {
        double_well_coeff = 1.;
        gradient_coeff    = sd.epsilon * sd.epsilon;
      }
      else
      {
        double_well_coeff = sd.sigma_tilde / sd.epsilon;
        gradient_coeff    = sd.sigma_tilde * sd.epsilon;
      }
      [[maybe_unused]] const double capillary_coeff =
        sd.sigma_tilde / sd.epsilon;
      const auto &body_force = sd.body_force;

      Tensor<1, dim> strong_residual_momentum;
      double         strong_residual_tracer;
      double         tau, tau_tracer;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_moving = sd.JxW_moving[q];
        const double rho        = sd.density[q];
        const double eta        = sd.dynamic_viscosity[q];

        const auto  &dudt       = sd.present_velocity_time_derivatives[q];
        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &lap_u      = sd.present_velocity_laplacians[q];
        const auto  &grad_div_u = sd.present_velocity_grad_div[q];
        const auto &source_u = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          // ALE contribution
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }

        const auto &p = sd.present_pressure_values[q];
        const auto &grad_p   = sd.present_pressure_gradients[q];
        const auto &source_p = sd.source_term_pressure[q];

        const auto &diffusive_flux = sd.diffusive_flux[q];
        const auto &phi            = sd.tracer_values[q];
        const auto &grad_phi       = sd.tracer_gradients[q];
        const auto &mu             = sd.potential_values[q];
        const auto &grad_mu        = sd.potential_gradients[q];
        const auto &source_phi     = sd.source_term_tracer[q];
        const auto &source_mu      = sd.source_term_potential[q];

        // Mobility M(q) and its derivative (both constant-model trivial).
        const double mobility       = sd.mobility_values[q];
        const double dmobility_dphi = sd.derivative_mobility_wrt_tracer[q];

        // Material marker m(phi) = q (abels_nlm) or phi (else): the
        // transported/conserved variable and the capillary marker are m, and
        // the potential mass factor is m'(phi). Identity marker (m=phi, m'=1)
        // reduces this to Abels/Ding-Horriche.
        const double m_marker  = sd.material_phase_values[q];
        const double dm_marker = sd.derivative_material_phase_wrt_tracer[q];
        const auto  &grad_m    = sd.material_phase_gradients[q];
        const double dmdt      = sd.material_phase_time_derivatives[q];

        // Capillary momentum force and diffusive inertia depend on the model:
        // Abels/abels_nlm use m*grad(mu) with the diffusive inertia J.grad(u);
        // Ding-Horriche uses -gamma*mu*grad(phi) and drops diffusive inertia.
        Tensor<1, dim> momentum_capillary;
        Tensor<1, dim> momentum_diffusive_inertia;
        if constexpr (BaseType::with_ding_horriche)
          momentum_capillary = -capillary_coeff * mu * grad_phi;
        else
        {
          momentum_capillary         = m_marker * grad_mu;
          momentum_diffusive_inertia = diffusive_flux;
        }

        const auto to_mult_by_phi_u_i =
          rho * (dudt + grad_u * u_conv - body_force) +
          momentum_diffusive_inertia + momentum_capillary + source_u;
        const auto to_mult_by_phi_phi_i = dmdt + u_conv * grad_m + source_phi;
        const auto to_mult_by_phi_mu_i =
          dm_marker * mu - double_well_coeff * phi * (phi * phi - 1.) +
          source_mu;

        const auto &phi_u = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];
        const auto &grad_phi_p     = sd.grad_phi_p[q];
        const auto &phi_phi      = sd.shape_phi[q];
        const auto &grad_phi_phi = sd.grad_shape_phi[q];
        const auto &phi_mu       = sd.shape_mu[q];
        const auto &grad_phi_mu  = sd.grad_shape_mu[q];

        double inv_rho = 0.;
        if constexpr (BaseType::with_stabilization)
        {
          tau                   = sd.tau_supg_velocity[q];
          inv_rho               = 1. / rho;
          const double detadphi = sd.derivative_dynamic_viscosity_wrt_tracer[q];

          // Compute strong residual of the momentum equation, in force units
          // (i.e. multiplied by the density). The consistent SUPG and PSPG test
          // operators are then (u_conv . grad(v)) and grad(q) / rho.
          strong_residual_momentum =
            rho * (dudt + grad_u * u_conv - body_force) +
            momentum_diffusive_inertia + momentum_capillary + grad_p +
            source_u - eta * (lap_u + grad_div_u) -
            2. * detadphi * (sym_grad_u * grad_phi);
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer          = sd.tau_supg_tracer[q];
          const double lap_mu = sd.potential_laplacians[q];

          // Strong residual of the transport equation (on the marker m).
          // div(M(q) grad mu) = M lap(mu) + (dM/dphi) grad(phi).grad(mu); the
          // second term vanishes when M is constant. Advection is on m
          // (grad_m = m' grad_phi), diffusion expands with grad_phi.
          strong_residual_tracer = dmdt + u_conv * grad_m -
                                   mobility * lap_mu -
                                   dmobility_dphi * (grad_phi * grad_mu) +
                                   source_phi;
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i   = sd.components[i];
          const bool         i_is_u   = this->ordering.is_velocity(comp_i);
          const bool         i_is_p   = this->ordering.is_pressure(comp_i);
          const bool         i_is_phi = this->ordering.is_tracer(comp_i);
          const bool         i_is_mu  = this->ordering.is_potential(comp_i);

          double local_rhs_i = i_is_p ? -phi_p[i] * (-div_u + source_p) : 0.;

          // Momentum equation
          if (i_is_u)
          {
            local_rhs_i -=
              phi_u[i] * to_mult_by_phi_u_i - div_phi_u[i] * p +
              2. * eta * scalar_product(sym_grad_phi_u[i], sym_grad_u);

            if constexpr (BaseType::with_stabilization)
              // SUPG stabilization
              local_rhs_i -=
                tau * (strong_residual_momentum * (grad_phi_u[i] * u_conv));
          }

          // Continuity equation
          else if (i_is_p)
          {
            if constexpr (BaseType::with_stabilization)
              // PSPG stabilization
              local_rhs_i +=
                tau * inv_rho * (strong_residual_momentum * grad_phi_p[i]);
          }

          // Tracer equation
          else if (i_is_phi)
          {
            local_rhs_i -= phi_phi[i] * to_mult_by_phi_phi_i +
                           mobility * (grad_phi_phi[i] * grad_mu);

            if constexpr (BaseType::with_tracer_stabilization)
              // Tracer SUPG stabilization
              local_rhs_i -= tau_tracer * (u_conv * grad_phi_phi[i]) *
                             strong_residual_tracer;
          }

          // Potential equation
          else if (i_is_mu)
          {
            local_rhs_i -= phi_mu[i] * to_mult_by_phi_mu_i -
                           gradient_coeff * (grad_phi_mu[i] * grad_phi);
          }

          local_rhs(i) += local_rhs_i * JxW_moving;
        }
      }

      if constexpr (BaseType::with_enlarged)
        assemble_psi_equation_rhs<dim>(this->ordering, sd, local_rhs);
    }

    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void VolumeAssembler<dim, ScratchData, CopyData, assembly_flags>::
      assemble_matrix(const ScratchData &scratch_data,
                      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      const double bdf_c0 = sd.bdf_c0;
      // Potential-equation coefficients (double-well and gradient terms) and
      // the Ding-Horriche capillary coefficient gamma; see the rhs assembler.
      double double_well_coeff, gradient_coeff;
      if constexpr (BaseType::with_ding_horriche)
      {
        double_well_coeff = 1.;
        gradient_coeff    = sd.epsilon * sd.epsilon;
      }
      else
      {
        double_well_coeff = sd.sigma_tilde / sd.epsilon;
        gradient_coeff    = sd.sigma_tilde * sd.epsilon;
      }
      [[maybe_unused]] const double capillary_coeff =
        sd.sigma_tilde / sd.epsilon;
      const auto &body_force = sd.body_force;

      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_momentum(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_potential(
        sd.dofs_per_cell);
      std::vector<double> phi_u_j_x_grad_phi(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_phi_i(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> strong_residual_momentum_variation(
        sd.dofs_per_cell);
      std::vector<double> strong_residual_tracer_variation(sd.dofs_per_cell);
      Tensor<1, dim>      strong_residual_momentum;
      Tensor<1, dim>      strong_residual_momentum_variation_phi_phi;
      double              strong_residual_tracer;
      Tensor<1, dim>      u_conv_dot_grad_phi_u_i, residual_dot_grad_phi_u_i;
      double              u_conv_dot_grad_phi_phi_i;
      Tensor<1, dim>      residual_tracer_dot_grad_phi_phi_i;
      double              tau, tau_tracer;

      const auto u_lower = this->ordering.u_lower;

      const SymmetricTensor<2, dim> identity_tensor =
        unit_symmetric_tensor<dim>();

      //
      // Moving mesh related data
      //
      const std::vector<Tensor<1, dim>> *phi_x;
      const std::vector<Tensor<2, dim>> *grad_phi_x_moving;
      const std::vector<Tensor<3, dim>> *hessian_phi_x_moving;
      // x-variation of the SUPG/PSPG strong residuals (frozen tau), used by the
      // momentum, continuity and tracer rows.
      std::vector<Tensor<1, dim>> strong_residual_momentum_x_variation(
        sd.dofs_per_cell);
      std::vector<double> strong_residual_tracer_x_variation(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> mesh_velocity_x_variation(sd.dofs_per_cell);
      std::vector<double>         trace_grad_phi_x_moving(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>>        to_mult_by_phi_u_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<Tensor<2, dim>> to_mult_by_grad_phi_u_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<double> p_x_tr_G_j(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_p_i_moving_mesh(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_phi_i_moving_mesh(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_grad_phi_phi_i_moving_mesh(
        sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_mu_i_moving_mesh(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_grad_phi_mu_i_moving_mesh(
        sd.dofs_per_cell);

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
      const Tensor<2, dim> *grad_source_term_velocity;
      const Tensor<1, dim> *grad_source_pressure;
      const Tensor<1, dim> *grad_source_tracer;
      const Tensor<1, dim> *grad_source_potential;
#endif

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_moving = sd.JxW_moving[q];
        const double rho        = sd.density[q];
        const double eta        = sd.dynamic_viscosity[q];
        const double drhodphi   = sd.derivative_density_wrt_tracer[q];
        const double detadphi   = sd.derivative_dynamic_viscosity_wrt_tracer[q];

        const auto  &dudt       = sd.present_velocity_time_derivatives[q];
        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &lap_u      = sd.present_velocity_laplacians[q];
        const auto  &grad_div_u = sd.present_velocity_grad_div[q];
        const auto &source_u = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;

        const auto &p = sd.present_pressure_values[q];
        const auto  &grad_p   = sd.present_pressure_gradients[q];
        const double source_p = sd.source_term_pressure[q];

        const auto  &diffusive_flux = sd.diffusive_flux[q];
        const auto  &phi        = sd.tracer_values[q];
        const auto  &grad_phi   = sd.tracer_gradients[q];
        const auto  &mu         = sd.potential_values[q];
        const auto  &grad_mu    = sd.potential_gradients[q];
        const double source_phi = sd.source_term_tracer[q];
        const double source_mu  = sd.source_term_potential[q];

        // Mobility M(phi) with its first two derivatives, and the Abels
        // diffusive-flux factor 0.5*(rho1 - rho0)*M(phi) with its derivative.
        // Every derivative is zero for the constant-mobility model.
        const double mobility         = sd.mobility_values[q];
        const double dmobility_dphi   = sd.derivative_mobility_wrt_tracer[q];
        const double d2mobility_dphi2 =
          sd.second_derivative_mobility_wrt_tracer[q];
        const double diffusive_flux_factor =
          sd.diffusive_flux_factor_values[q];
        const double ddiffusive_flux_factor_dphi =
          dmobility_dphi * 0.5 * (sd.density1 - sd.density0);

        // Material marker m(phi) = q (abels_nlm) or phi (else), its first two
        // derivatives, gradient and BDF time derivative. Identity marker
        // (m=phi, m'=1, m''=0) reduces every term below to Abels/Ding-Horriche.
        const double m_marker   = sd.material_phase_values[q];
        const double dm_marker  = sd.derivative_material_phase_wrt_tracer[q];
        const double d2m_marker =
          sd.second_derivative_material_phase_wrt_tracer[q];
        const auto  &grad_m     = sd.material_phase_gradients[q];
        const double dmdt       = sd.material_phase_time_derivatives[q];

        // Capillary momentum force and diffusive inertia (see the rhs
        // assembler): Abels/abels_nlm use m*grad(mu) with diffusive inertia,
        // Ding-Horriche uses -gamma*mu*grad(phi) and no diffusive inertia.
        Tensor<1, dim> momentum_capillary;
        Tensor<1, dim> momentum_diffusive_inertia;
        if constexpr (BaseType::with_ding_horriche)
          momentum_capillary = -capillary_coeff * mu * grad_phi;
        else
        {
          momentum_capillary         = m_marker * grad_mu;
          momentum_diffusive_inertia = diffusive_flux;
        }

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &laplacian_phi_u  = sd.laplacian_phi_u[q];
        const auto &grad_div_phi_u   = sd.grad_div_phi_u[q];
        const auto &phi_p = sd.phi_p[q];
        const auto &grad_phi_p       = sd.grad_phi_p[q];
        const auto &phi_phi      = sd.shape_phi[q];
        const auto &grad_phi_phi = sd.grad_shape_phi[q];
        const auto &phi_mu       = sd.shape_mu[q];
        const auto &grad_phi_mu  = sd.grad_shape_mu[q];
        const auto &laplacian_phi_mu = sd.laplacian_shape_mu[q];

        //
        // Moving mesh related data
        //
        if constexpr (BaseType::with_moving_mesh)
        {
          phi_x             = &sd.phi_x[q];
          grad_phi_x_moving = &sd.grad_phi_x_moving[q];
          if constexpr (BaseType::with_stabilization ||
                        BaseType::with_tracer_stabilization)
            hessian_phi_x_moving = &sd.hessian_phi_x_moving[q];

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
          grad_source_term_velocity = &sd.grad_source_velocity[q];
          grad_source_pressure      = &sd.grad_source_pressure[q];
          grad_source_tracer        = &sd.grad_source_tracer[q];
          grad_source_potential     = &sd.grad_source_potential[q];
#endif
        }

        // Precompute shape functions-independent terms. The last term is the
        // tracer variation of the Abels diffusive flux M(phi)*grad(u).grad(mu)
        // and is zero for a constant mobility.
        // Tracer (phi) column of the momentum equation, multiplied by the
        // scalar shape phi_phi[j]. For Abels the capillary derivative grad(mu)
        // and the diffusive-inertia derivative both scale with phi_phi[j]; for
        // Ding-Horriche the capillary derivative scales with grad(phi_phi[j])
        // instead and is added separately in the i/j loops below.
        Tensor<1, dim> to_mult_by_phi_u_i_phi_phi_j =
          drhodphi * (dudt + u_dot_grad_u_ale - body_force);
        if constexpr (!BaseType::with_ding_horriche)
          // Capillary m*grad(mu): d/dphi = m'*grad(mu) (m'=1 for Abels).
          to_mult_by_phi_u_i_phi_phi_j +=
            dm_marker * grad_mu +
            ddiffusive_flux_factor_dphi * (grad_u * grad_mu);

        const auto momentum_partial_residual =
          rho * (dudt - body_force + u_dot_grad_u_ale) + momentum_capillary +
          source_u;
        const auto phi_partial_residual = dmdt + u_conv * grad_m + source_phi;
        const auto mu_partial_residual =
          dm_marker * mu - double_well_coeff * phi * (phi * phi - 1.) +
          source_mu;
        double inv_rho      = 0.;
        double dinvrho_dphi = 0.;

        if constexpr (BaseType::with_stabilization)
        {
          tau          = sd.tau_supg_velocity[q];
          inv_rho      = 1. / rho;
          dinvrho_dphi = -drhodphi * inv_rho * inv_rho;

          // Compute strong residual of the momentum equation, in force units
          // (i.e. multiplied by the density); see the rhs assembler for the
          // corresponding test operators.
          strong_residual_momentum =
            rho * (dudt + grad_u * u_conv - body_force) +
            momentum_diffusive_inertia + momentum_capillary + grad_p +
            source_u - eta * (lap_u + grad_div_u) -
            2. * detadphi * (sym_grad_u * grad_phi);

          strong_residual_momentum_variation_phi_phi =
            to_mult_by_phi_u_i_phi_phi_j - detadphi * (lap_u + grad_div_u);
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer          = sd.tau_supg_tracer[q];
          const double lap_mu = sd.potential_laplacians[q];

          // Strong residual of the transport equation (on the marker m).
          // div(M(q) grad mu) = M lap(mu) + (dM/dphi) grad(phi).grad(mu); the
          // second term vanishes when M is constant. Advection is on m
          // (grad_m = m' grad_phi), diffusion expands with grad_phi.
          strong_residual_tracer = dmdt + u_conv * grad_m -
                                   mobility * lap_mu -
                                   dmobility_dphi * (grad_phi * grad_mu) +
                                   source_phi;
        }

        // Precompute quantities depending only on j
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto &phi_u_j       = phi_u[j];
          const auto &grad_phi_u_j  = grad_phi_u[j];
          const auto &grad_phi_mu_j = grad_phi_mu[j];

          to_mult_by_phi_u_i_momentum[j] =
            rho *
            (bdf_c0 * phi_u_j + grad_phi_u_j * u_conv + grad_u * phi_u_j);
          if constexpr (!BaseType::with_ding_horriche)
            // Diffusive-inertia velocity coupling (Abels only).
            to_mult_by_phi_u_i_momentum[j] +=
              diffusive_flux_factor * grad_phi_u_j * grad_mu;

          // Potential (mu) column of the momentum equation. Abels:
          // diffusive-inertia + capillary phi*grad(mu). Ding-Horriche:
          // capillary -gamma*mu*grad(phi), i.e. -gamma*phi_mu[j]*grad(phi).
          if constexpr (BaseType::with_ding_horriche)
            to_mult_by_phi_u_i_potential[j] =
              -capillary_coeff * phi_mu[j] * grad_phi;
          else
            to_mult_by_phi_u_i_potential[j] =
              diffusive_flux_factor * grad_u * grad_phi_mu_j +
              m_marker * grad_phi_mu_j;

          // Velocity column of the transport advection u.grad(m): d/du_j.
          phi_u_j_x_grad_phi[j] = phi_u_j * grad_m;

          // Tracer (phi) column of the transport dm/dt + u.grad(m). With the
          // marker m(phi): d/dphi_j = m'(bdf_c0 N + u.grad N)
          //                          + m'' N (u.grad phi). Identity -> Abels.
          to_mult_by_phi_phi_i[j] =
            dm_marker * (bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j]) +
            d2m_marker * phi_phi[j] * (u_conv * grad_phi);

          if constexpr (BaseType::with_stabilization)
          {
            // As in the stabilized NS assembler, tau is kept constant in the
            // Newton Jacobian; only the residual and test operator are
            // linearized.

            // Variation w.r.t. velocity and pressure
            strong_residual_momentum_variation[j] =
              to_mult_by_phi_u_i_momentum[j] + grad_phi_p[j] -
              eta * (laplacian_phi_u[j] + grad_div_phi_u[j]) -
              2. * detadphi * (sym_grad_phi_u[j] * grad_phi);

            // Variation w.r.t. tracer
            strong_residual_momentum_variation[j] +=
              phi_phi[j] * strong_residual_momentum_variation_phi_phi -
              2. * detadphi * (sym_grad_u * grad_phi_phi[j]);
            if constexpr (!BaseType::with_ding_horriche)
              // Non-linear-mixing second-order viscous cross term: the strong
              // residual has -2 eta'(phi)(d.grad phi) with eta'(phi)=eta_q m',
              // so d/dphi_j adds -2 eta_q m'' N_phi (d.grad phi). eta_q m'' = 0
              // for the identity marker, so this is byte-neutral for Abels.
              strong_residual_momentum_variation[j] +=
                -2. * 0.5 * (sd.dynamic_viscosity0 - sd.dynamic_viscosity1) *
                d2m_marker * phi_phi[j] * (sym_grad_u * grad_phi);
            if constexpr (BaseType::with_ding_horriche)
              // Ding-Horriche capillary derivative w.r.t. phi scales with
              // grad(phi_phi[j]) (see to_mult_by_phi_u_i_phi_phi_j).
              strong_residual_momentum_variation[j] +=
                -capillary_coeff * mu * grad_phi_phi[j];

            // Variation w.r.t. potential (same as the momentum mu column).
            strong_residual_momentum_variation[j] +=
              to_mult_by_phi_u_i_potential[j];
          }

          if constexpr (BaseType::with_tracer_stabilization)
          {
            // Transport dm/dt + u.grad(m) variation: bdf_c0 m' N + phi_u.grad m
            // + m'(u.grad N) + m'' N (u.grad phi), then the diffusion column.
            strong_residual_tracer_variation[j] =
              bdf_c0 * dm_marker * phi_phi[j] + phi_u_j * grad_m +
              dm_marker * (u_conv * grad_phi_phi[j]) +
              d2m_marker * phi_phi[j] * (u_conv * grad_phi) -
              mobility * laplacian_phi_mu[j];

            // Degenerate-mobility variations of -M(phi) lap(mu)
            // - M'(phi) grad(phi).grad(mu) (all zero for a constant mobility).
            const double lap_mu = sd.potential_laplacians[q];
            strong_residual_tracer_variation[j] -=
              dmobility_dphi * phi_phi[j] * lap_mu +
              d2mobility_dphi2 * phi_phi[j] * (grad_phi * grad_mu) +
              dmobility_dphi * (grad_phi_phi[j] * grad_mu) +
              dmobility_dphi * (grad_phi * grad_phi_mu_j);
          }

          // Variations w.r.t. mesh position
          if constexpr (BaseType::with_moving_mesh)
          {
            const auto  &phi_x_j     = (*phi_x)[j];
            const auto  &G           = (*grad_phi_x_moving)[j];
            const auto   transpose_G = transpose(G);
            const double trG         = trace(G);

            const auto grad_u_x_G_j = grad_u * G;

            p_x_tr_G_j[j] = p * trG;

            /**
             * Weak laplacian-like products (e.g., grad_phi_mu \cdot grad_phi in
             * the potential equation) vary like this with the mesh position:
             *
             * delta_x_j ((grad_phi_mu \cdot grad_phi) * dx) =
             *   (-G_j^T * grad_phi_mu) * grad_phi + grad_phi_mu * (-G_j^T *
             *   grad_phi)
             *     + (grad_phi_mu * grad_phi) * trace(G_j),
             *
             * which can be written as
             *   grad_phi_mu * ((-G_j - G_j^T + trace(G_j) * I) * grad_phi
             * = grad_phi_mu * ((- 2*sym(G_j) + trace(G_j) * I) * grad_phi.
             *
             * The quantity below is the one in parentheses.
             */
            const Tensor<2, dim> val =
              trG * identity_tensor - 2. * symmetrize(G);

            // x-variation of the SUPG/PSPG strong residuals (frozen tau).
            // Gradients transform as grad -> grad - G^T grad, the velocity
            // gradient as grad_u -> grad_u - grad_u G, and the ALE convective
            // velocity as u_conv -> u_conv - bdf_c0 phi_x.
            if constexpr (BaseType::with_stabilization ||
                          BaseType::with_tracer_stabilization)
            {
              const auto du_conv_dx  = -bdf_c0 * phi_x_j;
              const auto dgrad_u_dx  = -grad_u_x_G_j; // -(grad_u * G)
              const auto dgrad_mu_dx = -(transpose_G * grad_mu);
              const auto dgrad_phi_dx = -(transpose_G * grad_phi);

              mesh_velocity_x_variation[j] = du_conv_dx;
              trace_grad_phi_x_moving[j]   = trG;

              if constexpr (BaseType::with_stabilization)
              {
                const auto &H = sd.present_velocity_hessians[q];
                const auto &K = (*hessian_phi_x_moving)[j];
                const auto  dgrad_p_dx = -(transpose_G * grad_p);

                // d(lap_u + grad_div_u)/dx_j, with H[c][a][b] = d^2 u_c.
                Tensor<1, dim> d_lap_plus_graddiv_u_dx;
                for (unsigned int c = 0; c < dim; ++c)
                  for (unsigned int a = 0; a < dim; ++a)
                    for (unsigned int b = 0; b < dim; ++b)
                    {
                      double dH = 0.;
                      for (unsigned int m = 0; m < dim; ++m)
                        dH -= G[m][a] * H[c][m][b] + G[m][b] * H[c][a][m] +
                              K[m][a][b] * grad_u[c][m];
                      if (a == b)
                        d_lap_plus_graddiv_u_dx[c] += dH;
                      if (a == c)
                        d_lap_plus_graddiv_u_dx[b] += dH;
                    }

                const auto d_sym_grad_u_dx = symmetrize(dgrad_u_dx);
                const auto d_viscous_divergence_dx =
                  eta * d_lap_plus_graddiv_u_dx +
                  2. * detadphi *
                    (d_sym_grad_u_dx * grad_phi + sym_grad_u * dgrad_phi_dx);

                // Frozen-tau x-variation of the diffusive inertia and the
                // capillary force. Ding-Horriche drops the diffusive inertia
                // and its capillary varies through grad(phi) instead of
                // grad(mu).
                Tensor<1, dim> ddiffusive_flux_dx;
                Tensor<1, dim> dcapillary_dx;
                if constexpr (BaseType::with_ding_horriche)
                  dcapillary_dx = -capillary_coeff * mu * dgrad_phi_dx;
                else
                {
                  ddiffusive_flux_dx =
                    diffusive_flux_factor *
                    (dgrad_u_dx * grad_mu + grad_u * dgrad_mu_dx);
                  // Capillary m*grad(mu): m is a nodal value (invariant under
                  // the mesh x-variation), only grad(mu) transforms.
                  dcapillary_dx = m_marker * dgrad_mu_dx;
                }

                strong_residual_momentum_x_variation[j] =
                  rho * (dgrad_u_dx * u_conv + grad_u * du_conv_dx) +
                  dgrad_p_dx + ddiffusive_flux_dx + dcapillary_dx -
                  d_viscous_divergence_dx;
              }

              if constexpr (BaseType::with_tracer_stabilization)
              {
                const auto  &h = sd.potential_hessians[q];
                const auto  &K = (*hessian_phi_x_moving)[j];

                // d(laplacian mu)/dx_j = trace of the scalar hessian variation.
                double dlap_mu_dx = 0.;
                for (unsigned int i = 0; i < dim; ++i)
                  for (unsigned int a = 0; a < dim; ++a)
                    dlap_mu_dx -= G[a][i] * h[a][i] + G[a][i] * h[i][a] +
                                  K[a][i][i] * grad_mu[a];

                // Advection u.grad(m) = m' u.grad(phi): m' is invariant under
                // the mesh x-variation, so the advection x-variation scales by
                // m'; the diffusion column is unchanged.
                strong_residual_tracer_x_variation[j] =
                  dm_marker * (du_conv_dx * grad_phi + u_conv * dgrad_phi_dx) -
                  mobility * dlap_mu_dx -
                  dmobility_dphi *
                    (dgrad_phi_dx * grad_mu + grad_phi * dgrad_mu_dx);
              }
            }

            // Variation of momentum. The capillary trG part comes from
            // momentum_partial_residual*trG; the non-trG part and (for Abels)
            // the diffusive-inertia mesh variation are added here.
            to_mult_by_phi_u_i_moving_mesh[j] =
              momentum_partial_residual * trG +
              rho * (grad_u * (-bdf_c0 * phi_x_j - G * u_conv));
            if constexpr (BaseType::with_ding_horriche)
              to_mult_by_phi_u_i_moving_mesh[j] +=
                capillary_coeff * mu * (transpose_G * grad_phi);
            else
              to_mult_by_phi_u_i_moving_mesh[j] +=
                -m_marker * transpose_G * grad_mu +
                diffusive_flux_factor * grad_u * val * grad_mu;

            to_mult_by_grad_phi_u_i_moving_mesh[j] =
              p * transpose_G +
              2. * eta *
                (sym_grad_u * (trG * identity_tensor - transpose_G) -
                 symmetrize(grad_u_x_G_j));

            // Variation of continuity
            to_mult_by_phi_p_i_moving_mesh[j] =
              trace(grad_u_x_G_j) + (-div_u + source_p) * trG;

            // Variation of tracer (transport on m; grad_m = m' grad_phi).
            to_mult_by_phi_phi_i_moving_mesh[j] =
              phi_partial_residual * trG - bdf_c0 * (phi_x_j * grad_m) -
              u_conv * (transpose_G * grad_m);

            to_mult_by_grad_phi_phi_i_moving_mesh[j] =
              mobility * (val * grad_mu);

            // Variation of potential
            to_mult_by_phi_mu_i_moving_mesh[j] = mu_partial_residual * trG;

            to_mult_by_grad_phi_mu_i_moving_mesh[j] =
              -gradient_coeff * (val * grad_phi);

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
            to_mult_by_phi_u_i_moving_mesh[j] +=
              (*grad_source_term_velocity) * phi_x_j;
            to_mult_by_phi_p_i_moving_mesh[j] +=
              (*grad_source_pressure) * phi_x_j;
            to_mult_by_phi_phi_i_moving_mesh[j] +=
              (*grad_source_tracer) * phi_x_j;
            to_mult_by_phi_mu_i_moving_mesh[j] +=
              (*grad_source_potential) * phi_x_j;
#endif
          }
        }

        /**
         * Assemble the local matrix.
         * The loops over the j degrees of freedom are repeated for each
         * assembled equation: this removes the tests over i inside the j
         * loop, and is (ever so slightly) more efficient. Looping only over
         * the coupled dofs for each variable (obtained by creating a set
         * first in the scratch, for instance) does not yield any tremendous
         * additional gain though.
         */

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i = sd.components[i];
          const bool         i_is_x = this->ordering.is_position(comp_i);
          if (i_is_x)
            continue;

          // Iterator to the current matrix row
          auto matrix_row = local_matrix[i];

          const auto &phi_u_i          = phi_u[i];
          const auto &grad_phi_u_i     = grad_phi_u[i];
          const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
          const auto &div_phi_u_i      = div_phi_u[i];
          const auto &phi_p_i          = phi_p[i];
          const auto &grad_phi_p_i     = grad_phi_p[i];
          const auto &phi_phi_i      = phi_phi[i];
          const auto &grad_phi_phi_i = grad_phi_phi[i];
          const auto &phi_mu_i       = phi_mu[i];
          const auto &grad_phi_mu_i  = grad_phi_mu[i];

          /**
           * Momentum equation
           */
          if (this->ordering.is_velocity(comp_i))
          {
            if constexpr (BaseType::with_stabilization)
            {
              u_conv_dot_grad_phi_u_i = grad_phi_u_i * u_conv;
              residual_dot_grad_phi_u_i =
                strong_residual_momentum * grad_phi_u_i;
            }

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_u   = this->ordering.is_velocity(comp_j);
              const bool         j_is_p   = this->ordering.is_pressure(comp_j);
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &phi_phi_j = phi_phi[j];

              // Account for the pressure gradient when initializing
              double local_matrix_ij = j_is_p ? -div_phi_u_i * phi_p[j] : 0.;

              if (j_is_u)
              {
                local_matrix_ij += phi_u_i * to_mult_by_phi_u_i_momentum[j];

                // Diffusion: 2. * eta * scalar_product(sym_grad_phi_u[j],
                // sym_grad_phi_u_i), explicited for the symmetric gradient of
                // Lagrange shape functions.
                const auto &gui = grad_phi_u_i[comp_i - u_lower];
                const auto &guj = grad_phi_u[j][comp_j - u_lower];
                local_matrix_ij +=
                  eta * (gui[comp_j - u_lower] * guj[comp_i - u_lower]);
                if (comp_i == comp_j)
                  local_matrix_ij += eta * gui * guj;
              }
              else if (j_is_phi)
              {
                local_matrix_ij +=
                  phi_phi_j * (phi_u_i * to_mult_by_phi_u_i_phi_phi_j +
                               2. * detadphi *
                                 scalar_product(sym_grad_phi_u_i, sym_grad_u));
                if constexpr (BaseType::with_ding_horriche)
                  // Ding-Horriche capillary -gamma*mu*grad(phi) varies with
                  // grad(phi_phi[j]) rather than the scalar shape phi_phi[j].
                  local_matrix_ij +=
                    -capillary_coeff * mu *
                    (phi_u_i * grad_phi_phi[j]);
              }
              else if (j_is_mu)
                local_matrix_ij += phi_u_i * to_mult_by_phi_u_i_potential[j];

              if constexpr (BaseType::with_stabilization)
              {
                // SUPG stabilization : variation w.r.t. u and p
                local_matrix_ij +=
                  tau * (strong_residual_momentum_variation[j] *
                           u_conv_dot_grad_phi_u_i +
                         residual_dot_grad_phi_u_i * phi_u[j]);
              }

              if constexpr (BaseType::with_moving_mesh)
              {
                if (j_is_x)
                {
                  // Momentum : variation w.r.t. moving mesh position

                  // Simplification of the double contraction grad_phi_u_i : T,
                  // with T = to_mult_by_grad_phi_u_i_moving_mesh[j], valid
                  // for vector-valued Lagrange shape functions.
                  const auto &grad_phi_u_i_row = grad_phi_u_i[comp_i - u_lower];
                  const auto &t_row =
                    to_mult_by_grad_phi_u_i_moving_mesh[j][comp_i - u_lower];

                  local_matrix_ij +=
                    phi_u_i * to_mult_by_phi_u_i_moving_mesh[j] +
                    -div_phi_u_i * p_x_tr_G_j[j] + grad_phi_u_i_row * t_row;

                  // SUPG : variation w.r.t. mesh position (frozen tau). The
                  // trace term accounts for the variation of JxW_moving, which
                  // multiplies local_matrix_ij below.
                  if constexpr (BaseType::with_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    const auto  d_supg_test_dx =
                      -(grad_phi_u_i * G) * u_conv +
                      grad_phi_u_i * mesh_velocity_x_variation[j];
                    local_matrix_ij +=
                      tau *
                      (u_conv_dot_grad_phi_u_i *
                         strong_residual_momentum_x_variation[j] +
                       d_supg_test_dx * strong_residual_momentum +
                       u_conv_dot_grad_phi_u_i * strong_residual_momentum *
                         trace_grad_phi_x_moving[j]);
                  }
                }
              }

              // Increment local matrix
              matrix_row[j] += local_matrix_ij * JxW_moving;
            }
          }

          /**
           * Continuity equation
           */
          if (this->ordering.is_pressure(comp_i))
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = sd.components[j];
              const bool         j_is_u = this->ordering.is_velocity(comp_j);
              const bool         j_is_x = this->ordering.is_position(comp_j);

              if (j_is_u)
                matrix_row[j] += -phi_p_i * div_phi_u[j] * JxW_moving;

              if constexpr (BaseType::with_stabilization)
                // PSPG stabilization : variation w.r.t. u and p
                matrix_row[j] +=
                  -tau *
                  ((inv_rho * strong_residual_momentum_variation[j] +
                    dinvrho_dphi * phi_phi[j] * strong_residual_momentum) *
                   grad_phi_p_i) *
                  JxW_moving;

              if constexpr (BaseType::with_moving_mesh)
              {
                if (j_is_x)
                {
                  // Continuity : variation w.r.t. x
                  matrix_row[j] +=
                    phi_p_i * to_mult_by_phi_p_i_moving_mesh[j] * JxW_moving;

                  // PSPG : variation w.r.t. mesh position (frozen tau).
                  if constexpr (BaseType::with_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    // d(grad_phi_p_i)/dx_j = -(G^T grad_phi_p_i)
                    const auto dgrad_phi_p_i_dx = -(grad_phi_p_i * G);
                    matrix_row[j] -=
                      tau * inv_rho *
                      (strong_residual_momentum_x_variation[j] * grad_phi_p_i +
                       strong_residual_momentum * dgrad_phi_p_i_dx +
                       strong_residual_momentum * grad_phi_p_i *
                         trace_grad_phi_x_moving[j]) *
                      JxW_moving;
                  }
                }
              }
            }

          /**
           * Tracer equation
           */
          else if (this->ordering.is_tracer(comp_i))
          {
            if constexpr (BaseType::with_tracer_stabilization)
            {
              u_conv_dot_grad_phi_phi_i = u_conv * grad_phi_phi_i;
              residual_tracer_dot_grad_phi_phi_i =
                strong_residual_tracer * grad_phi_phi_i;
            }

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_u   = this->ordering.is_velocity(comp_j);
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &grad_phi_mu_j = grad_phi_mu[j];

              if (j_is_u)
              {
                matrix_row[j] += phi_phi_i * phi_u_j_x_grad_phi[j] * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  phi_phi_i * to_mult_by_phi_phi_i[j] * JxW_moving;
                // Tracer variation of the diffusion term M(phi) grad(v).grad(mu)
                // (zero for a constant mobility).
                matrix_row[j] += dmobility_dphi * phi_phi[j] *
                                 (grad_phi_phi_i * grad_mu) * JxW_moving;
              }
              else if (j_is_mu)
              {
                matrix_row[j] +=
                  mobility * (grad_phi_mu_j * grad_phi_phi_i) * JxW_moving;
              }

              if constexpr (BaseType::with_tracer_stabilization)
              {
                // Tracer SUPG stabilization : variation w.r.t. u and phi
                matrix_row[j] +=
                  tau_tracer *
                  (strong_residual_tracer_variation[j] *
                     u_conv_dot_grad_phi_phi_i +
                   residual_tracer_dot_grad_phi_phi_i * phi_u[j]) *
                  JxW_moving;
              }
              if constexpr (BaseType::with_moving_mesh)
              {
                // Tracer : variation w.r.t. x
                if (j_is_x)
                {
                  matrix_row[j] +=
                    (phi_phi_i * to_mult_by_phi_phi_i_moving_mesh[j] +
                     grad_phi_phi_i *
                       to_mult_by_grad_phi_phi_i_moving_mesh[j]) *
                    JxW_moving;

                  // Tracer-SUPG : variation w.r.t. mesh position (frozen tau).
                  if constexpr (BaseType::with_tracer_stabilization)
                  {
                    const auto &G = sd.grad_phi_x_moving[q][j];
                    const auto  dgrad_phi_phi_i_dx = -(grad_phi_phi_i * G);
                    const double d_supg_test_dx =
                      mesh_velocity_x_variation[j] * grad_phi_phi_i +
                      u_conv * dgrad_phi_phi_i_dx;
                    matrix_row[j] +=
                      tau_tracer *
                      (u_conv_dot_grad_phi_phi_i *
                         strong_residual_tracer_x_variation[j] +
                       d_supg_test_dx * strong_residual_tracer +
                       u_conv_dot_grad_phi_phi_i * strong_residual_tracer *
                         trace_grad_phi_x_moving[j]) *
                      JxW_moving;
                  }
                }
              }
            }
          }

          /**
           * Potential equation
           */
          else if (this->ordering.is_potential(comp_i))
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j   = sd.components[j];
              const bool         j_is_phi = this->ordering.is_tracer(comp_j);
              const bool         j_is_mu  = this->ordering.is_potential(comp_j);
              const bool         j_is_x   = this->ordering.is_position(comp_j);

              const auto &phi_phi_j      = phi_phi[j];
              const auto &grad_phi_phi_j = grad_phi_phi[j];
              const auto &phi_mu_j       = phi_mu[j];

              if (j_is_mu)
              {
                // Mass factor m'(phi) mu: d/dmu_j = m' N_mu (m'=1 for Abels).
                matrix_row[j] +=
                  dm_marker * phi_mu_i * phi_mu_j * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  (-double_well_coeff * phi_mu_i * phi_phi_j *
                     (3. * phi * phi - 1.) -
                   gradient_coeff * (grad_phi_mu_i * grad_phi_phi_j)) *
                  JxW_moving;
                // Non-linear-mixing: d/dphi_j of the mass factor m'(phi) mu is
                // m'' N_phi mu (zero for the identity marker).
                matrix_row[j] +=
                  d2m_marker * phi_mu_i * phi_phi_j * mu * JxW_moving;
              }

              if constexpr (BaseType::with_moving_mesh)
              {
                // Potential : variation w.r.t. x
                if (j_is_x)
                {
                  matrix_row[j] +=
                    (phi_mu_i * to_mult_by_phi_mu_i_moving_mesh[j] +
                     grad_phi_mu_i * to_mult_by_grad_phi_mu_i_moving_mesh[j]) *
                    JxW_moving;
                }
              }
            }
        }
      }

      if constexpr (BaseType::with_enlarged)
        assemble_psi_equation_matrix<dim, BaseType::with_moving_mesh>(
          this->ordering, this->coupling_table, sd, local_matrix);
    }

    template <int dim, typename ScratchData, typename CopyData>
    void ContactAngleBoundaryAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      if (!copy_data.cell_is_at_boundary)
        return;

      const auto  &sd        = scratch_data;
      auto        &local_rhs = copy_data.local_rhs(sd.active_fe_index);
      const double epsilon   = param.cahn_hilliard.epsilon_interface;
      const double coeff =
        CahnHilliard::contact_angle_surface_coefficient(param.cahn_hilliard,
                                                        sd.sigma_tilde);

      for (unsigned int i_face = 0; i_face < sd.n_faces; ++i_face)
      {
        if (!sd.face_at_boundary[i_face])
          continue;
        const auto bc_it = param.cahn_hilliard_bc.find(sd.face_boundary_id[i_face]);
        if (bc_it == param.cahn_hilliard_bc.end() ||
            bc_it->second.contact_angle < 0.)
          continue;
        const double theta = bc_it->second.contact_angle;

        for (unsigned int qf = 0; qf < sd.n_faces_q_points; ++qf)
        {
          const double g_phi = CahnHilliard::contact_angle_normal_derivative(
            sd.tracer_values_face[i_face][qf], epsilon, theta);
          const double weight = sd.face_JxW_moving[i_face][qf];

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
            if (ordering.is_potential(sd.components[i]))
              local_rhs(i) -=
                coeff * g_phi * sd.shape_mu_face[i_face][qf][i] * weight;
        }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    ContactAngleBoundaryAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      if (!copy_data.cell_is_at_boundary)
        return;

      const auto  &sd           = scratch_data;
      auto        &local_matrix = copy_data.local_matrix(sd.active_fe_index);
      const double epsilon      = param.cahn_hilliard.epsilon_interface;
      const double coeff =
        CahnHilliard::contact_angle_surface_coefficient(param.cahn_hilliard,
                                                        sd.sigma_tilde);

      for (unsigned int i_face = 0; i_face < sd.n_faces; ++i_face)
      {
        if (!sd.face_at_boundary[i_face])
          continue;
        const auto bc_it = param.cahn_hilliard_bc.find(sd.face_boundary_id[i_face]);
        if (bc_it == param.cahn_hilliard_bc.end() ||
            bc_it->second.contact_angle < 0.)
          continue;
        const double theta = bc_it->second.contact_angle;

        for (unsigned int qf = 0; qf < sd.n_faces_q_points; ++qf)
        {
          const double g_phi_prime =
            CahnHilliard::contact_angle_normal_derivative_jacobian(
              sd.tracer_values_face[i_face][qf], epsilon, theta);
          const double weight = sd.face_JxW_moving[i_face][qf];

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          {
            if (!ordering.is_potential(sd.components[i]))
              continue;
            const double test = sd.shape_mu_face[i_face][qf][i];
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
              if (ordering.is_tracer(sd.components[j]))
                local_matrix(i, j) += coeff * g_phi_prime *
                                      sd.shape_phi_face[i_face][qf][j] * test *
                                      weight;
          }
        }
      }
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

// Explicit instantiations
#include "incompressible_chns_assemblers.inst"
