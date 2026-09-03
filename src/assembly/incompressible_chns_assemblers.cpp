
#include <assembly/incompressible_chns_assemblers.h>
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

      const double mobility              = sd.mobility;
      const double sigma_tilde_over_eps  = sd.sigma_tilde / sd.epsilon;
      const double sigma_tilde_times_eps = sd.sigma_tilde * sd.epsilon;
      const auto  &body_force            = sd.body_force;

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
        const auto  &source_u   = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          // ALE contribution
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }

        const auto &p        = sd.present_pressure_values[q];
        const auto &grad_p   = sd.present_pressure_gradients[q];
        const auto &source_p = sd.source_term_pressure[q];

        const auto &diffusive_flux = sd.diffusive_flux[q];
        const auto &dphidt         = sd.tracer_time_derivatives[q];
        const auto &phi            = sd.tracer_values[q];
        const auto &grad_phi       = sd.tracer_gradients[q];
        const auto &mu             = sd.potential_values[q];
        const auto &grad_mu        = sd.potential_gradients[q];
        const auto &source_phi     = sd.source_term_tracer[q];
        const auto &source_mu      = sd.source_term_potential[q];

        /**
         * Model-dependent quantities. Abels and Stepien share the potential
         * equation (the coefficients sigma_tilde/eps and sigma_tilde*eps are
         * identical for both models), but differ in the momentum, continuity
         * and phase equations:
         *
         *  - momentum: Stepien uses the conservative form, which adds
         *    S_c * u with S_c = drho/dphi Dphi/Dt + rho div(u), replaces the
         *    Abels capillary force phi grad(mu) by (dpr - mu) grad(phi), adds
         *    the bulk-viscosity stress lambda (div u) I, and carries no
         *    diffusive inertia;
         *  - continuity: div(u) is no longer zero but equals
         *    -(drho/dphi / rho) Dphi/Dt;
         *  - phase: A Dphi/Dt - B div(M grad(q)) with q depending on p.
         */
        Tensor<1, dim> momentum_capillary_force;
        Tensor<1, dim> stepien_grad_q;
        double         stepien_A = 0., stepien_B = 0., bulk_viscosity = 0.;
        double         dbulk_viscosity_dphi = 0.;
        double         stepien_continuity_term = 0.;
        double         stepien_B_over_A = 0., stepien_lap_q = 0.;

        if constexpr (BaseType::with_stepien)
        {
          const double drhodphi = sd.derivative_density_wrt_tracer[q];
          const double dpr      = sd.stepien_dpr;
          const double Dphi     = dphidt + u_conv * grad_phi;

          stepien_A = 2. * sd.stepien_rho_product / (sd.stepien_rho_sum * rho);
          stepien_B = 2. / sd.stepien_rho_sum;

          // Bulk-viscosity stress lambda (div u) I temporarily disabled to
          // compare the Stepien model against Abels. Restore the Stokes
          // hypothesis below to reactivate it.
          // bulk_viscosity       = -2. / 3. * eta;
          // dbulk_viscosity_dphi =
          //   -2. / 3. * sd.derivative_dynamic_viscosity_wrt_tracer[q];

          stepien_grad_q = (rho * grad_mu - drhodphi * grad_p +
                            drhodphi * (mu - dpr) * grad_phi) /
                           sd.stepien_rho_product;

          stepien_continuity_term = (drhodphi / rho) * Dphi;

          if constexpr (BaseType::with_stepien_momentum)
          {
            // Mass residual S_c of the conservative momentum form.
            const double stepien_Sc = drhodphi * Dphi + rho * div_u;
            momentum_capillary_force = (dpr - mu) * grad_phi + stepien_Sc * u;
          }
          else
            momentum_capillary_force = diffusive_flux + phi * grad_mu;

          // B / A simplifies to rho / (rho0 rho1).
          stepien_B_over_A = rho / sd.stepien_rho_product;

          if constexpr (BaseType::with_tracer_stabilization)
            /**
             * Laplacian of the diffused quantity
             *   q = (rho mu - drho/dphi (p + p_r)) / (rho0 rho1).
             * The density is linear in the tracer, so drho/dphi is constant
             * and the product rule leaves the factor 2 on grad(phi).grad(mu).
             */
            stepien_lap_q =
              (rho * sd.potential_laplacians[q] +
               2. * drhodphi * (grad_phi * grad_mu) -
               drhodphi * sd.present_pressure_laplacians[q] +
               drhodphi * (mu - dpr) * sd.tracer_laplacians[q]) /
              sd.stepien_rho_product;
        }
        else
          momentum_capillary_force = diffusive_flux + phi * grad_mu;

        const auto to_mult_by_phi_u_i =
          rho * (dudt + grad_u * u_conv - body_force) +
          momentum_capillary_force + source_u;
        const double to_mult_by_phi_phi_i =
          BaseType::with_stepien ?
            stepien_A * (dphidt + u_conv * grad_phi) + source_phi :
            dphidt + u_conv * grad_phi + source_phi;
        const auto to_mult_by_phi_mu_i =
          mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];
        const auto &grad_phi_p     = sd.grad_phi_p[q];
        const auto &phi_phi        = sd.shape_phi[q];
        const auto &grad_phi_phi   = sd.grad_shape_phi[q];
        const auto &phi_mu         = sd.shape_mu[q];
        const auto &grad_phi_mu    = sd.grad_shape_mu[q];

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
            rho * (dudt + grad_u * u_conv - body_force) + grad_p + source_u -
            eta * (lap_u + grad_div_u) -
            2. * detadphi * (sym_grad_u * grad_phi) +
            momentum_capillary_force;

          if constexpr (BaseType::with_stepien_momentum)
            // Divergence of the bulk-viscosity stress lambda (div u) I:
            //   div(lambda (div u) I) = lambda grad(div u)
            //                           + dlambda/dphi (div u) grad(phi).
            strong_residual_momentum -=
              bulk_viscosity * grad_div_u +
              dbulk_viscosity_dphi * div_u * grad_phi;
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer = sd.tau_supg_tracer[q];

          /**
           * Strong residual of the tracer equation. The Stepien phase equation
           * is A Dphi/Dt - B div(M grad q) + source = 0; it is normalized by A
           * so that its leading term is Dphi/Dt, consistently with the way
           * tau_supg_tracer is built. Note that the source term is not scaled
           * by A in the weak form, hence the division here.
           */
          if constexpr (BaseType::with_stepien)
            strong_residual_tracer =
              dphidt + u_conv * grad_phi -
              stepien_B_over_A * mobility * stepien_lap_q +
              source_phi / stepien_A;
          else
            strong_residual_tracer = dphidt + u_conv * grad_phi -
                                     mobility * sd.potential_laplacians[q] +
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

            if constexpr (BaseType::with_stepien_momentum)
              // Bulk-viscosity stress lambda (div u) I, whose contribution is
              // non-zero because the Stepien velocity field is not solenoidal.
              local_rhs_i -= bulk_viscosity * div_u * div_phi_u[i];

            if constexpr (BaseType::with_stabilization)
              // SUPG stabilization
              local_rhs_i -=
                tau * (strong_residual_momentum * (grad_phi_u[i] * u_conv));
          }

          // Continuity equation
          else if (i_is_p)
          {
            if constexpr (BaseType::with_stepien)
              // Quasi-incompressible mass source: div(u) = -(drho/rho) Dphi/Dt
              local_rhs_i += phi_p[i] * stepien_continuity_term;

            if constexpr (BaseType::with_stabilization)
              // PSPG stabilization
              local_rhs_i +=
                tau * inv_rho * (strong_residual_momentum * grad_phi_p[i]);
          }

          // Tracer equation
          else if (i_is_phi)
          {
            local_rhs_i -= phi_phi[i] * to_mult_by_phi_phi_i;

            if constexpr (BaseType::with_stepien)
              // The Stepien phase equation diffuses q, not mu.
              local_rhs_i -=
                stepien_B * mobility * (grad_phi_phi[i] * stepien_grad_q);
            else
              local_rhs_i -= mobility * (grad_phi_phi[i] * grad_mu);

            if constexpr (BaseType::with_tracer_stabilization)
              // Tracer SUPG stabilization
              local_rhs_i -= tau_tracer * (u_conv * grad_phi_phi[i]) *
                             strong_residual_tracer;
          }

          // Potential equation
          else if (i_is_mu)
          {
            local_rhs_i -= phi_mu[i] * to_mult_by_phi_mu_i -
                           sigma_tilde_times_eps * (grad_phi_mu[i] * grad_phi);
          }

          local_rhs(i) += local_rhs_i * JxW_moving;
        }
      }
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

      const double bdf_c0                = sd.bdf_c0;
      const double mobility              = sd.mobility;
      const double sigma_tilde_over_eps  = sd.sigma_tilde / sd.epsilon;
      const double sigma_tilde_times_eps = sd.sigma_tilde * sd.epsilon;
      const double diffusive_flux_factor = sd.diffusive_flux_factor;
      const auto  &body_force            = sd.body_force;

      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_momentum(sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_mult_by_phi_u_i_potential(
        sd.dofs_per_cell);
      std::vector<double>         phi_u_j_x_grad_phi(sd.dofs_per_cell);
      std::vector<double>         to_mult_by_phi_phi_i(sd.dofs_per_cell);
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
        const auto  &source_u   = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;

        const auto  &p        = sd.present_pressure_values[q];
        const auto  &grad_p   = sd.present_pressure_gradients[q];
        const double source_p = sd.source_term_pressure[q];

        const auto  &diffusive_flux = sd.diffusive_flux[q];
        const auto  &dphidt         = sd.tracer_time_derivatives[q];
        const auto  &phi            = sd.tracer_values[q];
        const auto  &grad_phi       = sd.tracer_gradients[q];
        const auto  &mu             = sd.potential_values[q];
        const auto  &grad_mu        = sd.potential_gradients[q];
        const double source_phi     = sd.source_term_tracer[q];
        const double source_mu      = sd.source_term_potential[q];

        const auto &phi_u            = sd.phi_u[q];
        const auto &grad_phi_u       = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u   = sd.sym_grad_phi_u[q];
        const auto &div_phi_u        = sd.div_phi_u[q];
        const auto &laplacian_phi_u  = sd.laplacian_phi_u[q];
        const auto &grad_div_phi_u   = sd.grad_div_phi_u[q];
        const auto &phi_p            = sd.phi_p[q];
        const auto &grad_phi_p       = sd.grad_phi_p[q];
        const auto &phi_phi          = sd.shape_phi[q];
        const auto &grad_phi_phi     = sd.grad_shape_phi[q];
        const auto &phi_mu           = sd.shape_mu[q];
        const auto &grad_phi_mu      = sd.grad_shape_mu[q];
        const auto &laplacian_phi_mu = sd.laplacian_shape_mu[q];

        //
        // Moving mesh related data
        //
        if constexpr (BaseType::with_moving_mesh)
        {
          phi_x             = &sd.phi_x[q];
          grad_phi_x_moving = &sd.grad_phi_x_moving[q];

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
          grad_source_term_velocity = &sd.grad_source_velocity[q];
          grad_source_pressure      = &sd.grad_source_pressure[q];
          grad_source_tracer        = &sd.grad_source_tracer[q];
          grad_source_potential     = &sd.grad_source_potential[q];
#endif
        }

        // Precompute shape functions-independent terms. The trailing grad_mu
        // is the tracer derivative of the Abels capillary force phi grad(mu);
        // the Stepien capillary force is differentiated separately below.
        const Tensor<1, dim> to_mult_by_phi_u_i_phi_phi_j =
          BaseType::with_stepien_momentum ?
            drhodphi * (dudt + u_dot_grad_u_ale - body_force) :
            drhodphi * (dudt + u_dot_grad_u_ale - body_force) + grad_mu;

        const auto momentum_partial_residual =
          rho * (dudt - body_force + u_dot_grad_u_ale) + phi * grad_mu +
          source_u;
        const auto phi_partial_residual =
          dphidt + u_conv * grad_phi + source_phi;
        const auto mu_partial_residual =
          mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;
        double inv_rho      = 0.;
        double dinvrho_dphi = 0.;

        /**
         * Stepien per-quadrature coefficients, mirroring those of the rhs
         * assembler. stepien_Sc is the mass residual carried by the
         * conservative momentum form, and stepien_grad_q is the gradient of
         * the quantity diffused by the phase equation.
         */
        Tensor<1, dim> stepien_grad_q;
        double         stepien_A = 0., stepien_B = 0., stepien_Dphi = 0.;
        double         stepien_Sc = 0., stepien_inv_rho = 0.;
        double         bulk_viscosity = 0., dbulk_viscosity_dphi = 0.;
        double         stepien_B_over_A = 0., stepien_lap_q = 0.;
        double         lap_mu = 0., lap_phi = 0.;

        if constexpr (BaseType::with_stepien)
        {
          stepien_Dphi    = dphidt + u_conv * grad_phi;
          stepien_inv_rho = 1. / rho;
          stepien_A = 2. * sd.stepien_rho_product / (sd.stepien_rho_sum * rho);
          stepien_B = 2. / sd.stepien_rho_sum;

          // Bulk-viscosity stress lambda (div u) I temporarily disabled to
          // compare the Stepien model against Abels. Restore the Stokes
          // hypothesis below to reactivate it.
          // bulk_viscosity       = -2. / 3. * eta;
          // dbulk_viscosity_dphi = -2. / 3. * detadphi;

          stepien_Sc = drhodphi * stepien_Dphi + rho * div_u;

          stepien_grad_q = (rho * grad_mu - drhodphi * grad_p +
                            drhodphi * (mu - sd.stepien_dpr) * grad_phi) /
                           sd.stepien_rho_product;

          // B / A simplifies to rho / (rho0 rho1).
          stepien_B_over_A = rho / sd.stepien_rho_product;

          if constexpr (BaseType::with_tracer_stabilization)
          {
            lap_mu  = sd.potential_laplacians[q];
            lap_phi = sd.tracer_laplacians[q];

            stepien_lap_q =
              (rho * lap_mu + 2. * drhodphi * (grad_phi * grad_mu) -
               drhodphi * sd.present_pressure_laplacians[q] +
               drhodphi * (mu - sd.stepien_dpr) * lap_phi) /
              sd.stepien_rho_product;
          }
        }

        if constexpr (BaseType::with_stabilization)
        {
          tau          = sd.tau_supg_velocity[q];
          inv_rho      = 1. / rho;
          dinvrho_dphi = -drhodphi * inv_rho * inv_rho;

          // Compute strong residual of the momentum equation, in force units
          // (i.e. multiplied by the density); see the rhs assembler for the
          // corresponding test operators.
          strong_residual_momentum =
            rho * (dudt + grad_u * u_conv - body_force) + grad_p + source_u -
            eta * (lap_u + grad_div_u) -
            2. * detadphi * (sym_grad_u * grad_phi);

          strong_residual_momentum_variation_phi_phi =
            to_mult_by_phi_u_i_phi_phi_j - detadphi * (lap_u + grad_div_u);

          if constexpr (BaseType::with_stepien_momentum)
          {
            strong_residual_momentum +=
              (sd.stepien_dpr - mu) * grad_phi + stepien_Sc * u -
              bulk_viscosity * grad_div_u -
              dbulk_viscosity_dphi * div_u * grad_phi;

            /**
             * Tracer coefficient of the residual variation. Only the terms
             * proportional to dphi itself belong here; those proportional to
             * grad(dphi) are added per shape function below.
             *
             * The dynamic viscosity is linear in the tracer, so the second
             * derivative of the bulk viscosity vanishes.
             */
            strong_residual_momentum_variation_phi_phi -=
              dbulk_viscosity_dphi * grad_div_u;
            strong_residual_momentum_variation_phi_phi +=
              drhodphi * (bdf_c0 + div_u) * u;
          }
          else
            strong_residual_momentum += diffusive_flux + phi * grad_mu;
        }

        if constexpr (BaseType::with_tracer_stabilization)
        {
          tau_tracer = sd.tau_supg_tracer[q];

          // Compute strong residual of the tracer equation; see the rhs
          // assembler for the normalization used by the Stepien model.
          if constexpr (BaseType::with_stepien)
            strong_residual_tracer =
              dphidt + u_conv * grad_phi -
              stepien_B_over_A * mobility * stepien_lap_q +
              source_phi / stepien_A;
          else
            strong_residual_tracer = dphidt + u_conv * grad_phi -
                                     mobility * sd.potential_laplacians[q] +
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

          if constexpr (BaseType::with_stepien_momentum)
            // Stepien carries no diffusive inertia, and its capillary force
            // (dpr - mu) grad(phi) differentiates to -dmu grad(phi).
            to_mult_by_phi_u_i_potential[j] = -phi_mu[j] * grad_phi;
          else
          {
            to_mult_by_phi_u_i_momentum[j] +=
              diffusive_flux_factor * grad_phi_u_j * grad_mu;
            to_mult_by_phi_u_i_potential[j] =
              diffusive_flux_factor * grad_u * grad_phi_mu_j +
              phi * grad_phi_mu_j;
          }

          phi_u_j_x_grad_phi[j] = phi_u_j * grad_phi;

          if constexpr (BaseType::with_stepien)
            // A (c0 dphi + u.grad(dphi)) + dA/dphi dphi Dphi/Dt, with
            // dA/dphi = -A drho/dphi / rho.
            to_mult_by_phi_phi_i[j] =
              stepien_A * (bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j]) -
              stepien_A * drhodphi * stepien_inv_rho * phi_phi[j] *
                stepien_Dphi;
          else
            to_mult_by_phi_phi_i[j] =
              bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j];

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

            // Variation w.r.t. potential
            strong_residual_momentum_variation[j] +=
              to_mult_by_phi_u_i_potential[j];

            if constexpr (BaseType::with_stepien_momentum)
            {
              // Conservative correction S_c * u, w.r.t. the velocity in both
              // the S_c factor and the trailing u.
              strong_residual_momentum_variation[j] +=
                (drhodphi * phi_u_j_x_grad_phi[j] + rho * div_phi_u[j]) * u +
                stepien_Sc * phi_u_j;

              // Bulk-viscosity divergence, w.r.t. the velocity.
              strong_residual_momentum_variation[j] -=
                bulk_viscosity * grad_div_phi_u[j] +
                dbulk_viscosity_dphi * div_phi_u[j] * grad_phi;

              // Terms proportional to grad(dphi): capillary force, advective
              // part of S_c, and bulk-viscosity divergence.
              strong_residual_momentum_variation[j] +=
                (sd.stepien_dpr - mu) * grad_phi_phi[j] +
                drhodphi * (u_conv * grad_phi_phi[j]) * u -
                dbulk_viscosity_dphi * div_u * grad_phi_phi[j];
            }
          }

          if constexpr (BaseType::with_tracer_stabilization)
          {
            // Advective part, common to both models.
            strong_residual_tracer_variation[j] = bdf_c0 * phi_phi[j] +
                                                  phi_u_j * grad_phi +
                                                  u_conv * grad_phi_phi[j];

            if constexpr (BaseType::with_stepien)
            {
              /**
               * Variation of Delta(q), gathering the tracer, potential and
               * pressure shape functions. Only the shape function matching the
               * component of j is non-zero, so all contributions can be summed.
               */
              const double dlap_q =
                (drhodphi * phi_phi[j] * lap_mu +
                 2. * drhodphi * (grad_phi_phi[j] * grad_mu) +
                 drhodphi * (mu - sd.stepien_dpr) *
                   sd.laplacian_shape_phi[q][j] +
                 rho * laplacian_phi_mu[j] +
                 2. * drhodphi * (grad_phi * grad_phi_mu_j) +
                 drhodphi * lap_phi * phi_mu[j] -
                 drhodphi * sd.laplacian_phi_p[q][j]) /
                sd.stepien_rho_product;

              // d(B/A)/dphi = (drho/dphi) / (rho0 rho1)
              const double dB_over_A =
                drhodphi * phi_phi[j] / sd.stepien_rho_product;

              // d(1/A)/dphi = (rho0 + rho1) (drho/dphi) / (2 rho0 rho1)
              const double dinv_A = sd.stepien_rho_sum * drhodphi *
                                    phi_phi[j] /
                                    (2. * sd.stepien_rho_product);

              strong_residual_tracer_variation[j] -=
                mobility *
                (stepien_B_over_A * dlap_q + dB_over_A * stepien_lap_q);
              strong_residual_tracer_variation[j] += source_phi * dinv_A;
            }
            else
              strong_residual_tracer_variation[j] -=
                mobility * laplacian_phi_mu[j];
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

            // Variation of momentum
            to_mult_by_phi_u_i_moving_mesh[j] =
              momentum_partial_residual * trG +
              rho * (grad_u * (-bdf_c0 * phi_x_j - G * u_conv)) -
              phi * transpose_G * grad_mu +
              diffusive_flux_factor * grad_u * val * grad_mu;

            to_mult_by_grad_phi_u_i_moving_mesh[j] =
              p * transpose_G +
              2. * eta *
                (sym_grad_u * (trG * identity_tensor - transpose_G) -
                 symmetrize(grad_u_x_G_j));

            // Variation of continuity
            to_mult_by_phi_p_i_moving_mesh[j] =
              trace(grad_u_x_G_j) + (-div_u + source_p) * trG;

            // Variation of tracer
            to_mult_by_phi_phi_i_moving_mesh[j] =
              phi_partial_residual * trG - bdf_c0 * (phi_x_j * grad_phi) -
              u_conv * (transpose_G * grad_phi);

            to_mult_by_grad_phi_phi_i_moving_mesh[j] =
              mobility * (val * grad_mu);

            // Variation of potential
            to_mult_by_phi_mu_i_moving_mesh[j] = mu_partial_residual * trG;

            to_mult_by_grad_phi_mu_i_moving_mesh[j] =
              -sigma_tilde_times_eps * (val * grad_phi);

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
          const auto &phi_phi_i        = phi_phi[i];
          const auto &grad_phi_phi_i   = grad_phi_phi[i];
          const auto &phi_mu_i         = phi_mu[i];
          const auto &grad_phi_mu_i    = grad_phi_mu[i];

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

                if constexpr (BaseType::with_stepien_momentum)
                {
                  // Conservative correction (w.u) S_c, differentiated w.r.t.
                  // the velocity in both factors.
                  local_matrix_ij += (phi_u_i * phi_u[j]) * stepien_Sc;
                  local_matrix_ij +=
                    (phi_u_i * u) * (drhodphi * phi_u_j_x_grad_phi[j] +
                                     rho * div_phi_u[j]);
                  // Bulk-viscosity stress
                  local_matrix_ij +=
                    bulk_viscosity * div_phi_u[j] * div_phi_u_i;
                }
              }
              else if (j_is_phi)
              {
                local_matrix_ij +=
                  phi_phi_j * (phi_u_i * to_mult_by_phi_u_i_phi_phi_j +
                               2. * detadphi *
                                 scalar_product(sym_grad_phi_u_i, sym_grad_u));

                if constexpr (BaseType::with_stepien_momentum)
                {
                  // Capillary force (dpr - mu) grad(phi)
                  local_matrix_ij += (sd.stepien_dpr - mu) *
                                     (phi_u_i * grad_phi_phi[j]);
                  // Bulk viscosity through eta(phi)
                  local_matrix_ij += dbulk_viscosity_dphi * phi_phi_j * div_u *
                                     div_phi_u_i;
                  // Conservative correction (w.u) S_c w.r.t. the tracer
                  local_matrix_ij +=
                    (phi_u_i * u) * drhodphi *
                    (bdf_c0 * phi_phi_j + u_conv * grad_phi_phi[j] +
                     phi_phi_j * div_u);
                }
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

              if constexpr (BaseType::with_stepien)
              {
                /**
                 * The Stepien continuity equation carries the extra term
                 * -(drho/dphi / rho) Dphi/Dt, which couples the pressure test
                 * function to the velocity (through u.grad(phi)) and to the
                 * tracer (through 1/rho and Dphi/Dt).
                 */
                if (j_is_u)
                  matrix_row[j] += -phi_p_i * drhodphi * stepien_inv_rho *
                                   phi_u_j_x_grad_phi[j] * JxW_moving;

                if (this->ordering.is_tracer(comp_j))
                  matrix_row[j] +=
                    phi_p_i *
                    (drhodphi * drhodphi * stepien_inv_rho * stepien_inv_rho *
                       phi_phi[j] * stepien_Dphi -
                     drhodphi * stepien_inv_rho *
                       (bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j])) *
                    JxW_moving;
              }

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
                // Stepien scales the advection by the coefficient A.
                const double advection =
                  BaseType::with_stepien ?
                    stepien_A * phi_u_j_x_grad_phi[j] :
                    phi_u_j_x_grad_phi[j];
                matrix_row[j] += phi_phi_i * advection * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  phi_phi_i * to_mult_by_phi_phi_i[j] * JxW_moving;

                if constexpr (BaseType::with_stepien)
                {
                  // Tracer part of B M delta(grad q)
                  const Tensor<1, dim> delta_grad_q_phi_j =
                    (drhodphi * phi_phi[j] * grad_mu +
                     drhodphi * (mu - sd.stepien_dpr) * grad_phi_phi[j]) /
                    sd.stepien_rho_product;
                  matrix_row[j] += stepien_B * mobility *
                                   (grad_phi_phi_i * delta_grad_q_phi_j) *
                                   JxW_moving;
                }
              }
              else if (j_is_mu)
              {
                if constexpr (BaseType::with_stepien)
                {
                  // Potential part of B M delta(grad q)
                  const Tensor<1, dim> delta_grad_q_mu_j =
                    (rho * grad_phi_mu_j + drhodphi * phi_mu[j] * grad_phi) /
                    sd.stepien_rho_product;
                  matrix_row[j] += stepien_B * mobility *
                                   (grad_phi_phi_i * delta_grad_q_mu_j) *
                                   JxW_moving;
                }
                else
                  matrix_row[j] +=
                    mobility * (grad_phi_mu_j * grad_phi_phi_i) * JxW_moving;
              }

              if constexpr (BaseType::with_stepien)
                if (this->ordering.is_pressure(comp_j))
                {
                  /**
                   * Pressure part of B M delta(grad q). This tracer-pressure
                   * block has no counterpart in the Abels model: it exists
                   * only because the Stepien phase equation diffuses a
                   * quantity q that depends on the pressure.
                   */
                  const Tensor<1, dim> delta_grad_q_p_j =
                    -drhodphi * grad_phi_p[j] / sd.stepien_rho_product;
                  matrix_row[j] += stepien_B * mobility *
                                   (grad_phi_phi_i * delta_grad_q_p_j) *
                                   JxW_moving;
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
                // Mass
                matrix_row[j] += phi_mu_i * phi_mu_j * JxW_moving;
              }
              else if (j_is_phi)
              {
                matrix_row[j] +=
                  (-sigma_tilde_over_eps * phi_mu_i * phi_phi_j *
                     (3. * phi * phi - 1.) -
                   sigma_tilde_times_eps * (grad_phi_mu_i * grad_phi_phi_j)) *
                  JxW_moving;
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
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

// Explicit instantiations
#include "incompressible_chns_assemblers.inst"
