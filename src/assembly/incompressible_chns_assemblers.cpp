
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

      // //
      // // SUPG data
      // //
      // Tensor<1, dim> strong_residual_momentum;
      // double         tau;

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
        // const auto  &lap_u      = sd.present_velocity_laplacians[q];
        // const auto  &grad_div_u = sd.present_velocity_grad_div[q];
        const auto &source_u = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          // ALE contribution
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }

        const auto &p = sd.present_pressure_values[q];
        // const auto &grad_p   = sd.present_pressure_gradients[q];
        const auto &source_p = sd.source_term_pressure[q];

        const auto &diffusive_flux = sd.diffusive_flux[q];
        const auto &dphidt         = sd.tracer_time_derivatives[q];
        const auto &phi            = sd.tracer_values[q];
        const auto &grad_phi       = sd.tracer_gradients[q];
        const auto &mu             = sd.potential_values[q];
        const auto &grad_mu        = sd.potential_gradients[q];
        const auto &source_phi     = sd.source_term_tracer[q];
        const auto &source_mu      = sd.source_term_potential[q];

        const auto to_mult_by_phi_u_i =
          rho * (dudt + grad_u * u_conv - body_force) + diffusive_flux +
          phi * grad_mu + source_u;
        const auto to_mult_by_phi_phi_i =
          dphidt + u_conv * grad_phi + source_phi;
        const auto to_mult_by_phi_mu_i =
          mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;

        const auto &phi_u = sd.phi_u[q];
        // const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];
        // const auto &grad_phi_p     = sd.grad_phi_p[q];
        const auto &phi_phi      = sd.shape_phi[q];
        const auto &grad_phi_phi = sd.grad_shape_phi[q];
        const auto &phi_mu       = sd.shape_mu[q];
        const auto &grad_phi_mu  = sd.grad_shape_mu[q];

        //   if constexpr (BaseType::with_stabilization)
        //   {
        //     tau = sd.tau_supg_velocity[q];

        //     // Compute strong residual of the Navier-Stokes equations
        //     strong_residual_momentum =
        //       dudt + grad_u * u_conv + grad_p - nu * lap_u + source_u;

        //     if constexpr (BaseType::with_divergence_form)
        //       strong_residual_momentum -= nu * grad_div_u;
        //   }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i   = sd.components[i];
          const bool         i_is_u   = this->ordering.is_velocity(comp_i);
          const bool         i_is_p   = this->ordering.is_pressure(comp_i);
          const bool         i_is_phi = this->ordering.is_tracer(comp_i);
          const bool         i_is_mu  = this->ordering.is_potential(comp_i);

          // const auto &grad_phi_u_i = grad_phi_u[i];

          double local_rhs_i = i_is_p ? -phi_p[i] * (-div_u + source_p) : 0.;

          // Momentum equation
          if (i_is_u)
          {
            local_rhs_i -=
              phi_u[i] * to_mult_by_phi_u_i - div_phi_u[i] * p +
              2. * eta * scalar_product(sym_grad_phi_u[i], sym_grad_u);

            // if constexpr (BaseType::with_stabilization)
            // {
            //   // SUPG stabilization
            //   local_rhs_i -=
            //     tau * strong_residual_momentum * (grad_phi_u_i * u_conv);
            // }
          }

          // Continuity equation
          // if constexpr (BaseType::with_stabilization)
          // {
          //   // PSPG stabilization
          //   if (i_is_p)
          //     local_rhs_i -= -tau * strong_residual_momentum * grad_phi_p[i];
          // }

          // Tracer equation
          else if (i_is_phi)
          {
            local_rhs_i -= phi_phi[i] * to_mult_by_phi_phi_i +
                           mobility * (grad_phi_phi[i] * grad_mu);
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
      std::vector<double> phi_u_j_x_grad_phi(sd.dofs_per_cell);
      std::vector<double> to_mult_by_phi_phi_i(sd.dofs_per_cell);
      // std::vector<Tensor<1, dim>>
      // strong_residual_momentum_variation(sd.dofs_per_cell);

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

      // //
      // // SUPG data
      // //
      // Tensor<1, dim> strong_residual_momentum;
      // double         tau;
      // Tensor<1, dim> u_conv_dot_grad_phi_u_i,
      // residual_dot_grad_phi_u_i;

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
        // const auto  &lap_u      = sd.present_velocity_laplacians[q];
        // const auto  &grad_div_u = sd.present_velocity_grad_div[q];
        const auto &source_u = sd.source_term_velocity[q];

        auto u_conv = u;
        if constexpr (BaseType::with_moving_mesh)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;

        const auto &p = sd.present_pressure_values[q];
        // const auto &grad_p   = sd.present_pressure_gradients[q];
        const double source_p = sd.source_term_pressure[q];

        // const auto &diffusive_flux = sd.diffusive_flux[q];
        const auto  &dphidt     = sd.tracer_time_derivatives[q];
        const auto  &phi        = sd.tracer_values[q];
        const auto  &grad_phi   = sd.tracer_gradients[q];
        const auto  &mu         = sd.potential_values[q];
        const auto  &grad_mu    = sd.potential_gradients[q];
        const double source_phi = sd.source_term_tracer[q];
        const double source_mu  = sd.source_term_potential[q];

        // if constexpr (BaseType::with_stabilization)
        // {
        //   tau = sd.tau_supg_velocity[q];

        //   // Compute strong residual of the Navier-Stokes equations
        //   strong_residual_momentum =
        //     dudt + grad_u * u_conv + grad_p - nu * lap_u + source_u;

        //   if constexpr (BaseType::with_divergence_form)
        //     strong_residual_momentum -= nu * grad_div_u;
        // }

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        // const auto &laplacian_phi_u = sd.laplacian_phi_u[q];
        // const auto &grad_div_phi_u  = sd.grad_div_phi_u[q];
        const auto &phi_p = sd.phi_p[q];
        // const auto &grad_phi_p      = sd.grad_phi_p[q];
        const auto &phi_phi      = sd.shape_phi[q];
        const auto &grad_phi_phi = sd.grad_shape_phi[q];
        const auto &phi_mu       = sd.shape_mu[q];
        const auto &grad_phi_mu  = sd.grad_shape_mu[q];

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

        // Precompute shape functions-independent terms
        const auto to_mult_by_phi_u_i_phi_phi_j =
          (drhodphi * (dudt + u_dot_grad_u_ale - body_force) + grad_mu);

        const auto momentum_partial_residual =
          rho * (dudt - body_force + u_dot_grad_u_ale) + phi * grad_mu +
          source_u;
        const auto phi_partial_residual =
          dphidt + u_conv * grad_phi + source_phi;
        const auto mu_partial_residual =
          mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) + source_mu;

        // Precompute quantities depending only on j
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto &phi_u_j       = phi_u[j];
          const auto &grad_phi_u_j  = grad_phi_u[j];
          const auto &grad_phi_mu_j = grad_phi_mu[j];

          to_mult_by_phi_u_i_momentum[j] =
            rho *
              (bdf_c0 * phi_u_j + grad_phi_u_j * u_conv + grad_u * phi_u_j) +
            diffusive_flux_factor * grad_phi_u_j * grad_mu;

          to_mult_by_phi_u_i_potential[j] =
            diffusive_flux_factor * grad_u * grad_phi_mu_j +
            phi * grad_phi_mu_j;

          phi_u_j_x_grad_phi[j] = phi_u_j * grad_phi;

          to_mult_by_phi_phi_i[j] =
            bdf_c0 * phi_phi[j] + u_conv * grad_phi_phi[j];

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

          // if constexpr (BaseType::with_stabilization)
          // {
          //   strong_residual_momentum_variation[j] =
          //     to_mult_by_phi_u_i_momentum[j] + grad_phi_p[j] -
          //     nu * laplacian_phi_u[j];

          //   if constexpr (BaseType::with_divergence_form)
          //     strong_residual_momentum_variation[j] += -nu *
          //     grad_div_phi_u[j];
          // }
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
          // const auto &grad_phi_p_i     = grad_phi_p[i];
          const auto &phi_phi_i      = phi_phi[i];
          const auto &grad_phi_phi_i = grad_phi_phi[i];
          const auto &phi_mu_i       = phi_mu[i];
          const auto &grad_phi_mu_i  = grad_phi_mu[i];

          /**
           * Momentum equation
           */
          if (this->ordering.is_velocity(comp_i))
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
                local_matrix_ij +=
                  phi_phi_j * (phi_u_i * to_mult_by_phi_u_i_phi_phi_j +
                               2. * detadphi *
                                 scalar_product(sym_grad_phi_u_i, sym_grad_u));
              else if (j_is_mu)
                local_matrix_ij += phi_u_i * to_mult_by_phi_u_i_potential[j];

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
              }
              else if (j_is_mu)
              {
                matrix_row[j] +=
                  mobility * (grad_phi_mu_j * grad_phi_phi_i) * JxW_moving;
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
