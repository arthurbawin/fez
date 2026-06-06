
#include <assembly/incompressible_ns_assemblers.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <parameter_reader.h>
#include <scratch_data.h>

namespace Assembly
{
  namespace IncompressibleNavierStokes
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
      auto &sd        = scratch_data;
      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      const double                  nu = sd.kinematic_viscosity;
      const SymmetricTensor<2, dim> identity_tensor =
        unit_symmetric_tensor<dim>();

      //
      // Pseudo-solid related data
      //
      double                             JxW_fixed, lame_mu, lame_lambda;
      double                             present_displacement_divergence;
      double                             present_trace_strain;
      const Tensor<2, dim>              *present_position_gradients;
      Tensor<2, dim>                     present_strain;
      const Tensor<1, dim>              *source_term_position;
      const std::vector<Tensor<1, dim>> *phi_x;
      const std::vector<SymmetricTensor<2, dim>> *sym_grad_phi_x;
      const std::vector<double>                  *div_phi_x;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        //
        // Flow related data
        //
        const double JxW_moving = sd.JxW_moving[q];

        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &source_u   = sd.source_term_velocity[q];
        const auto  &dudt       = sd.present_velocity_time_derivatives[q];

        auto u_conv = u;
        if constexpr (this->with_pseudo_solid)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;
        const auto to_multiply_by_phi_u_i =
          (dudt + u_dot_grad_u_ale + source_u);

        const auto &p        = sd.present_pressure_values[q];
        const auto &source_p = sd.source_term_pressure[q];

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];

        //
        // Pseudo-solid related data
        //
        if constexpr (this->with_pseudo_solid)
        {
          JxW_fixed   = sd.JxW_fixed[q];
          lame_mu     = sd.lame_mu[q];
          lame_lambda = sd.lame_lambda[q];

          present_position_gradients      = &sd.present_position_gradients[q];
          present_displacement_divergence = trace(*present_position_gradients);
          present_strain =
            symmetrize(*present_position_gradients) - identity_tensor;
          present_trace_strain = present_displacement_divergence - (double)dim;

          source_term_position = &sd.source_term_position[q];

          phi_x          = &sd.phi_x[q];
          sym_grad_phi_x = &sd.sym_grad_phi_x[q];
          div_phi_x      = &sd.div_phi_x[q];
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i = sd.components[i];
          const bool         i_is_u =
            this->ordering.u_lower <= comp_i && comp_i < this->ordering.u_upper;
          const bool i_is_p = comp_i == this->ordering.p_lower;
          const bool i_is_x =
            this->ordering.x_lower <= comp_i && comp_i < this->ordering.x_upper;
          const bool i_is_l =
            this->ordering.l_lower <= comp_i && comp_i < this->ordering.l_upper;

          if (i_is_l)
            continue;

          //
          // Flow residual
          //
          double local_rhs_flow_i =
            i_is_p ? -phi_p[i] * (-div_u + source_p) : 0.;

          if (i_is_u)
          {
            local_rhs_flow_i -= (
              // Time derivative, convective acceleration and velocity source
              // term
              phi_u[i] * to_multiply_by_phi_u_i

              // Pressure gradient
              - div_phi_u[i] * p);

            // Diffusion
            // FIXME: more efficient double contraction
            if constexpr (this->with_divergence_form)
              local_rhs_flow_i -=
                2. * nu * scalar_product(sym_grad_u, sym_grad_phi_u[i]);
            else
              local_rhs_flow_i -= nu * scalar_product(grad_u, grad_phi_u[i]);
          }

          local_rhs_flow_i *= JxW_moving;

          //
          // Pseudo-solid residual
          //
          double local_rhs_ps_i = 0.;

          if constexpr (this->with_pseudo_solid)
          {
            if (i_is_x)
            {
              // Linear elasticity and source term
              // FIXME: more efficient double contraction
              local_rhs_ps_i -=
                (lame_lambda * present_trace_strain * (*div_phi_x)[i] +
                 2. * lame_mu *
                   scalar_product(present_strain, (*sym_grad_phi_x)[i]) +
                 (*phi_x)[i] * (*source_term_position));
              local_rhs_ps_i *= JxW_fixed;
            }
          }

          local_rhs(i) += local_rhs_flow_i + local_rhs_ps_i;
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

      const double bdf_c0 = sd.bdf_c0;
      const double nu     = sd.kinematic_viscosity;

      // FIXME: add to scratch?
      std::vector<Tensor<1, dim>> to_multiply_by_phi_u_i_momentum(
        sd.dofs_per_cell);
      std::vector<Tensor<1, dim>> to_multiply_by_phi_u_i_position(
        sd.dofs_per_cell);
      std::vector<double> trace_gradu_dot_grad_phi_x_moving_j(sd.dofs_per_cell);
      std::vector<Tensor<2, dim>> gradu_dot_grad_phi_x_moving_j(
        sd.dofs_per_cell);

      const auto u_lower = this->ordering.u_lower;
      const auto x_lower = this->ordering.x_lower;

      //
      // Pseudo-solid related data
      //
      double                             JxW_fixed, lame_mu, lame_lambda;
      const std::vector<Tensor<1, dim>> *phi_x;
      const std::vector<Tensor<2, dim>> *grad_phi_x;
      const Tensor<2, dim>              *grad_phi_x_i;
      const std::vector<Tensor<2, dim>> *grad_phi_x_moving;
      const std::vector<SymmetricTensor<2, dim>> *sym_grad_phi_x;
      const std::vector<double>                  *div_phi_x;
      double                                      div_phi_x_i;
      double                trace_sym_grad_u_dot_sym_grad_phi_u_i;
      const Tensor<1, dim> *source_term_velocity;
      double                source_term_pressure;

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
      const Tensor<2, dim> *grad_source_term_velocity;
      const Tensor<1, dim> *grad_source_term_pressure;
#endif

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        //
        // Flow related data
        //
        const double JxW_moving = sd.JxW_moving[q];

        const auto  &u          = sd.present_velocity_values[q];
        const auto  &grad_u     = sd.present_velocity_gradients[q];
        const auto  &sym_grad_u = sd.present_velocity_sym_gradients[q];
        const double div_u      = sd.present_velocity_divergence[q];
        const auto  &dudt       = sd.present_velocity_time_derivatives[q];

        auto u_conv = u;
        if constexpr (this->with_pseudo_solid)
        {
          const auto &dxdt = sd.present_mesh_velocity_values[q];
          u_conv -= dxdt;
        }
        const auto u_dot_grad_u_ale = grad_u * u_conv;

        const double p = sd.present_pressure_values[q];

        const auto &phi_u          = sd.phi_u[q];
        const auto &grad_phi_u     = sd.grad_phi_u[q];
        const auto &sym_grad_phi_u = sd.sym_grad_phi_u[q];
        const auto &div_phi_u      = sd.div_phi_u[q];
        const auto &phi_p          = sd.phi_p[q];

        //
        // Pseudo-solid related data
        //
        if constexpr (this->with_pseudo_solid)
        {
          JxW_fixed   = sd.JxW_fixed[q];
          lame_mu     = sd.lame_mu[q];
          lame_lambda = sd.lame_lambda[q];

          phi_x             = &sd.phi_x[q];
          grad_phi_x        = &sd.grad_phi_x[q];
          grad_phi_x_moving = &sd.grad_phi_x_moving[q];
          sym_grad_phi_x    = &sd.sym_grad_phi_x[q];
          div_phi_x         = &sd.div_phi_x[q];

          source_term_velocity = &sd.source_term_velocity[q];
          source_term_pressure = sd.source_term_pressure[q];
#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
          grad_source_term_velocity = &sd.grad_source_velocity[q];
          grad_source_term_pressure = &sd.grad_source_pressure[q];
#endif
        }

        // Precompute quantities depending only on j
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto &phi_u_j      = phi_u[j];
          const auto &grad_phi_u_j = grad_phi_u[j];

          to_multiply_by_phi_u_i_momentum[j] =
            bdf_c0 * phi_u_j + grad_phi_u_j * u_conv + grad_u * phi_u_j;

          if constexpr (this->with_pseudo_solid)
          {
            const auto &phi_x_j = (*phi_x)[j];
            const auto &G       = (*grad_phi_x_moving)[j];
            const auto  trG     = trace(G);

            gradu_dot_grad_phi_x_moving_j[j] = grad_u * G;
            trace_gradu_dot_grad_phi_x_moving_j[j] =
              trace(gradu_dot_grad_phi_x_moving_j[j]);

            to_multiply_by_phi_u_i_position[j] =
              (dudt + u_dot_grad_u_ale + *source_term_velocity) * trG -
              gradu_dot_grad_phi_x_moving_j[j] * u_conv -
              grad_u * bdf_c0 * phi_x_j;

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
            to_multiply_by_phi_u_i_position[j] +=
              (*grad_source_term_velocity) * phi_x_j;
#endif
          }
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i = sd.components[i];

          const bool i_is_l = this->ordering.is_lambda(comp_i);
          if (i_is_l)
            continue;
          const bool i_is_u = this->ordering.is_velocity(comp_i);
          const bool i_is_p = this->ordering.is_pressure(comp_i);
          const bool i_is_x = this->ordering.is_position(comp_i);
          // if (!(i_is_u or i_is_p or i_is_x))
          //   continue;

          const auto &coupling_row = this->coupling_table[comp_i];

          const auto &phi_u_i          = phi_u[i];
          const auto &grad_phi_u_i     = grad_phi_u[i];
          const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
          const auto &div_phi_u_i      = div_phi_u[i];
          const auto &phi_p_i          = phi_p[i];

          if constexpr (this->with_pseudo_solid)
          {
            grad_phi_x_i = &(*grad_phi_x)[i];
            div_phi_x_i  = (*div_phi_x)[i];

            trace_sym_grad_u_dot_sym_grad_phi_u_i =
              trace(sym_grad_u * Tensor<2, dim>(sym_grad_phi_u_i));
          }

          for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
          {
            const unsigned int comp_j = sd.components[j];

            // If lambda dof, continue before reading the coupling table
            const bool j_is_l = this->ordering.is_lambda(comp_j);
            if (j_is_l)
              continue;
            if (coupling_row[comp_j] != DoFTools::always)
              continue;
            const bool j_is_u = this->ordering.is_velocity(comp_j);
            const bool j_is_p = this->ordering.is_pressure(comp_j);
            const bool j_is_x = this->ordering.is_position(comp_j);

            // Account for the pressure gradient when initializing
            double local_flow_matrix_ij = j_is_p ? -div_phi_u_i * phi_p[j] : 0.;

            /**
             * Momentum equation
             */
            if (i_is_u)
            {
              if (j_is_u)
              {
                // Time derivative and convection (including ALE)
                local_flow_matrix_ij +=
                  phi_u_i * to_multiply_by_phi_u_i_momentum[j];

                // Diffusion
                if constexpr (this->with_divergence_form)
                {
                  // The following is the diffusion term
                  // 2. * nu * scalar_product(sym_grad_phi_u[j],
                  // sym_grad_phi_u_i), explicited for the symmetric gradient of
                  // Lagrange shape functions
                  const auto &gui = grad_phi_u_i[comp_i];
                  const auto &guj = grad_phi_u[j][comp_j];
                  local_flow_matrix_ij +=
                    nu * (gui[comp_j - u_lower] * guj[comp_i - u_lower]);
                  if (comp_i == comp_j)
                    local_flow_matrix_ij += nu * gui * guj;
                }
                else
                {
                  // Diffusion term for nu * grad * grad (laplacian form)
                  // FIXME: more efficient double contraction
                  local_flow_matrix_ij +=
                    nu * scalar_product(grad_phi_u_i, grad_phi_u[j]);
                }
              }

              if constexpr (this->with_pseudo_solid)
                if (j_is_x)
                {
                  const auto &G   = (*grad_phi_x_moving)[j];
                  const auto  trG = trace(G);

                  /**
                   * Variation of momentum terms on moving mesh w.r.t. position.
                   */
                  local_flow_matrix_ij +=
                    phi_u_i * to_multiply_by_phi_u_i_position[j] -
                    p * (trace(-grad_phi_u_i * G) + div_phi_u_i * trG);

                  // Diffusion term
                  if constexpr (this->with_divergence_form)
                  {
                    // FIXME: it may be possible to reduce this to a single
                    // double contraction (scalar_product), or even explicit the
                    // double contraction for efficiency.
                    local_flow_matrix_ij +=
                      2. * nu *
                      (scalar_product(-symmetrize(grad_phi_u_i * G),
                                      sym_grad_u) +
                       scalar_product(sym_grad_phi_u_i,
                                      -symmetrize(
                                        gradu_dot_grad_phi_x_moving_j[j])) +
                       trace_sym_grad_u_dot_sym_grad_phi_u_i * trG);
                  }
                  else
                  {
                    // Variation of the diffusion term in Laplacian form.
                    // Currently all solvers with a moving mesh involved use a
                    // Lagrange multiplier, and thus use the divergence
                    // formulation of the NS equations.
                    DEAL_II_NOT_IMPLEMENTED();
                  }
                }
            }

            /**
             * Continuity equation
             */
            if (i_is_p)
            {
              if (j_is_u)
              {
                // Continuity : variation w.r.t. u
                local_flow_matrix_ij += -phi_p_i * div_phi_u[j];
              }

              if constexpr (this->with_pseudo_solid)
                if (j_is_x)
                {
                  const auto &G   = (*grad_phi_x_moving)[j];
                  const auto  trG = trace(G);

                  // Continuity : variation w.r.t. mesh position x.
                  local_flow_matrix_ij +=
                    phi_p_i *
                    ((-div_u + source_term_pressure) * trG + trace(grad_u * G));

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
                  local_flow_matrix_ij +=
                    phi_p_i * (*grad_source_term_pressure) * (*phi_x)[j];
#endif
                }
            }

            /**
             * Pseudo-solid equation
             */
            double local_ps_matrix_ij = 0.;
            if constexpr (this->with_pseudo_solid)
              if (i_is_x && j_is_x)
              {
                const auto &gxi = (*grad_phi_x_i)[comp_i - x_lower];
                const auto &gxj = (*grad_phi_x)[j][comp_j - x_lower];

                // Linear elasticity
                local_ps_matrix_ij +=
                  lame_lambda * (*div_phi_x)[j] * div_phi_x_i

                  // The following is the double contraction
                  // 2. * lame_mu * scalar_product(sym_grad_phi_x_j,
                  // sym_grad_phi_x_i) explicited for the symmetric gradient of
                  // Lagrange shape functions
                  + lame_mu * gxi[comp_j - x_lower] * gxj[comp_i - x_lower];
                if (comp_i == comp_j)
                  local_ps_matrix_ij += lame_mu * gxi * gxj;

                local_ps_matrix_ij *= JxW_fixed;
              }

            local_flow_matrix_ij *= JxW_moving;
            local_matrix(i, j) += local_flow_matrix_ij + local_ps_matrix_ij;
          }
        }
      }
    }

    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void MMSTractionAssembler<dim, ScratchData, CopyData, assembly_flags>::
      assemble_rhs(const ScratchData &scratch_data, CopyData &copy_data) const
    {
      if (!copy_data.cell_is_at_boundary)
        return;

      auto &sd = scratch_data;

      for (unsigned int i_face = 0; i_face < sd.n_faces; ++i_face)
        if (sd.face_at_boundary[i_face])
        {
          const auto &fluid_bc = param.fluid_bc.at(sd.face_boundary_id[i_face]);

          // Traction/open boundary condition with prescribed manufactured
          // solution
          if (fluid_bc.type == BoundaryConditions::Type::open_mms)
          {
            auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

            const double nu = sd.kinematic_viscosity;

            for (unsigned int q = 0; q < sd.n_faces_q_points; ++q)
            {
              const double face_JxW_moving = sd.face_JxW_moving[i_face][q];
              const auto  &n               = sd.face_normals_moving[i_face][q];

              const auto &grad_u_exact =
                sd.exact_face_velocity_gradients[i_face][q];
              const double p_exact = sd.exact_face_pressure_values[i_face][q];

              /**
               * For the Navier-Stokes in Laplacian form (without the grad(div)
               * term), assemble only the term below. This is then an open
               * boundary condition, not a traction, involving only grad_u_exact
               * and not the symmetric gradient.
               */
              auto sigma_dot_n = -p_exact * n + nu * grad_u_exact * n;

              if constexpr (this->with_divergence_form)
                /*
                 * For the Navier-Stokes with full divergence of stress tensor,
                 * complete with the missing term (this is then a "real"
                 * traction boundary condition).
                 */
                sigma_dot_n += nu * transpose(grad_u_exact) * n;

              const auto &phi_u = sd.phi_u_face[i_face][q];

              for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
                local_rhs(i) -= -phi_u[i] * sigma_dot_n * face_JxW_moving;
            }
          }
        }
    }
  } // namespace IncompressibleNavierStokes
} // namespace Assembly

// Explicit instantiations
#include "incompressible_ns_assemblers.inst"
