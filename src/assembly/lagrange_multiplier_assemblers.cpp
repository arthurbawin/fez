
#include <assembly/lagrange_multiplier_assemblers.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <parameter_reader.h>
#include <scratch_data.h>

namespace Assembly
{
  namespace LagrangeMultiplier
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    void WeakNoSlipAssembler<dim, ScratchData, CopyData, with_moving_mesh>::
      assemble_rhs(const ScratchData &scratch_data, CopyData &copy_data) const
    {
      if (!copy_data.cell_has_lagrange_multiplier)
        return;
      if (!copy_data.cell_is_at_boundary)
        return;

      auto &sd = scratch_data;

      for (unsigned int i_face = 0; i_face < sd.n_faces; ++i_face)
        if (sd.face_at_boundary[i_face])
        {
          const auto &fluid_bc = param.fluid_bc.at(sd.face_boundary_id[i_face]);

          if (fluid_bc.type == BoundaryConditions::Type::weak_no_slip)
          {
            auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

            const auto &rotation_velocities =
              sd.input_face_rigid_body_rotation_velocity[i_face];

            for (unsigned int q = 0; q < sd.n_faces_q_points; ++q)
            {
              const double face_JxW_moving = sd.face_JxW_moving[i_face][q];

              const auto &phi_u = sd.phi_u_face[i_face][q];
              const auto &phi_l = sd.phi_l_face[i_face][q];

              const auto &fluid_velocity =
                sd.present_face_velocity_values[i_face][q];
              const auto &lambda = sd.present_face_lambda_values[i_face][q];

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
              const auto &face_velocity_source_term =
                sd.face_velocity_source_term[i_face][q];
#endif

              // Compute the no-slip velocity constraint, which may include
              // matching the velocity with the moving mesh and/or a rigid
              // rotation.
              auto velocity_constraint = fluid_velocity;
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
              Tensor<1, dim> exact_constraint =
                sd.exact_face_velocity_values[i_face][q];
#endif

              if constexpr (with_moving_mesh)
              {
                const auto &mesh_velocity =
                  sd.present_face_mesh_velocity_values[i_face][q];
                velocity_constraint -= mesh_velocity;

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
                exact_constraint -=
                  sd.exact_face_mesh_velocity_values[i_face][q];
#endif
              }

              if (fluid_bc.enable_rigid_body_rotation)
                velocity_constraint -= rotation_velocities[q];

              for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
              {
                double local_rhs_i = 0.;

                const unsigned int comp_i = sd.components[i];
                const bool         i_is_u = this->ordering.is_velocity(comp_i);
                const bool         i_is_l = this->ordering.is_lambda(comp_i);

                if (i_is_u)
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
                  local_rhs_i -=
                    -phi_u[i] * (lambda - face_velocity_source_term);
#else
                  local_rhs_i -= -phi_u[i] * lambda;
#endif

                if (i_is_l)
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
                  local_rhs_i -=
                    -(velocity_constraint - exact_constraint) * phi_l[i];
#else
                  local_rhs_i -= -velocity_constraint * phi_l[i];
#endif

                local_rhs(i) += local_rhs_i * face_JxW_moving;
              }
            }
          }
        }
    }

    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    void WeakNoSlipAssembler<dim, ScratchData, CopyData, with_moving_mesh>::
      assemble_matrix(const ScratchData &scratch_data,
                      CopyData          &copy_data) const
    {
      if (!copy_data.cell_has_lagrange_multiplier)
        return;
      if (!copy_data.cell_is_at_boundary)
        return;

      auto &sd = scratch_data;

      for (unsigned int i_face = 0; i_face < sd.n_faces; ++i_face)
        if (sd.face_at_boundary[i_face])
        {
          const auto &fluid_bc = param.fluid_bc.at(sd.face_boundary_id[i_face]);

          if (fluid_bc.type == BoundaryConditions::Type::weak_no_slip)
          {
            auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

            const unsigned int u_lower = this->ordering.u_lower;
            const unsigned int u_upper = this->ordering.u_upper;
            const unsigned int x_lower = this->ordering.x_lower;
            const unsigned int x_upper = this->ordering.x_upper;
            const unsigned int l_lower = this->ordering.l_lower;
            const unsigned int l_upper = this->ordering.l_upper;

            const Tensor<1, dim>       *lambda;
            const std::vector<double>  *delta_dx;
            double                      lambda_dot_phi_u_i;
            std::vector<Tensor<1, dim>> to_multiply_by_phi_l_i(
              sd.dofs_per_cell);

            const double bdf_c0 = sd.bdf_c0;

            for (unsigned int q = 0; q < sd.n_faces_q_points; ++q)
            {
              const double face_JxW_moving = sd.face_JxW_moving[i_face][q];

              const auto &phi_u = sd.phi_u_face[i_face][q];
              const auto &phi_l = sd.phi_l_face[i_face][q];

              // Precomputations
              if constexpr (with_moving_mesh)
              {
                const auto &phi_x = sd.phi_x_face[i_face][q];
                delta_dx          = &sd.delta_dx[i_face][q];
                const auto u_ale =
                  sd.present_face_velocity_values[i_face][q] -
                  sd.present_face_mesh_velocity_values[i_face][q];
                lambda = &sd.present_face_lambda_values[i_face][q];

                for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
                  to_multiply_by_phi_l_i[j] =
                    bdf_c0 * phi_x[j] - u_ale * (*delta_dx)[j];
              }

              for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
              {
                const unsigned int comp_i = sd.components[i];
                const bool i_is_u = u_lower <= comp_i && comp_i < u_upper;
                const bool i_is_l = l_lower <= comp_i && comp_i < l_upper;

                const auto &phi_u_i = phi_u[i];
                const auto &phi_l_i = phi_l[i];

                if constexpr (with_moving_mesh)
                  lambda_dot_phi_u_i = *lambda * phi_u_i;

                for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
                {
                  const unsigned int comp_j = sd.components[j];
                  const bool j_is_u = u_lower <= comp_j && comp_j < u_upper;
                  const bool j_is_l = l_lower <= comp_j && comp_j < l_upper;
                  const bool j_is_x = x_lower <= comp_j && comp_j < x_upper;

                  if constexpr (with_moving_mesh)
                  {
                    const bool assemble = (i_is_u and (j_is_x or j_is_l)) or
                                          (i_is_l and (j_is_u or j_is_x));
                    if (!assemble)
                      continue;
                  }
                  else
                  {
                    const bool assemble =
                      (i_is_u and j_is_l) or (i_is_l and j_is_u);
                    if (!assemble)
                      continue;
                  }

                  const auto &phi_u_j = phi_u[j];
                  const auto &phi_l_j = phi_l[j];

                  double local_matrix_ij = 0.;

                  if (i_is_u && j_is_l)
                    local_matrix_ij += -phi_l_j * phi_u_i;

                  if (i_is_l && j_is_u)
                    local_matrix_ij += -phi_u_j * phi_l_i;

                  if constexpr (with_moving_mesh)
                  {
                    if (j_is_x)
                    {
                      if (i_is_u)
                        local_matrix_ij += -lambda_dot_phi_u_i * (*delta_dx)[j];
                      if (i_is_l)
                        local_matrix_ij += phi_l_i * to_multiply_by_phi_l_i[j];
                    }
                  }
                  local_matrix(i, j) += local_matrix_ij * face_JxW_moving;
                }
              }
            }
          }
        }
    }
  } // namespace LagrangeMultiplier
} // namespace Assembly

// Explicit instantiations
#include "lagrange_multiplier_assemblers.inst"
