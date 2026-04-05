#ifndef ASSEMBLY_LAGRANGE_MULTIPLIER_H
#define ASSEMBLY_LAGRANGE_MULTIPLIER_H

#include <boundary_conditions.h>
#include <components_ordering.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <scratch_data.h>
#include <time_handler.h>

using namespace dealii;

namespace Assembly
{
  /**
   * - The scratch must have been reinited on the cell
   */
  template <bool with_moving_mesh = false, int dim, typename ScratchData>
  void weakly_enforced_no_slip_rhs(
    const ComponentOrdering                &component_ordering,
    const unsigned int                      i_face,
    const BoundaryConditions::FluidBC<dim> &fluid_boundary_condition,
    const ScratchData                      &scratch_data,
    Vector<double>                         &local_rhs);

  /**
   *
   */
  template <bool with_moving_mesh = false, int dim, typename ScratchData>
  void
  weakly_enforced_no_slip_matrix(const ComponentOrdering &component_ordering,
                                 const unsigned int       i_face,
                                 const ScratchData       &scratch_data,
                                 const TimeHandler       &time_handler,
                                 FullMatrix<double>      &local_matrix);
} // namespace Assembly

/* ---------------- Template functions ----------------- */

template <bool with_moving_mesh, int dim, typename ScratchData>
void Assembly::weakly_enforced_no_slip_rhs(
  const ComponentOrdering                &component_ordering,
  const unsigned int                      i_face,
  const BoundaryConditions::FluidBC<dim> &fluid_boundary_condition,
  const ScratchData                      &scratch_data,
  Vector<double>                         &local_rhs)
{
  const bool enable_rigid_body_rotation =
    fluid_boundary_condition.enable_rigid_body_rotation;
  const auto &rotation_velocities =
    scratch_data.input_face_rigid_body_rotation_velocity[i_face];

  for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
  {
    const double face_JxW_moving = scratch_data.face_JxW_moving[i_face][q];

    const auto &phi_u = scratch_data.phi_u_face[i_face][q];
    const auto &phi_l = scratch_data.phi_l_face[i_face][q];

    const auto &fluid_velocity =
      scratch_data.present_face_velocity_values[i_face][q];
    const auto &lambda = scratch_data.present_face_lambda_values[i_face][q];

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    const auto &face_velocity_source_term =
      scratch_data.face_velocity_source_term[i_face][q];
#endif

    auto velocity_constraint = fluid_velocity;
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    Tensor<1, dim> exact_constraint =
      scratch_data.exact_face_velocity_values[i_face][q];
#endif

    if constexpr (with_moving_mesh)
    {
      const auto &mesh_velocity =
        scratch_data.present_face_mesh_velocity_values[i_face][q];
      velocity_constraint -= mesh_velocity;

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
      exact_constraint -=
        scratch_data.exact_face_mesh_velocity_values[i_face][q];
#endif
    }

    if (enable_rigid_body_rotation)
      velocity_constraint -= rotation_velocities[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      double local_rhs_i = 0.;

      const unsigned int comp_i = scratch_data.components[i];
      const bool         i_is_u = component_ordering.is_velocity(comp_i);
      const bool         i_is_l = component_ordering.is_lambda(comp_i);

      if (i_is_u)
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
        local_rhs_i -= -phi_u[i] * (lambda - face_velocity_source_term);
#else
        local_rhs_i -= -phi_u[i] * lambda;
#endif

      if (i_is_l)
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
        local_rhs_i -= -(velocity_constraint - exact_constraint) * phi_l[i];
#else
        local_rhs_i -= -velocity_constraint * phi_l[i];
#endif

      local_rhs(i) += local_rhs_i * face_JxW_moving;
    }
  }
}

template <bool with_moving_mesh, int dim, typename ScratchData>
void Assembly::weakly_enforced_no_slip_matrix(
  const ComponentOrdering &component_ordering,
  const unsigned int       i_face,
  const ScratchData       &scratch_data,
  const TimeHandler       &time_handler,
  FullMatrix<double>      &local_matrix)
{
  const unsigned int u_lower = component_ordering.u_lower;
  const unsigned int u_upper = component_ordering.u_upper;
  const unsigned int x_lower = component_ordering.x_lower;
  const unsigned int x_upper = component_ordering.x_upper;
  const unsigned int l_lower = component_ordering.l_lower;
  const unsigned int l_upper = component_ordering.l_upper;

  const Tensor<1, dim>       *lambda;
  const std::vector<double>  *delta_dx;
  double                      lambda_dot_phi_u_i;
  std::vector<Tensor<1, dim>> to_multiply_by_phi_l_i(
    scratch_data.dofs_per_cell);

  const double bdf_c0 = time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
  {
    const double face_JxW_moving = scratch_data.face_JxW_moving[i_face][q];

    const auto &phi_u = scratch_data.phi_u_face[i_face][q];
    const auto &phi_l = scratch_data.phi_l_face[i_face][q];

    if constexpr (with_moving_mesh)
    {
      const auto &phi_x = scratch_data.phi_x_face[i_face][q];
      delta_dx          = &scratch_data.delta_dx[i_face][q];
      const auto u_ale =
        scratch_data.present_face_velocity_values[i_face][q] -
        scratch_data.present_face_mesh_velocity_values[i_face][q];
      lambda = &scratch_data.present_face_lambda_values[i_face][q];

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
        to_multiply_by_phi_l_i[j] = bdf_c0 * phi_x[j] + u_ale * (*delta_dx)[j];
    }

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i = scratch_data.components[i];
      const bool         i_is_u = u_lower <= comp_i && comp_i < u_upper;
      const bool         i_is_l = l_lower <= comp_i && comp_i < l_upper;

      const auto &phi_u_i = phi_u[i];
      const auto &phi_l_i = phi_l[i];

      if constexpr (with_moving_mesh)
        lambda_dot_phi_u_i = *lambda * phi_u_i;

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int comp_j = scratch_data.components[j];
        const bool         j_is_u = u_lower <= comp_j && comp_j < u_upper;
        const bool         j_is_l = l_lower <= comp_j && comp_j < l_upper;
        const bool         j_is_x = x_lower <= comp_j && comp_j < x_upper;

        if constexpr (with_moving_mesh)
        {
          const bool assemble =
            (i_is_u and (j_is_x or j_is_l)) or (i_is_l and (j_is_u or j_is_x));
          if (!assemble)
            continue;
        }
        else
        {
          const bool assemble = (i_is_u and j_is_l) or (i_is_l and j_is_u);
          if (!assemble)
            continue;
        }

        const auto &phi_u_j = phi_u[j];
        const auto &phi_l_j = phi_l[j];

        double local_matrix_ij = 0.;

        if (i_is_u && j_is_l)
        {
          local_matrix_ij += -phi_l_j * phi_u_i;
        }

        if (i_is_l && j_is_u)
        {
          local_matrix_ij += -phi_u_j * phi_l_i;
        }

        if constexpr (with_moving_mesh)
        {
          if (j_is_x)
          {
            if (i_is_u)
            {
              local_matrix_ij += -lambda_dot_phi_u_i * (*delta_dx)[j];
            }
            if (i_is_l)
            {
              local_matrix_ij += phi_l_i * to_multiply_by_phi_l_i[j];
            }
          }
        }

        local_matrix(i, j) += local_matrix_ij * face_JxW_moving;
      }
    }
  }
}

#endif
