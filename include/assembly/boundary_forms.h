#ifndef ASSEMBLY_BOUNDARY_FORMS_H
#define ASSEMBLY_BOUNDARY_FORMS_H

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
   *
   */
  template <typename ScratchData>
  void traction_boundary_mms_rhs(const ComponentOrdering &component_ordering,
                                 const unsigned int       i_face,
                                 const double             viscosity,
                                 const ScratchData       &scratch_data,
                                 Vector<double>          &local_rhs,
                                 const bool               full_traction = true);

} // namespace Assembly

/* ---------------- Template functions ----------------- */

template <typename ScratchData>
void Assembly::traction_boundary_mms_rhs(
  const ComponentOrdering & /*component_ordering*/,
  const unsigned int i_face,
  const double       viscosity,
  const ScratchData &scratch_data,
  Vector<double>    &local_rhs,
  const bool         full_traction)
{
  for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
  {
    const double face_JxW_moving = scratch_data.face_JxW_moving[i_face][q];
    const auto  &n               = scratch_data.face_normals_moving[i_face][q];

    const auto &grad_u_exact =
      scratch_data.exact_face_velocity_gradients[i_face][q];
    const double p_exact = scratch_data.exact_face_pressure_values[i_face][q];

    // This is an open boundary condition, not a traction,
    // involving only grad_u_exact and not the symmetric gradient.
    auto sigma_dot_n = -p_exact * n + viscosity * grad_u_exact * n;

    if (full_traction)
      // Full stress tensor
      sigma_dot_n += viscosity * transpose(grad_u_exact) * n;

    const auto &phi_u = scratch_data.phi_u_face[i_face][q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
      local_rhs(i) -= -phi_u[i] * sigma_dot_n * face_JxW_moving;
  }
}


#endif
