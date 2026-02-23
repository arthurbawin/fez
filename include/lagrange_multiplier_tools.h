#ifndef LAGRANGE_MULTIPLIER_TOOLS_H
#define LAGRANGE_MULTIPLIER_TOOLS_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <parameter_reader.h>
#include <scratch_data.h>

using namespace dealii;

/**
 * A collection of routines that are common to the flow solvers using a Lagrange
 * multiplier to enforce, e.g., a no-slip condition on some boundary.
 */
namespace LagrangeMultiplierTools
{
  /**
   * Check that the no-slip constraint is enforced on boundary_id, up to the
   * prescribed tolerance. Depending on solver and parameters, this constraint
   * is either
   *    u_fluid = 0,
   * or u_fluid = u_mesh if the boundary moves,
   * or u_fluid = u_rigid_rotation if a rigid rotation is applied,
   * or a combination of these.
   */
  template <int dim, typename ScratchData, typename VectorType>
  void check_no_slip_on_boundary(
    const ParameterReader<dim>           &param,
    ScratchData                          &scratch_data,
    const DoFHandler<dim>                &dof_handler,
    const VectorType                     &solution,
    const std::vector<VectorType>        &previous_solutions,
    const std::shared_ptr<Function<dim>> &source_terms,
    const std::shared_ptr<Function<dim>> &exact_solution,
    const types::boundary_id              boundary_id);

} // namespace LagrangeMultiplierTools

/* ---------------- Template functions ----------------- */

template <int dim, typename ScratchData, typename VectorType>
void LagrangeMultiplierTools::check_no_slip_on_boundary(
  const ParameterReader<dim>           &param,
  ScratchData                          &scratch_data,
  const DoFHandler<dim>                &dof_handler,
  const VectorType                     &solution,
  const std::vector<VectorType>        &previous_solutions,
  const std::shared_ptr<Function<dim>> &source_terms,
  const std::shared_ptr<Function<dim>> &exact_solution,
  const types::boundary_id              boundary_id)
{
  if (boundary_id == numbers::invalid_unsigned_int)
    return;

  // Check that there is a fluid BC on this boundary
  Assert(param.fluid_bc.count(boundary_id) > 0,
         ExcMessage("Cannot check for no-slip enforcement with Lagrange "
                    "multiplier on boundary" +
                    std::to_string(boundary_id) +
                    " because this boundary does not have a fluid boundary "
                    "condition assigned."));

  const auto &bc = param.fluid_bc.at(boundary_id);

  const bool enable_rigid_body_rotation = bc.enable_rigid_body_rotation;

  // Check difference between uh and dxhdt
  double l2_local = 0, li_local = 0;

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          scratch_data.reinit(
            cell, solution, previous_solutions, source_terms, exact_solution);

          const auto &fluid_velocity =
            scratch_data.present_face_velocity_values[i_face];

          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            Tensor<1, dim> constraint = fluid_velocity[q];

            // If the boundary moves, the constraint is
            // u_fluid = u_mesh -> u_fluid - u_mesh = 0
            if (scratch_data.enable_pseudo_solid)
              constraint -=
                scratch_data.present_face_mesh_velocity_values[i_face][q];

            // If in addition there is a rigid body rotation, the constraint
            // becomes u_fluid = u_mesh + u_rotation
            if (enable_rigid_body_rotation)
              constraint -=
                scratch_data.input_face_rigid_body_rotation_velocity[i_face][q];

            // Measure how well the constraint is enforced (target is zero)
            l2_local += constraint * constraint * scratch_data.JxW_fixed[q];
            li_local = std::max(li_local, constraint.norm());
          }
        }
      }

  MPI_Comm     mpi_communicator = dof_handler.get_mpi_communicator();
  const double l2_error =
    std::sqrt(Utilities::MPI::sum(l2_local, mpi_communicator));
  const double li_error = Utilities::MPI::max(li_local, mpi_communicator);

  const auto mpi_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  if (param.bc_data.fluid_verbosity == Parameters::Verbosity::verbose &&
      mpi_rank == 0)
  {
    std::cout << "Checking no-slip enforcement on cylinder:" << std::endl;
    // std::cout << "Checking no-slip enforcement on boundary with id "
    //           << boundary_id << ":" << std::endl;
    std::cout << "||uh - wh||_L2 = " << l2_error << std::endl;
    std::cout << "||uh - wh||_Li = " << li_error << std::endl;
  }

  AssertThrow(l2_error < bc.weak_no_slip_tolerance,
              ExcMessage(
                "L2 norm of no-slip constraint exceeds the given tolerance: " +
                std::to_string(l2_error)));
  AssertThrow(
    li_error < bc.weak_no_slip_tolerance,
    ExcMessage("Linf norm of no-slip constraint exceeds the given tolerance: " +
               std::to_string(li_error)));
}

#endif
