
#include <assembly/boundary_forms.h>
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_ns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
NSSolver<dim>::NSSolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim>(param)
{
  if (param.finite_elements.use_quads)
    fe = std::make_unique<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree), // Velocity
      dim,
      FE_Q<dim>(param.finite_elements.pressure_degree), // Pressure
      1);
  else
    fe = std::make_unique<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree), // Velocity
      dim,
      FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
      1);

  this->ordering = std::make_unique<ComponentOrderingNS<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);

  this->field_names_and_masks["velocity"] = this->velocity_mask;
  this->field_names_and_masks["pressure"] = this->pressure_mask;

  /**
   * This solver uses a fixed mapping only.
   */
  mapping = this->fixed_mapping.get();

  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);

  // Assign the exact solution
  this->exact_solution = std::make_shared<NSSolver<dim>::MMSSolution>(
    this->time_handler.current_time, *this->ordering, param.mms);

  if (param.mms_param.enable)
  {
    if (param.mms_param.force_source_term)
    {
      // Use the provided source term instead of the source term computed from
      // symbolic differentiation.
      this->source_terms = param.source_terms.fluid_source;
    }
    else
    {
      // Create the source term function for the given MMS and override source
      // terms
      this->source_terms = std::make_shared<NSSolver<dim>::MMSSourceTerm>(
        this->time_handler.current_time,
        *this->ordering,
        param.physical_properties,
        param.mms);
    }
  }
  else
  {
    this->source_terms = param.source_terms.fluid_source;
  }
}

template <int dim>
void NSSolver<dim>::MMSSourceTerm::vector_value(const Point<dim> &p,
                                                Vector<double>   &values) const
{
  const double nu = physical_properties.fluids[0].kinematic_viscosity;

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  // Use convention (grad_u)_ij := dvj/dxi
  Tensor<2, dim> grad_u    = mms.exact_velocity->gradient_vj_xi(p);
  Tensor<1, dim> lap_u     = mms.exact_velocity->vector_laplacian(p);
  Tensor<1, dim> grad_p    = mms.exact_pressure->gradient(p);
  Tensor<1, dim> uDotGradu = u * grad_u;

  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(dudt_eulerian + uDotGradu + grad_p - nu * lap_u);
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[p_lower] = mms.exact_velocity->divergence(p);
}

template <int dim>
void NSSolver<dim>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim>
void NSSolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints have been
  // defined
  //
#if defined(FEZ_WITH_PETSC)
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);
  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);
#else
  TrilinosWrappers::SparsityPattern dsp(this->locally_owned_dofs,
                                        this->locally_owned_dofs,
                                        this->locally_relevant_dofs,
                                        this->mpi_communicator);
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  dsp,
                                  this->nonzero_constraints);
  dsp.compress();
  this->system_matrix.reinit(dsp);
#endif
}

template <int dim>
void NSSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  CopyData copyData(fe->n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  auto assembly_ptr = this->param.nonlinear_solver.analytic_jacobian ?
                      &NSSolver::assemble_local_matrix :
                      &NSSolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  assembly_ptr,
                  &NSSolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copyData);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim>
void NSSolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &NSSolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    this->evaluation_point,
    this->local_evaluation_point);
}

template <int dim>
void NSSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW_moving[q];

    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];
    const auto &phi_p      = scratch_data.phi_p[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratch_data.components[i];
      const bool         i_is_u      = this->ordering->is_velocity(component_i);
      const bool         i_is_p      = this->ordering->is_pressure(component_i);

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratch_data.components[j];
        const bool         j_is_u = this->ordering->is_velocity(component_j);
        const bool         j_is_p = this->ordering->is_pressure(component_j);

        bool   assemble        = false;
        double local_matrix_ij = 0.;

        if (i_is_u && j_is_u)
        {
          assemble = true;

          // Time-dependent
          local_matrix_ij += bdf_c0 * phi_u[i] * phi_u[j];

          // Convection
          local_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                              present_velocity_gradients * phi_u[j]) *
                             phi_u[i];

          // Diffusion
          local_matrix_ij += nu * scalar_product(grad_phi_u[i], grad_phi_u[j]);
        }

        if (i_is_u && j_is_p)
        {
          assemble = true;

          // Pressure gradient
          local_matrix_ij += -div_phi_u[i] * phi_p[j];
        }

        if (i_is_p && j_is_u)
        {
          assemble = true;

          // Continuity : variation w.r.t. u
          local_matrix_ij += -phi_p[i] * div_phi_u[j];
        }

        if (assemble)
        {
          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void NSSolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                                    copy_data.local_dof_indices,
                                                    this->system_matrix);
}

template <int dim>
void NSSolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copyData(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &NSSolver::assemble_local_matrix,
    &NSSolver::assemble_local_rhs,
    *scratch_data,
    copyData,
    this->present_solution,
    this->evaluation_point,
    this->local_evaluation_point,
    this->mpi_communicator);

  this->pcout << "Max absolute error analytical vs fd matrix is "
              << errors.first << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > this->param.debug.analytical_jacobian_absolute_tolerance)
    this->pcout << "Max relative error analytical vs fd matrix is "
                << errors.second << std::endl;
}

template <int dim>
void NSSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  CopyData copyData(fe->n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &NSSolver::assemble_local_rhs,
                  &NSSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copyData);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void NSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW_moving[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];
    const auto  &source_term_velocity = scratch_data.source_term_velocity[q];
    const auto  &source_term_pressure = scratch_data.source_term_pressure[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const auto &phi_p      = scratch_data.phi_p[q];
    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      double local_rhs_i = -(
        // Transient
        dudt * phi_u[i]

        // Convection
        + (present_velocity_gradients * present_velocity_values) * phi_u[i]

        // Diffusion
        + nu * scalar_product(present_velocity_gradients, grad_phi_u[i])

        // Pressure gradient
        - div_phi_u[i] * present_pressure_values

        // Momentum source term
        + source_term_velocity * phi_u[i]

        // Continuity
        - present_velocity_divergence * phi_p[i]

        // Pressure source term
        + source_term_pressure * phi_p[i]);

      local_rhs_i *= JxW;
      local_rhs(i) += local_rhs_i;
    }
  }

  //
  // Face contributions
  //
  if (cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary())
      {
        const auto &fluid_bc = this->param.fluid_bc.at(face->boundary_id());

        // Open boundary condition with prescribed manufactured solution
        if (fluid_bc.type == BoundaryConditions::Type::open_mms)
        {
          Assembly::traction_boundary_mms_rhs(*this->ordering,
                                              i_face,
                                              nu,
                                              scratch_data,
                                              local_rhs,
                                              /* full_traction = */ false);
        }
      }
    }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void NSSolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                                    copy_data.local_dof_indices,
                                                    this->system_rhs);
}

// Explicit instantiation
template class NSSolver<2>;
template class NSSolver<3>;
