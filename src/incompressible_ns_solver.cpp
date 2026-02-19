
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
    fe = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree), // Velocity
      dim,
      FE_Q<dim>(param.finite_elements.pressure_degree), // Pressure
      1);
  else
    fe = std::make_shared<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree), // Velocity
      dim,
      FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
      1);

  this->ordering = std::make_shared<ComponentOrderingNS<dim>>();

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

  ScratchData scratchData(*this->ordering,
                          *fe,
                          *mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

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
                  scratchData,
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
  ScratchData                                          &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell,
                     this->evaluation_point,
                     this->previous_solutions,
                     this->source_terms,
                     this->exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW = scratchData.JxW_moving[q];

    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];
    const auto &phi_p      = scratchData.phi_p[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratchData.components[i];
      const bool         i_is_u      = this->ordering->is_velocity(component_i);
      const bool         i_is_p      = this->ordering->is_pressure(component_i);

      for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratchData.components[j];
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
  ScratchData scratchData(*this->ordering,
                          *fe,
                          *mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &NSSolver::assemble_local_matrix,
    &NSSolver::assemble_local_rhs,
    scratchData,
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

  ScratchData scratchData(*this->ordering,
                          *fe,
                          *mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &NSSolver::assemble_local_rhs,
                  &NSSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void NSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell,
                     this->evaluation_point,
                     this->previous_solutions,
                     this->source_terms,
                     this->exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW = scratchData.JxW_moving[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];
    const auto &present_pressure_values =
      scratchData.present_pressure_values[q];
    const auto  &source_term_velocity = scratchData.source_term_velocity[q];
    const auto  &source_term_pressure = scratchData.source_term_pressure[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratchData.previous_velocity_values);

    const auto &phi_p      = scratchData.phi_p[q];
    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
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
        // Open boundary condition with prescribed manufactured solution
        if (this->param.fluid_bc.at(scratchData.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
        {
          for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
          {
            const double face_JxW = scratchData.face_JxW_moving[i_face][q];
            const auto  &n        = scratchData.face_normals_moving[i_face][q];

            const auto &grad_u_exact =
              scratchData.exact_face_velocity_gradients[i_face][q];
            const double p_exact =
              scratchData.exact_face_pressure_values[i_face][q];

            // This is an open boundary condition, not a traction,
            // involving only grad_u_exact and not the symmetric gradient.
            const auto sigma_dot_n = -p_exact * n + nu * grad_u_exact * n;

            const auto &phi_u_face = scratchData.phi_u_face[i_face][q];

            for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
            {
              local_rhs(i) -= -phi_u_face[i] * sigma_dot_n * face_JxW;
            }
          }
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

template <int dim>
void NSSolver<dim>::output_results()
{
  TimerOutput::Scope t(this->computing_timer, "Write outputs");

  // ============================================================
  // 1) Full-domain VTU output
  // ============================================================
  if (this->param.output.write_results &&
      (this->time_handler.current_time_iteration %
           this->param.output.vtu_output_frequency ==
         0 ||
       this->time_handler.is_finished()))
  {
    // (u, p) => dim + 1 components
    std::vector<std::string> solution_names;
    solution_names.reserve(dim + 1);

    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation;
    data_component_interpretation.reserve(dim + 1);

    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(this->present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Partition
    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    const std::string pvtu_file = data_out.write_vtu_with_pvtu_record(
      this->param.output.output_dir,
      this->param.output.output_prefix,
      this->time_handler.current_time_iteration,
      this->mpi_communicator,
      2);

    this->visualization_times_and_names.emplace_back(
      this->time_handler.current_time, pvtu_file);
  }

  // ============================================================
  // 2) Skin (boundary-only) VTU output
  // ============================================================
  if (this->param.output.write_skin_results &&
      this->param.output.skin_boundary_id != numbers::invalid_unsigned_int &&
      (this->time_handler.current_time_iteration %
           this->param.output.skin_vtu_output_frequency ==
         0 ||
       this->time_handler.is_finished()))
  {
    // (u, p) on the boundary
    std::vector<std::string> solution_names_faces;
    solution_names_faces.reserve(dim + 1);

    for (unsigned int d = 0; d < dim; ++d)
      solution_names_faces.push_back("velocity");
    solution_names_faces.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation_faces;
    data_component_interpretation_faces.reserve(dim + 1);

    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation_faces.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation_faces.push_back(
      DataComponentInterpretation::component_is_scalar);

    PostProcessingTools::DataOutFacesOnBoundary<dim> data_out_faces(
      this->triangulation, this->param.output.skin_boundary_id);

    data_out_faces.attach_dof_handler(this->dof_handler);

    data_out_faces.add_data_vector(this->present_solution,
                                   solution_names_faces,
                                   DataOutFaces<dim>::type_dof_data,
                                   data_component_interpretation_faces);

    data_out_faces.build_patches(*mapping, 2);


    const std::string skin_prefix = this->param.output.output_prefix + "_skin";

    const std::string skin_pvtu_file =
      data_out_faces.write_vtu_with_pvtu_record(
        this->param.output.output_dir,
        skin_prefix,
        this->time_handler.current_time_iteration,
        this->mpi_communicator,
        2);

    this->visualization_times_and_names.emplace_back(
      this->time_handler.current_time, skin_pvtu_file);


    this->visualization_times_and_names.emplace_back(
      this->time_handler.current_time, skin_pvtu_file);
  }
}



// Explicit instantiation
template class NSSolver<2>;
template class NSSolver<3>;
