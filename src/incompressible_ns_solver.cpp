
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <incompressible_ns_solver.h>
#include <linear_direct_solver.h>
#include <mesh.h>
#include <scratch_data.h>

template <int dim>
IncompressibleNavierStokesSolver<dim>::IncompressibleNavierStokesSolver(
  const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver, param.timer)
  , param(param)
  , quadrature(QGaussSimplex<dim>(4))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , triangulation(mpi_communicator)
  , mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
  , fe(FE_SimplexP<dim>(param.finite_elements.velocity_degree), // Velocity
       dim,
       FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
       1)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(u_lower, n_components);

  if (param.mms_param.enable)
  {
    // Create the source term function for the given MMS and override source
    // terms this->source_terms =
    // std::make_shared<IncompressibleNavierStokesSolver::MMSSourceTerm>()
  }
  else
  {
    this->source_terms = param.source_terms.fluid_source;
  }
}

template <int dim>
IncompressibleNavierStokesSolver<dim>::MMSSourceTerm::MMSSourceTerm(
  const double       time,
  const unsigned int n_components,
  const ManufacturedSolution::ManufacturedSolution<dim> &mms)
  : Function<dim>(n_components, time)
, mms(mms)
{}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  // const double t  = this->get_time();
  // const double mu = VISCOSITY;

  Tensor<2, dim> grad_u;
  Tensor<1, dim> f, u, dudt_eulerian, uDotGradu, grad_p, lap_u;

  mms.exact_velocity->time_derivative(p, dudt_eulerian);
  mms.exact_velocity->gradient(p, grad_u);
  mms.exact_velocity->laplacian(p, lap_u);
  mms.exact_pressure->gradient(p, grad_p);

  // flow_mms.velocity_time_derivative(t, p, dudt_eulerian);
  // flow_mms.velocity(t, p, u);
  // // flow_mms.grad_velocity_ui_xj(t, p, grad_u);
  // flow_mms.grad_velocity_uj_xi(t, p, grad_u);
  // uDotGradu = u * grad_u;
  // flow_mms.grad_pressure(t, p, grad_p);
  // flow_mms.laplacian_velocity(t, p, lap_u);

  // // Stokes/Navier-Stokes source term
  // f = -(dudt_eulerian + uDotGradu + grad_p - mu * lap_u);

  // for (unsigned int d = 0; d < dim; ++d)
  //   values[u_lower + d] = f[d];

  // //
  // // Pressure
  // //
  // values[p_lower] = flow_mms.velocity_divergence(t, p);

  // //
  // // Pseudo-solid
  // //
  // // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
  // Tensor<1, dim> f_PS;
  // mesh_mms.divergence_stress_tensor(t, p, MU_PS, LAMBDA_PS, f_PS);

  // for (unsigned int d = 0; d < dim; ++d)
  //   values[x_lower + d] = f_PS[d];

  // //
  // // Lagrange multiplier: to have u = dxdt on boundary.
  // // dxdt must be evaluated on initial mesh!
  // // For now, return only u(x,t) on current mesh, and
  // // "assemble" the lambda source term where it is needed.
  // //
  // Tensor<1, dim> dxdt;
  // // mesh_mms.mesh_velocity(t, pInitial, dxdt);
  // // Tensor<1, dim> f_lambda = - (u - dxdt);
  // for (unsigned int d = 0; d < dim; ++d)
  //   values[l_lower + d] = u[d];
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::run()
{
  pcout << "param.mms.first         = " << param.mms_param.first_mesh_index
        << std::endl;
  pcout << "param.mms_param.last          = " << param.mms_param.last_mesh_index
        << std::endl;
  pcout << "param.mms_param.n_convergence = " << param.mms_param.n_convergence << std::endl;

  for (unsigned int i_conv = param.mms_param.first_mesh_index;
       i_conv <= param.mms_param.last_mesh_index;
       ++i_conv)
  {
    // If a manufactured solution test is run, bypass the given mesh file
    // and run the prescribed suffix.
    if (param.mms_param.enable)
    {
      param.mms_param.override_mesh_filename(param.mesh, i_conv);
      pcout << "Convergence test with manufactured solution:" << std::endl;
      pcout << "Mesh file was changed to " << param.mesh.filename << std::endl;
    }

    read_mesh(triangulation, param);
    setup_dofs();
    create_zero_constraints();
    create_nonzero_constraints();
    create_sparsity_pattern();
    set_initial_conditions();
    output_results();

    while (!time_handler.is_finished())
    {
      time_handler.advance();

      if (param.time_integration.verbosity == Parameters::Verbosity::verbose)
        pcout << std::endl
              << "Time step " << time_handler.current_time_iteration
              << " - Advancing to t = " << time_handler.current_time << '.'
              << std::endl;

      update_boundary_conditions();
      if (time_handler.current_time_iteration == 1 &&
          param.time_integration.scheme ==
            Parameters::TimeIntegration::Scheme::BDF2)
      {
        // FIXME: Start with BDF1
        set_initial_conditions();
      }
      else
      {
        // Entering the Newton solver with a solution satisfying the nonzero
        // constraints, which were applied in update_boundary_condition().
        solve_nonlinear_problem(false);
      }

      output_results();

      // Rotate solutions
      for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
        previous_solutions[j] = previous_solutions[j - 1];
      previous_solutions[0] = present_solution;
    }
  }
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(this->computing_timer, "Setup");

  auto &comm = this->mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(fe);

  this->pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Initialize parallel vectors
  this->present_solution.reinit(locally_owned_dofs,
                                locally_relevant_dofs,
                                comm);
  this->evaluation_point.reinit(locally_owned_dofs,
                                locally_relevant_dofs,
                                comm);

#if !defined(FEZ_WITH_TRILINOS) && !defined(FEZ_WITH_PETSC)
  this->local_evaluation_point.reinit(locally_owned_dofs,
                                      locally_relevant_dofs,
                                      comm);
  this->newton_update.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  this->system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
#else
  this->local_evaluation_point.reinit(locally_owned_dofs, comm);
  this->newton_update.reinit(locally_owned_dofs, comm);
  this->system_rhs.reinit(locally_owned_dofs, comm);
#endif

  // Allocate for previous BDF solutions
  previous_solutions.clear();
  previous_solutions.resize(time_handler.n_previous_solutions);
  for (auto &previous_sol : previous_solutions)
  {
    // previous_sol.clear();
    previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  }
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::create_zero_constraints()
{
  zero_constraints.clear();
  zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  const FEValuesExtractors::Vector velocity(u_lower);

  //
  // Velocity homogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    for (const auto &bc : this->param.fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::no_slip ||
          bc.type == BoundaryConditions::Type::input_function)
      {
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 zero_constraints,
                                                 fe.component_mask(velocity));
      }
      if (bc.type == BoundaryConditions::Type::slip)
        no_flux_boundaries.insert(bc.id);
    }

    // Add no velocity flux constraints
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_flux_boundaries, zero_constraints, *mapping);
  }

  zero_constraints.close();
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::create_nonzero_constraints()
{
  nonzero_constraints.clear();
  nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  const FEValuesExtractors::Vector velocity(u_lower);

  //
  // Velocity inhomogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    for (const auto &bc : this->param.fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::no_slip)
      {
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 nonzero_constraints,
                                                 fe.component_mask(velocity));
      }
      if (bc.type == BoundaryConditions::Type::input_function)
      {
        VectorTools::interpolate_boundary_values(
          *mapping,
          dof_handler,
          bc.id,
          ComponentwiseFlowVelocity<dim>(
            u_lower, n_components, bc.u, bc.v, bc.w),
          nonzero_constraints,
          fe.component_mask(velocity));
      }

      if (bc.type == BoundaryConditions::Type::slip)
        no_flux_boundaries.insert(bc.id);
    }

    // Add no velocity flux constraints
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_flux_boundaries, nonzero_constraints, *mapping);
  }

  nonzero_constraints.close();
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints have been
  // defined
  //
#if defined(FEZ_WITH_PETSC)
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
#else
  TrilinosWrappers::SparsityPattern dsp(locally_owned_dofs,
                                        locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
  dsp.compress();
  system_matrix.reinit(dsp);
#endif
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::set_initial_conditions()
{
  const FEValuesExtractors::Vector velocity(u_lower);

  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *this->param.initial_conditions.initial_velocity,
                           this->newton_update,
                           fe.component_mask(velocity));

#if defined(FORCE_DEAL_II_PARALLEL_VECTOR)
  this->newton_update.update_ghost_values();
#endif

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(this->newton_update);
  this->present_solution = this->newton_update;

  // FIXME: Dirty copy of the initial condition for BDF2 for now (-:
  for (auto &sol : previous_solutions)
    sol = this->present_solution;
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::update_boundary_conditions()
{
  // Re-create and distribute nonzero constraints:
  this->local_evaluation_point = this->present_solution;
  this->create_nonzero_constraints();
  nonzero_constraints.distribute(this->local_evaluation_point);
  this->present_solution = this->local_evaluation_point;
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  system_matrix = 0;

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 fe.n_dofs_per_cell(),
                                 time_handler.bdf_coefficients);
  CopyData           copyData(fe.n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(
    dof_handler.begin_active(),
    dof_handler.end(),
    *this,
    &IncompressibleNavierStokesSolver::assemble_local_matrix,
    &IncompressibleNavierStokesSolver::copy_local_to_global_matrix,
    scratchData,
    copyData);

  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell, evaluation_point, previous_solutions, source_terms);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double kinematic_viscosity =
    param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW = scratchData.JxW[q];

    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];
    const auto &phi_p      = scratchData.phi_p[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];

    // const auto &source_term_velocity = scratchData.source_term_velocity[q];
    // const auto &source_term_pressure = scratchData.source_term_pressure[q];
    // const auto &grad_source_velocity = scratchData.grad_source_velocity[q];
    // const auto &grad_source_pressure = scratchData.grad_source_pressure[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratchData.components[i];
      const bool         i_is_u      = is_velocity(component_i);
      const bool         i_is_p      = is_pressure(component_i);

      for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratchData.components[j];
        const bool         j_is_u      = is_velocity(component_j);
        const bool         j_is_p      = is_pressure(component_j);

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
          local_matrix_ij +=
            kinematic_viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);
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
void IncompressibleNavierStokesSolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              system_matrix);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 fe.n_dofs_per_cell(),
                                 time_handler.bdf_coefficients);
  CopyData           copyData(fe.n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &IncompressibleNavierStokesSolver::assemble_local_rhs,
                  &IncompressibleNavierStokesSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell, evaluation_point, previous_solutions, source_terms);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double kinematic_viscosity =
    param.physical_properties.fluids[0].kinematic_viscosity;

  const std::vector<double> bdf_coefficients = time_handler.bdf_coefficients;

  const unsigned int          nBDF = bdf_coefficients.size();
  std::vector<Tensor<1, dim>> velocity(nBDF);

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW = scratchData.JxW[q];

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

    // BDF
    velocity[0] = present_velocity_values;
    for (unsigned int i = 1; i < nBDF; ++i)
    {
      velocity[i] = scratchData.previous_velocity_values[i - 1][q];
    }

    const auto &phi_p      = scratchData.phi_p[q];
    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      double local_rhs_i = -(
        // Convection
        (present_velocity_gradients * present_velocity_values) * phi_u[i]

        // Diffusion
        + kinematic_viscosity *
            scalar_product(present_velocity_gradients, grad_phi_u[i])

        // Pressure gradient
        - div_phi_u[i] * present_pressure_values

        // Momentum source term
        + source_term_velocity * phi_u[i]

        // Continuity
        - present_velocity_divergence * phi_p[i]

        // Pressure source term
        + source_term_pressure * phi_p[i]);

      // Transient terms:
      for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
      {
        local_rhs_i -= bdf_coefficients[iBDF] * velocity[iBDF] * phi_u[i];
      }

      local_rhs_i *= JxW;
      local_rhs(i) += local_rhs_i;
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              this->system_rhs);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::solve_linear_system(
  const bool apply_inhomogeneous_constraints)
{
  solve_linear_system_direct(this,
                             system_matrix,
                             locally_owned_dofs,
                             zero_constraints);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::output_results() const
{
  if (param.output.write_results)
  {
    //
    // Plot FE solution
    //
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    //
    // Partition
    //
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    // Export regular time step
    data_out.write_vtu_with_pvtu_record(param.output.output_dir,
                                        param.output.output_prefix,
                                        time_handler.current_time_iteration,
                                        mpi_communicator,
                                        2);
  }
}

// Explicit instantiation
template class IncompressibleNavierStokesSolver<2>;
template class IncompressibleNavierStokesSolver<3>;