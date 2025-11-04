
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <incompressible_ns_solver.h>
#include <mesh.h>
#include <scratch_data.h>

template <int dim>
IncompressibleNavierStokesSolver<dim>::IncompressibleNavierStokesSolver(
  const ParameterReader<dim> &param)
  : GenericSolver<ParVectorType>(param.nonlinear_solver, param.timer)
  , param(param)
  , quadrature(QGaussSimplex<dim>(4))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , triangulation(this->mpi_communicator)
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
  this->param.initial_conditions.initial_velocity =
    std::make_shared<Parameters::InitialVelocity<dim>>(
      u_lower,
      n_components,
      this->param.initial_conditions.initial_velocity_callback);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::run()
{
  read_mesh(triangulation, this->param);
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

  this->local_evaluation_point.reinit(locally_owned_dofs, comm);
  this->newton_update.reinit(locally_owned_dofs, comm);
  this->system_rhs.reinit(locally_owned_dofs, comm);

  // Allocate for previous BDF solutions
  previous_solutions.clear();
  previous_solutions.resize(time_handler.n_previous_solutions);
  for (auto &previous_sol : previous_solutions)
  {
    previous_sol.clear();
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
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             this->mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       this->mpi_communicator);
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

  const bool first_step = false;

  system_matrix = 0;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

  // Data to compute matrix with finite differences
  // This is a particular case where automatic differentiation
  // cannot be used, since the mesh position is an unknown.
  // The local dofs values, which will be perturbed:
  std::vector<double> cell_dof_values(dofs_per_cell);
  Vector<double>      ref_local_rhs(dofs_per_cell);
  Vector<double>      perturbed_local_rhs(dofs_per_cell);

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 time_handler.bdf_coefficients);

  // CopyData copyData(dofs_per_cell);

  //
  // Multi-threaded assembly
  //
  if (this->param.nonlinear_solver.analytic_jacobian)
  {
    pcout << "Assembling matrix on " << MultithreadInfo::n_threads()
          << " threads out of " << MultithreadInfo::n_cores() << " cores"
          << std::endl;

#if defined(DEAL_II_WITH_PETSC)
    AssertThrow(MultithreadInfo::is_running_single_threaded(),
                ExcMessage(
                  "Solver is running with more than 1 thread, but was compiled "
                  "with PETSc and currently uses PETSc wrappers for parallel "
                  "matrix and vectors, which are not thread safe."));
#endif

    WorkStream::run(
      dof_handler.begin_active(),
      dof_handler.end(),
      *this,
      &IncompressibleNavierStokesSolver::assemble_local_matrix,
      &IncompressibleNavierStokesSolver::copy_local_to_global_matrix,
      scratchData,
      CopyData(dofs_per_cell));
  }

  // for (const auto &cell : dof_handler.active_cell_iterators() |
  //                           IteratorFilters::LocallyOwnedCell())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;

  //   cell->get_dof_indices(local_dof_indices);

  //   if (this->param.nonlinear_solver.analytic_jacobian)
  //   {
  //     //
  //     // Analytic jacobian matrix
  //     //
  //     const bool distribute = true;
  //     this->assemble_local_matrix_og(first_step,
  //                                 cell,
  //                                 scratchData,
  //                                 this->evaluation_point,
  //                                 previous_solutions,
  //                                 local_dof_indices,
  //                                 local_matrix,
  //                                 distribute);
  //   }
  //   else
  //   {
  //     //
  //     // Finite differences
  //     //
  //     const double h      = 1.e-8;
  //     local_matrix        = 0.;
  //     ref_local_rhs       = 0.;
  //     perturbed_local_rhs = 0.;

  //     // Get the local dofs values
  //     for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //       cell_dof_values[j] = this->evaluation_point[local_dof_indices[j]];

  //     const bool distribute_rhs    = false;
  //     const bool use_full_solution = false;

  //     // Compute non-perturbed RHS
  //     this->assemble_local_rhs(first_step,
  //                              cell,
  //                              scratchData,
  //                              this->evaluation_point,
  //                              previous_solutions,
  //                              local_dof_indices,
  //                              ref_local_rhs,
  //                              cell_dof_values,
  //                              distribute_rhs,
  //                              use_full_solution);

  //     for (unsigned int j = 0; j < dofs_per_cell; ++j)
  //     {
  //       const double       og_value = cell_dof_values[j];
  //       cell_dof_values[j] += h;

  //       // Compute perturbed RHS
  //       // Reinit is called in the local rhs function
  //       this->assemble_local_rhs(first_step,
  //                                cell,
  //                                scratchData,
  //                                this->evaluation_point,
  //                                previous_solutions,
  //                                local_dof_indices,
  //                                perturbed_local_rhs,
  //                                cell_dof_values,
  //                                distribute_rhs,
  //                                use_full_solution);

  //       // Finite differences (with sign change as residual is -NL(u))
  //       for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //       {
  //         local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) /
  //         h;
  //       }

  //       // Restore solution
  //       cell_dof_values[j] = og_value;
  //     }
  //   }
  // }

  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_local_matrix_og(
  bool                                                  first_step,
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  ParVectorType                                        &current_solution,
  std::vector<ParVectorType>                           &previous_solutions,
  std::vector<types::global_dof_index>                 &local_dof_indices,
  FullMatrix<double>                                   &local_matrix,
  bool                                                  distribute)
{
  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell, current_solution, previous_solutions);

  local_matrix = 0;

  const double kinematic_viscosity =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

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

  if (distribute)
  {
    cell->get_dof_indices(local_dof_indices);
    if (first_step)
    {
      throw std::runtime_error("First step");
      nonzero_constraints.distribute_local_to_global(local_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
    }
    else
    {
      zero_constraints.distribute_local_to_global(local_matrix,
                                                  local_dof_indices,
                                                  system_matrix);
    }
  }
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

  scratchData.reinit(cell, this->evaluation_point, this->previous_solutions);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double kinematic_viscosity =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

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
                                              this->system_matrix);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  const bool first_step = false;

  this->system_rhs = 0;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double>                       local_rhs(dofs_per_cell);
  std::vector<double>                  cell_dof_values(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 time_handler.bdf_coefficients);

  CopyData copyData(dofs_per_cell);

  pcout << "Assembling rhs on " << MultithreadInfo::n_threads()
        << " threads out of " << MultithreadInfo::n_cores() << " cores"
        << std::endl;
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &IncompressibleNavierStokesSolver::assemble_local_rhs,
                  &IncompressibleNavierStokesSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  // for (const auto &cell : dof_handler.active_cell_iterators() |
  //                           IteratorFilters::LocallyOwnedCell())
  // {
  //   local_rhs              = 0;
  //   bool distribute        = true;
  //   bool use_full_solution = true;
  //   this->assemble_local_rhs(first_step,
  //                            cell,
  //                            scratchData,
  //                            this->evaluation_point,
  //                            previous_solutions,
  //                            local_dof_indices,
  //                            local_rhs,
  //                            cell_dof_values,
  //                            distribute,
  //                            use_full_solution);
  // }

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleNavierStokesSolver<dim>::assemble_local_rhs_og(
  bool                                                  first_step,
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  ParVectorType                                        &current_solution,
  std::vector<ParVectorType>                           &previous_solutions,
  std::vector<types::global_dof_index>                 &local_dof_indices,
  Vector<double>                                       &local_rhs,
  std::vector<double>                                  &cell_dof_values,
  bool                                                  distribute,
  bool                                                  use_full_solution)
{
  if (use_full_solution)
  {
    scratchData.reinit(cell, current_solution, previous_solutions);
  }
  else
    scratchData.reinit(cell, cell_dof_values, previous_solutions);

  local_rhs = 0;

  const double kinematic_viscosity =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

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

  if (distribute)
  {
    cell->get_dof_indices(local_dof_indices);
    if (first_step)
    {
      throw std::runtime_error("First step");
      nonzero_constraints.distribute_local_to_global(local_rhs,
                                                     local_dof_indices,
                                                     this->system_rhs);
    }
    else
      zero_constraints.distribute_local_to_global(local_rhs,
                                                  local_dof_indices,
                                                  this->system_rhs);
  }
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

  scratchData.reinit(cell, evaluation_point, previous_solutions);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double kinematic_viscosity =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

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
  TimerOutput::Scope t(computing_timer, "Solve direct");

  ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                mpi_communicator);

  // Solve with MUMPS
  SolverControl                    solver_control;
  PETScWrappers::SparseDirectMUMPS solver(solver_control);
  solver.solve(system_matrix, completely_distributed_solution, system_rhs);

  newton_update = completely_distributed_solution;

  if (apply_inhomogeneous_constraints)
  {
    throw std::runtime_error("First step");
    nonzero_constraints.distribute(newton_update);
  }
  else
    zero_constraints.distribute(newton_update);
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