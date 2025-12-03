
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <scratch_data.h>

template <int dim>
IncompressibleCHNSSolver<dim>::IncompressibleCHNSSolver(
  const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param)
  , velocity_extractor(u_lower)
  , pressure_extractor(p_lower)
  , tracer_extractor(phi_lower)
  , potential_extractor(mu_lower)
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
  , velocity_mask(fe.component_mask(velocity_extractor))
  , pressure_mask(fe.component_mask(pressure_extractor))
{
  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(u_lower, n_components);

  if (param.mms_param.enable)
  {
    // // Assign the manufactured solution
    // exact_solution =
    //   std::make_shared<IncompressibleCHNSSolver<dim>::MMSSolution>(
    //     time_handler.current_time, param.mms);

    // // auto &stream = pcout.get_stream();
    // // pcout << "Pression" << std::endl;
    // // param.mms.exact_pressure->print_function(stream);
    // // param.mms.exact_pressure->print_time_derivative(stream);
    // // param.mms.exact_pressure->print_gradient(stream);
    // // pcout << "Vitesse" << std::endl;
    // // param.mms.exact_velocity->print_function(stream);
    // // param.mms.exact_velocity->print_time_derivative(stream);
    // // param.mms.exact_velocity->print_gradient(stream);
    // // param.mms.exact_velocity->print_hessian(stream);

    // if(mms_param.force_source_term)
    // {
    //   // Use the provided source term instead of the source term computed from
    //   // symbolic differentiation.
    //   pcout << "Forcing source term" << std::endl;
    //   source_terms = param.source_terms.fluid_source;
    // }
    // else
    // {
    //   // Create the source term function for the given MMS and override source
    //   // terms
    //   source_terms =
    //     std::make_shared<IncompressibleCHNSSolver<dim>::MMSSourceTerm>(
    //       time_handler.current_time, param.physical_properties, param.mms);
    // }

    // error_handler.create_entry("L2_u");
    // error_handler.create_entry("L2_p");
    // error_handler.create_entry("Li_u");
    // error_handler.create_entry("Li_p");
    // error_handler.create_entry("H1_u");
    // error_handler.create_entry("H1_p");
  }
  else
  {
    source_terms = param.source_terms.fluid_source;
    exact_solution =
      std::make_shared<Functions::ZeroFunction<dim>>(n_components);
  }
}

// template <int dim>
// void IncompressibleCHNSSolver<dim>::MMSSourceTerm::vector_value(
//   const Point<dim> &p,
//   Vector<double>   &values) const
// {
//   const double nu = physical_properties.fluids[0].kinematic_viscosity;

//   Tensor<2, dim> grad_u;
//   Tensor<1, dim> f, u, dudt_eulerian, uDotGradu, grad_p, lap_u;

//   mms.exact_velocity->time_derivative(p, dudt_eulerian);
//   mms.exact_velocity->value(p, u);
//   // mms.exact_velocity->gradient_vi_xj(p, grad_u);
//   mms.exact_velocity->gradient_vj_xi(p, grad_u);
//   mms.exact_velocity->laplacian(p, lap_u);
//   mms.exact_pressure->gradient(p, grad_p);

//   // FIXME: double, triple and quadruple-check index convention
//   uDotGradu = u * grad_u;

//   // Navier-Stokes momentum (velocity) source term
//   f = -(dudt_eulerian + uDotGradu + grad_p - nu * lap_u);

//   for (unsigned int d = 0; d < dim; ++d)
//     values[u_lower + d] = f[d];

//   // Mass conservation (pressure) source term,
//   // for - div(u) + f = 0 -> f = div(u_mms).
//   values[p_lower] = mms.exact_velocity->divergence(p);
// }

template <int dim>
void IncompressibleCHNSSolver<dim>::reset()
{
  // FIXME: This is not very clean: the derived class has the full parameters,
  // and the base class GenericSolver has a mesh and time param to be able to
  // modify the mesh file and/or time step in a convergence loop.
  this->param.mms_param.current_step = this->mms_param.current_step;
  this->param.mms_param.mesh_suffix  = this->mms_param.mesh_suffix;
  this->param.mesh.filename          = this->mesh_param.filename;
  this->param.time_integration.dt    = this->time_param.dt;

  // Mesh
  triangulation.clear();

  // Time handler (move assign a new time handler)
  time_handler = TimeHandler(param.time_integration);
  this->set_time();

  // Pressure DOF
  constrained_pressure_dof = numbers::invalid_dof_index;
}

template <int dim>
void IncompressibleCHNSSolver<dim>::set_time()
{
  // Update time in all relevant structures:
  // - relevant boundary conditions
  // - source terms, if any
  // - exact solution, if any
  for (auto &[id, bc] : param.fluid_bc)
    bc.set_time(time_handler.current_time);
  source_terms->set_time(time_handler.current_time);
  exact_solution->set_time(time_handler.current_time);
}

template <int dim>
void IncompressibleCHNSSolver<dim>::run()
{
  reset();
  read_mesh(triangulation, param);
  setup_dofs();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
  output_results();

  while (!time_handler.is_finished())
  {
    time_handler.advance(pcout);
    set_time();

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
      if (param.mms_param.enable)
      {
        // Convergence study: start with exact solution at first time step
        set_exact_solution();
      }
      else
      {
        // FIXME: Start with BDF1
        set_initial_conditions();
      }
    }
    else
    {
      // Entering the Newton solver with a solution satisfying the nonzero
      // constraints, which were applied in update_boundary_condition().
      // compare_analytical_matrix_with_fd();
      solve_nonlinear_problem(false);
    }

    if (param.mms_param.enable)
      compute_errors();

    output_results();

    if (!time_handler.is_steady())
    {
      // Rotate solutions
      for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
        previous_solutions[j] = previous_solutions[j - 1];
      previous_solutions[0] = present_solution;
    }
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(fe);

  ////////////////////////////////////////////////////////
  if(param.linear_solver.renumber)
    DoFRenumbering::Cuthill_McKee(dof_handler, true);
  ////////////////////////////////////////////////////////

  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Initialize parallel vectors
  present_solution.reinit(locally_owned_dofs,
                                locally_relevant_dofs,
                                comm);
  evaluation_point.reinit(locally_owned_dofs,
                                locally_relevant_dofs,
                                comm);

#if !defined(FEZ_WITH_TRILINOS) && !defined(FEZ_WITH_PETSC)
  local_evaluation_point.reinit(locally_owned_dofs,
                                      locally_relevant_dofs,
                                      comm);
  newton_update.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  system_rhs.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
#else
  local_evaluation_point.reinit(locally_owned_dofs, comm);
  newton_update.reinit(locally_owned_dofs, comm);
  system_rhs.reinit(locally_owned_dofs, comm);
#endif

  // Allocate for previous BDF solutions
  previous_solutions.clear();
  previous_solutions.resize(time_handler.n_previous_solutions);
  for (auto &previous_sol : previous_solutions)
  {
    previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  }

  // For unsteady simulation, add the number of elements, dofs and/or the time
  // step to the error handler, once per convergence run.
  if (!time_handler.is_steady() && param.mms_param.enable)
  {
    error_handler.add_reference_data("n_elm",
                                     triangulation.n_global_active_cells());
    error_handler.add_reference_data("n_dof", dof_handler.n_dofs());
    error_handler.add_time_step(time_handler.initial_dt);
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::constrain_pressure_point(
  AffineConstraints<double> &constraints,
  const bool                 set_to_zero)
{
  // Determine the pressure dof the first time
  if (constrained_pressure_dof == numbers::invalid_dof_index)
  {
    // Choose a fixed physical reference location
    // Here it's the origin (Point<dim> is initialized at 0)
    const Point<dim> reference_point;

    IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler, pressure_mask);

    // Get support points for locally relevant DoFs
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler, support_points);

    double                  local_min_dist = std::numeric_limits<double>::max();
    types::global_dof_index local_dof      = numbers::invalid_dof_index;

    for (auto idx : pressure_dofs)
    {
      if (!locally_relevant_dofs.is_element(idx))
        continue;

      const double dist = support_points[idx].distance(reference_point);
      if (dist < local_min_dist)
      {
        local_min_dist = dist;
        local_dof      = idx;
      }
    }

    // Prepare for MPI_MINLOC reduction
    struct MinLoc
    {
      double                  dist;
      types::global_dof_index dof;
    } local_pair{local_min_dist, local_dof}, global_pair;

    // MPI reduction to find the global closest DoF
    MPI_Allreduce(&local_pair,
                  &global_pair,
                  1,
                  MPI_DOUBLE_INT,
                  MPI_MINLOC,
                  mpi_communicator);

    constrained_pressure_dof = global_pair.dof;

    // Set support point for MMS evaluation
    if (locally_relevant_dofs.is_element(constrained_pressure_dof))
    {
      constrained_pressure_support_point =
        support_points[constrained_pressure_dof];
    }
  }

  // Constrain that DoF globally
  if (locally_relevant_dofs.is_element(constrained_pressure_dof))
  {
    constraints.add_line(constrained_pressure_dof);

    // The pressure DOF is set to 0 by default for the nonzero constraints,
    // unless there is a prescribed manufactured solution, in which case it is
    // prescribed to p_mms.
    if (set_to_zero || !param.mms_param.enable)
    {
      constraints.constrain_dof_to_zero(constrained_pressure_dof);
    }
    else
    {
      const double pAnalytic =
        exact_solution->value(constrained_pressure_support_point, p_lower);
      constraints.set_inhomogeneity(constrained_pressure_dof, pAnalytic);
    }
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::create_zero_constraints()
{
  zero_constraints.clear();
  zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  //
  // Velocity homogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    std::set<types::boundary_id> velocity_normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_normal_flux_functions;
    std::set<types::boundary_id> velocity_tangential_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_tangential_flux_functions;
    for (const auto &[id, bc] : param.fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::no_slip ||
          bc.type == BoundaryConditions::Type::input_function ||
          bc.type == BoundaryConditions::Type::velocity_mms)
      {
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 zero_constraints,
                                                 velocity_mask);
      }
      if (bc.type == BoundaryConditions::Type::slip)
        no_flux_boundaries.insert(bc.id);
      if (bc.type == BoundaryConditions::Type::velocity_flux_mms)
      {
        velocity_normal_flux_boundaries.insert(bc.id);
        velocity_normal_flux_functions[bc.id] = param.mms.exact_velocity.get();
        velocity_tangential_flux_boundaries.insert(bc.id);
        velocity_tangential_flux_functions[bc.id] = param.mms.exact_velocity.get();
      }
    }

    // Add no velocity flux constraints
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_flux_boundaries, zero_constraints, *mapping);
    // Add nonzero normal flux velocity constraints
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      u_lower,
      velocity_normal_flux_boundaries,
      velocity_normal_flux_functions,
      zero_constraints,
      *mapping);
    // Add nonzero tangential flux velocity constraints
    VectorTools::compute_nonzero_tangential_flux_constraints(
      dof_handler,
      u_lower,
      velocity_tangential_flux_boundaries,
      velocity_tangential_flux_functions,
      zero_constraints,
      *mapping);
  }

  if (param.bc_data.fix_pressure_constant)
  {
    bool set_to_zero = true;
    constrain_pressure_point(zero_constraints, set_to_zero);
  }

  zero_constraints.close();
}

template <int dim>
void IncompressibleCHNSSolver<dim>::create_nonzero_constraints()
{
  TimerOutput::Scope t(this->computing_timer, "Create constraints");

  nonzero_constraints.clear();
  nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  //
  // Velocity inhomogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    std::set<types::boundary_id> velocity_normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_normal_flux_functions;
    std::set<types::boundary_id> velocity_tangential_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_tangential_flux_functions;
    for (const auto &[id, bc] : param.fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::no_slip)
      {
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 nonzero_constraints,
                                                 velocity_mask);
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
          velocity_mask);
      }
      if (bc.type == BoundaryConditions::Type::velocity_mms)
      {
        VectorTools::interpolate_boundary_values(*mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 *exact_solution,
                                                 nonzero_constraints,
                                                 velocity_mask);
      }
      if (bc.type == BoundaryConditions::Type::slip)
        no_flux_boundaries.insert(bc.id);
      if (bc.type == BoundaryConditions::Type::velocity_flux_mms)
      {
        // Enforce both the normal and tangential flux to be well-posed
        velocity_normal_flux_boundaries.insert(bc.id);
        velocity_normal_flux_functions[bc.id] = param.mms.exact_velocity.get();
        velocity_tangential_flux_boundaries.insert(bc.id);
        velocity_tangential_flux_functions[bc.id] = param.mms.exact_velocity.get();
      }
    }

    // Add no velocity flux constraints
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_flux_boundaries, nonzero_constraints, *mapping);
    // Add nonzero normal flux velocity constraints
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      u_lower,
      velocity_normal_flux_boundaries,
      velocity_normal_flux_functions,
      nonzero_constraints,
      *mapping);
    // Add nonzero tangential flux velocity constraints
    VectorTools::compute_nonzero_tangential_flux_constraints(
      dof_handler,
      u_lower,
      velocity_tangential_flux_boundaries,
      velocity_tangential_flux_functions,
      nonzero_constraints,
      *mapping);
  }

  if (param.bc_data.fix_pressure_constant)
  {
    bool set_to_zero = false;
    constrain_pressure_point(nonzero_constraints, set_to_zero);
  }

  nonzero_constraints.close();
}

template <int dim>
void IncompressibleCHNSSolver<dim>::create_sparsity_pattern()
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

  ////////////////////////////////////////////////////////////////
  // Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  // for (unsigned int c = 0; c < dim + 1; ++c)
  //   for (unsigned int d = 0; d < dim + 1; ++d)
  //     if (!((c == dim) && (d == dim)))
  //       coupling[c][d] = DoFTools::always;
  //     else
  //       // Pressure-pressure dofs do not couple if nonstabilized
  //       coupling[c][d] = DoFTools::none;
  // DynamicSparsityPattern dsp2(locally_relevant_dofs);
  // DoFTools::make_sparsity_pattern(
  //   dof_handler, coupling, dsp2, nonzero_constraints, false);
  ////////////////////////////////////////////////////////////////

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
void IncompressibleCHNSSolver<dim>::set_initial_conditions()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           param.initial_conditions.set_to_mms ?
                             *exact_solution :
                             *param.initial_conditions.initial_velocity,
                           newton_update,
                           velocity_mask);

#if defined(FORCE_DEAL_II_PARALLEL_VECTOR)
  newton_update.update_ghost_values();
#endif

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;

  if (!time_handler.is_steady())
  {
    // Rotate solutions
    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::set_exact_solution()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           velocity_mask);
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           pressure_mask);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void IncompressibleCHNSSolver<dim>::update_boundary_conditions()
{
  // Re-create and distribute nonzero constraints:
  this->local_evaluation_point = this->present_solution;
  this->create_nonzero_constraints();
  nonzero_constraints.distribute(this->local_evaluation_point);
  this->present_solution = this->local_evaluation_point;
}

template <int dim>
void IncompressibleCHNSSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  system_matrix = 0;

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 fe.n_dofs_per_cell(),
                                 time_handler.bdf_coefficients,
                                 param);
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
    &IncompressibleCHNSSolver::assemble_local_matrix,
    &IncompressibleCHNSSolver::copy_local_to_global_matrix,
    scratchData,
    copyData);

  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleCHNSSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

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
void IncompressibleCHNSSolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              system_matrix);
}

template <int dim>
void IncompressibleCHNSSolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 fe.n_dofs_per_cell(),
                                 time_handler.bdf_coefficients,
                                 param);
  CopyData           copyData(fe.n_dofs_per_cell());

  double max_error_over_all_elements;

  Verification::compare_analytical_matrix_with_fd(
    dof_handler,
    fe.n_dofs_per_cell(),
    *this,
    &IncompressibleCHNSSolver::assemble_local_matrix,
    &IncompressibleCHNSSolver::assemble_local_rhs,
    scratchData,
    copyData,
    present_solution,
    evaluation_point,
    local_evaluation_point,
    mpi_communicator,
    max_error_over_all_elements);

  pcout << "Max error analytical vs fd matrix is "
        << max_error_over_all_elements << std::endl;
}

template <int dim>
void IncompressibleCHNSSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  ScratchDataNS<dim> scratchData(fe,
                                 quadrature,
                                 *mapping,
                                 face_quadrature,
                                 fe.n_dofs_per_cell(),
                                 time_handler.bdf_coefficients,
                                 param);
  CopyData           copyData(fe.n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &IncompressibleCHNSSolver::assemble_local_rhs,
                  &IncompressibleCHNSSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void IncompressibleCHNSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataNS<dim>                                   &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double nu = param.physical_properties.fluids[0].kinematic_viscosity;

  //
  // Volume contributions
  //
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

    const Tensor<1, dim> dudt =
      time_handler.compute_time_derivative_at_quadrature_node(
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
  if (scratchData.has_boundary_forms && cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary())
      {
        // Open boundary condition with prescribed manufactured solution
        if (param.fluid_bc.at(scratchData.face_boundary_id[i_face]).type ==
            BoundaryConditions::Type::open_mms)
        {
          for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
          {
            const double face_JxW = scratchData.face_JxW[i_face][q];
            const auto  &n        = scratchData.face_normals[i_face][q];

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
void IncompressibleCHNSSolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              this->system_rhs);
}

template <int dim>
void IncompressibleCHNSSolver<dim>::solve_linear_system(
  const bool /*apply_inhomogeneous_constraints*/)
{
  if (param.linear_solver.method ==
      Parameters::LinearSolver::Method::direct_mumps)
  {
    if(param.linear_solver.reuse)
      DEAL_II_NOT_IMPLEMENTED();
    else
      solve_linear_system_direct(this,
                                 param.linear_solver,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints);
  }
  else if (param.linear_solver.method ==
           Parameters::LinearSolver::Method::gmres)
  {
    solve_linear_system_iterative(this,
                                  param.linear_solver,
                                  system_matrix,
                                  locally_owned_dofs,
                                  zero_constraints);
  }
  else
  {
    AssertThrow(false, ExcMessage("No known resolution method"));
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::compute_errors()
{
  TimerOutput::Scope t(this->computing_timer, "Compute errors");

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  const ComponentSelectFunction<dim> velocity_comp_select(
    std::make_pair(u_lower, u_upper), n_components);
  const ComponentSelectFunction<dim> pressure_comp_select(p_lower,
                                                          n_components);

  // Choose another quadrature rule for error computation
  const unsigned int                  n_points_1D = (dim == 2) ? 6 : 5;
  const QWitherdenVincentSimplex<dim> err_quadrature(n_points_1D);

  // L2 - u
  const double l2_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_comp_select);
  // L2 - p
  const double l2_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_comp_select);
  // Linf - u
  const double li_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::Linfty_norm,
                                               &velocity_comp_select);
  // Linf - p
  const double li_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::Linfty_norm,
                                               &pressure_comp_select);
  // H1 seminorm - u
  const double h1semi_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::H1_seminorm,
                                               &velocity_comp_select);
  // H1 seminorm - p
  const double h1semi_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::H1_seminorm,
                                               &pressure_comp_select);

  if (time_handler.is_steady())
  {
    // Steady solver: simply add errors to convergence table
    error_handler.add_reference_data("n_elm",
                                     triangulation.n_global_active_cells());
    error_handler.add_reference_data("n_dof", dof_handler.n_dofs());
    error_handler.add_steady_error("L2_u", l2_u);
    error_handler.add_steady_error("L2_p", l2_p);
    error_handler.add_steady_error("Li_u", li_u);
    error_handler.add_steady_error("Li_p", li_p);
    error_handler.add_steady_error("H1_u", h1semi_u);
    error_handler.add_steady_error("H1_p", h1semi_p);
  }
  else
  {
    const double t = time_handler.current_time;
    error_handler.add_unsteady_error("L2_u", t, l2_u);
    error_handler.add_unsteady_error("L2_p", t, l2_p);
    error_handler.add_unsteady_error("Li_u", t, li_u);
    error_handler.add_unsteady_error("Li_p", t, li_p);
    error_handler.add_unsteady_error("H1_u", t, h1semi_u);
    error_handler.add_unsteady_error("H1_p", t, h1semi_p);
  }
}

template <int dim>
void IncompressibleCHNSSolver<dim>::output_results()
{
  TimerOutput::Scope t(this->computing_timer, "Write outputs");

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
template class IncompressibleCHNSSolver<2>;
template class IncompressibleCHNSSolver<3>;