
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_solver.h>
#include <mesh.h>
#include <navier_stokes_solver.h>
#include <utilities.h>

template <int dim>
NavierStokesSolver<dim>::NavierStokesSolver(const ParameterReader<dim> &param,
                                            const bool with_moving_mesh)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param)
  , param(param)
  , with_moving_mesh(with_moving_mesh)
  , quadrature(QGaussSimplex<dim>(4))
  , error_quadrature(QWitherdenVincentSimplex<dim>((dim == 2) ? 6 : 5))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , error_face_quadrature(QWitherdenVincentSimplex<dim - 1>((dim == 2) ? 6 : 5))
  , triangulation(mpi_communicator)
  , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  if (param.mms_param.enable)
  {
    for (auto norm : param.mms_param.norms_to_compute)
    {
      error_handlers[norm]->create_entry("u");
      error_handlers[norm]->create_entry("p");
      if (with_moving_mesh)
        error_handlers[norm]->create_entry("x");
    }
  }

  // Direct solver
  direct_solver_reuse =
    std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);
}

template <int dim>
void NavierStokesSolver<dim>::reset()
{
  // FIXME: This is not very clean: the derived class has the full parameters,
  // and the base class GenericSolver has a mesh and time param to be able to
  // modify the mesh file and/or time step in a convergence loop.
  param.mms_param.current_step = mms_param.current_step;
  param.mms_param.mesh_suffix  = mms_param.mesh_suffix;
  param.mesh.filename          = mesh_param.filename;
  param.time_integration.dt    = time_param.dt;

  // Mesh
  triangulation.clear();

  // Direct solver
  direct_solver_reuse =
    std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  // Time handler (move assign a new time handler)
  time_handler = TimeHandler(param.time_integration);
  this->set_time();

  // Pressure DOF
  constrained_pressure_dof = numbers::invalid_dof_index;

  // Initial mesh position
  initial_positions.clear();

  reset_solver_specific_data();
}

template <int dim>
void NavierStokesSolver<dim>::set_time()
{
  for (auto &[id, bc] : param.fluid_bc)
    bc.set_time(time_handler.current_time);

  if (with_moving_mesh)
    for (auto &[id, bc] : param.pseudosolid_bc)
      bc.set_time(time_handler.current_time);

  source_terms->set_time(time_handler.current_time);
  exact_solution->set_time(time_handler.current_time);
  param.physical_properties.set_time(time_handler.current_time);

  set_solver_specific_time();
}

template <int dim>
void NavierStokesSolver<dim>::run()
{
  reset();
  read_mesh(triangulation, param);
  setup_dofs();

  if (param.bc_data.enforce_zero_mean_pressure)
    create_zero_mean_pressure_constraints_data();
  create_solver_specific_constraints_data();

  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
  output_results();

  while (!time_handler.is_finished())
  {
    time_handler.advance(pcout);
    set_time();
    update_boundary_conditions();

    if (time_handler.is_starting_step() &&
        param.time_integration.bdfstart ==
          Parameters::TimeIntegration::BDFStart::initial_condition)
    {
      if (param.mms_param.enable || param.debug.apply_exact_solution)
        // Convergence study: start with exact solution at first time step
        set_exact_solution();
      else
        // Repeat initial condition
        set_initial_conditions();
    }
    else
    {
      // Entering the Newton solver with a solution satisfying the nonzero
      // constraints, which were applied in update_boundary_condition().
      if (param.debug.compare_analytical_jacobian_with_fd)
        compare_analytical_matrix_with_fd();

      if (param.debug.apply_exact_solution)
        set_exact_solution();
      else
        solve_nonlinear_problem(false);
    }

    postprocess_solution();

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
void NavierStokesSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(this->get_fe_system());

  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Initialize parallel vectors
  present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, comm);

  local_evaluation_point.reinit(locally_owned_dofs, comm);
  newton_update.reinit(locally_owned_dofs, comm);
  system_rhs.reinit(locally_owned_dofs, comm);

  // Allocate for previous BDF solutions
  previous_solutions.clear();
  previous_solutions.resize(time_handler.n_previous_solutions);
  for (auto &previous_sol : previous_solutions)
  {
    previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  }

  if (with_moving_mesh)
  {
    // Initialize mesh position directly from the triangulation.
    // The parallel vector storing the mesh position is local_evaluation_point,
    // because this is the one to modify when computing finite differences.
    VectorTools::get_position_vector(*fixed_mapping,
                                     dof_handler,
                                     local_evaluation_point,
                                     position_mask);
    local_evaluation_point.compress(VectorOperation::insert);
    evaluation_point = local_evaluation_point;

    // Also store them in initial_positions, for postprocessing:
    DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                         dof_handler,
                                         initial_positions,
                                         position_mask);

    // Create the solution-dependent mapping
    moving_mapping =
      std::make_shared<MappingFEField<dim, dim, LA::ParVectorType>>(
        dof_handler, evaluation_point, position_mask);
  }
  else
  {
    moving_mapping = fixed_mapping;
  }

  // For unsteady simulation, add the number of elements, dofs and/or the time
  // step to the error handler, once per convergence run.
  if (!time_handler.is_steady() && param.mms_param.enable)
    for (auto &[norm, handler] : error_handlers)
    {
      handler->add_reference_data("n_elm",
                                  triangulation.n_global_active_cells());
      handler->add_reference_data("n_dof", dof_handler.n_dofs());
      handler->add_time_step(time_handler.initial_dt);
    }
}

template <int dim>
void NavierStokesSolver<dim>::create_zero_mean_pressure_constraints_data()
{
  BoundaryConditions::create_zero_mean_pressure_constraints_data(
    triangulation,
    dof_handler,
    locally_relevant_dofs,
    *moving_mapping,
    quadrature,
    ordering->p_lower,
    constrained_pressure_dof,
    zero_mean_pressure_weights);
}

template <int dim>
void NavierStokesSolver<dim>::create_base_constraints(
  const bool                 homogeneous,
  AffineConstraints<double> &constraints)
{
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  /**
   * If relevant, apply mesh boundary conditions first, as they affect
   * the evaluation of the fields on the moving mesh.
   */
  if (with_moving_mesh)
  {
    BoundaryConditions::apply_mesh_position_boundary_conditions(
      homogeneous,
      ordering->x_lower,
      ordering->n_components,
      dof_handler,
      *fixed_mapping,
      param.pseudosolid_bc,
      *exact_solution,
      *param.mms.exact_mesh_position,
      constraints);
  }

  BoundaryConditions::apply_velocity_boundary_conditions(
    homogeneous,
    ordering->u_lower,
    ordering->n_components,
    dof_handler,
    *moving_mapping,
    param.fluid_bc,
    *exact_solution,
    *param.mms.exact_velocity,
    constraints);

  if (param.bc_data.fix_pressure_constant)
  {
    // The pressure DOF is set to 0 by default for the nonzero constraints too,
    // unless there is a prescribed manufactured solution, in which case it is
    // prescribed to p_mms.
    bool set_to_zero = true;
    if (!homogeneous && param.mms_param.enable)
      set_to_zero = false;

    BoundaryConditions::constrain_pressure_point(
      dof_handler,
      locally_relevant_dofs,
      *moving_mapping,
      *exact_solution,
      ordering->p_lower,
      set_to_zero,
      constraints,
      constrained_pressure_dof,
      constrained_pressure_support_point);
  }

  if (param.bc_data.enforce_zero_mean_pressure)
    BoundaryConditions::add_zero_mean_pressure_constraints(
      constraints,
      locally_relevant_dofs,
      constrained_pressure_dof,
      zero_mean_pressure_weights);

  /**
   * Do not close the constraints here, as derived solvers may need
   * to add boundary conditions on their own fields (e.g., Cahn Hilliard)
   */
  // constraints.close();
}

template <int dim>
void NavierStokesSolver<dim>::create_zero_constraints()
{
  create_base_constraints(true, zero_constraints);
  create_solver_specific_zero_constraints();
  zero_constraints.close();
}

template <int dim>
void NavierStokesSolver<dim>::create_nonzero_constraints()
{
  create_base_constraints(false, nonzero_constraints);
  create_solver_specific_nonzero_constraints();
  nonzero_constraints.close();
}

template <int dim>
void NavierStokesSolver<dim>::set_initial_conditions()
{
  /**
   * Mesh position should be evaluated and updated *BEFORE* evaluating fields on
   * moving mapping. This matters in the rare cases when the initial mesh
   * position is *not* the fixed_mapping.
   */

  const Function<dim> *velocity_fun =
    param.initial_conditions.set_to_mms ?
      exact_solution.get() :
      param.initial_conditions.initial_velocity.get();

  if (with_moving_mesh)
  {
    FixedMeshPosition<dim> fixed_mesh(ordering->x_lower,
                                      ordering->n_components);

    const Function<dim> *mesh_fun =
      param.initial_conditions.set_to_mms ? exact_solution.get() : &fixed_mesh;

    // Set mesh position with fixed mapping
    VectorTools::interpolate(
      *fixed_mapping, dof_handler, *mesh_fun, newton_update, position_mask);

    // Update MappingFEField *BEFORE* interpolating velocity
    evaluation_point = newton_update;
  }

  // Set velocity with moving mapping
  VectorTools::interpolate(
    *moving_mapping, dof_handler, *velocity_fun, newton_update, velocity_mask);

  // Set other solver-specific fields on moving mesh (e.g., CHNS tracer)
  set_solver_specific_initial_conditions();

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  evaluation_point = newton_update;

  if (!time_handler.is_steady())
  {
    // Rotate solutions
    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
}

template <int dim>
void NavierStokesSolver<dim>::set_exact_solution()
{
  if (with_moving_mesh)
  {
    // Update mesh position *BEFORE* evaluating fields on moving mapping.
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             *exact_solution,
                             local_evaluation_point,
                             position_mask);

    // Update MappingFEField *BEFORE* interpolating velocity/pressure
    evaluation_point = local_evaluation_point;
  }

  // Set velocity and pressure with moving mapping
  VectorTools::interpolate(*moving_mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           velocity_mask);
  VectorTools::interpolate(*moving_mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           pressure_mask);

  if (param.bc_data.enforce_zero_mean_pressure)
  {
    present_solution        = local_evaluation_point;
    const double p_mean     = VectorTools::compute_mean_value(*moving_mapping,
                                                          dof_handler,
                                                          quadrature,
                                                          present_solution,
                                                          ordering->p_lower);
    const double p_mms_mean = compute_global_mean_value(*exact_solution,
                                                        ordering->p_lower,
                                                        dof_handler,
                                                        *moving_mapping);

    pcout << "Before removing pressure: " << p_mean << std::endl;
    pcout << "Analytic mean is        : " << p_mms_mean << std::endl;
    BoundaryConditions::remove_mean_pressure(pressure_mask,
                                             dof_handler,
                                             p_mean,
                                             local_evaluation_point);
    present_solution     = local_evaluation_point;
    const double p_mean2 = VectorTools::compute_mean_value(*moving_mapping,
                                                           dof_handler,
                                                           quadrature,
                                                           present_solution,
                                                           ordering->p_lower);
    pcout << "After  removing pressure: " << p_mean2 << std::endl;
  }

  set_solver_specific_exact_solution();

  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void NavierStokesSolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = present_solution;

  if (with_moving_mesh)
  {
    // Create and apply the inhomogeneous constraints a first time
    // to apply mesh position boundary conditions.
    // Then update the moving mapping (through the evaluation point),
    // and evaluate the inhomogeneous velocity (and other) BC on the
    // updated mapping.
    create_nonzero_constraints();

    // Update the moving mapping
    nonzero_constraints.distribute(local_evaluation_point);
    evaluation_point = local_evaluation_point;
  }

  // Create and apply inhomogeneous BC for non-position fields.
  // The position BC are re-applied, but did not change.
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void NavierStokesSolver<dim>::solve_linear_system(
  const bool /*apply_inhomogeneous_constraints*/)
{
  if (param.linear_solver.method ==
      Parameters::LinearSolver::Method::direct_mumps)
  {
    if (param.linear_solver.reuse)
    {
      solve_linear_system_direct(this,
                                 param.linear_solver,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints,
                                 *direct_solver_reuse);
    }
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
void NavierStokesSolver<dim>::compute_and_add_errors(
  const Mapping<dim>                 &mapping,
  const Function<dim>                &exact_solution,
  Vector<double>                     &cellwise_errors,
  const ComponentSelectFunction<dim> &comp_function,
  const std::string                  &field_name)
{
  const double time = time_handler.current_time;
  for (auto norm : param.mms_param.norms_to_compute)
  {
    const double err =
      compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                                 mapping,
                                                 dof_handler,
                                                 present_solution,
                                                 exact_solution,
                                                 cellwise_errors,
                                                 error_quadrature,
                                                 norm,
                                                 &comp_function);
    error_handlers.at(norm)->add_error(field_name, err, time);
  }
}


template <int dim>
void NavierStokesSolver<dim>::compute_errors()
{
  TimerOutput::Scope t(this->computing_timer, "Compute errors");

  const unsigned int n_components = ordering->n_components;
  const unsigned int u_lower      = ordering->u_lower;
  const unsigned int p_lower      = ordering->p_lower;

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  if (time_handler.is_steady())
    for (auto norm : param.mms_param.norms_to_compute)
    {
      error_handlers.at(norm)->add_reference_data(
        "n_elm", triangulation.n_global_active_cells());
      error_handlers.at(norm)->add_reference_data("n_dof",
                                                  dof_handler.n_dofs());
    }

  /**
   * Set the function pointer to use, depending on whether the mean pressure
   * should be subtracted or not.
   */
  std::shared_ptr<Function<dim>> used_exact_solution = exact_solution;

  if (param.bc_data.enforce_zero_mean_pressure)
  {
    // Mean pressure value
    const double p_mean = VectorTools::compute_mean_value(
      *moving_mapping, dof_handler, quadrature, present_solution, p_lower);

    AssertThrow(std::abs(p_mean) < 1e-10,
                ExcMessage(
                  "Mean pressure should be zero, but it's not : p_mean = " +
                  std::to_string(p_mean)));

    if (param.mms_param.enable)
    {
      // Use a function wrapper where the pressure mean is subtracted
      const double p_mms_mean = compute_global_mean_value(*exact_solution,
                                                          p_lower,
                                                          dof_handler,
                                                          *moving_mapping);

      if (param.mms_param.subtract_mean_pressure)
        used_exact_solution =
          std::make_shared<PressureMeanSubtractedFunction<dim>>(*exact_solution,
                                                                p_mms_mean,
                                                                p_lower);
      else
        // Use the manufactured pressure which is then assumed to have zero
        // mean. Throw an error if it's not the case.
        AssertThrow(
          std::abs(p_mms_mean) < 1e-6,
          ExcMessage(
            "You are comparing a discrete zero-mean pressure with a "
            "manufactured "
            "pressure which is not zero-mean. The mean exact pressure is " +
            std::to_string(p_mms_mean)));
    }
  }

  /**
   * Compute errors on velocity, pressure and position if applicable
   */
  const ComponentSelectFunction<dim> velocity_comp_select(
    std::make_pair(u_lower, u_lower + dim), n_components);
  const ComponentSelectFunction<dim> pressure_comp_select(p_lower,
                                                          n_components);

  compute_and_add_errors(*moving_mapping,
                         *used_exact_solution,
                         cellwise_errors,
                         velocity_comp_select,
                         "u");
  compute_and_add_errors(*moving_mapping,
                         *used_exact_solution,
                         cellwise_errors,
                         pressure_comp_select,
                         "p");
  if (with_moving_mesh)
  {
    // Error on mesh position
    const unsigned int                 x_lower = ordering->x_lower;
    const ComponentSelectFunction<dim> position_comp_select(
      std::make_pair(x_lower, x_lower + dim), n_components);
    compute_and_add_errors(*fixed_mapping,
                           *exact_solution,
                           cellwise_errors,
                           position_comp_select,
                           "x");
  }

  compute_solver_specific_errors();
}

template <int dim>
void NavierStokesSolver<dim>::postprocess_solution()
{
  output_results();

  if (param.mms_param.enable)
    compute_errors();

  solver_specific_post_processing();
}

// Explicit instantiation
template class NavierStokesSolver<2>;
template class NavierStokesSolver<3>;