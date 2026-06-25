
#include <compare_matrix.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <error_estimation/patches.h>
#include <error_estimation/recovery_tools.h>
#include <error_estimation/solution_recovery.h>
#include <errors.h>
#include <heat_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <mesh_adaptation_tools.h>
#include <mesh_and_dof_tools.h>
#include <metric_field.h>
#include <post_processing_handler.h>
#include <post_processing_tools.h>
#include <solver_info.h>
#include <utilities.h>

template <int dim>
HeatSolver<dim>::HeatSolver(const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.output,
                                     param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param,
                                     SolverInfo::SolverType::main_physics)
  , ordering(std::make_unique<ComponentOrderingHeat>())
  , param(param)
  , time_handler(param.time_integration)
  , transient_fixed_point_data(this->param,
                               computing_timer,
                               param.time_integration.n_time_intervals,
                               mpi_communicator,
                               triangulation,
                               dof_handler,
                               present_solution,
                               previous_solutions,
                               metric_for_adaptation)
{
  create_quadrature_rules(param.finite_elements,
                          quadrature,
                          face_quadrature,
                          error_quadrature,
                          error_face_quadrature);

  if (param.finite_elements.use_quads)
  {
    mapping = std::make_unique<MappingQ<dim>>(1);
    fe      = std::make_unique<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.temperature_degree));
  }
  else
  {
    mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    fe      = std::make_unique<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.temperature_degree));
  }

  temperature_extractor = FEValuesExtractors::Scalar(0);
  temperature_mask      = fe->component_mask(temperature_extractor);

  this->param.initial_conditions.create_initial_temperature(0, 1);

  // Assign exact solution, if any
  exact_solution = param.mms.exact_temperature;

  if (param.mms_param.enable)
  {
    // Add the unknown to the error handlers
    if (param.mms_param.enable)
      for (auto &[norm, handler] : error_handlers)
        handler.create_entry("T");

    // Create source term function for the given MMS and override source terms
    source_terms = std::make_shared<HeatSolver<dim>::MMSSourceTerm>(
      time_handler.current_time, param.physical_properties, param.mms);
  }
  else
  {
    source_terms = param.source_terms.temperature_source;
  }
}

template <int dim>
void HeatSolver<dim>::MMSSourceTerm::vector_value(const Point<dim> &p,
                                                  Vector<double> &values) const
{
  const double dTdt     = mms.exact_temperature->time_derivative(p);
  const double lap_temp = mms.exact_temperature->laplacian(p);
  values[0]             = -(dTdt - lap_temp);
}

template <int dim>
void HeatSolver<dim>::reset()
{
  param.mms_param        = mms_param;
  param.mesh             = mesh_param;
  param.time_integration = time_param;

  // Clear list of files in pvd
  if (postproc_handler)
    postproc_handler->clear();

  // Clear mesh(es) and dof handler(s), and reassign immediately the
  // pointers for the first interval.
  if (!param.with_tree_based_adaptation())
    if (mms_param.current_step > 0)
      transient_fixed_point_data.reinit(param.time_integration.n_time_intervals,
                                        triangulation,
                                        dof_handler,
                                        present_solution,
                                        previous_solutions,
                                        metric_for_adaptation);
  
  // Time handler (move assign a new time handler)
  time_handler = TimeHandler(param.time_integration);
  set_time();
}

template <int dim>
void HeatSolver<dim>::set_time()
{
  // FIXME: simply add a set_time() function in the ParameterReader
  for (auto &[id, bc] : param.heat_bc)
    bc.set_time(time_handler.current_time);
  source_terms->set_time(time_handler.current_time);
  exact_solution->set_time(time_handler.current_time);
  param.physical_properties.set_time(time_handler.current_time);
  for (auto &metric_field : param.metric_fields)
    metric_field.set_time(time_handler.current_time);
}

template <int dim>
void HeatSolver<dim>::initialize()
{
  time_handler.validate_parameters(*ordering);
}

template <int dim>
void HeatSolver<dim>::initialize_interval()
{
  if (param.bc_data.n_metric_fields > 0)
    metric_for_adaptation->reinit(param.metrics.metric_for_adaptation,
                                  param,
                                  *triangulation);

  /**
   * Create the relevant patch handler and reconstruction data for each metric
   */
  ErrorEstimation::initialize_reconstruction_data(param,
                                                  *triangulation,
                                                  *mapping,
                                                  *dof_handler,
                                                  *present_solution,
                                                  *ordering,
                                                  metrics,
                                                  patch_handlers,
                                                  recoveries);
}

template <int dim>
void HeatSolver<dim>::finalize_interval()
{
  // Copy the metrics from the metric field chosen for adaptation into the
  // one in the transient fixed point data.
  // FIXME: ideally one of these is simply a non-owning pointer to the other,
  // probably the local metric here is a raw pointer to the metric for
  // adaptation

  if (param.bc_data.n_metric_fields > 0)
    for (unsigned int id = 0; id < param.metric_fields.size(); ++id)
      if (param.metric_fields[id].use_for_adaptation)
      {
        metric_for_adaptation->copy_metrics_from(*metrics[id]);
      }
}

template <int dim>
void HeatSolver<dim>::set_interval_data(const unsigned int interval_index)
{
  param.mesh.filename =
    transient_fixed_point_data.get_meshfile_name(interval_index);
  mesh_param.filename = param.mesh.filename;
  time_handler.set_time_interval(interval_index);

  if (param.time_integration.n_time_intervals > 1 &&
      param.time_integration.verbosity == Parameters::Verbosity::verbose)
  {
    pcout << std::endl;
    pcout << "Time sub-interval " << interval_index + 1 << "/"
          << param.time_integration.n_time_intervals << " : t in ["
          << time_handler.initial_time << ", " << time_handler.final_time << "]"
          << std::endl;
    pcout << "Reading mesh file: " << param.mesh.filename << std::endl;
    pcout << std::endl;
  }

  // Get the triangulation, dof handler, solution vectors and metric
  // for this time subinterval.
  transient_fixed_point_data.set_interval_data(interval_index,
                                               triangulation,
                                               dof_handler,
                                               present_solution,
                                               previous_solutions,
                                               metric_for_adaptation);

  // Create the post-processing handler.
  // This requires the description of the variable from the derived solvers,
  // so I don't think this can be done in the constructor of the base class.
  if (!postproc_handler)
  {
    const auto description = get_variables_description();
    postproc_handler       = std::make_unique<PostProcessingHandler<dim>>(
      param, *triangulation, *dof_handler, description);
  }
  else
    postproc_handler->attach_triangulation_and_dof_handler(*triangulation,
                                                           *dof_handler);

  // Create a direct solver for each interval
  direct_solver_reuse =
    std::make_unique<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);
}

template <int dim>
void HeatSolver<dim>::run_time_subinterval(const unsigned int interval_index)
{
  set_interval_data(interval_index);

  if (should_create_triangulation())
    MeshTools::read_mesh(*triangulation, param);

  setup_dofs();
  initialize_interval();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();

  if (interval_index == 0)
    set_initial_conditions();
  else
    transient_fixed_point_data.transfer_solution_between_intervals(
      interval_index,
      *mapping,
      *exact_solution,
      time_handler,
      locally_relevant_dofs,
      dofs_to_component);

  // For unsteady simulations, postprocess either the initial condition, or the
  // initial solution on this time interval. For unsteady simulations with mesh
  // adaptation with a Riemannian metric, this is needed to obtain an adapted
  // mesh that includes the initial condition.
  postprocess_solution();

  /**
   * Apply initial refinement.
   */
  if (!time_handler.is_steady() && param.with_tree_based_adaptation())
    for (unsigned int step = 0;
         step < param.mesh.adaptation.tree_amr.n_prerefinement_steps;
         ++step)
    {
      update_boundary_conditions();
      set_initial_conditions(false);
      adapt_mesh();
      output_results(/* is_prerefinement_step = */ true, step);
    }

  while (!time_handler.is_finished())
  {
    do
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
        if (param.nonlinear_solver.compare_jacobian_with_finite_differences)
          compare_analytical_matrix_with_fd();

        if (param.debug.apply_exact_solution)
          set_exact_solution();
        else
          solve_nonlinear_problem(time_handler);
      }
    }
    while (!time_handler.is_timestep_accepted(*present_solution,
                                              *previous_solutions));

    postprocess_solution();

    /**
     * Adapt the tree-based mesh during an unsteady simulation, if the current
     * time step iteration matches the prescribed frequency.
     *
     * For steady-state simulations, the mesh is adapted after the finalize()
     * function is called, so that the registered number of mesh elements
     * and dofs matches the computed error for convergence studies.
     */
    if (should_adapt_tree_based_mesh(time_handler))
      adapt_mesh();

    time_handler.rotate_solutions(*present_solution, *previous_solutions);
  }

  finalize_interval();
}

template <int dim>
void HeatSolver<dim>::run()
{
  reset();
  initialize();
  create_scratch_data();

  for (unsigned int i = 0; i < param.time_integration.n_time_intervals; ++i)
    run_time_subinterval(i);

  finalize();

  /**
   * If using a riemannian metric to adapt the mesh(es), perform all the
   * adaptations at the end of all time intervals (as it requires a global
   * scaling factor).
   */
  // if (time_handler.is_steady() || param.with_metric_based_adaptation())
  if (should_adapt_mesh_at_end_of_intervals(time_handler))
    adapt_mesh();
}

template <int dim>
void HeatSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler->distribute_dofs(*fe);

  pcout << "Number of degrees of freedom: " << dof_handler->n_dofs()
        << std::endl;

  locally_owned_dofs    = dof_handler->locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(*dof_handler);

  // Setup the dofs_to_component vector
  fill_dofs_to_component(*dof_handler,
                         locally_relevant_dofs,
                         dofs_to_component);

  // Attach data to time error estimator
  time_handler.attach_data_to_error_estimator(*ordering,
                                              locally_relevant_dofs,
                                              dofs_to_component);

  // Initialize parallel vectors
  present_solution->reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, comm);

  local_evaluation_point.reinit(locally_owned_dofs, comm);
  newton_update.reinit(locally_owned_dofs, comm);
  system_rhs.reinit(locally_owned_dofs, comm);

  // Allocate for previous BDF solutions
  previous_solutions->clear();
  previous_solutions->resize(time_handler.n_previous_solutions);
  for (auto &previous_sol : *previous_solutions)
    previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
}

template <int dim>
void HeatSolver<dim>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(
    *fe, *mapping, *quadrature, *face_quadrature, time_handler, param);
}

template <int dim>
void HeatSolver<dim>::create_base_constraints(
  const bool                 homogeneous,
  AffineConstraints<double> &constraints)
{
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  if (param.with_tree_based_adaptation())
    DoFTools::make_hanging_node_constraints(*dof_handler, constraints);

  Functions::ZeroFunction<dim> zero_fun(1);
  const Function<dim>         *fun_ptr;

  for (const auto &[id, bc] : param.heat_bc)
  {
    if (bc.type == BoundaryConditions::Type::input_function)
    {
      fun_ptr = homogeneous ?
                &zero_fun :
                static_cast<const Function<dim> *>(bc.temperature.get());
      VectorTools::interpolate_boundary_values(
        *mapping, *dof_handler, bc.id, *fun_ptr, constraints, temperature_mask);
    }
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      fun_ptr = homogeneous ? &zero_fun : exact_solution.get();
      VectorTools::interpolate_boundary_values(
        *mapping, *dof_handler, bc.id, *fun_ptr, constraints, temperature_mask);
    }
  }

  constraints.close();
}

template <int dim>
void HeatSolver<dim>::create_zero_constraints()
{
  create_base_constraints(true, zero_constraints);
}

template <int dim>
void HeatSolver<dim>::create_nonzero_constraints()
{
  create_base_constraints(false, nonzero_constraints);
}

template <int dim>
void HeatSolver<dim>::create_sparsity_pattern()
{
#if defined(FEZ_WITH_PETSC)
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(*dof_handler,
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
  DoFTools::make_sparsity_pattern(*dof_handler, dsp, nonzero_constraints);
  dsp.compress();
  system_matrix.reinit(dsp);
#endif
}

template <int dim>
void HeatSolver<dim>::set_initial_conditions(const bool rotate_solutions)
{
  const Function<dim> *temperature_fun =
    param.initial_conditions.set_to_mms ?
      exact_solution.get() :
      param.initial_conditions.initial_temperature.get();

  VectorTools::interpolate(
    *mapping, *dof_handler, *temperature_fun, newton_update, temperature_mask);

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  *present_solution = newton_update;
  evaluation_point  = newton_update;

  if (rotate_solutions)
    // FIXME: WHAT ABOUT THIS ROTATION?????????
    time_handler.rotate_solutions(*present_solution, *previous_solutions);
}

template <int dim>
void HeatSolver<dim>::set_exact_solution()
{
  TimerOutput::Scope t(computing_timer, "Set exact solution");

  VectorTools::interpolate(*mapping,
                           *dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           temperature_mask);
  evaluation_point  = local_evaluation_point;
  *present_solution = local_evaluation_point;
}

template <int dim>
void HeatSolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = *present_solution;
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point  = local_evaluation_point;
  *present_solution = local_evaluation_point;
}

template <int dim>
void HeatSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrix");

  system_matrix = 0;

  CopyData copy_data(*fe);

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(dof_handler->begin_active(),
                  dof_handler->end(),
                  *this,
                  &HeatSolver::assemble_local_matrix,
                  &HeatSolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void HeatSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(
    cell, evaluation_point, *previous_solutions, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix();
  local_matrix       = 0;

  const double bdf_c0 = time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW        = scratch_data.JxW[q];
    const auto  &phi_t      = scratch_data.phi_t[q];
    const auto  &grad_phi_t = scratch_data.grad_phi_t[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_t_i      = phi_t[i];
      const auto &grad_phi_t_i = grad_phi_t[i];

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        local_matrix(i, j) +=
          (bdf_c0 * phi_t_i * phi_t[j] + grad_phi_t_i * grad_phi_t[j]) * JxW;
      }
    }
  }
  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void HeatSolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                              copy_data.dof_indices(),
                                              system_matrix);
}

template <int dim>
void HeatSolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    &HeatSolver::assemble_local_matrix,
    &HeatSolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim>
void HeatSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler->begin_active(),
                  dof_handler->end(),
                  *this,
                  &HeatSolver::assemble_local_rhs,
                  &HeatSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void HeatSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(
    cell, evaluation_point, *previous_solutions, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs();
  local_rhs       = 0;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW[q];

    const auto &temperature_value    = scratch_data.temperature_values[q];
    const auto &temperature_gradient = scratch_data.temperature_gradients[q];
    const auto &source_term_temperature =
      scratch_data.source_term_temperature[q];

    const double dTdt = time_handler.compute_time_derivative_at_quadrature_node(
      q, temperature_value, scratch_data.previous_temperature_values);

    const auto &phi_t      = scratch_data.phi_t[q];
    const auto &grad_phi_t = scratch_data.grad_phi_t[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      local_rhs(i) -= (phi_t[i] * (dTdt + source_term_temperature) +
                       grad_phi_t[i] * temperature_gradient) *
                      JxW;
    }
  }

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void HeatSolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                              copy_data.dof_indices(),
                                              system_rhs);
}

template <int dim>
void HeatSolver<dim>::solve_linear_system()
{
  const auto &linear_solver_param = param.linear_solver.at(this->solver_type);

  if (linear_solver_param.method ==
      Parameters::LinearSolver::Method::direct_mumps)
  {
    if (linear_solver_param.reuse)
    {
      solve_linear_system_direct(this,
                                 linear_solver_param,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints,
                                 *direct_solver_reuse);
    }
    else
      solve_linear_system_direct(this,
                                 linear_solver_param,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints);
  }
  else if (linear_solver_param.method == Parameters::LinearSolver::Method::cg)
  {
    solve_linear_system_cg(this,
                           linear_solver_param,
                           system_matrix,
                           locally_owned_dofs,
                           zero_constraints);
  }
  else if (linear_solver_param.method ==
           Parameters::LinearSolver::Method::gmres)
  {
    solve_linear_system_iterative(this,
                                  linear_solver_param,
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
void HeatSolver<dim>::output_results(const bool         is_pre_refinement_step,
                                     const unsigned int pre_refinement_step)
{
  TimerOutput::Scope t(computing_timer, "Write outputs");
  postproc_handler->output_fields(*mapping,
                                  *present_solution,
                                  time_handler,
                                  is_pre_refinement_step,
                                  pre_refinement_step);
}

template <int dim>
void HeatSolver<dim>::compute_and_add_errors(
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
      compute_error_norm<dim, LA::ParVectorType>(*triangulation,
                                                 mapping,
                                                 *dof_handler,
                                                 *present_solution,
                                                 exact_solution,
                                                 cellwise_errors,
                                                 *error_quadrature,
                                                 norm,
                                                 &comp_function);
    error_handlers.at(norm).add_error(field_name, err, time);
  }
}

template <int dim>
void HeatSolver<dim>::compute_errors()
{
  TimerOutput::Scope t(computing_timer, "Compute errors");

  const unsigned int n_active_cells = triangulation->n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  /**
   * Compute errors on temperature
   */
  const ComponentSelectFunction<dim> temperature_comp_select(0, 1);
  compute_and_add_errors(
    *mapping, *exact_solution, cellwise_errors, temperature_comp_select, "T");
}

template <int dim>
void HeatSolver<dim>::compute_reconstructions()
{
  TimerOutput::Scope t(computing_timer, "Compute recovery");

  // Compute the reconstructions for this time step
  for (unsigned int i = 0; i < recoveries.size(); ++i)
  {
    Assert(recoveries[i], ExcInternalError());
    recoveries[i]->reconstruct_fields(*present_solution);
    recoveries[i]->write_pvtu(*mapping, "recovery_heat");
  }
}

template <int dim>
void HeatSolver<dim>::compute_riemannian_metric()
{
  TimerOutput::Scope t(computing_timer, "Compute Riemannian metric");

  Assert(param.bc_data.n_metric_fields > 0, ExcInternalError());

  // Update metric with its matching reconstruction operator
  for (unsigned int i = 0; i < metrics.size(); ++i)
  {
    Assert(metrics[i], ExcInternalError());
    Assert(recoveries[i], ExcInternalError());
    metrics[i]->increment_anisotropic_measure(*recoveries[i], time_handler);
  }

  // Intersect one at a time (order dependent!)
  for (unsigned int id = 0; id < param.metric_fields.size(); ++id)
    for (const unsigned int other_id :
         param.metric_fields[id].intersection.intersect_with)
      metrics[id]->intersect_with(*metrics[other_id]);
}

template <int dim>
void HeatSolver<dim>::compute_error_estimate()
{
  TimerOutput::Scope t(computing_timer, "Compute Kelly error estimate");

  temperature_error_on_cells.reinit(triangulation->n_active_cells());
  KellyErrorEstimator<dim>::estimate(
    *mapping,
    *dof_handler,
    *error_face_quadrature,
    std::map<types::boundary_id, const Function<dim> *>(),
    *present_solution,
    temperature_error_on_cells);
}

template <int dim>
void HeatSolver<dim>::postprocess_solution()
{
  if (should_compute_errors(time_handler))
    compute_errors();

  output_results();

  if (should_compute_reconstructions(param, time_handler))
    compute_reconstructions();

  if (should_compute_riemannian_metric(param, time_handler))
    compute_riemannian_metric();
}

template <int dim>
void HeatSolver<dim>::adapt_mesh()
{
  if (param.with_tree_based_adaptation())
    compute_error_estimate();

  // Adapt the mesh(es): either with a riemannian metric, or with the cellwise
  // error criteria.
  transient_fixed_point_data.adapt_meshes(time_handler,
                                          temperature_error_on_cells);

  // Re-setup up the dof_handler, constraints and linear algebra structures.
  // For steady-state convergence studies, we're doing the work twice, here
  // and at the beginning of the next convergence step, but it's OK.
  if (param.with_tree_based_adaptation())
  {
    setup_dofs();
    create_zero_constraints();
    create_nonzero_constraints();
    create_sparsity_pattern();
    postproc_handler->attach_triangulation_and_dof_handler(*triangulation,
                                                           *dof_handler);
    transient_fixed_point_data.transfer_solution_between_refinements(
      locally_relevant_dofs, nonzero_constraints);
  }
}

template <int dim>
void HeatSolver<dim>::finalize()
{
  // Add the reference data to the error handlers.
  // This is done at the end to have the number of time steps effectively done
  // available, when the simulation uses adaptive time stepping.
  if (should_add_error_reference_data(time_handler))
    for (auto &[norm, handler] : error_handlers)
      handler.add_reference_data(time_handler,
                                 transient_fixed_point_data,
                                 *triangulation,
                                 *dof_handler);

  // Write a summary of each time subinterval
  if (param.transient_fixed_point_adaptation_enabled() &&
      param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose)
    transient_fixed_point_data.write_summary(time_handler, std::cout);

  postproc_handler->write_pvd();
}

// Explicit instantiation
template class HeatSolver<2>;
template class HeatSolver<3>;
