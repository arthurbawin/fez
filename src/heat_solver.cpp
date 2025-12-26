
#include <compare_matrix.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <error_estimation/patches.h>
#include <errors.h>
#include <heat_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <utilities.h>

template <int dim>
HeatSolver<dim>::HeatSolver(const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param)
  , param(param)
  , fe(FE_SimplexP<dim>(param.finite_elements.temperature_degree), 1)
  , quadrature(QGaussSimplex<dim>(4))
  , error_quadrature(QWitherdenVincentSimplex<dim>((dim == 2) ? 6 : 5))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , error_face_quadrature(QWitherdenVincentSimplex<dim - 1>((dim == 2) ? 6 : 5))
  , triangulation(mpi_communicator)
  , mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  temperature_extractor = FEValuesExtractors::Scalar(0);
  temperature_mask      = fe.component_mask(temperature_extractor);

  this->param.initial_conditions.create_initial_temperature(0, 1);

  if (param.mms_param.enable)
  {
    // Add the unknown "u" to the error handlers
    if (param.mms_param.enable)
      for (auto norm : param.mms_param.norms_to_compute)
        error_handlers[norm]->create_entry("T");

    // Assign the manufactured solution
    exact_solution = param.mms.exact_temperature;

    // Create source term function for the given MMS and override source terms
    source_terms = std::make_shared<HeatSolver<dim>::MMSSourceTerm>(
      time_handler.current_time, param.physical_properties, param.mms);
  }
  else
  {
    source_terms   = param.source_terms.fluid_source;
    exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(1);
  }

  // Create direct solver
  direct_solver_reuse =
    std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);
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
  set_time();
}

template <int dim>
void HeatSolver<dim>::set_time()
{
  for (auto &[id, bc] : param.heat_bc)
    bc.set_time(time_handler.current_time);
  source_terms->set_time(time_handler.current_time);
  exact_solution->set_time(time_handler.current_time);
  param.physical_properties.set_time(time_handler.current_time);
}

template <int dim>
void HeatSolver<dim>::run()
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
void HeatSolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(fe);

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
void HeatSolver<dim>::create_base_constraints(
  const bool                 homogeneous,
  AffineConstraints<double> &constraints)
{
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

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
        *mapping, dof_handler, bc.id, *fun_ptr, constraints, temperature_mask);
    }
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      fun_ptr = homogeneous ? &zero_fun : exact_solution.get();
      VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, bc.id, *fun_ptr, constraints, temperature_mask);
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
void HeatSolver<dim>::set_initial_conditions()
{
  const Function<dim> *temperature_fun =
    param.initial_conditions.set_to_mms ?
      exact_solution.get() :
      param.initial_conditions.initial_temperature.get();

  VectorTools::interpolate(
    *mapping, dof_handler, *temperature_fun, newton_update, temperature_mask);

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
void HeatSolver<dim>::set_exact_solution()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           temperature_mask);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void HeatSolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = present_solution;
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void HeatSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrix");

  system_matrix = 0;

  ScratchDataHeat<dim> scratchData(fe,
                                   *mapping,
                                   quadrature,
                                   face_quadrature,
                                   time_handler.bdf_coefficients,
                                   param);
  CopyData             copyData(fe.n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &HeatSolver::assemble_local_matrix,
                  &HeatSolver::copy_local_to_global_matrix,
                  scratchData,
                  copyData);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void HeatSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataHeat<dim>                                 &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double bdf_c0 = time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW        = scratchData.JxW[q];
    const auto  &phi_t      = scratchData.phi_t[q];
    const auto  &grad_phi_t = scratchData.grad_phi_t[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      {
        local_matrix(i, j) +=
          (bdf_c0 * phi_t[i] * phi_t[j] + grad_phi_t[i] * grad_phi_t[j]) * JxW;
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void HeatSolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              system_matrix);
}

template <int dim>
void HeatSolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchDataHeat<dim> scratchData(fe,
                                   *mapping,
                                   quadrature,
                                   face_quadrature,
                                   time_handler.bdf_coefficients,
                                   param);
  CopyData             copyData(fe.n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    dof_handler,
    fe.n_dofs_per_cell(),
    *this,
    &HeatSolver::assemble_local_matrix,
    &HeatSolver::assemble_local_rhs,
    scratchData,
    copyData,
    present_solution,
    evaluation_point,
    local_evaluation_point,
    mpi_communicator);

  pcout << "Max absolute error analytical vs fd matrix is " << errors.first
        << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > param.debug.analytical_jacobian_absolute_tolerance)
    pcout << "Max relative error analytical vs fd matrix is " << errors.second
          << std::endl;
}

template <int dim>
void HeatSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  ScratchDataHeat<dim> scratchData(fe,
                                   *mapping,
                                   quadrature,
                                   face_quadrature,
                                   time_handler.bdf_coefficients,
                                   param);
  CopyData             copyData(fe.n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &HeatSolver::assemble_local_rhs,
                  &HeatSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void HeatSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataHeat<dim>                                 &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW = scratchData.JxW[q];

    const auto &temperature_value    = scratchData.temperature_values[q];
    const auto &temperature_gradient = scratchData.temperature_gradients[q];
    const auto &source_term_temperature =
      scratchData.source_term_temperature[q];

    const double dTdt = time_handler.compute_time_derivative_at_quadrature_node(
      q, temperature_value, scratchData.previous_temperature_values);

    const auto &phi_t      = scratchData.phi_t[q];
    const auto &grad_phi_t = scratchData.grad_phi_t[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      local_rhs(i) -= (phi_t[i] * (dTdt + source_term_temperature) +
                       grad_phi_t[i] * temperature_gradient) *
                      JxW;
    }
  }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void HeatSolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              system_rhs);
}

template <int dim>
void HeatSolver<dim>::solve_linear_system(
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
void HeatSolver<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Write outputs");

  if (param.output.write_results)
  {
    std::vector<std::string> solution_names(1, "temperature");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);

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
    data_out.add_data_vector(subdomain, "subdomain", DataOut<dim>::type_cell_data);

    data_out.build_patches(*mapping, 2);

    // Export regular time step
    data_out.write_vtu_with_pvtu_record(param.output.output_dir,
                                        param.output.output_prefix,
                                        time_handler.current_time_iteration,
                                        mpi_communicator,
                                        2);
  }
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
void HeatSolver<dim>::compute_errors()
{
  TimerOutput::Scope t(computing_timer, "Compute errors");

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
   * Compute errors on temperature
   */
  const ComponentSelectFunction<dim> temperature_comp_select(0, 1);
  compute_and_add_errors(
    *mapping, *exact_solution, cellwise_errors, temperature_comp_select, "T");
}

template <int dim>
void HeatSolver<dim>::compute_recovery()
{
  TimerOutput::Scope t(computing_timer, "Compute recovery");

  ErrorEstimation::Patches patches(
    triangulation, *mapping, dof_handler,  param.finite_elements.temperature_degree + 1, temperature_mask);
}

template <int dim>
void HeatSolver<dim>::postprocess_solution()
{
  output_results();

  if (param.mms_param.enable)
    compute_errors();

  compute_recovery();
}

// Explicit instantiation
template class HeatSolver<2>;
template class HeatSolver<3>;