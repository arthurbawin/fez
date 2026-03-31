#include <compare_matrix.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_elasticity_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <solver_info.h>
#include <utilities.h>

#include <cmath>
#include <deque>
#include <iomanip>

template <int dim>
LinearElasticitySolver<dim>::LinearElasticitySolver(
  const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.output,
                                     param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param,
                                     SolverInfo::SolverType::linear_elasticity)
  , param(param)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  create_quadrature_rules(param.finite_elements,
                          quadrature,
                          face_quadrature,
                          error_quadrature,
                          error_face_quadrature);

  if (param.finite_elements.use_quads)
  {
    mapping = std::make_shared<MappingQ<dim>>(1);
    fe      = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }
  else
  {
    mapping = std::make_shared<MappingFE<dim>>(FE_SimplexP<dim>(1));
    fe      = std::make_shared<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }

  position_extractor = FEValuesExtractors::Vector(0);
  position_mask      = fe->component_mask(position_extractor);

  if (param.mms_param.enable)
  {
    for (auto norm : param.mms_param.norms_to_compute)
      error_handlers[norm]->create_entry("x");

    // Assign the manufactured solution
    exact_solution = param.mms.exact_mesh_position;

    // Create source term function for the given MMS and override source terms
    source_terms = std::make_shared<LinearElasticitySolver<dim>::MMSSourceTerm>(
      param.physical_properties, param.mms);
  }
  else
  {
    source_terms   = param.source_terms.linear_elasticity_source;
    exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(dim);
  }

  // Create direct solver
  direct_solver_reuse =
    std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);
}

template <int dim>
void LinearElasticitySolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const auto    &pseudosolid = physical_properties.pseudosolids[0];
  Tensor<1, dim> f;

  if (pseudosolid.constitutive_model ==
      Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
    f = mms.exact_mesh_position
          ->divergence_neo_hookean_stress_variable_coefficients(
            p, pseudosolid.lame_mu_fun, pseudosolid.lame_lambda_fun);
  else
    f = mms.exact_mesh_position
          ->divergence_linear_elastic_stress_variable_coefficients(
            p, pseudosolid.lame_mu_fun, pseudosolid.lame_lambda_fun);

  for (unsigned int d = 0; d < dim; ++d)
    values[d] = f[d];
}

template <int dim>
void LinearElasticitySolver<dim>::reset()
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
}

template <int dim>
void LinearElasticitySolver<dim>::run()
{
  reset();
  read_mesh(triangulation, param);
  setup_dofs();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
  output_results();

  update_boundary_conditions();

  if (param.linear_elasticity.enable_source_term_on_current_mesh)
  {
    /**
     * Continuation method to handle possibly steep source terms evaluated
     * on the current (deformed) mesh.
     */
    const double c_min =
      param.linear_elasticity.min_current_mesh_source_term_multiplier;
    const double c_max =
      param.linear_elasticity.max_current_mesh_source_term_multiplier;
    const unsigned int n_steps = param.linear_elasticity.n_continuation_steps;
    // Progession
    const double step_size =
      n_steps > 1 ? (c_max - c_min) / static_cast<double>(n_steps - 1) : 0.0;
    const double min_gap =
      1e-8 * std::max(1.0, std::max(std::abs(c_min), std::abs(c_max)));

    source_term_fixed_mesh_multiplier = 0.;

    std::deque<double> continuation_targets;
    for (unsigned int n = 0; n < n_steps; ++n)
      continuation_targets.push_back(c_min + n * step_size);

    LA::ParVectorType accepted_solution(locally_owned_dofs,
                                        locally_relevant_dofs,
                                        mpi_communicator);
    LA::ParVectorType accepted_local_evaluation_point(locally_owned_dofs,
                                                      mpi_communicator);
    accepted_solution               = present_solution;
    accepted_local_evaluation_point = local_evaluation_point;
    double accepted_multiplier = 0.0;

    auto restore_accepted_state = [&]() {
      present_solution        = accepted_solution;
      evaluation_point        = accepted_solution;
      local_evaluation_point  = accepted_local_evaluation_point;
      newton_update           = 0.;
      system_rhs              = 0.;
    };

    auto reset_direct_solver = [&]() {
      direct_solver_reuse =
        std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);
    };

    while (!continuation_targets.empty())
    {
      const double trial_multiplier = continuation_targets.front();
      continuation_targets.pop_front();
      const double progress = 100.0 * trial_multiplier / c_max;

      restore_accepted_state();
      source_term_moving_mesh_multiplier = trial_multiplier;

      pcout << std::endl;
      pcout << "[Continuation] alpha = " << std::scientific
            << source_term_moving_mesh_multiplier << " / " << c_max << " ("
            << std::fixed << std::setprecision(2) << progress << "%)"
            << std::scientific << std::setprecision(8) << std::endl;
      pcout << std::endl;

      bool continuation_step_accepted = false;
      try
      {
        if (param.debug.compare_analytical_jacobian_with_fd)
          compare_analytical_matrix_with_fd();

        solve_nonlinear_problem(time_handler);
        const double solution_norm   = present_solution.l2_norm();
        const double evaluation_norm = evaluation_point.l2_norm();
        continuation_step_accepted =
          std::isfinite(solution_norm) && std::isfinite(evaluation_norm);
      }
      catch (const std::exception &exc)
      {
        pcout << "Continuation step rejected at multiplier "
              << trial_multiplier << " because Newton failed with: "
              << exc.what() << std::endl;
      }
      catch (...)
      {
        pcout << "Continuation step rejected at multiplier "
              << trial_multiplier
              << " because Newton failed with an unknown exception."
              << std::endl;
      }

      if (continuation_step_accepted)
      {
        accepted_solution               = present_solution;
        accepted_local_evaluation_point = local_evaluation_point;
        accepted_multiplier             = trial_multiplier;
        continue;
      }

      const double gap = trial_multiplier - accepted_multiplier;
      if (gap <= min_gap)
      {
        restore_accepted_state();
        source_term_moving_mesh_multiplier = accepted_multiplier;

        std::ostringstream message;
        message << "Continuation failed in LinearElasticitySolver: could not "
                   "reach source term multiplier "
                << trial_multiplier << " from last accepted multiplier "
                << accepted_multiplier << " because the continuation gap "
                << gap << " is already below the minimum allowed gap "
                << min_gap << ".";
        throw std::runtime_error(message.str());
      }

      const double reduced_trial = accepted_multiplier + 0.5 * gap;
      pcout << std::endl;
      pcout << "[Continuation] rejected" << std::endl;
      pcout << "  accepted alpha = " << std::scientific << accepted_multiplier
            << std::endl;
      pcout << "  failed alpha   = " << trial_multiplier << std::endl;
      pcout << "  retry alpha    = " << reduced_trial << std::endl;
      pcout << std::endl;

      restore_accepted_state();
      reset_direct_solver();

      continuation_targets.push_front(trial_multiplier);
      continuation_targets.push_front(reduced_trial);
    }

    source_term_moving_mesh_multiplier = accepted_multiplier;
    restore_accepted_state();
  }
  else
  {
    // Source term is evaluated on reference mesh and problem is linear
    // This is the case when performing a convergence study with a
    // manufactured solution, for example.
    source_term_moving_mesh_multiplier = 0.;
    source_term_fixed_mesh_multiplier  = 1.;

    if (param.debug.compare_analytical_jacobian_with_fd)
      compare_analytical_matrix_with_fd();
    solve_nonlinear_problem(time_handler);
  }

  postprocess_solution();
}

template <int dim>
void LinearElasticitySolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(*fe);

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
}

template <int dim>
void LinearElasticitySolver<dim>::create_base_constraints(
  const bool                 homogeneous,
  AffineConstraints<double> &constraints)
{
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  BoundaryConditions::apply_mesh_position_boundary_conditions(
    homogeneous,
    0,
    dim,
    dof_handler,
    *mapping,
    param.pseudosolid_bc,
    *exact_solution,
    *param.mms.exact_mesh_position,
    constraints);

  constraints.close();
}

template <int dim>
void LinearElasticitySolver<dim>::create_zero_constraints()
{
  create_base_constraints(true, zero_constraints);
}

template <int dim>
void LinearElasticitySolver<dim>::create_nonzero_constraints()
{
  create_base_constraints(false, nonzero_constraints);
}

template <int dim>
void LinearElasticitySolver<dim>::create_sparsity_pattern()
{
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
}

template <int dim>
void LinearElasticitySolver<dim>::set_initial_conditions()
{
  FixedMeshPosition<dim> fixed_mesh(0, dim);
  VectorTools::interpolate(
    *mapping, dof_handler, fixed_mesh, newton_update, position_mask);
  evaluation_point = newton_update;

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  evaluation_point = newton_update;
}

template <int dim>
void LinearElasticitySolver<dim>::set_exact_solution()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           position_mask);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void LinearElasticitySolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = present_solution;
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrix");

  system_matrix = 0;
  ScratchData scratch_data(*fe, *mapping, *quadrature, *face_quadrature, param);
  CopyData    copyData(fe->n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  auto assembly_ptr =
    this->param.nonlinear_solver.analytic_jacobian ?
      &LinearElasticitySolver::assemble_local_matrix :
      &LinearElasticitySolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  assembly_ptr,
                  &LinearElasticitySolver::copy_local_to_global_matrix,
                  scratch_data,
                  copyData);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &LinearElasticitySolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    this->evaluation_point,
    this->local_evaluation_point);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double alpha = source_term_moving_mesh_multiplier;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW         = scratch_data.JxW[q];
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const auto &ps = param.physical_properties.pseudosolids[0];

    const bool use_nh =
      (ps.constitutive_model ==
       Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean);

    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    const Tensor<2, dim> &grad_source_current_mesh =
      scratch_data.grad_source_term_position_current_mesh[q];

    // Neo-Hookean
    const double          J       = scratch_data.position_J[q];
    const Tensor<2, dim> &F_inv   = scratch_data.position_inv_gradients[q];
    const Tensor<2, dim> &F_inv_T = scratch_data.position_inv_gradients_T[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_x_i          = phi_x[i];
      const auto &grad_phi_x_i     = grad_phi_x[i];
      const auto &sym_grad_phi_x_i = symmetrize(grad_phi_x_i);
      const auto &div_phi_x_i      = div_phi_x[i];

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const auto &phi_x_j          = phi_x[j];
        const auto &grad_phi_x_j     = grad_phi_x[j];
        const auto &sym_grad_phi_x_j = symmetrize(grad_phi_x_j);
        const auto &div_phi_x_j      = div_phi_x[j];

        if (!use_nh)
        {
          local_matrix(i, j) +=
            (lame_lambda * div_phi_x_j * div_phi_x_i +
             2. * lame_mu *
               scalar_product(sym_grad_phi_x_j, sym_grad_phi_x_i) +
             alpha * phi_x_i * (grad_source_current_mesh * phi_x_j)) *
            JxW;
        }
        else
        {
          const Tensor<2, dim> dF = grad_phi_x_j;

          const Tensor<2, dim> dP =
            lame_mu * dF +
            (lame_mu - lame_lambda * std::log(J)) *
              (F_inv_T * transpose(dF) * F_inv_T) +
            lame_lambda * trace(F_inv * dF) * F_inv_T;

          local_matrix(i, j) +=
            (scalar_product(dP, grad_phi_x_i) +
             alpha * phi_x_i * (grad_source_current_mesh * phi_x_j)) *
            JxW;
        }
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void LinearElasticitySolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              system_matrix);
}

template <int dim>
void LinearElasticitySolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchData scratch_data(*fe, *mapping, *quadrature, *face_quadrature, param);
  CopyData    copyData(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &LinearElasticitySolver::assemble_local_matrix,
    &LinearElasticitySolver::assemble_local_rhs,
    scratch_data,
    copyData,
    present_solution,
    evaluation_point,
    local_evaluation_point,
    mpi_communicator,
    /*output_dir = */ "",
    /*print_problematic_elements = */ false,
    param.debug.analytical_jacobian_absolute_tolerance,
    param.debug.analytical_jacobian_relative_tolerance);

  pcout << "Max absolute error analytical vs fd matrix is " << errors.first
        << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > param.debug.analytical_jacobian_absolute_tolerance)
    pcout << "Max relative error analytical vs fd matrix is " << errors.second
          << std::endl;
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;
  ScratchData scratch_data(*fe, *mapping, *quadrature, *face_quadrature, param);
  CopyData    copyData(fe->n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &LinearElasticitySolver::assemble_local_rhs,
                  &LinearElasticitySolver::copy_local_to_global_rhs,
                  scratch_data,
                  copyData);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double alpha = source_term_moving_mesh_multiplier;
  const double gamma = source_term_fixed_mesh_multiplier;

  // alpha and gamma cannot both be nonzero
  Assert(!(std::abs(alpha) > 1e-14 && std::abs(gamma) > 1e-14),
         ExcInternalError());

  const SymmetricTensor<2, dim> identity_tensor = unit_symmetric_tensor<dim>();

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW         = scratch_data.JxW[q];
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const auto &ps = param.physical_properties.pseudosolids[0];

    const bool use_nh =
      (ps.constitutive_model ==
       Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean);

    const auto &position_sym_gradient = scratch_data.position_sym_gradients[q];
    const auto &source_term_position_moving_mesh =
      scratch_data.source_term_position_current_mesh[q];
    const auto &source_term_position_fixed_mesh =
      scratch_data.source_term_position[q];
    // The source term to use : using coefficients which cannot be both nonzero
    // avois using a condition
    const auto source_term = alpha * source_term_position_moving_mesh +
                             gamma * source_term_position_fixed_mesh;

    const auto strain       = position_sym_gradient - identity_tensor;
    const auto trace_strain = trace(strain);

    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    const Tensor<2, dim> &F       = scratch_data.position_gradients[q];
    const double          J       = scratch_data.position_J[q];
    const Tensor<2, dim> &F_inv_T = scratch_data.position_inv_gradients_T[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      if (!use_nh)
      {
        local_rhs(i) -=
          (lame_lambda * trace_strain * div_phi_x[i] +
           2. * lame_mu * scalar_product(strain, grad_phi_x[i]) +
           phi_x[i] * source_term) *
          JxW;
      }
      else
      {
        const Tensor<2, dim> P =
          lame_mu * (F - F_inv_T) + lame_lambda * std::log(J) * F_inv_T;

        local_rhs(i) -=
          (scalar_product(P, grad_phi_x[i]) + phi_x[i] * source_term) * JxW;
      }
    }
  }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void LinearElasticitySolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              system_rhs);
}

template <int dim>
void LinearElasticitySolver<dim>::solve_linear_system()
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
    solve_linear_system_unpreconditioned_cg(this,
                                            linear_solver_param,
                                            system_matrix,
                                            locally_owned_dofs,
                                            zero_constraints);
  }
  else if (linear_solver_param.method ==
           Parameters::LinearSolver::Method::gmres)
  {
    AssertThrow(false,
                ExcMessage("GMRES solver is not implemented for "
                           "LinearElasticitySolver. Use CG unstead."));
  }
  else
  {
    AssertThrow(false, ExcMessage("No known resolution method"));
  }
}

template <int dim>
void LinearElasticitySolver<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Write outputs");

  if (param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "position");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain,
                             "subdomain",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(*mapping, 2);
    data_out.write_vtu_with_pvtu_record(param.output.output_dir,
                                        param.output.output_prefix +
                                          "linear_elasticity",
                                        0,
                                        mpi_communicator,
                                        2);
  }
}

template <int dim>
void LinearElasticitySolver<dim>::move_mesh()
{
  present_solution.update_ghost_values();

  const IndexSet locally_owned = dof_handler.locally_owned_dofs();

  std::vector<bool> vertex_moved(triangulation.n_vertices(), false);
  std::vector<bool> vertex_locally_moved(triangulation.n_vertices(), false);

  for (auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto v : cell->vertex_indices())
      {
        const auto vid = cell->vertex_index(v);
        if (vertex_moved[vid])
          continue;

        // this rank owns the vertex if all dim position dofs are locally-owned
        bool i_own_vertex = true;
        for (unsigned int d = 0; d < dim; ++d)
          i_own_vertex = i_own_vertex &&
                         locally_owned.is_element(cell->vertex_dof_index(v, d));

        // mark as visited so we don't process it again
        vertex_moved[vid] = true;

        if (!i_own_vertex)
          continue;

        // move vertex to its new position
        for (unsigned int d = 0; d < dim; ++d)
          cell->vertex(v)[d] = present_solution(cell->vertex_dof_index(v, d));

        vertex_locally_moved[vid] = true;
      }

  // sync moved vertices across MPI boundaries
  triangulation.communicate_locally_moved_vertices(vertex_locally_moved);
}


template <int dim>
void LinearElasticitySolver<dim>::compute_errors()
{
  TimerOutput::Scope t(computing_timer, "Compute errors");

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);
  const ComponentSelectFunction<dim> position_comp_select(0, dim);

  for (auto norm : param.mms_param.norms_to_compute)
  {
    error_handlers.at(norm)->add_reference_data(
      "n_elm", triangulation.n_global_active_cells());
    error_handlers.at(norm)->add_reference_data("n_dof", dof_handler.n_dofs());
    const double err =
      compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                                 *mapping,
                                                 dof_handler,
                                                 present_solution,
                                                 *exact_solution,
                                                 cellwise_errors,
                                                 *error_quadrature,
                                                 norm,
                                                 &position_comp_select);
    error_handlers.at(norm)->add_error("x", err);
  }
}

template <int dim>
void LinearElasticitySolver<dim>::postprocess_solution()
{
  // Compute error *before* moving mesh for visualization (-:
  if (param.mms_param.enable)
    compute_errors();

  move_mesh();
  output_results();
}

// Explicit instantiation
template class LinearElasticitySolver<2>;
template class LinearElasticitySolver<3>;
