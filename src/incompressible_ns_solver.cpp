
#include <assembly/incompressible_ns_assemblers.h>
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
      FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^
                    dim),                                // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree)); // Pressure
  else
    fe = std::make_unique<FESystem<dim>>(
      FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                    dim),                                       // Velocity
      FE_SimplexP<dim>(param.finite_elements.pressure_degree)); // Pressure

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
                                               *mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim>
void NSSolver<dim>::reset_solver_specific_data()
{
  preconditioner.reset();
}

template <int dim>
void NSSolver<dim>::setup_assemblers()
{
  assemblers.clear();

  using namespace Assembly::IncompressibleNavierStokes;

  if (this->param.stabilization.enable_supg)
    Assembly::IncompressibleNavierStokes::
      setup_assemblers<dim, ScratchData, CopyData, stabilization>(
        this->param, *this->ordering, this->coupling_table, assemblers);
  else
    Assembly::IncompressibleNavierStokes::
      setup_assemblers<dim, ScratchData, CopyData, ns_laplace_form>(
        this->param, *this->ordering, this->coupling_table, assemblers);
}

template <int dim>
void NSSolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints have been
  // defined
  //

  const unsigned int n_components   = this->ordering->n_components;
  auto              &coupling_table = this->coupling_table;
  coupling_table.reinit(n_components, n_components);

  // If stabilized, (u,p) couples to (u,p).
  if (this->param.stabilization.enable_supg)
    coupling_table.fill(DoFTools::always);
  else
  {
    // Nonstabilized case
    for (unsigned int c = 0; c < n_components; ++c)
      for (unsigned int d = 0; d < n_components; ++d)
      {
        coupling_table[c][d] = DoFTools::none;

        // u couples to (u,p)
        if (this->ordering->is_velocity(c))
          coupling_table[c][d] = DoFTools::always;

        // p couples to u only
        if (this->ordering->is_pressure(c))
          if (this->ordering->is_velocity(d))
            coupling_table[c][d] = DoFTools::always;
      }
  }

#if defined(FEZ_WITH_PETSC)
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(*this->dof_handler,
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
  DoFTools::make_sparsity_pattern(*this->dof_handler,
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

  CopyData copy_data(*fe);

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
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  assembly_ptr,
                  &NSSolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim>
void NSSolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell, *this, &NSSolver::assemble_local_rhs, scratch_data, copy_data);
}

template <int dim>
void NSSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix = copy_data.local_matrix();
  local_matrix       = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_matrix(scratch_data, copy_data);

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void NSSolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim>
void NSSolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    &NSSolver::assemble_local_matrix,
    &NSSolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim>
void NSSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  &NSSolver::assemble_local_rhs,
                  &NSSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void NSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs = copy_data.local_rhs();
  local_rhs       = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_rhs(scratch_data, copy_data);

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void NSSolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                                    copy_data.dof_indices(),
                                                    this->system_rhs);
}

template <int dim>
void NSSolver<dim>::create_preconditioner()
{
  if (preconditioner)
    return;

  const auto &linear_solver_param =
    this->param.linear_solver.at(this->solver_type);

  if (linear_solver_param.preconditioner ==
      Parameters::LinearSolver::PreconditionerType::amg)
  {
    // Setup AMG preconditioner with ILU smoother
    TimerOutput::Scope t(this->computing_timer, "Setup AMG preconditioner");

#if defined(DEAL_II_WITH_PETSC)
#  if !defined(DEAL_II_PETSC_WITH_HYPRE)
    AssertThrow(false,
                ExcMessage("PETSc must be configured with hypre to use the "
                           "BoomerAMG preconditioner"));
#  endif
    // Set options not accessible through AdditionnalData
#  if DEAL_II_PETSC_VERSION_GTE(3, 22, 0)
    // ILU smoother is available from PETSc 3.22 onward
    PETScWrappers::set_option_value("-pc_hypre_boomeramg_smooth_type", "ILU");
    PETScWrappers::set_option_value("-pc_hypre_boomeramg_ilu_level",
                                    std::to_string(
                                      linear_solver_param.ilu_fill_level));
#  else
    AssertThrow(
      false,
      ExcMessage(
        "PETSc 3.22 onward is required to use BoomerAMG with ILU smoother"));
#  endif
#endif

    LA::MPI::PreconditionAMG::AdditionalData data;
    preconditioner =
      std::make_unique<LA::MPI::PreconditionAMG>(this->system_matrix, data);
  }
  else
    // Let the common function create other standard types of preconditioner
    LinearSolvers::create_preconditioner(linear_solver_param,
                                         this->system_matrix,
                                         preconditioner);
}

template <int dim>
void NSSolver<dim>::solve_linear_system_iterative()
{
  const auto &linear_solver_param =
    this->param.linear_solver.at(this->solver_type);

  // Create preconditioner once?
  this->create_preconditioner();

  switch (linear_solver_param.method)
  {
    case Parameters::LinearSolver::Method::gmres:
      LinearSolvers::solve_gmres(this,
                                 linear_solver_param,
                                 this->system_matrix,
                                 this->locally_owned_dofs,
                                 this->zero_constraints,
                                 preconditioner);
      break;
    default:
      AssertThrow(false,
                  ExcMessage("Only GMRES is available as iterative solver for "
                             "the incompressible_ns solver"));
  }
}

// Explicit instantiation
template class NSSolver<2>;
template class NSSolver<3>;
