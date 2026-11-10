
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
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_ns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <precondition_fieldsplit.h>
#include <scratch_data.h>
#include <utilities.h>

#if defined(DEAL_II_WITH_PETSC)
#  include <petscmat.h>

#  include <deal.II/lac/exceptions.h>

// Shorthand notation for PETSc error codes.
#  define AssertPETSc(code)                        \
    do                                             \
    {                                              \
      PetscErrorCode ierr = (code);                \
      AssertThrow(ierr == 0, ExcPETScError(ierr)); \
    }                                              \
    while (false)
#endif

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
  this->should_assemble_preconditioner = true;
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
void NSSolver<dim>::create_zero_constraints()
{
  NavierStokesSolver<dim>::create_zero_constraints();

  if (this->param.bc_data.pressure_null_space)
    create_pressure_null_vector();
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

  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(*this->dof_handler,
                                  coupling_table,
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

  const auto &linear_solver_param =
    this->param.linear_solver.at(this->solver_type);

#if defined(DEAL_II_WITH_PETSC)
  // For P1-P1 or Q1-Q1 elements, all dofs are located at the vertices, and the
  // (u, p) dofs of each vertex are numbered consecutively, in component order.
  // The matrix thus has a point block structure of size n_components. Expose
  // this structure to PETSc by setting the block size of the matrix, which
  // BoomerAMG then uses as its number of functions. Combined with the nodal
  // coarsening option set in create_preconditioner(), BoomerAMG coarsens the
  // vertices (nodes) rather than the individual dofs, using the Frobenius norm
  // of each nodal block of the matrix as connection strength, so that all
  // components of a node are kept or discarded together on the coarse levels.
  if (linear_solver_param.use_amg() &&
      this->param.finite_elements.velocity_degree == 1 &&
      this->param.finite_elements.pressure_degree == 1)
  {
    const PetscErrorCode ierr =
      MatSetBlockSize(this->system_matrix, n_components);
    AssertThrow(ierr == 0,
                ExcMessage("PETSc could not set the block size of the system "
                           "matrix to " +
                           std::to_string(this->ordering->n_components) +
                           " (error code " + std::to_string(ierr) + ")"));
  }
#endif

  // Data of the block Schur preconditioners: the dofs of each field, and the
  // sparsity pattern of the pressure mass matrix of the Cahouet-Chabard
  // preconditioner
  block_schur_data.is_assembled = false;
  if (linear_solver_param.use_fieldsplit())
  {
    // Locally owned dofs of each field
    block_schur_data.velocity_dofs =
      DoFTools::extract_dofs(*this->dof_handler, this->velocity_mask);
    block_schur_data.pressure_dofs =
      DoFTools::extract_dofs(*this->dof_handler, this->pressure_mask);
  }

  if (linear_solver_param.use_cc())
  {
    // Only the pressure couples to itself
    Table<2, DoFTools::Coupling> pressure_coupling(n_components, n_components);
    for (unsigned int c = 0; c < n_components; ++c)
      for (unsigned int d = 0; d < n_components; ++d)
        pressure_coupling[c][d] =
          (this->ordering->is_pressure(c) && this->ordering->is_pressure(d)) ?
            DoFTools::always :
            DoFTools::none;

    DynamicSparsityPattern mass_dsp(this->locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(*this->dof_handler,
                                    pressure_coupling,
                                    mass_dsp,
                                    this->nonzero_constraints,
                                    /* keep_constrained_dofs = */ false);
    // Constrained velocity dofs are not coupled in the pattern above, but
    // distribute_local_to_global() still writes their diagonal entry
    for (const auto i : this->locally_owned_dofs)
      mass_dsp.add(i, i);
    SparsityTools::distribute_sparsity_pattern(mass_dsp,
                                               this->locally_owned_dofs,
                                               this->mpi_communicator,
                                               this->locally_relevant_dofs);
    block_schur_data.pressure_mass_matrix.reinit(this->locally_owned_dofs,
                                                 this->locally_owned_dofs,
                                                 mass_dsp,
                                                 this->mpi_communicator);
  }
}

template <int dim>
void NSSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  CopyData copy_data(*fe);

  this->system_matrix = 0;
  this->do_matrix_assembly(*this,
                           this->dof_handler->begin_active(),
                           this->dof_handler->end(),
                           *scratch_data,
                           copy_data,
                           this->param.nonlinear_solver.analytic_jacobian);
  this->system_matrix.compress(VectorOperation::add);
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
  // Update heuristic indicators
  this->should_assemble_preconditioner       = false;
  this->preconditioner_rebuilt_at_last_solve = true;
  rebuild_data.bdf_c0_at_setup = this->time_handler.bdf_coefficients[0];
  rebuild_data.iterations_per_decade_after_setup = 0.;

  const auto &linear_solver_param =
    this->param.linear_solver.at(this->solver_type);

  if (linear_solver_param.use_amg())
  {
    // Setup AMG preconditioner with ILU smoother
    TimerOutput::Scope t(this->computing_timer, "Setup AMG preconditioner");

#if defined(DEAL_II_WITH_PETSC)
#  if !defined(DEAL_II_PETSC_WITH_HYPRE)
    AssertThrow(false,
                ExcMessage("PETSc must be configured with hypre to use the "
                           "BoomerAMG preconditioner, and PETSc 3.22 onward is "
                           "required to use BoomerAMG with ILU smoother"));
#  endif
    // Set options not accessible through AdditionnalData
#  if DEAL_II_PETSC_VERSION_GTE(3, 22, 0)
    // ILU smoother is available from PETSc 3.22 onward
    PETScWrappers::set_option_value("-pc_hypre_boomeramg_nodal_coarsen", "1");
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
#if defined(FEZ_WITH_PETSC)
    // With a pressure null space, the matrix is singular: avoid the direct
    // solve on the coarsest level
    if (this->param.bc_data.pressure_null_space)
      data.relaxation_type_coarse = LA::MPI::PreconditionAMG::AdditionalData::
        RelaxationType::symmetricSORJacobi;
#endif
    preconditioner =
      std::make_unique<LA::MPI::PreconditionAMG>(this->system_matrix, data);
  }
  else if (linear_solver_param.use_fieldsplit())
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Setup field split Schur preconditioner");

    const IndexSet &velocity_dofs = block_schur_data.velocity_dofs;
    const IndexSet &pressure_dofs = block_schur_data.pressure_dofs;

    using Precondition = PETScWrappers::PreconditionFieldSplitSchur;

    Precondition::AdditionalData data;
    data.velocity_preconditioner =
      linear_solver_param.velocity_block_preconditioner;
    data.ilu_fill_level = linear_solver_param.ilu_fill_level;
    if (this->param.bc_data.pressure_null_space)
      data.pressure_null_vector = &pressure_null_vector;

    // Diagonal of the time derivative term of the momentum block,
    // c0 diag(Mu), with the current leading BDF coefficient (Cahouet-Chabard)
    LA::ParVectorType time_derivative_diagonal;

    using SchurApproximation = Precondition::SchurApproximation;
    if (linear_solver_param.use_cc())
    {
      // Cahouet-Chabard approximation of S = −A10 * A00^-1 * A01, given by
      // Ŝ^-1 = (−Mp/ν)^-1 + (−A10 * (c0 diag(Mu))^-1 * A01)^-1,
      // with Mp and Mu the pressure and velocity mass matrices, respectively.
      // These matrices are only assembled after a change of their sparsity
      // pattern.
      const bool steady = this->time_handler.is_steady();
      if (!block_schur_data.is_assembled)
      {
        assemble_pressure_mass_matrix();
        if (!steady)
          assemble_velocity_mass_diagonal();
        block_schur_data.is_assembled = true;
      }

      data.scaled_pressure_mass_matrix = &block_schur_data.pressure_mass_matrix;
      if (steady)
        // Without time derivative, only the mass term remains
        data.schur_approximation = SchurApproximation::mass;
      else
      {
        time_derivative_diagonal = block_schur_data.velocity_mass_diagonal;
        time_derivative_diagonal *= this->time_handler.bdf_coefficients[0];
        data.schur_approximation      = SchurApproximation::cahouet_chabard;
        data.time_derivative_diagonal = &time_derivative_diagonal;
      }
    }
    else
    {
      // Least-squares commutator, scaled by diag(Mu), which is only assembled
      // after a change of the sparsity pattern
      if (!block_schur_data.is_assembled)
      {
        assemble_velocity_mass_diagonal();
        block_schur_data.is_assembled = true;
      }
      data.schur_approximation    = SchurApproximation::lsc;
      data.velocity_mass_diagonal = &block_schur_data.velocity_mass_diagonal;
    }

    preconditioner = std::make_unique<Precondition>(this->system_matrix,
                                                    velocity_dofs,
                                                    pressure_dofs,
                                                    data);
  }
  else
    // Let the common function create other standard types of preconditioner
    LinearSolvers::create_preconditioner(linear_solver_param,
                                         this->system_matrix,
                                         preconditioner);
}

template <int dim>
void NSSolver<dim>::assemble_pressure_mass_matrix()
{
  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  // Coefficient -1/nu on the pressure component and zero on the velocity
  // components. Zero velocity entries are computed, but not written to the
  // matrix.
  const ComponentSelectFunction<dim> coefficient(this->ordering->p_lower,
                                                 -1. / nu,
                                                 this->ordering->n_components);

  block_schur_data.pressure_mass_matrix = 0;
  MatrixCreator::create_mass_matrix(*mapping,
                                    *this->dof_handler,
                                    *this->quadrature,
                                    block_schur_data.pressure_mass_matrix,
                                    &coefficient,
                                    this->zero_constraints);
}

template <int dim>
void NSSolver<dim>::assemble_velocity_mass_diagonal()
{
  auto &diagonal = block_schur_data.velocity_mass_diagonal;
  diagonal.reinit(this->locally_owned_dofs, this->mpi_communicator);

  FEValues<dim> fe_values(*mapping,
                          *fe,
                          *this->quadrature,
                          update_values | update_JxW_values);

  const unsigned int                   dofs_per_cell = fe->n_dofs_per_cell();
  Vector<double>                       local_diagonal(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      local_diagonal = 0.;
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          if (this->ordering->is_velocity(
                fe->system_to_component_index(i).first))
          {
            // Only one component of each vector shape function is nonzero
            const double phi_i = fe_values.shape_value(i, q);
            local_diagonal(i) += phi_i * phi_i * fe_values.JxW(q);
          }

      cell->get_dof_indices(dof_indices);
      diagonal.add(dof_indices, local_diagonal);
    }
  diagonal.compress(VectorOperation::add);
}

template <int dim>
bool NSSolver<dim>::should_rebuild_preconditioner(
  const bool first_solve_of_time_step) const
{
  const bool verbose =
    this->param.linear_solver.at(this->solver_type).verbosity ==
    Parameters::Verbosity::verbose;

  // Preconditioner not created yet
  if (!preconditioner)
  {
    if (verbose)
      this->pcout << "Building initial preconditioner" << std::endl;
    return true;
  }

  // Preconditioner invalidated by a change of mesh, dof umbering or time
  // interval
  // FIXME: could connect to triangulation's signals instead
  if (this->should_assemble_preconditioner)
  {
    if (verbose)
      this->pcout << "Rebuilding preconditioner after mesh update" << std::endl;
    return true;
  }

  // Steady problems: always rebuild, as Jacobian matrix may change
  // significantly between Newton iterations
  if (this->time_handler.is_steady())
    return true;

  // The mass matrix contribution to the Jacobian scales with the leading BDF
  // coefficient, which changes with the time step or the BDF order
  const double bdf_c0 = this->time_handler.bdf_coefficients[0];
  if (std::abs(bdf_c0 - rebuild_data.bdf_c0_at_setup) >
      1e-10 * std::abs(bdf_c0))
  {
    if (verbose)
      this->pcout << "Rebuilding preconditioner: BDF coefficient changed"
                  << std::endl;
    return true;
  }

  // The remaining criteria are based on the last solve performed with the
  // current preconditioner, if any
  const auto &last_solve_info = rebuild_data.last_solve_info;
  if (last_solve_info.n_iterations == 0 && last_solve_info.converged)
    return false;

  if (!last_solve_info.converged)
  {
    if (verbose)
      this->pcout << "Rebuilding preconditioner: GMRES did not converge"
                  << std::endl;
    return true;
  }

  // Compare the first solves of the time steps only, see the documentation
  const double ref = rebuild_data.iterations_per_decade_after_setup;
  if (!first_solve_of_time_step || ref == 0.)
    return false;

  // Rebuild if the GMRES iterations per decade of residual reduction of the
  // first solve of the previous time step exceed both this factor times, and
  // this increment above, those of the first solve of a time step with the
  // current preconditioner. The increment (about 3 iterations over 6 decades)
  // only filters noise: with fast convergence (about 2 iterations per decade),
  // a larger increment would let the preconditioner degrade by half before
  // it is rebuilt
  constexpr double max_ratio     = 1.3;
  constexpr double max_increment = 0.5;
  const double current = rebuild_data.iterations_per_decade_at_last_first_solve;
  if (current > max_ratio * ref && current > ref + max_increment)
  {
    if (verbose)
      this->pcout << "Rebuilding preconditioner: " << current
                  << " GMRES iterations per decade of residual reduction at "
                     "the first Newton iteration, compared to "
                  << ref << " after its last setup" << std::endl;
    return true;
  }

  return false;
}

template <int dim>
void NSSolver<dim>::solve_linear_system_iterative()
{
  const auto &linear_solver_param =
    this->param.linear_solver.at(this->solver_type);

  AssertThrow(linear_solver_param.method ==
                Parameters::LinearSolver::Method::gmres,
              ExcMessage("Only GMRES is available as iterative solver for "
                         "the incompressible_ns solver"));

  if (this->param.bc_data.pressure_null_space)
    attach_pressure_null_space();

  // First linear solve of this time step, i.e. first Newton iteration
  const bool first_solve_of_time_step =
    this->time_handler.current_time_iteration !=
    rebuild_data.time_iteration_at_last_solve;
  rebuild_data.time_iteration_at_last_solve =
    this->time_handler.current_time_iteration;

  auto &last_solve_info = rebuild_data.last_solve_info;

  const auto solve = [&]() {
    last_solve_info = LinearSolvers::solve_gmres(this,
                                                 linear_solver_param,
                                                 this->system_matrix,
                                                 this->locally_owned_dofs,
                                                 this->zero_constraints,
                                                 preconditioner);
    // Store the convergence speed of the first solve of the time step, and
    // of the first one with this preconditioner (if the residual was reduced,
    // otherwise wait for the next time step)
    if (first_solve_of_time_step)
    {
      rebuild_data.iterations_per_decade_at_last_first_solve =
        last_solve_info.iterations_per_decade;
      if (rebuild_data.iterations_per_decade_after_setup == 0.)
        rebuild_data.iterations_per_decade_after_setup =
          last_solve_info.iterations_per_decade;
    }
  };

  // Set to true by create_preconditioner() if called during this solve
  this->preconditioner_rebuilt_at_last_solve = false;

  const bool rebuild = should_rebuild_preconditioner(first_solve_of_time_step);
  if (rebuild)
    create_preconditioner();
  solve();

  // GMRES solver may not have converged if the preconditioner was reused.
  // If so, rebuild the preconditioner and solve again once
  if (!rebuild && !last_solve_info.converged)
  {
    if (linear_solver_param.verbosity == Parameters::Verbosity::verbose)
      this->pcout << "Rebuilding preconditioner and solving again: GMRES did "
                     "not converge"
                  << std::endl;
    create_preconditioner();
    solve();
  }
}

template <int dim>
void NSSolver<dim>::create_pressure_null_vector()
{
  pressure_null_vector.reinit(this->locally_owned_dofs, this->mpi_communicator);
  for (const auto i : this->locally_owned_dofs)
    if (this->ordering->is_pressure(
          this->dofs_to_component[this->locally_relevant_dofs.index_within_set(
            i)]) &&
        !this->zero_constraints.is_constrained(i))
      pressure_null_vector[i] = 1.;
  pressure_null_vector.compress(VectorOperation::insert);
  pressure_null_vector /= pressure_null_vector.l2_norm();
}

template <int dim>
void NSSolver<dim>::attach_pressure_null_space()
{
#if defined(FEZ_WITH_PETSC)
  Vec          vector = static_cast<const Vec &>(pressure_null_vector);
  MatNullSpace null_space;
  AssertPETSc(MatNullSpaceCreate(
    this->mpi_communicator, PETSC_FALSE, 1, &vector, &null_space));
  AssertPETSc(
    MatSetNullSpace(static_cast<const Mat &>(this->system_matrix), null_space));
  AssertPETSc(
    MatSetTransposeNullSpace(static_cast<const Mat &>(this->system_matrix),
                             null_space));
  // MatNullSpaceCreate() references the vector (no copy), and the matrix
  // references the null space, so the local handle can be destroyed right away
  AssertPETSc(MatNullSpaceDestroy(&null_space));
#else
  AssertThrow(false,
              ExcMessage("Pressure null space handling requires PETSc."));
#endif
}

template <int dim>
void NSSolver<dim>::finalize_rhs()
{
  if (!this->param.bc_data.pressure_null_space)
    return;

  // Same projection as the one of the right-hand side by the linear solver
  this->system_rhs.add(-(pressure_null_vector * this->system_rhs),
                       pressure_null_vector);
}

// Explicit instantiation
template class NSSolver<2>;
template class NSSolver<3>;
