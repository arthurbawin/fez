
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>
#include <scratch_data.h>
#include <linear_direct_solver.h>

template <int dim>
MonolithicFSISolver<dim>::MonolithicFSISolver(const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver, param.timer, param.mesh, param.time_integration, param.mms_param)
  , param(param)
  , quadrature(QGaussSimplex<dim>(4))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , triangulation(this->mpi_communicator)
  , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
  , fe(FE_SimplexP<dim>(param.finite_elements.velocity_degree), // Velocity
       dim,
       FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
       1,
       FE_SimplexP<dim>(param.finite_elements.mesh_position_degree), // Position
       dim,
       FE_SimplexP<dim>(param.finite_elements
                          .no_slip_lagrange_mult_degree), // Lagrange multiplier
       dim)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  // Set the boundary id on which a weak no slip boundary condition is applied.
  // It is allowed *not* to prescribe a weak no slip on any boundary, to verify
  // that the solver produces the expected flow in the decoupled case.
  unsigned int n_weak_bc = 0;
  for (const auto &bc : param.fluid_bc)
    if (bc.type == BoundaryConditions::Type::weak_no_slip)
    {
      weak_no_slip_boundary_id = bc.id;
      n_weak_bc++;
    }
  AssertThrow(n_weak_bc <= 1,
              ExcMessage(
                "A weakly enforced no-slip boundary condition is enforced on "
                "more than 1 boundary, which is currently not supported."));

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
void MonolithicFSISolver<dim>::run()
{
  read_mesh(triangulation, this->param);
  setup_dofs();
  create_lagrange_multiplier_constraints();
  create_position_lagrange_mult_coupling_data();
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

    // Check position - lambda coupling if coupled
    if (param.fsi.enable_coupling)
      compare_forces_and_position_on_obstacle();

    // Always check that weak no-slip is satisfied
    check_velocity_boundary();

    const bool export_force_table =
      (time_handler.current_time_iteration % 5) == 0;
    compute_forces(export_force_table);
    const bool export_position_table =
      (time_handler.current_time_iteration % 5) == 0;
    write_cylinder_position(export_position_table);

    output_results();

    // Rotate solutions
    for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
      previous_solutions[j] = previous_solutions[j - 1];
    previous_solutions[0] = present_solution;
  }
}

template <int dim>
void MonolithicFSISolver<dim>::setup_dofs()
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
    // previous_sol.clear();
    previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  }

  // Initialize mesh position directly from the triangulation.
  // The parallel vector storing the mesh position is local_evaluation_point,
  // because this is the one to modify when computing finite differences.
  const FEValuesExtractors::Vector position(x_lower);
  VectorTools::get_position_vector(*fixed_mapping,
                                   dof_handler,
                                   this->local_evaluation_point,
                                   fe.component_mask(position));
  this->local_evaluation_point.compress(VectorOperation::insert);
  this->evaluation_point = this->local_evaluation_point;

  // Also store them in initial_positions, for postprocessing:
  DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                       dof_handler,
                                       this->initial_positions,
                                       fe.component_mask(position));

  // Create the solution-dependent mapping
  mapping = std::make_unique<MappingFEField<dim, dim, LA::ParVectorType>>(
    dof_handler, this->evaluation_point, fe.component_mask(position));
}

template <int dim>
void MonolithicFSISolver<dim>::create_lagrange_multiplier_constraints()
{
  lambda_constraints.clear();
  lambda_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  const FEValuesExtractors::Vector lambda(l_lower);

  // The lambda dofs which are not set to zero
  std::set<types::global_dof_index> unconstrained_lambda_dofs;

  // Flag the uncsontrained lambda dofs, lying on the boundary associated with a
  // weakly enforced no-slip BC. If there is no such boundary, constrain all
  // lambdas. This allows to keep the problem structure as is, to test with only
  // strong BC.
  if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
  {
    const unsigned int                   n_dofs_per_face = fe.n_dofs_per_face();
    std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);
    for (const auto &cell : dof_handler.active_cell_iterators())
      // if (cell->is_locally_owned())
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == weak_no_slip_boundary_id)
        {
          cell->face(f)->get_dof_indices(face_dofs);
          for (unsigned int idof = 0; idof < n_dofs_per_face; ++idof)
          {
            const unsigned int component =
              fe.face_system_to_component_index(idof).first;

            if (fe.has_support_on_face(idof, f) && is_lambda(component))
            {
              // Lambda DoF on the prescribed boundary: do not constrain
              unconstrained_lambda_dofs.insert(face_dofs[idof]);
            }
          }
        }
  }

  // Add zero constraints to all lambda DOFs *not* in the boundary set
  IndexSet lambda_dofs =
    DoFTools::extract_dofs(dof_handler, fe.component_mask(lambda));
  unsigned int n_constrained_local = 0;
  for (const auto dof : lambda_dofs)
  {
    // Only constrain owned DOFs
    if (locally_owned_dofs.is_element(dof))
      if (unconstrained_lambda_dofs.count(dof) == 0)
      {
        lambda_constraints.constrain_dof_to_zero(dof); // More readable (-:
        n_constrained_local++;
      }
  }
  lambda_constraints.close();

  const unsigned int n_unconstrained =
    Utilities::MPI::sum(unconstrained_lambda_dofs.size(),
                        this->mpi_communicator);
  const unsigned int n_constrained =
    Utilities::MPI::sum(n_constrained_local, this->mpi_communicator);

  this->pcout << n_unconstrained << " lambda DOFs are unconstrained"
              << std::endl;
  this->pcout << n_constrained << " lambda DOFs are constrained" << std::endl;
}

/**
 * On the cylinder, we have
 *
 * x = X - int_Gamma lambda dx,
 *
 * yielding the affine constraints
 *
 * x_i = X_i + sum_j c_ij * lambda_j, with c_ij = - int_Gamma phi_global_j dx.
 *
 * Each position DoF is linked to all lambda DoF on the cylinder, which may
 * not be owned of even ghosts of the current process.
 *
 * This function does the following:
 *
 * - It computes the coefficients c_ij of the coupling x_i = X_i + c_ij *
 * lambda_j, which are the integral of the global shape functions associated to
 * lambda_j.
 *
 * - It creates the DOF pairings (x_i, vector of lambda_j), which specify to
 * which lambda DOFs a position DOF on the cylinder is constrained (all of them
 * actually).
 *
 *   FIXME: THERE IS ONLY ONE VECTOR ACTUALLY
 */
template <int dim>
void MonolithicFSISolver<dim>::create_position_lagrange_mult_coupling_data()
{
  const FEValuesExtractors::Vector position(x_lower);
  const FEValuesExtractors::Vector lambda(l_lower);

  //
  // Get and synchronize the lambda DoFs on the cylinder
  //
  std::set<types::boundary_id> boundary_ids;
  boundary_ids.insert(weak_no_slip_boundary_id);

  IndexSet local_lambda_dofs =
    DoFTools::extract_boundary_dofs(dof_handler,
                                    fe.component_mask(lambda),
                                    boundary_ids);
  IndexSet local_position_dofs =
    DoFTools::extract_boundary_dofs(dof_handler,
                                    fe.component_mask(position),
                                    boundary_ids);

  const unsigned int n_local_lambda_dofs = local_lambda_dofs.n_elements();

  local_lambda_dofs   = local_lambda_dofs & locally_owned_dofs;
  local_position_dofs = local_position_dofs & locally_owned_dofs;

  // Gather all lists to all processes
  std::vector<std::vector<types::global_dof_index>> gathered_dofs =
    Utilities::MPI::all_gather(this->mpi_communicator,
                               local_lambda_dofs.get_index_vector());

  std::vector<types::global_dof_index> gathered_dofs_flattened;
  for (const auto &vec : gathered_dofs)
    gathered_dofs_flattened.insert(gathered_dofs_flattened.end(),
                                   vec.begin(),
                                   vec.end());

  std::sort(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());

  // Add the lambda DoFs to the list of locally relevant
  // DoFs: Do this only if partition contains a chunk of the cylinder
  if (n_local_lambda_dofs > 0)
  {
    locally_relevant_dofs.add_indices(gathered_dofs_flattened.begin(),
                                      gathered_dofs_flattened.end());
    locally_relevant_dofs.compress();
  }

  //
  // Compute the weights c_ij and identify the constrained position DOFs.
  // Done only once as cylinder is rigid and those weights will not change.
  //
  std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

  FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                         fe,
                                         face_quadrature,
                                         update_values |
                                           // update_quadrature_points |
                                           update_JxW_values);

  const unsigned int                   n_dofs_per_face = fe.n_dofs_per_face();
  std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        fe_face_values_fixed.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int q = 0; q < face_quadrature.size(); ++q)
        {
          const double JxW = fe_face_values_fixed.JxW(q);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
          {
            const unsigned int comp =
              fe.face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which
            // contribute to the integral on this element
            if (!locally_relevant_dofs.is_element(face_dofs[i_dof]))
              continue;

            // Lambda face dofs contribute to the weights
            if (is_lambda(comp))
            {
              const types::global_dof_index lambda_dof = face_dofs[i_dof];

              // Very, very, very important:
              // Even though fe_face_values_fixed is a FEFaceValues, the dof
              // index given to shape_value is still a CELL dof index.
              const unsigned int i_cell_dof =
                fe.face_to_cell_index(i_dof, i_face);

              const unsigned int d = comp - l_lower;
              const double       phi_i =
                fe_face_values_fixed.shape_value(i_cell_dof, q);
              coeffs[d][lambda_dof] +=
                -phi_i * JxW / this->param.fsi.spring_constant;
            }

            // Position face dofs are added to the list of coupled dofs
            if (is_position(comp))
            {
              const unsigned int d = comp - x_lower;
              coupled_position_dofs.insert({face_dofs[i_dof], d});
            }
          }
        }
      }

  //
  // Gather the constraint weights
  //
  position_lambda_coeffs.resize(dim);
  std::vector<std::map<unsigned int, double>> gathered_coeffs_map(dim);

  for (unsigned int d = 0; d < dim; ++d)
  {
    std::vector<std::pair<unsigned int, double>> coeffs_vector(
      coeffs[d].begin(), coeffs[d].end());
    std::vector<std::vector<std::pair<unsigned int, double>>> gathered =
      Utilities::MPI::all_gather(this->mpi_communicator, coeffs_vector);

    // Put back into map and sum contributions to same DoF from different
    // processes
    for (const auto &vec : gathered)
      for (const auto &pair : vec)
        gathered_coeffs_map[d][pair.first] += pair.second;

    position_lambda_coeffs[d].insert(position_lambda_coeffs[d].end(),
                                     gathered_coeffs_map[d].begin(),
                                     gathered_coeffs_map[d].end());
  }
}

template <int dim>
void MonolithicFSISolver<dim>::create_zero_constraints()
{
  zero_constraints.clear();
  zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  const FEValuesExtractors::Vector velocity(u_lower);
  const FEValuesExtractors::Vector position(x_lower);

  //
  // Mesh position homogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    for (const auto &bc : this->param.pseudosolid_bc)
    {
      if (bc.type == BoundaryConditions::Type::fixed ||
          bc.type == BoundaryConditions::Type::input_function)
      {
        VectorTools::interpolate_boundary_values(*fixed_mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 zero_constraints,
                                                 fe.component_mask(position));
      }

      if (bc.type == BoundaryConditions::Type::no_flux)
        no_flux_boundaries.insert(bc.id);
    }

    // Add no position flux constraints (tangential movement)
    VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                    x_lower,
                                                    no_flux_boundaries,
                                                    zero_constraints,
                                                    *fixed_mapping);
  }

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

  // Merge the zero lambda constraints
  zero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
    true);
}

template <int dim>
void MonolithicFSISolver<dim>::create_nonzero_constraints()
{
  nonzero_constraints.clear();
  nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  const FEValuesExtractors::Vector velocity(u_lower);
  const FEValuesExtractors::Vector position(x_lower);

  //
  // Mesh position inhomogeneous BC
  //
  {
    std::set<types::boundary_id> no_flux_boundaries;
    for (const auto &bc : this->param.pseudosolid_bc)
    {
      if (bc.type == BoundaryConditions::Type::fixed)
      {
        VectorTools::interpolate_boundary_values(
          *fixed_mapping,
          dof_handler,
          bc.id,
          FixedMeshPosition<dim>(x_lower, n_components),
          nonzero_constraints,
          fe.component_mask(position));
      }
      if (bc.type == BoundaryConditions::Type::input_function)
      {
        // TODO: Prescribed but non-fixed mesh position?
        AssertThrow(
          false,
          ExcMessage(
            "Input function for pseudosolid problem are not yet handled."));
      }
      if (bc.type == BoundaryConditions::Type::no_flux)
        no_flux_boundaries.insert(bc.id);
    }

    // Add no position flux constraints (tangential movement)
    VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                    x_lower,
                                                    no_flux_boundaries,
                                                    nonzero_constraints,
                                                    *fixed_mapping);
  }

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

  // Merge the zero lambda constraints
  nonzero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
    true);
}

template <int dim>
void MonolithicFSISolver<dim>::create_sparsity_pattern()
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

  // Add the position-lambda couplings explicitly
  // In a first (current) naive approach, each position dof is coupled to all
  // lambda dofs on cylinder
  for (const auto &[position_dof, d] : coupled_position_dofs)
    for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
      dsp.add(position_dof, lambda_dof);

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
void MonolithicFSISolver<dim>::set_initial_conditions()
{
  const FEValuesExtractors::Vector velocity(u_lower);
  const FEValuesExtractors::Vector position(x_lower);

  // Update mesh position *BEFORE* evaluating scalar field
  // with moving mapping (-:
  // This does not matter here, as the initial mesh position
  // is the fixed_mapping.

  // Set mesh position with fixed mapping
  VectorTools::interpolate(*fixed_mapping,
                           dof_handler,
                           FixedMeshPosition<dim>(x_lower, n_components),
                           this->newton_update,
                           fe.component_mask(position));

  // Set velocity with moving mapping (irrelevant for initial position)
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
void MonolithicFSISolver<dim>::update_boundary_conditions()
{
  // Re-create and distribute nonzero constraints:
  this->local_evaluation_point = this->present_solution;
  this->create_nonzero_constraints();
  nonzero_constraints.distribute(this->local_evaluation_point);
  this->present_solution = this->local_evaluation_point;
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_matrix()
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

  ScratchDataMonolithicFSI<dim> scratchData(fe,
                                            quadrature,
                                            *fixed_mapping,
                                            *mapping,
                                            face_quadrature,
                                            dofs_per_cell,
                                            weak_no_slip_boundary_id,
                                            time_handler.bdf_coefficients);

  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
  {
    if (!cell->is_locally_owned())
      continue;

    cell->get_dof_indices(local_dof_indices);

    if (this->param.nonlinear_solver.analytic_jacobian)
    {
      //
      // Analytic jacobian matrix
      //
      const bool distribute = true;
      this->assemble_local_matrix(first_step,
                                  cell,
                                  scratchData,
                                  this->evaluation_point,
                                  previous_solutions,
                                  local_dof_indices,
                                  local_matrix,
                                  distribute);
    }
    else
    {
      //
      // Finite differences
      //
      const double h      = 1.e-8;
      local_matrix        = 0.;
      ref_local_rhs       = 0.;
      perturbed_local_rhs = 0.;

      // Get the local dofs values
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        cell_dof_values[j] = this->evaluation_point[local_dof_indices[j]];

      const bool distribute_rhs    = false;
      const bool use_full_solution = false;

      // Compute non-perturbed RHS
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               this->evaluation_point,
                               previous_solutions,
                               local_dof_indices,
                               ref_local_rhs,
                               cell_dof_values,
                               distribute_rhs,
                               use_full_solution);

      for (unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        const unsigned int comp     = fe.system_to_component_index(j).first;
        const double       og_value = cell_dof_values[j];
        cell_dof_values[j] += h;

        if (is_position(comp))
        {
          // Also modify mapping_fe_field
          this->local_evaluation_point[local_dof_indices[j]] =
            cell_dof_values[j];
          this->local_evaluation_point.compress(VectorOperation::insert);
          this->evaluation_point = this->local_evaluation_point;
        }

        // Compute perturbed RHS
        // Reinit is called in the local rhs function
        this->assemble_local_rhs(first_step,
                                 cell,
                                 scratchData,
                                 this->evaluation_point,
                                 previous_solutions,
                                 local_dof_indices,
                                 perturbed_local_rhs,
                                 cell_dof_values,
                                 distribute_rhs,
                                 use_full_solution);

        // Finite differences (with sign change as residual is -NL(u))
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
        }

        // Restore solution
        cell_dof_values[j] = og_value;
        if (is_position(comp))
        {
          // Also modify mapping_fe_field
          this->local_evaluation_point[local_dof_indices[j]] = og_value;
          this->local_evaluation_point.compress(VectorOperation::insert);
          this->evaluation_point = this->local_evaluation_point;
        }
      }
    }
  }

  system_matrix.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    this->add_algebraic_position_coupling_to_matrix();
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_local_matrix(
  bool                                                  first_step,
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataMonolithicFSI<dim>                        &scratchData,
  LA::ParVectorType                                        &current_solution,
  std::vector<LA::ParVectorType>                           &previous_solutions,
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
  const double lame_lambda =
    this->param.physical_properties.pseudosolids[0].lame_lambda;
  const double lame_mu =
    this->param.physical_properties.pseudosolids[0].lame_mu;

  const double bdf_c0 = time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW_moving = scratchData.JxW_moving[q];
    const double JxW_fixed  = scratchData.JxW_fixed[q];

    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];
    const auto &phi_p      = scratchData.phi_p[q];
    const auto &phi_x      = scratchData.phi_x[q];
    const auto &grad_phi_x = scratchData.grad_phi_x[q];
    const auto &div_phi_x  = scratchData.div_phi_x[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);
    const double present_pressure_values =
      scratchData.present_pressure_values[q];

    const auto &dxdt = scratchData.present_mesh_velocity_values[q];

    // BDF: current dudt
    Tensor<1, dim> dudt =
      time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratchData.previous_velocity_values);
    // Tensor<1, dim> dudt = time_handler.bdf_coefficients[0] *
    // present_velocity_values; for (unsigned int i = 1; i <
    // time_handler.bdf_coefficients.size(); ++i)
    //   dudt += time_handler.bdf_coefficients[i] *
    //   scratchData.previous_velocity_values[i - 1][q];

    const auto &source_term_velocity = scratchData.source_term_velocity[q];
    const auto &source_term_pressure = scratchData.source_term_pressure[q];
    const auto &grad_source_velocity = scratchData.grad_source_velocity[q];
    const auto &grad_source_pressure = scratchData.grad_source_pressure[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratchData.components[i];
      const bool         i_is_u      = is_velocity(component_i);
      const bool         i_is_p      = is_pressure(component_i);
      const bool         i_is_x      = is_position(component_i);

      for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratchData.components[j];
        const bool         j_is_u      = is_velocity(component_j);
        const bool         j_is_p      = is_pressure(component_j);
        const bool         j_is_x      = is_position(component_j);

        bool   assemble             = false;
        double local_flow_matrix_ij = 0.;
        double local_ps_matrix_ij   = 0.;

        if (i_is_u && j_is_u)
        {
          assemble = true;

          // Time-dependent
          local_flow_matrix_ij += bdf_c0 * phi_u[i] * phi_u[j];

          // Convection
          local_flow_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                                   present_velocity_gradients * phi_u[j]) *
                                  phi_u[i];

          // Diffusion
          local_flow_matrix_ij +=
            kinematic_viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);

          // ALE acceleration : - w dot grad(delta u)
          local_flow_matrix_ij += grad_phi_u[j] * (-dxdt) * phi_u[i];
        }

        if (i_is_u && j_is_p)
        {
          assemble = true;

          // Pressure gradient
          local_flow_matrix_ij += -div_phi_u[i] * phi_p[j];
        }

        if (i_is_u && j_is_x)
        {
          assemble = true;

          // Variation of time-dependent term with mesh position
          local_flow_matrix_ij += dudt * phi_u[i] * trace(grad_phi_x[j]);

          // Variation of ALE term (dxdt cdot grad(u)) with mesh position
          local_flow_matrix_ij +=
            present_velocity_gradients * (-bdf_c0 * phi_x[j]) * phi_u[i];
          local_flow_matrix_ij +=
            (-present_velocity_gradients * grad_phi_x[j]) * (-dxdt) * phi_u[i];
          local_flow_matrix_ij += present_velocity_gradients * (-dxdt) *
                                  phi_u[i] * trace(grad_phi_x[j]);

          // Convection w.r.t. x
          local_flow_matrix_ij +=
            (-present_velocity_gradients * grad_phi_x[j]) *
            present_velocity_values * phi_u[i];
          local_flow_matrix_ij += present_velocity_gradients *
                                  present_velocity_values * phi_u[i] *
                                  trace(grad_phi_x[j]);

          // Diffusion
          const Tensor<2, dim> d_grad_u =
            -present_velocity_gradients * grad_phi_x[j];
          const Tensor<2, dim> d_grad_phi_u = -grad_phi_u[i] * grad_phi_x[j];
          local_flow_matrix_ij +=
            kinematic_viscosity * scalar_product(d_grad_u, grad_phi_u[i]);
          local_flow_matrix_ij +=
            kinematic_viscosity *
            scalar_product(present_velocity_gradients, d_grad_phi_u);
          local_flow_matrix_ij +=
            kinematic_viscosity *
            scalar_product(present_velocity_gradients, grad_phi_u[i]) *
            trace(grad_phi_x[j]);

          // Pressure gradient
          local_flow_matrix_ij +=
            -present_pressure_values * trace(-grad_phi_u[i] * grad_phi_x[j]);
          local_flow_matrix_ij +=
            -present_pressure_values * div_phi_u[i] * trace(grad_phi_x[j]);

          // Source term for velocity:
          // Variation of the source term integral with mesh position.
          // det J is accounted for at the end when multiplying by JxW(q).
          local_flow_matrix_ij += phi_u[i] * grad_source_velocity * phi_x[j];
          local_flow_matrix_ij +=
            source_term_velocity * phi_u[i] * trace(grad_phi_x[j]);
        }

        if (i_is_p && j_is_u)
        {
          assemble = true;

          // Continuity : variation w.r.t. u
          local_flow_matrix_ij += -phi_p[i] * div_phi_u[j];
        }

        if (i_is_p && j_is_x)
        {
          assemble = true;

          // Continuity : variation w.r.t. x
          local_flow_matrix_ij +=
            -trace(-present_velocity_gradients * grad_phi_x[j]) * phi_p[i];
          local_flow_matrix_ij +=
            -present_velocity_divergence * phi_p[i] * trace(grad_phi_x[j]);

          // Source term for pressure:
          local_flow_matrix_ij += phi_p[i] * grad_source_pressure * phi_x[j];
          local_flow_matrix_ij +=
            source_term_pressure * phi_p[i] * trace(grad_phi_x[j]);
        }

        //
        // Pseudo-solid
        //
        if (i_is_x && j_is_x)
        {
          assemble = true;

          // Linear elasticity
          local_ps_matrix_ij +=
            lame_lambda * div_phi_x[j] * div_phi_x[i] +
            // param.pseudo_solid_mu * scalar_product((grad_phi_x[j] +
            // transpose(grad_phi_x[j])), grad_phi_x[i]);
            lame_mu * scalar_product((grad_phi_x[i] + transpose(grad_phi_x[i])),
                                     grad_phi_x[j]);
        }

        if (assemble)
        {
          local_flow_matrix_ij *= JxW_moving;
          local_ps_matrix_ij *= JxW_fixed;
          local_matrix(i, j) += local_flow_matrix_ij + local_ps_matrix_ij;

          // Check that flow and pseudo-solid matrices don't overlap
          AssertThrow(!(std::abs(local_ps_matrix_ij) > 1e-14 &&
                        std::abs(local_flow_matrix_ij) > 1e-14),
                      ExcMessage("Mismatch"));
        }
      }
    }
  }

  //
  // Face contributions (Lagrange multiplier)
  //
  if (cell->at_boundary())
  {
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
        {
          const double face_JxW_moving = scratchData.face_JxW_moving[i_face][q];

          const auto &phi_u = scratchData.phi_u_face[i_face][q];
          const auto &phi_x = scratchData.phi_x_face[i_face][q];
          const auto &phi_l = scratchData.phi_l_face[i_face][q];

          const auto &present_u =
            scratchData.present_face_velocity_values[i_face][q];
          const auto &present_w =
            scratchData.present_face_mesh_velocity_values[i_face][q];
          const auto &present_l =
            scratchData.present_face_lambda_values[i_face][q];

          for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
          {
            const unsigned int component_i = scratchData.components[i];
            const bool         i_is_u      = is_velocity(component_i);
            const bool         i_is_l      = is_lambda(component_i);

            for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
            {
              const unsigned int component_j = scratchData.components[j];
              const bool         j_is_u      = is_velocity(component_j);
              const bool         j_is_x      = is_position(component_j);
              const bool         j_is_l      = is_lambda(component_j);

              const double delta_dx_j = scratchData.delta_dx[i_face][q][j];

              double local_matrix_ij = 0.;

              if (i_is_u && j_is_x)
              {
                local_matrix_ij += -(present_l * phi_u[i]) * delta_dx_j;
              }

              if (i_is_u && j_is_l)
              {
                local_matrix_ij += -(phi_l[j] * phi_u[i]);
              }

              if (i_is_l && j_is_u)
              {
                local_matrix_ij += -phi_u[j] * phi_l[i];
              }

              if (i_is_l && j_is_x)
              {
                local_matrix_ij += -(-bdf_c0 * phi_x[j] * phi_l[i]);
                local_matrix_ij +=
                  -((present_u - present_w) * phi_l[i] * delta_dx_j);
              }

              local_matrix_ij *= face_JxW_moving;
              local_matrix(i, j) += local_matrix_ij;
            }
          }
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
      // for (unsigned int ii = 0; ii < scratchData.dofs_per_cell; ++ii)
      //   for (unsigned int jj = 0; jj < scratchData.dofs_per_cell; ++jj)
      //     system_matrix.add(local_dof_indices[ii],
      //                       local_dof_indices[jj],
      //                       local_matrix(ii, jj));
    }
  }
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  const bool first_step = false;

  this->system_rhs = 0;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  Vector<double>                       local_rhs(dofs_per_cell);
  std::vector<double>                  cell_dof_values(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  ScratchDataMonolithicFSI<dim> scratchData(fe,
                                            quadrature,
                                            *fixed_mapping,
                                            *mapping,
                                            face_quadrature,
                                            dofs_per_cell,
                                            weak_no_slip_boundary_id,
                                            time_handler.bdf_coefficients);

  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
  {
    local_rhs              = 0;
    bool distribute        = true;
    bool use_full_solution = true;
    this->assemble_local_rhs(first_step,
                             cell,
                             scratchData,
                             this->evaluation_point,
                             previous_solutions,
                             local_dof_indices,
                             local_rhs,
                             cell_dof_values,
                             distribute,
                             use_full_solution);
  }

  this->system_rhs.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    this->add_algebraic_position_coupling_to_rhs();
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_local_rhs(
  bool                                                  first_step,
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataMonolithicFSI<dim>                        &scratchData,
  LA::ParVectorType                                        &current_solution,
  std::vector<LA::ParVectorType>                           &previous_solutions,
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
  const double lame_lambda =
    this->param.physical_properties.pseudosolids[0].lame_lambda;
  const double lame_mu =
    this->param.physical_properties.pseudosolids[0].lame_mu;

  const std::vector<double> bdf_coefficients = time_handler.bdf_coefficients;

  const unsigned int          nBDF = bdf_coefficients.size();
  std::vector<Tensor<1, dim>> velocity(nBDF);

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    //
    // Flow related data
    //
    const double JxW_moving = scratchData.JxW_moving[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];
    const auto &present_pressure_values =
      scratchData.present_pressure_values[q];
    const auto &present_mesh_velocity_values =
      scratchData.present_mesh_velocity_values[q];
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

    //
    // Pseudo-solid related data
    //
    const double JxW_fixed = scratchData.JxW_fixed[q];

    const auto &present_position_gradients =
      scratchData.present_position_gradients[q];
    const double present_displacement_divergence =
      trace(present_position_gradients);
    const auto present_displacement_gradient_sym =
      present_position_gradients + transpose(present_position_gradients);
    const auto &source_term_position = scratchData.source_term_position[q];

    const auto &phi_x      = scratchData.phi_x[q];
    const auto &grad_phi_x = scratchData.grad_phi_x[q];
    const auto &div_phi_x  = scratchData.div_phi_x[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      //
      // Flow residual
      //
      double local_rhs_flow_i = -(
        // Convection
        (present_velocity_gradients * present_velocity_values) * phi_u[i]

        // Mesh movement
        - (present_velocity_gradients * present_mesh_velocity_values) * phi_u[i]

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
        local_rhs_flow_i -= bdf_coefficients[iBDF] * velocity[iBDF] * phi_u[i];
      }

      local_rhs_flow_i *= JxW_moving;

      //
      // Pseudo-solid
      //
      double local_rhs_ps_i = -( // Linear elasticity
        lame_lambda * present_displacement_divergence * div_phi_x[i] +
        // param.pseudo_solid_mu * scalar_product(grad_phi_x[i],
        // present_displacement_gradient_sym)
        lame_mu *
          scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

        // Linear elasticity source term
        + phi_x[i] * source_term_position);

      local_rhs_ps_i *= JxW_fixed;

      local_rhs(i) += local_rhs_flow_i + local_rhs_ps_i;
    }
  }

  //
  // Face contributions (Lagrange multiplier)
  //
  if (cell->at_boundary())
  {
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
        {
          //
          // Flow related data (no-slip)
          //
          const double face_JxW_moving = scratchData.face_JxW_moving[i_face][q];
          const auto  &phi_u           = scratchData.phi_u_face[i_face][q];
          const auto  &phi_l           = scratchData.phi_l_face[i_face][q];

          const auto &present_u =
            scratchData.present_face_velocity_values[i_face][q];
          const auto &present_w =
            scratchData.present_face_mesh_velocity_values[i_face][q];
          const auto &present_l =
            scratchData.present_face_lambda_values[i_face][q];

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            double local_rhs_i = 0.;

            const unsigned int component_i = scratchData.components[i];
            const bool         i_is_u      = is_velocity(component_i);
            const bool         i_is_l      = is_lambda(component_i);

            if (i_is_u)
            {
              local_rhs_i -= -(phi_u[i] * present_l);
            }

            if (i_is_l)
            {
              local_rhs_i -= -(present_u - present_w) * phi_l[i];
            }

            local_rhs_i *= face_JxW_moving;
            local_rhs(i) += local_rhs_i;
          }
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
void MonolithicFSISolver<dim>::add_algebraic_position_coupling_to_matrix()
{
  //
  // Add algebraic constraints position-lambda
  //
  std::map<types::global_dof_index,
           std::vector<LA::ConstMatrixIterator>>
    position_row_entries;
  // Get row entries for each pos_dof
  for (const auto &[pos_dof, d] : coupled_position_dofs)
  {
    if (locally_owned_dofs.is_element(pos_dof))
    {
      std::vector<LA::ConstMatrixIterator> row_entries;
      for (auto it = system_matrix.begin(pos_dof);
           it != system_matrix.end(pos_dof);
           ++it)
        row_entries.push_back(it);
      position_row_entries[pos_dof] = row_entries;
    }
  }

  // Constrain matrix and RHS
  for (const auto &[pos_dof, d] : coupled_position_dofs)
  {
    if (locally_owned_dofs.is_element(pos_dof))
    {
      for (auto it : position_row_entries.at(pos_dof))
      {
        // std::cout << "zeroing " << pos_dof << " - " << it->column() <<
        // std::endl;
        system_matrix.set(pos_dof, it->column(), 0.0);
      }

      // Set constraint row: x_i - sum_j c_ij * lambda_j = 0
      system_matrix.set(pos_dof, pos_dof, 1.);
      for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
        system_matrix.set(pos_dof, lambda_dof, -weight);
    }
  }

  system_matrix.compress(VectorOperation::insert);
}

template <int dim>
void MonolithicFSISolver<dim>::add_algebraic_position_coupling_to_rhs()
{
  // Set RHS to zero for coupled position dofs
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (locally_owned_dofs.is_element(pos_dof))
      this->system_rhs(pos_dof) = 0.;

  this->system_rhs.compress(VectorOperation::insert);
}

template <int dim>
void MonolithicFSISolver<dim>::solve_linear_system(
  const bool /*apply_inhomogeneous_constraints*/)
{
  solve_linear_system_direct(this, system_matrix, locally_owned_dofs, zero_constraints);
  // TimerOutput::Scope t(computing_timer, "Solve direct");

  // LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
  //                                               this->mpi_communicator);

  // // Solve with MUMPS
  // SolverControl                    solver_control;
  // PETScWrappers::SparseDirectMUMPS solver(solver_control);
  // solver.solve(system_matrix,
  //              completely_distributed_solution,
  //              this->system_rhs);

  // this->newton_update = completely_distributed_solution;

  // if (apply_inhomogeneous_constraints)
  // {
  //   throw std::runtime_error("First step");
  //   nonzero_constraints.distribute(this->newton_update);
  // }
  // else
  //   zero_constraints.distribute(this->newton_update);
}

/**
 * Compute integral of lambda (fluid force), compare to position dofs
 */
template <int dim>
void MonolithicFSISolver<dim>::compare_forces_and_position_on_obstacle() const
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  const FEValuesExtractors::Vector lambda(l_lower);

  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values.reinit(cell, i_face);

        // Get FE solution values on the face
        fe_face_values[lambda].get_function_values(this->present_solution,
                                                   lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          // const Point<dim> &qpoint = fe_face_values.quadrature_point(q);

          // Increment the integral of lambda
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
        }
      }
    }
  }

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
  }

  //
  // Position BC
  //
  Tensor<1, dim> cylinder_displacement_local, max_diff_local;
  bool           first_displacement_x = true;
  bool           first_displacement_y = true;
  std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      if (!(face->at_boundary() &&
            face->boundary_id() == weak_no_slip_boundary_id))
        continue;

      face->get_dof_indices(face_dofs);

      for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
      {
        if (!locally_owned_dofs.is_element(face_dofs[i]))
          continue;

        const unsigned int comp =
          fe.face_system_to_component_index(i, i_face).first;

        //
        // Displacement or position coupling
        //
        if (is_position(comp))
        {
          const unsigned int d = comp - x_lower;
          if (d == 0 && first_displacement_x)
          {
            first_displacement_x = false;
            cylinder_displacement_local[d] =
              this->present_solution[face_dofs[i]] -
              this->initial_positions.at(face_dofs[i])[d];
          }
          if (d == 1 && first_displacement_y)
          {
            first_displacement_y = false;
            cylinder_displacement_local[d] =
              this->present_solution[face_dofs[i]] -
              this->initial_positions.at(face_dofs[i])[d];
          }
          if (!first_displacement_x && !first_displacement_y)
          {
            // Compare with cylinder_displacement_local
            const double displ = this->present_solution[face_dofs[i]] -
                                 this->initial_positions.at(face_dofs[i])[d];
            max_diff_local[d] =
              std::max(max_diff_local[d],
                       cylinder_displacement_local[d] - displ);
          }
        }
      }
    }
  }

  // To take the max displacement while preserving sign
  struct MaxAbsOp
  {
    static void
    apply(void *invec, void *inoutvec, int *len, MPI_Datatype * /*dtype*/)
    {
      double *in    = static_cast<double *>(invec);
      double *inout = static_cast<double *>(inoutvec);
      for (int i = 0; i < *len; ++i)
      {
        if (std::fabs(in[i]) > std::fabs(inout[i]))
          inout[i] = in[i];
      }
    }
  };
  MPI_Op mpi_maxabs;
  MPI_Op_create(&MaxAbsOp::apply, /*commutative=*/true, &mpi_maxabs);

  Tensor<1, dim> cylinder_displacement, max_diff, ratio;
  for (unsigned int d = 0; d < dim; ++d)
  {
    // cylinder_displacement[d] =
    //   Utilities::MPI::max(cylinder_displacement_local[d], mpi_communicator);

    // The cylinder displacement is trivially 0 on processes which do not own
    // a part of the boundary, and is nontrivial otherwise.
    // Taking the max to synchronize does not work because displacement
    // can be negative. Instead, we take the max while preserving the sign.
    MPI_Allreduce(&cylinder_displacement_local[d],
                  &cylinder_displacement[d],
                  1,
                  MPI_DOUBLE,
                  mpi_maxabs,
                  mpi_communicator);

    // Take the max between all max differences disp_i - disp_j
    // for x_i and x_j both on the cylinder.
    // Checks that all displacement are identical.
    max_diff[d] = Utilities::MPI::max(max_diff_local[d], mpi_communicator);

    // Check that the ratio of both terms in the position
    // boundary condition is -spring_constant
    if (std::abs(cylinder_displacement[d]) > 1e-10)
      ratio[d] = lambda_integral[d] / cylinder_displacement[d];
  }

  pcout << std::endl;
  pcout << std::scientific << std::setprecision(8) << std::showpos;
  pcout << "Checking consistency between lambda integral and position BC:"
        << std::endl;
  pcout << "Integral of lambda on cylinder is " << lambda_integral << std::endl;
  pcout << "Prescribed displacement        is " << cylinder_displacement
        << std::endl;
  pcout << "                         Ratio is " << ratio
        << " (expected: " << -param.fsi.spring_constant << ")" << std::endl;
  pcout << "Max diff between displacements is " << max_diff << std::endl;
  AssertThrow(max_diff.norm() <= 1e-10,
              ExcMessage(
                "Displacement values of the cylinder are not all the same."));

  //
  // Check relative error between lambda/disp ratio vs spring constant
  //
  for (unsigned int d = 0; d < dim; ++d)
  {
    if (std::abs(ratio[d]) < 1e-10)
      continue;

    const double absolute_error =
      std::abs(ratio[d] - (-param.fsi.spring_constant));

    if (absolute_error <= 1e-6)
      continue;

    const double relative_error = absolute_error / param.fsi.spring_constant;
    AssertThrow(relative_error <= 1e-2,
                ExcMessage("Ratio integral vs displacement values is not -k"));
  }
  pcout << std::endl;
}

template <int dim>
void MonolithicFSISolver<dim>::check_velocity_boundary() const
{
  // Check difference between uh and dxhdt
  double l2_local = 0;
  double li_local = 0;

  const FEValuesExtractors::Vector velocity(u_lower);
  const FEValuesExtractors::Vector position(x_lower);

  FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                         fe,
                                         face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);
  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  const unsigned int n_faces_q_points = face_quadrature.size();

  const auto &bdf_coefficients = time_handler.bdf_coefficients;

  std::vector<std::vector<Tensor<1, dim>>> position_values(
    bdf_coefficients.size(), std::vector<Tensor<1, dim>>(n_faces_q_points));
  std::vector<Tensor<1, dim>> mesh_velocity_values(n_faces_q_points);
  std::vector<Tensor<1, dim>> fluid_velocity_values(n_faces_q_points);
  Tensor<1, dim>              diff;

  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values_fixed.reinit(cell, i_face);
        fe_face_values.reinit(cell, i_face);

        // Get current and previous FE solution values on the face
        fe_face_values[velocity].get_function_values(present_solution,
                                                     fluid_velocity_values);
        fe_face_values_fixed[position].get_function_values(present_solution,
                                                           position_values[0]);
        for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
          fe_face_values_fixed[position].get_function_values(
            previous_solutions[iBDF - 1], position_values[iBDF]);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          // Compute FE mesh velocity at node
          mesh_velocity_values[q] = 0;
          for (unsigned int iBDF = 0; iBDF < bdf_coefficients.size(); ++iBDF)
            mesh_velocity_values[q] +=
              bdf_coefficients[iBDF] * position_values[iBDF][q];

          diff = mesh_velocity_values[q] - fluid_velocity_values[q];

          // std::cout << std::scientific << std::setprecision(8) <<
          // std::showpos;
          // // std::cout << "wh = " << mesh_velocity_values[q] << std::endl;
          // std::cout << "wh = " << mesh_velocity_values[q] << " - uh = " <<
          // fluid_velocity_values[q] << " - diff = " << diff << std::endl;

          // u_h - w_h
          l2_local += diff * diff * fe_face_values_fixed.JxW(q);
          li_local = std::max(li_local, std::abs(diff.norm()));
        }
      }
    }
  }

  const double l2_error =
    std::sqrt(Utilities::MPI::sum(l2_local, mpi_communicator));
  const double li_error = Utilities::MPI::max(li_local, mpi_communicator);
  pcout << "Checking no-slip enforcement on cylinder:" << std::endl;
  pcout << "||uh - wh||_L2 = " << l2_error << std::endl;
  pcout << "||uh - wh||_Li = " << li_error << std::endl;

  if (!(param.time_integration.scheme ==
          Parameters::TimeIntegration::Scheme::BDF2 &&
        time_handler.current_time_iteration == 1))
  {
    AssertThrow(l2_error < 1e-12,
                ExcMessage("L2 norm of uh - wh is too large."));
    AssertThrow(li_error < 1e-12,
                ExcMessage("Linf norm of uh - wh is too large."));
  }
}

template <int dim>
void MonolithicFSISolver<dim>::output_results() const
{
  //
  // Plot FE solution
  //
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("mesh_position");
  for (unsigned int d = 0; d < dim; ++d)
    solution_names.push_back("lambda");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);
  for (unsigned int d = 0; d < 2 * dim; ++d)
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(this->present_solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  //
  // Compute mesh velocity in post-processing
  // This is not ideal, this is done by modifying the displacement and
  // reexporting.
  //
  LA::ParVectorType mesh_velocity;
  mesh_velocity.reinit(locally_owned_dofs, this->mpi_communicator);
  const FEValuesExtractors::Vector position(x_lower);
  IndexSet                         disp_dofs =
    DoFTools::extract_dofs(dof_handler, fe.component_mask(position));

  for (const auto &i : disp_dofs)
    if (locally_owned_dofs.is_element(i))
      mesh_velocity[i] =
        time_handler.compute_time_derivative(i,
                                             this->present_solution,
                                             previous_solutions);
  mesh_velocity.compress(VectorOperation::insert);

  std::vector<std::string> mesh_velocity_name(dim, "ph_velocity");
  mesh_velocity_name.emplace_back("ph_pressure");
  for (unsigned int i = 0; i < dim; ++i)
    mesh_velocity_name.push_back("mesh_velocity");
  for (unsigned int i = 0; i < dim; ++i)
    mesh_velocity_name.push_back("ph_lambda");

  data_out.add_data_vector(mesh_velocity,
                           mesh_velocity_name,
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
  data_out.write_vtu_with_pvtu_record(this->param.output.output_dir,
                                      this->param.output.output_prefix,
                                      time_handler.current_time_iteration,
                                      this->mpi_communicator,
                                      2);
}

template <int dim>
void MonolithicFSISolver<dim>::compute_forces(const bool export_table)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  const FEValuesExtractors::Vector lambda(l_lower);

  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values.reinit(cell, i_face);

        // Get FE solution values on the face
        fe_face_values[lambda].get_function_values(present_solution,
                                                   lambda_values);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
      }
    }
  }

  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);

  // const double rho = param.physical_properties.fluids[0].density;
  // const double U   = boundary_description.U;
  // const double D   = boundary_description.D;
  // const double factor = 1. / (0.5 * rho * U * U * D);

  //
  // Forces on the cylinder are the NEGATIVE of the integral of lambda
  //
  forces_table.add_value("time", time_handler.current_time);
  forces_table.add_value("CFx", -lambda_integral[0]);
  forces_table.add_value("CFy", -lambda_integral[1]);
  if constexpr (dim == 3)
  {
    forces_table.add_value("CFz", -lambda_integral[2]);
  }

  if (export_table && mpi_rank == 0)
  {
    std::ofstream outfile(param.output.output_dir + "forces.txt");
    forces_table.write_text(outfile);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::write_cylinder_position(const bool export_table)
{
  Tensor<1, dim> average_position, position_integral_local;
  double         boundary_measure_local = 0.;

  const FEValuesExtractors::Vector position(x_lower);

  FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                         fe,
                                         face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values |
                                           update_normal_vectors);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  std::vector<Tensor<1, dim>> position_values(n_faces_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values_fixed.reinit(cell, i_face);

        // Get FE solution values on the face
        fe_face_values_fixed[position].get_function_values(present_solution,
                                                           position_values);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          boundary_measure_local += fe_face_values_fixed.JxW(q);
          position_integral_local +=
            position_values[q] * fe_face_values_fixed.JxW(q);
        }
      }
    }
  }

  const double boundary_measure =
    Utilities::MPI::sum(boundary_measure_local, mpi_communicator);
  for (unsigned int d = 0; d < dim; ++d)
    average_position[d] =
      1. / boundary_measure *
      Utilities::MPI::sum(position_integral_local[d], mpi_communicator);

  cylinder_position_table.add_value("time", time_handler.current_time);
  cylinder_position_table.add_value("xc", average_position[0]);
  cylinder_position_table.add_value("yc", average_position[1]);
  if constexpr (dim == 3)
    cylinder_position_table.add_value("zc", average_position[2]);

  if (export_table && mpi_rank == 0)
  {
    std::ofstream outfile(param.output.output_dir + "cylinder_center.txt");
    cylinder_position_table.write_text(outfile);
  }
}

// Explicit instantiation
template class MonolithicFSISolver<2>;
template class MonolithicFSISolver<3>;