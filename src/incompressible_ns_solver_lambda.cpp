
#include <boundary_conditions.h>
#include <compare_matrix.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <fe_simplex_p_with_3d_hp.h>
#include <incompressible_ns_solver_lambda.h>
#include <linear_solver.h>
#include <mesh.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
NSSolverLambda<dim>::NSSolverLambda(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim>(param)
{
  if (param.finite_elements.use_quads)
  {
    fe_with_lambda = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,
      FE_Q<dim>(param.finite_elements.pressure_degree),
      FE_Q<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^ dim);
    fe_without_lambda = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,
      FE_Q<dim>(param.finite_elements.pressure_degree),
      FE_Nothing<dim>() ^ dim);
  }
  else
  {
    /**
     * Currently, the FE_SimplexP class throws when applying hp dofs identities
     * for dim = 3. To bypass this, this solver uses a "FE_SimplexP_3D_hp"
     * space, an override of FE_SimplexP which is intended to implement the hp
     * dof identities for dim = 3. However, the hp_line_identities is never
     * entered at all, so these identities are created by hand here in
     * create_hp_line_dofs_identities(). So all in all, FE_SimplexP_3D_hp is for
     * now just a FE_SimplexP which does not throw when entering
     * hp_quad_dof_identities, but simply returns empty identities instead.
     */
    if constexpr (dim == 2)
    {
      fe_with_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP_3D_hp<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP_3D_hp<dim>(
          param.finite_elements.no_slip_lagrange_mult_degree) ^
          dim);
      fe_without_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP_3D_hp<dim>(param.finite_elements.pressure_degree),
        FE_Nothing<dim>(ReferenceCells::get_simplex<dim>()) ^ dim);
    }
    else
    {
      fe_with_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP_3D_hp<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP_3D_hp<dim>(
          param.finite_elements.no_slip_lagrange_mult_degree) ^
          dim);
      fe_without_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP_3D_hp<dim>(param.finite_elements.pressure_degree),
        FE_Nothing<dim>(ReferenceCells::get_simplex<dim>()) ^ dim);
    }
  }

  fe = std::make_shared<hp::FECollection<dim>>();

  // Ensure fe ordering is consistent throughout the solver
  if constexpr (index_fe_without_lambda == 0)
  {
    fe->push_back(*fe_without_lambda);
    fe->push_back(*fe_with_lambda);
  }
  else
  {
    fe->push_back(*fe_with_lambda);
    fe->push_back(*fe_without_lambda);
  }

  // Add the same fixed mapping and quadrature rules for both FESystems
  mapping_collection.push_back(*this->fixed_mapping);
  mapping_collection.push_back(*this->fixed_mapping);
  quadrature_collection.push_back(*this->quadrature);
  quadrature_collection.push_back(*this->quadrature);
  face_quadrature_collection.push_back(*this->face_quadrature);
  face_quadrature_collection.push_back(*this->face_quadrature);

  this->ordering = std::make_shared<ComponentOrderingNSLambda<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  this->lambda_extractor = FEValuesExtractors::Vector(this->ordering->l_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  this->lambda_mask   = fe->component_mask(this->lambda_extractor);

  // Set the boundary id on which a weak no slip boundary condition is applied.
  unsigned int n_weak_bc = 0;
  for (const auto &[id, bc] : param.fluid_bc)
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
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    this->exact_solution = std::make_shared<NSSolverLambda<dim>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

    // Create the source term function for the given MMS and override source
    // terms
    this->source_terms = std::make_shared<NSSolverLambda<dim>::MMSSourceTerm>(
      this->time_handler.current_time,
      *this->ordering,
      param.physical_properties,
      param.mms);

    // Create entry in error handler for Lagrange multiplier
    for (auto norm : this->param.mms_param.norms_to_compute)
      this->error_handlers[norm]->create_entry("l");
  }
  else
  {
    this->source_terms = std::make_shared<NSSolverLambda<dim>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
void NSSolverLambda<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
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
    values[ordering.u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[ordering.p_lower] = mms.exact_velocity->divergence(p);
}

template <int dim>
void NSSolverLambda<dim>::reset_solver_specific_data()
{
  hp_dof_identities.clear();
}

template <int dim>
void NSSolverLambda<dim>::setup_dofs()
{
  TimerOutput::Scope t(this->computing_timer, "Setup");

  auto &comm = this->mpi_communicator;

  // Mark the cells on which the Lagrange multiplier is defined
  // FIXME: MUST ALSO TAG CELLS WHO ONLY HAVE AN EDGE ON THE BOUNDARY, BUT NO
  // FACES
  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    cell->set_material_id(without_lambda_domain_id);
    if (cell->is_locally_owned())
      cell->set_active_fe_index(index_fe_without_lambda);
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        cell->set_material_id(with_lambda_domain_id);
        if (cell->is_locally_owned())
          cell->set_active_fe_index(index_fe_with_lambda);
        break;
      }
  }

  // Initialize dof handler
  this->dof_handler.distribute_dofs(*fe);

  this->pcout << "Number of degrees of freedom: " << this->dof_handler.n_dofs()
              << std::endl;

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  this->locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(this->dof_handler);

  // Initialize parallel vectors
  this->present_solution.reinit(this->locally_owned_dofs,
                                this->locally_relevant_dofs,
                                comm);
  this->evaluation_point.reinit(this->locally_owned_dofs,
                                this->locally_relevant_dofs,
                                comm);

  this->local_evaluation_point.reinit(this->locally_owned_dofs, comm);
  this->newton_update.reinit(this->locally_owned_dofs, comm);
  this->system_rhs.reinit(this->locally_owned_dofs, comm);

  // Allocate for previous BDF solutions
  this->previous_solutions.clear();
  this->previous_solutions.resize(this->time_handler.n_previous_solutions);
  for (auto &previous_sol : this->previous_solutions)
    previous_sol.reinit(this->locally_owned_dofs,
                        this->locally_relevant_dofs,
                        comm);

  // Unused in this solver
  this->moving_mapping = this->fixed_mapping;

  // For unsteady simulation, add the number of elements, dofs and/or the time
  // step to the error handler, once per convergence run.
  if (!this->time_handler.is_steady() && this->param.mms_param.enable)
    for (auto &[norm, handler] : this->error_handlers)
    {
      handler->add_reference_data("n_elm",
                                  this->triangulation.n_global_active_cells());
      handler->add_reference_data("n_dof", this->dof_handler.n_dofs());
      handler->add_time_step(this->time_handler.initial_dt);
    }
}

template <int dim>
void NSSolverLambda<dim>::create_lagrange_multiplier_constraints()
{
  // Out of the cells where a Lagrange multiplier is defined,
  // mark the lambda dofs to set to zero (those not on faces with
  // the prescribed boundary id).
  lambda_constraints.reinit(this->locally_owned_dofs,
                            this->locally_relevant_dofs);

  // If there is no weakly enforced no slip boundary, this set remains empty and
  // all lambda dofs are constrained.
  IndexSet relevant_boundary_dofs;

  if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
  {
    relevant_boundary_dofs =
      DoFTools::extract_boundary_dofs(this->dof_handler,
                                      lambda_mask,
                                      {weak_no_slip_boundary_id});
  }

  // There does not seem to be a 2-3 liner way to extract the locally
  // relevant dofs on a boundary for a given component (extract_dofs
  // returns owned dofs).
  std::vector<types::global_dof_index> local_dofs(
    fe_with_lambda->n_dofs_per_cell());
  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;
    if (cell_has_lambda(cell))
    {
      cell->get_dof_indices(local_dofs);
      for (unsigned int i = 0; i < local_dofs.size(); ++i)
      {
        types::global_dof_index dof = local_dofs[i];
        unsigned int comp = fe_with_lambda->system_to_component_index(i).first;
        if (this->ordering->is_lambda(comp))
          if (this->locally_relevant_dofs.is_element(dof))
            if (!relevant_boundary_dofs.is_element(dof))
              lambda_constraints.constrain_dof_to_zero(dof);
      }
    }
  }
  lambda_constraints.close();

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
  {
    // Print number of owned and constrained lambda dofs
    IndexSet lambda_dofs =
      DoFTools::extract_dofs(this->dof_handler, lambda_mask);
    unsigned int constrained_owned_dofs   = 0;
    unsigned int unconstrained_owned_dofs = 0;
    for (const auto &dof : lambda_dofs)
    {
      if (!lambda_constraints.is_constrained(dof))
        unconstrained_owned_dofs++;
      else
        constrained_owned_dofs++;
    }

    const unsigned int total_constrained_owned_dofs =
      Utilities::MPI::sum(constrained_owned_dofs, this->mpi_communicator);
    std::cout << total_constrained_owned_dofs
              << " constrained owned lambda dofs" << std::endl;
    const unsigned int total_unconstrained_owned_dofs =
      Utilities::MPI::sum(unconstrained_owned_dofs, this->mpi_communicator);
    std::cout << total_unconstrained_owned_dofs
              << " unconstrained owned lambda dofs" << std::endl;

    {
      // Print constrainted lambda dofs
      std::map<types::global_dof_index, Point<dim>> support_points =
        DoFTools::map_dofs_to_support_points(mapping_collection,
                                             this->dof_handler);

      std::ofstream outfile(this->param.output.output_dir +
                            "constrained_lambda_dofs_proc" +
                            std::to_string(this->mpi_rank) + ".pos");
      outfile << "View \"constrained_lambda_dofs_proc" << this->mpi_rank
              << "\"{" << std::endl;
      for (const auto dof : lambda_dofs)
        if (lambda_constraints.is_constrained(dof))
        {
          const Point<dim> &pt = support_points.at(dof);
          if constexpr (dim == 2)
            outfile << "SP(" << pt[0] << "," << pt[1] << ", 0.){1};"
                    << std::endl;
          else
            outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
                    << std::endl;
        }
      outfile << "};" << std::endl;
      outfile.close();
    }
  }
}

template <int dim>
void NSSolverLambda<dim>::create_hp_line_dof_identities()
{
  TimerOutput::Scope t(this->computing_timer, "Hp line identities");

  const unsigned int deg_p   = this->param.finite_elements.pressure_degree;
  const unsigned int u_lower = this->ordering->u_lower;
  std::vector<std::set<types::global_dof_index>> velocity_dofs_to_match(dim);
  std::set<types::global_dof_index>              pressure_dofs_to_match;

  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(mapping_collection, this->dof_handler);

  /**
   * If cell has lambda FESystem, accumulate its dofs,
   * then match them with the adjacent cells without lambda.
   */
  {
    const auto &fe_lambda = (*fe)[index_fe_with_lambda];
    std::vector<types::global_dof_index> dof_indices(
      fe_lambda.n_dofs_per_cell());
    for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (!cell_has_lambda(cell))
        continue;
      cell->get_dof_indices(dof_indices);

      // Simply grab all the velocity and pressure dofs on the lambda cells,
      // without distinguishing if they're associated to vertices, lines, faces
      // or interiors. The hp dof identities that were correctly applied already
      // matched the global dof index of some of these dofs.
      // Then we'll see if the adjacent dofs at the same support point have
      // a different global dof index or not, and if so match their values as a
      // constraint.
      for (unsigned int i = 0; i < fe_lambda.n_dofs_per_cell(); ++i)
      {
        const unsigned int comp = fe_lambda.system_to_component_index(i).first;
        if (this->ordering->is_velocity(comp))
          velocity_dofs_to_match[comp - u_lower].insert(dof_indices[i]);
        // If pressure is linear, there is no line dof to match
        if (deg_p > 1 && this->ordering->is_pressure(comp))
          pressure_dofs_to_match.insert(dof_indices[i]);
      }
    }
  }

  // Safety checks
  if constexpr (running_in_debug_mode())
  {
    std::vector<std::vector<types::global_dof_index>> vec_velocity_dofs(dim);
    for (unsigned int d = 0; d < dim; ++d)
    {
      vec_velocity_dofs[d] =
        std::vector<types::global_dof_index>(velocity_dofs_to_match.begin(),
                                             velocity_dofs_to_match.end());
      Assert(vec_velocity_dofs[0].size() == vec_velocity_dofs[d].size(),
             ExcInternalError());
    }
    for (unsigned int i = 0; i < vec_velocity_dofs[0].size(); ++i)
      // FIXME: This assumes that velocity dofs for u,v,w are contiguous
      for (unsigned int d = 1; d < dim; ++d)
        Assert(vec_velocity_dofs[d][i] == vec_velocity_dofs[d - 1][i] + 1,
               ExcInternalError());
  }

  hp_dof_identities.reserve(pressure_dofs_to_match.size() +
                            dim * velocity_dofs_to_match[0].size());

  /**
   * Then loop over non-lambda cells, and if one of their neighbouring cell has
   * lambda dofs, match the dofs of same component located at the same support
   * point.
   */
  {
    const auto &fe_without_lambda = (*fe)[index_fe_without_lambda];
    std::vector<types::global_dof_index> dof_indices(
      fe_without_lambda.n_dofs_per_cell());
    for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      if (cell_has_lambda(cell))
        continue;

      bool has_neighbor_with_lambda = false;

      for (const auto i_face : cell->face_indices())
      {
        // Check if neighbouring cell is valid and has lambda defined
        auto neighbor = cell->neighbor(i_face);
        if (neighbor->state() != IteratorState::IteratorStates::valid)
          continue;
        if (!cell_has_lambda(cell->neighbor(i_face)))
          continue;
        has_neighbor_with_lambda = true;
        break;
      }

      if (has_neighbor_with_lambda)
      {
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < fe_without_lambda.n_dofs_per_cell(); ++i)
        {
          const auto        &pt_i = support_points.at(dof_indices[i]);
          const unsigned int comp =
            fe_without_lambda.system_to_component_index(i).first;

          if (this->ordering->is_velocity(comp))
            for (const auto dof : velocity_dofs_to_match[comp - u_lower])
              if (dof != dof_indices[i])
                if (support_points.at(dof).distance_square(pt_i) < 1e-15)
                {
                  hp_dof_identities.push_back({dof, dof_indices[i]});
                  break;
                }
          if (this->ordering->is_pressure(comp))
            for (const auto dof : pressure_dofs_to_match)
              if (dof != dof_indices[i])
                if (support_points.at(dof).distance_square(pt_i) < 1e-15)
                {
                  hp_dof_identities.push_back({dof, dof_indices[i]});
                  break;
                }
        }
      }
    }
  }
}

template <int dim>
void NSSolverLambda<dim>::remove_cylinder_velocity_constraints(
  AffineConstraints<double> &constraints) const
{
  if (weak_no_slip_boundary_id == numbers::invalid_unsigned_int)
    return;

  IndexSet relevant_boundary_velocity_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->velocity_mask,
                                    {weak_no_slip_boundary_id});
  /**
   * There is a tricky corner case that happens when a partition has ghost dofs
   * on a boundary edge, but the faces sharing this edge do not belong to this
   * boundary (for instance, tets making an angle, and the tet whose face is on
   * the boundary belongs to another rank). In that case, the ghost dofs on the
   * boundary are not collected with DoFTools::extract_boundary_dofs, since the
   * ghost faces are simply not on the given boundary.
   *
   * We have to exchange the boundary dofs, and add the missing ghost ones from
   * other ranks.
   */
  {
    std::vector<std::vector<types::global_dof_index>> gathered_vel_bdr_dofs =
      Utilities::MPI::all_gather(
        this->mpi_communicator,
        relevant_boundary_velocity_dofs.get_index_vector());
    for (const auto &vec : gathered_vel_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          relevant_boundary_velocity_dofs.add_index(dof);
  }

  if constexpr (running_in_debug_mode())
  {
    // Check consistency of constraints for RELEVANT (not active) dofs before
    // removing
    {
      const bool consistent = constraints.is_consistent_in_parallel(
        Utilities::MPI::all_gather(this->mpi_communicator,
                                   this->locally_owned_dofs),
        this->locally_relevant_dofs,
        // DoFTools::extract_locally_active_dofs(this->dof_handler),
        this->mpi_communicator,
        true);
      Assert(consistent,
             ExcMessage("Constraints are not consistent before removing"));
    }
  }

  /**
   * Now actually remove the constraints
   */
  {
    AffineConstraints<double> filtered;
    filtered.reinit(this->locally_owned_dofs, this->locally_relevant_dofs);

    for (const auto &line : constraints.get_lines())
    {
      if (relevant_boundary_velocity_dofs.is_element(line.index))
        continue;

      filtered.add_constraint(line.index, line.entries, line.inhomogeneity);

      // Check that entries do not involve an absent velocity dof
      // With the get_view() function, this is done automatically
      for (const auto &entry : line.entries)
      {
        AssertThrow(!relevant_boundary_velocity_dofs.is_element(entry.first),
                    ExcMessage("Constraint involves a cylinder velocity dof"));
      }
    }

    filtered.close();
    constraints.clear();
    constraints = std::move(filtered);
  }

  if constexpr (running_in_debug_mode())
  {
    // Check consistency of constraints for RELEVANT (not active) dofs after
    // removing
    {
      const bool consistent = constraints.is_consistent_in_parallel(
        Utilities::MPI::all_gather(this->mpi_communicator,
                                   this->locally_owned_dofs),
        this->locally_relevant_dofs,
        // DoFTools::extract_locally_active_dofs(this->dof_handler),
        this->mpi_communicator,
        true);
      Assert(consistent,
             ExcMessage("Constraints are not consistent after removing"));
    }

    // Check that boundary dofs were correctly removed
    for (const auto &dof : relevant_boundary_velocity_dofs)
      Assert(!constraints.is_constrained(dof),
             ExcMessage(
               "On rank " + std::to_string(this->mpi_rank) +
               " : "
               "Velocity dof " +
               std::to_string(dof) +
               " on a boundary with weak no-slip remains "
               "constrained by a boundary condition. This can happen if "
               "velocity dofs lying on both the cylinder and a face "
               "boundary have conflicting prescribed boundary conditions."));
  }
}

template <int dim>
void NSSolverLambda<dim>::add_hp_identities_constraints(
  AffineConstraints<double> &constraints) const
{
  AffineConstraints<double> hp_constraints(this->locally_owned_dofs,
                                           this->locally_relevant_dofs);

  // Apply dof_1 = dof_2 to each pair of identified dofs
  for (const auto &[dof1, dof2] : hp_dof_identities)
    hp_constraints.add_constraint(dof1, {{dof2, 1.}}, 0.);
  hp_constraints.close();

  /**
   * The current constraints may already constrain some boundary dofs, which
   * also need to be identified to an adjacent cell dof to maintain continuity.
   * Continuity takes precedence, so the hp_constraints wins if there is a clash
   * between cosntraints. The other constrain will be applied anyways to the
   * identified dof.
   */
  constraints.merge(
    hp_constraints,
    AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
}

template <int dim>
void NSSolverLambda<dim>::create_solver_specific_zero_constraints()
{
  this->zero_constraints.close();

  // Merge the zero lambda constraints
  this->zero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
    /** FIXME: Instead of dim = 3, the test should be whether dofs
     * belong to multiple boundaries, but for now this only happens for the
     * 3D fsi test case.
     */
    if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->zero_constraints);
    }

  // Add the hp dof identities as constraints
  // This won't be required as soon as the line dof identities are applied in
  // deal.II (in dof_handler.distribute_dofs())
  add_hp_identities_constraints(this->zero_constraints);
}

template <int dim>
void NSSolverLambda<dim>::create_solver_specific_nonzero_constraints()
{
  this->nonzero_constraints.close();

  // Merge the zero lambda constraints
  this->nonzero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
    if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->nonzero_constraints);
    }

  // Add the hp dof identities as constraints
  // This won't be required as soon as the line dof identities are applied in
  // deal.II (in dof_handler.distribute_dofs())
  add_hp_identities_constraints(this->nonzero_constraints);
}

template <int dim>
void NSSolverLambda<dim>::create_sparsity_pattern()
{
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);

  const unsigned int           n_components = this->ordering->n_components;
  Table<2, DoFTools::Coupling> coupling(n_components, n_components);
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int d = 0; d < n_components; ++d)
    {
      coupling[c][d] = DoFTools::none;

      // u couples to all variables
      if (this->ordering->is_velocity(c))
        coupling[c][d] = DoFTools::always;

      // p couples to u
      if (this->ordering->is_pressure(c))
        if (this->ordering->is_velocity(d))
          coupling[c][d] = DoFTools::always;

      // lambda couples to u
      if (this->ordering->is_lambda(c))
        if (this->ordering->is_velocity(d))
          coupling[c][d] = DoFTools::always;
    }
  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling,
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
}

template <int dim>
void NSSolverLambda<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           mapping_collection,
                           quadrature_collection,
                           face_quadrature_collection,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(*fe);

  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &NSSolverLambda::assemble_local_matrix,
                  &NSSolverLambda::copy_local_to_global_matrix,
                  scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim>
void NSSolverLambda<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell,
                     this->evaluation_point,
                     this->previous_solutions,
                     this->source_terms,
                     this->exact_solution);

  const unsigned int fe_index    = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto &local_matrix             = copy_data.matrices[fe_index];
  auto &local_dof_indices        = copy_data.local_dof_indices[fe_index];

  local_matrix = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
  {
    const double JxW_moving = scratchData.JxW_moving[q];

    const auto &phi_u      = scratchData.phi_u[q];
    const auto &grad_phi_u = scratchData.grad_phi_u[q];
    const auto &div_phi_u  = scratchData.div_phi_u[q];
    const auto &phi_p      = scratchData.phi_p[q];

    const auto &present_velocity_values =
      scratchData.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratchData.present_velocity_gradients[q];

    for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
    {
      const unsigned int comp_i = scratchData.components[i];
      const bool         i_is_u = this->ordering->is_velocity(comp_i);
      const bool         i_is_p = this->ordering->is_pressure(comp_i);

      for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      {
        const unsigned int comp_j = scratchData.components[j];
        const bool         j_is_u = this->ordering->is_velocity(comp_j);
        const bool         j_is_p = this->ordering->is_pressure(comp_j);

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
          local_matrix_ij += nu * scalar_product(grad_phi_u[j], grad_phi_u[i]);
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
          local_matrix_ij *= JxW_moving;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }
  }

  //
  // Face contributions (Lagrange multiplier)
  //
  if (cell->at_boundary() && cell_has_lambda(cell))
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
          const auto &phi_l = scratchData.phi_l_face[i_face][q];

          for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
          {
            const unsigned int comp_i = scratchData.components[i];
            const bool         i_is_u = this->ordering->is_velocity(comp_i);
            const bool         i_is_l = this->ordering->is_lambda(comp_i);

            for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = scratchData.components[j];
              const bool         j_is_u = this->ordering->is_velocity(comp_j);
              const bool         j_is_l = this->ordering->is_lambda(comp_j);

              double local_matrix_ij = 0.;

              if (i_is_u && j_is_l)
                local_matrix_ij += -(phi_l[j] * phi_u[i]);

              if (i_is_l && j_is_u)
                local_matrix_ij += -phi_u[j] * phi_l[i];

              local_matrix_ij *= face_JxW_moving;
              local_matrix(i, j) += local_matrix_ij;
            }
          }
        }
      }
    }
  }
  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void NSSolverLambda<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  const auto i = copy_data.last_active_fe_index;
  this->zero_constraints.distribute_local_to_global(
    copy_data.matrices[i], copy_data.local_dof_indices[i], this->system_matrix);
}

template <int dim>
void NSSolverLambda<dim>::compare_analytical_matrix_with_fd()
{
  // ScratchData scratch_data(*this->ordering,
  //                                  *fe,
  //                                  *this->fixed_mapping,
  //                                  *this->moving_mapping,
  //                                  *this->quadrature,
  //                                  *this->face_quadrature,
  //                                  this->time_handler.bdf_coefficients,
  //                                  this->param);
  // CopyData            copy_data(fe->n_dofs_per_cell());

  // auto errors = Verification::compare_analytical_matrix_with_fd(
  //   this->dof_handler,
  //   fe->n_dofs_per_cell(),
  //   *this,
  //   &NSSolverLambda::assemble_local_matrix,
  //   &NSSolverLambda::assemble_local_rhs,
  //   scratch_data,
  //   copy_data,
  //   this->present_solution,
  //   this->evaluation_point,
  //   this->local_evaluation_point,
  //   this->mpi_communicator,
  //   this->param.output.output_dir,
  //   true,
  //   this->param.debug.analytical_jacobian_absolute_tolerance,
  //   this->param.debug.analytical_jacobian_relative_tolerance);

  // this->pcout << "Max absolute error analytical vs fd matrix is "
  //             << errors.first << std::endl;

  // // Only print relative error if absolute is too large
  // if (errors.first >
  // this->param.debug.analytical_jacobian_absolute_tolerance)
  //   this->pcout << "Max relative error analytical vs fd matrix is "
  //               << errors.second << std::endl;
}

template <int dim>
void NSSolverLambda<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           mapping_collection,
                           quadrature_collection,
                           face_quadrature_collection,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(*fe);

  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &NSSolverLambda::assemble_local_rhs,
                  &NSSolverLambda::copy_local_to_global_rhs,
                  scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void NSSolverLambda<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(cell,
                     this->evaluation_point,
                     this->previous_solutions,
                     this->source_terms,
                     this->exact_solution);

  const unsigned int fe_index    = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto &local_rhs                = copy_data.vectors[fe_index];
  auto &local_dof_indices        = copy_data.local_dof_indices[fe_index];

  local_rhs = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

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
    const auto  &source_term_velocity = scratchData.source_term_velocity[q];
    const auto  &source_term_pressure = scratchData.source_term_pressure[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
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
        + phi_u[i] * source_term_velocity

        // Continuity
        - present_velocity_divergence * phi_p[i]

        // Pressure source term
        + source_term_pressure * phi_p[i]);

      local_rhs_i *= JxW_moving;
      local_rhs(i) += local_rhs_i;
    }
  }

  //
  // Face contributions
  //
  if (cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary())
      {
        //
        // Lagrange multiplier for no-slip
        //
        if (cell_has_lambda(cell))
          if (face->boundary_id() == weak_no_slip_boundary_id)
            for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
            {
              //
              // Flow related data (no-slip)
              //
              const double face_JxW_moving =
                scratchData.face_JxW_moving[i_face][q];
              const auto &phi_u = scratchData.phi_u_face[i_face][q];
              const auto &phi_l = scratchData.phi_l_face[i_face][q];

              const auto &present_u =
                scratchData.present_face_velocity_values[i_face][q];
              const auto &present_l =
                scratchData.present_face_lambda_values[i_face][q];

              for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
              {
                double local_rhs_i = 0.;

                const unsigned int comp_i = scratchData.components[i];
                const bool         i_is_u = this->ordering->is_velocity(comp_i);
                const bool         i_is_l = this->ordering->is_lambda(comp_i);

                if (i_is_u)
                  local_rhs_i -= -phi_u[i] * present_l;

                if (i_is_l)
                  local_rhs_i -= -present_u * phi_l[i];

                local_rhs_i *= face_JxW_moving;
                local_rhs(i) += local_rhs_i;
              }
            }

        /**
         * Open boundary condition with prescribed manufactured solution.
         * Applied on moving mesh.
         */
        if (this->param.fluid_bc.at(scratchData.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
          for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
          {
            const double face_JxW_moving =
              scratchData.face_JxW_moving[i_face][q];
            const auto &n = scratchData.face_normals_moving[i_face][q];

            const auto &grad_u_exact =
              scratchData.exact_face_velocity_gradients[i_face][q];
            const double p_exact =
              scratchData.exact_face_pressure_values[i_face][q];

            // This is an open boundary condition, not a traction,
            // involving only grad_u_exact and not the symmetric gradient.
            const auto quasisigma_dot_n = -p_exact * n + nu * grad_u_exact * n;

            const auto &phi_u = scratchData.phi_u_face[i_face][q];

            for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
              local_rhs(i) -= -phi_u[i] * quasisigma_dot_n * face_JxW_moving;
          }
      }
    }
  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void NSSolverLambda<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  const auto i = copy_data.last_active_fe_index;
  this->zero_constraints.distribute_local_to_global(
    copy_data.vectors[i], copy_data.local_dof_indices[i], this->system_rhs);
}

template <int dim>
void NSSolverLambda<dim>::check_velocity_boundary() const
{
  IndexSet relevant_boundary_velocity_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->velocity_mask,
                                    {weak_no_slip_boundary_id});
  double local_max_boundary_velocity = 0.;
  for (auto dof : relevant_boundary_velocity_dofs)
    local_max_boundary_velocity =
      std::max(local_max_boundary_velocity,
               std::abs(this->present_solution[dof]));

  const double max_boundary_velocity =
    Utilities::MPI::max(local_max_boundary_velocity, this->mpi_communicator);

  if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  {
    this->pcout << "Checking no-slip enforcement:" << std::endl;
    this->pcout << "max velocity on boundary = " << max_boundary_velocity
                << std::endl;
  }

  AssertThrow(max_boundary_velocity < 1e-12,
              ExcMessage("Velocity on weak no-slip boundary is too large : " +
                         std::to_string(max_boundary_velocity)));
}

template <int dim>
void NSSolverLambda<dim>::compute_lambda_error_on_boundary(
  double         &lambda_l2_error,
  double         &lambda_linf_error,
  Tensor<1, dim> &error_on_integral)
{
  // double lambda_l2_local   = 0;
  // double lambda_linf_local = 0;

  // Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local,
  //   exact_integral_local;
  // lambda_integral_local = 0;
  // exact_integral_local  = 0;

  // const double rho = this->param.physical_properties.fluids[0].density;
  // const double nu =
  //   this->param.physical_properties.fluids[0].kinematic_viscosity;
  // const double mu = nu * rho;

  // FEFaceValues<dim> fe_face_values(*this->moving_mapping,
  //                                  *fe,
  //                                  *this->face_quadrature,
  //                                  update_values | update_quadrature_points |
  //                                    update_JxW_values | update_normal_vectors);

  // const unsigned int          n_faces_q_points =
  // this->face_quadrature->size(); std::vector<Tensor<1, dim>>
  // lambda_values(n_faces_q_points); Tensor<1, dim>              diff, exact;

  // // std::ofstream out("normals.pos");
  // // out << "View \"normals\" {\n";

  // for (auto cell : this->dof_handler.active_cell_iterators())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;

  //   for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  //   {
  //     const auto &face = cell->face(i_face);

  //     if (face->at_boundary() &&
  //         face->boundary_id() == weak_no_slip_boundary_id)
  //     {
  //       fe_face_values.reinit(cell, i_face);

  //       // Get FE solution values on the face
  //       fe_face_values[lambda_extractor].get_function_values(
  //         this->present_solution, lambda_values);

  //       // Evaluate exact solution at quadrature points
  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         const Point<dim> &qpoint         =
  //         fe_face_values.quadrature_point(q); const auto normal_to_mesh =
  //         fe_face_values.normal_vector(q); const auto        normal_to_solid
  //         = -normal_to_mesh;

  //         // Careful:
  //         // int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
  //         //                                                   =
  //         //                                             normal_to_mesh
  //         //                                                   =
  //         //                                            -normal_to_solid
  //         //
  //         // Got to take the consistent normal to compare int lambda_h with
  //         // solution.
  //         //
  //         // Solution<dim> computes lambda_exact = - sigma cdot ns, where n
  //         is
  //         // expected to be the normal to the SOLID.

  //         // out << "VP(" << qpoint[0] << "," << qpoint[1] << "," << 0. <<
  //         "){"
  //         //   << normal[0] << "," << normal[1] << "," << 0. << "};\n";

  //         // exact_solution is a pointer to base class Function<dim>,
  //         // so we have to ruse to use the specific function for lambda.
  //         std::static_pointer_cast<NSSolverLambda<dim>::MMSSolution>(
  //           this->exact_solution)
  //           ->lagrange_multiplier(qpoint, mu, normal_to_solid, exact);

  //         diff = lambda_values[q] - exact;

  //         lambda_l2_local += diff * diff * fe_face_values.JxW(q);
  //         lambda_linf_local =
  //           std::max(lambda_linf_local, std::abs(diff.norm()));

  //         // Increment the integral of lambda
  //         lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
  //         exact_integral_local += exact * fe_face_values.JxW(q);
  //       }
  //     }
  //   }
  // }

  // // out << "};\n";
  // // out.close();

  // lambda_l2_error =
  //   Utilities::MPI::sum(lambda_l2_local, this->mpi_communicator);
  // lambda_l2_error = std::sqrt(lambda_l2_error);

  // lambda_linf_error =
  //   Utilities::MPI::max(lambda_linf_local, this->mpi_communicator);

  // for (unsigned int d = 0; d < dim; ++d)
  // {
  //   lambda_integral[d] =
  //     Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
  //   exact_integral[d] =
  //     Utilities::MPI::sum(exact_integral_local[d], this->mpi_communicator);
  //   error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
  // }
}

template <int dim>
void NSSolverLambda<dim>::compute_solver_specific_errors()
{
  // double         l2_l = 0., li_l = 0.;
  // Tensor<1, dim> error_on_integral;
  // this->compute_lambda_error_on_boundary(l2_l, li_l, error_on_integral);
  // // linf_error_Fx = std::max(linf_error_Fx, error_on_integral[0]);
  // // linf_error_Fy = std::max(linf_error_Fy, error_on_integral[1]);

  // const double t = this->time_handler.current_time;
  // for (auto &[norm, handler] : this->error_handlers)
  // {
  //   if (norm == VectorTools::L2_norm)
  //     handler->add_error("l", l2_l, t);
  //   if (norm == VectorTools::Linfty_norm)
  //     handler->add_error("l", li_l, t);
  // }
}

template <int dim>
void NSSolverLambda<dim>::output_results()
{
  if (this->param.output.write_results)
  {
    //
    // Plot FE solution
    //
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("lambda");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(this->present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    //
    // Partition
    //
    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*this->moving_mapping, 2);
    const std::string pvtu_file = data_out.write_vtu_with_pvtu_record(
      this->param.output.output_dir,
      this->param.output.output_prefix,
      this->time_handler.current_time_iteration,
      this->mpi_communicator,
      2);

    this->visualization_times_and_names.emplace_back(
      this->time_handler.current_time, pvtu_file);
  }
}

template <int dim>
void NSSolverLambda<dim>::compute_forces(const bool export_table)
{
  // Tensor<1, dim> lambda_integral, lambda_integral_local;
  // lambda_integral_local = 0;

  // FEFaceValues<dim> fe_face_values(*this->moving_mapping,
  //                                  *fe,
  //                                  *this->face_quadrature,
  //                                  update_values | update_quadrature_points |
  //                                    update_JxW_values | update_normal_vectors);

  // const unsigned int          n_faces_q_points =
  // this->face_quadrature->size(); std::vector<Tensor<1, dim>>
  // lambda_values(n_faces_q_points);

  // for (auto cell : this->dof_handler.active_cell_iterators())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;

  //   for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  //   {
  //     const auto &face = cell->face(i_face);

  //     if (face->at_boundary() &&
  //         face->boundary_id() == weak_no_slip_boundary_id)
  //     {
  //       fe_face_values.reinit(cell, i_face);

  //       // Get FE solution values on the face
  //       fe_face_values[lambda_extractor].get_function_values(
  //         this->present_solution, lambda_values);

  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //         lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
  //     }
  //   }
  // }

  // for (unsigned int d = 0; d < dim; ++d)
  //   lambda_integral[d] =
  //     Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);

  // // const double rho = param.physical_properties.fluids[0].density;
  // // const double U   = boundary_description.U;
  // // const double D   = boundary_description.D;
  // // const double factor = 1. / (0.5 * rho * U * U * D);

  // //
  // // Forces on the cylinder are the NEGATIVE of the integral of lambda
  // //
  // this->forces_table.add_value("time", this->time_handler.current_time);
  // this->forces_table.add_value("CFx", -lambda_integral[0]);
  // this->forces_table.add_value("CFy", -lambda_integral[1]);
  // if constexpr (dim == 3)
  // {
  //   this->forces_table.add_value("CFz", -lambda_integral[2]);
  // }

  // if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
  //   this->pcout << "Computed forces: " << -lambda_integral << std::endl;

  // if (export_table && this->param.output.write_results && this->mpi_rank ==
  // 0)
  // {
  //   std::ofstream outfile(this->param.output.output_dir + "forces.txt");
  //   this->forces_table.write_text(outfile);
  // }
}

template <int dim>
void NSSolverLambda<dim>::solver_specific_post_processing()
{
  /**
   * Check that no-slip condition is satisfied.
   *
   * When applying the exact solution, the fluid velocity will be exact,
   * but the mesh velocity is only precise up to time integration order.
   * So these velocities differ by some power of the time step, rather
   * than the machine epsilon as checked in this function, thus the
   * no-slip is not checked in this case.
   *
   * Also, not checking when using BDF2 and starting with the initial
   * condition, as it will generally not respect the no-slip condition.
   */
  if (!this->param.debug.apply_exact_solution)
  {
    if (!(this->time_handler.is_starting_step() &&
          this->param.time_integration.bdfstart ==
            Parameters::TimeIntegration::BDFStart::initial_condition))
      check_velocity_boundary();
  }

  const bool export_force_table =
    this->time_handler.is_steady() ||
    ((this->time_handler.current_time_iteration % 5) == 0);
  compute_forces(export_force_table);
}

// Explicit instantiation
template class NSSolverLambda<2>;
template class NSSolverLambda<3>;