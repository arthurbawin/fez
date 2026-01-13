
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
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_ns_solver_lambda.h>
#include <linear_solver.h>
#include <mesh.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
NSSolverLambda<dim>::NSSolverLambda(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim>(param, /* moving_mesh = */ false)
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
    fe_with_lambda = std::make_shared<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim,
      FE_SimplexP<dim>(param.finite_elements.pressure_degree),
      FE_SimplexP<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim);
    fe_without_lambda = std::make_shared<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim,
      FE_SimplexP<dim>(param.finite_elements.pressure_degree),
      FE_Nothing<dim>() ^ dim);
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
      outfile << "View \"constrained_lambda_dofs_proc" << this->mpi_rank <<
      "\"{"
              << std::endl;
      for (const auto dof : lambda_dofs)
        if (lambda_constraints.is_constrained(dof))
        {
          const Point<dim> &pt = support_points.at(dof);
          outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
                  << std::endl;
        }
      outfile << "};" << std::endl;
      outfile.close();
    }
  }
}

template <int dim>
void NSSolverLambda<dim>::remove_cylinder_velocity_constraints(
  AffineConstraints<double> &constraints,
  const bool                 remove_velocity_constraints,
  const bool                 remove_position_constraints) const
{
  //   if (weak_no_slip_boundary_id == numbers::invalid_unsigned_int)
  //     return;

  //   IndexSet relevant_boundary_velocity_dofs =
  //     DoFTools::extract_boundary_dofs(this->dof_handler,
  //                                     this->velocity_mask,
  //                                     {weak_no_slip_boundary_id});
  //   IndexSet relevant_boundary_position_dofs =
  //     DoFTools::extract_boundary_dofs(this->dof_handler,
  //                                     this->position_mask,
  //                                     {weak_no_slip_boundary_id});

  //   /**
  //    * There is a tricky corner case that happens when a partition has ghost
  //    dofs
  //    * on a boundary edge, but the faces sharing this edge do not belong to
  //    this
  //    * boundary (for instance, tets making an angle, and the tet whose face
  //    is on
  //    * the boundary belongs to another rank). In that case, the ghost dofs on
  //    the
  //    * boundary are not collected with DoFTools::extract_boundary_dofs, since
  //    the
  //    * ghost faces are simply not on the given boundary.
  //    *
  //    * We have to exchange the boundary dofs, and add the missing ghost ones
  //    from
  //    * other ranks.
  //    */
  //   {
  //     std::vector<std::vector<types::global_dof_index>> gathered_vel_bdr_dofs
  //     =
  //       Utilities::MPI::all_gather(
  //         this->mpi_communicator,
  //         relevant_boundary_velocity_dofs.get_index_vector());
  //     std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs
  //     =
  //       Utilities::MPI::all_gather(
  //         this->mpi_communicator,
  //         relevant_boundary_position_dofs.get_index_vector());

  //     for (const auto &vec : gathered_vel_bdr_dofs)
  //       for (const auto dof : vec)
  //         if (this->locally_relevant_dofs.is_element(dof))
  //           relevant_boundary_velocity_dofs.add_index(dof);
  //     for (const auto &vec : gathered_pos_bdr_dofs)
  //       for (const auto dof : vec)
  //         if (this->locally_relevant_dofs.is_element(dof))
  //           relevant_boundary_position_dofs.add_index(dof);
  //   }

  //   // Check consistency of constraints for RELEVANT (not active) dofs before
  //   // removing
  //   {
  //     const bool consistent = constraints.is_consistent_in_parallel(
  //       Utilities::MPI::all_gather(this->mpi_communicator,
  //                                  this->locally_owned_dofs),
  //       // this->locally_relevant_dofs,
  //       DoFTools::extract_locally_active_dofs(this->dof_handler),
  //       this->mpi_communicator,
  //       true);
  //     AssertThrow(consistent,
  //                 ExcMessage("Constraints are not consistent before
  //                 removing"));
  //   }

  //   /**
  //    * Now actually remove the constraints
  //    */
  //   {
  //     AffineConstraints<double> filtered;
  //     filtered.reinit(this->locally_owned_dofs, this->locally_relevant_dofs);

  //     for (const auto &line : constraints.get_lines())
  //     {
  //       if (remove_velocity_constraints &&
  //           relevant_boundary_velocity_dofs.is_element(line.index))
  //         continue;
  //       if (remove_position_constraints &&
  //           relevant_boundary_position_dofs.is_element(line.index))
  //         continue;

  //       filtered.add_constraint(line.index, line.entries,
  //       line.inhomogeneity);

  //       // Check that entries do not involve an absent velocity dof
  //       // With the get_view() function, this is done automatically
  //       for (const auto &entry : line.entries)
  //       {
  //         if (remove_velocity_constraints)
  //           AssertThrow(!relevant_boundary_velocity_dofs.is_element(entry.first),
  //                       ExcMessage(
  //                         "Constraint involves a cylinder velocity dof"));
  //         if (remove_position_constraints)
  //           AssertThrow(!relevant_boundary_position_dofs.is_element(entry.first),
  //                       ExcMessage(
  //                         "Constraint involves a cylinder position dof"));
  //       }
  //     }

  //     filtered.close();
  //     constraints.clear();
  //     constraints = std::move(filtered);
  //   }

  //   // Check consistency of constraints for RELEVANT (not active) dofs after
  //   // removing
  //   {
  //     const bool consistent = constraints.is_consistent_in_parallel(
  //       Utilities::MPI::all_gather(this->mpi_communicator,
  //                                  this->locally_owned_dofs),
  //       // this->locally_relevant_dofs,
  //       DoFTools::extract_locally_active_dofs(this->dof_handler),
  //       this->mpi_communicator,
  //       true);
  //     AssertThrow(consistent,
  //                 ExcMessage("Constraints are not consistent after
  //                 removing"));
  //   }

  //   // Check that boundary dofs were correctly removed
  //   if (remove_velocity_constraints)
  //     for (const auto &dof : relevant_boundary_velocity_dofs)
  //       AssertThrow(
  //         !constraints.is_constrained(dof),
  //         ExcMessage(
  //           "On rank " + std::to_string(this->mpi_rank) +
  //           " : "
  //           "Velocity dof " +
  //           std::to_string(dof) +
  //           " on a boundary with weak no-slip remains "
  //           "constrained by a boundary condition. This can happen if "
  //           "velocity dofs lying on both the cylinder and a face "
  //           "boundary have conflicting prescribed boundary conditions."));
  //   if (remove_position_constraints)
  //     for (const auto &dof : relevant_boundary_position_dofs)
  //       AssertThrow(
  //         !constraints.is_constrained(dof),
  //         ExcMessage(
  //           "On rank " + std::to_string(this->mpi_rank) +
  //           " : "
  //           "Position dof " +
  //           std::to_string(dof) +
  //           " on a boundary with weak no-slip remains "
  //           "constrained by a boundary condition. This can happen if "
  //           "position dofs lying on both the cylinder and a face "
  //           "boundary have conflicting prescribed boundary conditions."));
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
      remove_cylinder_velocity_constraints(this->zero_constraints, true, false);
    }
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
      remove_cylinder_velocity_constraints(this->nonzero_constraints,
                                           true,
                                           false);
    }
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
  CopyData copy_data(*fe);

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

  const unsigned int fe_index          = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto &local_matrix = copy_data.matrices[fe_index];
  auto              &local_dof_indices = copy_data.local_dof_indices[fe_index];

  local_matrix       = 0;

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

        bool   assemble             = false;
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
            nu * scalar_product(grad_phi_u[j], grad_phi_u[i]);
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
          const double face_JxW_moving =
          scratchData.face_JxW_moving[i_face][q];

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
      copy_data.matrices[i],
      copy_data.local_dof_indices[i],
      this->system_matrix);
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
  CopyData copy_data(*fe);

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

  const unsigned int fe_index          = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto              &local_rhs         = copy_data.vectors[fe_index];
  auto              &local_dof_indices = copy_data.local_dof_indices[fe_index];

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

/**
 * Compute integral of lambda (fluid force), compare to position dofs
 */
template <int dim>
void NSSolverLambda<dim>::compare_forces_and_position_on_obstacle() const
{
  // Tensor<1, dim> lambda_integral, lambda_integral_local;
  // lambda_integral_local = 0;

  // FEFaceValues<dim> fe_face_values(*this->moving_mapping,
  //                                  *fe,
  //                                  *this->face_quadrature,
  //                                  update_values | update_JxW_values);

  // // Compute integral of lambda on owned boundary
  // const unsigned int n_faces_q_points = this->face_quadrature->size();
  // std::vector<types::global_dof_index> face_dofs(fe->n_dofs_per_face());

  // std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  // Tensor<1, dim>    cylinder_displacement_local, max_diff_local;
  // std::vector<bool> first_computed_displacement(dim, true);

  // for (auto cell : this->dof_handler.active_cell_iterators())
  //   if (cell->is_locally_owned() && cell->at_boundary())
  //     for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  //     {
  //       const auto &face = cell->face(i_face);
  //       if (face->at_boundary() &&
  //           face->boundary_id() == weak_no_slip_boundary_id)
  //       {
  //         fe_face_values.reinit(cell, i_face);

  //         // Increment lambda integral
  //         fe_face_values[lambda_extractor].get_function_values(
  //           this->present_solution, lambda_values);
  //         for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //           lambda_integral_local += lambda_values[q] *
  //           fe_face_values.JxW(q);

  //         /**
  //          * Cylinder is rigid, so all displacements should be identical for
  //          a
  //          * given component. If first position dof, save displacement,
  //          * otherwise compare with saved displacement.
  //          */
  //         face->get_dof_indices(face_dofs);

  //         for (unsigned int i_dof = 0; i_dof < fe->n_dofs_per_face();
  //         ++i_dof)
  //           if (this->locally_owned_dofs.is_element(face_dofs[i_dof]))
  //           {
  //             const unsigned int comp =
  //               fe->face_system_to_component_index(i_dof, i_face).first;
  //             if (this->ordering->is_position(comp))
  //             {
  //               const unsigned int d = comp - this->ordering->x_lower;

  //               if (first_computed_displacement[d])
  //               {
  //                 // Save displacement
  //                 first_computed_displacement[d] = false;
  //                 cylinder_displacement_local[d] =
  //                   this->present_solution[face_dofs[i_dof]] -
  //                   this->initial_positions.at(face_dofs[i_dof])[d];
  //               }
  //               else
  //               {
  //                 // Compare with saved displacement
  //                 const double displ =
  //                   this->present_solution[face_dofs[i_dof]] -
  //                   this->initial_positions.at(face_dofs[i_dof])[d];
  //                 max_diff_local[d] =
  //                   std::max(max_diff_local[d],
  //                            cylinder_displacement_local[d] - displ);
  //               }
  //             }
  //           }
  //       }
  //     }

  // for (unsigned int d = 0; d < dim; ++d)
  //   lambda_integral[d] =
  //     Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);

  // // To take the max displacement while preserving sign
  // struct MaxAbsOp
  // {
  //   static void
  //   apply(void *invec, void *inoutvec, int *len, MPI_Datatype * /*dtype*/)
  //   {
  //     double *in    = static_cast<double *>(invec);
  //     double *inout = static_cast<double *>(inoutvec);
  //     for (int i = 0; i < *len; ++i)
  //     {
  //       if (std::fabs(in[i]) > std::fabs(inout[i]))
  //         inout[i] = in[i];
  //     }
  //   }
  // };
  // MPI_Op mpi_maxabs;
  // MPI_Op_create(&MaxAbsOp::apply, /*commutative=*/true, &mpi_maxabs);

  // Tensor<1, dim> cylinder_displacement, max_diff, ratio;
  // for (unsigned int d = 0; d < dim; ++d)
  // {
  //   /**
  //    * Cylinder displacement is trivially 0 on processes which do not own a
  //    part
  //    * of the boundary, and is nontrivial otherwise.     Taking the max to
  //    * synchronize does not work because displacement can be negative.
  //    Instead,
  //    * we take the max while preserving the sign.
  //    */
  //   MPI_Allreduce(&cylinder_displacement_local[d],
  //                 &cylinder_displacement[d],
  //                 1,
  //                 MPI_DOUBLE,
  //                 mpi_maxabs,
  //                 this->mpi_communicator);

  //   // Take the max between all max differences disp_i - disp_j
  //   // for x_i and x_j both on the cylinder.
  //   // Checks that all displacement are identical.
  //   max_diff[d] =
  //     Utilities::MPI::max(max_diff_local[d], this->mpi_communicator);

  //   // Check that the ratio of both terms in the position
  //   // boundary condition is -spring_constant
  //   if (std::abs(cylinder_displacement[d]) > 1e-10)
  //     ratio[d] = lambda_integral[d] / cylinder_displacement[d];
  // }

  // if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  // {
  //   this->pcout << std::endl;
  //   this->pcout << std::scientific << std::setprecision(8) << std::showpos;
  //   this->pcout
  //     << "Checking consistency between lambda integral and position BC:"
  //     << std::endl;
  //   this->pcout << "Integral of lambda on cylinder is " << lambda_integral
  //               << std::endl;
  //   this->pcout << "Prescribed displacement        is " <<
  //   cylinder_displacement
  //               << std::endl;
  //   this->pcout << "                         Ratio is " << ratio
  //               << " (expected: " << -this->param.fsi.spring_constant << ")"
  //               << std::endl;
  //   this->pcout << "Max diff between displacements is " << max_diff
  //               << std::endl;
  //   this->pcout << std::endl;
  // }

  // AssertThrow(max_diff.norm() <= 1e-10,
  //             ExcMessage(
  //               "Displacement values of the cylinder are not all the
  //               same."));

  // //
  // // Check relative error between lambda/disp ratio vs spring constant
  // //
  // for (unsigned int d = 0; d < dim; ++d)
  // {
  //   if (std::abs(ratio[d]) < 1e-10)
  //     continue;

  //   const double absolute_error =
  //     std::abs(ratio[d] - (-this->param.fsi.spring_constant));

  //   if (absolute_error <= 1e-6)
  //     continue;

  //   const double relative_error =
  //     absolute_error / this->param.fsi.spring_constant;
  //   AssertThrow(relative_error <= 1e-2,
  //               ExcMessage("Ratio integral vs displacement values is not
  //               -k"));
  // }
}

template <int dim>
void NSSolverLambda<dim>::check_velocity_boundary() const
{
  // // Check difference between uh and dxhdt
  // double l2_local = 0;
  // double li_local = 0;

  // FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
  //                                        *fe,
  //                                        *this->face_quadrature,
  //                                        update_values |
  //                                          update_quadrature_points |
  //                                          update_JxW_values);
  // FEFaceValues<dim> fe_face_values(*this->moving_mapping,
  //                                  *fe,
  //                                  *this->face_quadrature,
  //                                  update_values | update_quadrature_points |
  //                                    update_JxW_values);

  // const unsigned int n_faces_q_points = this->face_quadrature->size();

  // const auto &bdf_coefficients = this->time_handler.bdf_coefficients;

  // std::vector<std::vector<Tensor<1, dim>>> position_values(
  //   bdf_coefficients.size(), std::vector<Tensor<1, dim>>(n_faces_q_points));
  // std::vector<Tensor<1, dim>> mesh_velocity_values(n_faces_q_points);
  // std::vector<Tensor<1, dim>> fluid_velocity_values(n_faces_q_points);
  // Tensor<1, dim>              diff;

  // for (auto cell : this->dof_handler.active_cell_iterators())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;

  //   for (const auto i_face : cell->face_indices())
  //   {
  //     const auto &face = cell->face(i_face);

  //     if (face->at_boundary() &&
  //         face->boundary_id() == weak_no_slip_boundary_id)
  //     {
  //       fe_face_values_fixed.reinit(cell, i_face);
  //       fe_face_values.reinit(cell, i_face);

  //       // Get current and previous FE solution values on the face
  //       fe_face_values[this->velocity_extractor].get_function_values(
  //         this->present_solution, fluid_velocity_values);
  //       fe_face_values_fixed[this->position_extractor].get_function_values(
  //         this->present_solution, position_values[0]);
  //       for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
  //         fe_face_values_fixed[this->position_extractor].get_function_values(
  //           this->previous_solutions[iBDF - 1], position_values[iBDF]);

  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         // Compute FE mesh velocity at node
  //         mesh_velocity_values[q] = 0;
  //         for (unsigned int iBDF = 0; iBDF < bdf_coefficients.size(); ++iBDF)
  //           mesh_velocity_values[q] +=
  //             bdf_coefficients[iBDF] * position_values[iBDF][q];

  //         diff = mesh_velocity_values[q] - fluid_velocity_values[q];

  //         // u_h - w_h
  //         l2_local += diff * diff * fe_face_values_fixed.JxW(q);
  //         li_local = std::max(li_local, std::abs(diff.norm()));
  //       }
  //     }
  //   }
  // }

  // const double l2_error =
  //   std::sqrt(Utilities::MPI::sum(l2_local, this->mpi_communicator));
  // const double li_error = Utilities::MPI::max(li_local,
  // this->mpi_communicator);

  // if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  // {
  //   this->pcout << "Checking no-slip enforcement on cylinder:" << std::endl;
  //   this->pcout << "||uh - wh||_L2 = " << l2_error << std::endl;
  //   this->pcout << "||uh - wh||_Li = " << li_error << std::endl;
  // }

  // if (!this->param.debug.fsi_apply_erroneous_coupling)
  // {
  //   AssertThrow(l2_error < 1e-12,
  //               ExcMessage("L2 norm of uh - wh is too large : " +
  //                          std::to_string(l2_error)));
  //   AssertThrow(li_error < 1e-12,
  //               ExcMessage("Linf norm of uh - wh is too large : " +
  //                          std::to_string(li_error)));
  // }
}

template <int dim>
void NSSolverLambda<dim>::check_manufactured_solution_boundary()
{
  // Tensor<1, dim> lambdaMMS_integral, lambdaMMS_integral_local;
  // Tensor<1, dim> lambda_integral, lambda_integral_local;
  // Tensor<1, dim> pns_integral, pns_integral_local;
  // lambdaMMS_integral_local = 0;
  // lambda_integral_local    = 0;
  // pns_integral_local       = 0;

  // const double rho = this->param.physical_properties.fluids[0].density;
  // const double nu =
  //   this->param.physical_properties.fluids[0].kinematic_viscosity;
  // const double mu = nu * rho;

  // FEFaceValues<dim> fe_face_values(*this->moving_mapping,
  //                                  *fe,
  //                                  *this->face_quadrature,
  //                                  update_values | update_quadrature_points |
  //                                    update_JxW_values | update_normal_vectors);
  // FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
  //                                        *fe,
  //                                        *this->face_quadrature,
  //                                        update_values |
  //                                          update_quadrature_points |
  //                                          update_JxW_values);

  // const unsigned int          n_faces_q_points =
  // this->face_quadrature->size(); Tensor<1, dim>              lambda_MMS;
  // std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  // //
  // // First compute integral over cylinder of lambda_MMS
  // //
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
  //         const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
  //         const auto        normal_to_solid =
  //         -fe_face_values.normal_vector(q);

  //         const double p_MMS =
  //           this->exact_solution->value(qpoint, this->ordering->p_lower);

  //         std::static_pointer_cast<NSSolverLambda<dim>::MMSSolution>(
  //           this->exact_solution)
  //           ->lagrange_multiplier(qpoint, mu, normal_to_solid, lambda_MMS);

  //         // Increment the integrals of lambda:

  //         // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
  //         lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

  //         // This is int lambda := int sigma(u_MMS, p_MMS) cdot
  //         normal_to_fluid
  //         // -normal_to_solid lambda_integral_local += lambda_values[q] *
  //         fe_face_values.JxW(q);

  //         // Increment integral of p * n_solid
  //         pns_integral_local += p_MMS * normal_to_solid *
  //         fe_face_values.JxW(q);
  //       }
  //     }
  //   }
  // }

  // for (unsigned int d = 0; d < dim; ++d)
  // {
  //   lambdaMMS_integral[d] =
  //     Utilities::MPI::sum(lambdaMMS_integral_local[d],
  //     this->mpi_communicator);
  //   lambda_integral[d] =
  //     Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
  // }
  // pns_integral =
  //   Utilities::MPI::sum(pns_integral_local, this->mpi_communicator);

  // // // Reference solution for int_Gamma p*n_solid dx is - k * d * f(t).
  // // Tensor<1, dim> translation;
  // // translation[0] = 0.1;
  // // translation[1] = 0.05;
  // const Tensor<1, dim> ref_pns;
  // // const Tensor<1, dim> ref_pns =
  // //   -param.fsi.spring_constant * translation *
  // //   std::static_pointer_cast<NSSolverLambda<dim>::MMSSolution>(
  // //
  // exact_solution)->mms.exact_mesh_position->time_function->value(this->time_handler.current_time);
  // // const double err_pns = (ref_pns - pns_integral).norm();
  // const double err_pns = -1.;

  // //
  // // Check x_MMS
  // //
  // Tensor<1, dim> x_MMS;
  // double         max_x_error = 0.;
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
  //       fe_face_values_fixed.reinit(cell, i_face);

  //       // Evaluate exact solution at quadrature points
  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         const Point<dim> &qpoint_fixed =
  //           fe_face_values_fixed.quadrature_point(q);

  //         for (unsigned int d = 0; d < dim; ++d)
  //           x_MMS[d] = this->exact_solution->value(qpoint_fixed,
  //                                                  this->ordering->x_lower +
  //                                                  d);

  //         const Tensor<1, dim> ref =
  //           -1. / this->param.fsi.spring_constant * lambdaMMS_integral;
  //         const double err = ((x_MMS - qpoint_fixed) - ref).norm();
  //         // std::cout << "x_MMS - X0 at quad node is " << x_MMS  -
  //         qpoint_fixed
  //         // << " - diff = " << err << std::endl;
  //         max_x_error = std::max(max_x_error, err);
  //       }
  //     }
  //   }
  // }

  // //
  // // Check u_MMS
  // //
  // Tensor<1, dim> u_MMS, w_MMS;
  // double         max_u_error = -1;
  // // for (auto cell : this->dof_handler.active_cell_iterators())
  // // {
  // //   if (!cell->is_locally_owned())
  // //     continue;
  // //   for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  // //   {
  // //     const auto &face = cell->face(i_face);
  // //     if (face->at_boundary() && face->boundary_id() == boundary_id)
  // //     {
  // //       fe_face_values.reinit(cell, i_face);
  // //       fe_face_values_fixed.reinit(cell, i_face);

  // //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  // //       {
  // //         const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
  // //         const Point<dim> &qpoint_fixed  =
  // //         fe_face_values_fixed.quadrature_point(q);

  // //         for (unsigned int d = 0; d < dim; ++d)
  // //         {
  // //           u_MMS[d] = solution_fun.value(qpoint, u_lower + d);
  // //           w_MMS[d] = mesh_velocity_fun.value(qpoint_fixed, x_lower + d);
  // //         }

  // //         const double err = (u_MMS - w_MMS).norm();
  // //         // std::cout << "u_MMS & w_MMS at quad node are " << u_MMS << "
  // , "
  // //         << w_MMS << " - norm diff = " << err << std::endl; max_u_error =
  // //         std::max(max_u_error, err);
  // //       }
  // //     }
  // //   }
  // // }

  // // if(VERBOSE)
  // // {
  // this->pcout << std::endl;
  // this->pcout << "Checking manufactured solution for k = "
  //             << this->param.fsi.spring_constant << " :" << std::endl;
  // this->pcout << "integral lambda         = " << lambda_integral <<
  // std::endl; this->pcout << "integral lambdaMMS      = " <<
  // lambdaMMS_integral
  //             << std::endl;
  // this->pcout << "integral pMMS * n_solid = " << pns_integral << std::endl;
  // this->pcout << "reference: -k*d*f(t)    = " << ref_pns
  //             << " - err = " << err_pns << std::endl;
  // this->pcout << "max error on (x_MMS -    X0) vs -1/k * integral lambda = "
  //             << max_x_error << std::endl;
  // this->pcout << "max error on  u_MMS          vs w_MMS                  = "
  //             << max_u_error << std::endl;
  // this->pcout << std::endl;
  // // }
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
void NSSolverLambda<dim>::write_cylinder_position(const bool export_table)
{
  // Tensor<1, dim> average_position, position_integral_local;
  // double         boundary_measure_local = 0.;

  // FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
  //                                        *fe,
  //                                        *this->face_quadrature,
  //                                        update_values |
  //                                          update_quadrature_points |
  //                                          update_JxW_values |
  //                                          update_normal_vectors);

  // const unsigned int          n_faces_q_points =
  // this->face_quadrature->size(); std::vector<Tensor<1, dim>>
  // position_values(n_faces_q_points);

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
  //       fe_face_values_fixed.reinit(cell, i_face);

  //       // Get FE solution values on the face
  //       fe_face_values_fixed[this->position_extractor].get_function_values(
  //         this->present_solution, position_values);

  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         boundary_measure_local += fe_face_values_fixed.JxW(q);
  //         position_integral_local +=
  //           position_values[q] * fe_face_values_fixed.JxW(q);
  //       }
  //     }
  //   }
  // }

  // const double boundary_measure =
  //   Utilities::MPI::sum(boundary_measure_local, this->mpi_communicator);
  // for (unsigned int d = 0; d < dim; ++d)
  //   average_position[d] =
  //     1. / boundary_measure *
  //     Utilities::MPI::sum(position_integral_local[d],
  //     this->mpi_communicator);

  // cylinder_position_table.add_value("time", this->time_handler.current_time);
  // cylinder_position_table.add_value("xc", average_position[0]);
  // cylinder_position_table.add_value("yc", average_position[1]);
  // if constexpr (dim == 3)
  //   cylinder_position_table.add_value("zc", average_position[2]);

  // if (export_table && this->param.output.write_results && this->mpi_rank ==
  // 0)
  // {
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "cylinder_center.txt");
  //   cylinder_position_table.write_text(outfile);
  // }
}

template <int dim>
void NSSolverLambda<dim>::solver_specific_post_processing()
{
  // output_results();

  if (this->param.mms_param.enable)
  {
    // compute_errors();

    if (this->param.debug.fsi_check_mms_on_boundary)
      check_manufactured_solution_boundary();
  }

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
  const bool export_position_table =
    this->time_handler.is_steady() ||
    ((this->time_handler.current_time_iteration % 5) == 0);
  write_cylinder_position(export_position_table);
}

// Explicit instantiation
template class NSSolverLambda<2>;
template class NSSolverLambda<3>;