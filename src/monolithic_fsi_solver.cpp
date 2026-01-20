
#include <compare_matrix.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_solver.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
FSISolver<dim>::FSISolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, true>(param)
{
  if (param.finite_elements.use_quads)
    fe = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,      // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree),            // Pressure
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim, // Position
      FE_Q<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier
  else
    fe = std::make_shared<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim, // Velocity
      FE_SimplexP<dim>(param.finite_elements.pressure_degree),       // Pressure
      FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^
        dim, // Position
      FE_SimplexP<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier

  this->ordering = std::make_shared<ComponentOrderingFSI<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  this->position_extractor =
    FEValuesExtractors::Vector(this->ordering->x_lower);
  this->lambda_extractor = FEValuesExtractors::Vector(this->ordering->l_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  this->position_mask = fe->component_mask(this->position_extractor);
  this->lambda_mask   = fe->component_mask(this->lambda_extractor);

  // Set the boundary id on which a weak no slip boundary condition is applied.
  // It is allowed *not* to prescribe a weak no slip on any boundary, to verify
  // that the solver produces the expected flow in the decoupled case.
  unsigned int n_weak_bc = 0;
  for (const auto &[id, bc] : param.fluid_bc)
    if (bc.type == BoundaryConditions::Type::weak_no_slip)
    {
      weak_no_slip_boundary_id = bc.id;
      n_weak_bc++;

      for (const auto &[id, bc] : param.pseudosolid_bc)
        if (bc.type == BoundaryConditions::Type::coupled_to_fluid)
          AssertThrow(
            bc.id == weak_no_slip_boundary_id,
            ExcMessage(
              "A pseudosolid boundary condition was set to "
              "\"coupled_to_fluid\" on boundary \"" +
              bc.gmsh_name +
              "\", but the fluid boundary condition on this boundary was not "
              "set to \"weak_no_slip\". For now, fluid-structure coupling can "
              "only be done through a Lagrange multiplier, which requires "
              "weakly enforced no-slip condition on the coupled boundary."));
    }
  AssertThrow(n_weak_bc <= 1,
              ExcMessage(
                "A weakly enforced no-slip boundary condition is enforced on "
                "more than 1 boundary, which is currently not supported."));

  /**
   * Enforcing zero-mean pressure on moving mesh is not trivial, since
   * the constraint weights depend on the mesh position.
   */
  AssertThrow(!param.bc_data.enforce_zero_mean_pressure,
              ExcMessage("Enforcing zero mean pressure on moving mesh is "
                         "currently not implemented."));

  /**
   * While different coupling schemes are still being tested, keep the debug
   * flag to change the scheme at runtime, but do not allow using the first,
   * inefficient coupling.
   */
  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "Using coupling scheme : "
                << this->param.debug.fsi_coupling_option << std::endl;
  AssertThrow(this->param.debug.fsi_coupling_option != 0,
              ExcMessage(
                "This parameter file still uses the inefficient coupling "
                "scheme used for prototyping. Use a better coupling by setting "
                "fsi_coupling_option = 1 in the Debug subsection."));

  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    this->exact_solution = std::make_shared<FSISolver<dim>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

    // Create the source term function for the given MMS and override source
    // terms
    this->source_terms = std::make_shared<FSISolver<dim>::MMSSourceTerm>(
      this->time_handler.current_time,
      *this->ordering,
      param.physical_properties,
      param.mms);

    // Create entry in error handler for Lagrange multiplier
    for (auto norm : this->param.mms_param.norms_to_compute)
    {
      this->error_handlers[norm]->create_entry("l");
      if (this->param.fsi.compute_error_on_forces)
        for (unsigned int d = 0; d < dim; ++d)
          this->error_handlers[norm]->create_entry("F_comp" +
                                                   std::to_string(d));
    }
  }
  else
  {
    this->source_terms = std::make_shared<FSISolver<dim>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
void FSISolver<dim>::MMSSourceTerm::vector_value(const Point<dim> &p,
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

  // Velocity source term
  Tensor<1, dim> f = -(dudt_eulerian + uDotGradu + grad_p - nu * lap_u);
  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.u_lower + d] = f[d];

  // Mass conservation (pressure) source term
  values[ordering.p_lower] = mms.exact_velocity->divergence(p);

  // Pseudosolid (mesh position) source term
  // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
  Tensor<1, dim> f_PS =
    mms.exact_mesh_position
      ->divergence_linear_elastic_stress_variable_coefficients(
        p,
        physical_properties.pseudosolids[0].lame_mu_fun,
        physical_properties.pseudosolids[0].lame_lambda_fun);

  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.x_lower + d] = f_PS[d];

  // Lagrange multiplier source term (none)
  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.l_lower + d] = 0.;
}

template <int dim>
void FSISolver<dim>::reset_solver_specific_data()
{
  // Position - lambda constraints
  for (auto &vec : position_lambda_coeffs)
    vec.clear();
  position_lambda_coeffs.clear();
  coupled_position_dofs.clear();
  has_chunk_of_cylinder           = false;
  has_global_master_position_dofs = false;
  for (unsigned int d = 0; d < dim; ++d)
  {
    local_position_master_dofs[d]  = numbers::invalid_unsigned_int;
    global_position_master_dofs[d] = numbers::invalid_unsigned_int;
  }
}

template <int dim>
void FSISolver<dim>::create_lagrange_multiplier_constraints()
{
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
  std::vector<types::global_dof_index> local_dofs(fe->n_dofs_per_cell());
  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;
    cell->get_dof_indices(local_dofs);
    for (unsigned int i = 0; i < local_dofs.size(); ++i)
    {
      types::global_dof_index dof  = local_dofs[i];
      unsigned int            comp = fe->system_to_component_index(i).first;
      if (this->ordering->is_lambda(comp))
        if (this->locally_relevant_dofs.is_element(dof))
          if (!relevant_boundary_dofs.is_element(dof))
            lambda_constraints.constrain_dof_to_zero(dof);
    }
  }
  lambda_constraints.close();

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
  {
    // Print number of owned and constrained lambda dofs
    IndexSet lambda_dofs =
      DoFTools::extract_dofs(this->dof_handler, lambda_mask);
    unsigned int unconstrained_owned_dofs = 0;
    for (const auto &dof : lambda_dofs)
      if (!lambda_constraints.is_constrained(dof))
        unconstrained_owned_dofs++;

    const unsigned int total_unconstrained_owned_dofs =
      Utilities::MPI::sum(unconstrained_owned_dofs, this->mpi_communicator);
    std::cout << total_unconstrained_owned_dofs
              << " unconstrained owned lambda dofs" << std::endl;
  }
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
void FSISolver<dim>::create_position_lagrange_mult_coupling_data()
{
  //
  // Get and synchronize the lambda DoFs on the cylinder
  //
  IndexSet local_lambda_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    lambda_mask,
                                    {weak_no_slip_boundary_id});
  IndexSet local_position_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->position_mask,
                                    {weak_no_slip_boundary_id});

  /**
   * We might be missing some owned dofs, on boundary edges
   * (see also comment in the remove_constraints function below).
   * Add them here.
   */
  {
    std::vector<std::vector<types::global_dof_index>> gathered_lambda_bdr_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 local_lambda_dofs.get_index_vector());
    std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 local_position_dofs.get_index_vector());

    for (const auto &vec : gathered_lambda_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          local_lambda_dofs.add_index(dof);
    for (const auto &vec : gathered_pos_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          local_position_dofs.add_index(dof);
  }

  const unsigned int n_local_lambda_dofs = local_lambda_dofs.n_elements();

  local_lambda_dofs   = local_lambda_dofs & this->locally_owned_dofs;
  local_position_dofs = local_position_dofs & this->locally_owned_dofs;

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
    additional_relevant_dofs.set_size(this->dof_handler.n_dofs());
    additional_relevant_dofs.add_indices(gathered_dofs_flattened.begin(),
                                         gathered_dofs_flattened.end());
    this->locally_relevant_dofs.add_indices(additional_relevant_dofs);
    this->locally_relevant_dofs.compress();
  }

  /**
   * Reinitialize the ghosted parallel vectors with the additional ghosts.
   */
  this->present_solution.reinit(this->locally_owned_dofs,
                                this->locally_relevant_dofs,
                                this->mpi_communicator);
  this->evaluation_point.reinit(this->locally_owned_dofs,
                                this->locally_relevant_dofs,
                                this->mpi_communicator);
  this->present_solution = this->local_evaluation_point;
  this->evaluation_point = this->local_evaluation_point;

  for (auto &previous_sol : this->previous_solutions)
  {
    // Create a temporary, fully distributed copy of the previous solution to
    // reapply after resizing. This is needed for checkpointing, because the
    // previous solutions won't be zero when restarting.
    LA::ParVectorType tmp_prev_sol(this->locally_owned_dofs,
                                   this->mpi_communicator);
    tmp_prev_sol = previous_sol;
    previous_sol.reinit(this->locally_owned_dofs,
                        this->locally_relevant_dofs,
                        this->mpi_communicator);
    previous_sol = tmp_prev_sol;
  }

  if (this->param.debug.fsi_coupling_option == 1)
  {
    // Set the local_position_master_dofs
    // Simply take the first owned position dofs on the cylinder
    // Here it's assumed that local_position_dofs is organized as
    // x_0, y_0, z_0, x_1, y_1, z_1, ...,
    // and we take the first dim.
    const auto index_vector = local_position_dofs.get_index_vector();
    if (index_vector.size() > 0)
    {
      has_chunk_of_cylinder = true;
      AssertThrow(index_vector.size() >= dim,
                  ExcMessage(
                    "This partition has position dofs on the cylinder, but has "
                    "less than dim position dofs, which should not happen. It "
                    "should have n * dim position dofs on this boundary."));
      for (unsigned int d = 0; d < dim; ++d)
        local_position_master_dofs[d] = index_vector[d];
    }
  }
  else if (this->param.debug.fsi_coupling_option == 2)
  {
    // Set local master position dofs (smallest on cylinder on this rank)
    // Then set global master position dofs (smallest of all on cylinder)
    const auto index_vector = local_position_dofs.get_index_vector();
    if (index_vector.size() > 0)
    {
      has_chunk_of_cylinder = true;
      AssertThrow(index_vector.size() >= dim,
                  ExcMessage(
                    "This partition has position dofs on the cylinder, but has "
                    "less than dim position dofs, which should not happen. It "
                    "should have n * dim position dofs on this boundary."));
      for (unsigned int d = 0; d < dim; ++d)
        local_position_master_dofs[d] = index_vector[d];
    }

    // The global master dofs are those on the lowest rank among the ranks on
    // which the local position dofs is defined
    const unsigned int candidate_rank =
      has_chunk_of_cylinder ? this->mpi_rank :
                              std::numeric_limits<unsigned int>::max();
    const unsigned int owner_rank =
      Utilities::MPI::min(candidate_rank, this->mpi_communicator);
    has_global_master_position_dofs = (this->mpi_rank == owner_rank);

    for (unsigned int d = 0; d < dim; ++d)
      global_position_master_dofs[d] = local_position_master_dofs[d];

    if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
      this->pcout << "Global master is on rank " << owner_rank << std::endl;

    Utilities::MPI::broadcast(global_position_master_dofs.data(),
                              dim,
                              owner_rank,
                              this->mpi_communicator);
  }

  //
  // Compute the weights c_ij and identify the constrained position DOFs.
  // Done only once as cylinder is rigid and those weights will not change.
  //
  std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

  FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values | update_JxW_values);

  const unsigned int                   n_dofs_per_face = fe->n_dofs_per_face();
  std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    /**
     * Loop only on the owned cells for 2 reasons :
     *
     * - Only owned cells contribute to the integral of lambda on this partition
     *
     * - The force-position coupling is done by hand by modifying the linear
     * system directly. Since each rank only stores its *owned* lines in the
     * matrix/rhs, we are only interested in the *owned* position dofs that are
     * coupled to the lambda dofs. Some ghost dofs are added here, but we only
     * care for the owned.
     *
     *   Important (see below) : since we loop over cell *faces*, we can miss
     * owned position dofs which should be coupled. This happens when owned dofs
     * are located on the edges of slanted tets, whose faces do *not* lie on the
     *   boundary of the obstacle. Thus, we never loop over these faces and
     * cannot get these owned dofs. They are added afterwards after gathering
     * the coupled dofs from other ranks.
     *
     *   TODO: Check if it's possible to loop over edges, but I doubt it since
     * the boundary ID is ill defined on edges.
     */
    if (cell->is_locally_owned())
    {
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        fe_face_values_fixed.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int q = 0; q < this->face_quadrature->size(); ++q)
        {
          const double JxW = fe_face_values_fixed.JxW(q);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
          {
            const unsigned int comp =
              fe->face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which
            // contribute to the integral on this element
            if (!this->locally_relevant_dofs.is_element(face_dofs[i_dof]))
              continue;

            /**
             * Lambda face dofs contribute to the weights
             */
            if (this->ordering->is_lambda(comp))
            {
              const unsigned int            d = comp - this->ordering->l_lower;
              const types::global_dof_index lambda_dof = face_dofs[i_dof];

              // Very, very, very important:
              // Even though fe_face_values_fixed is a FEFaceValues, the dof
              // index given to shape_value is still a CELL dof index.
              const unsigned int i_cell_dof =
                fe->face_to_cell_index(i_dof, i_face);
              const double phi_i =
                fe_face_values_fixed.shape_value(i_cell_dof, q);
              coeffs[d][lambda_dof] +=
                -phi_i * JxW / this->param.fsi.spring_constant;

              if constexpr (dim == 3)
                if (d == 2 && this->param.fsi.fix_z_component)
                  coeffs[d][lambda_dof] = 0.;
            }

            /**
             * Position face dofs are added to the list of coupled dofs
             */
            if (this->ordering->is_position(comp))
            {
              const unsigned int d = comp - this->ordering->x_lower;
              coupled_position_dofs.insert({face_dofs[i_dof], d});
            }
          }
        }
      }
    }
  }

  /**
   * We might be missing some owned coupled dofs, on boundary edges
   * (see also comment in the remove_constraints function below).
   * Add them here.
   */
  {
    using MessageType =
      std::vector<std::pair<types::global_dof_index, unsigned int>>;
    MessageType coupled_position_dofs_vec(coupled_position_dofs.begin(),
                                          coupled_position_dofs.end());

    std::vector<MessageType> gathered_coupled_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 coupled_position_dofs_vec);

    for (const auto &vec : gathered_coupled_dofs)
      for (const auto &[dof, dimension] : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          coupled_position_dofs.insert({dof, dimension});
  }

  /**
   * Sanity check on the weights
   * Expected sum is -1/k * |Cylinder|
   */
  {
    const double k                    = this->param.fsi.spring_constant;
    const double r                    = this->param.fsi.cylinder_radius;
    double       expected_weights_sum = -1 / k * 2. * M_PI * r;
    if constexpr (dim == 3)
      expected_weights_sum *= this->param.fsi.cylinder_length;

    const double expected_discrete_weights_sum =
      -1. / k *
      compute_boundary_volume(this->dof_handler,
                              *this->moving_mapping,
                              *this->face_quadrature,
                              weak_no_slip_boundary_id);

    for (unsigned int d = 0; d < dim; ++d)
    {
      // Do not compare for dim = 2 if fixed
      if (d == 2 && this->param.fsi.fix_z_component)
        continue;

      double local_weights_sum = 0.;
      for (const auto &[lambda_dof, weight] : coeffs[d])
        local_weights_sum += weight;

      const double weights_sum =
        Utilities::MPI::sum(local_weights_sum, this->mpi_communicator);

      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
      {
        this->pcout << "Dim " << d << " : Sum of weights = " << weights_sum
                    << " - expected from mesh : "
                    << expected_discrete_weights_sum
                    << " - expected theoretical : " << expected_weights_sum
                    << std::endl;
      }

      AssertThrow(
        std::abs(weights_sum - expected_discrete_weights_sum) < 1e-10,
        ExcMessage(
          "The sum of weights for component " + std::to_string(d) +
          " of lambda coupling should be -1/k * |Cylinder|, but it's not."));
    }
  }

  //
  // Gather the constraint weights
  //
  position_lambda_coeffs.resize(dim);
  std::vector<std::map<types::global_dof_index, double>> gathered_coeffs_map(
    dim);

  for (unsigned int d = 0; d < dim; ++d)
  {
    std::vector<std::pair<types::global_dof_index, double>> coeffs_vector(
      coeffs[d].begin(), coeffs[d].end());
    std::vector<std::vector<std::pair<types::global_dof_index, double>>>
      gathered =
        Utilities::MPI::all_gather(this->mpi_communicator, coeffs_vector);

    // Put back into map and sum contributions to same DoF from different
    // processes
    for (const auto &vec : gathered)
      for (const auto &[lambda_dof, weight] : vec)
        gathered_coeffs_map[d][lambda_dof] += weight;

    position_lambda_coeffs[d].insert(position_lambda_coeffs[d].end(),
                                     gathered_coeffs_map[d].begin(),
                                     gathered_coeffs_map[d].end());
  }
}

template <int dim>
void FSISolver<dim>::remove_cylinder_velocity_constraints(
  AffineConstraints<double> &constraints,
  const bool                 remove_velocity_constraints,
  const bool                 remove_position_constraints) const
{
  if (weak_no_slip_boundary_id == numbers::invalid_unsigned_int)
    return;

  IndexSet relevant_boundary_velocity_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->velocity_mask,
                                    {weak_no_slip_boundary_id});
  IndexSet relevant_boundary_position_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->position_mask,
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
    std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs =
      Utilities::MPI::all_gather(
        this->mpi_communicator,
        relevant_boundary_position_dofs.get_index_vector());

    for (const auto &vec : gathered_vel_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          relevant_boundary_velocity_dofs.add_index(dof);
    for (const auto &vec : gathered_pos_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          relevant_boundary_position_dofs.add_index(dof);
  }

  // ///////////////////////////////////////////////////////////////////////////
  // // Get the support points for the relevant dofs
  // std::map<types::global_dof_index, Point<dim>> support_points =
  //   DoFTools::map_dofs_to_support_points(*this->fixed_mapping,
  //                                        this->dof_handler);

  // /**
  //  * For debug: Create a dof to component map (relevant dofs only,
  //  * because looping over owned and ghost cells)
  //  */
  // const types::global_dof_index n_dofs        = this->dof_handler.n_dofs();
  // const unsigned int            dofs_per_cell = fe.dofs_per_cell;

  // std::vector<int>                     dof_to_component(n_dofs, -1);
  // std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // for (const auto &cell : this->dof_handler.active_cell_iterators())
  // {
  //   cell->get_dof_indices(local_dof_indices);
  //   for (unsigned int i = 0; i < dofs_per_cell; ++i)
  //   {
  //     const unsigned int component      =
  //     fe->system_to_component_index(i).first; const types::global_dof_index
  //     dof = local_dof_indices[i];

  //     // Check that this is compatible with value already there, if any
  //     AssertThrow(dof_to_component[dof] == -1 ||
  //                   dof_to_component[dof] == component,
  //                 ExcMessage("Mismatch in dof component"));

  //     dof_to_component[dof] = component;
  //   }
  // }

  // /**
  //  * Print owned, ghost and relevant boundary VELOCITY dofs
  //  */
  // {
  //   // Print owned velocity dofs
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "owned_velocity_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"owned_velocity_dofs_proc" << this->mpi_rank << "\"{"
  //           << std::endl;
  //   for (const auto dof : this->locally_owned_dofs)
  //   {
  //     if (this->ordering->is_velocity(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // {
  //   // Print ghost velocity dofs
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "ghost_velocity_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"ghost_velocity_dofs_proc" << this->mpi_rank << "\"{"
  //           << std::endl;
  //   for (const auto dof : this->locally_relevant_dofs)
  //   {
  //     if (this->locally_owned_dofs.is_element(dof))
  //       continue;

  //     if (this->ordering->is_velocity(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // {
  //   // Print relevant boundary velocity dofs from extraction
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "extracted_velocity_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"extracted_velocity_dofs_proc" << this->mpi_rank <<
  //   "\"{"
  //           << std::endl;
  //   for (const auto dof : relevant_boundary_velocity_dofs)
  //   {
  //     if (this->ordering->is_velocity(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // /**
  //  * Print owned, ghost and relevant boundary POSITION dofs
  //  */
  // {
  //   // Print owned position dofs
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "owned_position_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"owned_position_dofs_proc" << this->mpi_rank << "\"{"
  //           << std::endl;
  //   for (const auto dof : this->locally_owned_dofs)
  //   {
  //     if (this->ordering->is_position(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // {
  //   // Print ghost position dofs
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "ghost_position_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"ghost_position_dofs_proc" << this->mpi_rank << "\"{"
  //           << std::endl;
  //   for (const auto dof : this->locally_relevant_dofs)
  //   {
  //     if (this->locally_owned_dofs.is_element(dof))
  //       continue;

  //     if (this->ordering->is_position(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // {
  //   // Print relevant boundary position dofs from extraction
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "extracted_position_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"extracted_position_dofs_proc" << this->mpi_rank <<
  //   "\"{"
  //           << std::endl;
  //   for (const auto dof : relevant_boundary_position_dofs)
  //   {
  //     if (this->ordering->is_position(dof_to_component[dof]))
  //     {
  //       const Point<dim> &pt = support_points.at(dof);
  //       outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //               << std::endl;
  //     }
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // {
  //   // Print the COUPLED POSITION DOFS on this partition
  //   std::ofstream outfile(this->param.output.output_dir +
  //                         "coupled_position_dofs_proc" +
  //                         std::to_string(this->mpi_rank) + ".pos");
  //   outfile << "View \"coupled_position_dofs_proc" << this->mpi_rank << "\"{"
  //           << std::endl;
  //   for (const auto &[dof, dimension] : coupled_position_dofs)
  //   {
  //     const Point<dim> &pt = support_points.at(dof);
  //     outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2] << "){1};"
  //             << std::endl;
  //   }
  //   outfile << "};" << std::endl;
  //   outfile.close();
  // }
  // MPI_Barrier(this->mpi_communicator);
  // {
  //   // Check that all coupled position dofs are indeed relevant on the
  //   // boundary and vice versa
  //   for (const auto &[pos_dof, d] : coupled_position_dofs)
  //   {
  //     AssertThrow(relevant_boundary_position_dofs.is_element(pos_dof),
  //                 ExcMessage("A coupled position dof was not extracted"));
  //   }
  //   for (const auto &pos_dof : relevant_boundary_position_dofs)
  //   {
  //     AssertThrow(coupled_position_dofs.count(pos_dof) > 0,
  //                 ExcMessage(
  //                   "An extract position dof is not in the coupled map"));
  //   }
  // }

  // const auto og_relevant =
  //   DoFTools::extract_locally_relevant_dofs(this->dof_handler);

  // Print
  // for (unsigned int r = 0; r < this->mpi_size; ++r)
  // {
  //   MPI_Barrier(this->mpi_communicator);
  //   if (r == this->mpi_rank)
  //   {
  //     std::cout << "In remove : Rank " << this->mpi_rank << " has "
  //               << this->locally_relevant_dofs.n_elements() << " relevant and
  //               "
  //               << additional_relevant_dofs.n_elements()
  //               << " additional and og has " << og_relevant.n_elements()
  //               << std::endl;

  //     for (unsigned int i = 0; i < n_dofs; ++i)
  //     {
  //       // Support points are defined only for relevant dofs
  //       if (!this->locally_relevant_dofs.is_element(i))
  //         continue;

  //       // Support points are not defined for the additional ghost lambda
  //       dofs if (additional_relevant_dofs.is_element(i))
  //         continue;

  //       if (!og_relevant.is_element(i))
  //       {
  //         std::cout << "B: Rank " << r << " : dof " << i
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned       : "
  //                   << this->locally_owned_dofs.is_element(i)
  //                   << " is relevant    : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is og relevant : " << og_relevant.is_element(i)
  //                   << " is additional  : "
  //                   << additional_relevant_dofs.is_element(i) << std::endl;
  //         AssertThrow(false,
  //                     ExcMessage("Dof is not additional but not og
  //                     relevant"));
  //       }

  //       if (relevant_boundary_velocity_dofs.is_element(i) ||
  //           relevant_boundary_position_dofs.is_element(i))
  //       {
  //         std::cout << "B: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << std::endl;

  //         // Faces are at z = 0 and z = 0.5
  //         if (constraints.is_constrained(i))
  //         {
  //           const double z = support_points.at(i)[2];
  //           AssertThrow(std::abs(z) < 1e-10 || std::abs(z - 0.5) < 1e-10,
  //                       ExcMessage("Unexpected constrained dof"));
  //         }
  //       }
  //       else
  //         std::cout << "B: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << " (not u/x or not on boundary)" << std::endl;
  //     }
  //   }
  // }
  ///////////////////////////////////////////////////////////////////////////

  // Check consistency of constraints for RELEVANT (not active) dofs before
  // removing
  {
    const bool consistent = constraints.is_consistent_in_parallel(
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 this->locally_owned_dofs),
      // this->locally_relevant_dofs,
      DoFTools::extract_locally_active_dofs(this->dof_handler),
      this->mpi_communicator,
      true);
    AssertThrow(consistent,
                ExcMessage("Constraints are not consistent before removing"));
  }

  /**
   * Now actually remove the constraints
   */
  {
    AffineConstraints<double> filtered;
    filtered.reinit(this->locally_owned_dofs, this->locally_relevant_dofs);

    for (const auto &line : constraints.get_lines())
    {
      if (remove_velocity_constraints &&
          relevant_boundary_velocity_dofs.is_element(line.index))
        continue;
      if (remove_position_constraints &&
          relevant_boundary_position_dofs.is_element(line.index))
        continue;

      filtered.add_constraint(line.index, line.entries, line.inhomogeneity);

      // Check that entries do not involve an absent velocity dof
      // With the get_view() function, this is done automatically
      for (const auto &entry : line.entries)
      {
        if (remove_velocity_constraints)
          AssertThrow(!relevant_boundary_velocity_dofs.is_element(entry.first),
                      ExcMessage(
                        "Constraint involves a cylinder velocity dof"));
        if (remove_position_constraints)
          AssertThrow(!relevant_boundary_position_dofs.is_element(entry.first),
                      ExcMessage(
                        "Constraint involves a cylinder position dof"));
      }
    }

    filtered.close();
    constraints.clear();
    constraints = std::move(filtered);
  }

  // {
  //   // This does not work:

  //   // IndexSet local_lines = zero_constraints.get_local_lines();
  //   // local_lines.compress();
  //   // this->pcout << local_lines.n_intervals() << std::endl;
  //   // this->pcout << local_lines.n_elements() << std::endl;
  //   // this->pcout << local_lines.size() << std::endl;
  //   // this->pcout << weak_velocity_dofs.n_intervals() << std::endl;
  //   // this->pcout << weak_velocity_dofs.n_elements() << std::endl;
  //   // this->pcout << weak_velocity_dofs.size() << std::endl;
  //   // local_lines.get_view(weak_velocity_dofs);

  //   IndexSet velocity_to_keep = this->locally_relevant_dofs;
  //   velocity_to_keep.subtract_set(relevant_boundary_velocity_dofs);
  //   IndexSet position_to_keep = this->locally_relevant_dofs;
  //   position_to_keep.subtract_set(relevant_boundary_position_dofs);
  //   IndexSet to_keep = velocity_to_keep;
  //   to_keep.add_indices(position_to_keep.begin(), position_to_keep.end());

  //   auto tmp_constraints = constraints.get_view(to_keep);
  //   constraints.reinit(this->locally_owned_dofs,
  //   this->locally_relevant_dofs); constraints.close();
  //   constraints.merge(tmp_constraints);
  // }

  // {
  //   // This does not work either: (test for velocity only)
  //   // Keep everything (relevant) but the relevant boundary dofs
  //   IndexSet to_keep = this->locally_relevant_dofs;
  //   to_keep.subtract_set(relevant_boundary_velocity_dofs);
  //   AffineConstraints<double> tmp;
  //   tmp.copy_from(constraints);

  //   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  //   constraints.add_selected_constraints(tmp, to_keep);
  //   constraints.close();
  // }

  ///////////////////////////////////////////////////////////////////////////
  // Print the relevant dofs after removing the constraints.
  // No relevant velocity dof on the boundary should be constrained
  // for (unsigned int r = 0; r < this->mpi_size; ++r)
  // {
  //   MPI_Barrier(this->mpi_communicator);
  //   if (r == this->mpi_rank)
  //     for (unsigned int i = 0; i < n_dofs; ++i)
  //     {
  //       // Support points are defined only for relevant dofs
  //       if (!this->locally_relevant_dofs.is_element(i))
  //         continue;

  //       // Support points are not defined for the additional ghost lambda
  //       dofs if (additional_relevant_dofs.is_element(i))
  //         continue;

  //       if (relevant_boundary_velocity_dofs.is_element(i) ||
  //           relevant_boundary_position_dofs.is_element(i))
  //       {
  //         std::cout << "A: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << std::endl;
  //         AssertThrow(!constraints.is_constrained(i),
  //                     ExcMessage("Constrained dof remains"));
  //       }
  //       else
  //       {
  //         std::cout << "A: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << " (not u/x or not on boundary)" << std::endl;
  //       }
  //     }
  // }
  ///////////////////////////////////////////////////////////////////////////

  // Check consistency of constraints for RELEVANT (not active) dofs after
  // removing
  {
    const bool consistent = constraints.is_consistent_in_parallel(
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 this->locally_owned_dofs),
      // this->locally_relevant_dofs,
      DoFTools::extract_locally_active_dofs(this->dof_handler),
      this->mpi_communicator,
      true);
    AssertThrow(consistent,
                ExcMessage("Constraints are not consistent after removing"));
  }

  // Check that boundary dofs were correctly removed
  if (remove_velocity_constraints)
    for (const auto &dof : relevant_boundary_velocity_dofs)
      AssertThrow(
        !constraints.is_constrained(dof),
        ExcMessage(
          "On rank " + std::to_string(this->mpi_rank) +
          " : "
          "Velocity dof " +
          std::to_string(dof) +
          " on a boundary with weak no-slip remains "
          "constrained by a boundary condition. This can happen if "
          "velocity dofs lying on both the cylinder and a face "
          "boundary have conflicting prescribed boundary conditions."));
  if (remove_position_constraints)
    for (const auto &dof : relevant_boundary_position_dofs)
      AssertThrow(
        !constraints.is_constrained(dof),
        ExcMessage(
          "On rank " + std::to_string(this->mpi_rank) +
          " : "
          "Position dof " +
          std::to_string(dof) +
          " on a boundary with weak no-slip remains "
          "constrained by a boundary condition. This can happen if "
          "position dofs lying on both the cylinder and a face "
          "boundary have conflicting prescribed boundary conditions."));
}

template <int dim>
void FSISolver<dim>::create_solver_specific_zero_constraints()
{
  this->zero_constraints.close();

  // Merge the zero lambda constraints
  this->zero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
  {
    /** FIXME: Instead of dim = 3, the test should be whether dofs
     * belong to multiple boundaries, but for now this only happens for the
     * 3D fsi test case.
     */
    if (this->param.fsi.enable_coupling)
    {
      /**
       * Remove both position and velocity constraints on the moving boundary:
       *
       * - Position because it is coupled to the Lagrange multiplier.
       *   If the force-position constraints were handled with an
       *   AffineConstraints, this would be checked by the merge() and
       *   specifying "no_conflicts_allowed". But the constraints are enforced
       *   "by hand", so we have to manually check and remove constrained
       *   position dofs from adjacent faces.
       *
       * - Velocity because a Lagrange multiplier enforces no slip.
       *   If velocity is set by another constraint, the lambda will have
       *   garbage values since the constraint cannot be satisfied.
       */
      this->pcout << "Removing zero constraints on cylinder" << std::endl;
      remove_cylinder_velocity_constraints(this->zero_constraints, true, true);
    }
    else if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->zero_constraints, true, false);
    }
  }
}

template <int dim>
void FSISolver<dim>::create_solver_specific_nonzero_constraints()
{
  this->nonzero_constraints.close();

  // Merge the zero lambda constraints
  this->nonzero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
  {
    if (this->param.fsi.enable_coupling)
    {
      this->pcout << "Removing nonzero constraints on cylinder" << std::endl;
      remove_cylinder_velocity_constraints(this->nonzero_constraints,
                                           true,
                                           true);
    }
    else if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->nonzero_constraints,
                                           true,
                                           false);
    }
  }
}

template <int dim>
void FSISolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints are defined
  //
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);

  const unsigned int n_components   = this->ordering->n_components;
  auto              &coupling_table = this->coupling_table;
  coupling_table = Table<2, DoFTools::Coupling>(n_components, n_components);
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int d = 0; d < n_components; ++d)
    {
      coupling_table[c][d] = DoFTools::none;

      // u couples to all variables
      if (this->ordering->is_velocity(c))
        coupling_table[c][d] = DoFTools::always;

      // p couples to u and x
      if (this->ordering->is_pressure(c))
        if (this->ordering->is_velocity(d) || this->ordering->is_position(d))
          coupling_table[c][d] = DoFTools::always;

      // x couples to itself
      if (this->ordering->is_position(c) && this->ordering->is_position(d))
        coupling_table[c][d] = DoFTools::always;

      // Lambda couples only on the relevant boundary faces:
      // add these coupling on faces only, below.
    }

  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling_table,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);

  {
    // Manually add the lambda coupling on the relevant boundary faces
    const unsigned int n_dofs_per_face = fe->n_dofs_per_face();
    std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;
        face->get_dof_indices(face_dofs);
        for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
        {
          const unsigned int comp_i =
            fe->face_system_to_component_index(i_dof, i_face).first;
          const unsigned int d_i = comp_i - this->ordering->l_lower;

          if (this->ordering->is_lambda(comp_i))
            for (unsigned int j_dof = 0; j_dof < n_dofs_per_face; ++j_dof)
            {
              const unsigned int comp_j =
                fe->face_system_to_component_index(j_dof, i_face).first;

              // Lambda couples to u and x on faces where no-slip is enforced
              // weakly
              if (this->ordering->is_velocity(comp_j))
              {
                const unsigned int d_j = comp_j - this->ordering->u_lower;
                if (d_i == d_j)
                {
                  // Lambda couples to u and vice versa
                  dsp.add(face_dofs[i_dof], face_dofs[j_dof]);
                  dsp.add(face_dofs[j_dof], face_dofs[i_dof]);
                }
              }
              if (this->ordering->is_position(comp_j))
              {
                const unsigned int d_j = comp_j - this->ordering->x_lower;
                if (d_i == d_j)
                  // In the PDEs, lambda couples to x, but x does not couple to
                  // lambda. The x - lambda boundary coupling is applied
                  // directly in the add_algebraic_position_coupling routines.
                  dsp.add(face_dofs[i_dof], face_dofs[j_dof]);
              }
            }
        }
      }
  }

  /**
   * FIXME: Still testing for better coupling betwen x and lambda.
   */
  switch (this->param.debug.fsi_coupling_option)
  {
    case 0:
    {
      // Add the position-lambda couplings explicitly
      // In a first (current) naive approach, each position dof is coupled to
      // all lambda dofs on cylinder
      // Note : this is highly inefficient, and will be removed atfer testing
      // for alternatives.
      for (const auto &[position_dof, d] : coupled_position_dofs)
        for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
          dsp.add(position_dof, lambda_dof);
      break;
    }
    case 1:
    {
      if (has_chunk_of_cylinder)
      {
        // Add position-lambda couplings *only* for local master position dofs
        for (unsigned int d = 0; d < dim; ++d)
        {
          // Couple the local master position dof in dimension d to the lambda
          // of same dimension (one-way coupling)
          for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
            dsp.add(local_position_master_dofs[d], lambda_dof);
        }
        // Couple the remaining owned position dofs to the local master dof
        // This coupling is local to the partition (also a one-way coupling)
        for (const auto &[position_dof, d] : coupled_position_dofs)
          dsp.add(position_dof, local_position_master_dofs[d]);
      }
      break;
    }
    case 2:
    {
      if (has_chunk_of_cylinder)
      {
        if (has_global_master_position_dofs)
        {
          // Add position-lambda couplings *only* for global master pos dofs
          for (unsigned int d = 0; d < dim; ++d)
          {
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
              dsp.add(global_position_master_dofs[d], lambda_dof);
          }
        }
        else
        {
          // If this rank does not own the global master position dofs,
          // couple its position dofs to it
          for (unsigned int d = 0; d < dim; ++d)
          {
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            dsp.add(local_position_master_dofs[d],
                    global_position_master_dofs[d]);
          }
        }

        // For all ranks with a piece of cylinder, couple the remaining owned
        // position dofs to the local master dof On the rank with the global
        // master, the local masters are also the global. This coupling is local
        // to the partition (also a one-way coupling)
        for (const auto &[position_dof, d] : coupled_position_dofs)
          dsp.add(position_dof, local_position_master_dofs[d]);
      }
      break;
    }
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);

  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "Matrix has " << this->system_matrix.n_nonzero_elements()
                << " nnz" << std::endl;
}

template <int dim>
void FSISolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           *this->fixed_mapping,
                           *this->moving_mapping,
                           *this->quadrature,
                           *this->face_quadrature,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(fe->n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &FSISolver::assemble_local_matrix,
                  &FSISolver::copy_local_to_global_matrix,
                  scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_matrix();
}

template <int dim>
void FSISolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      this->source_terms,
                      this->exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const double JxW_moving = scratch_data.JxW_moving[q];
    const double JxW_fixed  = scratch_data.JxW_fixed[q];

    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];
    const auto &phi_p      = scratch_data.phi_p[q];
    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);
    const double present_pressure_values =
      scratch_data.present_pressure_values[q];

    const auto &dxdt = scratch_data.present_mesh_velocity_values[q];

    const auto u_ale            = present_velocity_values - dxdt;
    const auto u_dot_grad_u_ale = present_velocity_gradients * u_ale;

    // BDF: current dudt
    Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i = scratch_data.components[i];
      const bool         i_is_u =
        const_ordering.u_lower <= comp_i && comp_i < const_ordering.u_upper;
      const bool i_is_p = comp_i == const_ordering.p_lower;
      const bool i_is_x =
        const_ordering.x_lower <= comp_i && comp_i < const_ordering.x_upper;

      const auto &phi_u_i      = phi_u[i];
      const auto &grad_phi_u_i = grad_phi_u[i];
      const auto &div_phi_u_i  = div_phi_u[i];
      const auto &phi_p_i      = phi_p[i];
      const auto &grad_phi_x_i = grad_phi_x[i];
      const auto &div_phi_x_i  = div_phi_x[i];

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int comp_j = scratch_data.components[j];
        bool               assemble =
          this->coupling_table[comp_i][comp_j] == DoFTools::always;
        if (!assemble)
          continue;
        const bool j_is_u =
          const_ordering.u_lower <= comp_j && comp_j < const_ordering.u_upper;
        const bool j_is_p = comp_j == const_ordering.p_lower;
        const bool j_is_x =
          const_ordering.x_lower <= comp_j && comp_j < const_ordering.x_upper;

        const auto &phi_u_j            = phi_u[j];
        const auto &grad_phi_u_j       = grad_phi_u[j];
        const auto &div_phi_u_j        = div_phi_u[j];
        const auto &phi_p_j            = phi_p[j];
        const auto &phi_x_j            = phi_x[j];
        const auto &grad_phi_x_j       = grad_phi_x[j];
        const auto  trace_grad_phi_x_j = trace(grad_phi_x_j);
        const auto &div_phi_x_j        = div_phi_x[j];

        double local_flow_matrix_ij = 0.;
        double local_ps_matrix_ij   = 0.;

        if (i_is_u)
        {
          if (j_is_u)
          {
            // Time derivative, convection (including ALE) and diffusion
            local_flow_matrix_ij +=
              phi_u_i * (bdf_c0 * phi_u_j + grad_phi_u_j * u_ale +
                         present_velocity_gradients * phi_u_j) +
              nu * scalar_product(grad_phi_u_j, grad_phi_u_i);
          }

          if (j_is_p)
          {
            // Pressure gradient
            local_flow_matrix_ij += -div_phi_u_i * phi_p_j;
          }

          if (j_is_x)
          {
            // Variation of all momentum terms on moving mesh w.r.t. position
            // Does not include variation of the velocity source term : this
            // term is only needed for manufactured solutions, and slows down
            // the assembly. Ommitting it does not seem to affect convergence of
            // the nonlinear solver.
            const Tensor<2, dim> d_grad_phi_u = -grad_phi_u_i * grad_phi_x_j;
            const auto           gradu_dot_grad_phi_x_j =
              present_velocity_gradients * grad_phi_x_j;
            local_flow_matrix_ij +=
              phi_u_i * ((dudt + u_dot_grad_u_ale) * trace_grad_phi_x_j -
                         gradu_dot_grad_phi_x_j * u_ale -
                         present_velocity_gradients * bdf_c0 * phi_x_j) +
              nu * scalar_product(-gradu_dot_grad_phi_x_j, grad_phi_u_i) +
              nu * scalar_product(present_velocity_gradients,
                                  d_grad_phi_u +
                                    grad_phi_u_i * trace_grad_phi_x_j) -
              present_pressure_values *
                (trace(d_grad_phi_u) + div_phi_u_i * trace_grad_phi_x_j);
          }
        }

        if (i_is_p)
        {
          if (j_is_u)
          {
            // Continuity : variation w.r.t. u
            local_flow_matrix_ij += -phi_p_i * div_phi_u_j;
          }

          if (j_is_x)
          {
            // Continuity : variation w.r.t. x
            // Does not include variation of the pressure source term : this
            // term is only needed for manufactured solutions, and slows down
            // the assembly. Ommitting it does not seem to affect convergence of
            // the nonlinear solver.
            const auto gradu_dot_grad_phi_x_j =
              present_velocity_gradients * grad_phi_x_j;
            local_flow_matrix_ij +=
              phi_p_i * (trace(gradu_dot_grad_phi_x_j) -
                         present_velocity_divergence * trace_grad_phi_x_j);
          }
        }

        //
        // Pseudo-solid
        //
        if (i_is_x && j_is_x)
        {
          // Linear elasticity
          local_ps_matrix_ij +=
            lame_lambda * div_phi_x_j * div_phi_x_i +
            lame_mu * scalar_product((grad_phi_x_j + transpose(grad_phi_x_j)),
                                     grad_phi_x_i);
        }

        local_flow_matrix_ij *= JxW_moving;
        local_ps_matrix_ij *= JxW_fixed;
        local_matrix(i, j) += local_flow_matrix_ij + local_ps_matrix_ij;
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
        for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
        {
          const double face_JxW_moving =
            scratch_data.face_JxW_moving[i_face][q];

          const auto &phi_u = scratch_data.phi_u_face[i_face][q];
          const auto &phi_x = scratch_data.phi_x_face[i_face][q];
          const auto &phi_l = scratch_data.phi_l_face[i_face][q];

          const auto &present_u =
            scratch_data.present_face_velocity_values[i_face][q];
          const auto &present_w =
            scratch_data.present_face_mesh_velocity_values[i_face][q];
          const auto  u_ale = present_u - present_w;
          const auto &present_l =
            scratch_data.present_face_lambda_values[i_face][q];

          for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
          {
            const unsigned int comp_i = scratch_data.components[i];
            const bool         i_is_u = const_ordering.u_lower <= comp_i &&
                                comp_i < const_ordering.u_upper;
            const bool i_is_l = const_ordering.l_lower <= comp_i &&
                                comp_i < const_ordering.l_upper;

            const auto &phi_u_i = phi_u[i];
            const auto &phi_l_i = phi_l[i];

            const auto lambda_dot_phi_u_i = present_l * phi_u_i;

            for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = scratch_data.components[j];
              const bool         j_is_u = const_ordering.u_lower <= comp_j &&
                                  comp_j < const_ordering.u_upper;
              const bool j_is_x = const_ordering.x_lower <= comp_j &&
                                  comp_j < const_ordering.x_upper;
              const bool j_is_l = const_ordering.l_lower <= comp_j &&
                                  comp_j < const_ordering.l_upper;

              const bool assemble = (i_is_u and (j_is_x or j_is_l)) or
                                    (i_is_l and (j_is_u or j_is_x));
              if (!assemble)
                continue;

              const auto &phi_u_j = phi_u[j];
              const auto &phi_x_j = phi_x[j];
              const auto &phi_l_j = phi_l[j];

              const double delta_dx_j = scratch_data.delta_dx[i_face][q][j];

              double local_matrix_ij = 0.;

              if (i_is_u && j_is_x)
              {
                local_matrix_ij += -lambda_dot_phi_u_i * delta_dx_j;
              }

              if (i_is_u && j_is_l)
              {
                local_matrix_ij += -phi_l_j * phi_u_i;
              }

              if (i_is_l && j_is_u)
              {
                local_matrix_ij += -phi_u_j * phi_l_i;
              }

              if (i_is_l && j_is_x)
              {
                local_matrix_ij +=
                  phi_l_i * (bdf_c0 * phi_x_j + u_ale * delta_dx_j);
              }

              local_matrix_ij *= face_JxW_moving;
              local_matrix(i, j) += local_matrix_ij;
            }
          }
        }
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void FSISolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                                    copy_data.local_dof_indices,
                                                    this->system_matrix);
}

template <int dim>
void FSISolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchData scratch_data(*this->ordering,
                           *fe,
                           *this->fixed_mapping,
                           *this->moving_mapping,
                           *this->quadrature,
                           *this->face_quadrature,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &FSISolver::assemble_local_matrix,
    &FSISolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    this->present_solution,
    this->evaluation_point,
    this->local_evaluation_point,
    this->mpi_communicator,
    this->param.output.output_dir,
    true,
    this->param.debug.analytical_jacobian_absolute_tolerance,
    this->param.debug.analytical_jacobian_relative_tolerance);

  this->pcout << "Max absolute error analytical vs fd matrix is "
              << errors.first << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > this->param.debug.analytical_jacobian_absolute_tolerance)
    this->pcout << "Max relative error analytical vs fd matrix is "
                << errors.second << std::endl;
}

template <int dim>
void FSISolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           *this->fixed_mapping,
                           *this->moving_mapping,
                           *this->quadrature,
                           *this->face_quadrature,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(fe->n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &FSISolver::assemble_local_rhs,
                  &FSISolver::copy_local_to_global_rhs,
                  scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_rhs();
}

template <int dim>
void FSISolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      this->source_terms,
                      this->exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;
  const SymmetricTensor<2, dim> identity_tensor = unit_symmetric_tensor<dim>();

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    //
    // Flow related data
    //
    const double JxW_moving = scratch_data.JxW_moving[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];
    const auto &present_mesh_velocity_values =
      scratch_data.present_mesh_velocity_values[q];
    const auto  &source_term_velocity = scratch_data.source_term_velocity[q];
    const auto  &source_term_pressure = scratch_data.source_term_pressure[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const auto &phi_p      = scratch_data.phi_p[q];
    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];

    //
    // Pseudo-solid related data
    //
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const double JxW_fixed = scratch_data.JxW_fixed[q];

    const auto &present_position_gradients =
      scratch_data.present_position_gradients[q];
    const double present_displacement_divergence =
      trace(present_position_gradients);
    const auto present_strain =
      symmetrize(present_position_gradients) - identity_tensor;
    const double present_trace_strain =
      present_displacement_divergence - (double)dim;
    const auto &source_term_position = scratch_data.source_term_position[q];

    const auto u_dot_grad_u_ale =
      present_velocity_gradients *
      (present_velocity_values - present_mesh_velocity_values);
    const auto to_multiply_by_phi_u_i =
      (dudt + u_dot_grad_u_ale + source_term_velocity);

    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i = scratch_data.components[i];
      const bool         i_is_u =
        const_ordering.u_lower <= comp_i && comp_i < const_ordering.u_upper;
      const bool i_is_x =
        const_ordering.x_lower <= comp_i && comp_i < const_ordering.x_upper;

      const auto &phi_u_i      = phi_u[i];
      const auto &grad_phi_u_i = grad_phi_u[i];
      const auto &div_phi_u_i  = div_phi_u[i];
      const auto &phi_p_i      = phi_p[i];

      //
      // Flow residual
      //

      double local_rhs_flow_i =

        // Time derivative, convective acceleration and velocity source term
        -(phi_u_i * to_multiply_by_phi_u_i

          // Pressure gradient
          - div_phi_u_i * present_pressure_values

          // Continuity and pressure source term
          + phi_p_i * (-present_velocity_divergence + source_term_pressure));

      if (i_is_u)
      {
        // Diffusion : only compute double contraction if needed
        local_rhs_flow_i -=
          nu * scalar_product(present_velocity_gradients, grad_phi_u_i);
      }

      local_rhs_flow_i *= JxW_moving;

      //
      // Pseudo-solid
      //
      double local_rhs_ps_i = 0.;

      if (i_is_x)
      {
        local_rhs_ps_i -= (
          // Linear elasticity : only compute double contraction if needed
          lame_lambda * present_trace_strain * div_phi_x[i] +
          2. * lame_mu * scalar_product(present_strain, grad_phi_x[i])
          // Linear elasticity source term
          + phi_x[i] * source_term_position);
        local_rhs_ps_i *= JxW_fixed;
      }

      local_rhs(i) += local_rhs_flow_i + local_rhs_ps_i;
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
        if (face->boundary_id() == weak_no_slip_boundary_id)
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            //
            // Flow related data (no-slip)
            //
            const double face_JxW_moving =
              scratch_data.face_JxW_moving[i_face][q];
            const auto &phi_u = scratch_data.phi_u_face[i_face][q];
            const auto &phi_l = scratch_data.phi_l_face[i_face][q];

            const auto &present_u =
              scratch_data.present_face_velocity_values[i_face][q];
            const auto &present_w =
              scratch_data.present_face_mesh_velocity_values[i_face][q];
            const auto &present_l =
              scratch_data.present_face_lambda_values[i_face][q];
            const auto u_ale = present_u - present_w;

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
            {
              double local_rhs_i = 0.;

              const unsigned int comp_i = scratch_data.components[i];
              const bool         i_is_u = this->ordering->is_velocity(comp_i);
              const bool         i_is_l = this->ordering->is_lambda(comp_i);

              if (i_is_u)
                local_rhs_i -= -(phi_u[i] * present_l);

              if (i_is_l)
                local_rhs_i -= -u_ale * phi_l[i];

              local_rhs_i *= face_JxW_moving;
              local_rhs(i) += local_rhs_i;
            }
          }

        /**
         * Open boundary condition with prescribed manufactured solution.
         * Applied on moving mesh.
         */
        if (this->param.fluid_bc.at(scratch_data.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            const double face_JxW_moving =
              scratch_data.face_JxW_moving[i_face][q];
            const auto &n = scratch_data.face_normals_moving[i_face][q];

            const auto &grad_u_exact =
              scratch_data.exact_face_velocity_gradients[i_face][q];
            const double p_exact =
              scratch_data.exact_face_pressure_values[i_face][q];

            // This is an open boundary condition, not a traction,
            // involving only grad_u_exact and not the symmetric gradient.
            const auto quasisigma_dot_n = -p_exact * n + nu * grad_u_exact * n;

            const auto &phi_u = scratch_data.phi_u_face[i_face][q];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
              local_rhs(i) -= -phi_u[i] * quasisigma_dot_n * face_JxW_moving;
          }
      }
    }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void FSISolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                                    copy_data.local_dof_indices,
                                                    this->system_rhs);
}

template <int dim>
void FSISolver<dim>::add_algebraic_position_coupling_to_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Apply constraints to matrix");

  //
  // Add algebraic constraints position-lambda
  //
  std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
    position_row_entries;
  // Get row entries for each pos_dof
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
    {
      std::vector<LA::ConstMatrixIterator> row_entries;
      for (auto it = this->system_matrix.begin(pos_dof);
           it != this->system_matrix.end(pos_dof);
           ++it)
        row_entries.push_back(it);
      position_row_entries[pos_dof] = row_entries;
    }

  switch (this->param.debug.fsi_coupling_option)
  {
    case 0:
    {
      // Constrain matrix
      // Constrain each owned coupled position dof to the sum of lambdas
      for (const auto &[pos_dof, d] : coupled_position_dofs)
        if (this->locally_owned_dofs.is_element(pos_dof))
        {
          for (auto it : position_row_entries.at(pos_dof))
            this->system_matrix.set(pos_dof, it->column(), 0.0);

          // Set constraint row: x_i - sum_j c_ij * lambda_j = 0
          this->system_matrix.set(pos_dof, pos_dof, 1.);
          for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
            this->system_matrix.set(pos_dof, lambda_dof, -weight);
        }
      break;
    }
    case 1:
    {
      if (has_chunk_of_cylinder)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master

        // Get the rows for the local master position dofs
        std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
          master_position_row_entries;
        for (unsigned int d = 0; d < dim; ++d)
        {
          std::vector<LA::ConstMatrixIterator> row_entries;
          for (auto it =
                 this->system_matrix.begin(local_position_master_dofs[d]);
               it != this->system_matrix.end(local_position_master_dofs[d]);
               ++it)
            row_entries.push_back(it);
          master_position_row_entries[local_position_master_dofs[d]] =
            row_entries;
        }

        for (unsigned int d = 0; d < dim; ++d)
        {
          for (auto it :
               master_position_row_entries.at(local_position_master_dofs[d]))
            this->system_matrix.set(local_position_master_dofs[d],
                                    it->column(),
                                    0.0);

          // Set constraint row: x_master - sum_j c_j * lambda_j = 0
          this->system_matrix.set(local_position_master_dofs[d],
                                  local_position_master_dofs[d],
                                  1.);
          for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
            this->system_matrix.set(local_position_master_dofs[d],
                                    lambda_dof,
                                    -weight);
        }
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
          {
            for (auto it : position_row_entries.at(pos_dof))
              this->system_matrix.set(pos_dof, it->column(), 0.0);

            this->system_matrix.set(pos_dof, pos_dof, 1.);
            this->system_matrix.set(pos_dof,
                                    local_position_master_dofs[d],
                                    -1.);
          }
      }
      break;
    }
    case 2:
    {
      if (has_chunk_of_cylinder)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master

        if (has_global_master_position_dofs)
        {
          // Get the rows for the global master position dofs
          std::map<types::global_dof_index,
                   std::vector<LA::ConstMatrixIterator>>
            master_position_row_entries;
          for (unsigned int d = 0; d < dim; ++d)
          {
            std::vector<LA::ConstMatrixIterator> row_entries;
            for (auto it =
                   this->system_matrix.begin(global_position_master_dofs[d]);
                 it != this->system_matrix.end(global_position_master_dofs[d]);
                 ++it)
              row_entries.push_back(it);
            master_position_row_entries[global_position_master_dofs[d]] =
              row_entries;
          }

          for (unsigned int d = 0; d < dim; ++d)
          {
            for (auto it :
                 master_position_row_entries.at(global_position_master_dofs[d]))
              this->system_matrix.set(global_position_master_dofs[d],
                                      it->column(),
                                      0.0);

            // Set constraint row: x_master - sum_j c_j * lambda_j = 0
            this->system_matrix.set(global_position_master_dofs[d],
                                    global_position_master_dofs[d],
                                    1.);
            for (const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
              this->system_matrix.set(global_position_master_dofs[d],
                                      lambda_dof,
                                      -weight);
          }
        }
        else
        {
          // Get the rows for the local master position dofs
          std::map<types::global_dof_index,
                   std::vector<LA::ConstMatrixIterator>>
            master_position_row_entries;
          for (unsigned int d = 0; d < dim; ++d)
          {
            std::vector<LA::ConstMatrixIterator> row_entries;
            for (auto it =
                   this->system_matrix.begin(local_position_master_dofs[d]);
                 it != this->system_matrix.end(local_position_master_dofs[d]);
                 ++it)
              row_entries.push_back(it);
            master_position_row_entries[local_position_master_dofs[d]] =
              row_entries;
          }

          // Constrain local to global
          for (unsigned int d = 0; d < dim; ++d)
          {
            for (auto it :
                 master_position_row_entries.at(local_position_master_dofs[d]))
              this->system_matrix.set(local_position_master_dofs[d],
                                      it->column(),
                                      0.0);

            this->system_matrix.set(local_position_master_dofs[d],
                                    local_position_master_dofs[d],
                                    1.);
            this->system_matrix.set(local_position_master_dofs[d],
                                    global_position_master_dofs[d],
                                    -1.);
          }
        }

        // In any case, set remaining pos dofs to local master
        // On the rank with the global master, the local is also the global
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
          {
            for (auto it : position_row_entries.at(pos_dof))
              this->system_matrix.set(pos_dof, it->column(), 0.0);

            this->system_matrix.set(pos_dof, pos_dof, 1.);
            this->system_matrix.set(pos_dof,
                                    local_position_master_dofs[d],
                                    -1.);
          }
      }
      break;
    }
  }

  this->system_matrix.compress(VectorOperation::insert);
}

template <int dim>
void FSISolver<dim>::add_algebraic_position_coupling_to_rhs()
{
  // Set RHS to zero for coupled position dofs
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
      this->system_rhs(pos_dof) = 0.;

  this->system_rhs.compress(VectorOperation::insert);
}

/**
 * Compute integral of lambda (fluid force), compare to position dofs
 */
template <int dim>
void FSISolver<dim>::compare_forces_and_position_on_obstacle() const
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_JxW_values);

  // Compute integral of lambda on owned boundary
  const unsigned int n_faces_q_points = this->face_quadrature->size();
  std::vector<types::global_dof_index> face_dofs(fe->n_dofs_per_face());

  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  Tensor<1, dim>    cylinder_displacement_local, max_diff_local;
  std::vector<bool> first_computed_displacement(dim, true);

  for (auto cell : this->dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() &&
            face->boundary_id() == weak_no_slip_boundary_id)
        {
          fe_face_values.reinit(cell, i_face);

          // Increment lambda integral
          fe_face_values[lambda_extractor].get_function_values(
            this->present_solution, lambda_values);
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);

          /**
           * Cylinder is rigid, so all displacements should be identical for a
           * given component. If first position dof, save displacement,
           * otherwise compare with saved displacement.
           */
          face->get_dof_indices(face_dofs);

          for (unsigned int i_dof = 0; i_dof < fe->n_dofs_per_face(); ++i_dof)
            if (this->locally_owned_dofs.is_element(face_dofs[i_dof]))
            {
              const unsigned int comp =
                fe->face_system_to_component_index(i_dof, i_face).first;
              if (this->ordering->is_position(comp))
              {
                const unsigned int d = comp - this->ordering->x_lower;

                if (first_computed_displacement[d])
                {
                  // Save displacement
                  first_computed_displacement[d] = false;
                  cylinder_displacement_local[d] =
                    this->present_solution[face_dofs[i_dof]] -
                    this->initial_positions.at(face_dofs[i_dof])[d];
                }
                else
                {
                  // Compare with saved displacement
                  const double displ =
                    this->present_solution[face_dofs[i_dof]] -
                    this->initial_positions.at(face_dofs[i_dof])[d];
                  max_diff_local[d] =
                    std::max(max_diff_local[d],
                             cylinder_displacement_local[d] - displ);
                }
              }
            }
        }
      }

  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);

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
    /**
     * Cylinder displacement is trivially 0 on processes which do not own a
     * part of the boundary, and is nontrivial otherwise.     Taking the max
     * to synchronize does not work because displacement can be negative.
     * Instead, we take the max while preserving the sign.
     */
    MPI_Allreduce(&cylinder_displacement_local[d],
                  &cylinder_displacement[d],
                  1,
                  MPI_DOUBLE,
                  mpi_maxabs,
                  this->mpi_communicator);

    // Take the max between all max differences disp_i - disp_j
    // for x_i and x_j both on the cylinder.
    // Checks that all displacement are identical.
    max_diff[d] =
      Utilities::MPI::max(max_diff_local[d], this->mpi_communicator);

    // Check that the ratio of both terms in the position
    // boundary condition is -spring_constant
    if (std::abs(cylinder_displacement[d]) > 1e-10)
      ratio[d] = lambda_integral[d] / cylinder_displacement[d];
  }

  if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  {
    this->pcout << std::endl;
    this->pcout << std::scientific << std::setprecision(8) << std::showpos;
    this->pcout
      << "Checking consistency between lambda integral and position BC:"
      << std::endl;
    this->pcout << "Integral of lambda on cylinder is " << lambda_integral
                << std::endl;
    this->pcout << "Prescribed displacement        is " << cylinder_displacement
                << std::endl;
    this->pcout << "                         Ratio is " << ratio
                << " (expected: " << -this->param.fsi.spring_constant << ")"
                << std::endl;
    this->pcout << "Max diff between displacements is " << max_diff
                << std::endl;
    this->pcout << std::endl;
  }

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
      std::abs(ratio[d] - (-this->param.fsi.spring_constant));

    if (absolute_error <= 1e-6)
      continue;

    const double relative_error =
      absolute_error / this->param.fsi.spring_constant;
    AssertThrow(relative_error <= 1e-2,
                ExcMessage("Ratio integral vs displacement values is not -k"));
  }
}

template <int dim>
void FSISolver<dim>::check_velocity_boundary() const
{
  // Check difference between uh and dxhdt
  double l2_local = 0;
  double li_local = 0;

  FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);
  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values);

  const unsigned int n_faces_q_points = this->face_quadrature->size();

  const auto &bdf_coefficients = this->time_handler.bdf_coefficients;

  std::vector<std::vector<Tensor<1, dim>>> position_values(
    bdf_coefficients.size(), std::vector<Tensor<1, dim>>(n_faces_q_points));
  std::vector<Tensor<1, dim>> mesh_velocity_values(n_faces_q_points);
  std::vector<Tensor<1, dim>> fluid_velocity_values(n_faces_q_points);
  Tensor<1, dim>              diff;

  for (auto cell : this->dof_handler.active_cell_iterators())
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
        fe_face_values[this->velocity_extractor].get_function_values(
          this->present_solution, fluid_velocity_values);
        fe_face_values_fixed[this->position_extractor].get_function_values(
          this->present_solution, position_values[0]);
        for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
          fe_face_values_fixed[this->position_extractor].get_function_values(
            this->previous_solutions[iBDF - 1], position_values[iBDF]);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          // Compute FE mesh velocity at node
          mesh_velocity_values[q] = 0;
          for (unsigned int iBDF = 0; iBDF < bdf_coefficients.size(); ++iBDF)
            mesh_velocity_values[q] +=
              bdf_coefficients[iBDF] * position_values[iBDF][q];

          diff = mesh_velocity_values[q] - fluid_velocity_values[q];

          // u_h - w_h
          l2_local += diff * diff * fe_face_values_fixed.JxW(q);
          li_local = std::max(li_local, std::abs(diff.norm()));
        }
      }
    }
  }

  const double l2_error =
    std::sqrt(Utilities::MPI::sum(l2_local, this->mpi_communicator));
  const double li_error = Utilities::MPI::max(li_local, this->mpi_communicator);

  if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  {
    this->pcout << "Checking no-slip enforcement on cylinder:" << std::endl;
    this->pcout << "||uh - wh||_L2 = " << l2_error << std::endl;
    this->pcout << "||uh - wh||_Li = " << li_error << std::endl;
  }

  if (!this->param.debug.fsi_apply_erroneous_coupling)
  {
    AssertThrow(l2_error < 1e-12,
                ExcMessage("L2 norm of uh - wh is too large : " +
                           std::to_string(l2_error)));
    AssertThrow(li_error < 1e-12,
                ExcMessage("Linf norm of uh - wh is too large : " +
                           std::to_string(li_error)));
  }
}

template <int dim>
void FSISolver<dim>::check_manufactured_solution_boundary()
{
  Tensor<1, dim> lambdaMMS_integral, lambdaMMS_integral_local;
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  Tensor<1, dim> pns_integral, pns_integral_local;
  lambdaMMS_integral_local = 0;
  lambda_integral_local    = 0;
  pns_integral_local       = 0;

  const double rho = this->param.physical_properties.fluids[0].density;
  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;
  const double mu = nu * rho;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  Tensor<1, dim>              lambda_MMS;
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  //
  // First compute integral over cylinder of lambda_MMS
  //
  for (auto cell : this->dof_handler.active_cell_iterators())
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
        fe_face_values[lambda_extractor].get_function_values(
          this->present_solution, lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
          const auto        normal_to_solid = -fe_face_values.normal_vector(q);

          const double p_MMS =
            this->exact_solution->value(qpoint, this->ordering->p_lower);

          std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
            this->exact_solution)
            ->lagrange_multiplier(qpoint, mu, normal_to_solid, lambda_MMS);

          // Increment the integrals of lambda:

          // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
          lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

          /**
           * This is int lambda := int sigma(u_MMS, p_MMS) cdot normal_to_fluid
           *                                                   -normal_to_solid
           */
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);

          // Increment integral of p * n_solid
          pns_integral_local += p_MMS * normal_to_solid * fe_face_values.JxW(q);
        }
      }
    }
  }

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambdaMMS_integral[d] =
      Utilities::MPI::sum(lambdaMMS_integral_local[d], this->mpi_communicator);
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
  }
  pns_integral =
    Utilities::MPI::sum(pns_integral_local, this->mpi_communicator);

  // // Reference solution for int_Gamma p*n_solid dx is - k * d * f(t).
  // Tensor<1, dim> translation;
  // translation[0] = 0.1;
  // translation[1] = 0.05;
  const Tensor<1, dim> ref_pns;
  // const Tensor<1, dim> ref_pns =
  //   -param.fsi.spring_constant * translation *
  //   std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
  //     exact_solution)->mms.exact_mesh_position->time_function->value(this->time_handler.current_time);
  // const double err_pns = (ref_pns - pns_integral).norm();
  const double err_pns = -1.;

  //
  // Check x_MMS
  //
  Tensor<1, dim> x_MMS;
  double         max_x_error = 0.;
  for (auto cell : this->dof_handler.active_cell_iterators())
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

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint_fixed =
            fe_face_values_fixed.quadrature_point(q);

          for (unsigned int d = 0; d < dim; ++d)
            x_MMS[d] = this->exact_solution->value(qpoint_fixed,
                                                   this->ordering->x_lower + d);

          const Tensor<1, dim> ref =
            -1. / this->param.fsi.spring_constant * lambdaMMS_integral;
          const double err = ((x_MMS - qpoint_fixed) - ref).norm();
          max_x_error      = std::max(max_x_error, err);
        }
      }
    }
  }

  //
  // Check u_MMS
  //
  Tensor<1, dim> u_MMS, w_MMS;
  double         max_u_error = -1;
  // for (auto cell : this->dof_handler.active_cell_iterators())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;
  //   for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  //   {
  //     const auto &face = cell->face(i_face);
  //     if (face->at_boundary() && face->boundary_id() == boundary_id)
  //     {
  //       fe_face_values.reinit(cell, i_face);
  //       fe_face_values_fixed.reinit(cell, i_face);

  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
  //         const Point<dim> &qpoint_fixed  =
  //         fe_face_values_fixed.quadrature_point(q);

  //         for (unsigned int d = 0; d < dim; ++d)
  //         {
  //           u_MMS[d] = solution_fun.value(qpoint, u_lower + d);
  //           w_MMS[d] = mesh_velocity_fun.value(qpoint_fixed, x_lower + d);
  //         }

  //         const double err = (u_MMS - w_MMS).norm();
  //         // std::cout << "u_MMS & w_MMS at quad node are " << u_MMS << " ,
  //         "
  //         << w_MMS << " - norm diff = " << err << std::endl; max_u_error =
  //         std::max(max_u_error, err);
  //       }
  //     }
  //   }
  // }

  // if(VERBOSE)
  // {
  this->pcout << std::endl;
  this->pcout << "Checking manufactured solution for k = "
              << this->param.fsi.spring_constant << " :" << std::endl;
  this->pcout << "integral lambda         = " << lambda_integral << std::endl;
  this->pcout << "integral lambdaMMS      = " << lambdaMMS_integral
              << std::endl;
  this->pcout << "integral pMMS * n_solid = " << pns_integral << std::endl;
  this->pcout << "reference: -k*d*f(t)    = " << ref_pns
              << " - err = " << err_pns << std::endl;
  this->pcout << "max error on (x_MMS -    X0) vs -1/k * integral lambda = "
              << max_x_error << std::endl;
  this->pcout << "max error on  u_MMS          vs w_MMS                  = "
              << max_u_error << std::endl;
  this->pcout << std::endl;
  // }
}

template <int dim>
void FSISolver<dim>::compute_lambda_error_on_boundary(
  double         &lambda_l2_error,
  double         &lambda_linf_error,
  Tensor<1, dim> &error_on_integral)
{
  double lambda_l2_local   = 0;
  double lambda_linf_local = 0;

  Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local,
    exact_integral_local;
  lambda_integral_local = 0;
  exact_integral_local  = 0;

  const double rho = this->param.physical_properties.fluids[0].density;
  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;
  const double mu = nu * rho;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);
  Tensor<1, dim>              diff, exact;

  // std::ofstream out("normals.pos");
  // out << "View \"normals\" {\n";

  for (auto cell : this->dof_handler.active_cell_iterators())
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
        fe_face_values[lambda_extractor].get_function_values(
          this->present_solution, lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint         = fe_face_values.quadrature_point(q);
          const auto        normal_to_mesh = fe_face_values.normal_vector(q);
          const auto        normal_to_solid = -normal_to_mesh;

          // Careful:
          // int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
          //                                                   =
          //                                             normal_to_mesh
          //                                                   =
          //                                            -normal_to_solid
          //
          // Got to take the consistent normal to compare int lambda_h with
          // solution.
          //
          // Solution<dim> computes lambda_exact = - sigma cdot ns, where n is
          // expected to be the normal to the SOLID.

          // out << "VP(" << qpoint[0] << "," << qpoint[1] << "," << 0. <<
          // "){"
          //   << normal[0] << "," << normal[1] << "," << 0. << "};\n";

          // exact_solution is a pointer to base class Function<dim>,
          // so we have to ruse to use the specific function for lambda.
          std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
            this->exact_solution)
            ->lagrange_multiplier(qpoint, mu, normal_to_solid, exact);

          diff = lambda_values[q] - exact;

          lambda_l2_local += diff * diff * fe_face_values.JxW(q);
          lambda_linf_local =
            std::max(lambda_linf_local, std::abs(diff.norm()));

          // Increment the integral of lambda
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
          exact_integral_local += exact * fe_face_values.JxW(q);
        }
      }
    }
  }

  // out << "};\n";
  // out.close();

  lambda_l2_error =
    Utilities::MPI::sum(lambda_l2_local, this->mpi_communicator);
  lambda_l2_error = std::sqrt(lambda_l2_error);

  lambda_linf_error =
    Utilities::MPI::max(lambda_linf_local, this->mpi_communicator);

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
    exact_integral[d] =
      Utilities::MPI::sum(exact_integral_local[d], this->mpi_communicator);
    error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
  }
}

template <int dim>
void FSISolver<dim>::compute_solver_specific_errors()
{
  double         l2_l = 0., li_l = 0.;
  Tensor<1, dim> error_on_integral;
  this->compute_lambda_error_on_boundary(l2_l, li_l, error_on_integral);
  // linf_error_Fx = std::max(linf_error_Fx, error_on_integral[0]);
  // linf_error_Fy = std::max(linf_error_Fy, error_on_integral[1]);

  const double t = this->time_handler.current_time;
  for (auto &[norm, handler] : this->error_handlers)
  {
    if (norm == VectorTools::L2_norm)
      handler->add_error("l", l2_l, t);
    if (norm == VectorTools::Linfty_norm)
      handler->add_error("l", li_l, t);

    if (this->param.fsi.compute_error_on_forces)
    {
      // The error on the forces is |F_h - F_exact|, there is no need to
      // distinguish between L^p norms.
      for (unsigned int d = 0; d < dim; ++d)
        handler->add_error("F_comp" + std::to_string(d),
                           error_on_integral[d],
                           t);
    }
  }
}

template <int dim>
void FSISolver<dim>::output_results()
{
  if (this->param.output.write_results)
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
    data_out.attach_dof_handler(this->dof_handler);
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
    mesh_velocity.reinit(this->locally_owned_dofs, this->mpi_communicator);
    IndexSet disp_dofs =
      DoFTools::extract_dofs(this->dof_handler, this->position_mask);

    for (const auto &i : disp_dofs)
      if (this->locally_owned_dofs.is_element(i))
        mesh_velocity[i] =
          this->time_handler.compute_time_derivative(i,
                                                     this->present_solution,
                                                     this->previous_solutions);
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
void FSISolver<dim>::compute_forces(const bool export_table)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  for (auto cell : this->dof_handler.active_cell_iterators())
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
        fe_face_values[lambda_extractor].get_function_values(
          this->present_solution, lambda_values);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
      }
    }
  }

  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);

  // const double rho = param.physical_properties.fluids[0].density;
  // const double U   = boundary_description.U;
  // const double D   = boundary_description.D;
  // const double factor = 1. / (0.5 * rho * U * U * D);

  //
  // Forces on the cylinder are the NEGATIVE of the integral of lambda
  //
  this->forces_table.add_value("time", this->time_handler.current_time);
  this->forces_table.add_value("CFx", -lambda_integral[0]);
  this->forces_table.add_value("CFy", -lambda_integral[1]);
  if constexpr (dim == 3)
  {
    this->forces_table.add_value("CFz", -lambda_integral[2]);
  }

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "Computed forces: " << -lambda_integral << std::endl;

  if (export_table && this->param.output.write_results && this->mpi_rank == 0)
  {
    std::ofstream outfile(this->param.output.output_dir + "forces.txt");
    this->forces_table.write_text(outfile);
  }
}

template <int dim>
void FSISolver<dim>::write_cylinder_position(const bool export_table)
{
  Tensor<1, dim> average_position, position_integral_local;
  double         boundary_measure_local = 0.;

  FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values |
                                           update_normal_vectors);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  std::vector<Tensor<1, dim>> position_values(n_faces_q_points);

  for (auto cell : this->dof_handler.active_cell_iterators())
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
        fe_face_values_fixed[this->position_extractor].get_function_values(
          this->present_solution, position_values);

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
    Utilities::MPI::sum(boundary_measure_local, this->mpi_communicator);
  for (unsigned int d = 0; d < dim; ++d)
    average_position[d] =
      1. / boundary_measure *
      Utilities::MPI::sum(position_integral_local[d], this->mpi_communicator);

  cylinder_position_table.add_value("time", this->time_handler.current_time);
  cylinder_position_table.add_value("xc", average_position[0]);
  cylinder_position_table.add_value("yc", average_position[1]);
  if constexpr (dim == 3)
    cylinder_position_table.add_value("zc", average_position[2]);

  if (export_table && this->param.output.write_results && this->mpi_rank == 0)
  {
    std::ofstream outfile(this->param.output.output_dir +
                          "cylinder_center.txt");
    cylinder_position_table.write_text(outfile);
  }
}

template <int dim>
void FSISolver<dim>::solver_specific_post_processing()
{
  // output_results();

  if (this->param.mms_param.enable)
  {
    // compute_errors();

    if (this->param.debug.fsi_check_mms_on_boundary)
      check_manufactured_solution_boundary();
  }

  // Check position - lambda coupling if coupled
  if (this->param.fsi.enable_coupling)
    compare_forces_and_position_on_obstacle();

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
template class FSISolver<2>;
template class FSISolver<3>;