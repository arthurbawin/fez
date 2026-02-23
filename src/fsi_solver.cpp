
#include <compare_matrix.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <fe_simplex_p_with_3d_hp.h>
#include <fsi_solver.h>
#include <lagrange_multiplier_tools.h>
#include <linear_solver.h>
#include <mapping_fe_field_hp2.h>
#include <mesh.h>
#include <mesh_and_dof_tools.h>
#include <post_processing_tools.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
FSISolverLessLambda<dim>::FSISolverLessLambda(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, true>(param)
  , all_lambda_accumulators(dim)
{
  if (param.finite_elements.use_quads)
  {
    fe_with_lambda = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,      // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree),            // Pressure
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim, // Position
      FE_Q<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier
    fe_without_lambda = std::make_shared<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,
      FE_Q<dim>(param.finite_elements.pressure_degree),
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim,
      FE_Nothing<dim>() ^ dim);
  }
  else
  {
    if constexpr (dim == 2)
    {
      fe_with_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
          dim,                                                   // Velocity
        FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^
          dim, // Position
        FE_SimplexP<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
          dim); // Lagrange multiplier
      fe_without_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim,
        FE_Nothing<dim>(ReferenceCells::get_simplex<dim>()) ^ dim);
    }
    else
    {
      fe_with_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^
          dim, // Velocity
        FE_SimplexP_3D_hp<dim>(
          param.finite_elements.pressure_degree), // Pressure
        FE_SimplexP_3D_hp<dim>(param.finite_elements.mesh_position_degree) ^
          dim, // Position
        FE_SimplexP_3D_hp<dim>(
          param.finite_elements.no_slip_lagrange_mult_degree) ^
          dim); // Lagrange multiplier
      fe_without_lambda = std::make_shared<FESystem<dim>>(
        FE_SimplexP_3D_hp<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP_3D_hp<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP_3D_hp<dim>(param.finite_elements.mesh_position_degree) ^
          dim,
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
  fixed_mapping_collection.push_back(*this->fixed_mapping);
  fixed_mapping_collection.push_back(*this->fixed_mapping);
  quadrature_collection.push_back(*this->quadrature);
  quadrature_collection.push_back(*this->quadrature);
  face_quadrature_collection.push_back(*this->face_quadrature);
  face_quadrature_collection.push_back(*this->face_quadrature);

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
    this->exact_solution =
      std::make_shared<FSISolverLessLambda<dim>::MMSSolution>(
        this->time_handler.current_time, *this->ordering, param.mms);

    // Create the source term function for the given MMS and override source
    // terms
    this->source_terms =
      std::make_shared<FSISolverLessLambda<dim>::MMSSourceTerm>(
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
    this->source_terms = std::make_shared<FSISolverLessLambda<dim>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
void FSISolverLessLambda<dim>::MMSSourceTerm::vector_value(
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
void FSISolverLessLambda<dim>::setup_dofs()
{
  TimerOutput::Scope t(this->computing_timer, "Setup");

  auto &comm = this->mpi_communicator;

  // Mark the cells on which the Lagrange multiplier is defined,
  // based on if they have a vertex touching the boundary.
  // See also comments in incompressible_ns_solver_lambda.
  {
    std::set<Point<dim>, PointComparator<dim>> vertices_on_boundary =
      get_mesh_vertices_on_boundary(this->dof_handler,
                                    weak_no_slip_boundary_id);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
    {
      cell->set_material_id(without_lambda_domain_id);
      if (cell->is_locally_owned())
        cell->set_active_fe_index(index_fe_without_lambda);

      for (const auto v : cell->vertex_indices())
        if (vertices_on_boundary.count(cell->vertex(v)) > 0)
        {
          cell->set_material_id(with_lambda_domain_id);
          if (cell->is_locally_owned())
            cell->set_active_fe_index(index_fe_with_lambda);
          break;
        }
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

  // Initialize mesh position directly from the triangulation.
  // The parallel vector storing the mesh position is local_evaluation_point,
  // because this is the one to modify when computing finite differences.

  // FIXME: does get_position_vector work for hp context?
  // Use interpolate instead
  VectorTools::interpolate(fixed_mapping_collection,
                           this->dof_handler,
                           FixedMeshPosition<dim>(this->ordering->x_lower,
                                                  this->ordering->n_components),
                           this->local_evaluation_point,
                           this->position_mask);
  // VectorTools::get_position_vector(*fixed_mapping,
  //                                  dof_handler,
  //                                  local_evaluation_point,
  //                                  position_mask);
  this->local_evaluation_point.compress(VectorOperation::insert);
  this->evaluation_point = this->local_evaluation_point;

  // Also store them in initial_positions, for postprocessing:
  this->initial_positions =
    DoFTools::map_dofs_to_support_points(fixed_mapping_collection,
                                         this->dof_handler,
                                         this->position_mask);

  // Create the solution-dependent mapping
  this->moving_mapping =
    std::make_shared<MappingFEFieldHp2<dim, dim, LA::ParVectorType>>(
      this->dof_handler,
      fixed_mapping_collection,
      quadrature_collection,
      face_quadrature_collection,
      this->evaluation_point,
      this->position_mask);

  moving_mapping_collection.push_back(*this->moving_mapping);
  moving_mapping_collection.push_back(*this->moving_mapping);

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
void FSISolverLessLambda<dim>::reset_solver_specific_data()
{
  // Position - lambda constraints
  for (auto &vec : lambda_integral_coeffs)
    vec.clear();
  lambda_integral_coeffs.clear();
  coupled_position_dofs.clear();
  has_local_position_master       = false;
  has_local_lambda_accumulator    = false;
  has_global_master_position_dofs = false;
  for (unsigned int d = 0; d < dim; ++d)
  {
    local_position_master_dofs[d]  = numbers::invalid_unsigned_int;
    global_position_master_dofs[d] = numbers::invalid_unsigned_int;
    local_lambda_accumulators[d]   = numbers::invalid_unsigned_int;
    all_lambda_accumulators[d].clear();
  }
  // hp_dof_identities.clear();
}

template <int dim>
void FSISolverLessLambda<dim>::create_lagrange_multiplier_constraints()
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

        // If using the coupling method with accumulators and if dof is a local
        // accumulator, do not constrain it
        bool skip_dof = false;
        if (this->param.debug.fsi_coupling_option == 3)
          for (unsigned int d = 0; d < dim; ++d)
            if (local_lambda_accumulators[d] == dof)
            {
              skip_dof = true;
              break;
            }

        if (!skip_dof)
        {
          unsigned int comp =
            fe_with_lambda->system_to_component_index(i).first;
          if (this->ordering->is_lambda(comp))
            if (this->locally_relevant_dofs.is_element(dof))
              if (!relevant_boundary_dofs.is_element(dof))
                lambda_constraints.constrain_dof_to_zero(dof);
        }
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
        DoFTools::map_dofs_to_support_points(fixed_mapping_collection,
                                             this->dof_handler);

      {
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
              outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2]
                      << "){1};" << std::endl;
          }
        outfile << "};" << std::endl;
        outfile.close();
      }

      // Print unconstrained
      {
        std::ofstream outfile(this->param.output.output_dir +
                              "unconstrained_lambda_dofs_proc" +
                              std::to_string(this->mpi_rank) + ".pos");
        outfile << "View \"unconstrained_lambda_dofs_proc" << this->mpi_rank
                << "\"{" << std::endl;
        for (const auto dof : lambda_dofs)
          if (!lambda_constraints.is_constrained(dof))
          {
            const Point<dim> &pt = support_points.at(dof);
            if constexpr (dim == 2)
              outfile << "SP(" << pt[0] << "," << pt[1] << ", 0.){1};"
                      << std::endl;
            else
              outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2]
                      << "){1};" << std::endl;
          }
        outfile << "};" << std::endl;
        outfile.close();
      }
    }
  }
}

template <int dim>
void FSISolverLessLambda<dim>::create_hp_line_dof_identities()
{
  TimerOutput::Scope t(this->computing_timer, "Hp line identities");

  const unsigned int deg_p   = this->param.finite_elements.pressure_degree;
  const unsigned int deg_x   = this->param.finite_elements.mesh_position_degree;
  const unsigned int u_lower = this->ordering->u_lower;
  const unsigned int x_lower = this->ordering->x_lower;
  std::vector<std::set<types::global_dof_index>> velocity_dofs_to_match(dim);
  std::set<types::global_dof_index>              pressure_dofs_to_match;
  std::vector<std::set<types::global_dof_index>> position_dofs_to_match(dim);

  std::map<types::global_dof_index, Point<dim>> support_points =
    DoFTools::map_dofs_to_support_points(fixed_mapping_collection,
                                         this->dof_handler);

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

      // Simply grab all the velocity, pressure and position dofs on the lambda
      // cells, without distinguishing if they're associated to vertices, lines,
      // faces or interiors. The hp dof identities that were correctly applied
      // already matched the global dof index of some of these dofs. Then we'll
      // see if the adjacent dofs at the same support point have a different
      // global dof index or not, and if so match their values as a constraint.
      for (unsigned int i = 0; i < fe_lambda.n_dofs_per_cell(); ++i)
      {
        const unsigned int comp = fe_lambda.system_to_component_index(i).first;
        if (this->ordering->is_velocity(comp))
          velocity_dofs_to_match[comp - u_lower].insert(dof_indices[i]);
        // If pressure is linear, there is no line dof to match
        if (deg_p > 1 && this->ordering->is_pressure(comp))
          pressure_dofs_to_match.insert(dof_indices[i]);
        if (deg_x > 1 && this->ordering->is_position(comp))
          position_dofs_to_match[comp - x_lower].insert(dof_indices[i]);
      }
    }
  }

  // Safety checks
  if constexpr (running_in_debug_mode())
  {
    // Check that the velocity and position dofs are contiguous
    std::vector<std::vector<std::set<types::global_dof_index>>> to_check = {
      velocity_dofs_to_match, position_dofs_to_match};

    for (const auto &vec : to_check)
    {
      std::vector<std::vector<types::global_dof_index>> vec_dofs(dim);
      for (unsigned int d = 0; d < dim; ++d)
      {
        vec_dofs[d] =
          std::vector<types::global_dof_index>(vec[d].begin(), vec[d].end());
        Assert(vec_dofs[0].size() == vec_dofs[d].size(), ExcInternalError());
      }
      for (unsigned int i = 0; i < vec_dofs[0].size(); ++i)
        for (unsigned int d = 1; d < dim; ++d)
          Assert(vec_dofs[d][i] == vec_dofs[d - 1][i] + 1, ExcInternalError());
    }
  }

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
                  hp_dof_identities.insert({dof, dof_indices[i]});
                  break;
                }
          if (this->ordering->is_pressure(comp))
            for (const auto dof : pressure_dofs_to_match)
              if (dof != dof_indices[i])
                if (support_points.at(dof).distance_square(pt_i) < 1e-15)
                {
                  hp_dof_identities.insert({dof, dof_indices[i]});
                  break;
                }
          if (this->ordering->is_position(comp))
            for (const auto dof : position_dofs_to_match[comp - x_lower])
              if (dof != dof_indices[i])
                if (support_points.at(dof).distance_square(pt_i) < 1e-15)
                {
                  hp_dof_identities.insert({dof, dof_indices[i]});
                  break;
                }
        }
      }
    }
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
void FSISolverLessLambda<dim>::create_position_lagrange_mult_coupling_data()
{
  /**
   * Get the owned position dofs on the cylinder.
   * We might be missing some owned dofs, e.g., on boundary edges for which
   * no cell face touches the cylinder in 3D. Also add them here.
   */
  IndexSet local_position_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->position_mask,
                                    {weak_no_slip_boundary_id});
  local_position_dofs = local_position_dofs & this->locally_owned_dofs;
  {
    std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 local_position_dofs.get_index_vector());
    for (const auto &vec : gathered_pos_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_owned_dofs.is_element(dof))
          local_position_dofs.add_index(dof);
  }

  const bool has_owned_position_dofs_on_boundary =
    local_position_dofs.n_elements() > 0;

  /**
   * The coupling options 0, 1 and 2 all couple one or more position dofs
   * to *all* lambda dofs, which requires adding these dofs as ghosts on
   * the partitions which own a piece of the cylinder.
   */
  const bool requires_lambda_ghosts = this->param.debug.fsi_coupling_option < 3;

  if (requires_lambda_ghosts)
  {
    // Collect the (relevant) lambda dofs
    IndexSet boundary_lambda_dofs =
      DoFTools::extract_boundary_dofs(this->dof_handler,
                                      this->lambda_mask,
                                      {weak_no_slip_boundary_id});

    std::vector<std::vector<types::global_dof_index>> gathered =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 boundary_lambda_dofs.get_index_vector());

    std::vector<types::global_dof_index> all_boundary_lambda_dofs;
    for (const auto &vec : gathered)
      all_boundary_lambda_dofs.insert(all_boundary_lambda_dofs.end(),
                                      vec.begin(),
                                      vec.end());

    if (has_owned_position_dofs_on_boundary)
    {
      this->locally_relevant_dofs.add_indices(all_boundary_lambda_dofs.begin(),
                                              all_boundary_lambda_dofs.end());
      this->locally_relevant_dofs.compress();
    }

    // Reinitialize the ghosted parallel vectors with the additional ghosts.
    this->reinit_vectors();
  }

  if (this->param.debug.fsi_coupling_option > 0)
  {
    // Set the local_position_master_dofs
    // Simply take the first owned position dofs on the cylinder
    // Here it's assumed that local_position_dofs is organized as
    // x_0, y_0, z_0, x_1, y_1, z_1, ...,
    // and we take the first dim.
    const auto pos_index_vector = local_position_dofs.get_index_vector();
    if (pos_index_vector.size() > 0)
    {
      has_local_position_master = true;
      AssertThrow(pos_index_vector.size() >= dim,
                  ExcMessage(
                    "This partition has position dofs on the cylinder, but has "
                    "less than dim position dofs, which should not happen. It "
                    "should have n * dim position dofs on this boundary."));
      for (unsigned int d = 0; d < dim; ++d)
      {
        local_position_master_dofs[d] = pos_index_vector[d];
        Assert(this->locally_owned_dofs.is_element(pos_index_vector[d]),
               ExcMessage("Local position master dof " +
                          std::to_string(pos_index_vector[d]) +
                          " is not owned. This should not happen!"));
      }
    }

    if constexpr (running_in_debug_mode())
    {
      n_ranks_with_position_master =
        Utilities::MPI::sum(has_local_position_master ? 1 : 0,
                            this->mpi_communicator);
      this->pcout << "There are " << n_ranks_with_position_master
                  << " ranks with local position master dofs" << std::endl;
    }
  }

  switch (this->param.debug.fsi_coupling_option)
  {
    case 1:
      // Couple all local position dofs to local master
      // Then couple local master to all (global) lambda dofs
      // Nothing else to do : the local position master was set,
      // and the lambda dofs on cylinder have been added as ghosts
      break;
    case 2:
    {
      // Couple all local position dofs to local master, then to global master
      // Couple global master to all (global) lambda dofs
      // We still have to determine the global position master :
      // it is simply taken as the local master on the lowest rank
      // among those with a local position master:
      const unsigned int candidate_rank =
        has_local_position_master ? this->mpi_rank :
                                    std::numeric_limits<unsigned int>::max();
      const unsigned int owner_rank =
        Utilities::MPI::min(candidate_rank, this->mpi_communicator);
      has_global_master_position_dofs = (this->mpi_rank == owner_rank);

      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "Global position master is on rank " << owner_rank
                    << std::endl;

      // Set the global position dofs and broadcast them to all ranks
      for (unsigned int d = 0; d < dim; ++d)
      {
        global_position_master_dofs[d] = numbers::invalid_unsigned_int;
        if (has_global_master_position_dofs)
          global_position_master_dofs[d] = local_position_master_dofs[d];
      }

      Utilities::MPI::broadcast(global_position_master_dofs.data(),
                                dim,
                                owner_rank,
                                this->mpi_communicator);

      if constexpr (running_in_debug_mode())
      {
        for (unsigned int d = 0; d < dim; ++d)
          Assert(global_position_master_dofs[d] !=
                   numbers::invalid_unsigned_int,
                 ExcMessage(
                   "The global position master is invalid after broadcast"));
      }

      break;
    }
    case 3:
    {
      // Couple all local position dofs to local master,
      // and use local accumulators to sum the lambda integrals on owned faces
      // Then, couple each local master to all lambda accumulators

      // Normally, this coupling would require adding "dim" dofs per
      // partition to store the integral of each component (accumulators).
      // But we can ruse a little bit: since we are already storing more lambda
      // dofs than required (even in hp mode), we can just use "dim" of these
      // useless dofs to store the force on this proc,
      // while being careful not to affect the no-slip constraint.
      // The global dof indices of these dofs are stored in
      // local_lambda_accumulators.

      // Set the accumulator dofs from among the unused lambda dofs:
      // This rank should have a lambda accumulator if it has at least
      // one owned face on the cylinder
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          if (cell->at_boundary())
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary() &&
                  face->boundary_id() == weak_no_slip_boundary_id)
              {
                has_local_lambda_accumulator = true;
                goto reduce_accumulators;
              }
    reduce_accumulators:
      n_ranks_with_lambda_accumulator =
        Utilities::MPI::sum(has_local_lambda_accumulator ? 1 : 0,
                            this->mpi_communicator);
      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "There are " << n_ranks_with_lambda_accumulator
                    << " ranks with local lambda accumulators" << std::endl;

      for (unsigned int d = 0; d < dim; ++d)
        local_lambda_accumulators[d] = numbers::invalid_unsigned_int;

      // Now actually set the dofs for the accumulators, if possible
      if (has_local_lambda_accumulator)
      {
        // We can take as accumulators the first dim lambda dofs on this
        // partition that would otherwise be constrained to zero

        // Impact on the no-slip enforcement:
        // The lambda equations on the relevant boundaries are assembled by
        // looping over the cell dofs, not only the face dofs, so if we choose
        // lambda dofs from a cell adjacent to these boundaries, this will
        // affect the no-slip enforcement. To avoid that we can take lambda dofs
        // from a cell that touches the boundary by a vertex only, and take dofs
        // which are not shared which a directly adjacent cell to the boundary.

        std::vector<types::global_dof_index> face_dofs(
          fe_with_lambda->n_dofs_per_face());
        unsigned int n_accumulators = 0;
        for (const auto &cell : this->dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            if (cell_has_lambda(cell))
            {
              bool skip_cell = false;

              // Skip if this cell altogether if it touches the target boundary
              // with a face
              for (const auto &face : cell->face_iterators())
                if (face->at_boundary() &&
                    face->boundary_id() == weak_no_slip_boundary_id)
                {
                  skip_cell = true;
                  break;
                };

              if (!skip_cell)
                for (const auto i_face : cell->face_indices())
                {
                  const auto &face      = cell->face(i_face);
                  bool        skip_face = false;

                  // Skip face if neighbouring cell through this face touches
                  // the target boundary
                  auto neighbor = cell->neighbor(i_face);
                  if (neighbor->state() == IteratorState::IteratorStates::valid)
                    for (const auto neighbor_i_face : neighbor->face_indices())
                    {
                      const auto &neighbor_face =
                        neighbor->face(neighbor_i_face);
                      if (neighbor_face->at_boundary() &&
                          neighbor_face->boundary_id() ==
                            weak_no_slip_boundary_id)
                      {
                        skip_face = true;
                        break;
                      }
                    }

                  if (!skip_face)
                  {
                    face->get_dof_indices(face_dofs, index_fe_with_lambda);
                    for (unsigned int i = 0; i < face_dofs.size(); ++i)
                    {
                      types::global_dof_index dof = face_dofs[i];
                      unsigned int            comp =
                        fe_with_lambda
                          ->face_system_to_component_index(i, i_face)
                          .first;
                      unsigned int base =
                        fe_with_lambda
                          ->face_system_to_component_index(i, i_face)
                          .second;

                      // FIXME:
                      // Hardcoded to the first P2 dof of the first face whose
                      // neighbouring cell does not touch the boundary Its shape
                      // functions index (base) is 2 in 2D (P2 dof on a line)
                      // and 3 in 3D (P2 dof on triangle). This is for simplices
                      // only...
                      unsigned int target_base = (dim == 2) ? 2 : 3;
                      if (base == target_base)
                        if (this->ordering->is_lambda(comp))
                          /**
                           * The accumulator must be an owned dof. It might not
                           * be possible to assign an accumulator, based on the
                           * partition used, see the assert below.
                           */
                          if (this->locally_owned_dofs.is_element(dof))
                          {
                            local_lambda_accumulators[n_accumulators++] = dof;
                            if (n_accumulators == dim)
                              goto accumulators_found;
                          }
                    }
                  }
                }
            }
      accumulators_found:
        if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        {
          if constexpr (dim == 2)
          {
            std::cout << "Set lambda accumulator at dof "
                      << local_lambda_accumulators[0] << " - "
                      << local_lambda_accumulators[1] << std::endl;
          }
          else
          {
            std::cout << "Set lambda accumulator at dof "
                      << local_lambda_accumulators[0] << " - "
                      << local_lambda_accumulators[1] << " - "
                      << local_lambda_accumulators[2] << std::endl;
          }
        }
        for (unsigned int d = 0; d < dim; ++d)
          /**
           * On some weird partitions (typically with "too many" MPI procs),
           * there are owned cells on the boundary, but no owned lambda dof that
           * can be used to accumulate the local integral.
           *
           * This is technically an issue with the coupling method itself, as
           * accumulators should be defined on their own, without using
           * otherwise unused lambda dofs. Note that allowing accumulators on a
           * non-boundary face of elements touching the boundary is not
           * sufficient, because in some cases the *only* owned lambda dofs are
           * on a boundary face, and there is really no way to define an
           * accumulator without modifying the flow solution.
           */
          AssertThrow(
            local_lambda_accumulators[d] != numbers::invalid_unsigned_int,
            ExcMessage(
              "\n This rank owns at least one cell touching a boundary where "
              "no-slip should be enforced with a Lagrange multiplier (lambda). "
              "But it doesn't own any lambda degree of freedom that can be "
              "used to safely accumulate the force integral on this rank (all "
              "its lambda dofs are either ghosts, or owned but on the "
              "prescribed "
              "boundary)."
              "\n\n This can happen on somewhat pathological mesh partitions "
              "with isolated elements touching the boundary, and it probably "
              "indicates that the mesh has too few elements for the number of "
              "MPI processes used."
              "\n\n To go around this issue, try running with another number "
              "of MPI processes."));

#if defined(DEBUG_PRINTS)
        {
          // Print accumulators
          std::map<types::global_dof_index, Point<dim>> support_points =
            DoFTools::map_dofs_to_support_points(fixed_mapping_collection,
                                                 this->dof_handler);

          {
            std::ofstream outfile(this->param.output.output_dir +
                                  "accumulators_dofs" +
                                  std::to_string(this->mpi_rank) + ".pos");
            outfile << "View \"accumulators_dofs" << this->mpi_rank << "\"{"
                    << std::endl;
            for (const auto dof : local_lambda_accumulators)
            {
              const Point<dim> &pt = support_points.at(dof);
              if constexpr (dim == 2)
                outfile << "SP(" << pt[0] << "," << pt[1] << ", 0.){1};"
                        << std::endl;
              else
                outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2]
                        << "){1};" << std::endl;
            }
            outfile << "};" << std::endl;
            outfile.close();
          }
        }
#endif
      }

      // Lastly, add accumulators as ghosts on all procs with a position master
      // and reinit the parallel vectors with these additional ghosts
      {
        std::vector<std::array<types::global_dof_index, dim>> gathered =
          Utilities::MPI::all_gather(this->mpi_communicator,
                                     local_lambda_accumulators);
        // all_lambda_accumulators.resize(dim);
        for (unsigned int rank = 0; rank < gathered.size(); ++rank)
          for (unsigned int d = 0; d < dim; ++d)
            if (gathered[rank][d] != numbers::invalid_unsigned_int)
              all_lambda_accumulators[d].push_back(gathered[rank][d]);
      }

      if constexpr (running_in_debug_mode())
      {
        // Check that there are indeed n_ranks_with_lambda_accumulator dofs for
        // each dimension
        for (unsigned int d = 0; d < dim; ++d)
          Assert(all_lambda_accumulators[d].size() ==
                   n_ranks_with_lambda_accumulator,
                 ExcMessage(
                   "There are " +
                   std::to_string(all_lambda_accumulators[d].size()) +
                   "lambda accumulators in the local vector on this rank, "
                   "but there are " +
                   std::to_string(n_ranks_with_lambda_accumulator) +
                   " ranks with an accumulator."));
      }

      if (has_local_lambda_accumulator)
      {
        for (unsigned int d = 0; d < dim; ++d)
          this->locally_relevant_dofs.add_indices(
            all_lambda_accumulators[d].begin(),
            all_lambda_accumulators[d].end());
        this->locally_relevant_dofs.compress();
      }

      this->reinit_vectors();

      break;
    }
    default:
      DEAL_II_NOT_IMPLEMENTED();
  }

  /**
   * Compute the weights c_ij and identify the constrained position DOFs.
   * Done only once as cylinder is rigid and those weights will not change.
   */
  std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

  hp::FEFaceValues hp_fe_face_values_fixed(fixed_mapping_collection,
                                           *fe,
                                           face_quadrature_collection,
                                           update_values | update_JxW_values);

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
     */
    if (cell->is_locally_owned())
    {
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        const unsigned int fe_index = cell->active_fe_index();

        hp_fe_face_values_fixed.reinit(cell, face);

        const FEFaceValues<dim> &fe_face_values_fixed =
          hp_fe_face_values_fixed.get_present_fe_values();
        const FESystem<dim> &active_fe = fe_face_values_fixed.get_fe();

        const unsigned int n_dofs_per_face = active_fe.n_dofs_per_face();
        std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);

        face->get_dof_indices(face_dofs, fe_index);

        for (unsigned int q = 0; q < this->face_quadrature->size(); ++q)
        {
          const double JxW = fe_face_values_fixed.JxW(q);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
          {
            const unsigned int comp =
              active_fe.face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which
            // contribute to the integral on this element
            // FIXME: This should never happen, to check and remove
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
                active_fe.face_to_cell_index(i_dof, i_face);
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
   * Once again we might be missing some owned coupled dofs, on boundary edges
   * Add them here.
   * They are added only if they are already relevant (does not add ghosts).
   * FIXME: Can this be done only once instead?
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
                              moving_mapping_collection,
                              face_quadrature_collection,
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

  /**
   * Either store the local coefficients of the lambda integral (option 3),
   * or reduce them into a single vector
   */
  lambda_integral_coeffs.resize(dim);
  if (this->param.debug.fsi_coupling_option == 3)
  {
    for (unsigned int d = 0; d < dim; ++d)
      lambda_integral_coeffs[d] =
        std::vector<std::pair<unsigned int, double>>(coeffs[d].begin(),
                                                     coeffs[d].end());
  }
  else
  {
    for (unsigned int d = 0; d < dim; ++d)
    {
      const auto gathered = Utilities::MPI::all_gather(
        this->mpi_communicator,
        std::vector<std::pair<types::global_dof_index, double>>(
          coeffs[d].begin(), coeffs[d].end()));

      std::map<types::global_dof_index, double> coeffs_map;

      // Accumulate contributions
      for (const auto &vec : gathered)
        for (const auto &[lambda_dof, weight] : vec)
          coeffs_map[lambda_dof] += weight;

      lambda_integral_coeffs[d].insert(lambda_integral_coeffs[d].end(),
                                       coeffs_map.begin(),
                                       coeffs_map.end());
    }
  }
}

template <int dim>
void FSISolverLessLambda<dim>::remove_cylinder_velocity_constraints(
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

// template <int dim>
// void FSISolverLessLambda<dim>::add_hp_identities_constraints(
//   AffineConstraints<double> &constraints) const
// {
//   // AssertThrow(false, ExcMessage("Not needed"));

//   AffineConstraints<double> hp_constraints(this->locally_owned_dofs,
//                                            this->locally_relevant_dofs);

//   // Apply dof_1 = dof_2 to each pair of identified dofs
//   for (const auto &[dof1, dof2] : hp_dof_identities)
//     hp_constraints.add_constraint(dof1, {{dof2, 1.}}, 0.);
//   hp_constraints.close();

//   // See also comments in incompressible_ns_solver_lambda.cpp
//   constraints.merge(
//     hp_constraints,
//     AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
// }

template <int dim>
void FSISolverLessLambda<dim>::create_solver_specific_zero_constraints()
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

  // Add the hp dof identities as constraints
  // This won't be required as soon as the line dof identities are applied in
  // deal.II (in dof_handler.distribute_dofs())
  // add_hp_identities_constraints(this->zero_constraints);
}

template <int dim>
void FSISolverLessLambda<dim>::create_solver_specific_nonzero_constraints()
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

  // Add the hp dof identities as constraints
  // This won't be required as soon as the line dof identities are applied in
  // deal.II (in dof_handler.distribute_dofs())
  // add_hp_identities_constraints(this->nonzero_constraints);
}

template <int dim>
void FSISolverLessLambda<dim>::create_sparsity_pattern()
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

  const unsigned int s0 = dsp.n_nonzero_elements();
  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "DSP has " << dsp.n_nonzero_elements() << " nnz"
                << std::endl;

  {
    // Manually add the lambda coupling on the relevant boundary faces
    std::vector<types::global_dof_index> cell_dofs;
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        // Add coupling based on cell, rather than based on faces.
        // This is because in the assembly, we loop on the cell dofs
        // even for face terms, as the FEFaceValues functions run from
        // 0 to n_dofs_per_cell even on faces.
        const unsigned int fe_index        = cell->active_fe_index();
        const auto        &active_fe       = this->dof_handler.get_fe(fe_index);
        const unsigned int n_dofs_per_cell = active_fe.n_dofs_per_cell();
        cell_dofs.resize(n_dofs_per_cell);
        cell->get_dof_indices(cell_dofs);

        for (unsigned int i_dof = 0; i_dof < n_dofs_per_cell; ++i_dof)
        {
          const unsigned int comp_i =
            active_fe.system_to_component_index(i_dof).first;

          if (this->ordering->is_lambda(comp_i))
            for (unsigned int j_dof = 0; j_dof < n_dofs_per_cell; ++j_dof)
            {
              const unsigned int comp_j =
                active_fe.system_to_component_index(j_dof).first;

              // Lambda couples to u and x on faces where no-slip is enforced
              // weakly
              if (this->ordering->is_velocity(comp_j))
              {
                // Lambda couples to u and vice versa
                dsp.add(cell_dofs[i_dof], cell_dofs[j_dof]);
                dsp.add(cell_dofs[j_dof], cell_dofs[i_dof]);
              }
              if (this->ordering->is_position(comp_j))
              {
                // In the PDEs, lambda couples to x, but x does not couple to
                // lambda. The x - lambda boundary coupling is applied
                // directly in the add_algebraic_position_coupling routines.
                dsp.add(cell_dofs[i_dof], cell_dofs[j_dof]);
              }
            }
        }
      }
  }

  const unsigned int s1 = dsp.n_nonzero_elements();
  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "DSP has " << dsp.n_nonzero_elements() << " nnz - "
                << s1 - s0 << std::endl;

  // Add the couplings on the cylinder depending on the chosen coupling scheme
  // Regardless of the method, couple position dofs to local master if there is
  // one Local position masters are already coupled to themselves from the
  // coupling table
  if (has_local_position_master)
    for (const auto &[position_dof, d] : coupled_position_dofs)
      dsp.add(position_dof, local_position_master_dofs[d]);

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
        for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
          dsp.add(position_dof, lambda_dof);
      break;
    }
    case 1:
    {
      // Add position-lambda couplings only for local master position dofs
      if (has_local_position_master)
        for (unsigned int d = 0; d < dim; ++d)
          // Couple the local master position dof in dimension d to the lambda
          // of same dimension (one-way coupling)
          for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
            dsp.add(local_position_master_dofs[d], lambda_dof);
      break;
    }
    case 2:
    {
      if (has_local_position_master)
      {
        if (has_global_master_position_dofs)
          // Add position-lambda couplings *only* for global master pos dofs
          for (unsigned int d = 0; d < dim; ++d)
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
              dsp.add(global_position_master_dofs[d], lambda_dof);
        else
          // If this rank does not own the global master position dofs,
          // couple its position dofs to it
          for (unsigned int d = 0; d < dim; ++d)
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            dsp.add(local_position_master_dofs[d],
                    global_position_master_dofs[d]);
      }
      break;
    }
    case 3:
    {
      if (has_local_position_master)
        // Couple local position master to all lambda accumulators (one way)
        for (unsigned int d = 0; d < dim; ++d)
        {
          // dsp.add(local_position_master_dofs[d],
          // local_position_master_dofs[d]);
          for (const auto &lambda_accumulator : all_lambda_accumulators[d])
            dsp.add(local_position_master_dofs[d], lambda_accumulator);
        }

      if (has_local_lambda_accumulator)
        // Couple local lambda accumulators (one per dimension) to themselves
        // and to local lambdas of same dimension
        for (unsigned int d = 0; d < dim; ++d)
        {
          dsp.add(local_lambda_accumulators[d], local_lambda_accumulators[d]);
          for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
            dsp.add(local_lambda_accumulators[d], lambda_dof);
        }
      break;
    }
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }

  const unsigned int s2 = dsp.n_nonzero_elements();
  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "DSP has " << dsp.n_nonzero_elements() << " nnz - "
                << s2 - s1 << std::endl;

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
                << " nnz and size " << this->system_matrix.m() << " x "
                << this->system_matrix.n() << std::endl;
}

template <int dim>
void FSISolverLessLambda<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           fixed_mapping_collection,
                           moving_mapping_collection,
                           quadrature_collection,
                           face_quadrature_collection,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(*fe);

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
                  &FSISolverLessLambda::assemble_local_matrix,
                  &FSISolverLessLambda::copy_local_to_global_matrix,
                  scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_matrix();
}

template <int dim>
void FSISolverLessLambda<dim>::assemble_local_matrix(
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

  const unsigned int fe_index    = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto &local_matrix             = copy_data.matrices[fe_index];
  auto &local_dof_indices        = copy_data.local_dof_indices[fe_index];
  local_matrix                   = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  // std::cout << "Assembling on volume" << std::endl;
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
    // for (auto i : scratch_data.dofs_without_lambda)
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
      // for (auto j : scratch_data.dofs_without_lambda)
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
  cell->get_dof_indices(local_dof_indices);
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
  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void FSISolverLessLambda<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  const auto i = copy_data.last_active_fe_index;
  this->zero_constraints.distribute_local_to_global(
    copy_data.matrices[i], copy_data.local_dof_indices[i], this->system_matrix);
}

template <int dim>
void FSISolverLessLambda<dim>::compare_analytical_matrix_with_fd()
{
  // ScratchData scratch_data(*this->ordering,
  //                          *fe,
  //                          fixed_mapping_collection,
  //                          moving_mapping_collection,
  //                          quadrature_collection,
  //                          face_quadrature_collection,
  //                          this->time_handler.bdf_coefficients,
  //                          this->param);
  // CopyData    copy_data(*fe);

  // auto errors = Verification::compare_analytical_matrix_with_fd(
  //   this->dof_handler,
  //   fe->n_dofs_per_cell(),
  //   *this,
  //   &FSISolverLessLambda::assemble_local_matrix,
  //   &FSISolverLessLambda::assemble_local_rhs,
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
void FSISolverLessLambda<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  ScratchData scratch_data(*this->ordering,
                           *fe,
                           fixed_mapping_collection,
                           moving_mapping_collection,
                           quadrature_collection,
                           face_quadrature_collection,
                           this->time_handler.bdf_coefficients,
                           this->param);
  CopyData    copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &FSISolverLessLambda::assemble_local_rhs,
                  &FSISolverLessLambda::copy_local_to_global_rhs,
                  scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_rhs();
}

template <int dim>
void FSISolverLessLambda<dim>::assemble_local_rhs(
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

  const unsigned int fe_index    = cell->active_fe_index();
  copy_data.last_active_fe_index = fe_index;
  auto &local_rhs                = copy_data.vectors[fe_index];
  auto &local_dof_indices        = copy_data.local_dof_indices[fe_index];
  local_rhs                      = 0;

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
    // for (auto i : scratch_data.dofs_without_lambda)
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
        {
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
  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void FSISolverLessLambda<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  const auto i = copy_data.last_active_fe_index;
  this->zero_constraints.distribute_local_to_global(
    copy_data.vectors[i], copy_data.local_dof_indices[i], this->system_rhs);
}

template <int dim>
void FSISolverLessLambda<dim>::add_algebraic_position_coupling_to_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Apply constraints to matrix");

  //
  // Add algebraic constraints position-lambda
  //
  // Get row entries for each pos_dof
  std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
    position_rows, master_position_rows;
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
      position_rows[pos_dof] = get_matrix_rows(this->system_matrix, pos_dof);

  if (has_global_master_position_dofs)
  {
    for (unsigned int d = 0; d < dim; ++d)
      master_position_rows[global_position_master_dofs[d]] =
        get_matrix_rows(this->system_matrix, global_position_master_dofs[d]);
  }
  else if (has_local_position_master)
  {
    for (unsigned int d = 0; d < dim; ++d)
      master_position_rows[local_position_master_dofs[d]] =
        get_matrix_rows(this->system_matrix, local_position_master_dofs[d]);
  }

  switch (this->param.debug.fsi_coupling_option)
  {
    case 0:
    {
      // Constrain matrix
      // Constrain each owned coupled position dof to the sum of lambdas
      for (const auto &[pos_dof, d] : coupled_position_dofs)
        if (this->locally_owned_dofs.is_element(pos_dof))
          constrain_matrix_row(this->system_matrix,
                               pos_dof,
                               position_rows.at(pos_dof),
                               lambda_integral_coeffs[d]);
      break;
    }
    case 1:
    {
      if (has_local_position_master)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master
        for (unsigned int d = 0; d < dim; ++d)
          constrain_matrix_row(this->system_matrix,
                               local_position_master_dofs[d],
                               master_position_rows.at(
                                 local_position_master_dofs[d]),
                               lambda_integral_coeffs[d]);

        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);
      }
      break;
    }
    case 2:
    {
      if (has_local_position_master)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master

        if (has_global_master_position_dofs)
        {
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 global_position_master_dofs[d],
                                 master_position_rows.at(
                                   global_position_master_dofs[d]),
                                 lambda_integral_coeffs[d]);
        }
        else
        {
          // Constrain local to global
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 local_position_master_dofs[d],
                                 master_position_rows.at(
                                   local_position_master_dofs[d]),
                                 global_position_master_dofs[d],
                                 -1.);
        }

        // In any case, set remaining pos dofs to local master
        // On the rank with the global master, the local is also the global
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);
      }
      break;
    }
    case 3:
    {
      std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
        accumulator_rows;
      if (has_local_lambda_accumulator)
      {
        for (unsigned int d = 0; d < dim; ++d)
          if (local_lambda_accumulators[d] != numbers::invalid_unsigned_int)
          {
            AssertThrow(
              this->locally_owned_dofs.is_element(local_lambda_accumulators[d]),
              ExcMessage("Local accumulator is not locally owned " +
                         std::to_string(local_lambda_accumulators[d])));

            accumulator_rows[local_lambda_accumulators[d]] =
              get_matrix_rows(this->system_matrix,
                              local_lambda_accumulators[d]);
          }
      }

      if (has_local_position_master)
      {
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);

        // Couple local master to all lambda accumulators:
        // Constrain: x_master - sum_{i_rank} accumulator_{i_rank} = 0
        for (unsigned int d = 0; d < dim; ++d)
        {
          if (this->locally_owned_dofs.is_element(
                local_position_master_dofs[d]))
          {
            std::vector<std::pair<types::global_dof_index, double>>
              accumulator_coeffs;
            for (auto lambda_accumulator : all_lambda_accumulators[d])
              accumulator_coeffs.push_back({lambda_accumulator, 1.});
            constrain_matrix_row(this->system_matrix,
                                 local_position_master_dofs[d],
                                 master_position_rows.at(
                                   local_position_master_dofs[d]),
                                 accumulator_coeffs);
          }
        }
      }

      if (has_local_lambda_accumulator)
      {
        // Couple local accumulator to local lambda dofs
        // Constrain: local_accumulator - sum_j c_j * lambda_j = 0
        for (unsigned int d = 0; d < dim; ++d)
          if (this->locally_owned_dofs.is_element(local_lambda_accumulators[d]))
            constrain_matrix_row(this->system_matrix,
                                 local_lambda_accumulators[d],
                                 accumulator_rows.at(
                                   local_lambda_accumulators[d]),
                                 lambda_integral_coeffs[d]);
      }

      break;
    }
  }

  this->system_matrix.compress(VectorOperation::insert);
}

template <int dim>
void FSISolverLessLambda<dim>::add_algebraic_position_coupling_to_rhs()
{
  // Set RHS to zero for coupled position dofs
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
      this->system_rhs(pos_dof) = 0.;

  // Set RHS to zero for local lambda accumulator
  if (this->param.debug.fsi_coupling_option == 3)
    for (const auto accumulator_dof : local_lambda_accumulators)
      if (this->locally_owned_dofs.is_element(accumulator_dof))
        this->system_rhs(accumulator_dof) = 0.;

  this->system_rhs.compress(VectorOperation::insert);
}

/**
 * Compute integral of lambda (fluid force), compare to position dofs
 */
template <int dim>
void FSISolverLessLambda<dim>::compare_forces_and_position_on_obstacle() const
{
  AssertThrow(
    face_quadrature_collection.size() == 2,
    ExcMessage(
      "Assuming a face quadrature collection with exactly 2 entries."));
  AssertThrow(face_quadrature_collection[0].size() ==
                face_quadrature_collection[1].size(),
              ExcMessage(
                "Assuming the same two copies of the same face quadrature."));

  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  hp::FEFaceValues<dim> hp_fe_face_values(moving_mapping_collection,
                                          *fe,
                                          face_quadrature_collection,
                                          update_values | update_JxW_values);

  // Compute integral of lambda on owned boundary
  const unsigned int n_faces_q_points = face_quadrature_collection[0].size();
  std::vector<types::global_dof_index> face_dofs; //(fe->n_dofs_per_face());

  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  Tensor<1, dim>    cylinder_displacement_local, max_diff_local;
  std::vector<bool> is_first_computed_displacement(dim, true);

  for (auto cell : this->dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() &&
            face->boundary_id() == weak_no_slip_boundary_id)
        {
          const unsigned int fe_index = cell->active_fe_index();
          hp_fe_face_values.reinit(cell, i_face);
          const FEFaceValues<dim> &fe_face_values =
            hp_fe_face_values.get_present_fe_values();
          const FESystem<dim> &active_fe       = fe_face_values.get_fe();
          const unsigned int   n_dofs_per_face = active_fe.n_dofs_per_face();
          face_dofs.resize(n_dofs_per_face);

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
          face->get_dof_indices(face_dofs, fe_index);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
            if (this->locally_owned_dofs.is_element(face_dofs[i_dof]))
            {
              const unsigned int comp =
                active_fe.face_system_to_component_index(i_dof, i_face).first;
              if (this->ordering->is_position(comp))
              {
                const unsigned int d = comp - this->ordering->x_lower;

                if (is_first_computed_displacement[d])
                {
                  // Save displacement
                  is_first_computed_displacement[d] = false;
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
void FSISolverLessLambda<dim>::check_velocity_boundary() const
{
  ScratchData scratch_data(*this->ordering,
                           *fe,
                           fixed_mapping_collection,
                           moving_mapping_collection,
                           quadrature_collection,
                           face_quadrature_collection,
                           this->time_handler.bdf_coefficients,
                           this->param);

  LagrangeMultiplierTools::check_no_slip_on_boundary<dim>(
    this->param,
    scratch_data,
    this->dof_handler,
    this->evaluation_point,
    this->previous_solutions,
    this->source_terms,
    this->exact_solution,
    weak_no_slip_boundary_id);
}

template <int dim>
void FSISolverLessLambda<dim>::check_manufactured_solution_boundary()
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

  //         std::static_pointer_cast<FSISolverLessLambda<dim>::MMSSolution>(
  //           this->exact_solution)
  //           ->lagrange_multiplier(qpoint, mu, normal_to_solid, lambda_MMS);

  //         // Increment the integrals of lambda:

  //         // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
  //         lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

  //         /**
  //          * This is int lambda := int sigma(u_MMS, p_MMS) cdot
  //          normal_to_fluid
  //          * -normal_to_solid
  //          */
  //         lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);

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
  // //   std::static_pointer_cast<FSISolverLessLambda<dim>::MMSSolution>(
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
  //         max_x_error      = std::max(max_x_error, err);
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
  // ,
  // //         "
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
void FSISolverLessLambda<dim>::compute_lambda_error_on_boundary(
  double         &lambda_l2_error,
  double         &lambda_linf_error,
  Tensor<1, dim> &error_on_integral)
{
  double lambda_l2_local   = 0;
  double lambda_linf_local = 0;

  Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local,
    exact_integral_local;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  hp::FEFaceValues hp_fe_face_values(moving_mapping_collection,
                                     *fe,
                                     face_quadrature_collection,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

  Tensor<1, dim> diff, exact;

  for (auto cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;
    if (!cell_has_lambda(cell))
      continue;

    const auto         fe_index = cell->active_fe_index();
    const unsigned int n_faces_q_points =
      face_quadrature_collection[fe_index].size();
    std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        hp_fe_face_values.reinit(cell, i_face);
        const auto &fe_face_values = hp_fe_face_values.get_present_fe_values();

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
          AssertThrow(dynamic_cast<FSISolverLessLambda<dim>::MMSSolution *>(
                        this->exact_solution.get()) != nullptr,
                      ExcInternalError());
          const FSISolverLessLambda<dim>::MMSSolution &sol =
            static_cast<FSISolverLessLambda<dim>::MMSSolution &>(
              *this->exact_solution);
          sol.lagrange_multiplier(qpoint, nu, normal_to_solid, exact);

          diff = lambda_values[q] - exact;

          lambda_l2_local += diff * diff * fe_face_values.JxW(q);
          lambda_linf_local = std::max(lambda_linf_local, diff.norm());

          // Increment the integral of lambda
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
          exact_integral_local += exact * fe_face_values.JxW(q);
        }
      }
    }
  }

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
void FSISolverLessLambda<dim>::compute_solver_specific_errors()
{
  double         l2_l = 0., li_l = 0.;
  Tensor<1, dim> error_on_integral;
  this->compute_lambda_error_on_boundary(l2_l, li_l, error_on_integral);

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
void FSISolverLessLambda<dim>::solver_specific_post_processing()
{
  if (this->param.mms_param.enable)
  {
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
}

// Explicit instantiation
template class FSISolverLessLambda<2>;
template class FSISolverLessLambda<3>;
