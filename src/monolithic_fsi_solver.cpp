
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>

template <int dim>
MonolithicFSISolver<dim>::MonolithicFSISolver(const ParameterReader<dim> &param)
  : GenericSolver<dim, ParVectorType>(param)
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
}

template <int dim>
void MonolithicFSISolver<dim>::run()
{
  // Read mesh
  this->pcout << "Reading mesh" << std::endl;
  read_mesh(triangulation, this->param);

  // SET TIME OF FUNCTIONS, EITHER BEFORE OR AFTER ROTATE

  this->pcout << "Rotating time" << std::endl;
  time_handler.rotate();

  this->pcout << "Setting up dofs" << std::endl;
  setup_dofs();

  this->pcout << "Creating lambda constraints" << std::endl;
  create_lagrange_multiplier_constraints();

  create_position_lagrange_mult_coupling_data();

  create_zero_constraints();
  create_nonzero_constraints();
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
    previous_sol.clear();
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
  mapping = std::make_unique<MappingFEField<dim, dim, ParVectorType>>(
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
      // if (bc.type == BoundaryConditions::Type::input_function)
      // {
      //   VectorTools::interpolate_boundary_values(*fixed_mapping,
      //                                            dof_handler,
      //                                            bc.id,
      //                                            Functions::ZeroFunction<dim>(
      //                                              n_components),
      //                                            nonzero_constraints,
      //                                            fe.component_mask(position));
      // }
      // if (bc.type == BoundaryConditions::Type::no_flux)
      //   no_flux_boundaries.insert(bc.id);
    }

    // // Add no position flux constraints (tangential movement)
    // VectorTools::compute_no_normal_flux_constraints(dof_handler,
    //                                                 x_lower,
    //                                                 no_flux_boundaries,
    //                                                 nonzero_constraints,
    //                                                 *fixed_mapping);
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

  // nonzero_constraints.close();

  // // Merge the zero lambda constraints
  // nonzero_constraints.merge(
  //   lambda_constraints,
  //   AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
  //   true);
}

template <int dim>
void MonolithicFSISolver<dim>::create_sparsity_pattern()
{}

template <int dim>
void MonolithicFSISolver<dim>::assemble_matrix()
{}

template <int dim>
void MonolithicFSISolver<dim>::assemble_rhs()
{}

template <int dim>
void MonolithicFSISolver<dim>::solve_linear_system()
{}

template <int dim>
void MonolithicFSISolver<dim>::output_results() const
{}

// Explicit instantiation
template class MonolithicFSISolver<2>;
template class MonolithicFSISolver<3>;