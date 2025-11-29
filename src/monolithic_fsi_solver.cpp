
#include <compare_matrix.h>
#include <copy_data.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_solver.h>
#include <mesh.h>
// #include <mms.h>
#include <monolithic_fsi_solver.h>
#include <scratch_data.h>

template <int dim>
MonolithicFSISolver<dim>::MonolithicFSISolver(const ParameterReader<dim> &param)
  : GenericSolver<LA::ParVectorType>(param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param)
  , velocity_extractor(u_lower)
  , pressure_extractor(p_lower)
  , position_extractor(x_lower)
  , lambda_extractor(l_lower)
  , param(param)
  , quadrature(QGaussSimplex<dim>(4))
  , face_quadrature(QGaussSimplex<dim - 1>(4))
  , triangulation(mpi_communicator)
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
  , velocity_mask(fe.component_mask(velocity_extractor))
  , pressure_mask(fe.component_mask(pressure_extractor))
  , position_mask(fe.component_mask(position_extractor))
  , lambda_mask(fe.component_mask(lambda_extractor))
{
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

  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(u_lower, n_components);

  // Direct solver
  direct_solver_reuse =
    std::make_shared<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    exact_solution = std::make_shared<MonolithicFSISolver<dim>::MMSSolution>(
      time_handler.current_time, param.mms);

    // Create the source term function for the given MMS and override source
    // terms
    source_terms = std::make_shared<MonolithicFSISolver<dim>::MMSSourceTerm>(
      time_handler.current_time, param.physical_properties, param.mms);

    error_handler.create_entry("L2_u");
    error_handler.create_entry("L2_p");
    error_handler.create_entry("L2_x");
    error_handler.create_entry("L2_l");
    error_handler.create_entry("Li_u");
    error_handler.create_entry("Li_p");
    error_handler.create_entry("Li_x");
    error_handler.create_entry("Li_l");
    if(time_handler.is_steady())
    {
      error_handler.create_entry("H1_u");
      error_handler.create_entry("H1_p");
      error_handler.create_entry("H1_x");
    }
  }
  else
  {
    // FIXME: this is the source term for fluid problem only
    source_terms = param.source_terms.fluid_source;
    exact_solution =
      std::make_shared<Functions::ZeroFunction<dim>>(n_components);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const double nu          = physical_properties.fluids[0].kinematic_viscosity;
  const double lame_mu     = physical_properties.pseudosolids[0].lame_mu;
  const double lame_lambda = physical_properties.pseudosolids[0].lame_lambda;

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
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term
  values[p_lower] = mms.exact_velocity->divergence(p);

  // Pseudosolid (mesh position) source term
  // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
  Tensor<1, dim> f_PS =
    mms.exact_mesh_position->divergence_linear_elastic_stress(p,
                                                                  lame_mu,
                                                                  lame_lambda);

  for (unsigned int d = 0; d < dim; ++d)
    values[x_lower + d] = f_PS[d];

  // Lagrange multiplier source term (none)
  for (unsigned int d = 0; d < dim; ++d)
    values[l_lower + d] = 0.;
}

template <int dim>
void MonolithicFSISolver<dim>::run()
{
  reset();
  read_mesh(triangulation, param);
  setup_dofs();

  create_lagrange_multiplier_constraints();
  if (param.fsi.enable_coupling)
    create_position_lagrange_mult_coupling_data();

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

    if (time_handler.is_starting_step())
    {
      if (param.mms_param.enable)
      {
        if (param.time_integration.bdfstart ==
            Parameters::TimeIntegration::BDFStart::initial_condition)
        {
          // Convergence study: start with exact solution at first time step
          set_exact_solution();
        }
        else
        {
          //////////////////////////////////////////////////////////
          // Start with BDF1
          solve_nonlinear_problem(false);
          output_results();
          if (param.debug.fsi_check_mms_on_boundary)
            check_manufactured_solution_boundary();
          if (!time_handler.is_steady())
          {
            // Rotate solutions
            for (unsigned int j = previous_solutions.size() - 1; j >= 1; --j)
              previous_solutions[j] = previous_solutions[j - 1];
            previous_solutions[0] = present_solution;
          }
          continue;
          //////////////////////////////////////////////////////////
        }
      }
      else
      {
        // FIXME: Start with BDF1
        set_initial_conditions();
      }
    }
    else
    {
      // Entering the Newton solver with a solution satisfying the nonzero
      // constraints, which were applied in update_boundary_condition().
      if (param.debug.compare_analytical_jacobian_with_fd)
        compare_analytical_matrix_with_fd();
      // throw std::runtime_error("Debug");

      if (param.debug.apply_exact_solution)
        set_exact_solution();
      else
        solve_nonlinear_problem(false);
    }

    output_results();

    if (!param.mms_param.enable)
    {
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
    }
    else
    {
      if (param.debug.fsi_check_mms_on_boundary)
      {
        check_manufactured_solution_boundary();

        /**
         * When applying the exact solution, the fluid velocity will be exact,
         * but the mesh velocity is only precise up to time integration order.
         * So these velocities differ by some power of the time step, rather
         * than the machine epsilon as checked in this function.
         */
        if (!param.debug.apply_exact_solution)
          check_velocity_boundary();
      }
      compute_errors();
    }

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
void MonolithicFSISolver<dim>::setup_dofs()
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

  // Initialize mesh position directly from the triangulation.
  // The parallel vector storing the mesh position is local_evaluation_point,
  // because this is the one to modify when computing finite differences.
  const FEValuesExtractors::Vector position(x_lower);
  VectorTools::get_position_vector(*fixed_mapping,
                                   dof_handler,
                                   local_evaluation_point,
                                   position_mask);
  local_evaluation_point.compress(VectorOperation::insert);
  evaluation_point = local_evaluation_point;

  // Also store them in initial_positions, for postprocessing:
  DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                       dof_handler,
                                       initial_positions,
                                       position_mask);

  // Create the solution-dependent mapping
  mapping = std::make_shared<MappingFEField<dim, dim, LA::ParVectorType>>(
    dof_handler, evaluation_point, position_mask);

  // For unsteady simulation, add the number of elements, dofs and/or the time
  // step to the error handler, once per convergence run.
  if (!time_handler.is_steady() && param.mms_param.enable)
  {
    error_handler.add_reference_data("n_elm",
                                     triangulation.n_global_active_cells());
    error_handler.add_reference_data("n_dof", dof_handler.n_dofs());
    error_handler.add_time_step(time_handler.initial_dt);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::reset()
{
  // FIXME: This is not very clean: the derived class has the full parameters,
  // and the base class GenericSolver has a mesh and time param to be able to
  // modify the mesh file and/or time step in a convergence loop.
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
  this->set_time();

  // Pressure DOF
  constrained_pressure_dof = numbers::invalid_dof_index;

  // Position - lambda constraints
  for (auto &vec : position_lambda_coeffs)
    vec.clear();
  position_lambda_coeffs.clear();
  initial_positions.clear();
  coupled_position_dofs.clear();
}

template <int dim>
void MonolithicFSISolver<dim>::set_time()
{
  // Update time in all relevant structures:
  // - relevant boundary conditions
  // - source terms, if any
  // - exact solution, if any
  for (auto &[id, bc] : param.fluid_bc)
    bc.set_time(time_handler.current_time);
  for (auto &[id, bc] : param.pseudosolid_bc)
    bc.set_time(time_handler.current_time);
  source_terms->set_time(time_handler.current_time);
  exact_solution->set_time(time_handler.current_time);
}

template <int dim>
void MonolithicFSISolver<dim>::create_lagrange_multiplier_constraints()
{
  // lambda_constraints.clear();
  // lambda_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  // const FEValuesExtractors::Vector lambda(l_lower);

  // // The lambda dofs which are not set to zero
  // std::set<types::global_dof_index> unconstrained_lambda_dofs;

  // // Flag the unconstrained lambda dofs, lying on the boundary associated
  // with a
  // // weakly enforced no-slip BC. If there is no such boundary, constrain all
  // // lambdas. This allows to keep the problem structure as is, to test with
  // only
  // // strong BC.
  // if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
  // {
  //   const unsigned int                   n_dofs_per_face =
  //   fe.n_dofs_per_face(); std::vector<types::global_dof_index>
  //   face_dofs(n_dofs_per_face); for (const auto &cell :
  //   dof_handler.active_cell_iterators())
  //   {
  //     // if (cell->is_locally_owned())
  //     // {
  //     for (const auto f : cell->face_indices())
  //       if (cell->face(f)->at_boundary() &&
  //           cell->face(f)->boundary_id() == weak_no_slip_boundary_id)
  //       {
  //         cell->face(f)->get_dof_indices(face_dofs);
  //         for (unsigned int idof = 0; idof < n_dofs_per_face; ++idof)
  //         {
  //           if (!locally_relevant_dofs.is_element(face_dofs[idof]))
  //             continue;

  //           const unsigned int component =
  //             fe.face_system_to_component_index(idof).first;

  //           if (fe.has_support_on_face(idof, f) && is_lambda(component))
  //           {
  //             // Lambda DoF on the prescribed boundary: do not constrain
  //             unconstrained_lambda_dofs.insert(face_dofs[idof]);
  //           }
  //         }
  //       }
  //     // }
  //   }
  // }

  // // Add zero constraints to all lambda DOFs *not* in the boundary set
  // IndexSet     lambda_dofs = DoFTools::extract_dofs(dof_handler,
  // lambda_mask); unsigned int n_constrained_local = 0; for (const auto dof :
  // lambda_dofs)
  // {
  //   if (locally_relevant_dofs.is_element(dof))
  //     if (unconstrained_lambda_dofs.count(dof) == 0)
  //     {
  //       lambda_constraints.constrain_dof_to_zero(dof); // More readable (-:
  //       n_constrained_local++;
  //     }
  // }
  // lambda_constraints.close();

  ///////////////////////////////////////////////////////////////////////
  // Get the relevant lambda dofs on the boundary and constrain them
  lambda_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  // If there is no weakly enforced no slip boundary, this set remains empty and
  // all lambda dofs are constrained.
  IndexSet relevant_boundary_dofs;

  if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
  {
    relevant_boundary_dofs =
      DoFTools::extract_boundary_dofs(dof_handler,
                                      lambda_mask,
                                      {weak_no_slip_boundary_id});
  }

  // There does not seem to be a 2-3 liner way to extract the locally
  // relevant dofs on a boundary for a given component (extract_dofs
  // returns owned dofs).
  std::vector<types::global_dof_index> local_dofs(fe.n_dofs_per_cell());
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;
    cell->get_dof_indices(local_dofs);
    for (unsigned int i = 0; i < local_dofs.size(); ++i)
    {
      types::global_dof_index dof  = local_dofs[i];
      unsigned int            comp = fe.system_to_component_index(i).first;
      if (is_lambda(comp))
        if (locally_relevant_dofs.is_element(dof))
          if (!relevant_boundary_dofs.is_element(dof))
            lambda_constraints.constrain_dof_to_zero(dof);
    }
  }
  lambda_constraints.close();
  ///////////////////////////////////////////////////////////////////////

  // Show the number of owned and constrained lambda dofs
  IndexSet     lambda_dofs = DoFTools::extract_dofs(dof_handler, lambda_mask);
  unsigned int unconstrained_owned_dofs = 0;
  for (const auto &dof : lambda_dofs)
    if (!lambda_constraints.is_constrained(dof))
      unconstrained_owned_dofs++;

  // const unsigned int total_unconstrained_owned_dofs =
  //   Utilities::MPI::sum(unconstrained_owned_dofs, mpi_communicator);

  // std::cout << total_unconstrained_owned_dofs
  //           << " unconstrained owned lambda dofs" << std::endl;
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
  //
  // Get and synchronize the lambda DoFs on the cylinder
  //
  std::set<types::boundary_id> boundary_ids;
  boundary_ids.insert(weak_no_slip_boundary_id);

  IndexSet local_lambda_dofs =
    DoFTools::extract_boundary_dofs(dof_handler, lambda_mask, boundary_ids);
  IndexSet local_position_dofs =
    DoFTools::extract_boundary_dofs(dof_handler, position_mask, boundary_ids);

  const unsigned int n_local_lambda_dofs = local_lambda_dofs.n_elements();

  local_lambda_dofs   = local_lambda_dofs & locally_owned_dofs;
  local_position_dofs = local_position_dofs & locally_owned_dofs;

  // Gather all lists to all processes
  std::vector<std::vector<types::global_dof_index>> gathered_dofs =
    Utilities::MPI::all_gather(mpi_communicator,
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
  {
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
              const unsigned int d = comp - l_lower;

              // Only move in x and y, even in 3D
              if (d < 2)
              {
                const types::global_dof_index lambda_dof = face_dofs[i_dof];

                // Very, very, very important:
                // Even though fe_face_values_fixed is a FEFaceValues, the dof
                // index given to shape_value is still a CELL dof index.
                const unsigned int i_cell_dof =
                  fe.face_to_cell_index(i_dof, i_face);

                const double phi_i =
                  fe_face_values_fixed.shape_value(i_cell_dof, q);
                coeffs[d][lambda_dof] +=
                  -phi_i * JxW / this->param.fsi.spring_constant;
              }
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
    }
  }

  // ////////////////////////////////////////////////////////////////////////
  // // // Expected sum is -1/k * |Cylinder| (surface)
  // double       expected_weights_sum;
  // const double radius = 0.5;
  // if constexpr (dim == 2)
  //   expected_weights_sum = -1. / param.fsi.spring_constant * 2. * M_PI *
  //   radius;
  // else
  // {
  //   const double width = 8.;
  //   expected_weights_sum =
  //     -1. / param.fsi.spring_constant * 2. * M_PI * radius * width;
  // }

  // for (unsigned int d = 0; d < dim; ++d)
  // {
  //   double local_weights_sum = 0.;
  //   // std::cout << "Weights for dim = " << d << std::endl;
  //   for (const auto &[lambda_dof, weight] : coeffs[d])
  //   {
  //     // std::cout << "Lambda dof: " << lambda_dof << " - weight: " << weight
  //     <<
  //     // std::endl;
  //     local_weights_sum += weight;
  //   }
  //   // std::cout << "Sum is = " << local_weights_sum << " - Expected : " <<
  //   // expected_weights_sum << std::endl;

  //   const double weights_sum =
  //     Utilities::MPI::sum(local_weights_sum, mpi_communicator);
  //   // std::cout << "Sum of lambda weights for dim " << d << " = " <<
  //   weights_sum
  //   //           << " - Expected : " << expected_weights_sum << std::endl;
  // }
  ////////////////////////////////////////////////////////////////////////

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
      Utilities::MPI::all_gather(mpi_communicator, coeffs_vector);

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
void MonolithicFSISolver<dim>::apply_erroneous_position_lambda_constraints(
  const bool homogeneous)
{
  // Resize the position constraints with the updated locally_relevant_dofs
  erroneous_position_constraints.clear();
  erroneous_position_constraints.reinit(locally_owned_dofs,
                                        locally_relevant_dofs);

  std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());

  FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                         fe,
                                         face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

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

      fe_face_values_fixed.reinit(cell, face);
      face->get_dof_indices(face_dofs);

      for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
      {
        if (!locally_owned_dofs.is_element(face_dofs[i]))
          continue;

        const unsigned int comp =
          fe.face_system_to_component_index(i, i_face).first;

        if (is_position(comp))
        {
          const unsigned int d = comp - x_lower;
          erroneous_position_constraints.add_line(face_dofs[i]);
          erroneous_position_constraints.add_entries(face_dofs[i],
                                                     position_lambda_coeffs[d]);

          if (!homogeneous)
            erroneous_position_constraints.set_inhomogeneity(
              face_dofs[i], this->initial_positions.at(face_dofs[i])[d]);
        }
      }
    }
  }
  erroneous_position_constraints.make_consistent_in_parallel(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  erroneous_position_constraints.close();
}

template <int dim>
void MonolithicFSISolver<dim>::remove_cylinder_velocity_constraints(
  AffineConstraints<double> &constraints) const
{
  if (weak_no_slip_boundary_id == numbers::invalid_unsigned_int)
    return;

  IndexSet weak_velocity_dofs =
    DoFTools::extract_boundary_dofs(dof_handler,
                                    velocity_mask,
                                    {weak_no_slip_boundary_id});

  {
    AffineConstraints<double> filtered;
    filtered.reinit(locally_owned_dofs, locally_relevant_dofs);

    for (const auto &line : constraints.get_lines())
    {
      if (weak_velocity_dofs.is_element(line.index))
        continue;

      filtered.add_line(line.index);
      filtered.add_entries(line.index, line.entries);
      filtered.set_inhomogeneity(line.index, line.inhomogeneity);

      // Check that entries do not involve an absent velocity dof
      // With the get_view() function, this is done automatically
      for (const auto &entry : line.entries)
        AssertThrow(!weak_velocity_dofs.is_element(entry.first),
                    ExcMessage("Constraint involve a cylinder velocity dof"));
    }
    filtered.close();
    constraints = std::move(filtered);
    // constraints.make_consistent_in_parallel(locally_owned_dofs,
    //                           constraints.get_local_lines(),
    //                           mpi_communicator);
  }

  // {
  //   // Get the dofs that are not the cylinder fluid velocity
  //   // IndexSet subset = constraints.get_local_lines();
  //   // subset.subtract_set(weak_velocity_dofs);

  //   IndexSet local_lines = zero_constraints.get_local_lines();
  //   local_lines.compress();
  //   pcout << local_lines.n_intervals() << std::endl;
  //   pcout << local_lines.n_elements() << std::endl;
  //   pcout << local_lines.size() << std::endl;
  //   pcout << weak_velocity_dofs.n_intervals() << std::endl;
  //   pcout << weak_velocity_dofs.n_elements() << std::endl;
  //   pcout << weak_velocity_dofs.size() << std::endl;
  //   local_lines.get_view(weak_velocity_dofs);

  //   // IndexSet subset = constraints.get_local_lines() & weak_velocity_dofs;

  //   // This does not work:
  //   // auto tmp_constraints = constraints.get_view(subset);
  //   // constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  //   // constraints.close();
  //   // constraints.merge(tmp_constraints);
  // }

  // {
  //   This does not work either:
  //   AffineConstraints<double> tmp;
  //   tmp.copy_from(constraints);
  //   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  //   constraints.add_selected_constraints(tmp, subset);
  //   constraints.close();
  // }
}

template <int dim>
void MonolithicFSISolver<dim>::constrain_pressure_point(
  AffineConstraints<double> &constraints,
  const bool                 set_to_zero)
{
  // Determine the pressure dof the first time
  if (constrained_pressure_dof == numbers::invalid_dof_index)
  {
    // Choose a fixed physical reference location
    // Here it's the origin (Point<dim> is initialized at 0)
    const Point<dim> reference_point;

    IndexSet pressure_dofs = DoFTools::extract_dofs(dof_handler, pressure_mask);

    // Get support points for locally relevant DoFs
    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler, support_points);

    double                  local_min_dist = std::numeric_limits<double>::max();
    types::global_dof_index local_dof      = numbers::invalid_dof_index;

    for (auto idx : pressure_dofs)
    {
      if (!locally_relevant_dofs.is_element(idx))
        continue;

      const double dist = support_points[idx].distance(reference_point);
      if (dist < local_min_dist)
      {
        local_min_dist = dist;
        local_dof      = idx;
      }
    }

    // Prepare for MPI_MINLOC reduction
    struct MinLoc
    {
      double                  dist;
      types::global_dof_index dof;
    } local_pair{local_min_dist, local_dof}, global_pair;

    // MPI reduction to find the global closest DoF
    MPI_Allreduce(&local_pair,
                  &global_pair,
                  1,
                  MPI_DOUBLE_INT,
                  MPI_MINLOC,
                  mpi_communicator);

    constrained_pressure_dof = global_pair.dof;

    // Set support point for MMS evaluation
    if (locally_relevant_dofs.is_element(constrained_pressure_dof))
    {
      constrained_pressure_support_point =
        support_points[constrained_pressure_dof];
    }
  }

  // Constrain that DoF globally
  if (locally_relevant_dofs.is_element(constrained_pressure_dof))
  {
    constraints.add_line(constrained_pressure_dof);

    // The pressure DOF is set to 0 by default for the nonzero constraints,
    // unless there is a prescribed manufactured solution, in which case it is
    // prescribed to p_mms.
    if (set_to_zero || !param.mms_param.enable)
    {
      constraints.constrain_dof_to_zero(constrained_pressure_dof);
    }
    else
    {
      const double pAnalytic =
        exact_solution->value(constrained_pressure_support_point, p_lower);
      // std::cout << "Constraining pressure DOF " << constrained_pressure_dof
      //           << " at " << constrained_pressure_support_point << " to "
      //           << pAnalytic << std::endl;
      constraints.set_inhomogeneity(constrained_pressure_dof, pAnalytic);
    }
  }
}

template <int dim>
void MonolithicFSISolver<dim>::create_zero_constraints()
{
  zero_constraints.clear();
  zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  //
  // Mesh position homogeneous BC
  //
  {
    FixedMeshPosition<dim>                              fixed_mesh_fun(0, dim);
    std::set<types::boundary_id>                        normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> position_flux_functions;
    for (const auto &[id, bc] : this->param.pseudosolid_bc)
    {
      if (bc.type == BoundaryConditions::Type::fixed ||
          bc.type == BoundaryConditions::Type::input_function ||
          bc.type == BoundaryConditions::Type::position_mms)
      {
        VectorTools::interpolate_boundary_values(*fixed_mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 zero_constraints,
                                                 position_mask);
      }

      if (bc.type == BoundaryConditions::Type::no_flux)
      {
        normal_flux_boundaries.insert(bc.id);
        position_flux_functions[bc.id] = &fixed_mesh_fun;
      }
    }

    // Add position nonzero flux constraints (tangential movement)
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      x_lower,
      normal_flux_boundaries,
      position_flux_functions,
      zero_constraints,
      *fixed_mapping);
  }

  BoundaryConditions::apply_velocity_boundary_conditions(true,
                                     u_lower,
                                     n_components,
                                     dof_handler,
                                     *mapping,
                                     param.fluid_bc,
                                     *exact_solution,
                                     *param.mms.exact_velocity,
                                     zero_constraints);

  if (param.bc_data.fix_pressure_constant)
  {
    bool set_to_zero = true;
    constrain_pressure_point(zero_constraints, set_to_zero);
  }

  zero_constraints.close();

  if constexpr (dim == 3)
    remove_cylinder_velocity_constraints(zero_constraints);

  // Merge the zero lambda constraints
  zero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
    true);

  if (param.fsi.enable_coupling && param.debug.fsi_apply_erroneous_coupling)
  {
    // Apply the "wrong" coupling between lambda and position on cylinder,
    // to compare with previous solver.
    const bool homogeneous = true;
    this->apply_erroneous_position_lambda_constraints(homogeneous);
    zero_constraints.merge(
      erroneous_position_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::create_nonzero_constraints()
{
  nonzero_constraints.clear();
  nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  //
  // Mesh position inhomogeneous BC
  //
  {
    FixedMeshPosition<dim>                              fixed_mesh_fun(0, dim);
    std::set<types::boundary_id>                        normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> position_flux_functions;
    for (const auto &[id, bc] : param.pseudosolid_bc)
    {
      if (bc.type == BoundaryConditions::Type::fixed)
      {
        VectorTools::interpolate_boundary_values(
          *fixed_mapping,
          dof_handler,
          bc.id,
          FixedMeshPosition<dim>(x_lower, n_components),
          nonzero_constraints,
          position_mask);
      }
      if (bc.type == BoundaryConditions::Type::input_function)
      {
        // TODO: Prescribed but non-fixed mesh position?
        AssertThrow(
          false,
          ExcMessage(
            "Input function for pseudosolid problem are not yet handled."));
      }
      if (bc.type == BoundaryConditions::Type::position_mms)
      {
        VectorTools::interpolate_boundary_values(*fixed_mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 *exact_solution,
                                                 nonzero_constraints,
                                                 position_mask);
      }
      if (bc.type == BoundaryConditions::Type::no_flux)
      {
        normal_flux_boundaries.insert(bc.id);
        position_flux_functions[bc.id] = &fixed_mesh_fun;
      }
    }

    // Add position nonzero flux constraints (tangential movement)
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      x_lower,
      normal_flux_boundaries,
      position_flux_functions,
      nonzero_constraints,
      *fixed_mapping);
  }

  BoundaryConditions::apply_velocity_boundary_conditions(false,
                                     u_lower,
                                     n_components,
                                     dof_handler,
                                     *mapping,
                                     param.fluid_bc,
                                     *exact_solution,
                                     *param.mms.exact_velocity,
                                     nonzero_constraints);

  if (param.bc_data.fix_pressure_constant)
  {
    bool set_to_zero = false;
    constrain_pressure_point(nonzero_constraints, set_to_zero);
  }

  nonzero_constraints.close();

  //
  if constexpr (dim == 3)
    remove_cylinder_velocity_constraints(nonzero_constraints);

  // Merge the zero lambda constraints
  nonzero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
    true);

  if (param.fsi.enable_coupling && param.debug.fsi_apply_erroneous_coupling)
  {
    // Apply the "wrong" coupling between lambda and position on cylinder,
    // to compare with previous solver.
    const bool homogeneous = false;
    this->apply_erroneous_position_lambda_constraints(homogeneous);
    nonzero_constraints.merge(
      erroneous_position_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints have been
  // defined
  //
  DynamicSparsityPattern dsp(locally_relevant_dofs);

  Table<2, DoFTools::Coupling> coupling(n_components, n_components);

  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int d = 0; d < n_components; ++d)
    {
      coupling[c][d] = DoFTools::none;

      // u couples to all variables
      if (is_velocity(c))
        coupling[c][d] = DoFTools::always;

      // p couples to u and x
      if (is_pressure(c))
        if (is_velocity(d) || is_position(d))
          coupling[c][d] = DoFTools::always;

      // x couples to itself
      if (is_position(c) && is_position(d))
        coupling[c][d] = DoFTools::always;

      if (is_lambda(c))
        if (is_velocity(d) || is_position(d))
          coupling[c][d] = DoFTools::always;
    }

  DoFTools::make_sparsity_pattern(dof_handler,
                                  coupling,
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
                                             mpi_communicator,
                                             locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);
}

template <int dim>
void MonolithicFSISolver<dim>::set_initial_conditions()
{
  // Update mesh position *BEFORE* evaluating fields on moving mapping.
  // This does not matter here though, as the initial mesh position is the
  // fixed_mapping.

  if (param.initial_conditions.set_to_mms)
  {
    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             *exact_solution,
                             newton_update,
                             position_mask);

    // Update MappingFEField *BEFORE* interpolating velocity
    evaluation_point = newton_update;

    // Set velocity with moving mapping (irrelevant for initial position)
    VectorTools::interpolate(
      *mapping, dof_handler, *exact_solution, newton_update, velocity_mask);
  }
  else
  {
    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             FixedMeshPosition<dim>(x_lower, n_components),
                             newton_update,
                             position_mask);

    // Update MappingFEField *BEFORE* interpolating velocity
    evaluation_point = newton_update;

    // Set velocity with moving mapping (irrelevant for initial position)
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             *param.initial_conditions.initial_velocity,
                             newton_update,
                             velocity_mask);
  }

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
void MonolithicFSISolver<dim>::set_exact_solution()
{
  // Update mesh position *BEFORE* evaluating fields on moving mapping.
  VectorTools::interpolate(*fixed_mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           position_mask);

  // Update MappingFEField *BEFORE* interpolating velocity/pressure
  evaluation_point = local_evaluation_point;

  // Set velocity and pressure with moving mapping
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           velocity_mask);
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           pressure_mask);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void MonolithicFSISolver<dim>::update_boundary_conditions()
{
  // Re-create and distribute nonzero constraints:
  local_evaluation_point = present_solution;

  // Create and apply the inhomogeneous constraints a first time
  // to apply mesh position boundary conditions.
  // Then update the moving mapping (through the evaluation point),
  // and evaluate the inhomogeneous velocity (and other) BC on the
  // updated mapping.
  create_nonzero_constraints();

  // Update the moving mapping
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;

  // Create and apply inhomogeneous BC for non-position fields.
  // The position BC are re-applied, but did not change.
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  system_matrix = 0;

  ScratchDataMonolithicFSI<dim> scratch_data(fe,
                                             quadrature,
                                             *fixed_mapping,
                                             *mapping,
                                             face_quadrature,
                                             fe.n_dofs_per_cell(),
                                             weak_no_slip_boundary_id,
                                             time_handler.bdf_coefficients);
  CopyData                      copy_data(fe.n_dofs_per_cell());

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
                  &MonolithicFSISolver::assemble_local_matrix,
                  &MonolithicFSISolver::copy_local_to_global_matrix,
                  scratch_data,
                  copy_data);

  system_matrix.compress(VectorOperation::add);

  if (param.fsi.enable_coupling && !param.debug.fsi_apply_erroneous_coupling)
    add_algebraic_position_coupling_to_matrix();
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataMonolithicFSI<dim>                        &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  const double nu =
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

          // ALE acceleration : - w dot grad(delta u)
          local_flow_matrix_ij += grad_phi_u[j] * (-dxdt) * phi_u[i];

          // Diffusion
          local_flow_matrix_ij +=
            nu * scalar_product(grad_phi_u[j], grad_phi_u[i]);
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

          // Convection w.r.t. x
          local_flow_matrix_ij +=
            (-present_velocity_gradients * grad_phi_x[j]) *
            present_velocity_values * phi_u[i];
          local_flow_matrix_ij += present_velocity_gradients *
                                  present_velocity_values * phi_u[i] *
                                  trace(grad_phi_x[j]);

          // Variation of ALE term (dxdt cdot grad(u)) with mesh position
          local_flow_matrix_ij +=
            present_velocity_gradients * (-bdf_c0 * phi_x[j]) * phi_u[i];
          local_flow_matrix_ij +=
            (-present_velocity_gradients * grad_phi_x[j]) * (-dxdt) * phi_u[i];
          local_flow_matrix_ij += present_velocity_gradients * (-dxdt) *
                                  phi_u[i] * trace(grad_phi_x[j]);

          // Diffusion
          const Tensor<2, dim> d_grad_u =
            -present_velocity_gradients * grad_phi_x[j];
          const Tensor<2, dim> d_grad_phi_u = -grad_phi_u[i] * grad_phi_x[j];
          local_flow_matrix_ij += nu * scalar_product(d_grad_u, grad_phi_u[i]);
          local_flow_matrix_ij +=
            nu * scalar_product(present_velocity_gradients, d_grad_phi_u);
          local_flow_matrix_ij +=
            nu * scalar_product(present_velocity_gradients, grad_phi_u[i]) *
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

          // local_flow_matrix_ij += trace(grad_phi_x[j]);
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
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void MonolithicFSISolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                              copy_data.local_dof_indices,
                                              system_matrix);
}

template <int dim>
void MonolithicFSISolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchDataMonolithicFSI<dim> scratch_data(fe,
                                             quadrature,
                                             *fixed_mapping,
                                             *mapping,
                                             face_quadrature,
                                             fe.n_dofs_per_cell(),
                                             weak_no_slip_boundary_id,
                                             time_handler.bdf_coefficients);
  CopyData                      copy_data(fe.n_dofs_per_cell());

  double max_error_over_all_elements;

  Verification::compare_analytical_matrix_with_fd(
    dof_handler,
    fe.n_dofs_per_cell(),
    *this,
    &MonolithicFSISolver::assemble_local_matrix,
    &MonolithicFSISolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    present_solution,
    evaluation_point,
    local_evaluation_point,
    mpi_communicator,
    max_error_over_all_elements,
    param.output.output_dir,
    true,
    param.debug.analytical_jacobian_absolute_tolerance,
    param.debug.analytical_jacobian_relative_tolerance);

  pcout << "Max error analytical vs fd matrix is "
        << max_error_over_all_elements << std::endl;
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  ScratchDataMonolithicFSI<dim> scratch_data(fe,
                                             quadrature,
                                             *fixed_mapping,
                                             *mapping,
                                             face_quadrature,
                                             fe.n_dofs_per_cell(),
                                             weak_no_slip_boundary_id,
                                             time_handler.bdf_coefficients);
  CopyData                      copy_data(fe.n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &MonolithicFSISolver::assemble_local_rhs,
                  &MonolithicFSISolver::copy_local_to_global_rhs,
                  scratch_data,
                  copy_data);

  system_rhs.compress(VectorOperation::add);

  if (param.fsi.enable_coupling && !param.debug.fsi_apply_erroneous_coupling)
    add_algebraic_position_coupling_to_rhs();
}

template <int dim>
void MonolithicFSISolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataMonolithicFSI<dim>                        &scratchData,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratchData.reinit(
    cell, evaluation_point, previous_solutions, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;
  const double lame_lambda =
    this->param.physical_properties.pseudosolids[0].lame_lambda;
  const double lame_mu =
    this->param.physical_properties.pseudosolids[0].lame_mu;

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

    const Tensor<1, dim> dudt =
      time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratchData.previous_velocity_values);

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
        // Transient
        dudt * phi_u[i]

        // Convection
        + (present_velocity_gradients * present_velocity_values) * phi_u[i]

        // Mesh movement
        - (present_velocity_gradients * present_mesh_velocity_values) * phi_u[i]

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

      // double local_rhs_flow_i = 0.;
      // if(is_pressure(fe.system_to_component_index(i).first))
      //   local_rhs_flow_i = -1.;

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
              local_rhs_i -= -(phi_u[i] * present_l);

            if (i_is_l)
              local_rhs_i -= -(present_u - present_w) * phi_l[i];

            local_rhs_i *= face_JxW_moving;
            local_rhs(i) += local_rhs_i;
          }
        }
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void MonolithicFSISolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                              copy_data.local_dof_indices,
                                              this->system_rhs);
}

template <int dim>
void MonolithicFSISolver<dim>::add_algebraic_position_coupling_to_matrix()
{
  //
  // Add algebraic constraints position-lambda
  //
  std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
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

  if (!time_handler.is_starting_step() &&
      !param.debug.fsi_apply_erroneous_coupling)
  {
    AssertThrow(l2_error < 1e-12,
                ExcMessage("L2 norm of uh - wh is too large."));
    AssertThrow(li_error < 1e-12,
                ExcMessage("Linf norm of uh - wh is too large."));
  }
}

template <int dim>
void MonolithicFSISolver<dim>::check_manufactured_solution_boundary()
{
  Tensor<1, dim> lambdaMMS_integral, lambdaMMS_integral_local;
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  Tensor<1, dim> pns_integral, pns_integral_local;
  lambdaMMS_integral_local = 0;
  lambda_integral_local    = 0;
  pns_integral_local       = 0;

  const double rho = param.physical_properties.fluids[0].density;
  const double nu  = param.physical_properties.fluids[0].kinematic_viscosity;
  const double mu  = nu * rho;

  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                         fe,
                                         face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  Tensor<1, dim>              lambda_MMS;
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  //
  // First compute integral over cylinder of lambda_MMS
  //
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
        fe_face_values[lambda_extractor].get_function_values(present_solution,
                                                             lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
          const auto        normal_to_solid = -fe_face_values.normal_vector(q);

          const double p_MMS = exact_solution->value(qpoint, p_lower);

          std::static_pointer_cast<MonolithicFSISolver<dim>::MMSSolution>(
            exact_solution)
            ->lagrange_multiplier(qpoint, mu, normal_to_solid, lambda_MMS);

          // Increment the integrals of lambda:

          // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
          lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

          // This is int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
          //                                                    -normal_to_solid
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
      Utilities::MPI::sum(lambdaMMS_integral_local[d], mpi_communicator);
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
  }
  pns_integral = Utilities::MPI::sum(pns_integral_local, mpi_communicator);

  // // Reference solution for int_Gamma p*n_solid dx is - k * d * f(t).
  // Tensor<1, dim> translation;
  // translation[0] = 0.1;
  // translation[1] = 0.05;
  const Tensor<1, dim> ref_pns;
  // const Tensor<1, dim> ref_pns =
  //   -param.fsi.spring_constant * translation *
  //   std::static_pointer_cast<MonolithicFSISolver<dim>::MMSSolution>(
  //     exact_solution)->mms.exact_mesh_position->time_function->value(time_handler.current_time);
  // const double err_pns = (ref_pns - pns_integral).norm();
  const double err_pns = -1.;

  //
  // Check x_MMS
  //
  Tensor<1, dim> x_MMS;
  double         max_x_error = 0.;
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

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint_fixed =
            fe_face_values_fixed.quadrature_point(q);

          for (unsigned int d = 0; d < dim; ++d)
            x_MMS[d] = exact_solution->value(qpoint_fixed, x_lower + d);

          const Tensor<1, dim> ref =
            -1. / param.fsi.spring_constant * lambdaMMS_integral;
          const double err = ((x_MMS - qpoint_fixed) - ref).norm();
          // std::cout << "x_MMS - X0 at quad node is " << x_MMS  - qpoint_fixed
          // << " - diff = " << err << std::endl;
          max_x_error = std::max(max_x_error, err);
        }
      }
    }
  }

  //
  // Check u_MMS
  //
  Tensor<1, dim> u_MMS, w_MMS;
  double         max_u_error = -1;
  // for (auto cell : dof_handler.active_cell_iterators())
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
  //         // std::cout << "u_MMS & w_MMS at quad node are " << u_MMS << " , "
  //         << w_MMS << " - norm diff = " << err << std::endl; max_u_error =
  //         std::max(max_u_error, err);
  //       }
  //     }
  //   }
  // }

  // if(VERBOSE)
  // {
  pcout << std::endl;
  pcout << "Checking manufactured solution for k = "
        << param.fsi.spring_constant << " :" << std::endl;
  pcout << "integral lambda         = " << lambda_integral << std::endl;
  pcout << "integral lambdaMMS      = " << lambdaMMS_integral << std::endl;
  pcout << "integral pMMS * n_solid = " << pns_integral << std::endl;
  pcout << "reference: -k*d*f(t)    = " << ref_pns << " - err = " << err_pns
        << std::endl;
  pcout << "max error on (x_MMS -    X0) vs -1/k * integral lambda = "
        << max_x_error << std::endl;
  pcout << "max error on  u_MMS          vs w_MMS                  = "
        << max_u_error << std::endl;
  pcout << std::endl;
  // }
}

template <int dim>
void MonolithicFSISolver<dim>::compute_lambda_error_on_boundary(
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

  const double rho = param.physical_properties.fluids[0].density;
  const double nu  = param.physical_properties.fluids[0].kinematic_viscosity;
  const double mu  = nu * rho;

  FEFaceValues<dim> fe_face_values(*mapping,
                                   fe,
                                   face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);
  Tensor<1, dim>              diff, exact;

  // std::ofstream out("normals.pos");
  // out << "View \"normals\" {\n";

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
        fe_face_values[lambda_extractor].get_function_values(present_solution,
                                                             lambda_values);

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

          // out << "VP(" << qpoint[0] << "," << qpoint[1] << "," << 0. << "){"
          //   << normal[0] << "," << normal[1] << "," << 0. << "};\n";

          // exact_solution is a pointer to base class Function<dim>,
          // so we have to ruse to use the specific function for lambda.
          std::static_pointer_cast<MonolithicFSISolver<dim>::MMSSolution>(
            exact_solution)
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

  lambda_l2_error = Utilities::MPI::sum(lambda_l2_local, mpi_communicator);
  lambda_l2_error = std::sqrt(lambda_l2_error);

  lambda_linf_error = Utilities::MPI::max(lambda_linf_local, mpi_communicator);

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
    exact_integral[d] =
      Utilities::MPI::sum(exact_integral_local[d], mpi_communicator);
    error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::compute_errors()
{
  TimerOutput::Scope t(this->computing_timer, "Compute errors");

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  ComponentSelectFunction<dim> velocity_comp_select(std::make_pair(u_lower,
                                                                   u_upper),
                                                    n_components);
  ComponentSelectFunction<dim> pressure_comp_select(p_lower, n_components);
  ComponentSelectFunction<dim> position_comp_select(std::make_pair(x_lower,
                                                                   x_upper),
                                                    n_components);
  ComponentSelectFunction<dim> lambda_comp_select(std::make_pair(l_lower,
                                                                 l_upper),
                                                  n_components);

  // Choose another quadrature rule for error computation
  const unsigned int                  n_points_1D = (dim == 2) ? 6 : 5;
  const QWitherdenVincentSimplex<dim> err_quadrature(n_points_1D);

  // L2 - u
  const double l2_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::L2_norm,
                                               &velocity_comp_select);
  // L2 - p
  const double l2_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::L2_norm,
                                               &pressure_comp_select);
  // L2 - x
  const double l2_x =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *fixed_mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::L2_norm,
                                               &position_comp_select);
  // Linf - u
  const double li_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::Linfty_norm,
                                               &velocity_comp_select);
  // Linf - p
  const double li_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::Linfty_norm,
                                               &pressure_comp_select);
  // Linf - x
  const double li_x =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *fixed_mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::Linfty_norm,
                                               &position_comp_select);
  // H1 seminorm - u
  const double h1semi_u =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::H1_seminorm,
                                               &velocity_comp_select);
  // H1 seminorm - p
  const double h1semi_p =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::H1_seminorm,
                                               &pressure_comp_select);
  // H1 seminorm - x
  const double h1semi_x =
    compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                               *fixed_mapping,
                                               dof_handler,
                                               present_solution,
                                               *exact_solution,
                                               cellwise_errors,
                                               err_quadrature,
                                               VectorTools::H1_seminorm,
                                               &position_comp_select);

  //
  // Errors for lambda on the relevant boundaries
  //
  // Do not compute at first time step for BDF2
  double         l2_l = 0., li_l = 0.;
  Tensor<1, dim> error_on_integral;
  // if (!time_handler.is_starting_step())
  // {
  this->compute_lambda_error_on_boundary(l2_l, li_l, error_on_integral);
  // linf_error_Fx = std::max(linf_error_Fx, error_on_integral[0]);
  // linf_error_Fy = std::max(linf_error_Fy, error_on_integral[1]);
  // }

  if (time_handler.is_steady())
  {
    // Steady solver: simply add errors to convergence table
    error_handler.add_reference_data("n_elm",
                                     triangulation.n_global_active_cells());
    error_handler.add_reference_data("n_dof", dof_handler.n_dofs());
    error_handler.add_steady_error("L2_u", l2_u);
    error_handler.add_steady_error("L2_p", l2_p);
    error_handler.add_steady_error("L2_x", l2_x);
    error_handler.add_steady_error("Li_u", li_u);
    error_handler.add_steady_error("Li_p", li_p);
    error_handler.add_steady_error("Li_x", li_x);
    error_handler.add_steady_error("H1_u", h1semi_u);
    error_handler.add_steady_error("H1_p", h1semi_p);
    error_handler.add_steady_error("H1_x", h1semi_x);
  }
  else
  {
    const double t = time_handler.current_time;
    error_handler.add_unsteady_error("L2_u", t, l2_u);
    error_handler.add_unsteady_error("L2_p", t, l2_p);
    error_handler.add_unsteady_error("L2_x", t, l2_x);
    error_handler.add_unsteady_error("L2_l", t, l2_l);
    error_handler.add_unsteady_error("Li_u", t, li_u);
    error_handler.add_unsteady_error("Li_p", t, li_p);
    error_handler.add_unsteady_error("Li_x", t, li_x);
    error_handler.add_unsteady_error("Li_l", t, li_l);
    // error_handler.add_unsteady_error("H1_u", t, h1semi_u);
    // error_handler.add_unsteady_error("H1_p", t, h1semi_p);
    // error_handler.add_unsteady_error("H1_x", t, h1semi_x);
  }
}

template <int dim>
void MonolithicFSISolver<dim>::output_results() const
{
  if (param.output.write_results)
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
    mesh_velocity.reinit(locally_owned_dofs, mpi_communicator);
    IndexSet disp_dofs = DoFTools::extract_dofs(dof_handler, position_mask);

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
                                        mpi_communicator,
                                        2);
  }
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

  if (export_table && param.output.write_results && mpi_rank == 0)
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

  if (export_table && param.output.write_results && mpi_rank == 0)
  {
    std::ofstream outfile(param.output.output_dir + "cylinder_center.txt");
    cylinder_position_table.write_text(outfile);
  }
}

// Explicit instantiation
template class MonolithicFSISolver<2>;
template class MonolithicFSISolver<3>;