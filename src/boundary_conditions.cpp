
#include <boundary_conditions.h>
#include <deal.II/grid/grid_tools_geometry.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/exceptions.h>
#include <mpi.h>
#include <sstream>
#include <deal.II/fe/mapping_q1.h>


namespace BoundaryConditions
{
  void BoundaryCondition::declare_parameters(ParameterHandler &prm)
  {
    prm.declare_entry(
      "id",
      "-1",
      Patterns::Integer(),
      "Gmsh tag of the physical entity associated to this boundary");
    prm.declare_entry(
      "name",
      "",
      Patterns::Anything(),
      "Name of the Gmsh physical entity associated to this boundary");
  }

  void BoundaryCondition::read_parameters(ParameterHandler &prm)
  {
    id        = prm.get_integer("id");
    gmsh_name = prm.get("name");
  }

  template <int dim>
  void FluidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|input_function|outflow|no_slip|weak_no_slip|slip|"
                        "velocity_mms|velocity_flux_mms|open_mms"),
                      "Type of fluid boundary condition");

    // Imposed functions, if any
    prm.enter_subsection("u");
    u->declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("v");
    v->declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("w");
    w->declare_parameters(prm);
    prm.leave_subsection();
  }

  template <int dim>
  void FluidBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::fluid;
    physics_str                   = "fluid";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "input_function")
      type = Type::input_function;
    if (parsed_type == "outflow")
      type = Type::outflow;
    if (parsed_type == "no_slip")
      type = Type::no_slip;
    if (parsed_type == "weak_no_slip")
      type = Type::weak_no_slip;
    if (parsed_type == "slip")
      type = Type::slip;
    if (parsed_type == "velocity_mms")
      type = Type::velocity_mms;
    if (parsed_type == "velocity_flux_mms")
      type = Type::velocity_flux_mms;
    if (parsed_type == "open_mms")
      type = Type::open_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Fluid boundary condition for boundary " + std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed fluid boundary conditions is smaller than "
        "the specified \"number\" field.");

    prm.enter_subsection("u");
    u->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("v");
    v->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("w");
    w->parse_parameters(prm);
    prm.leave_subsection();
  }

  template <int dim>
  void PseudosolidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|fixed|coupled_to_fluid|no_flux|input_function|"
                        "position_mms|position_flux_mms"),
                      "Type of pseudosolid boundary condition");
  }

  template <int dim>
  void PseudosolidBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::pseudosolid;
    physics_str                   = "pseudosolid";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "fixed")
      type = Type::fixed;
    if (parsed_type == "coupled_to_fluid")
      type = Type::coupled_to_fluid;
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "input_function")
      type = Type::input_function;
    if (parsed_type == "position_mms")
      type = Type::position_mms;
    if (parsed_type == "position_flux_mms")
      type = Type::position_flux_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Pseudosolid boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");
  }

  template <int dim>
  void CahnHilliardBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection("none|no_flux|dirichlet_mms"),
                      "Type of Cahn-Hilliard boundary condition");
  }

  template <int dim>
  void CahnHilliardBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::cahn_hilliard;
    physics_str                   = "cahn_hilliard";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "dirichlet_mms")
      type = Type::dirichlet_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Cahn-Hilliard boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed Cahn-Hilliard boundary conditions is smaller than "
        "the specified \"number\" field.");
  }

  template <int dim>
  void HeatBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection("none|input_function|dirichlet_mms"),
                      "Type of temperature boundary condition");
    prm.enter_subsection("temperature");
    temperature->declare_parameters(prm);
    prm.leave_subsection();
  }

  template <int dim>
  void HeatBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::heat;
    physics_str                   = "heat";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "input_function")
      type = Type::input_function;
    if (parsed_type == "dirichlet_mms")
      type = Type::dirichlet_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Temperature boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed temperature boundary conditions is smaller than "
        "the specified \"number\" field.");
    prm.enter_subsection("temperature");
    temperature->parse_parameters(prm);
    prm.leave_subsection();
  }

  // Explicit instantiation
  template class FluidBC<2>;
  template class FluidBC<3>;
  template class PseudosolidBC<2>;
  template class PseudosolidBC<3>;
  template class CahnHilliardBC<2>;
  template class CahnHilliardBC<3>;
  template class HeatBC<2>;
  template class HeatBC<3>;

  template <int dim>
  void apply_velocity_boundary_conditions(
    const bool             homogeneous,
    const unsigned int     u_lower,
    const unsigned int     n_components,
    const DoFHandler<dim> &dof_handler,
    const Mapping<dim>    &mapping,
    const std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>>
                              &fluid_bc,
    const Function<dim>       &exact_solution,
    const Function<dim>       &exact_velocity,
    AffineConstraints<double> &constraints,
    unsigned int rank)
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const ComponentMask velocity_mask = dof_handler.get_fe().component_mask(velocity);



    std::set<types::boundary_id> no_flux_boundaries;
    std::set<types::boundary_id> outflow_boundaries;
    std::set<types::boundary_id> velocity_normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_normal_flux_functions;
    std::set<types::boundary_id> velocity_tangential_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> velocity_tangential_flux_functions;

    // Compteurs utiles
    unsigned int n_no_slip = 0;
    unsigned int n_input_function = 0;
    unsigned int n_velocity_mms = 0;
    unsigned int n_slip = 0;
    unsigned int n_velocity_flux_mms = 0;
    unsigned int n_outflow = 0 ; 


    for (const auto &[id, bc] : fluid_bc)
    {


      if (bc.type == BoundaryConditions::Type::no_slip)
      {
        ++n_no_slip;

        VectorTools::interpolate_boundary_values(mapping,
                                                dof_handler,
                                                bc.id,
                                                Functions::ZeroFunction<dim>(n_components),
                                                constraints,
                                                velocity_mask);
      }

      if (bc.type == BoundaryConditions::Type::input_function)
      {
        ++n_input_function;

        if (homogeneous)
        {

          VectorTools::interpolate_boundary_values(mapping,
                                                  dof_handler,
                                                  bc.id,
                                                  Functions::ZeroFunction<dim>(n_components),
                                                  constraints,
                                                  velocity_mask);
        }
        else
        {
          VectorTools::interpolate_boundary_values(mapping,
                                                  dof_handler,
                                                  bc.id,
                                                  ComponentwiseFlowVelocity<dim>(u_lower,
                                                                                  n_components,
                                                                                  bc.u,
                                                                                  bc.v,
                                                                                  bc.w),
                                                  constraints,
                                                  velocity_mask);
        }
      }

      if (bc.type == BoundaryConditions::Type::velocity_mms)
      {
        ++n_velocity_mms;

        if (homogeneous)
        {

          VectorTools::interpolate_boundary_values(mapping,
                                                  dof_handler,
                                                  bc.id,
                                                  Functions::ZeroFunction<dim>(n_components),
                                                  constraints,
                                                  velocity_mask);
        }
        else
        {
          VectorTools::interpolate_boundary_values(mapping,
                                                  dof_handler,
                                                  bc.id,
                                                  exact_solution,
                                                  constraints,
                                                  velocity_mask);
        }


      }

      if (bc.type == BoundaryConditions::Type::slip)
      {
        ++n_slip;
        no_flux_boundaries.insert(bc.id);
      }

      if (bc.type == BoundaryConditions::Type::outflow)
      {
        ++n_outflow;
        outflow_boundaries.insert(bc.id);

      }

      if (bc.type == BoundaryConditions::Type::velocity_flux_mms)
      {
        ++n_velocity_flux_mms;

        // Enforce both the normal and tangential flux to be well-posed
        velocity_normal_flux_boundaries.insert(bc.id);
        velocity_normal_flux_functions[bc.id] = &exact_velocity;

        velocity_tangential_flux_boundaries.insert(bc.id);
        velocity_tangential_flux_functions[bc.id] = &exact_velocity;
      }
    }
    auto comm = dof_handler.get_mpi_communicator();

    MPI_Barrier(comm);
    VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                    u_lower,
                                                    no_flux_boundaries,
                                                    constraints,
                                                    mapping,
                                                    /*use_manifold_for_normal=*/false);
    MPI_Barrier(comm);

    VectorTools::compute_normal_flux_constraints(dof_handler,
                                                    u_lower,
                                                    outflow_boundaries,
                                                    constraints,
                                                    mapping,
                                                    /*use_manifold_for_normal=*/false);

    VectorTools::compute_nonzero_normal_flux_constraints(dof_handler,
                                                        u_lower,
                                                        velocity_normal_flux_boundaries,
                                                        velocity_normal_flux_functions,
                                                        constraints,
                                                        mapping,/*use_manifold_for_normal=*/false);

    VectorTools::compute_nonzero_tangential_flux_constraints(dof_handler,
                                                            u_lower,
                                                            velocity_tangential_flux_boundaries,
                                                            velocity_tangential_flux_functions,
                                                            constraints,
                                                            mapping,
                                                            /*use_manifold_for_normal=*/false);
  }

  template <int dim>
  void apply_mesh_position_boundary_conditions(
    const bool             homogeneous,
    const unsigned int     x_lower,
    const unsigned int     n_components,
    const DoFHandler<dim> &dof_handler,
    const Mapping<dim>    &mapping,
    const std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
                              &pseudosolid_bc,
    const Function<dim>       &exact_solution,
    const Function<dim>       &exact_mesh_position,
    AffineConstraints<double> &constraints)
  {

    const FEValuesExtractors::Vector position(x_lower);
    const ComponentMask              position_mask =
      dof_handler.get_fe().component_mask(position);

    Functions::ZeroFunction<dim> zero_fun(n_components);
    FixedMeshPosition<dim>       fixed_mesh(x_lower, n_components);
    const Function<dim>         *fun_ptr;

    FixedMeshPosition<dim>       fixed_mesh_for_flux(0, dim);
    std::set<types::boundary_id> normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *> position_flux_functions;
    std::set<types::boundary_id> mms_normal_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *>
      mms_position_flux_functions;
    for (const auto &[id, bc] : pseudosolid_bc)
    {
      if (bc.type == BoundaryConditions::Type::fixed)
      {
        if (homogeneous)
          fun_ptr = &zero_fun;
        else
          fun_ptr = &fixed_mesh;
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler, bc.id, *fun_ptr, constraints, position_mask);
      }
      if (bc.type == BoundaryConditions::Type::input_function)
      {
        // TODO: Prescribed but non-fixed mesh position?
        if (!homogeneous)
          DEAL_II_NOT_IMPLEMENTED();

        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 position_mask);
      }
      if (bc.type == BoundaryConditions::Type::position_mms)
      {
        fun_ptr = homogeneous ? &zero_fun : &exact_solution;
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler, bc.id, *fun_ptr, constraints, position_mask);
      }

      if (bc.type == BoundaryConditions::Type::no_flux)
      {
        normal_flux_boundaries.insert(bc.id);
        position_flux_functions[bc.id] = &fixed_mesh_for_flux;
      }
      if (bc.type == BoundaryConditions::Type::position_flux_mms)
      {
        mms_normal_flux_boundaries.insert(bc.id);
        mms_position_flux_functions[bc.id] = &exact_mesh_position;
      }
      // FIXME: Error if BC not handled?
    }

    // Add position nonzero flux constraints (tangential movement)
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      x_lower,
      normal_flux_boundaries,
      position_flux_functions,
      constraints,
      mapping,
      /*use_manifold_for_normal=*/false);

    // Add position nonzero flux constraints from manufactured solution
    // (tangential movement)
    VectorTools::compute_nonzero_normal_flux_constraints(
      dof_handler,
      x_lower,
      mms_normal_flux_boundaries,
      mms_position_flux_functions,
      constraints,
      mapping,
      /*use_manifold_for_normal=*/false);
  }

  template <int dim>
  void
  constrain_pressure_point(const DoFHandler<dim>     &dof_handler,
                           const IndexSet            &locally_relevant_dofs,
                           const Mapping<dim>        &mapping,
                           const Function<dim>       &exact_solution,
                           const unsigned int         p_lower,
                           const bool                 set_to_zero,
                           AffineConstraints<double> &constraints,
                           types::global_dof_index   &constrained_pressure_dof,
                           Point<dim>       &constrained_pressure_support_point,
                           const Point<dim> &reference_point)
  {
    // Determine the pressure dof the first time
    if (constrained_pressure_dof == numbers::invalid_dof_index)
    {
      const FEValuesExtractors::Scalar pressure(p_lower);
      const ComponentMask              pressure_mask =
        dof_handler.get_fe().component_mask(pressure);

      IndexSet pressure_dofs =
        DoFTools::extract_dofs(dof_handler, pressure_mask);

      // Get support points for locally relevant DoFs
      std::map<types::global_dof_index, Point<dim>> support_points =
        DoFTools::map_dofs_to_support_points(mapping, dof_handler);

      double local_min_dist             = std::numeric_limits<double>::max();
      types::global_dof_index local_dof = numbers::invalid_dof_index;

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
                    dof_handler.get_mpi_communicator());

      constrained_pressure_dof = global_pair.dof;

      // Set support point for MMS evaluation
      if (locally_relevant_dofs.is_element(constrained_pressure_dof))
      {
        constrained_pressure_support_point =
          support_points[constrained_pressure_dof];
      }
    }

    // Constrain that DoF if owned or ghosted
    if (locally_relevant_dofs.is_element(constrained_pressure_dof))
    {
      if (constraints.can_store_line(constrained_pressure_dof) &&
          !constraints.is_constrained(constrained_pressure_dof))
      {
        constraints.add_line(constrained_pressure_dof);
        if (set_to_zero)
          constraints.constrain_dof_to_zero(constrained_pressure_dof);
        else
        {
          const double pAnalytic =
            exact_solution.value(constrained_pressure_support_point, p_lower);
          constraints.set_inhomogeneity(constrained_pressure_dof, pAnalytic);
        }
      }
    }
  }

  template <int dim>
  void create_zero_mean_pressure_constraints_data(
    const Triangulation<dim>   &tria,
    const DoFHandler<dim>      &dof_handler,
    IndexSet                   &locally_relevant_dofs,
    std::vector<unsigned char> &dofs_to_component,
    const Mapping<dim>         &mapping,
    const Quadrature<dim>      &quadrature,
    const unsigned int          p_lower,
    types::global_dof_index    &constrained_pressure_dof,
    std::vector<std::pair<types::global_dof_index, double>> &constraint_weights)
  {
    const FEValuesExtractors::Scalar pressure(p_lower);
    const ComponentMask              pressure_mask =
      dof_handler.get_fe().component_mask(pressure);

    /**
     * One pressure dof will be coupled with all other pressure dofs,
     * which are not in the list of locally relevant dofs. Add them.
     */
    IndexSet local_pressure_dofs =
      DoFTools::extract_dofs(dof_handler, pressure_mask);

    // const unsigned int n_local_pressure_dofs =
    // local_pressure_dofs.n_elements();

    // Gather all lists to all processes
    std::vector<std::vector<types::global_dof_index>> gathered_dofs =
      Utilities::MPI::all_gather(dof_handler.get_mpi_communicator(),
                                 local_pressure_dofs.get_index_vector());

    std::vector<types::global_dof_index> gathered_dofs_flattened;
    for (const auto &vec : gathered_dofs)
      gathered_dofs_flattened.insert(gathered_dofs_flattened.end(),
                                     vec.begin(),
                                     vec.end());

    std::sort(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());

    // Add the pressure DoFs to the list of locally relevant dofs
    // FIXME: do this only if the proc has the constrained pressure dof has
    // owned or relevant?
    locally_relevant_dofs.add_indices(gathered_dofs_flattened.begin(),
                                      gathered_dofs_flattened.end());
    locally_relevant_dofs.compress();

    // If the dofs_to_component map was not created, create it and specify that
    // the added non-local dofs are pressure dofs
    if (dofs_to_component.empty())
      fill_dofs_to_component(dof_handler,
                             locally_relevant_dofs,
                             dofs_to_component);
    AssertDimension(dofs_to_component.size(), locally_relevant_dofs.n_elements());
    for (const auto dof : gathered_dofs_flattened)
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)] = p_lower;

    //
    // Compute integral of p over partition
    //
    std::map<types::global_dof_index, double> coeffs;

    const auto   &fe = dof_handler.get_fe();
    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_JxW_values);

    const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dofs);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          const double JxW = fe_values.JxW(q);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_cell; ++i_dof)
          {
            const unsigned int comp = fe.system_to_component_index(i_dof).first;

            // Here we need to account for ghost DoF (not only owned), which
            // contribute to the integral on this element
            if (!locally_relevant_dofs.is_element(local_dofs[i_dof]))
              continue;

            if (comp == p_lower)
            {
              const types::global_dof_index pressure_dof = local_dofs[i_dof];
              const double phi_i = fe_values.shape_value(i_dof, q);
              coeffs[pressure_dof] += phi_i * JxW;
            }
          }
        }
      }
    }

    //
    // Gather the constraint weights
    //
    {
      std::vector<std::pair<types::global_dof_index, double>> coeffs_vec(
        coeffs.begin(), coeffs.end());
      std::vector<std::vector<std::pair<unsigned int, double>>> gathered =
        Utilities::MPI::all_gather(dof_handler.get_mpi_communicator(),
                                   coeffs_vec);

      // Sum contributions to same DoF from different processes
      coeffs.clear();
      for (const auto &vec : gathered)
        for (const auto &[p_dof, partial_weight] : vec)
          coeffs[p_dof] += partial_weight;
    }

    // Sanity check : sum of coefficients should be measure of domain
    const double vol = GridTools::volume(tria, mapping);
    double       sum = 0.;
    for (const auto &[p_dof, val] : coeffs)
      sum += val;
    AssertThrow(
      (std::abs(sum - vol) / std::abs(vol)) < 1e-5,
      ExcMessage(
        "Sum of the constraints weights to enforce zero-mean pressure should "
        "be equal to the domain's volume, but it's not: sum of weights = " +
        std::to_string(sum) + " and domain volume = " + std::to_string(vol)));

    // First global pressure dof will be constrained, on the procs
    // for which it is owned or ghosted
    std::vector<std::pair<types::global_dof_index, double>> coeffs_vec(
      coeffs.begin(), coeffs.end());
    constrained_pressure_dof = coeffs_vec[0].first;
    const double a_0         = coeffs_vec[0].second;

    coeffs_vec.erase(coeffs_vec.begin());

    for (auto &[p_dof, val] : coeffs_vec)
      val /= -a_0;

    constraint_weights = coeffs_vec;
  }

  void add_zero_mean_pressure_constraints(
    AffineConstraints<double>     &constraints,
    const IndexSet                &locally_relevant_dofs,
    const types::global_dof_index &constrained_pressure_dof,
    const std::vector<std::pair<types::global_dof_index, double>>
      &constraint_weights)
  {
    if (locally_relevant_dofs.is_element(constrained_pressure_dof))
    {
      constraints.add_line(constrained_pressure_dof);
      constraints.add_entries(constrained_pressure_dof, constraint_weights);
    }
  }

} // namespace BoundaryConditions

// Explicit instantiations
#include "boundary_conditions.inst"