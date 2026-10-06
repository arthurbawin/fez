
#include <boundary_conditions.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_tools_geometry.h>

#include <limits>

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
    prm.declare_entry(
      "type",
      "none",
      Patterns::Selection(
        "none|input_function|outflow|no_slip|weak_no_slip|slip|"
        "weak_pressure|dirichlet_pressure|"
        "velocity_mms|velocity_flux_mms|pressure_mms|open_mms|"
        "no_tangential_flow|no_tangential_flow_with_weak_pressure"),
      "Type of fluid boundary condition");

    // Select which velocity components an input_function boundary constrains
    // strongly. A component left unconstrained can instead be handled by a no
    // normal flux added on the same boundary.
    prm.declare_entry("constrain_u",
                      "true",
                      Patterns::Bool(),
                      "Constrain x-velocity component on this boundary");
    prm.declare_entry("constrain_v",
                      "true",
                      Patterns::Bool(),
                      "Constrain y-velocity component on this boundary");
    prm.declare_entry(
      "constrain_w",
      "true",
      Patterns::Bool(),
      "Constrain z-velocity component on this boundary (3D only)");

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

    prm.enter_subsection("p");
    p->declare_parameters(prm);
    prm.leave_subsection();

    prm.declare_entry(
      "weak no slip tolerance",
      "1e-12",
      Patterns::Double(),
      "Throw an error if the velocity constraint on a no-slip boundary "
      "enforced with a Lagrange multiplier exceeds this tolerance");

    prm.enter_subsection("rigid body rotation");
    {
      prm.declare_entry("enable",
                        "false",
                        Patterns::Bool(),
                        "Enable/disable rigid-body rotation on this boundary");
      const std::string default_point = (dim == 2) ? "0, 0" : "0, 0, 0";
      prm.declare_entry("center of rotation",
                        default_point,
                        Patterns::List(Patterns::Double(), dim, dim, ","),
                        "Center of rotation");
      prm.enter_subsection("angular velocity");
      angular_velocity->declare_parameters(prm, (dim == 2) ? 1 : dim);
      prm.leave_subsection();
    }
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
    else if (parsed_type == "outflow")
      type = Type::outflow;
    else if (parsed_type == "no_tangential_flow")
      type = Type::no_tangential_flow;
    else if (parsed_type == "no_slip")
      type = Type::no_slip;
    else if (parsed_type == "weak_no_slip")
      type = Type::weak_no_slip;
    else if (parsed_type == "slip")
      type = Type::slip;
    else if (parsed_type == "velocity_mms")
      type = Type::velocity_mms;
    else if (parsed_type == "velocity_flux_mms")
      type = Type::velocity_flux_mms;
    else if (parsed_type == "pressure_mms")
      type = Type::pressure_mms;
    else if (parsed_type == "open_mms")
      type = Type::open_mms;
    else if (parsed_type == "weak_pressure")
      type = Type::weak_pressure;
    else if (parsed_type == "dirichlet_pressure")
      type = Type::dirichlet_pressure;
    else if (parsed_type == "no_tangential_flow_with_weak_pressure")
      type = Type::no_tangential_flow_with_weak_pressure;
    else if (parsed_type == "none")
      throw std::runtime_error(
        "Fluid boundary condition for boundary " + std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed fluid boundary conditions is smaller than "
        "the specified \"number\" field.");

    constrain_u = prm.get_bool("constrain_u");
    constrain_v = prm.get_bool("constrain_v");
    constrain_w = prm.get_bool("constrain_w");

    if constexpr (dim == 2)
      constrain_w = false;

    AssertThrow(
      type != Type::input_function || constrain_u || constrain_v || constrain_w,
      ExcMessage("Fluid BC " + std::to_string(this->id) +
                 ": at least one velocity component must be constrained."));

    prm.enter_subsection("u");
    u->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("v");
    v->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("w");
    w->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("p");
    p->parse_parameters(prm);
    prm.leave_subsection();

    weak_no_slip_tolerance = prm.get_double("weak no slip tolerance");

    prm.enter_subsection("rigid body rotation");
    {
      enable_rigid_body_rotation = prm.get_bool("enable");
      center_of_rotation =
        parse_rank_1_tensor<dim>(prm.get("center of rotation"));
      prm.enter_subsection("angular velocity");
      angular_velocity->parse_parameters(prm);
      prm.leave_subsection();
    }
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

    // Input component functions, same pattern as FluidBC u/v/w
    // For an input_function boundary, each component
    // subsection may additionally set "type = no_flux" to leave that component
    // free (handled by the pseudosolid solver) instead of constraining it to
    // its function. The default "input_function" constrains the component.
    const auto declare_component = [&](const std::string              &name,
                                       Functions::ParsedFunction<dim> &fun) {
      prm.enter_subsection(name);
      fun.declare_parameters(prm);
      prm.declare_entry(
        "type",
        "input_function",
        Patterns::Selection("input_function|no_flux"),
        "Per-component constraint: input_function constrains "
        "this component to its function; no_flux leaves it free");
      prm.leave_subsection();
    };

    declare_component("x", *x);
    declare_component("y", *y);
    declare_component("z", *z);
  }

  template <int dim>
  void PseudosolidBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type = PhysicsType::pseudosolid;
    physics_str  = "pseudosolid";

    const std::string parsed_type = prm.get("type");
    if (parsed_type == "fixed")
      type = Type::fixed;
    else if (parsed_type == "coupled_to_fluid")
      type = Type::coupled_to_fluid;
    else if (parsed_type == "no_flux")
      type = Type::no_flux;
    else if (parsed_type == "input_function")
      type = Type::input_function;
    else if (parsed_type == "position_mms")
      type = Type::position_mms;
    else if (parsed_type == "position_flux_mms")
      type = Type::position_flux_mms;
    else if (parsed_type == "none")
      throw std::runtime_error(
        "Pseudosolid boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");

    // Parse each component function and its optional per-component type. A
    // component whose type is "no_flux" is left free (not constrained to its
    // function); the default "input_function" constrains it.
    const auto parse_component = [&](const std::string              &name,
                                     Functions::ParsedFunction<dim> &fun,
                                     bool &constrain) {
      prm.enter_subsection(name);
      fun.parse_parameters(prm);
      constrain = (prm.get("type") == "input_function");
      prm.leave_subsection();
    };

    parse_component("x", *x, constrain_x);
    parse_component("y", *y, constrain_y);
    parse_component("z", *z, constrain_z);

    if constexpr (dim < 3)
      constrain_z = false;

    if (type == Type::input_function)
      AssertThrow(constrain_x || constrain_y || constrain_z,
                  ExcMessage("Pseudosolid BC " + std::to_string(this->id) +
                             ": at least one position component must be "
                             "constrained."));
  }

  template <int dim>
  void CahnHilliardBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|no_flux|dirichlet_mms|input_function"),
                      "Type of Cahn-Hilliard boundary condition");
    prm.enter_subsection("tracer");
    tracer->declare_parameters(prm);
    prm.leave_subsection();
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
    if (parsed_type == "input_function")
      type = Type::input_function;
    prm.enter_subsection("tracer");
    tracer->parse_parameters(prm);
    prm.leave_subsection();
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
                      Patterns::Selection(
                        "none|input_function|dirichlet_mms|no_flux|heat_flux"),
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
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "heat_flux")
      type = Type::heat_flux;
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

  // Wrapper around a Function that evaluates the wrapped function and records
  // the maximum L1 norm of the evaluated values.
  template <int dim>
  class FunctionWithMagnitude : public Function<dim>
  {
  public:
    FunctionWithMagnitude(const Function<dim> &function)
      : Function<dim>(function.n_components)
      , function(function)
    {}

    void vector_value(const Point<dim> &point,
                      Vector<double>   &values) const override
    {
      function.vector_value(point, values);
      magnitude = std::max(magnitude, values.l1_norm());
    }

    double get_magnitude() const { return magnitude; }

  private:
    const Function<dim> &function;
    mutable double       magnitude = 0.;
  };

  /**
   * Adds the flux constraints stored in @p boundary_constraints to
   * @p flux_constraints, an existing set of constraints. This set may already
   * store flux constraints involving the same vector components, in which case
   * the new constraints are inserted into the existing ones, resolving chains
   * of constraints if needed.
   *
   * By default, the behavior of deal.II's
   * VectorTools::compute_*_normal_flux_constraints is as follows:
   * - when providing the full list of boundaries with flux BC, the normal
   * vectors at corners are averaged if these corners lie on different cells
   * (they are not if the corner lies in a single cell).
   * - when providing individual boundaries one at a time, already constrained
   * vector components ("pivots") are skipped, and the subsequent flux
   * constraints are silently ignored.
   *
   * None of these are satisfactory when enforcing e.g. two zero flux conditions
   * on touching boundaries, which should yield a fully constrained vector at
   * the corner, which motivates the use of this function.
   *
   * Each deal.II affine constraint is written as
   *
   *     x_i = b + sum_j a_j x_j.
   *
   * Before combining boundary conditions, rewrite it as
   *
   *     x_i - sum_j a_j x_j = b.
   *
   * If one of these DoFs is already constrained by a previously processed
   * boundary, substitute its existing affine relation into the new equation.
   * This eliminates previously constrained pivots and leaves an equation
   * involving the remaining DoFs. If no independent equation remains, check
   * that the right-hand side is compatible with the existing constraints.
   */
  void add_independent_flux_constraints(
    AffineConstraints<double> &boundary_constraints,
    AffineConstraints<double> &flux_constraints,
    const double               prescribed_value_scale = 0.)
  {
    boundary_constraints.close();

    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    // As in deal.II, break near-ties consistently between MPI ranks.
    // Compare relative to the largest remaining coefficient below.
    constexpr double pivot_tolerance = 1e-10;

    std::vector<std::pair<types::global_dof_index, double>> coefficients,
      pending;
    for (const auto &line : boundary_constraints.get_lines())
    {
      coefficients.clear();
      pending.clear();
      pending.emplace_back(line.index, 1.);
      for (const auto &[index, weight] : line.entries)
        pending.emplace_back(index, -weight);
      double       rhs               = line.inhomogeneity;
      double       rhs_scale         = std::abs(rhs);
      double       coefficient_scale = 0.;
      unsigned int n_operations      = 1;
      while (!pending.empty())
      {
        // Not a structured binding: index is captured by a lambda below.
        const types::global_dof_index index  = pending.back().first;
        const double                  weight = pending.back().second;
        pending.pop_back();
        coefficient_scale += std::abs(weight);
        if (flux_constraints.is_constrained(index))
        {
          const double shift =
            weight * flux_constraints.get_inhomogeneity(index);
          rhs -= shift;
          rhs_scale += std::abs(shift);
          n_operations += 2;
          for (const auto &[other, coefficient] :
               *flux_constraints.get_constraint_entries(index))
          {
            pending.emplace_back(other, weight * coefficient);
            ++n_operations;
          }
        }
        else
        {
          const auto existing = std::find_if(coefficients.begin(),
                                             coefficients.end(),
                                             [index](const auto &entry) {
                                               return entry.first == index;
                                             });
          if (existing == coefficients.end())
            coefficients.emplace_back(index, weight);
          else
          {
            existing->second += weight;
            ++n_operations;
          }
        }
      }
      // Estimate cancellation roundoff from the number of operations and
      // the sizes of their terms. For prescribed positions, also account
      // for cancellation in deal.II's projection onto the normal.
      // A redundant equation must have a compatible right-hand side.
      const double roundoff =
        n_operations * epsilon / (1. - n_operations * epsilon);
      double largest_coefficient = 0.;
      for (const auto &[index, coefficient] : coefficients)
        largest_coefficient =
          std::max(largest_coefficient, std::abs(coefficient));
      if (largest_coefficient <= roundoff * coefficient_scale)
      {
        AssertThrow(
          std::abs(rhs) <=
            roundoff * (rhs_scale + coefficient_scale * prescribed_value_scale),
          ExcMessage(
            "Incompatible flux conditions at a boundary intersection."));
        continue;
      }
      // Scaling the comparison keeps a small but independent equation from
      // selecting a much smaller pivot just because both coefficients are
      // tiny.
      std::sort(coefficients.begin(), coefficients.end());
      auto pivot =
        std::find_if(coefficients.begin(),
                     coefficients.end(),
                     [largest_coefficient](const auto &entry) {
                       return std::abs(entry.second) / largest_coefficient >=
                              1. - pivot_tolerance;
                     });
      // Preserve deal.II's original pivot when it is still a largest entry.
      // Elimination can make a previously good pivot arbitrarily small.
      const auto original = std::find_if(coefficients.begin(),
                                         coefficients.end(),
                                         [&line](const auto &entry) {
                                           return entry.first == line.index;
                                         });
      if (original != coefficients.end() &&
          std::abs(original->second) / largest_coefficient >=
            1. - pivot_tolerance)
        pivot = original;
      flux_constraints.add_line(pivot->first);
      flux_constraints.set_inhomogeneity(pivot->first, rhs / pivot->second);
      for (const auto &[index, coefficient] : coefficients)
        if (index != pivot->first &&
            std::abs(coefficient / pivot->second) > epsilon)
          flux_constraints.add_entry(pivot->first,
                                     index,
                                     -coefficient / pivot->second);
    }
  }

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
    AffineConstraints<double> &constraints)
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const ComponentMask              velocity_mask =
      dof_handler.get_fe().component_mask(velocity);

    // Build the component mask of the velocity components an input_function
    // boundary constrains strongly. Components left out can be handled by a no
    // normal flux added on the same boundary.
    const auto make_partial_velocity_mask =
      [&](const BoundaryConditions::FluidBC<dim> &bc) -> ComponentMask {
      std::vector<bool> mask(n_components, false);
      if (bc.constrain_u)
        mask[u_lower + 0] = true;
      if constexpr (dim >= 2)
        if (bc.constrain_v)
          mask[u_lower + 1] = true;
      if constexpr (dim == 3)
        if (bc.constrain_w)
          mask[u_lower + 2] = true;
      return ComponentMask(mask);
    };

    std::set<types::boundary_id> no_flux_boundaries;
    std::set<types::boundary_id> no_tangential_flow_boundaries;
    std::map<types::boundary_id, const Function<dim> *>
                                 velocity_normal_flux_functions;
    std::set<types::boundary_id> velocity_tangential_flux_boundaries;
    std::map<types::boundary_id, const Function<dim> *>
      velocity_tangential_flux_functions;

    for (const auto &[id, bc] : fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::no_slip)
      {
        VectorTools::interpolate_boundary_values(mapping,
                                                 dof_handler,
                                                 bc.id,
                                                 Functions::ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 velocity_mask);
      }
      if (bc.type == BoundaryConditions::Type::input_function)
      {
        const ComponentMask partial_mask = make_partial_velocity_mask(bc);
        if (homogeneous)
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc.id,
                                                   Functions::ZeroFunction<dim>(
                                                     n_components),
                                                   constraints,
                                                   partial_mask);
        else
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            bc.id,
            VectorFunctionFromComponents<dim>(
              u_lower, n_components, *bc.u, *bc.v, *bc.w),
            constraints,
            partial_mask);
      }
      if (bc.type == BoundaryConditions::Type::velocity_mms)
      {
        if (homogeneous)
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc.id,
                                                   Functions::ZeroFunction<dim>(
                                                     n_components),
                                                   constraints,
                                                   velocity_mask);
        else
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc.id,
                                                   exact_solution,
                                                   constraints,
                                                   velocity_mask);
      }
      if (bc.type == BoundaryConditions::Type::slip)
        no_flux_boundaries.insert(bc.id);
      if (bc.type == BoundaryConditions::Type::no_tangential_flow ||
          bc.type ==
            BoundaryConditions::Type::no_tangential_flow_with_weak_pressure)
        no_tangential_flow_boundaries.insert(bc.id);
      if (bc.type == BoundaryConditions::Type::velocity_flux_mms)
      {
        // Enforce both the normal and tangential flux to be well-posed
        velocity_normal_flux_functions[bc.id] = &exact_velocity;
        velocity_tangential_flux_boundaries.insert(bc.id);
        velocity_tangential_flux_functions[bc.id] = &exact_velocity;
      }
    }

    // Add no velocity flux constraints.
    // Do this one boundary at a time, and substitute new constraints into
    // existing ones to prevent skipping already constrained dofs and guarantee
    // compatible flux constraints at corners.
    AffineConstraints<double> flux_constraints, boundary_constraints;
    flux_constraints.reinit(constraints.get_locally_owned_indices(),
                            constraints.get_local_lines());
    for (const auto boundary_id : no_flux_boundaries)
    {
      boundary_constraints.reinit(constraints.get_locally_owned_indices(),
                                  constraints.get_local_lines());
      VectorTools::compute_no_normal_flux_constraints(
        dof_handler,
        u_lower,
        {boundary_id},
        boundary_constraints,
        mapping,
        /*use_manifold_for_normal=*/false);
      add_independent_flux_constraints(boundary_constraints, flux_constraints);
    }

    // Add normal velocity constraints
    for (const auto boundary_id : no_tangential_flow_boundaries)
    {
      boundary_constraints.reinit(constraints.get_locally_owned_indices(),
                                  constraints.get_local_lines());
      VectorTools::compute_normal_flux_constraints(
        dof_handler,
        u_lower,
        {boundary_id},
        boundary_constraints,
        mapping,
        /*use_manifold_for_normal=*/false);
      add_independent_flux_constraints(boundary_constraints, flux_constraints);
    }

    // Add prescribed normal flux constraints
    for (const auto &[id, fun] : velocity_normal_flux_functions)
    {
      boundary_constraints.reinit(constraints.get_locally_owned_indices(),
                                  constraints.get_local_lines());
      VectorTools::compute_nonzero_normal_flux_constraints(
        dof_handler,
        u_lower,
        {id},
        {{id, fun}},
        boundary_constraints,
        mapping,
        /*use_manifold_for_normal=*/false);
      add_independent_flux_constraints(boundary_constraints, flux_constraints);
    }

    // Add nonzero tangential flux velocity constraints
    VectorTools::compute_nonzero_tangential_flux_constraints(
      dof_handler,
      u_lower,
      velocity_tangential_flux_boundaries,
      velocity_tangential_flux_functions,
      constraints,
      mapping,
      /*use_manifold_for_normal=*/false);

    // Keep the existing precedence of strong Dirichlet constraints.
    if (flux_constraints.n_constraints() > 0)
      constraints.merge(flux_constraints,
                        AffineConstraints<double>::left_object_wins);
  }

  template <int dim>
  void apply_pressure_boundary_conditions(
    const bool             homogeneous,
    const unsigned int     p_lower,
    const unsigned int     n_components,
    const DoFHandler<dim> &dof_handler,
    const Mapping<dim>    &mapping,
    const std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>>
                              &fluid_bc,
    const Function<dim>       &exact_solution,
    AffineConstraints<double> &constraints)
  {
    (void)exact_solution;

    const FEValuesExtractors::Scalar pressure(p_lower);
    const ComponentMask              pressure_mask =
      dof_handler.get_fe().component_mask(pressure);

    for (const auto &[boundary_id, bc] : fluid_bc)
    {
      (void)boundary_id;

      if (bc.type == BoundaryConditions::Type::dirichlet_pressure)
      {
        if (homogeneous)
        {
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc.id,
                                                   Functions::ZeroFunction<dim>(
                                                     n_components),
                                                   constraints,
                                                   pressure_mask);
        }
        else
        {
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            bc.id,
            ScalarFunctionFromComponents<dim>(p_lower, n_components, *bc.p),
            constraints,
            pressure_mask);
        }
      }
    }
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
    FixedMeshPosition<dim>       fixed_mesh_for_flux(0, dim);
    const Function<dim>         *fun_ptr;

    // Function objects used when enforcing a prescribed position flux (i.e.,
    // movement is only allowed to slide along the boundary). These functions
    // store the position scale, to check compatibility of nearly cancelling
    // normal constraints. See also the translated-wall case in the
    // boundary_flux_constraints.cc test.
    FunctionWithMagnitude<dim> fixed_position(fixed_mesh_for_flux);
    FunctionWithMagnitude<dim> mms_position(exact_mesh_position);

    std::map<types::boundary_id, const Function<dim> *> position_flux_functions;
    std::map<types::boundary_id, const Function<dim> *>
      mms_position_flux_functions;

    // Build the component mask of the position components an input_function
    // boundary constrains strongly. A component left out (type = no_flux in its
    // subsection) is determined by the pseudosolid solver instead.
    const auto make_partial_position_mask =
      [&](const BoundaryConditions::PseudosolidBC<dim> &bc) -> ComponentMask {
      std::vector<bool> mask(n_components, false);
      if (bc.constrain_x)
        mask[x_lower + 0] = true;
      if constexpr (dim >= 2)
        if (bc.constrain_y)
          mask[x_lower + 1] = true;
      if constexpr (dim == 3)
        if (bc.constrain_z)
          mask[x_lower + 2] = true;
      return ComponentMask(mask);
    };

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
        const ComponentMask partial_mask = make_partial_position_mask(bc);
        if (homogeneous)
          VectorTools::interpolate_boundary_values(
            mapping, dof_handler, bc.id, zero_fun, constraints, partial_mask);
        else
          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            bc.id,
            VectorFunctionFromComponents<dim>(
              x_lower, n_components, *bc.x, *bc.y, *bc.z),
            constraints,
            partial_mask);
      }
      if (bc.type == BoundaryConditions::Type::position_mms)
      {
        fun_ptr = homogeneous ? &zero_fun : &exact_solution;
        VectorTools::interpolate_boundary_values(
          mapping, dof_handler, bc.id, *fun_ptr, constraints, position_mask);
      }

      if (bc.type == BoundaryConditions::Type::no_flux)
        position_flux_functions[bc.id] = &fixed_position;
      if (bc.type == BoundaryConditions::Type::position_flux_mms)
        mms_position_flux_functions[bc.id] = &mms_position;
      // FIXME: Error if BC not handled?
    }

    AffineConstraints<double> flux_constraints, boundary_constraints;
    flux_constraints.reinit(constraints.get_locally_owned_indices(),
                            constraints.get_local_lines());

    // As for velocity, apply each wall separately to preserve corner
    // conditions.
    const auto add_position_flux_constraints =
      [&](
        const std::map<types::boundary_id, const Function<dim> *> &functions) {
        for (const auto &[boundary_id, function] : functions)
        {
          boundary_constraints.reinit(constraints.get_locally_owned_indices(),
                                      constraints.get_local_lines());
          VectorTools::compute_nonzero_normal_flux_constraints(
            dof_handler,
            x_lower,
            {boundary_id},
            {{boundary_id, function}},
            boundary_constraints,
            mapping,
            /*use_manifold_for_normal=*/false);
          // Use the same tolerance scale on all ranks so they agree on
          // whether shared boundary constraints are compatible.
          const double local_scale = std::max(fixed_position.get_magnitude(),
                                              mms_position.get_magnitude());
          const double prescribed_value_scale =
            Utilities::MPI::max(local_scale,
                                dof_handler.get_mpi_communicator());
          add_independent_flux_constraints(boundary_constraints,
                                           flux_constraints,
                                           prescribed_value_scale);
        }
      };

    // Add position nonzero flux constraints (tangential movement)
    add_position_flux_constraints(position_flux_functions);

    // Add position nonzero flux constraints from manufactured solution
    // (tangential movement)
    add_position_flux_constraints(mms_position_flux_functions);

    if (flux_constraints.n_constraints() > 0)
      constraints.merge(flux_constraints,
                        AffineConstraints<double>::left_object_wins);
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

    // (Re-)create the dofs_to_component map and specify that
    // the added non-local dofs are pressure dofs
    fill_dofs_to_component(dof_handler,
                           locally_relevant_dofs,
                           dofs_to_component);
    AssertDimension(dofs_to_component.size(),
                    locally_relevant_dofs.n_elements());
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
