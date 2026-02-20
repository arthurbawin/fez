#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <utilities.h>

using namespace dealii;

/**
 *
 */
namespace BoundaryConditions
{
  /**
   * Physics for which boundary conditions are available
   */
  enum class PhysicsType
  {
    fluid,
    pseudosolid,
    cahn_hilliard,
    heat
  };

  /**
   * Available boundary conditions
   */
  enum class Type
  {
    // Common
    none,
    input_function,
    dirichlet_mms,

    // Flow
    outflow,      // Do nothing
    no_slip,      // Enforce given functions
    weak_no_slip, // Check that lagrange mult is defined, couple
    slip,         // Enforce no_flux

    // These boundary conditions are for flow verification purposes:
    // Set velocity to prescribed manufactured solution
    velocity_mms,
    // Set both the normal and tangential flux to u dot n/t = u_mms dot n/t
    velocity_flux_mms,
    // Set (-pI + nu*grad(u)) \cdot n = (-p_mmsI + nu*grad(u_mms)) \cdot n
    open_mms,

    // Pseudo_solid
    fixed, // Enforce 0 displacement. Default when no BC is prescribed?
    coupled_to_fluid,  // Couple to lagrange mult
    no_flux,           // Slip. Have to check what happens at corners, etc.
    position_mms,      // Enforce x = x_mms
    position_flux_mms, // Enforce x \cdot n = x_mms \cdot n

    // Cahn-Hilliard
    // no_flux
  };

  /**
   * Base class for boundary conditions
   */
  class BoundaryCondition
  {
  public:
    // Physics associated to this boundary condition (type and string)
    PhysicsType physics_type;
    std::string physics_str;

    // Type of boundary condition
    Type type;

    // Id of the associated boundary
    types::boundary_id id;

    // The Gmsh name of the boundary entity
    std::string gmsh_name;

    /**
     * Declare the parameters common to all boundary conditions
     */
    virtual void declare_parameters(ParameterHandler &prm);
    virtual void read_parameters(ParameterHandler &prm);

    /**
     * Update the time in the underlying functions, if applicable.
     */
    virtual void set_time(const double new_time) = 0;
  };

  /**
   * A boundary condition for the incompressible Navier-Stokes equations.
   */
  template <int dim>
  class FluidBC : public BoundaryCondition
  {
  public:
    /**
     * The flow components imposed by a user function. Only used for bc of type
     * input_function. When declaring the possible entries of the parameter
     * file, we do not know beforehand which bc will be associated to a user
     * function. For now, *all* boundary conditions have functions.
     */
    std::shared_ptr<Functions::ParsedFunction<dim>> u;
    std::shared_ptr<Functions::ParsedFunction<dim>> v;
    std::shared_ptr<Functions::ParsedFunction<dim>> w;

  public:
    // Constructor. Allocates the pointers to the user functions.
    FluidBC()
    {
      u = std::make_shared<Functions::ParsedFunction<dim>>();
      v = std::make_shared<Functions::ParsedFunction<dim>>();
      w = std::make_shared<Functions::ParsedFunction<dim>>();
    };

    virtual void set_time(const double new_time) override
    {
      u->set_time(new_time);
      v->set_time(new_time);
      w->set_time(new_time);
    }

  public:
    virtual void declare_parameters(ParameterHandler &prm) override;
    virtual void read_parameters(ParameterHandler &prm) override;
  };

  /**
   * A boundary condition for the linear elasticity equations,
   * for the mesh movement analogy.
   */
  template <int dim>
  class PseudosolidBC : public BoundaryCondition
  {
  public:
    /// Prescribed mesh position (used only for input_function)
    std::shared_ptr<Functions::ParsedFunction<dim>> x;
    std::shared_ptr<Functions::ParsedFunction<dim>> y;
    std::shared_ptr<Functions::ParsedFunction<dim>> z;

  public:
    PseudosolidBC()
    {
      x = std::make_shared<Functions::ParsedFunction<dim>>();
      y = std::make_shared<Functions::ParsedFunction<dim>>();
      z = std::make_shared<Functions::ParsedFunction<dim>>();
    }

    virtual void set_time(const double new_time) override
    {
      x->set_time(new_time);
      y->set_time(new_time);
      z->set_time(new_time);
    }

    virtual void declare_parameters(ParameterHandler &prm) override;
    virtual void read_parameters(ParameterHandler &prm) override;
  };

  /**
   * A boundary condition for the Cahn-Hilliard tracer and potential.
   */
  template <int dim>
  class CahnHilliardBC : public BoundaryCondition
  {
  public:
    virtual void declare_parameters(ParameterHandler &prm) override;
    virtual void read_parameters(ParameterHandler &prm) override;
    virtual void set_time(const double) override {}
  };

  /**
   * A boundary condition for the heat equation.
   */
  template <int dim>
  class HeatBC : public BoundaryCondition
  {
  public:
    // User-defined temperature function for input_function boundary
    std::shared_ptr<Functions::ParsedFunction<dim>> temperature;

  public:
    HeatBC()
      : temperature(std::make_shared<Functions::ParsedFunction<dim>>())
    {}

    virtual void declare_parameters(ParameterHandler &prm) override;
    virtual void read_parameters(ParameterHandler &prm) override;
    virtual void set_time(const double new_time) override
    {
      temperature->set_time(new_time);
    }
  };

  /**
   * Declare a family of boundary conditions (fluid, pseudosolid, ...)
   */
  template <typename BCType>
  void declare_boundary_conditions(ParameterHandler  &prm,
                                   const unsigned int n_bc,
                                   const std::string &bc_type_name);
  /**
   * Parse a family of boundary conditions from the parameter file
   */
  template <typename BCType>
  void read_boundary_conditions(
    ParameterHandler                     &prm,
    const unsigned int                    n_bc,
    const std::string                    &bc_type_name,
    std::map<types::boundary_id, BCType> &boundary_conditions);

  /**
   *
   *
   */
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
    AffineConstraints<double> &constraints);

  /**
   *
   *
   */
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
    AffineConstraints<double> &constraints);

  /**
   *
   */
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
                           const Point<dim> &reference_point = Point<dim>());

  /**
   * Enforcing zero-mean pressure yields the linear constraint:
   *
   *  int_\Omega p dx = 0  -> sum_j a_j * p_j = 0,
   *
   * where the c_j are obtained from integrating the pressure shape functions
   * over \Omega. This is satisfied by constraining a single pressure DoF to
   *
   *  p_0 = - sum_{j != 0} a_j/a_0 * p_j := sum_{j != 0} c_j * p_j.
   *
   * This function computes the weights c_j and sets the pressure dof to
   * constrain "constrained_pressure_dof", which is simply the (globally) first
   * pressure DoF.
   *
   * Important: since this is a global constraint, this pressure dof will be
   * coupled to *all* other pressure dofs (on MPI processes which have this dof
   * as owned or ghost). This fills the sparsity pattern and is thus highly
   * inefficient. Enforcing zero-mean this way is only meant for verification on
   * corner-case (pun intended, see further) convergence studies, in particular
   * for 3D tests with some specific non-divergence-free velocity fields, which
   * show either reduced convergence order for the pressure or even no
   * convergence at all when setting a corner pressure DoF to zero.
   *
   * Also important: Note that this function adds all the non-local pressure
   * dofs to the vector of locally relevant dofs on ranks for which the
   * constrained pressure dof is relevant. This should be kept in mind when
   * performing operations on locally_relevant_dofs. For example, newly added
   * relevant pressure dofs do not have a matching support point in the map
   * retrieved from map_dofs_to_support_points.
   */
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
    std::vector<std::pair<types::global_dof_index, double>>
      &constraint_weights);

  /**
   * Given the weights and dof computed with the function above,
   * add the single zero-mean constraint on the specified pressure dof to
   * the passed constraints.
   *
   * Important: this yields a very inefficient sparsity pattern and should only
   * be used for specific verification tests, see above.
   */
  void add_zero_mean_pressure_constraints(
    AffineConstraints<double>     &constraints,
    const IndexSet                &locally_relevant_dofs,
    const types::global_dof_index &constrained_pressure_dof,
    const std::vector<std::pair<types::global_dof_index, double>>
      &constraint_weights);

  /**
   *
   */
  template <int dim, typename VectorType>
  void remove_mean_pressure(const ComponentMask   &pressure_mask,
                            const DoFHandler<dim> &dof_handler,
                            const double           mean_pressure,
                            VectorType            &solution)
  {
    const IndexSet owned_pressure_dofs =
      DoFTools::extract_dofs(dof_handler, pressure_mask);
    for (const auto i : owned_pressure_dofs)
      solution[i] -= mean_pressure;
    solution.compress(VectorOperation::add);
  }

  /**
   * Apply the given field name as solution for the whole volume and boundaries.
   * Not a boundary condition per se, but a similar constraint on the field
   * dofs.
   */
  template <int dim, typename VectorType>
  void apply_field_as_solution_on_volume_and_boundaries(
    const DoFHandler<dim>      &dof_handler,
    const Mapping<dim>         &mapping,
    const Function<dim>        &exact_solution,
    VectorType                 &present_solution,
    VectorType                 &local_present_solution,
    const IndexSet             &locally_relevant_dofs,
    std::vector<unsigned char> &dofs_to_component,
    const ComponentMask        &component_mask,
    const bool                  homogeneous,
    AffineConstraints<double>  &constraints);

} // namespace BoundaryConditions

/**
 * Vector-valued function described by up to 3 individual ParsedFunctions.
 * The parameter @p lower is the first vector component in the solution vector.
 *
 * This Function does no override the set_time function, instead it assumes
 * that time has been correctly updated in each underlying ParsedFunction.
 */
template <int dim>
class VectorFunctionFromComponents : public Function<dim>
{
public:
  const unsigned int                              lower;
  std::shared_ptr<Functions::ParsedFunction<dim>> x_component;
  std::shared_ptr<Functions::ParsedFunction<dim>> y_component;
  std::shared_ptr<Functions::ParsedFunction<dim>> z_component;

public:
  VectorFunctionFromComponents(
    const unsigned int                              lower,
    const unsigned int                              n_components,
    std::shared_ptr<Functions::ParsedFunction<dim>> x_component,
    std::shared_ptr<Functions::ParsedFunction<dim>> y_component,
    std::shared_ptr<Functions::ParsedFunction<dim>> z_component)
    : Function<dim>(n_components)
    , lower(lower)
    , x_component(x_component)
    , y_component(y_component)
    , z_component(z_component)
  {}

  virtual double value(const Point<dim> &p,
                       unsigned int      component) const override
  {
    if (component == lower + 0)
      return x_component->value(p);
    if (component == lower + 1)
      return y_component->value(p);
    if (component == lower + 2)
      return z_component->value(p);
    return 0.;
  }
};

/**
 * This function is meant to represent the spatial identity function,
 * and is used to enforce a no displacement boundary condition for a
 * pseudosolid mesh movement problem, that is, x(X) = X.
 *
 * The only trick is that depending on the problem, the index of the
 * position variables in the solution vector may vary, so this needs
 * to be specified by the @p x_lower parameter.
 *
 * Note that if we were solving for the displacement instead of the
 * position, this function would simply be the zero function.
 */
template <int dim>
class FixedMeshPosition : public Function<dim>
{
public:
  // Lower bound of the mesh position variable (first component)
  const unsigned int x_lower;

public:
  FixedMeshPosition(const unsigned int x_lower, const unsigned int n_components)
    : Function<dim>(n_components)
    , x_lower(x_lower)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &values) const override
  {
    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = p[d];
  }
};


/* ---------------- template and inline functions ----------------- */

template <typename BCType>
void BoundaryConditions::declare_boundary_conditions(
  ParameterHandler  &prm,
  const unsigned int n_bc,
  const std::string &bc_type_name)
{
  // These boundary conditions are used to declare the generic parameters
  // (problem is that map keys are immutable after they are created, although
  // we could alternatively move or swap the map entries).
  std::vector<BCType> tmp_bc(n_bc);

  prm.enter_subsection(bc_type_name + " boundary conditions");
  {
    // The number was already parsed in a first dry run
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of " + bc_type_name + " boundary conditions");

    for (unsigned int i = 0; i < n_bc; ++i)
    {
      prm.enter_subsection("boundary " + std::to_string(i));
      {
        tmp_bc[i].declare_parameters(prm);
      }
      prm.leave_subsection();
    }
  }
  prm.leave_subsection();
}

template <typename BCType>
void BoundaryConditions::read_boundary_conditions(
  ParameterHandler                     &prm,
  const unsigned int                    n_bc,
  const std::string                    &bc_type_name,
  std::map<types::boundary_id, BCType> &boundary_conditions)
{
  /**
   * FIXME: If the number of bc actually added in the parameter file is
   * greater than the specified number, deal.ii throws a parse error.
   * Ideally, we should be able to return an error here if the user specified
   * too few boundary conditions.
   */
  prm.enter_subsection(bc_type_name + " boundary conditions");
  {
    for (unsigned int i = 0; i < n_bc; ++i)
    {
      prm.enter_subsection("boundary " + std::to_string(i));
      {
        unsigned int id = prm.get_integer("id");
        AssertThrow(
          id != numbers::invalid_unsigned_int,
          ExcMessage(
            bc_type_name + " boundary condition " + std::to_string(i) +
            " could not be read, possibly because a boundary condition "
            "sequential number (not the id) is repeated, for instance:\n\n"
            "subsection boundary 1\n"
            "  set id   = 2\n"
            "  set name = boundary_2\n"
            "  set type = type\n"
            "  end\n"
            "subsection boundary 1 <====== 1 is repeated\n"
            "  set id   = 3\n"
            "  set name = boundary_3\n"
            "  set type = type\n"
            "end"));
        boundary_conditions[id].read_parameters(prm);
      }
      prm.leave_subsection();
    }
  }
  prm.leave_subsection();
}

template <int dim, typename VectorType>
void BoundaryConditions::apply_field_as_solution_on_volume_and_boundaries(
  const DoFHandler<dim>      &dof_handler,
  const Mapping<dim>         &mapping,
  const Function<dim>        &exact_solution,
  VectorType                 &present_solution,
  VectorType                 &local_present_solution,
  const IndexSet             &locally_relevant_dofs,
  std::vector<unsigned char> &dofs_to_component,
  const ComponentMask        &component_mask,
  const bool                  homogeneous,
  AffineConstraints<double>  &constraints)
{
  const auto zero_fun =
    Functions::ZeroFunction<dim>(exact_solution.n_components);
  const Function<dim> &f = homogeneous ? zero_fun : exact_solution;
  VectorTools::interpolate(
    mapping, dof_handler, f, local_present_solution, component_mask);

  present_solution = local_present_solution;

  if (dofs_to_component.empty())
    fill_dofs_to_component(dof_handler,
                           locally_relevant_dofs,
                           dofs_to_component);

  for (const auto &dof : locally_relevant_dofs)
  {
    const unsigned char comp =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];

    /**
     * Non-local dofs may have been added to locally_relevant_dofs after its
     * creation : for these dofs, the component index cannot be determined by
     * looping over the local cells, thus we cannot decide here if the missing
     * component index is the field currently being constrained... The map
     * dofs_to_component should be updated when adding these dofs to
     * locally_relevant_dofs.
     *
     * Currently, this happens when enforcing a zero mean pressure (which adds
     * all pressure dofs to some ranks), and when adding lambda dofs as relevant
     * in the FSI solver. The first case is prevented by checking the options in
     * parameter_reader.cpp, the other should not come up because it does not
     * make too much sense to set an exact Lagrange multiplier (and there is no
     * currently no way to do it in the parameter file).
     */
    Assert(
      comp != static_cast<unsigned char>(-1),
      ExcMessage(
        "You are trying to apply a prescribed exact field in parallel, but the "
        "component index for some dofs on this partition could not be "
        "determined, and "
        "thus the constraints for this field may be incomplete. This is likely "
        "because at some point, non-local dofs for this field were added to "
        "the vector of locally relevant dofs, but their components in the "
        "dofs_to_component map were not updated accordingly. To solve this, "
        "the dofs_to_component should be updated whenever ghost dofs are added "
        "to locally_relevant_dofs."));
    if (component_mask[comp])
      if (constraints.can_store_line(dof) && !constraints.is_constrained(dof))
        constraints.add_constraint(dof, {}, present_solution[dof]);
  }
}

#endif
