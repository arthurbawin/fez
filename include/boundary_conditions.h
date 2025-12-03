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
    cahn_hilliard
  };

  /**
   * Available boundary conditions
   */
  enum class Type
  {
    // Common
    none,
    input_function,

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
    coupled_to_fluid, // Couple to lagrange mult
    no_flux,          // Slip. Have to check what happens at corners, etc.
    position_mms,     // Enforce x = x_mms
    position_flux_mms // Enforce x \cdot n = x_mms \cdot n

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
    virtual void declare_parameters(ParameterHandler &prm) override;
    virtual void read_parameters(ParameterHandler &prm) override;
    virtual void set_time(const double) override {}
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
   */
  template <int dim>
  void create_zero_mean_pressure_constraints_data(
    const Triangulation<dim> &tria,
    const DoFHandler<dim>    &dof_handler,
    IndexSet                 &locally_relevant_dofs,
    const Mapping<dim>       &mapping,
    const Quadrature<dim>    &quadrature,
    const unsigned int        p_lower,
    types::global_dof_index  &constrained_pressure_dof,
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

} // namespace BoundaryConditions

/**
 * Flow velocity prescribed by individual ParsedFunctions u,v,w.
 * The parameter @p u_lower is the first velocity component in the
 * solution vector (zero-based).
 *
 * Note that the time dependency (if any) is accounted for by
 * setting the time of the underlying ParsedFunctions through set_time(t),
 * and calling e.g. u->value(p), and not u->value(p,t).
 */
template <int dim>
class ComponentwiseFlowVelocity : public Function<dim>
{
public:
  const unsigned int                              u_lower;
  std::shared_ptr<Functions::ParsedFunction<dim>> u;
  std::shared_ptr<Functions::ParsedFunction<dim>> v;
  std::shared_ptr<Functions::ParsedFunction<dim>> w;

public:
  ComponentwiseFlowVelocity(const unsigned int u_lower,
                            const unsigned int n_components,
                            std::shared_ptr<Functions::ParsedFunction<dim>> u,
                            std::shared_ptr<Functions::ParsedFunction<dim>> v,
                            std::shared_ptr<Functions::ParsedFunction<dim>> w)
    : Function<dim>(n_components)
    , u_lower(u_lower)
    , u(u)
    , v(v)
    , w(w)
  {}

  virtual double value(const Point<dim> &p,
                       unsigned int      component) const override
  {
    if (component == u_lower + 0)
      return u->value(p);
    if (component == u_lower + 1)
      return v->value(p);
    if (component == u_lower + 2)
      return w->value(p);
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
        boundary_conditions[id].read_parameters(prm);
      }
      prm.leave_subsection();
    }
  }
  prm.leave_subsection();
}

#endif