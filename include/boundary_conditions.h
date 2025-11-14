#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/lac/vector.h>

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
    pseudosolid
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
    velocity_mms,      // Set velocity to prescribed manufactured solution
    velocity_flux_mms, // Set u dot n = u_mms dot n
    // Set (-pI + nu*grad(u)) \cdot n = (-p_mmsI + nu*grad(u_mms)) \cdot n
    open_mms,

    // Pseudo_solid
    fixed, // Enforce 0 displacement. Default when no BC is prescribed?
    coupled_to_fluid, // Couple to lagrange mult
    no_flux           // Slip. Have to check what happens at corners, etc.
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

  // FIXME: templatize the "declare" and "read" functions below,
  // so that one function handles all types of BC vectors.
  template <int dim>
  void declare_fluid_boundary_conditions(ParameterHandler  &prm,
                                         const unsigned int n_fluid_bc)
  {
    // These boundary conditions are used to declare the generic parameters
    // (problem is that map keys are immutable after they are created, although
    // we could alternatively move or swap the map entries).
    std::vector<FluidBC<dim>> tmp_bc(n_fluid_bc);

    prm.enter_subsection("Fluid boundary conditions");
    {
      // The number was already parsed in a first dry run
      prm.declare_entry("number",
                        "0",
                        Patterns::Integer(),
                        "Number of fluid boundary conditions");

      for (unsigned int i = 0; i < n_fluid_bc; ++i)
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

  template <int dim>
  void read_fluid_boundary_conditions(
    ParameterHandler                           &prm,
    const unsigned int                          n_fluid_bc,
    std::map<types::boundary_id, FluidBC<dim>> &fluid_bc)
  {
    /**
     * FIXME: If the number of bc actually added in the parameter file is
     * greater than the specified number, deal.ii throws a parse error.
     Ideally,
     * we should be able to return an error here if the user specified too
     few
     * boundary conditions.
     */
    prm.enter_subsection("Fluid boundary conditions");
    {
      for (unsigned int i = 0; i < n_fluid_bc; ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          unsigned int id = prm.get_integer("id");
          fluid_bc[id].read_parameters(prm);
        }
        prm.leave_subsection();
      }
    }
    prm.leave_subsection();
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
  void
  declare_pseudosolid_boundary_conditions(ParameterHandler  &prm,
                                          const unsigned int n_pseudosolid_bc)
  {
    std::vector<PseudosolidBC<dim>> tmp_bc(n_pseudosolid_bc);

    prm.enter_subsection("Pseudosolid boundary conditions");
    {
      prm.declare_entry("number",
                        "0",
                        Patterns::Integer(),
                        "Number of pseudosolid boundary conditions");

      for (unsigned int i = 0; i < n_pseudosolid_bc; ++i)
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

  template <int dim>
  void read_pseudosolid_boundary_conditions(
    ParameterHandler                                 &prm,
    const unsigned int                                n_pseudosolid_bc,
    std::map<types::boundary_id, PseudosolidBC<dim>> &pseudosolid_bc)
  {
    prm.enter_subsection("Pseudosolid boundary conditions");
    {
      for (unsigned int i = 0; i < n_pseudosolid_bc; ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          unsigned int id = prm.get_integer("id");
          pseudosolid_bc[id].read_parameters(prm);
        }
        prm.leave_subsection();
      }
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
                        "none|fixed|coupled_to_fluid|no_flux|input_function"),
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
    if (parsed_type == "none")
      throw std::runtime_error(
        "Pseudosolid boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");
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

#endif