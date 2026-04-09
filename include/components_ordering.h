#ifndef COMPONENT_ORDERING_H
#define COMPONENT_ORDERING_H

#include <deal.II/base/types.h>
#include <solver_info.h>

/**
 * A ComponentOrdering describes the lower and upper component indices for a
 * given solver. For now, all solvers work in a monolithic fashion, solving
 * their complete system of PDEs in a unique coupled system.
 *
 * Each derived class must specify the [lower, upper) interval for its
 * variables.
 *
 * FIXME: These orderings should be known at compile time.
 */
class ComponentOrdering
{
public:
  ComponentOrdering() {}

  static constexpr unsigned int invalid = dealii::numbers::invalid_unsigned_int;

  unsigned int n_components = invalid;
  // Fluid velocity
  unsigned int u_lower = invalid;
  unsigned int u_upper = invalid;
  // Pressure
  unsigned int p_lower = invalid;
  unsigned int p_upper = invalid;
  // Mesh position
  unsigned int x_lower = invalid;
  unsigned int x_upper = invalid;
  // Lagrange multiplier
  unsigned int l_lower = invalid;
  unsigned int l_upper = invalid;
  // Cahn-Hilliard tracer
  unsigned int phi_lower = invalid;
  unsigned int phi_upper = invalid;
  // Cahn-Hililard potential
  unsigned int mu_lower = invalid;
  unsigned int mu_upper = invalid;
  // Enlarged phase marker
  unsigned int psi_lower = invalid;
  unsigned int psi_upper = invalid;
  // Temperature
  unsigned int t_lower = invalid;
  unsigned int t_upper = invalid;

  /**
   * Return true if @p component is the queried variable
   */
  inline bool is_velocity(const unsigned int component) const
  {
    return u_lower <= component && component < u_upper;
  }
  inline bool is_pressure(const unsigned int component) const
  {
    return p_lower == component;
  }
  inline bool is_position(const unsigned int component) const
  {
    return x_lower <= component && component < x_upper;
  }
  inline bool is_lambda(const unsigned int component) const
  {
    return l_lower <= component && component < l_upper;
  }
  inline bool is_tracer(const unsigned int component) const
  {
    return phi_lower == component;
  }
  inline bool is_potential(const unsigned int component) const
  {
    return mu_lower == component;
  }
  inline bool is_psi(const unsigned int component) const
  {
    return psi_lower == component;
  }
  inline bool is_temperature(const unsigned int component) const
  {
    return t_lower == component;
  }

  /**
   * Return true if the solver with this ordering support the queried variable
   */
  inline bool has_variable(const SolverInfo::VariableType variable_type) const
  {
    using Type = SolverInfo::VariableType;
    if (variable_type == Type::velocity)
      return u_lower != invalid;
    else if (variable_type == Type::pressure)
      return p_lower != invalid;
    else if (variable_type == Type::mesh_position)
      return x_lower != invalid;
    else if (variable_type == Type::lagrange_mult)
      return l_lower != invalid;
    else if (variable_type == Type::phase_tracer)
      return phi_lower != invalid;
    else if (variable_type == Type::phase_potential)
      return mu_lower != invalid;
    else if (variable_type == Type::phase_enlarged)
      return psi_lower != invalid;
    else if (variable_type == Type::temperature)
      return t_lower != invalid;
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }

  inline SolverInfo::VariableType
  component_to_variable_type(const unsigned int component) const
  {
    if (is_velocity(component))
      return SolverInfo::VariableType::velocity;
    else if (is_pressure(component))
      return SolverInfo::VariableType::pressure;
    else if (is_position(component))
      return SolverInfo::VariableType::mesh_position;
    else if (is_lambda(component))
      return SolverInfo::VariableType::lagrange_mult;
    else if (is_tracer(component))
      return SolverInfo::VariableType::phase_tracer;
    else if (is_potential(component))
      return SolverInfo::VariableType::phase_potential;
    else if (is_psi(component))
      return SolverInfo::VariableType::phase_enlarged;
    else if (is_temperature(component))
      return SolverInfo::VariableType::temperature;
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }
};

/**
 * Components ordering for the heat equation solver.
 */
class ComponentOrderingHeat : public ComponentOrdering
{
public:
  ComponentOrderingHeat()
    : ComponentOrdering()
  {
    n_components = 1;
    t_lower      = 0;
    t_upper      = 1;
  }
};

/**
 * Components ordering for the incompressible Navier-Stokes solver without ALE.
 */
template <int dim>
class ComponentOrderingNS : public ComponentOrdering
{
public:
  ComponentOrderingNS()
    : ComponentOrdering()
  {
    n_components = dim + 1;
    u_lower      = 0;
    u_upper      = dim;
    p_lower      = dim;
    p_upper      = dim + 1;
  }
};

/**
 * Components ordering for the incompressible Navier-Stokes solver with Lagrange
 * multiplier.
 */
template <int dim>
class ComponentOrderingNSLambda : public ComponentOrdering
{
public:
  ComponentOrderingNSLambda()
    : ComponentOrdering()
  {
    n_components = 2 * dim + 1;
    u_lower      = 0;
    u_upper      = dim;
    p_lower      = dim;
    p_upper      = dim + 1;
    l_lower      = dim + 1;
    l_upper      = 2 * dim + 1;
  }
};

/**
 * Components ordering for the compressible Navier-Stokes solver.
 */
template <int dim>
class ComponentOrderingCompressibleNS : public ComponentOrdering
{
public:
  ComponentOrderingCompressibleNS()
    : ComponentOrdering()
  {
    n_components = dim + 2;
    u_lower      = 0;
    u_upper      = dim;
    p_lower      = dim;
    p_upper      = dim + 1;
    t_lower      = dim + 1;
    t_upper      = dim + 2;
  }
};

/**
 * Components ordering for the fluid-structure solver with ALE.
 */
template <int dim>
class ComponentOrderingFSI : public ComponentOrdering
{
public:
  ComponentOrderingFSI()
    : ComponentOrdering()
  {
    n_components = 3 * dim + 1;
    u_lower      = 0;
    u_upper      = dim;
    p_lower      = dim;
    p_upper      = dim + 1;
    x_lower      = dim + 1;
    x_upper      = 2 * dim + 1;
    l_lower      = 2 * dim + 1;
    l_upper      = 3 * dim + 1;
  }
};

template <int dim>
class ConstexprComponentOrderingFSI
{
public:
  constexpr ConstexprComponentOrderingFSI() = default;

  static constexpr unsigned int n_components = 3 * dim + 1;
  static constexpr unsigned int u_lower      = 0;
  static constexpr unsigned int u_upper      = dim;
  static constexpr unsigned int p_lower      = dim;
  static constexpr unsigned int p_upper      = dim + 1;
  static constexpr unsigned int x_lower      = dim + 1;
  static constexpr unsigned int x_upper      = 2 * dim + 1;
  static constexpr unsigned int l_lower      = 2 * dim + 1;
  static constexpr unsigned int l_upper      = 3 * dim + 1;
};

/**
 * Components ordering for the quasi-incompressible Cahn-Hilliard Navier-Stokes
 * solver without ALE.
 */



template <int dim,
          bool with_moving_mesh = false,
          bool with_enlarged    = false>
class ConstexprComponentOrderingCHNS
{
public:
  constexpr ConstexprComponentOrderingCHNS() = default;

  // The enlarged ALE layout appends psi after mu to minimize disruption in
  // the existing CHNS component ordering.
  static constexpr unsigned int n_components =
    with_moving_mesh ?
      (with_enlarged ? (2 * dim + 4) : (2 * dim + 3)) :
      (dim + 3);
  static constexpr unsigned int u_lower = 0;
  static constexpr unsigned int u_upper = dim;
  static constexpr unsigned int p_lower = dim;
  static constexpr unsigned int p_upper = dim + 1;
  static constexpr unsigned int x_lower = dim + 1;
  static constexpr unsigned int x_upper =
    with_moving_mesh ? 2 * dim + 1 : dim + 1;
  static constexpr unsigned int phi_lower =
    with_moving_mesh ? 2 * dim + 1 : dim + 1;
  static constexpr unsigned int phi_upper =
    with_moving_mesh ? 2 * dim + 2 : dim + 2;
  static constexpr unsigned int mu_lower =
    with_moving_mesh ? 2 * dim + 2 : dim + 2;
  static constexpr unsigned int mu_upper =
    with_moving_mesh ? 2 * dim + 3 : dim + 3;
  static constexpr unsigned int psi_lower =
    (with_moving_mesh && with_enlarged) ?
      (2 * dim + 3) :
      dealii::numbers::invalid_unsigned_int;
  static constexpr unsigned int psi_upper =
    (with_moving_mesh && with_enlarged) ?
      (2 * dim + 4) :
      dealii::numbers::invalid_unsigned_int;
};

template <int dim, bool with_moving_mesh, bool with_enlarged = false>
class ComponentOrderingCHNS : public ComponentOrdering
{
public:
  ComponentOrderingCHNS()
  {
    using C      =
      ConstexprComponentOrderingCHNS<dim, with_moving_mesh, with_enlarged>;
    n_components = C::n_components;

    u_lower   = C::u_lower;
    u_upper   = C::u_upper;
    p_lower   = C::p_lower;
    p_upper   = C::p_upper;
    x_lower   = C::x_lower;
    x_upper   = C::x_upper;
    phi_lower = C::phi_lower;
    phi_upper = C::phi_upper;
    mu_lower  = C::mu_lower;
    mu_upper  = C::mu_upper;
    psi_lower = C::psi_lower;
    psi_upper = C::psi_upper;
  }
};

#endif
