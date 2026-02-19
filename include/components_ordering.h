#ifndef COMPONENT_ORDERING_H
#define COMPONENT_ORDERING_H

#include <deal.II/base/types.h>

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
  // Temperature
  unsigned int t_lower = invalid;
  unsigned int t_upper = invalid;

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
  inline bool is_temperature(const unsigned int component) const
  {
    return t_lower == component;
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
template <int dim>
class ComponentOrderingCHNS : public ComponentOrdering
{
public:
  ComponentOrderingCHNS()
    : ComponentOrdering()
  {
    n_components = dim + 3;
    u_lower      = 0;
    u_upper      = dim;
    p_lower      = dim;
    p_upper      = dim + 1;
    phi_lower    = dim + 1;
    phi_upper    = dim + 2;
    mu_lower     = dim + 2;
    mu_upper     = dim + 3;
  }
};

template <int dim, bool with_moving_mesh = false>
class ConstexprComponentOrderingCHNS
{
public:
  constexpr ConstexprComponentOrderingCHNS() = default;

  static constexpr unsigned int n_components = dim + 3;
  static constexpr unsigned int u_lower      = 0;
  static constexpr unsigned int u_upper      = dim;
  static constexpr unsigned int p_lower      = dim;
  static constexpr unsigned int p_upper      = dim + 1;
  static constexpr unsigned int x_lower      = dim + 1;
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
};

#endif
