#ifndef COMPONENT_ORDERING_H
#define COMPONENT_ORDERING_H

#include <deal.II/base/types.h>

using namespace dealii;

/**
 *
 */
class ComponentOrdering
{
public:
  ComponentOrdering() {}

  static constexpr unsigned int invalid = numbers::invalid_unsigned_int;

  unsigned int n_components = invalid;
  unsigned int u_lower      = invalid;
  unsigned int u_upper      = invalid;
  unsigned int p_lower      = invalid;
  unsigned int p_upper      = invalid;
  unsigned int x_lower      = invalid;
  unsigned int x_upper      = invalid;
  unsigned int l_lower      = invalid;
  unsigned int l_upper      = invalid;
  unsigned int phi_lower    = invalid;
  unsigned int phi_upper    = invalid;
  unsigned int mu_lower     = invalid;
  unsigned int mu_upper     = invalid;

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

#endif