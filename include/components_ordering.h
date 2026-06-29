#ifndef COMPONENT_ORDERING_H
#define COMPONENT_ORDERING_H

#include <deal.II/base/types.h>
#include <deal.II/fe/fe_values_extractors.h>
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
  ComponentOrdering() = default;

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

  // Lagrange multiplier (for velocity constraints)
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
  inline bool is_temperature(const unsigned int component) const
  {
    return t_lower == component;
  }

  /**
   * Return true if the solver with this ordering support the queried variable
   */
  inline bool has_variable(const SolverInfo::VariableType variable_type) const
  {
    switch (variable_type)
    {
      case SolverInfo::VariableType::velocity:
        return u_lower != invalid;
      case SolverInfo::VariableType::pressure:
        return p_lower != invalid;
      case SolverInfo::VariableType::mesh_position:
        return x_lower != invalid;
      case SolverInfo::VariableType::lagrange_mult:
        return l_lower != invalid;
      case SolverInfo::VariableType::phase_tracer:
        return phi_lower != invalid;
      case SolverInfo::VariableType::phase_potential:
        return mu_lower != invalid;
      case SolverInfo::VariableType::temperature:
        return t_lower != invalid;
    }
    DEAL_II_ASSERT_UNREACHABLE();
  }

  /**
   * Return the first variable component. Return an invalid number if the solver
   * does not store the required variable, so it's best to check first if
   * has_variable() above returns true for that variable.
   */
  inline unsigned int variable_to_first_component(
    const SolverInfo::VariableType variable_type) const
  {
    switch (variable_type)
    {
      case SolverInfo::VariableType::velocity:
        return u_lower;
      case SolverInfo::VariableType::pressure:
        return p_lower;
      case SolverInfo::VariableType::mesh_position:
        return x_lower;
      case SolverInfo::VariableType::lagrange_mult:
        return l_lower;
      case SolverInfo::VariableType::phase_tracer:
        return phi_lower;
      case SolverInfo::VariableType::phase_potential:
        return mu_lower;
      case SolverInfo::VariableType::temperature:
        return t_lower;
    }
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
    else if (is_temperature(component))
      return SolverInfo::VariableType::temperature;
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }

  /**
   * Return true if the variable is scalar-valued, false otherwise.
   * Used to return a FEValueExtractor::Scalar.
   */
  inline bool is_scalar(const SolverInfo::VariableType variable_type) const
  {
    switch (variable_type)
    {
      case SolverInfo::VariableType::velocity:
        return false;
      case SolverInfo::VariableType::pressure:
        return true;
      case SolverInfo::VariableType::mesh_position:
        return false;
      case SolverInfo::VariableType::lagrange_mult:
        return false;
      case SolverInfo::VariableType::phase_tracer:
        return true;
      case SolverInfo::VariableType::phase_potential:
        return true;
      case SolverInfo::VariableType::temperature:
        return true;
    }
    DEAL_II_ASSERT_UNREACHABLE();
  }

  /**
   * Return a FEValueExtractor::Scalar for this variable. Expects a
   * scalar-valued variable, which can be checked with the function above.
   */
  inline dealii::FEValuesExtractors::Scalar
  get_scalar_extractor(const SolverInfo::VariableType variable_type) const
  {
    Assert(
      is_scalar(variable_type),
      dealii::StandardExceptions::ExcMessage(
        "Cannot return a FEValuesExtractors::Scalar for non-scalar variable"));
    switch (variable_type)
    {
      case SolverInfo::VariableType::velocity:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::pressure:
        return dealii::FEValuesExtractors::Scalar(p_lower);
      case SolverInfo::VariableType::mesh_position:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::lagrange_mult:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::phase_tracer:
        return dealii::FEValuesExtractors::Scalar(phi_lower);
      case SolverInfo::VariableType::phase_potential:
        return dealii::FEValuesExtractors::Scalar(mu_lower);
      case SolverInfo::VariableType::temperature:
        return dealii::FEValuesExtractors::Scalar(t_lower);
    }
    DEAL_II_ASSERT_UNREACHABLE();
  }

  /**
   * Return true if the variable is vector-valued, false otherwise.
   * Used to return a FEValueExtractor::Vector.
   */
  inline bool is_vector(const SolverInfo::VariableType variable_type) const
  {
    // There is no tensor-valued variable for now, so simply return !is_scalar
    return !is_scalar(variable_type);
  }

  /**
   * Return a FEValueExtractor::Vector for this variable. Expects a
   * vector-valued variable, which can be checked with the function above.
   */
  inline dealii::FEValuesExtractors::Vector
  get_vector_extractor(const SolverInfo::VariableType variable_type) const
  {
    Assert(
      is_vector(variable_type),
      dealii::StandardExceptions::ExcMessage(
        "Cannot return a FEValuesExtractors::Vector for non-vector variable"));
    switch (variable_type)
    {
      case SolverInfo::VariableType::velocity:
        return dealii::FEValuesExtractors::Vector(u_lower);
      case SolverInfo::VariableType::pressure:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::mesh_position:
        return dealii::FEValuesExtractors::Vector(x_lower);
      case SolverInfo::VariableType::lagrange_mult:
        return dealii::FEValuesExtractors::Vector(l_lower);
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::phase_tracer:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::phase_potential:
        DEAL_II_ASSERT_UNREACHABLE();
      case SolverInfo::VariableType::temperature:
        DEAL_II_ASSERT_UNREACHABLE();
    }
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



template <int dim, bool with_moving_mesh = false>
class ConstexprComponentOrderingCHNS
{
public:
  constexpr ConstexprComponentOrderingCHNS() = default;

  static constexpr unsigned int n_components =
    with_moving_mesh ? (2 * dim + 3) : (dim + 3);
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
};

template <int dim, bool with_moving_mesh>
class ComponentOrderingCHNS : public ComponentOrdering
{
public:
  ComponentOrderingCHNS()
  {
    using C      = ConstexprComponentOrderingCHNS<dim, with_moving_mesh>;
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
  }
};

/**
 * Components ordering for the elasticity solver.
 */
template <int dim>
class ComponentOrderingElasticity : public ComponentOrdering
{
public:
  ComponentOrderingElasticity()
    : ComponentOrdering()
  {
    n_components = dim;
    x_lower      = 0;
    x_upper      = dim;
  }
};

#endif
