#ifndef PRESOLVER_TOOLS_H
#define PRESOLVER_TOOLS_H

#include <linear_elasticity_solver.h>
#include <parameter_reader.h>

#include <memory>

template <int dim, bool with_enlarged = false>
std::unique_ptr<LinearElasticitySolver<dim>>
create_linear_elasticity_presolver(const ParameterReader<dim> &param)
{
  if (!param.linear_elasticity.use_as_presolver &&
      !param.linear_elasticity.chns_presolver_enable)
    return nullptr;

  using CacheMode =
    Parameters::LinearElasticity::PresolvedMeshPositionCache::Mode;

  using PresolvedCHNSFields =
    typename LinearElasticitySolver<dim>::PresolvedCHNSFields;
  constexpr PresolvedCHNSFields presolved_fields =
    with_enlarged ? PresolvedCHNSFields::phi_psi : PresolvedCHNSFields::phi;

  auto elasticity_solver =
    std::make_unique<LinearElasticitySolver<dim>>(param, presolved_fields);

  const auto mode = param.linear_elasticity.presolved_mesh_position_cache.mode;
  bool       loaded_from_cache = false;

  if (mode == CacheMode::automatic || mode == CacheMode::read_only)
    loaded_from_cache =
      elasticity_solver->try_load_presolved_mesh_position_cache();

  if (!loaded_from_cache)
    elasticity_solver->run();

  return elasticity_solver;
}

#endif
