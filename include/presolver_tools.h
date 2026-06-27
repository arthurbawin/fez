#ifndef PRESOLVER_TOOLS_H
#define PRESOLVER_TOOLS_H

#include <elasticity_solver.h>
#include <parameter_reader.h>

#include <memory>

/**
 * Build and run the elasticity presolver that pre-positions the mesh for a
 * CHNS-ALE simulation. Returns nullptr when the presolver is disabled. The
 * returned solver owns the presolved mesh position, which the CHNS solver then
 * injects as its initial mesh.
 */
template <int dim>
std::unique_ptr<ElasticitySolver<dim>>
create_elasticity_presolver(const ParameterReader<dim> &param)
{
  if (!param.cahn_hilliard.use_presolver)
    return nullptr;

  auto presolver = std::make_unique<ElasticitySolver<dim>>(param);
  presolver->run();
  return presolver;
}

#endif
