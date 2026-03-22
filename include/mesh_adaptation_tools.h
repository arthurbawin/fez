#ifndef MESH_ADAPTATION_TOOLS_H
#define MESH_ADAPTATION_TOOLS_H

#include <deal.II/grid/tria.h>
#include <metric_field.h>
#include <parameter_reader.h>

using namespace dealii;

namespace MeshAdaptation
{
  /**
   * Adapt the mesh with the MMG library.
   */
  template <int dim>
  void adapt_with_mmg(const ParameterReader<dim> &param,
                      const Triangulation<dim>   &triangulation,
                      const MetricField<dim>     &metric_field);
} // namespace MeshAdaptation

#endif
