#ifndef MESH_ADAPTATION_TOOLS_H
#define MESH_ADAPTATION_TOOLS_H

#include <deal.II/grid/tria.h>
#include <metric_field.h>
#include <parameter_reader.h>

using namespace dealii;

namespace MeshTools
{
  /**
   * Adapt the mesh given by @p input_meshfile with the MMG library, using the
   * Riemannian metric stored in @p metric_field. The adapted mesh will be
   * written as @p output_meshfile in @p adapt_directory, together with
   * auxiliary files.
   */
  template <int dim>
  void adapt_with_mmg(const ParameterReader<dim> &param,
                      const MetricField<dim>     &metric_field,
                      const std::string          &adapt_directory,
                      const std::string          &input_meshfile,
                      const std::string          &output_meshfile,
                      const unsigned int          interval_index = 0);
} // namespace MeshTools

#endif
