#ifndef MESH_ADAPTATION_TOOLS_H
#define MESH_ADAPTATION_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <metric_field.h>
#include <parameter_reader.h>

#include <algorithm>
#include <cmath>

using namespace dealii;

namespace MeshTools
{
  template <int dim>
  double mapped_cell_diameter(
    const Mapping<dim>                                   &mapping,
    const typename DoFHandler<dim>::active_cell_iterator &cell)
  {
    const auto vertices = mapping.get_vertices(cell);
    double     diameter = 0.;
    for (unsigned int i = 0; i < vertices.size(); ++i)
      for (unsigned int j = i + 1; j < vertices.size(); ++j)
        diameter = std::max(diameter, vertices[i].distance(vertices[j]));
    return diameter;
  }

  /**
   * Select independently for each nonzero field, then combine refinement by
   * union and coarsening by intersection. Apply the cell budget to the union.
   */
  template <int dim>
  void mark_multifield_adaptation(
    parallel::DistributedTriangulationBase<dim> &tria,
    const std::vector<Vector<float>>            &indicators,
    const Parameters::Mesh::Adaptation::TreeAMR &param);

  /**
   * Mark a physical band around the sampled zero contour of a scalar field.
   * Criteria are +1 (refine), 0 (keep), -1 (coarsen). Cell sizes use the
   * mapped mesh for ALE and the reference mesh otherwise. The contour and cell
   * geometry are piecewise linear approximations.
   */
  template <bool with_moving_mesh>
  void compute_interface_band_criterion(const DoFHandler<2>     &dof_handler,
                                        const Mapping<2>        &mapping,
                                        const LA::ParVectorType &solution,
                                        const unsigned int       component,
                                        const double             half_width,
                                        const double   target_diameter,
                                        Vector<float> &criteria);

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
