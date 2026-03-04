#ifndef MESH_ADAPTATION_TOOLS_H
#define MESH_ADAPTATION_TOOLS_H

#include <deal.II/grid/tria.h>
#include <parameter_reader.h>

using namespace dealii;

/**
 *
 */
template <int dim>
void adapt_mesh_mmg(const ParameterReader<dim> &param,
                    const Triangulation<dim>   &triangulation);

#endif
