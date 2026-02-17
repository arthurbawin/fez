#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/types.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/data_out_faces.h>

/**
 * Collection of utilities for post-processing.
 */
namespace PostProcessingTools
{
  using namespace dealii;

  /**
   * Small specialization of deal.II's DataOutFaces to output a single boundary.
   */
  template <int dim>
  class DataOutFacesOnBoundary : public DataOutFaces<dim>
  {
  public:
    /**
     * Constructor
     */
    DataOutFacesOnBoundary(const Triangulation<dim> &triangulation,
                           const types::boundary_id  boundary_id);

    /**
     * A FaceDescriptor is a pair [cell : number of the face].
     */
    using FaceDescriptor = typename DataOutFaces<dim>::FaceDescriptor;

    /**
     * Return the first face which we want output for.
     */
    virtual FaceDescriptor first_face() override;

    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     */
    virtual FaceDescriptor next_face(const FaceDescriptor &face) override;

  protected:
    const Triangulation<dim> &triangulation;
    const types::boundary_id  boundary_id;
  };
} // namespace PostProcessingTools

#endif
