#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
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


    FaceDescriptor first_face() override;

    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     */
    FaceDescriptor next_face(const FaceDescriptor &face) override;

  protected:
    const Triangulation<dim> &triangulation;
    const types::boundary_id  boundary_id;
  };

  enum class SliceAxis : unsigned int
  {
    x = 0,
    y = 1,
    z = 2
  };


  /**
   * Compute slice indices for degrees of freedom located on a given boundary.
   *
   *  The boundary @p boundary_id is partitioned into @p n_slices along the
   * coordinate direction @p axis. Each degree of freedom on this boundary
   * is assigned to one slice based on its geometric position.
   *
   * The function returns a Vector<double> containing the slice index
   * associated with each degree of freedom. A floating-point vector is used
   * for compatibility with deal.II data structures and post-processing.
   *
   * @return Vector of slice indices for the specified boundary.
   */
  template <int dim>
  Vector<double>
  compute_slice_index_on_boundary(const DoFHandler<dim>   &dof_handler,
                                  const types::boundary_id boundary_id,
                                  const unsigned int       n_slices,
                                  const SliceAxis          axis,
                                  const MPI_Comm           mpi_comm);

} // namespace PostProcessingTools

#endif
