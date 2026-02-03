#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_faces.h>

namespace PostProcessingTools
{
  using namespace dealii;

  template <int dim>
  class DataOutFacesOnBoundary : public DataOutFaces<dim>
  {
  public:
    DataOutFacesOnBoundary(const Triangulation<dim> &triangulation,
                           const types::boundary_id  boundary_id);


    using FaceDescriptor = typename DataOutFaces<dim>::FaceDescriptor;


    FaceDescriptor first_face() override;

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

  template <int dim>
  Vector<double>
  compute_slice_index_on_boundary(const DoFHandler<dim>   &dof_handler,
                                  const types::boundary_id boundary_id,
                                  const unsigned int       n_slices,
                                  const SliceAxis          axis,
                                  const MPI_Comm           mpi_comm);

} // namespace PostProcessingTools

#endif // POST_PROCESSING_TOOLS_H
