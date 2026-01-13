#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_faces.h>

#include <limits>

namespace PostProcessingTools
{
  template <int dim>
  class BoundaryDataOutFaces : public dealii::DataOutFaces<dim>
  {
  public:
    using Base           = dealii::DataOutFaces<dim>;
    using FaceDescriptor = typename Base::FaceDescriptor;
    using cell_iterator  = typename Base::cell_iterator;

    BoundaryDataOutFaces(const dealii::DoFHandler<dim>   &dof_handler,
                         const dealii::types::boundary_id boundary_id,
                         const bool                       surface_only = true);

    FaceDescriptor first_face() override;
    FaceDescriptor next_face(const FaceDescriptor &face) override;

  private:
    const dealii::DoFHandler<dim>   *dof_handler;
    const dealii::types::boundary_id boundary_id;
  };

  enum class SliceAxis : unsigned int
  {
    x = 0,
    y = 1,
    z = 2
  };


  template <int dim>
  dealii::Vector<double>
  compute_slice_index_on_boundary(const dealii::DoFHandler<dim>   &dof_handler,
                                  const dealii::types::boundary_id boundary_id,
                                  const unsigned int               n_slices,
                                  const SliceAxis                  axis,
                                  const MPI_Comm                   mpi_comm);


} // namespace PostProcessingTools

#endif // POST_PROCESSING_TOOLS_H
