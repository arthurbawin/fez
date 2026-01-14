#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <optional>
#include <vector>

#include <fstream>
#include <limits>
#include <string>

using namespace dealii;
namespace PostProcessingTools
{
  template <int dim>
  class BoundaryDataOutFaces : public DataOutFaces<dim>
  {
  public:
    using Base           = DataOutFaces<dim>;
    using FaceDescriptor = typename Base::FaceDescriptor;
    using cell_iterator  = typename Base::cell_iterator;

    BoundaryDataOutFaces(const DoFHandler<dim>   &dof_handler,
                         const types::boundary_id boundary_id,
                         const bool                       surface_only = true);

    FaceDescriptor first_face() override;
    FaceDescriptor next_face(const FaceDescriptor &face) override;

  private:
    const DoFHandler<dim>   *dof_handler;
    const types::boundary_id boundary_id;
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
                                  const unsigned int               n_slices,
                                  const SliceAxis                  axis,
                                  const MPI_Comm                   mpi_comm);
                                  
} // namespace PostProcessingTools

#endif // POST_PROCESSING_TOOLS_H
