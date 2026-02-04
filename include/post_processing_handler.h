#ifndef POST_PROCESSING_HANDLER_H
#define POST_PROCESSING_HANDLER_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <parameters.h>
#include <post_processing_tools.h>

#include <algorithm>
#include <string>

using namespace dealii;

template <int dim>
class PostProcessingHandler
{
public:
  using SliceAxis = PostProcessingTools::SliceAxis;

  PostProcessingHandler(
    const Parameters::PostProcessing &postprocessing_parameters,
    const Parameters::Output         &output_parameters)
    : post_proc_param(postprocessing_parameters)
  {}

  void setup_slices(const DoFHandler<dim> &dof_handler)
  {
    // If slices are disabled, do nothing
    if (!post_proc_param.enable_slicing)
      return;

    AssertThrow(
      post_proc_param.slicing_boundary_id != numbers::invalid_unsigned_int,
      ExcMessage("slicing_boundary_id is invalid."));

    const unsigned int n_slices =
      std::max(1u, post_proc_param.number_of_slices);

    AssertThrow(n_slices >= 1,
                ExcMessage("n_slices must be >= 1."));

    const std::string &dir = post_proc_param.slicing_direction;

    if constexpr (dim == 2)
      AssertThrow(
        dir == "x" || dir == "y",
        ExcMessage(
          "setup_slices(): slicing_direction must be 'x' or 'y' in 2D."));
    else
      AssertThrow(
        dir == "x" || dir == "y" || dir == "z",
        ExcMessage(
          "setup_slices(): slicing_direction must be 'x', 'y' or 'z' in 3D."));

    const SliceAxis axis =
      (dir == "x" ? SliceAxis::x :
      dir == "y" ? SliceAxis::y :
                    SliceAxis::z);

    const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();

    slice_index =
      PostProcessingTools::compute_slice_index_on_boundary<dim>(
        dof_handler,
        post_proc_param.slicing_boundary_id,
        n_slices,
        axis,
        mpi_communicator);
  }


  const Vector<double> &get_slice_index() const
  {
    if (post_proc_param.enable_slicing)
    {
      AssertThrow(!slice_index.empty(),
                  ExcMessage(
                    "get_slice_index(): slice_index is empty "
                    "(setup_slices() was not called or slicing is disabled)."));
    }

    return slice_index;
  }

private:
  const Parameters::PostProcessing &post_proc_param;
  Vector<double> slice_index;
};

#endif
