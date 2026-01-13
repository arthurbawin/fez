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

template <int dim>
class PostProcessingHandler
{
public:
  using SliceAxis = PostProcessingTools::SliceAxis;

  PostProcessingHandler(
    const Parameters::PostProcessing &postprocessing_parameters,
    const Parameters::Output         &output_parameters)
    : pp_param(postprocessing_parameters)
  {}

  /**
   * Single entry point:
   * - stores dof_handler + communicator
   * - computes slice_index once
   */
  void setup_slices(const dealii::DoFHandler<dim> &dof_handler_,
                    const MPI_Comm                 mpi_communicator_)
  {
    dof_handler      = &dof_handler_;
    mpi_communicator = mpi_communicator_;
    is_initialized   = true;

    // If slices are disabled, do nothing (and don't mark computed)
    if (!pp_param.enable_slicing)
      return;

    AssertThrow(
      pp_param.slicing_boundary_id != dealii::numbers::invalid_unsigned_int,
      dealii::ExcMessage("setup_slices(): slicing_boundary_id is invalid."));

    const unsigned int n_slices = std::max(1u, pp_param.number_of_slices);

    AssertThrow(n_slices >= 1, ExcMessage("n_slices must be >= 1."));
    const std::string &dir = pp_param.slicing_direction;

    if constexpr (dim == 2)
      AssertThrow(
        dir == "x" || dir == "y",
        dealii::ExcMessage(
          "setup_slices(): slicing_direction must be 'x' or 'y' in 2D."));
    else
      AssertThrow(
        dir == "x" || dir == "y" || dir == "z",
        dealii::ExcMessage(
          "setup_slices(): slicing_direction must be 'x', 'y' or 'z' in 3D."));

    const SliceAxis axis = (dir == "x" ? SliceAxis::x :
                            dir == "y" ? SliceAxis::y :
                                         SliceAxis::z);

    slice_index = PostProcessingTools::compute_slice_index_on_boundary<dim>(
      *dof_handler,
      pp_param.slicing_boundary_id,
      n_slices,
      axis,
      mpi_communicator);

    slices_computed = true;
  }

  const dealii::Vector<double> &get_slice_index() const
  {
    AssertThrow(
      is_initialized,
      dealii::ExcMessage(
        "get_slice_index(): handler not initialized (call setup_slices())."));

    // Si slices désactivées: au choix.
    // Ici je force une erreur pour éviter des sorties incohérentes.
    AssertThrow(pp_param.enable_slicing,
                dealii::ExcMessage(
                  "get_slice_index(): slices disabled (enable_slices=false)."));

    AssertThrow(slices_computed,
                dealii::ExcMessage("get_slice_index(): slices not computed "
                                   "(setup_slices did not compute)."));

    return slice_index;
  }

private:
  const Parameters::PostProcessing &pp_param;

  const dealii::DoFHandler<dim> *dof_handler      = nullptr;
  MPI_Comm                       mpi_communicator = MPI_COMM_NULL;

  bool is_initialized  = false;
  bool slices_computed = false;

  dealii::Vector<double> slice_index;
};

#endif
