#include <post_processing_tools.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>

#include <limits>
#include <cmath>
#include <algorithm>

namespace PostProcessingTools
{
  template <int dim>
  BoundaryDataOutFaces<dim>::BoundaryDataOutFaces(
    const dealii::DoFHandler<dim> &       dof_handler_,
    const dealii::types::boundary_id      boundary_id_,
    const bool                            surface_only)
    : Base(surface_only)
    , dof_handler(&dof_handler_)
    , boundary_id(boundary_id_)
  {}

  template <int dim>
  typename BoundaryDataOutFaces<dim>::FaceDescriptor
  BoundaryDataOutFaces<dim>::first_face()
  {
    for (auto cell = dof_handler->begin_active(); cell != dof_handler->end(); ++cell)
      {
        if (!cell->is_locally_owned())
          continue;

        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == boundary_id)
            return FaceDescriptor(cell, f);
      }

    return FaceDescriptor(dof_handler->end(), 0U);
  }

  template <int dim>
  typename BoundaryDataOutFaces<dim>::FaceDescriptor
  BoundaryDataOutFaces<dim>::next_face(const FaceDescriptor &face)
  {
    cell_iterator cell = face.first;
    unsigned int  f    = face.second;

    if (cell == dof_handler->end())
      return face;

    // 1) faces suivantes sur la même cellule
    for (unsigned int ff = f + 1; ff < cell->n_faces(); ++ff)
      if (cell->face(ff)->at_boundary() &&
          cell->face(ff)->boundary_id() == boundary_id)
        return FaceDescriptor(cell, ff);

    // 2) cellules suivantes
    for (++cell; cell != dof_handler->end(); ++cell)
      {
        if (!cell->is_locally_owned())
          continue;

        for (const auto ff : cell->face_indices())
          if (cell->face(ff)->at_boundary() &&
              cell->face(ff)->boundary_id() == boundary_id)
            return FaceDescriptor(cell, ff);
      }

    return FaceDescriptor(dof_handler->end(), 0U);
  }

  template <int dim>
  dealii::Vector<float>
  compute_slice_index_on_boundary(
    const dealii::DoFHandler<dim> &      dof_handler,
    const dealii::types::boundary_id     boundary_id,
    const unsigned int                   n_slices,
    const SliceAxis                      axis,
    const MPI_Comm                       mpi_comm)
  {
    AssertThrow(n_slices >= 1, dealii::ExcMessage("n_slices must be >= 1."));
    const unsigned int axis_id = static_cast<unsigned int>(axis);
    AssertThrow(axis_id < dim, dealii::ExcMessage("Slice axis invalid for this dim."));

    const auto &tria = dof_handler.get_triangulation();

    dealii::Vector<float> slice_index(tria.n_active_cells());
    for (unsigned int i = 0; i < slice_index.size(); ++i)
      slice_index(i) = -1.0f;

    if constexpr (dim < 2)
      return slice_index;

    // ---------------------------
    // 1) min/max sur la frontière (via SOMMETS)
    // ---------------------------
    double coord_min_local = std::numeric_limits<double>::max();
    double coord_max_local = -std::numeric_limits<double>::max();

    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto f : cell->face_indices())
      {
        const auto &face = cell->face(f);
        if (!face->at_boundary() || face->boundary_id() != boundary_id)
          continue;

        for (unsigned int v = 0; v < face->n_vertices(); ++v)
        {
          const double coord = face->vertex(v)[axis_id];
          coord_min_local = std::min(coord_min_local, coord);
          coord_max_local = std::max(coord_max_local, coord);
        }
      }
    }

    const double coord_min = dealii::Utilities::MPI::min(coord_min_local, mpi_comm);
    const double coord_max = dealii::Utilities::MPI::max(coord_max_local, mpi_comm);

    AssertThrow(coord_max > coord_min,
                dealii::ExcMessage("Could not determine coordinate range on boundary."));

    const double inv_range = 1.0 / (coord_max - coord_min);

    // petite tolérance pour gérer les cas "pile sur un plan"
    const double eps = 1e-12;

    // ---------------------------
    // 2) affectation par face : centroid (moyenne sommets)
    // ---------------------------
    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto f : cell->face_indices())
      {
        const auto &face = cell->face(f);
        if (!face->at_boundary() || face->boundary_id() != boundary_id)
          continue;

        double coord_sum = 0.0;
        const unsigned int nv = face->n_vertices();

        for (unsigned int v = 0; v < nv; ++v)
          coord_sum += face->vertex(v)[axis_id];

        const double coord_face = coord_sum / static_cast<double>(nv);

        double s = (coord_face - coord_min) * inv_range;
        // clamp + epsilon
        s = std::max(0.0, std::min(1.0, s));

        // IMPORTANT: éviter le cas s=1 -> floor(n_slices) :
        double t = s * static_cast<double>(n_slices);
        if (t > 0.0)
          t -= eps;

        int k = static_cast<int>(std::floor(t));
        k = std::max(0, std::min(static_cast<int>(n_slices) - 1, k));

        slice_index[cell->active_cell_index()] = static_cast<float>(k);
      }
    }

    return slice_index;
  }


  // ---------------- explicit instantiation ----------------
  template dealii::Vector<float>
  compute_slice_index_on_boundary<2>(const dealii::DoFHandler<2> &,
                                     const dealii::types::boundary_id,
                                     const unsigned int,
                                     const SliceAxis,
                                     const MPI_Comm);

  template dealii::Vector<float>
  compute_slice_index_on_boundary<3>(const dealii::DoFHandler<3> &,
                                     const dealii::types::boundary_id,
                                     const unsigned int,
                                     const SliceAxis,
                                     const MPI_Comm);

} // namespace PostProcessingTools

// --------- explicit instantiation for BoundaryDataOutFaces ---------
template class PostProcessingTools::BoundaryDataOutFaces<2>;
template class PostProcessingTools::BoundaryDataOutFaces<3>;
