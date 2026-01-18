#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <post_processing_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out_faces.h>


#include <algorithm>
#include <cmath>
#include <limits>

namespace PostProcessingTools
{
  using namespace dealii;

  template <int dim>
  DataOutFacesOnBoundary<dim>::DataOutFacesOnBoundary(
    const Triangulation<dim> &triangulation_,
    const types::boundary_id  boundary_id_)
    : DataOutFaces<dim>(/*surface_only=*/true)
    , triangulation(triangulation_)
    , boundary_id(boundary_id_)
  {}

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::first_face()
  {
    for (const auto &cell : this->triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const unsigned int f : cell->face_indices())
        {
          const auto &face = cell->face(f);
          if (face->at_boundary() && face->boundary_id() == this->boundary_id)
            return FaceDescriptor(cell, f);
        }

    // Peut arriver en parallèle si toutes les faces de cette boundary_id
    // sont possédées par d'autres ranks.
    return FaceDescriptor();
  }

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::next_face(const FaceDescriptor &old_face)
  {
    FaceDescriptor face = old_face;

    if (face.first == this->triangulation.end())
      return face;

    Assert(face.first->is_locally_owned(), ExcInternalError());

    // 1) faces suivantes sur la même cellule
    for (unsigned int f = face.second + 1; f < face.first->n_faces(); ++f)
    {
      const auto &face_iter = face.first->face(f);
      if (face_iter->at_boundary() &&
          face_iter->boundary_id() == this->boundary_id)
      {
        face.second = f;
        return face;
      }
    }

    // 2) cellules suivantes
    typename Triangulation<dim>::active_cell_iterator active_cell = face.first;
    ++active_cell;

    while (active_cell != this->triangulation.end())
    {
      if (active_cell->is_locally_owned())
        for (const unsigned int f : active_cell->face_indices())
          if (active_cell->face(f)->at_boundary() &&
              active_cell->face(f)->boundary_id() == this->boundary_id)
          {
            face.first  = active_cell;
            face.second = f;
            return face;
          }
      ++active_cell;
    }

    face.first  = this->triangulation.end();
    face.second = 0;
    return face;
  }

  template <int dim>
  Vector<double>
  compute_slice_index_on_boundary(const DoFHandler<dim>   &dof_handler,
                                  const types::boundary_id boundary_id,
                                  const unsigned int       n_slices,
                                  const SliceAxis          axis,
                                  const MPI_Comm           mpi_comm)
  {
    const unsigned int axis_id = static_cast<unsigned int>(axis);

    const auto &tria = dof_handler.get_triangulation();

    Vector<double> slice_index(tria.n_active_cells());
    for (unsigned int i = 0; i < slice_index.size(); ++i)
      slice_index(i) = numbers::invalid_unsigned_int;

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
          coord_min_local    = std::min(coord_min_local, coord);
          coord_max_local    = std::max(coord_max_local, coord);
        }
      }
    }

    const double coord_min = Utilities::MPI::min(coord_min_local, mpi_comm);
    const double coord_max = Utilities::MPI::max(coord_max_local, mpi_comm);

    AssertThrow(coord_max > coord_min,
                ExcMessage(
                  "Could not determine coordinate range on boundary."));

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

        double             coord_sum = 0.0;
        const unsigned int nv        = face->n_vertices();

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
        k     = std::max(0, std::min(static_cast<int>(n_slices) - 1, k));

        slice_index[cell->active_cell_index()] = static_cast<float>(k);
      }
    }

    return slice_index;
  }


  // ---------------- explicit instantiation ----------------
  template Vector<double>
  compute_slice_index_on_boundary<2>(const DoFHandler<2> &,
                                     const types::boundary_id,
                                     const unsigned int,
                                     const SliceAxis,
                                     const MPI_Comm);

  template Vector<double>
  compute_slice_index_on_boundary<3>(const DoFHandler<3> &,
                                     const types::boundary_id,
                                     const unsigned int,
                                     const SliceAxis,
                                     const MPI_Comm);

} // namespace PostProcessingTools

// --------- explicit instantiation for DataOutFacesOnBoundary ---------
template class PostProcessingTools::DataOutFacesOnBoundary<2>;
template class PostProcessingTools::DataOutFacesOnBoundary<3>;
