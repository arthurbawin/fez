#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out_faces.h>
#include <post_processing_tools.h>

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

} // namespace PostProcessingTools

// --------- explicit instantiation for DataOutFacesOnBoundary ---------
template class PostProcessingTools::DataOutFacesOnBoundary<2>;
template class PostProcessingTools::DataOutFacesOnBoundary<3>;
