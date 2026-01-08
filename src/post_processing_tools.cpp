
#include <post_processing_tools.h>

namespace PostProcessingTools
{
  template <int dim>
  DataOutFacesOnBoundary<dim>::DataOutFacesOnBoundary(
    const Triangulation<dim> &triangulation,
    const types::boundary_id  boundary_id)
    : DataOutFaces<dim>(/*surface_only =*/true)
    , triangulation(triangulation)
    , boundary_id(boundary_id)
  {}

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::first_face()
  {
    // Find first active cell with a face on the prescribed boundary
    for (const auto &cell : this->triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const unsigned int f : cell->face_indices())
        {
          const auto &face = cell->face(f);
          if (face->at_boundary() && face->boundary_id() == this->boundary_id)
            return FaceDescriptor(cell, f);
        }

    // just return an invalid descriptor if we haven't found a locally
    // owned face. this can happen in parallel where all boundary
    // faces are owned by other processors
    return FaceDescriptor();
  }

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::next_face(const FaceDescriptor &old_face)
  {
    FaceDescriptor face = old_face;

    // first check whether the present cell has more faces on the boundary.
    // since we started with this face, its cell must clearly be locally owned
    Assert(face.first->is_locally_owned(), ExcInternalError());
    for (unsigned int f = face.second + 1; f < face.first->n_faces(); ++f)
    {
      const auto &face_iter = face.first->face(f);
      if (face_iter->at_boundary() &&
          face_iter->boundary_id() == this->boundary_id)
      {
        // Return this face
        face.second = f;
        return face;
      }
    }

    // otherwise find the next active cell that has a face on the boundary
    typename Triangulation<dim>::active_cell_iterator active_cell = face.first;
    ++active_cell;

    while (active_cell != this->triangulation.end())
    {
      if (active_cell->is_locally_owned())
        for (const unsigned int f : face.first->face_indices())
          if (active_cell->face(f)->at_boundary() &&
              active_cell->face(f)->boundary_id() == this->boundary_id)
          {
            face.first  = active_cell;
            face.second = f;
            return face;
          }
      ++active_cell;
    }

    // If no face was found, return with invalid pointer
    face.first  = this->triangulation.end();
    face.second = 0;
    return face;
  }

  template class DataOutFacesOnBoundary<2>;
  template class DataOutFacesOnBoundary<3>;
} // namespace PostProcessingTools