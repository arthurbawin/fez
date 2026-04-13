
#include <deal.II/dofs/dof_handler.h>
#include <mesh_and_dof_tools.h>

template <int dim>
void get_owned_mesh_vertices(const Triangulation<dim> &triangulation,
                             const types::subdomain_id subdomain_id,
                             std::vector<bool>        &owned_vertices)
{
  owned_vertices.resize(triangulation.n_vertices(), false);

  // Start by marking all cells touching an owned cell as owned
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const unsigned int v : cell->vertex_indices())
        owned_vertices[cell->vertex_index(v)] = true;

  // If a ghost cell with lesser id touches a vertex, mark it non-owned
  for (const auto &cell : triangulation.active_cell_iterators())
    if (cell->is_artificial() ||
        (cell->is_ghost() && cell->subdomain_id() < subdomain_id))
      for (const unsigned int v : cell->vertex_indices())
        owned_vertices[cell->vertex_index(v)] = false;
}

template void get_owned_mesh_vertices(const Triangulation<2> &,
                                      const types::subdomain_id,
                                      std::vector<bool> &);
template void get_owned_mesh_vertices(const Triangulation<3> &,
                                      const types::subdomain_id,
                                      std::vector<bool> &);

template <int dim>
std::set<Point<dim>, PointComparator<dim>>
get_mesh_vertices_on_boundary(const DoFHandler<dim>   &dof_handler,
                              const types::boundary_id boundary_id)
{
  const MPI_Comm comm = dof_handler.get_mpi_communicator();
  std::set<Point<dim>, PointComparator<dim>> vertices_on_boundary;
  std::vector<Point<dim>>                    local_vertices_on_boundary;

  // Mark based on faces
  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == boundary_id)
        for (unsigned int i = 0; i < face->n_vertices(); ++i)
          local_vertices_on_boundary.push_back(face->vertex(i));

  std::vector<std::vector<Point<dim>>> gathered =
    Utilities::MPI::all_gather(comm, local_vertices_on_boundary);

  for (const auto &vec : gathered)
    for (const auto pt : vec)
      vertices_on_boundary.insert(pt);

  return vertices_on_boundary;
}

template std::set<Point<2>, PointComparator<2>>
get_mesh_vertices_on_boundary(const DoFHandler<2> &, const types::boundary_id);
template std::set<Point<3>, PointComparator<3>>
get_mesh_vertices_on_boundary(const DoFHandler<3> &, const types::boundary_id);
