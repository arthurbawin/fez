#ifndef PATCHES_H
#define PATCHES_H

#include <deal.II/base/types.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * Element and vertex patches around each mesh vertex, to compute a more
   * accurate solution by least-square projection.
   *
   * Each processor stores the patches associated to the mesh vertices of
   * its partition of the mesh. The vertices have a global numbering
   * on each partition, but (i) it does not seem to be consistent across
   * partitions (that is, one vertex stored on multiple partitions does not
   * have the same index), and (ii) the index does not necessarily start at 0.
   */
  template <int dim>
  class Patches
  {
  public:
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    Patches(const parallel::DistributedTriangulationBase<dim> &triangulation,
            const Mapping<dim>                                &mapping,
            const DoFHandler<dim>                             &dof_handler,
            const unsigned int                                 degree,
            const ComponentMask                               &mask);

    void write_element_patch_gmsh(
      const types::global_vertex_index vertex_index,
      const unsigned int layer) const;
    void write_support_points_patch(const unsigned int layer);

    void print_owned_vertices() const;
    void print_ghost_dofs() const;
    void print_ghost_cells() const;

  private:
    /**
     * Add a layer of elements (and their vertices) to the patch of the given
     * mesh vertex.
     */
    void add_element_layer(types::global_vertex_index vertex_index);
    void add_element_nth_layer(const unsigned int layer,
                               const unsigned     n_required_vertices,
                               bool              &needs_further_expansion);
    void exchange_ghost_layer_dofs(
      const std::map<types::subdomain_id, std::set<types::global_dof_index>>
        &dofs_to_request,
      std::map<types::global_dof_index, std::vector<Point<dim>>>
        &connected_support_points_to_requested_dofs);

    /**
     * Compute the scaling of each patch : max_i |x_i - x|, with x the vertex
     * (center of the patch) and x_i a vertex of the patch.
     */
    void compute_scalings();

  protected:
    const parallel::DistributedTriangulationBase<dim> &triangulation;
    const DoFHandler<dim>                             &dof_handler;
    const FESystem<dim>                               &fe;

    MPI_Comm mpi_communicator;

    const ComponentMask mask;

    const unsigned int my_rank;
    const unsigned int size;
    // The id of this mesh partition (= the MPI rank)
    const types::subdomain_id subdomain_id;

    // Total number of mesh vertices
    unsigned int n_vertices;

    /**
     * Mask for owned mesh vertices. A mesh vertex is owned if it is inside the
     * partition, or, at boundaries, if it belongs to the subdomain (MPI proc)
     * with lowest id.
     *
     * FIXME: This does not seem to be the case though? Boundary vertices are
     * owned by multiple partitions...
     */
    std::vector<bool> owned_vertices;

    // Number of owned mesh vertices on this process
    unsigned int n_owned_vertices;

    /**
     * Patches for the Zhang & Naga recovery are vertex-centered, and consist
     * of the dof support points of the surrounding layers of elements.
     *
     * Of course, this is only valid for nodal finite elements, for which the
     * notion of dof support point makes sense.
     *
     * Instead of storing the support points, the global dof index is stored.
     */
    std::vector<std::set<CellIterator>>            patches_of_elements;
    std::vector<std::set<types::global_dof_index>> patches_of_dofs;
    std::vector<std::vector<Point<dim>>>           patches_of_support_points;

    std::set<types::subdomain_id> neighbouring_subdomains;

    // owning subdomain : dof to request
    std::map<types::subdomain_id, std::set<types::global_dof_index>>
      dofs_in_ghost_layer;
    // For each requested dof, the owned vertices containing this dof
    // and for which queried support points must be added
    std::map<types::global_dof_index, std::set<types::global_vertex_index>>
      to_add_after_request;

    std::vector<unsigned int> n_layers;
    std::vector<Point<dim>>   scalings;

    std::map<types::global_dof_index, Point<dim>> relevant_dofs_support_points;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    IndexSet ghost_dofs;
  };
} // namespace ErrorEstimation

#endif