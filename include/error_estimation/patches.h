#ifndef PATCHES_H
#define PATCHES_H

#include <deal.II/base/types.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * Patches of elements and dof support points around mesh vertices for the
   * computation of a more accurate solution by least-square projection,
   * following the Polynomial-Preserving Recovery (PPR) of Zhang & Naga.
   *
   * The least-square recovery uses nearby information from the numerical
   * solution to fit a polynomial of order p + 1. Thus, the information stored
   * in the patches are the \emph{support points} of the degrees of freedom of
   * the recovered field (i.e., not mesh vertices). Of course, this only makes
   * sense for nodel finite elements, for which support points are well defined.
   *
   * These recoveries and their derivatives are used as anisotropic error
   * estimates to compute metric tensors fields for mesh adaptation. Since
   * we are using MMG for mesh adaptation, the recoveries are performed for each
   * mesh vertex, as MMG takes a vertex-based file of metric tensors.
   *
   * Patches are constructed by adding layers of mesh elements, then by adding
   * the support points of each dof of the given field lying on these elements.
   */
  template <int dim>
  class Patches
  {
  public:
    /**
     * Constructor
     */
    Patches(const parallel::DistributedTriangulationBase<dim> &triangulation,
            const Mapping<dim>                                &mapping,
            const DoFHandler<dim>                             &dof_handler,
            const unsigned int                                 degree,
            const ComponentMask                               &mask);

    void write_element_patch_gmsh(const types::global_vertex_index vertex_index,
                                  const unsigned int               layer) const;
    void write_support_points_patch(std::ostream &out = std::cout);

  private:
    /**
     * Add a layer of elements (and their support points for the given mask) to
     * the patch of the given mesh vertex.
     */
    void add_element_layer(const unsigned int layer,
                           const unsigned     n_required_vertices,
                           bool              &needs_further_expansion);

    /**
     * When adding a layer of elements to a patch, cells touching the already
     * stored support points are added. If these support points are in the ghost
     * layer, the connected cells may not exist on the partition, and we need
     * to ask the other ranks for the list of support points in these cells.
     *
     * This is done in this function, by first exchanging the list of dofs
     * for which ranks request the neighbours, the creating and sending back
     * the list of neighbouring support points.
     */
    void exchange_ghost_layer_dofs(
      const std::map<types::subdomain_id, std::set<types::global_dof_index>>
        &dofs_to_request,
      std::map<types::global_dof_index, std::vector<Point<dim>>>
        &connected_support_points_to_requested_dofs);

    /**
     * Compute the scaling s of each patch, defined as s_i = max_i |x_i - x|,
     * with x the vertex (center of the patch) and x_i a support points of the
     * patch.
     *
     * On anisotropic meshes, the least-square projection is computed on scaled
     * patches for better stability and accuracy.
     */
    void compute_scalings();

  protected:
    const parallel::DistributedTriangulationBase<dim> &triangulation;
    const DoFHandler<dim>                             &dof_handler;
    const ComponentMask                                mask;

    MPI_Comm mpi_communicator;

    // The id of this mesh partition (= the MPI rank)
    const types::subdomain_id subdomain_id;

    // Number of mesh vertices on this partition
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
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    std::vector<std::set<CellIterator>>            patches_of_elements;
    std::vector<std::set<types::global_dof_index>> patches_of_dofs;
    std::vector<std::vector<Point<dim>>>           patches_of_support_points;

    // owning subdomain : dof to request
    std::map<types::subdomain_id, std::set<types::global_dof_index>>
      dofs_in_ghost_layer;
    // For each requested dof, the owned vertices containing this dof
    // and for which queried support points must be added
    std::map<types::global_dof_index, std::set<types::global_vertex_index>>
      to_add_after_request;

    std::vector<Point<dim>> scalings;

    std::map<types::global_dof_index, Point<dim>> relevant_dofs_support_points;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    IndexSet ghost_dofs;
  };
} // namespace ErrorEstimation

#endif