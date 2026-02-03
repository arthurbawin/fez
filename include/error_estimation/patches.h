#ifndef PATCHES_H
#define PATCHES_H

#include <deal.II/base/types.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <types.h>

namespace ErrorEstimation
{
  using namespace dealii;

  // Forward declaration
  template <int dim>
  class PatchHandler;

  /**
   * This struct represents a "patch" of data surrounding a mesh vertex. These
   * patches are used to fit a locally more accurate polynomial representing the
   * FE solution around the vertex, using the Polynomial-Preserving Recovery
   * (PPR) of Zhang & Naga [ref]. Each patch is centered at an owned mesh
   * vertex, and consists of the dof support points of the surrounding layers of
   * elements. The solution at these dofs is used to fit a polynomial of degree
   * p + 1 with least squares.
   *
   * Of course, this definition of a patch in terms of dof support points only
   * makes sense for nodal finite elements.
   *
   * The "neighbours"
   * map stores pairs [dof index : support points], and contains both local
   * (relevant) and non-local dof indices. Non-local dof indices must be added
   * to the index set of relevant dofs to gather the solution value at these
   * dofs.
   */
  template <int dim>
  struct Patch
  {
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;
    using DofData = typename std::pair<types::global_dof_index, Point<dim>>;

    Point<dim> center;
    Point<dim> scaling;

    std::vector<DofData> neighbours;
    std::vector<DofData> neighbours_local_coordinates;

    // Boost serialization for MPI communications through deal.II wrappers
    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar & center & scaling & neighbours & neighbours_local_coordinates;
    }

  private:
    /**
     * Set of local (owned or ghost) cells around the vertex, and map of
     * neighbouring dofs (local and non-local). These are used only during patch
     * creation.
     */
    std::set<CellIterator> elements;

    std::unordered_map<types::global_dof_index, Point<dim>> neighbours_map;

    friend class PatchHandler<dim>;
  };

  /**
   * This class manages the creation of patches, for polynomial fitting with
   * the Polynomial-Preserving Recovery (PPR) of Zhang & Naga [ref].
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
   *
   * The patches for, say, a P1 and P2 fields typically differ (the P1 patches
   * require more elements), so for now each set of patches is created for a
   * single component mask.
   */
  template <int dim>
  class PatchHandler
  {
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

  public:
    /**
     * Constructor
     */
    PatchHandler(
      const parallel::DistributedTriangulationBase<dim> &triangulation,
      const Mapping<dim>                                &mapping,
      const DoFHandler<dim>                             &dof_handler,
      const unsigned int                                 degree,
      const ComponentMask                               &mask);

    void write_element_patch_gmsh(const types::global_vertex_index vertex_index,
                                  const unsigned int               layer) const;
    void write_support_points_patch(const LA::ParVectorType &solution,
                                    std::ostream &out = std::cout) const;

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
      std::map<types::global_dof_index,
               std::vector<std::pair<types::global_dof_index, Point<dim>>>>
        &connected_dofs_to_requested_dofs);

    /**
     * Compute the scaling s of each patch, defined as s_i = max_i |x_i - x|,
     * with x the vertex (center of the patch) and x_i a support points of the
     * patch.
     *
     * On anisotropic meshes, the least-square projection is computed on scaled
     * patches for better stability and accuracy.
     */
    void compute_scalings_and_local_coordinates();

  public:
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

    // The patches associated to the owned mesh vertices on this partition
    std::vector<Patch<dim>> patches;

    /**
     * Map to request neighbouring dofs to other ranks.
     * For each neighbouring rank, a set of dof indices for which the neighbours
     * are requested. These dofs may be ghosted on the specified rank, but
     * they'll still know the neighbouring dofs to these dofs.
     */
    std::map<types::subdomain_id, std::set<types::global_dof_index>>
      dofs_in_ghost_layer;

    /**
     * For each requested dof in the ghost layer, the owned vertices containing
     * this dof and for which queried support points must be added once known.
     */
    std::map<types::global_dof_index, std::set<types::global_vertex_index>>
      to_add_after_request;

    std::map<types::global_dof_index, Point<dim>> relevant_dofs_support_points;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;
    IndexSet ghost_dofs;
  };
} // namespace ErrorEstimation

#endif