#ifndef PATCHES_H
#define PATCHES_H

#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/types.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_system.h>
#include <types.h>

#include <tuple>

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

    /**
     * Center of this patch. This point is also an owned mesh vertex.
     */
    Point<dim> center;

    /**
     * Scaling of this patch defined as s_i = max_i |x_i - center|.
     */
    Point<dim> scaling;

    /**
     * The dof index associated with the vertex center of this patch if this
     * patch was created for a mask selecting a scalar field.
     */
    types::global_dof_index center_scalar_dof = numbers::invalid_unsigned_int;

    /**
     * The dofs associated with the center of this patch if it was created for
     * a mask selecting the components of a vector-valued field.
     */
    std::array<types::global_dof_index, dim> center_vector_dofs;

    /**
     * Dofs forming this patch and their support points in absolute coordinates.
     */
    std::vector<DofData> neighbours;

    /**
     * Dofs forming this patch and their support points in scaled coordinates:
     *
     * x_i,loc := (x_i,abs - center) / scaling_i.
     */
    std::vector<DofData> neighbours_local_coordinates;

    /**
     * For each dof in the patch, the rank owning this dof.
     */
    std::vector<types::subdomain_id> neighbours_owners;

    /**
     * For each dof in the patch, its weight used when averaging the polynomials
     * evaluation from the center to define the PPR operator.
     */
    std::vector<double> averaging_weights;

    /**
     * First layer of mesh cells around the center.
     */
    std::vector<CellIterator> first_cell_layer;

    // Boost serialization for MPI communications through deal.II wrappers
    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar &center &scaling &neighbours &neighbours_local_coordinates;
    }

  private:
    /**
     * Set of local (owned or ghost) cells around the vertex, and map of
     * neighbouring dofs (local and non-local). These are used only during patch
     * creation.
     */
    std::set<CellIterator> elements;

    /**
     * Map used during patch creation.
     */
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
     * Constructor.
     */
    PatchHandler(
      const parallel::DistributedTriangulationBase<dim> &triangulation,
      const Mapping<dim>                                &mapping,
      const DoFHandler<dim>                             &dof_handler,
      const unsigned int                                 degree,
      const ComponentMask                               &mask = {});

    /**
     * Create the patches, with enough dofs/support points around each (owned)
     * mesh vertex to fit a degree "degree" + 1 polynomial in a least-squares
     * sense.
     *
     * If @p enforce_full_rank_least_squares_matrices is true, then also build
     * the least-squares matrices during patches creation and make sure that
     * each matrix is full rank. Since this is the default intended use for
     * these patches, this is true by default.
     *
     * If @p enforce_full_rank_least_squares_matrices is false, @p n_layers
     * specifies the number of layers of cells used to create the patch around
     * each (owned) mesh vertex, and the dofs from only those cells constitute
     * the patch. If @p enforce_full_rank_least_squares_matrices is true, this
     * argument is ignored altogether. This argument is intended for testing
     * only, to ensure that the first n layers are correclty computed on
     * arbitrary mesh partitions.
     */
    void
    build_patches(const bool enforce_full_rank_least_squares_matrices = true,
                  const unsigned int n_layers                         = 1);

    /**
     * Return the patches associated with each (owned) mesh vertex on this
     * partition.
     */
    const std::vector<Patch<dim>> &get_patches() const;

    /**
     * Gather the patches to the root process and write them to @p out.
     * This function is intended for debug and unit tests.
     *
     * If @p write_for_gmsh is true, also write these patches as .pos files
     * to be visualized as views with Gmsh.
     */
    void write_support_points_patch(const std::string       &output_dir,
                                    const LA::ParVectorType &solution,
                                    std::ostream            &out = std::cout,
                                    bool write_for_gmsh          = false) const;

  private:
    /**
     * Add a layer of elements (and their support points for the given mask) to
     * the patch of the given mesh vertex.
     */
    void add_element_layer(const unsigned int layer,
                           const bool         enforce_full_rank,
                           bool              &at_least_one_patch_modified);

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
      std::map<
        types::global_dof_index,
        std::vector<
          std::tuple<types::global_dof_index, Point<dim>, types::subdomain_id>>>
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

    MPI_Comm           mpi_communicator;
    const unsigned int mpi_rank;

    // Number of mesh vertices on this partition
    unsigned int n_vertices;

    /**
     * Polynomial basis to fit a degree p + 1 polynomial
     */
    std::unique_ptr<PolynomialSpace<dim>> monomials_recovery;

    /**
     * The dimension of the basis of monomials_recovery
     */
    unsigned int dim_recovery_basis;

    /**
     * Minimal number of vertices to fit a polynomial of order "degree".
     * This numbered is not guaranteed to yield a unique solution to the
     * least-squares fitting if the vertices are not arranged in a way that
     * yields a full-rank least-squares matrix, in which case the patch will be
     * enlarged.
     */
    unsigned int n_required_vertices;

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
     * The least squares matrix for each (owned) mesh vertex.
     */
    std::vector<FullMatrix<double>> least_squares_matrices;

    /**
     * Rank of each least squares matrix.
     */
    std::vector<unsigned int> least_squares_matrices_rank;

    /**
     * Map to request neighbouring dofs to other ranks.
     * For each neighbouring rank, a set of dof indices for which the neighbours
     * are requested. These dofs may be ghosted on the specified rank, but
     * they'll still know the neighbouring dofs to these dofs.
     */
    std::map<types::subdomain_id, std::set<types::global_dof_index>>
      dofs_to_request;

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

    /**
     * A map storing which rank owns which ghost dof on this partition.
     */
    std::map<types::global_dof_index, types::subdomain_id> ghost_dof_to_owner;
  };
} // namespace ErrorEstimation

/* ---------------- template and inline functions ----------------- */

namespace ErrorEstimation
{
  template <int dim>
  const std::vector<Patch<dim>> &PatchHandler<dim>::get_patches() const
  {
    return patches;
  }
} // namespace ErrorEstimation

#endif
