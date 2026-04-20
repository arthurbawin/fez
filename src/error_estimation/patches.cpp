
#include "error_estimation/patches.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

#include <tuple>

#include "mesh_and_dof_tools.h"
#include "utilities.h"
#include <Eigen/Dense>

namespace ErrorEstimation
{
  template <int dim>
  PatchHandler<dim>::PatchHandler(
    const parallel::DistributedTriangulationBase<dim> &triangulation,
    const Mapping<dim>                                &mapping,
    const DoFHandler<dim>                             &dof_handler,
    const LA::ParVectorType                           &solution,
    const unsigned int                                 degree,
    const ComponentMask                               &mask)
    : triangulation(triangulation)
    , dof_handler(dof_handler)
    , mask(mask)
    , mpi_communicator(triangulation.get_mpi_communicator())
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_vertices(triangulation.n_vertices())
    , patches(n_vertices)
    , least_squares_matrices(n_vertices)
    , least_squares_matrices_rank(n_vertices)
    , least_squares_matrices_were_computed(false)
  {
    // Patches are to be used for the Polynomial Preserving Recovery, so it only
    // makes sense to define them for a scalar or vector-valued mask
    const unsigned int n_mask_components =
      mask.n_selected_components(dof_handler.get_fe().n_components());
    AssertThrow(
      n_mask_components == 1 || n_mask_components == dim,
      ExcMessage(
        "You are trying to create a PatchHandler by providing a "
        "component mask selecting " +
        std::to_string(n_mask_components) +
        ", but a PatchHandler expects a mask that selects the components "
        "of a single scalar or vector-valued field, thus selecting "
        "either 1 or dim vector components of the full FESystem."));

    // Mark the locally owned mesh vertices
    get_owned_mesh_vertices(triangulation, mpi_rank, owned_vertices);
    n_owned_vertices =
      std::count(owned_vertices.begin(), owned_vertices.end(), true);

    relevant_dofs_support_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler, mask);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);

    // Map the ghost dofs to their owner
    std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
      all_local_ranges =
        Utilities::MPI::all_gather(mpi_communicator, solution.local_range());

    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);
    for (const auto dof : ghost_dofs)
      for (unsigned int rank = 0; rank < mpi_size; ++rank)
      {
        const auto &[start, end] = all_local_ranges[rank];
        if (dof >= start && dof < end)
        {
          ghost_dof_to_owner[dof] = rank;
          break;
        }
      }

    // Create the 1d monomials of degree 0 up to "degree"
    std::vector<Polynomials::Monomial<double>> monomials_1d;
    for (unsigned int i = 0; i <= degree; ++i)
      monomials_1d.push_back(Polynomials::Monomial<double>(i));

    // Create the dim-dimensional polynomial space for a fitting
    monomials_recovery = std::make_unique<PolynomialSpace<dim>>(monomials_1d);

    // Number of vertices to fit a polynomial of order "degree"
    dim_recovery_basis = monomials_recovery->n_polynomials(monomials_1d.size());
    n_required_vertices = dim_recovery_basis;

    AssertThrow(dim_recovery_basis == monomials_recovery->n(),
                ExcInternalError());

    /**
     * Cannot create patches if there are too few vertices in the mesh.
     */
    AssertThrow(
      n_vertices >= n_required_vertices,
      ExcMessage(
        "Each vertices patch for solution recovery requires at least " +
        std::to_string(n_required_vertices) +
        ", but either the total number of vertices in the mesh (" +
        std::to_string(n_vertices) + ") or in this subdomain (" +
        std::to_string(n_owned_vertices) +
        ") is lower than this number (or both)."));
  }

  template <int dim>
  void PatchHandler<dim>::build_patches(
    const bool         enforce_full_rank_least_squares_matrices,
    const unsigned int n_layers)
  {
    /**
     * Construct the patches of support points.
     * For each owned mesh vertex, keep adding layers of elements until there
     * are enough dof support points in the patch.
     */
    bool stop = false, at_least_one_patch_modified = true;
    for (unsigned int layer = 0; !stop; ++layer)
    {
      add_element_layer(layer,
                        enforce_full_rank_least_squares_matrices,
                        at_least_one_patch_modified);

      /**
       * If any rank needs expansion, all ranks must continue exchanging
       * support points info (otherwise program will hang).
       */
      at_least_one_patch_modified =
        Utilities::MPI::max(at_least_one_patch_modified ? 1 : 0,
                            mpi_communicator) > 0;

      if (enforce_full_rank_least_squares_matrices)
        // Stop if patches are no longer modified
        stop = !at_least_one_patch_modified;
      else
        // If we're not requesting that least-squares matrices are full rank,
        // then the prescribed number of layers controls the loop. Stop if n
        // layers were added.
        stop = layer + 1 == n_layers;

      // Add a maximum number of layers to avoid loops.
      // Unless on bizarre meshes, 1 to 3 layers of elements should be enough.
      AssertThrow(layer < 5,
                  ExcMessage(
                    "The maximum number of cell layers to define a patch for "
                    "the polynomial-preserving recovery operator is for now "
                    "capped to 5 layers. If you believe this value is too low "
                    "based for your specific mesh, this cap can be removed."));
    }

    /**
     * Convert map of neighbours to vector and sort.
     * FIXME: For now, the pairs are sorted based on the Points<dim> and in
     * lexicographic order, it would be more intuitive (and robust) to sort them
     * based on their dof indices (-: (but then the test results must be
     * re-written)
     */
    for (auto &patch : patches)
    {
      patch.neighbours.clear();
      patch.neighbours.reserve(patch.neighbours_map.size());
      for (const auto &[dof, data] : patch.neighbours_map)
        patch.neighbours.emplace_back(data);

      std::sort(patch.neighbours.begin(),
                patch.neighbours.end(),
                [](const auto &a, const auto &b) {
                  PointComparator<dim> comp;
                  return comp(a.pt, b.pt);
                });
      auto last = std::unique(patch.neighbours.begin(),
                              patch.neighbours.end(),
                              [](const auto &a, const auto &b) {
                                PointEquality<dim> comp;
                                return comp(a.pt, b.pt);
                              });
      patch.neighbours.erase(last, patch.neighbours.end());
    }

    compute_scalings_and_local_coordinates();

    if (enforce_full_rank_least_squares_matrices)
      least_squares_matrices_were_computed = true;
  }

  /**
   * Fill the Vandermonde matrix at mesh vertex v, according to the scaling
   * vector stored in the patch at v.
   */
  template <int dim>
  void fill_vandermonde_matrix(const Patch<dim>           &patch,
                               const unsigned int          dim_recovery_basis,
                               const PolynomialSpace<dim> &basis,
                               FullMatrix<double>         &mat)
  {
    const auto &neighbours = patch.neighbours;

    unsigned int i = 0;
    for (const auto &data : neighbours)
    {
      // Evaluate each monomial at local_coordinates
      for (unsigned int j = 0; j < dim_recovery_basis; ++j)
        mat(i, j) = basis.compute_value(j, data.local_pt);
      ++i;
    }
  }

  /**
   * Use Eigen to get the rank of a FullMatrix
   */
  template <int dim>
  unsigned int get_rank(const FullMatrix<double> &full_matrix,
                        Eigen::MatrixXd          &eigen_matrix)
  {
    const unsigned int m = full_matrix.m();
    const unsigned int n = full_matrix.n();
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
        eigen_matrix(i, j) = full_matrix(i, j);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(eigen_matrix);
    return lu_decomp.rank();
  }

  // Construct (A^T*A)^-1 * A^T
  template <int dim>
  unsigned int
  compute_least_squares_matrix(Patch<dim>                 &patch,
                               const unsigned int          dim_recovery_basis,
                               const PolynomialSpace<dim> &basis,
                               FullMatrix<double>         &workspace_AtA,
                               Eigen::MatrixXd            &workspace_eigenAtA,
                               FullMatrix<double>         &least_squares_mat)
  {
    const unsigned int n_adjacent = patch.neighbours.size();

    FullMatrix<double> A(n_adjacent, dim_recovery_basis);
    fill_vandermonde_matrix(patch, dim_recovery_basis, basis, A);
    A.Tmmult(workspace_AtA, A);

    const unsigned int rank = get_rank<dim>(workspace_AtA, workspace_eigenAtA);

    // If A^T * A is full rank, actually compute the least-squares matrix.
    // left_invert() throws in debug if matrix is singular, so we need to check
    // the rank ourselves
    if (rank >= dim_recovery_basis)
    {
      least_squares_mat.reinit(dim_recovery_basis, n_adjacent);
      least_squares_mat.left_invert(A);
    }
    return rank;
  }

  template <int dim>
  void compute_scaling_and_local_coordinates(Patch<dim> &patch)
  {
    const Point<dim> &center     = patch.center;
    Point<dim>       &scaling    = patch.scaling;
    auto             &neighbours = patch.neighbours;

    // Compute scaling (bounding box)
    scaling = Point<dim>();
    for (const auto &data : neighbours)
      for (unsigned int d = 0; d < dim; ++d)
        scaling[d] = std::max(scaling[d], std::abs(data.pt[d] - center[d]));

    // Compute local coordinates : (x_i - center) / scaling
    for (auto &data : neighbours)
    {
      // Although pt - center is a Tensor<1, dim> (as the difference of two
      // Points), it denotes local coordinates w.r.t. an origin, so it makes
      // sense to convert it to a Point.
      Point<dim> local_coordinates(data.pt - center);
      for (unsigned int d = 0; d < dim; ++d)
        local_coordinates[d] /= scaling[d];
      data.local_pt = local_coordinates;
    }
  }

  template <int dim>
  void PatchHandler<dim>::compute_scalings_and_local_coordinates()
  {
    for (auto &patch : patches)
      compute_scaling_and_local_coordinates(patch);
  }

  template <int dim>
  void PatchHandler<dim>::add_element_layer(const unsigned int layer,
                                            const bool enforce_full_rank,
                                            bool &at_least_one_patch_modified)
  {
    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();
    const FESystem<dim>           &fe       = dof_handler.get_fe();
    const unsigned int             n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

    at_least_one_patch_modified = false;

    std::vector<bool> needs_expansion(n_vertices, false);

    // Workspace matrices to compute least-squares matrices.
    // The least-squares matrices (A^T*A)^-1 * A^T are of size
    // dim_recovery_basis x n_adjacent, but n_adjacent varies and can change if
    // the patch is increased. The matrix A^T * A, however, is
    // dim_recovery_basis x dim_recovery_basis
    FullMatrix<double> AtA(dim_recovery_basis, dim_recovery_basis);
    Eigen::MatrixXd    eigenAtA =
      Eigen::MatrixXd::Zero(dim_recovery_basis, dim_recovery_basis);

    /**
     * For each mesh vertex, check if its patch needs to be enlarged,
     * which is the case if
     *   - there are not enough dofs to compute a least-squares matrix
     *   - the least-squares matrix is not full-rank
     *
     * If either is true and enforce_full_rank is true, then an additional layer
     * of mesh cells is added to this patch. If enforce_full_rank is false,
     * an additional layer is always added.
     *
     * Adding an additional layer of dofs is done in three steps:
     * - first add the locally available dofs (owned or ghosts) on the cells
     *   containing any of the already stored dofs in the patch.
     *   Add the ghost dofs of the previous layer to a list of dofs whose
     *   neighbours are not known on this partition and must be requested to
     *   neighbouring partitions, or even neighbours of neighbours, etc.
     * - send and receive the neighbours requests to other partitions.
     * - complete the additional layer of dofs with the dofs received from
     *   other partitions.
     */
    for (types::global_vertex_index v = 0; v < n_vertices; ++v)
    {
      if (owned_vertices[v])
      {
        auto &patch = patches[v];

        // Check if patch needs to be expanded
        if (enforce_full_rank)
        {
          // Expand patch if it does not have enough nodes to compute a
          // least-squares matrix
          const bool has_enough_neighbours =
            patch.neighbours_map.size() >= n_required_vertices;

          if (!has_enough_neighbours)
          {
            needs_expansion[v] = true;
          }
          else
          {
            // If patch has enough neighbours, compute least-squares matrix
            // and check if it is full rank.

            // Convert neighbours map to vector to compute least-squares matrix
            patch.neighbours.clear();
            patch.neighbours.reserve(patch.neighbours_map.size());
            for (const auto &[dof, data] : patch.neighbours_map)
              patch.neighbours.emplace_back(data);

            std::sort(patch.neighbours.begin(),
                      patch.neighbours.end(),
                      [](const DofData &a, const DofData &b) {
                        PointComparator<dim> comp;
                        return comp(a.pt, b.pt);
                      });
            auto last = std::unique(patch.neighbours.begin(),
                                    patch.neighbours.end(),
                                    [](const DofData &a, const DofData &b) {
                                      PointEquality<dim> comp;
                                      return comp(a.pt, b.pt);
                                    });
            patch.neighbours.erase(last, patch.neighbours.end());

            AssertDimension(patch.neighbours.size(),
                            patch.neighbours_map.size());

            // Compute scaling and local neighbours coordinates
            compute_scaling_and_local_coordinates(patch);

            // Compute least-squares matrix.
            // If rank is sufficient, least_squares_matrices[v] contains the
            // matrix, if not we will try again with a larger patch.
            const unsigned int rank =
              compute_least_squares_matrix(patch,
                                           dim_recovery_basis,
                                           *monomials_recovery,
                                           AtA,
                                           eigenAtA,
                                           least_squares_matrices[v]);
            least_squares_matrices_rank[v] = rank;

            if (rank < dim_recovery_basis)
            {
              needs_expansion[v] = true;
            }
          }
        }
        else
        {
          // Not enforcing full rank least-squares matrices, but simply adding
          // layers for testing. Simply mark that all patches need expansion,
          // regardless of the polynomial basis used.
          needs_expansion[v] = true;
        }

        // Now add the locally available dofs to the patch and prepare the list
        // of requests to the other ranks.
        if (needs_expansion[v])
        {
          at_least_one_patch_modified = true;

          std::set<CellIterator>            new_cells;
          std::set<types::global_dof_index> new_dofs;

          if (layer == 0)
          {
            Assert(patch.elements.size() == 0,
                   ExcMessage("Initial patch should be empty"));

            // Set center
            patch.center = vertices[v];

            /**
             * First layer: add cells containing the vertex directly.
             * There is always at least 1 ghost cell layer, so the first cell
             * layer is local to the partition.
             */
            for (const auto &cell : dof_handler.active_cell_iterators())
              for (unsigned int i = 0; i < cell->n_vertices(); ++i)
                if (cell->vertex_index(i) == v)
                {
                  new_cells.insert(cell);
                  break;
                }

            // On one of the cells in the first layer, find with which dof(s)
            // the center of the patch is associated, if any (only valid for
            // nodal FE spaces, for which there is a dof at the mesh vertices).
            // For higher-order geometry, there may not be a dof at the mesh
            // vertices (e.g., P2 geometry and P1 or P3 field).
            {
              const bool mask_is_scalar =
                mask.n_selected_components(fe.n_components()) == 1;
              const auto &cell = *(new_cells.begin());
              cell->get_dof_indices(local_dofs);
              unsigned int d = 0;
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                if (mask[fe.system_to_component_index(i).first])
                  // Compute distance between support point and mesh vertex
                  if (relevant_dofs_support_points.at(local_dofs[i])
                        .distance(patch.center) < 1e-12)
                  {
                    if (mask_is_scalar)
                      patch.center_scalar_dof = local_dofs[i];
                    else
                      patch.center_vector_dofs[d++] = local_dofs[i];
                  }
            }

            // Store this first layer. It will be used to define the PPR
            // operator.
            patch.first_cell_layer =
              std::vector<CellIterator>(new_cells.begin(), new_cells.end());
          }
          else
          {
            /**
             * Add the local cells touching any of the already stored support
             * points.
             */
            for (const auto &cell : dof_handler.active_cell_iterators())
            {
              cell->get_dof_indices(local_dofs);
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                if (mask[fe.system_to_component_index(i).first])
                {
                  // Add cell if one of its dofs is in current patch
                  if (patch.neighbours_map.count(local_dofs[i]) > 0)
                  {
                    new_cells.insert(cell);
                    break;
                  }
                }
            }

            /**
             * Prepare the request for the non-local cells touching any of the
             * stored neighbour. List the patch dofs already added and lying in
             * the ghost layer. These dofs might not have an adjacent cell in
             * this partition, and a request will be done to other partitions.
             */
            for (const auto &cell : patch.elements)
              if (cell->is_ghost())
              {
                cell->get_dof_indices(local_dofs);
                for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
                  if (mask[fe.system_to_component_index(i).first])
                    if (ghost_dofs.is_element(local_dofs[i]))
                    {
                      dofs_to_request[cell->subdomain_id()].insert(
                        local_dofs[i]);
                      to_add_after_request[local_dofs[i]].insert(v);
                    }
              }
          }

          // Add new (incomplete) layer of elements
          patch.elements.insert(new_cells.begin(), new_cells.end());

          // Add new (incomplete) set of dofs with their support points
          for (const auto &cell : new_cells)
          {
            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              if (mask[fe.system_to_component_index(i).first])
                new_dofs.insert(local_dofs[i]);
          }
          for (const auto dof : new_dofs)
          {
            DofData data;
            data.dof   = dof;
            data.pt    = relevant_dofs_support_points.at(dof);
            data.owner = locally_owned_dofs.is_element(dof) ?
                           mpi_rank :
                           ghost_dof_to_owner.at(dof);

            patch.neighbours_map.insert({dof, data});
          }
        } // if patch needs expansion
      }
    }

    /**
     * Request the neighbouring dof to ghost dofs, and add these non-owned,
     * non-ghosted support points to the patch.
     */
    if (layer > 0)
    {
      std::map<types::global_dof_index, std::vector<DofData>>
        connected_dofs_to_requested_dofs;

      exchange_ghost_layer_dofs(dofs_to_request,
                                connected_dofs_to_requested_dofs);

      dofs_to_request.clear();

      std::map<types::global_dof_index, std::set<types::global_vertex_index>>
        to_add_after_next_request;

      // Add the received neighbours to all patches who needed them on this
      // partition. Also prepare for the next request, by adding all received
      // non-local dofs to the list of dofs whose neighbours will be requested
      // next.
      for (const auto &[dof, vertex_indices] : to_add_after_request)
      {
        Assert(connected_dofs_to_requested_dofs.count(dof) > 0,
               ExcInternalError());
        const auto &connected_data = connected_dofs_to_requested_dofs.at(dof);

        for (const auto v : vertex_indices)
        {
          // Dofs are typically neighbours in multiple patches, and one may have
          // requested its neighbours while other patches do not need more dofs.
          // Thus, only add the received neighbours if this patch actually
          // needed expansion.
          if (needs_expansion[v])
          {
            auto &patch = patches[v];
            for (const DofData &data : connected_data)
            {
              const unsigned int connected_dof = data.dof;
              patch.neighbours_map.insert({connected_dof, data});

              // The neighbours of this connected dof will need to be added
              // again to this vertex patch if the patch must be enlarged again
              if (!locally_relevant_dofs.is_element(connected_dof))
              {
                to_add_after_next_request[connected_dof].insert(v);
                dofs_to_request[data.owner].insert(connected_dof);
              }
            }
          }
        }
      }

      to_add_after_request = to_add_after_next_request;
    }
  }

  /**
   * A small struct to share by MPI::some_to_some :
   * the requesting dof, one of its neighbour and its support point.
   */
  template <int dim>
  struct NeighbourDofData
  {
    types::global_dof_index requested_dof;
    types::global_dof_index neighbouring_dof;
    Point<dim>              support_point;
    types::subdomain_id     neighbour_owner;

    // Boost serialization
    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar &requested_dof;
      ar &neighbouring_dof;
      ar &support_point;
      ar &neighbour_owner;
    }
  };

  /**
   * Given a list of dofs located in the ghost layer of this partition,
   * we want to get from its owning proc a list of dofs belonging to the cell(s)
   * touching this dof.
   */
  template <int dim>
  void PatchHandler<dim>::exchange_ghost_layer_dofs(
    const std::map<types::subdomain_id, std::set<types::global_dof_index>>
      &dofs_to_request,
    std::map<types::global_dof_index, std::vector<DofData>>
      &connected_dofs_to_requested_dofs)
  {
    const FESystem<dim> &fe = dof_handler.get_fe();
    std::map<types::subdomain_id, std::vector<types::global_dof_index>>
      requested_dofs_by_others;
    // Also dump all the requests from others into a unique set
    std::set<types::global_dof_index> all_requested_dofs;

    /**
     * First, exchange the lists of dofs for which each partition needs the
     * connected dof support points, with MPI::some_to_some.
     */
    {
      using MessageType = std::vector<types::global_dof_index>;

      // Convert the requested set to a vector of dof indices
      // because set has no built-in serialization
      std::map<types::subdomain_id, MessageType> dofs_to_request_vec;
      for (const auto &[dest, requested_dofs] : dofs_to_request)
        dofs_to_request_vec[dest] =
          MessageType(requested_dofs.begin(), requested_dofs.end());

      std::map<unsigned int, MessageType> received_data =
        Utilities::MPI::some_to_some(mpi_communicator, dofs_to_request_vec);

      for (const auto &[source, requested_dofs] : received_data)
      {
        if constexpr (running_in_debug_mode())
        {
          // Check that requests were sent to the right ranks
          for (auto dof : requested_dofs)
            Assert(
              locally_relevant_dofs.is_element(dof),
              ExcMessage(
                "Rank " + std::to_string(mpi_rank) +
                " received request for DoF " + std::to_string(dof) +
                " from rank " + std::to_string(source) +
                " but does not have this dof as either owned or ghosted."));
        }

        requested_dofs_by_others[source] = requested_dofs;
        all_requested_dofs.insert(requested_dofs.begin(), requested_dofs.end());
      }
    }

    /**
     * Get the dof indices and the support points of (owned or relevant) dofs
     * connected to the requested dofs.
     *
     * These should be interpreted as : if you are using a ghost dof that I own,
     * and thus would have expanded the patch with a cell that I own and is not
     * ghosted on your rank, here are the dofs and support points that you would
     * have added.
     */
    const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

    std::unordered_map<
      types::global_dof_index,
      std::vector<
        std::tuple<types::global_dof_index, Point<dim>, types::subdomain_id>>>
      connected_dofs_data;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
      {
        cell->get_dof_indices(local_dofs);
        for (const auto requested_dof : all_requested_dofs)
        {
          auto it =
            std::find(local_dofs.begin(), local_dofs.end(), requested_dof);
          if (it != local_dofs.end())
          {
            // Add all dofs on this element for the prescribed field
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              if (mask[fe.system_to_component_index(i).first])
                if (local_dofs[i] != requested_dof)
                {
                  const auto &pt =
                    relevant_dofs_support_points.at(local_dofs[i]);

                  // Store the owner of this dof to send.
                  // Dofs on an owned cell are not necessarily owned, and
                  // similarly dofs on a ghost cell are not necessarily ghosted,
                  // so check explicitly if the dof is owned.
                  const types::subdomain_id owner =
                    locally_owned_dofs.is_element(local_dofs[i]) ?
                      mpi_rank :
                      ghost_dof_to_owner.at(local_dofs[i]);

                  connected_dofs_data[requested_dof].push_back(
                    {local_dofs[i], pt, owner});
                }
          }
        }
      }

    /**
     * Send the connected dofs and support points back.
     */
    {
      using MessageType = std::vector<NeighbourDofData<dim>>;

      /**
       * For each requesting rank, add relevant NeighbourDofData to a vector
       * This repeats the requested_dof across each NeighbourDofData, but should
       * be faster than exchanging maps
       */
      std::map<types::subdomain_id, MessageType> data_to_send;

      for (const auto &[dest, requested_dofs] : requested_dofs_by_others)
      {
        MessageType connected_data;
        for (const auto requested_dof : requested_dofs)
          for (const auto &[dof, pt, owner] :
               connected_dofs_data.at(requested_dof))
          {
            NeighbourDofData<dim> data;
            data.requested_dof    = requested_dof;
            data.neighbouring_dof = dof;
            data.support_point    = pt;
            data.neighbour_owner  = owner;
            connected_data.push_back(data);
          }
        data_to_send[dest] = connected_data;
      }

      std::map<unsigned int, MessageType> received_data =
        Utilities::MPI::some_to_some(mpi_communicator, data_to_send);

      for (const auto &[source, connected_data] : received_data)
        for (const auto &data : connected_data)
        {
          DofData dof_data;
          dof_data.dof   = data.neighbouring_dof;
          dof_data.pt    = data.support_point;
          dof_data.owner = data.neighbour_owner;

          connected_dofs_to_requested_dofs[data.requested_dof].push_back(
            dof_data);
        }
    }
  }

  template <int dim>
  void PatchHandler<dim>::write_support_points_patch(
    const std::string       &output_dir,
    const LA::ParVectorType &solution,
    std::ostream            &out,
    bool                     write_for_gmsh) const
  {
    std::vector<Point<dim>> global_vertices;
    std::vector<Patch<dim>> global_patches;

    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

    // Gather the mesh vertices
    {
      std::vector<Point<dim>> local_vertices;
      for (unsigned int i = 0; i < n_vertices; ++i)
      {
        if (owned_vertices[i])
          local_vertices.push_back(vertices[i]);
      }

      std::vector<std::vector<Point<dim>>> gathered_vertices =
        Utilities::MPI::all_gather(mpi_communicator, local_vertices);
      for (const auto &vec : gathered_vertices)
        global_vertices.insert(global_vertices.end(), vec.begin(), vec.end());
      std::sort(global_vertices.begin(),
                global_vertices.end(),
                PointComparator<dim>());
    }

    // Gather the patches
    {
      std::vector<std::vector<Patch<dim>>> gathered_patches =
        Utilities::MPI::all_gather(mpi_communicator, patches);

      global_patches.resize(global_vertices.size());
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        bool found = false;
        for (unsigned int r = 0; r < mpi_size; ++r)
          for (const auto &patch : gathered_patches[r])
          {
            if (patch.neighbours.size() > 0)
              if (global_vertices[i].distance(patch.center) < 1e-12)
              {
                global_patches[i] = patch;
                found             = true;
                break;
              }
          }
        AssertThrow(found, ExcMessage("Vertex not found"));
      }
    }

    /**
     * To print the FE solution at the points of the patches, create a vector of
     * relevant dofs containing in addition the non-local dofs. Since the print
     * is done from rank 0, this amounts to adding *all* dofs as relevant. This
     * is only done for testing on small meshes, where a single rank can store
     * the complete set of dofs.
     */
    IndexSet          all_dofs = complete_index_set(dof_handler.n_dofs());
    LA::ParVectorType local_solution;
    local_solution.reinit(locally_owned_dofs, all_dofs, mpi_communicator);
    local_solution = solution;

    if (mpi_rank == 0)
    {
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        const auto &patch = global_patches[i];

        AssertThrow(patch.neighbours.size() > 0, ExcMessage("No neighbours"));

        out << "Mesh vertex " << global_vertices[i] << std::endl;
        out << "Center      " << patch.center << std::endl;
        out << "Scaling     " << patch.scaling << std::endl;
        for (const auto &data : patch.neighbours)
          out << data.pt << " : sol = " << local_solution[data.dof]
              << std::endl;
      }

      if (write_for_gmsh)
      {
        for (unsigned int i = 0; i < global_vertices.size(); ++i)
        {
          std::ofstream out(output_dir + "patches_" + std::to_string(mpi_size) +
                            "ranks_vertex" + std::to_string(i) + ".pos");
          out << "View \"patches_vertex_" << i << "\"{\n";

          const auto &patch = global_patches[i];
          const auto &c     = patch.center;
          out << "SP(" << c[0] << "," << c[1] << "," << (dim == 3 ? c[2] : 0.)
              << "){2.};\n"
              << std::endl;
          for (const auto &data : patch.neighbours)
          {
            out << "SP(" << data.pt[0] << "," << data.pt[1] << ","
                << (dim == 3 ? data.pt[2] : 0.) << "){1.};\n"
                << std::endl;
          }
          out << "};\n";
          out.close();
        }
      }
    }
  }

  template class PatchHandler<2>;
  template class PatchHandler<3>;
} // namespace ErrorEstimation
