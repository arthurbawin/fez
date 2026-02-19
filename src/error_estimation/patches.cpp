
#include "error_estimation/patches.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

#include "utilities.h"

namespace ErrorEstimation
{
  template <int dim>
  PatchHandler<dim>::PatchHandler(
    const parallel::DistributedTriangulationBase<dim> &triangulation,
    const Mapping<dim>                                &mapping,
    const DoFHandler<dim>                             &dof_handler,
    const unsigned int                                 degree,
    const ComponentMask                               &mask)
    : triangulation(triangulation)
    , dof_handler(dof_handler)
    , mask(mask)
    , mpi_communicator(triangulation.get_mpi_communicator())
    , subdomain_id(Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_vertices(triangulation.n_vertices())
    , owned_vertices(n_vertices, false)
    , patches(n_vertices)
  {
    {
      /**
       * GridTools::get_locally_owned_vertices(triangulation) provides mesh
       * vertices with multiple owners at the boundary of a partition, which is
       * not what we want. Re-create the owned vector here.
       */
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
      // This leaves out vertices touching ONLY ghost-cells,
      // which would be marked as owned with
      // GridTools::get_locally_owned_vertices
      n_owned_vertices =
        std::count(owned_vertices.begin(), owned_vertices.end(), true);
    }

    relevant_dofs_support_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler, mask);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);

    // Create the 1d monomials of degree 0 to degree
    std::vector<Polynomials::Monomial<double>> monomials;
    for (unsigned int i = 0; i <= degree; ++i)
      monomials.push_back(Polynomials::Monomial<double>(i));

    // for(const auto &m : monomials)
    // {
    // 	std::cout << "Monomial:"<< std::endl;
    // 	m.print(std::cout);
    // }

    PolynomialSpace<dim> poly(monomials);

    // poly.output_indices(std::cout);
    // std::cout << "n_polynomials" << std::endl;
    // std::cout << poly.n_polynomials(monomials.size()) << std::endl;

    // Number of vertices to fit a polynomial of order "degree"
    unsigned int n_required_vertices = poly.n_polynomials(monomials.size());

    /**
     * Cannot create patches if there are too few vertices in the mesh
     * or in the partition.
     */
    AssertThrow(
      n_vertices >= n_required_vertices &&
        n_owned_vertices >= n_required_vertices,
      ExcMessage(
        "Each vertices patch for solution recovery requires at least " +
        std::to_string(n_required_vertices) +
        ", but either the total number of vertices in the mesh (" +
        std::to_string(n_vertices) + ") or in this subdomain (" +
        std::to_string(n_owned_vertices) +
        ") is lower than this number (or both)."));

    /**
     * Construct the patches of support points.
     * For each owned mesh vertex, keep adding layers of elements until there
     * are enough dof support points in the patch.
     */
    bool needs_further_expansion = true;
    for (unsigned int layer = 0; needs_further_expansion; ++layer)
    {
      add_element_layer(layer, n_required_vertices, needs_further_expansion);

      /**
       * If any rank needs expansion, all ranks must continue exhchanging
       * support points info (otherwise program will hang).
       */
      needs_further_expansion =
        Utilities::MPI::max(needs_further_expansion ? 1 : 0, mpi_communicator) >
        0;
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
      patch.neighbours =
        std::vector<std::pair<types::global_dof_index, Point<dim>>>(
          patch.neighbours_map.begin(), patch.neighbours_map.end());
      std::sort(patch.neighbours.begin(),
                patch.neighbours.end(),
                [](const auto &a, const auto &b) {
                  PointComparator<dim> comp;
                  return comp(a.second, b.second);
                });
      auto last = std::unique(patch.neighbours.begin(),
                              patch.neighbours.end(),
                              [](const auto &a, const auto &b) {
                                PointEquality<dim> comp;
                                return comp(a.second, b.second);
                              });
      patch.neighbours.erase(last, patch.neighbours.end());
    }

    compute_scalings_and_local_coordinates();
  }

  template <int dim>
  void PatchHandler<dim>::add_element_layer(const unsigned int layer,
                                            const unsigned n_required_vertices,
                                            bool &needs_further_expansion)
  {
    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();
    const FESystem<dim>           &fe       = dof_handler.get_fe();
    const unsigned int             n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

    needs_further_expansion = false;

    std::vector<bool> needs_expansion(n_vertices, false);

    for (types::global_vertex_index v = 0; v < n_vertices; ++v)
      if (owned_vertices[v])
        needs_expansion[v] =
          patches[v].neighbours_map.size() < n_required_vertices;

    /**
     * If previous layer included ghost cells, this layer may need to
     * request neighbouring dof support points to other ranks. Start by
     * exchanging this info, and add the non-owned, non-ghosted support points
     * to the patch.
     */
    if (layer > 0)
    {
      std::map<types::global_dof_index,
               std::vector<std::pair<types::global_dof_index, Point<dim>>>>
        connected_dofs_to_requested_dofs;

      exchange_ghost_layer_dofs(dofs_in_ghost_layer,
                                connected_dofs_to_requested_dofs);

      for (const auto &[dof, vertex_indices] : to_add_after_request)
      {
        const auto &connected_data = connected_dofs_to_requested_dofs.at(dof);

        // All owned vertices containing this dof
        for (const auto v : vertex_indices)
        {
          auto &patch = patches[v];
          if (needs_expansion[v])
            for (const auto &pair : connected_data)
              /**
               * FIXME: Here we check that the added dof is non-local, because
               * otherwise there is a difference with the layers obtained by
               * considering local dofs only (patch.dofs), I'm not sure why.
               */
              if (!locally_relevant_dofs.is_element(pair.first))
                patch.neighbours_map.insert(pair);
        }
      }
    }

    /**
     * Then extend the current patches (at owned mesh vertices) with the dof
     * support points in the next layer of elements.
     */
    for (types::global_vertex_index v = 0; v < n_vertices; ++v)
    {
      // Skip this patch if it already has enough support points
      if (!owned_vertices[v] || !needs_expansion[v])
        continue;

      auto &patch = patches[v];

      std::set<CellIterator>            new_cells;
      std::set<types::global_dof_index> new_dofs;

      if (layer == 0)
      {
        AssertThrow(patch.elements.size() == 0,
                    ExcMessage("Initial patch should be empty"));
        /**
         * First layer: add cells containing the vertex directly.
         * There is always at least 1 ghost cell layer, so the first cell layer
         * is local to the partition.
         */
        patch.center = vertices[v];
        for (const auto &cell : dof_handler.active_cell_iterators())
          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
            if (cell->vertex_index(i) == v)
            {
              new_cells.insert(cell);
              break;
            }
      }
      else
      {
        /**
         * Add cells touching any of the already stored support points.
         */
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dofs);
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            if (mask[fe.system_to_component_index(i).first])
            {
              // Add cell if one of its dofs is in current patch
              if (patch.neighbours_map.count(local_dofs[i]) > 0)
              // if (patch.dofs.count(local_dofs[i]) > 0)
              {
                new_cells.insert(cell);
                break;
              }
            }
        }
      }

      // Add new elements
      patch.elements.insert(new_cells.begin(), new_cells.end());

      // Add new dofs
      for (const auto &cell : new_cells)
      {
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          if (mask[fe.system_to_component_index(i).first])
            new_dofs.insert(local_dofs[i]);
      }
      // patch.dofs.insert(new_dofs.begin(), new_dofs.end());

      // Add neighbouring dofs and their support points
      for (const auto dof : new_dofs)
        patch.neighbours_map.insert(
          {dof, relevant_dofs_support_points.at(dof)});

      // Check if an additional layer will be required for at least one patch
      if (patch.neighbours_map.size() < n_required_vertices)
      {
        needs_further_expansion = true;

        /**
         * List the patch dofs already added and lying in the ghost layer. These
         * dofs might not have an adjacent cell in this partition, and a request
         * will be done to other partitions.
         */
        for (const auto &cell : patch.elements)
          if (cell->is_ghost())
          {
            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              if (mask[fe.system_to_component_index(i).first])
                if (ghost_dofs.is_element(local_dofs[i]))
                {
                  dofs_in_ghost_layer[cell->subdomain_id()].insert(
                    local_dofs[i]);
                  to_add_after_request[local_dofs[i]].insert(v);
                }
          }
      }
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

    // Boost serialization
    template <class Archive>
    void serialize(Archive &ar, const unsigned int)
    {
      ar &requested_dof;
      ar &neighbouring_dof;
      ar &support_point;
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
    std::map<types::global_dof_index,
             std::vector<std::pair<types::global_dof_index, Point<dim>>>>
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
        // for (auto dof : requested_dofs)
        //   if (!locally_relevant_dofs.is_element(dof))
        //     std::cerr << "ERROR: Rank " << mpi_rank
        //               << " received request for DoF " << dof << " from Rank "
        //               << source << " but I don't own it!" << std::endl;

        requested_dofs_by_others[source] = requested_dofs;
        all_requested_dofs.insert(requested_dofs.begin(), requested_dofs.end());

        // std::cout << "Rank " << subdomain_id << " received request from "
        //           << source << " for " << dof_list.size() << " DOFs."
        //           << std::endl;
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

    std::unordered_map<types::global_dof_index,
                       std::unordered_map<types::global_dof_index, Point<dim>>>
      connected_dofs_and_support_points;

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
              {
                const auto &pt = relevant_dofs_support_points.at(local_dofs[i]);
                connected_dofs_and_support_points[requested_dof].insert(
                  {local_dofs[i], pt});
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
          for (const auto &[dof, pt] :
               connected_dofs_and_support_points.at(requested_dof))
          {
            NeighbourDofData<dim> data;
            data.requested_dof    = requested_dof;
            data.neighbouring_dof = dof;
            data.support_point    = pt;
            connected_data.push_back(data);
          }
        data_to_send[dest] = connected_data;
      }

      std::map<unsigned int, MessageType> received_data =
        Utilities::MPI::some_to_some(mpi_communicator, data_to_send);

      for (const auto &[source, connected_data] : received_data)
      {
        for (const auto &data : connected_data)
        {
          const auto  requested_dof = data.requested_dof;
          const auto  connected_dof = data.neighbouring_dof;
          const auto &pt            = data.support_point;
          connected_dofs_to_requested_dofs[requested_dof].push_back(
            {connected_dof, pt});
        }
      }
    }
  }

  template <int dim>
  void PatchHandler<dim>::compute_scalings_and_local_coordinates()
  {
    for (auto &patch : patches)
    {
      const Point<dim> &center           = patch.center;
      Point<dim>       &scaling          = patch.scaling;
      const auto       &neighbours       = patch.neighbours;
      auto &neighbours_local_coordinates = patch.neighbours_local_coordinates;

      // Compute scaling (bounding box)
      for (const auto &[dof, pt] : neighbours)
        for (unsigned int d = 0; d < dim; ++d)
          scaling[d] = std::max(scaling[d], std::abs(pt[d] - center[d]));

      // Compute local coordinates : (x_i - center) / scaling
      for (const auto &[dof, pt] : neighbours)
      {
        // Although pt - center is a Tensor<1, dim> (as the difference of two
        // Points), it denotes local coordinates w.r.t. an origin, so it makes
        // sense to convert it to a Point.
        Point<dim> local_coordinates(pt - center);
        for (unsigned int d = 0; d < dim; ++d)
          local_coordinates[d] /= scaling[d];
        neighbours_local_coordinates.push_back({dof, local_coordinates});
      }
    }
  }

  template <int dim>
  void PatchHandler<dim>::write_element_patch_gmsh(
    const types::global_vertex_index vertex_index,
    const unsigned int               layer) const
  {
    // const auto &patch_elements = patches_of_elements[vertex_index];
    // const auto &patches_support_points =
    //   patches_of_support_points[vertex_index];
    // const auto       &scaling = scalings[vertex_index];
    // const Point<dim> &v       = triangulation.get_vertices()[vertex_index];

    // if (patch_elements.size() == 0)
    //   return;

    // std::ofstream out(
    //   "patch_elements_subdomain_" + std::to_string(subdomain_id) + "_vertex_"
    //   + std::to_string(vertex_index) + "_" + std::to_string(layer) + ".pos");
    // out << "View \"Subdomain " << subdomain_id << " - Vertex " <<
    // vertex_index
    //     << " - Layer " << layer << "\" {\n";

    // out << "SP(" << v[0] << "," << v[1] << "," << (dim == 3 ? v[2] : 0.)
    //     << "){2.};\n"
    //     << std::endl;

    // for (const auto &cell : patch_elements)
    // {
    //   // FIXME: Add for 3D
    //   out << "ST(";
    //   for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
    //   {
    //     const Point<dim> &pt = cell->vertex(iv);
    //     out << pt[0] << "," << pt[1] << ",0";
    //     if (iv < 2)
    //       out << ",";
    //   }

    //   // Color elements by owning partition (subdomain id)
    //   const unsigned int id = cell->subdomain_id();
    //   out << "){" << id << "," << id << "," << id << "};\n";
    // }
    // for (const auto &pt : patches_support_points)
    // {
    //   // Get the support point of this dof
    //   out << "SP(" << pt[0] << "," << pt[1] << "," << (dim == 3 ? pt[2] : 0.)
    //       << "){1.};\n"
    //       << std::endl;
    // }

    // // The bounding box
    // const double xmin = v[0] - scaling[0];
    // const double xmax = v[0] + scaling[0];
    // const double ymin = v[1] - scaling[1];
    // const double ymax = v[1] + scaling[1];
    // out << "SL(" << xmin << "," << ymin << ",0.," << xmax << "," << ymin
    //     << ",0.){1., 1.};\n";
    // out << "SL(" << xmax << "," << ymin << ",0.," << xmax << "," << ymax
    //     << ",0.){1., 1.};\n";
    // out << "SL(" << xmax << "," << ymax << ",0.," << xmin << "," << ymax
    //     << ",0.){1., 1.};\n";
    // out << "SL(" << xmin << "," << ymax << ",0.," << xmin << "," << ymin
    //     << ",0.){1., 1.};\n";

    // out << "};\n";
    // out.close();
  }

  template <int dim>
  void PatchHandler<dim>::write_support_points_patch(
    const LA::ParVectorType &solution,
    std::ostream            &out) const
  {
    std::vector<Point<dim>> global_vertices;
    std::vector<Patch<dim>> global_patches;

    const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(mpi_communicator);
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
        for (const auto &[dof, pt] : patch.neighbours)
          out << pt << " : sol = " << local_solution[dof] << std::endl;
      }
    }
  }

  template class PatchHandler<2>;
  template class PatchHandler<3>;
} // namespace ErrorEstimation
