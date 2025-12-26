
#include "error_estimation/patches.h"

#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>

namespace ErrorEstimation
{
  template <int dim>
  Patches<dim>::Patches(
    const parallel::DistributedTriangulationBase<dim> &triangulation,
    const Mapping<dim>                                &mapping,
    const DoFHandler<dim>                             &dof_handler,
    const unsigned int                                 degree,
    const ComponentMask                               &mask)
    : triangulation(triangulation)
    , dof_handler(dof_handler)
    , fe(dof_handler.get_fe())
    , mpi_communicator(triangulation.get_mpi_communicator())
    , mask(mask)
    , my_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , size(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , subdomain_id(Utilities::MPI::this_mpi_process(mpi_communicator))
    , n_vertices(triangulation.n_vertices())
    // , owned_vertices(GridTools::get_locally_owned_vertices(triangulation))
    , owned_vertices(n_vertices, false)
    // , n_owned_vertices(owned_vertices.size())
    , patches_of_elements(n_vertices)
    , patches_of_dofs(n_vertices)
    , patches_of_support_points(n_vertices)
    , n_layers(n_vertices)
    , scalings(n_vertices)
  {
    /**
     * GridTools::get_locally_owned_vertices(triangulation) gives mesh vertices
     * with multiple owners at the boundary of a partition.
     * Re-create the owned vector here.
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
    // which would be marked as owned with GridTools::get_locally_owned_vertices

    relevant_dofs_support_points =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);

    print_ghost_dofs();
    print_ghost_cells();

    MPI_Barrier(mpi_communicator);
    for (unsigned int r = 0;
         r < Utilities::MPI::n_mpi_processes(mpi_communicator);
         ++r)
      if (r == subdomain_id)
        for (auto dof : ghost_dofs)
          std::cout << "Rank " << r << " : ghost " << dof << std::endl;
    MPI_Barrier(mpi_communicator);

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

    // std::cout << "Patches require " << n_required_vertices
    //           << " vertices minimum" << std::endl;

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

    // Get neighbouring subdomains. We only exchange information with these
    // ranks.
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_ghost())
        neighbouring_subdomains.insert(cell->subdomain_id());

    // Check that neighbours are symmetric
    MPI_Barrier(mpi_communicator);
    for (unsigned int r = 0;
         r < Utilities::MPI::n_mpi_processes(mpi_communicator);
         ++r)
      if (r == subdomain_id)
        for (auto n : neighbouring_subdomains)
          std::cout << "Rank " << my_rank << " has rank " << n
                    << " as neighbour" << std::endl
                    << std::flush;
    MPI_Barrier(mpi_communicator);

    // Initialize the request to neighbours (they might not be needed)
    for (const auto id : neighbouring_subdomains)
      dofs_in_ghost_layer[id];

    std::cout << "Rank " << my_rank << " first barrier" << std::endl;
    MPI_Barrier(mpi_communicator);

    bool needs_further_expansion = true;
    for (unsigned int layer = 0;; ++layer)
    {
      std::cout << "Rank " << my_rank << " entering add layer " << layer
                << std::endl;
      add_element_nth_layer(layer,
                            n_required_vertices,
                            needs_further_expansion);
      for (unsigned int i = 0; i < n_vertices; ++i)
        if (owned_vertices[i])
          write_element_patch_gmsh(i, layer);

      write_support_points_patch(layer);
      std::cout << "Rank " << my_rank << " finished loop " << layer
                << std::endl;

      // 2. AGREE GLOBALLY: Do we need another layer?
      // If ANY rank needs expansion, ALL ranks must continue to keep MPI in
      // sync.
      const bool globally_needs_expansion =
        Utilities::MPI::max(needs_further_expansion ? 1 : 0, mpi_communicator) >
        0;

      if (!globally_needs_expansion)
        break;

      needs_further_expansion = globally_needs_expansion;
    }

    // compute_scalings();
    // print_owned_vertices();
  }

  template <int dim>
  void Patches<dim>::add_element_nth_layer(const unsigned int layer,
                                           const unsigned n_required_vertices,
                                           bool &needs_further_expansion)
  {
    const unsigned int                   n_dofs_per_cell = fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

    needs_further_expansion = false;

    std::vector<bool> needs_expansion(n_vertices, false);

    for (types::global_vertex_index vertex_index = 0; vertex_index < n_vertices;
         ++vertex_index)
    {
      needs_expansion[vertex_index] =
        patches_of_support_points[vertex_index].size() < n_required_vertices;
    }

    /**
     * If the previous layer included ghost cells, this layer may need to
     * request neighbouring dof support points to other ranks. Start by
     * exchanging this info, and add the non-owned, non-ghosted support points
     * to the patch.
     */
    if (layer > 0)
    {
      std::map<types::global_dof_index, std::vector<Point<dim>>>
        connected_support_points_to_requested_dofs;

      std::cout << "Rank " << my_rank << " entering exchange" << std::endl;
      exchange_ghost_layer_dofs(dofs_in_ghost_layer,
                                connected_support_points_to_requested_dofs);

      for (const auto &[dof, vertex_indices] : to_add_after_request)
      {
        const auto &connected_support_points =
          connected_support_points_to_requested_dofs.at(dof);

        // All owned vertices containing this dof
        for (const auto vertex_index : vertex_indices)
        {
          auto &patches_support_points =
            patches_of_support_points[vertex_index];

          // if (patches_support_points.size() < n_required_vertices)
          if (needs_expansion[vertex_index])
            for (const auto &pt : connected_support_points)
              patches_support_points.push_back(pt);
        }
      }
    }

    /**
     * Then extend the current patches (at owned mesh vertices) with the dof
     * support points in the next layer of elements.
     */
    for (types::global_vertex_index vertex_index = 0; vertex_index < n_vertices;
         ++vertex_index)
    {
      if (!owned_vertices[vertex_index])
        continue;

      auto &patch_elements         = patches_of_elements[vertex_index];
      auto &patch_dofs             = patches_of_dofs[vertex_index];
      auto &patches_support_points = patches_of_support_points[vertex_index];

      // Skip this patch if it already has enough support points
      // if (patches_support_points.size() >= n_required_vertices)
      if (!needs_expansion[vertex_index])
        continue;

      std::set<CellIterator>            new_cells;
      std::set<types::global_dof_index> new_dofs;

      if (layer == 0)
      {
        AssertThrow(patch_elements.size() == 0,
                    ExcMessage("Initial patch should be empty"));
        AssertThrow(patch_dofs.size() == 0,
                    ExcMessage("Initial patch should be empty"));
        AssertThrow(patches_support_points.size() == 0,
                    ExcMessage("Initial patch should be empty"));

        // if (my_rank == 0)
        // {
        //   std::cout << "Patch dofs - Layer " << layer << " Vertex "
        //             << vertex_index << std::endl;
        //   for (auto dof : patch_dofs)
        //     std::cout << dof << " ";
        //   std::cout << std::endl;
        // }

        /**
         * First layer: add cells containing the vertex directly.
         * There is always at least 1 ghost cell layer, so the first cell layer
         * is local to the partition.
         */
        for (const auto &cell : dof_handler.active_cell_iterators())
          for (unsigned int i = 0; i < cell->n_vertices(); ++i)
            if (cell->vertex_index(i) == vertex_index)
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
        if (my_rank == 0)
        {
          std::cout << "Patch dofs - Layer " << layer << " Vertex "
                    << vertex_index << std::endl;
          for (auto dof : patch_dofs)
            std::cout << dof << " ";
          std::cout << std::endl;
        }

        for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dofs);

          if (my_rank == 0)
            std::cout << my_rank << " Checking elem " << cell->id()
                      << std::endl;

          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            const unsigned int comp = fe.system_to_component_index(i).first;
            if (mask[comp])
            {
              if (my_rank == 0)
                std::cout << local_dofs[i] << " ";

              /**
               * Add cell if one of its dofs is in current patch
               */
              if (patch_dofs.count(local_dofs[i]) > 0)
              {
                // if (patch_elements.count(cell) == 0)
                new_cells.insert(cell);

                if (my_rank == 0)
                  std::cout << my_rank << " Added elem " << cell->id()
                            << std::endl;

                break;
              }
            }
          }
        }
      }

      // Add new elements
      patch_elements.insert(new_cells.begin(), new_cells.end());

      // Add new dofs
      for (const auto &cell : new_cells)
      {
        cell->get_dof_indices(local_dofs);
        for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
        {
          const unsigned int comp = fe.system_to_component_index(i).first;
          if (mask[comp])
            new_dofs.insert(local_dofs[i]);
        }
      }
      patch_dofs.insert(new_dofs.begin(), new_dofs.end());

      // Add support points from dof indices
      for (const auto dof : new_dofs)
        patches_support_points.push_back(relevant_dofs_support_points.at(dof));

      // Check if an additional layer will be required for at least one patch
      if (patches_support_points.size() < n_required_vertices)
      {
        needs_further_expansion = true;

        if (my_rank == 0)
          std::cout << "Patch at vertex " << vertex_index
                    << " will need expansion" << std::endl;

        // /**
        //  * List the patch dofs already added and lying in the ghost
        //  layer.
        //  * These dofs might not have an adjacent cell in this partition,
        //  and
        //  * a request will be done to other partitions.
        //  */
        for (const auto &cell : patch_elements)
          if (cell->is_ghost())
          {
            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            {
              const unsigned int comp = fe.system_to_component_index(i).first;
              if (mask[comp])
                if (ghost_dofs.is_element(local_dofs[i]))
                {
                  // if (my_rank == 0)
                  std::cout << "Rank " << my_rank << " - Vertex "
                            << vertex_index << " needs neighbours of dof "
                            << local_dofs[i] << " from proc "
                            << cell->subdomain_id() << std::endl;

                  dofs_in_ghost_layer[cell->subdomain_id()].insert(
                    local_dofs[i]);
                  to_add_after_request[local_dofs[i]].insert(vertex_index);
                }
            }
          }
      }
    }
  }

  // template <int dim>
  // struct SupportPointData
  // {
  //   types::global_dof_index global_dof_index;
  //   std::vector<Point<dim>> support_points_on_cell;
  //   // double                          value;

  //   // Boost serialization
  //   template <class Archive>
  //   void serialize(Archive &ar, const unsigned int version)
  //   {
  //     ar &global_dof_index;
  //     ar &support_points_on_cell;
  //     // ar &value;
  //   }
  // };

  /**
   * Given a list of dofs located in the ghost layer of this partition,
   * we want to get from its owning proc a list of dofs belonging to the cell(s)
   * touching this dof.
   *
   * -
   */
  template <int dim>
  void Patches<dim>::exchange_ghost_layer_dofs(
    const std::map<types::subdomain_id, std::set<types::global_dof_index>>
      &dofs_to_request,
    std::map<types::global_dof_index, std::vector<Point<dim>>>
      &connected_support_points_to_requested_dofs)
  {
    auto comm = mpi_communicator;

    std::cout << "Rank " << my_rank << " ready (1)" << std::endl;
    MPI_Barrier(comm);
    std::cout << "Rank " << my_rank << " ready (2)" << std::endl;
    MPI_Barrier(comm);

    /**
     * Send to their owner and receive the list of ghost dofs whose connected
     * dof support points are required to construct the patches on this rank.
     */
    std::map<types::subdomain_id, std::vector<types::global_dof_index>>
      requested_dofs_by_others;
    // Also dump all the requests from others into a unique set
    std::set<types::global_dof_index> all_requested_dofs;

    // {
    //   using MessageType = std::vector<types::global_dof_index>;

    //   // Prepare to receive from the requested owners
    //   std::multimap<unsigned int, Utilities::MPI::Future<MessageType>>
    //     receive_futures;

    //   // Prepare to receive from neighbouring partitions only
    //   for (const auto source : neighbouring_subdomains)
    //     receive_futures.emplace(source,
    //                             Utilities::MPI::irecv<MessageType>(comm,
    //                                                                source));

    //   // Send the list of requests to their owners
    //   std::vector<Utilities::MPI::Future<void>> send_futures;
    //   for (const auto dest : neighbouring_subdomains)
    //   {
    //     const auto &dofs_to_request = dofs_in_ghost_layer.at(dest);
    //     MessageType dofs_vector(dofs_to_request.begin(),
    //     dofs_to_request.end()); send_futures.emplace_back(
    //       Utilities::MPI::isend(dofs_vector, comm, dest));
    //   }

    //   // Wait for the receives to be done
    //   for (auto &receive_future : receive_futures)
    //   {
    //     // Wait for the receive and obtain the message
    //     const MessageType  dof_list      = receive_future.second.get();
    //     const unsigned int source        = receive_future.first;
    //     requested_dofs_by_others[source] = dof_list;
    //     all_requested_dofs.insert(dof_list.begin(), dof_list.end());
    //     for (auto dof : dof_list)
    //       std::cout << "Rank " << subdomain_id << " received request from "
    //                 << source << " : " << dof << std::endl;
    //   }

    //   // Wait for the sends to be done as well.
    //   for (auto &send_future : send_futures)
    //   {
    //     send_future.wait();
    //     std::cout << "Send future (1) returned successfully." << std::endl;
    //   }
    // }

    /**
     * Using some_to_some to exchange global_dof_indices.
     * This function handles all irecv/isend/waits internally and is
     * deadlock-safe.
     */
    {
      using MessageType = std::vector<types::global_dof_index>;

      // 1. Prepare the data to be sent
      // Map key: destination rank, Map value: the list of DOFs I am requesting
      // from them
      std::map<unsigned int, MessageType> dofs_to_send;

      for (const auto dest : neighbouring_subdomains)
      {
        // Ensure the key exists in your source map (dofs_in_ghost_layer)
        if (dofs_in_ghost_layer.find(dest) != dofs_in_ghost_layer.end())
        {
          const auto &dof_set = dofs_in_ghost_layer.at(dest);
          dofs_to_send[dest]  = MessageType(dof_set.begin(), dof_set.end());
        }
      }

      // 2. Perform the communication
      // This returns a map where:
      // Key: the rank that sent data TO me
      // Value: the MessageType (vector of DOFs) they sent
      std::cout << "Rank " << my_rank << " starting some_to_some" << std::endl;
      std::map<unsigned int, MessageType> received_data =
        Utilities::MPI::some_to_some(mpi_communicator, dofs_to_send);

      // 3. Process the results
      requested_dofs_by_others.clear();
      all_requested_dofs.clear();

      for (const auto &[source, dof_list] : received_data)
      {
        for (auto dof : dof_list)
        {
          if (!locally_relevant_dofs.is_element(dof))
          {
            std::cerr << "ERROR: Rank " << my_rank
                      << " received request for DoF " << dof << " from Rank "
                      << source << " but I don't own it!" << std::endl;
          }
        }

        requested_dofs_by_others[source] = dof_list;
        all_requested_dofs.insert(dof_list.begin(), dof_list.end());

        // Optional: Debug print (Rank 0 only or carefully buffered)
        std::cout << "Rank " << subdomain_id << " received request from "
                  << source << " for " << dof_list.size() << " DOFs."
                  << std::endl;
      }
    }

    std::cout << "Rank " << my_rank << " at barrier (3)" << std::endl;
    MPI_Barrier(comm);

    // /**
    //  * Once the lists have been exchanged, send and receive the dof support
    //  * points connected to the requested dofs.
    //  *
    //  * These should be interpreted as : if you are using a ghost dof that I
    //  own,
    //  * and thus would have expanded the patch with a cell that I own and is
    //  not
    //  * ghosted on your rank, here are the dof support points that you would
    //  have
    //  * added.
    //  */
    // {
    //   // Get the dofs connected to the requested dofs
    //   const unsigned int n_dofs_per_cell =
    //     fe.n_dofs_per_cell();
    //   std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

    //   // The dofs on the element touching the requested dof
    //   std::map<types::global_dof_index, std::set<types::global_dof_index>>
    //     connected_dofs;

    //   /**
    //    * Loop over cells of this partition. For each cell, check if it
    //    contains
    //    * a requested dof, and if so add the relevant dofs to the map.
    //    */
    //   for (const auto &cell : dof_handler.active_cell_iterators())
    //     if (cell->is_locally_owned())
    //     {
    //       cell->get_dof_indices(local_dofs);

    //       for (const auto requested_dof : all_requested_dofs)
    //         if (std::find(local_dofs.begin(),
    //                       local_dofs.end(),
    //                       requested_dof) != local_dofs.end())
    //         {
    //           for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    //           {
    //             const unsigned int comp =
    //               fe.system_to_component_index(i).first;
    //             if (mask[comp])
    //             {
    //               connected_dofs[requested_dof].insert(local_dofs[i]);
    //             }
    //           }
    //         }
    //     }

    //   /**
    //    * For each requested dof, send and receive a list of support points.
    //    * The dof indices are not useful on other partitions, so send support
    //    * points directly instead.
    //    */
    //   using MessageType =
    //     std::map<types::global_dof_index, std::vector<Point<dim>>>;
    //   // using MessageType = std::vector<std::pair<types::global_dof_index,
    //   // std::vector<types::global_dof_index>>>;

    //   // Prepare to receive from the requested owners
    //   std::multimap<unsigned int, Utilities::MPI::Future<MessageType>>
    //     receive_futures;

    //   // Prepare to receive from neighbouring partitions only
    //   for (const auto source : neighbouring_subdomains)
    //     receive_futures.emplace(source,
    //                             Utilities::MPI::irecv<MessageType>(comm,
    //                                                                source));

    //   // Send the connected support points
    //   std::vector<Utilities::MPI::Future<void>> send_futures;
    //   for (const auto dest : neighbouring_subdomains)
    //   {
    //     // For the requests of this other rank, convert the dof indices to
    //     their
    //     // support points
    //     MessageType support_points_to_send;
    //     for (const auto dof : requested_dofs_by_others.at(dest))
    //     {
    //       // Draw
    //       std::ofstream out(
    //         "requested_dofs_for_" + std::to_string(subdomain_id) +
    //         "_connected_to_dof_" + std::to_string(dof) + ".pos");
    //       out << "View \"Subdomain " << subdomain_id
    //           << " - Requested support points for dof " << dof << "\" {\n";
    //       const auto &pt_dof = relevant_dofs_support_points.at(dof);
    //       out << "SP(" << pt_dof[0] << "," << pt_dof[1] << ","
    //           << (dim == 3 ? pt_dof[2] : 0.) << "){2.};\n"
    //           << std::endl;

    //       for (const auto connected_dof : connected_dofs.at(dof))
    //       {
    //         support_points_to_send[dof].push_back(
    //           relevant_dofs_support_points.at(connected_dof));
    //         const auto &pt = relevant_dofs_support_points.at(connected_dof);
    //         out << "SP(" << pt[0] << "," << pt[1] << ","
    //             << (dim == 3 ? pt[2] : 0.) << "){1.};\n"
    //             << std::endl;
    //       }

    //       out << "};\n";
    //       out.close();
    //     }

    //     send_futures.emplace_back(
    //       Utilities::MPI::isend(support_points_to_send, comm, dest));
    //   }

    //   // Wait for the receives to be done
    //   for (auto &receive_future : receive_futures)
    //   {
    //     const MessageType support_points_to_receive =
    //       receive_future.second.get();

    //     // Each requested dof is owned by a single rank, so we can just
    //     assign
    //     // the vector of support points to the received vector.
    //     for (const auto &[requested_dof, support_points] :
    //          support_points_to_receive)
    //       connected_support_points_to_requested_dofs[requested_dof] =
    //         support_points;
    //   }

    //   // Wait for the sends to be done as well.
    //   for (auto &send_future : send_futures)
    //   {
    //     send_future.wait();
    //     std::cout << "Send future (2) returned successfully." << std::endl;
    //   }
    // }

    {
      const unsigned int n_dofs_per_cell = fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

      // 1. Identify which owned DoFs are connected to the DoFs others requested
      // from us
      std::map<types::global_dof_index, std::set<types::global_dof_index>>
        connected_dofs;

      for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() || cell->is_ghost())
        {
          cell->get_dof_indices(local_dofs);
          for (const auto requested_dof : all_requested_dofs)
          {
            auto it =
              std::find(local_dofs.begin(), local_dofs.end(), requested_dof);
            if (it != local_dofs.end())
            {
              for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
              {
                const unsigned int comp = fe.system_to_component_index(i).first;
                if (mask[comp])
                  connected_dofs[requested_dof].insert(local_dofs[i]);
              }
            }
          }
        }
      }

      // 2. Prepare the data for some_to_some
      // MessageType is std::map<types::global_dof_index,
      // std::vector<Point<dim>>>
      using MessageType =
        std::map<types::global_dof_index, std::vector<Point<dim>>>;
      std::map<unsigned int, MessageType> data_to_send;

      for (const auto &[dest_rank, dofs_requested_by_them] :
           requested_dofs_by_others)
      {
        MessageType support_points_map;
        for (const auto dof : dofs_requested_by_them)
        {
          // It is possible a rank requested a DoF we own, but it isn't
          // in our 'connected_dofs' yet if we haven't looped over the right
          // cells.
          if (connected_dofs.count(dof) > 0)
          {
            for (const auto connected_dof : connected_dofs.at(dof))
            {
              support_points_map[dof].push_back(
                relevant_dofs_support_points.at(connected_dof));
            }
          }
        }
        data_to_send[dest_rank] = support_points_map;
      }

      // 3. Perform the global exchange
      std::cout << "Rank " << my_rank << " starting some_to_some (2)"
                << std::endl;

      std::map<unsigned int, MessageType> received_data =
        Utilities::MPI::some_to_some(mpi_communicator, data_to_send);

      // 4. Unpack the received support points into your patch structure
      for (const auto &[sender_rank, support_points_map] : received_data)
      {
        for (const auto &[requested_dof, points_vec] : support_points_map)
        {
          connected_support_points_to_requested_dofs[requested_dof] =
            points_vec;
        }
      }

      std::cout << "Rank " << my_rank << " finished exchange (2)" << std::endl;
    }
  }

  template <int dim>
  void Patches<dim>::compute_scalings()
  {
    // const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

    // for (unsigned int i = 0; i < n_vertices; ++i)
    //   if (owned_vertices[i])
    //   {
    //     const Point<dim> &vi                = vertices[i];
    //     Point<dim>       &scaling           = scalings[i];
    //     const auto       &patch_of_vertices = patches_of_vertices[i];

    //     for (const auto &connected_vert : patch_of_vertices)
    //     {
    //       const Point<dim> &vj = vertices[connected_vert];
    //       for (unsigned int k = 0; k < dim; ++k)
    //         scaling[k] = std::max(scaling[k], std::abs(vi[k] - vj[k]));
    //     }
    //   }
  }

  template <int dim>
  void Patches<dim>::write_element_patch_gmsh(
    const types::global_vertex_index vertex_index,
    const unsigned int               layer) const
  {
    const auto &patch_elements = patches_of_elements[vertex_index];
    const auto &patches_support_points =
      patches_of_support_points[vertex_index];
    const auto       &scaling = scalings[vertex_index];
    const Point<dim> &v       = triangulation.get_vertices()[vertex_index];

    if (patch_elements.size() == 0)
      return;

    std::ofstream out(
      "patch_elements_subdomain_" + std::to_string(subdomain_id) + "_vertex_" +
      std::to_string(vertex_index) + "_" + std::to_string(layer) + ".pos");
    out << "View \"Subdomain " << subdomain_id << " - Vertex " << vertex_index
        << " - Layer " << layer << "\" {\n";

    out << "SP(" << v[0] << "," << v[1] << "," << (dim == 3 ? v[2] : 0.)
        << "){2.};\n"
        << std::endl;

    for (const auto &cell : patch_elements)
    {
      // FIXME: Add for 3D
      out << "ST(";
      for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
      {
        const Point<dim> &pt = cell->vertex(iv);
        out << pt[0] << "," << pt[1] << ",0";
        if (iv < 2)
          out << ",";
      }

      // Color elements by owning partition (subdomain id)
      const unsigned int id = cell->subdomain_id();
      out << "){" << id << "," << id << "," << id << "};\n";
    }
    for (const auto &pt : patches_support_points)
    {
      // Get the support point of this dof
      out << "SP(" << pt[0] << "," << pt[1] << "," << (dim == 3 ? pt[2] : 0.)
          << "){1.};\n"
          << std::endl;
    }

    // The bounding box
    const double xmin = v[0] - scaling[0];
    const double xmax = v[0] + scaling[0];
    const double ymin = v[1] - scaling[1];
    const double ymax = v[1] + scaling[1];
    out << "SL(" << xmin << "," << ymin << ",0.," << xmax << "," << ymin
        << ",0.){1., 1.};\n";
    out << "SL(" << xmax << "," << ymin << ",0.," << xmax << "," << ymax
        << ",0.){1., 1.};\n";
    out << "SL(" << xmax << "," << ymax << ",0.," << xmin << "," << ymax
        << ",0.){1., 1.};\n";
    out << "SL(" << xmin << "," << ymax << ",0.," << xmin << "," << ymin
        << ",0.){1., 1.};\n";

    out << "};\n";
    out.close();
  }

  template <int dim>
  struct PointComparator
  {
    bool operator()(const Point<dim> &a, const Point<dim> &b) const
    {
      for (unsigned int d = 0; d < dim; ++d)
      {
        if (std::abs(a[d] - b[d]) > 1e-14)
          return a[d] < b[d];
      }
      return false;
    }
  };

  template <int dim>
  struct PointEquality
  {
    bool operator()(const Point<dim> &a, const Point<dim> &b) const
    {
      return a.distance(b) < 1e-14;
    }
  };

  template <int dim>
  void Patches<dim>::write_support_points_patch(const unsigned int layer)
  {
    std::vector<Point<dim>>              global_vertices;
    std::vector<std::vector<Point<dim>>> global_patches;

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

    // Sort and unique the patches
    for (auto &patch : patches_of_support_points)
    {
      std::sort(patch.begin(), patch.end(), PointComparator<dim>());
      auto last = std::unique(patch.begin(), patch.end(), PointEquality<dim>());
      patch.erase(last, patch.end());
    }

    // Gather the patches
    {
      // Attach each patch to its mesh vertex Point<dim>
      std::map<Point<dim>, std::vector<Point<dim>>, PointComparator<dim>>
        patches_of_support_points_with_vertex;
      for (unsigned int i = 0; i < n_vertices; ++i)
        if (owned_vertices[i])
          patches_of_support_points_with_vertex[vertices[i]] =
            patches_of_support_points[i];

      std::vector<
        std::map<Point<dim>, std::vector<Point<dim>>, PointComparator<dim>>>
        gathered_patches =
          Utilities::MPI::all_gather(mpi_communicator,
                                     patches_of_support_points_with_vertex);

      // global_patches : mesh_vertex -> patch of support points
      global_patches.resize(global_vertices.size());
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        bool found = false;
        for (unsigned int r = 0; r < size; ++r)
          for (const auto &[pt, patch] : gathered_patches[r])
            if (global_vertices[i].distance(pt) < 1e-12)
            {
              global_patches[i] = patch;
              found             = true;
              break;
            }
        AssertThrow(found, ExcMessage("Vertex not found"));
      }
    }

    if (my_rank == 0)
    {
      std::ofstream out("patches_support_points_" + std::to_string(size) +
                        "procs_layer_" + std::to_string(layer) + ".txt");

      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        out << "Mesh vertex " << global_vertices[i] << std::endl;

        const auto &patch_support_points = global_patches[i];

        for (const auto pt : patch_support_points)
          out << pt << std::endl;
      }
      out.close();
    }
  }

  template <int dim>
  void Patches<dim>::print_owned_vertices() const
  {
    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

    std::ofstream out("owned_vertices_subdomain_" +
                      std::to_string(subdomain_id) + ".pos");
    out << "View \"Subdomain " << subdomain_id << " - Owned\" {\n";

    for (unsigned int i = 0; i < n_vertices; ++i)
      if (owned_vertices[i])
        out << "SP(" << vertices[i][0] << "," << vertices[i][1] << ","
            << (dim == 3 ? vertices[i][2] : 0.) << "){" << i << "};\n"
            << std::endl;

    out << "};\n";
    out.close();
  }

  template <int dim>
  void Patches<dim>::print_ghost_dofs() const
  {
    std::ofstream out("ghost_dofs_subdomain_" + std::to_string(subdomain_id) +
                      ".pos");
    out << "View \"Subdomain " << subdomain_id << " - Ghost\" {\n";

    for (auto dof : ghost_dofs)
    {
      const auto &pt = relevant_dofs_support_points.at(dof);
      out << "SP(" << pt[0] << "," << pt[1] << "," << (dim == 3 ? pt[2] : 0.)
          << "){1.};\n"
          << std::endl;
    }
    out << "};\n";
    out.close();
  }

  template <int dim>
  void Patches<dim>::print_ghost_cells() const
  {
    std::ofstream out("ghost_cells_subdomain_" + std::to_string(subdomain_id) +
                      ".pos");
    out << "View \"Subdomain " << subdomain_id << " - Ghost cells\" {\n";

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_ghost())
      {
        out << "ST(";
        for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
        {
          const Point<dim> &pt = cell->vertex(iv);
          out << pt[0] << "," << pt[1] << ",0";
          if (iv < 2)
            out << ",";
        }
        const unsigned int id = cell->subdomain_id();
        out << "){" << id << "," << id << "," << id << "};\n";
      }
    }
    out << "};\n";
    out.close();
  }

  template class Patches<2>;
  template class Patches<3>;
} // namespace ErrorEstimation