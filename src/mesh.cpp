
#include <boundary_conditions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/numerics/data_out.h>
#include <mesh.h>
#include <parameter_reader.h>

// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/manifold_lib.h>
// #include <deal.II/grid/tria_accessor.h>
// #include <deal.II/grid/tria_iterator.h>

// template <int dim>
// Triangulation<dim> read_mesh(const std::string &meshFile,
//                              const MPI_Comm &comm)
// {
//   Triangulation<dim> serial_tria;

//   GridIn<dim> grid_in;
//   grid_in.attach_triangulation(serial_tria);
//   std::ifstream input(meshFile);
//   AssertThrow(input, ExcMessage("Could not open mesh file: " + meshFile));
//   grid_in.read_msh(input);

//   // Partition serial triangulation:
//   GridTools::partition_triangulation(
//     Utilities::MPI::n_mpi_processes(comm), serial_tria);

//   // Create building blocks:
//   const TriangulationDescription::Description<dim> description =
//     TriangulationDescription::Utilities::
//       create_description_from_triangulation(serial_tria, comm);

//   // Create a fully distributed triangulation:
//   parallel::fullydistributed::Triangulation<dim> distr_tria(comm);
//   distr_tria.create_triangulation(description);

//   // // Optional: visualize partition
//   // DoFHandler<dim> dof_handler(distr_tria);
//   // FE_SimplexP<dim> fe(1);
//   // dof_handler.distribute_dofs(fe);
//   // MappingFE<dim> mapping(FE_SimplexP<dim>(1));

//   // Vector<double> cell_data(distr_tria.n_active_cells());

//   // for (const auto &cell : distr_tria.active_cell_iterators())
//   // {
//   //   cell_data[cell->active_cell_index()] = (double) cell->subdomain_id();
//   // }

//   // DataOut<dim> data_out;
//   // data_out.attach_dof_handler(dof_handler);
//   // data_out.add_data_vector(cell_data, "subdomain",
//   DataOut<2>::type_cell_data);

//   // data_out.build_patches(mapping);

//   // data_out.write_vtu_with_pvtu_record(
//   //   "./", "mesh", 0, MPI_COMM_WORLD, 2);

//   Vector<float> subdomain(distr_tria.n_active_cells());
//   for (unsigned int i = 0; i < subdomain.size(); ++i)
//     subdomain(i) = distr_tria.locally_owned_subdomain();

//   DoFHandler<dim> dof_handler(distr_tria);
//   FE_SimplexP<dim> fe(1);
//   dof_handler.distribute_dofs(fe);
//   MappingFE<dim> mapping(FE_SimplexP<dim>(1));

//   DataOut<dim> data_out;
//   data_out.attach_dof_handler(dof_handler);
//   data_out.add_data_vector(subdomain, "subdomain");

//   data_out.build_patches(mapping);

//   data_out.write_vtu_with_pvtu_record(
//     "./", "meshPartition", 0, comm, 2);

//   return distr_tria;
// }

/**
 * Deal.II does not read the names of the physical entities (I think).
 * Read them here and assign them to their boundary id.
 */
void read_gmsh_physical_names(const std::string                   &meshFile,
                              std::map<unsigned int, std::string> &tag2name,
                              std::map<std::string, unsigned int> &name2tag)
{
  std::ifstream in(meshFile);
  if (!in)
    throw std::runtime_error("Could not open file " + meshFile);

  std::string line;
  while (std::getline(in, line))
  {
    if (line == "$PhysicalNames")
    {
      // Next line contains the number of entries
      unsigned int num;
      if (!(in >> num))
        throw std::runtime_error("Invalid $PhysicalNames section in " +
                                 meshFile);

      // Each of the next 'num' lines: <dim> <id> "<name>"
      for (unsigned int i = 0; i < num; ++i)
      {
        unsigned int dim, id;
        std::string  name;
        in >> dim >> id;
        in >> std::ws; // skip spaces before name
        std::getline(in, name);

        // Gmsh usually puts the name in quotes; strip them
        if (!name.empty() && name.front() == '"')
        {
          name.erase(0, 1);
          if (!name.empty() && name.back() == '"')
            name.pop_back();
        }

        tag2name[id]   = name;
        name2tag[name] = id;
      }

      // Skip until end of section
      while (std::getline(in, line) && line != "$EndPhysicalNames")
        ;
      break;
    }
  }
}

template <int dim>
void check_single_boundary_condition(
  const BoundaryConditions::BoundaryCondition &bc,
  const ParameterReader<dim>                  &param)
{
  // Check that boundary id given in parameter file exists in the mesh
  AssertThrow(param.mesh.id2name.count(bc.id) == 1,
              ExcMessage(
                "A " + bc.physics_str +
                " boundary condition is prescribed on a mesh "
                "domain (Physical Entity) with id " +
                std::to_string(bc.id) +
                ", but this id either does not exist in the mesh or appears "
                "more than one time, which is not supported."));

  // Check that the boundary name exists
  AssertThrow(param.mesh.name2id.count(bc.gmsh_name) == 1,
              ExcMessage(
                "A " + bc.physics_str +
                " boundary condition is prescribed on a mesh domain (Physical "
                "Entity) named \"" +
                bc.gmsh_name +
                "\", but this entity either does not exist in the mesh or "
                "appears more than one time, which is not supported."));

  // Check that the prescribed name and id match in the mesh
  AssertThrow(param.mesh.id2name.at(bc.id) == bc.gmsh_name,
              ExcMessage(
                "A " + bc.physics_str +
                " boundary condition is prescribed on entity \"" +
                bc.gmsh_name + "\" with id " + std::to_string(bc.id) +
                ", but this id does not match this entity in the "
                "mesh. Instead, the entity exists in the mesh with id " +
                std::to_string(param.mesh.name2id.at(bc.gmsh_name)) + "."));
}

/**
 * Checks that the boundary conditions are compatible with the mesh:
 *
 * - all prescribed ids exist in the mesh
 * - all prescribed names exist and are attached to matching ids
 */
template <int dim>
void check_boundary_conditions_compatibility(const ParameterReader<dim> &param)
{
  // Check fluid conditions
  for (const auto &bc : param.fluid_bc)
    check_single_boundary_condition(bc, param);
  // Check pseudosolid conditions
  for (const auto &bc : param.pseudosolid_bc)
    check_single_boundary_condition(bc, param);
}

template <int dim>
void print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string        &filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto &face : triangulation.active_face_iterators())
      if (face->at_boundary())
        boundary_count[face->boundary_id()]++;

    std::cout << " boundary indicators: ";
    for (const std::pair<const types::boundary_id, unsigned int> &pair :
         boundary_count)
    {
      std::cout << pair.first << '(' << pair.second << " times) ";
    }
    std::cout << std::endl;
  }

  std::ofstream out(filename);
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << " written to " << filename << std::endl << std::endl;
}

template <int dim, int spacedim>
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<dim>                                  &param)
{
  MPI_Comm comm = triangulation.get_mpi_communicator();

  Triangulation<dim> serial_tria;

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(serial_tria);

  std::ifstream input(param.mesh.filename);
  AssertThrow(input,
              ExcMessage("Could not open mesh file: " + param.mesh.filename));
  grid_in.read_msh(input);

  // Partition serial triangulation:
  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     serial_tria);

  // Create building blocks:
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_tria, comm);

  // Create a fully distributed triangulation:
  // copy_triangulation does not seems to work, so maybe give reference to the
  // mesh
  triangulation.create_triangulation(description);

  // // Save initial position of the mesh vertices
  // initial_mesh_position.resize(triangulation.n_vertices());
  // for (auto &cell : dof_handler.active_cell_iterators())
  // {
  //   for (const auto v : cell->vertex_indices())
  //   {
  //     const unsigned int global_vertex_index     = cell->vertex_index(v);
  //     initial_mesh_position[global_vertex_index] = cell->vertex(v);
  //   }
  // }

  read_gmsh_physical_names(param.mesh.filename,
                           param.mesh.id2name,
                           param.mesh.name2id);

  check_boundary_conditions_compatibility(param);

  // // Print mesh info
  // bool VERBOSE = true;
  // if (mpi_rank == 0)
  // {
  //   if (VERBOSE)
  //   {
  //     std::cout << "Mesh info:" << std::endl
  //               << " dimension: " << dim << std::endl
  //               << " no. of cells: " << serial_tria.n_active_cells()
  //               << std::endl;
  //   }

  //   std::map<types::boundary_id, unsigned int> boundary_count;
  //   for (const auto &face : serial_tria.active_face_iterators())
  //     if (face->at_boundary())
  //       boundary_count[face->boundary_id()]++;

  //   if (VERBOSE)
  //   {
  //     std::cout << " boundary indicators: ";
  //     for (const std::pair<const types::boundary_id, unsigned int> &pair :
  //          boundary_count)
  //     {
  //       std::cout << pair.first << '(' << pair.second << " times) ";
  //     }
  //     std::cout << std::endl;
  //   }

  //   // Check that all boundary indices found in the mesh
  //   // have a matching name, to make sure we're not forgetting
  //   // a boundary.
  //   for (const auto &[id, count] : boundary_count)
  //   {
  //     if (mesh_domains_tag2name.count(id) == 0)
  //       throw std::runtime_error("Deal.ii read a boundary entity with tag " +
  //                                std::to_string(id) +
  //                                " in the mesh, but no physical entity with "
  //                                "this tag was read from the mesh file.");
  //   }

  //   if (VERBOSE)
  //   {
  //     for (const auto &[id, name] : mesh_domains_tag2name)
  //       std::cout << "ID " << id << " -> " << name << "\n";
  //   }
  // }

  // std::vector<std::string> all_boundaries;
  // for (auto str : boundary_description.position_fixed_boundary_names)
  //   all_boundaries.push_back(str);
  // for (auto str : boundary_description.position_moving_boundary_names)
  //   all_boundaries.push_back(str);
  // for (auto str : boundary_description.strong_velocity_boundary_names)
  //   all_boundaries.push_back(str);
  // for (auto str : boundary_description.weak_velocity_boundary_names)
  //   all_boundaries.push_back(str);
  // for (auto str : boundary_description.noflux_velocity_boundary_names)
  //   all_boundaries.push_back(str);

  // // Check that specified boundaries exist
  // for (auto str : all_boundaries)
  // {
  //   if (mesh_domains_name2tag.count(str) == 0)
  //   {
  //     throw std::runtime_error("A boundary condition should be prescribed "
  //                              "on the boundary named \"" +
  //                              str +
  //                              "\", but no physical entity with this name "
  //                              "was read from the mesh file.");
  //   }
  // }

  // if (boundary_description.weak_velocity_boundary_names.size() > 1)
  //   throw std::runtime_error(
  //     "Only considering a single boundary for weak velocity BC for now.");

  // for (auto str : boundary_description.weak_velocity_boundary_names)
  // {
  //   weak_bc_boundary_id = mesh_domains_name2tag.at(str);
  // }
}

// template Triangulation<2> read_mesh<2>(const std::string &meshFile,
//                                        const MPI_Comm    &comm);
// template Triangulation<3> read_mesh<3>(const std::string &meshFile,
//                                        const MPI_Comm    &comm);

template void
read_mesh(parallel::DistributedTriangulationBase<2> &triangulation,
          ParameterReader<2>                        &param);

template void
read_mesh(parallel::DistributedTriangulationBase<3> &triangulation,
          ParameterReader<3>                        &param);