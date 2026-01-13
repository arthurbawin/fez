
#include <boundary_conditions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
#include <grid_generator_simplex.h>
#include <mesh.h>
#include <parameter_reader.h>
#include <parameters.h>

/**
 * Read sequential mesh from Gmsh file.
 */
template <int dim>
void read_gmsh_mesh(Triangulation<dim> &serial_triangulation,
                    const std::string  &mesh_file)
{
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(serial_triangulation);

  std::ifstream input(mesh_file);
  AssertThrow(input, ExcMessage("Could not open mesh file: " + mesh_file));
  grid_in.read_msh(input);
}

/**
 * FIXME: the whole mesh is first read on all processes, then partitioned
 * and distributed. This won't work for really big meshes.
 */
template <int dim, int spacedim>
void partition_and_create_parallel_mesh(
  Triangulation<dim>                                    &serial_triangulation,
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation)
{
  MPI_Comm comm = triangulation.get_mpi_communicator();

  // Partition serial triangulation:
  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     serial_triangulation);

  // Create building blocks:
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      serial_triangulation, comm);

  // Create a fully distributed triangulation:
  // copy_triangulation does not seems to work, so maybe give reference to the
  // mesh
  triangulation.create_triangulation(description);
}

/**
 * Deal.II does not read the names of the physical entities (I think).
 * Read them here and assign them to their boundary id.
 */
void read_gmsh_physical_names(const std::string                   &meshFile,
                              std::map<unsigned int, std::string> &tag2name,
                              std::map<std::string, unsigned int> &name2tag)
{
  std::ifstream in(meshFile);
  AssertThrow(in, ExcMessage("Could not open file " + meshFile));

  std::string line;
  while (std::getline(in, line))
  {
    // Trim the possible trailing whitespaces
    line.erase(std::find_if(line.rbegin(),
                            line.rend(),
                            [](unsigned char ch) { return !std::isspace(ch); })
                 .base(),
               line.end());

    if (line == "$PhysicalNames")
    {
      // Next line contains the number of entries
      unsigned int num;
      AssertThrow(in >> num,
                  ExcMessage("Invalid $PhysicalNames section in " + meshFile));

      // Each of the next 'num' lines: <dim> <id> "<name>"
      for (unsigned int i = 0; i < num; ++i)
      {
        unsigned int dim, id;
        std::string  name;
        in >> dim >> id;
        in >> std::ws; // skip spaces before name
        std::getline(in, name);

        // Trim the possible trailing whitespaces
        name.erase(std::find_if(name.rbegin(),
                                name.rend(),
                                [](unsigned char ch) {
                                  return !std::isspace(ch);
                                })
                     .base(),
                   name.end());

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
void create_cube(Triangulation<dim> &tria,
                 Parameters::Mesh   &mesh_param,
                 const double        min_corner,
                 const double        max_corner,
                 const unsigned int  refinements_per_direction,
                 const bool          convert_to_tets = false)
{
  GridGenerator::subdivided_hyper_cube(
    tria, refinements_per_direction, min_corner, max_corner, true);

  for (auto &cell : tria.active_cell_iterators())
    for (unsigned int f = 0; f < cell->n_faces(); ++f)
      if (cell->face(f)->at_boundary())
      {
        const auto   c   = cell->face(f)->center();
        const double tol = 1e-12;

        if (std::fabs(c[0] - min_corner) < tol)
          cell->face(f)->set_boundary_id(1); // x=0
        else if (std::fabs(c[0] - max_corner) < tol)
          cell->face(f)->set_boundary_id(2); // x=1
        else if (std::fabs(c[1] - min_corner) < tol)
          cell->face(f)->set_boundary_id(3); // y=0
        else if (std::fabs(c[1] - max_corner) < tol)
          cell->face(f)->set_boundary_id(4); // y=1

        if constexpr (dim == 3)
        {
          if (std::fabs(c[2] - min_corner) < tol)
            cell->face(f)->set_boundary_id(5); // z=0
          if (std::fabs(c[2] - max_corner) < tol)
            cell->face(f)->set_boundary_id(6); // z=1
        }
      }

  mesh_param.id2name.insert({1, "x_min"});
  mesh_param.id2name.insert({2, "x_max"});
  mesh_param.id2name.insert({3, "y_min"});
  mesh_param.id2name.insert({4, "y_max"});
  mesh_param.name2id.insert({"x_min", 1});
  mesh_param.name2id.insert({"x_max", 2});
  mesh_param.name2id.insert({"y_min", 3});
  mesh_param.name2id.insert({"y_max", 4});
  if constexpr (dim == 3)
  {
    mesh_param.id2name.insert({5, "z_min"});
    mesh_param.id2name.insert({6, "z_max"});
    mesh_param.name2id.insert({"z_min", 5});
    mesh_param.name2id.insert({"z_max", 6});
  }

  if (convert_to_tets)
  {
    const unsigned int n_divisions = (dim == 2) ? 2u : 6u;
    GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria, n_divisions);
  }
}

template <int dim>
void create_rectangle(Triangulation<dim> &tria,
                      Parameters::Mesh   &mesh_param,
                      const std::string  &parameters,
                      const unsigned int  refinement,
                      const bool          convert_to_tets = false)
{
  // Parse the "repetitions" part of the parameters and
  // multiply by the global refinement (for e.g. convergence studies)
  auto blocks      = Utilities::split_string_list(parameters, ':');
  auto repetitions = Utilities::split_string_list(blocks[0], ',');

  AssertThrow(blocks.size() > 1,
              ExcMessage(
                "The parsed arguments to create a rectangle/box mesh do not "
                "contain any \":\" separator. Please separate the arguments by "
                "a colon. The parsed parameters are : " +
                parameters));
  AssertThrow(
    repetitions.size() > 1,
    ExcMessage(
      "The parsed arguments to create a rectangle/box mesh should start by a "
      "comma-separated list of refinement per direction (e.g., nx, ny [, nz] : "
      "[remaining parameters]), but the given parameters are not separated by "
      "commas. The parsed parameters are : " +
      parameters));

  std::string updated_param = "";
  for (unsigned int d = 0; d < dim; ++d)
  {
    updated_param += std::to_string(std::stoi(repetitions[d]) * refinement);
    if (d < dim - 1)
      updated_param += ", ";
  }
  for (unsigned int i = 1; i < blocks.size(); ++i)
    updated_param += " : " + blocks[i];

  GridGenerator::generate_from_name_and_arguments(tria,
                                                  "subdivided_hyper_rectangle",
                                                  updated_param);

  // Use the boundary pattern obtained with "colorize = true" for
  // subdivided_hyper_rectangle.
  mesh_param.id2name.insert({0, "x_min"});
  mesh_param.id2name.insert({1, "x_max"});
  mesh_param.id2name.insert({2, "y_min"});
  mesh_param.id2name.insert({3, "y_max"});
  mesh_param.name2id.insert({"x_min", 0});
  mesh_param.name2id.insert({"x_max", 1});
  mesh_param.name2id.insert({"y_min", 2});
  mesh_param.name2id.insert({"y_max", 3});
  if constexpr (dim == 3)
  {
    mesh_param.id2name.insert({4, "z_min"});
    mesh_param.id2name.insert({5, "z_max"});
    mesh_param.name2id.insert({"z_min", 4});
    mesh_param.name2id.insert({"z_max", 5});
  }

  if (convert_to_tets)
  {
    const unsigned int n_divisions = (dim == 2) ? 2u : 6u;
    GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria, n_divisions);
  }

  GridOut grid_out;
  grid_out.write_msh(tria, "tria.msh");
}

template <int dim>
void create_holed_plate(Triangulation<dim> &tria,
                        Parameters::Mesh   &mesh_param,
                        const unsigned int  refinement_level,
                        const bool          convert_to_tets = false)
{
  GridGenerator::plate_with_a_hole(tria,
                                   0.15,
                                   0.25,
                                   0.25,
                                   0.25,
                                   0.25,
                                   0.25,
                                   Point<dim>(0.5, 0.5),
                                   0,
                                   1,
                                   1.,
                                   4,
                                   true);

  tria.refine_global(refinement_level);

  // Use the boundary pattern obtained with "colorize = true"
  mesh_param.id2name.insert({0, "x_min"});
  mesh_param.id2name.insert({1, "x_max"});
  mesh_param.id2name.insert({2, "y_min"});
  mesh_param.id2name.insert({3, "y_max"});
  mesh_param.id2name.insert({4, "hole"});
  mesh_param.name2id.insert({"x_min", 0});
  mesh_param.name2id.insert({"x_max", 1});
  mesh_param.name2id.insert({"y_min", 2});
  mesh_param.name2id.insert({"y_max", 3});
  mesh_param.name2id.insert({"hole", 4});
  if constexpr (dim == 3)
  {
    mesh_param.id2name.insert({5, "z_min"});
    mesh_param.id2name.insert({6, "z_max"});
    mesh_param.name2id.insert({"z_min", 5});
    mesh_param.name2id.insert({"z_max", 6});
  }

  if (convert_to_tets)
  {
    const unsigned int n_divisions = (dim == 2) ? 2u : 6u;
    GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria, n_divisions);
  }

  GridOut grid_out;
  grid_out.write_msh(tria, "tria.msh");
}


/**
 * Perform checks on the mesh boundary ids and entities.
 */
template <int dim>
void check_boundary_ids(Triangulation<dim>         &serial_triangulation,
                        const ParameterReader<dim> &param)
{
  std::map<types::boundary_id, unsigned int> boundary_count;
  for (const auto &face : serial_triangulation.active_face_iterators())
    if (face->at_boundary())
      boundary_count[face->boundary_id()]++;

  /**
   * TODO: In some meshes, there remains boundary geometric entities which are
   * not part of any physical entity, which are read by deal.II with boundary id
   * 0. Filtering them out of the mesh can be very tedious, here they are simply
   * set as unused, but associated boundary conditions are still required in the
   * parameter file. This should be treated here.
   */
  // for (const auto &[id, count] : boundary_count)
  //   if(id == 0 && param.mesh.id2name.count(id) == 0)
  //   {
  //     param.mesh.id2name[id] = "BOUNDARY_IS_UNUSED";
  //     param.mesh.name2id["BOUNDARY_IS_UNUSED"] = id;
  //   }

  /**
   * Check that all boundary ids found in the mesh have a matching name in the
   * mesh file. That is, all boundaries must be part of a named Gmsh
   * Physical Entity.
   */
  for (const auto &[id, count] : boundary_count)
    AssertThrow(
      param.mesh.id2name.count(id) == 1,
      ExcMessage(
        "In mesh file " + param.mesh.filename +
        " :\n"
        "Deal.ii read a boundary entity with id " +
        std::to_string(id) +
        " in the mesh, but no named Physical Entity with this tag "
        "was read from the mesh file. This typically happens if there is a "
        "geometric entity (i.e., a boundary curve or surface) in the mesh that "
        "is not associated to any named Physical Entity. Make sure that all "
        "boundaries are part of a named group."));

  /**
   * Check that all boundary ids found in the mesh have a matching name in the
   * parameter file.
   * For now, all dim-1 dimensional boundary entity should have an assigned
   * boundary condition in the parameter file.
   */
  for (const auto &[id, count] : boundary_count)
  {
    if (param.fluid_bc.size() > 0)
    {
      // Check that each boundary id appears in the fluid boundary conditions
      AssertThrow(param.fluid_bc.find(id) != param.fluid_bc.end(),
                  ExcMessage(
                    "In mesh file " + param.mesh.filename +
                    " :\n"
                    "No fluid boundary condition was assigned to boundary " +
                    std::to_string(id) + " (" + param.mesh.id2name.at(id) +
                    "). For now, all boundaries must be assigned a boundary "
                    "condition for all relevant fields."));
    }

    if (param.pseudosolid_bc.size() > 0)
    {
      // Check that each boundary id appears in the pseudosolid boundary
      // conditions
      AssertThrow(
        param.pseudosolid_bc.find(id) != param.pseudosolid_bc.end(),
        ExcMessage(
          "In mesh file " + param.mesh.filename +
          " :\n"
          "No pseudosolid boundary condition was assigned to boundary " +
          std::to_string(id) + " (" + param.mesh.id2name.at(id) +
          "). For now, all boundaries must be assigned a boundary "
          "condition for all relevant fields."));
    }

    if (param.cahn_hilliard_bc.size() > 0)
    {
      // Check that each boundary id appears in the CH boundary conditions
      AssertThrow(
        param.cahn_hilliard_bc.find(id) != param.cahn_hilliard_bc.end(),
        ExcMessage(
          "In mesh file " + param.mesh.filename +
          " :\n"
          "No Cahn-Hilliard boundary condition was assigned to boundary " +
          std::to_string(id) + " (" + param.mesh.id2name.at(id) +
          "). For now, all boundaries must be assigned a boundary "
          "condition for all relevant fields."));
    }

    if (param.heat_bc.size() > 0)
    {
      // Check that each boundary id appears in the heat boundary conditions
      AssertThrow(param.heat_bc.find(id) != param.heat_bc.end(),
                  ExcMessage(
                    "In mesh file " + param.mesh.filename +
                    " :\n"
                    "No heat boundary condition was assigned to boundary " +
                    std::to_string(id) + " (" + param.mesh.id2name.at(id) +
                    "). For now, all boundaries must be assigned a boundary "
                    "condition for all relevant fields."));
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
                "In mesh file " + param.mesh.filename +
                " :\n"
                "A " +
                bc.physics_str +
                " boundary condition is prescribed on a mesh "
                "domain (Physical Entity) with id " +
                std::to_string(bc.id) +
                ", but this id either does not exist in the mesh or appears "
                "more than one time, which is not supported."));

  // Check that the boundary name exists
  AssertThrow(param.mesh.name2id.count(bc.gmsh_name) == 1,
              ExcMessage(
                "In mesh file " + param.mesh.filename +
                " :\n"
                "A " +
                bc.physics_str +
                " boundary condition is prescribed on a mesh domain (Physical "
                "Entity) named \"" +
                bc.gmsh_name +
                "\", but this entity either does not exist in the mesh or "
                "appears more than one time, which is not supported."
                " This entity appears exactly " +
                std::to_string(param.mesh.name2id.count(bc.gmsh_name)) +
                " times in the mesh."));

  // Check that the prescribed name and id match in the mesh
  AssertThrow(param.mesh.id2name.at(bc.id) == bc.gmsh_name,
              ExcMessage(
                "In mesh file " + param.mesh.filename +
                " :\n"
                "A " +
                bc.physics_str +
                " boundary condition is prescribed on entity \"" +
                bc.gmsh_name + "\" with id " + std::to_string(bc.id) +
                ", but this id does not match this entity in the "
                "mesh. Instead, the entity exists in the mesh with id " +
                std::to_string(param.mesh.name2id.at(bc.gmsh_name)) +
                " and id " + std::to_string(bc.id) + " exists with name \"" +
                param.mesh.id2name.at(bc.id) + "\"."));
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
  for (const auto &[id, bc] : param.fluid_bc)
    check_single_boundary_condition(bc, param);
  // Check pseudosolid conditions
  for (const auto &[id, bc] : param.pseudosolid_bc)
    check_single_boundary_condition(bc, param);
}

template <int dim, int spacedim>
void print_mesh_info(
  const Triangulation<dim> &serial_triangulation,
  const parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  const ParameterReader<dim>                                  &param)
{
  MPI_Comm           comm = triangulation.get_mpi_communicator();
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

  if (rank == 0 && param.mesh.verbosity == Parameters::Verbosity::verbose)
  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto &face : serial_triangulation.active_face_iterators())
      if (face->at_boundary())
        boundary_count[face->boundary_id()]++;

    std::cout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << serial_triangulation.n_active_cells()
              << std::endl;

    std::cout << " boundary indicators: ";
    for (const auto &[id, count] : boundary_count)
      std::cout << id << '(' << count << " faces) ";
    std::cout << std::endl;

    for (const auto &[id, name] : param.mesh.id2name)
      std::cout << "ID " << id << " -> " << name << "\n";
  }
}

template <int dim, int spacedim>
void write_partition_gmsh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  const ParameterReader<spacedim>                       &param)
{
  MPI_Comm           comm = triangulation.get_mpi_communicator();
  const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

  std::ofstream outfile(param.output.output_dir + "partitions_proc" +
                        std::to_string(rank) + ".pos");
  outfile << "View \"partitions_proc" + std::to_string(rank) + "\"{"
          << std::endl;

  for (const auto &cell : triangulation.active_cell_iterators())
  {
    const std::string id = std::to_string(cell->subdomain_id());
    outfile << ((dim == 2) ? "ST(" : "SS(");
    for (unsigned int v = 0; v < cell->n_vertices(); ++v)
    {
      const Point<spacedim> &p = cell->vertex(v);
      if constexpr (spacedim == 2)
        outfile << p[0] << "," << p[1] << ",0."
                << ((v == cell->n_vertices() - 1) ? "" : ",");
      else
        outfile << p[0] << "," << p[1] << "," << p[2]
                << ((v == cell->n_vertices() - 1) ? "" : ",");
    }
    if constexpr (dim == 2)
      outfile << "){" << id << "," << id << "," << id << "};" << std::endl;
    else
      outfile << "){" << id << "," << id << "," << id << "," << id << "};"
              << std::endl;
  }

  outfile << "};" << std::endl;
  outfile.close();
}

template <int dim, int spacedim>
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<spacedim>                             &param)
{
  Triangulation<dim> serial_triangulation;

  bool use_deal_ii_mesh =
    param.mesh.deal_ii_preset_mesh != "none" ||
    param.mesh.use_deal_ii_cube_mesh ||
    (param.mms_param.enable && param.mms_param.use_deal_ii_cube_mesh) ||
    (param.mms_param.enable && param.mms_param.use_deal_ii_holed_plate_mesh);

  if (use_deal_ii_mesh)
  {
    const bool convert_to_simplices = !param.finite_elements.use_quads;

    if (param.mesh.deal_ii_preset_mesh == "cube" ||
        param.mesh.use_deal_ii_cube_mesh ||
        param.mms_param.use_deal_ii_cube_mesh)
    {
      const double       min_corner = (dim == 2) ? 0. : 0.;
      const double       max_corner = 1.;
      const unsigned int refinement_level =
        param.mms_param.enable ? pow(2, param.mms_param.mesh_suffix + 1) :
                                 param.mesh.refinement_level;
      create_cube(serial_triangulation,
                  param.mesh,
                  min_corner,
                  max_corner,
                  refinement_level,
                  convert_to_simplices);
    }
    else if (param.mesh.deal_ii_preset_mesh == "rectangle")
    {
      const unsigned int refinement_level =
        param.mms_param.enable ? pow(2, param.mms_param.mesh_suffix) :
                                 param.mesh.refinement_level;
      create_rectangle(serial_triangulation,
                       param.mesh,
                       param.mesh.deal_ii_mesh_param,
                       refinement_level,
                       convert_to_simplices);
    }
    else if (param.mesh.deal_ii_preset_mesh == "holed plate" ||
             param.mms_param.use_deal_ii_holed_plate_mesh)
    {
      const unsigned int refinement_level = param.mms_param.enable ?
                                              param.mms_param.mesh_suffix :
                                              param.mesh.refinement_level;
      create_holed_plate(serial_triangulation,
                         param.mesh,
                         refinement_level,
                         convert_to_simplices);
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Mesh creation for deal.II preset geometry \"" +
                             param.mesh.deal_ii_preset_mesh +
                             "\" is not implemented."));
    }
  }
  else
  {
    // Read Gmsh .msh4 mesh file
    read_gmsh_mesh(serial_triangulation, param.mesh.filename);

    // Manually read the Gmsh entity names and store them
    read_gmsh_physical_names(param.mesh.filename,
                             param.mesh.id2name,
                             param.mesh.name2id);
  }

  partition_and_create_parallel_mesh(serial_triangulation, triangulation);
  if (param.debug.write_partition_pos_gmsh)
    write_partition_gmsh(triangulation, param);
  print_mesh_info(serial_triangulation, triangulation, param);
  check_boundary_ids(serial_triangulation, param);
  check_boundary_conditions_compatibility(param);
}

template void
read_mesh(parallel::DistributedTriangulationBase<2> &triangulation,
          ParameterReader<2>                        &param);

template void
read_mesh(parallel::DistributedTriangulationBase<3> &triangulation,
          ParameterReader<3>                        &param);