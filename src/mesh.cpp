
#include <boundary_conditions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
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
  MPI_Comm comm =
    triangulation.get_mpi_communicator();

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
void create_cube(Triangulation<dim> &tria,
                 const unsigned int  refinements_per_direction,
                 const bool          convert_to_tets = false)
{
  const double corner_min =  0.;
  const double corner_max =  1.;

  GridGenerator::subdivided_hyper_cube(
    tria, refinements_per_direction, corner_min, corner_max, true);

  for (auto &cell : tria.active_cell_iterators())
    for (unsigned int f = 0; f < cell->n_faces(); ++f)
      if (cell->face(f)->at_boundary())
      {
        const auto   c   = cell->face(f)->center();
        const double tol = 1e-12;

        if (std::fabs(c[0] - corner_min) < tol)
          cell->face(f)->set_boundary_id(1); // x=0
        else if (std::fabs(c[0] - corner_max) < tol)
          cell->face(f)->set_boundary_id(2); // x=1
        else if (std::fabs(c[1] - corner_min) < tol)
          cell->face(f)->set_boundary_id(3); // y=0
        else if (std::fabs(c[1] - corner_max) < tol)
          cell->face(f)->set_boundary_id(4); // y=1
        else if (std::fabs(c[2] - corner_min) < tol)
          cell->face(f)->set_boundary_id(5); // z=0
        else if (std::fabs(c[2] - corner_max) < tol)
          cell->face(f)->set_boundary_id(6); // z=1
      }

  if (convert_to_tets)
    GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria);
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
    // Check that each boundary id appears in the fluid boundary conditions
    AssertThrow(
      // std::count_if(param.fluid_bc.begin(),
      //               param.fluid_bc.end(),
      //               [id](const auto &bc) { return bc.id == id; }) == 1,
      param.fluid_bc.find(id) != param.fluid_bc.end(),
      ExcMessage("In mesh file " + param.mesh.filename +
                 " :\n"
                 "No fluid boundary condition was assigned to boundary " +
                 std::to_string(id) + " (" + param.mesh.id2name.at(id) +
                 "). For now, all boundaries must be assigned a boundary "
                 "condition for all relevant fields."));

    if (param.physical_properties.n_pseudosolids > 0)
    {
      // Check that each boundary id appears in the pseudosolid boundary
      // conditions
      AssertThrow(
        // std::count_if(param.pseudosolid_bc.begin(),
        //               param.pseudosolid_bc.end(),
        //               [id](const auto &bc) { return bc.id == id; }) == 1,
        param.pseudosolid_bc.find(id) != param.pseudosolid_bc.end(),
        ExcMessage(
          "In mesh file " + param.mesh.filename +
          " :\n"
          "No pseudosolid boundary condition was assigned to boundary " +
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
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<dim>                                  &param)
{
  Triangulation<dim> serial_triangulation;
  if constexpr (dim == 3)
  {
    if (param.mms_param.enable)
    {
      // There seems to be a bug in deal.II when reading a transfinite cube mesh
      // file. Use deal.II's routines to create a subdivided cube mesh.
      const unsigned int refinement_level = pow(2, param.mms_param.current_step);
      const bool convert_to_tetrahedra = true;
      create_cube(serial_triangulation, refinement_level, convert_to_tetrahedra);
      partition_and_create_parallel_mesh(serial_triangulation, triangulation);
      return;
    }
  }

  read_gmsh_mesh(serial_triangulation, param.mesh.filename);
  partition_and_create_parallel_mesh(serial_triangulation, triangulation);

  // Manually read the Gmsh entity names and store them
  read_gmsh_physical_names(param.mesh.filename,
                           param.mesh.id2name,
                           param.mesh.name2id);

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