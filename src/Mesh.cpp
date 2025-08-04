
#include <Mesh.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/manifold_lib.h>
// #include <deal.II/grid/tria_accessor.h>
// #include <deal.II/grid/tria_iterator.h>



template <int dim>
Triangulation<dim> read_mesh(const std::string &meshFile,
                             const MPI_Comm &comm)
{
  Triangulation<dim> serial_tria;

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(serial_tria);
  std::ifstream input(meshFile);
  AssertThrow(input, ExcMessage("Could not open mesh file: " + meshFile));
  grid_in.read_msh(input);
   
  // Partition serial triangulation:
  GridTools::partition_triangulation(
    Utilities::MPI::n_mpi_processes(comm), serial_tria);
   
  // Create building blocks:
  const TriangulationDescription::Description<dim> description =
    TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_tria, comm);
       
  // Create a fully distributed triangulation:
  parallel::fullydistributed::Triangulation<dim> distr_tria(comm);
  distr_tria.create_triangulation(description);

  // // Optional: visualize partition
  // DoFHandler<dim> dof_handler(distr_tria);
  // FE_SimplexP<dim> fe(1);
  // dof_handler.distribute_dofs(fe);
  // MappingFE<dim> mapping(FE_SimplexP<dim>(1));

  // Vector<double> cell_data(distr_tria.n_active_cells());

  // for (const auto &cell : distr_tria.active_cell_iterators())
  // {
  //   cell_data[cell->active_cell_index()] = (double) cell->subdomain_id();
  // }

  // DataOut<dim> data_out;
  // data_out.attach_dof_handler(dof_handler);
  // data_out.add_data_vector(cell_data, "subdomain", DataOut<2>::type_cell_data);

  // data_out.build_patches(mapping);

  // data_out.write_vtu_with_pvtu_record(
  //   "./", "mesh", 0, MPI_COMM_WORLD, 2);

  Vector<float> subdomain(distr_tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = distr_tria.locally_owned_subdomain();

  DoFHandler<dim> dof_handler(distr_tria);
  FE_SimplexP<dim> fe(1);
  dof_handler.distribute_dofs(fe);
  MappingFE<dim> mapping(FE_SimplexP<dim>(1));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches(mapping);

  data_out.write_vtu_with_pvtu_record(
    "./", "meshPartition", 0, comm, 2);

  return distr_tria;
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

template Triangulation<2> read_mesh<2>(
  const std::string &meshFile,
  const MPI_Comm &comm);
template Triangulation<3> read_mesh<3>(
  const std::string &meshFile,
  const MPI_Comm &comm);
