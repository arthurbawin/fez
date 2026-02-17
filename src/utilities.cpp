
#include <deal.II/base/quadrature_lib.h>
#include <mpi.h>
#include <utilities.h>

#include <filesystem>

template <int dim>
void create_quadrature_rules(
  const Parameters::FiniteElements<dim> &fe_param,
  std::shared_ptr<Quadrature<dim>>      &quadrature,
  std::shared_ptr<Quadrature<dim - 1>>  &face_quadrature,
  std::shared_ptr<Quadrature<dim>>      &error_quadrature,
  std::shared_ptr<Quadrature<dim - 1>>  &error_face_quadrature)
{
  using RuleType =
    typename Parameters::FiniteElements<dim>::QuadratureRule::Type;

  // Rules for quads are hardcoded to n_points_1d = 4 for now.
  if (fe_param.use_quads)
  {
    quadrature            = std::make_shared<QGauss<dim>>(4);
    error_quadrature      = std::make_shared<QGauss<dim>>(4);
    face_quadrature       = std::make_shared<QGauss<dim - 1>>(4);
    error_face_quadrature = std::make_shared<QGauss<dim - 1>>(4);
    return;
  }

  const auto make_simplex_quadrature = [](auto  rule_type,
                                          auto  n_cell,
                                          auto  n_face,
                                          auto &cell_quad,
                                          auto &face_quad) {
    switch (rule_type)
    {
      case RuleType::GaussSimplex:
        cell_quad = std::make_shared<QGaussSimplex<dim>>(n_cell);
        face_quad = std::make_shared<QGaussSimplex<dim - 1>>(n_face);
        break;
      case RuleType::WitherdenVincent:
        cell_quad = std::make_shared<QWitherdenVincentSimplex<dim>>(n_cell);
        face_quad = std::make_shared<QWitherdenVincentSimplex<dim - 1>>(n_face);
        break;
      default:
        AssertThrow(false,
                    ExcMessage("Could not assign a valid quadrature rule!"));
    }
  };

  make_simplex_quadrature(fe_param.rule.type,
                          fe_param.rule.n_pts_1D_simplex_cell_quad,
                          fe_param.rule.n_pts_1D_simplex_face_quad,
                          quadrature,
                          face_quadrature);

  make_simplex_quadrature(fe_param.rule_for_error.type,
                          fe_param.rule_for_error.n_pts_1D_simplex_cell_quad,
                          fe_param.rule_for_error.n_pts_1D_simplex_face_quad,
                          error_quadrature,
                          error_face_quadrature);
}

template void create_quadrature_rules(const Parameters::FiniteElements<3> &,
                                      std::shared_ptr<Quadrature<3>> &,
                                      std::shared_ptr<Quadrature<2>> &,
                                      std::shared_ptr<Quadrature<3>> &,
                                      std::shared_ptr<Quadrature<2>> &);
template void create_quadrature_rules(const Parameters::FiniteElements<2> &,
                                      std::shared_ptr<Quadrature<2>> &,
                                      std::shared_ptr<Quadrature<1>> &,
                                      std::shared_ptr<Quadrature<2>> &,
                                      std::shared_ptr<Quadrature<1>> &);

void replace_temporary_files(const std::string directory,
                             const std::string temporary_filename_prefix,
                             const std::string final_filename_prefix,
                             const MPI_Comm   &mpi_communicator)
{
  // Create a shared-memory communicator (one per node)
  MPI_Comm shm_comm;
  MPI_Comm_split_type(
    mpi_communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);

  int shm_rank;
  MPI_Comm_rank(shm_comm, &shm_rank);

  // Only one rank per shared-memory node touches the filesystem
  if (shm_rank == 0)
  {
    std::list<std::string> tmp_checkpoint_files;
    for (const auto &dir_entry : std::filesystem::directory_iterator(directory))
      if (dir_entry.is_regular_file() &&
          (dir_entry.path().filename().string().find(
             temporary_filename_prefix) == 0))
        tmp_checkpoint_files.push_back(dir_entry.path().filename().string());

    for (const std::string &filename : tmp_checkpoint_files)
    {
      const std::string final_filename =
        directory + final_filename_prefix +
        filename.substr(temporary_filename_prefix.size(), std::string::npos);
      std::filesystem::rename(directory + filename, final_filename);
    }
  }

  // Ensure all ranks see the renamed files
  MPI_Barrier(mpi_communicator);
  MPI_Comm_free(&shm_comm);
}

template <int dim>
void fill_dofs_to_component(const DoFHandler<dim>      &dof_handler,
                            const IndexSet             &locally_relevant_dofs,
                            std::vector<unsigned char> &dofs_to_component)
{
  /**
   * Note that non-local dofs may have been added to locally_relevant_dofs
   * (e.g., to add a mean pressure constraint). Because dofs_to_component is
   * filled by looping over relevant cells, and since these non-local dofs do
   * not belong to ghost cells, dofs_to_component will have the default value at
   * these dofs.
   */
  const unsigned int n_relevant_dofs = locally_relevant_dofs.n_elements();
  dofs_to_component.resize(n_relevant_dofs, static_cast<unsigned char>(-1));
  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    dof_indices.resize(cell->get_fe().n_dofs_per_cell());
    cell->get_dof_indices(dof_indices);

    for (unsigned int i = 0; i < dof_indices.size(); ++i)
    {
      const types::global_dof_index dof = dof_indices[i];
      AssertThrow(locally_relevant_dofs.is_element(dof), ExcInternalError());
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)] =
        cell->get_fe().system_to_component_index(i).first;
    }
  }
}

template void fill_dofs_to_component(const DoFHandler<2> &,
                                     const IndexSet &,
                                     std::vector<unsigned char> &);
template void fill_dofs_to_component(const DoFHandler<3> &,
                                     const IndexSet &,
                                     std::vector<unsigned char> &);
