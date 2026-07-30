#include <deal.II/base/tensor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include "post_processing_tools.h"

class SolidBodyRotation : public Function<3>
{
public:
  SolidBodyRotation()
    : Function<3>(4)
  {}

  void vector_value(const Point<3> &point,
                    Vector<double> &values) const override
  {
    values[0] = -point[1];
    values[1] = point[0];
    values[2] = 0.;
    values[3] = 0.;
  }
};

int main(int argc, char *argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    Tensor<2, 3> grad_u;
    grad_u[0][0] = 1.;
    grad_u[0][1] = 2.;
    grad_u[0][2] = 3.;
    grad_u[1][0] = 4.;
    grad_u[1][1] = 5.;
    grad_u[1][2] = 6.;
    grad_u[2][0] = 7.;
    grad_u[2][1] = 8.;
    grad_u[2][2] = 9.;

    const Tensor<1, 3> vorticity =
      PostProcessingTools::compute_vorticity_from_velocity_gradient<3>(grad_u);
    const double qcriterion =
      PostProcessingTools::compute_qcriterion_from_velocity_gradient<3>(grad_u);

    deallog << "vorticity " << vorticity << std::endl;
    deallog << "Qcriterion " << qcriterion << std::endl;

    Triangulation<3> serial_triangulation;
    GridGenerator::subdivided_hyper_cube(serial_triangulation, 2);
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
      serial_triangulation);
    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_triangulation,
                                            MPI_COMM_WORLD);
    parallel::fullydistributed::Triangulation<3> triangulation(MPI_COMM_WORLD);
    triangulation.create_triangulation(description);

    FESystem<3>  fe(FE_Q<3>(2), 3, FE_Q<3>(1), 1);
    DoFHandler<3> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    LA::ParVectorType solution;
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    MPI_COMM_WORLD);
    const MappingQ1<3> mapping;
    LA::ParVectorType owned_solution;
    owned_solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    VectorTools::interpolate(
      mapping,
      dof_handler,
      SolidBodyRotation(),
      owned_solution);
    owned_solution.compress(VectorOperation::insert);
    solution = owned_solution;
    solution.update_ghost_values();

    LA::ParVectorType nodal_vorticity;
    LA::ParVectorType nodal_qcriterion;
    PostProcessingTools::compute_nodal_flow_diagnostics<3>(
      dof_handler,
      mapping,
      solution,
      fe,
      FEValuesExtractors::Vector(0),
      true,
      true,
      nodal_vorticity,
      nodal_qcriterion);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        std::vector<types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        cell->get_dof_indices(dof_indices);
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
          const auto [component, shape] = fe.system_to_component_index(i);
          (void)shape;
          const auto dof = dof_indices[i];
          if (!locally_owned_dofs.is_element(dof) || component >= 3)
            continue;

          const double expected_vorticity = component == 2 ? 2. : 0.;
          AssertThrow(std::abs(nodal_vorticity[dof] - expected_vorticity) <
                        1e-12,
                      ExcInternalError());
          if (component == 0)
            AssertThrow(std::abs(nodal_qcriterion[dof] - 1.) < 1e-12,
                        ExcInternalError());
        }
      }

    deallog << "nodal diagnostics solid rotation OK" << std::endl;
  }
  catch (const std::exception &exc)
  {
    std::cerr << exc.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    return 1;
  }

  return 0;
}
