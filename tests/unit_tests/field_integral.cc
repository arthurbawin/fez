#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_tools.h>

#include "../tests.h"

// Integrate a scalar and a vector with different components on [0, 2]^dim.
// Nonconstant fields and MPI partitions test quadrature and global sums.
template <int dim>
class Fields : public Function<dim>
{
public:
  Fields()
    : Function<dim>(dim + 1)
  {}

  double value(const Point<dim>  &p,
               const unsigned int component = 0) const override
  {
    return component == dim ? 3. * p[0] : (component + 1.) * p[component];
  }
};

template <int dim>
void test_integrals()
{
  parallel::shared::Triangulation<dim> triangulation(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(triangulation, 0., 2.);
  triangulation.refine_global(1);
  FESystem<dim>   fe(FE_Q<dim>(1), dim + 1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  MappingQ1<dim> mapping;
  QGauss<dim>    quadrature(2);

  const auto owned    = dof_handler.locally_owned_dofs();
  const auto relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  LA::ParVectorType local_solution, solution;
  local_solution.reinit(owned, MPI_COMM_WORLD);
  solution.reinit(owned, relevant, MPI_COMM_WORLD);
  VectorTools::interpolate(mapping, dof_handler, Fields<dim>(), local_solution);
  solution = local_solution;

  const auto scalar =
    PostProcessingTools::compute_field_integral(dof_handler,
                                                mapping,
                                                quadrature,
                                                solution,
                                                FEValuesExtractors::Scalar(
                                                  dim));
  const auto vector = PostProcessingTools::compute_field_integral(
    dof_handler, mapping, quadrature, solution, FEValuesExtractors::Vector(0));
  const double volume = std::pow(2., dim);
  AssertThrow(std::abs(scalar - 3. * volume) < 1e-12, ExcInternalError());
  for (unsigned int d = 0; d < dim; ++d)
    AssertThrow(std::abs(vector[d] - (d + 1.) * volume) < 1e-12,
                ExcInternalError());
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << dim << "D scalar/vector integrals OK" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test_integrals<2>();
  test_integrals<3>();
}
