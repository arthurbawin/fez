#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_handler.h>

#include <fstream>
#include <sstream>

#include "../tests.h"

// Integrate a scalar and a vector with different components on [0, 2]^dim.
// Nonconstant fields and MPI partitions exercise quadrature and global sums.
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
  // Exercise parameter parsing, separate scalar/vector tables, output
  // frequency, and the final write when the last step is off-frequency.
  Parameters::BoundaryConditionsData boundary_conditions;
  ParameterReader<dim>               parameters(boundary_conditions);
  ParameterHandler                   prm;
  parameters.declare(prm);
  prm.parse_input_from_string("subsection Postprocessing\n"
                              "  subsection field integral\n"
                              "    set enable = true\n"
                              "    set variables = temperature, velocity\n"
                              "    set output frequency = 2\n"
                              "  end\n"
                              "end\n");
  parameters.read(prm);
  parameters.output.write_results   = false;
  parameters.output.output_dir      = "";
  auto &integral_parameters         = parameters.postprocessing.field_integral;
  integral_parameters.output_prefix = "integral_" + std::to_string(dim) + "d";
  auto &time_parameters             = parameters.time_integration;
  time_parameters.scheme            = Parameters::TimeIntegration::Scheme::BDF1;
  time_parameters.t_initial         = 0.;
  time_parameters.t_end             = 3.;
  time_parameters.dt                = 1.;
  TimeHandler       time_handler(time_parameters);
  ComponentOrdering ordering;
  ordering.n_components = dim + 1;
  ordering.u_lower      = 0;
  ordering.u_upper      = dim;
  ordering.t_lower      = dim;
  ordering.t_upper      = dim + 1;
  PostProcessingHandler<dim> handler(
    ordering, parameters, triangulation, dof_handler, {});
  for (unsigned int step = 0; step <= 3; ++step)
  {
    time_handler.current_time_iteration = step;
    time_handler.current_time           = step;
    handler.compute_field_integrals(mapping,
                                    quadrature,
                                    solution,
                                    time_handler);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      for (const auto &variable : {"temperature", "velocity"})
      {
        std::ifstream file(integral_parameters.output_prefix + "_" + variable +
                           ".txt");
        AssertThrow(file.is_open(), ExcInternalError());
        std::string line;
        std::getline(file, line);
        const bool scalar_field = std::string(variable) == "temperature";
        AssertThrow(line.find(scalar_field ? "temperature" : "velocity_x") !=
                      std::string::npos,
                    ExcInternalError());
        unsigned int rows = 0;
        while (std::getline(file, line))
        {
          std::istringstream row(line);
          double             time, value;
          AssertThrow(bool(row >> time) && time == rows, ExcInternalError());
          for (unsigned int d = 0; d < (scalar_field ? 1 : dim); ++d)
          {
            const double expected = (scalar_field ? 3. : d + 1.) * volume;
            AssertThrow(bool(row >> value) &&
                          std::abs(value - expected) < 1e-12,
                        ExcInternalError());
          }
          ++rows;
        }
        AssertThrow(rows == (step == 1 ? 1 : step + 1), ExcInternalError());
      }
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << dim << "D scalar/vector integrals and output frequency OK"
            << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test_integrals<2>();
  test_integrals<3>();
}
