#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_handler.h>

#include "../tests.h"

using PP = Parameters::PostProcessing;

void test_parameters()
{
  for (const std::string section :
       {"forces computation", "structure position", "slicing"})
  {
    ParameterHandler prm;
    PP               pp;
    pp.declare_parameters(prm);
    prm.enter_subsection("Postprocessing");
    prm.enter_subsection(section);
    prm.set("boundary ids", "3, 1");
    prm.leave_subsection();
    prm.leave_subsection();
    pp.read_parameters(prm);
    const auto &ids =
      section == "forces computation" ? pp.forces.boundary_ids :
      section == "structure position" ? pp.structure_position.boundary_ids :
                                        pp.slices.boundary_ids;
    AssertThrow(ids == std::vector<types::boundary_id>({1, 3}),
                ExcInternalError());
    prm.enter_subsection("Postprocessing");
    prm.enter_subsection(section);
    prm.set("boundary id", "4");
    prm.leave_subsection();
    prm.leave_subsection();
    pp.read_parameters(prm);
    AssertThrow(ids == std::vector<types::boundary_id>({4}),
                ExcInternalError());
    prm.enter_subsection("Postprocessing");
    prm.enter_subsection(section);
    prm.set("boundary ids", "1, 1");
    prm.leave_subsection();
    prm.leave_subsection();
    bool rejected = false;
    try
    {
      pp.read_parameters(prm);
    }
    catch (const ExceptionBase &)
    {
      rejected = true;
    }
    AssertThrow(rejected, ExcInternalError());
  }
}

template <int dim>
class Fields : public Function<dim>
{
public:
  Fields()
    : Function<dim>(3 * dim + 1)
  {}
  double value(const Point<dim> &p, const unsigned int c) const override
  {
    if (c == dim)
      return 2.;
    if (c > dim && c <= 2 * dim)
      return -double(c - dim);
    if (c > 2 * dim)
      return p[c - 2 * dim - 1];
    return 0.;
  }
};

template <int dim>
void test_boundaries()
{
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD);
  // Shift the domain so that slice indices must account for its origin.
  GridGenerator::hyper_cube(tria, 2., 4., true);
  tria.refine_global(1);
  FESystem<dim>   fe(FE_Q<dim>(1), 3 * dim + 1);
  DoFHandler<dim> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<dim>    mapping;
  QGauss<dim - 1>   quadrature(2);
  LA::ParVectorType owned, solution;
  owned.reinit(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
  solution.reinit(dofs.locally_owned_dofs(),
                  DoFTools::extract_locally_relevant_dofs(dofs),
                  MPI_COMM_WORLD);
  VectorTools::interpolate(mapping, dofs, Fields<dim>(), owned);
  solution = owned;

  Parameters::BoundaryConditionsData bc{};
  ParameterReader<dim>               param(bc);
  ParameterHandler                   prm;
  param.postprocessing.declare_parameters(prm);
  param.time_integration.declare_parameters(prm);
  param.postprocessing.read_parameters(prm);
  param.time_integration.read_parameters(prm);
  param.time_integration.n_time_intervals          = 1;
  param.time_integration.n_steady_adaptation_steps = 0;
  param.output.write_results                       = false;
  param.output.skin.write_results                  = false;
  param.physical_properties.fluids.resize(1);
  param.physical_properties.fluids[0].density           = 1.234;
  param.physical_properties.fluids[0].dynamic_viscosity = 1.;
  auto &pp                                              = param.postprocessing;
  for (auto *p : std::array<PP::PostProcessingFileBoundary *, 3>{
         {&pp.forces, &pp.structure_position, &pp.slices}})
  {
    p->enable        = true;
    p->write_results = false;
    p->verbosity     = Parameters::Verbosity::quiet;
    p->boundary_ids  = {0, 1};
    p->precision     = 12;
  }
  pp.slices.along_which_axis         = "y";
  pp.slices.n_slices                 = 2;
  pp.slices.compute_forces_on_slices = true;
  ComponentOrdering ordering;
  ordering.u_lower = 0;
  ordering.p_lower = dim;
  ordering.l_lower = dim + 1;
  ordering.x_lower = 2 * dim + 1;
  TimeHandler time(param.time_integration);

  for (const auto method : {PP::Forces::ComputationMethod::stress_vector,
                            PP::Forces::ComputationMethod::lagrange_multiplier})
  {
    pp.forces.method = method;
    PostProcessingHandler<dim> handler(ordering, param, tria, dofs, {});
    handler.output_fields(mapping, solution, time, {});
    // Repeating the computation also checks that face forces are reset.
    for (unsigned int step = 0; step < 2; ++step)
    {
      if (step == 0)
        handler.compute_forces(
          ordering, dofs, mapping, quadrature, solution, time);
      else
        handler.compute_forces(ordering,
                               dofs,
                               hp::MappingCollection<dim>(mapping),
                               hp::QCollection<dim - 1>(quadrature),
                               solution,
                               time);
      handler.compute_structure_mean_position(
        ordering, dofs, mapping, quadrature, solution, time);
    }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ostringstream forces, positions;
      handler.write_forces(forces);
      handler.write_structure_mean_position(positions);
      const double area = std::pow(2., dim - 1);
      for (const bool is_force : {true, false})
      {
        std::istringstream table(is_force ? forces.str() : positions.str());
        std::string        header;
        std::getline(table, header);
        AssertThrow(header.find("boundary") != std::string::npos,
                    ExcInternalError());
        for (unsigned int row = 0; row < 4; ++row)
        {
          double       t;
          unsigned int id;
          AssertThrow(bool(table >> t >> id), ExcInternalError());
          AssertThrow(id == row % 2, ExcInternalError());
          for (unsigned int d = 0; d < dim; ++d)
          {
            double actual;
            AssertThrow(bool(table >> actual), ExcInternalError());
            const double expected =
              !is_force ? (d == 0 ? 2. + 2. * id : 3.) :
              method == PP::Forces::ComputationMethod::lagrange_multiplier ?
                          1.234 * (d + 1.) * area :
              d == 0 ? (id == 0 ? -1. : 1.) * 2. * 1.234 * area :
                       0.;
            AssertThrow(std::abs(actual - expected) < 1e-10,
                        ExcInternalError());
          }
        }
        table >> std::ws;
        AssertThrow(table.eof(), ExcInternalError());
      }
    }
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << dim << "D boundary forces, slices and positions OK" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  test_parameters();
  test_boundaries<2>();
  test_boundaries<3>();
}
