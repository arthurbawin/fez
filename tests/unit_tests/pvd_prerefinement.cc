#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <post_processing_handler.h>

#include <filesystem>
#include <fstream>
#include <regex>

#include "../tests.h"

using Record = std::pair<double, std::string>;

std::vector<Record> read_records(const std::string &filename)
{
  std::ifstream input(filename);
  AssertThrow(input, ExcMessage("Missing PVD file: " + filename));
  const std::regex    entry("timestep=\"([^\"]+)\".*file=\"([^\"]+)\"");
  std::vector<Record> records;
  std::string         line;
  while (std::getline(input, line))
  {
    std::smatch match;
    if (std::regex_search(line, match, entry))
    {
      records.emplace_back(std::stod(match[1]), match[2]);
      AssertThrow(std::filesystem::exists(match[2].str()), ExcInternalError());
    }
  }
  return records;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();
  Parameters::BoundaryConditionsData bc;
  ParameterHandler                   prm;
  ParameterReader<2>                 param(bc);
  param.declare(prm);
  prm.enter_subsection("Time integration");
  prm.set("scheme", "BDF1");
  prm.leave_subsection();
  param.read(prm);
  param.time_integration.t_initial     = 0.;
  param.output.output_dir              = "./";
  param.output.output_prefix           = "volume";
  param.output.write_results           = true;
  param.output.vtu_output_frequency    = 1;
  param.output.skin.write_results      = true;
  param.output.skin.output_prefix      = "skin";
  param.output.skin.boundary_id        = 0;
  param.output.skin.output_frequency   = 1;
  param.finite_elements.mapping_degree = 1;

  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  FE_Q<2>       fe(1);
  DoFHandler<2> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<2>             mapping;
  ComponentOrderingHeat    ordering;
  PostProcessingHandler<2> output(
    ordering, param, tria, dofs, {{"temperature", 1}});
  TimeHandler                          time(param.time_integration);
  PostProcessingHandler<2>::PrefixData prefix;
  prefix.is_convergence_step = true;
  prefix.convergence_step    = 3;
  Vector<double> solution;
  const auto     write = [&]() {
    solution.reinit(dofs.n_dofs());
    output.output_fields(mapping, solution, time, prefix);
  };
  const auto refine = [&]() {
    tria.refine_global(1);
    dofs.distribute_dofs(fe);
    output.attach_triangulation_and_dof_handler(tria, dofs);
  };

  prefix.is_prerefinement_step = true;
  write();
  refine();
  prefix.prerefinement_step = 1;
  write();
  prefix.is_prerefinement_step = false;
  write();
  refine();
  time.current_time_iteration = 1;
  time.current_time           = 0.25;
  write();

  for (const std::string name : {"volume", "skin"})
  {
    const auto physical = read_records(name + "_convergence_step_3.pvd");
    AssertThrow(physical.size() == 2, ExcInternalError());
    AssertThrow(physical[0].second.find("prerefinement") == std::string::npos,
                ExcMessage("Prerefinement overwrote the physical PVD"));
    AssertThrow(physical[0].first == 0., ExcInternalError());
    AssertThrow(physical[1].first == 0.25, ExcInternalError());
    const auto initial =
      read_records(name + "_convergence_step_3_prerefinement.pvd");
    AssertThrow(initial.size() == 2, ExcInternalError());
    AssertThrow(initial[0].first == 0. && initial[1].first == 1.,
                ExcInternalError());
    AssertThrow(initial[1].second.find("prerefinement_step_1") !=
                  std::string::npos,
                ExcInternalError());
  }

  // A new run must not inherit prerefinement records from the previous one.
  output.clear();
  prefix.is_prerefinement_step = true;
  prefix.prerefinement_step    = 0;
  time.current_time_iteration  = 0;
  time.current_time            = 0.;
  write();
  prefix.is_prerefinement_step = false;
  write();
  for (const std::string name : {"volume", "skin"})
  {
    AssertThrow(read_records(name + "_convergence_step_3.pvd").size() == 1,
                ExcInternalError());
    AssertThrow(
      read_records(name + "_convergence_step_3_prerefinement.pvd").size() == 1,
      ExcInternalError());
  }
  deallog << "Physical and prerefinement PVD collections OK" << std::endl;
}
