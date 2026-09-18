#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <mesh_adaptation_tools.h>

#include "../tests.h"

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize        mpi(argc, argv, 1);
  const auto                              comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<2> tria(comm);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);
  Parameters::Mesh::Adaptation::TreeAMR param{};
  param.fraction_to_refine  = .25;
  param.fraction_to_coarsen = .25;
  param.max_n_cells         = 1000;
  param.min_level           = 0;
  param.max_level           = 8;
  std::vector<Vector<float>> fields(2, Vector<float>(tria.n_active_cells()));
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const auto i = cell->active_cell_index();
      const auto p = cell->center();
      fields[0][i] = 1 + static_cast<unsigned int>(p[0] * 4) +
                     4 * static_cast<unsigned int>(p[1] * 4);
      fields[1][i] = 17 - fields[0][i];
    }
  using Strategy = Parameters::Mesh::Adaptation::TreeAMR::RefinementStrategy;
  for (unsigned int pattern = 0; pattern < 3; ++pattern)
  {
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const auto i = cell->active_cell_index();
        fields[1][i] =
          pattern == 0 ?
            17 - fields[0][i] :
          pattern == 1 ?
            fields[0][i] :
            1 + (static_cast<unsigned int>(std::round(fields[0][i])) + 1) % 16;
      }
    for (const auto strategy : {Strategy::FixedNumber, Strategy::FixedFraction})
    {
      param.refinement_strategy = strategy;
      std::vector<unsigned int> votes(tria.n_active_cells(), 0);
      std::vector<bool>         coarsen(tria.n_active_cells(), true);
      for (const auto &field : fields)
      {
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            cell->clear_refine_flag();
            cell->clear_coarsen_flag();
          }
        if (strategy == Strategy::FixedNumber)
          parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_number(tria, field, .25, .25);
        else
          parallel::distributed::GridRefinement::
            refine_and_coarsen_fixed_fraction(tria, field, .25, .25);
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            const auto i = cell->active_cell_index();
            votes[i] +=
              cell->refine_flag_set() != RefinementCase<2>::no_refinement;
            coarsen[i] = coarsen[i] && cell->coarsen_flag_set();
          }
      }
      const auto check = [&]() {
        MeshTools::mark_multifield_adaptation(tria, fields, param);
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
          {
            const auto i = cell->active_cell_index();
            AssertThrow((cell->refine_flag_set() !=
                         RefinementCase<2>::no_refinement) == (votes[i] > 0),
                        ExcMessage("Refinement must be the union of "
                                   "independent selections: pattern=" +
                                   std::to_string(pattern) + " strategy=" +
                                   std::to_string(static_cast<int>(strategy)) +
                                   " fields=" + std::to_string(fields.size()) +
                                   " cell=" + cell->id().to_string() +
                                   " votes=" + std::to_string(votes[i]) +
                                   " actual=" +
                                   std::to_string(cell->refine_flag_set())));
            AssertThrow(cell->coarsen_flag_set() ==
                          (coarsen[i] && votes[i] == 0),
                        ExcMessage("Coarsening must be the intersection"));
          }
      };
      check();
      std::swap(fields[0], fields[1]);
      fields[0] *= std::ldexp(1., -40);
      fields[1] *= std::ldexp(1., 40);
      fields.emplace_back(tria.n_active_cells());
      check(); // Scaling, order and an inactive field must not change
               // decisions.
      fields.pop_back();
      fields[0] *= std::ldexp(1., 40);
      fields[1] *= std::ldexp(1., -40);
      std::swap(fields[0], fields[1]);

      param.max_n_cells = tria.n_global_active_cells();
      MeshTools::mark_multifield_adaptation(tria, fields, param);
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          AssertThrow(!cell->refine_flag_set() &&
                        (!cell->coarsen_flag_set() ||
                         coarsen[cell->active_cell_index()]),
                      ExcMessage("Budget must not force extra coarsening"));
      param.max_n_cells = tria.n_global_active_cells() + 6;
      MeshTools::mark_multifield_adaptation(tria, fields, param);
      unsigned int selected = 0, lowest_selected_vote = 3,
                   highest_rejected_vote = 0;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const auto i = cell->active_cell_index();
          if (cell->refine_flag_set())
          {
            ++selected;
            lowest_selected_vote = std::min(lowest_selected_vote, votes[i]);
          }
          else
            highest_rejected_vote = std::max(highest_rejected_vote, votes[i]);
        }
      AssertThrow(
        Utilities::MPI::sum(selected, comm) == 2,
        ExcMessage(
          "Limited budget must allow exactly two refinement requests"));
      AssertThrow(Utilities::MPI::min(lowest_selected_vote, comm) >=
                    Utilities::MPI::max(highest_rejected_vote, comm),
                  ExcMessage(
                    "Budget must prioritize cells requested by more fields"));
      param.max_n_cells = 1000;
    }
  }
  for (auto &field : fields)
    field = 0.;
  MeshTools::mark_multifield_adaptation(tria, fields, param);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      AssertThrow(!cell->refine_flag_set() && !cell->coarsen_flag_set(),
                  ExcMessage("All-zero fields must leave the mesh unchanged"));
  // Some ranks have no nonzero indicator: skipping a field must be collective.
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned() && Utilities::MPI::this_mpi_process(comm) == 0)
      fields[0][cell->active_cell_index()] = 1 + cell->active_cell_index();
  MeshTools::mark_multifield_adaptation(tria, fields, param);
  param.min_level = 2;
  param.max_level = 2;
  MeshTools::mark_multifield_adaptation(tria, fields, param);
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
      AssertThrow(!cell->refine_flag_set() && !cell->coarsen_flag_set(),
                  ExcMessage("Level bounds must apply to the combined flags"));
  param.min_level           = 0;
  param.max_level           = 8;
  param.fraction_to_coarsen = 0.;
  param.refinement_strategy = Strategy::FixedNumber;
  for (const auto &cell : tria.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      const auto p = cell->center();
      const auto i = cell->active_cell_index();
      fields[0][i] = 1 + static_cast<unsigned int>(p[0] * 4) +
                     4 * static_cast<unsigned int>(p[1] * 4);
      fields[1][i] = 17 - fields[0][i];
    }
  const auto original_cells = tria.n_global_active_cells();
  param.max_n_cells         = original_cells + 6;
  MeshTools::mark_multifield_adaptation(tria, fields, param);
  tria.execute_coarsening_and_refinement();
  AssertThrow(tria.n_global_active_cells() == original_cells + 6,
              ExcMessage(
                "The two budgeted requests must actually refine two cells"));
  initlog();
  deallog << "Independent multifield AMR selections OK" << std::endl;
}
