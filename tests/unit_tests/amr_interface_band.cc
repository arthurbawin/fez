#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <heat_solver.h>
#include <incompressible_chns_solver.h>
#include <mesh_adaptation_tools.h>

#include "../tests.h"

class BandState : public Function<2>
{
public:
  BandState(const unsigned int components,
            const double       scale,
            const double       offset = .43,
            const bool         circle = false)
    : Function<2>(components)
    , scale(scale)
    , offset(offset)
    , circle(circle)
  {}

  double value(const Point<2> &p, const unsigned int c) const override
  {
    if (c == 3 || c == 4)
      return scale * p[c - 3];
    if (c == 5)
      return circle ? (p - Point<2>(.5, .5)).norm_square() - .0625 :
                      p[0] - offset;
    return 0.;
  }

private:
  const double scale, offset;
  const bool   circle;
};

class BandSolver : public CHNSSolver<2, true, false>
{
public:
  using CHNSSolver<2, true, false>::CHNSSolver;

  void check()
  {
    initialize();
    GridGenerator::subdivided_hyper_cube(*triangulation, 4);
    setup_dofs();
    setup_mappings();
    for (double offset : {.43, .5})
      for (double scale : {1., .25, 2.})
      {
        VectorTools::interpolate(*fixed_mapping,
                                 *dof_handler,
                                 BandState(ordering->n_components,
                                           scale,
                                           offset),
                                 local_evaluation_point);
        *present_solution = local_evaluation_point;
        evaluation_point  = local_evaluation_point;
        compute_error_estimate();
        Vector<float> reference_criteria;
        MeshTools::compute_interface_band_criterion<false>(*dof_handler,
                                                           *moving_mapping,
                                                           *present_solution,
                                                           ordering->phi_lower,
                                                           .08,
                                                           .2,
                                                           reference_criteria);
        for (const auto &cell : dof_handler->active_cell_iterators())
          if (cell->is_locally_owned())
          {
            const double left  = cell->vertex(0)[0];
            const double right = cell->vertex(1)[0];
            const double distance =
              scale * std::max({left - offset, offset - right, 0.});
            const bool in_band      = distance <= .08;
            const bool above_target = scale * std::sqrt(2.) / 4. > .2;
            AssertThrow(reference_criteria[cell->active_cell_index()] ==
                          (in_band ? 1.f : 0.f),
                        ExcMessage(
                          "Fixed band must use the reference diameter"));
            AssertThrow(
              cellwise_refinement_criterion[cell->active_cell_index()] ==
                (in_band && above_target ? 1.f : 0.f),
              ExcMessage(
                "ALE band must use physical distance and moving diameter"));
          }
      }

    VectorTools::interpolate(*fixed_mapping,
                             *dof_handler,
                             BandState(ordering->n_components, 1., .43, true),
                             local_evaluation_point);
    *present_solution = local_evaluation_point;
    evaluation_point  = local_evaluation_point;
    compute_error_estimate();
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const auto center = cell->center();
        const bool corner =
          std::abs(center[0] - .5) > .3 && std::abs(center[1] - .5) > .3;
        AssertThrow(cellwise_refinement_criterion[cell->active_cell_index()] ==
                      (corner ? 0.f : 1.f),
                    ExcMessage("Curved interface band is not detected"));
      }

    create_scratch_data();
    create_solver_specific_constraints_data();
    create_zero_constraints();
    create_nonzero_constraints();
    create_sparsity_pattern();
    set_state(.43);
    const auto original = triangulation->n_global_active_cells();
    param.mesh.adaptation.tree_amr.max_n_cells = original;
    adapt_mesh();
    AssertThrow(triangulation->n_global_active_cells() == original,
                ExcMessage("Cell budget must block requested refinement"));
    param.mesh.adaptation.tree_amr.max_n_cells = 10000;
    param.mesh.adaptation.tree_amr.max_level   = 0;
    adapt_mesh();
    AssertThrow(triangulation->n_global_active_cells() == original,
                ExcMessage("Maximum level must block requested refinement"));
    param.mesh.adaptation.tree_amr.max_level = 8;
    adapt_mesh();
    const auto refined = triangulation->n_global_active_cells();
    AssertThrow(refined > original,
                ExcMessage("Band must refine coarse cells"));
    for (unsigned int i = 0; i < 3; ++i)
      adapt_mesh();
    AssertThrow(triangulation->n_global_active_cells() == refined,
                ExcMessage("Stationary band must stop refining at the target"));
    compute_error_estimate();
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        AssertThrow(cellwise_refinement_criterion[cell->active_cell_index()] <=
                      0.f,
                    ExcMessage("Target diameter not reached"));
    set_state(.82);
    for (unsigned int i = 0; i < 3; ++i)
      adapt_mesh();
    for (const auto &cell : dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const double distance =
          std::max({cell->vertex(0)[0] - .82, .82 - cell->vertex(1)[0], 0.});
        if (distance <= .08)
          AssertThrow(
            MeshTools::mapped_cell_diameter(*moving_mapping, cell) <= .2,
            ExcMessage("Refinement must follow the moving interface"));
      }
    set_state(2.); // No zero contour remains in the domain.
    for (unsigned int i = 0; i < 3; ++i)
      adapt_mesh();
    AssertThrow(triangulation->n_global_active_cells() == original,
                ExcMessage("Cells left behind by the band must coarsen"));
  }

private:
  void set_state(const double offset)
  {
    VectorTools::interpolate(*fixed_mapping,
                             *dof_handler,
                             BandState(ordering->n_components, 1., offset),
                             local_evaluation_point);
    nonzero_constraints.distribute(local_evaluation_point);
    *present_solution = local_evaluation_point;
    evaluation_point  = local_evaluation_point;
    for (auto &previous : *previous_solutions)
      previous = *present_solution;
  }
};

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize   mpi(argc, argv, 1);
  Parameters::BoundaryConditionsData bc;
  ParameterHandler                   prm;
  ParameterReader<2>                 param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Output
  set write vtu results = false
end
subsection FiniteElements
  set use quads = true
end
subsection Cahn Hilliard
  set interface thickness = 0.1
end
subsection Mesh
  subsection Adaptation
    set enable = true
    set strategy = local refinement
    subsection Local refinement
      set refinement strategy = interface band
      set interface band half width over epsilon = 0.8
      set interface band diameter over epsilon = 2
    end
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  BandSolver(param).check();
  bool rejected = false;
  try
  {
    HeatSolver<2> heat(param);
  }
  catch (const ExceptionBase &exception)
  {
    rejected =
      std::string(exception.what()).find("interface band") != std::string::npos;
  }
  AssertThrow(rejected,
              ExcMessage(
                "Heat must reject interface band without a phase tracer"));
  initlog();
  deallog << "Interface band AMR uses physical distance and moving diameter"
          << std::endl;
}
