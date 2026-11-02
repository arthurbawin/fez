#include <deal.II/fe/fe_values.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

struct CheckedInitialState
{};
class BoundaryProblem : public CHNSSolver<2, true>
{
public:
  BoundaryProblem(const ParameterReader<2> &p)
    : CHNSSolver<2, true>(p)
  {
    pcout.set_condition(false);
  }
  void solver_specific_post_processing() override
  {
    time_handler.current_time = .375;
    set_solver_specific_time();
    const auto                &fe = get_fe_system();
    const Quadrature<2>        nodes(fe.get_unit_support_points());
    FEValues<2>                values(*fixed_mapping,
                       fe,
                       nodes,
                       update_values | update_quadrature_points);
    FEValuesExtractors::Vector x(ordering->x_lower);
    FEValuesExtractors::Scalar phi(ordering->phi_lower);
    std::vector<Tensor<1, 2>>  positions(nodes.size());
    std::vector<double>        phase(nodes.size());
    const auto                 points =
      DoFTools::map_dofs_to_support_points(*fixed_mapping, *dof_handler);
    for (double alpha : {1., .5, .125})
    {
      local_evaluation_point = *present_solution;
      for (const auto i : locally_owned_dofs)
        if (ordering->is_position(
              dofs_to_component[locally_relevant_dofs.index_within_set(i)]))
        {
          const auto d =
            dofs_to_component[locally_relevant_dofs.index_within_set(i)] -
            ordering->x_lower;
          const double reference = points.at(i)[d];
          local_evaluation_point[i] =
            reference + .1 * alpha * reference * (1. - reference);
        }
      for (const auto i : locally_owned_dofs)
        if (ordering->is_tracer(
              dofs_to_component[locally_relevant_dofs.index_within_set(i)]))
          local_evaluation_point[i] = .123;
      // This is the production operation invoked by every Newton trial.
      local_evaluation_point.compress(VectorOperation::insert);
      distribute_nonzero_constraints();
      evaluation_point = local_evaluation_point;
      double error = 0., displacement = 0.;
      for (const auto &cell : dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
          values.reinit(cell);
          values[x].get_function_values(evaluation_point, positions);
          values[phi].get_function_values(evaluation_point, phase);
          for (unsigned int i = 0; i < nodes.size(); ++i)
            if (ordering->is_tracer(fe.system_to_component_index(i).first))
            {
              const auto &p = values.quadrature_point(i);
              if (p[0] < 1e-12 || p[0] > 1. - 1e-12 || p[1] < 1e-12 ||
                  p[1] > 1. - 1e-12)
              {
                error =
                  std::max(error,
                           std::abs(phase[i] - (positions[i][0] +
                                                2. * positions[i][1] + .375)));
                displacement =
                  std::max(displacement,
                           (positions[i] - Tensor<1, 2>(p)).norm());
              }
              else
                error = std::max(error, std::abs(phase[i] - .123));
            }
        }
      error        = Utilities::MPI::max(error, mpi_communicator);
      displacement = Utilities::MPI::max(displacement, mpi_communicator);
      AssertThrow(error < 1e-12,
                  ExcMessage(
                    "Prescribed phase uses stale ALE coordinates or time"));
      AssertThrow(displacement > 1e-4,
                  ExcMessage("The trial did not move boundary support points"));
    }
    if (mpi_rank == 0)
      deallog
        << "Prescribed phase follows the time and every trial ALE geometry"
        << std::endl;
    throw CheckedInitialState{};
  }
};
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    std::ostringstream overrides;
    overrides << "subsection Cahn Hilliard\nset use presolver=false\nset mff "
                 "source term=off\nend\n";
    for (const std::string section : {"Pseudosolid boundary conditions",
                                      "CahnHilliard boundary conditions"})
    {
      overrides << "subsection " << section << "\n";
      for (unsigned int b = 0; b < 4; ++b)
      {
        overrides << "subsection boundary " << b << "\nset type="
                  << (section == "Pseudosolid boundary conditions" ?
                        "no_flux" :
                        "input_function")
                  << "\n";
        if (section == "CahnHilliard boundary conditions")
          overrides
            << "subsection tracer\nset Function expression=x+2*y+t\nend\n";
        overrides << "end\n";
      }
      overrides << "end\n";
    }
    auto            p = presolver_test_parameters<2>(overrides.str());
    BoundaryProblem problem(p);
    problem.run();
    AssertThrow(false, ExcMessage("Initial state check was not executed"));
  }
  catch (const CheckedInitialState &)
  {
    return 0;
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
