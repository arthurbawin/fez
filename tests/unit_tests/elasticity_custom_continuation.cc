#include <elasticity_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

// Check the solved deformation even though standalone elasticity moves the
// triangulation vertices during its final postprocessing.
class CustomElasticity : public ElasticitySolver<2>
{
public:
  CustomElasticity(const ParameterReader<2> &p)
    : ElasticitySolver<2>(p)
  {}
  void output_results(const Mapping<2> * = nullptr) override
  {
    if (!initialized)
    {
      for (const auto i : locally_owned_dofs)
        initial[i] = present_solution[i];
      initialized = true;
      return;
    }
    double displacement = 0.;
    for (const auto &[i, x] : initial)
      displacement = std::max(displacement, std::abs(present_solution[i] - x));
    displacement = Utilities::MPI::max(displacement, mpi_communicator);
    AssertThrow(
      displacement > .01 && displacement < .4,
      ExcMessage(
        "Custom forcing must produce a significant, bounded deformation"));
    if (mpi_rank == 0)
      deallog
        << "Custom elasticity continuation deforms the mesh by more than 1%"
        << std::endl;
    checked = true;
  }
  bool initialized = false, checked = false;

private:
  std::map<types::global_dof_index, double> initial;
};
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    auto p = presolver_test_parameters<2>(R"(subsection Time integration
 set scheme=stationary
end
subsection Cahn Hilliard
 set use presolver=false
 set mff source term=custom
 set mff physics compression factor=1000
end
subsection Elasticity
 subsection source term
  set Function expression=-12*x*(1-x)*y*(1-y);6*x*(1-x)*y*(1-y)
 end
 subsection current mesh source term
  set enable=true
  set min multiplier=1
  set max multiplier=4
  set continuation steps=4
 end
end
)");
    CustomElasticity solver(p);
    solver.run();
    AssertThrow(solver.initialized && solver.checked, ExcInternalError());
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
