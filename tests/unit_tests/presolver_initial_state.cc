#include <deal.II/fe/fe_values.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

// Inspect the initial state through the real CHNS postprocessing hook, then
// let the solver advance through its BDF startup and subsequent time steps.
template <int dim>
class InitialStateProblem : public CHNSSolver<dim, true>
{
public:
  InitialStateProblem(const ParameterReader<dim> &param)
    : CHNSSolver<dim, true>(param)
  {
    this->pcout.set_condition(false);
  }

  void solver_specific_post_processing() override
  {
    if (this->time_handler.current_time_iteration != 0)
    {
      if (this->time_handler.current_time_iteration == 1)
      {
        double error = 0.;
        for (const auto &[i, position] : initial_positions)
          error =
            std::max(error, std::abs((*this->present_solution)[i] - position));
        AssertThrow(Utilities::MPI::max(error, this->mpi_communicator) < 1e-12,
                    ExcMessage("BDF startup changed the presolved position"));
        checked_startup = true;
      }
      return;
    }
    ++checks;
    const auto                      &fe = this->get_fe_system();
    const Quadrature<dim>            nodes(fe.get_unit_support_points());
    FEValues<dim>                    values(*this->fixed_mapping,
                         fe,
                         nodes,
                         update_values | update_quadrature_points);
    const FEValuesExtractors::Vector position(this->ordering->x_lower);
    const FEValuesExtractors::Vector velocity(this->ordering->u_lower);
    const FEValuesExtractors::Scalar phase(this->ordering->phi_lower);
    std::vector<Tensor<1, dim>>      x(nodes.size()), u(nodes.size());
    std::vector<double>              phi(nodes.size());
    double                           displacement = 0., error = 0.;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        values[position].get_function_values(*this->present_solution, x);
        values[velocity].get_function_values(*this->present_solution, u);
        values[phase].get_function_values(*this->present_solution, phi);
        for (unsigned int i = 0; i < nodes.size(); ++i)
        {
          const auto component = fe.system_to_component_index(i).first;
          if (this->ordering->is_tracer(component))
            error = std::max(error,
                             std::abs(phi[i] - std::tanh((x[i][0] - .4) / .2)));
          if (this->ordering->is_velocity(component))
          {
            double expected = .1;
            for (unsigned int d = 0; d < dim; ++d)
              expected *= x[i][d] * (1. - x[i][d]);
            const unsigned int d = component - this->ordering->u_lower;
            error =
              std::max(error, std::abs(u[i][d] - (d == 0 ? expected : 0.)));
          }
          if (this->ordering->is_position(component))
            displacement = std::max(
              displacement,
              (x[i] - Tensor<1, dim>(values.quadrature_point(i))).norm());
        }
        // The reference vertices remain on the original regular grid.
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          for (unsigned int d = 0; d < dim; ++d)
            error = std::max(error,
                             std::abs(3. * cell->vertex(v)[d] -
                                      std::round(3. * cell->vertex(v)[d])));
      }
    for (const auto i : this->locally_owned_dofs)
      if (this->ordering->is_position(
            this->dofs_to_component[this->locally_relevant_dofs
                                      .index_within_set(i)]))
        for (const auto &previous : *this->previous_solutions)
          error =
            std::max(error,
                     std::abs(previous[i] - (*this->present_solution)[i]));
    for (const auto i : this->locally_owned_dofs)
      if (this->ordering->is_position(
            this->dofs_to_component[this->locally_relevant_dofs
                                      .index_within_set(i)]))
        initial_positions[i] = (*this->present_solution)[i];
    error        = Utilities::MPI::max(error, this->mpi_communicator);
    displacement = Utilities::MPI::max(displacement, this->mpi_communicator);
    AssertThrow(error < 1e-10,
                ExcMessage(
                  "Initial geometry, fields or BDF history are inconsistent"));
    AssertThrow(displacement > 1e-5,
                ExcMessage("The presolver did not compress the mesh"));
    if (this->mpi_rank == 0)
      deallog << dim << "D initial position, fields and BDF histories OK"
              << std::endl;
  }

  unsigned int                              checks          = 0;
  bool                                      checked_startup = false;
  std::map<types::global_dof_index, double> initial_positions;
};

template <int dim>
void check_initial_state()
{
  auto                     param = presolver_test_parameters<dim>();
  InitialStateProblem<dim> problem(param);
  problem.run();
  AssertThrow(problem.checks == 1 && problem.checked_startup,
              ExcInternalError());
  if (problem.mpi_rank == 0)
    deallog << dim << "D BDF time steps converged" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  check_initial_state<2>();
  check_initial_state<3>();
}
