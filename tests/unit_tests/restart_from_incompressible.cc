
#include <compressible_ns_solver.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <incompressible_ns_solver.h>
#include <parameter_reader.h>
#include <parameters.h>

#include <array>
#include <cmath>
#include <iostream>

#include "../tests.h"

template <int dim>
class InspectableNSSolver : public NSSolver<dim>
{
public:
  using NSSolver<dim>::NSSolver;

  const DoFHandler<dim> &get_dof_handler() const { return *this->dof_handler; }
  const Mapping<dim>    &get_mapping() const { return *this->fixed_mapping; }
  LA::ParVectorType     &solution() { return *this->present_solution; }
  LA::ParVectorType     &previous_solution(const unsigned int i)
  {
    return (*this->previous_solutions)[i];
  }
};

template <int dim>
class InspectableCompressibleNSSolver : public CompressibleNSSolver<dim>
{
public:
  using CompressibleNSSolver<dim>::CompressibleNSSolver;

  const DoFHandler<dim> &get_dof_handler() const { return *this->dof_handler; }
  const Mapping<dim>    &get_mapping() const { return *this->fixed_mapping; }
  const IndexSet    &get_owned_dofs() const { return this->locally_owned_dofs; }
  LA::ParVectorType &solution() { return *this->present_solution; }
  LA::ParVectorType &previous_solution(const unsigned int i)
  {
    return (*this->previous_solutions)[i];
  }
  const TimeHandler &time() const { return this->time_handler; }
};

template <int dim>
void set_common_parameters(ParameterHandler &prm,
                           const bool        enable_checkpoint,
                           const bool        restart)
{
  prm.enter_subsection("Dimension");
  prm.set("dimension", std::to_string(dim));
  prm.leave_subsection();

  prm.enter_subsection("Timer");
  prm.set("enable timer", "false");
  prm.leave_subsection();

  prm.enter_subsection("Output");
  prm.set("write vtu results", "false");
  prm.set("output directory", "./");
  prm.set("output prefix", "solution");
  prm.leave_subsection();

  prm.enter_subsection("Mesh");
  prm.set("verbosity", "quiet");
  prm.leave_subsection();

  prm.enter_subsection("Time integration");
  prm.set("verbosity", "quiet");
  prm.set("dt", "0.1");
  prm.set("t_initial", "0");
  prm.set("t_end", "0.3");
  prm.set("scheme", "BDF2");
  prm.set("bdf start method", "initial condition");
  prm.leave_subsection();

  prm.enter_subsection("Nonlinear solver");
  prm.set("verbosity", "quiet");
  prm.set("tolerance", "1e-8");
  prm.set("divergence_tolerance", "1e+4");
  prm.set("max_iterations", "50");
  prm.set("enable_line_search", "true");
  prm.set("analytic_jacobian", "true");
  prm.leave_subsection();

  prm.enter_subsection("Linear solver");
  prm.enter_subsection("main physics");
  prm.set("verbosity", "quiet");
  prm.set("method", "direct_mumps");
  prm.leave_subsection();
  prm.leave_subsection();

  prm.enter_subsection("FiniteElements");
  prm.set("Velocity degree", "2");
  prm.set("Pressure degree", "1");
  prm.leave_subsection();

  prm.enter_subsection("Initial conditions");
  prm.set("to mms", "true");
  prm.leave_subsection();

  prm.enter_subsection("Fluid boundary conditions");
  prm.set("number", "4");
  prm.set("fix pressure constant", "true");
  const std::array<std::string, 4> boundary_names = {
    {"x_min", "x_max", "y_min", "y_max"}};
  for (unsigned int i = 0; i < 4; ++i)
  {
    prm.enter_subsection("boundary " + std::to_string(i));
    prm.set("id", std::to_string(i + 1));
    prm.set("name", boundary_names[i]);
    prm.set("type", "velocity_mms");
    prm.leave_subsection();
  }
  prm.leave_subsection();

  prm.enter_subsection("Exact solution");
  prm.enter_subsection("exact velocity");
  prm.set("Function expression",
          "(x*y + y^2) * cos(t); (x*y + x^2) * 1/(1 + t^2)");
  prm.leave_subsection();
  prm.enter_subsection("exact pressure");
  prm.set("Function expression", "(x + y) * sin(t)");
  prm.leave_subsection();
  prm.leave_subsection();

  prm.enter_subsection("Manufactured solution");
  prm.set("enable", "true");
  prm.set("type", "time");
  prm.set("convergence steps", "1");
  prm.enter_subsection("Space convergence");
  prm.set("use dealii cube mesh", "true");
  prm.set("norms to compute", "L2_norm");
  prm.leave_subsection();
  prm.enter_subsection("Time convergence");
  prm.set("norm", "Linfty");
  prm.set("use spatial mesh", "true");
  prm.set("spatial mesh index", "1");
  prm.set("time step reduction", "0.5");
  prm.leave_subsection();
  prm.leave_subsection();

  prm.enter_subsection("Checkpoint Restart");
  prm.set("enable checkpoint", enable_checkpoint ? "true" : "false");
  prm.set("restart", restart ? "true" : "false");
  prm.set("checkpoint file", "checkpoint");
  prm.set("checkpoint frequency", "3");
  prm.leave_subsection();
}

template <int dim>
ParameterReader<dim>
make_incompressible_parameters(const bool enable_checkpoint, const bool restart)
{
  Parameters::BoundaryConditionsData bc_data;
  bc_data.n_fluid_bc = 4;

  ParameterHandler     prm;
  ParameterReader<dim> dummy_param(bc_data);
  dummy_param.declare(prm);

  set_common_parameters<dim>(prm, enable_checkpoint, restart);

  prm.enter_subsection("Physical properties");
  prm.set("number of fluids", "1");
  prm.enter_subsection("Fluid 0");
  prm.set("density", "1.2345");
  prm.set("kinematic viscosity", "9.8765");
  prm.leave_subsection();
  prm.leave_subsection();

  ParameterReader<dim> param(bc_data);
  ParameterHandler     dummy_prm;
  param.declare(dummy_prm);
  param.read(prm);
  return param;
}

template <int dim>
ParameterReader<dim> make_compressible_parameters()
{
  Parameters::BoundaryConditionsData bc_data;
  bc_data.n_fluid_bc = 4;
  bc_data.n_heat_bc  = 4;

  ParameterHandler     prm;
  ParameterReader<dim> dummy_param(bc_data);
  dummy_param.declare(prm);

  set_common_parameters<dim>(prm, false, true);

  prm.enter_subsection("FiniteElements");
  prm.set("Temperature degree", "1");
  prm.leave_subsection();

  prm.enter_subsection("Physical properties");
  prm.set("number of fluids", "1");
  prm.enter_subsection("Fluid 0");
  prm.set("density", "1.2345");
  prm.set("thermal conductivity", "0.1234");
  prm.set("heat capacity at constant pressure", "1234");
  prm.set("dynamic viscosity", "9.8765");
  prm.set("pressure reference", "101325");
  prm.set("temperature reference", "293");
  prm.leave_subsection();
  prm.leave_subsection();

  prm.enter_subsection("Heat boundary conditions");
  prm.set("number", "4");
  const std::array<std::string, 4> boundary_names = {
    {"x_min", "x_max", "y_min", "y_max"}};
  for (unsigned int i = 0; i < 4; ++i)
  {
    prm.enter_subsection("boundary " + std::to_string(i));
    prm.set("id", std::to_string(i + 1));
    prm.set("name", boundary_names[i]);
    prm.set("type", "dirichlet_mms");
    prm.leave_subsection();
  }
  prm.leave_subsection();

  prm.enter_subsection("Exact solution");
  prm.enter_subsection("exact temperature");
  prm.set("Function expression", "x + y");
  prm.leave_subsection();
  prm.leave_subsection();

  ParameterReader<dim> param(bc_data);
  ParameterHandler     dummy_prm;
  param.declare(dummy_prm);
  param.read(prm);
  return param;
}

template <int dim>
class TestIncompressibleRestartAdapter : public Function<dim>
{
public:
  TestIncompressibleRestartAdapter(
    Functions::FEFieldFunction<dim, LA::ParVectorType> &fe_field,
    const double                                        rho_ref,
    const double                                        p_ref,
    const double                                        T_ref)
    : Function<dim>(dim + 2)
    , fe_field(fe_field)
    , rho_ref(rho_ref)
    , p_ref(p_ref)
    , T_ref(T_ref)
  {}

  virtual double value(const Point<dim>  &point,
                       const unsigned int component) const override
  {
    if (component < dim)
      return fe_field.value(point, component);

    const double kinematic_pressure = fe_field.value(point, dim);
    if (component == dim)
      return rho_ref * kinematic_pressure;

    return rho_ref * T_ref / p_ref * kinematic_pressure;
  }

private:
  Functions::FEFieldFunction<dim, LA::ParVectorType> &fe_field;
  const double                                        rho_ref;
  const double                                        p_ref;
  const double                                        T_ref;
};

template <int dim>
struct MappingError
{
  double       value          = 0.;
  unsigned int component      = numbers::invalid_unsigned_int;
  double       actual_value   = 0.;
  double       expected_value = 0.;
  Point<dim>   point;
};

template <int dim>
MappingError<dim>
max_compressible_error(InspectableCompressibleNSSolver<dim> &reference,
                       LA::ParVectorType                    &reference_solution,
                       InspectableCompressibleNSSolver<dim> &candidate,
                       LA::ParVectorType                    &candidate_solution)
{
  Functions::FEFieldFunction<dim, LA::ParVectorType> reference_field(
    reference.get_dof_handler(), reference_solution, reference.get_mapping());
  Functions::FEFieldFunction<dim, LA::ParVectorType> candidate_field(
    candidate.get_dof_handler(), candidate_solution, candidate.get_mapping());

  MappingError<dim> local_error;
  for (const auto &cell : reference.get_dof_handler().active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    const Point<dim> point = cell->center();
    for (unsigned int component = 0; component < dim + 2; ++component)
    {
      const double expected_value = reference_field.value(point, component);
      const double actual_value   = candidate_field.value(point, component);
      const double point_error    = std::abs(actual_value - expected_value);
      if (point_error > local_error.value)
      {
        local_error.value          = point_error;
        local_error.component      = component;
        local_error.actual_value   = actual_value;
        local_error.expected_value = expected_value;
        local_error.point          = point;
      }
    }
  }

  local_error.value = Utilities::MPI::max(local_error.value, MPI_COMM_WORLD);
  return local_error;
}

template <int dim>
void map_incompressible_solution(
  InspectableNSSolver<dim>             &incompressible,
  LA::ParVectorType                    &incompressible_solution,
  InspectableCompressibleNSSolver<dim> &compressible,
  LA::ParVectorType                    &compressible_solution)
{
  Functions::FEFieldFunction<dim, LA::ParVectorType> incomp_field(
    incompressible.get_dof_handler(),
    incompressible_solution,
    incompressible.get_mapping());
  TestIncompressibleRestartAdapter<dim> test_adapter(incomp_field,
                                                     1.2345,
                                                     101325.,
                                                     293.);

  VectorTools::interpolate(compressible.get_mapping(),
                           compressible.get_dof_handler(),
                           test_adapter,
                           compressible_solution);
  compressible_solution.compress(VectorOperation::insert);
}

template <int dim>
MappingError<dim>
max_incompressible_error(InspectableNSSolver<dim> &reference,
                         LA::ParVectorType        &reference_solution,
                         InspectableNSSolver<dim> &candidate,
                         LA::ParVectorType        &candidate_solution)
{
  Functions::FEFieldFunction<dim, LA::ParVectorType> reference_field(
    reference.get_dof_handler(), reference_solution, reference.get_mapping());
  Functions::FEFieldFunction<dim, LA::ParVectorType> candidate_field(
    candidate.get_dof_handler(), candidate_solution, candidate.get_mapping());

  MappingError<dim> local_error;
  for (const auto &cell : reference.get_dof_handler().active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    const Point<dim> point = cell->center();
    for (unsigned int component = 0; component < dim + 1; ++component)
    {
      const double expected_value = reference_field.value(point, component);
      const double actual_value   = candidate_field.value(point, component);
      const double point_error    = std::abs(actual_value - expected_value);
      if (point_error > local_error.value)
      {
        local_error.value          = point_error;
        local_error.component      = component;
        local_error.actual_value   = actual_value;
        local_error.expected_value = expected_value;
        local_error.point          = point;
      }
    }
  }

  local_error.value = Utilities::MPI::max(local_error.value, MPI_COMM_WORLD);
  return local_error;
}

template <int dim>
void test_restart_from_incompressible()
{
  auto incomp_param = make_incompressible_parameters<dim>(true, false);
  InspectableNSSolver<dim> incompressible(incomp_param);
  incompressible.run();

  auto incomp_restart_param = make_incompressible_parameters<dim>(false, true);
  InspectableNSSolver<dim> incompressible_restart(incomp_restart_param);
  incompressible_restart.reset();
  incompressible_restart.initialize();
  incompressible_restart.restart();
  const auto incomp_restart_error =
    max_incompressible_error(incompressible,
                             incompressible.solution(),
                             incompressible_restart,
                             incompressible_restart.solution());
  AssertThrow(incomp_restart_error.value < 1e-12,
              ExcMessage(
                "Incompressible restart changed the checkpoint "
                "solution: " +
                std::to_string(incomp_restart_error.value) + " on component " +
                std::to_string(incomp_restart_error.component) + " actual " +
                std::to_string(incomp_restart_error.actual_value) +
                " expected " +
                std::to_string(incomp_restart_error.expected_value)));

  auto comp_param = make_compressible_parameters<dim>();
  InspectableCompressibleNSSolver<dim> compressible(comp_param);
  compressible.reset();
  compressible.initialize();
  compressible.restart();

  LA::ParVectorType direct_mapped(compressible.get_owned_dofs(),
                                  MPI_COMM_WORLD);
  map_incompressible_solution(incompressible,
                              incompressible.solution(),
                              compressible,
                              direct_mapped);

  AssertThrow(std::abs(compressible.time().current_time - 0.3) < 1e-14,
              ExcMessage("Compressible restart did not load checkpoint time."));
  AssertThrow(compressible.time().current_time_iteration == 3,
              ExcMessage("Compressible restart did not load checkpoint step."));
  AssertThrow(compressible.time().bdf_coefficients.size() == 3,
              ExcMessage(
                "Compressible restart did not load BDF2 coefficients."));
  deallog << "time handler OK" << std::endl;

  const auto present_error = max_compressible_error(compressible,
                                                    direct_mapped,
                                                    compressible,
                                                    compressible.solution());
  AssertThrow(present_error.value < 1e-12,
              ExcMessage(
                "Present solution was not mapped correctly: " +
                std::to_string(present_error.value) + " on component " +
                std::to_string(present_error.component) + " actual " +
                std::to_string(present_error.actual_value) + " expected " +
                std::to_string(present_error.expected_value)));
  deallog << "present solution OK" << std::endl;

  for (unsigned int i = 0; i < 2; ++i)
  {
    LA::ParVectorType direct_mapped_previous(compressible.get_owned_dofs(),
                                             MPI_COMM_WORLD);
    map_incompressible_solution(incompressible,
                                incompressible.previous_solution(i),
                                compressible,
                                direct_mapped_previous);
    const auto previous_error =
      max_compressible_error(compressible,
                             direct_mapped_previous,
                             compressible,
                             compressible.previous_solution(i));
    AssertThrow(previous_error.value < 1e-12,
                ExcMessage("Previous solution was not mapped correctly."));
  }
  deallog << "previous solutions OK" << std::endl;

  deallog << "OK" << std::endl;
}

int main(int argc, char *argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPILogInitAll                    log;
    test_restart_from_incompressible<2>();
  }
  catch (const std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
