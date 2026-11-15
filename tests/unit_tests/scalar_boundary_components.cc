#include <boundary_conditions.h>
#include <compressible_ns_solver.h>
#include <deal.II/dofs/dof_tools.h>
#include <mesh.h>

#include <array>
#include <cmath>

#include "../tests.h"

namespace
{
  template <int dim>
  double pressure(const Point<dim> &p, const double time)
  {
    double value = 7. + 4. * time;
    for (unsigned int d = 0; d < dim; ++d)
      value += (d + 1.) * p[d];
    return value;
  }

  template <int dim>
  double temperature(const Point<dim> &p, const double time)
  {
    double value = -3. - 2. * p[0] + 3. * p[1] - 2. * time;
    if constexpr (dim == 3)
      value += 4. * p[2];
    return value;
  }

  void close(const double actual, const double expected)
  {
    AssertThrow(std::isfinite(actual) && std::abs(actual - expected) < 1e-11,
                ExcMessage("Unexpected scalar boundary value: " +
                           std::to_string(actual) +
                           " != " + std::to_string(expected)));
  }

  template <int dim>
  std::string expression(const bool is_temperature)
  {
    if (is_temperature)
      return dim == 2 ? "-3-2*x+3*y-2*t" : "-3-2*x+3*y+4*z-2*t";
    return dim == 2 ? "7+x+2*y+4*t" : "7+x+2*y+3*z+4*t";
  }

  template <int dim>
  void test_evaluation()
  {
    Functions::ParsedFunction<dim> scalar;
    ParameterHandler               prm;
    scalar.declare_parameters(prm);
    prm.set("Function expression", expression<dim>(false));
    scalar.parse_parameters(prm);
    const unsigned int      n_components = dim + 2;
    std::vector<Point<dim>> points(3);
    for (unsigned int q = 0; q < points.size(); ++q)
      for (unsigned int d = 0; d < dim; ++d)
        points[q][d] = .2 * (q + 1) - .3 * d;

    // Exercise arbitrary placements, including the first and last components.
    for (unsigned int selected = 0; selected < n_components; ++selected)
    {
      ScalarFunctionFromComponents<dim> embedded(selected,
                                                 n_components,
                                                 scalar);
      for (const double time : {.25, 1.5})
      {
        scalar.set_time(time);
        std::vector<Vector<double>> list(points.size(),
                                         Vector<double>(n_components));
        // Repeated vector evaluation must overwrite unselected entries too.
        for (auto &values : list)
          values = -99.;
        embedded.vector_value_list(points, list);
        for (unsigned int q = 0; q < points.size(); ++q)
        {
          Vector<double> values(n_components);
          values = 123.;
          embedded.vector_value(points[q], values);
          for (unsigned int c = 0; c < n_components; ++c)
          {
            const double expected =
              c == selected ? pressure(points[q], time) : 0.;
            close(embedded.value(points[q], c), expected);
            close(values[c], expected);
            close(list[q][c], expected);
          }
        }
      }
    }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      deallog << dim << "D scalar and vector evaluation OK" << std::endl;
  }

  template <int dim>
  ParameterReader<dim> make_parameters()
  {
    Parameters::BoundaryConditionsData bc{};
    bc.n_fluid_bc = bc.n_heat_bc = 2 * dim;
    ParameterReader<dim> param(bc);
    ParameterHandler     prm;
    param.declare(prm);
    prm.parse_input_from_string(R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
  set output directory = ./
end
subsection Mesh
  set verbosity = quiet
  set use dealii cube mesh = true
  set refinement level = 2
end
subsection Time integration
  set verbosity = quiet
  set scheme = stationary
end
subsection FiniteElements
  set use quads = true
  set Velocity degree = 2
  set Pressure degree = 1
  set Temperature degree = 2
end
subsection Physical properties
  set number of fluids = 1
  subsection Fluid 0
    set density = 2
    set dynamic viscosity = 0.2
    set thermal conductivity = 0.3
    set heat capacity at constant pressure = 20
    set pressure reference = 100
    set temperature reference = 10
  end
end
subsection Fluid boundary conditions
  set fix pressure constant = false
end
)");
    prm.enter_subsection("Dimension");
    prm.set("dimension", std::to_string(dim));
    prm.leave_subsection();
    const std::array<std::string, 6> names{
      {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}};
    for (const bool heat : {false, true})
    {
      prm.enter_subsection(heat ? "Heat boundary conditions" :
                                  "Fluid boundary conditions");
      prm.set("number", std::to_string(2 * dim));
      for (unsigned int i = 0; i < 2 * dim; ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        prm.set("id", std::to_string(i + 1));
        prm.set("name", names[i]);
        prm.set("type",
                heat ? (i == 1 ? "input_function" : "no_flux") :
                       (i == 0 ? "dirichlet_pressure" : "outflow"));
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
    prm.enter_subsection("Fluid boundary conditions");
    prm.enter_subsection("boundary 0");
    prm.enter_subsection("p");
    prm.set("Function expression", expression<dim>(false));
    prm.leave_subsection();
    prm.leave_subsection();
    prm.leave_subsection();
    prm.enter_subsection("Heat boundary conditions");
    prm.enter_subsection("boundary 1");
    prm.enter_subsection("temperature");
    prm.set("Function expression", expression<dim>(true));
    prm.leave_subsection();
    prm.leave_subsection();
    prm.leave_subsection();
    param.read(prm);
    return param;
  }

  template <int dim>
  class ConstraintProbe : public CompressibleNSSolver<dim>
  {
  public:
    using CompressibleNSSolver<dim>::CompressibleNSSolver;

    void check()
    {
      this->reset();
      this->initialize();
      this->set_interval_data(0);
      MeshTools::read_mesh(*this->triangulation, this->param);
      this->setup_dofs();
      this->setup_mappings();
      const auto points =
        DoFTools::map_dofs_to_support_points(*this->fixed_mapping,
                                             *this->dof_handler);
      AssertThrow(Utilities::MPI::min(this->locally_owned_dofs.n_elements(),
                                      MPI_COMM_WORLD) > 0,
                  ExcMessage("Every MPI rank must own DoFs"));
      for (const double time : {.25, 1.5})
      {
        // Set the callback times explicitly: this test targets evaluation and
        // constraint construction, rather than the solver's time integration.
        for (auto &[id, bc] : this->param.fluid_bc)
          bc.set_time(time);
        for (auto &[id, bc] : this->param.heat_bc)
          bc.set_time(time);
        this->create_zero_constraints();
        this->create_nonzero_constraints();

        for (const bool homogeneous : {false, true})
        {
          const auto &constraints =
            homogeneous ? this->zero_constraints : this->nonzero_constraints;
          Vector<double> values(this->dof_handler->n_dofs());
          for (unsigned int i = 0; i < values.size(); ++i)
            values[i] = -100. - i;
          constraints.distribute(values);
          unsigned int pressure_count = 0, temperature_count = 0;
          for (const auto i : this->locally_owned_dofs)
          {
            const auto  c = this->dofs_to_component[this->locally_relevant_dofs
                                                     .index_within_set(i)];
            const auto &point = points.at(i);
            const bool  is_pressure =
              c == this->ordering->p_lower && std::abs(point[0]) < 1e-12;
            const bool is_temperature =
              c == this->ordering->t_lower && std::abs(point[0] - 1.) < 1e-12;
            AssertThrow(
              constraints.is_constrained(i) == (is_pressure || is_temperature),
              ExcMessage("Constraint applied to the wrong field or face"));
            double expected = -100. - i;
            if (is_pressure)
            {
              ++pressure_count;
              expected = homogeneous ? 0. : pressure(point, time);
            }
            if (is_temperature)
            {
              ++temperature_count;
              expected = homogeneous ? 0. : temperature(point, time);
            }
            close(values[i], expected);
          }
          AssertThrow(
            Utilities::MPI::sum(pressure_count, MPI_COMM_WORLD) > 0 &&
              Utilities::MPI::sum(temperature_count, MPI_COMM_WORLD) > 0,
            ExcMessage("The test did not exercise both scalar boundaries"));
        }
      }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog << dim << "D pressure and temperature constraints OK"
                << std::endl;
    }
  };
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  try
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      initlog();
    test_evaluation<2>();
    test_evaluation<3>();
    ConstraintProbe<2>(make_parameters<2>()).check();
    ConstraintProbe<3>(make_parameters<3>()).check();
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
