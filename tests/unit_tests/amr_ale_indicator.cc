#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

// Reference-coordinate kinks stay aligned with interior faces as ALE moves.
template <int dim>
class State : public Function<dim>
{
public:
  State(const unsigned int components,
        const double       scale,
        const bool         rigid,
        const bool         local)
    : Function<dim>(components)
    , scale(scale)
    , rigid(rigid)
    , local(local)
  {}

  double value(const Point<dim> &p, const unsigned int c) const override
  {
    if (c == 0)
      return std::abs(p[1] - 0.5);
    if (c == 2 * dim + 1)
      return std::abs(p[0] - 0.5);
    if (c >= dim + 1 && c < 2 * dim + 1)
    {
      const unsigned int d = c - dim - 1;
      if (local && d == 0)
        return p[0] <= 0.5 ? 0.5 * p[0] : 1.5 * p[0] - 0.5;
      if (rigid && d < 2)
        return d == 0 ? 2. - scale * p[1] : scale * p[0] - 1.;
      return scale * p[d];
    }
    return 0.;
  }

private:
  const double scale;
  const bool   rigid, local;
};

template <int dim>
class TestSolver : public CHNSSolver<dim, true, false>
{
public:
  using CHNSSolver<dim, true, false>::CHNSSolver;

  void check()
  {
    this->initialize();
    GridGenerator::subdivided_hyper_cube(*this->triangulation, 2);
    this->setup_dofs();
    this->setup_mappings();
    for (const double scale : {1., 0.25, 2.})
      check_geometry(scale, false, false);
    check_geometry(1., true, false);
    check_geometry(1., false, true);
  }

private:
  void check_geometry(const double scale, const bool rigid, const bool local)
  {
    VectorTools::interpolate(
      *this->fixed_mapping,
      *this->dof_handler,
      State<dim>(
        this->dof_handler->get_fe().n_components(), scale, rigid, local),
      this->local_evaluation_point);
    *this->present_solution = this->local_evaluation_point;
    this->evaluation_point  = this->local_evaluation_point;

    using V                                     = SolverInfo::VariableType;
    const std::vector<std::vector<V>> variables = {
      {V::phase_tracer},
      {V::velocity},
      {V::pressure},
      {V::phase_tracer, V::velocity},
      {V::velocity, V::phase_tracer},
      {V::pressure, V::phase_tracer}};

    // Analytic face jump integrals on the two columns. For local compression,
    // widths are 1/4 and 3/4, and the tracer gradient jump is 2 + 2/3.
    double tracer[2], velocity[2];
    for (unsigned int column = 0; column < 2; ++column)
    {
      const double a           = local ? (column == 0 ? 0.5 : 1.5) : scale;
      const double b           = local ? 1. : scale;
      const double diameter    = 0.5 * std::sqrt(a * a + (dim - 1) * b * b);
      const double tracer_jump = local ? 8. / 3. : 2. / scale;
      tracer[column]   = std::sqrt(diameter / 24. * std::pow(b / 2., dim - 1) *
                                 tracer_jump * tracer_jump);
      velocity[column] = std::sqrt(diameter / 24. * (a / 2.) *
                                   std::pow(b / 2., dim - 2) * 4. / (b * b));
    }
    for (const auto &fields : variables)
    {
      this->param.mesh.adaptation.tree_amr.variables_for_adaptation = fields;
      this->compute_error_estimate();
      for (const auto &cell : this->dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const unsigned int column = cell->center()[0] < 0.5 ? 0 : 1;
          for (unsigned int f = 0; f < fields.size(); ++f)
          {
            const auto   field     = fields[f];
            const double expected  = field == V::phase_tracer ? tracer[column] :
                                     field == V::velocity ? velocity[column] :
                                                            0.;
            const auto  &criterion = fields.size() > 1 ?
                                       this->field_refinement_criteria[f] :
                                       this->cellwise_refinement_criterion;
            const double actual    = criterion[cell->active_cell_index()];
            AssertThrow(std::isfinite(actual) &&
                          std::abs(actual - expected) < 2e-6,
                        ExcMessage("ALE Kelly field indicator: expected " +
                                   std::to_string(expected) + ", got " +
                                   std::to_string(actual)));
          }
        }
    }
  }
};

template <int dim>
void check()
{
  Parameters::BoundaryConditionsData bc;
  ParameterHandler                   prm;
  ParameterReader<dim>               param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Output
  set write vtu results = false
end
subsection FiniteElements
  set use quads = true
end
subsection Mesh
  subsection Adaptation
    set enable = true
    set strategy = local refinement
    subsection Local refinement
      set variables for adaptation = phase_tracer
    end
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  TestSolver<dim>(param).check();
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  check<2>();
  check<3>();
  deallog << "ALE Kelly indicators use physical geometry before independent "
             "field selection"
          << std::endl;
}
