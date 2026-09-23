#include <assembly/incompressible_chns_assemblers.h>
#include <copy_data.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <scratch_data.h>

#include <array>

#include "../tests.h"

// Check the actual CHNS cell Jacobian, including curved mesh perturbations.
// Central differences keep tau fixed, as assumed by the Newton linearization.
template <int dim>
class Fields : public Function<dim>
{
public:
  Fields(const double time)
    : Function<dim>(2 * dim + 3)
    , time(time)
  {}

  double value(const Point<dim> &p, const unsigned int c = 0) const override
  {
    const ComponentOrderingCHNS<dim, true> ordering;
    if (ordering.is_velocity(c))
      return 0.2 * (c + 1) + (0.3 + time) * p[c] * p[c] + 0.1 * p[0] * p[1];
    if (ordering.is_position(c))
    {
      const unsigned int d = c - ordering.x_lower;
      return (1. + 0.1 * (d + 1)) * p[d] + 0.07 * p[(d + 1) % dim] +
             (0.03 + 0.02 * time) * p[d] * p[d] + 0.04 * time;
    }
    if (ordering.is_pressure(c))
      return 0.4 * p[0] + 0.2 * p[1] * p[1];
    if (ordering.is_tracer(c))
      return -0.3 + 0.2 * p[0] * p[1] + 0.1 * time * p[0];
    return 0.1 + 0.3 * p[0] * p[0] - 0.2 * p[1] * p[1] + 0.07 * p[0] * p[1];
  }

private:
  const double time;
};

// Exercise spatial source derivatives when that optional ALE path is enabled.
// The default build freezes sources in its Jacobian, so use constant sources.
template <int dim>
class Source : public Function<dim>
{
public:
  Source(const unsigned int n_components)
    : Function<dim>(n_components)
  {}

  double value(const Point<dim> &p, const unsigned int c = 0) const override
  {
#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
    return 0.13 + 0.04 * (c + 1) * (p[0] + 0.2 * p[1]);
#else
    return 0.13;
#endif
  }

  Tensor<1, dim> gradient(const Point<dim> &,
                          const unsigned int c = 0) const override
  {
    Tensor<1, dim> result;
#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
    result[0] = 0.04 * (c + 1);
    result[1] = 0.008 * (c + 1);
#endif
    return result;
  }
};

template <int dim>
void test_jacobian(const bool supg, const bool tracer_supg)
{
  using Scratch = NavierStokesScratch::ScratchDataCHNS<dim, true>;
  using Copy    = CopyDataBase<1>;
  const ComponentOrderingCHNS<dim, true>   ordering;
  const Parameters::BoundaryConditionsData bc_data{};
  ParameterReader<dim>                     param(bc_data);
  ParameterHandler                         prm;
  param.finite_elements.declare_parameters(prm);
  param.physical_properties.declare_parameters(prm);
  param.time_integration.declare_parameters(prm);
  param.stabilization.declare_parameters(prm);
  param.cahn_hilliard.declare_parameters(prm);
  prm.parse_input_from_string(R"(
subsection Physical properties
  set number of fluids = 2
  set number of pseudosolids = 1
  subsection Fluid 0
    set density = 1.3
    set kinematic viscosity = 0.7
  end
  subsection Fluid 1
    set density = 0.8
    set kinematic viscosity = 1.1
  end
end
subsection Time integration
  set verbosity = quiet
  set scheme = BDF2
  set bdf start method = BDF1
  set dt = 0.2
  set t_initial = 0
  set t_end = 1
end
subsection Cahn Hilliard
  set mobility = 0.4
  set interface thickness = 0.7
  set surface tension = 1.1
  set enable tracer limiter = false
end
)");
  param.finite_elements.read_parameters(prm);
  param.physical_properties.read_parameters(prm);
  param.time_integration.read_parameters(prm);
  param.stabilization.read_parameters(prm);
  param.cahn_hilliard.read_parameters(prm);
  param.finite_elements.use_quads         = true;
  param.finite_elements.velocity_degree   = 2;
  param.time_integration.n_time_intervals = 1;
  param.stabilization.enable_supg         = supg;
  param.stabilization.enable_tracer_supg  = tracer_supg;
  param.fluid_bc[0].type                  = BoundaryConditions::Type::no_slip;

  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 3);
  const FESystem<dim> fe(FE_Q<dim>(2), ordering.n_components);
  DoFHandler<dim>     dofs(triangulation);
  dofs.distribute_dofs(fe);
  Vector<double> solution(dofs.n_dofs());
  VectorTools::interpolate(dofs, Fields<dim>(0.), solution);
  TimeHandler                 time(param.time_integration);
  std::vector<Vector<double>> history(time.n_previous_solutions, solution);
  MappingQ<dim>               fixed_mapping(1);
  MappingFEField<dim, dim, Vector<double>> moving_mapping(
    dofs,
    solution,
    fe.component_mask(FEValuesExtractors::Vector(ordering.x_lower)));
  Scratch scratch(ordering,
                  fe,
                  fixed_mapping,
                  moving_mapping,
                  QGauss<dim>(3),
                  QGauss<dim - 1>(3),
                  time,
                  param);

  Table<2, DoFTools::Coupling> coupling(ordering.n_components,
                                        ordering.n_components);
  for (unsigned int i = 0; i < ordering.n_components; ++i)
    for (unsigned int j = 0; j < ordering.n_components; ++j)
      coupling[i][j] = DoFTools::always;
  using namespace Assembly::IncompressibleCHNS;
  // Direct assemblers let this test expose missing derivatives on the old
  // code, independently of the solver's unsupported-combination guard.
  std::unique_ptr<Assembly::AssemblerBase<Scratch, Copy>> assembler;
  if (supg && tracer_supg)
    assembler = std::make_unique<
      VolumeAssembler<dim,
                      Scratch,
                      Copy,
                      moving_mesh | stabilization | tracer_stabilization>>(
      ordering, coupling);
  else if (supg)
    assembler = std::make_unique<
      VolumeAssembler<dim, Scratch, Copy, moving_mesh | stabilization>>(
      ordering, coupling);
  else if (tracer_supg)
    assembler = std::make_unique<
      VolumeAssembler<dim, Scratch, Copy, moving_mesh | tracer_stabilization>>(
      ordering, coupling);
  else
    assembler =
      std::make_unique<VolumeAssembler<dim, Scratch, Copy, moving_mesh>>(
        ordering, coupling);

  const Source<dim> source(ordering.n_components);
  // Test volume terms on an interior cell; 3D hexahedral pseudo-solid face
  // assembly is not implemented on master and is outside this test's scope.
  auto cell = dofs.begin_active();
  while (cell != dofs.end() && cell->at_boundary())
    ++cell;
  AssertThrow(cell != dofs.end(), ExcInternalError());
  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  cell->get_dof_indices(indices);
  const double epsilon = 2.e-6;
  for (unsigned int step = 0; step < 2; ++step)
  {
    time.advance(ConditionalOStream(std::cout, false));
    VectorTools::interpolate(dofs, Fields<dim>(time.current_time), solution);
    scratch.reinit(cell, solution, history, source, source);
    const auto tau        = scratch.tau_supg_velocity;
    const auto tau_tracer = scratch.tau_supg_tracer;
    Copy       matrix(fe);
    assembler->assemble_matrix(scratch, matrix);
    std::array<double, 4> errors = {};
    for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
    {
      if (!ordering.is_position(fe.system_to_component_index(j).first))
        continue;
      const double saved = solution[indices[j]];
      Copy         plus(fe), minus(fe);
      solution[indices[j]] = saved + epsilon;
      scratch.reinit(cell, solution, history, source, source);
      scratch.tau_supg_velocity = tau;
      scratch.tau_supg_tracer   = tau_tracer;
      assembler->assemble_rhs(scratch, plus);
      solution[indices[j]] = saved - epsilon;
      scratch.reinit(cell, solution, history, source, source);
      scratch.tau_supg_velocity = tau;
      scratch.tau_supg_tracer   = tau_tracer;
      assembler->assemble_rhs(scratch, minus);
      solution[indices[j]] = saved;
      for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
      {
        const unsigned int c = fe.system_to_component_index(i).first;
        if (ordering.is_position(c))
          continue;
        const unsigned int row = ordering.is_velocity(c) ? 0 :
                                 ordering.is_pressure(c) ? 1 :
                                 ordering.is_tracer(c)   ? 2 :
                                                           3;
        // The assembled right-hand side is minus the residual.
        const double fd =
          -(plus.local_rhs()[i] - minus.local_rhs()[i]) / (2. * epsilon);
        const double analytic = matrix.local_matrix()(i, j);
        errors[row] =
          std::max(errors[row], std::abs(analytic - fd) / (1. + std::abs(fd)));
      }
    }
    for (unsigned int row = 0; row < errors.size(); ++row)
      AssertThrow(errors[row] < 2.e-7,
                  ExcMessage(
                    std::to_string(dim) + "D row " + std::to_string(row) +
                    " mesh Jacobian error: " + std::to_string(errors[row])));
    deallog << dim << "D SUPG=" << supg << " tracer=" << tracer_supg
            << " step=" << step + 1 << ": OK" << std::endl;
    history[1] = history[0];
    history[0] = solution;
  }
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const auto supg : {false, true})
    for (const auto tracer : {false, true})
    {
      test_jacobian<2>(supg, tracer);
      test_jacobian<3>(supg, tracer);
    }
}
