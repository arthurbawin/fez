#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_handler.h>
#include <postprocessors_and_evaluators.h>

#include <filesystem>
#include <fstream>

#include "../tests.h"

namespace
{
  template <int dim>
  class TracerState : public Function<dim>
  {
  public:
    explicit TracerState(const double shift = 0.)
      : Function<dim>(3)
      , shift(shift)
    {}
    double value(const Point<dim> &p, const unsigned int c) const override
    {
      // Other components deliberately disagree with the physical tracer.
      return c == 1 ? -1.01 + 2.02 * p[0] + 0.5 * p[0] * (1. - p[0]) + shift :
                      0.4;
    }

  private:
    const double shift;
  };

  template <int dim>
  void check(const std::string &model,
             const bool         limited,
             const std::string &method    = "direct",
             const bool         simplices = false)
  {
    Parameters::BoundaryConditionsData bc;
    ParameterHandler                   prm;
    ParameterReader<dim>               param(bc);
    param.declare(prm);
    std::istringstream input(R"(
subsection Time integration
  set scheme = BDF1
  set dt = 0.1
  set t_end = 1
end
subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density = 1000
  end
  subsection Fluid 1
    set density = 1
  end
end
subsection Postprocessing
  subsection density
    set enable = true
    set output frequency = 20
    set degree = 1
  end
end
)");
    prm.parse_input(input);
    prm.enter_subsection("Postprocessing");
    prm.enter_subsection("density");
    prm.set("computation method", method);
    prm.leave_subsection();
    prm.leave_subsection();
    prm.enter_subsection("Cahn Hilliard");
    prm.set("CHNS model", model);
    prm.set("enable tracer limiter", limited ? "true" : "false");
    prm.leave_subsection();
    param.read(prm);
    param.finite_elements.use_quads = !simplices;
    param.output.write_results      = true;
    param.output.n_subdivisions     = simplices ? 2 : 4;
    param.output.output_prefix      = "density";
    param.output.output_dir = "density_" + std::to_string(dim) + "_" + model +
                              (limited ? "_limited_" : "_unlimited_") + method +
                              (simplices ? "_simplices/" : "/");
    std::filesystem::create_directories(param.output.output_dir);

    Triangulation<dim>                  mesh;
    std::unique_ptr<FiniteElement<dim>> scalar;
    std::unique_ptr<Mapping<dim>>       mapping_ptr;
    Quadrature<dim>                     quadrature;
    if (simplices)
    {
      Triangulation<dim> cubes;
      GridGenerator::hyper_cube(cubes);
      GridGenerator::convert_hypercube_to_simplex_mesh(cubes, mesh);
      scalar      = std::make_unique<FE_SimplexP<dim>>(2);
      mapping_ptr = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
      quadrature  = QGaussSimplex<dim>(3);
    }
    else
    {
      GridGenerator::hyper_cube(mesh);
      scalar      = std::make_unique<FE_Q<dim>>(2);
      mapping_ptr = std::make_unique<MappingQ1<dim>>();
      quadrature  = QGauss<dim>(3);
    }
    const FESystem<dim> fe(*scalar, 3);
    DoFHandler<dim>     dofs(mesh);
    dofs.distribute_dofs(fe);
    const auto       &mapping = *mapping_ptr;
    ComponentOrdering ordering;
    ordering.n_components = 3;
    ordering.phi_lower    = 1;
    ordering.phi_upper    = 2;
    ordering.psi_lower    = 2;
    ordering.psi_upper    = 3;
    PostProcessingHandler<dim> handler(
      ordering, param, mesh, dofs, {{"dummy", 1}, {"tracer", 1}, {"psi", 1}});
    TimeHandler time(param.time_integration);
    TimerOutput timer(std::cout, TimerOutput::never, TimerOutput::wall_times);
    for (unsigned int frame = 0; frame < 2; ++frame)
    {
      if (frame == 1 && !simplices)
      {
        mesh.refine_global(1);
        dofs.distribute_dofs(fe);
      }
      // Reattachment/recreation is also performed after adaptation in solvers.
      handler.attach_triangulation_and_dof_handler(mesh, dofs);
      handler.create_field_postprocessors(param, mapping, quadrature, false);
      LA::ParVectorType solution(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
      VectorTools::interpolate(mapping,
                               dofs,
                               TracerState<dim>(0.05 * frame),
                               solution);
      // Check values after actual patch construction, not just the scalar
      // helper. This catches interpolation-before-clamping and wrong-component
      // errors for a quadratic tracer and both saturated tails.
      PostProcessingTools::DensityPostProcessor<dim> density(ordering, param);
      DataOut<dim>                                   patches;
      patches.attach_dof_handler(dofs);
      patches.add_data_vector(solution, {"dummy", "tracer", "psi"});
      patches.add_data_vector(solution, density);
      patches.build_patches(mapping, param.output.n_subdivisions);
      for (const auto &patch : patches.get_patches())
        for (unsigned int p = 0; p < patch.data.n_cols(); ++p)
        {
          double phi = patch.data(1, p);
          if (limited)
            phi = std::max(-1., std::min(1., phi));
          const double marker =
            model == "abels_nlm" ? std::tanh(5. * phi) / std::tanh(5.) : phi;
          const double expected = 0.5 * (1001. + 999. * marker);
          const double actual   = patch.data(3, p);
          AssertThrow(
            std::abs(actual - expected) < 3e-4 + 2e-6 * std::abs(expected),
            ExcMessage("Density does not match the visualized tracer"));
          if (limited)
            AssertThrow(actual >= 1. - 1e-4 && actual <= 1000. + 1e-4,
                        ExcMessage("Direct density outside fluid bounds"));
        }
      const std::vector<LA::ParVectorType> previous;
      time.current_time_iteration = frame;
      time.current_time           = frame * 0.1;
      handler.compute_field_postprocessors(timer, solution, previous, time);
      handler.output_fields(mapping,
                            solution,
                            time,
                            typename PostProcessingHandler<dim>::PrefixData());
    }
    unsigned int files = 0;
    for (const auto &entry :
         std::filesystem::directory_iterator(param.output.output_dir))
      if (entry.path().extension() == ".vtu")
      {
        std::ifstream     stream(entry.path(), std::ios::binary);
        const std::string data((std::istreambuf_iterator<char>(stream)), {});
        const std::string name  = "Name=\"density\"";
        const auto        first = data.find(name);
        AssertThrow(first != std::string::npos &&
                      data.find(name, first + name.size()) == std::string::npos,
                    ExcMessage(
                      "Export must contain exactly one density field"));
        ++files;
      }
    AssertThrow(files == 2, ExcMessage("Expected both density output frames"));
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  for (const std::string model : {"abels", "ding_horriche", "abels_nlm"})
    for (const bool limited : {true, false})
    {
      check<2>(model, limited);
      check<3>(model, limited);
    }
  check<2>("abels", true, "direct", true);
  check<3>("abels_nlm", true, "direct", true);
  check<2>("abels", true, "discontinuous");
  check<2>("abels", true, "l2 projection");
  check<2>("abels", true, "weighted average");
  initlog();
  deallog << "Direct density output: all CHNS models, limiter on/off, 2D/3D OK"
          << std::endl;
}
