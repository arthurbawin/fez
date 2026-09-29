#pragma once

#include <assembly/incompressible_chns_assemblers.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

namespace
{
  unsigned int failed_directions = 0;

  template <bool moving>
  class State : public Function<2>
  {
  public:
    State(const unsigned int components,
          const unsigned int direction = numbers::invalid_unsigned_int)
      : Function<2>(components)
      , direction(direction)
    {}

    double value(const Point<2> &p, const unsigned int c) const override
    {
      const double x = p[0], y = p[1];
      if (direction != numbers::invalid_unsigned_int)
        return c == direction ?
                 0.3 + 0.2 * x - 0.1 * y + 0.15 * x * y + 0.1 * x * x :
                 0.;
      if (c == 0)
        return 0.4 + 0.3 * x + 0.2 * y * y;
      if (c == 1)
        return -0.2 + 0.2 * y - 0.1 * x * x;
      if (c == 2)
        return 0.2 * x - 0.1 * y;
      if constexpr (moving)
      {
        if (c == 3)
          return x + 0.08 * x * y;
        if (c == 4)
          return y + 0.05 * x * x;
      }
      if (c == (moving ? 5 : 3))
        return -0.3 + 0.5 * x + 0.1 * y * y;
      if (c == (moving ? 6 : 4))
        return 0.1 + 0.3 * x * x + 0.2 * y * y + 0.1 * x * y;
      return 0.1 + 0.2 * x;
    }

  private:
    const unsigned int direction;
  };

  template <bool moving, bool enlarged = false>
  class TestSolver : public CHNSSolver<2, moving, enlarged>
  {
  public:
    using CHNSSolver<2, moving, enlarged>::CHNSSolver;

    void check(const std::string &label)
    {
      this->initialize();
      GridGenerator::hyper_cube(*this->triangulation);
      this->setup_dofs();
      this->setup_mappings();
      this->create_scratch_data();
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();
      const auto        &fe           = this->dof_handler->get_fe();
      const unsigned int n_components = fe.n_components();
      VectorTools::interpolate(*this->fixed_mapping,
                               *this->dof_handler,
                               State<moving>(n_components),
                               this->local_evaluation_point);
      this->evaluation_point  = this->local_evaluation_point;
      *this->present_solution = this->local_evaluation_point;
      for (auto &previous : *this->previous_solutions)
        previous = this->local_evaluation_point;
      ConditionalOStream quiet(std::cout, false);
      this->time_handler.advance(quiet);
      this->set_time();

      using Scratch = NavierStokesScratch::ScratchDataCHNS<2, moving, enlarged>;
      std::vector<
        std::unique_ptr<Assembly::AssemblerBase<Scratch, CopyDataBase<>>>>
        assemblers;
      Assembly::IncompressibleCHNS::
        setup_assemblers<2, Scratch, CopyDataBase<>, moving, enlarged>(
          this->param, *this->ordering, this->coupling_table, assemblers);
      const auto &volume  = *assemblers.front();
      auto       &scratch = *this->scratch_data;
      // The accepted-state global coefficient is frozen throughout Newton.
      scratch.profile_correction_reference_mobility = 0.37;
      const auto cell   = this->dof_handler->begin_active();
      auto       reinit = [&]() {
        scratch.reinit(cell,
                       this->evaluation_point,
                       *this->previous_solutions,
                       *this->source_terms,
                       *this->exact_solution);
      };
      reinit();
      const auto     tau_u   = scratch.tau_supg_velocity;
      const auto     tau_phi = scratch.tau_supg_tracer;
      CopyDataBase<> copy(fe);
      copy.local_matrix() = 0.;
      volume.assemble_matrix(scratch, copy);
      const FullMatrix<double>             matrix(copy.local_matrix());
      std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
      cell->get_dof_indices(indices);
      LA::ParVectorType base(this->locally_owned_dofs, this->mpi_communicator);
      LA::ParVectorType direction(this->locally_owned_dofs,
                                  this->mpi_communicator);
      base = this->local_evaluation_point;

      // Separate component directions prevent the large momentum/time terms
      // from hiding a missing tracer or pressure coupling.
      for (unsigned int component = 0; component < n_components; ++component)
      {
        VectorTools::interpolate(*this->fixed_mapping,
                                 *this->dof_handler,
                                 State<moving>(n_components, component),
                                 direction);
        Vector<double> local_direction(indices.size()),
          analytic(indices.size());
        for (unsigned int j = 0; j < indices.size(); ++j)
          local_direction[j] = direction[indices[j]];
        matrix.vmult(analytic, local_direction);
        double             best_error = std::numeric_limits<double>::max();
        std::ostringstream diagnostics;
        diagnostics << std::scientific << std::setprecision(8);
        for (const double h : {2e-5, 5e-6, 1e-6})
        {
          auto rhs = [&](const double step) {
            this->local_evaluation_point = base;
            this->local_evaluation_point.add(step, direction);
            this->evaluation_point = this->local_evaluation_point;
            reinit();
            // Production deliberately linearizes with frozen stabilization.
            scratch.tau_supg_velocity = tau_u;
            scratch.tau_supg_tracer   = tau_phi;
            copy.local_rhs()          = 0.;
            volume.assemble_rhs(scratch, copy);
            return Vector<double>(copy.local_rhs());
          };
          Vector<double> numerical = rhs(h);
          numerical -= rhs(-h);
          numerical *= -0.5 / h;
          double maximum_error = 0.;
          for (unsigned int row_component = 0; row_component < n_components;
               ++row_component)
          {
            double error_sq = 0., scale_sq = 0., numerical_sq = 0.;
            for (unsigned int i = 0; i < indices.size(); ++i)
              if (fe.system_to_component_index(i).first == row_component)
              {
                error_sq +=
                  Utilities::fixed_power<2>(numerical[i] - analytic[i]);
                scale_sq += Utilities::fixed_power<2>(analytic[i]);
                numerical_sq += Utilities::fixed_power<2>(numerical[i]);
              }
            const double row_error =
              std::sqrt(error_sq) / std::max(1e-8, std::sqrt(scale_sq));
            maximum_error = std::max(maximum_error, row_error);
            if (row_error >= 2e-7)
              diagnostics << "  h=" << h << " row=" << row_component
                          << " error_norm=" << std::sqrt(error_sq)
                          << " analytic_norm=" << std::sqrt(scale_sq)
                          << " numerical_norm=" << std::sqrt(numerical_sq)
                          << " relative_error=" << row_error << '\n';
          }
          best_error = std::min(best_error, maximum_error);
        }
        if (best_error >= 2e-7)
        {
          ++failed_directions;
          std::cerr << label << ": column component=" << component
                    << " best directional error=" << best_error << '\n'
                    << diagnostics.str();
        }
      }
    }
  };

  template <bool moving, bool enlarged = false>
  void run_case(const std::string &model,
                const std::string &correction,
                const bool         momentum_supg,
                const bool         tracer_supg,
                const std::string &mobility_model = "adaptative_mobility_3",
                const double       gradient_coefficient = 0.)
  {
    Parameters::BoundaryConditionsData bc;
    ParameterHandler                   prm;
    ParameterReader<2>                 param(bc);
    param.declare(prm);
    std::ostringstream input;
    input << R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
end
subsection Time integration
  set scheme = BDF1
  set dt = 0.2
  set t_end = 1
  set verbosity = quiet
end
subsection FiniteElements
  set use quads = true
  set Velocity degree = 2
  set Pressure degree = 1
  set Mesh position degree = 2
  set Tracer degree = 2
  set Potential degree = 2
end
subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density = 3
    set kinematic viscosity = 0.05
  end
  subsection Fluid 1
    set density = 1
    set kinematic viscosity = 0.05
  end
  set number of pseudosolids = 1
  subsection Pseudosolid 0
    subsection lame lambda
      set Function expression = 1
    end
    subsection lame mu
      set Function expression = 1
    end
  end
end
subsection Cahn Hilliard
  set adaptive mobility n = 2
  set adaptive mobility delta = 0.15
  set adaptive mobility 3 n = 2
  set adaptive mobility 3 delta = 0.15
  set surface tension = 1
  set interface thickness = 0.5
  set enable tracer limiter = false
  set enable mobility tracer limiter = false
  set profile correction strength = 0.2
)";
    input << "  set mobility model = " << mobility_model << '\n'
          << "  set adaptive mobility m = " << gradient_coefficient << '\n'
          << "  set CHNS model = " << model << '\n'
          << "  set interface profile correction = " << correction << "\nend\n"
          << "subsection Stabilization\n  set enable supg = "
          << (momentum_supg ? "true" : "false")
          << "\n  set enable tracer supg = " << (tracer_supg ? "true" : "false")
          << "\nend\n";
    std::istringstream parameters(input.str());
    prm.parse_input(parameters);
    param.read(prm);
    param.bc_data.fix_pressure_constant      = false;
    param.bc_data.enforce_zero_mean_pressure = false;
    const std::string label =
      model + "/" + correction + (moving ? "/ALE" : "/fixed") +
      (enlarged ? "/enlarged" : "") + (momentum_supg ? "/momentum-SUPG" : "") +
      (tracer_supg ? "/tracer-SUPG" : "");
    const std::string mobility_label =
      mobility_model == "adaptative_mobility_3" ?
        label :
        mobility_model + "/m=" + std::to_string(gradient_coefficient) + "/" +
          label;
    TestSolver<moving, enlarged>(param).check(mobility_label);
  }
} // namespace
