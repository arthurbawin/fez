#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

template <bool enlarged>
class TestSolver : public CHNSSolver<2, true, enlarged>
{
public:
  using CHNSSolver<2, true, enlarged>::CHNSSolver;

  void check_initialization(const unsigned int refinements)
  {
    this->initialize();
    this->set_time();
    const Point<2> sample(0.3, 0.6);
    const double   initial_time = this->param.time_integration.t_initial;
    AssertThrow(std::abs(this->param.initial_conditions.initial_velocity->value(
                           sample, this->ordering->u_lower) -
                         (sample[0] * sample[1] + initial_time)) < 1e-12,
                ExcMessage(
                  "Prerefinement velocity uses the wrong initial time"));
    AssertThrow(std::abs(this->param.initial_conditions.initial_chns_tracer
                           ->value(sample, this->tracer_extractor.component) -
                         std::tanh((sample[0] - 0.43 - 0.1 * initial_time) /
                                   0.2)) < 1e-12,
                ExcMessage("Prerefinement phase uses the wrong initial time"));
    GridGenerator::subdivided_hyper_cube(*this->triangulation, 4, 0., 1., true);
    this->setup_dofs();
    this->setup_mappings();
    this->create_scratch_data();
    if (this->param.bc_data.enforce_zero_mean_pressure)
      this->create_zero_mean_pressure_constraints_data();
    this->create_solver_specific_constraints_data();
    this->create_zero_constraints();
    this->create_nonzero_constraints();
    this->create_sparsity_pattern();
    const auto initial_cells = this->triangulation->n_global_active_cells();
    for (unsigned int i = 0; i < refinements; ++i)
    {
      this->set_initial_conditions(false);
      this->adapt_mesh();
    }
    AssertThrow(refinements == 0 ||
                  this->triangulation->n_global_active_cells() > initial_cells,
                ExcMessage("Prerefinement must change the topology"));
    const auto reference = snapshot();
    this->initialize_solution();
    AssertThrow(snapshot() == reference,
                ExcMessage("Presolver changed the reference or topology"));

    FEValues<2>         values(*this->moving_mapping,
                       this->dof_handler->get_fe(),
                       QGauss<2>(3),
                       update_values | update_JxW_values |
                         update_quadrature_points);
    double              displacement = 0., psi_norm = 0.;
    std::vector<double> psi(9);
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        for (unsigned int q = 0; q < 9; ++q)
          AssertThrow(values.JxW(q) > 0., ExcMessage("Inverted initial cell"));
        const Point<2> unit(0.5, 0.5);
        displacement =
          std::max(displacement,
                   this->moving_mapping->transform_unit_to_real_cell(cell, unit)
                     .distance(this->fixed_mapping->transform_unit_to_real_cell(
                       cell, unit)));
        if constexpr (enlarged)
        {
          values[this->psi_extractor].get_function_values(
            *this->present_solution, psi);
          for (const auto value : psi)
            psi_norm = std::max(psi_norm, std::abs(value));
        }
      }
    AssertThrow(Utilities::MPI::max(displacement, this->mpi_communicator) >
                  1e-5,
                ExcMessage("Initial mesh was not compressed by the presolver"));
    if constexpr (enlarged)
      AssertThrow(Utilities::MPI::max(psi_norm, this->mpi_communicator) > 1e-3,
                  ExcMessage("Presolver psi was lost during initialization"));

    // Independently interpolate the prescribed phase in physical coordinates.
    LA::ParVectorType expected(this->locally_owned_dofs,
                               this->mpi_communicator);
    expected = *this->present_solution;
    VectorTools::interpolate(
      *this->moving_mapping,
      *this->dof_handler,
      *this->param.initial_conditions.initial_chns_tracer,
      expected,
      this->tracer_mask);
    this->nonzero_constraints.distribute(expected);
    expected -= *this->present_solution;
    AssertThrow(expected.linfty_norm() < 1e-11,
                ExcMessage("Initial phase uses the reference geometry"));
    expected = *this->present_solution;
    VectorTools::interpolate(*this->moving_mapping,
                             *this->dof_handler,
                             *this->param.initial_conditions.initial_velocity,
                             expected,
                             this->velocity_mask);
    this->nonzero_constraints.distribute(expected);
    expected -= *this->present_solution;
    AssertThrow(expected.linfty_norm() < 1e-11,
                ExcMessage("Initial velocity uses the reference geometry"));
    for (const auto &previous : *this->previous_solutions)
    {
      expected = previous;
      expected -= *this->present_solution;
      AssertThrow(expected.linfty_norm() < 1e-11,
                  ExcMessage("BDF history lost the compressed initial state"));
    }
    if (refinements == 2)
    {
      const double initial_mass = phase_mass();
      for (unsigned int step = 0; step < 2; ++step)
      {
        const auto before = this->triangulation->n_global_active_cells();
        this->param.mesh.adaptation.tree_amr.fraction_to_refine =
          step == 0 ? 0.3 : 0.;
        this->param.mesh.adaptation.tree_amr.fraction_to_coarsen =
          step == 0 ? 0. : 1.;
        this->adapt_mesh();
        const auto after = this->triangulation->n_global_active_cells();
        AssertThrow(step == 0 ? after > before : after < before,
                    ExcMessage("Compressed fixture must refine and coarsen"));
        const double mass = phase_mass();
        if (step == 0)
          AssertThrow(std::abs(mass - initial_mass) < 1e-10,
                      ExcMessage("Refinement changed represented phase mass"));
        // Coarsening interpolates: it need not preserve mass exactly.
        AssertThrow(std::isfinite(mass) && std::abs(mass) <= 1. + 1e-10,
                    ExcMessage("Transferred phase mass is outside its bounds"));
        expected.reinit(this->locally_owned_dofs, this->mpi_communicator);
        for (const auto &previous : *this->previous_solutions)
        {
          expected = previous;
          expected -= *this->present_solution;
          AssertThrow(expected.linfty_norm() < 1e-11,
                      ExcMessage(
                        "AMR created a spurious initial mesh velocity"));
        }
        const auto *additional_solution =
          this->time_handler.get_additional_solution();
        AssertThrow(additional_solution, ExcInternalError());
        expected = *additional_solution;
        expected -= *this->present_solution;
        AssertThrow(expected.linfty_norm() < 1e-11,
                    ExcMessage("AMR lost the additional BDF history"));
      }
      this->param.mesh.adaptation.tree_amr.fraction_to_refine  = 0.3;
      this->param.mesh.adaptation.tree_amr.fraction_to_coarsen = 0.;
    }
  }

  void check_repeated_startup()
  {
    check_startup = true;
    this->run();
    AssertThrow(initializations == 1 && postprocess_calls == 2,
                ExcMessage("Startup must initialize once and repeat once"));
  }

protected:
  void initialize_solution() override
  {
    ++initializations;
    if (check_startup)
      AssertThrow(this->triangulation->n_global_active_cells() == 16,
                  ExcMessage("Startup must presolve before prerefinement"));
    CHNSSolver<2, true, enlarged>::initialize_solution();
    if (check_startup)
    {
      std::vector<double> local;
      FEValues<2>         values(*this->moving_mapping,
                         this->dof_handler->get_fe(),
                         QTrapezoid<2>(),
                         update_values);
      std::vector<double> psi(4);
      for (const auto &cell : this->dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const auto center = cell->center();
          local.push_back(static_cast<unsigned int>(4 * center[0]) +
                          4 * static_cast<unsigned int>(4 * center[1]));
          const auto vertices = this->moving_mapping->get_vertices(cell);
          if constexpr (enlarged)
          {
            values.reinit(cell);
            values[this->psi_extractor].get_function_values(
              *this->present_solution, psi);
          }
          for (unsigned int v = 0; v < 4; ++v)
            local.insert(local.end(), {vertices[v][0], vertices[v][1], psi[v]});
        }
      coarse_geometry.resize(16 * 12);
      for (const auto &rank :
           Utilities::MPI::all_gather(this->mpi_communicator, local))
        for (unsigned int i = 0; i < rank.size(); i += 13)
          std::copy_n(rank.begin() + i + 1,
                      12,
                      coarse_geometry.begin() +
                        12 * static_cast<unsigned int>(rank[i]));
    }
  }

  void solver_specific_post_processing() override
  {
    CHNSSolver<2, true, enlarged>::solver_specific_post_processing();
    if (!check_startup)
      return;
    if (postprocess_calls == 0)
    {
      AssertThrow(this->triangulation->n_global_active_cells() > 16,
                  ExcMessage("AMR must prerefine the compressed mesh"));
      check_transferred_geometry();
      LA::ParVectorType expected(this->locally_owned_dofs,
                                 this->mpi_communicator);
      expected = *this->present_solution;
      VectorTools::interpolate(
        *this->moving_mapping,
        *this->dof_handler,
        *this->param.initial_conditions.initial_chns_tracer,
        expected,
        this->tracer_mask);
      VectorTools::interpolate(*this->moving_mapping,
                               *this->dof_handler,
                               *this->param.initial_conditions.initial_velocity,
                               expected,
                               this->velocity_mask);
      this->nonzero_constraints.distribute(expected);
      expected -= *this->present_solution;
      AssertThrow(
        expected.linfty_norm() < 1e-11,
        ExcMessage(
          "Initial fields must be refreshed on the refined physical mesh"));
      for (const auto &previous : *this->previous_solutions)
      {
        expected = previous;
        expected -= *this->present_solution;
        AssertThrow(expected.linfty_norm() < 1e-11,
                    ExcMessage(
                      "Post-presolve AMR must initialize every BDF history"));
      }
      initial_state.reinit(this->locally_owned_dofs, this->mpi_communicator);
      initial_state     = *this->present_solution;
      initial_reference = snapshot();
    }
    else
    {
      LA::ParVectorType error(this->locally_owned_dofs, this->mpi_communicator);
      error = *this->present_solution;
      error -= initial_state;
      AssertThrow(error.linfty_norm() < 1e-11,
                  ExcMessage("BDF initial-condition startup lost compression"));
      AssertThrow(snapshot() == initial_reference,
                  ExcMessage("Startup changed the reference mesh"));
      for (const auto &previous : *this->previous_solutions)
      {
        error = previous;
        error -= initial_state;
        AssertThrow(error.linfty_norm() < 1e-11,
                    ExcMessage("Startup changed an initial history"));
      }
    }
    ++postprocess_calls;
  }

private:
  bool                check_startup   = false;
  unsigned int        initializations = 0, postprocess_calls = 0;
  LA::ParVectorType   initial_state;
  std::string         initial_reference;
  std::vector<double> coarse_geometry;

  void check_transferred_geometry() const
  {
    FEValues<2>         values(*this->moving_mapping,
                       this->dof_handler->get_fe(),
                       QTrapezoid<2>(),
                       update_values);
    std::vector<double> psi(4);
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const unsigned int i = static_cast<unsigned int>(4 * cell->center()[0]);
        const unsigned int j = static_cast<unsigned int>(4 * cell->center()[1]);
        const auto         vertices = this->moving_mapping->get_vertices(cell);
        if constexpr (enlarged)
        {
          values.reinit(cell);
          values[this->psi_extractor].get_function_values(
            *this->present_solution, psi);
        }
        for (unsigned int v = 0; v < 4; ++v)
        {
          const double x = 4 * cell->vertex(v)[0] - i;
          const double y = 4 * cell->vertex(v)[1] - j;
          for (unsigned int component = 0; component < (enlarged ? 3u : 2u);
               ++component)
          {
            double expected = 0.;
            for (unsigned int k = 0; k < 4; ++k)
              expected += (k % 2 ? x : 1 - x) * (k / 2 ? y : 1 - y) *
                          coarse_geometry[12 * (i + 4 * j) + 3 * k + component];
            const double actual =
              component < 2 ? vertices[v][component] : psi[v];
            AssertThrow(
              std::abs(actual - expected) < 1e-10,
              ExcMessage(
                "Prerefinement changed the presolved geometry or psi"));
          }
        }
      }
    phase_mass(); // Also checks positive geometric Jacobians.
  }

  double phase_mass() const
  {
    FEValues<2>         values(*this->moving_mapping,
                       this->dof_handler->get_fe(),
                       QGauss<2>(3),
                       update_values | update_JxW_values | update_jacobians);
    std::vector<double> phi(9);
    double              mass = 0.;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        values.reinit(cell);
        values[this->tracer_extractor].get_function_values(
          *this->present_solution, phi);
        for (unsigned int q = 0; q < 9; ++q)
        {
          AssertThrow(values.jacobian(q).determinant() > 0.,
                      ExcMessage("AMR inverted a compressed cell"));
          mass += phi[q] * values.JxW(q);
        }
        for (const auto v : cell->vertex_indices())
          for (unsigned int d = 0; d < 2; ++d)
            AssertThrow(cell->vertex(v)[d] >= 0. && cell->vertex(v)[d] <= 1.,
                        ExcMessage(
                          "AMR moved a reference vertex outside the domain"));
      }
    return Utilities::MPI::sum(mass, this->mpi_communicator);
  }

  std::string snapshot() const
  {
    std::ostringstream out;
    out << std::setprecision(17);
    for (const auto &cell : this->triangulation->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        out << cell->id() << ' ' << cell->material_id();
        for (const auto v : cell->vertex_indices())
          out << ' ' << cell->vertex(v);
        for (const auto f : cell->face_indices())
          out << ' ' << cell->face(f)->boundary_id();
        out << '\n';
      }
    return out.str();
  }
};

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  Parameters::BoundaryConditionsData bc;
  bc.n_fluid_bc         = 4;
  bc.n_pseudosolid_bc   = 4;
  bc.n_cahn_hilliard_bc = 4;
  ParameterHandler   prm;
  ParameterReader<2> param(bc);
  param.declare(prm);
  std::istringstream input(R"(

# Integration test: an elasticity presolver pre-positions the mesh (via the
# Cahn-Hilliard "chns form" forcing built from the prescribed phase) and its
# mesh position is injected as the initial mesh of a short CHNS-ALE run.

subsection Dimension
  set dimension = 2
end

subsection Timer
  set enable timer = false
end

subsection Output
  set write vtu results = false
end

subsection Mesh
  set dealii preset mesh     = rectangle
  set dealii mesh parameters = 4, 4 : 0., 0. : 1., 1. : true
  set refinement level       = 1
  subsection Adaptation
    set enable = true
    set strategy = local refinement
    subsection Local refinement
      set variables for adaptation = phase_tracer
      set fraction to refine = 0.3
      set fraction to coarsen = 0
      set max grid level = 4
    end
  end
end

subsection Time integration
  set verbosity = quiet
  set dt        = 0.05
  set t_initial = 0.2
  set t_end     = 0.35
  set scheme    = BDF2
  set bdf start method = BDF1
  subsection Adaptation
    set enable = true
    set adaptation strategy = cfl
  end
end

subsection Nonlinear solver
  set verbosity            = quiet
  set tolerance            = 1e-9
  set divergence_tolerance = 1e+4
  set max_iterations       = 25
  set enable_line_search   = true
  set analytic_jacobian    = true
end

subsection Linear solver
  subsection main physics
    set verbosity = quiet
    set method    = direct_mumps
    set reuse     = false
  end
  subsection elasticity
    set verbosity = quiet
    set method    = direct_mumps
    set reuse     = false
  end
end

subsection FiniteElements
  set use quads = true
  set Velocity degree      = 2
  set Pressure degree      = 1
  set Mesh position degree = 1
  set Tracer degree        = 1
  set Potential degree     = 1
end

subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density             = 1
    set kinematic viscosity = 1
  end
  subsection Fluid 1
    set density             = 1
    set kinematic viscosity = 1
  end
  set number of pseudosolids = 1
  subsection Pseudosolid 0
    set constitutive model = neo hookean
    subsection lame lambda
      set Function expression = 1
    end
    subsection lame mu
      set Function expression = 1
    end
  end
  set body force = 0, 0
end

subsection Cahn Hilliard
  set mobility model                 = constant
  set mobility                       = 0.1
  set surface tension                = 1
  set interface thickness            = 0.2
  set mff source term                = chns form
  set mff physics compression factor = 0.5
  set mff transport factor           = 0
  set mff regularization gamma       = 0.8
  set use presolver                  = true
end

subsection Initial conditions
  set to mms = false
  subsection velocity
    set Function expression = x*y+t; -0.5*y*y
  end
  subsection cahn hilliard tracer
    set Function expression = tanh((x - 0.43 - 0.1*t) / 0.2)
  end
end

subsection Elasticity
  subsection current mesh source term
    set enable = false
  end
  subsection presolver
    set initial compression multiplier = 0.1
    set continuation steps             = 4
  end
end

subsection Fluid boundary conditions
  set number = 4
  set fix pressure constant      = true
  set enforce zero mean pressure = false
  subsection boundary 0
    set id   = 0
    set name = x_min
    set type = no_slip
  end
  subsection boundary 1
    set id   = 1
    set name = x_max
    set type = no_slip
  end
  subsection boundary 2
    set id   = 2
    set name = y_min
    set type = no_slip
  end
  subsection boundary 3
    set id   = 3
    set name = y_max
    set type = no_slip
  end
end

subsection Pseudosolid boundary conditions
  set number = 4
  subsection boundary 0
    set id   = 0
    set name = x_min
    set type = fixed
  end
  subsection boundary 1
    set id   = 1
    set name = x_max
    set type = fixed
  end
  subsection boundary 2
    set id   = 2
    set name = y_min
    set type = fixed
  end
  subsection boundary 3
    set id   = 3
    set name = y_max
    set type = fixed
  end
end

subsection CahnHilliard boundary conditions
  set number = 4
  subsection boundary 0
    set id   = 0
    set name = x_min
    set type = no_flux
  end
  subsection boundary 1
    set id   = 1
    set name = x_max
    set type = no_flux
  end
  subsection boundary 2
    set id   = 2
    set name = y_min
    set type = no_flux
  end
  subsection boundary 3
    set id   = 3
    set name = y_max
    set type = no_flux
  end
end

subsection Manufactured solution
  set enable = false
end

)");
  prm.parse_input(input);
  param.read(prm);
  for (const unsigned int refinements : {0, 2})
  {
    TestSolver<false>(param).check_initialization(refinements);
    TestSolver<true>(param).check_initialization(refinements);
  }
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = true;
  TestSolver<false>(param).check_initialization(0);
  TestSolver<true>(param).check_initialization(0);
  param.bc_data.fix_pressure_constant      = true;
  param.bc_data.enforce_zero_mean_pressure = false;
  using Time                               = Parameters::TimeIntegration;
  param.time_integration.scheme            = Time::Scheme::BDF1;
  TestSolver<false>(param).check_initialization(0);
  TestSolver<true>(param).check_initialization(0);
  param.time_integration.scheme   = Time::Scheme::BDF2;
  param.time_integration.bdfstart = Time::BDFStart::initial_condition;
  TestSolver<false>(param).check_initialization(0);
  TestSolver<true>(param).check_initialization(0);
  param.time_integration.bdf_starting_step_ratio = 1.;
  param.time_integration.t_end =
    param.time_integration.t_initial + param.time_integration.dt;
  param.mesh.adaptation.tree_amr.n_prerefinement_steps = 2;
  TestSolver<false>(param).check_repeated_startup();
  TestSolver<true>(param).check_repeated_startup();
  using Strategy = Parameters::Mesh::Adaptation::TreeAMR::RefinementStrategy;
  param.mesh.adaptation.tree_amr.refinement_strategy = Strategy::InterfaceBand;
  param.mesh.adaptation.tree_amr.interface_band_half_width_over_epsilon = 0.5;
  param.mesh.adaptation.tree_amr.interface_band_diameter_over_epsilon   = 1.;
  TestSolver<false>(param).check_repeated_startup();
  TestSolver<true>(param).check_repeated_startup();
  using Mode = Parameters::Elasticity::PresolvedMeshPositionMode;
  for (const auto mode : {Mode::reuse, Mode::force_recompute})
  {
    param.elasticity.presolved_mesh_position_mode = mode;
    bool rejected                                 = false;
    try
    {
      TestSolver<false> solver(param);
    }
    catch (const ExceptionBase &)
    {
      rejected = true;
    }
    AssertThrow(rejected,
                ExcMessage("AMR presolver must reject its old cache"));
  }
  param.elasticity.presolved_mesh_position_mode = Mode::off;
  param.elasticity.write_final_msh              = true;
  bool rejected                                 = false;
  try
  {
    TestSolver<true> solver(param);
  }
  catch (const ExceptionBase &)
  {
    rejected = true;
  }
  AssertThrow(rejected, ExcMessage("AMR presolver must reject Gmsh export"));
  deallog << "Presolver compression, phase, psi and BDF history survive AMR "
             "initialization"
          << std::endl;
}
