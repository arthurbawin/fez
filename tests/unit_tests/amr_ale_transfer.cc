#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

// A piecewise affine tracer drives local refinement. All fields and the
// time-dependent ALE positions are exactly representable on the coarse mesh.
class State : public Function<2>
{
public:
  State(const unsigned int components, const double time)
    : Function<2>(components)
    , time(time)
  {}

  double value(const Point<2> &p, const unsigned int c) const override
  {
    if (c == 2)
      return p[0] - 0.5;
    if (c == 3)
      return 1.2 * p[0] + 0.1 * p[1] + time;
    if (c == 4)
      return 0.9 * p[1] + 0.2 * time;
    if (c == 5)
      return std::abs(p[0] - 0.5) + time;
    return 0.1 * (c + 1) + p[0] + 0.2 * p[1] + time;
  }

private:
  const double time;
};

template <bool enlarged>
class TestSolver : public CHNSSolver<2, true, enlarged>
{
public:
  using CHNSSolver<2, true, enlarged>::CHNSSolver;

  void check_boundary_geometry()
  {
    this->initialize();
    this->time_handler.current_time = 0.3;
    this->set_time();
    GridGenerator::subdivided_hyper_cube(*this->triangulation, 2);
    this->setup_dofs();
    this->setup_mappings();
    VectorTools::interpolate(*this->fixed_mapping,
                             *this->dof_handler,
                             State(this->dof_handler->get_fe().n_components(),
                                   0.3),
                             this->local_evaluation_point);
    *this->present_solution = this->local_evaluation_point;
    this->evaluation_point  = this->local_evaluation_point;
    for (auto &previous : *this->previous_solutions)
      previous = this->local_evaluation_point;
    this->adapt_mesh();
    FEFaceValues<2>           values(*this->moving_mapping,
                           this->dof_handler->get_fe(),
                           QGauss<1>(3),
                           update_values | update_quadrature_points);
    std::vector<Tensor<1, 2>> velocity(3);
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary())
          {
            values.reinit(cell, face);
            values[this->velocity_extractor].get_function_values(
              *this->present_solution, velocity);
            for (unsigned int q = 0; q < 3; ++q)
              for (unsigned int d = 0; d < 2; ++d)
                AssertThrow(std::abs(velocity[q][d] -
                                     values.quadrature_point(q)[d]) < 1e-11,
                            ExcMessage(
                              "Boundary values use the old ALE geometry"));
          }
  }

  void check_transfer()
  {
    this->initialize();
    this->time_handler.current_time = 0.3;
    this->set_time();
    GridGenerator::subdivided_hyper_cube(*this->triangulation, 2);
    this->setup_dofs();
    this->setup_mappings();
    const auto nc = this->dof_handler->get_fe().n_components();
    VectorTools::interpolate(*this->fixed_mapping,
                             *this->dof_handler,
                             State(nc, 0.3),
                             this->local_evaluation_point);
    *this->present_solution = this->local_evaluation_point;
    this->evaluation_point  = this->local_evaluation_point;
    for (unsigned int i = 0; i < this->previous_solutions->size(); ++i)
    {
      VectorTools::interpolate(*this->fixed_mapping,
                               *this->dof_handler,
                               State(nc, 0.2 - 0.1 * i),
                               this->local_evaluation_point);
      (*this->previous_solutions)[i] = this->local_evaluation_point;
    }
    this->local_evaluation_point = *this->present_solution;
    for (unsigned int step = 0; step < 3; ++step)
    {
      const auto before = this->triangulation->n_global_active_cells();
      if (step == 2)
      {
        this->param.mesh.adaptation.tree_amr.fraction_to_refine  = 0.;
        this->param.mesh.adaptation.tree_amr.fraction_to_coarsen = 1.;
      }
      this->adapt_mesh();
      const auto after = this->triangulation->n_global_active_cells();
      AssertThrow(step == 2 ? after < before : after > before,
                  ExcMessage("The test must refine and coarsen the mesh"));
      check_state(*this->present_solution, 0.3);
      check_state(this->evaluation_point, 0.3);
      for (unsigned int i = 0; i < this->previous_solutions->size(); ++i)
        check_state((*this->previous_solutions)[i], 0.2 - 0.1 * i);
      for (const auto &cell : this->dof_handler->active_cell_iterators())
        if (cell->is_locally_owned())
        {
          const Point<2> unit(0.3, 0.7);
          const auto     ref =
            this->fixed_mapping->transform_unit_to_real_cell(cell, unit);
          const auto mapped =
            this->moving_mapping->transform_unit_to_real_cell(cell, unit);
          const Point<2> expected(State(nc, 0.3).value(ref, 3),
                                  State(nc, 0.3).value(ref, 4));
          AssertThrow(mapped.distance(expected) < 1e-11,
                      ExcMessage("AMR lost the deformed ALE geometry"));
        }
    }
  }

private:
  void check_state(const LA::ParVectorType &solution, const double time)
  {
    LA::ParVectorType expected(this->locally_owned_dofs,
                               this->mpi_communicator);
    VectorTools::interpolate(*this->fixed_mapping,
                             *this->dof_handler,
                             State(this->dof_handler->get_fe().n_components(),
                                   time),
                             expected);
    expected -= solution;
    AssertThrow(expected.linfty_norm() < 1e-11,
                ExcMessage("AMR lost an ALE field or its BDF history"));
  }
};

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  Parameters::BoundaryConditionsData bc;
  bc.n_fluid_bc       = 1;
  bc.n_pseudosolid_bc = 1;
  ParameterHandler   prm;
  ParameterReader<2> param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Time integration
  set scheme = BDF2
  set t_initial = 0
  set t_end = 1
  set dt = 0.1
end
subsection Output
  set write vtu results = false
end
subsection FiniteElements
  set use quads = true
end
subsection Fluid boundary conditions
  set number = 1
  subsection boundary 0
    set id = 0
    set type = input_function
    subsection u
      set Function expression = (x-0.1*(y-0.2*t)/0.9-t)/1.2+0.2*(y-0.2*t)/0.9+t+0.1
    end
    subsection v
      set Function expression = (x-0.1*(y-0.2*t)/0.9-t)/1.2+0.2*(y-0.2*t)/0.9+t+0.2
    end
  end
end
subsection Pseudosolid boundary conditions
  set number = 1
  subsection boundary 0
    set id = 0
    set type = input_function
    subsection x
      set Function expression = 1.2*x+0.1*y+t
    end
    subsection y
      set Function expression = 0.9*y+0.2*t
    end
  end
end
subsection Mesh
  subsection Adaptation
    set enable = true
    set strategy = local refinement
    subsection Local refinement
      set variables for adaptation = phase_tracer
      set fraction to refine = 0.5
      set fraction to coarsen = 0
      set max grid level = 4
    end
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  TestSolver<false>(param).check_transfer();
  TestSolver<true>(param).check_transfer();
  param.bc_data.enforce_zero_mean_pressure = true;
  TestSolver<false>(param).check_transfer();
  TestSolver<true>(param).check_transfer();
  prm.enter_subsection("Pseudosolid boundary conditions");
  prm.enter_subsection("boundary 0");
  prm.enter_subsection("x");
  prm.set("Function expression", "1.2*x+0.1*y+t+0.1*x*x");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.leave_subsection();
  prm.enter_subsection("Fluid boundary conditions");
  prm.enter_subsection("boundary 0");
  prm.enter_subsection("u");
  prm.set("Function expression", "x");
  prm.leave_subsection();
  prm.enter_subsection("v");
  prm.set("Function expression", "y");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.leave_subsection();
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  TestSolver<false>(param).check_boundary_geometry();
  TestSolver<true>(param).check_boundary_geometry();
  deallog
    << "ALE geometry, fields and BDF history survive refinement and coarsening"
    << std::endl;
}
