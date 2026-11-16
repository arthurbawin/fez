#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fsi_solver.h>
#include <incompressible_chns_solver.h>
#include <linear_elasticity_solver.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>

#include <limits>
#include <type_traits>

#include "../tests.h"

// Exercise each real solver's hook without assembling a constitutive law.
template <int dim, typename Solver>
class GeometryProblem : public Solver
{
  static constexpr bool fluid_solver =
    std::is_base_of_v<NavierStokesSolver<dim, true>, Solver>;

public:
  GeometryProblem(const ParameterReader<dim> &param)
    : Solver(param)
  {
    this->pcout.set_condition(false);
    if constexpr (fluid_solver)
      MeshTools::read_mesh(*this->triangulation, this->param);
    else
      MeshTools::read_mesh(this->triangulation, this->param);
    this->setup_dofs();
    if constexpr (!fluid_solver)
    {
      // This fixture bypasses run(), which initializes the source multipliers.
      this->scratch_data->source_term_fixed_mesh_multiplier  = 1.;
      this->scratch_data->source_term_moving_mesh_multiplier = 0.;
    }
  }

  double rhs_norm() const { return this->system_rhs.l2_norm(); }

  class Position : public Function<dim>
  {
  public:
    Position(unsigned int components, unsigned int first, double scale)
      : Function<dim>(components)
      , first(first)
      , scale(scale)
    {}
    double value(const Point<dim>  &p,
                 const unsigned int component) const override
    {
      if (component < first || component >= first + dim)
        return 0.;
      return component == first ? scale * p[0] : p[component - first];
    }
    const unsigned int first;
    const double       scale;
  };

  void set_position(const double scale)
  {
    const auto    &dofs    = this->get_dof_handler();
    const auto    &mapping = reference_mapping();
    const Position position(dofs.get_fe_collection()[0].n_components(),
                            this->position_extractor.first_vector_component,
                            scale);
    VectorTools::interpolate(hp::MappingCollection<dim>(mapping),
                             dofs,
                             position,
                             this->local_evaluation_point);
    this->evaluation_point = this->local_evaluation_point;
  }

  void set_nonfinite_position(const double value)
  {
    // Only one rank supplies non-finite dofs. The decision must be collective.
    if (this->mpi_rank == 0)
      AssertThrow(VecSet(this->local_evaluation_point, value) == 0,
                  ExcInternalError());
    this->evaluation_point = this->local_evaluation_point;
  }

  const Mapping<dim> &reference_mapping() const
  {
    if constexpr (fluid_solver)
      return *this->fixed_mapping;
    else
      return *this->mapping;
  }
};

template <int dim, typename Solver>
void check_solver(const ParameterReader<dim> &param,
                  const char                 *name,
                  const unsigned int          degree)
{
  GeometryProblem<dim, Solver> solver(param);
  for (const double scale : {1., .01, 0., -1.})
  {
    solver.set_position(scale);
    AssertThrow(solver.is_trial_admissible() == (scale > 0.),
                ExcMessage("Incorrect deformation determinant check"));
  }
  // An inverted linear deformation can still have a finite residual.
  if constexpr (std::is_same_v<Solver, LinearElasticitySolver<dim>>)
  {
    solver.setup_assemblers();
    solver.create_zero_constraints();
    solver.assemble_rhs();
    AssertThrow(std::isfinite(solver.rhs_norm()) && solver.rhs_norm() > 0.,
                ExcMessage(
                  "The inverted linear example must have a finite RHS"));
  }
  for (const double value : {std::numeric_limits<double>::quiet_NaN()})
  {
    solver.set_position(1.);
    // Suspend traps only for the intentional NaN, using PETSc's trap state.
    AssertThrow(PetscFPTrapPush(PETSC_FP_TRAP_OFF) == 0, ExcInternalError());
    solver.set_nonfinite_position(value);
    const bool admissible = solver.is_trial_admissible();
    AssertThrow(PetscFPTrapPop() == 0, ExcInternalError());
    AssertThrow(!admissible, ExcMessage("Non-finite deformation was accepted"));
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << name << " dim=" << dim << " degree=" << degree << " OK"
            << std::endl;
}

// Actual neo-Hookean residual and Jacobian, with a constant body force.
// Only instrument the hook/assembly boundary: do not prescribe Newton updates.
class CompressionProblem : public LinearElasticitySolver<2>
{
public:
  using LinearElasticitySolver<2>::LinearElasticitySolver;
  double rhs_norm() const { return system_rhs.l2_norm(); }
  bool   is_trial_admissible() const override
  {
    const bool valid = LinearElasticitySolver<2>::is_trial_admissible();
    ++checks;
    if (checks == 2 || checks == 3)
    {
      LA::ParVectorType increment(local_evaluation_point);
      increment.add(-1., present_solution);
      const double alpha = increment.l2_norm() / newton_update.l2_norm();
      AssertThrow(std::abs(alpha - (checks == 2 ? 1. : .5)) < 1e-10 &&
                    valid == (checks == 3),
                  ExcMessage(
                    "Expected an inverted full step, then a valid half step"));
    }
    return valid;
  }
  void assemble_rhs() override
  {
    AssertThrow(LinearElasticitySolver<2>::is_trial_admissible(),
                ExcMessage("An invalid trial reached RHS assembly"));
    LinearElasticitySolver<2>::assemble_rhs();
    AssertThrow(std::isfinite(system_rhs.l2_norm()),
                ExcMessage(
                  "The safeguarded physical residual must stay finite"));
  }
  mutable unsigned int checks = 0;
};

void check_compression(ParameterHandler &prm, ParameterReader<2> &param)
{
  prm.parse_input_from_string(R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
end
subsection Mesh
  set dealii mesh parameters = 4,4:0,0:1,1:true
end
subsection Time integration
  set scheme = stationary
end
subsection Nonlinear solver
  set verbosity = quiet
  set tolerance = 1e-9
  set max_iterations = 20
end
subsection Linear solver
  subsection linear elasticity
    set method = direct_mumps
    set reuse = false
  end
end
subsection Physical properties
  subsection Pseudosolid 0
    set constitutive model = neo hookean
    subsection lame lambda
      set Function expression = 0.002
    end
    subsection lame mu
      set Function expression = 0.5
    end
  end
end
subsection Linear elasticity
  subsection source term
    set Function expression = 5; 0
  end
end
)");
  param.read(prm);
  for (unsigned int id = 0; id < 4; ++id)
  {
    param.pseudosolid_bc[id].id = id;
    param.pseudosolid_bc[id].gmsh_name =
      std::string(1, "xy"[id / 2]) + (id % 2 == 0 ? "_min" : "_max");
    param.pseudosolid_bc[id].type = BoundaryConditions::Type::fixed;
  }
  CompressionProblem solver(param);
  solver.pcout.set_condition(false);
  solver.run();
  AssertThrow(solver.checks >= 3 &&
                solver.rhs_norm() < param.nonlinear_solver.tolerance,
              ExcMessage("The physical compression did not recover/converge"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << "neo-Hookean compression: alpha=1 rejected, alpha=0.5 accepted, "
               "converged"
            << std::endl;
}

template <int dim>
void check_geometry(const unsigned int degree)
{
  Parameters::BoundaryConditionsData boundaries;
  ParameterReader<dim>               param(boundaries);
  ParameterHandler                   prm;
  param.declare(prm);
  prm.enter_subsection("Dimension");
  prm.set("dimension", std::to_string(dim));
  prm.leave_subsection();
  prm.enter_subsection("Mesh");
  prm.set("dealii preset mesh", "rectangle");
  prm.set("dealii mesh parameters",
          dim == 2 ? "2, 2 : 0, 0 : 1, 1 : true" :
                     "2, 2, 2 : 0, 0, 0 : 1, 1, 1 : true");
  prm.leave_subsection();
  prm.enter_subsection("FiniteElements");
  prm.set("use quads", "true");
  prm.set("Mesh position degree", std::to_string(degree));
  prm.leave_subsection();
  prm.enter_subsection("Physical properties");
  prm.set("number of pseudosolids", "1");
  prm.enter_subsection("Pseudosolid 0");
  for (const char *coefficient : {"lame lambda", "lame mu"})
  {
    prm.enter_subsection(coefficient);
    prm.set("Function expression", "1");
    prm.leave_subsection();
  }
  prm.leave_subsection();
  prm.leave_subsection();
  param.read(prm);
  param.stabilization.enable_supg        = false;
  param.stabilization.enable_tracer_supg = false;
  param.debug.fsi_coupling_option        = 1;
  // The hp FSI solver uses both FE systems near/away from this boundary.
  for (unsigned int id = 0; id < 2 * dim; ++id)
  {
    param.fluid_bc[id].gmsh_name =
      std::string(1, "xyz"[id / 2]) + (id % 2 == 0 ? "_min" : "_max");
    param.fluid_bc[id].id   = id;
    param.fluid_bc[id].type = BoundaryConditions::Type::no_slip;
  }
  param.fluid_bc[0].type = BoundaryConditions::Type::weak_no_slip;
  if constexpr (dim == 2)
  {
    if (degree == 1)
    {
      check_solver<dim, LinearElasticitySolver<dim>>(param,
                                                     "elasticity",
                                                     degree);
      check_solver<dim, CHNSSolver<dim, true>>(param, "CHNS-ALE", degree);
      check_compression(prm, param);
    }
    else
      check_solver<dim, FSISolverLessLambda<dim>>(param, "hp FSI", degree);
  }
  else
    check_solver<dim, FSISolver<dim>>(param, "monolithic FSI", degree);
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  check_geometry<2>(1);
  check_geometry<3>(1);
  check_geometry<2>(2);
}
