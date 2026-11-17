#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fsi_solver.h>
#include <incompressible_chns_solver.h>
#include <linear_elasticity_solver.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>

#include <limits>
#include <type_traits>

#include "../tests.h"

/**
 * Test evaluation_point_is_valid() for the solvers whose mesh position is
 * part of the solution, or is the solution:
 *
 * - for each solver, the position x(X) = (s X_0, X_1, ...) stored in the
 *   evaluation point must be valid if and only if s > 0, and a NaN position
 *   must be invalid,
 * - for the linear elasticity solver, an inverted mesh still yields a finite
 *   residual, so that checking the residual for NaNs would not detect it,
 * - for a neo-Hookean compression solved by the Newton solver, an inverted
 *   full step must be rejected before RHS assembly, and the line search must
 *   recover with a half step.
 */

/**
 * Give access to the evaluation point of @p Solver after setting up its
 * mesh, dofs and mappings, without calling run(), i.e., without assembling or
 * solving.
 */
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
    // The moving mapping is built on evaluation_point, so it sees the positions
    // prescribed by set_position() below.
    if constexpr (fluid_solver)
      this->setup_mappings();
    else
    {
      // This fixture bypasses run(), which initializes the source multipliers.
      this->scratch_data->source_term_fixed_mesh_multiplier  = 1.;
      this->scratch_data->source_term_moving_mesh_multiplier = 0.;
    }
  }

  /**
   * Position x(X) = (scale * X_0, X_1, ..., X_{dim-1}) in the dim components
   * starting at @p first, and zero in the other components. The determinant
   * of its gradient is scale.
   */
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

  // Store Position(scale) in the evaluation point. It is interpolated with the
  // fixed mapping, so that X are the reference coordinates.
  void set_position(const double scale)
  {
    const Mapping<dim> *reference_mapping;
    if constexpr (fluid_solver)
      reference_mapping = this->fixed_mapping.get();
    else
      reference_mapping = this->mapping.get();
    const auto    &dofs = this->get_dof_handler();
    const Position position(dofs.get_fe_collection()[0].n_components(),
                            this->position_extractor.first_vector_component,
                            scale);
    VectorTools::interpolate(hp::MappingCollection<dim>(*reference_mapping),
                             dofs,
                             position,
                             this->local_evaluation_point);
    this->evaluation_point = this->local_evaluation_point;
  }

  // Set all owned dofs of the evaluation point to NaN on rank 0 only. The
  // other ranks keep their positions, but must reach the same decision.
  void set_nan_position()
  {
    if (this->mpi_rank == 0)
      AssertThrow(VecSet(this->local_evaluation_point,
                         std::numeric_limits<double>::quiet_NaN()) == 0,
                  ExcInternalError());
    this->evaluation_point = this->local_evaluation_point;
  }
};

/**
 * Check evaluation_point_is_valid() of @p Solver for valid, degenerate,
 * inverted and NaN positions.
 */
template <int dim, typename Solver>
void check_solver(const ParameterReader<dim> &param,
                  const char                 *name,
                  const unsigned int          degree)
{
  GeometryProblem<dim, Solver> solver(param);

  // Identity, strongly compressed (but valid), degenerate and inverted meshes
  for (const double scale : {1., .01, 0., -1.})
  {
    solver.set_position(scale);
    AssertThrow(solver.evaluation_point_is_valid() == (scale > 0.),
                ExcMessage("Incorrect deformation determinant check"));
  }
  // An inverted mesh can still have a finite residual for linear elasticity,
  // so the residual alone cannot detect the inversion.
  if constexpr (std::is_same_v<Solver, LinearElasticitySolver<dim>>)
  {
    solver.set_position(-1.);
    solver.setup_assemblers();
    solver.create_zero_constraints();
    solver.assemble_rhs();
    const double rhs_norm = solver.get_system_rhs().l2_norm();
    AssertThrow(std::isfinite(rhs_norm) && rhs_norm > 0.,
                ExcMessage(
                  "The inverted linear example must have a finite RHS"));
  }

  // NaN position on rank 0, on top of a valid mesh, so that the check can only
  // fail because of the NaN.
  // In debug builds on glibc, tests.h enables floating-point exceptions,
  // which PETSc detects at initialization. Suspend them only for the
  // intentional NaN.
  solver.set_position(1.);
  AssertThrow(PetscFPTrapPush(PETSC_FP_TRAP_OFF) == 0, ExcInternalError());
  solver.set_nan_position();
  const bool valid = solver.evaluation_point_is_valid();
  AssertThrow(PetscFPTrapPop() == 0, ExcInternalError());
  AssertThrow(!valid, ExcMessage("Non-finite deformation was accepted"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << name << " dim=" << dim << " degree=" << degree << " OK"
            << std::endl;
}

/**
 * Neo-Hookean solid on the unit square with fixed boundaries, compressed by a
 * constant body force along x. The Newton updates are the solver's own: the
 * full first step inverts the mesh, but the half step does not. The validity
 * check and the RHS assembly are instrumented to check that:
 *
 * - the first line search trial (alpha = 1) is invalid and the second
 *   (alpha = 1/2) is valid. These are the 2nd and 3rd validity checks, the
 *   1st being the one of the initial guess,
 * - an invalid evaluation point never reaches RHS assembly,
 * - the assembled residual is always finite.
 */
class CompressionProblem : public LinearElasticitySolver<2>
{
public:
  using LinearElasticitySolver<2>::LinearElasticitySolver;
  bool evaluation_point_is_valid() const override
  {
    const bool valid = LinearElasticitySolver<2>::evaluation_point_is_valid();
    ++checks;
    if (checks == 2 || checks == 3)
    {
      // Recover alpha from the trial step, which is alpha * newton_update
      // from the last accepted iterate.
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
    AssertThrow(LinearElasticitySolver<2>::evaluation_point_is_valid(),
                ExcMessage("An invalid trial reached RHS assembly"));
    LinearElasticitySolver<2>::assemble_rhs();
    AssertThrow(std::isfinite(system_rhs.l2_norm()),
                ExcMessage(
                  "The safeguarded physical residual must stay finite"));
  }
  // Number of calls to evaluation_point_is_valid() by the Newton solver
  mutable unsigned int checks = 0;
};

/**
 * Solve the CompressionProblem. The parameters set by check_geometry() (quads,
 * position degree 1, one pseudosolid) are completed and overridden by the ones
 * below.
 */
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
  // Fix the position on the four sides of the square
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
  AssertThrow(solver.checks >= 3 && solver.get_system_rhs().l2_norm() <
                                      param.nonlinear_solver.tolerance,
              ExcMessage("The physical compression did not recover/converge"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << "neo-Hookean compression: alpha=1 rejected, alpha=0.5 accepted, "
               "converged"
            << std::endl;
}

/**
 * Set up the parameters for the unit square or cube with 2 cells per
 * direction and a position field of degree @p degree, then run the checks of
 * the solvers tested for this dimension and degree.
 */
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
  // No-slip on all boundaries, named x_min, x_max, y_min, etc.
  for (unsigned int id = 0; id < 2 * dim; ++id)
  {
    param.fluid_bc[id].gmsh_name =
      std::string(1, "xyz"[id / 2]) + (id % 2 == 0 ? "_min" : "_max");
    param.fluid_bc[id].id   = id;
    param.fluid_bc[id].type = BoundaryConditions::Type::no_slip;
  }
  // Weak no-slip on x_min: the hp FSI solver then uses both of its FE
  // systems, near and away from this boundary.
  param.fluid_bc[0].type = BoundaryConditions::Type::weak_no_slip;
  // 2D, degree 1: elasticity, CHNS-ALE and the compression problem
  // 2D, degree 2: hp FSI
  // 3D, degree 1: monolithic FSI
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
