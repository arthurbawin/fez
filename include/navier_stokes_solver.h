#ifndef NAVIER_STOKES_SOLVER_H
#define NAVIER_STOKES_SOLVER_H

#include <components_ordering.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/lac/affine_constraints.h>
#include <generic_solver.h>
#include <mumps_solver.h>
#include <parameter_reader.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * A base class for Navier-Stokes solvers.
 *
 * This class handles the common tasks to all Navier-Stokes related solvers,
 * namely :
 *
 *  - creating and handling the mesh (fixed and/or moving)
 *  - creating the system parallel matrix and vectors
 *  - creating the mesh- and NS-related boundary conditions (constraints)
 *  - advancing the time integration and solving the nonlinear system
 *  - computing errors on velocity, pressure and mesh position, if applicable
 *
 * Solver-specific tasks (additional boundary conditions, outputting results,
 * computing errors, etc.) are handled by overloading the "solver_specific"
 * routines, which by default do nothing in this base class.
 * Other functions, such as setup_dofs, are marked virtual to allow overloading
 * by solvers requiring a specific treatment, but not all solvers are required
 * to overload these functions.
 *
 * This class does *not* handle the following, which must then be
 * implemented by each derived solver:
 *
 * - creation of the FiniteElements (e.g., FESystem or FECollection), which
 *   depend on the fields layout and specificities of each solver
 * - matrix and rhs assembly
 *   FIXME: This means that currently, each solver assembles its whole set of
 *   equations. In particular, the Navier-Stokes eq. are duplicated across the
 *   solvers, which is not ideal.
 *
 */
template <int dim>
class NavierStokesSolver : public GenericSolver<LA::ParVectorType>
{
public:
  NavierStokesSolver(const ParameterReader<dim> &param,
                     const bool                  with_moving_mesh);

  virtual ~NavierStokesSolver() {}

public:
  /**
   * Solve: either solve for the steady-state solution, or integrate
   * in time until the end of the simulation.
   */
  virtual void run() override;

  /**
   * Reset the solver between two runs. This is typically useful when running
   * convergence studies, to properly reset the mesh, time integration data,
   * etc.
   */
  void         reset();
  virtual void reset_solver_specific_data() {}

  /**
   * Update time in all relevant structures:
   *  - boundary conditions
   *  - source terms
   *  - exact solution
   *  - physical properties
   */
  void         set_time();
  virtual void set_solver_specific_time() {}

  /**
   * Distribute (number) the degrees of freedom and allocate the parallel matrix
   * and vectors.
   */
  virtual void setup_dofs();

  /**
   * Create the data needed to enforce zero-mean pressure.
   *
   * Note that, as it is done for now, enforcing zero-mean is an expensive
   * operation, because it couples a pressure dof on a partition to *all*
   * other pressure dofs, thus filling its matrix entries. See also the comments
   * in boundary_conditions.h.
   */
  void create_zero_mean_pressure_constraints_data();

  /**
   * If a derived solver requires additional constraints data that need to be
   * created only once, they should be created within an overload of this
   * function. For instance, the FSI solver creates here the data to couple the
   * fluid forces on an obstacle to the mesh position.
   */
  virtual void create_solver_specific_constraints_data() {}

  /**
   * Create the velocity, pressure and mesh position boundary conditions.
   */
  virtual void create_base_constraints(const bool                 homogeneous,
                                       AffineConstraints<double> &constraints);

  /**
   * Create the homogeneous boundary conditions.
   */
  virtual void create_zero_constraints();
  virtual void create_solver_specific_zero_constraints() {}

  /**
   * Create the inhomogeneous boundary conditions.
   */
  virtual void create_nonzero_constraints();
  virtual void create_solver_specific_nonzero_constraints() {}

  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   * Update the inhomogeneous boundary conditions for the current time, after
   * time has been updated.
   */
  void update_boundary_conditions();

  /**
   * Create the matrix sparsity pattern, given the finite element spaces and
   * constraints.
   */
  virtual void create_sparsity_pattern() = 0;

  /**
   * Apply the initial conditions for velocity, pressure and mesh position.
   * Initial conditions on additional fields must be set in the solver-specific
   * overload.
   */
  void         set_initial_conditions();
  virtual void set_solver_specific_initial_conditions() {}

  /**
   * Idem as initial conditions, but applies the prescribed exact solution.
   */
  void         set_exact_solution();
  virtual void set_solver_specific_exact_solution() {}

  /**
   * Compare each Jacobian matrix computed in assemble_local_matrix to its
   * finite differences approximation obtained by perturbing the right-hand side
   * (Newton residual). To allow comparing for multiple Newton iterations and/or
   * time steps, this function does not throw if the difference between the
   * analytical matrix entries and their FD counterpart exceeds a prescribed
   * tolerance, but instead prints the local matrices and the problematic
   * entries.
   */
  virtual void compare_analytical_matrix_with_fd() = 0;

  /**
   * Solve the linear system for a single nonlinear solver iteration.
   */
  virtual void solve_linear_system(const bool /* */) override;

  /**
   * Post-process the numerical solution: output for visualization,
   * compute errors, forces, etc.
   */
  void         postprocess_solution();
  virtual void solver_specific_post_processing() {}

  /**
   * For each prescribed Sobolev norm, compute the error on the given field
   * and add it to the error handler.
   */
  void compute_and_add_errors(const Mapping<dim>  &mapping,
                              const Function<dim> &exact_solution,
                              Vector<double>      &cellwise_errors,
                              const ComponentSelectFunction<dim> &comp_function,
                              const std::string                  &field_name);

  /**
   * Compute the error on the velocity, pressure and mesh position for each of
   * the prescribed Sobolev norms. Errors on additional fields must be computed
   * in the overloaded function.
   */
  void         compute_errors();
  virtual void compute_solver_specific_errors() {}

  /**
   *
   */
  void compute_forces();

  /**
   * Write the results to a vtu/pvtu file for visualization.
   */
  virtual void output_results() = 0;

  /**
   * Perform actions after the end of the simulation loop, such as writing
   * .pvd output.
   */
  void finalize();

  /**
   * Get the FESystem of the derived solver
   */
  virtual const FESystem<dim> &get_fe_system() const = 0;

protected:
  std::shared_ptr<ComponentOrdering> ordering;

  ParameterReader<dim> param;

  const bool with_moving_mesh;

  // Choose another quadrature rule for error computation
  std::shared_ptr<Quadrature<dim>>     quadrature;
  std::shared_ptr<Quadrature<dim>>     error_quadrature;
  std::shared_ptr<Quadrature<dim - 1>> face_quadrature;
  std::shared_ptr<Quadrature<dim - 1>> error_face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  fixed_mapping;
  std::shared_ptr<Mapping<dim>>                  moving_mapping;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  FEValuesExtractors::Vector velocity_extractor;
  FEValuesExtractors::Scalar pressure_extractor;
  FEValuesExtractors::Vector position_extractor;

  ComponentMask velocity_mask;
  ComponentMask pressure_mask;
  ComponentMask position_mask;

  Table<2, DoFTools::Coupling> coupling_table;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  types::global_dof_index constrained_pressure_dof = numbers::invalid_dof_index;
  Point<dim>              constrained_pressure_support_point;
  std::vector<std::pair<types::global_dof_index, double>>
    zero_mean_pressure_weights;

  std::map<types::global_dof_index, Point<dim>> initial_positions;

  LA::ParMatrixType              system_matrix;
  std::vector<LA::ParVectorType> previous_solutions;

  std::shared_ptr<Function<dim>> source_terms;
  std::shared_ptr<Function<dim>> exact_solution;

  TableHandler forces_table;

  SolverControl                                          solver_control;
  std::shared_ptr<PETScWrappers::SparseDirectMUMPSReuse> direct_solver_reuse;
};

#endif