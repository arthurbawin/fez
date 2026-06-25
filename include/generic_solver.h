#ifndef GENERIC_SOLVER_H
#define GENERIC_SOLVER_H

/**
 * This structure is borrowed from the Lethe project.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/observer_pointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <error_handler.h>
#include <newton_solver.h>
#include <nonlinear_solver.h>
#include <parameter_reader.h>
#include <solver_info.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Abstract base class for a generic solver with the following common members:
 *
 * - simulation parameters
 * - mpi communicator
 * - nonlinear solver
 */
template <typename VectorType>
class GenericSolver : public EnableObserverPointer
{
public:
  /**
   * Constructor.
   */
  GenericSolver(const Parameters::Output          &output_param,
                const Parameters::NonLinearSolver &nonlinear_solver_param,
                const Parameters::Timer           &timer_param,
                const Parameters::Mesh            &mesh_param,
                const Parameters::TimeIntegration &time_param,
                const Parameters::MMS             &mms_param,
                const SolverInfo::SolverType       solver_type);

  /**
   * Destructor.
   */
  virtual ~GenericSolver() = default;

  /**
   * Solve the problem. This function is the only one that is called in the
   * solvers' main function, and thus contains all the steps required to solve
   * the problem at hand, namely, create the mesh, setup the matrix and vectors,
   * handle the time integration loop, postprocess the solution, etc.
   *
   * It must be overloaded by the various derived solvers.
   */
  virtual void run() = 0;

  /**
   * Run a convergence study with manufactured or exact solutions. This function
   * is run instead of the run() above if the "Manufactured solution" is enabled
   * in the parameter file. This functions takes care of refining the time step
   * and/or replacing the name of the mesh file by the refined one at each step
   * of the convergence study, stores the computed errors in error tables, and
   * output the convergence rates.
   */
  template <int dim>
  void run_convergence_loop();

  /**
   * Run the problem within a fixed point loop, performing (possibly)
   * anisotropic mesh adaptation based on a Riemannian metric. The aim of the
   * fixed point loop is to converge the solution-mesh pair together.
   *
   * This function is called within the main function of each executable if mesh
   * adaptation with a Riemannian metric is enabled.
   */
  void run_fixed_point_loop();

  /**
   * Assemble the Jacobian matrix of the nonlinear problem, that is, its
   * linearization at the current solution. This function is overloaded by each
   * derived solver.
   */
  virtual void assemble_matrix() = 0;

  /**
   * Assemble the (negative of the) nonlinear residual -NL(U), evaluated at the
   * current solution. This function is overloaded by each derived solver.
   */
  virtual void assemble_rhs() = 0;

  /**
   * Solve the linear system described by the assembly functions above.
   */
  virtual void solve_linear_system() = 0;

  /**
   * Solve the nonlinear problem NL(U) = 0, where NL is the weak formulation of
   * the system of PDEs of interest, typically the Navier-Stokes equations
   * augmented with additional equations for the mesh movement, phase tracer,
   * etc., depending on the derived solver.
   */
  void solve_nonlinear_problem(const TimeHandler &time_handler);

  /**
   * Return the nonzero constraints of the derived solver, that is, the
   * inhomogeneous boundary conditions and constraints on the dofs.
   */
  virtual AffineConstraints<double> &get_nonzero_constraints() = 0;

  /**
   * Apply the inhomogeneous boundary conditions and constraints to the
   * local evaluation point.
   */
  void distribute_nonzero_constraints();

  /**
   *
   */
  virtual void adapt_mesh();

  /**
   * Return true if the triangulation should be (re-)created on this time
   * interval.
   */
  virtual bool should_create_triangulation() const;

  /**
   * Return true if the triangulation should be refined, i.e., if the simulation
   * is unsteady and this is a time step matching the prescribed frequency.
   * Only used to adapt tree-based meshes.
   */
  virtual bool
  should_adapt_tree_based_mesh(const TimeHandler &time_handler) const;

  /**
   * Return true if the mesh(es) should be adapted, after all time steps have
   * been computed on all time subintervals.
   *
   * With metric-based mesh adaptation, this is always true, as the transient
   * fixed-point method requires scaling the metrics with a global scaling
   * factor that can only be computed at the end of a fixed-point iteration
   * (i.e., the whole simulation time interval).
   *
   * With tree-based adaptation, this is true only for steady-state convergence
   * studies, and if there is another convergence step after this one. Because
   * the number of mesh cells changes when calling the refinement and coarsening
   * routines, this function should be called *after* registering the number of
   * cells/dofs in the error handler (with add_reference_adata(...), which is
   * typically done in a finalize() function), to provide matching cells/dofs
   * and measured error norms in the convergence table.
   */
  virtual bool
  should_adapt_mesh_at_end_of_intervals(const TimeHandler &time_handler) const;

  /**
   * Return true if the solver should evaluate error norms. This is typically
   * always true, except when adapting the mesh with a Riemannian metric, in
   * which case a few fixed-point iterations are performed to converge to a
   * mesh-solution pair, and the errors are computed only on the last solution.
   */
  virtual bool should_compute_errors(const TimeHandler &time_handler) const;

  /**
   * Return true if reference data (number of mesh elements, vertices, dofs,
   * time step, ...) should be added to the error handler.
   */
  virtual bool
  should_add_error_reference_data(const TimeHandler &time_handler) const;

  /**
   * Return true if velocity constraints enforced with a Lagrange multiplier
   * should be checked for accuracy.
   */
  virtual bool
  should_check_weakly_enforced_velocity(const TimeHandler &time_handler) const;

  /**
   * Return true if the solver should reconstruct the solution (and optionally
   * its derivatives) with the polynomial-preserving operator.
   */
  template <int dim>
  bool should_compute_reconstructions(const ParameterReader<dim> &param,
                                      const TimeHandler &time_handler) const;

  /**
   * Return true if the solver should compute or update its Riemannian
   * metric(s).
   */
  template <int dim>
  bool should_compute_riemannian_metric(const ParameterReader<dim> &param,
                                        const TimeHandler &time_handler) const;

  /**
   * Return the (ghosted) solution vector.
   */
  virtual VectorType &get_present_solution() = 0;

  /**
   * Return the (ghosted) evaluation point.
   */
  VectorType &get_evaluation_point();

  /**
   * Return the (fully distributed) evaluation point.
   */
  VectorType &get_local_evaluation_point();

  /**
   * Return the (fully distributed) newton update.
   */
  VectorType &get_newton_update();

  /**
   * Return the (fully distributed) system right-hand side.
   */
  VectorType &get_system_rhs();

  /**
   * Return the time integration parameters. Currently, this is only used by the
   * Newton solver to determine the reassembly heuristic, as the system is
   * always reassembled for steady-state computations for instance.
   */
  const Parameters::TimeIntegration &get_time_parameters() const;

  /**
   * Return a pointer to the ErrorHandler associated to the given norm @p type.
   * Throws an error if the given type was not stored, i.e., if it was not
   * specified in the parameter file in the "Manufactured solution" section.
   */
  const ErrorHandler &get_error_handler(const VectorTools::NormType type) const;

public:
  // MPI communicator
  MPI_Comm mpi_communicator;

  // MPI rank of this process
  const unsigned int mpi_rank;

  // Number of MPI processes
  const unsigned int mpi_size;

  // Conditional stream to print from the root process
  ConditionalOStream pcout;

  // Timer
  TimerOutput computing_timer;

protected:
  /**
   * Solver type
   */
  SolverInfo::SolverType solver_type;

  /**
   * Copy of the (ghosted) solution that is modified during the nonlinear solve,
   * whereas the solution vector is modified only at the end of the nonlinear
   * solve, or during the line search (if applicable).
   */
  VectorType evaluation_point;

  /**
   * Fully distributed (i.e., without ghost entries) evaluation point.
   */
  VectorType local_evaluation_point;

  /**
   * Fully distributed increment in the Newton-Raphson method.
   */
  VectorType newton_update;

  /**
   * Fully distributed right-hand side in the Newton-Raphson method.
   */
  VectorType system_rhs;

  /**
   * Nonlinear solver.
   */
  std::unique_ptr<NonLinearSolver<VectorType>> nonlinear_solver;

  /**
   * Data to perform a space and/or time convergence study
   * Some must be modified during the convergence loop, so store a copy
   */
  const Parameters::Output   &output_param;
  Parameters::Mesh            mesh_param;
  Parameters::TimeIntegration time_param;
  Parameters::MMS             mms_param;

  /**
   * ErrorHandlers, one for each required norm.
   * Each ErrorHandler stores the error in the prescribed norm for all fields.
   */
  std::map<VectorTools::NormType, ErrorHandler> error_handlers;

  // FIXME: Not really useful to set the Newton solver as friend if there are
  // getter functions.
  friend class NewtonSolver<VectorType>;
};

/* ---------------- Template functions ----------------- */

template <typename VectorType>
VectorType &GenericSolver<VectorType>::get_evaluation_point()
{
  return evaluation_point;
}

template <typename VectorType>
VectorType &GenericSolver<VectorType>::get_local_evaluation_point()
{
  return local_evaluation_point;
}

template <typename VectorType>
VectorType &GenericSolver<VectorType>::get_newton_update()
{
  return newton_update;
}

template <typename VectorType>
VectorType &GenericSolver<VectorType>::get_system_rhs()
{
  return system_rhs;
}

#endif
