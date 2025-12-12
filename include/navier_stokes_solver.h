#ifndef NAVIER_STOKES_SOLVER_H
#define NAVIER_STOKES_SOLVER_H

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
#include <scratch_data.h>
#include <time_handler.h>
#include <types.h>
#include <components_ordering.h>

using namespace dealii;

/**
 * A base class for Navier-Stokes solvers
 *
 * This class handles:
 *
 *  -
 *
 * This class does not handle:
 *
 * - the FESystem, which depends on the fields layout of each specific solver
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
   * Solve!
   */
  virtual void run() override;

  /**
   *
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
   *
   */
  void setup_dofs();

  /**
   * 
   */
  void create_zero_mean_pressure_constraints_data();

  /**
   *
   */
  virtual void create_solver_specific_constraints_data() {}

  /**
   *
   */
  void create_base_constraints(const bool                 homogeneous,
                               AffineConstraints<double> &constraints);

  /**
   *
   */
  void         create_zero_constraints();
  virtual void create_solver_specific_zero_constraints() {}

  /**
   *
   */
  void         create_nonzero_constraints();
  virtual void create_solver_specific_nonzero_constraints() {}

  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   * 
   */
  void update_boundary_conditions();

  /**
   *
   */
  virtual void create_sparsity_pattern() = 0;

  /**
   *
   */
  void         set_initial_conditions();
  virtual void set_solver_specific_initial_conditions() {}

  /**
   * 
   */
  void set_exact_solution();

  /**
   * 
   */
  virtual void compare_analytical_matrix_with_fd() = 0;

  /**
   * 
   */
  virtual void
  solve_linear_system(const bool /* */) override;

  /**
   *
   */
  void postprocess_solution();
  virtual void solver_specific_post_processing() {}

  /**
   * 
   */
  void compute_errors();
  virtual void compute_solver_specific_errors() {}

  /**
   * 
   */
  void compute_forces();

  /**
   *
   */
  virtual void output_results() = 0;

  /**
   * Get the FESystem of the derived solver
   */
  virtual const FESystem<dim> &get_fe_system() const = 0;

protected:
  std::shared_ptr<ComponentOrdering> ordering;

  ParameterReader<dim> param;

  const bool with_moving_mesh;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  fixed_mapping;
  std::shared_ptr<Mapping<dim>>                  moving_mapping;
  DoFHandler<dim> dof_handler;
  TimeHandler     time_handler;

  FEValuesExtractors::Vector velocity_extractor;
  FEValuesExtractors::Scalar pressure_extractor;
  FEValuesExtractors::Vector position_extractor;

  ComponentMask velocity_mask;
  ComponentMask pressure_mask;
  ComponentMask position_mask;

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