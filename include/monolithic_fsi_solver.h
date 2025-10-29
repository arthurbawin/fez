#ifndef MONOLITHIC_FSI_SOLVER_H
#define MONOLITHIC_FSI_SOLVER_H

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
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <generic_solver.h>
#include <parameter_reader.h>
#include <time_handler.h>
#include <types.h>
#include <scratch_data.h>

using namespace dealii;

/**
 * Derived class for the monolithic fluid-structure interaction solver.
 * It is a somewhat niche class, which treats a single obstacle for now.
 */
template <int dim>
class MonolithicFSISolver : public GenericSolver<ParVectorType>
{
public:
  MonolithicFSISolver(const ParameterReader<dim> &param);

public:
  virtual ~MonolithicFSISolver() {}

public:
  /**
   * Solve the FSI problem
   */
  virtual void run() override;

  /**
   *
   */
  void setup_dofs();

  /**
   *
   */
  void create_zero_constraints();

  /**
   *
   */
  void create_nonzero_constraints();

  /**
   * 
   */
  virtual AffineConstraints<double>& get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   *
   */
  void create_sparsity_pattern();

  /**
   * Create the AffineConstraints storing the lambda = 0
   * constraints everywhere, except on the boundary of interest
   * on which a weakly enforced no-slip condition is prescribed.
   */
  void create_lagrange_multiplier_constraints();

  /**
   *
   */
  void create_position_lagrange_mult_coupling_data();

  /**
   *
   */
  void set_initial_conditions();

  /**
   *
   */
  void update_boundary_conditions();

  /**
   *
   */
  void add_algebraic_position_coupling_to_matrix();

  /**
   *
   */
  void add_algebraic_position_coupling_to_rhs();

  /**
   *
   */
  void assemble_local_matrix(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    ParVectorType                                      &current_solution,
    std::vector<ParVectorType>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute);

  /**
   *
   */
  virtual void assemble_matrix() override;

  /**
   *
   */
  void
  assemble_local_rhs(bool first_step,
                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchData<dim>                     &scratchData,
                     ParVectorType                      &current_solution,
                     std::vector<ParVectorType>         &previous_solutions,
                     std::vector<types::global_dof_index> &local_dof_indices,
                     Vector<double>                       &local_rhs,
                     std::vector<double>                  &cell_dof_values,
                     bool                                  distribute,
                     bool                                  use_full_solution);

  /**
   *
   */
  virtual void assemble_rhs() override;

  /**
   *
   */
  virtual void solve_linear_system(const bool apply_inhomogeneous_constraints) override;

  /**
   *
   */
  void output_results() const;

  /**
   * 
   */
  void compare_forces_and_position_on_obstacle() const;

  /**
   * 
   */
  void check_velocity_boundary() const;

  /**
   * Compute the "raw" forces on the obstacle.
   * These need to nondimensionalized to obtain the force coefficients.
   */
  void compute_forces(const bool export_table);

  /**
   * 
   */
  void write_cylinder_position(const bool export_table);

protected:
  // Ordering of the FE system for the FSI solver.
  // Each field is in the half-open [lower, upper)
  // Check for matching component by doing e.g.:
  // if(u_lower <= comp && comp < u_upper)
  const unsigned int n_components = 3 * dim + 1;
  const unsigned int u_lower      = 0;
  const unsigned int u_upper      = dim;
  const unsigned int p_lower      = dim;
  const unsigned int p_upper      = dim + 1;
  const unsigned int x_lower      = dim + 1;
  const unsigned int x_upper      = 2 * dim + 1;
  const unsigned int l_lower      = 2 * dim + 1;
  const unsigned int l_upper      = 3 * dim + 1;

  /**
   * Quality-of-life functions to check which field a given component is
   */
  bool is_velocity(const unsigned int component) const
  {
    return u_lower <= component && component < u_upper;
  }
  bool is_pressure(const unsigned int component) const
  {
    return p_lower <= component && component < p_upper;
  }
  bool is_position(const unsigned int component) const
  {
    return x_lower <= component && component < x_upper;
  }
  bool is_lambda(const unsigned int component) const
  {
    return l_lower <= component && component < l_upper;
  }

protected:
  ParameterReader<dim> param;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::unique_ptr<Mapping<dim>>                  fixed_mapping;
  std::unique_ptr<Mapping<dim>>                  mapping;
  FESystem<dim>                                  fe;

  DoFHandler<dim> dof_handler;

  TimeHandler time_handler;

  /**
   * The id of the mesh boundary on which the weak no-slip condition
   * is enforced. Currently, this is limited to a single boundary.
   */
  types::boundary_id weak_no_slip_boundary_id = numbers::invalid_unsigned_int;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;
  AffineConstraints<double> lambda_constraints;

  // Position-lambda constraints on the cylinder
  // The affine coefficients c_ij: [dim][{lambdaDOF_j : c_ij}]
  std::vector<std::vector<std::pair<unsigned int, double>>>
                                                  position_lambda_coeffs;
  std::map<types::global_dof_index, Point<dim>>   initial_positions;
  std::map<types::global_dof_index, unsigned int> coupled_position_dofs;

  dealii::LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;
  std::vector<ParVectorType>                    previous_solutions;


  TableHandler forces_table;
  TableHandler cylinder_position_table;
};

#endif