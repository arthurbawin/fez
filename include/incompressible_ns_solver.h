#ifndef INCOMPRESSIBLE_NS_SOLVER_H
#define INCOMPRESSIBLE_NS_SOLVER_H

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <generic_solver.h>
#include <parameter_reader.h>
#include <scratch_data.h>
#include <copy_data.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Derived class for the incompressible Navier-Stokes solver
 */
template <int dim>
class IncompressibleNavierStokesSolver : public GenericSolver<ParVectorType>
{
public:
  IncompressibleNavierStokesSolver(const ParameterReader<dim> &param);

public:
  virtual ~IncompressibleNavierStokesSolver() {}

public:
  /**
   * Solve the flow problem
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
  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   *
   */
  void create_sparsity_pattern();

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
  void assemble_local_matrix_og(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataNS<dim>                                   &scratchData,
    ParVectorType                                        &current_solution,
    std::vector<ParVectorType>                           &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute);

  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataNS<dim>                                   &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_matrix(const CopyData &copy_data);

  /**
   *
   */
  virtual void assemble_matrix() override;

  /**
   *
   */
  void
  assemble_local_rhs_og(bool first_step,
                     const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchDataNS<dim>                   &scratchData,
                     ParVectorType                        &current_solution,
                     std::vector<ParVectorType>           &previous_solutions,
                     std::vector<types::global_dof_index> &local_dof_indices,
                     Vector<double>                       &local_rhs,
                     std::vector<double>                  &cell_dof_values,
                     bool                                  distribute,
                     bool                                  use_full_solution);

  void
  assemble_local_rhs(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataNS<dim>                                   &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_rhs(const CopyData &copy_data);

  /**
   *
   */
  virtual void assemble_rhs() override;

  /**
   *
   */
  virtual void
  solve_linear_system(const bool apply_inhomogeneous_constraints) override;

  /**
   *
   */
  void output_results() const;

protected:
  // Ordering of the FE system for the incompressible NS solver.
  // Each field is in the half-open [lower, upper)
  // Check for matching component by doing e.g.:
  // if(u_lower <= comp && comp < u_upper)
  const unsigned int n_components = dim + 1;
  const unsigned int u_lower      = 0;
  const unsigned int u_upper      = dim;
  const unsigned int p_lower      = dim;
  const unsigned int p_upper      = dim + 1;

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

protected:
  ParameterReader<dim> param;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::unique_ptr<Mapping<dim>>                  mapping;
  FESystem<dim>                                  fe;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  dealii::LinearAlgebraPETSc::MPI::SparseMatrix system_matrix;
  std::vector<ParVectorType>                    previous_solutions;

  TableHandler forces_table;
  TableHandler cylinder_position_table;
};

#endif