#ifndef INCOMPRESSIBLE_NS_SOLVER_H
#define INCOMPRESSIBLE_NS_SOLVER_H

#include <copy_data.h>
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
#include <generic_solver.h>
#include <parameter_reader.h>
#include <scratch_data.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Incompressible Navier-Stokes solver.
 * Solves the nonstabilized Navier-Stokes equations with div(u) = 0. Because
 * the system is not stabilized with e.g. PSPG terms, LBB stable mixed finite
 * elements should be used, the most straightforward being the P2-P1 Taylor-Hood
 * element.
 */
template <int dim>
class IncompressibleNavierStokesSolver : public GenericSolver<LA::ParVectorType>
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
   * Initialize the dof handler and allocate parallel vectors
   */
  void setup_dofs();

  /**
   * Create the homogeneous constraints
   */
  void create_zero_constraints();

  /**
   * (Re-)create the nonhomogeneous constraints
   */
  void create_nonzero_constraints();

  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   * Create the sparsity pattern and allocate matrix
   */
  void create_sparsity_pattern();

  /**
   * Apply initial conditions
   */
  void set_initial_conditions();

  /**
   * Recreate and apply nonhomogeneous constraints
   */
  void update_boundary_conditions();

  /**
   * Assemble the linearized Jacobian matrix at the current evaluation point
   */
  virtual void assemble_matrix() override;

  /**
   * Compute the element-wise matrix. This function is passed to
   * WorkStream::run to perform multithreaded assembly if supported
   * (i.e., if using thread-safe matrix and vector wrappers).
   */
  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataNS<dim>                                   &scratchData,
    CopyData                                             &copy_data);

  /**
   * Assemble the element-wise matrix computed with assemble_local_matrix
   * into the global matrix. Passed to WorkStream::run (see above).
   */
  void copy_local_to_global_matrix(const CopyData &copy_data);

  /**
   * Assemble the Newton residual at the current evluation point
   */
  virtual void assemble_rhs() override;

  /**
   * See assemble_local_matrix.
   */
  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchDataNS<dim> &scratchData,
                     CopyData           &copy_data);

  /**
   * See copy_local_to_global_matrix.
   */
  void copy_local_to_global_rhs(const CopyData &copy_data);

  /**
   * Solve the linear system J(u) * du = -NL(u).
   */
  virtual void
  solve_linear_system(const bool apply_inhomogeneous_constraints) override;

  /**
   * Write the velocity and pressure to vtu file.
   */
  void output_results() const;

  /**
   * Set source terms explicitly after creation and override the source terms
   * read from the parameter file, for instance when performing manufactured
   * solutions tests.
   */
  void set_source_terms(const std::shared_ptr<Function<dim>> source_terms)
  {
    this->source_terms = source_terms;
  }

  /**
   * Source term called when performing a convergence study.
   * This function calls the derivatives functions from the given
   * pre-set manufactured solution.
   */
  class MMSSourceTerm : public Function<dim>
  {
  protected:
    const ManufacturedSolution::ManufacturedSolution<dim> &mms;
  public:
    MMSSourceTerm(const double time,
      const unsigned int n_components,
      const ManufacturedSolution::ManufacturedSolution<dim> &mms);

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;
  };

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
  inline bool is_velocity(const unsigned int component) const
  {
    return u_lower <= component && component < u_upper;
  }
  inline bool is_pressure(const unsigned int component) const
  {
    return p_lower <= component && component < p_upper;
  }

protected:
  ParameterReader<dim> param;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  mapping;
  FESystem<dim>                                  fe;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  LA::ParMatrixType              system_matrix;
  std::vector<LA::ParVectorType> previous_solutions;

  std::shared_ptr<Function<dim>> source_terms;

  TableHandler forces_table;
  TableHandler cylinder_position_table;

  ConvergenceTable mms_errors;
};

#endif