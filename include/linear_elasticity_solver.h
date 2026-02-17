#ifndef LINEAR_ELASTICITY_SOLVER_H
#define LINEAR_ELASTICITY_SOLVER_H

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
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/lac/affine_constraints.h>
#include <generic_solver.h>
#include <mumps_solver.h>
#include <parameter_reader.h>
#include <scratch_data_linear_elasticity.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Solve the linear elasticity equation written for the mesh position :
 *
 * - \nabla \cdot \sigma(x) + f = 0,
 *
 * with \sigma(x) = 2\mu\epsilon(x) + \lambda\tr{\epsilon(x)} I.
 * The infinitesimal strain tensor \epsilon(x) is written in terms of the mesh
 * position, and not in terms of displacement u, and thus writes:
 *
 * \epsilon(x) = (\nabla(u) + \nabla(u)^T)/2
 *             = (\nabla(x) + \nabla(x)^T)/2 - I
 *
 * The source term f can either be evaluated on the reference configuration X,
 * or on the current configuration x(X), yielding a nonlinear problem.
 * For the latter, the nonlinear solver may fail to find a solution if the
 * source term on the current mesh is too steep. To counteract this, a
 * continuation method is used, to progressively account for the source term,
 * and we actually solve :
 *
 * - \nabla \cdot \sigma(x) + alpha * f(x(X)) = 0.
 *
 * This feature is controlled by the "Linear elasticity" subsection of the
 * parameter file. The continuation parameter alpha lies in the provided
 * [min_coeff, max_coeff] bracket, so that the last solved position field
 * satisfies :
 *
 * - \nabla \cdot \sigma(x) + max_coeff * f(x(X)) = 0.
 *
 */
template <int dim>
class LinearElasticitySolver : public GenericSolver<LA::ParVectorType>
{
  using ScratchData = ScratchDataLinearElasticity<dim>;

public:
  /**
   * Constructor
   */
  LinearElasticitySolver(const ParameterReader<dim> &param);

  virtual ~LinearElasticitySolver() {}

public:
  /**
   * Solve the elasticity problem
   */
  virtual void run() override;

  void reset();

  void set_time();

  void setup_dofs();

  void create_base_constraints(const bool                 homogeneous,
                               AffineConstraints<double> &constraints);

  void create_zero_constraints();
  void create_nonzero_constraints();

  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  void update_boundary_conditions();

  virtual void create_sparsity_pattern();

  void set_initial_conditions();
  void set_exact_solution();

  virtual void solve_linear_system(const bool /* */) override;

  virtual void assemble_matrix() override;

  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratchData,
    CopyData                                             &copy_data);

  void assemble_local_matrix_finite_differences(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_matrix(const CopyData &copy_data);

  void compare_analytical_matrix_with_fd();

  virtual void assemble_rhs() override;

  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchData &scratchData,
                     CopyData    &copy_data);

  void copy_local_to_global_rhs(const CopyData &copy_data);

  void move_mesh();

  void postprocess_solution();

  void compute_errors();

  virtual void output_results();

protected:
  ParameterReader<dim> param;

  std::shared_ptr<FESystem<dim>> fe;

  std::shared_ptr<Quadrature<dim>>     quadrature;
  std::shared_ptr<Quadrature<dim>>     error_quadrature;
  std::shared_ptr<Quadrature<dim - 1>> face_quadrature;
  std::shared_ptr<Quadrature<dim - 1>> error_face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  mapping;
  DoFHandler<dim>                                dof_handler;

  FEValuesExtractors::Vector position_extractor;
  ComponentMask              position_mask;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  LA::ParMatrixType system_matrix;

  std::shared_ptr<Function<dim>> source_terms;
  std::shared_ptr<Function<dim>> exact_solution;

  SolverControl                                          solver_control;
  std::shared_ptr<PETScWrappers::SparseDirectMUMPSReuse> direct_solver_reuse;

  double source_term_moving_mesh_multiplier;
  double source_term_fixed_mesh_multiplier;

protected:
  /**
   * Source term called when performing a convergence study.
   * This function calls the derivatives functions from the given
   * pre-set manufactured solution.
   */
  class MMSSourceTerm : public Function<dim>
  {
  public:
    MMSSourceTerm(
      const Parameters::PhysicalProperties<dim> &physical_properties,
      const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(dim)
      , physical_properties(physical_properties)
      , mms(mms)
    {}

    /**
     * Evaluate source term for the linear elasticity equation.
     */
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;

    /**
     * Fill the given list of gradients.
     */
    virtual void vector_gradient_list(
      const std::vector<Point<dim>> & /*points*/,
      std::vector<std::vector<Tensor<1, dim>>> & /*gradients*/) const override
    {
      // Do nothing: this function is only there to be able to call
      // vector_gradient_list in the ScratchData without throwing in debug mode.
      // Source term gradient is only needed when the source term is evaluated
      // on the current configuration (mesh), which is never the case for an MMS
      // source term.
    }

  protected:
    const Parameters::PhysicalProperties<dim>              &physical_properties;
    const ManufacturedSolutions::ManufacturedSolution<dim> &mms;
  };
};

#endif