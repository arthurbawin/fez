#ifndef HEAT_SOLVER_H
#define HEAT_SOLVER_H

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
#include <scratch_data_heat.h>
#include <time_handler.h>
#include <copy_data.h>
#include <types.h>

using namespace dealii;

/**
 * Solve the unsteady heat diffusion equation (concrete class).
 *
 * This solver is meant as a toy solver, to test error estimation
 * and mesh adaptation methods.
 */
template <int dim>
class HeatSolver : public GenericSolver<LA::ParVectorType>
{
public:
  HeatSolver(const ParameterReader<dim> &param);

  virtual ~HeatSolver() {}

public:
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

  void         set_initial_conditions();
  void         set_exact_solution();

  virtual void solve_linear_system(const bool /* */) override;

  virtual void assemble_matrix() override;

  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataHeat<dim>                                 &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_matrix(const CopyData &copy_data);

  void compare_analytical_matrix_with_fd();

  virtual void assemble_rhs() override;

  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchDataHeat<dim> &scratchData,
                     CopyData             &copy_data);

  void copy_local_to_global_rhs(const CopyData &copy_data);

  void postprocess_solution();

  void compute_and_add_errors(const Mapping<dim>  &mapping,
                              const Function<dim> &exact_solution,
                              Vector<double>      &cellwise_errors,
                              const ComponentSelectFunction<dim> &comp_function,
                              const std::string                  &field_name);

  void compute_errors();

  virtual void output_results();

protected:
  std::shared_ptr<ComponentOrdering> ordering;

  ParameterReader<dim> param;

  FESystem<dim> fe;

  // Choose another quadrature rule for error computation
  QSimplex<dim>     quadrature;
  QSimplex<dim>     error_quadrature;
  QSimplex<dim - 1> face_quadrature;
  QSimplex<dim - 1> error_face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  mapping;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  FEValuesExtractors::Scalar temperature_extractor;
  ComponentMask              temperature_mask;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  LA::ParMatrixType              system_matrix;
  std::vector<LA::ParVectorType> previous_solutions;

  std::shared_ptr<Function<dim>> source_terms;
  std::shared_ptr<Function<dim>> exact_solution;

  SolverControl                                          solver_control;
  std::shared_ptr<PETScWrappers::SparseDirectMUMPSReuse> direct_solver_reuse;

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
      const double                               time,
      const Parameters::PhysicalProperties<dim> &physical_properties,
      const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(1, time)
      , physical_properties(physical_properties)
      , mms(mms)
    {}

    // Update time in the mms functions
    virtual void set_time(const double new_time) override
    {
      mms.set_time(new_time);
    }

    /**
     * Evaluate source term for the heat equation.
     */
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;

  protected:
    const Parameters::PhysicalProperties<dim> &physical_properties;
    // MMS cannot be const since its internal time must be updated
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };
};

#endif