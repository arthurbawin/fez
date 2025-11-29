#ifndef MONOLITHIC_FSI_SOLVER_H
#define MONOLITHIC_FSI_SOLVER_H

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
// #include <mms.h>
#include <mumps_solver.h>
#include <parameter_reader.h>
#include <scratch_data.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Derived class for the monolithic fluid-structure interaction solver.
 * It is a somewhat "niche" class, which treats a single obstacle for now.
 */
template <int dim>
class MonolithicFSISolver : public GenericSolver<LA::ParVectorType>
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
  void constrain_pressure_point(AffineConstraints<double> &constraints,
                                const bool                 set_to_zero);

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

  void apply_erroneous_position_lambda_constraints(const bool homogeneous);

  /**
   *
   */
  void remove_cylinder_velocity_constraints(
    AffineConstraints<double> &constraints) const;

  /**
   *
   */
  void set_initial_conditions();

  /**
   * Set solution to exact solution, if provided
   */
  void set_exact_solution();

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
    ScratchDataMonolithicFSI<dim>                        &scratchData,
    LA::ParVectorType                                    &current_solution,
    std::vector<LA::ParVectorType>                       &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute);

  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchDataMonolithicFSI<dim>                        &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_matrix(const CopyData &copy_data);

  void compare_analytical_matrix_with_fd();

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
                     ScratchDataMonolithicFSI<dim>        &scratchData,
                     LA::ParVectorType                    &current_solution,
                     std::vector<LA::ParVectorType>       &previous_solutions,
                     std::vector<types::global_dof_index> &local_dof_indices,
                     Vector<double>                       &local_rhs,
                     std::vector<double>                  &cell_dof_values,
                     bool                                  distribute,
                     bool                                  use_full_solution);

  /**
   *
   */
  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchDataMonolithicFSI<dim> &scratchData,
                     CopyData                      &copy_data);

  /**
   * See copy_local_to_global_matrix.
   */
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
  void compute_lambda_error_on_boundary(double         &lambda_l2_error,
                                        double         &lambda_linf_error,
                                        Tensor<1, dim> &error_on_integral);

  void check_manufactured_solution_boundary();

  /**
   *
   */
  void compute_errors();

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

  /**
   * Reset the resolution related structures (mesh, dof_handler, etc.) in
   * between two runs, e.g. when performing convergence tests.
   */
  void reset();

  /**
   * Update time in all relevant structures (boundary conditions, source terms,
   * exact solution).
   */
  void set_time();

protected:
  // Ordering of the FE system for the FSI solver.
  // Each field is in the half-open [lower, upper)
  // Check for matching component by doing e.g.:
  // if(u_lower <= comp && comp < u_upper)
  static constexpr unsigned int n_components = 3 * dim + 1;
  static constexpr unsigned int u_lower      = 0;
  static constexpr unsigned int u_upper      = dim;
  static constexpr unsigned int p_lower      = dim;
  static constexpr unsigned int p_upper      = dim + 1;
  static constexpr unsigned int x_lower      = dim + 1;
  static constexpr unsigned int x_upper      = 2 * dim + 1;
  static constexpr unsigned int l_lower      = 2 * dim + 1;
  static constexpr unsigned int l_upper      = 3 * dim + 1;

  const FEValuesExtractors::Vector velocity_extractor;
  const FEValuesExtractors::Scalar pressure_extractor;
  const FEValuesExtractors::Vector position_extractor;
  const FEValuesExtractors::Vector lambda_extractor;

  /**
   * Quality-of-life functions to check which field a given component is
   */
  static bool is_velocity(const unsigned int component)
  {
    return u_lower <= component && component < u_upper;
  }
  static bool is_pressure(const unsigned int component)
  {
    return p_lower <= component && component < p_upper;
  }
  static bool is_position(const unsigned int component)
  {
    return x_lower <= component && component < x_upper;
  }
  static bool is_lambda(const unsigned int component)
  {
    return l_lower <= component && component < l_upper;
  }

protected:
  ParameterReader<dim> param;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  fixed_mapping;
  std::shared_ptr<Mapping<dim>>                  mapping;
  FESystem<dim>                                  fe;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  const ComponentMask velocity_mask;
  const ComponentMask pressure_mask;
  const ComponentMask position_mask;
  const ComponentMask lambda_mask;

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
  AffineConstraints<double> erroneous_position_constraints;

  types::global_dof_index constrained_pressure_dof = numbers::invalid_dof_index;
  Point<dim>              constrained_pressure_support_point;

  // Position-lambda constraints on the cylinder
  // The affine coefficients c_ij: [dim][{lambdaDOF_j : c_ij}]
  std::vector<std::vector<std::pair<unsigned int, double>>>
                                                  position_lambda_coeffs;
  std::map<types::global_dof_index, Point<dim>>   initial_positions;
  std::map<types::global_dof_index, unsigned int> coupled_position_dofs;

  LA::ParMatrixType              system_matrix;
  std::vector<LA::ParVectorType> previous_solutions;

  std::shared_ptr<Function<dim>> source_terms;
  std::shared_ptr<Function<dim>> exact_solution;

  TableHandler forces_table;
  TableHandler cylinder_position_table;

  SolverControl                                          solver_control;
  std::shared_ptr<PETScWrappers::SparseDirectMUMPSReuse> direct_solver_reuse;

protected:
  /**
   * Exact solution when performing a convergence study with a manufactured
   * solution.
   */
  class MMSSolution : public Function<dim>
  {
  public:
    MMSSolution(const double                                            time,
                const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(n_components, time)
      , mms(mms)
    {}

    virtual void set_time(const double new_time) override
    {
      FunctionTime<double>::set_time(new_time);
      mms.set_time(new_time);
    }

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      if (is_velocity(component))
        return mms.exact_velocity->value(p, component - u_lower);
      else if (is_pressure(component))
        return mms.exact_pressure->value(p);
      else if (is_position(component))
        return mms.exact_mesh_position->value(p, component - x_lower);
      else if (is_lambda(component))
        // For the exact Lagrange multiplier, call the function below.
        // It can only be called at quadrature nodes on faces, where
        // the normal is well-defined.
        return 0.;
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

    /**
     * Exact Lagrange multiplier requires local unit normal vector
     */
    void lagrange_multiplier(const Point<dim>     &p,
                             const double          mu_viscosity,
                             const Tensor<1, dim> &normal_to_solid,
                             Tensor<1, dim>       &lambda) const
    {
      Tensor<2, dim> sigma;
      sigma                 = 0;
      const double pressure = mms.exact_pressure->value(p);
      for (unsigned int d = 0; d < dim; ++d)
        sigma[d][d] = -pressure;
      // Tensor<2, dim> grad_u;
      // mms.exact_velocity->gradient_vj_xi(p, grad_u);
      Tensor<2, dim> grad_u = mms.exact_velocity->gradient_vj_xi(p);
      sigma += mu_viscosity * (grad_u + transpose(grad_u));
      lambda = -sigma * normal_to_solid;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      if (is_velocity(component))
        return mms.exact_velocity->gradient(p, component - u_lower);
      else if (is_pressure(component))
        return mms.exact_pressure->gradient(p);
      else if (is_position(component))
        return mms.exact_mesh_position->gradient(p, component - x_lower);
      else if (is_lambda(component))
        return Tensor<1, dim>();
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

  public:
    // MMS cannot be const since its internal time must be updated
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };

  /**
   * Source term called when performing a convergence study.
   */
  class MMSSourceTerm : public Function<dim>
  {
  public:
    MMSSourceTerm(const double                          time,
                  const Parameters::PhysicalProperties &physical_properties,
                  const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(n_components, time)
      , physical_properties(physical_properties)
      , mms(mms)
    {}

    virtual void set_time(const double new_time) override
    {
      FunctionTime<double>::set_time(new_time);
      mms.set_time(new_time);
    }

    /**
     * Evaluate the combined velocity-pressure-position-lambda source terms.
     */
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;

    /**
     * Gradient of source term, using finite differences
     */
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      const double h = 1e-8;

      Vector<double> vals_plus(gradients.size()), vals_minus(gradients.size());

      for (unsigned int d = 0; d < dim; ++d)
      {
        Point<dim> p_plus = p, p_minus = p;
        p_plus[d] += h;
        p_minus[d] -= h;

        this->vector_value(p_plus, vals_plus);
        this->vector_value(p_minus, vals_minus);

        // Centered finite differences
        for (unsigned int c = 0; c < gradients.size(); ++c)
          gradients[c][d] = (vals_plus[c] - vals_minus[c]) / (2.0 * h);
      }
    }

  protected:
    const Parameters::PhysicalProperties &physical_properties;

    // MMS cannot be const since its internal time must be updated
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };
};

#endif