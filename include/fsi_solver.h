#ifndef FSI_SOLVER_H
#define FSI_SOLVER_H

#include <components_ordering.h>
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
#include <navier_stokes_solver.h>
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
class FSISolverLessLambda : public NavierStokesSolver<dim, true>
{
protected:
  static constexpr unsigned int n_hp_partitions         = 2;
  static constexpr unsigned int index_fe_without_lambda = 0;
  static constexpr unsigned int index_fe_with_lambda    = 1;

  using ScratchData = ScratchDataFSI_hp<dim>;
  using CopyData    = MyCopyData<dim, n_hp_partitions>;

public:
  /**
   * Constructor
   */
  FSISolverLessLambda(const ParameterReader<dim> &param);

  virtual ~FSISolverLessLambda() {}

public:
  virtual void reset_solver_specific_data() override;

  /**
   * Create the AffineConstraints storing the lambda = 0
   * constraints everywhere, except on the boundary of interest
   * on which a weakly enforced no-slip condition is prescribed.
   */
  void create_lagrange_multiplier_constraints();

  /**
   * Create the missing hp dof identities on lines.
   * See incompressible_ns_solver_lambda.h for the full comment.
   */
  void create_hp_line_dof_identities();

  /**
   * Distribute dofs for the hp::FECollection and allocate vectors
   */
  virtual void setup_dofs() override;

  void check_dofs(const AffineConstraints<double> &constraints) const;

  /**
   *
   */
  void create_position_lagrange_mult_coupling_data();

  virtual void create_solver_specific_constraints_data() override
  {
    if (this->param.fsi.enable_coupling)
      create_position_lagrange_mult_coupling_data();
    create_lagrange_multiplier_constraints();
    // create_hp_line_dof_identities();
  }

  /**
   *
   */
  void remove_cylinder_velocity_constraints(
    AffineConstraints<double> &constraints,
    const bool                 remove_velocity_constraints,
    const bool                 remove_position_constraints) const;

  /**
   * Constrain duplicated hp dofs to be the same. For each pair (dof1, dof2)
   * identified by create_hp_line_dof_identities() and stored in
   * hp_dof_identities, it adds to the constraints argument a constraint dof1 =
   * dof2.
   */
  // void
  // add_hp_identities_constraints(AffineConstraints<double> &constraints)
  // const;

  virtual void create_solver_specific_zero_constraints() override;
  virtual void create_solver_specific_nonzero_constraints() override;

  /**
   *
   */
  virtual void create_sparsity_pattern() override;

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
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratchData,
    CopyData                                             &copy_data);

  void copy_local_to_global_matrix(const CopyData &copy_data);

  virtual void compare_analytical_matrix_with_fd() override;

  /**
   *
   */
  virtual void assemble_matrix() override;

  /**
   *
   */
  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchData &scratchData,
                     CopyData    &copy_data);

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
  void compute_lambda_error_on_boundary(double         &lambda_l2_error,
                                        double         &lambda_linf_error,
                                        Tensor<1, dim> &error_on_integral);

  void check_manufactured_solution_boundary();

  /**
   * Errors for lambda on the relevant boundaries
   */
  virtual void compute_solver_specific_errors() override;

  /**
   *
   */
  void compare_forces_and_position_on_obstacle() const;

  /**
   *
   */
  void check_velocity_boundary() const;

  virtual void solver_specific_post_processing() override;

protected:
  virtual std::vector<std::pair<std::string, unsigned int>>
  get_additional_variables_description() const override
  {
    std::vector<std::pair<std::string, unsigned int>> description;
    description.push_back({"lambda", dim});
    return description;
  }

  /**
   * FIXME: This function should probably be templated to work with
   * both a FiniteElement<dim> and a hp::FECollection<dim>
   */
  virtual const FESystem<dim> &get_fe_system() const override
  {
    AssertThrow(false,
                ExcMessage(
                  "You are trying to get the FESystem from the FSISolver, but "
                  "this solver uses a hp::FECollection instead."));
    return *fe_with_lambda;
  }

  virtual bool uses_hp_capabilities() const override { return true; };

  virtual const hp::FECollection<dim> *get_fe_collection() const override
  {
    return fe.get();
  };

  virtual const hp::MappingCollection<dim> *
  get_fixed_mapping_collection() const override
  {
    return &fixed_mapping_collection;
  };

  virtual const hp::MappingCollection<dim> *
  get_moving_mapping_collection() const override
  {
    return &moving_mapping_collection;
  };

  virtual const hp::QCollection<dim> *
  get_cell_quadrature_collection() const override
  {
    return &quadrature_collection;
  };

  virtual const hp::QCollection<dim - 1> *
  get_face_quadrature_collection() const override
  {
    return &face_quadrature_collection;
  };

protected:
  enum
  {
    with_lambda_domain_id,
    without_lambda_domain_id
  };

  static bool
  cell_has_lambda(const typename DoFHandler<dim>::cell_iterator &cell)
  {
    return cell->material_id() == with_lambda_domain_id;
  }

  std::shared_ptr<FESystem<dim>>         fe_with_lambda;
  std::shared_ptr<FESystem<dim>>         fe_without_lambda;
  std::shared_ptr<hp::FECollection<dim>> fe;

  hp::MappingCollection<dim> fixed_mapping_collection;
  hp::MappingCollection<dim> moving_mapping_collection;
  hp::QCollection<dim>       quadrature_collection;
  hp::QCollection<dim - 1>   face_quadrature_collection;

  static constexpr ConstexprComponentOrderingFSI<dim> const_ordering = {};

  FEValuesExtractors::Vector lambda_extractor;
  ComponentMask              lambda_mask;

  /**
   * The id of the mesh boundary on which the weak no-slip condition
   * is enforced. Currently, this is limited to a single boundary.
   */
  types::boundary_id weak_no_slip_boundary_id = numbers::invalid_unsigned_int;

  AffineConstraints<double> lambda_constraints;

  std::set<std::pair<types::global_dof_index, types::global_dof_index>>
    hp_dof_identities;

  // Position-lambda constraints on the cylinder
  // The affine coefficients c_ij: [dim][{lambdaDOF_j : c_ij}]
  std::vector<std::vector<std::pair<unsigned int, double>>>
                                                  lambda_integral_coeffs;
  std::map<types::global_dof_index, unsigned int> coupled_position_dofs;

  bool         has_local_position_master       = false;
  bool         has_local_lambda_accumulator    = false;
  bool         has_global_master_position_dofs = false;
  unsigned int n_ranks_with_position_master;
  unsigned int n_ranks_with_lambda_accumulator;

  std::array<types::global_dof_index, dim> local_position_master_dofs;
  std::array<types::global_dof_index, dim> global_position_master_dofs;

  std::array<types::global_dof_index, dim>          local_lambda_accumulators;
  std::vector<std::vector<types::global_dof_index>> all_lambda_accumulators;

  TableHandler cylinder_position_table;

protected:
  /**
   * Source term.
   */
  class SourceTerm : public Function<dim>
  {
  public:
    SourceTerm(const double                        time,
               const ComponentOrdering            &ordering,
               const Parameters::SourceTerms<dim> &source_terms)
      : Function<dim>(ordering.n_components, time)
      , ordering(ordering)
      , source_terms(source_terms)
    {}

    virtual void set_time(const double new_time) override
    {
      source_terms.set_time(new_time);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      // source_terms.fluid_source is a function with dim+1 components
      for (unsigned int d = 0; d < dim; ++d)
        values[ordering.u_lower + d] = source_terms.fluid_source->value(p, d);
      values[ordering.p_lower] = source_terms.fluid_source->value(p, dim);
      for (unsigned int d = 0; d < dim; ++d)
        values[ordering.x_lower + d] =
          source_terms.pseudosolid_source->value(p, d);
    }

  protected:
    const ComponentOrdering     &ordering;
    Parameters::SourceTerms<dim> source_terms;
  };

  /**
   * Exact solution when performing a convergence study with a manufactured
   * solution.
   */
  class MMSSolution : public Function<dim>
  {
  public:
    MMSSolution(const double             time,
                const ComponentOrdering &ordering,
                const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(ordering.n_components, time)
      , ordering(ordering)
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
      if (ordering.is_velocity(component))
        return mms.exact_velocity->value(p, component - ordering.u_lower);
      else if (ordering.is_pressure(component))
        return mms.exact_pressure->value(p);
      else if (ordering.is_position(component))
        return mms.exact_mesh_position->value(p, component - ordering.x_lower);
      else if (ordering.is_lambda(component))
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
      Tensor<2, dim> grad_u = mms.exact_velocity->gradient_vj_xi(p);
      sigma += mu_viscosity * (grad_u + transpose(grad_u));
      lambda = -sigma * normal_to_solid;
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      if (ordering.is_velocity(component))
        return mms.exact_velocity->gradient(p, component - ordering.u_lower);
      else if (ordering.is_pressure(component))
        return mms.exact_pressure->gradient(p);
      else if (ordering.is_position(component))
        return mms.exact_mesh_position->gradient(p,
                                                 component - ordering.x_lower);
      else if (ordering.is_lambda(component))
        return Tensor<1, dim>();
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

  public:
    const ComponentOrdering                         &ordering;
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };

  /**
   * Source term called when performing a convergence study.
   */
  class MMSSourceTerm : public Function<dim>
  {
  public:
    MMSSourceTerm(
      const double                               time,
      const ComponentOrdering                   &ordering,
      const Parameters::PhysicalProperties<dim> &physical_properties,
      const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(ordering.n_components, time)
      , ordering(ordering)
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
    const ComponentOrdering                         &ordering;
    const Parameters::PhysicalProperties<dim>       &physical_properties;
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };
};

#endif
