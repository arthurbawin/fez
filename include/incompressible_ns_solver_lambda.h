#ifndef INCOMPRESSIBLE_NS_SOLVER_LAMBDA_H
#define INCOMPRESSIBLE_NS_SOLVER_LAMBDA_H

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
#include <deal.II/hp/fe_collection.h>
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
 * Variant of the incompressible NS solver allowing to prescribe weak
 * no-slip boundary conditions with a Lagrange multiplier ("lambda");
 *
 * The Lagrange multiplier is defined on boundary entities of codimension 1,
 * but FESystems can only consist of FE spaces defined on entities of the same
 * dimension. Instead of defining lambda in the whole domain, we do as in
 * deal.II's step 46, and use the hp capabilities to define two cell partitions,
 * one with lambda, and one with a FENothing instead. A FENothing space does not
 * contribute to the number of dofs on those cells, and thus doesn't add any
 * additional dof to the global system. The only useless additional dofs are the
 * non-boundary lambda dofs on the cells touching the prescribed boundary (that
 * is, their interior dofs and those on non-boundary faces). Those dofs are
 * constrained to zero.
 */
template <int dim>
class NSSolverLambda : public NavierStokesSolver<dim>
{
protected:
  static constexpr unsigned int n_hp_partitions         = 2;
  static constexpr unsigned int index_fe_without_lambda = 0;
  static constexpr unsigned int index_fe_with_lambda    = 1;

  using ScratchData = ScratchDataIncompressibleNSLambda<dim>;
  using CopyData    = MyCopyData<dim, n_hp_partitions>;

public:
  /**
   * Constructor
   */
  NSSolverLambda(const ParameterReader<dim> &param);

  virtual ~NSSolverLambda() {}

public:
  /**
   * Create the AffineConstraints storing the lambda = 0
   * constraints everywhere, except on the boundary of interest
   * on which a weakly enforced no-slip condition is prescribed.
   */
  void create_lagrange_multiplier_constraints();

  /**
   * When defining an hp partition, deal.II first creates dofs on the elements
   * as if they were discontinuous, then identifies dofs from adjacent elements
   * that should be the same and receive a unique global dof index. This is done
   * in the hp_*_dof_identities, where * = vertex, line or quad (face). But it
   * seems that for the FESystem considered here and on tetrahedra, it never
   * enters the line dof identities function, and as a result the, e.g., P2
   * velocity field is discontinuous on the elements at the boundary of the
   * partitions.
   *
   * This function identifies those line dofs that were missed and stores them
   * in the vector hp_dof_identities. This vector is afterward used to enforce
   * the identities as constraints.
   *
   * As soon as this is fixed on the deal.II side, these calls won't be
   * necessary anymore.
   */
  void create_hp_line_dof_identities();

  /**
   * Reset the vector hp_dof_identities.
   * Will not be needed once the hp dof identification is fixed in deal.II.
   */
  virtual void reset_solver_specific_data() override;

  virtual void setup_dofs() override;

  /**
   * Create the constraints specific to this solver:
   * - Constrain the lambda in the lambda partition but not on the boundary to
   * zero
   * - Identify and constrain the hp degrees of freedom on lines
   */
  virtual void create_solver_specific_constraints_data() override
  {
    create_lagrange_multiplier_constraints();
    create_hp_line_dof_identities();
  }

  /**
   * FIXME: This function should be more generic, and concerns the geometries
   * where boundaries intersect each other on edges.
   *
   * Remove the velocity constraints from the boundary on which a no-slip
   * condition is enforced, as otherwise the Lagrange multiplier cannot satisfy
   * the no-slip constraint and gets nonsense values.
   */
  void remove_cylinder_velocity_constraints(
    AffineConstraints<double> &constraints) const;

  /**
   * Constrain duplicated hp dofs to be the same. For each pair (dof1, dof2)
   * identified by create_hp_line_dof_identities() and stored in
   * hp_dof_identities, it adds to the constraints argument a constraint dof1 =
   * dof2.
   */
  void
  add_hp_identities_constraints(AffineConstraints<double> &constraints) const;

  virtual void create_solver_specific_zero_constraints() override;
  virtual void create_solver_specific_nonzero_constraints() override;

  /**
   *
   */
  virtual void create_sparsity_pattern() override;

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
  virtual void output_results() override;

  /**
   * Check that the maximum velocity dof on the boundary with weakly enforced
   * no-slip is small emough.
   */
  void check_velocity_boundary() const;

  virtual void solver_specific_post_processing() override;

  /**
   * Compute the "raw" forces on the obstacle.
   * These need to nondimensionalized to obtain the force coefficients.
   */
  void compute_forces(const bool export_table);

  virtual const FESystem<dim> &get_fe_system() const override
  {
    // FIXME: This function is only called when distributing dofs in the base
    // class, but this solver has its own setup_dofs function, and thus should
    // never be called.
    AssertThrow(
      false,
      ExcMessage(
        "NS solver with Lagrange multiplier for no-slip enforcement uses "
        "deal.II's hp tools, and does not have a unique FESystem. This "
        "function should not be called because this solver takes care of "
        "distributing the hp dofs."));
    return *fe_with_lambda;
  }

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

  hp::MappingCollection<dim> mapping_collection;
  hp::QCollection<dim>       quadrature_collection;
  hp::QCollection<dim - 1>   face_quadrature_collection;

  FEValuesExtractors::Vector lambda_extractor;
  ComponentMask              lambda_mask;
  AffineConstraints<double>  lambda_constraints;

  std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
    hp_dof_identities;

  /**
   * The id of the mesh boundary on which the weak no-slip condition
   * is enforced. Currently, this is limited to a single boundary.
   */
  types::boundary_id weak_no_slip_boundary_id = numbers::invalid_unsigned_int;

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
      // No source term for Lagrange multiplier
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
     * Evaluate the combined velocity-pressure-lambda source terms.
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