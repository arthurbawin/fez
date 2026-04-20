#ifndef SOLUTION_RECOVERY_H
#define SOLUTION_RECOVERY_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/numerics/vector_tools_common.h>

#include "error_estimation/patches.h"
#include "parameter_reader.h"
#include "types.h"

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * A namespace for the reconstruction of the numerical solution and its
   * derivatives using successive applications of the Polynomial Preserving
   * Recovery (PPR) from Zhang and Naga [ref].
   *
   * This namespace includes a Scalar and Vector derived classes, to reconstruct
   * scalar or vector-valued fields and their derivatives.
   *
   * TODO: Currently only Scalar is implemented, to reconstruct e.g. a
   * temperature field.
   *
   * A typical use of these classes is as follows:
   *
   * ErrorEstimation::SolutionRecovery::Scalar recovery(...);
   * recovery.compute_least_squares_matrices();
   * recovery.reconstruct_fields();
   * recovery.write_pvtu("recovery.pvtu");
   *
   * The least-squares matrices depend only on the mesh geometry. They need only
   * be computed once, unless the mesh has moved. Once computed, solution and/or
   * derivatives can be reconstructed by simply multiplying the solution at the
   * patch of neighbouring dofs by the least-square matrix of each vertex.
   *
   * The main goal of this namespace is to provide an estimation of the p+1-th
   * order derivatives of a numerical solution of degree p, to use as an
   * anisotropic error estimate and derive a Riemannian metric controlling the
   * interpolation error. Since the Riemannian metric is stored at the mesh
   * vertices using an isoparametric representation, the same choice is made
   * here, and the recoveries are associated with the mesh vertices.
   *
   * FIXME: This means that currently, the patches and reconstructed fields are
   * stored assuming an isoparametric representation. This differs from the
   * original PPR formulation, where the reconstructed gradient is an operator
   * from V_h to V_h x V_h. In the original formulation, however, patches are
   * also only defined at the mesh vertices, then the operator is extended to
   * V_h x V_h by evaluating and averaging the fitted polynomials at the higher
   * order interpolation nodes. For now, there is no evaluation/average yet with
   * the isoparametric approach For instance, assume a P2 finite element
   * solution and a P1 geometry are used. Then the patches are defined only at
   * the P1 nodes (matching the mesh vertices), and these patches include the P2
   * dofs of the FE solution. Fitted polynomials are computed for each P1 node,
   * and, e.g., the reconstructed gradient operator is currently represented as
   * an isoparametric P1 field, and is not extended to be used as a P2 field
   * like the FE solution. To extend it, we would need to represent the
   * reconstructed fields as fields in V_h as well, using a dof_handler of the
   * same degree of the incoming FE solution, with an FESystem including the
   * solution, gradient, hessian, etc. This is not particularly hard to do, I
   * just went for the isoparametric route first, then realized this in the
   * process (-:
   */
  namespace SolutionRecovery
  {
    /**
     * Type describing the nature of the field(s) to reconstruct with the PPR.
     * Each type requires reconstructing the derivatives of lesser order, e.g.,
     * choosing "hessian" will require reconstructing both the solution and its
     * gradient, yielding 1 + dim reconstructions for a scalar field, and dim +
     * dim^2 for a vector-valued field.
     */
    enum RecoveryType
    {
      solution,
      gradient,
      hessian,
      // third_derivatives
    };

    /**
     * Abstract base class for solution and recoveries. Common data and
     * functions to the scalar and vector-valued classes.
     *
     * The base class mainly:
     *  - sets up the data related to the polynomial degree of the solution
     * (number of recoveries/derivatives needed, polynomial bases)
     *  - computes the least-squares matrices, which depend only on the mesh
     * (through the passed patches) and are independent of the nature of the
     * solution (scalar or vector)
     *
     * Then each derived class performs the vmult between the least-squares
     * matrices and their data.
     */
    template <int dim>
    class Base
    {
    public:
      /**
       * Constructor.
       */
      Base(const unsigned int          highest_recovered_derivative,
           const ParameterReader<dim> &param,
           PatchHandler<dim>          &patch_handler,
           const DoFHandler<dim>      &dof_handler,
           const LA::ParVectorType    &solution,
           const FiniteElement<dim>   &fe,
           const Mapping<dim>         &mapping,
           const ComponentMask        &mask);

      /**
       * Perform the reconstruction of the solution and derivatives up to
       * the prescribed @p highest_recovered_derivative given in the constructor.
       */
      void reconstruct_fields();

      /**
       * Compute the integral L^p norm of the error between the reconstructed
       * field
       * of prescribed @p type and the exact field. @p exact_solution must be
       * a function with a number of vector components equal to the total number
       * of reconstructed components, which depends on
       * highest_recovered_derivative:
       *
       * - for scalar variables:
       *
       *   highest_recovered_derivative  |  # of components
       *                0                |          1
       *                1                |        1 + dim
       *                2                |     1 + dim + dim^2
       *               etc.
       * - for vector-valued variables:
       *
       *   highest_recovered_derivative  |  # of components
       *                0                |         dim
       *                1                |       dim + dim^2
       *                2                |    dim + dim^2 + dim^3
       *               etc.
       */
      double
      compute_integral_error(const RecoveryType          type,
                             const VectorTools::NormType norm_type,
                             const Function<dim>        &exact_solution,
                             const Quadrature<dim>      &cell_quadrature) const;

      /**
       * Compute the nodal error between the reconstructed solution and the
       * exact solution at the isoparametric dofs, that is, the vector \ell^p
       * norm of u_exact - u_reconstructed as vectors of \mathbb{R}^{N_dofs}.
       *
       * This function exists because even if the reconstructed solution is more
       * accurate than the FE solution (say, if it is its exact interpolant),
       * computing the L^p norm of its error
       * still requires interpolating with the isoparametric FE, which caps
       * the observed convergence order. For instance, if the reconstructed
       * solution interpolates exactly the exact solution, then this error
       * is zero, whereas the result of compute_integral_error() converges at
       * order p + 1, but is nonzero.
       */
      double compute_nodal_error(const RecoveryType          type,
                                 const VectorTools::NormType norm_type,
                                 const Function<dim> &exact_solution) const;

      /**
       * Return the highest order of the derivatives stored in this object.
       *
       * FIXME: this value is actually highest_recovered_derivative + 1, since
       * the derivative of one additional order is readily available once a
       * lower order derivative has been recovered by polynomial fitting.
       * This should be made consistent throughout this class.
       */
      unsigned int get_highest_stored_derivative() const;

      /**
       * Return the reconstructed @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<double> &
      get_reconstructed_solution(const unsigned int component = 0) const = 0;

      /**
       * Return the reconstructed gradient of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<Tensor<1, dim>> &
      get_reconstructed_gradient(const unsigned int component = 0) const = 0;

      /**
       * Return the reconstructed hessian of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       *
       * FIXME: Maybe it would be better to symmetrize the recovered hessian
       */
      virtual const std::vector<Tensor<2, dim>> &
      get_reconstructed_hessian(const unsigned int component = 0) const = 0;

      /**
       * Write all the reconstructed fields to a pvtu file for visualization.
       */
      virtual void write_pvtu(const std::string &filename) const = 0;

      /**
       * Write the least-squares matrices and polynomials associated
       * representing the reconstructed solution. This function is only intended
       * for debug and unit tests.
       */
      void write_least_squares_systems(std::ostream &out = std::cout) const;

    protected:
      /**
       * Reconstruct a single field (solution or derivatives).
       */
      virtual void reconstruct_field(const unsigned int derivative_order) = 0;

      /**
       * Copy the scalar data stored at the (owned) mesh vertices in @p vertex_data
       * to its isoparametric FE representation stored in @p local_dof_data and @p
       * dof_data.
       *
       * FIXME: we could probably use only the function for tensors, using
       * Tensor<0, dim> for scalar data.
       */
      void vertex_to_isoparametric(
        const std::vector<double> &vertex_data,
        LA::ParVectorType         &local_dof_data,
        LA::ParVectorType         &dof_data,
        const std::vector<std::array<types::global_dof_index, 1>>
          &vertex_to_dofs);

      /**
       * Copy the tensor-valued data stored at the (owned) mesh vertices in
       * @p vertex_data to its isoparametric FE representation stored in
       * @p local_dof_data and @p dof_data.
       */
      template <int rank, int n_components>
      void vertex_to_isoparametric(
        const std::vector<Tensor<rank, dim>> &vertex_data,
        LA::ParVectorType                    &local_dof_data,
        LA::ParVectorType                    &dof_data,
        const std::vector<std::array<types::global_dof_index, n_components>>
          &vertex_to_dofs);

      /**
       * Evaluate at @p p the gradient of the polynomial described by the basis
       * @p polynomial_space and the coefficients @p polynomial_coeffs.
       * @p basis_gradients is a used as a temporary vector to store the
       * gradient of the polynomial basis at @p p.
       */
      Tensor<1, dim> evaluate_polynomial_gradient(
        const Point<dim>             &p,
        const PolynomialSpace<dim>   &polynomial_space,
        const dealii::Vector<double> &polynomial_coeffs,
        std::vector<Tensor<1, dim>>  &basis_gradients);

    protected:
      /**
       * Highest derivative order to reconstruct (1 = gradient, 2 = hessian,
       * etc.)
       */
      const unsigned int highest_recovered_derivative;

      /**
       * Reference to the parameters
       */
      const ParameterReader<dim> &param;

      /**
       * Reference to the PatchHandler and to the vector of patches at the
       * (owned) mesh vertices. The PatchHandler is not const because the
       * patches may be increased when computing the least-squares matrices, if
       * the initial matrix is not full rank.
       */
      PatchHandler<dim>       &patch_handler;
      std::vector<Patch<dim>> &patches;

      /**
       * The dof handler, FE space and mapping from the FE solution
       */
      const DoFHandler<dim>    &dof_handler;
      const FiniteElement<dim> &fe;
      const Mapping<dim>       &mapping;

      /**
       * Component mask for the field to reconstruct.
       */
      const ComponentMask mask;

      /**
       * Isoparametric dof handler, fe and mapping, to store the recovery data
       * at the mesh vertices used to describe the geometry.
       */
      DoFHandler<dim>                     isoparam_dh;
      std::unique_ptr<Mapping<dim>>       isoparam_mapping;
      std::unique_ptr<FiniteElement<dim>> isoparam_fe;

      /**
       * Total number of vector components in the isoparametric solution (see
       * also comments for compute_integral_error())
       */
      unsigned int n_isoparam_components;

      /**
       * Communicator and conditional ostream
       */
      MPI_Comm           mpi_communicator;
      ConditionalOStream pcout;

      /**
       * Number of mesh vertices on this partition and mask
       */
      const unsigned int n_vertices;
      std::vector<bool>  owned_vertices;

      /**
       * The set of owned dofs from the FE solution, and the set of relevant
       * dofs, augmented with the non-local patch dofs that were gathered from
       * other ranks. This set is used to initialize
       * solution_with_additional_ghosts, so that the solution at non-local dofs
       * in patches can be updated.
       */
      IndexSet          locally_owned_dofs;
      IndexSet          relevant_dofs;
      LA::ParVectorType local_solution;
      LA::ParVectorType solution_with_additional_ghosts;

      /**
       * Sets of owned and relevant dofs from the isoparametric dof_handler
       */
      IndexSet locally_owned_isoparam_dofs;
      IndexSet locally_relevant_isoparam_dofs;

      /**
       * Non-hosted and ghosted isoparametric representation of the recovered
       * fields.
       */
      LA::ParVectorType local_isoparam_solution;
      LA::ParVectorType isoparam_solution;

      /**
       * Masks and component select functions for the reconstructed fields.
       */
      ComponentMask                                 solution_mask;
      ComponentMask                                 gradient_mask;
      ComponentMask                                 hessian_mask;
      std::unique_ptr<ComponentSelectFunction<dim>> solution_comp_select;
      std::unique_ptr<ComponentSelectFunction<dim>> gradient_comp_select;
      std::unique_ptr<ComponentSelectFunction<dim>> hessian_comp_select;

      /**
       * Degree of the FE field whose derivatives are reconstructed.
       * Polynomials of degree "degree" + 1 will be fitted.
       */
      const unsigned int degree;

      /**
       * FIXME: almost all of these are unused and can be removed
       */
      unsigned int dim_recovery_basis;
      unsigned int dim_gradient_basis;
      unsigned int n_fields_to_recover;
      unsigned int n_recovered_fields;
      unsigned int n_derivatives_to_store;
      unsigned int n_derivatives_computed;

      /**
       * Polynomial basis to fit a degree p + 1 polynomial
       */
      std::unique_ptr<PolynomialSpace<dim>> monomials_recovery;

      /**
       * Gradients at the origin of the monomials forming the recovery basis
       */
      std::vector<Tensor<1, dim>> gradients_of_recovery_monomials;

      /**
       * These vectors are dummies, needed to use the evaluate(...) function on
       * a PolynomialSpace. They must remain empty so as to only evaluate the
       * gradient.
       */
      std::vector<double>         empty_polynomial_space_values;
      std::vector<Tensor<2, dim>> empty_polynomial_space_grad_grads;
      std::vector<Tensor<3, dim>> empty_polynomial_space_third_derivatives;
      std::vector<Tensor<4, dim>> empty_polynomial_space_fourth_derivatives;

      /**
       * The least squares matrix for each (owned) mesh vertex.
       */
      const std::vector<FullMatrix<double>> &least_squares_matrices;

      /**
       * For each owned mesh vertex and each recovered field, the multivariate
       * polynomial centered at the vertex.
       */
      std::vector<Vector<double>> recoveries_coefficients;

      // FIXME : add homogeneous error polynomials?
      // // These coefficients already include the binomial coefficients.
      // // [vertex][component][error_coefficients]
      // std::vector<std::vector<Vector<double>>> homogeneous_errors;
    };

    /**
     * Solution and derivatives recovery for a scalar variable.
     */
    template <int dim>
    class Scalar : public Base<dim>
    {
    public:
      /**
       * An alias for the data type of values this class
       * represents. Since we deal with a single components, the value type is a
       * scalar double.
       */
      using value_type = double;

      /**
       * An alias for the type of gradients this class represents.
       * Here, for a scalar component of the finite element, the gradient is a
       * <code>Tensor@<1,dim@></code>.
       */
      using gradient_type = Tensor<1, dim>;

      /**
       * An alias for the type of second derivatives this class
       * represents. Here, for a scalar component of the finite element, the
       * Hessian is a <code>Tensor@<2,dim@></code>.
       */
      using hessian_type = Tensor<2, dim>;

      /**
       * An alias for the type of third derivatives this class
       * represents. Here, for a scalar component of the finite element, the
       * Third derivative is a <code>Tensor@<3,dim@></code>.
       */
      using third_derivative_type = Tensor<3, dim>;

      /**
       * Constructor.
       */
      Scalar(const unsigned int          highest_recovered_derivative,
             const ParameterReader<dim> &param,
             PatchHandler<dim>          &patch_handler,
             const DoFHandler<dim>      &dof_handler,
             const LA::ParVectorType    &solution,
             const FiniteElement<dim>   &fe,
             const Mapping<dim>         &mapping,
             const ComponentMask        &mask = {});

      /**
       * Return the reconstructed @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<double> &get_reconstructed_solution(
        const unsigned int component = 0) const override;

      /**
       * Return the reconstructed gradient of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<Tensor<1, dim>> &get_reconstructed_gradient(
        const unsigned int component = 0) const override;

      /**
       * Return the reconstructed hessian of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<Tensor<2, dim>> &get_reconstructed_hessian(
        const unsigned int component = 0) const override;

      /**
       * Write all the reconstructed fields to a pvtu file for visualization.
       */
      virtual void write_pvtu(const std::string &filename) const override;

    protected:
      /**
       * Reconstruct a single field (solution or derivatives).
       */
      virtual void
      reconstruct_field(const unsigned int derivative_order) override;

      /**
       * Compute the weights used in the PPR operator to define the gradient
       * at non-vertices dofs.
       *
       * TODO: Add more comments.
       */
      void compute_patches_averaging_weights();

      /**
       * Update the copy of the solution vector with the values of the
       * @p component-th derivative component to reconstruct.
       * This allows updating the ghosted values through the solution vector's
       * mechanisms.
       *
       * TODO: Add more comments.
       */
      void update_local_solution(const unsigned int derivative_order,
                                 const unsigned int component);

    private:
      /**
       * The reconstructed fields, stored at (owned) mesh vertices.
       */
      std::vector<value_type>    recovered_solution_at_vertices;
      std::vector<gradient_type> recovered_gradient_at_vertices;
      std::vector<hessian_type>  recovered_hessian_at_vertices;

      /**
       * Maps to go from data stored at (owned) mesh vertices to their
       * representation as an isoparametric FE solution:
       *
       * Map from reconstructed solution at vertices to dof representation, and
       * vice versa.
       */
      std::vector<std::array<types::global_dof_index, 1>>
        vertices_to_solution_dofs;
      std::vector<std::pair<types::global_vertex_index, unsigned int>>
        solution_dofs_to_vertices;

      /**
       * Map from reconstructed gradient at vertices to dof representation, and
       * vice versa.
       */
      std::vector<std::array<types::global_dof_index,
                             gradient_type::n_independent_components>>
        vertices_to_gradient_dofs;
      std::vector<std::pair<types::global_vertex_index, unsigned int>>
        gradient_dofs_to_vertices;

      /**
       * Map from reconstructed hessian at vertices to dof representation, and
       * vice versa.
       */
      std::vector<std::array<types::global_dof_index,
                             hessian_type::n_independent_components>>
        vertices_to_hessian_dofs;
      std::vector<std::pair<types::global_vertex_index, unsigned int>>
        hessian_dofs_to_vertices;
    };

    /**
     * Solution and derivatives recovery for a vector-valued variable
     */
    template <int dim>
    class Vector : public Base<dim>
    {
    public:
      Vector(const unsigned int          highest_recovered_derivative,
             const ParameterReader<dim> &param,
             PatchHandler<dim>          &patch_handler,
             const DoFHandler<dim>      &dof_handler,
             const LA::ParVectorType    &solution,
             const FiniteElement<dim>   &fe,
             const Mapping<dim>         &mapping,
             const ComponentMask        &mask = {})
        : Base<dim>(highest_recovered_derivative,
                    param,
                    patch_handler,
                    dof_handler,
                    solution,
                    fe,
                    mapping,
                    mask)
      {
        // TODO
        DEAL_II_NOT_IMPLEMENTED();
      }
    };
  } // namespace SolutionRecovery
} // namespace ErrorEstimation

/* ---------------- template and inline functions ----------------- */

namespace ErrorEstimation
{
  namespace SolutionRecovery
  {
    template <int dim>
    unsigned int Base<dim>::get_highest_stored_derivative() const
    {
      return highest_recovered_derivative;
    }

    template <int dim>
    void Base<dim>::vertex_to_isoparametric(
      const std::vector<double>                                 &vertex_data,
      LA::ParVectorType                                         &local_dof_data,
      LA::ParVectorType                                         &dof_data,
      const std::vector<std::array<types::global_dof_index, 1>> &vertex_to_dofs)
    {
      AssertDimension(vertex_data.size(), vertex_to_dofs.size());
      AssertDimension(vertex_to_dofs.size(), n_vertices);

      for (types::global_vertex_index v = 0; v < n_vertices; ++v)
      {
        if (owned_vertices[v])
        {
          const double val = vertex_data[v];
          const auto  &dof = vertex_to_dofs[v][0];
          if (locally_owned_isoparam_dofs.is_element(dof))
            local_dof_data[dof] = val;
        }
      }
      local_dof_data.compress(VectorOperation::insert);
      dof_data = local_dof_data;
    }

    template <int dim>
    template <int rank, int n_components>
    void Base<dim>::vertex_to_isoparametric(
      const std::vector<Tensor<rank, dim>> &vertex_data,
      LA::ParVectorType                    &local_dof_data,
      LA::ParVectorType                    &dof_data,
      const std::vector<std::array<types::global_dof_index, n_components>>
        &vertex_to_dofs)
    {
      static_assert(
        std::vector<Tensor<rank, dim>>::value_type::n_independent_components ==
          n_components,
        "n_components must match the number of independent components of a "
        "Tensor<rank, dim>");
      AssertDimension(vertex_data.size(), vertex_to_dofs.size());
      AssertDimension(vertex_to_dofs.size(), n_vertices);

      for (types::global_vertex_index v = 0; v < n_vertices; ++v)
      {
        if (owned_vertices[v])
        {
          const auto &t    = vertex_data[v];
          const auto &dofs = vertex_to_dofs[v];
          for (unsigned int c = 0; c < n_components; ++c)
            if (locally_owned_isoparam_dofs.is_element(dofs[c]))
              local_dof_data[dofs[c]] = t[t.unrolled_to_component_indices(c)];
        }
      }
      local_dof_data.compress(VectorOperation::insert);
      dof_data = local_dof_data;
    }

    template <int dim>
    const std::vector<double> &
    Scalar<dim>::get_reconstructed_solution(const unsigned int) const
    {
      return recovered_solution_at_vertices;
    }

    template <int dim>
    const std::vector<Tensor<1, dim>> &
    Scalar<dim>::get_reconstructed_gradient(const unsigned int) const
    {
      return recovered_gradient_at_vertices;
    }

    template <int dim>
    const std::vector<Tensor<2, dim>> &
    Scalar<dim>::get_reconstructed_hessian(const unsigned int) const
    {
      return recovered_hessian_at_vertices;
    }

  } // namespace SolutionRecovery
} // namespace ErrorEstimation

#endif
