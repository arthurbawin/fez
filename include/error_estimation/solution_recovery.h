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
   * derivatives using the polynomial-preserving recovery (PPR) from Zhang and
   * Naga [ref], an alternative to the classic Zhu-Zienkiewicz gradient recovery
   * operator. This operator is based on local reconstructions of the numerical
   * solution around mesh vertices, fitting polynomial of degree p + 1 in a
   * least-squares sense.
   *
   * This operator provides a C^1 field for the solution and its derivatives,
   * and defines in particular degrees of freedom for the gradient, hessian,
   * etc., which are not typically defined in a finite element setting.
   *
   * Possible use cases of this operator are as follows :
   *
   * - using the reconstructed solution as a spatial error estimator. Since the
   * reconstructed solution typically converges faster than the numerical
   * solution, it can be used as an error indicator.
   * - using the reconstructed derivatives of degree p + 1 as an anisotropic
   * interpolation error estimate, to define a Riemannian metric controlling the
   * interpolation error.
   * - using the reconstructed gradients to define, e.g., a C^1 vorticity field
   * for visualization.
   *
   *
   * <h3>Definition of the gradient recovery operator</h3>
   *
   * From Zhang and Naga's original paper, the PPR gradient recovery operator is
   * defined in two steps:
   *
   * - first, least-squares polynomial fittings are computed at each mesh
   * vertex, to obtain a polynomial representation of the FE solution one degree
   * higher, based on neighbouring solution values. The gradient of each
   * polynomial at 0 in the local frame defines the value of the gradient
   * operator at that mesh vertex, that is, the value of its associated dofs
   * (one dof per space dimension). For scalar fields, this step defines an
   * operator from V_h to (P_geo)^dim, where P_geo is the polynomial space used
   * to represent the geometry. Thus, this step defines an isoparametric
   * representation of the smoothed gradient (e.g., (P_1)^dim for linear
   * meshes).
   *
   * - the operator is then extended to (V_h)^dim, by defining its values at the
   * non-vertices dofs (e.g., P_2 edge dofs, P_3 edge and element dofs). This is
   * done by averaging the evaluations at these dofs' support points, the
   * gradient of the fitted polynomials associated with their adjacent mesh
   * vertices, using barycentric weights.
   *
   * In some applications, defining the full operator in (V_h)^dim in not
   * actually needed, as the smoothed derivatives are only needed at the mesh
   * vertices. This is for instance the case when computing a Riemannian metric
   * for anisotropic mesh adaptation, since metric-based remeshers typically
   * only support a linear background mesh and thus require the metric at the
   * mesh vertices only, requiring in turn the higher-order derivatives at the
   * mesh vertices only. In that case, it is sufficient to keep only an
   * isoparametric representation of the smoothed derivatives,
   * which can be achieved with the @p isoparametric constructor argument.
   *
   *
   * <h3>Definition of the solution recovery operator</h3>
   *
   * The solution recovery operator is defined in a similar way, by first
   * evaluating the fitted polynomials at the origin, then extending the
   * operator to V_h (or (V_h)^dim in the case of vector-valued fields).
   *
   * <h3>Definition of the recovery operator for higher order derivatives</h3>
   *
   * Higher-order derivatives are recovered through successive application of
   * the PPR operator, first recovering the solution and obtaining its smoothed
   * gradient, then recovering the gradient and obtaining its smoothed hessian,
   * etc.
   *
   * Successive differentiations of numerical data are known to quickly become
   * noisy, especially near the boundaries. In the case where the higher-order
   * derivatives are only needed at the mesh vertices (that is, if @p isoparametric
   * is set to true), these derivatives can be computed at these vertices with a
   * single polynomial fitting, taking the derivatives of order up to p + 1 of
   * the fitted polynomial of degree p + 1. For instance, in the case of a
   * linear FE field, the approximate hessians can be obtained through a single
   * fitting of a quadratic polynomial p(x,y) = a + bx + cy + dx^2 + exy + fy^2,
   * then setting the constant hessian hess(p) = [2d e; e 2f] as the hessian at
   * that
   * vertex. This option can be enabled by setting @p single_reconstruction to true.
   *
   *
   * <h3>How to use this operator</h3>
   *
   * This namespace includes a Scalar and Vector derived classes, to reconstruct
   * scalar or vector-valued fields and their derivatives.
   *
   * TODO: Currently only Scalar is implemented, to reconstruct e.g. a
   * temperature field.
   *
   * This operator requires a PatchHandler, which handles the creation of
   * patches of dof and support points around each mesh vertex. Once the patches
   * have been created with patch_handler.build_patches (and optionally updated
   * with update_patches(mapping) if using a moving mapping), a typical use of
   * this namespace is as follows:
   *
   * ErrorEstimation::SolutionRecovery::Scalar recovery(...);
   * recovery.reconstruct_fields(solution);
   * recovery.write_pvtu("recovery.pvtu");
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
      third_derivatives
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
           const ComponentMask        &mask,
           const bool                  isoparametric,
           const bool                  single_reconstruction);

      /**
       * Perform the reconstruction of the solution and derivatives up to
       * the prescribed @p highest_recovered_derivative given in the constructor.
       */
      void reconstruct_fields(const LA::ParVectorType &solution);

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
                             const Mapping<dim>         &mapping,
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
                                 const Mapping<dim>         &mapping,
                                 const Function<dim> &exact_solution) const;

      /**
       * Return the polynomial degree of the solution used to create this
       * recovery.
       */
      unsigned int get_solution_degree() const;

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
       * Return a vector of booleans stating which local mesh vertices are
       * owned.
       */
      const std::vector<bool> &get_owned_vertices() const;

      /**
       * Return the number of owned vertices on this rank.
       */
      unsigned int get_n_owned_vertices() const;

      /**
       * Return true if this object stores data for mesh vertex @p v.
       * Simply return whether this mesh vertex is owned on this partition.
       */
      bool has_mesh_vertex(const types::global_vertex_index v) const;

      /**
       * Return the DofHandler associated with this object.
       */
      const DoFHandler<dim> &get_dof_handler() const;

      /**
       * Return the FESystem representing the reconstructed fields.
       */
      const FiniteElement<dim> &get_fe() const;

      /**
       * Return a new FEValues initialized with this object's FESystem and its
       * appropriate UpdateFlags.
       */
      FEValues<dim> get_fe_values(const Mapping<dim>    &mapping,
                                  const Quadrature<dim> &quadrature) const;

      /**
       * Return the mask to select the reconstructed solution from this object's
       * solution vector.
       */
      ComponentMask get_solution_mask() const;

      /**
       * Return the mask to select the reconstructed gradient from this object's
       * solution vector.
       */
      ComponentMask get_gradient_mask() const;

      /**
       * Return the mask to select the reconstructed hessian from this object's
       * solution vector.
       */
      ComponentMask get_hessian_mask() const;

      /**
       * Return the vector of reconstructed FE fields.
       */
      const LA::ParVectorType &get_reconstructions() const;

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
       * Return the reconstructed 3rd derivatives of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<Tensor<3, dim>> &
      get_reconstructed_third_derivatives(
        const unsigned int component = 0) const = 0;

      /**
       * Write all the reconstructed fields to a pvtu file for visualization.
       */
      virtual void
      write_pvtu(const Mapping<dim> &mapping,
                 const std::string  &filename_without_extension) const = 0;

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
      template <int rank, int n_tensor_components>
      void vertex_to_isoparametric(
        const std::vector<Tensor<rank, dim>>               &vertex_data,
        LA::ParVectorType                                  &local_dof_data,
        LA::ParVectorType                                  &dof_data,
        const std::vector<std::array<types::global_dof_index,
                                     n_tensor_components>> &vertex_to_dofs);

      /**
       * Convenience function to solve the least square problem at a mesh
       * vertex, and store the result in @p scaled_polynomial_coeffs.
       *
       * The least-squares matrix (from the PatchHandler) is expected to have
       * been computed from the scaled position of the patch support points,
       * in the local frame centered at the patch center. Thus, this function
       * computes the scaled polynomial \hat{p}(\hat{x}), where
       *
       * \hat{x}_i := (x_i - x_center,i) / scaling_i.
       *
       * This polynomial is such that p(x) = \hat{p}(\hat{x}), so there is no
       * need to scale back to evaluate to polynomial itself, but its
       * derivatives must be scaled back according to the chain rule, for
       * instance:
       *
       * grad p(x)
       *  = (\partial \hat{x}/\partial x)^T \cdot \hat{grad}\hat{p}(\hat{x}).
       *
       * The transformation \partial \hat{x}/\partial x is diagonal, so one
       * simply has to divide each component of the gradient, hessian, etc. by
       * the corresponding scaling component, e.g.:
       *
       * (grad p(x))_i = (\hat{grad}\hat{p}(\hat{x}))_i / scaling_i,
       *
       * (hess p(x))_ij
       *  = (\hat{hess}\hat{p}(\hat{x}))_ij / (scaling_i * scaling_j),
       *
       * and so on.
       */
      void solve_least_squares_problem(
        const unsigned int      vertex_index,
        dealii::Vector<double> &scaled_polynomial_coeffs);

      /**
       * Evaluate at @p p the scaled polynomial described by the basis
       * @p polynomial_space and the coefficients @p polynomial_coeffs.
       * @p basis is used as a temporary vector to store the polynomial basis
       * at @p p.
       */
      double
      evaluate_polynomial(const Point<dim>             &p,
                          const PolynomialSpace<dim>   &polynomial_space,
                          const dealii::Vector<double> &polynomial_coeffs,
                          std::vector<double>          &basis);

      /**
       * Evaluate at @p p the gradient of the scaled polynomial described by the
       * basis @p polynomial_space and the coefficients @p scaled_polynomial_coeffs,
       * then scale back the result to yield the gradient in physical
       * coordinates according to the comments for the
       * solve_least_squares_problem function.
       * @p basis_gradients is used as a temporary vector to store the
       * gradient of the polynomial basis at @p p.
       */
      Tensor<1, dim> evaluate_polynomial_gradient(
        const Point<dim>             &p,
        const PolynomialSpace<dim>   &polynomial_space,
        const dealii::Vector<double> &scaled_polynomial_coeffs,
        std::vector<Tensor<1, dim>>  &basis_gradients,
        const Point<dim>             &scaling);

      /**
       * Evaluate at 0 the gradient of the scaled polynomial described by the
       * local frame coefficients @p scaled_polynomial_coeffs, and scale back the
       * result to yield the gradient in physical coordinates according to the
       * comments for the solve_least_squares_problem function.
       */
      Tensor<1, dim>
      gradient_at_origin(const dealii::Vector<double> &scaled_polynomial_coeffs,
                         const Point<dim>             &scaling) const;

      /**
       * Evaluate at 0 the hessian of the scaled polynomial described by the
       * local frame coefficients @p scaled_polynomial_coeffs, and scale back
       * the result.
       */
      Tensor<2, dim>
      hessian_at_origin(const dealii::Vector<double> &scaled_polynomial_coeffs,
                        const Point<dim>             &scaling) const;

      /**
       * Evaluate at 0 the third derivatives of the scaled polynomial described
       * by the local frame coefficients @p scaled_polynomial_coeffs, and scale
       * back the result.
       */
      Tensor<3, dim> third_derivatives_at_origin(
        const dealii::Vector<double> &scaled_polynomial_coeffs,
        const Point<dim>             &scaling) const;

    protected:
      /**
       * Highest derivative order to reconstruct (1 = gradient, 2 = hessian,
       * etc.)
       */
      const unsigned int highest_recovered_derivative;

      /**
       * A bool specifying whether the PPR operator should be stored only at
       * the mesh vertices, and thus define an isoparametric field.
       */
      const bool isoparametric;

      /**
       * A bool specifying whether the higher-order derivatives should be
       * computed by taking the higher-order derivatives a single polynomial (if
       * true), or by successive reconstructions-first derivatives (if false).
       *
       * Only relevant if isoparametric is true.
       */
      const bool single_reconstruction;

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
      const DoFHandler<dim>    &solution_dh;
      const FiniteElement<dim> &solution_fe;
      const Mapping<dim>       &solution_mapping;

      /**
       * Component mask for the field to reconstruct.
       */
      const ComponentMask mask;

      /**
       * The dof handler and fe used to represent the solution and derivatives.
       * For a scalar field, the FESystem consists for instance of [FE_P,
       * FE_P^dim, FE_P^(dim*dim), etc.], to represent the reconstructed
       * solution, gradient, hessian, etc. at each dof. This FESystem uses the
       * same FE space as the one for the geometry if isoparemtric is true.
       *
       * In any case, an isoparametric space (isoparam_fe) is needed to compute
       * the averaging weights used to define the operator at non-vertices dofs.
       */
      DoFHandler<dim>                     dh;
      std::unique_ptr<FiniteElement<dim>> fe;
      std::unique_ptr<FiniteElement<dim>> isoparam_fe;

      /**
       * Total number of vector components in the isoparametric solution (see
       * also comments for compute_integral_error())
       */
      unsigned int n_components;

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
      IndexSet locally_owned_dofs;
      IndexSet relevant_dofs;

      /**
       * A copy of the main solver's solution vectors. Used to assign the values
       * of each field to recover at patches dofs.
       */
      LA::ParVectorType local_solution;
      LA::ParVectorType solution_with_additional_ghosts;

      /**
       * Sets of owned and relevant dofs from the isoparametric dof_handler
       */
      IndexSet locally_owned_recovery_dofs;
      IndexSet locally_relevant_recovery_dofs;

      /**
       * Representations of the recovered fields (e.g., u, grad(u), hess(u)) as
       * a combination of scalar and vector-valued fields.
       */
      LA::ParVectorType local_recovery_solution;
      LA::ParVectorType recovery_solution;

      /**
       * A map storing the owning MPI rank of each dof of the parallel recovery
       * solution
       */
      std::map<types::global_dof_index, types::subdomain_id> ghost_owners;

      /**
       * Masks and component select functions for the reconstructed fields.
       */
      ComponentMask                                 solution_mask;
      ComponentMask                                 gradient_mask;
      ComponentMask                                 hessian_mask;
      ComponentMask                                 third_derivatives_mask;
      std::unique_ptr<ComponentSelectFunction<dim>> solution_comp_select;
      std::unique_ptr<ComponentSelectFunction<dim>> gradient_comp_select;
      std::unique_ptr<ComponentSelectFunction<dim>> hessian_comp_select;
      std::unique_ptr<ComponentSelectFunction<dim>>
        third_derivatives_comp_select;

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
       * Derivatives at the origin of the monomials forming the recovery basis
       */
      std::vector<Tensor<1, dim>> gradients_of_recovery_monomials;
      std::vector<Tensor<2, dim>> hessians_of_recovery_monomials;
      std::vector<Tensor<3, dim>> third_derivatives_of_recovery_monomials;

      /**
       * These vectors are dummies, needed to use the evaluate(...) function on
       * a PolynomialSpace. They must remain empty so as to only evaluate the
       * gradient.
       */
      std::vector<double>         empty_polynomial_space_values;
      std::vector<Tensor<1, dim>> empty_polynomial_space_grads;
      std::vector<Tensor<2, dim>> empty_polynomial_space_grad_grads;
      std::vector<Tensor<3, dim>> empty_polynomial_space_third_derivatives;
      std::vector<Tensor<4, dim>> empty_polynomial_space_fourth_derivatives;

      /**
       * The least squares matrix for each (owned) mesh vertex.
       */
      const std::vector<FullMatrix<double>> &least_squares_matrices;

      /**
       * For each derivative order (starting at 0), tensor component, and owned
       * mesh vertex, the coefficients of the multivariate polynomial centered
       * at the vertex and obtained by least-squares fitting.
       */
      std::vector<std::vector<std::vector<Vector<double>>>>
        recoveries_coefficients;
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
       * Offsets in the various vectors for each quantity.
       */
      static constexpr unsigned int solution_offset = 0;
      static constexpr unsigned int gradient_offset = 1;
      static constexpr unsigned int hessian_offset  = 1 + dim;
      static constexpr unsigned int third_derivative_offset =
        1 + dim + dim * dim;

      /**
       * Number of components for the solution (since we cannot use
       * double::n_independent_components)
       */
      static constexpr unsigned int n_solution_components = 1;

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
             const ComponentMask        &mask                  = {},
             const bool                  isoparametric         = true,
             const bool                  single_reconstruction = false);

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
       * Return the reconstructed 3rd derivatives of the @p component-th component of the solution,
       * stored at the (owned) mesh vertices of this partition.
       */
      virtual const std::vector<Tensor<3, dim>> &
      get_reconstructed_third_derivatives(
        const unsigned int component = 0) const override;

      /**
       * Return the map from the solver's solution dof to the recovery dof
       * in this object's solution vector.
       */
      const std::map<types::global_dof_index,
                     std::array<types::global_dof_index, 1>> &
      get_solution_to_recovery_dof_map() const;

      /**
       * Return the map from the solver's solution dof to the gradients dof
       * in this object's solution vector.
       */
      const std::map<types::global_dof_index,
                     std::array<types::global_dof_index,
                                gradient_type::n_independent_components>> &
      get_solution_to_gradient_dof_map() const;

      /**
       * Return the map from the solver's solution dof to the hessians dof
       * in this object's solution vector.
       */
      const std::map<types::global_dof_index,
                     std::array<types::global_dof_index,
                                hessian_type::n_independent_components>> &
      get_solution_to_hessian_dof_map() const;

      /**
       * Write all the reconstructed fields to a pvtu file for visualization.
       */
      virtual void
      write_pvtu(const Mapping<dim> &mapping,
                 const std::string  &filename_without_extension) const override;

    protected:
      /**
       * Map each solution dof (for e.g. a field u) to the dofs of the
       * reconstructed u, grad(u), hess(u), etc. at the same support point.
       */
      void create_solution_dofs_to_recovery_dofs_map();

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
                                 const unsigned int reconstruction_component,
                                 const unsigned int gradient_component);

      /**
       * When using a non-isoparametric representation of the PPR operator, this
       * functions completes the operator by assigning the missing values at
       * non-vertex dofs, by evaluating the polynomials from adjacent vertices
       * (stored in recoveries_coefficients) and averaging these evaluations,
       * using the pre-computed weights. If @p compute_gradient is true, then
       * the gradient of these polynomials are evaluated and averaged instead of
       * the polynomials themselves.
       *
       * Communication is involved to exchange contributions from patch dofs on
       * remote partitions.
       *
       * The result of these evaluations is stored in the @p component-th vector
       * component of local_recovery_solution, then the ghost values are updated
       * in recovery_solution.
       */
      void evaluate_and_average_recovery_solution(
        const RecoveryType type,
        const unsigned int derivative_order,
        const unsigned int component,
        const bool         compute_gradient = true);

      /**
       * Update the recovery solution with the last polynomials computed.
       * Transfer the values at vertices to their dof
       * representation, and if not isoparametric, extend the operator by
       * calling evaluate_and_average_recovery_solution.
       */
      void
      update_recovery_solution(const RecoveryType type,
                               const unsigned int derivative_order,
                               const unsigned int reconstruction_component);

    private:
      /**
       * The reconstructed fields, stored at (owned) mesh vertices.
       */
      std::vector<value_type>    recovered_solution_at_vertices;
      std::vector<gradient_type> recovered_gradient_at_vertices;
      std::vector<hessian_type>  recovered_hessian_at_vertices;
      std::vector<third_derivative_type>
        recovered_third_derivatives_at_vertices;

      /**
       * Maps to go from data stored at (owned) mesh vertices to their
       * representation as an isoparametric FE solution:
       */
      std::vector<std::array<types::global_dof_index, 1>>
        vertices_to_solution_dofs;
      std::vector<std::array<types::global_dof_index,
                             gradient_type::n_independent_components>>
        vertices_to_gradient_dofs;
      std::vector<std::array<types::global_dof_index,
                             hessian_type::n_independent_components>>
        vertices_to_hessian_dofs;
      std::vector<std::array<types::global_dof_index,
                             third_derivative_type::n_independent_components>>
        vertices_to_third_derivatives_dofs;

      /**
       * Maps from FE solution dofs (in local_solution) to the dofs of their
       * reconstructed fields (in local_recovery_solution).
       */
      std::map<types::global_dof_index, std::array<types::global_dof_index, 1>>
        solution_dofs_to_recovery_dofs;
      std::map<types::global_dof_index,
               std::array<types::global_dof_index,
                          gradient_type::n_independent_components>>
        solution_dofs_to_gradient_dofs;
      std::map<types::global_dof_index,
               std::array<types::global_dof_index,
                          hessian_type::n_independent_components>>
        solution_dofs_to_hessian_dofs;
      std::map<types::global_dof_index,
               std::array<types::global_dof_index,
                          third_derivative_type::n_independent_components>>
        solution_dofs_to_third_derivatives_dofs;
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
             const ComponentMask        &mask                  = {},
             const bool                  isoparametric         = true,
             const bool                  single_reconstruction = false)
        : Base<dim>(highest_recovered_derivative,
                    param,
                    patch_handler,
                    dof_handler,
                    solution,
                    fe,
                    mapping,
                    mask,
                    isoparametric,
                    single_reconstruction)
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
    inline unsigned int Base<dim>::get_solution_degree() const
    {
      return degree;
    }

    template <int dim>
    inline unsigned int Base<dim>::get_highest_stored_derivative() const
    {
      return highest_recovered_derivative;
    }

    template <int dim>
    inline unsigned int Base<dim>::get_n_owned_vertices() const
    {
      return std::count(owned_vertices.begin(), owned_vertices.end(), true);
    }

    template <int dim>
    inline const std::vector<bool> &Base<dim>::get_owned_vertices() const
    {
      return owned_vertices;
    }

    template <int dim>
    inline bool
    Base<dim>::has_mesh_vertex(const types::global_vertex_index v) const
    {
      AssertIndexRange(v, n_vertices);
      return owned_vertices[v];
    }

    template <int dim>
    inline const DoFHandler<dim> &Base<dim>::get_dof_handler() const
    {
      return dh;
    }

    template <int dim>
    inline const FiniteElement<dim> &Base<dim>::get_fe() const
    {
      return *fe;
    }

    template <int dim>
    inline const LA::ParVectorType &Base<dim>::get_reconstructions() const
    {
      return recovery_solution;
    }

    template <int dim>
    inline FEValues<dim>
    Base<dim>::get_fe_values(const Mapping<dim>    &mapping,
                             const Quadrature<dim> &quadrature) const
    {
      return FEValues<dim>(mapping,
                           *this->fe,
                           quadrature,
                           update_values | update_JxW_values);
    }

    template <int dim>
    inline ComponentMask Base<dim>::get_solution_mask() const
    {
      return solution_mask;
    }

    template <int dim>
    inline ComponentMask Base<dim>::get_gradient_mask() const
    {
      return gradient_mask;
    }

    template <int dim>
    inline ComponentMask Base<dim>::get_hessian_mask() const
    {
      return hessian_mask;
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
          if (locally_owned_recovery_dofs.is_element(dof))
            local_dof_data[dof] = val;
        }
      }
      local_dof_data.compress(VectorOperation::insert);
      dof_data = local_dof_data;
    }

    template <int dim>
    template <int rank, int n_tensor_components>
    void Base<dim>::vertex_to_isoparametric(
      const std::vector<Tensor<rank, dim>>               &vertex_data,
      LA::ParVectorType                                  &local_dof_data,
      LA::ParVectorType                                  &dof_data,
      const std::vector<std::array<types::global_dof_index,
                                   n_tensor_components>> &vertex_to_dofs)
    {
      static_assert(
        std::vector<Tensor<rank, dim>>::value_type::n_independent_components ==
          n_tensor_components,
        "n_tensor_components must match the number of independent components "
        "of a "
        "Tensor<rank, dim>");
      AssertDimension(vertex_data.size(), vertex_to_dofs.size());
      AssertDimension(vertex_to_dofs.size(), n_vertices);

      for (types::global_vertex_index v = 0; v < n_vertices; ++v)
      {
        if (owned_vertices[v])
        {
          const auto &t    = vertex_data[v];
          const auto &dofs = vertex_to_dofs[v];
          for (unsigned int c = 0; c < n_tensor_components; ++c)
            if (locally_owned_recovery_dofs.is_element(dofs[c]))
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

    template <int dim>
    const std::vector<Tensor<3, dim>> &
    Scalar<dim>::get_reconstructed_third_derivatives(const unsigned int) const
    {
      return recovered_third_derivatives_at_vertices;
    }

    template <int dim>
    const std::map<types::global_dof_index,
                   std::array<types::global_dof_index, 1>> &
    Scalar<dim>::get_solution_to_recovery_dof_map() const
    {
      return solution_dofs_to_recovery_dofs;
    }

    template <int dim>
    const std::map<
      types::global_dof_index,
      std::array<types::global_dof_index,
                 Scalar<dim>::gradient_type::n_independent_components>> &
    Scalar<dim>::get_solution_to_gradient_dof_map() const
    {
      return solution_dofs_to_gradient_dofs;
    }

    template <int dim>
    const std::map<
      types::global_dof_index,
      std::array<types::global_dof_index,
                 Scalar<dim>::hessian_type::n_independent_components>> &
    Scalar<dim>::get_solution_to_hessian_dof_map() const
    {
      return solution_dofs_to_hessian_dofs;
    }

  } // namespace SolutionRecovery
} // namespace ErrorEstimation

#endif
