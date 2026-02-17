#ifndef SOLUTION_RECOVERY_H
#define SOLUTION_RECOVERY_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/polynomial_space.h>

#include "error_estimation/patches.h"
#include "types.h"

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * TODO: Add documentation
   */
  template <int dim>
  class SolutionRecovery
  {
  public:
    /**
     * Constructor
     */
    SolutionRecovery(PatchHandler<dim>        &patch_handler,
                     const LA::ParVectorType  &solution,
                     const FiniteElement<dim> &fe,
                     const Mapping<dim>       &mapping);

    // void write_derivatives_to_vtu(const unsigned int order) const;

    /**
     * Write the least-squares matrices for debug and testing
     */
    void write_least_squares_systems(std::ostream &out = std::cout) const;

  private:
    /**
     * For each owned mesh vertex (each patch), compute the least squares
     * matrix.
     */
    void compute_least_squares_matrices();

    /**
     * Fill the Vandermonde matrix at mesh vertex v, according to the scaling
     * vector stored in the patch at v.
     */
    void fill_vandermonde_matrix(const Patch<dim>   &patch,
                                 FullMatrix<double> &mat) const;

  private:
    // Patches are not const as they may be increased when computing the
    // least-squares matrices
    PatchHandler<dim>             &patch_handler;
    const std::vector<Patch<dim>> &patches;
    const LA::ParVectorType       &solution;
    const FiniteElement<dim>      &fe;
    const Mapping<dim>            &mapping;

    MPI_Comm           mpi_communicator;
    ConditionalOStream pcout;

    /**
     * Relevant dofs and local solution vector.
     *
     * This index set is the usual set of relevant dofs, augmented with
     * the non-local patch dofs that were gathered from other ranks.
     */
    IndexSet          relevant_dofs;
    LA::ParVectorType local_solution;

    // Number of mesh vertices on this partition and mask
    const unsigned int n_vertices;
    std::vector<bool>  owned_vertices;

    // const unsigned int n_components;
    // const unsigned int degree;
    unsigned int dim_recovery_basis;
    unsigned int dim_gradient_basis;
    unsigned int n_fields_to_recover;
    unsigned int n_recovered_fields;
    unsigned int n_derivatives_to_store;
    unsigned int n_derivatives_computed;

    // // The polynomial basis for the fitting of degree p+1 and p respectively
    // std::vector<std::array<unsigned int, dim>> monomials,
    // monomials_derivatives;

    // Polynomial bases for fitting of degree p+1 and its derivatives of degree
    // p
    std::shared_ptr<PolynomialSpace<dim>> monomials_recovery;
    std::shared_ptr<PolynomialSpace<dim>> monomials_gradient;

    // // A map from monomials of degree p to their position in
    // // monomials_derivatives.
    // std::map<std::array<unsigned int, dim>, unsigned int>
    //   monomials_derivatives_index;

    // The least squares matrix for each mesh vertex
    std::vector<FullMatrix<double>> least_squares_matrices;

    // owned vertex : # recovery : multivariate polynomial centered at vertex
    std::vector<std::vector<PolynomialSpace<dim>>> recoveries;
    std::vector<std::vector<Vector<double>>>       recoveries_coefficients;

    // The recovered solution and derivatives evaluated at the vertex (= local
    // origin)
    std::vector<double>         recovered_solution;
    std::vector<Tensor<1, dim>> recovered_gradient;
    std::vector<Tensor<2, dim>> recovered_hessian;
    std::vector<Tensor<3, dim>> recovered_third_derivatives;
    std::vector<Tensor<4, dim>> recovered_fourth_derivatives;

    // std::vector<std::vector<PolynomialSpace<dim>>> gradients;


    // // For each vertex and each component : the recovered polynomials,
    // // that is, solution and derivatives up to order "degree",
    // // where "degree" is the polynomial degree of the given solution.
    // //
    // // Ordered as [vertex][component][indexRecovery][coefficients].
    // //
    // // with indexRecovery as follows:
    // //
    // // 0 : Recovered solution u
    // // 1 -> dim : Recovered gradient ux, uy, uz
    // // dim + 1 -> dim + 1 + dim^2 : Recovered hessian uxx, uxy, uxz, ...
    // std::vector<std::vector<std::vector<Vector<double>>>>
    // recovered_polynomials;
    // std::vector<std::vector<std::vector<Vector<double>>>>
    // recovered_derivatives;

    // // These coefficients already include the binomial coefficients.
    // // [vertex][component][error_coefficients]
    // std::vector<std::vector<Vector<double>>> homogeneous_errors;

    void recover_from_solution(const unsigned i_recovered_derivative);
    // void compute_derivatives(const Vector<double>        &coeffs,
    //                          std::vector<Vector<double>> &derivatives);
  };
} // namespace ErrorEstimation

#endif
