#ifndef SOLUTION_RECOVERY_H
#define SOLUTION_RECOVERY_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include <array>

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   *
   */
  template <int dim>
  class ErrorPatches
  {
  public:
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    /**
     * Constructor.
     *
     * @param dof_handler The DoFHandler associated with the mesh.
     * @param degree The degree of the polynomial solution. A polynomial
     *        of order degree + 1 will be fitted on the patches.
     */
    ErrorPatches(const Triangulation<dim> &triangulation,
            const DoFHandler<dim>    &dof_handler,
            unsigned int              degree);

    void increase_patch_size(unsigned int vertex_index);

    void write_patch_to_pos(const unsigned int i,
                            const unsigned int posTag) const;

  public:
    const Triangulation<dim> &triangulation;
    const DoFHandler<dim>    &dof_handler;

    // Patches of elements and vertices
    std::vector<std::set<CellIterator>>               patches;
    std::vector<std::set<types::global_vertex_index>> patches_of_vertices;
    std::vector<unsigned int>                         num_layers;

    std::vector<Point<dim>> scalings;

  private:
    void addLayer(unsigned int vertex_index);
    // Compute the scaling of each patch : max_i |x_i - x|,
    // with x the vertex (center of the patch) and x_i a vertex of the patch.
    void compute_scalings();
  };

  /**
   * A small convenience class to define monomials.
   */
  template <int dim>
  class PolynomialSpace_not_deal_ii
  {
  public:
    using MultiIndex = std::array<unsigned int, dim>;

    // Return all monomial multi-indices up to total degree `degree`
    static std::vector<MultiIndex> generate_monomials(unsigned int degree)
    {
      std::vector<MultiIndex> monomials;

      // Recursive function to enumerate multi-indices
      std::function<void(MultiIndex &, unsigned int, unsigned int)> enumerate;
      enumerate =
        [&](MultiIndex &current, unsigned int pos, unsigned int remaining) {
          if (pos == dim - 1)
          {
            current[pos] = remaining;
            monomials.push_back(current);
            return;
          }

          for (unsigned int i = 0; i <= remaining; ++i)
          {
            current[pos] = i;
            enumerate(current, pos + 1, remaining - i);
          }
        };

      MultiIndex current{};
      for (unsigned int total_deg = 0; total_deg <= degree; ++total_deg)
        enumerate(current, 0, total_deg);

      return monomials;
    }

    // Return the total number of monomials of total degree <= degree
    static unsigned int dim_polynomial_basis(unsigned int degree)
    {
      if constexpr (dim == 2)
        return (degree + 1) * (degree + 2) / 2;
      else
        return (degree + 1) * (degree + 2) * (degree + 3) / 6;
    }
  };

  /**
   *
   */
  template <int dim>
  class SolutionRecovery
  {
  public:
    /**
     * Constructor.
     *
     * @param triangulation The triangulation on which the finite element space is defined.
     * @param fe The finite element describing the finite element space.
     * @param dof_handler The DoFHandler associated with the finite element space.
     * @param solution The finite element solution vector.
     */
    SolutionRecovery(
      ErrorPatches<dim>             &patches,
      const Vector<double>     &solution,
      const FiniteElement<dim> &fe,
      const Mapping<dim>       &mapping = MappingFE<dim>(FE_SimplexP<dim>(1)));

    void write_derivatives_to_vtu(const unsigned int order) const;

  private:
    // Patches may be increased when computing the least-squares matrices
    ErrorPatches<dim>             &patches;
    const Triangulation<dim> &triangulation;
    const FiniteElement<dim> &fe;
    const Mapping<dim>       &mapping;
    const DoFHandler<dim>    &dof_handler;
    const Vector<double>     &solution;

    const unsigned int n_vertices;
    const unsigned int n_components;
    const unsigned int degree;
    const unsigned int dim_recovery_basis;
    const unsigned int dim_derivative_basis;
    unsigned int       n_fields_to_recover;
    unsigned int       n_recovered_fields;
    unsigned int       n_derivatives_to_store;
    unsigned int       n_derivatives_computed;

    // The polynomial basis for the fitting of degree p+1 and p respectively
    std::vector<std::array<unsigned int, dim>> monomials, monomials_derivatives;
    // A map from monomials of degree p to their position in
    // monomials_derivatives.
    std::map<std::array<unsigned int, dim>, unsigned int>
      monomials_derivatives_index;

    // The least squares matrix for each mesh vertex
    std::vector<FullMatrix<double>> least_squares_matrices;

    // For each vertex and each component : the recovered polynomials,
    // that is, solution and derivatives up to order "degree",
    // where "degree" is the polynomial degree of the given solution.
    //
    // Ordered as [vertex][component][indexRecovery][coefficients].
    //
    // with indexRecovery as follows:
    //
    // 0 : Recovered solution u
    // 1 -> dim : Recovered gradient ux, uy, uz
    // dim + 1 -> dim + 1 + dim^2 : Recovered hessian uxx, uxy, uxz, ...
    std::vector<std::vector<std::vector<Vector<double>>>> recovered_polynomials;
    std::vector<std::vector<std::vector<Vector<double>>>> recovered_derivatives;

    // These coefficients already include the binomial coefficients.
    // [vertex][component][error_coefficients]
    std::vector<std::vector<Vector<double>>> homogeneous_errors;

    // [vertex][component] -> global DoF index
    // Probably does not work for parallel mesh
    std::vector<std::vector<types::global_dof_index>> vertex_to_dof;

    void computeLeastSquaresMatrices();
    void fill_vandermonde_matrix(const unsigned int  vertex,
                                 FullMatrix<double> &mat) const;
    void build_vertex_to_dof_map();
    void recover_from_solution(const unsigned i_recovered_derivative);
    void compute_derivatives(const Vector<double>        &coeffs,
                             std::vector<Vector<double>> &derivatives);
  };
} // namespace ErrorEstimation

#endif