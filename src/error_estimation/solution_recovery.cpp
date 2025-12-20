
#include <error_estimation/solution_recovery.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>

#include <Eigen/Dense>

namespace ErrorEstimation
{
  template <int dim>
  std::vector<unsigned int>
  build_commuting_derivative_to_symmetric_index_map(const unsigned int degree,
    const std::vector<std::array<unsigned int, dim>> &monomials)
  {
    const unsigned int n_entries = std::pow(dim, degree);
    std::vector<unsigned int> result(n_entries, numbers::invalid_unsigned_int);

    // Subset of highest degree monomials
    std::vector<std::array<unsigned int, dim>> highest_monomials;
    for(const auto &m : monomials)
    {
      unsigned int sum;
      if constexpr (dim == 2)
        sum = m[0] + m[1];
      else
        sum = m[0] + m[1] + m[2];
      if(sum == degree)
        // TODO: Check ordering compatibility with metric computations
        highest_monomials.push_back(m);
    }

    for(const auto &m : highest_monomials)
    {
      std::cout << "Highest monomial " << m[0] << " " << m[1] << std::endl;
    }


    // Build mapping from symmetric multi-index (alpha) to its position in monomials
    std::map<std::array<unsigned int, dim>, unsigned int> alpha_to_index;
    for (unsigned int i = 0; i < highest_monomials.size(); ++i)
      alpha_to_index[highest_monomials[i]] = i;

    for (unsigned int i = 0; i < n_entries; ++i)
    {
      // Decode i to its tensor multi-index (like [0,1] for d²u/dxdy)
      unsigned int temp = i;
      std::array<unsigned int, dim> alpha = {}; // zero init
      for (unsigned int j = 0; j < degree; ++j)
      {
        const unsigned int d = temp % dim;
        ++alpha[d];
        temp /= dim;
      }

      // Now alpha is the symmetric version (e.g., dxdy and dydx → (1,1))
      auto it = alpha_to_index.find(alpha);
      Assert(it != alpha_to_index.end(), ExcMessage("Symmetric multi-index not found in monomials"));
      result[i] = it->second;
    }

    return result;
  }

  template <int dim>
  SolutionRecovery<dim>::SolutionRecovery(Patches<dim>             &patches,
                                          const Vector<double>     &solution,
                                          const FiniteElement<dim> &fe,
                                          const Mapping<dim>       &mapping)
    : patches(patches)
    , triangulation(patches.triangulation)
    , fe(fe)
    , mapping(mapping)
    , dof_handler(patches.dof_handler)
    , solution(solution)
    , n_vertices(triangulation.n_vertices())
    , n_components(fe.n_components())
    , degree(fe.degree)
    , dim_recovery_basis(PolynomialSpace<dim>::dim_polynomial_basis(degree + 1))
    , dim_derivative_basis(PolynomialSpace<dim>::dim_polynomial_basis(degree))
  {
    monomials = PolynomialSpace<dim>::generate_monomials(degree + 1);
    monomials_derivatives = PolynomialSpace<dim>::generate_monomials(degree);

    // Map exponent -> index for reduced basis
    for (unsigned int i = 0; i < dim_derivative_basis; ++i)
      monomials_derivatives_index[monomials_derivatives[i]] = i;

    // For each vector component, the number of fields to reconstruct
    // (the field + all its derivatives up to order "degree") and the number of
    // derivatives to store (sum of dim^i, i > 0, until degree + 1).
    n_fields_to_recover = 1;
    for (unsigned int i = 1; i <= degree; ++i)
    {
      n_fields_to_recover += std::pow(dim, i);
    }
    n_derivatives_to_store =
      n_fields_to_recover - 1 + std::pow(dim, degree + 1);

    recovered_polynomials.resize(
      n_vertices, std::vector<std::vector<Vector<double>>>(n_components));
    recovered_derivatives.resize(
      n_vertices, std::vector<std::vector<Vector<double>>>(n_components));
    for (unsigned int v = 0; v < n_vertices; ++v)
    {
      for (unsigned int c = 0; c < n_components; ++c)
      {
        recovered_polynomials[v][c].resize(n_fields_to_recover,
                                           Vector<double>());
        recovered_derivatives[v][c].resize(n_derivatives_to_store,
                                           Vector<double>());
      }
    }

    std::cout << "The monomials are:" << std::endl;
    for (const auto &m : monomials)
    {
      std::cout << m[0] << " - " << m[1] << std::endl;
    }

    computeLeastSquaresMatrices();
    std::cout << "Computed least squares matrices" << std::endl;

    build_vertex_to_dof_map();
    std::cout << "Built vertex to dof map" << std::endl;

    n_recovered_fields     = 0;
    n_derivatives_computed = 0;
    for (unsigned int i = 0; i < n_fields_to_recover; ++i)
    {
      // If i = 0, then a more accurate solution is fitted.
      // If i > 0, then a more accurate derivative is fitted.
      recover_from_solution(i);
    }

    std::vector<unsigned int> index_map = build_commuting_derivative_to_symmetric_index_map<dim>(3, monomials);
    for (auto &val : index_map)
    {
      std::cout << "index_map " << val << std::endl;
    }
  }

  template <int dim>
  unsigned int getRank(const FullMatrix<double> &mat, Eigen::MatrixXd &eigenMat)
  {
    const unsigned int m = mat.m();
    const unsigned int n = mat.n();
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
        eigenMat(i, j) = mat(i, j);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(eigenMat);
    return lu_decomp.rank();
  }

  template <int dim>
  void SolutionRecovery<dim>::computeLeastSquaresMatrices()
  {
    // The matrices are of size dim_recovery_basis x n_adjacent,
    // but n_adjacent varies and can change if the patch is increased...
    least_squares_matrices.resize(n_vertices);

    FullMatrix<double> AtA(dim_recovery_basis, dim_recovery_basis);
    Eigen::MatrixXd    eigenAtA =
      Eigen::MatrixXd::Zero(dim_recovery_basis, dim_recovery_basis);

    // Construct (A^T*A)^-1 * A^T
    for (unsigned int i = 0; i < n_vertices; ++i)
    {
      bool         isFullRank = false;
      unsigned int rank, num_patch_increases = 0, max_patch_increases = 2;

      do
      {
        const auto        &patch_of_vertices = patches.patches_of_vertices[i];
        const unsigned int n_adjacent        = patch_of_vertices.size();

        AssertThrow(
          n_adjacent >= dim_recovery_basis,
          ExcMessage(
            "Cannot create least-squares matrix because a patch of adjacent "
            "vertices has less vertices than the dimension of the polynomial "
            "basis for the fitting. This should not have happened, as the "
            "patches are created with at least that many vertices..."));

        FullMatrix<double> A(n_adjacent, dim_recovery_basis);
        this->fill_vandermonde_matrix(i, A);
        A.Tmmult(AtA, A);
        rank = getRank<dim>(AtA, eigenAtA);

        if (rank >= dim_recovery_basis)
        {
          isFullRank = true;
          FullMatrix<double> least_squares_mat(dim_recovery_basis, n_adjacent);
          least_squares_mat.left_invert(A);
          least_squares_matrices[i] = least_squares_mat;
        }
        else
        {
          if (num_patch_increases++ > max_patch_increases)
          {
            throw std::runtime_error(
              "Could not create least-squares matrix of full rank even "
              "after increasing the patch size several times.");
          }

          // Increase patch size
          patches.increase_patch_size(i);
        }
      }
      while (!isFullRank);
    }
  }

  template <int dim>
  void
  SolutionRecovery<dim>::fill_vandermonde_matrix(const unsigned int  vertex,
                                                 FullMatrix<double> &mat) const
  {
    const auto        &vertices          = triangulation.get_vertices();
    const auto        &patch_of_vertices = patches.patches_of_vertices[vertex];
    const Point<dim>  &center            = vertices[vertex];
    const Point<dim>  &scaling           = patches.scalings[vertex];
    const unsigned int dim_basis         = monomials.size();
    double             xLoc[dim];

    unsigned int i = 0;
    for (const types::global_vertex_index &ind_i : patch_of_vertices)
    {
      const Point<dim> &vi = vertices[ind_i];

      // Compute local scaled coordinates
      for (unsigned int d = 0; d < dim; ++d)
        xLoc[d] = (vi[d] - center[d]) / scaling[d];

      // Evaluate all monomials at xLoc
      for (unsigned int j = 0; j < dim_basis; ++j)
      {
        double val = 1.0;
        for (unsigned int d = 0; d < dim; ++d)
          val *= std::pow(xLoc[d], monomials[j][d]);

        mat(i, j) = val;
      }
      ++i;
    }
  }

  template <int dim>
  void SolutionRecovery<dim>::build_vertex_to_dof_map()
  {
    AssertThrow(n_components == 1,
                ExcMessage(
                  "Complete build_vertex_to_dof_map for vector-valued FE"));

    vertex_to_dof.resize(n_vertices,
                         std::vector<types::global_dof_index>(
                           n_components, numbers::invalid_dof_index));

    // Get support points
    // std::map<types::global_dof_index, Point<dim>> dof_to_point;
    const unsigned int      n_dofs = dof_handler.n_dofs();
    std::vector<Point<dim>> dof_to_points(n_dofs);
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, dof_to_points);

    // for (const auto &[dof_idx, pt] : dof_to_points)
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const Point<dim> &pt = dof_to_points[i];

      // Use GridTools::find_closest_vertex if needed
      types::global_vertex_index vertex_idx =
        GridTools::find_closest_vertex(dof_handler.get_triangulation(), pt);

      // const unsigned int comp = fe.system_to_component_index(i).first;
      const unsigned int comp = 0;

      vertex_to_dof[vertex_idx][comp] = i;
    }
  }

  // Get the solution field u_h at the vertices of a patch
  void get_component_values_on_patch(
    const Vector<double>                       &solution,
    const std::set<types::global_vertex_index> &patch_vertex_indices,
    const std::vector<std::vector<types::global_dof_index>> &vertex_to_dof,
    const unsigned int                                       component,
    Vector<double>                                          &values_out)
  {
    Assert(values_out.size() == patch_vertex_indices.size(),
           ExcMessage("Output vector size mismatch"));

    unsigned int i = 0;
    for (const types::global_vertex_index &v : patch_vertex_indices)
    {
      types::global_dof_index dof = vertex_to_dof[v][component];

      AssertThrow(dof != numbers::invalid_dof_index,
                  ExcMessage("No DoF associated to vertex/component"));

      values_out[i] = solution[dof];
      ++i;
    }
  }

  // Get the previously computed derivative of u_h at the vertices of a patch
  void get_component_derivative_on_patch(
    const unsigned int i_derivative,
    const std::vector<std::vector<std::vector<Vector<double>>>>
                                               &recovered_derivatives,
    const std::set<types::global_vertex_index> &patch_vertex_indices,
    const unsigned int                          component,
    Vector<double>                             &values_out)
  {
    Assert(values_out.size() == patch_vertex_indices.size(),
           ExcMessage("Output vector size mismatch"));

    // For each vertex of the patch, evaluate its derivative polynomial
    // at (0,0) (= the independent term).
    unsigned int i = 0;
    for (const types::global_vertex_index &v : patch_vertex_indices)
    {
      values_out[i] = recovered_derivatives[v][component][i_derivative][0];
      ++i;
    }
  }

  template <int dim>
  void SolutionRecovery<dim>::recover_from_solution(
    const unsigned int i_recovered_derivative)
  {
    Vector<double> coeffs(dim_recovery_basis);

    for (unsigned int v = 0; v < n_vertices; ++v)
    {
      const Point<dim> &scaling = patches.scalings[v];
      const auto       &ls_mat  = least_squares_matrices[v];
      Vector<double>    patch_values(ls_mat.n());

      const auto &patch_vertices = patches.patches_of_vertices[v];

      // Extract local solution values for each component
      for (unsigned int c = 0; c < n_components; ++c)
      {
        if (i_recovered_derivative == 0)
          get_component_values_on_patch(
            solution, patch_vertices, vertex_to_dof, c, patch_values);
        else
          get_component_derivative_on_patch(i_recovered_derivative - 1,
                                            recovered_derivatives,
                                            patch_vertices,
                                            c,
                                            patch_values);

        ls_mat.vmult(coeffs, patch_values);

        for (unsigned int i = 0; i < dim_recovery_basis; ++i)
        {
          // Scale back
          for (unsigned int d = 0; d < dim; ++d)
            coeffs[i] /= std::pow(scaling[d], monomials[i][d]);
        }

        // Store coefficients
        recovered_polynomials[v][c][i_recovered_derivative] = coeffs;

        // Compute derivatives
        compute_derivatives(coeffs, recovered_derivatives[v][c]);
      }
    }

    n_recovered_fields++;
    n_derivatives_computed += dim;
  }

  template <int dim>
  void SolutionRecovery<dim>::compute_derivatives(
    const Vector<double>        &coeffs,
    std::vector<Vector<double>> &derivatives)
  {
    AssertDimension(coeffs.size(), dim_recovery_basis);

    Vector<double> deriv;

    for (unsigned int d = 0; d < dim; ++d)
    {
      deriv.reinit(dim_derivative_basis);

      // Compute derivative of coeffs with respect to d-th coordinate
      for (unsigned int i = 0; i < dim_recovery_basis; ++i)
      {
        const auto &alpha = monomials[i];

        if (alpha[d] == 0)
          continue;

        // TODO: Check if the small terms can be capped,
        // or if it becomes unstable
        // if(std::abs(coeffs[i]) < 1e-10)
        // continue;

        std::array<unsigned int, dim> beta = alpha;
        beta[d] -= 1;

        auto it = monomials_derivatives_index.find(beta);
        if (it != monomials_derivatives_index.end())
          deriv[it->second] += coeffs[i] * alpha[d];
      }

      derivatives[n_derivatives_computed + d] = deriv;
    }
  }

  // template <int dim>
  // void SolutionRecovery<dim>::compute_homogeneous_error_polynomials()
  // {
  //   unsigned int       derivative_offset = 0;
  //   for (unsigned int i = 1; i < degree + 1; ++i)
  //     derivative_offset += std::pow(dim, i);

  //   // Number of monomials of degree "degree + 1":
  //   constexpr unsigned int n_coeffs = (dim == 2) ? degree + 1 :
  //     (degree + 2) * (degree + 3) / 2;

  //   // Get the monomials of degree "degree + 1"
  //   std::vector<std::array<unsigned int, dim>> highest_monomials;
  //   for(const auto &m : monomials)
  //   {
  //     unsigned int sum;
  //     if constexpr (dim == 2)
  //       sum = m[0] + m[1];
  //     else
  //       sum = m[0] + m[1] + m[2];
  //     if(sum == degree + 1)
  //       // TODO: Check ordering compatibility with metric computations
  //       highest_monomials.push_back(m);
  //   }

  //   AssertDimension(highest_monomials.size(), n_coeffs);

  //   homogeneous_error_polynomials.resize(n_vertices, std::vector<Vector<double>>(n_components));

  //   Vector<double> coeffs(n_coeffs);

  //   for (types::global_vertex_index v = 0; v < n_vertices; ++v)
  //   {
  //     for (unsigned int c = 0; c < n_components; ++c)
  //     {
  //       for (unsigned int i = 0; i < n_coeffs; ++i)
  //       {
  //         const auto &alpha = highest_monomials[i];

  //         // Compute multinomial denominator a!
  //         double alpha_fact = 1.0;
  //         for (const auto a : alpha)
  //           alpha_fact *= dealii::Utilities::factorial<double>(a);

  //         AssertIndexRange(derivative_offset + i, recovered_derivatives[v][c].size());

  //         coeffs[i] = recovered_derivatives[v][c][derivative_offset + i][0] / alpha_fact;
  //       }

  //       homogeneous_error_polynomials[v][c] = std::move(coeffs);
  //     }
  //   }
  // }

  // #include <algorithm> // std::next_permutation, std::sort
  // #include <numeric>   // std::accumulate
  // #include <map>
  // #include <vector>
  // #include <array>

  // template <int dim>
  // void SolutionRecovery<dim>::compute_homogeneous_error_polynomials()
  // {
  //   const unsigned int n_terms = monomials.size();

  //   // Helper: factorial for unsigned int
  //   auto factorial = [](unsigned int n) -> unsigned int {
  //     unsigned int f = 1;
  //     for (unsigned int i = 2; i <= n; ++i)
  //       f *= i;
  //     return f;
  //   };

  //   // Helper: multinomial coefficient = (sum(alpha)!) / (alpha_1! * ... * alpha_dim!)
  //   auto multinomial_coeff = [&](const std::array<unsigned int, dim> &alpha) -> unsigned int {
  //     unsigned int numerator = factorial(std::accumulate(alpha.begin(), alpha.end(), 0u));
  //     unsigned int denominator = 1;
  //     for (auto a : alpha)
  //       denominator *= factorial(a);
  //     return numerator / denominator;
  //   };

  //   // Helper: generate all distinct permutations of a multiset (alpha)
  //   auto generate_permutations = [&](const std::array<unsigned int, dim> &alpha)
  //     -> std::vector<std::array<unsigned int, dim>> {
  //     // Convert array to vector for easier permutation
  //     std::vector<unsigned int> v(alpha.begin(), alpha.end());
  //     // Sort to prepare for std::next_permutation
  //     std::sort(v.begin(), v.end());

  //     std::vector<std::array<unsigned int, dim>> permutations;
  //     do {
  //       std::array<unsigned int, dim> perm{};
  //       std::copy(v.begin(), v.end(), perm.begin());
  //       permutations.push_back(perm);
  //     } while (std::next_permutation(v.begin(), v.end()));

  //     return permutations;
  //   };

  //   // Build a map from multi-index to position in monomials vector for quick lookup
  //   // Using std::map with vector key since std::array supports operator< by default
  //   std::map<std::array<unsigned int, dim>, unsigned int> multiindex_to_pos;
  //   for (unsigned int i = 0; i < n_terms; ++i)
  //     multiindex_to_pos[monomials[i]] = i;

  //   // Resize output: [vertex][component] with length = n_terms (highest degree only)
  //   homogeneous_error_polynomials.resize(recovered_derivatives.size(),
  //                                        std::vector<Vector<double>>(n_components));

  //   for (types::global_vertex_index v = 0; v < recovered_derivatives.size(); ++v)
  //   {
  //     for (unsigned int c = 0; c < n_components; ++c)
  //     {
  //       Vector<double> coeffs(n_terms);
  //       coeffs = 0.0;

  //       // Offset for highest order derivatives in recovered_derivatives[v][c]
  //       const unsigned int derivative_offset =
  //         (p == 0 ? 0 :
  //          p == 1 ? dim :
  //          p == 2 ? dim + dim * dim :
  //          throw dealii::ExcNotImplemented());

  //       for (unsigned int i = 0; i < n_terms; ++i)
  //       {
  //         const auto &alpha = monomials[i];

  //         // Generate all distinct permutations of alpha
  //         std::vector<std::array<unsigned int, dim>> permutations = generate_permutations(alpha);

  //         // Sum recovered derivatives over all permutations
  //         double derivative_sum = 0.0;
  //         for (const auto &perm : permutations)
  //         {
  //           // Find index of perm in monomials vector
  //           auto it = multiindex_to_pos.find(perm);
  //           if (it == multiindex_to_pos.end())
  //             AssertThrow(false, dealii::ExcMessage("Permutation multi-index not found in monomials vector"));

  //           const unsigned int perm_index = it->second;
  //           const unsigned int deriv_index = derivative_offset + perm_index;
  //           AssertIndexRange(deriv_index, recovered_derivatives[v][c].size());

  //           // Add constant term of recovered derivative polynomial at vertex
  //           derivative_sum += recovered_derivatives[v][c][deriv_index][0];
  //         }

  //         // Divide by multinomial coefficient (number of permutations)
  //         const unsigned int alpha_factorial = multinomial_coeff(alpha);

  //         coeffs[i] = derivative_sum / alpha_factorial;
  //       }

  //       homogeneous_error_polynomials[v][c] = std::move(coeffs);
  //     }
  //   }
  // }

  template <int dim>
  void SolutionRecovery<dim>::write_derivatives_to_vtu(
    const unsigned int order) const
  {
    std::string msg = "Could not write derivatives of order " +
                      std::to_string(order) +
                      " because only derivatives of order up to " +
                      std::to_string(degree + 1) + " were computed.";
    AssertThrow(order <= degree + 1, ExcMessage(msg));

    const unsigned int n_derivatives     = std::pow(dim, order);
    unsigned int       derivative_offset = 0;
    for (unsigned int i = 1; i < order; ++i)
      derivative_offset += std::pow(dim, i);

    const unsigned int total_components = n_components * n_derivatives;

    // One linear DoF per component at each vertex
    FESystem<dim> fe_output(FE_SimplexP<dim>(1), total_components);

    DoFHandler<dim> dof_handler_out(triangulation);
    dof_handler_out.distribute_dofs(fe_output);

    Vector<double> output_vector(dof_handler_out.n_dofs());

    // Fill the vector: loop over cells and vertex DoFs
    for (const auto &cell : dof_handler_out.active_cell_iterators())
    {
      for (unsigned int v = 0; v < cell->n_vertices(); ++v)
      {
        const types::global_vertex_index vertex_idx = cell->vertex_index(v);

        AssertIndexRange(vertex_idx, recovered_derivatives.size());

        for (unsigned int c = 0; c < n_components; ++c)
        {
          AssertIndexRange(c, recovered_derivatives[vertex_idx].size());

          for (unsigned int d = 0; d < n_derivatives; ++d)
          {
            const unsigned int global_component = c * n_derivatives + d;
            const unsigned int dof_index =
              cell->vertex_dof_index(v, global_component);

            AssertIndexRange(derivative_offset + d,
                             recovered_derivatives[vertex_idx][c].size());

            // Evaluate at vertex center = constant term
            output_vector[dof_index] =
              recovered_derivatives[vertex_idx][c][derivative_offset + d][0];
          }
        }
      }
    }

    // Build output
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler_out);

    std::vector<std::string> names;
    for (unsigned int c = 0; c < n_components; ++c)
      for (unsigned int d = 0; d < n_derivatives; ++d)
        names.emplace_back("u" + std::to_string(c) + "_d" +
                           std::to_string(order) + "_" + std::to_string(d));

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(total_components,
                     DataComponentInterpretation::component_is_scalar);

    data_out.add_data_vector(output_vector,
                             names,
                             DataOut<dim>::type_dof_data,
                             interpretation);
    data_out.build_patches();

    std::ofstream out("recovered_derivatives_order_" + std::to_string(order) +
                      ".vtu");
    data_out.write_vtu(out);
  }

  template class SolutionRecovery<2>;
  template class SolutionRecovery<3>;

} // namespace SolutionRecoveryNamespace