
#include <deal.II/base/polynomial_space.h>
#include <deal.II/dofs/dof_tools.h>
#include <error_estimation/patches.h>
#include <error_estimation/solution_recovery.h>
#include <utilities.h>
// #include <deal.II/fe/fe_system.h>
// #include <deal.II/grid/grid_tools.h>
// #include <deal.II/numerics/data_out.h>

#include <Eigen/Dense>

namespace ErrorEstimation
{
  //   template <int dim>
  //   std::vector<unsigned int>
  //   build_commuting_derivative_to_symmetric_index_map(const unsigned int
  //   degree,
  //     const std::vector<std::array<unsigned int, dim>> &monomials)
  //   {
  //     const unsigned int n_entries = std::pow(dim, degree);
  //     std::vector<unsigned int> result(n_entries,
  //     numbers::invalid_unsigned_int);

  //     // Subset of highest degree monomials
  //     std::vector<std::array<unsigned int, dim>> highest_monomials;
  //     for(const auto &m : monomials)
  //     {
  //       unsigned int sum;
  //       if constexpr (dim == 2)
  //         sum = m[0] + m[1];
  //       else
  //         sum = m[0] + m[1] + m[2];
  //       if(sum == degree)
  //         // TODO: Check ordering compatibility with metric computations
  //         highest_monomials.push_back(m);
  //     }

  //     for(const auto &m : highest_monomials)
  //     {
  //       std::cout << "Highest monomial " << m[0] << " " << m[1] << std::endl;
  //     }


  //     // Build mapping from symmetric multi-index (alpha) to its position in
  //     monomials std::map<std::array<unsigned int, dim>, unsigned int>
  //     alpha_to_index; for (unsigned int i = 0; i < highest_monomials.size();
  //     ++i)
  //       alpha_to_index[highest_monomials[i]] = i;

  //     for (unsigned int i = 0; i < n_entries; ++i)
  //     {
  //       // Decode i to its tensor multi-index (like [0,1] for d²u/dxdy)
  //       unsigned int temp = i;
  //       std::array<unsigned int, dim> alpha = {}; // zero init
  //       for (unsigned int j = 0; j < degree; ++j)
  //       {
  //         const unsigned int d = temp % dim;
  //         ++alpha[d];
  //         temp /= dim;
  //       }

  //       // Now alpha is the symmetric version (e.g., dxdy and dydx → (1,1))
  //       auto it = alpha_to_index.find(alpha);
  //       Assert(it != alpha_to_index.end(), ExcMessage("Symmetric multi-index
  //       not found in monomials")); result[i] = it->second;
  //     }

  //     return result;
  //   }

  // /**
  //  * Generates a flat vector of exponents for a P_p space.
  //  * Layout: [exp0_dim0, exp0_dim1, exp1_dim0, exp1_dim1, ...]
  //  */
  // template <int dim>
  // std::vector<unsigned int> create_numbering(const unsigned int degree)
  // {
  //     std::vector<unsigned int> exponents;
  //     std::array<unsigned int, dim> current_idx;

  //     for (unsigned int total_deg = 0; total_deg <= degree; ++total_deg) {
  //         current_idx.fill(0);
  //         current_idx[0] = total_deg;

  //         while (true) {
  //             // Push all dimensions for the current polynomial term
  //             for (unsigned int d = 0; d < dim; ++d) {
  //                 exponents.push_back(current_idx[d]);
  //             }

  //             if (dim == 1) break;

  //             // Graded Lexicographical logic to find the next combination
  //             int target = -1;
  //             for (int d = static_cast<int>(dim) - 2; d >= 0; --d) {
  //                 if (current_idx[d] > 0) {
  //                     target = d;
  //                     break;
  //                 }
  //             }

  //             if (target == -1) break;

  //             current_idx[target]--;
  //             current_idx[target + 1]++;

  //             for (unsigned int d = target + 2; d < dim; ++d) {
  //                 current_idx[target + 1] += current_idx[d];
  //                 current_idx[d] = 0;
  //             }
  //         }
  //     }
  //     return exponents;
  // }

  template <int dim>
  void parse_dealii_indices(
    const PolynomialSpace<dim>             &poly,
    std::vector<std::vector<unsigned int>> &monomials_exponents)
  {
    std::stringstream ss;
    poly.output_indices(ss);

    monomials_exponents.resize(dim);

    // 3. Parse the stream
    // The format is: Index  Exp_0  Exp_1 ...
    unsigned int val;
    unsigned int count = 0;

    while (ss >> val)
    {
      // output_indices prints: [RowIndex] [Dim0] [Dim1] ...
      // We want to skip the RowIndex, which occurs every (dim + 1) entries.
      const auto i = count % (dim + 1);
      if (i != 0)
        monomials_exponents[i - 1].push_back(val);
      count++;
    }
  }

  template <int dim>
  SolutionRecovery<dim>::SolutionRecovery(PatchHandler<dim> &patch_handler,
                                          const LA::ParVectorType  &solution,
                                          const FiniteElement<dim> &fe,
                                          const Mapping<dim>       &mapping)
    : patch_handler(patch_handler)
    , patches(patch_handler.patches)
    , solution(solution)
    , fe(fe)
    , mapping(mapping)
    , mpi_communicator(patch_handler.mpi_communicator)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , n_vertices(patch_handler.n_vertices)
    , owned_vertices(patch_handler.owned_vertices)
    , least_squares_matrices(patch_handler.n_vertices)
    , recoveries(patch_handler.n_vertices)
    , recoveries_coefficients(patch_handler.n_vertices)
  {
    /**
     * Create the set of locally relevant dofs
     */
    relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(patch_handler.dof_handler);
    for (const auto &patch : patches)
    {
      for (const auto &[dof, pt] : patch.neighbours)
      {
        relevant_dofs.add_index(dof);
      }
    }

    local_solution.reinit(patch_handler.dof_handler.locally_owned_dofs(),
                          relevant_dofs,
                          mpi_communicator);
    local_solution = solution;

    /**
     * Create the polynomial bases for polynomial fitting of degree p + 1 and
     * for the gradients
     */
    std::vector<Polynomials::Monomial<double>> monomials_1d_recovery;
    for (unsigned int i = 0; i <= fe.degree + 1; ++i)
      monomials_1d_recovery.push_back(Polynomials::Monomial<double>(i));
    std::vector<Polynomials::Monomial<double>> monomials_1d_gradient;
    for (unsigned int i = 0; i <= fe.degree; ++i)
      monomials_1d_gradient.push_back(Polynomials::Monomial<double>(i));

    for (const auto &m : monomials_1d_recovery)
    {
      pcout << "Monomial for recovery:" << std::endl;
      m.print(pcout.get_stream());
    }
    for (const auto &m : monomials_1d_gradient)
    {
      pcout << "Monomial for gradient:" << std::endl;
      m.print(pcout.get_stream());
    }

    monomials_recovery =
      std::make_shared<PolynomialSpace<dim>>(monomials_1d_recovery);
    monomials_gradient =
      std::make_shared<PolynomialSpace<dim>>(monomials_1d_gradient);

    pcout << "Rec" << std::endl;
    monomials_recovery->output_indices(std::cout);
    std::cout << "n_polynomials" << std::endl;
    std::cout << monomials_recovery->n_polynomials(monomials_1d_recovery.size())
              << std::endl;
    pcout << "Grad" << std::endl;
    monomials_gradient->output_indices(std::cout);
    std::cout << "n_polynomials" << std::endl;
    std::cout << monomials_gradient->n_polynomials(monomials_1d_gradient.size())
              << std::endl;

    dim_recovery_basis =
      monomials_recovery->n_polynomials(monomials_1d_recovery.size());
    dim_gradient_basis =
      monomials_gradient->n_polynomials(monomials_1d_gradient.size());

    // For each vector component, the number of fields to reconstruct (the field
    // + all its derivatives up to order "degree") and the number of derivatives
    // to store (sum of dim^i, i > 0, until degree + 1).
    n_fields_to_recover = 1;
    for (unsigned int i = 1; i <= fe.degree; ++i)
    {
      n_fields_to_recover += std::pow(dim, i);
    }
    n_derivatives_to_store =
      n_fields_to_recover - 1 + std::pow(dim, fe.degree + 1);

    for (types::global_vertex_index i = 0; i < n_vertices; ++i)
      if (owned_vertices[i])
        recoveries_coefficients[i].resize(n_fields_to_recover);

    pcout << "Computing least squares matrices" << std::endl;
    compute_least_squares_matrices();
    pcout << "Done computing least squares matrices" << std::endl;

    n_recovered_fields     = 0;
    n_derivatives_computed = 0;
    // for (unsigned int i = 0; i < n_fields_to_recover; ++i)
    for (unsigned int i = 0; i < 1; ++i)
    {
      // If i = 0, then a more accurate solution is fitted.
      // If i > 0, then a more accurate derivative is fitted.
      recover_from_solution(i);
    }

    // std::vector<unsigned int> index_map =
    //   build_commuting_derivative_to_symmetric_index_map<dim>(3, monomials);
    // for (auto &val : index_map)
    // {
    //   std::cout << "index_map " << val << std::endl;
    // }
  }

  /**
   * Use Eigen to get the rank of a FullMatrix
   */
  template <int dim>
  unsigned int get_rank(const FullMatrix<double> &full_matrix,
                        Eigen::MatrixXd          &eigen_matrix)
  {
    const unsigned int m = full_matrix.m();
    const unsigned int n = full_matrix.n();
    for (unsigned int i = 0; i < m; ++i)
      for (unsigned int j = 0; j < n; ++j)
        eigen_matrix(i, j) = full_matrix(i, j);
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(eigen_matrix);
    return lu_decomp.rank();
  }

  template <int dim>
  void SolutionRecovery<dim>::compute_least_squares_matrices()
  {
    // The matrices are of size dim_recovery_basis x n_adjacent,
    // but n_adjacent varies and can change if the patch is increased.
    // The least-squares matrix A^T * A, however, is dim_recovery_basis x
    // dim_recovery_basis

    FullMatrix<double> AtA(dim_recovery_basis, dim_recovery_basis);
    Eigen::MatrixXd    eigenAtA =
      Eigen::MatrixXd::Zero(dim_recovery_basis, dim_recovery_basis);

    // Construct (A^T*A)^-1 * A^T
    for (types::global_vertex_index i = 0; i < n_vertices; ++i)
    {
      if (!owned_vertices[i])
        continue;

      const Patch<dim> &patch = patches[i];

      bool         is_full_rank = false;
      unsigned int rank;
      // unsigned int num_patch_increases = 0;
      // unsigned int max_patch_increases = 2;

      do
      {
        const unsigned int n_adjacent = patch.neighbours.size();

        AssertThrow(
          n_adjacent >= dim_recovery_basis,
          ExcMessage(
            "Internal error: "
            "Cannot create least-squares matrix because a patch of support "
            "points "
            "has fewer vertices than the dimension of the polynomial "
            "basis for the polynomial fitting. This should not have "
            "happened, as the "
            "patches are created with at least that many vertices."));

        FullMatrix<double> A(n_adjacent, dim_recovery_basis);
        fill_vandermonde_matrix(patch, A);
        A.Tmmult(AtA, A);
        rank = get_rank<dim>(AtA, eigenAtA);

        pcout << "rank is " << rank << std::endl;

        AssertThrow(rank >= dim_recovery_basis,
                    ExcMessage("Matrix is not full rank at vertex"));

        // if (rank >= dim_recovery_basis)
        // {
        is_full_rank = true;
        FullMatrix<double> least_squares_mat(dim_recovery_basis, n_adjacent);
        least_squares_mat.left_invert(A);
        least_squares_matrices[i] = least_squares_mat;
        // }
        // else
        // {
        //   if (num_patch_increases++ > max_patch_increases)
        //   {
        //     throw std::runtime_error(
        //       "Could not create least-squares matrix of full rank even "
        //       "after increasing the patch size several times.");
        //   }

        //   // Increase patch size
        //   patches.increase_patch_size(i);
        // }
      }
      while (!is_full_rank);
    }
  }

  template <int dim>
  void
  SolutionRecovery<dim>::fill_vandermonde_matrix(const Patch<dim>   &patch,
                                                 FullMatrix<double> &mat) const
  {
    const auto &neighbours = patch.neighbours_local_coordinates;

    unsigned int i = 0;
    for (const auto &[dof, pt] : neighbours)
    {
      // Evaluate each monomial at local_coordinates
      for (unsigned int j = 0; j < dim_recovery_basis; ++j)
        mat(i, j) = monomials_recovery->compute_value(j, pt);
      ++i;
    }
  }

  // Get the solution field u_h at the vertices of a patch
  template <int dim>
  void get_component_values_on_patch(const LA::ParVectorType &local_solution,
                                     const Patch<dim>        &patch,
                                     Vector<double>          &values_out)
  {
    unsigned int i = 0;
    for (const auto &[dof, pt] : patch.neighbours_local_coordinates)
    {
      values_out[i] = local_solution[dof];
      ++i;
    }
  }

  //   // Get the previously computed derivative of u_h at the vertices of a
  //   patch void get_component_derivative_on_patch(
  //     const unsigned int i_derivative,
  //     const std::vector<std::vector<std::vector<Vector<double>>>>
  //                                                &recovered_derivatives,
  //     const std::set<types::global_vertex_index> &patch_vertex_indices,
  //     const unsigned int                          component,
  //     Vector<double>                             &values_out)
  //   {
  //     Assert(values_out.size() == patch_vertex_indices.size(),
  //            ExcMessage("Output vector size mismatch"));

  //     // For each vertex of the patch, evaluate its derivative polynomial
  //     // at (0,0) (= the independent term).
  //     unsigned int i = 0;
  //     for (const types::global_vertex_index &v : patch_vertex_indices)
  //     {
  //       values_out[i] = recovered_derivatives[v][component][i_derivative][0];
  //       ++i;
  //     }
  //   }

  /**
   *
   */
  // template <int dim>
  // PolynomialSpace<dim> coeffs_to_polynomial(const Vector<double> &coeffs)
  // {
  //   std::vector<Polynomials::Polynomial<double>> pols(dim);
  //   for (unsigned int d = 0; d < dim; ++d)
  //     pols[d] = Polynomials::Polynomial<double>()
  // }

  template <int dim>
  void SolutionRecovery<dim>::recover_from_solution(
    const unsigned int i_recovered_derivative)
  {
    Vector<double> coeffs(dim_recovery_basis);

    for (types::global_vertex_index v = 0; v < n_vertices; ++v)
    {
      if (!owned_vertices[v])
        continue;

      const Patch<dim> &patch  = patches[v];
      const auto       &ls_mat = least_squares_matrices[v];
      Vector<double>    rhs(ls_mat.n());

      // pcout << "Vertex " << patch.center << std::endl;
      // pcout << "Neighbours" << std::endl;
      // for (unsigned int i = 0; i < patch.neighbours_local_coordinates.size();
      // ++i)
      // {
      //   pcout << patch.neighbours[i].second << " - local: " <<
      //   patch.neighbours_local_coordinates[i].second << std::endl;
      // }
      // pcout << "LSM" << std::endl;
      // ls_mat.print_formatted(pcout.get_stream(), 4, true, 0, " ", 1., 0., "
      // ");

      // Extract local solution values for each component
      // if (i_recovered_derivative == 0)
      get_component_values_on_patch(local_solution, patch, rhs);
      // else
      //   get_component_derivative_on_patch(i_recovered_derivative - 1,
      //                                     recovered_derivatives,
      //                                     patch_vertices,
      //                                     c,
      //                                     rhs);

      ls_mat.vmult(coeffs, rhs);

      // Scale back
      for (unsigned int i = 0; i < dim_recovery_basis; ++i)
        coeffs[i] /= monomials_recovery->compute_value(i, patch.scaling);

      if (i_recovered_derivative == 0)
        recoveries_coefficients[v][0] = coeffs;

      // pcout << "RHS" << std::endl;
      // pcout << rhs << std::endl;
      // pcout << "Coeffs" << std::endl;
      // pcout << coeffs << std::endl;

      // // Store coefficients
      // recovered_polynomials[v][c][i_recovered_derivative] = coeffs;

      // // Compute derivatives
      // compute_derivatives(coeffs, recovered_derivatives[v][c]);
    }

    n_recovered_fields++;
    n_derivatives_computed += dim;
  }

  //   template <int dim>
  //   void SolutionRecovery<dim>::compute_derivatives(
  //     const Vector<double>        &coeffs,
  //     std::vector<Vector<double>> &derivatives)
  //   {
  //     AssertDimension(coeffs.size(), dim_recovery_basis);

  //     Vector<double> deriv;

  //     for (unsigned int d = 0; d < dim; ++d)
  //     {
  //       deriv.reinit(dim_derivative_basis);

  //       // Compute derivative of coeffs with respect to d-th coordinate
  //       for (unsigned int i = 0; i < dim_recovery_basis; ++i)
  //       {
  //         const auto &alpha = monomials[i];

  //         if (alpha[d] == 0)
  //           continue;

  //         // TODO: Check if the small terms can be capped,
  //         // or if it becomes unstable
  //         // if(std::abs(coeffs[i]) < 1e-10)
  //         // continue;

  //         std::array<unsigned int, dim> beta = alpha;
  //         beta[d] -= 1;

  //         auto it = monomials_derivatives_index.find(beta);
  //         if (it != monomials_derivatives_index.end())
  //           deriv[it->second] += coeffs[i] * alpha[d];
  //       }

  //       derivatives[n_derivatives_computed + d] = deriv;
  //     }
  //   }

  //   // template <int dim>
  //   // void SolutionRecovery<dim>::compute_homogeneous_error_polynomials()
  //   // {
  //   //   unsigned int       derivative_offset = 0;
  //   //   for (unsigned int i = 1; i < degree + 1; ++i)
  //   //     derivative_offset += std::pow(dim, i);

  //   //   // Number of monomials of degree "degree + 1":
  //   //   constexpr unsigned int n_coeffs = (dim == 2) ? degree + 1 :
  //   //     (degree + 2) * (degree + 3) / 2;

  //   //   // Get the monomials of degree "degree + 1"
  //   //   std::vector<std::array<unsigned int, dim>> highest_monomials;
  //   //   for(const auto &m : monomials)
  //   //   {
  //   //     unsigned int sum;
  //   //     if constexpr (dim == 2)
  //   //       sum = m[0] + m[1];
  //   //     else
  //   //       sum = m[0] + m[1] + m[2];
  //   //     if(sum == degree + 1)
  //   //       // TODO: Check ordering compatibility with metric computations
  //   //       highest_monomials.push_back(m);
  //   //   }

  //   //   AssertDimension(highest_monomials.size(), n_coeffs);

  //   //   homogeneous_error_polynomials.resize(n_vertices,
  //   std::vector<Vector<double>>(n_components));

  //   //   Vector<double> coeffs(n_coeffs);

  //   //   for (types::global_vertex_index v = 0; v < n_vertices; ++v)
  //   //   {
  //   //     for (unsigned int c = 0; c < n_components; ++c)
  //   //     {
  //   //       for (unsigned int i = 0; i < n_coeffs; ++i)
  //   //       {
  //   //         const auto &alpha = highest_monomials[i];

  //   //         // Compute multinomial denominator a!
  //   //         double alpha_fact = 1.0;
  //   //         for (const auto a : alpha)
  //   //           alpha_fact *= dealii::Utilities::factorial<double>(a);

  //   //         AssertIndexRange(derivative_offset + i,
  //   recovered_derivatives[v][c].size());

  //   //         coeffs[i] = recovered_derivatives[v][c][derivative_offset +
  //   i][0] / alpha_fact;
  //   //       }

  //   //       homogeneous_error_polynomials[v][c] = std::move(coeffs);
  //   //     }
  //   //   }
  //   // }

  //   // #include <algorithm> // std::next_permutation, std::sort
  //   // #include <numeric>   // std::accumulate
  //   // #include <map>
  //   // #include <vector>
  //   // #include <array>

  //   // template <int dim>
  //   // void SolutionRecovery<dim>::compute_homogeneous_error_polynomials()
  //   // {
  //   //   const unsigned int n_terms = monomials.size();

  //   //   // Helper: factorial for unsigned int
  //   //   auto factorial = [](unsigned int n) -> unsigned int {
  //   //     unsigned int f = 1;
  //   //     for (unsigned int i = 2; i <= n; ++i)
  //   //       f *= i;
  //   //     return f;
  //   //   };

  //   //   // Helper: multinomial coefficient = (sum(alpha)!) / (alpha_1! * ...
  //   * alpha_dim!)
  //   //   auto multinomial_coeff = [&](const std::array<unsigned int, dim>
  //   &alpha) -> unsigned int {
  //   //     unsigned int numerator = factorial(std::accumulate(alpha.begin(),
  //   alpha.end(), 0u));
  //   //     unsigned int denominator = 1;
  //   //     for (auto a : alpha)
  //   //       denominator *= factorial(a);
  //   //     return numerator / denominator;
  //   //   };

  //   //   // Helper: generate all distinct permutations of a multiset (alpha)
  //   //   auto generate_permutations = [&](const std::array<unsigned int, dim>
  //   &alpha)
  //   //     -> std::vector<std::array<unsigned int, dim>> {
  //   //     // Convert array to vector for easier permutation
  //   //     std::vector<unsigned int> v(alpha.begin(), alpha.end());
  //   //     // Sort to prepare for std::next_permutation
  //   //     std::sort(v.begin(), v.end());

  //   //     std::vector<std::array<unsigned int, dim>> permutations;
  //   //     do {
  //   //       std::array<unsigned int, dim> perm{};
  //   //       std::copy(v.begin(), v.end(), perm.begin());
  //   //       permutations.push_back(perm);
  //   //     } while (std::next_permutation(v.begin(), v.end()));

  //   //     return permutations;
  //   //   };

  //   //   // Build a map from multi-index to position in monomials vector for
  //   quick lookup
  //   //   // Using std::map with vector key since std::array supports
  //   operator< by default
  //   //   std::map<std::array<unsigned int, dim>, unsigned int>
  //   multiindex_to_pos;
  //   //   for (unsigned int i = 0; i < n_terms; ++i)
  //   //     multiindex_to_pos[monomials[i]] = i;

  //   //   // Resize output: [vertex][component] with length = n_terms (highest
  //   degree only)
  //   //   homogeneous_error_polynomials.resize(recovered_derivatives.size(),
  //   // std::vector<Vector<double>>(n_components));

  //   //   for (types::global_vertex_index v = 0; v <
  //   recovered_derivatives.size(); ++v)
  //   //   {
  //   //     for (unsigned int c = 0; c < n_components; ++c)
  //   //     {
  //   //       Vector<double> coeffs(n_terms);
  //   //       coeffs = 0.0;

  //   //       // Offset for highest order derivatives in
  //   recovered_derivatives[v][c]
  //   //       const unsigned int derivative_offset =
  //   //         (p == 0 ? 0 :
  //   //          p == 1 ? dim :
  //   //          p == 2 ? dim + dim * dim :
  //   //          throw dealii::ExcNotImplemented());

  //   //       for (unsigned int i = 0; i < n_terms; ++i)
  //   //       {
  //   //         const auto &alpha = monomials[i];

  //   //         // Generate all distinct permutations of alpha
  //   //         std::vector<std::array<unsigned int, dim>> permutations =
  //   generate_permutations(alpha);

  //   //         // Sum recovered derivatives over all permutations
  //   //         double derivative_sum = 0.0;
  //   //         for (const auto &perm : permutations)
  //   //         {
  //   //           // Find index of perm in monomials vector
  //   //           auto it = multiindex_to_pos.find(perm);
  //   //           if (it == multiindex_to_pos.end())
  //   //             AssertThrow(false, dealii::ExcMessage("Permutation
  //   multi-index not found in monomials vector"));

  //   //           const unsigned int perm_index = it->second;
  //   //           const unsigned int deriv_index = derivative_offset +
  //   perm_index;
  //   //           AssertIndexRange(deriv_index,
  //   recovered_derivatives[v][c].size());

  //   //           // Add constant term of recovered derivative polynomial at
  //   vertex
  //   //           derivative_sum +=
  //   recovered_derivatives[v][c][deriv_index][0];
  //   //         }

  //   //         // Divide by multinomial coefficient (number of permutations)
  //   //         const unsigned int alpha_factorial = multinomial_coeff(alpha);

  //   //         coeffs[i] = derivative_sum / alpha_factorial;
  //   //       }

  //   //       homogeneous_error_polynomials[v][c] = std::move(coeffs);
  //   //     }
  //   //   }
  //   // }

  //   template <int dim>
  //   void SolutionRecovery<dim>::write_derivatives_to_vtu(
  //     const unsigned int order) const
  //   {
  //     std::string msg = "Could not write derivatives of order " +
  //                       std::to_string(order) +
  //                       " because only derivatives of order up to " +
  //                       std::to_string(degree + 1) + " were computed.";
  //     AssertThrow(order <= degree + 1, ExcMessage(msg));

  //     const unsigned int n_derivatives     = std::pow(dim, order);
  //     unsigned int       derivative_offset = 0;
  //     for (unsigned int i = 1; i < order; ++i)
  //       derivative_offset += std::pow(dim, i);

  //     const unsigned int total_components = n_components * n_derivatives;

  //     // One linear DoF per component at each vertex
  //     FESystem<dim> fe_output(FE_SimplexP<dim>(1), total_components);

  //     DoFHandler<dim> dof_handler_out(triangulation);
  //     dof_handler_out.distribute_dofs(fe_output);

  //     Vector<double> output_vector(dof_handler_out.n_dofs());

  //     // Fill the vector: loop over cells and vertex DoFs
  //     for (const auto &cell : dof_handler_out.active_cell_iterators())
  //     {
  //       for (unsigned int v = 0; v < cell->n_vertices(); ++v)
  //       {
  //         const types::global_vertex_index vertex_idx =
  //         cell->vertex_index(v);

  //         AssertIndexRange(vertex_idx, recovered_derivatives.size());

  //         for (unsigned int c = 0; c < n_components; ++c)
  //         {
  //           AssertIndexRange(c, recovered_derivatives[vertex_idx].size());

  //           for (unsigned int d = 0; d < n_derivatives; ++d)
  //           {
  //             const unsigned int global_component = c * n_derivatives + d;
  //             const unsigned int dof_index =
  //               cell->vertex_dof_index(v, global_component);

  //             AssertIndexRange(derivative_offset + d,
  //                              recovered_derivatives[vertex_idx][c].size());

  //             // Evaluate at vertex center = constant term
  //             output_vector[dof_index] =
  //               recovered_derivatives[vertex_idx][c][derivative_offset +
  //               d][0];
  //           }
  //         }
  //       }
  //     }

  //     // Build output
  //     DataOut<dim> data_out;
  //     data_out.attach_dof_handler(dof_handler_out);

  //     std::vector<std::string> names;
  //     for (unsigned int c = 0; c < n_components; ++c)
  //       for (unsigned int d = 0; d < n_derivatives; ++d)
  //         names.emplace_back("u" + std::to_string(c) + "_d" +
  //                            std::to_string(order) + "_" +
  //                            std::to_string(d));

  //     std::vector<DataComponentInterpretation::DataComponentInterpretation>
  //       interpretation(total_components,
  //                      DataComponentInterpretation::component_is_scalar);

  //     data_out.add_data_vector(output_vector,
  //                              names,
  //                              DataOut<dim>::type_dof_data,
  //                              interpretation);
  //     data_out.build_patches();

  //     std::ofstream out("recovered_derivatives_order_" +
  //     std::to_string(order) +
  //                       ".vtu");
  //     data_out.write_vtu(out);
  //   }

  template <int dim>
  void
  SolutionRecovery<dim>::write_least_squares_systems(std::ostream &out) const
  {
    std::vector<Point<dim>>         global_vertices;
    std::vector<FullMatrix<double>> global_ls_matrices;
    std::vector<Vector<double>>     global_recovery_coeffs;

    const unsigned int mpi_rank =
      Utilities::MPI::this_mpi_process(mpi_communicator);
    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    const std::vector<Point<dim>> &vertices =
      patch_handler.triangulation.get_vertices();

    // Gather the mesh vertices
    {
      std::vector<Point<dim>> local_vertices;
      for (unsigned int i = 0; i < n_vertices; ++i)
        if (owned_vertices[i])
          local_vertices.push_back(vertices[i]);
      std::vector<std::vector<Point<dim>>> gathered_vertices =
        Utilities::MPI::all_gather(mpi_communicator, local_vertices);
      for (const auto &vec : gathered_vertices)
        global_vertices.insert(global_vertices.end(), vec.begin(), vec.end());
      std::sort(global_vertices.begin(),
                global_vertices.end(),
                PointComparator<dim>());
    }
    // Gather the least-squares matrices
    {
      using MessageType =
        std::vector<std::pair<Point<dim>, FullMatrix<double>>>;

      MessageType local_matrices;

      for (types::global_vertex_index i = 0; i < n_vertices; ++i)
        if (owned_vertices[i])
          local_matrices.push_back({vertices[i], least_squares_matrices[i]});

      std::vector<MessageType> gathered_matrices =
        Utilities::MPI::all_gather(mpi_communicator, local_matrices);

      global_ls_matrices.resize(global_vertices.size());
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        bool found = false;
        for (unsigned int r = 0; r < mpi_size; ++r)
          for (const auto &[pt, mat] : gathered_matrices[r])
            if (global_vertices[i].distance(pt) < 1e-12)
            {
              global_ls_matrices[i] = mat;
              found                 = true;
              break;
            }
        AssertThrow(found, ExcMessage("Vertex not found"));
      }
    }
    // Gather the coefficients of the solution recovery of degree p + 1
    {
      using MessageType = std::vector<std::pair<Point<dim>, Vector<double>>>;

      MessageType local_coeffs;

      for (types::global_vertex_index i = 0; i < n_vertices; ++i)
        if (owned_vertices[i])
          local_coeffs.push_back({vertices[i], recoveries_coefficients[i][0]});

      std::vector<MessageType> gathered_coeffs =
        Utilities::MPI::all_gather(mpi_communicator, local_coeffs);

      global_recovery_coeffs.resize(global_vertices.size());
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        bool found = false;
        for (unsigned int r = 0; r < mpi_size; ++r)
          for (const auto &[pt, coeffs] : gathered_coeffs[r])
            if (global_vertices[i].distance(pt) < 1e-12)
            {
              global_recovery_coeffs[i] = coeffs;
              found                     = true;
              break;
            }
        AssertThrow(found, ExcMessage("Vertex not found"));
      }
    }
    // Print
    if (mpi_rank == 0)
    {
      for (unsigned int i = 0; i < global_vertices.size(); ++i)
      {
        out << "Mesh vertex " << global_vertices[i] << std::endl;
        out << "Least-squares matrix" << std::endl;
        global_ls_matrices[i].print_formatted(
          out, 3, true, 0, " ", 1., 0., " ");
        out << "Polynomial coefficients" << std::endl;
        global_recovery_coeffs[i].print(out, 3, true, true);
      }
    }
  }

  template class SolutionRecovery<2>;
  template class SolutionRecovery<3>;

} // namespace ErrorEstimation