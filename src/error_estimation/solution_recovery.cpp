
#include <deal.II/base/polynomial_space.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>
#include <error_estimation/patches.h>
#include <error_estimation/solution_recovery.h>
#include <mesh_and_dof_tools.h>
#include <utilities.h>

#include <Eigen/Dense>

namespace ErrorEstimation
{
  namespace SolutionRecovery
  {
    template <int dim>
    SolutionRecoveryBase<dim>::SolutionRecoveryBase(
      const unsigned int          highest_recovered_derivative,
      const ParameterReader<dim> &param,
      PatchHandler<dim>          &patch_handler,
      const DoFHandler<dim>      &dof_handler,
      const LA::ParVectorType    &solution,
      const FiniteElement<dim>   &fe,
      const Mapping<dim>         &mapping)
      : highest_recovered_derivative(highest_recovered_derivative)
      , param(param)
      , patch_handler(patch_handler)
      , patches(patch_handler.patches)
      , dof_handler(dof_handler)
      , fe(fe)
      , mapping(mapping)
      , isoparam_dh(dof_handler.get_triangulation())
      , mpi_communicator(patch_handler.mpi_communicator)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
      , n_vertices(patch_handler.n_vertices)
      , owned_vertices(patch_handler.owned_vertices)
      , least_squares_matrices(patch_handler.n_vertices)
      , recoveries_coefficients(patch_handler.n_vertices)
    {
      // Create the set of locally relevant dofs, including the additional dofs
      // needed to recover data on the patches on this partition
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      relevant_dofs      = DoFTools::extract_locally_relevant_dofs(dof_handler);
      for (const auto &patch : patches)
        for (const auto &[dof, pt] : patch.neighbours)
          relevant_dofs.add_index(dof);

      local_solution.reinit(locally_owned_dofs, mpi_communicator);
      solution_with_additional_ghosts.reinit(locally_owned_dofs,
                                             relevant_dofs,
                                             mpi_communicator);
      local_solution                  = solution;
      solution_with_additional_ghosts = solution;

      // Create the polynomial bases for polynomial fitting of degree p + 1 and
      // for the gradients
      std::vector<Polynomials::Monomial<double>> monomials_1d_recovery;
      for (unsigned int i = 0; i <= fe.degree + 1; ++i)
        monomials_1d_recovery.push_back(Polynomials::Monomial<double>(i));

      // for (const auto &m : monomials_1d_recovery)
      // {
      //   pcout << "Monomial for recovery:" << std::endl;
      //   m.print(pcout.get_stream());
      // }

      monomials_recovery =
        std::make_unique<PolynomialSpace<dim>>(monomials_1d_recovery);
      // monomials_recovery->output_indices(std::cout);
      dim_recovery_basis =
        monomials_recovery->n_polynomials(monomials_1d_recovery.size());

      // Vectors to call evaluate() from the PolynomialSpace.
      // Only those actually wanted (the gradients) are allocated.
      std::vector<double> recovery_monomials_values;
      gradients_of_recovery_monomials.resize(dim_recovery_basis);
      std::vector<Tensor<2, dim>> recovery_monomials_grad_grads;
      std::vector<Tensor<3, dim>> recovery_monomials_third_derivatives;
      std::vector<Tensor<4, dim>> recovery_monomials_fourth_derivatives;

      // Evaluate grad(monomials) at the origin.
      // The gradient of each reconstructed component at zero is given by
      // sum_i coeffs_i * nabla(P_i)(0), where nabla(P_i)(0) is
      // gradients_of_recovery_monomials.
      monomials_recovery->evaluate(Point<dim>(),
                                   recovery_monomials_values,
                                   gradients_of_recovery_monomials,
                                   recovery_monomials_grad_grads,
                                   recovery_monomials_third_derivatives,
                                   recovery_monomials_fourth_derivatives);

      // For each vector component, the number of fields to reconstruct (the
      // field
      // + all its derivatives up to order "degree") and the number of
      // derivatives to store (sum of dim^i, i > 0, until degree + 1).
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
    void SolutionRecoveryBase<dim>::fill_vandermonde_matrix(
      const Patch<dim>   &patch,
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

    template <int dim>
    void SolutionRecoveryBase<dim>::compute_least_squares_matrices()
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

          // pcout << "rank is " << rank << std::endl;

          // FIXME: Enlarge patches if not of full rank
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
    void SolutionRecoveryBase<dim>::reconstruct_fields()
    {
      for (unsigned int i = 0; i < highest_recovered_derivative; ++i)
      {
        // If i = 0, then a more accurate solution is fitted.
        // If i > 0, then a more accurate derivative is fitted.
        reconstruct_field(i);
      }
    }

    template <int dim>
    double SolutionRecoveryBase<dim>::compute_integral_error(
      const RecoveryType          type,
      const VectorTools::NormType norm_type,
      const Function<dim>        &exact_solution,
      const Quadrature<dim>      &cell_quadrature) const
    {
      // Get the component mask for the required field type, and the mask
      // consisting of all but its components, to set all other dofs to zero
      const ComponentSelectFunction<dim> *component_select = nullptr;
      switch (type)
      {
        case RecoveryType::solution:
          component_select = solution_comp_select.get();
          break;
        case RecoveryType::gradient:
          AssertThrow(
            highest_recovered_derivative > 0,
            ExcMessage(
              "You are trying to compute the nodal error over a reconstructed "
              "gradient, but this quantity was not reconstructed by this "
              "SolutionRecovery. Set highest_recovered_derivative to at least "
              "1 to reconstruct the gradient of the solution."));
          component_select = gradient_comp_select.get();
          break;
        case RecoveryType::hessian:
          AssertThrow(
            highest_recovered_derivative > 1,
            ExcMessage(
              "You are trying to compute the nodal error over a reconstructed "
              "Hessian, but this quantity was not reconstructed by this "
              "SolutionRecovery. Set highest_recovered_derivative to at least "
              "2 to reconstruct the Hessian of the solution."));
          component_select = hessian_comp_select.get();
          break;
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

      const auto            &tria = isoparam_dh.get_triangulation();
      dealii::Vector<double> cellwise_errors(tria.n_active_cells());

      // Error between recovered solution and exact solution
      VectorTools::integrate_difference(*isoparam_mapping,
                                        isoparam_dh,
                                        isoparam_solution,
                                        exact_solution,
                                        cellwise_errors,
                                        cell_quadrature,
                                        norm_type,
                                        component_select);
      double norm =
        VectorTools::compute_global_error(tria, cellwise_errors, norm_type);
      return norm;
    }

    template <int dim>
    double SolutionRecoveryBase<dim>::compute_nodal_error(
      const RecoveryType          type,
      const VectorTools::NormType norm_type,
      const Function<dim>        &exact_solution) const
    {
      // Get the component mask for the required field type, and the mask
      // consisting of all but its components, to set all other dofs to zero
      const ComponentMask *mask = nullptr;
      switch (type)
      {
        case RecoveryType::solution:
          mask = &solution_mask;
          break;
        case RecoveryType::gradient:
          AssertThrow(
            highest_recovered_derivative > 0,
            ExcMessage(
              "You are trying to compute the nodal error over a reconstructed "
              "gradient, but this quantity was not reconstructed by this "
              "SolutionRecovery. Set highest_recovered_derivative to at least "
              "1 to reconstruct the gradient of the solution."));
          mask = &gradient_mask;
          break;
        case RecoveryType::hessian:
          AssertThrow(
            highest_recovered_derivative > 1,
            ExcMessage(
              "You are trying to compute the nodal error over a reconstructed "
              "Hessian, but this quantity was not reconstructed by this "
              "SolutionRecovery. Set highest_recovered_derivative to at least "
              "2 to reconstruct the Hessian of the solution."));
          mask = &hessian_mask;
          break;
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }

      std::vector<bool> all_but_type_component_mask(n_isoparam_components,
                                                    true);
      for (unsigned int i = 0; i < n_isoparam_components; ++i)
        all_but_type_component_mask[i] = !(*mask)[i];
      const ComponentMask all_but_type_mask(all_but_type_component_mask);

      // Interpolate the exact solution or its derivatives at the isoparametric
      // dofs
      LA::ParVectorType local_nodal_error, nodal_error;
      local_nodal_error.reinit(locally_owned_isoparam_dofs, mpi_communicator);
      nodal_error.reinit(locally_owned_isoparam_dofs,
                         locally_relevant_isoparam_dofs,
                         mpi_communicator);
      VectorTools::interpolate(*isoparam_mapping,
                               isoparam_dh,
                               exact_solution,
                               local_nodal_error,
                               *mask);

      // Subtract all the reconstructed fields
      local_nodal_error -= local_isoparam_solution;

      // Interpolate the zero function everywhere except at the required dofs,
      // to overwrite the dofs that are not of the required RecoveryType
      VectorTools::interpolate(*isoparam_mapping,
                               isoparam_dh,
                               Functions::ZeroFunction<dim>(
                                 n_isoparam_components),
                               local_nodal_error,
                               all_but_type_mask);
      nodal_error = local_nodal_error;

      // Compute the \ell^p error between the solution vectors
      switch (norm_type)
      {
        case VectorTools::NormType::L1_norm:
          return nodal_error.l1_norm();
        case VectorTools::NormType::L2_norm:
          return nodal_error.l2_norm();
        case VectorTools::NormType::Linfty_norm:
          return nodal_error.linfty_norm();
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
      DEAL_II_ASSERT_UNREACHABLE();
      return -1;
    }

    template <int dim>
    void SolutionRecoveryBase<dim>::write_least_squares_systems(
      std::ostream &out) const
    {
      std::vector<Point<dim>>             global_vertices;
      std::vector<FullMatrix<double>>     global_ls_matrices;
      std::vector<dealii::Vector<double>> global_recovery_coeffs;

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
        using MessageType =
          std::vector<std::pair<Point<dim>, dealii::Vector<double>>>;

        MessageType local_coeffs;

        for (types::global_vertex_index i = 0; i < n_vertices; ++i)
          if (owned_vertices[i])
            local_coeffs.push_back(
              {vertices[i], recoveries_coefficients[i][0]});

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
            out, 3, true, 0, "0.", 1., 0., " ");
          out << "Polynomial coefficients" << std::endl;
          global_recovery_coeffs[i].print(out, 3, true, true);
        }
      }
    }

    template <int dim>
    Scalar<dim>::Scalar(const unsigned int highest_recovered_derivative,
                        const ParameterReader<dim> &param,
                        PatchHandler<dim>          &patch_handler,
                        const DoFHandler<dim>      &dof_handler,
                        const LA::ParVectorType    &solution,
                        const FiniteElement<dim>   &fe,
                        const Mapping<dim>         &mapping)
      : SolutionRecoveryBase<dim>(highest_recovered_derivative,
                                  param,
                                  patch_handler,
                                  dof_handler,
                                  solution,
                                  fe,
                                  mapping)
      , recovered_solution_at_vertices(this->n_vertices)
      , recovered_gradient_at_vertices(this->n_vertices)
      , recovered_hessian_at_vertices(this->n_vertices)
    {
      constexpr unsigned int mapping_degree = 1;

      // Isoparametric representation to associate recovered data to each mesh
      // vertex
      if (param.finite_elements.use_quads)
      {
        this->isoparam_mapping =
          std::make_unique<MappingQ<dim>>(mapping_degree);
        switch (highest_recovered_derivative)
        {
          case 0:
            this->isoparam_fe =
              std::make_unique<FESystem<dim>>(FE_Q<dim>(mapping_degree));
            break;
          case 1:
            this->isoparam_fe = std::make_unique<FESystem<dim>>(
              FE_Q<dim>(mapping_degree),
              FE_Q<dim>(mapping_degree) ^
                gradient_type::n_independent_components);
            break;
          case 2:
            this->isoparam_fe = std::make_unique<FESystem<dim>>(
              FE_Q<dim>(mapping_degree),
              FE_Q<dim>(mapping_degree) ^
                gradient_type::n_independent_components,
              FE_Q<dim>(mapping_degree) ^
                hessian_type::n_independent_components);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }
      else
      {
        this->isoparam_mapping =
          std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(mapping_degree));
        switch (highest_recovered_derivative)
        {
          case 0:
            this->isoparam_fe =
              std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(mapping_degree));
            break;
          case 1:
            this->isoparam_fe = std::make_unique<FESystem<dim>>(
              FE_SimplexP<dim>(mapping_degree),
              FE_SimplexP<dim>(mapping_degree) ^
                gradient_type::n_independent_components);
            break;
          case 2:
            this->isoparam_fe = std::make_unique<FESystem<dim>>(
              FE_SimplexP<dim>(mapping_degree),
              FE_SimplexP<dim>(mapping_degree) ^
                gradient_type::n_independent_components,
              FE_SimplexP<dim>(mapping_degree) ^
                hessian_type::n_independent_components);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }
      this->isoparam_dh.distribute_dofs(*this->isoparam_fe);

      // Total number of components in the isoparametric FE solution
      // = 1 + dim + dim^2 + ... = (dim^(N+1) - 1) / (dim - 1)
      this->n_isoparam_components =
        (std::pow(dim, highest_recovered_derivative + 1) - 1) / (dim - 1);

      const auto comm     = this->mpi_communicator;
      auto      &owned    = this->locally_owned_isoparam_dofs;
      auto      &relevant = this->locally_relevant_isoparam_dofs;

      // Initialize vectors using the isoparametric dof handler
      owned    = this->isoparam_dh.locally_owned_dofs();
      relevant = DoFTools::extract_locally_relevant_dofs(this->isoparam_dh);

      // Create parallel vectors and maps for the FE isoparametric
      // representation of the reconstructed solution, gradient, hessian, etc.
      this->isoparam_solution.reinit(owned, relevant, comm);
      this->local_isoparam_solution.reinit(owned, comm);

      {
        const unsigned int               solution_offset = 0;
        const FEValuesExtractors::Scalar solution_extractor(0);
        this->solution_mask =
          this->isoparam_fe->component_mask(solution_extractor);
        this->solution_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(
            0, this->n_isoparam_components);

        create_mesh_vertex_to_tensor_dofs_maps<dim, 1>(
          solution_offset,
          this->isoparam_dh,
          relevant,
          vertices_to_solution_dofs,
          solution_dofs_to_vertices);
      }

      if (highest_recovered_derivative > 0)
      {
        const unsigned int               gradient_offset = 1;
        const FEValuesExtractors::Vector gradient_extractor(1);
        this->gradient_mask =
          this->isoparam_fe->component_mask(gradient_extractor);
        this->gradient_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(
            std::make_pair(gradient_offset,
                           gradient_offset +
                             gradient_type::n_independent_components),
            this->n_isoparam_components);

        create_mesh_vertex_to_tensor_dofs_maps<
          dim,
          gradient_type::n_independent_components>(gradient_offset,
                                                   this->isoparam_dh,
                                                   relevant,
                                                   vertices_to_gradient_dofs,
                                                   gradient_dofs_to_vertices);
      }

      if (highest_recovered_derivative > 1)
      {
        const unsigned int hessian_offset = 1 + dim;
        std::vector<bool>  hessian_component_mask(this->n_isoparam_components,
                                                 false);
        for (unsigned int i = 0; i < hessian_type::n_independent_components;
             ++i)
          hessian_component_mask[hessian_offset + i] = true;
        this->hessian_mask = ComponentMask(hessian_component_mask);
        this->hessian_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(
            std::make_pair(hessian_offset,
                           hessian_offset +
                             hessian_type::n_independent_components),
            this->n_isoparam_components);

        create_mesh_vertex_to_tensor_dofs_maps<
          dim,
          hessian_type::n_independent_components>(hessian_offset,
                                                  this->isoparam_dh,
                                                  relevant,
                                                  vertices_to_hessian_dofs,
                                                  hessian_dofs_to_vertices);
      }
    }

    // Get the solution field u_h at the vertices of a patch
    template <int dim>
    void get_component_values_on_patch(
      const LA::ParVectorType &solution_with_additional_ghosts,
      const Patch<dim>        &patch,
      dealii::Vector<double>  &values_out)
    {
      unsigned int i = 0;
      for (const auto &[dof, pt] : patch.neighbours_local_coordinates)
      {
        values_out[i] = solution_with_additional_ghosts[dof];
        ++i;
      }
    }

    /**
     * FIXME: for nwo this does the same as get_component_values_on_patch(),
     * if the local solution vector is updated with the components to
     * reconstruct...
     */
    template <int dim>
    void get_gradient_on_patch(
      const unsigned int       component,
      const LA::ParVectorType &solution_with_additional_ghosts,
      const Patch<dim>        &patch,
      dealii::Vector<double>  &values_out)
    {
      // FIXME:
      // The patches dof are from the FE dofhandler, not from the
      // isoparametric... But more importantly, the patch will have more dofs
      // than the mesh vertices: we need to evaluate the reconstructed fields at
      // the Point<dim> of the patch nodes?
      unsigned int i = 0;
      for (const auto &[dof, pt] : patch.neighbours_local_coordinates)
      {
        // values_out[i] = recovered_gradient_at_vertices[v][component];
        values_out[i] = solution_with_additional_ghosts[dof];
        ++i;
      }
    }

    template <int dim>
    void Scalar<dim>::reconstruct_field(const unsigned int derivative_order)
    {
      const unsigned int n = this->dim_recovery_basis;
      const unsigned int n_components_to_recover =
        std::pow(dim, derivative_order);
      dealii::Vector<double> coeffs(n);

      // Reconstruct each component of the solution, gradient, etc.
      for (unsigned int i_comp = 0; i_comp < n_components_to_recover; ++i_comp)
      {
        if (derivative_order == 1)
        {
          // Update the copy of the FE solution vector with the gradient of the
          // reconstructed solution We need the dof index of the center of the
          // patch
          for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
            if (this->owned_vertices[v])
            {
              const Patch<dim> &patch = this->patches[v];

              // Get the dof of the neighbour that is actually the center
              // For patches defined for a scalar-valued field, there is only
              // one such dof
              types::global_dof_index center_dof =
                numbers::invalid_unsigned_int;
              for (const auto &neighbour : patch.neighbours)
                if (neighbour.second.distance(patch.center) < 1e-12)
                {
                  center_dof = neighbour.first;
                  break;
                }
              AssertThrow(center_dof != numbers::invalid_unsigned_int,
                          ExcMessage(
                            "Could not find a neighbouring dof in the patch "
                            "that matches the center of the patch"));
              AssertThrow(this->locally_owned_dofs.is_element(center_dof),
                          ExcMessage("Center dof is not owned"));

              // Update the copy of the FE solution
              this->local_solution[center_dof] =
                recovered_gradient_at_vertices[v][i_comp];
            }

          // Update ghosts
          this->local_solution.compress(VectorOperation::insert);
          this->solution_with_additional_ghosts = this->local_solution;
        }
        else if (derivative_order == 2)
        {
          // Update the local solution vector with the current hessian values,
          // which will then be reconstructed.
          DEAL_II_NOT_IMPLEMENTED();
        }

        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
        {
          if (!this->owned_vertices[v])
            continue;

          const Patch<dim>      &patch  = this->patches[v];
          const auto            &ls_mat = this->least_squares_matrices[v];
          dealii::Vector<double> rhs(ls_mat.n());

          // Extract local solution values for each component
          if (derivative_order == 0)
            get_component_values_on_patch(this->solution_with_additional_ghosts,
                                          patch,
                                          rhs);
          else if (derivative_order == 1)
            get_gradient_on_patch(i_comp,
                                  this->solution_with_additional_ghosts,
                                  patch,
                                  rhs);
          else
            DEAL_II_NOT_IMPLEMENTED();

          // Solve the least-squares system
          ls_mat.vmult(coeffs, rhs);

          // Scale back
          for (unsigned int i = 0; i < n; ++i)
            coeffs[i] /=
              this->monomials_recovery->compute_value(i, patch.scaling);

          if (derivative_order == 0)
          {
            // Store reconstructed solution evaluated at the origin
            this->recoveries_coefficients[v][0] = coeffs;
            recovered_solution_at_vertices[v]   = coeffs[0];

            // Compute the gradient at the origin = sum_i coeffs_i *
            // grad(P_i)(0)
            Tensor<1, dim> grad;
            for (unsigned int i = 0; i < n; ++i)
              grad += coeffs[i] * this->gradients_of_recovery_monomials[i];
            recovered_gradient_at_vertices[v] = grad;
          }
          else if (derivative_order == 1)
          {
            // Store reconstructed gradient evaluated at the origin
            recovered_gradient_at_vertices[v][i_comp] = coeffs[0];

            // Compute grad(grad_icomp) at the origin = sum_i coeffs_i *
            // grad(P_i)(0)
            Tensor<1, dim> grad;
            for (unsigned int i = 0; i < n; ++i)
              grad += coeffs[i] * this->gradients_of_recovery_monomials[i];
            for (unsigned int d = 0; d < dim; ++d)
              recovered_hessian_at_vertices[v][i_comp][d] = grad[d];
          }
        }
      }

      if (derivative_order == 0)
      {
        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
          if (this->owned_vertices[v])
            std::cout << "Gradient from recovered solution: "
                      << recovered_gradient_at_vertices[v] << std::endl;

        // Store vertex-based solution as an FE isoparametric solution
        this->vertex_to_isoparametric(recovered_solution_at_vertices,
                                      this->local_isoparam_solution,
                                      this->isoparam_solution,
                                      vertices_to_solution_dofs);
        // Store vertex-based gradient as an FE isoparametric gradient
        this->template vertex_to_isoparametric<
          1,
          gradient_type::n_independent_components>(
          recovered_gradient_at_vertices,
          this->local_isoparam_solution,
          this->isoparam_solution,
          vertices_to_gradient_dofs);
      }
      else if (derivative_order == 1)
      {
        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
          if (this->owned_vertices[v])
            std::cout << "Hessian from recovered gradient: "
                      << recovered_hessian_at_vertices[v] << std::endl;

        // Store vertex-based hessian as an FE isoparametric hessian
        this->template vertex_to_isoparametric<
          2,
          hessian_type::n_independent_components>(recovered_hessian_at_vertices,
                                                  this->local_isoparam_solution,
                                                  this->isoparam_solution,
                                                  vertices_to_hessian_dofs);
      }
    }

    template <int dim>
    void Scalar<dim>::write_pvtu(const std::string &filename) const
    {
      std::vector<std::string> data_names(this->n_isoparam_components);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interpretation(this->n_isoparam_components);

      data_names[0]          = "solution";
      data_interpretation[0] = DataComponentInterpretation::component_is_scalar;

      if (this->highest_recovered_derivative > 0)
        for (unsigned int i = 0; i < gradient_type::n_independent_components;
             ++i)
        {
          data_names[1 + i] = "gradient";
          data_interpretation[1 + i] =
            DataComponentInterpretation::component_is_part_of_vector;
        }
      if (this->highest_recovered_derivative > 1)
        for (unsigned int i = 0; i < hessian_type::n_independent_components;
             ++i)
        {
          data_names[1 + dim + i] = "hessian";
          data_interpretation[1 + dim + i] =
            DataComponentInterpretation::component_is_part_of_tensor;
        }

      DataOut<dim> data_out;
      data_out.attach_dof_handler(this->isoparam_dh);
      data_out.add_data_vector(this->isoparam_solution,
                               data_names,
                               DataOut<dim>::type_dof_data,
                               data_interpretation);
      data_out.build_patches(*this->isoparam_mapping, 2);
      data_out.write_vtu_with_pvtu_record(
        "./", filename, 0, this->isoparam_dh.get_mpi_communicator(), 2);
    }

    template class SolutionRecoveryBase<2>;
    template class SolutionRecoveryBase<3>;
    template class Scalar<2>;
    template class Scalar<3>;
    template class Vector<2>;
    template class Vector<3>;
  } // namespace SolutionRecovery
} // namespace ErrorEstimation
