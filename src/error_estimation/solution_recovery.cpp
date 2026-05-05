
#include <deal.II/base/mpi.h>
#include <deal.II/base/polynomial_space.h>
#include <deal.II/distributed/tria_base.h>
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
    Base<dim>::Base(const unsigned int          highest_recovered_derivative,
                    const ParameterReader<dim> &param,
                    PatchHandler<dim>          &patch_handler,
                    const DoFHandler<dim>      &dof_handler,
                    const LA::ParVectorType    &solution,
                    const FiniteElement<dim>   &fe,
                    const Mapping<dim>         &mapping,
                    const ComponentMask        &mask,
                    const bool                  isoparametric,
                    const bool                  single_reconstruction)
      : highest_recovered_derivative(highest_recovered_derivative)
      , isoparametric(isoparametric)
      , single_reconstruction(single_reconstruction)
      , param(param)
      , patch_handler(patch_handler)
      , patches(patch_handler.patches)
      , solution_dh(dof_handler)
      , solution_fe(fe)
      , solution_mapping(mapping)
      , mask(mask)
      , dh(dof_handler.get_triangulation())
      , mpi_communicator(patch_handler.mpi_communicator)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
      , n_vertices(patch_handler.n_vertices)
      , owned_vertices(patch_handler.owned_vertices)
      , degree(fe.get_sub_fe(mask).degree)
      , least_squares_matrices(patch_handler.get_least_squares_matrices())
      , recoveries_coefficients(highest_recovered_derivative)
    {
      // Check that the least-squares matrices were computed during patches
      // creation
      AssertThrow(
        patch_handler.has_least_squares_matrices(),
        ExcMessage(
          "The PatchHandler provided to reconstruct the solution and "
          "derivatives does not store the least-squares matrices at each "
          "vertex patch. When creating the patches with build_patches(...), "
          "set enforce_full_rank_least_squares_matrices to true, or simply "
          "call build_patches() with the default arguments."));

      // Create the set of locally relevant dofs, including the additional dofs
      // needed to recover data on the patches on this partition
      locally_owned_dofs = solution_dh.locally_owned_dofs();
      relevant_dofs      = DoFTools::extract_locally_relevant_dofs(solution_dh);
      for (const auto &patch : patches)
        for (const auto &data : patch.neighbours)
          relevant_dofs.add_index(data.dof);

      local_solution.reinit(locally_owned_dofs, mpi_communicator);
      solution_with_additional_ghosts.reinit(locally_owned_dofs,
                                             relevant_dofs,
                                             mpi_communicator);
      local_solution                  = solution;
      solution_with_additional_ghosts = solution;

      // Polynomial bases of degree "degree + 1"
      std::vector<Polynomials::Monomial<double>> monomials_1d_recovery;
      for (unsigned int i = 0; i <= degree + 1; ++i)
        monomials_1d_recovery.push_back(Polynomials::Monomial<double>(i));

      monomials_recovery =
        std::make_unique<PolynomialSpace<dim>>(monomials_1d_recovery);
      dim_recovery_basis =
        monomials_recovery->n_polynomials(monomials_1d_recovery.size());

      // Evaluate grad(monomials) at the origin.
      // The gradient of each reconstructed component at zero is given by
      // sum_i coeffs_i * nabla(P_i)(0), where nabla(P_i)(0) is
      // gradients_of_recovery_monomials.
      // Only the vectors actually wanted are allocated.
      gradients_of_recovery_monomials.resize(dim_recovery_basis);
      hessians_of_recovery_monomials.resize(dim_recovery_basis);
      third_derivatives_of_recovery_monomials.resize(dim_recovery_basis);
      monomials_recovery->evaluate(Point<dim>(),
                                   empty_polynomial_space_values,
                                   gradients_of_recovery_monomials,
                                   hessians_of_recovery_monomials,
                                   third_derivatives_of_recovery_monomials,
                                   empty_polynomial_space_fourth_derivatives);

      // For each vector component, the number of fields to reconstruct (the
      // field
      // + all its derivatives up to order "degree") and the number of
      // derivatives to store (sum of dim^i, i > 0, until degree + 1).
      n_fields_to_recover = 1;
      for (unsigned int i = 1; i <= degree; ++i)
      {
        n_fields_to_recover += std::pow(dim, i);
      }
      n_derivatives_to_store =
        n_fields_to_recover - 1 + std::pow(dim, degree + 1);
    }

    template <int dim>
    void Base<dim>::reconstruct_fields(const LA::ParVectorType &solution)
    {
      // FIXME: add verbosity condition
      pcout << std::endl;
      pcout << "-- Reconstructing solution and derivatives of order up to "
            << highest_recovered_derivative << "..." << std::endl;

      local_solution                  = solution;
      solution_with_additional_ghosts = solution;

      recoveries_coefficients.resize(highest_recovered_derivative + 1);

      if (isoparametric && single_reconstruction)
      {
        /**
         * Reconstruct only the solution, then compute all derivatives from it.
         */
        reconstruct_field(0);
      }
      else
      {
        /**
         * Successively reconstruct and take the gradient, starting with the
         * given field. If i = 0, a more accurate solution is fitted.
         * If i > 0, a more accurate derivative is fitted.
         */
        for (unsigned int i = 0; i < std::max(highest_recovered_derivative, 1u);
             ++i)
        {
          reconstruct_field(i);
        }
      }
    }

    template <int dim>
    double Base<dim>::compute_integral_error(
      const RecoveryType          type,
      const VectorTools::NormType norm_type,
      const Mapping<dim>         &mapping,
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

      const auto            &tria = dh.get_triangulation();
      dealii::Vector<double> cellwise_errors(tria.n_active_cells());

      // Error between recovered solution and exact solution
      VectorTools::integrate_difference(mapping,
                                        dh,
                                        recovery_solution,
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
    double
    Base<dim>::compute_nodal_error(const RecoveryType          type,
                                   const VectorTools::NormType norm_type,
                                   const Mapping<dim>         &mapping,
                                   const Function<dim> &exact_solution) const
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

      // Interpolate the exact solution or its derivatives at the isoparametric
      // dofs
      LA::ParVectorType local_nodal_error, nodal_error;
      local_nodal_error.reinit(locally_owned_recovery_dofs, mpi_communicator);
      nodal_error.reinit(locally_owned_recovery_dofs,
                         locally_relevant_recovery_dofs,
                         mpi_communicator);
      VectorTools::interpolate(
        mapping, dh, exact_solution, local_nodal_error, *mask);

      // Subtract all the reconstructed fields
      local_nodal_error -= local_recovery_solution;

      if (n_components > 1)
      {
        // Interpolate the zero function everywhere except at the required dofs,
        // to overwrite the dofs that are not of the required RecoveryType
        std::vector<bool> all_but_type_component_mask(n_components, true);
        for (unsigned int i = 0; i < n_components; ++i)
          all_but_type_component_mask[i] = !(*mask)[i];
        const ComponentMask all_but_type_mask(all_but_type_component_mask);

        VectorTools::interpolate(mapping,
                                 dh,
                                 Functions::ZeroFunction<dim>(n_components),
                                 local_nodal_error,
                                 all_but_type_mask);
      }

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
    void Base<dim>::write_least_squares_systems(std::ostream &out) const
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
              {vertices[i], recoveries_coefficients[0][0][i]});

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
    void Scalar<dim>::create_solution_dofs_to_recovery_dofs_map()
    {
      vertices_to_solution_dofs.clear();
      vertices_to_gradient_dofs.clear();
      vertices_to_hessian_dofs.clear();

      vertices_to_solution_dofs.resize(this->n_vertices);
      vertices_to_gradient_dofs.resize(this->n_vertices);
      vertices_to_hessian_dofs.resize(this->n_vertices);

      solution_dofs_to_recovery_dofs.clear();
      solution_dofs_to_gradient_dofs.clear();
      solution_dofs_to_hessian_dofs.clear();

      const unsigned int n_isoparam_dofs_per_cell =
        this->isoparam_fe->n_dofs_per_cell();

      const unsigned int n_rec_dofs_per_cell = this->fe->n_dofs_per_cell();
      std::vector<types::global_dof_index> local_rec_dof_indices(
        n_rec_dofs_per_cell);

      const unsigned int n_sol_dofs_per_cell =
        this->solution_fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> local_sol_dof_indices(
        n_sol_dofs_per_cell);

      const unsigned int n_dofs_selected_component =
        this->solution_fe.get_sub_fe(this->mask).n_dofs_per_cell();
      std::vector<types::global_dof_index> solution_dofs(
        n_dofs_selected_component);

      // Loop over cells using the *recovery* dof handler
      for (const auto &cell : this->dh.active_cell_iterators())
      {
        cell->get_dof_indices(local_rec_dof_indices);

        // Get this cell as a solution dh iterator
        const auto solution_cell =
          cell->as_dof_handler_iterator(this->solution_dh);
        solution_cell->get_dof_indices(local_sol_dof_indices);

        // Get the solution dofs for this mask on this cell
        for (unsigned int i = 0; i < n_sol_dofs_per_cell; ++i)
        {
          const auto component_shape =
            this->solution_fe.system_to_component_index(i);
          const unsigned int comp  = component_shape.first;
          const unsigned int shape = component_shape.second;

          if (this->mask[comp])
            solution_dofs[shape] = local_sol_dof_indices[i];
        }

        // For each recovery dof on this cell, map it to the associated solution
        // dof if their shape indices match.
        for (unsigned int i = 0; i < n_rec_dofs_per_cell; ++i)
        {
          const auto component_shape = this->fe->system_to_component_index(i);
          const unsigned int comp    = component_shape.first;
          const unsigned int shape   = component_shape.second;

          AssertIndexRange(shape, n_dofs_selected_component);

          // Map the smoothed solution dofs
          const int c_sol = comp - solution_offset;
          if (c_sol == 0)
          {
            solution_dofs_to_recovery_dofs[solution_dofs[shape]][c_sol] =
              local_rec_dof_indices[i];

            // Also map the vertex data to isoparametric recovery dofs
            // use "shape" as vertex_index
            if (shape < n_isoparam_dofs_per_cell)
            {
              const auto vertex_index = cell->vertex_index(shape);
              vertices_to_solution_dofs[vertex_index][c_sol] =
                local_rec_dof_indices[i];
            }
          }

          // Map the smoothed gradient dofs
          if (this->highest_recovered_derivative > 0)
          {
            const int c_grad = comp - gradient_offset;
            if (0 <= c_grad &&
                c_grad < (int)gradient_type::n_independent_components)
            {
              solution_dofs_to_gradient_dofs[solution_dofs[shape]][c_grad] =
                local_rec_dof_indices[i];

              // Map the vertex data to isoparametric recovery dofs
              if (shape < n_isoparam_dofs_per_cell)
              {
                const auto vertex_index = cell->vertex_index(shape);
                vertices_to_gradient_dofs[vertex_index][c_grad] =
                  local_rec_dof_indices[i];
              }
            }
          }

          // Map the smoothed hessian dofs
          if (this->highest_recovered_derivative >= 1)
          {
            const int c_hess = comp - hessian_offset;
            if (0 <= c_hess &&
                c_hess < (int)hessian_type::n_independent_components)
            {
              solution_dofs_to_hessian_dofs[solution_dofs[shape]][c_hess] =
                local_rec_dof_indices[i];

              // Map the vertex data to isoparametric recovery dofs
              if (shape < n_isoparam_dofs_per_cell)
              {
                const auto vertex_index = cell->vertex_index(shape);
                vertices_to_hessian_dofs[vertex_index][c_hess] =
                  local_rec_dof_indices[i];
              }
            }
          }
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
                        const Mapping<dim>         &mapping,
                        const ComponentMask        &mask,
                        const bool                  isoparametric,
                        const bool                  single_reconstruction)
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
      , recovered_solution_at_vertices(this->n_vertices)
      , recovered_gradient_at_vertices(this->n_vertices)
      , recovered_hessian_at_vertices(this->n_vertices)
    {
      AssertThrow(
        mask.n_selected_components(fe.n_components()) == 1,
        ExcMessage(
          "You are trying to create a SolutionRecovery::Scalar, but the "
          "provided ComponentMask selects more than a single solution "
          "component. Use a mask for a single scalar component, or "
          "alternatively create a SolutionRecovery::Vector to reconstruct the "
          "derivatives of a vector-valued field."));

      constexpr unsigned int mapping_degree = 1;

      // The degree of the finite element spaces used to represent the
      // reconstructed solution and derivatives. If isoparametric, this is the
      // degree of the mapping, otherwise it is the degree of the finite element
      // solution. This degree is *not* the degree of the least-squares
      // polynomials used to smooth the solution.
      const unsigned int recovery_degree =
        isoparametric ? mapping_degree : this->degree;

      // Finite element space used to represent to smoothed data.
      // This FESystem holds a scalar FE space for the smoothed solution, a
      // vector-valued FE space for the smoothed gradient, for the smoothed
      // hessian, etc.
      if (param.finite_elements.use_quads)
      {
        this->isoparam_fe =
          std::make_unique<FESystem<dim>>(FE_Q<dim>(mapping_degree));
        switch (highest_recovered_derivative)
        {
          case 0:
            this->fe =
              std::make_unique<FESystem<dim>>(FE_Q<dim>(recovery_degree));
            break;
          case 1:
            this->fe = std::make_unique<FESystem<dim>>(
              FE_Q<dim>(recovery_degree),
              FE_Q<dim>(recovery_degree) ^
                gradient_type::n_independent_components);
            break;
          case 2:
            this->fe = std::make_unique<FESystem<dim>>(
              FE_Q<dim>(recovery_degree),
              FE_Q<dim>(recovery_degree) ^
                gradient_type::n_independent_components,
              FE_Q<dim>(recovery_degree) ^
                hessian_type::n_independent_components);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }
      else
      {
        this->isoparam_fe =
          std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(mapping_degree));
        switch (highest_recovered_derivative)
        {
          case 0:
            this->fe = std::make_unique<FESystem<dim>>(
              FE_SimplexP<dim>(recovery_degree));
            break;
          case 1:
            this->fe = std::make_unique<FESystem<dim>>(
              FE_SimplexP<dim>(recovery_degree),
              FE_SimplexP<dim>(recovery_degree) ^
                gradient_type::n_independent_components);
            break;
          case 2:
            this->fe = std::make_unique<FESystem<dim>>(
              FE_SimplexP<dim>(recovery_degree),
              FE_SimplexP<dim>(recovery_degree) ^
                gradient_type::n_independent_components,
              FE_SimplexP<dim>(recovery_degree) ^
                hessian_type::n_independent_components);
            break;
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }
      }
      this->dh.distribute_dofs(*this->fe);

      // Total number of vector components in this object's FESystem
      // = 1 + dim + dim^2 + ... = (dim^(N+1) - 1) / (dim - 1)
      this->n_components =
        (std::pow(dim, highest_recovered_derivative + 1) - 1) / (dim - 1);

      const auto comm     = this->mpi_communicator;
      auto      &owned    = this->locally_owned_recovery_dofs;
      auto      &relevant = this->locally_relevant_recovery_dofs;

      // Initialize vectors using the isoparametric dof handler
      owned    = this->dh.locally_owned_dofs();
      relevant = DoFTools::extract_locally_relevant_dofs(this->dh);

      // Parallel vectors for the FE representation of the smoothed data.
      this->recovery_solution.reinit(owned, relevant, comm);
      this->local_recovery_solution.reinit(owned, comm);

      // Get the owners of the ghost recovery dofs, as they are unrelated to the
      // owners of the solution ghost dofs.
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>
        all_local_ranges =
          Utilities::MPI::all_gather(this->mpi_communicator,
                                     this->recovery_solution.local_range());

      const unsigned int mpi_size =
        Utilities::MPI::n_mpi_processes(this->mpi_communicator);
      IndexSet ghost_dofs = relevant;
      ghost_dofs.subtract_set(owned);
      for (const auto dof : ghost_dofs)
        for (unsigned int rank = 0; rank < mpi_size; ++rank)
        {
          const auto &[start, end] = all_local_ranges[rank];
          if (dof >= start && dof < end)
          {
            this->ghost_owners[dof] = rank;
            break;
          }
        }

      // Masks for each derivative
      {
        const FEValuesExtractors::Scalar solution_extractor(0);
        this->solution_mask = this->fe->component_mask(solution_extractor);
        this->solution_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(0, this->n_components);
      }
      if (highest_recovered_derivative > 0)
      {
        const FEValuesExtractors::Vector gradient_extractor(1);
        this->gradient_mask = this->fe->component_mask(gradient_extractor);
        this->gradient_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(
            std::make_pair(gradient_offset,
                           gradient_offset +
                             gradient_type::n_independent_components),
            this->n_components);
      }
      if (highest_recovered_derivative > 1)
      {
        std::vector<bool> hessian_component_mask(this->n_components, false);
        for (unsigned int i = 0; i < hessian_type::n_independent_components;
             ++i)
          hessian_component_mask[hessian_offset + i] = true;
        this->hessian_mask = ComponentMask(hessian_component_mask);
        this->hessian_comp_select =
          std::make_unique<ComponentSelectFunction<dim>>(
            std::make_pair(hessian_offset,
                           hessian_offset +
                             hessian_type::n_independent_components),
            this->n_components);
      }

      create_solution_dofs_to_recovery_dofs_map();

      // Compute the weights associated with the closest dofs on each patch
      compute_patches_averaging_weights();
    }

    template <int dim>
    double Base<dim>::evaluate_polynomial(
      const Point<dim>             &p,
      const PolynomialSpace<dim>   &polynomial_space,
      const dealii::Vector<double> &polynomial_coeffs,
      std::vector<double>          &basis)
    {
      AssertDimension(polynomial_coeffs.size(), basis.size());

      // Evaluate the polynomial space at p.
      // Since the empty_* vectors are empty, only the basis is computed.
      polynomial_space.evaluate(p,
                                basis,
                                empty_polynomial_space_grads,
                                empty_polynomial_space_grad_grads,
                                empty_polynomial_space_third_derivatives,
                                empty_polynomial_space_fourth_derivatives);
      double             res = 0;
      const unsigned int n   = basis.size();
      for (unsigned int i = 0; i < n; ++i)
        res += polynomial_coeffs[i] * basis[i];
      return res;
    }

    template <int dim>
    Tensor<1, dim> Base<dim>::evaluate_polynomial_gradient(
      const Point<dim>             &p,
      const PolynomialSpace<dim>   &polynomial_space,
      const dealii::Vector<double> &polynomial_coeffs,
      std::vector<Tensor<1, dim>>  &basis_gradients)
    {
      AssertDimension(polynomial_coeffs.size(), basis_gradients.size());

      // Evaluate the gradient of the polynomial space at p.
      // Since the empty_* vectors are empty, only the gradient is computed.
      polynomial_space.evaluate(p,
                                empty_polynomial_space_values,
                                basis_gradients,
                                empty_polynomial_space_grad_grads,
                                empty_polynomial_space_third_derivatives,
                                empty_polynomial_space_fourth_derivatives);
      Tensor<1, dim>     res;
      const unsigned int n = basis_gradients.size();
      for (unsigned int i = 0; i < n; ++i)
        res += polynomial_coeffs[i] * basis_gradients[i];
      return res;
    }

    template <int dim>
    void Scalar<dim>::compute_patches_averaging_weights()
    {
      using DofData = typename Patch<dim>::DofData;

      const auto &reference_support_points =
        this->solution_fe.get_sub_fe(this->mask).get_unit_support_points();

      // Weighting of the polynomials evaluations is done using a scalar
      // isoparametric FE, i.e., the base FE associated with the solution. In
      // the vector-valued case this might get trickier, and we'll probably need
      // to add a scalar-valued isoparametric FE, just for this purpose.
      // const auto &solution_isoparam_fe = this->fe->base_element(0);

      // The patches use dof indices from the main solver's dof handler
      //  Must use this dof handler and the solver's FESystem.
      const unsigned int n_dofs_per_cell = this->solution_fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

      std::vector<DofData> contributions_to_ghosts;

      for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
        if (this->owned_vertices[v])
        {
          auto &patch = this->patches[v];

          std::vector<unsigned int> n_contributions(patch.neighbours.size(), 0);

          // Reset all the weights
          for (auto &dof_data : patch.neighbours)
            dof_data.averaging_weight = 0.;

          for (const auto &cell : patch.first_cell_layer)
          {
            // Find which cell vertex the current vertex is
            unsigned int iv = numbers::invalid_unsigned_int;
            for (const auto i : cell->vertex_indices())
              if (cell->vertex_index(i) == v)
              {
                iv = i;
                break;
              }
            Assert(iv != numbers::invalid_unsigned_int,
                   ExcMessage("This vertex is not a vertex of one of its "
                              "first layer cells..."));

            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            {
              const auto comp_shape =
                this->solution_fe.system_to_component_index(i);
              const unsigned int comp = comp_shape.first;

              if (this->mask[comp])
              {
                const auto dof = local_dofs[i];

                // Check if dof is in the patch
                // FIXME: This should always be the case, since these are the
                // first layer of elements around the vertices, so all dofs of
                // the chosen field should be in the patch
                auto it =
                  std::find_if(patch.neighbours.begin(),
                               patch.neighbours.end(),
                               [&](const DofData &d) { return d.dof == dof; });

                Assert(it != patch.neighbours.end(),
                       ExcMessage(
                         "Dof of the chosen field in the first layer is "
                         "not in the patch of a cell vertex"));

                // Evaluate the weight from the isoparametric FE
                // This dof is at the "shape"-th point in the solver FE's
                // reference cell
                const unsigned int shape = comp_shape.second;

                AssertIndexRange(shape, reference_support_points.size());
                const Point<dim> &ref_coord = reference_support_points[shape];

                // Now evaluate the vertex-th shape function of the
                // isoparametric FE at these reference coordinates
                AssertIndexRange(iv, this->isoparam_fe->n_dofs_per_cell());
                const double weight =
                  this->isoparam_fe->shape_value(iv, ref_coord);
                it->averaging_weight += weight;

                const unsigned int neighbour_index =
                  std::distance(patch.neighbours.begin(), it);
                n_contributions[neighbour_index]++;
              }
            }
          }

          // Divide the weight at each dof by the total number of contributions
          // from this vertex
          for (unsigned int i = 0; i < patch.neighbours.size(); ++i)
          {
            DofData &data = patch.neighbours[i];

            if (n_contributions[i] > 0)
              data.averaging_weight /= n_contributions[i];

            // In debug, add the nonzero weight contributions to ghosts to send
            // to their owners
            if constexpr (running_in_debug_mode())
            {
              if (!this->locally_owned_dofs.is_element(data.dof) &&
                  std::abs(data.averaging_weight) > 1e-14)
                contributions_to_ghosts.push_back(data);
            }
          }
        }

      /**
       * In debug, check that each owned dof has weights that sum to 1 across
       * all patches, that is, check that it will receive complete contributions
       * to its gradients.
       */
      if constexpr (running_in_debug_mode())
      {
        using MessageType = std::vector<DofData>;

        std::map<types::global_dof_index, double> sum_weights;

        std::map<types::subdomain_id, MessageType> contributions_to_send;
        for (const auto &data : contributions_to_ghosts)
          contributions_to_send[data.owner].push_back(data);

        std::map<unsigned int, MessageType> received_contributions =
          Utilities::MPI::some_to_some(this->mpi_communicator,
                                       contributions_to_send);

        // Add the weight contribution from this partition
        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
          if (this->owned_vertices[v])
            for (const auto &data : this->patches[v].neighbours)
              if (this->locally_owned_dofs.is_element(data.dof))
                sum_weights[data.dof] += data.averaging_weight;

        // Add the weight contribution from other partitions
        for (const auto &[source_rank, contributions_to_add] :
             received_contributions)
          for (const auto &data_to_add : contributions_to_add)
          {
            // Make sure this rank is the right owner
            Assert(this->locally_owned_dofs.is_element(data_to_add.dof),
                   ExcInternalError());
            // if (sum_weights.at(data_to_add.dof) > 0)
            sum_weights[data_to_add.dof] += data_to_add.averaging_weight;
          }

        // Check that weights sum to 1
        for (const auto &[dof, sum_weight] : sum_weights)
          Assert(std::abs(sum_weight - 1.) < 1e-12,
                 ExcMessage("Proc " +
                            std::to_string(Utilities::MPI::this_mpi_process(
                              this->mpi_communicator)) +
                            " : owned dof " + std::to_string(dof) +
                            " has sum of weights " +
                            std::to_string(sum_weight) +
                            " but it should sum to 1."));
      }
    }

    inline void exchange_contributions_and_update_ghosts(
      const IndexSet    &locally_owned_dofs,
      const MPI_Comm     mpi_communicator,
      LA::ParVectorType &local_solution,
      LA::ParVectorType &solution,
      std::map<types::subdomain_id,
               std::vector<std::pair<types::global_dof_index, double>>>
        &contributions_to_ghosts)
    {
      // Send contributions to ghosts
      std::map<unsigned int,
               std::vector<std::pair<types::global_dof_index, double>>>
        received_contributions =
          Utilities::MPI::some_to_some(mpi_communicator,
                                       contributions_to_ghosts);

      for (const auto &[source_rank, contributions_to_add] :
           received_contributions)
        for (const auto &[dof, contribution] : contributions_to_add)
          if (locally_owned_dofs.is_element(dof))
            // Increment local solution vector
            local_solution[dof] += contribution;

      // Update ghosts
      local_solution.compress(VectorOperation::add);
      solution = local_solution;
    }

    template <int dim>
    void Scalar<dim>::update_local_solution(
      const unsigned int derivative_order,
      const unsigned int reconstruction_component,
      const unsigned int gradient_component)
    {
      AssertIndexRange(gradient_component, dim);
      const auto &coefficients =
        this->recoveries_coefficients[derivative_order - 1]
                                     [reconstruction_component];

      std::vector<Tensor<1, dim>> basis_gradients(this->dim_recovery_basis);

      // Reset the local copy of the solution
      this->local_solution = 0;

      std::map<types::subdomain_id,
               std::vector<std::pair<types::global_dof_index, double>>>
        contributions_to_ghosts;

      for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
        if (this->owned_vertices[v])
        {
          const auto &patch  = this->patches[v];
          const auto &coeffs = coefficients[v];

          for (const auto &data : patch.neighbours)
            if (std::abs(data.averaging_weight) > 1e-14)
            {
              Point<dim> pt = data.local_pt;

              // Scale back?
              // FIXME: it's probably more robust to keep the coefficients
              // scaled in reconstruct_field(), and to keep the local coord
              // here as well, instead of scaling back both...
              for (unsigned int d = 0; d < dim; ++d)
                pt[d] *= patch.scaling[d];

              // Evaluate the gradient of the polynomial reconstruction at
              // this dof's support point
              const Tensor<1, dim> grad = this->evaluate_polynomial_gradient(
                pt, *this->monomials_recovery, coeffs, basis_gradients);

              // Add weighted average
              if (this->locally_owned_dofs.is_element(data.dof))
                this->local_solution[data.dof] +=
                  data.averaging_weight * grad[gradient_component];
              else
                contributions_to_ghosts[data.owner].push_back(
                  {data.dof, data.averaging_weight * grad[gradient_component]});
            }
        }

      /**
       * Send contributions to dof owners, which will increment their local part
       * of the solution vector.
       */
      exchange_contributions_and_update_ghosts(
        this->locally_owned_dofs,
        this->mpi_communicator,
        this->local_solution,
        this->solution_with_additional_ghosts,
        contributions_to_ghosts);
    }

    /**
     * Update the solution storing the smoothed data.
     *
     * Given the coefficients of the polynomials stored in
     * this->recoveries_coefficients, evaluate the polynomials at the
     * non-vertices dofs and average them using the
     * pre-computed weights. Store the result in the @p component-th vector
     * component of @p local_dof_data, and update the ghosts
     * in @p dof_data.
     *
     * This function only fills the non-vertices dofs in local_dof_data, that
     * is, the dofs whose support point is not also a mesh vertex.
     *
     * If @p gradient is true, then we actually evaluate and average the
     * gradient of the polynomials stored in this->recoveries_coefficients.
     */
    template <int dim>
    void Scalar<dim>::evaluate_and_average_recovery_solution(
      const RecoveryType type,
      const unsigned int derivative_order,
      const unsigned int reconstruction_component,
      const bool         gradient)
    {
      const auto &coefficients =
        this
          ->recoveries_coefficients[derivative_order][reconstruction_component];

      std::vector<double>         basis(this->dim_recovery_basis);
      std::vector<Tensor<1, dim>> basis_gradients(this->dim_recovery_basis);

      std::map<types::subdomain_id,
               std::vector<std::pair<types::global_dof_index, double>>>
        contributions_to_ghosts;

      // Reset this component in local_recovery_solution
      std::vector<bool> component_mask(this->n_components, false);
      if (gradient)
      {
        unsigned int offset = 1; // components of solution
        if (type == RecoveryType::gradient)
          offset += gradient_type::n_independent_components +
                    reconstruction_component * dim;
        else if (type == RecoveryType::hessian)
          DEAL_II_NOT_IMPLEMENTED();

        for (unsigned int d = 0; d < dim; ++d)
          component_mask[offset + d] = true;
      }
      else
      {
        if (type == RecoveryType::solution)
          component_mask[solution_offset + reconstruction_component] = true;
        else if (type == RecoveryType::gradient)
          component_mask[gradient_offset + reconstruction_component] = true;
        else if (type == RecoveryType::hessian)
          component_mask[hessian_offset + reconstruction_component] = true;
        else
          DEAL_II_NOT_IMPLEMENTED();
      }

      VectorTools::interpolate(this->solution_mapping,
                               this->dh,
                               Functions::ZeroFunction<dim>(this->n_components),
                               this->local_recovery_solution,
                               ComponentMask(component_mask));

      for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
        if (this->owned_vertices[v])
        {
          const auto &patch  = this->patches[v];
          const auto &coeffs = coefficients[v];

          for (const auto &data : patch.neighbours)
            if (std::abs(data.averaging_weight) > 1e-14)
            {
              Point<dim> pt = data.local_pt;

              // Scale back?
              // FIXME: it's probably more robust to keep the coefficients
              // scaled in reconstruct_field(), and to keep the local coord
              // here as well, instead of scaling back both...
              for (unsigned int d = 0; d < dim; ++d)
                pt[d] *= patch.scaling[d];

              if (gradient)
              {
                // Evaluate the gradient of the polynomial reconstruction at
                // this dof's support point.
                const Tensor<1, dim> grad = this->evaluate_polynomial_gradient(
                  pt, *this->monomials_recovery, coeffs, basis_gradients);

                for (unsigned int d = 0; d < dim; ++d)
                {
                  const double val = data.averaging_weight * grad[d];

                  // Get the dof to increment in the recovery solution
                  // Since this is the gradient, take the dof from the
                  // appropriate derviative
                  unsigned int dof = numbers::invalid_unsigned_int;
                  if (type == RecoveryType::solution)
                  {
                    Assert(solution_dofs_to_gradient_dofs.count(data.dof) > 0,
                           ExcInternalError());
                    dof = solution_dofs_to_gradient_dofs.at(data.dof)[d];
                  }
                  else if (type == RecoveryType::gradient)
                  {
                    Assert(solution_dofs_to_hessian_dofs.count(data.dof) > 0,
                           ExcInternalError());
                    dof = solution_dofs_to_hessian_dofs.at(
                      data.dof)[reconstruction_component *
                                  gradient_type::n_independent_components +
                                d];
                  }
                  else if (type == RecoveryType::hessian)
                  {
                    DEAL_II_NOT_IMPLEMENTED();
                  }

                  Assert(dof != numbers::invalid_unsigned_int,
                         ExcInternalError());

                  // Add weighted average
                  if (this->locally_relevant_recovery_dofs.is_element(dof))
                    this->local_recovery_solution[dof] += val;
                  else
                    contributions_to_ghosts[this->ghost_owners.at(dof)]
                      .push_back({dof, val});
                }
              }
              else
              {
                // Evaluate the polynomial reconstruction
                const double val =
                  data.averaging_weight *
                  this->evaluate_polynomial(pt,
                                            *this->monomials_recovery,
                                            coeffs,
                                            basis);

                // Get the dof to increment in the recovery solution
                unsigned int dof = numbers::invalid_unsigned_int;
                if (type == RecoveryType::solution)
                {
                  Assert(solution_dofs_to_recovery_dofs.count(data.dof) > 0,
                         ExcInternalError());
                  dof = solution_dofs_to_recovery_dofs.at(
                    data.dof)[reconstruction_component];
                }
                else if (type == RecoveryType::gradient)
                {
                  Assert(solution_dofs_to_gradient_dofs.count(data.dof) > 0,
                         ExcInternalError());
                  dof = solution_dofs_to_gradient_dofs.at(
                    data.dof)[reconstruction_component];
                }
                else if (type == RecoveryType::hessian)
                {
                  Assert(solution_dofs_to_hessian_dofs.count(data.dof) > 0,
                         ExcInternalError());
                  dof = solution_dofs_to_hessian_dofs.at(
                    data.dof)[reconstruction_component];
                }

                Assert(dof != numbers::invalid_unsigned_int,
                       ExcInternalError());

                // Add weighted average
                if (this->locally_relevant_recovery_dofs.is_element(dof))
                  this->local_recovery_solution[dof] += val;
                else
                  contributions_to_ghosts[this->ghost_owners.at(dof)].push_back(
                    {dof, val});
              }
            }
        }

      exchange_contributions_and_update_ghosts(
        this->locally_owned_recovery_dofs,
        this->mpi_communicator,
        this->local_recovery_solution,
        this->recovery_solution,
        contributions_to_ghosts);
    }

    template <int dim>
    void Scalar<dim>::update_recovery_solution(
      const RecoveryType type,
      const unsigned int derivative_order,
      const unsigned int reconstruction_component)
    {
      if (this->isoparametric)
      {
        /**
         * The polynomials were fitted at the vertices, so to obtain an
         * isoparametric representation of the PPR operator, we just need to
         * transfer the data stored at the mesh vertices to data known at
         * isoparametric dofs.
         *
         * Since the gradient is available "for free" whenever a field is
         * reconstructed, we transfer the vertex-based data for the last
         * reconstructed field and the gradient of each of its components.
         */
        if (type == RecoveryType::solution)
        {
          // Store vertex-based solution in FE solution
          this->vertex_to_isoparametric(recovered_solution_at_vertices,
                                        this->local_recovery_solution,
                                        this->recovery_solution,
                                        vertices_to_solution_dofs);

          if (this->highest_recovered_derivative >= derivative_order + 1)
            // Store vertex-based gradient in FE solution
            this->template vertex_to_isoparametric<
              1,
              gradient_type::n_independent_components>(
              recovered_gradient_at_vertices,
              this->local_recovery_solution,
              this->recovery_solution,
              vertices_to_gradient_dofs);
        }
        else if (type == RecoveryType::gradient)
        {
          /**
           * At this point, each component of the gradient has been
           * reconstructed as well with a polynomial of degree p + 1, so we
           * could update the gradient with these values. They don't seem to be
           * always more accurate though, so it's not done for now.
           */

          // // Store vertex-based gradient in FE solution
          // this->template vertex_to_isoparametric<
          //   1,
          //   gradient_type::n_independent_components>(
          //   recovered_gradient_at_vertices,
          //   this->local_recovery_solution,
          //   this->recovery_solution,
          //   vertices_to_gradient_dofs);

          if (this->highest_recovered_derivative >= derivative_order + 1)
            // Store vertex-based hessian in FE solution
            this->template vertex_to_isoparametric<
              2,
              hessian_type::n_independent_components>(
              recovered_hessian_at_vertices,
              this->local_recovery_solution,
              this->recovery_solution,
              vertices_to_hessian_dofs);
        }
        else
          DEAL_II_NOT_IMPLEMENTED();

        if (this->single_reconstruction)
        {
          if (this->degree >= 1)
            this->template vertex_to_isoparametric<
              2,
              hessian_type::n_independent_components>(
              recovered_hessian_at_vertices,
              this->local_recovery_solution,
              this->recovery_solution,
              vertices_to_hessian_dofs);
        }
      }
      else
      {
        /**
         * We want a PPR operator in (V_h)^dim, where V_h is the discrete space
         * of the given scalar field. The values of this operator are available
         * at the dofs associated with mesh vertices (the isoparametric dofs),
         * and the values at the remaining dofs (edges and interior) are
         * obtained by averaging the evaluations of the least-squares
         * polynomials from adjacent mesh vertices.
         */

        if (derivative_order == 0)
          // Average the evaluations of the smoothed solution
          evaluate_and_average_recovery_solution(type,
                                                 derivative_order,
                                                 reconstruction_component,
                                                 false);

        if (this->highest_recovered_derivative >= derivative_order + 1)
        {
          // Average the evaluations of the smoothed gradient
          evaluate_and_average_recovery_solution(type,
                                                 derivative_order,
                                                 reconstruction_component,
                                                 true);
        }
      }
    }

    // Get the solution field u_h at the vertices of a patch
    template <int dim>
    void get_solution_values_on_patch(
      const LA::ParVectorType &solution_with_additional_ghosts,
      const Patch<dim>        &patch,
      dealii::Vector<double>  &values_out)
    {
      unsigned int i = 0;
      for (const auto &data : patch.neighbours)
      {
        values_out[i] = solution_with_additional_ghosts[data.dof];
        ++i;
      }
    }

    template <int dim>
    void Base<dim>::solve_least_squares_problem(const unsigned int vertex_index,
                                                dealii::Vector<double> &coeffs)
    {
      const Patch<dim> &patch  = this->patches[vertex_index];
      const auto       &ls_mat = this->least_squares_matrices[vertex_index];
      dealii::Vector<double> rhs(ls_mat.n());

      // Extract local solution values for each component
      get_solution_values_on_patch(this->solution_with_additional_ghosts,
                                   patch,
                                   rhs);

      // Solve the least-squares system
      ls_mat.vmult(coeffs, rhs);

      // Scale back
      // FIXME: see other comment on scaling.
      for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
        coeffs[i] /= this->monomials_recovery->compute_value(i, patch.scaling);
    }

    // Compute gradient at origin = sum_i coeffs_i * grad(P_i)(0)
    template <int dim>
    Tensor<1, dim> Base<dim>::gradient_at_origin(
      const dealii::Vector<double> &polynomial_coeffs) const
    {
      Tensor<1, dim> grad;
      for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
        grad += polynomial_coeffs[i] * this->gradients_of_recovery_monomials[i];
      return grad;
    }

    // Compute hessian at origin = sum_i coeffs_i * hess(P_i)(0)
    template <int dim>
    Tensor<2, dim> Base<dim>::hessian_at_origin(
      const dealii::Vector<double> &polynomial_coeffs) const
    {
      Tensor<2, dim> hess;
      for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
        hess += polynomial_coeffs[i] * this->hessians_of_recovery_monomials[i];
      return hess;
    }

    template <int dim>
    Tensor<3, dim> Base<dim>::third_derivatives_at_origin(
      const dealii::Vector<double> &polynomial_coeffs) const
    {
      Tensor<3, dim> res;
      for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
        res += polynomial_coeffs[i] *
               this->third_derivatives_of_recovery_monomials[i];
      return res;
    }

    template <int dim>
    void Scalar<dim>::reconstruct_field(const unsigned int derivative_order)
    {
      /**
       * Set some bookkeeping variables.
       *
       * The cases are written explicitly because the first reconstruction
       * reconstructs a single component, the scalar field itself, whereas for
       * all subsequent reconstructions, the *dim* components of the gradient of
       * each previously reconstructed component is reconstructed.
       */
      RecoveryType type;
      unsigned int n_components_to_reconstruct, n_components_to_obtain;
      if (derivative_order == 0)
      {
        // Reconstruct the scalar field, yielding a scalar field.
        // Store 1 polynomial at each mesh vertex.
        type                        = RecoveryType::solution;
        n_components_to_reconstruct = n_solution_components;
        n_components_to_obtain      = n_solution_components;
      }
      else if (derivative_order == 1)
      {
        // Reconstruct the gradient of a scalar field.
        // Store *dim* polynomials at each mesh vertex.
        type                        = RecoveryType::gradient;
        n_components_to_reconstruct = n_solution_components;
        n_components_to_obtain      = gradient_type::n_independent_components;
      }
      else if (derivative_order == 2)
      {
        // Reconstruct the gradient of the gradient of a scalar field.
        // Store *dim*dim* polynomials at each mesh vertex.
        type                        = RecoveryType::hessian;
        n_components_to_reconstruct = gradient_type::n_independent_components;
        n_components_to_obtain      = hessian_type::n_independent_components;
      }
      else
        DEAL_II_NOT_IMPLEMENTED();

      this->recoveries_coefficients[derivative_order].resize(
        n_components_to_obtain);

      for (unsigned int i_comp = 0; i_comp < n_components_to_reconstruct;
           ++i_comp)
      {
        /**
         * For the first reconstruction (the numerical field), we only need to
         * reconstruct 1 component, the scalar field itself. Then, we store the
         * gradient of this field and fit polynomials for each gradient
         * component of the previously recovered quantity.
         */
        const unsigned int dim_to_reconstruct = derivative_order == 0 ? 1 : dim;

        for (unsigned int d = 0; d < dim_to_reconstruct; ++d)
        {
          const unsigned int comp_to_reconstruct =
            i_comp * dim_to_reconstruct + d;
          auto &coefficients =
            this
              ->recoveries_coefficients[derivative_order][comp_to_reconstruct];
          coefficients.resize(this->n_vertices);

          if (derivative_order > 0)
          {
            /**
             * Set local copy of the solution as the d-th component of the
             * gradient of the recovered solution. This way, we use the copy to
             * update the ghost values, and we use a single function to evaluate
             * the values of all fields to reconstruct on a patch, getting the
             * field at the patch dofs.
             */
            update_local_solution(derivative_order, i_comp, d);
          }

          for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
          {
            if (!this->owned_vertices[v])
              continue;

            auto &coeffs = coefficients[v];
            coeffs.reinit(this->dim_recovery_basis);
            this->solve_least_squares_problem(v, coeffs);

            // Store the reconstructed solution evaluated at the origin
            if (derivative_order == 0)
            {
              recovered_solution_at_vertices[v] = coeffs[0];
              recovered_gradient_at_vertices[v] =
                this->gradient_at_origin(coeffs);

              if (this->single_reconstruction)
              {
                /**
                 * Evaluate directly the derivatives of order p + 1
                 */
                if (this->degree >= 1)
                  recovered_hessian_at_vertices[v] =
                    this->hessian_at_origin(coeffs);
                if (this->degree >= 2)
                {
                  // FIXME: uncomment when third derivatives are added
                  // recovered_third_derivatives_at_vertices[v] =
                  // this->third_derivatives_at_origin(coeffs);
                }
              }
            }
            else if (derivative_order == 1)
            {
              // The gradient was already stored, by it has been reconstructed
              // with a polynomial of degree p + 1. Update its values with the
              // more accurate values
              recovered_gradient_at_vertices[v][d] = coeffs[0];
              recovered_hessian_at_vertices[v][d] =
                this->gradient_at_origin(coeffs);
            }
            else if (derivative_order == 2)
            {
              DEAL_II_NOT_IMPLEMENTED();
            }
            else
              DEAL_II_NOT_IMPLEMENTED();
          }

          /**
           * Update the local_recovery_solution vector with the computed values.
           * This function assigns the values at dofs and defines the PPR
           * operator per se.
           */
          update_recovery_solution(type, derivative_order, d);
        }
      }
    }

    template <int dim>
    void Scalar<dim>::write_pvtu(const Mapping<dim> &mapping,
                                 const std::string  &filename) const
    {
      std::vector<std::string> data_names(this->n_components);
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_interpretation(this->n_components);

      data_names[solution_offset] = "solution";
      data_interpretation[solution_offset] =
        DataComponentInterpretation::component_is_scalar;

      if (this->highest_recovered_derivative > 0)
        for (unsigned int i = 0; i < gradient_type::n_independent_components;
             ++i)
        {
          data_names[gradient_offset + i] = "gradient";
          data_interpretation[gradient_offset + i] =
            DataComponentInterpretation::component_is_part_of_vector;
        }
      if (this->highest_recovered_derivative > 1)
        for (unsigned int i = 0; i < hessian_type::n_independent_components;
             ++i)
        {
          data_names[hessian_offset + i] = "hessian";
          data_interpretation[hessian_offset + i] =
            DataComponentInterpretation::component_is_part_of_tensor;
        }

      DataOut<dim> data_out;
      data_out.attach_dof_handler(this->dh);
      data_out.add_data_vector(this->recovery_solution,
                               data_names,
                               DataOut<dim>::type_dof_data,
                               data_interpretation);
      data_out.build_patches(mapping, 2);
      data_out.write_vtu_with_pvtu_record(
        "./", filename, 0, this->mpi_communicator, 2);
    }

    template class Base<2>;
    template class Base<3>;
    template class Scalar<2>;
    template class Scalar<3>;
    template class Vector<2>;
    template class Vector<3>;
  } // namespace SolutionRecovery
} // namespace ErrorEstimation
