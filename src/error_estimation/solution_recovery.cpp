
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
                    const Mapping<dim> & /*mapping*/,
                    const ComponentMask &mask)
      : highest_recovered_derivative(highest_recovered_derivative)
      , param(param)
      , patch_handler(patch_handler)
      , patches(patch_handler.patches)
      , dof_handler(dof_handler)
      , fe(fe)
      , mask(mask)
      , isoparam_dh(dof_handler.get_triangulation())
      , mpi_communicator(patch_handler.mpi_communicator)
      , pcout(std::cout,
              (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
      , n_vertices(patch_handler.n_vertices)
      , owned_vertices(patch_handler.owned_vertices)
      , degree(fe.get_sub_fe(mask).degree)
      , least_squares_matrices(patch_handler.get_least_squares_matrices())
      , recoveries_coefficients(n_vertices)
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
      locally_owned_dofs = dof_handler.locally_owned_dofs();
      relevant_dofs      = DoFTools::extract_locally_relevant_dofs(dof_handler);
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
      // Only the vectors actually wanted (the gradients) are allocated.
      gradients_of_recovery_monomials.resize(dim_recovery_basis);
      monomials_recovery->evaluate(Point<dim>(),
                                   empty_polynomial_space_values,
                                   gradients_of_recovery_monomials,
                                   empty_polynomial_space_grad_grads,
                                   empty_polynomial_space_third_derivatives,
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
    void Base<dim>::reconstruct_fields()
    {
      for (unsigned int i = 0; i < highest_recovered_derivative; ++i)
      {
        // If i = 0, then a more accurate solution is fitted.
        // If i > 0, then a more accurate derivative is fitted.
        reconstruct_field(i);
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

      const auto            &tria = isoparam_dh.get_triangulation();
      dealii::Vector<double> cellwise_errors(tria.n_active_cells());

      // Error between recovered solution and exact solution
      VectorTools::integrate_difference(mapping,
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
      VectorTools::interpolate(
        mapping, isoparam_dh, exact_solution, local_nodal_error, *mask);

      // Subtract all the reconstructed fields
      local_nodal_error -= local_isoparam_solution;

      // Interpolate the zero function everywhere except at the required dofs,
      // to overwrite the dofs that are not of the required RecoveryType
      VectorTools::interpolate(mapping,
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
            local_coeffs.push_back({vertices[i], recoveries_coefficients[i]});

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
                        const Mapping<dim>         &mapping,
                        const ComponentMask        &mask)
      : Base<dim>(highest_recovered_derivative,
                  param,
                  patch_handler,
                  dof_handler,
                  solution,
                  fe,
                  mapping,
                  mask)
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
        constexpr unsigned int           solution_offset = 0;
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
        constexpr unsigned int           gradient_offset = 1;
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
        constexpr unsigned int hessian_offset = 1 + dim;
        std::vector<bool> hessian_component_mask(this->n_isoparam_components,
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

      if (Utilities::MPI::this_mpi_process(comm) == 0)
      {
        std::cout << "Reconstructing solution and derivatives of order up to "
                  << highest_recovered_derivative << std::endl;
        std::cout << "Degree of the polynomial solution : " << this->degree
                  << std::endl;
        std::cout << "Fitting polynomials of degree     : " << this->degree + 1
                  << std::endl;
      }

      // Compute the weights associated with the closest dofs on each patch
      compute_patches_averaging_weights();
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
        this->fe.get_sub_fe(this->mask).get_unit_support_points();

      // Weighting of the polynomials evaluations is done using a scalar
      // isoparametric FE, i.e., the base FE associated with the solution. In
      // the vector-valued case this might get trickier, and we'll probably need
      // to add a scalar-valued isoparametric FE, just for this purpose.
      const auto &solution_isoparam_fe = this->isoparam_fe->base_element(0);

      // The patches use dof indices from the main solver's dof handler
      //  Must use this dof handler and the solver's FESystem.
      const unsigned int n_dofs_per_cell = this->fe.n_dofs_per_cell();
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
            AssertThrow(iv != numbers::invalid_unsigned_int,
                        ExcMessage("This vertex is not a vertex of one of its "
                                   "first layer cells..."));

            cell->get_dof_indices(local_dofs);
            for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
            {
              const auto comp_shape   = this->fe.system_to_component_index(i);
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

                AssertThrow(it != patch.neighbours.end(),
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
                AssertIndexRange(iv, solution_isoparam_fe.n_dofs_per_cell());
                const double weight =
                  solution_isoparam_fe.shape_value(iv, ref_coord);
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

    template <int dim>
    void Scalar<dim>::update_local_solution(const unsigned int derivative_order,
                                            const unsigned int component)
    {
      if (derivative_order == 1)
      {
        std::vector<Tensor<1, dim>> basis_gradients(this->dim_recovery_basis);

        // Reset the local copy of the solution
        this->local_solution = 0;

        // The patches use dof indices from the main solver's dof handler
        //  Must use this dof handler and the solver's FESystem.
        const unsigned int n_dofs_per_cell = this->fe.n_dofs_per_cell();
        std::vector<types::global_dof_index> local_dofs(n_dofs_per_cell);

        std::map<types::subdomain_id,
                 std::vector<std::pair<types::global_dof_index, double>>>
          contributions_to_ghosts;

        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
          if (this->owned_vertices[v])
          {
            const auto &patch  = this->patches[v];
            const auto &coeffs = this->recoveries_coefficients[v];

            for (const auto &data : patch.neighbours)
            {
              Point<dim> pt = data.local_pt;

              // Scale back?
              // FIXME: it's probably more robust to keep the coefficients
              // scaled in reconstruct_field(), and to keep the local coord
              // here as well, instead of scaling back both...
              for (unsigned int d = 0; d < dim; ++d)
                pt[d] *= patch.scaling[d];

              if (std::abs(data.averaging_weight) > 1e-14)
              {
                // Evaluate the gradient of the polynomial reconstruction at
                // this dof's support point
                const Tensor<1, dim> grad = this->evaluate_polynomial_gradient(
                  pt, *this->monomials_recovery, coeffs, basis_gradients);

                // Add weighted average
                if (this->locally_owned_dofs.is_element(data.dof))
                  this->local_solution[data.dof] +=
                    data.averaging_weight * grad[component];
                else
                  contributions_to_ghosts[data.owner].push_back(
                    {data.dof, data.averaging_weight * grad[component]});
              }
            }
          }

        // Processes may store contributions that need to be added to ghost
        // dofs, but we cannot increment the local vector at this dof. We need
        // to send this contribution to the dof owner, which will then write
        // into its local part of the solution vector.
        //
        // Each mesh vertex contributes to the gradient of the the first layer
        // of mesh elements only, so at most a single layer of ghost elements in
        // involved. A possible strategy is to send to neighbouring ranks the
        // whole list of ghost dofs involved and their contribution, then the
        // ranks will add the contributions if they own those dofs.
        {
          std::map<unsigned int,
                   std::vector<std::pair<types::global_dof_index, double>>>
            received_contributions =
              Utilities::MPI::some_to_some(this->mpi_communicator,
                                           contributions_to_ghosts);

          for (const auto &[source_rank, contributions_to_add] :
               received_contributions)
            for (const auto &[dof, contribution] : contributions_to_add)
              if (this->locally_owned_dofs.is_element(dof))
                // Increment local solution vector
                this->local_solution[dof] += contribution;
        }

        // Update ghosts
        this->local_solution.compress(VectorOperation::add);
        this->solution_with_additional_ghosts = this->local_solution;
      }
      else if (derivative_order == 2)
      {
        // Update the local solution vector with the current hessian values,
        // which will then be reconstructed.
        DEAL_II_NOT_IMPLEMENTED();
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
    void Scalar<dim>::reconstruct_field(const unsigned int derivative_order)
    {
      const unsigned int n_components_to_recover =
        std::pow(dim, derivative_order);
      dealii::Vector<double> coeffs(this->dim_recovery_basis);

      // Reconstruct each component of the solution, gradient, etc.
      for (unsigned int i_comp = 0; i_comp < n_components_to_recover; ++i_comp)
      {
        if (derivative_order > 0)
        {
          // Update the copy of the solution with the reconstructed derivative
          // of order "derivative_order - 1"
          update_local_solution(derivative_order, i_comp);
        }

        for (types::global_vertex_index v = 0; v < this->n_vertices; ++v)
        {
          if (!this->owned_vertices[v])
            continue;

          const Patch<dim>      &patch  = this->patches[v];
          const auto            &ls_mat = this->least_squares_matrices[v];
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
            coeffs[i] /=
              this->monomials_recovery->compute_value(i, patch.scaling);

          if (derivative_order == 0)
          {
            // Store reconstructed solution evaluated at the origin
            this->recoveries_coefficients[v]  = coeffs;
            recovered_solution_at_vertices[v] = coeffs[0];

            // Compute the gradient at the origin = sum_i coeffs_i *
            // grad(P_i)(0)
            Tensor<1, dim> grad;
            for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
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
            for (unsigned int i = 0; i < this->dim_recovery_basis; ++i)
              grad += coeffs[i] * this->gradients_of_recovery_monomials[i];
            for (unsigned int d = 0; d < dim; ++d)
              recovered_hessian_at_vertices[v][i_comp][d] = grad[d];
          }
        }
      }

      if (derivative_order == 0)
      {
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
    void Scalar<dim>::write_pvtu(const Mapping<dim> &mapping,
                                 const std::string  &filename) const
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
      data_out.build_patches(mapping, 2);
      data_out.write_vtu_with_pvtu_record(
        "./", filename, 0, this->isoparam_dh.get_mpi_communicator(), 2);
    }

    template class Base<2>;
    template class Base<3>;
    template class Scalar<2>;
    template class Scalar<3>;
    template class Vector<2>;
    template class Vector<3>;
  } // namespace SolutionRecovery
} // namespace ErrorEstimation
