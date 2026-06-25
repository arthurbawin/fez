#ifndef SCRATCH_DATA_H
#define SCRATCH_DATA_H

#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <fsi_exact_solution.h>
#include <parameter_reader.h>
#include <stabilization_tools.h>
#include <time_handler.h>
#include <types.h>

/**
 * A namespace for the ScratchData used in the Navier-Stokes derived solvers.
 *
 * ScratchDatas are helper temporary structures containing all sorts of
 * quantities that must be computed on each elements and faces during
 * assembly, e.g., the current solution and its gradients interpolated at the
 * quadrature nodes, the shape functions and their derivatives, among many
 * others. See also the deal.II doc of meshworker/scratch_data.h and the run()
 * functions in work_stream.h. A ScratchData is thus essentially a collection
 * of FEValues and FEFaceValues, together with vectors of data used for
 * assembly.
 */
namespace NavierStokesScratch
{
  using namespace dealii;

  /**
   * These flags are used to specify which updates are needed when calling the
   * reinit function of the ScratchData template below. These are similar to
   * deal.II's UpdateFlags, which are used to reinit FEValues and such, and they
   * are used in the same way.
   *
   * For instance, ScratchData<dim, pseudo_solid | cahn_hilliard> specifies that
   * both the pseudo-solid and Cahn-Hilliard related data (physical properties,
   * shape functions, current values and gradients, etc.) should be computed for
   * this scratch, in addition to the data related to the incompressible Navier-
   * Stokes system, which are always computed. This combinations of flags is
   * used for the incompressible CHNS solver with mesh movement, for instance.
   *
   * Enabling SUPG/PSPG stabilization is for now not a template flag, but a run
   * time option passed to the constructor of the ScratchData. This allows
   * enabling/disabling stabilization without refactoring either the scratch
   * base class, or the solvers (although such a refactor will probably happen
   * to reduce the number of assemblers instantiations needed).
   *
   * To add another value, one simply needs to add a bitshift to the list. The
   * limit is the number of bits in the representation of an unsigned int (32
   * or 64), which should be plenty (-:
   */
  enum ScratchFlags : unsigned int
  {
    /**
     * Reinit data required to assemble the incompressible Navier-Stokes
     * system. The gradients needed for both the laplacian form and the
     * divergence form are computed (velocity gradients and symmetric
     * gradients).
     *
     * This is always active and does not need to be specified.
     */
    ns_only = 0,

    /**
     * Reinit data to assemble the mesh movement elasticity equation.
     */
    pseudo_solid = 1 << 0,

    /**
     * Reinit data to assemble constraints using a Lagrange multiplier.
     */
    lagrange_multiplier = 1 << 1,

    /**
     * Reinit data to assemble the Cahn-Hilliard system, used to form the CHNS
     * system.
     */
    cahn_hilliard = 1 << 2,

    /**
     * Reinit data to assemble the compressible Navier-Stokes system (including
     * the energy equation).
     */
    compressible = 1 << 3,

    /**
     * Specifies if this ScratchData is for a solver using the hp tools.
     * In that case, hp::FEValues/FEFaceValues are stored and the appropriate
     * FEValues/FEFaceValues are selected on the current cell when the scratch
     * is reinit'ed.
     *
     * Not an update flag per se, but avoids defining an extra template
     * parameter.
     */
    with_hp_capabilities = 1 << 4
  };

  // Forward declaration of base class below
  template <int dim, unsigned int update_flags = ns_only>
  class ScratchData;

  /**
   * Scratch data for the incompressible NS solver on fixed mesh.
   */
  template <int dim>
  using ScratchDataIncompressibleNS = ScratchData<dim>;

  /**
   * Scratch data for the compressible NS solver.
   */
  template <int dim>
  using ScratchDataCompressibleNS = ScratchData<dim, compressible>;

  /**
   * hp Scratch data for the incompressible NS solver with Lagrange multiplier.
   */
  template <int dim>
  using ScratchDataIncompressibleNSLambda =
    ScratchData<dim, lagrange_multiplier | with_hp_capabilities>;

  /**
   * Scratch data for the FSI solver on moving mesh.
   */
  template <int dim>
  using ScratchDataFSI = ScratchData<dim, pseudo_solid | lagrange_multiplier>;

  /**
   * hp Scratch data for the FSI solver.
   */
  template <int dim>
  using ScratchDataFSI_hp =
    ScratchData<dim, pseudo_solid | lagrange_multiplier | with_hp_capabilities>;

  /**
   * Scratch data for the quasi-incompressible Cahn_hilliard Navier-Stokes
   * solver on fixed mesh.
   */
  template <int dim, bool with_moving_mesh = false>
  using ScratchDataCHNS =
    ScratchData<dim,
                (with_moving_mesh ? pseudo_solid : ns_only) | cahn_hilliard>;

  /**
   * This base class is a ScratchData common to all solvers. The pre-computation
   * of specific fields is enabled or disabled using the update flags defined
   * above. By default, the Navier-Stokes related quantities are computed for
   * all solvers. They are computed at the quadrature nodes located inside the
   * mesh elements, and at those located on their faces if the element touches a
   * boundary.
   *
   * This scratch has FEValues/FEFaceValues defined on both the fixed and moving
   * mesh: only quantities used to assemble the pseudo-solid equation are
   * evaluated on the fixed mesh, whereas *all* other fields are evaluated on
   * the moving mesh. Of course, when the passed mapping representing the moving
   * mesh is also the fixed mapping, then everything is evaluated on a unique
   * fixed mesh.
   */
  template <int dim, unsigned int update_flags>
  class ScratchData
  {
  public:
    /**
     * Constructor
     */
    ScratchData(const ComponentOrdering    &ordering,
                const FESystem<dim>        &fe,
                const Mapping<dim>         &fixed_mapping,
                const Mapping<dim>         &moving_mapping,
                const Quadrature<dim>      &cell_quadrature,
                const Quadrature<dim - 1>  &face_quadrature,
                const TimeHandler          &time_handler,
                const ParameterReader<dim> &param);

    /**
     * Constructor with hp capabilities
     */
    ScratchData(const ComponentOrdering          &ordering,
                const hp::FECollection<dim>      &fe_collection,
                const hp::MappingCollection<dim> &fixed_mapping_collection,
                const hp::MappingCollection<dim> &moving_mapping_collection,
                const hp::QCollection<dim>       &cell_quadrature_collection,
                const hp::QCollection<dim - 1>   &face_quadrature_collection,
                const TimeHandler                &time_handler,
                const ParameterReader<dim>       &param);

    /**
     * Copy constructor
     */
    ScratchData(const ScratchData &other);

  private:
    /**
     * Allocate the class vectors.
     */
    void allocate();

    /**
     * Set up data related to the incompressible Navier-Stokes system.
     */
    void initialize_navier_stokes();

    /**
     * Set up data related to the mesh movement elasticity equations.
     */
    void initialize_pseudo_solid();

    /**
     * Set up data related to the Lagrange multiplier fields.
     */
    void initialize_lagrange_multiplier();

    /**
     * Set up data related to the Cahn-Hilliard system.
     */
    void initialize_cahn_hilliard();

    /**
     * Set up data related to the compressible Navier-Stokes system.
     */
    void initialize_compressible();

    /**
     * Reinit FEValues or FEFaceValues and return a reference to it.
     * Adapted from the reinit routines in deal.II's meshworker/scratch_data.cc,
     * for the hp case and with spacedim = dim.
     */
    const FEValues<dim> *
    reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
           const bool                                            fixed_mapping);

    const FEFaceValues<dim> *
    reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
           const unsigned int                                    face_no,
           const bool                                            fixed_mapping);

    template <typename VectorType>
    void
    reinit_navier_stokes_cell(const FEValues<dim>           &fe_values,
                              const VectorType              &current_solution,
                              const std::vector<VectorType> &previous_solutions,
                              const Function<dim>           &source_terms,
                              const Function<dim> & /*exact_solution*/)
    {
      fe_values[velocity].get_function_values(current_solution,
                                              present_velocity_values);
      fe_values[velocity].get_function_gradients(current_solution,
                                                 present_velocity_gradients);
      fe_values[velocity].get_function_symmetric_gradients(
        current_solution, present_velocity_sym_gradients);
      fe_values[velocity].get_function_divergences(current_solution,
                                                   present_velocity_divergence);
      fe_values[pressure].get_function_values(current_solution,
                                              present_pressure_values);
      if (enable_stabilization)
      {
        fe_values[velocity].get_function_hessians(current_solution,
                                                  present_velocity_hessians);
        fe_values[pressure].get_function_gradients(current_solution,
                                                   present_pressure_gradients);

        // Compute the velocity laplacian and grad(div) from the hessians
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          present_velocity_laplacians[q] = Tensor<1, dim>();
          present_velocity_grad_div[q]   = Tensor<1, dim>();
          for (unsigned int c = 0; c < dim; ++c)
            for (unsigned int d = 0; d < dim; ++d)
            {
              present_velocity_laplacians[q][c] +=
                present_velocity_hessians[q][c][d][d];
              present_velocity_grad_div[q][d] +=
                present_velocity_hessians[q][c][c][d];
            }
        }
      }

      // Previous solutions
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
        fe_values[velocity].get_function_values(previous_solutions[i],
                                                previous_velocity_values[i]);

      // Source terms with layout u-v-(w)-p
      source_terms.vector_value_list(fe_values.get_quadrature_points(),
                                     source_term_full_moving);

      // Get jacobian, shape functions and set source terms
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW_moving[q] = fe_values.JxW(q);

        // Time derivatives
        present_velocity_time_derivatives[q] =
          time_handler.compute_time_derivative_at_quadrature_node(
            q, present_velocity_values[q], previous_velocity_values);

        for (int d = 0; d < dim; ++d)
          source_term_velocity[q][d] = source_term_full_moving[q](u_lower + d);
        source_term_pressure[q] = source_term_full_moving[q](p_lower);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]          = fe_values[velocity].value(k, q);
          grad_phi_u[q][k]     = fe_values[velocity].gradient(k, q);
          sym_grad_phi_u[q][k] = symmetrize(grad_phi_u[q][k]);
          div_phi_u[q][k]      = fe_values[velocity].divergence(k, q);
          phi_p[q][k]          = fe_values[pressure].value(k, q);

          if (enable_stabilization)
          {
            grad_phi_p[q][k] = fe_values[pressure].gradient(k, q);

            auto &lap      = laplacian_phi_u[q][k];
            auto &grad_div = grad_div_phi_u[q][k];

            // (Δφ_u)_c     = Σ_d Hk[c][d][d]   (trace on last two indices)
            // (∇div φ_u)_d = Σ_c Hk[c][c][d]   (trace on first two indices)
            const Tensor<3, dim> hessian_phi_u =
              fe_values[velocity].hessian(k, q);
            lap      = 0;
            grad_div = 0;
            for (unsigned int c = 0; c < dim; ++c)
              for (unsigned int d = 0; d < dim; ++d)
              {
                lap[c] += hessian_phi_u[c][d][d];
                grad_div[d] += hessian_phi_u[c][c][d];
              }

            // Get the gradient of the shape functions for the x-component of
            // the velocity to compute the cell length for stabilization. If
            // using Lagrange shape functions, we can take any component since
            // the space is juste d copies of the scalar shape functions.
            if constexpr (running_in_debug_mode())
            {
              // FIXME: this assumes velocity is always the first base element!
              const auto &velocity_fe = fe_values.get_fe().base_element(0);
              Assert(
                velocity_fe.is_primitive(),
                ExcMessage(
                  "The computation of the cell length for SUPG stabilization "
                  "of the Navier-Stokes equation uses the gradient of the "
                  "velocity shape functions. Since scalar-valued shape "
                  "functions are required, this is done by taking the gradient "
                  "of the scalar-valued shape functions of the first velocity "
                  "component, and this thus assumes that the velocity finite "
                  "element space can be split into meaningful scalar "
                  "components (i.e., that it is \"primitive\" in deal.II "
                  "terms. Actually, by choosing an *arbitrary* velocity "
                  "component (here, the first one), we even assume that the FE "
                  "space is Lagrange and consists of identical copies of the "
                  "scalar Lagrange space for each velocity component). "
                  "However, the velocity space used is *not* primitive, and "
                  "thus this operation does not make sense."));
            }
            grad_phi_u_first_component[k] = grad_phi_u[q][k][0];
          }
        }

        // When the Cahn-Hilliard part is enabled, u_conv and tau are instead
        // computed in reinit_cahn_hilliard, where the kinematic viscosity
        // (which depends on the density, and thus on the tracer) is known.
        if constexpr (!enable_cahn_hilliard)
          if (enable_stabilization)
          {
            // Compute stabilization parameter tau.
            // Mesh velocity has already been computed, so ALE velocity is well
            // defined.
            auto u_conv = present_velocity_values[q];
            if constexpr (enable_pseudo_solid)
              u_conv -= present_mesh_velocity_values[q];

            tau_supg_velocity[q] = StabilizationTools::compute_tau_supg(
              time_handler,
              dofs_per_cell,
              cell_diameter,
              param.finite_elements.velocity_degree,
              kinematic_viscosity,
              u_conv,
              grad_phi_u_first_component);
          }
      }
    }

    template <typename VectorType>
    void reinit_navier_stokes_face(
      const unsigned int       i_face,
      const FEFaceValues<dim> &fe_face_values,
      const VectorType        &current_solution,
      const std::vector<VectorType> & /*previous_solutions*/,
      const Function<dim> & /*source_terms*/,
      const Function<dim> &exact_solution)
    {
      fe_face_values[velocity].get_function_values(
        current_solution, present_face_velocity_values[i_face]);

      fe_face_values[velocity].get_function_gradients(
        current_solution, present_face_velocity_gradients[i_face]);

      // Exact solution with layout u-v-(w-)p and its gradient
      exact_solution.vector_value_list(fe_face_values.get_quadrature_points(),
                                       exact_solution_full);
      exact_solution.vector_gradient_list(
        fe_face_values.get_quadrature_points(), grad_exact_solution_full);

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        face_JxW_moving[i_face][q]     = fe_face_values.JxW(q);
        face_normals_moving[i_face][q] = fe_face_values.normal_vector(q);

        present_face_velocity_sym_gradients[i_face][q] =
          symmetrize(present_face_velocity_gradients[i_face][q]);

        for (int di = 0; di < dim; ++di)
          for (int dj = 0; dj < dim; ++dj)
            exact_face_velocity_gradients[i_face][q][di][dj] =
              grad_exact_solution_full[q][u_lower + di][dj];
        exact_face_velocity_divergences[i_face][q] =
          trace(exact_face_velocity_gradients[i_face][q]);
        exact_face_pressure_values[i_face][q] = exact_solution_full[q](p_lower);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u_face[i_face][q][k] = fe_face_values[velocity].value(k, q);
          phi_p_face[i_face][q][k] = fe_face_values[pressure].value(k, q);
          grad_phi_u_face[i_face][q][k] =
            fe_face_values[velocity].gradient(k, q);
          sym_grad_phi_u_face[i_face][q][k] =
            symmetrize(grad_phi_u_face[i_face][q][k]);
          div_phi_u_face[i_face][q][k] =
            fe_face_values[velocity].divergence(k, q);
        }
      }
    }

    template <typename VectorType>
    void
    reinit_compressible_cell(const FEValues<dim>           &fe_values,
                             const VectorType              &current_solution,
                             const std::vector<VectorType> &previous_solutions,
                             const Function<dim> & /*source_terms*/,
                             const Function<dim> &exact_solution)
    {
      fe_values[temperature].get_function_values(current_solution,
                                                 present_temperature_values);
      fe_values[temperature].get_function_gradients(
        current_solution, present_temperature_gradients);
      fe_values[pressure].get_function_values(current_solution,
                                              present_pressure_values);
      fe_values[pressure].get_function_gradients(current_solution,
                                                 present_pressure_gradients);

      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_values[pressure].get_function_values(previous_solutions[i],
                                                previous_pressure_values[i]);
        fe_values[temperature].get_function_values(
          previous_solutions[i], previous_temperature_values[i]);
      }

      // Exact solution at cell quadrature points (layout u-v-(w-)p-T)
      exact_solution.vector_value_list(fe_values.get_quadrature_points(),
                                       exact_solution_full_cell);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          exact_velocity_values_cell[q][d] =
            exact_solution_full_cell[q](u_lower + d);
        exact_pressure_values_cell[q]    = exact_solution_full_cell[q](p_lower);
        exact_temperature_values_cell[q] = exact_solution_full_cell[q](t_lower);
      }

      // Get jacobian, shape functions and set source terms
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        source_term_temperature[q] = source_term_full_moving[q](t_lower);

        const double p_star = present_pressure_values[q];
        const double T_star = present_temperature_values[q];

        // Total pressure p = p_ref + p_star, and similarly for temperature
        present_pressure_absolute_values[q]    = pressure_ref + p_star;
        present_temperature_absolute_values[q] = temperature_ref + T_star;

        a_p[q] = alpha_r / (alpha_r * p_star + 1.0);
        b_T[q] = beta_r / (beta_r * T_star + 1.0);

        density[q] =
          density_ref * ((alpha_r * p_star + 1.0) / (beta_r * T_star + 1.0));

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          grad_phi_p[q][k] = fe_values[pressure].gradient(k, q);
          phi_T[q][k]      = fe_values[temperature].value(k, q);
          grad_phi_T[q][k] = fe_values[temperature].gradient(k, q);
        }
      }
    }

    template <typename VectorType>
    void reinit_compressible_face(
      const unsigned int       i_face,
      const FEFaceValues<dim> &fe_face_values,
      const VectorType        &current_solution,
      const std::vector<VectorType> & /*previous_solutions*/,
      const Function<dim> & /*source_terms*/,
      const Function<dim> &exact_solution)
    {
      fe_face_values[pressure].get_function_values(
        current_solution, present_face_pressure_values[i_face]);

      fe_face_values[temperature].get_function_values(
        current_solution, present_face_temperature_values[i_face]);

      fe_face_values[temperature].get_function_gradients(
        current_solution, present_face_temperature_gradients[i_face]);

      exact_solution.vector_gradient_list(
        fe_face_values.get_quadrature_points(), grad_exact_solution_full);

      const auto &quad_points = fe_face_values.get_quadrature_points();

      // Fluid and heat boundary condition on this face
      // In the compressible solver, both fluid and heat boundary conditions are
      // expected to be defined on all boundaries.
      Assert(param.fluid_bc.count(face_boundary_id[i_face]) > 0,
             ExcInternalError());
      Assert(param.heat_bc.count(face_boundary_id[i_face]) > 0,
             ExcInternalError());
      const auto &fluid_bc = param.fluid_bc.at(face_boundary_id[i_face]);
      const auto &bc_heat  = param.heat_bc.at(face_boundary_id[i_face]);

      // Imposed pressure if pressure is weakly enforced
      if (fluid_bc.type == BoundaryConditions::Type::weak_pressure)
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
          face_input_pressure_values[i_face][q] =
            fluid_bc.p->value(quad_points[q]);

      // Imposed heat flux
      if (bc_heat.type == BoundaryConditions::Type::heat_flux)
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
          face_input_heat_flux_values[i_face][q] =
            bc_heat.temperature->value(quad_points[q]);

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        present_face_velocity_divergence[i_face][q] =
          trace(present_face_velocity_gradients[i_face][q]);

        present_face_temperature_absolute_values[i_face][q] =
          temperature_ref + present_face_temperature_values[i_face][q];

        exact_face_temperature_gradients[i_face][q] =
          grad_exact_solution_full[q][t_lower];

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_T_face[i_face][q][k] = fe_face_values[temperature].value(k, q);
          grad_phi_T_face[i_face][q][k] =
            fe_face_values[temperature].gradient(k, q);
        }
      }
    }

    template <typename VectorType>
    void
    reinit_pseudo_solid_cell(const FEValues<dim>           &fe_values_fixed,
                             const FEValues<dim>           &fe_values_moving,
                             const VectorType              &current_solution,
                             const std::vector<VectorType> &previous_solutions,
                             const Function<dim>           &source_terms,
                             const Function<dim> & /*exact_solution*/)
    {
      fe_values_fixed[position].get_function_values(current_solution,
                                                    present_position_values);
      fe_values_fixed[position].get_function_gradients(
        current_solution, present_position_gradients);

      // Previous solutions
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_values_fixed[position].get_function_values(
          previous_solutions[i], previous_position_values[i]);
      }

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Compute mesh velocity from mesh position
        present_mesh_velocity_values[q] =
          time_handler.compute_time_derivative_at_quadrature_node(
            q, present_position_values[q], previous_position_values);
      }

      const auto &fixed_quadrature_points =
        fe_values_fixed.get_quadrature_points();

      // Source terms on fixed mapping for x
      source_terms.vector_value_list(fixed_quadrature_points,
                                     source_term_full_fixed);

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
      /**
       * The gradient of the source terms contributes to the Jacobian matrix
       * when the mesh is moving. Currently, the solvers that need it implement
       * this gradient using finite differences, which is quite slow.
       * It is typically required for convergence studies with manufactured
       * solutions, for which the source term is non uniform.
       *
       * For most other simulations, the source term is usually constant, so the
       * computation of the gradient is disabled by default. The main
       * consequence is that this slows down the convergence of the MMS tests.
       *
       * A solution would be to implement the analytic source term gradient for
       * MMS (which requires adding a few exact derivatives functions to the
       * MMSFunction class), and compute the symbolic gradient of the parsed
       * source term.
       */
      source_terms.vector_gradient_list(
        fe_values_moving.get_quadrature_points(), grad_source_term_full);
#endif

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW_fixed[q] = fe_values_fixed.JxW(q);

        const Point<dim> &q_point = fixed_quadrature_points[q];
        lame_mu[q] =
          physical_properties.pseudosolids[0].lame_mu_fun->value(q_point);
        lame_lambda[q] =
          physical_properties.pseudosolids[0].lame_lambda_fun->value(q_point);

        AssertThrow(lame_mu[q] >= 0,
                    ExcMessage("Lamé coefficient mu should be positive"));

        // Data for hyperelastic models
        const Tensor<2, dim> &F               = present_position_gradients[q];
        present_position_J[q]                 = determinant(F);
        present_position_inverse_gradients[q] = invert(F);
        present_position_inverse_gradients_T[q] =
          transpose(present_position_inverse_gradients[q]);

        for (int d = 0; d < dim; ++d)
          source_term_position[q][d] = source_term_full_fixed[q](x_lower + d);

#if defined(WITH_GRADIENT_OF_SOURCE_TERMS)
        // Fill the gradients of the source term
        // Layout: grad_source_velocity[q] = df_i/dx_j
        for (int di = 0; di < dim; ++di)
        {
          grad_source_pressure[q][di] = grad_source_term_full[q][p_lower][di];
          for (int dj = 0; dj < dim; ++dj)
            grad_source_velocity[q][di][dj] =
              grad_source_term_full[q][u_lower + di][dj];

          if constexpr (enable_cahn_hilliard)
          {
            grad_source_tracer[q][di] = grad_source_term_full[q][phi_lower][di];
            grad_source_potential[q][di] =
              grad_source_term_full[q][mu_lower][di];
          }
        }
#endif

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_x[q][k]             = fe_values_fixed[position].value(k, q);
          grad_phi_x[q][k]        = fe_values_fixed[position].gradient(k, q);
          sym_grad_phi_x[q][k]    = symmetrize(grad_phi_x[q][k]);
          trace_grad_phi_x[q][k]  = trace(grad_phi_x[q][k]);
          div_phi_x[q][k]         = fe_values_fixed[position].divergence(k, q);
          grad_phi_x_moving[q][k] = fe_values_moving[position].gradient(k, q);
          if (enable_stabilization || enable_tracer_stabilization)
            hessian_phi_x_moving[q][k] =
              fe_values_moving[position].hessian(k, q);
        }
      }
    }

    template <typename VectorType>
    void
    reinit_pseudo_solid_face(const unsigned int       i_face,
                             const FEFaceValues<dim> &fe_face_values_fixed,
                             const FEFaceValues<dim> &fe_face_values,
                             const VectorType        &current_solution,
                             const std::vector<VectorType> &previous_solutions,
                             const Function<dim> & /*source_terms*/,
                             const Function<dim> & /*exact_solution*/)
    {
      fe_face_values_fixed[position].get_function_values(
        current_solution, present_face_position_values[i_face]);
      fe_face_values_fixed[position].get_function_gradients(
        current_solution, present_face_position_gradient[i_face]);

      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_face_values_fixed[position].get_function_values(
          previous_solutions[i], previous_face_position_values[i_face][i]);
      }

      const auto &bdf_coefficients = time_handler.get_bdf_coefficients();

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        face_JxW_fixed[i_face][q] = fe_face_values_fixed.JxW(q);

        //
        // Jacobian of geometric transformation is needed to compute
        // tangent matrix for the lambda equation, on moving mapping
        //
        const Tensor<2, dim> J = fe_face_values.jacobian(q);

        if constexpr (dim == 2)
        {
          if (use_quads)
            switch (i_face)
            {
              case 0:
                dxsids_array[0][0] = 0.;
                dxsids_array[0][1] = 2.;
                break;
              case 1:
                dxsids_array[0][0] = 0.;
                dxsids_array[0][1] = 2.;
                break;
              case 2:
                dxsids_array[0][0] = 2.;
                dxsids_array[0][1] = 0.;
                break;
              case 3:
                dxsids_array[0][0] = 2.;
                dxsids_array[0][1] = 0.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          else
            switch (i_face)
            {
              case 0:
                dxsids_array[0][0] = 1.;
                dxsids_array[0][1] = 0.;
                break;
              case 1:
                dxsids_array[0][0] = -1.;
                dxsids_array[0][1] = 1.;
                break;
              case 2:
                dxsids_array[0][0] = 0.;
                dxsids_array[0][1] = -1.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
        }
        else
        {
          if (use_quads)
            // TODO!
            DEAL_II_NOT_IMPLEMENTED();
          else
            switch (i_face)
            {
              // Using dealii's face ordering
              case 3: // Opposite to v0
                dxsids_array[0][0] = -1.;
                dxsids_array[1][0] = -1.;
                dxsids_array[0][1] = 1.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 2: // Opposite to v1
                dxsids_array[0][0] = 0.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 1.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 1: // Opposite to v2
                dxsids_array[0][0] = 1.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 0: // Opposite to v3
                dxsids_array[0][0] = 1.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.;
                dxsids_array[1][1] = 1.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 0.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
        }

        Tensor<2, dim - 1> G;
        G = 0;
        for (unsigned int di = 0; di < dim - 1; ++di)
          for (unsigned int dj = 0; dj < dim - 1; ++dj)
            for (unsigned int im = 0; im < dim; ++im)
              for (unsigned int in = 0; in < dim; ++in)
                for (unsigned int ip = 0; ip < dim; ++ip)
                  G[di][dj] += dxsids_array[di][im] * J[in][im] * J[in][ip] *
                               dxsids_array[dj][ip];
        const Tensor<2, dim - 1> G_inverse = invert(G);

        // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
        Tensor<2, dim - 1> res;

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_x_face[i_face][q][k] = fe_face_values_fixed[position].value(k, q);

          // FIXME: this might be the gradient obtained from
          // fe_face_values (moving)
          grad_phi_x_face[i_face][q][k] =
            fe_face_values_fixed[position].gradient(k, q);

          const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

          Tensor<2, dim> A =
            transpose(J) *
            (transpose(present_face_position_gradient[i_face][q]) * grad_phi_x +
             transpose(grad_phi_x) *
               present_face_position_gradient[i_face][q]) *
            J;

          res = 0;
          for (unsigned int di = 0; di < dim - 1; ++di)
            for (unsigned int dj = 0; dj < dim - 1; ++dj)
              for (unsigned int im = 0; im < dim - 1; ++im)
                for (unsigned int in = 0; in < dim; ++in)
                  for (unsigned int io = 0; io < dim; ++io)
                    res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] *
                                   A[in][io] * dxsids_array[dj][io];

          // Choose this delta_dx if multiplying by JxW in the matrix
          delta_dx[i_face][q][k] = 0.5 * trace(res);

          // If instead the matrix term is multiplied only by the weight,
          // use this delta_dx (this is the actual delta_dx, but the other
          // is used to multiply only once the whole local matrix).
          // delta_dx[i_face][q][k] = 0.5 * sqrt_det_G * trace(res);
        }

        // Face mesh velocity
        present_face_mesh_velocity_values[i_face][q] =
          bdf_coefficients[0] * present_face_position_values[i_face][q];
        for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
        {
          present_face_mesh_velocity_values[i_face][q] +=
            bdf_coefficients[iBDF] *
            previous_face_position_values[i_face][iBDF - 1][q];
        }
      }
    }

    template <typename VectorType>
    void reinit_lagrange_multiplier_face(
      const unsigned int       i_face,
      const FEFaceValues<dim> &fe_face_values_fixed,
      const FEFaceValues<dim> &fe_face_values,
      const VectorType        &current_solution,
      const std::vector<VectorType> & /*previous_solutions*/,
      const Function<dim> & /*source_terms*/,
      const Function<dim> &exact_solution)
    {
      fe_face_values[lambda].get_function_values(
        current_solution, present_face_lambda_values[i_face]);

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
      /**
       * Compute the data required to add source terms for the momentum
       * and Lagrange multiplier equation.
       */
      const auto &quadrature_points = fe_face_values.get_quadrature_points();

      // Get exact velocity on face
      exact_solution.vector_value_list(quadrature_points, exact_solution_full);
      exact_solution.vector_gradient_list(quadrature_points,
                                          grad_exact_solution_full);

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        exact_face_pressure_values[i_face][q] = exact_solution_full[q][p_lower];
        for (int d = 0; d < dim; ++d)
        {
          exact_face_velocity_values[i_face][q][d] =
            exact_solution_full[q][u_lower + d];
          exact_face_lambda_values[i_face][q][d] =
            exact_solution_full[q][l_lower + d];
          for (int dj = 0; dj < dim; ++dj)
            exact_face_velocity_gradients[i_face][q][d][dj] =
              grad_exact_solution_full[q][u_lower + d][dj];
        }

        Tensor<2, dim> sigma;
        for (unsigned int d = 0; d < dim; ++d)
          sigma[d][d] = -exact_face_pressure_values[i_face][q];
        const Tensor<2, dim> &grad_u = exact_face_velocity_gradients[i_face][q];
        const double          nu     = kinematic_viscosity;
        sigma += nu * (grad_u + transpose(grad_u));
        const auto &normal_to_solid = -fe_face_values.normal_vector(q);
        const auto  stress_vector   = -sigma * normal_to_solid;

        face_velocity_source_term[i_face][q] =
          exact_face_lambda_values[i_face][q] - stress_vector;
      }

      if constexpr (enable_pseudo_solid)
      {
        // If using an FSI solver, get the exact mesh velocity
        // FIXME: The exact_solution should be an MMSFunction, which
        // has a time_derivative function.
        const FSIExactSolution<dim> *sol = nullptr;
        if (dynamic_cast<const FSIExactSolution<dim> *>(&exact_solution) !=
            nullptr)
          sol = dynamic_cast<const FSIExactSolution<dim> *>(&exact_solution);
        if (sol != nullptr)
        {
          const auto &fixed_quadrature_points =
            fe_face_values_fixed.get_quadrature_points();

          // Compute time derivative of mesh position
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const auto &qpoint = fixed_quadrature_points[q];
            for (int d = 0; d < dim; ++d)
              exact_face_mesh_velocity_values[i_face][q][d] =
                sol->time_derivative(qpoint, x_lower + d);
          }
        }
      }
#else
      (void)exact_solution;
#endif

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          phi_l_face[i_face][q][k] = fe_face_values[lambda].value(k, q);

      // Rigid-body rotation
      const auto &fluid_bc = param.fluid_bc.at(face_boundary_id[i_face]);
      if (fluid_bc.enable_rigid_body_rotation)
      {
        const Point<dim> &center = fluid_bc.center_of_rotation;

        /**
         * Rigid-body rotation can be applied on fixed or moving mesh: if
         * enable_pseudo_solid is true, then the rotation velocity is evaluated
         * on the fixed mesh, with constant center of rotation. If false, then
         * rotation velocity is evaluated on "moving" mesh, but the model does
         * not solve for the pseudosolid and thus does not move the mesh, and
         * the center of rotation is also constant.
         *
         * This allows keeping a constant center of rotation for both cases.
         *
         * The TL;DR is that "moving" refers to the current configuration, and
         * is a misnomer for solvers which do not solve for the mesh position,
         * as in that case the "moving" mesh remains fixed for the whole
         * simulation...
         */
        const auto &qpoints = enable_pseudo_solid ?
                                fe_face_values_fixed.get_quadrature_points() :
                                fe_face_values.get_quadrature_points();

        Tensor<1, dim>        pos_vector;
        angular_velocity_type omega;
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          pos_vector = qpoints[q] - center;

          // Cartesian velocity on fixed mesh is omega \times (X - Xc)
          if constexpr (dim == 2)
          {
            omega = fluid_bc.angular_velocity->value(qpoints[q]);

            // cross_product_2d returns the product with (0, 0, 1)
            input_face_rigid_body_rotation_velocity[i_face][q] =
              -omega * cross_product_2d(pos_vector);
          }
          else
          {
            for (unsigned int d = 0; d < dim; ++d)
              omega[d] = fluid_bc.angular_velocity->value(qpoints[q], d);
            input_face_rigid_body_rotation_velocity[i_face][q] =
              cross_product_3d(omega, pos_vector);
          }
        }
      }
    }

    template <typename VectorType>
    void
    reinit_cahn_hilliard_cell(const FEValues<dim>           &fe_values_fixed,
                              const FEValues<dim>           &fe_values_moving,
                              const VectorType              &current_solution,
                              const std::vector<VectorType> &previous_solutions,
                              const Function<dim>           &source_terms,
                              const Function<dim> & /*exact_solution*/)
    {
      fe_values_moving[tracer].get_function_values(current_solution,
                                                   tracer_values);
      fe_values_moving[tracer].get_function_gradients(current_solution,
                                                      tracer_gradients);

      if constexpr (enable_pseudo_solid)
      {
        fe_values_fixed[tracer].get_function_values(current_solution,
                                                    tracer_values_fixed);
        fe_values_fixed[tracer].get_function_gradients(current_solution,
                                                       tracer_gradients_fixed);
      }

      fe_values_moving[potential].get_function_values(current_solution,
                                                      potential_values);
      fe_values_moving[potential].get_function_gradients(current_solution,
                                                         potential_gradients);
      if (enable_tracer_stabilization)
      {
        fe_values_moving[potential].get_function_laplacians(
          current_solution, potential_laplacians);
        if constexpr (enable_pseudo_solid)
          // The moving-mesh x-variation of the tracer SUPG residual needs the
          // full potential hessian, not only its laplacian.
          fe_values_moving[potential].get_function_hessians(current_solution,
                                                            potential_hessians);
      }
      // Previous solutions
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
        fe_values_moving[tracer].get_function_values(previous_solutions[i],
                                                     previous_tracer_values[i]);

      source_terms.vector_value_list(fe_values_moving.get_quadrature_points(),
                                     source_term_full_moving);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Time derivatives
        tracer_time_derivatives[q] =
          time_handler.compute_time_derivative_at_quadrature_node(
            q, tracer_values[q], previous_tracer_values);

        // Physical properties based on tracer, filter if applicable
        const double filtered_phi = tracer_limiter(tracer_values[q]);
        density[q] =
          CahnHilliard::linear_mixing(filtered_phi, density0, density1);
        dynamic_viscosity[q] = CahnHilliard::linear_mixing(filtered_phi,
                                                           dynamic_viscosity0,
                                                           dynamic_viscosity1);
        derivative_density_wrt_tracer[q] =
          CahnHilliard::linear_mixing_derivative(filtered_phi,
                                                 density0,
                                                 density1);
        derivative_dynamic_viscosity_wrt_tracer[q] =
          CahnHilliard::linear_mixing_derivative(filtered_phi,
                                                 dynamic_viscosity0,
                                                 dynamic_viscosity1);

        source_term_tracer[q]    = source_term_full_moving[q](phi_lower);
        source_term_potential[q] = source_term_full_moving[q](mu_lower);

        diffusive_flux[q] = diffusive_flux_factor *
                            present_velocity_gradients[q] *
                            potential_gradients[q];

        Tensor<1, dim> u_conv = present_velocity_values[q];
        if constexpr (enable_pseudo_solid)
          u_conv -= present_mesh_velocity_values[q];
        if (enable_stabilization)
        {
          Assert(density[q] > 0.,
                 ExcMessage("The density must be strictly positive to compute "
                            "the kinematic viscosity for SUPG stabilization."));
          const double kinematic_viscosity = dynamic_viscosity[q] / density[q];
          tau_supg_velocity[q] = StabilizationTools::compute_tau_supg(
            time_handler,
            dofs_per_cell,
            cell_diameter,
            param.finite_elements.velocity_degree,
            kinematic_viscosity,
            u_conv,
            grad_phi_u_first_component);
        }

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          // Shape functions on moving mesh
          shape_phi[q][k]      = fe_values_moving[tracer].value(k, q);
          grad_shape_phi[q][k] = fe_values_moving[tracer].gradient(k, q);
          shape_mu[q][k]       = fe_values_moving[potential].value(k, q);
          grad_shape_mu[q][k]  = fe_values_moving[potential].gradient(k, q);
          if (enable_tracer_stabilization)
            laplacian_shape_mu[q][k] =
              trace(fe_values_moving[potential].hessian(k, q));

          // Shape functions on fixed mesh
          if constexpr (enable_pseudo_solid)
          {
            shape_phi_fixed[q][k]      = fe_values_fixed[tracer].value(k, q);
            grad_shape_phi_fixed[q][k] = fe_values_fixed[tracer].gradient(k, q);
          }
        }

        if (enable_tracer_stabilization)
          tau_supg_tracer[q] = StabilizationTools::compute_tau_supg(
            time_handler,
            dofs_per_cell,
            cell_diameter,
            param.finite_elements.tracer_degree,
            mobility,
            u_conv,
            grad_shape_phi[q]);
      }
    }

  public:
    /**
     * Reinit this ScratchData on the given cell.
     * Default function to call when using a ScratchData.
     */
    template <typename VectorType>
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const VectorType              &current_solution,
                const std::vector<VectorType> &previous_solutions,
                const Function<dim>           &source_terms,
                const Function<dim>           &exact_solution)
    {
      /**
       * Reinit the fe_values on moving mesh on the current cell, and possibly
       * the fe_values on fixed mesh if enable_pseudo_solid is true.
       *
       * In the hp setting, these are the FEValues which are appropriate for the
       * current cell, extracted from the hp::FEValues with
       * get_present_fe_values().
       */
      active_fe_index  = cell->active_fe_index();
      active_fe_values = this->reinit(cell, false);
      if constexpr (enable_pseudo_solid)
        active_fe_values_fixed = this->reinit(cell, true);

      dofs_per_cell = active_fe_values->dofs_per_cell;
      for (const unsigned int i : active_fe_values->dof_indices())
        components[i] =
          active_fe_values->get_fe().system_to_component_index(i).first;

      bdf_c0 = time_handler.bdf_coefficients[0];

      if (enable_stabilization || enable_tracer_stabilization)
        cell_diameter = cell->diameter();

      /**
       * Volume contributions.
       *
       * Compute pseudo-solid data *before* Navier-Stokes data, since the mesh
       * velocity is required to compute the ALE velocity in the strong residual
       * of the NS equation.
       */
      if constexpr (enable_pseudo_solid)
        reinit_pseudo_solid_cell(*active_fe_values_fixed,
                                 *active_fe_values,
                                 current_solution,
                                 previous_solutions,
                                 source_terms,
                                 exact_solution);

      reinit_navier_stokes_cell(*active_fe_values,
                                current_solution,
                                previous_solutions,
                                source_terms,
                                exact_solution);

      if constexpr (enable_compressible)
        reinit_compressible_cell(*active_fe_values,
                                 current_solution,
                                 previous_solutions,
                                 source_terms,
                                 exact_solution);
      if constexpr (enable_cahn_hilliard)
        reinit_cahn_hilliard_cell(*active_fe_values_fixed,
                                  *active_fe_values,
                                  current_solution,
                                  previous_solutions,
                                  source_terms,
                                  exact_solution);
      /**
       * Face contributions
       */
      if (cell->at_boundary())
        for (const auto i_face : cell->face_indices())
        {
          const auto &face         = cell->face(i_face);
          face_at_boundary[i_face] = face->at_boundary();

          if (face_at_boundary[i_face])
          {
            face_boundary_id[i_face] = face->boundary_id();

            /**
             * Reinit the fe_face_values on moving (and possibly fixed) mesh,
             * with the same remark as for cells regarding the hp setting.
             */
            active_fe_face_values = this->reinit(cell, i_face, false);
            if constexpr (enable_pseudo_solid)
              active_fe_face_values_fixed = this->reinit(cell, i_face, true);

            reinit_navier_stokes_face(i_face,
                                      *active_fe_face_values,
                                      current_solution,
                                      previous_solutions,
                                      source_terms,
                                      exact_solution);
            if constexpr (enable_compressible)
              reinit_compressible_face(i_face,
                                       *active_fe_face_values,
                                       current_solution,
                                       previous_solutions,
                                       source_terms,
                                       exact_solution);
            if constexpr (enable_pseudo_solid)
              reinit_pseudo_solid_face(i_face,
                                       *active_fe_face_values_fixed,
                                       *active_fe_face_values,
                                       current_solution,
                                       previous_solutions,
                                       source_terms,
                                       exact_solution);
            if constexpr (enable_lagrange_multiplier)
              reinit_lagrange_multiplier_face(i_face,
                                              *active_fe_face_values_fixed,
                                              *active_fe_face_values,
                                              current_solution,
                                              previous_solutions,
                                              source_terms,
                                              exact_solution);
          }
        }
    }

  private:
    const ParameterReader<dim> &param;
    const bool                  use_quads;
    const ComponentOrdering     ordering;

    unsigned int n_components;
    unsigned int u_lower;
    unsigned int p_lower;
    unsigned int x_lower;
    unsigned int l_lower;
    unsigned int phi_lower;
    unsigned int mu_lower;
    unsigned int t_lower;

  public:
    static constexpr bool enable_pseudo_solid =
      (update_flags & pseudo_solid) != 0;
    static constexpr bool enable_lagrange_multiplier =
      (update_flags & lagrange_multiplier) != 0;
    static constexpr bool enable_cahn_hilliard =
      (update_flags & cahn_hilliard) != 0;
    static constexpr bool enable_compressible =
      (update_flags & compressible) != 0;
    static constexpr bool has_hp_capabilities =
      (update_flags & with_hp_capabilities) != 0;

    bool enable_stabilization;
    bool enable_tracer_stabilization;

  private:
    Parameters::PhysicalProperties<dim> physical_properties;
    Parameters::CahnHilliard<dim>       cahn_hilliard_param;

  public:
    unsigned int active_fe_index;

  private:
    // Non-owning pointers for active FEValues/FaceValues
    const FEValues<dim>     *active_fe_values;
    const FEValues<dim>     *active_fe_values_fixed;
    const FEFaceValues<dim> *active_fe_face_values;
    const FEFaceValues<dim> *active_fe_face_values_fixed;

    std::unique_ptr<FEValues<dim>>     fe_values;
    std::unique_ptr<FEValues<dim>>     fe_values_fixed;
    std::unique_ptr<FEFaceValues<dim>> fe_face_values;
    std::unique_ptr<FEFaceValues<dim>> fe_face_values_fixed;

    std::unique_ptr<hp::FEValues<dim>>     hp_fe_values;
    std::unique_ptr<hp::FEValues<dim>>     hp_fe_values_fixed;
    std::unique_ptr<hp::FEFaceValues<dim>> hp_fe_face_values;
    std::unique_ptr<hp::FEFaceValues<dim>> hp_fe_face_values_fixed;

  public:
    const unsigned int n_q_points;
    const unsigned int n_faces;
    const unsigned int n_faces_q_points;
    unsigned int       dofs_per_cell;
    const unsigned int max_dofs_per_cell;

    const TimeHandler &time_handler;

    std::vector<unsigned int>                components;
    std::vector<double>                      JxW_moving;
    std::vector<double>                      JxW_fixed;
    std::vector<bool>                        face_at_boundary;
    std::vector<unsigned int>                face_boundary_id;
    std::vector<std::vector<double>>         face_JxW_moving;
    std::vector<std::vector<double>>         face_JxW_fixed;
    std::vector<std::vector<Tensor<1, dim>>> face_normals_moving;

    // First of the BDF coefficients
    double bdf_c0;

    /**
     * Navier-Stokes
     */
    double kinematic_viscosity;

    FEValuesExtractors::Vector velocity;
    FEValuesExtractors::Scalar pressure;

    // Current and previous values and gradients for each quad node
    std::vector<Tensor<1, dim>>              present_velocity_values;
    std::vector<Tensor<2, dim>>              present_velocity_gradients;
    std::vector<SymmetricTensor<2, dim>>     present_velocity_sym_gradients;
    std::vector<double>                      present_velocity_divergence;
    std::vector<Tensor<1, dim>>              present_velocity_laplacians;
    std::vector<Tensor<3, dim>>              present_velocity_hessians;
    std::vector<Tensor<1, dim>>              present_velocity_grad_div;
    std::vector<Tensor<1, dim>>              present_velocity_time_derivatives;
    std::vector<double>                      present_pressure_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

    // Current values on faces (each face, each quad node)
    std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;
    std::vector<std::vector<Tensor<2, dim>>> present_face_velocity_gradients;
    std::vector<std::vector<SymmetricTensor<2, dim>>>
                                     present_face_velocity_sym_gradients;
    std::vector<std::vector<double>> present_face_velocity_divergence;

    std::vector<std::vector<double>> present_face_pressure_values;

    // Shape functions in volume (each quad node and each dof)
    std::vector<std::vector<Tensor<1, dim>>>          phi_u;
    std::vector<std::vector<Tensor<2, dim>>>          grad_phi_u;
    std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi_u;
    std::vector<std::vector<double>>                  div_phi_u;
    std::vector<std::vector<Tensor<1, dim>>>          laplacian_phi_u;
    std::vector<std::vector<Tensor<1, dim>>>          grad_div_phi_u;
    std::vector<std::vector<double>>                  phi_p;
    std::vector<std::vector<Tensor<1, dim>>>          grad_phi_p;

    // Shape functions on faces (each face, quad node and dof)
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;
    std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_u_face;
    std::vector<std::vector<std::vector<SymmetricTensor<2, dim>>>>
                                                  sym_grad_phi_u_face;
    std::vector<std::vector<std::vector<double>>> div_phi_u_face;
    std::vector<std::vector<std::vector<double>>> phi_p_face;

    // Source term in volume
    std::vector<Vector<double>> source_term_full_moving;
    std::vector<Tensor<1, dim>> source_term_velocity;
    std::vector<double>         source_term_pressure;

    // Exact solution (cell/volume quadrature points)
    std::vector<Vector<double>> exact_solution_full_cell;
    std::vector<Tensor<1, dim>> exact_velocity_values_cell;
    std::vector<double>         exact_pressure_values_cell;
    std::vector<double>         exact_temperature_values_cell;

    // Exact solution (faces)
    std::vector<Vector<double>>              exact_solution_full;
    std::vector<std::vector<Tensor<1, dim>>> grad_exact_solution_full;
    std::vector<std::vector<Tensor<2, dim>>> exact_face_velocity_gradients;
    std::vector<std::vector<double>>         exact_face_velocity_divergences;
    std::vector<std::vector<double>>         exact_face_pressure_values;
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    std::vector<std::vector<Tensor<1, dim>>> exact_face_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> exact_face_lambda_values;
    std::vector<std::vector<Tensor<1, dim>>> exact_face_mesh_velocity_values;
#endif

    // Stabilization data
    std::vector<Tensor<1, dim>> strong_residual_momentum;
    std::vector<Tensor<1, dim>> grad_phi_u_first_component;
    std::vector<double>         tau_supg_velocity;
    double                      cell_diameter;

    /**
     * Compressible NS
     */
    FEValuesExtractors::Scalar temperature;
    double                     density_ref;
    double                     pressure_ref;
    double                     temperature_ref;
    double                     alpha_r;
    double                     beta_r;

    std::vector<std::vector<double>> face_input_pressure_values;
    std::vector<std::vector<double>> face_input_heat_flux_values;

    // Variable density, also used by CHNS models
    std::vector<double> density;
    std::vector<double> a_p; // alpha_r/(alpha_r p* + 1)
    std::vector<double> b_T; // beta_r /(beta_r  T* + 1)

    std::vector<std::vector<double>> previous_pressure_values;
    std::vector<Tensor<1, dim>>      present_pressure_gradients;
    std::vector<double>              present_temperature_values;
    std::vector<Tensor<1, dim>>      present_temperature_gradients;
    std::vector<std::vector<double>> previous_temperature_values;

    // Thermodynamic fields: p = p^* + p_ref, T = T^* + T_ref
    std::vector<double> present_pressure_absolute_values;
    std::vector<double> present_temperature_absolute_values;

    std::vector<std::vector<double>>         phi_T;
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_T;

    std::vector<double> source_term_temperature;

    // Temperature fields on faces (T = T^* + T_ref for absolute values)
    std::vector<std::vector<double>> present_face_temperature_values;
    std::vector<std::vector<double>> present_face_temperature_absolute_values;
    std::vector<std::vector<Tensor<1, dim>>> exact_face_temperature_gradients;
    std::vector<std::vector<std::vector<double>>>         phi_T_face;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> grad_phi_T_face;
    std::vector<std::vector<Tensor<1, dim>>> present_face_temperature_gradients;

    /**
     * Pseudo-solid and ALE
     */
    FEValuesExtractors::Vector position;

    std::vector<double> lame_mu;
    std::vector<double> lame_lambda;

    std::vector<Tensor<1, dim>>              present_position_values;
    std::vector<Tensor<2, dim>>              present_position_gradients;
    std::vector<Tensor<1, dim>>              present_mesh_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values;

    // Data for hyperelastic models
    std::vector<double>         present_position_J;                   // det(F)
    std::vector<Tensor<2, dim>> present_position_inverse_gradients;   // F^{-1}
    std::vector<Tensor<2, dim>> present_position_inverse_gradients_T; // F^{-T}

    // Current and previous values on faces
    std::vector<std::vector<Tensor<1, dim>>> present_face_position_values;
    std::vector<std::vector<Tensor<2, dim>>> present_face_position_gradient;
    std::vector<std::vector<Tensor<1, dim>>> present_face_mesh_velocity_values;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>>
      previous_face_position_values;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<Tensor<1, dim>>>          phi_x;
    std::vector<std::vector<Tensor<2, dim>>>          grad_phi_x;
    std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi_x;
    std::vector<std::vector<Tensor<2, dim>>>          grad_phi_x_moving;
    std::vector<std::vector<Tensor<3, dim>>>          hessian_phi_x_moving;
    std::vector<std::vector<double>>                  div_phi_x;
    std::vector<std::vector<double>>                  trace_grad_phi_x;

    // Shape functions on faces for relevant faces, each quad node and each dof
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_x_face;
    std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_x_face;

    std::vector<Vector<double>> source_term_full_fixed;
    std::vector<Tensor<1, dim>> source_term_position;

    // Gradient of source term, at each quad node, for each dof component,
    // result is a Tensor<1, dim>. Only needed for the Jacobian matrix of NS
    // with moving mesh.
    std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
    std::vector<Tensor<2, dim>>              grad_source_velocity;
    std::vector<Tensor<1, dim>>              grad_source_pressure;
    std::vector<Tensor<1, dim>>              grad_source_tracer;
    std::vector<Tensor<1, dim>>              grad_source_potential;

    // If the elasticity source term is written w.r.t. the current mesh position
    // x, this is its gradient w.r.t. x. Used in the Jacobian matrix.
    std::vector<Tensor<2, dim>> grad_source_term_position_current_mesh;

    // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
    std::array<Tensor<1, dim>, dim - 1> dxsids_array;

    // At face x quad node x phi_position_j
    std::vector<std::vector<std::vector<double>>> delta_dx;

    /**
     * Lagrange multiplier
     */
    FEValuesExtractors::Vector               lambda;
    std::vector<std::vector<Tensor<1, dim>>> present_face_lambda_values;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_l_face;
#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    std::vector<std::vector<Tensor<1, dim>>> face_velocity_source_term;
#endif

    // Rigid-body rotation enforced with Lagrange multiplier
    // The prescribed angular velocity is scalar in 2D and a vector in 3D
    using angular_velocity_type = Tensor<dim == 2 ? 0 : 1, dim>;
    std::vector<std::vector<Tensor<1, dim>>>
      input_face_rigid_body_rotation_velocity;

    /**
     * Cahn-Hilliard
     */
    FEValuesExtractors::Scalar tracer;
    FEValuesExtractors::Scalar potential;

    double         density0;
    double         density1;
    double         dynamic_viscosity0;
    double         dynamic_viscosity1;
    double         mobility;
    double         epsilon;
    double         sigma_tilde;
    double         diffusive_flux_factor;
    Tensor<1, dim> body_force;

    CahnHilliard::TracerLimiterFunction tracer_limiter;

    std::vector<double> derivative_density_wrt_tracer;
    std::vector<double> dynamic_viscosity;
    std::vector<double> derivative_dynamic_viscosity_wrt_tracer;

    // Tracer on current and fixed (reference) mesh
    std::vector<double>              tracer_values;
    std::vector<double>              tracer_time_derivatives;
    std::vector<Tensor<1, dim>>      tracer_gradients;
    std::vector<double>              tracer_values_fixed;
    std::vector<Tensor<1, dim>>      tracer_gradients_fixed;
    std::vector<std::vector<double>> previous_tracer_values;
    // Potential on current mesh
    std::vector<double>         potential_values;
    std::vector<Tensor<1, dim>> potential_gradients;
    std::vector<double>         potential_laplacians;
    // Only used for the moving-mesh x-variation of the tracer SUPG residual.
    std::vector<Tensor<2, dim>> potential_hessians;

    std::vector<Tensor<1, dim>> diffusive_flux;
    std::vector<double>         tau_supg_tracer;

    std::vector<std::vector<double>>         shape_phi;
    std::vector<std::vector<Tensor<1, dim>>> grad_shape_phi;
    std::vector<std::vector<double>>         shape_phi_fixed;
    std::vector<std::vector<Tensor<1, dim>>> grad_shape_phi_fixed;
    std::vector<std::vector<double>>         shape_mu;
    std::vector<std::vector<Tensor<1, dim>>> grad_shape_mu;
    std::vector<std::vector<double>>         laplacian_shape_mu;

    std::vector<double> source_term_tracer;
    std::vector<double> source_term_potential;
  };
} // namespace NavierStokesScratch

#endif
