#ifndef INCOMPRESSIBLE_CHNS_ASSEMBLERS_H
#define INCOMPRESSIBLE_CHNS_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <parameter_reader.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace Assembly
{
  /**
   * A namespace gathering the assembly routines for the (quasi-)incompressible
   * Cahn-Hilliard Navier-Stokes models.
   */
  namespace IncompressibleCHNS
  {
    /**
     * These flags are used to specify which terms to assemble in addition to
     * the base incompressible CHNS system.
     */
    enum AssemblyFlags : unsigned int
    {
      /**
       * Assemble the incompressible CHNS system without any stabilization.
       */
      chns = 0,

      /**
       * Add SUPG/PSPG stabilization forms for the Navier-Stokes equations.
       */
      stabilization = 1 << 0,

      /**
       * Add SUPG stabilization form for the phase tracer equation.
       */
      tracer_stabilization = 1 << 1,

      /**
       * Account for moving mesh contributions (ALE).
       */
      moving_mesh = 1 << 2,

      /**
       * Assemble the enlarged (psi) tracer Helmholtz reconstruction. Implies
       * moving_mesh (the enlarged solver is ALE only).
       */
      enlarged = 1 << 3,

      /**
       * Assemble the Ding-Horriche CHNS model instead of the default Abels
       * model. This changes the potential scaling, the capillary momentum
       * force (mu*grad(phi) instead of phi*grad(mu)) and drops the diffusive
       * inertia term.
       */
      ding_horriche = 1 << 4
    };

    /* ---------- Enlarged (psi) tracer Helmholtz reconstruction ----------
     *
     * The enlarged tracer psi is a widened phase marker reconstructed by the
     * Helmholtz equation
     *
     *     psi - L^2 lap(psi) = phi - mu_correction,
     *
     * with L the widening length scale (precomputed as scratch.psi_length_scale
     * _sq). The optional mu-correction localizes a chemical-potential term near
     * the interface. These forms are templated on the ScratchData so they can
     * be reused both by the full CHNS solver (phi a finite-element unknown,
     * field mode) and, later, by the elasticity presolver (phi an analytic
     * function, function mode). The sign convention matches the rest of the
     * CHNS assembler: the rhs holds the negative residual and the matrix the
     * residual Jacobian.
     */

    // Smooth band weight (1 - phi^2)^2 keeping the mu-correction localized near
    // the diffuse interface, and its derivative w.r.t. phi.
    inline double psi_mu_correction_eta(const double tracer_value)
    {
      const double band = 1. - tracer_value * tracer_value;
      return band * band;
    }
    inline double psi_mu_correction_eta_jacobian(const double tracer_value)
    {
      const double band = 1. - tracer_value * tracer_value;
      return -4. * tracer_value * band;
    }

    // Prefactor of the opt-in mu-correction. Returns 0 (no cost) when disabled.
    // Abels form; the Ding-Horriche variant (L^2 / epsilon^2) will be added
    // together with the model-switching feature.
    template <int dim>
    inline double compute_psi_mu_correction_prefactor(
      const double psi_mu_correction_factor,
      const double sigma_tilde,
      const double epsilon,
      const double length_scale_sq)
    {
      if (std::abs(psi_mu_correction_factor) < 1e-14)
        return 0.;
      return psi_mu_correction_factor * length_scale_sq /
             (epsilon * sigma_tilde);
    }

    template <int dim, typename ScratchData, typename VectorType>
    inline void assemble_psi_equation_rhs(const ComponentOrdering &ordering,
                                          const ScratchData       &scratch,
                                          VectorType              &local_rhs)
    {
      const double length_scale_sq = scratch.psi_length_scale_sq;
      const double correction_prefactor =
        compute_psi_mu_correction_prefactor<dim>(scratch.psi_mu_correction_factor,
                                                 scratch.sigma_tilde,
                                                 scratch.epsilon,
                                                 length_scale_sq);

      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double psi_mu_correction =
            correction_prefactor *
            psi_mu_correction_eta(scratch.tracer_values[q]) *
            scratch.potential_values[q];

          const double residual_i =
            scratch.shape_psi[q][i] *
              (scratch.psi_values[q] - scratch.tracer_values[q] +
               scratch.source_term_psi[q] - psi_mu_correction) +
            length_scale_sq * dealii::scalar_product(scratch.grad_shape_psi[q][i],
                                                     scratch.psi_gradients[q]);

          local_rhs(i) -= residual_i * scratch.JxW_moving[q];
        }
    }

    template <int  dim,
              bool with_moving_mesh,
              typename ScratchData,
              typename MatrixType>
    inline void assemble_psi_equation_matrix(
      const ComponentOrdering            &ordering,
      const dealii::Table<2, dealii::DoFTools::Coupling> &coupling_table,
      const ScratchData                  &scratch,
      MatrixType                         &local_matrix)
    {
      using namespace dealii;
      const double length_scale_sq = scratch.psi_length_scale_sq;
      const double correction_prefactor =
        compute_psi_mu_correction_prefactor<dim>(scratch.psi_mu_correction_factor,
                                                 scratch.sigma_tilde,
                                                 scratch.epsilon,
                                                 length_scale_sq);

      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double          phi_i  = scratch.shape_psi[q][i];
          const Tensor<1, dim> &grad_i = scratch.grad_shape_psi[q][i];
          const double          eta_weight =
            psi_mu_correction_eta(scratch.tracer_values[q]);
          const double eta_weight_jacobian =
            psi_mu_correction_eta_jacobian(scratch.tracer_values[q]);
          const double psi_mu_correction =
            correction_prefactor * eta_weight * scratch.potential_values[q];

          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          {
            if (coupling_table[scratch.components[i]][scratch.components[j]] !=
                DoFTools::always)
              continue;

            const unsigned int comp_j          = scratch.components[j];
            double             local_matrix_ij = 0.;

            if (ordering.is_psi(comp_j))
            {
              local_matrix_ij += phi_i * scratch.shape_psi[q][j];
              local_matrix_ij +=
                length_scale_sq *
                scalar_product(grad_i, scratch.grad_shape_psi[q][j]);
            }

            if (ordering.is_tracer(comp_j))
              local_matrix_ij -=
                phi_i * (scratch.shape_phi[q][j] +
                         correction_prefactor * eta_weight_jacobian *
                           scratch.potential_values[q] * scratch.shape_phi[q][j]);

            if (ordering.is_potential(comp_j))
              local_matrix_ij -=
                phi_i * correction_prefactor * eta_weight * scratch.shape_mu[q][j];

            if constexpr (with_moving_mesh)
              if (ordering.is_position(comp_j))
              {
                // Mesh-position (x) variation of the psi residual. Field values
                // do not transform, so the value term only picks up the JxW
                // trace; the Helmholtz gradient term varies as a weak laplacian
                // (gradients transform as grad -> grad - G^T grad). The source
                // x-variation is omitted, as elsewhere in this assembler.
                const Tensor<2, dim> &G   = scratch.grad_phi_x_moving[q][j];
                const double          trG = trace(G);

                local_matrix_ij +=
                  phi_i *
                  (scratch.psi_values[q] - scratch.tracer_values[q] -
                   psi_mu_correction) *
                  trG;

                const Tensor<1, dim> dgrad_i   = -transpose(G) * grad_i;
                const Tensor<1, dim> dgrad_psi =
                  -transpose(G) * scratch.psi_gradients[q];
                local_matrix_ij +=
                  length_scale_sq *
                  (scalar_product(dgrad_i, scratch.psi_gradients[q]) +
                   scalar_product(grad_i, dgrad_psi) +
                   scalar_product(grad_i, scratch.psi_gradients[q]) * trG);
              }

            local_matrix(i, j) += local_matrix_ij * scratch.JxW_moving[q];
          }
        }
    }

    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh,
              bool with_enlarged = false>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     * Abstract base class for the incompressible CHNS forms.
     */
    template <typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    class Base : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      /**
       * Constructor
       */
      Base(const ComponentOrdering &ordering)
        : ordering(ordering)
      {}

    public:
      static constexpr bool with_stabilization =
        (assembly_flags & stabilization) != 0;
      static constexpr bool with_tracer_stabilization =
        (assembly_flags & tracer_stabilization) != 0;
      static constexpr bool with_moving_mesh =
        (assembly_flags & moving_mesh) != 0;
      static constexpr bool with_enlarged =
        (assembly_flags & enlarged) != 0;
      static constexpr bool with_ding_horriche =
        (assembly_flags & ding_horriche) != 0;

      const ComponentOrdering &ordering;
    };

    /**
     * Assembler for the incompressible CHNS system in the volume.
     *
     * TODO: Add expression of the weak form.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags = chns>
    class VolumeAssembler : public Base<ScratchData, CopyData, assembly_flags>
    {
      using BaseType = Base<ScratchData, CopyData, assembly_flags>;

    public:
      VolumeAssembler(const ComponentOrdering            &ordering,
                      const Table<2, DoFTools::Coupling> &coupling_table)
        : Base<ScratchData, CopyData, assembly_flags>(ordering)
        , coupling_table(coupling_table)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const Table<2, DoFTools::Coupling> &coupling_table;
    };

    /**
     * Boundary assembler for the static contact-angle (wetting) condition. On
     * Cahn-Hilliard boundary faces carrying a contact angle, the natural Neumann
     * condition n.grad(phi) = 0 of the potential equation is replaced by the
     * wetting condition n.grad(phi) = g(phi), adding the boundary term
     *
     *   - coeff * g(phi) * w_mu        (residual, mu rows)
     *   + coeff * g'(phi) * N_phi * w_mu   (Jacobian, mu <- phi),
     *
     * with coeff = contact_angle_surface_coefficient (= sigma_tilde * epsilon).
     * Following the rest of the framework, the boundary contribution is added
     * inside assemble_rhs/assemble_matrix by looping over the cell's boundary
     * faces (using the moving-mesh Cahn-Hilliard face data).
     */
    template <int dim, typename ScratchData, typename CopyData>
    class ContactAngleBoundaryAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      ContactAngleBoundaryAssembler(const ParameterReader<dim> &param,
                                    const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };
  } // namespace IncompressibleCHNS
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh,
              bool with_enlarged>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      // Perform some static checks where possible:
      static_assert(with_moving_mesh == ScratchData::enable_pseudo_solid,
                    "To enable moving_mesh computations in the CHNS "
                    "assemblers, the provided ScratchData should be "
                    "initialized with a pseudo-solid update flag.");
      static_assert(with_enlarged == ScratchData::enable_enlarged,
                    "To enable the enlarged (psi) tracer in the CHNS "
                    "assemblers, the provided ScratchData should be "
                    "initialized with the enlarged update flag.");
      static_assert(!with_enlarged || with_moving_mesh,
                    "The enlarged CHNS solver is ALE only.");

      using namespace BoundaryConditions;

      const bool supg        = param.stabilization.enable_supg;
      const bool tracer_supg = param.stabilization.enable_tracer_supg;
      const bool use_ding_horriche =
        CahnHilliard::is_ding_horriche_model(param.cahn_hilliard);
      constexpr unsigned int moving_mesh_flag =
        with_moving_mesh ? moving_mesh : chns;
      constexpr unsigned int enlarged_flag = with_enlarged ? enlarged : chns;

      // Instantiate the volume assembler for a compile-time base flag set,
      // adding the Ding-Horriche model flag when that model is selected. This
      // keeps the model a compile-time assembly flag (branched with
      // if constexpr in the assembler) rather than an extra template parameter.
      auto emplace_volume_assembler = [&](auto base_flags_constant) {
        constexpr unsigned int base_flags = decltype(base_flags_constant)::value;
        if (use_ding_horriche)
          assemblers.emplace_back(
            std::make_unique<VolumeAssembler<dim,
                                             ScratchData,
                                             CopyData,
                                             base_flags | ding_horriche>>(
              ordering, coupling_table));
        else
          assemblers.emplace_back(
            std::make_unique<
              VolumeAssembler<dim, ScratchData, CopyData, base_flags>>(
              ordering, coupling_table));
      };

      // Assign the volume assembler
      if (supg)
      {
        if (tracer_supg)
          emplace_volume_assembler(
            std::integral_constant<unsigned int,
                                   stabilization | tracer_stabilization |
                                     moving_mesh_flag | enlarged_flag>{});
        else
          emplace_volume_assembler(
            std::integral_constant<unsigned int,
                                   stabilization | moving_mesh_flag |
                                     enlarged_flag>{});
      }
      else
      {
        if (tracer_supg)
          emplace_volume_assembler(
            std::integral_constant<unsigned int,
                                   tracer_stabilization | moving_mesh_flag |
                                     enlarged_flag>{});
        else
          emplace_volume_assembler(
            std::integral_constant<unsigned int,
                                   moving_mesh_flag | enlarged_flag>{});
      }

      // Assign the relevant boundary assemblers.
      // Static contact-angle (wetting) condition: registered only when at least
      // one Cahn-Hilliard boundary carries a contact angle.
      const bool any_contact_angle = std::any_of(
        param.cahn_hilliard_bc.begin(),
        param.cahn_hilliard_bc.end(),
        [](const auto &id_bc) { return id_bc.second.contact_angle >= 0.; });
      if (any_contact_angle)
        assemblers.emplace_back(
          std::make_unique<
            ContactAngleBoundaryAssembler<dim, ScratchData, CopyData>>(
            param, ordering));
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

#endif
