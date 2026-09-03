#ifndef INCOMPRESSIBLE_CHNS_ASSEMBLERS_H
#define INCOMPRESSIBLE_CHNS_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/base/table.h>
#include <parameter_reader.h>

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
       * Assemble the Stepien quasi-incompressible model instead of the default
       * Abels model. The two models share the potential (chemical potential)
       * equation but differ in the momentum, continuity and phase equations;
       * see the volume assembler for the corresponding weak forms.
       */
      stepien = 1 << 3
    };

    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
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
      static constexpr bool with_stepien = (assembly_flags & stepien) != 0;

      /**
       * Whether the momentum equation carries the Stepien forcing, i.e. the
       * capillary force (dpr - mu) grad(phi), the conservative correction
       * S_c u and the bulk-viscosity stress. It is turned off by the
       * Stepien-Abels validation mode, which falls back to the Abels momentum
       * forcing; see CahnHilliard::stepien_abels_validation. The continuity,
       * phase and potential equations need no such switch: they reduce to the
       * Abels ones on their own once rho0 = rho1.
       */
      static constexpr bool with_stepien_momentum =
        with_stepien && !CahnHilliard::stepien_abels_validation;

      static_assert(!(with_stepien && with_moving_mesh),
                    "The Stepien forms are only implemented on a fixed mesh.");

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
  } // namespace IncompressibleCHNS
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    namespace internal
    {
      /**
       * Emplace the volume assembler matching the runtime SUPG switches
       * @p supg and @p tracer_supg, on top of the compile-time @p model_flags
       * which carry the CHNS model and the moving-mesh bit.
       *
       * Keeping the model and mesh bits as a template parameter avoids
       * spelling out one branch per combination of the four flags.
       */
      template <int dim,
                typename ScratchData,
                typename CopyData,
                unsigned int model_flags,
                bool         allow_tracer_stabilization = true>
      void emplace_volume_assembler(
        const ComponentOrdering            &ordering,
        const Table<2, DoFTools::Coupling> &coupling_table,
        const bool                          supg,
        const bool                          tracer_supg,
        std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
          &assemblers)
      {
        const auto emplace = [&](auto flags) {
          assemblers.emplace_back(
            std::make_unique<VolumeAssembler<dim,
                                             ScratchData,
                                             CopyData,
                                             decltype(flags)::value>>(
              ordering, coupling_table));
        };

        if constexpr (allow_tracer_stabilization)
        {
          if (supg && tracer_supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization |
                                             tracer_stabilization>{});
          else if (tracer_supg)
            emplace(
              std::integral_constant<unsigned int,
                                     model_flags | tracer_stabilization>{});
          else if (supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization>{});
          else
            emplace(std::integral_constant<unsigned int, model_flags>{});
        }
        else
        {
          // The caller guarantees that tracer_supg is false here, so the
          // corresponding assemblers are never instantiated.
          if (supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization>{});
          else
            emplace(std::integral_constant<unsigned int, model_flags>{});
        }
      }
    } // namespace internal

    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
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

      using namespace BoundaryConditions;

      const bool supg        = param.stabilization.enable_supg;
      const bool tracer_supg = param.stabilization.enable_tracer_supg;
      constexpr unsigned int moving_mesh_flag =
        with_moving_mesh ? moving_mesh : chns;

      if constexpr (with_moving_mesh)
        AssertThrow(
          !(supg || tracer_supg),
          ExcMessage(
            "CHNS stabilization on a moving mesh is not implemented yet."));

      /**
       * The volume assembler reads the mobility as a single scalar
       * (ScratchData::mobility) rather than per quadrature point, so a
       * degenerate mobility would silently be evaluated as the constant one.
       */
      AssertThrow(
        param.cahn_hilliard.mobility_model ==
          Parameters::CahnHilliard<dim>::MobilityModel::constant,
        ExcMessage("The \"Stabilization\" subsection is only implemented for a "
                   "constant mobility."));

      // Assign the volume assembler
      if (CahnHilliard::is_stepien_model(param.cahn_hilliard))
      {
        // The Stepien forms are written for a fixed mesh only, so the ALE
        // instantiations are never requested.
        if constexpr (with_moving_mesh)
          AssertThrow(false, ExcMessage("Stepien ALE is not implemented."));
        else
          internal::emplace_volume_assembler<dim, ScratchData, CopyData,
                                             stepien>(
            ordering, coupling_table, supg, tracer_supg, assemblers);
      }
      else
        internal::emplace_volume_assembler<dim,
                                           ScratchData,
                                           CopyData,
                                           moving_mesh_flag>(
          ordering, coupling_table, supg, tracer_supg, assemblers);

      // Assign the relevant boundary assemblers
      // ...
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

#endif
