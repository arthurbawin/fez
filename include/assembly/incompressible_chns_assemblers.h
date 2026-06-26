#ifndef INCOMPRESSIBLE_CHNS_ASSEMBLERS_H
#define INCOMPRESSIBLE_CHNS_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <components_ordering.h>
#include <deal.II/base/table.h>
#include <parameter_reader.h>

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
      moving_mesh = 1 << 2
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

      // Assign the volume assembler
      if (supg)
      {
        if (tracer_supg)
          assemblers.emplace_back(
            std::make_unique<VolumeAssembler<
              dim,
              ScratchData,
              CopyData,
              stabilization | tracer_stabilization | moving_mesh_flag>>(
              ordering, coupling_table));
        else
          assemblers.emplace_back(
            std::make_unique<VolumeAssembler<dim,
                                             ScratchData,
                                             CopyData,
                                             stabilization | moving_mesh_flag>>(
              ordering, coupling_table));
      }
      else
      {
        if (tracer_supg)
        {
          assemblers.emplace_back(
            std::make_unique<
              VolumeAssembler<dim,
                              ScratchData,
                              CopyData,
                              tracer_stabilization | moving_mesh_flag>>(
              ordering, coupling_table));
        }
        else
          assemblers.emplace_back(
            std::make_unique<
              VolumeAssembler<dim, ScratchData, CopyData, moving_mesh_flag>>(
              ordering, coupling_table));
      }

      // Assign the relevant boundary assemblers
      // ...
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

#endif
