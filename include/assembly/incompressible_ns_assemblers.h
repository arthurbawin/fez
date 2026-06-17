#ifndef INCOMPRESSIBLE_NS_ASSEMBLERS_H
#define INCOMPRESSIBLE_NS_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <components_ordering.h>
#include <deal.II/base/table.h>
#include <parameter_reader.h>

namespace Assembly
{
  /**
   * A namespace gathering the incompressible Navier-Stokes assembly routines,
   * including the volume and various boundary contributions.
   *
   * For efficiency, the volume contributions of the various physics are not
   * split into separate routines. For instance, the volume assembler in this
   * namespace is also in charge of assembling the elasticity equation for mesh
   * movement, if required. This avoids accessing the scratch data multiple
   * times to retrieve, e.g., the mesh position shape functions, which appear
   * both in the elasticity equation block and in the off-diagonal coupling
   * blocks of the Jacobian matrix.
   */
  namespace IncompressibleNavierStokes
  {
    /**
     * These flags are used to specify which terms to assemble in addition to
     * the base incompressible Navier-Stokes weak form written in laplacian
     * form.
     *
     * FIXME: these are redundant with the ScratchData flags, since the scratch
     * should agree with the system of PDEs.
     */
    enum AssemblyFlags : unsigned int
    {
      /**
       * Assemble the incompressible Navier-Stokes equations in laplacian form
       * (i.e., without the grad(div) velocity term). Default.
       */
      ns_laplace_form = 0,

      /**
       * Assemble the Navier-Stokes equations in divergence form.
       */
      divergence_form = 1 << 1,

      /**
       * Add SUPG/PSPG stabilization forms.
       */
      stabilization = 1 << 2,

      /**
       * Assemble the mesh movement elasticity equations.
       */
      pseudo_solid = 1 << 3
    };

    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     * Abstract base class for the incompressible Navier-Stokes forms.
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
      static constexpr bool with_divergence_form =
        (assembly_flags & divergence_form) != 0;
      static constexpr bool with_stabilization =
        (assembly_flags & stabilization) != 0;
      static constexpr bool with_pseudo_solid =
        (assembly_flags & pseudo_solid) != 0;
      static constexpr bool with_moving_mesh =
        (assembly_flags & pseudo_solid) != 0;

      const ComponentOrdering &ordering;
    };

    /**
     * Assembler for the incompressible Navier-Stokes in the volume.
     *
     * TODO: Add expression of the weak form.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags = ns_laplace_form>
    class VolumeAssembler : public Base<ScratchData, CopyData, assembly_flags>
    {
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
     * Assembler for the traction/open boundary condition for which the stress
     * flux is prescribed by a manufactured solution. The assembled weak form
     * writes:
     *
     * (\phi_{u,i}, - \sigma(u_exact, p_exact) \cdot n)_\Gamma.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags = ns_laplace_form>
    class MMSTractionAssembler
      : public Base<ScratchData, CopyData, assembly_flags>
    {
    public:
      MMSTractionAssembler(const ParameterReader<dim> &param,
                           const ComponentOrdering    &ordering)
        : Base<ScratchData, CopyData, assembly_flags>(ordering)
        , param(param)
      {}

      /**
       * This form has no local matrix to assemble.
       */
      virtual void assemble_matrix(const ScratchData &,
                                   CopyData &) const override
      {}

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const ParameterReader<dim> &param;
    };
  } // namespace IncompressibleNavierStokes
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace IncompressibleNavierStokes
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags = ns_laplace_form>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      // Perform compile-time checks when possible: if this function is called
      // with specific assembly flags, make sure that the template flags for the
      // given ScratchData agree.
      // This is only possible for the pseudo-solid flag, since the terms for
      // the divergence form of the NS equations are always computed in the
      // scratch, and computing the additional terms for SUPG stabilization is a
      // runtime flag (the latter is checked in the assembly routine in debug).
      //
      // FIXME: When the dedicated elasticity assemblers are added, this check
      // will be moved there.
      static_assert(
        !(assembly_flags & pseudo_solid) || ScratchData::enable_pseudo_solid,
        "The assemblers for the incompressible Navier-Stokes equations are set "
        "to assemble the pseudo-solid equations, but computation of the "
        "required data was not enabled in the provided ScratchData.");

      using namespace BoundaryConditions;

      // Assign the volume assembler
      assemblers.emplace_back(
        std::make_unique<
          VolumeAssembler<dim, ScratchData, CopyData, assembly_flags>>(
          ordering, coupling_table));

      // Assign the relevant boundary assemblers
      if (has_boundary_condition(param.fluid_bc, Type::open_mms))
      {
        assemblers.emplace_back(
          std::make_unique<
            MMSTractionAssembler<dim, ScratchData, CopyData, assembly_flags>>(
            param, ordering));
      }
    }
  } // namespace IncompressibleNavierStokes
} // namespace Assembly

#endif
