#ifndef LAGRANGE_MULTIPLIER_ASSEMBLERS_H
#define LAGRANGE_MULTIPLIER_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <components_ordering.h>
#include <parameter_reader.h>

namespace Assembly
{
  /**
   * A namespace for the assembly routines involving a Lagrange multiplier
   * to enforce a velocity-related constraint, typically a weakly enforced
   * no-slip boundary condition.
   */
  namespace LagrangeMultiplier
  {
    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh = false>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     * Assembler for a weakly enforced no-slip boundary condition.
     * This assembles the following terms:
     *
     * Momentum:   (\phi_{u,i}, - \lambda)_\Gamma
     *
     * and
     *
     * Multiplier: (\phi_{l,i}, - velocity_constraint)_\Gamma,
     *
     * where velocity_constraint takes the general form (u - dxdt - u_rot),
     * with dxdt the mesh velocity and u_rot an imposed rotation velocity.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    class WeakNoSlipAssembler : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      WeakNoSlipAssembler(const ParameterReader<dim> &param,
                          const ComponentOrdering    &ordering)
        : AssemblerBase<ScratchData, CopyData>()
        , param(param)
        , ordering(ordering)
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
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };
  } // namespace LagrangeMultiplier
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace LagrangeMultiplier
  {
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      if (has_boundary_condition(param.fluid_bc,
                                 BoundaryConditions::Type::weak_no_slip))
      {
        assemblers.emplace_back(
          std::make_unique<
            WeakNoSlipAssembler<dim, ScratchData, CopyData, with_moving_mesh>>(
            param, ordering));
      }
    }
  } // namespace LagrangeMultiplier
} // namespace Assembly

#endif
