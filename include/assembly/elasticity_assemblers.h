#ifndef ELASTICITY_ASSEMBLERS_H
#define ELASTICITY_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <components_ordering.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <scratch_data.h>

namespace Assembly
{
  namespace Elasticity
  {
    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim, typename ScratchData, typename CopyData>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class LinearElasticityAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      LinearElasticityAssembler(const ParameterReader<dim> &param,
                                const ComponentOrdering    &ordering)
        : param(param)
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

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class NeoHookeanAssembler : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      NeoHookeanAssembler(const ParameterReader<dim> &param,
                          const ComponentOrdering    &ordering)
        : param(param)
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

    /**
     *
     */
    template <int dim, typename ScratchData, typename CopyData>
    class OgdenHyperelasticityAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      OgdenHyperelasticityAssembler(const ParameterReader<dim> &param,
                                    const ComponentOrdering    &ordering)
        : param(param)
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

    /**
     * Assemble the extra matrix contribution when the source term depends on
     * the current mesh position f(x(X)).
     */
    template <int dim, typename ScratchData, typename CopyData>
    class CurrentMeshSourceAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      CurrentMeshSourceAssembler(const ParameterReader<dim> &param,
                                 const ComponentOrdering    &ordering)
        : param(param)
        , ordering(ordering)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * This assembler has no additional rhs to assemble, as the source term
       * is already assembled in the main elasticity assembler.
       */
      virtual void assemble_rhs(const ScratchData &, CopyData &) const override
      {}

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };

    /**
     * Elasticity source term depending on the Cahn-Hilliard Navier-Stokes phase
     * marker (tracer) and velocity.
     *
     * Note: this is only instantiated for the CHNS scratch data with moving
     * mesh.
     */
    template <int dim, typename ScratchData, typename CopyData>
    class SourceFromCHNSTracerAssembler
      : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      SourceFromCHNSTracerAssembler(const ParameterReader<dim> &param,
                                    const ComponentOrdering    &ordering)
        : param(param)
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
      virtual void assemble_rhs(const ScratchData &, CopyData &) const override;

    public:
      const ParameterReader<dim> &param;
      const ComponentOrdering    &ordering;
    };
  } // namespace Elasticity
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace Elasticity
  {
    template <int dim, typename ScratchData, typename CopyData>
    void setup_assemblers(
      const ParameterReader<dim> &param,
      const ComponentOrdering    &ordering,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      const auto &solid = param.physical_properties.pseudosolids[0];

      switch (solid.constitutive_model)
      {
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::linear_elasticity:
          assemblers.emplace_back(
            std::make_unique<
              LinearElasticityAssembler<dim, ScratchData, CopyData>>(param,
                                                                     ordering));
          break;
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean:
          assemblers.emplace_back(
            std::make_unique<NeoHookeanAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
          break;
        case Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden:
          assemblers.emplace_back(
            std::make_unique<
              OgdenHyperelasticityAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
          break;
        default:
          DEAL_II_ASSERT_UNREACHABLE();
      }

      if (param.elasticity.enable_source_term_on_current_mesh)
        assemblers.emplace_back(
          std::make_unique<
            CurrentMeshSourceAssembler<dim, ScratchData, CopyData>>(param,
                                                                    ordering));

      if constexpr (std::is_same_v<
                      ScratchData,
                      NavierStokesScratch::ScratchDataCHNS<dim, true>>)
      {
        // FIXME: add an "enable" flag to assemble this only when needed
        if (false)
          assemblers.emplace_back(
            std::make_unique<
              SourceFromCHNSTracerAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
      }
    }
  } // namespace Elasticity
} // namespace Assembly

#endif
