#ifndef ELASTICITY_ASSEMBLERS_H
#define ELASTICITY_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <components_ordering.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <scratch_data.h>

#include <algorithm>
#include <cmath>

template <int dim>
class ScratchDataElasticity;

namespace Assembly
{
  namespace Elasticity
  {
    /**
     * Regularized compression coefficient of the Cahn-Hilliard moving-mesh
     * forcing. The raw coefficient phi / (1 - gamma^2 phi^2) is singular as
     * gamma*phi -> 1, so phi is saturated through a tanh before being used.
     * Both the value and its derivative w.r.t. the phase are returned (the
     * derivative is used to linearize the forcing).
     */
    struct MeshForcingFactor
    {
      double value;
      double derivative;
    };

    inline MeshForcingFactor
    mesh_forcing_factor(const double phase_value, const double gamma)
    {
      constexpr double phi_max_user = 0.998;
      const double     phi_max_safe =
        std::min(phi_max_user, 0.98 / std::max(gamma, 1e-14));

      const double z                    = phase_value / phi_max_safe;
      const double tanh_z               = std::tanh(z);
      const double regularized_phase    = phi_max_safe * tanh_z;
      const double regularized_jacobian = 1. - tanh_z * tanh_z;
      const double denominator =
        1. - gamma * gamma * regularized_phase * regularized_phase;
      const double support            = 1. / denominator;
      const double support_derivative = 2. * gamma * gamma * regularized_phase *
                                        regularized_jacobian /
                                        (denominator * denominator);

      return {phase_value * support, support + phase_value * support_derivative};
    }

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
     * Cahn-Hilliard moving-mesh forcing source term of the pseudosolid
     * equation, f = compression * eps * factor(phi) * grad phi. This is the
     * "chns form" path (as opposed to a user-defined custom source term, which
     * goes through CurrentMeshSourceAssembler).
     *
     * It is meant to support two ways of providing the phase marker:
     *  - field mode  : phi/psi is an unknown FE field (full CHNS-ALE solver and
     *                  the enlarged presolver). The Jacobian is exact and
     *                  Hessian-free (material gradient remapping -G^T grad phi).
     *  - function mode: phi is an analytic function to evaluate (the ALE
     *                  elasticity presolver). The Jacobian is exact thanks to
     *                  the analytic Hessian of phi (symbolic differentiation).
     * Only the function mode (ScratchDataElasticity) is implemented for now.
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

      // Custom (user-defined) source term on the current mesh.
      if (param.elasticity.enable_source_term_on_current_mesh)
        assemblers.emplace_back(
          std::make_unique<
            CurrentMeshSourceAssembler<dim, ScratchData, CopyData>>(param,
                                                                    ordering));

      // TODO: wire the "custom" mff source term path, where the moving-mesh
      // forcing is a user-defined expression assembled through
      // CurrentMeshSourceAssembler. For now only "off" and "chns form" are
      // functional; "custom" is parsed but assembles no forcing.

      // Cahn-Hilliard moving-mesh forcing ("chns form" path): function mode for
      // the elasticity presolver (analytic phi) and field mode for the full
      // CHNS-ALE solver (FE phi). Both share this assembler.
      const bool with_chns_form_forcing =
        param.cahn_hilliard.mff_source_term ==
        Parameters::CahnHilliard<dim>::MeshForcingSourceTerm::chns_form;

      if constexpr (std::is_same_v<ScratchData, ScratchDataElasticity<dim>> ||
                    std::is_same_v<
                      ScratchData,
                      NavierStokesScratch::ScratchDataCHNS<dim, true>>)
      {
        if (with_chns_form_forcing)
          assemblers.emplace_back(
            std::make_unique<
              SourceFromCHNSTracerAssembler<dim, ScratchData, CopyData>>(
              param, ordering));
      }
    }
  } // namespace Elasticity
} // namespace Assembly

#endif
