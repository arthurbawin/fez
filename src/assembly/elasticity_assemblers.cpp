
#include <assembly/elasticity_assemblers.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/symmetric_tensor.h>
#include <parameter_reader.h>
#include <scratch_data.h>
#include <scratch_data_elasticity.h>

namespace Assembly
{
  namespace Elasticity
  {
    template <int dim, typename ScratchData, typename CopyData>
    void LinearElasticityAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd        = scratch_data;
      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      const SymmetricTensor<2, dim> identity_tensor =
        unit_symmetric_tensor<dim>();

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const auto  &grad_x       = sd.present_position_gradients[q];
        const double div_x        = trace(grad_x);
        const auto   strain       = symmetrize(grad_x) - identity_tensor;
        const double trace_strain = div_x - (double)dim;

        const auto &source_term_position = sd.source_term_position[q];

        const auto &phi_x          = sd.phi_x[q];
        const auto &sym_grad_phi_x = sd.sym_grad_phi_x[q];
        const auto &div_phi_x      = sd.div_phi_x[q];

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          if (!ordering.is_position(sd.components[i]))
            continue;

          // Linear elasticity and source term
          // FIXME: more efficient double contraction
          local_rhs(i) -= (lambda * trace_strain * div_phi_x[i] +
                           2. * mu * scalar_product(strain, sym_grad_phi_x[i]) +
                           phi_x[i] * source_term_position) *
                          JxW_fixed;
        }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void LinearElasticityAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      const auto x_lower = ordering.x_lower;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const auto &grad_phi_x = sd.grad_phi_x[q];
        const auto &div_phi_x  = sd.div_phi_x[q];

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          const unsigned int comp_i = sd.components[i];
          if (!ordering.is_position(comp_i))
            continue;

          const auto  &grad_phi_x_i = grad_phi_x[i];
          const double div_phi_x_i  = div_phi_x[i];

          for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
          {
            const unsigned int comp_j = sd.components[j];
            if (!ordering.is_position(comp_j))
              continue;

            const auto &gxi = grad_phi_x_i[comp_i - x_lower];
            const auto &gxj = grad_phi_x[j][comp_j - x_lower];

            double local_matrix_ij =
              lambda * div_phi_x[j] * div_phi_x_i

              // The following is the double contraction
              // 2. * mu * scalar_product(sym_grad_phi_x_j,
              // sym_grad_phi_x_i) explicited for the symmetric gradient of
              // Lagrange shape functions:
              + mu * gxi[comp_j - x_lower] * gxj[comp_i - x_lower];
            if (comp_i == comp_j)
              local_matrix_ij += mu * gxi * gxj;

            local_matrix(i, j) += local_matrix_ij * JxW_fixed;
          }
        }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void NeoHookeanAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd        = scratch_data;
      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const double lnJ     = std::log(sd.present_position_J[q]);
        const auto  &F       = sd.present_position_gradients[q];
        const auto  &F_inv_T = sd.present_position_inverse_gradients_T[q];

        const auto &source_term_position = sd.source_term_position[q];

        const auto &phi_x      = sd.phi_x[q];
        const auto &grad_phi_x = sd.grad_phi_x[q];

        // First Piola-Kirchhoff stress tensor
        const auto P = mu * (F - F_inv_T) + lambda * lnJ * F_inv_T;

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          if (ordering.is_position(sd.components[i]))
            // FIXME: more efficient double contraction
            local_rhs(i) -= (scalar_product(grad_phi_x[i], P) +
                             phi_x[i] * source_term_position) *
                            JxW_fixed;
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void NeoHookeanAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      std::vector<Tensor<2, dim>> variation_piola(sd.dofs_per_cell);

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const double lnJ     = std::log(sd.present_position_J[q]);
        const auto  &F_inv   = sd.present_position_inverse_gradients[q];
        const auto  &F_inv_T = sd.present_position_inverse_gradients_T[q];

        const auto &grad_phi_x = sd.grad_phi_x[q];

        // Precompute the variations of the Piola tensor
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto &grad_phi_x_j = grad_phi_x[j];

          variation_piola[j] = mu * grad_phi_x_j +
                               (mu - lambda * lnJ) *
                                 (F_inv_T * transpose(grad_phi_x_j) * F_inv_T) +
                               lambda * trace(F_inv * grad_phi_x_j) * F_inv_T;
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          if (ordering.is_position(sd.components[i]))
          {
            const auto &grad_phi_x_i = grad_phi_x[i];
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
              if (ordering.is_position(sd.components[j]))
                local_matrix(i, j) +=
                  scalar_product(variation_piola[j], grad_phi_x_i) * JxW_fixed;
          }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    OgdenHyperelasticityAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd        = scratch_data;
      auto &local_rhs = copy_data.local_rhs(sd.active_fe_index);

      const double beta = param.physical_properties.pseudosolids[0].ogden_beta;
      Assert(std::abs(beta) > 1e-14, ExcInternalError());

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const double Jm_beta = std::pow(sd.present_position_J[q], -beta);
        const auto  &F       = sd.present_position_gradients[q];
        const auto  &F_inv_T = sd.present_position_inverse_gradients_T[q];

        const double volumetric_stress = (1. - Jm_beta) / beta;

        const auto &source_term_position = sd.source_term_position[q];

        const auto &phi_x      = sd.phi_x[q];
        const auto &grad_phi_x = sd.grad_phi_x[q];

        // First Piola-Kirchhoff stress tensor
        const auto P =
          mu * (F - F_inv_T) + lambda * volumetric_stress * F_inv_T;

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          if (ordering.is_position(sd.components[i]))
            // FIXME: more efficient double contraction
            local_rhs(i) -= (scalar_product(grad_phi_x[i], P) +
                             phi_x[i] * source_term_position) *
                            JxW_fixed;
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    OgdenHyperelasticityAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      const double beta = param.physical_properties.pseudosolids[0].ogden_beta;

      std::vector<Tensor<2, dim>> variation_piola(sd.dofs_per_cell);

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double JxW_fixed = sd.JxW_fixed[q];
        const double mu        = sd.lame_mu[q];
        const double lambda    = sd.lame_lambda[q];

        const double Jm_beta = std::pow(sd.present_position_J[q], -beta);
        const auto  &F_inv   = sd.present_position_inverse_gradients[q];
        const auto  &F_inv_T = sd.present_position_inverse_gradients_T[q];

        const double volumetric_stress = (1. - Jm_beta) / beta;

        const auto &grad_phi_x = sd.grad_phi_x[q];

        // Precompute the variations of the Piola tensor
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
        {
          const auto          &grad_phi_x_j = grad_phi_x[j];
          const Tensor<2, dim> dF_inv_T =
            -F_inv_T * transpose(grad_phi_x_j) * F_inv_T;

          variation_piola[j] =
            mu * (grad_phi_x_j - dF_inv_T) +
            lambda * (Jm_beta * trace(F_inv * grad_phi_x_j) * F_inv_T +
                      volumetric_stress * dF_inv_T);
        }

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          if (ordering.is_position(sd.components[i]))
          {
            const auto &grad_phi_x_i = grad_phi_x[i];
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
              if (ordering.is_position(sd.components[j]))
                local_matrix(i, j) +=
                  scalar_product(variation_piola[j], grad_phi_x_i) * JxW_fixed;
          }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    CurrentMeshSourceAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto &sd           = scratch_data;
      auto &local_matrix = copy_data.local_matrix(sd.active_fe_index);

      std::vector<Tensor<1, dim>> grad_source_dot_phi_x_j(sd.dofs_per_cell);

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
      {
        const double          JxW_fixed = sd.JxW_fixed[q];
        const auto           &phi_x     = sd.phi_x[q];
        const Tensor<2, dim> &grad_source_current_mesh =
          sd.grad_source_term_position_current_mesh[q];

        // Precompute the product grad_source_current_mesh * phi_x_j
        for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
          if (ordering.is_position(sd.components[j]))
            grad_source_dot_phi_x_j[j] = grad_source_current_mesh * phi_x[j];

        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          if (ordering.is_position(sd.components[i]))
          {
            const auto &phi_x_i = phi_x[i];
            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
              if (ordering.is_position(sd.components[j]))
                local_matrix(i, j) +=
                  phi_x_i * grad_source_dot_phi_x_j[j] * JxW_fixed;
          }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    SourceFromCHNSTracerAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      // Function mode: phi is the prescribed (analytic) phase, evaluated on the
      // current mesh. The forcing is f = compression * eps * factor(phi) * grad
      // phi, integrated against the position test functions. In the enlarged
      // presolver (hybrid third case) the widened marker psi is an FE field, so
      // an extra enlarged compression C_psi * L * factor_eq(psi) * grad psi is
      // added on top of the analytic-phi physical compression; the transport
      // term is intentionally OFF in the presolver.
      if constexpr (std::is_same_v<ScratchData, ScratchDataElasticity<dim>>)
      {
        auto        &sd        = scratch_data;
        auto        &local_rhs = copy_data.local_rhs(sd.active_fe_index);
        const double epsilon   = param.cahn_hilliard.epsilon_interface;
        const double gamma     = param.cahn_hilliard.mff_regularization_gamma;
        const double compression =
          param.cahn_hilliard.mff_physics_compression_factor *
          sd.chns_compression_multiplier;

        for (unsigned int q = 0; q < sd.n_q_points; ++q)
        {
          const double          phi      = sd.chns_tracer_values[q];
          const Tensor<1, dim> &grad_phi = sd.chns_tracer_gradients[q];
          Tensor<1, dim>        forcing =
            compression * epsilon * mesh_forcing_factor(phi, gamma).value *
            grad_phi;

          if (sd.with_enlarged_psi)
          {
            const double marker_epsilon =
              param.cahn_hilliard.psi_interface_width_factor * epsilon;
            const double enlarged_compression =
              param.cahn_hilliard.mff_enlarged_compression_factor *
              sd.chns_compression_multiplier;
            const MeshForcingFactor psi_factor = enlarged_mesh_forcing_factor(
              sd.psi_values[q],
              gamma,
              param.cahn_hilliard.mff_enlarged_factor_equalization_exponent);
            forcing += enlarged_compression * marker_epsilon * psi_factor.value *
                       sd.psi_gradients[q];
          }

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
            if (ordering.is_position(sd.components[i]))
              local_rhs(i) -= sd.phi_x[q][i] * forcing * sd.JxW_fixed[q];
        }
      }
      else
      {
        // Field mode: the phase markers are unknown finite-element fields on the
        // moving mesh (full CHNS-ALE solver). The physical compression term
        // f_phys = compression * eps * factor(phi) * grad phi always uses the
        // sharp tracer phi. In the enlarged solver the compression marker is the
        // widened field psi (with its own length scale L = factor * eps and the
        // equalized factor), and the transport term acts on psi instead of phi;
        // otherwise both use phi. The sign convention (rhs -=) matches the
        // elasticity source term and the presolver, so the presolved mesh is a
        // coherent equilibrium here.
        auto        &sd        = scratch_data;
        auto        &local_rhs = copy_data.local_rhs(sd.active_fe_index);
        const double epsilon   = param.cahn_hilliard.epsilon_interface;
        const double gamma     = param.cahn_hilliard.mff_regularization_gamma;
        const double physics_compression =
          param.cahn_hilliard.mff_physics_compression_factor;
        // Negated relative to the compression term so that the transport
        // factor keeps its usual sign convention (the whole forcing enters the
        // residual with rhs -=, which flips the compression factor's sign).
        const double transport = -param.cahn_hilliard.mff_transport_factor;

        for (unsigned int q = 0; q < sd.n_q_points; ++q)
        {
          const double            phi      = sd.tracer_values[q];
          const Tensor<1, dim>   &grad_phi = sd.tracer_gradients[q];
          const MeshForcingFactor phi_factor = mesh_forcing_factor(phi, gamma);

          const Tensor<1, dim> convective_velocity =
            sd.present_velocity_values[q] - sd.present_mesh_velocity_values[q];

          // Physical (sharp tracer) compression: always present.
          Tensor<1, dim> forcing =
            physics_compression * epsilon * phi_factor.value * grad_phi;

          if constexpr (ScratchData::enable_enlarged)
          {
            const double marker_epsilon =
              param.cahn_hilliard.psi_interface_width_factor * epsilon;
            const double enlarged_compression =
              param.cahn_hilliard.mff_enlarged_compression_factor;
            const double exponent =
              param.cahn_hilliard.mff_enlarged_factor_equalization_exponent;

            const double            psi      = sd.psi_values[q];
            const Tensor<1, dim>   &grad_psi = sd.psi_gradients[q];
            const MeshForcingFactor psi_factor =
              enlarged_mesh_forcing_factor(psi, gamma, exponent);

            forcing +=
              enlarged_compression * marker_epsilon * psi_factor.value * grad_psi;
            forcing += transport * marker_epsilon * marker_epsilon *
                       ((convective_velocity * grad_psi) * grad_psi);
          }
          else
          {
            forcing += transport * epsilon * epsilon *
                       ((convective_velocity * grad_phi) * grad_phi);
          }

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
            if (ordering.is_position(sd.components[i]))
              local_rhs(i) -= sd.phi_x[q][i] * forcing * sd.JxW_fixed[q];
        }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void
    SourceFromCHNSTracerAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      // Function mode: the forcing depends on the mesh position only through
      // phi(x(X)), so df/dx = compression * eps * (factor'(phi) grad phi (x)
      // grad phi + factor(phi) Hess phi), using the analytic gradient and
      // Hessian of the prescribed phase. The variation of the quadrature point
      // is the position shape function value phi_x[j].
      if constexpr (std::is_same_v<ScratchData, ScratchDataElasticity<dim>>)
      {
        auto        &sd           = scratch_data;
        auto        &local_matrix = copy_data.local_matrix(sd.active_fe_index);
        const double epsilon      = param.cahn_hilliard.epsilon_interface;
        const double gamma        = param.cahn_hilliard.mff_regularization_gamma;
        const double compression =
          param.cahn_hilliard.mff_physics_compression_factor *
          sd.chns_compression_multiplier;

        for (unsigned int q = 0; q < sd.n_q_points; ++q)
        {
          const double                   phi      = sd.chns_tracer_values[q];
          const Tensor<1, dim>          &grad_phi  = sd.chns_tracer_gradients[q];
          const SymmetricTensor<2, dim> &hess_phi  = sd.chns_tracer_hessians[q];
          const MeshForcingFactor        factor    = mesh_forcing_factor(phi, gamma);

          // x <- x from the analytic phi physical compression (exact via the
          // analytic Hessian of phi).
          const Tensor<2, dim> dforcing_dx =
            compression * epsilon *
            (factor.derivative * outer_product(grad_phi, grad_phi) +
             factor.value * Tensor<2, dim>(hess_phi));

          // Enlarged (hybrid) presolver: the psi compression is an FE field, so
          // it varies w.r.t. the mesh position through the ALE remapping of grad
          // psi (-G^T grad psi) and w.r.t. the psi dofs through the shapes.
          const bool   with_psi = sd.with_enlarged_psi;
          double       enlarged_compression = 0.;
          double       marker_epsilon       = 0.;
          MeshForcingFactor psi_factor{0., 0.};
          if (with_psi)
          {
            marker_epsilon =
              param.cahn_hilliard.psi_interface_width_factor * epsilon;
            enlarged_compression =
              param.cahn_hilliard.mff_enlarged_compression_factor *
              sd.chns_compression_multiplier;
            psi_factor = enlarged_mesh_forcing_factor(
              sd.psi_values[q],
              gamma,
              param.cahn_hilliard.mff_enlarged_factor_equalization_exponent);
          }

          const auto &phi_x = sd.phi_x[q];

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          {
            if (!ordering.is_position(sd.components[i]))
              continue;
            const auto &phi_x_i = phi_x[i];

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = sd.components[j];
              Tensor<1, dim>     dforcing;

              if (ordering.is_position(comp_j))
              {
                dforcing += dforcing_dx * phi_x[j];
                if (with_psi)
                {
                  const Tensor<2, dim> &G = sd.grad_phi_x_moving[q][j];
                  const Tensor<1, dim>  transported_grad_psi =
                    -transpose(G) * sd.psi_gradients[q];
                  dforcing += enlarged_compression * marker_epsilon *
                              psi_factor.value * transported_grad_psi;
                }
              }

              if (with_psi && ordering.is_psi(comp_j))
                dforcing += enlarged_compression * marker_epsilon *
                            (psi_factor.derivative * sd.shape_psi[q][j] *
                               sd.psi_gradients[q] +
                             psi_factor.value * sd.grad_shape_psi[q][j]);

              local_matrix(i, j) += phi_x_i * dforcing * sd.JxW_fixed[q];
            }
          }
        }
      }
      else
      {
        // Field mode: the markers are unknown FE fields, so the forcing varies
        // w.r.t. the marker (factor'(m) shape + factor(m) grad shape), the mesh
        // position (the moving-mesh gradient remaps as -G^T grad m), and the
        // velocity (transport term). The physical compression always linearizes
        // against the sharp tracer phi; in the enlarged solver the enlarged
        // compression and the transport linearize against psi (length scale L,
        // equalized factor) instead of phi, adding an x<->psi block. The sign
        // (matrix +=) matches the field mode rhs (rhs -=).
        auto        &sd           = scratch_data;
        auto        &local_matrix = copy_data.local_matrix(sd.active_fe_index);
        const double epsilon      = param.cahn_hilliard.epsilon_interface;
        const double gamma        = param.cahn_hilliard.mff_regularization_gamma;
        const double physics_compression =
          param.cahn_hilliard.mff_physics_compression_factor;
        // Negated relative to the compression term so that the transport
        // factor keeps its usual sign convention (the whole forcing enters the
        // residual with rhs -=, which flips the compression factor's sign).
        const double transport = -param.cahn_hilliard.mff_transport_factor;

        for (unsigned int q = 0; q < sd.n_q_points; ++q)
        {
          const double            phi      = sd.tracer_values[q];
          const Tensor<1, dim>   &grad_phi = sd.tracer_gradients[q];
          const MeshForcingFactor phi_factor = mesh_forcing_factor(phi, gamma);

          const Tensor<1, dim> convective_velocity =
            sd.present_velocity_values[q] - sd.present_mesh_velocity_values[q];
          const double velocity_dot_grad_phi = convective_velocity * grad_phi;

          // Transport acts on the marker: psi when enlarged, phi otherwise.
          double         marker_epsilon = epsilon;
          Tensor<1, dim> grad_marker    = grad_phi;
          if constexpr (ScratchData::enable_enlarged)
          {
            marker_epsilon =
              param.cahn_hilliard.psi_interface_width_factor * epsilon;
            grad_marker = sd.psi_gradients[q];
          }
          const double velocity_dot_grad_marker =
            convective_velocity * grad_marker;

          for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
          {
            if (!ordering.is_position(sd.components[i]))
              continue;
            const auto &test = sd.phi_x[q][i];

            for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
            {
              const unsigned int comp_j = sd.components[j];
              Tensor<1, dim>     forcing_variation;

              if (ordering.is_velocity(comp_j))
                forcing_variation +=
                  transport * marker_epsilon * marker_epsilon *
                  ((sd.phi_u[q][j] * grad_marker) * grad_marker);

              if (ordering.is_position(comp_j))
              {
                const Tensor<2, dim> &G = sd.grad_phi_x_moving[q][j];
                const Tensor<1, dim>  transported_grad_phi =
                  -transpose(G) * grad_phi;
                const Tensor<1, dim>  transported_grad_marker =
                  -transpose(G) * grad_marker;

                // Physical compression against the sharp tracer.
                forcing_variation += physics_compression * epsilon *
                                     phi_factor.value * transported_grad_phi;

                const Tensor<1, dim> convective_velocity_variation =
                  -sd.bdf_c0 * sd.phi_x[q][j];
                forcing_variation +=
                  transport * marker_epsilon * marker_epsilon *
                  ((convective_velocity_variation * grad_marker) * grad_marker +
                   (convective_velocity * transported_grad_marker) * grad_marker +
                   velocity_dot_grad_marker * transported_grad_marker);

                if constexpr (ScratchData::enable_enlarged)
                {
                  const double enlarged_compression =
                    param.cahn_hilliard.mff_enlarged_compression_factor;
                  const MeshForcingFactor psi_factor =
                    enlarged_mesh_forcing_factor(
                      sd.psi_values[q],
                      gamma,
                      param.cahn_hilliard
                        .mff_enlarged_factor_equalization_exponent);
                  forcing_variation += enlarged_compression * marker_epsilon *
                                       psi_factor.value *
                                       transported_grad_marker;
                }
              }

              if (ordering.is_tracer(comp_j))
              {
                const double          shape = sd.shape_phi[q][j];
                const Tensor<1, dim> &shape_gradient = sd.grad_shape_phi[q][j];
                forcing_variation +=
                  physics_compression * epsilon *
                  (phi_factor.derivative * shape * grad_phi +
                   phi_factor.value * shape_gradient);

                if constexpr (!ScratchData::enable_enlarged)
                  forcing_variation +=
                    transport * epsilon * epsilon *
                    ((convective_velocity * shape_gradient) * grad_phi +
                     velocity_dot_grad_phi * shape_gradient);
              }

              if constexpr (ScratchData::enable_enlarged)
                if (ordering.is_psi(comp_j))
                {
                  const double enlarged_compression =
                    param.cahn_hilliard.mff_enlarged_compression_factor;
                  const MeshForcingFactor psi_factor =
                    enlarged_mesh_forcing_factor(
                      sd.psi_values[q],
                      gamma,
                      param.cahn_hilliard
                        .mff_enlarged_factor_equalization_exponent);
                  const double          shape = sd.shape_psi[q][j];
                  const Tensor<1, dim> &shape_gradient =
                    sd.grad_shape_psi[q][j];
                  forcing_variation +=
                    enlarged_compression * marker_epsilon *
                    (psi_factor.derivative * shape * grad_marker +
                     psi_factor.value * shape_gradient);
                  forcing_variation +=
                    transport * marker_epsilon * marker_epsilon *
                    ((convective_velocity * shape_gradient) * grad_marker +
                     velocity_dot_grad_marker * shape_gradient);
                }

              local_matrix(i, j) +=
                test * forcing_variation * sd.JxW_fixed[q];
            }
          }
        }
      }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void PresolverPsiAssembler<dim, ScratchData, CopyData>::assemble_rhs(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      auto        &sd              = scratch_data;
      auto        &local_rhs       = copy_data.local_rhs(sd.active_fe_index);
      const double length_scale_sq = sd.psi_length_scale_sq;

      // Helmholtz residual of psi on the moving mesh, with the analytic phi_0
      // as source: R = w_psi (psi - phi_0) + L^2 grad(w_psi) . grad(psi).
      for (unsigned int q = 0; q < sd.n_q_points; ++q)
        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(sd.components[i]))
            continue;

          const double residual_i =
            sd.shape_psi[q][i] *
              (sd.psi_values[q] - sd.chns_tracer_values[q]) +
            length_scale_sq * dealii::scalar_product(sd.grad_shape_psi[q][i],
                                                     sd.psi_gradients[q]);

          local_rhs(i) -= residual_i * sd.JxW_moving[q];
        }
    }

    template <int dim, typename ScratchData, typename CopyData>
    void PresolverPsiAssembler<dim, ScratchData, CopyData>::assemble_matrix(
      const ScratchData &scratch_data,
      CopyData          &copy_data) const
    {
      using namespace dealii;
      auto        &sd              = scratch_data;
      auto        &local_matrix    = copy_data.local_matrix(sd.active_fe_index);
      const double length_scale_sq = sd.psi_length_scale_sq;

      for (unsigned int q = 0; q < sd.n_q_points; ++q)
        for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(sd.components[i]))
            continue;

          const double          test      = sd.shape_psi[q][i];
          const Tensor<1, dim> &grad_test = sd.grad_shape_psi[q][i];

          for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
          {
            const unsigned int comp_j   = sd.components[j];
            double             local_ij = 0.;

            // psi <- psi: Helmholtz operator (mass + L^2 stiffness).
            if (ordering.is_psi(comp_j))
              local_ij +=
                test * sd.shape_psi[q][j] +
                length_scale_sq *
                  scalar_product(grad_test, sd.grad_shape_psi[q][j]);

            // psi <- x: moving-domain variation. The psi value is nodal (does
            // not transform), but the analytic source phi_0(x) does, giving the
            // extra term -grad(phi_0).delta x; the value term also picks up the
            // JxW trace, and the Helmholtz stiffness term varies as a weak
            // laplacian under the ALE remapping grad -> grad - G^T grad.
            if (ordering.is_position(comp_j))
            {
              const Tensor<2, dim> &G       = sd.grad_phi_x_moving[q][j];
              const double          trG     = trace(G);
              const Tensor<1, dim> &delta_x = sd.phi_x[q][j];

              const double delta_phi_0 = sd.chns_tracer_gradients[q] * delta_x;

              local_ij +=
                test * (-delta_phi_0 +
                        (sd.psi_values[q] - sd.chns_tracer_values[q]) * trG);

              const Tensor<1, dim> dgrad_test = -transpose(G) * grad_test;
              const Tensor<1, dim> dgrad_psi =
                -transpose(G) * sd.psi_gradients[q];
              local_ij +=
                length_scale_sq *
                (scalar_product(dgrad_test, sd.psi_gradients[q]) +
                 scalar_product(grad_test, dgrad_psi) +
                 scalar_product(grad_test, sd.psi_gradients[q]) * trG);
            }

            local_matrix(i, j) += local_ij * sd.JxW_moving[q];
          }
        }
    }
  } // namespace Elasticity
} // namespace Assembly

// Explicit instantiations
#include "elasticity_assemblers.inst"
