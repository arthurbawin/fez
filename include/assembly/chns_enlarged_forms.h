#ifndef ASSEMBLY_CHNS_ENLARGED_FORMS_H
#define ASSEMBLY_CHNS_ENLARGED_FORMS_H

#include <components_ordering.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>

using namespace dealii;

namespace Assembly
{
  template <int dim, typename ScratchDataType, typename VectorType>
  inline void
  assemble_psi_equation_rhs(const ComponentOrdering &ordering,
                            const ScratchDataType   &scratch,
                            const double             length_scale_sq,
                            VectorType              &local_rhs)
  {
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double local_rhs_i =
            scratch.shape_psi[q][i] *
              (scratch.psi_values[q] - scratch.tracer_values[q] +
               scratch.source_term_psi[q]) +
            length_scale_sq * scalar_product(scratch.grad_shape_psi[q][i],
                                             scratch.psi_gradients[q]);

          local_rhs(i) -= local_rhs_i * scratch.JxW_moving[q];
        }
  }

  template <int dim,
            bool with_moving_mesh,
            typename ScratchDataType,
            typename CouplingTableType,
            typename MatrixType>
  inline void
  assemble_psi_equation_matrix(const ComponentOrdering &ordering,
                               const CouplingTableType &coupling_table,
                               const ScratchDataType   &scratch,
                               const double             length_scale_sq,
                               MatrixType              &local_matrix)
  {
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        {
          if (!ordering.is_psi(scratch.components[i]))
            continue;

          const double          phi_i  = scratch.shape_psi[q][i];
          const Tensor<1, dim> &grad_i = scratch.grad_shape_psi[q][i];

          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            {
              if (coupling_table[scratch.components[i]][scratch.components[j]] !=
                  DoFTools::always)
                continue;

              const unsigned int comp_j = scratch.components[j];
              double             local_matrix_ij = 0.;

              if (ordering.is_psi(comp_j))
                {
                  local_matrix_ij += phi_i * scratch.shape_psi[q][j];
                  local_matrix_ij +=
                    length_scale_sq *
                    scalar_product(grad_i, scratch.grad_shape_psi[q][j]);
                }

              if (ordering.is_tracer(comp_j))
                local_matrix_ij -= phi_i * scratch.shape_phi[q][j];

              if constexpr (with_moving_mesh)
                if (ordering.is_position(comp_j))
                  {
                    const Tensor<2, dim> &G   = scratch.grad_phi_x_moving[q][j];
                    const double          trG = trace(G);

                    local_matrix_ij +=
                      phi_i * (scratch.psi_values[q] - scratch.tracer_values[q]) * trG;
                    local_matrix_ij +=
                      length_scale_sq *
                      (scalar_product(-transpose(G) * grad_i,
                                      scratch.psi_gradients[q]) +
                       scalar_product(grad_i,
                                      -transpose(G) * scratch.psi_gradients[q]) +
                       scalar_product(grad_i, scratch.psi_gradients[q]) * trG);
                  }

              local_matrix(i, j) += local_matrix_ij * scratch.JxW_moving[q];
            }
        }
  }
} // namespace Assembly

#endif
