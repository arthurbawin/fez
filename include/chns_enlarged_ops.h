#ifndef CHNS_ENLARGED_OPS_H
#define CHNS_ENLARGED_OPS_H

#include <assembly/chns_enlarged_forms.h>
#include <components_ordering.h>
#include <deal.II/base/table.h>
#include <deal.II/dofs/dof_tools.h>

#include <utility>
#include <vector>

using namespace dealii;

template <int dim, bool with_moving_mesh, bool with_enlarged>
struct CHNSEnlargedOps;

template <int dim, bool with_moving_mesh>
struct CHNSEnlargedOps<dim, with_moving_mesh, false>
{
  // No-op routing layer for the standard CHNS solver.
  static void extend_additional_variables_description(
    std::vector<std::pair<std::string, unsigned int>> & /*description*/)
  {}

  static void extend_coupling_table(
    const ComponentOrdering & /*ordering*/,
    const bool /*stabilization*/,
    Table<2, DoFTools::Coupling> & /*coupling_table*/)
  {}

  template <typename ScratchDataType>
  static const double &phase_marker_value(const ScratchDataType &scratch_data,
                                          const unsigned int     q)
  {
    return scratch_data.tracer_values[q];
  }

  template <typename ScratchDataType>
  static const Tensor<1, dim> &
  phase_marker_gradient(const ScratchDataType &scratch_data,
                        const unsigned int     q)
  {
    return scratch_data.tracer_gradients[q];
  }

  template <typename ScratchDataType>
  static const double &phase_shape_value(const ScratchDataType &scratch_data,
                                         const unsigned int     q,
                                         const unsigned int     j)
  {
    return scratch_data.shape_phi[q][j];
  }

  template <typename ScratchDataType>
  static const Tensor<1, dim> &
  phase_shape_gradient(const ScratchDataType &scratch_data,
                       const unsigned int     q,
                       const unsigned int     j)
  {
    return scratch_data.grad_shape_phi[q][j];
  }

  static unsigned int phase_component(const ComponentOrdering &ordering)
  {
    return ordering.phi_lower;
  }

  template <typename ScratchDataType, typename VectorType>
  static void assemble_rhs_terms(const ComponentOrdering & /*ordering*/,
                                 const ScratchDataType & /*scratch_data*/,
                                 const Parameters::CahnHilliard<dim> & /*cahn_hilliard*/,
                                 const double /*length_scale_sq*/,
                                 VectorType & /*local_rhs*/)
  {}

  template <typename ScratchDataType,
            typename CouplingTableType,
            typename MatrixType>
  static void assemble_matrix_terms(const ComponentOrdering & /*ordering*/,
                                    const CouplingTableType & /*coupling_table*/,
                                    const ScratchDataType   & /*scratch_data*/,
                                    const Parameters::CahnHilliard<dim> & /*cahn_hilliard*/,
                                    const double /*length_scale_sq*/,
                                    MatrixType & /*local_matrix*/)
  {}
};

template <int dim>
struct CHNSEnlargedOps<dim, true, true>
{
  // Lightweight routing layer for enlarged-only data and local forms.
  static void extend_additional_variables_description(
    std::vector<std::pair<std::string, unsigned int>> &description)
  {
    description.push_back({"psi", 1});
  }

  static void extend_coupling_table(const ComponentOrdering &ordering,
                                    const bool /*stabilization*/,
                                    Table<2, DoFTools::Coupling> &table)
  {
    table[ordering.psi_lower][ordering.phi_lower] = DoFTools::always;
    table[ordering.psi_lower][ordering.mu_lower]  = DoFTools::always;
    table[ordering.psi_lower][ordering.psi_lower] = DoFTools::always;
    for (unsigned int d = ordering.x_lower; d < ordering.x_upper; ++d)
      {
        table[ordering.psi_lower][d] = DoFTools::always;
        table[d][ordering.psi_lower] = DoFTools::always;
      }
  }

  template <typename ScratchDataType>
  static const double &phase_marker_value(const ScratchDataType &scratch_data,
                                          const unsigned int     q)
  {
    return scratch_data.psi_values[q];
  }

  template <typename ScratchDataType>
  static const Tensor<1, dim> &
  phase_marker_gradient(const ScratchDataType &scratch_data,
                        const unsigned int     q)
  {
    return scratch_data.psi_gradients[q];
  }

  template <typename ScratchDataType>
  static const double &phase_shape_value(const ScratchDataType &scratch_data,
                                         const unsigned int     q,
                                         const unsigned int     j)
  {
    return scratch_data.shape_psi[q][j];
  }

  template <typename ScratchDataType>
  static const Tensor<1, dim> &
  phase_shape_gradient(const ScratchDataType &scratch_data,
                       const unsigned int     q,
                       const unsigned int     j)
  {
    return scratch_data.grad_shape_psi[q][j];
  }

  static unsigned int phase_component(const ComponentOrdering &ordering)
  {
    return ordering.psi_lower;
  }

  template <typename ScratchDataType, typename VectorType>
  static void assemble_rhs_terms(const ComponentOrdering &ordering,
                                 const ScratchDataType   &scratch,
                                 const Parameters::CahnHilliard<dim> &cahn_hilliard,
                                 const double             length_scale_sq,
                                 VectorType              &local_rhs)
  {
    Assembly::assemble_psi_equation_rhs<dim>(
      ordering, scratch, cahn_hilliard, length_scale_sq, local_rhs);
  }

  template <typename ScratchDataType,
            typename CouplingTableType,
            typename MatrixType>
  static void assemble_matrix_terms(const ComponentOrdering &ordering,
                                    const CouplingTableType &coupling_table,
                                    const ScratchDataType   &scratch,
                                    const Parameters::CahnHilliard<dim> &cahn_hilliard,
                                    const double             length_scale_sq,
                                    MatrixType              &local_matrix)
  {
    Assembly::assemble_psi_equation_matrix<dim, true>(
      ordering,
      coupling_table,
      scratch,
      cahn_hilliard,
      length_scale_sq,
      local_matrix);
  }
};

#endif
