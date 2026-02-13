#ifndef SCRATCH_DATA_HEAT_H
#define SCRATCH_DATA_HEAT_H

#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <parameter_reader.h>
#include <types.h>

using namespace dealii;

/**
 * Small scratch data for the heat equation on fixed mesh.
 */
template <int dim>
class ScratchDataHeat
{
public:
  /**
   * Constructor
   */
  ScratchDataHeat(const FESystem<dim>        &fe,
                  const Mapping<dim>         &mapping,
                  const Quadrature<dim>      &cell_quadrature,
                  const Quadrature<dim - 1>  &face_quadrature,
                  const std::vector<double>  &bdf_coefficients,
                  const ParameterReader<dim> &param)
    : physical_properties(param.physical_properties)
    , fe_values(mapping,
                fe,
                cell_quadrature,
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , fe_face_values(mapping,
                     fe,
                     face_quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values |
                       update_normal_vectors)
    , n_q_points(cell_quadrature.size())
    , n_faces(fe.reference_cell().n_faces())
    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(fe.dofs_per_cell)
    , bdf_coefficients(bdf_coefficients)
  {
    temperature.component = 0;
    allocate();
  }

  /**
   * Copy constructor
   */
  ScratchDataHeat(const ScratchDataHeat &other)
    : physical_properties(other.physical_properties)
    , fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                other.fe_values.get_update_flags())
    , fe_face_values(other.fe_face_values.get_mapping(),
                     other.fe_face_values.get_fe(),
                     other.fe_face_values.get_quadrature(),
                     other.fe_face_values.get_update_flags())
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
    , bdf_coefficients(other.bdf_coefficients)
  {
    temperature.component = 0;
    allocate();
  }

private:
  void allocate()
  {
    JxW.resize(n_q_points);
    face_boundary_id.resize(n_faces);
    face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_normals.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    temperature_values.resize(n_q_points);
    temperature_gradients.resize(n_q_points);
    previous_temperature_values.resize(bdf_coefficients.size() - 1,
                                       std::vector<double>(n_q_points));
    phi_t.resize(n_q_points, std::vector<double>(dofs_per_cell));
    grad_phi_t.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));

    source_term_full.resize(n_q_points, Vector<double>(1));
    source_term_temperature.resize(n_q_points);
  }

public:
  template <typename VectorType>
  void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const VectorType                     &current_solution,
              const std::vector<VectorType>        &previous_solutions,
              const std::shared_ptr<Function<dim>> &source_terms,
              const std::shared_ptr<Function<dim>> & /*exact_solution*/)
  {
    fe_values.reinit(cell);

    /**
     * Volume contributions
     */
    fe_values[temperature].get_function_values(current_solution,
                                               temperature_values);
    fe_values[temperature].get_function_gradients(current_solution,
                                                  temperature_gradients);
    // Previous solutions
    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      fe_values[temperature].get_function_values(
        previous_solutions[i], previous_temperature_values[i]);

    source_terms->vector_value_list(fe_values.get_quadrature_points(),
                                    source_term_full);

    // Get jacobian, shape functions and set source terms
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW[q]                     = fe_values.JxW(q);
      source_term_temperature[q] = source_term_full[q](0);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_t[q][k]      = fe_values[temperature].value(k, q);
        grad_phi_t[q][k] = fe_values[temperature].gradient(k, q);
      }
    }

    /**
     * Face contributions
     */
    if (cell->at_boundary())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary())
        {
          face_boundary_id[i_face] = face->boundary_id();
          fe_face_values.reinit(cell, face);

          /**
           * TODO
           */
        }
      }
  }

private:
  Parameters::PhysicalProperties<dim> physical_properties;

  FEValues<dim>     fe_values;
  FEFaceValues<dim> fe_face_values;

public:
  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  const std::vector<double> bdf_coefficients;

  std::vector<double>                      JxW;
  std::vector<unsigned int>                face_boundary_id;
  std::vector<std::vector<double>>         face_JxW;
  std::vector<std::vector<Tensor<1, dim>>> face_normals;

  FEValuesExtractors::Scalar temperature;

  std::vector<double>              temperature_values;
  std::vector<Tensor<1, dim>>      temperature_gradients;
  std::vector<std::vector<double>> previous_temperature_values;

  std::vector<std::vector<double>>         phi_t;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_t;

  std::vector<Vector<double>> source_term_full;
  std::vector<double>         source_term_temperature;
};

#endif