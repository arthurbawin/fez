#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <assembly/pseudosolid_forms.h>
#include <h_target_fsi_solver.h>
#include <h_target_tools.h>

namespace
{
  template <int dim, typename TriangulationType>
  double
  average_reference_mesh_size(const TriangulationType &triangulation,
                              const MPI_Comm          comm,
                              const double            h_min)
  {
    double       local_measure = 0.;
    unsigned int local_cells   = 0;

    for (const auto &cell : triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        local_measure += cell->measure();
        ++local_cells;
      }

    const double global_measure =
      Utilities::MPI::sum(local_measure, comm);
    const unsigned int global_cells =
      Utilities::MPI::sum(local_cells, comm);

    const double average_measure =
      global_cells > 0 ? global_measure / global_cells : 0.;

    return std::max(HTargetTools::reference_size_from_cell_measure<dim>(
                      average_measure),
                    h_min * (1. + 1e-8));
  }

}

template <int dim>
FSISolverHTarget<dim>::FSISolverHTarget(const ParameterReader<dim> &param)
  : FSISolver<dim>(param)
{
  AssertThrow(param.h_target.enable_h_target_equation,
              ExcMessage("FSISolverHTarget requires the h_target equation."));

  if (param.finite_elements.use_quads)
    this->fe = std::make_unique<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,      // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree),            // Pressure
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim, // Position
      FE_Q<dim>(param.finite_elements.h_target_degree),            // h_target
      FE_Q<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier
  else
    this->fe = std::make_unique<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
        dim,                                                   // Velocity
      FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
      FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^
        dim,                                                   // Position
      FE_SimplexP<dim>(param.finite_elements.h_target_degree), // h_target
      FE_SimplexP<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier

  this->ordering = std::make_unique<ComponentOrderingFSIHTarget<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  this->position_extractor =
    FEValuesExtractors::Vector(this->ordering->x_lower);
  h_target_extractor =
    FEValuesExtractors::Scalar(this->ordering->h_lower);
  this->lambda_extractor =
    FEValuesExtractors::Vector(this->ordering->l_lower);

  this->velocity_mask =
    this->fe->component_mask(this->velocity_extractor);
  this->pressure_mask =
    this->fe->component_mask(this->pressure_extractor);
  this->position_mask =
    this->fe->component_mask(this->position_extractor);
  h_target_mask =
    this->fe->component_mask(h_target_extractor);
  this->lambda_mask =
    this->fe->component_mask(this->lambda_extractor);

  this->field_names_and_masks.clear();
  this->field_names_and_masks["velocity"]      = this->velocity_mask;
  this->field_names_and_masks["pressure"]      = this->pressure_mask;
  this->field_names_and_masks["mesh position"] = this->position_mask;
  this->field_names_and_masks["eta_h_target"]  = h_target_mask;

  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower,
    this->ordering->n_components);

  if (param.mms_param.enable)
  {
    this->exact_solution =
      std::make_shared<typename FSISolver<dim>::MMSSolution>(
        this->time_handler.current_time,
        *this->ordering,
        param.mms);

    this->source_terms =
      std::make_shared<typename FSISolver<dim>::MMSSourceTerm>(
        this->time_handler.current_time,
        *this->ordering,
        param.physical_properties,
        param.mms);
  }
  else
  {
    this->source_terms =
      std::make_shared<typename FSISolver<dim>::SourceTerm>(
        this->time_handler.current_time,
        *this->ordering,
        param.source_terms);

    this->exact_solution =
      std::make_shared<Functions::ZeroFunction<dim>>(
        this->ordering->n_components);
  }
}

template <int dim>
void FSISolverHTarget<dim>::create_scratch_data()
{
  this->scratch_data =
    std::make_unique<ScratchData>(*this->ordering,
                                  *this->fe,
                                  *this->fixed_mapping,
                                  *this->moving_mapping,
                                  *this->quadrature,
                                  *this->face_quadrature,
                                  this->time_handler,
                                  this->param);
}

template <int dim>
void FSISolverHTarget<dim>::set_solver_specific_initial_conditions()
{
  const double h_background =
    average_reference_mesh_size<dim>(this->triangulation,
                                     this->mpi_communicator,
                                     this->param.h_target.h_min);

  ConstantHTarget h_initial(
    *this->ordering,
    HTargetTools::unbounded_variable_from_target_size(
      h_background,
      this->param.h_target.h_min));

  VectorTools::interpolate(*this->moving_mapping,
                           this->dof_handler,
                           h_initial,
                           this->newton_update,
                           h_target_mask);
}

template <int dim>
void FSISolverHTarget<dim>::set_solver_specific_exact_solution()
{
  const double h_background =
    average_reference_mesh_size<dim>(this->triangulation,
                                     this->mpi_communicator,
                                     this->param.h_target.h_min);

  ConstantHTarget h_initial(
    *this->ordering,
    HTargetTools::unbounded_variable_from_target_size(
      h_background,
      this->param.h_target.h_min));

  VectorTools::interpolate(*this->moving_mapping,
                           this->dof_handler,
                           h_initial,
                           this->local_evaluation_point,
                           h_target_mask);
}

template <int dim>
void FSISolverHTarget<dim>::add_solver_specific_postprocessing_data()
{
  FSISolver<dim>::add_solver_specific_postprocessing_data();

  if (!this->postproc_handler->should_output_volume_fields(this->time_handler))
    return;

  Vector<float> h_target_cell(this->triangulation.n_active_cells());
  Vector<float> h_current_cell(this->triangulation.n_active_cells());
  Vector<float> p_size_cell(this->triangulation.n_active_cells());

  FEValues<dim> fe_values_moving(*this->moving_mapping,
                                 *this->fe,
                                 *this->quadrature,
                                 update_values | update_JxW_values);
  FEValues<dim> fe_values_fixed(*this->fixed_mapping,
                                *this->fe,
                                *this->quadrature,
                                update_values | update_JxW_values);

  std::vector<double> eta_h_values(this->quadrature->size());

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    fe_values_moving.reinit(cell);
    fe_values_fixed.reinit(cell);

    fe_values_fixed[h_target_extractor].get_function_values(this->present_solution,
                                                            eta_h_values);

    const double reference_cell_size =
      HTargetTools::reference_size_from_cell_measure<dim>(cell->measure());
    const double h_background =
      std::max(reference_cell_size, this->param.h_target.h_min * (1. + 1e-8));
    const double h_current_reference = reference_cell_size;

    double h_integral         = 0.0;
    double h_current_integral = 0.0;
    double p_size_integral    = 0.0;
    double volume             = 0.0;

    for (unsigned int q = 0; q < this->quadrature->size(); ++q)
    {
      const double h_q =
        HTargetTools::target_size_from_unbounded_variable(
          eta_h_values[q],
          this->param.h_target.h_min);

      const double JxW = fe_values_moving.JxW(q);

      h_integral += h_q * JxW;

      if (this->param.h_target.enable_mesh_concentration_stress)
      {
        const double J =
          std::max(fe_values_moving.JxW(q) /
                     std::max(fe_values_fixed.JxW(q), 1e-30),
                   1e-14);
        const double h_current =
          std::max(h_current_reference * std::pow(J, 1.0 / dim), 1e-14);

        const double lame_mu =
          this->param.physical_properties.pseudosolids[0].lame_mu_fun->value(
            cell->center());
        const double lame_lambda =
          this->param.physical_properties.pseudosolids[0].lame_lambda_fun->value(
            cell->center());
        const double c =
          Assembly::Pseudosolid::MeshConcentration::pressure_coefficient(
            HTargetTools::smooth_time_ramp(
              this->time_handler.current_time,
              this->param.h_target.mesh_concentration_ramp_time),
            this->param.h_target.size_pressure_coefficient,
            Assembly::Pseudosolid::MeshConcentration::
              equivalent_size_stiffness(
                this->param.physical_properties.pseudosolids[0],
                lame_mu,
                lame_lambda));

        const double p_size =
          Assembly::Pseudosolid::MeshConcentration::size_pressure(
            c,
            this->param.h_target.current_size_weight,
            h_background,
            h_current,
            h_q,
            this->param.h_target.h_min);

        h_current_integral += h_current * JxW;
        p_size_integral += p_size * JxW;
      }

      volume += JxW;
    }

    if (volume > 0.0)
    {
      h_target_cell[cell->active_cell_index()] =
        static_cast<float>(h_integral / volume);

      if (this->param.h_target.enable_mesh_concentration_stress)
      {
        h_current_cell[cell->active_cell_index()] =
          static_cast<float>(h_current_integral / volume);
        p_size_cell[cell->active_cell_index()] =
          static_cast<float>(p_size_integral / volume);
      }
    }
  }

  this->postproc_handler->add_cell_data_vector(h_target_cell,
                                               "h_target");
  if (this->param.h_target.enable_mesh_concentration_stress)
  {
    this->postproc_handler->add_cell_data_vector(h_current_cell,
                                                 "h_current");
    this->postproc_handler->add_cell_data_vector(
      p_size_cell,
      "mesh_concentration_pressure");
  }
}

template class FSISolverHTarget<2>;
template class FSISolverHTarget<3>;
