#ifndef UTILITIES_H
#define UTILITIES_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <parameters.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

using namespace dealii;

template <int dim>
struct PointComparator
{
  bool operator()(const Point<dim> &a, const Point<dim> &b) const
  {
    for (unsigned int d = 0; d < dim; ++d)
    {
      if (std::abs(a[d] - b[d]) > 1e-14)
        return a[d] < b[d];
    }
    return false;
  }
};

template <int dim>
struct PointEquality
{
  bool operator()(const Point<dim> &a, const Point<dim> &b) const
  {
    return a.distance(b) < 1e-14;
  }
};

/**
 * Perform a dry run to read the run-time problem dimension set in the
 * "Dimension" block of the given parameter file.
 */
inline unsigned int read_problem_dimension(const std::string &parameter_file)
{
  ParameterHandler prm;

  // Declare
  prm.enter_subsection("Dimension");
  prm.declare_entry("dimension",
                    "2",
                    Patterns::Integer(2, 3),
                    "Problem dimension (2 or 3)",
                    true);
  prm.leave_subsection();

  // Read only this structure from the file
  // Parse will fail if the "Dimension" block is not specified
  prm.parse_input(parameter_file,
                  /*last_line=*/"",
                  /*skip_undefined=*/true,
                  /*assert_mandatory_entries_are_found=*/true);

  // Parse
  prm.enter_subsection("Dimension");
  unsigned int dim = prm.get_integer("dimension");
  prm.leave_subsection();

  return dim;
}

/**
 * Perform a dry run to read the number of boundary conditions of each type.
 */
inline void
read_number_of_boundary_conditions(const std::string &parameter_file,
                                   Parameters::BoundaryConditionsData &bc_data)
{
  ParameterHandler prm;

  // Declare all possible boundary conditions.
  // They do not all need to be present in the parameter file.
  prm.enter_subsection("Fluid boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for the flow problem "
                      "(Navier-Stokes equations)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Pseudosolid boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for the pseudosolid mesh "
                      "movement problem");
  }
  prm.leave_subsection();

  prm.enter_subsection("CahnHilliard boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for two-phase flows with "
                      "the Cahn-Hilliard Navier-Stokes model");
  }
  prm.leave_subsection();

  prm.enter_subsection("Heat boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for the heat equation");
  }
  prm.leave_subsection();

  // Read only these structures from the file
  prm.parse_input(parameter_file, /*last_line=*/"", /*skip_undefined=*/true);

  // Parse
  prm.enter_subsection("Fluid boundary conditions");
  bc_data.n_fluid_bc = prm.get_integer("number");
  prm.leave_subsection();

  prm.enter_subsection("Pseudosolid boundary conditions");
  bc_data.n_pseudosolid_bc = prm.get_integer("number");
  prm.leave_subsection();

  prm.enter_subsection("CahnHilliard boundary conditions");
  bc_data.n_cahn_hilliard_bc = prm.get_integer("number");
  prm.leave_subsection();

  prm.enter_subsection("Heat boundary conditions");
  bc_data.n_heat_bc = prm.get_integer("number");
  prm.leave_subsection();
}

template <int dim>
inline Tensor<1, dim> parse_rank_1_tensor(const std::string &values,
                                          const std::string &delimiter = ",")
{
  const std::vector<double> parsed = Utilities::string_to_double(
    Utilities::split_string_list(values, delimiter));

  AssertThrow(parsed.size() == dim,
              ExcMessage("Could not read rank-1 tensor from input " + values));

  Tensor<1, dim> res;
  for (unsigned int d = 0; d < dim; ++d)
    res[d] = parsed[d];
  return res;
}

/**
 *
 */
inline std::pair<double, double>
compute_relative_error(const double A,
                       const double B,
                       const double tol_to_ignore = 1e-14)
{
  AssertThrow(
    std::isfinite(A) && std::isfinite(B),
    ExcMessage(
      "Taking relative error of values which are not finite or numbers."));

  const double abs_err = std::abs(B - A);

  // If both values are small enough, return 0 as relative error
  if (std::abs(A) < tol_to_ignore && std::abs(B) < tol_to_ignore)
    return std::make_pair(abs_err, 0.);

  const double rel_err = abs_err / std::max({std::abs(A), DBL_EPSILON});
  return std::make_pair(abs_err, rel_err);
}

/**
 * Compute the mean value of the given function f on the mesh.
 */
template <int dim>
double compute_global_mean_value(const Function<dim>   &f,
                                 const unsigned int     component,
                                 const DoFHandler<dim> &dof_handler,
                                 const Mapping<dim>    &mapping,
                                 const unsigned int     n_q_points = 4)
{
  double             I_local = 0., vol_local = 0.;
  QGaussSimplex<dim> quadrature(n_q_points);
  FEValues<dim>      fe_values(mapping,
                          dof_handler.get_fe(),
                          quadrature,
                          update_quadrature_points | update_JxW_values);

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        const Point<dim> &p = fe_values.quadrature_point(q);
        I_local += f.value(p, component) * fe_values.JxW(q);
        vol_local += fe_values.JxW(q);
      }
    }

  // Reduce across all processes
  const double I_global =
    Utilities::MPI::sum(I_local, dof_handler.get_mpi_communicator());
  const double vol_global =
    Utilities::MPI::sum(vol_local, dof_handler.get_mpi_communicator());

  return I_global / vol_global;
}

/**
 * A wrapper to subtract the mean pressure from a given function,
 * typically the exact solution.
 */
template <int dim>
class PressureMeanSubtractedFunction : public Function<dim>
{
public:
  PressureMeanSubtractedFunction(const Function<dim> &base_function,
                                 const double         mean_pressure,
                                 const unsigned int   p_lower)
    : Function<dim>(base_function.n_components)
    , base(base_function)
    , mean_pressure(mean_pressure)
    , p_lower(p_lower)
  {}

  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override
  {
    if (component == p_lower)
      return base.value(p, component) - mean_pressure;
    else
      return base.value(p, component);
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int component = 0) const override
  {
    return base.gradient(p, component);
  }

private:
  const Function<dim> &base;
  const double         mean_pressure;
  const unsigned int   p_lower;
};

/**
 * Compute the measure (surface or length) of a given boundary.
 */
template <int dim>
double compute_boundary_volume(const DoFHandler<dim>     &dof_handler,
                               const Mapping<dim>        &mapping,
                               const Quadrature<dim - 1> &face_quadrature,
                               const types::boundary_id   boundary_id)
{
  double I = 0.;

  FEFaceValues<dim> fe_face_values(mapping,
                                   dof_handler.get_fe(),
                                   face_quadrature,
                                   update_JxW_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == boundary_id)
        {
          fe_face_values.reinit(cell, f);
          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            I += fe_face_values.JxW(q);
        }
  return Utilities::MPI::sum(I, dof_handler.get_communicator());
}

/**
 * Same as above but in a hp context
 */
template <int dim>
double compute_boundary_volume(const DoFHandler<dim>            &dof_handler,
                               const hp::MappingCollection<dim> &mapping,
                               const hp::QCollection<dim - 1> &face_quadrature,
                               const types::boundary_id        boundary_id)
{
  double I = 0.;

  hp::FEFaceValues<dim> hp_fe_face_values(mapping,
                                          dof_handler.get_fe_collection(),
                                          face_quadrature,
                                          update_JxW_values);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == boundary_id)
        {
          const unsigned int fe_index = cell->active_fe_index();
          hp_fe_face_values.reinit(cell, f);
          const FEFaceValues<dim> &fe_face_values =
            hp_fe_face_values.get_present_fe_values();
          for (unsigned int q = 0; q < face_quadrature[fe_index].size(); ++q)
            I += fe_face_values.JxW(q);
        }
  return Utilities::MPI::sum(I, dof_handler.get_communicator());
}

/**
 * Rename temporary files containing the root "temporary_filename_prefix" to the
 * root "final_filename_prefix", while maintaining the suffixes. This is used to
 * overwrite the checkpoint save files, without risking to corrupt the previous
 * files, see also deal.II step 83.
 */
void replace_temporary_files(const std::string directory,
                             const std::string temporary_filename_prefix,
                             const std::string final_filename_prefix,
                             const MPI_Comm   &mpi_communicator);

/**
 * Fill the vector dofs_to_component, which contains for each relevant dof its
 * component index, similarly to what is done in
 * DoFTools::internal::get_component_association.
 */
template <int dim>
void fill_dofs_to_component(const DoFHandler<dim>      &dof_handler,
                            const IndexSet             &locally_relevant_dofs,
                            std::vector<unsigned char> &dofs_to_component);


#endif