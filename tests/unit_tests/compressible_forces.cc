#include <deal.II/distributed/shared_tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <post_processing_handler.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "../tests.h"

using PP = Parameters::PostProcessing;

enum class ForceCase
{
  pressure,
  shear,
  divergence
};

constexpr double            viscosity = 3.;
constexpr double            density   = 2.75; // Deliberately distinct from one.
const std::array<double, 3> lengths   = {{2., 3., 4.}};

template <int dim>
class ForceFields : public Function<dim>
{
public:
  ForceFields(const ForceCase test_case)
    : Function<dim>(dim + 2)
    , test_case(test_case)
  {}

  double value(const Point<dim> &p, const unsigned int c) const override
  {
    if (c == dim)
      return test_case == ForceCase::pressure ? 6. : 0.;
    if (c == dim + 1)
      return 100. + p[0]; // Detect accidental use of temperature as pressure.
    if (c == 0 && test_case == ForceCase::shear)
      return 2. * p[1];
    if (c == 0 && test_case == ForceCase::divergence)
      return p[0];
    return 0.;
  }

private:
  const ForceCase test_case;
};

// Analytical force per unit area on an axis-aligned face. With mu=3:
// pressure: p=6; shear: tau_xy=tau_yx=6; divergence: tau=diag(4,-2[,-2]).
// This oracle does not reconstruct the production stress tensor.
template <int dim>
Tensor<1, dim> expected_traction(const ForceCase    test_case,
                                 const unsigned int boundary,
                                 const bool         compressible = true)
{
  Tensor<1, dim>     result;
  const unsigned int axis = boundary / 2;
  const double       sign = boundary % 2 ? 1. : -1.;
  if (test_case == ForceCase::pressure)
    result[axis] = sign * 6. * (compressible ? 1. : density);
  else if (test_case == ForceCase::shear)
  {
    if (axis == 0)
      result[1] = -sign * 6.;
    if (axis == 1)
      result[0] = -sign * 6.;
  }
  else
    result[axis] = compressible ? sign * (axis == 0 ? -4. : 2.) :
                                  sign * (axis == 0 ? -6. : 0.);
  return result;
}

template <int dim>
double face_area(const unsigned int boundary)
{
  double area = 1.;
  for (unsigned int d = 0; d < dim; ++d)
    if (d != boundary / 2)
      area *= lengths[d];
  return area;
}

void assert_error(const double local_error, const std::string &context)
{
  const double error =
    Utilities::MPI::max(std::isfinite(local_error) ?
                          local_error :
                          std::numeric_limits<double>::infinity(),
                        MPI_COMM_WORLD);
  AssertThrow(std::isfinite(error) && error < 1e-10,
              ExcMessage(context +
                         ": force mismatch, error=" + std::to_string(error)));
}

template <int dim>
void check_handler(const ForceCase                       test_case,
                   const bool                            compressible,
                   const bool                            use_hp,
                   parallel::shared::Triangulation<dim> &tria,
                   const DoFHandler<dim>                &dofs,
                   const MappingQ1<dim>                 &mapping,
                   const hp::MappingCollection<dim>     &mappings,
                   const hp::QCollection<dim - 1>       &quadratures,
                   const LA::ParVectorType              &solution)
{
  Parameters::BoundaryConditionsData bc{};
  ParameterReader<dim>               param(bc);
  ParameterHandler                   prm;
  param.postprocessing.declare_parameters(prm);
  param.time_integration.declare_parameters(prm);
  prm.enter_subsection("Time integration");
  prm.set("scheme", "stationary");
  prm.leave_subsection();
  param.postprocessing.read_parameters(prm);
  param.time_integration.read_parameters(prm);
  param.time_integration.n_time_intervals          = 1;
  param.time_integration.n_steady_adaptation_steps = 0;
  param.output.write_results                       = false;
  param.output.skin.write_results                  = false;
  param.physical_properties.fluids.resize(1);
  param.physical_properties.fluids[0].density           = density;
  param.physical_properties.fluids[0].dynamic_viscosity = viscosity;
  auto &pp                                              = param.postprocessing;
  pp.forces.enable                                      = true;
  pp.forces.write_results                               = false;
  pp.forces.verbosity = Parameters::Verbosity::quiet;
  pp.forces.method    = PP::Forces::ComputationMethod::stress_vector;
  pp.forces.precision = 16;
  pp.forces.boundary_ids.clear();
  for (unsigned int id = 0; id < 2 * dim; ++id)
    pp.forces.boundary_ids.push_back(id);
  pp.slices.enable                   = true;
  pp.slices.write_results            = false;
  pp.slices.verbosity                = Parameters::Verbosity::quiet;
  pp.slices.boundary_ids             = {0, 1};
  pp.slices.along_which_axis         = "y";
  pp.slices.n_slices                 = 2;
  pp.slices.compute_forces_on_slices = true;

  ComponentOrdering ordering;
  ordering.n_components = dim + 2;
  ordering.u_lower      = 0;
  ordering.u_upper      = dim;
  ordering.p_lower      = dim;
  ordering.p_upper      = dim + 1;
  if (compressible)
  {
    ordering.t_lower = dim + 1;
    ordering.t_upper = dim + 2;
  }
  const TimeHandler          time(param.time_integration);
  PostProcessingHandler<dim> handler(ordering, param, tria, dofs, {});
  handler.output_fields(mapping, solution, time, {}); // Assign slice indices.
  const QGauss<dim - 1> quadrature(2);
  for (unsigned int repeat = 0; repeat < 2; ++repeat)
    if (use_hp)
      handler.compute_forces(
        ordering, dofs, mappings, quadratures, solution, time);
    else
      handler.compute_forces(
        ordering, dofs, mapping, quadrature, solution, time);

  double error = 0.;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ostringstream output;
    handler.write_forces(output);
    std::istringstream table(output.str());
    std::string        header;
    std::getline(table, header);
    for (unsigned int row = 0; row < 4 * dim; ++row)
    {
      double       t;
      unsigned int id;
      if (!(table >> t >> id) || !std::isfinite(t) || id != row % (2 * dim))
      {
        error = std::numeric_limits<double>::infinity();
        break;
      }
      const auto expected = face_area<dim>(id) *
                            expected_traction<dim>(test_case, id, compressible);
      for (unsigned int d = 0; d < dim; ++d)
      {
        double actual;
        if (!(table >> actual) || !std::isfinite(actual))
        {
          error = std::numeric_limits<double>::infinity();
          break;
        }
        error = std::max(error, std::abs(actual - expected[d]));
      }
    }
    std::string extra;
    if (table >> extra)
      error = std::numeric_limits<double>::infinity();
  }
  assert_error(error,
               compressible ? "Compressible handler dispatch" :
                              "Incompressible pressure scaling");
}

template <int dim>
void check_dimension(const bool use_hp)
{
  parallel::shared::Triangulation<dim> tria(MPI_COMM_WORLD);
  Point<dim>                           lower, upper;
  for (unsigned int d = 0; d < dim; ++d)
    upper[d] = lengths[d];
  GridGenerator::hyper_rectangle(tria, lower, upper, true);
  tria.refine_global(1);
  hp::FECollection<dim> elements;
  elements.push_back(FESystem<dim>(FE_Q<dim>(1), dim + 2));
  elements.push_back(FESystem<dim>(FE_Q<dim>(2), dim + 2));
  DoFHandler<dim> dofs(tria);
  if (use_hp)
  {
    for (const auto &cell : dofs.active_cell_iterators())
      if (cell->is_locally_owned())
        cell->set_active_fe_index(cell->center()[0] > 1. ? 1 : 0);
    dofs.distribute_dofs(elements);
  }
  else
    dofs.distribute_dofs(elements[0]);
  const MappingQ1<dim>       mapping;
  hp::MappingCollection<dim> mappings;
  mappings.push_back(mapping);
  mappings.push_back(mapping);
  hp::QCollection<dim - 1> quadratures;
  quadratures.push_back(QGauss<dim - 1>(2));
  quadratures.push_back(QGauss<dim - 1>(3));
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);
  LA::ParVectorType owned(dofs.locally_owned_dofs(), MPI_COMM_WORLD);
  LA::ParVectorType solution(dofs.locally_owned_dofs(),
                             DoFTools::extract_locally_relevant_dofs(dofs),
                             MPI_COMM_WORLD);

  for (const auto test_case :
       {ForceCase::pressure, ForceCase::shear, ForceCase::divergence})
  {
    if (use_hp)
      VectorTools::interpolate(mappings,
                               dofs,
                               ForceFields<dim>(test_case),
                               owned);
    else
      VectorTools::interpolate(mapping,
                               dofs,
                               ForceFields<dim>(test_case),
                               owned);
    owned.compress(VectorOperation::insert);
    solution = owned;
    solution.update_ghost_values();
    std::vector<Tensor<1, dim>> per_face(tria.n_faces());
    for (auto &f : per_face)
      for (unsigned int d = 0; d < dim; ++d)
        f[d] = 123. + d;
    for (unsigned int repeat = 0; repeat < 2; ++repeat)
      for (unsigned int id = 0; id < 2 * dim; ++id)
      {
        const auto force =
          use_hp ? PostProcessingTools::compute_compressible_forces_on_boundary(
                     dofs,
                     mappings,
                     quadratures,
                     solution,
                     id,
                     velocity,
                     pressure,
                     viscosity,
                     per_face) :
                   PostProcessingTools::compute_compressible_forces_on_boundary(
                     dofs,
                     mapping,
                     QGauss<dim - 1>(2),
                     solution,
                     id,
                     velocity,
                     pressure,
                     viscosity,
                     per_face);
        const auto traction = expected_traction<dim>(test_case, id);
        double     error    = (force - face_area<dim>(id) * traction).norm();
        for (const auto &cell : dofs.active_cell_iterators())
          if (cell->is_locally_owned())
            for (const auto &face : cell->face_iterators())
              if (face->at_boundary() && face->boundary_id() == id)
                error = std::max(error,
                                 (per_face[face->index()] -
                                  face->measure() * traction)
                                   .norm());
        assert_error(error, "Analytical boundary and per-face forces");
      }
    const auto missing =
      use_hp ? PostProcessingTools::compute_compressible_forces_on_boundary(
                 dofs,
                 mappings,
                 quadratures,
                 solution,
                 99,
                 velocity,
                 pressure,
                 viscosity,
                 per_face) :
               PostProcessingTools::compute_compressible_forces_on_boundary(
                 dofs,
                 mapping,
                 QGauss<dim - 1>(2),
                 solution,
                 99,
                 velocity,
                 pressure,
                 viscosity,
                 per_face);
    assert_error(missing.norm(), "Absent boundary");
    check_handler(test_case,
                  true,
                  use_hp,
                  tria,
                  dofs,
                  mapping,
                  mappings,
                  quadratures,
                  solution);
    check_handler(test_case,
                  false,
                  use_hp,
                  tria,
                  dofs,
                  mapping,
                  mappings,
                  quadratures,
                  solution);
  }
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << dim << "d " << (use_hp ? "hp" : "standard")
            << " analytical forces, face reset, slices and dispatch OK"
            << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  try
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      initlog();
    check_dimension<2>(false);
    check_dimension<2>(true);
    check_dimension<3>(false);
    check_dimension<3>(true);
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
