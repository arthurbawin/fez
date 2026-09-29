#include <chns_phase_diffusion.h>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "../tests.h"

namespace
{
  template <int dim>
  using PointData = CahnHilliard::PhaseDiffusionPoint<dim>;

  template <int dim>
  PointData<dim> shifted(const PointData<dim> &point,
                         const PointData<dim> &direction,
                         const double          step)
  {
    auto result = point;
    result.tracer += step * direction.tracer;
    result.tracer_gradient += step * direction.tracer_gradient;
    result.tracer_hessian += step * direction.tracer_hessian;
    result.potential_gradient += step * direction.potential_gradient;
    result.potential_hessian += step * direction.potential_hessian;
    result.mobility += step * direction.mobility;
    result.mobility_gradient += step * direction.mobility_gradient;
    return result;
  }

  template <int dim>
  PointData<dim> base_point()
  {
    PointData<dim> point;
    point.tracer   = 0.35;
    point.mobility = 0.8;
    for (unsigned int i = 0; i < dim; ++i)
    {
      point.tracer_gradient[i]    = i == 0 ? 0.7 : -0.4;
      point.potential_gradient[i] = -0.6 + 0.5 * i;
      point.mobility_gradient[i]  = 0.2 - 0.15 * i;
      for (unsigned int j = 0; j < dim; ++j)
      {
        point.tracer_hessian[i][j] =
          i == j ? 0.11 * (i + 1.) : 0.07 * (i + j + 1.);
        point.potential_hessian[i][j] =
          i == j ? -0.17 + 0.12 * i : -0.09 * (i + j + 1.);
      }
    }
    return point;
  }

  template <int dim>
  std::vector<PointData<dim>> test_points(const double epsilon)
  {
    std::vector<PointData<dim>> points = {base_point<dim>()};
    const double tail = CahnHilliard::profile_correction_tail_scale();
    const double g0   = 1. / (std::sqrt(2.) * epsilon);

    auto tail_point               = base_point<dim>();
    tail_point.tracer             = -0.999;
    tail_point.tracer_gradient    = {};
    tail_point.tracer_gradient[0] = 0.1 * g0 * tail;
    points.push_back(tail_point);

    // Exercise the second derivatives of both quintic activations, first
    // separately and then together. Saturated activations cannot catch a
    // missing derivative of the cutoff weights.
    auto phase_transition   = base_point<dim>();
    phase_transition.tracer = -std::sqrt(1. - 0.375 * tail);
    points.push_back(phase_transition);

    auto gradient_transition               = base_point<dim>();
    gradient_transition.tracer_gradient    = {};
    gradient_transition.tracer_gradient[0] = 0.375 * g0 * tail;
    points.push_back(gradient_transition);

    auto both_transitions            = phase_transition;
    both_transitions.tracer_gradient = gradient_transition.tracer_gradient;
    points.push_back(both_transitions);

    auto zero_gradient            = base_point<dim>();
    zero_gradient.tracer_gradient = {};
    points.push_back(zero_gradient);

    for (const double tracer : {-1., 1., -1.01, 1.01})
    {
      auto inactive   = base_point<dim>();
      inactive.tracer = tracer;
      points.push_back(inactive);
    }

    // Jet of an exact tanh profile of a linear signed distance. The profile
    // flux and its divergence must vanish even though its Hessian is nonzero.
    PointData<dim> equilibrium;
    equilibrium.tracer             = -0.999;
    equilibrium.mobility           = 0.8;
    equilibrium.tracer_gradient[0] = g0 * (1. - 0.999 * 0.999);
    equilibrium.tracer_hessian[0][0] =
      -2. * equilibrium.tracer * g0 * equilibrium.tracer_gradient[0];
    points.push_back(equilibrium);
    return points;
  }

  template <int dim>
  double spatial_divergence_fd(const Parameters::CahnHilliard<dim> &parameters,
                               const PointData<dim>                &point,
                               const double                         reference,
                               const double                         step)
  {
    // Independent check: construct quadratic tracer/potential and linear
    // mobility fields with the prescribed jets, then differentiate the
    // EXISTING physical flux, without using the new divergence helper.
    double divergence = 0.;
    for (unsigned int d = 0; d < dim; ++d)
    {
      dealii::Tensor<1, dim> offset;
      offset[d]          = step;
      const auto flux_at = [&](const dealii::Tensor<1, dim> &x) {
        const double phi = point.tracer + point.tracer_gradient * x +
                           0.5 * x * (point.tracer_hessian * x);
        const auto gradient = point.tracer_gradient + point.tracer_hessian * x;
        const auto potential_gradient =
          point.potential_gradient + point.potential_hessian * x;
        const double mobility = point.mobility + point.mobility_gradient * x;
        return CahnHilliard::phase_diffusion_flux_driver<dim>(
          parameters, phi, gradient, potential_gradient, mobility, reference);
      };
      divergence += (flux_at(offset)[d] - flux_at(-offset)[d]) / (2. * step);
    }
    return divergence;
  }

  template <int dim>
  std::vector<PointData<dim>>
  independent_directions(const PointData<dim> &point, const double epsilon)
  {
    std::vector<PointData<dim>> directions;
    PointData<dim>              direction;
    const double tail = CahnHilliard::profile_correction_tail_scale();
    const double g0   = 1. / (std::sqrt(2.) * epsilon);

    // Scale probes inside narrow activation bands. This keeps a central
    // difference local without allowing cancellation to dominate it.
    direction.tracer = std::abs(point.tracer) > 0.99 ? 0.13 * tail : 0.13;
    directions.push_back(direction);
    for (unsigned int d = 0; d < dim; ++d)
    {
      direction = {};
      direction.tracer_gradient[d] =
        point.tracer_gradient.norm() < g0 * tail ? 0.17 * g0 * tail : 0.17;
      directions.push_back(direction);
      direction                       = {};
      direction.potential_gradient[d] = -0.23;
      directions.push_back(direction);
      direction                      = {};
      direction.mobility_gradient[d] = 0.19;
      directions.push_back(direction);
      for (unsigned int e = d; e < dim; ++e)
      {
        direction                      = {};
        direction.tracer_hessian[d][e] = 0.21;
        direction.tracer_hessian[e][d] = 0.21;
        directions.push_back(direction);
        direction                         = {};
        direction.potential_hessian[d][e] = -0.31;
        direction.potential_hessian[e][d] = -0.31;
        directions.push_back(direction);
      }
    }
    direction          = {};
    direction.mobility = 0.29;
    directions.push_back(direction);
    return directions;
  }

  template <int dim>
  void check_divergence_and_linearization(const std::string &mode)
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<dim> parameters;
    parameters.declare_parameters(prm);
    prm.enter_subsection("Cahn Hilliard");
    prm.set("interface profile correction", mode);
    prm.set("interface thickness", "0.6");
    prm.set("surface tension", "2.");
    prm.leave_subsection();
    parameters.read_parameters(prm);

    const auto points = test_points<dim>(parameters.epsilon_interface);
    for (const double reference : {0., 0.37})
      for (unsigned int k = 0; k < points.size(); ++k)
      {
        const auto &point = points[k];
        const auto  linearization =
          CahnHilliard::evaluate_phase_diffusion_divergence(parameters,
                                                            point,
                                                            reference);
        double best_divergence_error = std::numeric_limits<double>::max();
        for (const double step : {2e-6, 5e-7, 1e-7, 2e-8})
        {
          const double fd =
            spatial_divergence_fd(parameters, point, reference, step);
          best_divergence_error =
            std::min(best_divergence_error,
                     std::abs(linearization.value - fd) /
                       std::max(1., std::abs(linearization.value)));
        }
        AssertThrow(best_divergence_error < 3e-7,
                    ExcMessage(mode +
                               ": div(K) differs from spatial flux FD, "
                               "dim=" +
                               std::to_string(dim) + ", point=" +
                               std::to_string(k) + ", scaled error=" +
                               std::to_string(best_divergence_error)));

        if (k + 1 == points.size())
          AssertThrow(std::abs(linearization.value) < 1e-13,
                      ExcMessage("Exact planar tanh profile has nonzero "
                                 "profile-flux divergence"));

        const auto directions =
          independent_directions(point, parameters.epsilon_interface);
        for (unsigned int j = 0; j < directions.size(); ++j)
        {
          const auto  &direction       = directions[j];
          const double derivative      = linearization.variation(direction);
          double best_derivative_error = std::numeric_limits<double>::max();
          for (const double step : {2e-4, 5e-5, 1e-5})
          {
            const double plus =
              CahnHilliard::evaluate_phase_diffusion_divergence(
                parameters, shifted(point, direction, step), reference)
                .value;
            const double minus =
              CahnHilliard::evaluate_phase_diffusion_divergence(
                parameters, shifted(point, direction, -step), reference)
                .value;
            const double fd = (plus - minus) / (2. * step);
            best_derivative_error =
              std::min(best_derivative_error,
                       std::abs(derivative - fd) /
                         std::max(1., std::abs(derivative)));
          }
          AssertThrow(best_derivative_error < 5e-7,
                      ExcMessage(
                        mode +
                        ": div(K) variation differs from FD, "
                        "dim=" +
                        std::to_string(dim) + ", point=" + std::to_string(k) +
                        ", direction=" + std::to_string(j) + ", scaled error=" +
                        std::to_string(best_derivative_error)));
        }
      }
    deallog << "Phase diffusion " << mode
            << " divergence and all variations in " << dim << "D OK"
            << std::endl;
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  for (const std::string mode : {"none", "profile", "profile_flux"})
  {
    check_divergence_and_linearization<2>(mode);
    check_divergence_and_linearization<3>(mode);
  }
}
