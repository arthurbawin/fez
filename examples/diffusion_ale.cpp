
#include <Mesh.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <cmath>
#include <fstream>
#include <iostream>

/**
 * Solves the Poisson equation on a moving grid.
 * The Poisson equation is either steady or unsteady,
 * and the mesh movement is prescribed by a manufactured solution
 * for the displacement and solving the linear elasticity equation.
 */

// Diffusion equation
#define CONDUCTIVITY 1. // 0.123

// Pseudo-solid
#define LAMBDA_PS 1. // 1.234
#define MU_PS 1.     // 2.987

// Enable time-dependent BC for mesh displacement
#define WITH_TRANSIENT_DISPLACEMENT

// Enable dudt in Poisson equation
#define WITH_TRANSIENT_POISSON

// Enable ALE (mesh velocity term w cdot grad(u) in Poisson equation)
// If WITH_TRANSIENT_POISSON is defined, then WITH_ALE should
// be enabled. It can be disabled for debug purposes.
#define WITH_ALE

// Set a quadratic displacement (if not, use chi from Hay et al.)
#define LINEAR_DISPLACEMENT
// #define QUADRATIC_DISPLACEMENT

#define SOLVE_FOR_POSITION

namespace NS_MMS
{
  using namespace dealii;

  double G_fun(const double t)
  {
    return 1.;
    // return 1.23 + t;
    // return t*t;
    // return t*t*t;
    // return -sin(2. * M_PI * t);
  }

  double Gdot_fun(const double t)
  {
    return 0.;
    // return 1.;
    // return 2.*t;
    // return 3.*t*t;
    // return -2 * M_PI * cos(2. * M_PI * t);
  }

  double phi_fun(const double t)
  {
#if defined(WITH_TRANSIENT_DISPLACEMENT)
    // return t;
    // return t*t;
    return (t > 0) ? 4. * (t - tanh(t)) : 0.;
    // return sin(M_PI / 2. * t);
    // return t*t/10.;
#else
    return 1.;
#endif
  }

  double phidot_fun(const double t)
  {
#if defined(WITH_TRANSIENT_DISPLACEMENT)
    // return 1.;
    // return 2.*t;
    return (t > 0) ? 4. * tanh(t) * tanh(t) : 0.;
    // return M_PI / 2. * cos(M_PI / 2. * t);
    // return t/5.;
#else
    return 0.;
#endif
  }

  template <int dim>
  double u_fun(const double G, const Point<dim> &p)
  {
    const double x = p[0];
    const double y = p[1];

    if constexpr (dim == 2)
      // return G;
      // return G * x;
      // return G * y;
      return G / (2. * CONDUCTIVITY) * (y * (y - 1.) + x);
    else
      return G / (2. * CONDUCTIVITY) * (y * (y - 1.) + 2. * x + p[2]);
  }

  // Displacement manufactured solution
  template <int dim>
  double
  chi_fun(const double phi, const Point<dim> &p, const unsigned int component)
  {
#if defined(LINEAR_DISPLACEMENT)

    // Linear displacement in each component
    return phi / 4. * p[component];

#elif defined(QUADRATIC_DISPLACEMENT)
    // Quadratic displacement in each component
    if constexpr (dim == 2)
      return phi / 4. * p[component] * (p[component] - 1.);
    else
      return phi / 4. * p[component] * (p[component] - 1.);
#else
    // Displacement from Hay et al.
    if constexpr (dim == 2)
      return phi / 4. * p[0] * p[1] * (p[component] - 1.);
    else
      return phi / 4. * p[0] * p[1] * p[2] * (p[component] - 1.);
#endif
  }

  // Mesh position manufactured solution
  template <int dim>
  double
  pos_fun(const double phi, const Point<dim> &p, const unsigned int component)
  {
    return p[component] + chi_fun(phi, p, component);
  }

  // Mesh velocity from manufactured solution
  template <int dim>
  double w_fun(const double       phidot,
               const Point<dim>  &p,
               const unsigned int component)
  {
#if defined(LINEAR_DISPLACEMENT)
    
    // Linear displacement
    return phidot / 4. * p[component];

#elif defined(QUADRATIC_DISPLACEMENT)

    // Quadratic displacement in each component
    if constexpr (dim == 2)
      return phidot / 4. * p[component] * (p[component] - 1.);
    else
      return phidot / 4. * p[component] * (p[component] - 1.);
#else
    // Displacement from Hay et al.
    if constexpr (dim == 2)
      return phidot / 4. * p[0] * p[1] * (p[component] - 1.);
    else
      return phidot / 4. * p[0] * p[1] * p[2] * (p[component] - 1.);
#endif
  }

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t   = this->get_time();
      const double G   = G_fun(t);
      const double phi = phi_fun(t);

      if constexpr (dim == 2)
      {
        // Scalar field
        values[0] = u_fun(G, p);
        // Mesh position
        values[1] = pos_fun(phi, p, 0);
        values[2] = pos_fun(phi, p, 1);
      }
      else
      {
        // Scalar field
        values[0] = u_fun(G, p);
        // Mesh position
        values[1] = pos_fun(phi, p, 0);
        values[2] = pos_fun(phi, p, 1);
        values[3] = pos_fun(phi, p, 2);
      }
    }
  };

  template <int dim>
  class MeshVelocity : public Function<dim>
  {
  public:
    MeshVelocity(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t      = this->get_time();
      const double phidot = phidot_fun(t);

      if constexpr (dim == 2)
      {
        // Scalar field
        values[0] = 0.;
        // Mesh position
        values[1] = w_fun(phidot, p, 0);
        values[2] = w_fun(phidot, p, 1);
      }
      else
      {
        // Scalar field
        values[0] = 0.;
        // Mesh position
        values[1] = w_fun(phidot, p, 0);
        values[2] = w_fun(phidot, p, 1);
        values[3] = w_fun(phidot, p, 2);
      }
    }
  };

  template <int dim>
  class SourceTerm : public Function<dim>
  {
  public:
    SourceTerm(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t        = this->get_time();
      const double G        = G_fun(t);
      const double Gdot     = Gdot_fun(t);
      const double phi      = phi_fun(t);
      const double x        = p[0];
      const double y        = p[1];
      const double k        = CONDUCTIVITY;
      const double lambda_s = LAMBDA_PS;
      const double mu_s     = MU_PS;

      if constexpr (dim == 2)
      {
        // For u = G(t)
        // gradu[0]           = 0.;
        // gradu[1]           = 0.;
        // const double dudt_eulerian  = Gdot;
        // const double lap_u = 0.;

        // For u = G(t) * x
        // gradu[0] = G;
        // gradu[1] = 0.;
        // const double dudt_eulerian  = Gdot * x;
        // const double lap_u = 0.;

        // For u = G / (2. * CONDUCTIVITY) * (y * (y - 1.) + x);
        // gradu[0] = G / (2. * k);
        // gradu[1] = G / (2. * k) * (2.*y - 1.);
        const double dudt_eulerian  = Gdot / (2. * k) * (y * (y - 1.) + x);
        const double lap_u = G / k;

        // Poisson equation
#if defined(WITH_TRANSIENT_POISSON)
        values[0] = -(dudt_eulerian - k * lap_u);
#else
        values[0] = -(-k * lap_u);
#endif

        // Pseudo-solid
#if defined(LINEAR_DISPLACEMENT)
        values[1] = 0.;
        values[2] = 0.;
#elif defined(QUADRATIC_DISPLACEMENT)
        // Source term for quadratic displacement
        values[1] = (phi * (lambda_s + 2 * mu_s)) / 2;
        values[2] = (phi * (lambda_s + 2 * mu_s)) / 2;
#else
        // Source term for displacement from Hay et al.
        values[1] = -(phi / 4. * (lambda_s * (h - 4 * y) + mu_s * (h - 6 * y)));
        values[2] = -(phi / 4. * (lambda_s * (h - 4 * x) + mu_s * (h - 6 * x)));
#endif
      }
      else // dim = 3
      {
        const double z = p[2];

        // For u = G / (2. * CONDUCTIVITY) * (y * (y - 1.) + 2. * x + z);
        const double dudt_eulerian  = Gdot / (2. * k) * (y * (y - 1.) + 2. * x + z);
        const double lap_u = G / k;

        // Poisson equation
#if defined(WITH_TRANSIENT_POISSON)
        values[0] = -(dudt_eulerian - k * lap_u);
#else
        values[0] = -(-k * lap_u);
#endif

        // Pseudo-solid
#if defined(LINEAR_DISPLACEMENT)
        values[1] = 0.;
        values[2] = 0.;
        values[3] = 0.;
#elif defined(QUADRATIC_DISPLACEMENT)
        // Source term for quadratic displacement
        values[1] = phi / 2. * (2 * mu_s + lambda_s);
        values[2] = phi / 2. * (2 * mu_s + lambda_s);
        values[3] = phi / 2. * (2 * mu_s + lambda_s);
#else
        DEAL_II_NOT_IMPLEMENTED();
        // Source term for displacement from Hay et al.
        values[1] = -(phi / 4. * (lambda_s * (h - 4 * y) + mu_s * (h - 6 * y)));
        values[2] = -(phi / 4. * (lambda_s * (h - 4 * x) + mu_s * (h - 6 * x)));
#endif
      }
    }

    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      const double t      = this->get_time();
      const double G      = G_fun(t);
      const double Gdot   = Gdot_fun(t);
      const double phi    = phi_fun(t);
      const double x      = p[0];
      const double y      = p[1];
      const double k      = CONDUCTIVITY;

      if constexpr (dim == 2)
      {
        // Scalar field u
        const unsigned int u_lower = 0;
// #if defined(WITH_ALE)
// #  if defined(LINEAR_DISPLACEMENT)
        gradients[u_lower + 0][0] = 0.;
        gradients[u_lower + 0][1] = 0.;
// #elif defined(QUADRATIC_DISPLACEMENT)
//         gradients[u_lower + 0][0] = -(Gdot / (2.0 * k)) - (phi * G * (2.0 * x - 1.0)) / (8.0 * k);
//         gradients[u_lower + 0][1] =
//           -(Gdot * (2.0 * y - 1.0)) / (2.0 * k) -
//           (phi * G * (6.0 * y * y - 6.0 * y + 1.0)) / (8.0 * k);
// #  else
//         DEAL_II_NOT_IMPLEMENTED();
//         // Gradient of displacement from Hay et al.
//         gradients[u_lower + 0][0] = 0.;
//         gradients[u_lower + 0][1] = 0.;
// #  endif
// #else
//         // Gradient of G - ((x + y*(y - 1))*Gdot)/(2*k);
//         gradients[u_lower + 0][0] = -Gdot / (2 * k);
//         gradients[u_lower + 0][1] = -Gdot / (2 * k) * (2 * y - 1.);
//         // gradients[u_lower + 0][0] = 0.;
//         // gradients[u_lower + 0][1] = 0.;
// #endif

        // Gradient of position source term is not needed
        // as the integral of F cdot phi_x is always computed
        // on the initial mesh.
        const unsigned int x_lower = 1;
        gradients[x_lower + 0][0]  = 0.;
        gradients[x_lower + 0][1]  = 0.;

        gradients[x_lower + 1][0] = 0.;
        gradients[x_lower + 1][1] = 0.;
      }
      else // dim = 3
      {
        const double z = p[2];

        // Gradient of source term for scalar field u
        const unsigned int u_lower = 0;

        // Change if needed
        Tensor<1, dim> grad_k_lapu;
        grad_k_lapu[0] = 0.;
        grad_k_lapu[1] = 0.;
        grad_k_lapu[2] = 0.;

#if defined(WITH_TRANSIENT_POISSON)
        Tensor<1, dim> grad_dudt_eulerian;
        grad_dudt_eulerian[0] = Gdot / k;
        grad_dudt_eulerian[1] = Gdot / (2. * k) * (2.*y - 1.);
        grad_dudt_eulerian[1] = Gdot / (2. * k);

        gradients[u_lower + 0][0] = -(grad_dudt_eulerian[0] - grad_k_lapu[0]); 
        gradients[u_lower + 0][1] = -(grad_dudt_eulerian[1] - grad_k_lapu[1]);
        gradients[u_lower + 0][2] = -(grad_dudt_eulerian[2] - grad_k_lapu[2]);
# else
        gradients[u_lower + 0][0] = -(- grad_k_lapu[0]); 
        gradients[u_lower + 0][1] = -(- grad_k_lapu[1]);
        gradients[u_lower + 0][2] = -(- grad_k_lapu[2]);
# endif

        // Gradient of position source term is not needed
        // as the integral of F cdot phi_x is always computed
        // on the initial mesh.
        const unsigned int x_lower = 1;
        for(unsigned int d1 = 0; d1 < dim; ++d1)
          for(unsigned int d2 = 0; d2 < dim; ++d2)
            gradients[x_lower + d1][d2] = 0.;
      }
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    ScratchData(const FESystem<dim>       &fe,
                const Quadrature<dim>     &cell_quadrature,
                const Mapping<dim>        &fixed_mapping,
                const Mapping<dim>        &mapping,
                const Quadrature<dim - 1> &face_quadrature,
                const unsigned int         dofs_per_cell,
                const std::vector<double> &bdfCoeffs)
      : fe_values(mapping,
                  fe,
                  cell_quadrature,
                  update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_jacobians |
                    update_inverse_jacobians)
      , fe_values_fixed_mapping(fixed_mapping,
                  fe,
                  cell_quadrature,
                  update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_jacobians |
                    update_inverse_jacobians)
      , fe_face_values(mapping,
                       fe,
                       face_quadrature,
                       update_values | update_gradients |
                         update_quadrature_points | update_JxW_values |
                         update_jacobians | update_inverse_jacobians)
      , n_q_points(cell_quadrature.size())
      , n_faces_q_points(face_quadrature.size())
      , dofs_per_cell(dofs_per_cell)
      , bdfCoeffs(bdfCoeffs)
    {
      this->allocate();
    }

    void allocate();

  private:
    template <typename VectorType>
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const VectorType                   &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions,
                const Function<dim>                &source_term_fun)
    {
      active_fe_values->reinit(cell);

      for (const unsigned int i : active_fe_values->dof_indices())
        components[i] = active_fe_values->get_fe().system_to_component_index(i).first;

      const FEValuesExtractors::Scalar scalar_field(0);
      const FEValuesExtractors::Vector displacement(1);

      if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
      {
        (*active_fe_values)[scalar_field].get_function_values(current_solution,
                                                    present_field_values);
        (*active_fe_values)[scalar_field].get_function_gradients(current_solution,
                                                       present_field_gradients);
        (*active_fe_values)[displacement].get_function_values(
          current_solution, present_displacement_values);
        (*active_fe_values)[displacement].get_function_gradients(
          current_solution, present_displacement_gradients);
      }
      else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
      {
        (*active_fe_values)[scalar_field].get_function_values_from_local_dof_values(
          current_solution, present_field_values);
        (*active_fe_values)[scalar_field].get_function_gradients_from_local_dof_values(
          current_solution, present_field_gradients);
        (*active_fe_values)[displacement].get_function_values_from_local_dof_values(
          current_solution, present_displacement_values);
        (*active_fe_values)[displacement].get_function_gradients_from_local_dof_values(
          current_solution, present_displacement_gradients);
      }
      else
      {
        static_assert(false,
                      "reinit expects LA::MPI::Vector or std::vector<double>");
      }

      // Source term
      source_term_fun.vector_value_list(active_fe_values->get_quadrature_points(),
                                        source_term_full);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        source_term_field[q] = source_term_full[q](0);

        for (int d = 0; d < dim; ++d)
        {
          source_term_displacement[q][d] = source_term_full[q](d + 1);
        }
      }

      // Gradient of source term
      // Only need to fill in for the scalar field
      source_term_fun.vector_gradient_list(active_fe_values->get_quadrature_points(),
                                           grad_source_term_full);
      const unsigned int u_lower = 0;
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (int d = 0; d < dim; ++d)
          grad_source_field[q][d] = grad_source_term_full[q][u_lower][d];

      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        (*active_fe_values)[scalar_field].get_function_values(previous_solutions[i],
                                                    previous_field_values[i]);
        (*active_fe_values)[displacement].get_function_values(
          previous_solutions[i], previous_displacement_values[i]);
      }

      // Current mesh velocity from displacement
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        present_mesh_velocity_values[q] =
          bdfCoeffs[0] * present_displacement_values[q];
        for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        {
          present_mesh_velocity_values[q] +=
            bdfCoeffs[iBDF] * previous_displacement_values[iBDF - 1][q];
        }
      }

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW[q] = active_fe_values->JxW(q);
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]         = (*active_fe_values)[scalar_field].value(k, q);
          grad_phi_u[q][k]    = (*active_fe_values)[scalar_field].gradient(k, q);
          phi_disp[q][k]      = (*active_fe_values)[displacement].value(k, q);
          grad_phi_disp[q][k] = (*active_fe_values)[displacement].gradient(k, q);
          div_phi_disp[q][k]  = (*active_fe_values)[displacement].divergence(k, q);
        }
      }
    }

  public:
    template <typename VectorType>
    void reinit_current_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                   &current_solution,
      const std::vector<LA::MPI::Vector> &previous_solutions,
      const Function<dim>                &source_term_fun)
    {
      active_fe_values = &fe_values;
      this->reinit(cell, current_solution, previous_solutions, source_term_fun);
    }

    template <typename VectorType>
    void reinit_fixed_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                   &current_solution,
      const std::vector<LA::MPI::Vector> &previous_solutions,
      const Function<dim>                &source_term_fun)
    {
      active_fe_values = &fe_values_fixed_mapping;
      this->reinit(cell, current_solution, previous_solutions, source_term_fun);
    }

    const FEValues<dim> &get_current_fe_values() const
    {
      return fe_values;
    }
    const FEValues<dim> &get_fixed_fe_values() const
    {
      return fe_values_fixed_mapping;
    }

  public:
    FEValues<dim>    *active_fe_values;
    FEValues<dim>     fe_values;
    FEValues<dim>     fe_values_fixed_mapping;
    FEFaceValues<dim> fe_face_values;

    const unsigned int         n_q_points;
    const unsigned int         n_faces_q_points;
    const unsigned int         dofs_per_cell;
    const std::vector<double> &bdfCoeffs;

    std::vector<double> JxW;
    std::vector<double> face_JxW;

    std::vector<unsigned int> components;

    // Current and previous values and gradients for each quad node
    std::vector<double>              present_field_values;
    std::vector<Tensor<1, dim>>      present_field_gradients;
    std::vector<std::vector<double>> previous_field_values;

    std::vector<Tensor<1, dim>>              present_displacement_values;
    std::vector<Tensor<2, dim>>              present_displacement_gradients;
    std::vector<std::vector<Tensor<1, dim>>> previous_displacement_values;
    std::vector<Tensor<1, dim>>              present_mesh_velocity_values;

    // Source term on cell
    std::vector<Vector<double>>
                        source_term_full; // The source term with n_components
    std::vector<double> source_term_field;
    std::vector<Tensor<1, dim>> source_term_displacement;

    // Gradient of source term,
    // at each quad node, for each dof component, result is a Tensor<1, dim>
    std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
    std::vector<Tensor<1, dim>>              grad_source_field;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<double>>         phi_u;
    std::vector<std::vector<Tensor<1, dim>>> grad_phi_u;
    std::vector<std::vector<Tensor<1, dim>>> phi_disp;
    std::vector<std::vector<double>>         div_phi_disp;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_disp;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_field_values.resize(n_q_points);
    present_field_gradients.resize(n_q_points);
    present_displacement_values.resize(n_q_points);
    present_displacement_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);

    source_term_full.resize(n_q_points, Vector<double>(dim + 1));
    source_term_field.resize(n_q_points);
    source_term_displacement.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(dim + 1));
    grad_source_field.resize(n_q_points);

    // BDF
    previous_field_values.resize(2, std::vector<double>(n_q_points));
    previous_displacement_values.resize(
      2, std::vector<Tensor<1, dim>>(n_q_points));

    phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_disp.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_disp.resize(n_q_points,
                         std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_disp.resize(n_q_points, std::vector<double>(dofs_per_cell));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces_q_points);
  }

  class SimulationParameters
  {
  public:
    unsigned int field_degree;
    unsigned int displacement_degree;
    double       conductivity;
    double       pseudo_solid_mu;
    double       pseudo_solid_lambda;
    unsigned int bdf_order;
    double       t0;
    double       t1;
    double       dt;
    double       prev_dt;
    unsigned int nTimeSteps;
    unsigned int nConvergenceCycles;

  public:
    SimulationParameters(){};
  };

  template <int dim>
  class MMS
  {
  public:
    MMS(const SimulationParameters &param);

    void run();

  private:
    void set_bdf_coefficients(const unsigned int order);
    void make_grid(const unsigned int iMesh);
    void setup_system();
    void create_zero_constraints();
    void create_nonzero_constraints();
    void create_sparsity_pattern();
    void set_initial_condition();
    void apply_nonzero_constraints();
    void set_exact_solution();

    void assemble_matrix(bool first_step);
    void assemble_local_matrix(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      bool                                                  distribute);
    void assemble_local_matrix_fd(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      Vector<double>                                       &ref_local_rhs,
      Vector<double>                                       &perturbed_local_rhs,
      std::vector<double>                                  &cell_dof_values);
    void assemble_local_matrix_pseudo_solid(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      bool                                                  distribute);

    void assemble_rhs(bool first_step);
    void assemble_local_rhs(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      Vector<double>                                       &local_rhs,
      std::vector<double>                                  &cell_dof_values,
      bool                                                  distribute,
      bool update_cell_dof_values,
      bool use_full_solution);
    void assemble_local_rhs_pseudo_solid(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      Vector<double>                                       &local_rhs,
      std::vector<double>                                  &cell_dof_values,
      bool                                                  distribute,
      bool update_cell_dof_values,
      bool use_full_solution);

    void solve_direct(bool first_step);
    void solve_newton();
    void output_results(const unsigned int convergence_index,
                        const unsigned int time_step,
                        const int          n_newton_iter = -1);
    void compute_errors(const unsigned int time_step);
    void reset();

    SimulationParameters param;

    MPI_Comm           mpi_communicator;
    const unsigned int mpi_rank;

    FESystem<dim> fe;

    // Ordering of the FE system
    // Each field is in the half-open [lower, upper)
    // Check for matching component by doing e.g.:
    // if(u_lower <= comp && comp < u_upper)
    const unsigned int n_components = dim + 1;
    const unsigned int u_lower      = 0;
    const unsigned int u_upper      = 1;
    const unsigned int x_lower      = 1;
    const unsigned int x_upper      = dim + 1;

  public:
    bool is_field(const unsigned int component) const
    {
      return u_lower <= component && component < u_upper;
    }
    bool is_position_or_displacement(const unsigned int component) const
    {
      return x_lower <= component && component < x_upper;
    }

  public:
    QSimplex<dim> quadrature;
    QSimplex<dim-1> face_quadrature;

    parallel::fullydistributed::Triangulation<dim> triangulation;
    std::unique_ptr<Mapping<dim>>                  fixed_mapping;
    std::unique_ptr<Mapping<dim>>                  mapping;

    // Description of the .msh mesh entities
    std::map<unsigned int, std::string> mesh_domains_tag2name;
    std::map<std::string, unsigned int> mesh_domains_name2tag;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    LA::MPI::SparseMatrix system_matrix;

    // With ghosts (read only)
    LA::MPI::Vector present_solution;
    LA::MPI::Vector evaluation_point;

    // Without ghosts (owned)
    LA::MPI::Vector local_evaluation_point;
    LA::MPI::Vector newton_update;
    LA::MPI::Vector system_rhs;

    LA::MPI::Vector local_mesh_velocity;
    LA::MPI::Vector mesh_velocity;

    LA::MPI::Vector exact_solution;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    std::vector<Point<dim>> initial_mesh_position;
    IndexSet                pos_dof_indices;

    std::vector<LA::MPI::Vector> previous_solutions;
    std::vector<double>          bdfCoeffs;

    double current_time;

    Solution<dim>     solution_fun;
    SourceTerm<dim>   source_term_fun;
    MeshVelocity<dim> mesh_velocity_fun;

    // L1 in time, L2 in space
    double l2_err_u;
    double l2_err_disp;
    double l2_err_w;
    // Linf in space and time
    double           linf_error_u;
    double           linf_error_disp;
    double           linf_error_w;
    ConvergenceTable convergence_table;
  };

  template <int dim>
  MMS<dim>::MMS(const SimulationParameters &param)
    : param(param)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , fe(FE_SimplexP<dim>(param.field_degree), // Scalar field
         1,
         FE_SimplexP<dim>(
           param.displacement_degree), // Displacement or position
         dim)
    , quadrature(QGaussSimplex<dim>(4))
    , face_quadrature(QGaussSimplex<dim-1>(4))
    , triangulation(mpi_communicator)
    , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
    , dof_handler(triangulation)
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , current_time(param.t0)
    , solution_fun(Solution<dim>(current_time, n_components))
    , source_term_fun(SourceTerm<dim>(current_time, n_components))
    , mesh_velocity_fun(MeshVelocity<dim>(current_time, n_components))
  {}

  template <int dim>
  void MMS<dim>::set_bdf_coefficients(const unsigned int order)
  {
    const double dt      = param.dt;
    const double prev_dt = param.prev_dt;

    switch (order)
    {
      case 0: // Stationary
        bdfCoeffs.resize(1);
        bdfCoeffs[0] = 0.;
        break;
      case 1:
        bdfCoeffs.resize(2);
        bdfCoeffs[0] = 1. / dt;
        bdfCoeffs[1] = -1. / dt;
        break;
      case 2:
        bdfCoeffs.resize(3);
        // bdfCoeffs[0] =  3. / (2. * dt);
        // bdfCoeffs[1] = -2. / dt;
        // bdfCoeffs[2] =  1. / (2. * dt);
        bdfCoeffs[0] = 1.0 / dt + 1.0 / (dt + prev_dt);
        bdfCoeffs[1] = -1.0 / dt - 1.0 / (prev_dt);
        bdfCoeffs[2] = dt / prev_dt * 1. / (dt + prev_dt);
        break;
      default:
        throw std::runtime_error(
          "Can only choose BDF1 or BDF2 time integration method");
    }
  }

  template <int dim>
  void MMS<dim>::make_grid(const unsigned int iMesh)
  {
    pcout << "Making grid..." << std::endl;

    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      meshFile = "../data/meshes/square" + std::to_string(iMesh) + ".msh";
    }
    else
    {
      meshFile = "../data/meshes/cube" + std::to_string(iMesh) + ".msh";
    }

    std::ifstream input(meshFile);
    AssertThrow(input, ExcMessage("Could not open mesh file: " + meshFile));
    grid_in.read_msh(input);

    // Partition serial triangulation:
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(mpi_communicator), serial_tria);

    // Create building blocks:
    const TriangulationDescription::Description<dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_tria, mpi_communicator);

    // Create a fully distributed triangulation:
    // copy_triangulation does not seems to work, so maybe give reference to the
    // mesh
    triangulation.create_triangulation(description);

    // Save initial position of the mesh vertices
    initial_mesh_position.resize(triangulation.n_vertices());
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index     = cell->vertex_index(v);
        initial_mesh_position[global_vertex_index] = cell->vertex(v);
      }
    }

    read_gmsh_physical_names(meshFile,
                             mesh_domains_tag2name,
                             mesh_domains_name2tag);

    // Print mesh info
    if (mpi_rank == 0)
    {
      std::cout << "Mesh info:" << std::endl
                << " dimension: " << dim << std::endl
                << " no. of cells: " << serial_tria.n_active_cells()
                << std::endl;

      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : serial_tria.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;

      std::cout << " boundary indicators: ";
      for (const std::pair<const types::boundary_id, unsigned int> &pair :
           boundary_count)
      {
        std::cout << pair.first << '(' << pair.second << " times) ";
      }
      std::cout << std::endl;

      // Check that all boundary indices found in the mesh
      // have a matching name, to make sure we're not forgetting
      // a boundary.
      for (const auto &[id, count] : boundary_count)
      {
        if (mesh_domains_tag2name.count(id) == 0)
          throw std::runtime_error("Deal.ii read a boundary entity with tag " +
                                   std::to_string(id) +
                                   " in the mesh, but no physical entity with "
                                   "this tag was read from the mesh file.");
      }

      for (const auto &[id, name] : mesh_domains_tag2name)
        std::cout << "ID " << id << " -> " << name << "\n";
    }
  }

  template <int dim>
  void MMS<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);

    local_evaluation_point.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    local_mesh_velocity.reinit(locally_owned_dofs, mpi_communicator);
    mesh_velocity.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);

    exact_solution.reinit(locally_owned_dofs, mpi_communicator);

    // BDF solutions
    previous_solutions.clear();
    previous_solutions.resize(bdfCoeffs.size() - 1); // 1 or 2
    for (auto &previous_sol : previous_solutions)
    {
      previous_sol.clear();
      previous_sol.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);
    }

#if defined(SOLVE_FOR_POSITION)
    // Mesh position
    // Initialize directly from the triangulation
    // The parallel vector storing the mesh position is local_evaluation_point,
    // because this is the one to modify when computing finite differences.
    const FEValuesExtractors::Vector mesh_position(x_lower);
    VectorTools::get_position_vector(*fixed_mapping,
                                     dof_handler,
                                     local_evaluation_point,
                                     fe.component_mask(mesh_position));
    local_evaluation_point.compress(VectorOperation::insert);
    evaluation_point = local_evaluation_point;

    // Reset the mapping as a solution-dependent mapping
    mapping = std::make_unique<MappingFEField<dim, dim, LA::MPI::Vector>>(
      dof_handler, evaluation_point, fe.component_mask(mesh_position));
#endif
  }

  template <int dim>
  void MMS<dim>::create_sparsity_pattern()
  {
    //
    // Sparsity pattern and allocate matrix
    // After the constraints have been defined
    //
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    nonzero_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }

  template <int dim>
  void MMS<dim>::set_initial_condition()
  {
    const FEValuesExtractors::Scalar scalar_field(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Update mesh position *BEFORE* evaluating scalar field
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             solution_fun,
                             newton_update,
                             fe.component_mask(position));

    // Set scalar field with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             newton_update,
                             fe.component_mask(scalar_field));

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    present_solution = newton_update;

    // Dirty copy of the initial condition for BDF2 for now (-:
    for (auto &sol : previous_solutions)
      sol = present_solution;
  }

  template <int dim>
  void MMS<dim>::create_zero_constraints()
  {
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    // Boundaries where Dirichlet BC are applied,
    // where the Newton increment should be zero.
    const FEValuesExtractors::Scalar scalar_field(u_lower);
    VectorTools::interpolate_boundary_values(*mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             zero_constraints,
                                             fe.component_mask(scalar_field));

    const FEValuesExtractors::Vector disp_position(x_lower);
    VectorTools::interpolate_boundary_values(*fixed_mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             zero_constraints,
                                             fe.component_mask(disp_position));
    zero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector disp_position(x_lower);
    VectorTools::interpolate_boundary_values(*fixed_mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             solution_fun,
                                             nonzero_constraints,
                                             fe.component_mask(disp_position));

    const FEValuesExtractors::Scalar scalar_field(u_lower);
    VectorTools::interpolate_boundary_values(*mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             solution_fun,
                                             nonzero_constraints,
                                             fe.component_mask(scalar_field));
    nonzero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::apply_nonzero_constraints()
  {
    nonzero_constraints.distribute(local_evaluation_point);
    evaluation_point = local_evaluation_point;
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void MMS<dim>::set_exact_solution()
  {
    const FEValuesExtractors::Scalar scalar_field(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Update mesh position *BEFORE* evaluating scalar field
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point,
                             fe.component_mask(position));

    // Set scalar field with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point,
                             fe.component_mask(scalar_field));
    evaluation_point = local_evaluation_point;
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void MMS<dim>::assemble_matrix(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
    Vector<double>     ref_local_rhs(dofs_per_cell);
    Vector<double>     perturbed_local_rhs(dofs_per_cell);

    FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);

    // The local dofs values, which will be perturbed
    std::vector<double> cell_dof_values(dofs_per_cell);

    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 bdfCoeffs);

    // Scratch with fixed mapping for elasticity
    // ScratchData<dim> fixed_scratchData(fe,
    //                                    cell_quadrature,
    //                                    *fixed_mapping,
    //                                    face_quadrature,
    //                                    dofs_per_cell,
    //                                    bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute = true;
      this->assemble_local_matrix_pseudo_solid(first_step,
                                               cell,
                                               scratchData,
                                               evaluation_point,
                                               previous_solutions,
                                               local_dof_indices,
                                               local_matrix,
                                               distribute);
    }

    // Scratch with varying mapping for flow
    // ScratchData<dim> moving_scratchData(
    //   fe, cell_quadrature, *mapping, face_quadrature, dofs_per_cell, bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute = true;
      this->assemble_local_matrix(first_step,
                                  cell,
                                  scratchData,
                                  evaluation_point,
                                  previous_solutions,
                                  local_dof_indices,
                                  local_matrix,
                                  distribute);

      // std::cout << "Analytical non-elasticity matrix is " << std::endl;
      // local_matrix.print(std::cout, 10, 5);

      // // Compare with FD matrix
      // FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
      // FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);
      // Vector<double> ref_local_rhs(dofs_per_cell),
      // perturbed_local_rhs(dofs_per_cell);
      // this->assemble_local_matrix_fd(first_step, cell, moving_scratchData,
      // evaluation_point,
      //   previous_solutions, local_dof_indices, local_matrix_fd,
      //   ref_local_rhs, perturbed_local_rhs, cell_dof_values);

      // std::cout << "FD         non-elasticity matrix is " << std::endl;
      // local_matrix_fd.print(std::cout, 10, 5);

      // diff_matrix.equ(1.0, local_matrix);
      // diff_matrix.add(-1.0, local_matrix_fd);
      // std::cout << "Max difference is " << diff_matrix.linfty_norm() <<
      // std::endl;

      // throw std::runtime_error("Testing FD");
    }

    system_matrix.compress(VectorOperation::add);

    ////////////////////////////////////////////////////
    // Check that move_mesh is accounted for:
    // Print the individual cell areas (variable) and total area (constant)
    // double totalArea = 0.;
    // for (const auto &cell : dof_handler.active_cell_iterators() |
    //                           IteratorFilters::LocallyOwnedCell())
    // {
    //   scratchData.reinit(cell,
    //                      evaluation_point,
    //                      previous_solutions,
    //                      source_term_fun);
    //   double area = 0.;
    //   for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    //   {
    //     area += scratchData.JxW[q];
    //   }
    //   totalArea += area;
    //   std::cout << "Cell area is " << area << std::endl;
    // }
    // std::cout << "Total area is " << totalArea << std::endl;
    ////////////////////////////////////////////////////
  }

  template <int dim>
  void MMS<dim>::assemble_local_matrix_fd(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    Vector<double>                                       &ref_local_rhs,
    Vector<double>                                       &perturbed_local_rhs,
    std::vector<double>                                  &cell_dof_values)
  {
    throw std::runtime_error("FD matrix function must be modified in parallel first, "
      "to update evaluation_point as local_evaluation_point is modified");

    if (!cell->is_locally_owned())
      return;

    local_matrix        = 0.;
    ref_local_rhs       = 0.;
    perturbed_local_rhs = 0.;

    const double h = 1e-8;

    cell->get_dof_indices(local_dof_indices);

    const unsigned int dofs_per_cell = local_dof_indices.size();

    // Get the local dofs values
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      cell_dof_values[j] = evaluation_point[local_dof_indices[j]];

    // Compute the non-perturbed residual, do not distribute in global RHS
    // Actually: we can probably distribute here, and save an extra residual
    // computation
    bool distribute             = false;
    bool update_cell_dof_values = false;
    bool use_full_solution      = false;
    this->assemble_local_rhs(first_step,
                             cell,
                             scratchData,
                             current_solution,
                             previous_solutions,
                             local_dof_indices,
                             ref_local_rhs,
                             cell_dof_values,
                             distribute,
                             update_cell_dof_values,
                             use_full_solution);

    //
    // Solve for the mesh position, not displacement.
    // The non-position dofs are perturbed by modifying the local cell dof
    // values (cell_dof_values) The position dofs are perturbed by modyfing both
    // cell_dof_values and the local_evaluation_point, which then affects the
    // fe_values reinits through the MappingFEField.
    //
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      const unsigned int comp = fe.system_to_component_index(j).first;

      const double og_value = cell_dof_values[j];
      cell_dof_values[j] += h;

      if (is_position_or_displacement(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = cell_dof_values[j];
        local_evaluation_point.compress(VectorOperation::insert);
      }

      // Reinit is called in the local rhs function
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               current_solution,
                               previous_solutions,
                               local_dof_indices,
                               perturbed_local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);

      // Finite differences (with sign change as residual is -NL(u))
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
      }

      // Restore solution
      cell_dof_values[j] = og_value;
      if (is_position_or_displacement(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = og_value;
        local_evaluation_point.compress(VectorOperation::insert);
      }
    }

    // cell->get_dof_indices(local_dof_indices);
    // if (first_step)
    // {
    //   nonzero_constraints.distribute_local_to_global(local_matrix,
    //                                                  local_dof_indices,
    //                                                  system_matrix);
    // }
    // else
    // {
    //   zero_constraints.distribute_local_to_global(local_matrix,
    //                                               local_dof_indices,
    //                                               system_matrix);
    // }
  }

  template <int dim>
  void MMS<dim>::assemble_local_matrix(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit_current_mapping(cell,
                       current_solution,
                       previous_solutions,
                       source_term_fun);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];

      const auto &present_field_values = scratchData.present_field_values[q];
      const auto &present_field_gradients =
        scratchData.present_field_gradients[q];

      const auto &dxdt = scratchData.present_mesh_velocity_values[q];

#if defined(WITH_TRANSIENT_POISSON)
      // BDF: current dudt
      double dudt = bdfCoeffs[0] * present_field_values;
      for (unsigned int i = 1; i < bdfCoeffs.size(); ++i)
        dudt += bdfCoeffs[i] * scratchData.previous_field_values[i - 1][q];
#endif

      const auto &source_term_field = scratchData.source_term_field[q];
      const auto &grad_source_field = scratchData.grad_source_field[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_u      = is_field(component_i);
        // const bool         i_is_d = is_position_or_displacement(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_u      = is_field(component_j);
          const bool         j_is_d = is_position_or_displacement(component_j);

          double local_matrix_ij = 0.;

          // Scalar field - scalar field block
          if (i_is_u && j_is_u)
          {
#if defined(WITH_TRANSIENT_POISSON)
            // Time-dependent
            local_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];
#endif

            local_matrix_ij +=
              param.conductivity * grad_phi_u[i] * grad_phi_u[j];

#if defined(WITH_ALE)
            // ALE acceleration : - w dot grad(delta u)
            local_matrix_ij += - (dxdt * grad_phi_u[j]) * phi_u[i];
#endif
          }

          if (i_is_u && j_is_d)
          {
#if defined(WITH_TRANSIENT_POISSON)
            // Variation of time-dependent term with mesh position
            local_matrix_ij += dudt * phi_u[i] * trace(grad_phi_x[j]);
#endif

#if defined(WITH_ALE)
            // Variation of ALE term (dxdt cdot grad(u)) with mesh position
            local_matrix_ij +=
              - bdfCoeffs[0] * phi_x[j] * present_field_gradients * phi_u[i];
            local_matrix_ij +=
              - dxdt * (-present_field_gradients * grad_phi_x[j]) * phi_u[i];
            local_matrix_ij +=
              - dxdt * present_field_gradients * phi_u[i] * trace(grad_phi_x[j]);
#endif
            // Diffusion term:
            // Variation of the diffusion weak form with mesh position.
            // det J is accounted for at the end when multiplying by JxW(q).
            local_matrix_ij += param.conductivity *
                               (-present_field_gradients * grad_phi_x[j]) *
                               grad_phi_u[i];
            local_matrix_ij += param.conductivity * present_field_gradients *
                               (-grad_phi_u[i] * grad_phi_x[j]);
            local_matrix_ij += param.conductivity * present_field_gradients *
                               grad_phi_u[i] * trace(grad_phi_x[j]);

            // Source term:
            // Variation of the source term integral with mesh position.
            // det J is accounted for at the end when multiplying by JxW(q).
            local_matrix_ij += grad_source_field * phi_x[j] * phi_u[i];
            local_matrix_ij +=
              source_term_field * phi_u[i] * trace(grad_phi_x[j]);
          }

          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
      {
        nonzero_constraints.distribute_local_to_global(local_matrix,
                                                       local_dof_indices,
                                                       system_matrix);
      }
      else
      {
        zero_constraints.distribute_local_to_global(local_matrix,
                                                    local_dof_indices,
                                                    system_matrix);
      }
    }
  }

  template <int dim>
  void MMS<dim>::assemble_local_matrix_pseudo_solid(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit_fixed_mapping(cell,
                       current_solution,
                       previous_solutions,
                       source_term_fun);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_d = is_position_or_displacement(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_d = is_position_or_displacement(component_j);

          double local_matrix_ij = 0.;

          // Displacement - displacement block (chi-chi)
          if (i_is_d && j_is_d)
          {
            // Linear elasticity for pseudo-solid
            local_matrix_ij +=
              param.pseudo_solid_lambda * div_phi_x[j] * div_phi_x[i] +
              param.pseudo_solid_mu *
                scalar_product((grad_phi_x[i] + transpose(grad_phi_x[i])),
                               grad_phi_x[j]);
          }

          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
      {
        nonzero_constraints.distribute_local_to_global(local_matrix,
                                                       local_dof_indices,
                                                       system_matrix);
      }
      else
      {
        zero_constraints.distribute_local_to_global(local_matrix,
                                                    local_dof_indices,
                                                    system_matrix);
      }
    }
  }

  template <int dim>
  void MMS<dim>::assemble_rhs(bool first_step)
  {
    // pcout << "Assembling RHS" << std::endl;
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    // const QGaussSimplex<dim>     cell_quadrature(4);
    // const QGaussSimplex<dim - 1> face_quadrature(4);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    // ScratchData<dim> scratchData(
    //   fe, cell_quadrature, *mapping, face_quadrature, dofs_per_cell,
    //   bdfCoeffs);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<double> cell_dof_values(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Assemble pseudo-solid residual on initial mesh
    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 bdfCoeffs);
    // ScratchData<dim> fixed_scratchData(fe,
    //                                    cell_quadrature,
    //                                    *fixed_mapping,
    //                                    face_quadrature,
    //                                    dofs_per_cell,
                                       // bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute             = true;
      bool update_cell_dof_values = false;
      bool use_full_solution      = true;
      this->assemble_local_rhs_pseudo_solid(first_step,
                                            cell,
                                            scratchData,
                                            evaluation_point,
                                            previous_solutions,
                                            local_dof_indices,
                                            local_rhs,
                                            cell_dof_values,
                                            distribute,
                                            update_cell_dof_values,
                                            use_full_solution);
    }

    // Then assemble residuals on moving mesh
    // ScratchData<dim> moving_scratchData(
      // fe, cell_quadrature, *mapping, face_quadrature, dofs_per_cell, bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute             = true;
      bool update_cell_dof_values = false;
      bool use_full_solution      = true;
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               evaluation_point,
                               previous_solutions,
                               local_dof_indices,
                               local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);
    }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void MMS<dim>::assemble_local_rhs(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    Vector<double>                                       &local_rhs,
    std::vector<double>                                  &cell_dof_values,
    bool                                                  distribute,
    bool update_cell_dof_values,
    bool use_full_solution)
  {
    if (update_cell_dof_values)
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
        cell_dof_values[j] = local_evaluation_point[local_dof_indices[j]];
    }

    if (use_full_solution)
    {
      scratchData.reinit_current_mapping(cell,
                         current_solution,
                         previous_solutions,
                         source_term_fun);
    }
    else
      scratchData.reinit_current_mapping(cell,
                         cell_dof_values,
                         previous_solutions,
                         source_term_fun);

    local_rhs = 0;

    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<double>         field(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      // Evaluate the exact mesh velocity at this quad node for debug
      // Get physical position of quad node
      // const Point<dim> &quad_point = scratchData.get_current_fe_values().get_quadrature_points()[q];
      // Tensor<1, dim> w;
      // const double phidot = phidot_fun(this->current_time);
      // w[0] = w_fun(phidot, quad_point, 0);
      // w[1] = w_fun(phidot, quad_point, 1);

      const double JxW = scratchData.JxW[q];

      const auto &present_field_values = scratchData.present_field_values[q];
      const auto &present_field_gradients =
        scratchData.present_field_gradients[q];
#if defined(WITH_ALE)
      const auto &present_mesh_velocity_values =
        scratchData.present_mesh_velocity_values[q];
#endif
      const auto &source_term_field = scratchData.source_term_field[q];

      // BDF
      field[0]        = present_field_values;
      for (unsigned int i = 1; i < nBDF; ++i)
      {
        field[i]        = scratchData.previous_field_values[i - 1][q];
      }

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        double local_rhs_i =
          -(
#if defined(WITH_ALE)
            //
            // Mesh movement
            // The mesh velocity is evaluated on the moving mapping,
            // but it is defined on the fixed mapping :/
            // Does it matter? It seems to converge well as is.
            //
            // Actually, I believe this is correct:
            // the mesh velocity at this DoF is associated to
            // the moving mesh vertex, even though the mesh position
            // (and thus velocity) was computed on the fixed mesh.
            // This is dxdt|_X, mesh velocity at fixed initial position.
            //
            - phi_u[i] * present_mesh_velocity_values * present_field_gradients
            // - phi_u[i] * w * present_field_gradients
#endif

            // Poisson diffusion
            + param.conductivity * present_field_gradients * grad_phi_u[i]

            // Poisson source term
            + phi_u[i] * source_term_field) *
          JxW;

#if defined(WITH_TRANSIENT_POISSON)
        // Transient terms:
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
        {
          local_rhs_i -= bdfCoeffs[iBDF] * field[iBDF] * phi_u[i] * JxW;
        }
#endif

        local_rhs(i) += local_rhs_i;
      }
    }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
        nonzero_constraints.distribute_local_to_global(local_rhs,
                                                       local_dof_indices,
                                                       system_rhs);
      else
        zero_constraints.distribute_local_to_global(local_rhs,
                                                    local_dof_indices,
                                                    system_rhs);
    }
  }

  template <int dim>
  void MMS<dim>::assemble_local_rhs_pseudo_solid(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    Vector<double>                                       &local_rhs,
    std::vector<double>                                  &cell_dof_values,
    bool                                                  distribute,
    bool update_cell_dof_values,
    bool use_full_solution)
  {
    if (update_cell_dof_values)
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
        cell_dof_values[j] = local_evaluation_point[local_dof_indices[j]];
    }

    if (use_full_solution)
    {
      scratchData.reinit_fixed_mapping(cell,
                         current_solution,
                         previous_solutions,
                         source_term_fun);
    }
    else
      scratchData.reinit_fixed_mapping(cell,
                         cell_dof_values,
                         previous_solutions,
                         source_term_fun);

    local_rhs = 0;

    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<double>         field(nBDF);
    std::vector<Tensor<1, dim>> displacement(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_displacement_gradients =
        scratchData.present_displacement_gradients[q];

      const auto &source_term_displacement =
        scratchData.source_term_displacement[q];

      double present_displacement_divergence =
        trace(present_displacement_gradients);

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const auto present_displacement_gradient_sym =
          present_displacement_gradients +
          transpose(present_displacement_gradients);

        double local_rhs_i =
          -(
            // Linear elasticity
            +param.pseudo_solid_lambda * present_displacement_divergence *
              div_phi_x[i] +
            param.pseudo_solid_mu *
              scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

            // Linear elasticity source term
            + phi_x[i] * source_term_displacement) *
          JxW;

        local_rhs(i) += local_rhs_i;
      }
    }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
        nonzero_constraints.distribute_local_to_global(local_rhs,
                                                       local_dof_indices,
                                                       system_rhs);
      else
        zero_constraints.distribute_local_to_global(local_rhs,
                                                    local_dof_indices,
                                                    system_rhs);
    }
  }

  template <int dim>
  void MMS<dim>::solve_direct(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    newton_update = completely_distributed_solution;

    if (first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void MMS<dim>::solve_newton()
  {
    double             current_res = 1e6;
    double             norm_correction;
    bool               first_step = true;
    unsigned int       iter       = 0;
    const unsigned int max_iter   = 50;
    const double       tol        = 1e-9;
    bool               converged  = false;

    while (current_res > tol && iter <= max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_rhs(first_step);

      // If residual norm is low enough, return
      current_res = system_rhs.linfty_norm();
      if (current_res <= tol)
      {
        pcout << "Converged in " << iter
              << " iteration(s) because next nonlinear residual is below "
                 "tolerance: "
              << current_res << " < " << tol << std::endl;
        converged = true;
        break;
      }

      this->assemble_matrix(first_step);
      this->solve_direct(first_step);
      first_step = false;

      iter++;

      norm_correction = newton_update.linfty_norm(); // On this proc only!
      pcout << std::scientific << std::setprecision(8)
            << "Newton iteration: " << iter << " - ||du|| = " << norm_correction
            << " - ||NL(u)|| = " << current_res << std::endl;

      if (norm_correction > 1e10 || current_res > 1e10)
      {
        pcout << "Diverged after " << iter << " iteration(s)" << std::endl;
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          throw std::runtime_error("Nonlinear solver diverged");
      }

      local_evaluation_point = present_solution;
      local_evaluation_point.add(1., newton_update);
      nonzero_constraints.distribute(local_evaluation_point);
      evaluation_point = local_evaluation_point;

      //////////////////////////////////////////////
      // Recreate the nonzero constraints as the mesh may have moved
      // WHY DOES THIS WORK? (e.g., with u = x) :/
      this->create_nonzero_constraints();
      this->apply_nonzero_constraints();
      //////////////////////////////////////////////

      // this->assemble_rhs(first_step);
      // current_res = system_rhs.linfty_norm();

      // if (current_res <= tol)
      // {
      //   pcout << "Converged in " << iter
      //         << " iteration(s) because next nonlinear residual is below "
      //            "tolerance: "
      //         << current_res << " < " << tol << std::endl;
      //   converged = true;
      // }

      present_solution = evaluation_point;
    }

    if (!converged && iter == max_iter + 1)
    {
      pcout << "Did not converge after " << iter << " iteration(s)"
            << std::endl;
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        throw std::runtime_error("Nonlinear solver did not convege");
    }
  }

  template <int dim>
  void MMS<dim>::output_results(const unsigned int convergence_index,
                                const unsigned int time_step,
                                const int          n_newton_iter)
  {
    // Plot FE solution
    std::vector<std::string> solution_names(1, "scalar_field");
#if defined(SOLVE_FOR_POSITION)
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("mesh_position");
#else
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("displacement");
#endif

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        1, DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Plot exact solution
    std::vector<std::string> exact_solution_names(1, "exact_scalar_field");
#if defined(SOLVE_FOR_POSITION)
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_mesh_position");
#else
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_displacement");
#endif

    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             exact_solution);
    data_out.add_data_vector(exact_solution,
                             exact_solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    std::string root =
      "./diffusion_ale" + std::to_string(convergence_index) + "_" +
      std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) +
      "proc/";

    if (n_newton_iter == -1)
    {
      data_out.write_vtu_with_pvtu_record(
        root, "solution", time_step, mpi_communicator, 2);
    }
    else
    {
      data_out.write_vtu_with_pvtu_record(
        root, "solution", n_newton_iter, mpi_communicator, 2);
    }
  }

  template <int dim>
  void MMS<dim>::compute_errors(const unsigned int time_step)
  {
    const unsigned int n_active_cells = triangulation.n_active_cells();

    const ComponentSelectFunction<dim> scalar_field_mask(u_lower, n_components);
    const ComponentSelectFunction<dim> disp_position_mask(
      std::make_pair(x_lower, x_upper), n_components);

    Vector<double> cellwise_errors(n_active_cells);

    // Choose another quadrature rule for error computation
    const QWitherdenVincentSimplex<dim> err_quadrature(6);

    //
    // Linfty errors
    //
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::Linfty_norm,
                                      &scalar_field_mask);
    const double u_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_u = std::max(linf_error_u, u_linf);

    VectorTools::integrate_difference(*fixed_mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::Linfty_norm,
                                      &disp_position_mask);
    const double disp_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_disp = std::max(linf_error_disp, disp_linf);

    // Compute mesh velocity
    const FEValuesExtractors::Vector position(x_lower);
    IndexSet                         pos_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(position));

    for (const auto &i : pos_dofs)
    {
      if (!locally_owned_dofs.is_element(i))
        continue;

      double value = bdfCoeffs[0] * present_solution[i];
      for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        value += bdfCoeffs[iBDF] * previous_solutions[iBDF - 1][i];
      local_mesh_velocity[i] = value;
    }
    local_mesh_velocity.compress(VectorOperation::insert);
    mesh_velocity = local_mesh_velocity;

    // Do not compute mesh velocity at first time step for BDF2,
    // as the derivative is ill-defined...
    bool compute_mesh_velocity_error = !(param.bdf_order == 2 && time_step == 0);

    double w_linf = 0.;
    if(compute_mesh_velocity_error)
    {
      // Error on mesh velocity
      VectorTools::integrate_difference(*fixed_mapping,
                                        dof_handler,
                                        mesh_velocity,
                                        mesh_velocity_fun,
                                        cellwise_errors,
                                        err_quadrature,
                                        VectorTools::Linfty_norm,
                                        &disp_position_mask);
      w_linf =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::Linfty_norm);
      linf_error_w = std::max(linf_error_w, w_linf);
    }

    //
    // L2 errors
    //
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::L2_norm,
                                      &scalar_field_mask);
    const double u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(*fixed_mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::L2_norm,
                                      &disp_position_mask);
    const double disp_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    double w_l2_error = 0.;
    if(compute_mesh_velocity_error)
    {
      VectorTools::integrate_difference(*fixed_mapping,
                                        dof_handler,
                                        mesh_velocity,
                                        mesh_velocity_fun,
                                        cellwise_errors,
                                        err_quadrature,
                                        VectorTools::L2_norm,
                                        &disp_position_mask);
      w_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);
    }

    l2_err_u += param.dt * u_l2_error;
    l2_err_disp += param.dt * disp_l2_error;
    l2_err_w += param.dt * w_l2_error;

    pcout << "Current L2 errors: "
          << "||e_u||_L2 = " << u_l2_error << " - "
          << "||e_d||_L2 = " << disp_l2_error << " - "
          << "||e_w||_L2 = " << w_l2_error << std::endl;
    pcout << "Cumul.  L2 errors: "
          << "||e_u||_L2 = " << l2_err_u << " - "
          << "||e_d||_L2 = " << l2_err_disp << " - "
          << "||e_w||_L2 = " << l2_err_w << std::endl;
    pcout << "Current Li errors: "
          << "||e_u||_Li = " << u_linf << " - "
          << "||e_d||_Li = " << disp_linf << " - "
          << "||e_w||_Li = " << w_linf << std::endl;
    pcout << "Cumul.  Li errors: "
          << "||e_u||_Li = " << linf_error_u << " - "
          << "||e_d||_Li = " << linf_error_disp << " - "
          << "||e_w||_Li = " << linf_error_w << std::endl;
  }

  //
  // Reset solver between two solves with different meshes
  //
  template <int dim>
  void MMS<dim>::reset()
  {
    // Mesh
    triangulation.clear();

    // Errors
    linf_error_u    = 0.;
    linf_error_disp = 0.;
    linf_error_w    = 0.;
    l2_err_u        = 0.;
    l2_err_disp     = 0.;
    l2_err_w        = 0.;
  }

  template <int dim>
  void MMS<dim>::run()
  {
    unsigned int iMesh        = 1;
    unsigned int nConvergence = param.nConvergenceCycles;

    if (param.bdf_order == 0)
    {
      param.nTimeSteps = 1;
      nConvergence     = 1;
    }

    for (unsigned int iT = 1; iT <= nConvergence; ++iT, this->param.dt /= 2., this->param.nTimeSteps *= 2.)
    // for (unsigned int iT = 1; iT <=
    // nConvergence; ++iT, ++iMesh)
    // for (unsigned int iT = 1; iT <= nConvergence;
    //      ++iT, ++iMesh, this->param.dt /= 2., this->param.nTimeSteps *= 2.)
    {
      this->reset();

      this->param.prev_dt =
        this->param.dt; // Change this if using variable timestep
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->solution_fun.set_time(current_time);
      this->source_term_fun.set_time(current_time);
      this->mesh_velocity_fun.set_time(current_time);

      this->make_grid(iMesh);
      this->setup_system();
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();
      this->set_initial_condition();

      this->output_results(iT, 0);

      for (unsigned int i = 0; i < param.nTimeSteps; ++i)
      {
        this->current_time += param.dt;
        this->solution_fun.set_time(current_time);
        this->source_term_fun.set_time(current_time);
        this->mesh_velocity_fun.set_time(current_time);

        pcout << std::endl
              << "Time step " << i + 1 << " - Advancing to t = " << current_time
              << '.' << std::endl;

        this->create_nonzero_constraints();
        this->apply_nonzero_constraints();

        if (i == 0 && param.bdf_order == 2)
        {
          // For BDF2: set first step to exact solution
          this->set_exact_solution();

          // TODO: Or compute first step with BDF1 and smaller time step
        }
        else
        {
          this->solve_newton();
        }

        this->compute_errors(i);
        this->output_results(iT, i + 1);

        // Rotate solutions
        if (param.bdf_order > 0)
        {
          for (unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
            previous_solutions[i] = previous_solutions[i - 1];
          previous_solutions[0] = present_solution;
        }
      }

      // Add Linf error and L2 error at last time step
      convergence_table.add_value("nElm", triangulation.n_active_cells());
      convergence_table.add_value("dt", param.dt);
      convergence_table.add_value("L2_u", l2_err_u);
      convergence_table.add_value("Li_u", linf_error_u);
      convergence_table.add_value("L2_d", l2_err_disp);
      convergence_table.add_value("Li_d", linf_error_disp);
      convergence_table.add_value("L2_w", l2_err_w);
      convergence_table.add_value("Li_w", linf_error_w);
    }

    convergence_table.evaluate_convergence_rates(
      "L2_u", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_d", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_d", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_w", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_w", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_u", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_d", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_d", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_w", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_w", ConvergenceTable::reduction_rate_log2);

    // Arrange convergence table
    convergence_table.set_precision("L2_u", 4);
    convergence_table.set_precision("L2_d", 4);
    convergence_table.set_precision("L2_u", 4);
    convergence_table.set_precision("Li_u", 4);
    convergence_table.set_precision("Li_d", 4);
    convergence_table.set_precision("Li_w", 4);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.set_scientific("L2_d", true);
    convergence_table.set_scientific("L2_w", true);
    convergence_table.set_scientific("Li_u", true);
    convergence_table.set_scientific("Li_d", true);
    convergence_table.set_scientific("Li_w", true);

    pcout << std::endl;
    pcout << "BDF order: " << param.bdf_order << std::endl;
    pcout << "Scalar field  P" << param.field_degree << std::endl;
    pcout << "Mesh position P" << param.displacement_degree << std::endl;
    pcout << std::endl;
    if (mpi_rank == 0)
      convergence_table.write_text(std::cout);
  }
} // namespace NS_MMS

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace NS_MMS;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    SimulationParameters param;

    param.field_degree        = 2;
    param.displacement_degree = 1;

    param.conductivity        = CONDUCTIVITY;
    param.pseudo_solid_mu     = MU_PS;
    param.pseudo_solid_lambda = LAMBDA_PS;

    // Time integration
    param.bdf_order  = 2;
    param.t0         = 0.;
    param.dt         = 0.1;
    param.nTimeSteps = 11;
    param.t1         = param.dt * param.nTimeSteps;

    param.nConvergenceCycles = 8;

    MMS<2> problem2D(param);
    problem2D.run();

    // MMS<3> problem3D(param);
    // problem3D.run();
  }
  catch (const std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
