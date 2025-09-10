
#include <MMS.h>
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

bool VERBOSE = false;

// Fluid
#define VISCOSITY 1. // 0.123

// Pseudo-solid
#define LAMBDA_PS 1. // 1.234
#define MU_PS 1.     // 2.987

// Enable time-dependent BC for mesh displacement
bool WITH_MESH_DISPLACEMENT = true;

// Enable dudt in Navier-Stokes
#define WITH_TRANSIENT_NS

#define WITH_CONVECTION

// #define RIGID_MOTION
// #define LINEAR_DISPLACEMENT
#define QUADRATIC_DISPLACEMENT
// #define NO_BOUNDARY_DISPLACEMENT

// #define DISCONTINUOUS_LAMBDA

namespace NS_MMS
{
  using namespace dealii;
  using namespace ManufacturedSolution;

  void print_ifdefs()
  {
    if (WITH_MESH_DISPLACEMENT)
      std::cout << "WITH_MESH_DISPLACEMENT: ON" << std::endl;
    else
      std::cout << "WITH_MESH_DISPLACEMENT: OFF" << std::endl;

#if defined(WITH_TRANSIENT_NS)
    std::cout << "WITH_TRANSIENT_NS:           ON" << std::endl;
#else
    std::cout << "WITH_TRANSIENT_NS:           OFF" << std::endl;
#endif
#if defined(WITH_CONVECTION)
    std::cout << "WITH_CONVECTION:             ON" << std::endl;
#else
    std::cout << "WITH_CONVECTION:             OFF" << std::endl;
#endif
#if defined(LINEAR_DISPLACEMENT)
    std::cout << "LINEAR_DISPLACEMENT:         ON" << std::endl;
#else
    std::cout << "LINEAR_DISPLACEMENT:         OFF" << std::endl;
#endif
#if defined(QUADRATIC_DISPLACEMENT)
    std::cout << "QUADRATIC_DISPLACEMENT:      ON" << std::endl;
#else
    std::cout << "QUADRATIC_DISPLACEMENT:      OFF" << std::endl;
#endif
  }

  double phi_fun(const double t)
  {
    if (WITH_MESH_DISPLACEMENT)
      // return t;
      return sin(2 * M_PI * t);
    // return (t > 0) ? 4. * (t - tanh(t)) : 0.;
    else
      return 0.;
  }

  double phidot_fun(const double t)
  {
    if (WITH_MESH_DISPLACEMENT)
      // return 1.;
      return 2 * M_PI * cos(2 * M_PI * t);
    // return (t > 0) ? 4. * tanh(t) * tanh(t) : 0.;
    else
      return 0.;
  }

  //
  // Displacement manufactured solution
  //
  template <int dim>
  double
  chi_fun(const double phi, const Point<dim> &p, const unsigned int component)
  {
#if defined(RIGID_MOTION)

    return (component == 0) ? (phi / 4.) : 0.;

#elif defined(LINEAR_DISPLACEMENT)

    // Linear displacement in each component
    return phi / 4. * p[component];

#elif defined(QUADRATIC_DISPLACEMENT)

    // Quadratic displacement in each component
    if constexpr (dim == 2)
      return phi / 4. * p[component] * (p[component] - 1.);
    else
      return phi / 4. * p[component] * (p[component] - 1.);

#elif defined(NO_BOUNDARY_DISPLACEMENT)

    const double x = p[0];
    const double y = p[1];

    if constexpr (dim == 2)
    {
      if (component == 0)
        return phi * x * (x - 1) * y * (y - 1);
      else
        return phi * x * (x - 1) * y * (y - 1);
    }
    else
      DEAL_II_NOT_IMPLEMENTED();

#else

    // Displacement from Hay et al.
    if constexpr (dim == 2)
      return phi / 4. * p[0] * p[1] * (p[component] - 1.);
    else
      return phi / 4. * p[0] * p[1] * p[2] * (p[component] - 1.);

#endif
  }

  //
  // Mesh position manufactured solution
  //
  template <int dim>
  double
  pos_fun(const double phi, const Point<dim> &p, const unsigned int component)
  {
    return p[component] + chi_fun(phi, p, component);
  }

  //
  // Mesh velocity from manufactured solution
  //
  template <int dim>
  double wMesh_fun(const double       phidot,
                   const Point<dim>  &p,
                   const unsigned int component)
  {
#if defined(RIGID_MOTION)

    return (component == 0) ? (phidot / 4.) : 0.;

#elif defined(LINEAR_DISPLACEMENT)

    // Linear displacement
    return phidot / 4. * p[component];

#elif defined(QUADRATIC_DISPLACEMENT)

    // Quadratic displacement in each component
    if constexpr (dim == 2)
      return phidot / 4. * p[component] * (p[component] - 1.);
    else
      return phidot / 4. * p[component] * (p[component] - 1.);

#elif defined(NO_BOUNDARY_DISPLACEMENT)

    if constexpr (dim == 2)
    {
      if (component == 0)
        return phidot * p[0] * (p[0] - 1) * p[1] * (p[1] - 1);
      else
        return phidot * p[0] * (p[0] - 1) * p[1] * (p[1] - 1);
    }
    else
      DEAL_II_NOT_IMPLEMENTED();

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
    const FlowManufacturedSolutionBase<dim> &flow_mms;
    const unsigned int                       n_components;
    const unsigned int                       u_lower = 0;
    const unsigned int                       p_lower = dim;
    const unsigned int                       x_lower = dim + 1;
    const unsigned int                       l_lower = 2 * dim + 1;

  public:
    Solution(const double                             time,
             const unsigned int                       n_components,
             const FlowManufacturedSolutionBase<dim> &flow_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
      , n_components(n_components)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      // Used only to return the pressure value when constraining
      // pressure DoF
      if (component != p_lower)
        DEAL_II_ASSERT_UNREACHABLE();

      const double t   = this->get_time();
      const double phi = phi_fun(t);

      Vector<double> values(n_components);

      // Velocity
      values[0] = flow_mms.velocity(t, p, 0);
      values[1] = flow_mms.velocity(t, p, 1);
      // Pressure
      values[2] = flow_mms.pressure(t, p);
      // Mesh position
      values[3] = pos_fun(phi, p, 0);
      values[4] = pos_fun(phi, p, 1);
      // Lagrange multiplier:
      // Use dedicated "value" function below and give normal vector
      values[5] = 0.;
      values[6] = 0.;

      return values[component];
    }

    double value(const Point<dim>     &p,
                 const Tensor<1, dim> &normal,
                 const unsigned int    component = 0) const
    {
      // Used only to evaluate Lagrange multiplier on the
      // interior of boundary elements
      if (!(l_lower <= component && component < l_lower + dim))
        DEAL_II_ASSERT_UNREACHABLE();

      const double t   = this->get_time();
      const double phi = phi_fun(t);

      Tensor<1, dim> lambda;
      Tensor<2, dim> sigma;
      flow_mms.newtonian_stress(t, p, VISCOSITY, sigma);
      lambda = -sigma * normal;

      return lambda[component - l_lower];
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t   = this->get_time();
      const double phi = phi_fun(t);

      if constexpr (dim == 2)
      {
        // Velocity
        values[0] = flow_mms.velocity(t, p, 0);
        values[1] = flow_mms.velocity(t, p, 1);
        // Pressure
        values[2] = flow_mms.pressure(t, p);
        // Mesh position
        values[3] = pos_fun(phi, p, 0);
        values[4] = pos_fun(phi, p, 1);
        // Lagrange multiplier : use dedicated "value" function
        values[5] = 0.;
        values[6] = 0.;
      }
      else
      {
        // Velocity
        values[0] = flow_mms.velocity(t, p, 0);
        values[1] = flow_mms.velocity(t, p, 1);
        values[2] = flow_mms.velocity(t, p, 2);
        // Pressure
        values[3] = flow_mms.pressure(t, p);
        // Mesh position
        values[4] = pos_fun(phi, p, 0);
        values[5] = pos_fun(phi, p, 1);
        values[6] = pos_fun(phi, p, 2);
      }
    }

    // Gradient of solution, using finite differences
    // Only used to compute the matrix associated to (u - u_0) in weak BCs
    // Could also use the exact velocity gradient,
    // as only the velocity gradient is required.
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      const double h = 1e-8;

      Vector<double> vals_plus(gradients.size()), vals_minus(gradients.size());

      for (unsigned int d = 0; d < dim; ++d)
      {
        // perturbation direction
        Point<dim> p_plus = p, p_minus = p;
        p_plus[d] += h;
        p_minus[d] -= h;

        // evaluate at perturbed points
        this->vector_value(p_plus, vals_plus);
        this->vector_value(p_minus, vals_minus);

        // central difference
        for (unsigned int c = 0; c < gradients.size(); ++c)
          gradients[c][d] = (vals_plus[c] - vals_minus[c]) / (2.0 * h);
      }
    }
  };

  //
  // This function evaluates the velocity and pressure values
  // at the prescribed position value for this time.
  // Thus, instead of evaluating
  //
  //  u(p,t), v(p,t), w(p,t), pressure(p,t),
  //
  // it evaluates the values at the future position x(p,t):
  //
  //  u(x(p),t), v(x(p),t), w(x(p),t), pressure(x(p),t)
  //
  // The values of x are inchanged, and actually this function is
  // not meant to be called for position evaluations, only
  // for velocity and pressure values.
  //
  template <int dim>
  class SolutionAtFutureMeshPosition : public Function<dim>
  {
  public:
    const FlowManufacturedSolutionBase<dim> &flow_mms;

  public:
    SolutionAtFutureMeshPosition(
      const double                             time,
      const unsigned int                       n_components,
      const FlowManufacturedSolutionBase<dim> &flow_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      const double t   = this->get_time();
      const double phi = phi_fun(t);

      // Get prescribed mesh position
      Point<dim> pFinal;
      for (unsigned int d = 0; d < dim; ++d)
        pFinal[d] = pos_fun(phi, p, d);

      // Used only to return the pressure value when constraining
      // pressure DoF
      if (component == dim)
        return flow_mms.pressure(t, pFinal);
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t   = this->get_time();
      const double phi = phi_fun(t);

      // Get prescribed mesh position for this point
      Point<dim> pFinal;
      for (unsigned int d = 0; d < dim; ++d)
        pFinal[d] = pos_fun(phi, p, d);

      if constexpr (dim == 2)
      {
        // Velocity
        values[0] = flow_mms.velocity(t, pFinal, 0);
        values[1] = flow_mms.velocity(t, pFinal, 1);
        // Pressure
        values[2] = flow_mms.pressure(t, pFinal);
        // Mesh position
        values[3] = pos_fun(phi, p, 0);
        values[4] = pos_fun(phi, p, 1);
      }
      else
      {
        // Velocity
        values[0] = flow_mms.velocity(t, pFinal, 0);
        values[1] = flow_mms.velocity(t, pFinal, 1);
        values[2] = flow_mms.velocity(t, pFinal, 2);
        // Pressure
        values[3] = flow_mms.pressure(t, pFinal);
        // Mesh position
        values[4] = pos_fun(phi, p, 0);
        values[5] = pos_fun(phi, p, 1);
        values[6] = pos_fun(phi, p, 2);
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
        // Velocity
        values[0] = 0.;
        values[1] = 0.;
        // Pressure
        values[2] = 0.;
        // Mesh position
        values[3] = wMesh_fun(phidot, p, 0);
        values[4] = wMesh_fun(phidot, p, 1);
      }
      else
      {
        // Velocity
        values[0] = 0.;
        values[1] = 0.;
        values[2] = 0.;
        // Pressure
        values[3] = 0.;
        // Mesh position
        values[4] = wMesh_fun(phidot, p, 0);
        values[5] = wMesh_fun(phidot, p, 1);
        values[6] = wMesh_fun(phidot, p, 2);
      }
    }
  };

  template <int dim>
  class SourceTerm : public Function<dim>
  {
  public:
    const FlowManufacturedSolutionBase<dim> &flow_mms;

  public:
    SourceTerm(const double                             time,
               const unsigned int                       n_components,
               const FlowManufacturedSolutionBase<dim> &flow_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t        = this->get_time();
      const double phi      = phi_fun(t);
      const double mu       = VISCOSITY;
      const double lambda_s = LAMBDA_PS;
      const double mu_s     = MU_PS;

      Tensor<2, dim> grad_u;
      Tensor<1, dim> f, u, dudt_eulerian, uDotGradu, grad_p, lap_u;

      flow_mms.velocity_time_derivative(t, p, dudt_eulerian);
      flow_mms.velocity(t, p, u);
      // flow_mms.grad_velocity_ui_xj(t, p, grad_u);
      flow_mms.grad_velocity_uj_xi(t, p, grad_u);
      uDotGradu = u * grad_u;
      flow_mms.grad_pressure(t, p, grad_p);
      flow_mms.laplacian_velocity(t, p, lap_u);

      // Stokes/Navier-Stokes source term
#if defined(WITH_TRANSIENT_NS) && defined(WITH_CONVECTION)

      f = -(dudt_eulerian + uDotGradu + grad_p - mu * lap_u);

#elif defined(WITH_TRANSIENT_NS) && !defined(WITH_CONVECTION)

      f = -(dudt_eulerian + grad_p - mu * lap_u);

#elif !defined(WITH_TRANSIENT_NS) && defined(WITH_CONVECTION)

      f = -(uDotGradu + grad_p - mu * lap_u);

#else

      f = -(grad_p - mu * lap_u);

#endif

      for (unsigned int d = 0; d < dim; ++d)
        values[d] = f[d];

      // Pressure
      values[dim] = 0.;

      // Pseudo-solid
#if defined(RIGID_MOTION) || defined(LINEAR_DISPLACEMENT)

      for (unsigned int d = 0; d < dim; ++d)
        values[dim + 1 + d] = 0.;

#elif defined(QUADRATIC_DISPLACEMENT)

      for (unsigned int d = 0; d < dim; ++d)
        values[dim + 1 + d] = (phi * (lambda_s + 2 * mu_s)) / 2;

#elif defined(NO_BOUNDARY_DISPLACEMENT)

      const double x = p[0];
      const double y = p[1];

      if constexpr (dim == 2)
      {
        values[3] =
          (phi * (lambda_s + mu_s - 2 * lambda_s * x - 4 * lambda_s * y -
                  4 * mu_s * x - 6 * mu_s * y + 2 * lambda_s * y * y +
                  2 * mu_s * x * x + 4 * mu_s * y * y + 4 * lambda_s * x * y +
                  4 * mu_s * x * y));
        values[4] =
          (phi * (lambda_s + mu_s - 4 * lambda_s * x - 2 * lambda_s * y -
                  6 * mu_s * x - 4 * mu_s * y + 2 * lambda_s * x * x +
                  4 * mu_s * x * x + 2 * mu_s * y * y + 4 * lambda_s * x * y +
                  4 * mu_s * x * y));
      }
      else
      {
        const double z = p[2];
        DEAL_II_NOT_IMPLEMENTED();
      }
#else

      const double x = p[0];
      const double y = p[1];
      const double h = 1.;

      if constexpr (dim == 2)
      {
        // Source term for displacement from Hay et al.
        values[3] = -(phi / 4. * (lambda_s * (h - 4 * y) + mu_s * (h - 6 * y)));
        values[4] = -(phi / 4. * (lambda_s * (h - 4 * x) + mu_s * (h - 6 * x)));
      }
      else
      {
        DEAL_II_NOT_IMPLEMENTED();
      }
#endif
    }

    // Gradient of source term, using finite differences
    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      const double h = 1e-8;

      Vector<double> vals_plus(gradients.size()), vals_minus(gradients.size());

      for (unsigned int d = 0; d < dim; ++d)
      {
        // perturbation direction
        Point<dim> p_plus = p, p_minus = p;
        p_plus[d] += h;
        p_minus[d] -= h;

        // evaluate at perturbed points
        this->vector_value(p_plus, vals_plus);
        this->vector_value(p_minus, vals_minus);

        // central difference
        for (unsigned int c = 0; c < gradients.size(); ++c)
          gradients[c][d] = (vals_plus[c] - vals_minus[c]) / (2.0 * h);
      }
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    const unsigned int n_components = 3 * dim + 1;
    const unsigned int u_lower      = 0;
    const unsigned int p_lower      = dim;
    const unsigned int x_lower      = dim + 1;
    const unsigned int l_lower      = 2 * dim + 1;

  public:
    ScratchData(const FESystem<dim>       &fe,
                const Quadrature<dim>     &cell_quadrature,
                const Mapping<dim>        &fixed_mapping,
                const Mapping<dim>        &mapping,
                const Quadrature<dim - 1> &face_quadrature,
                const unsigned int         dofs_per_cell,
                const unsigned int         boundary_id,
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
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values |
                                  update_jacobians | update_inverse_jacobians)
      , fe_face_values(mapping,
                       fe,
                       face_quadrature,
                       update_values | update_gradients |
                         update_quadrature_points | update_JxW_values |
                         update_jacobians | update_inverse_jacobians)
      , fe_face_values_fixed_mapping(fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values | update_jacobians |
                                       update_inverse_jacobians)
      , n_q_points(cell_quadrature.size())

      // We assume that simplicial meshes with all tris or tets
      , n_faces((dim == 3) ? 4 : 3)

      , n_faces_q_points(face_quadrature.size())
      , dofs_per_cell(dofs_per_cell)
      , boundary_id(boundary_id)
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
                const Function<dim>                &solution_fun,
                const Function<dim>                &source_term_fun)
    {
      active_fe_values->reinit(cell);

      for (const unsigned int i : active_fe_values->dof_indices())
        components[i] =
          active_fe_values->get_fe().system_to_component_index(i).first;

      const FEValuesExtractors::Vector velocity(u_lower);
      const FEValuesExtractors::Scalar pressure(p_lower);
      const FEValuesExtractors::Vector position(x_lower);
      const FEValuesExtractors::Vector lambda(l_lower);

      if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
      {
        (*active_fe_values)[velocity].get_function_values(
          current_solution, present_velocity_values);
        (*active_fe_values)[velocity].get_function_gradients(
          current_solution, present_velocity_gradients);
        (*active_fe_values)[pressure].get_function_values(
          current_solution, present_pressure_values);
        (*active_fe_values)[position].get_function_values(
          current_solution, present_position_values);
        (*active_fe_values)[position].get_function_gradients(
          current_solution, present_position_gradients);
      }
      else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
      {
        (*active_fe_values)[velocity].get_function_values_from_local_dof_values(
          current_solution, present_velocity_values);
        (*active_fe_values)[velocity]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_velocity_gradients);
        (*active_fe_values)[pressure].get_function_values_from_local_dof_values(
          current_solution, present_pressure_values);
        (*active_fe_values)[position].get_function_values_from_local_dof_values(
          current_solution, present_position_values);
        (*active_fe_values)[position]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_position_gradients);
      }
      else
      {
        static_assert(false,
                      "reinit expects LA::MPI::Vector or std::vector<double>");
      }

      // Source term
      source_term_fun.vector_value_list(
        active_fe_values->get_quadrature_points(), source_term_full);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (int d = 0; d < dim; ++d)
          source_term_velocity[q][d] = source_term_full[q](u_lower + d);
        for (int d = 0; d < dim; ++d)
          source_term_position[q][d] = source_term_full[q](x_lower + d);
      }

      // Gradient of source term
      // Only need to fill in for the scalar field
      source_term_fun.vector_gradient_list(
        active_fe_values->get_quadrature_points(), grad_source_term_full);

      // Layout: grad_source_velocity[q] = df_i/dx_j
      for (unsigned int q = 0; q < n_q_points; ++q)
        for (int di = 0; di < dim; ++di)
          for (int dj = 0; dj < dim; ++dj)
            grad_source_velocity[q][di][dj] =
              grad_source_term_full[q][u_lower + di][dj];

      // Previous solutions
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        (*active_fe_values)[velocity].get_function_values(
          previous_solutions[i], previous_velocity_values[i]);
        (*active_fe_values)[position].get_function_values(
          previous_solutions[i], previous_position_values[i]);
      }

      // Current mesh velocity from displacement
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        present_mesh_velocity_values[q] =
          bdfCoeffs[0] * present_position_values[q];
        for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        {
          present_mesh_velocity_values[q] +=
            bdfCoeffs[iBDF] * previous_position_values[iBDF - 1][q];
        }
      }

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW[q] = active_fe_values->JxW(q);
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]      = (*active_fe_values)[velocity].value(k, q);
          grad_phi_u[q][k] = (*active_fe_values)[velocity].gradient(k, q);
          div_phi_u[q][k]  = (*active_fe_values)[velocity].divergence(k, q);
          phi_p[q][k]      = (*active_fe_values)[pressure].value(k, q);
          phi_x[q][k]      = (*active_fe_values)[position].value(k, q);
          grad_phi_x[q][k] = (*active_fe_values)[position].gradient(k, q);
          div_phi_x[q][k]  = (*active_fe_values)[position].divergence(k, q);
        }
      }

      //
      // Face values and shape functions,
      // for the faces touching the prescribed boundary_id
      //
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        active_fe_face_values->reinit(cell, face);

        if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
        {
          (*active_fe_face_values)[velocity].get_function_values(
            current_solution, present_face_velocity_values[i_face]);
          (*active_fe_face_values)[position].get_function_values(
            current_solution, present_face_position_values[i_face]);
          (*active_fe_face_values)[position].get_function_gradients(
            current_solution, present_face_position_gradient[i_face]);
          (*active_fe_face_values)[lambda].get_function_values(
            current_solution, present_face_lambda_values[i_face]);
        }
        else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
        {
          (*active_fe_face_values)[velocity]
            .get_function_values_from_local_dof_values(
              current_solution, present_face_velocity_values[i_face]);
          (*active_fe_face_values)[position]
            .get_function_values_from_local_dof_values(
              current_solution, present_face_position_values[i_face]);
          (*active_fe_face_values)[position]
            .get_function_gradients_from_local_dof_values(
              current_solution, present_face_position_gradient[i_face]);
          (*active_fe_face_values)[lambda]
            .get_function_values_from_local_dof_values(
              current_solution, present_face_lambda_values[i_face]);
        }
        else
        {
          static_assert(
            false, "reinit expects LA::MPI::Vector or std::vector<double>");
        }

        for (unsigned int i = 0; i < previous_solutions.size(); ++i)
        {
          (*active_fe_face_values)[position].get_function_values(
            previous_solutions[i], previous_face_position_values[i_face][i]);
        }

        // ////////////////////////////////////////////
        // std::cout << "Cell with vertices:\n";
        // for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
        //   std::cout << std::setprecision(10) << "  " << cell->vertex(iv) << "\n";
        // std::cout << "  Face " << i_face << " vertices:\n";
        // for (unsigned int iv = 0; iv < face->n_vertices(); ++iv)
        //   std::cout << std::setprecision(10) << "    " << face->vertex(iv) << "\n";
        // std::cout << std::setprecision(10) << "J = " << active_fe_face_values->jacobian(0) << std::endl;
        // ////////////////////////////////////////////

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          face_JxW[i_face][q]       = active_fe_face_values->JxW(q);
          face_jacobians[i_face][q] = active_fe_face_values->jacobian(q);
          const Tensor<2, dim> J = active_fe_face_values->jacobian(q);

          if constexpr (dim == 2)
          {
            switch (i_face)
            {
              case 0:
                dxsids[0] = 1.;
                dxsids[1] = 0.;

                dxsids_array[0][0] = 1.;
                dxsids_array[0][1] = 0.;
                break;
              case 1:
                dxsids[0] = -1.;
                dxsids[1] = 1.;

                dxsids_array[0][0] = -1.;
                dxsids_array[0][1] =  1.;
                break;
              case 2:
                dxsids[0] = 0.;
                dxsids[1] = -1.;

                dxsids_array[0][0] = 0.;
                dxsids_array[0][1] = -1.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          }
          else
          {
            switch (i_face)
            {
              // Using dealii's face ordering
              case 3: // Opposite to v0
                dxsids_array[0][0] = -1.; dxsids_array[1][0] = -1.;
                dxsids_array[0][1] =  1.; dxsids_array[1][1] =  0.;
                dxsids_array[0][2] =  0.; dxsids_array[1][2] =  1.;
                break;
              case 2: // Opposite to v1
                dxsids_array[0][0] =  0.; dxsids_array[1][0] =  0.;
                dxsids_array[0][1] =  1.; dxsids_array[1][1] =  0.;
                dxsids_array[0][2] =  0.; dxsids_array[1][2] =  1.;
                break;
              case 1: // Opposite to v2
                dxsids_array[0][0] =  1.; dxsids_array[1][0] =  0.;
                dxsids_array[0][1] =  0.; dxsids_array[1][1] =  0.;
                dxsids_array[0][2] =  0.; dxsids_array[1][2] =  1.;
                break;
              case 0: // Opposite to v3
                dxsids_array[0][0] =  1.; dxsids_array[1][0] =  0.;
                dxsids_array[0][1] =  0.; dxsids_array[1][1] =  1.;
                dxsids_array[0][2] =  0.; dxsids_array[1][2] =  0.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          }

          face_dXds[i_face][q] = face_jacobians[i_face][q] * dxsids;

          Tensor<2, dim-1> G;
          G = 0;
          for(unsigned int di = 0; di < dim-1; ++di)
            for(unsigned int dj = 0; dj < dim-1; ++dj)
              for(unsigned int im = 0; im < dim; ++im)
                for(unsigned int in = 0; in < dim; ++in)
                  for(unsigned int ip = 0; ip < dim; ++ip)
                    G[di][dj] += dxsids_array[di][im] * J[in][im] * J[in][ip] * dxsids_array[dj][ip];
          face_G[i_face][q] = G;
          const double sqrt_det_G = sqrt(determinant(G));
          const Tensor<2, dim-1> G_inverse = invert(G);
          // std::cout << "G is " << G << std::endl;

          // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
          Tensor<2, dim-1> res;

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            phi_u_face[i_face][q][k] =
              (*active_fe_face_values)[velocity].value(k, q);
            phi_x_face[i_face][q][k] =
              (*active_fe_face_values)[position].value(k, q);
            grad_phi_x_face[i_face][q][k] =
              (*active_fe_face_values)[position].gradient(k, q);
            phi_l_face[i_face][q][k] =
              (*active_fe_face_values)[lambda].value(k, q);

            const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

            // std::cout << "grad_phi_x  is " << grad_phi_x << std::endl;
            // std::cout << "pres_grad_x is " << present_face_position_gradient[i_face][q] << std::endl;


            Tensor<2, dim> A = transpose(J) * (transpose(present_face_position_gradient[i_face][q]) * grad_phi_x 
                                             + transpose(grad_phi_x) * present_face_position_gradient[i_face][q]) * J;

            // std::cout << "A           is " << A << std::endl;

            res = 0;
            for(unsigned int di = 0; di < dim-1; ++di)
              for(unsigned int dj = 0; dj < dim-1; ++dj)
                for(unsigned int im = 0; im < dim-1; ++im)
                  for(unsigned int in = 0; in < dim; ++in)
                      for(unsigned int io = 0; io < dim; ++io)
                        res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] * A[in][io] * dxsids_array[dj][io];
            // std::cout << "G_inverse   is " << G_inverse << std::endl;
            // std::cout << "res         is " << res << std::endl;
            // delta_dx[i_face][q][k] = 0.5 * sqrt_det_G * trace(res); // Choose this if multiplying by W
            delta_dx[i_face][q][k] = 0.5 * trace(res); // Choose this if multiplying by JxW in the matrix
          }

          // throw std::runtime_error("foo");

          // Face mesh velocity
          present_face_mesh_velocity_values[i_face][q] =
            bdfCoeffs[0] * present_face_position_values[i_face][q];
          for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
          {
            present_face_mesh_velocity_values[i_face][q] +=
              bdfCoeffs[iBDF] *
              previous_face_position_values[i_face][iBDF - 1][q];
          }
        }

        // Prescribed velocity values on the weak bc boundary
        solution_fun.vector_value_list(
          active_fe_face_values->get_quadrature_points(),
          solution_on_weak_bc_full[i_face]);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          for (int d = 0; d < dim; ++d)
            prescribed_velocity_weak_bc[i_face][q][d] =
              solution_on_weak_bc_full[i_face][q](u_lower + d);
        }

        // Gradient of prescribed solution
        // Only need to fill in for the velocity field,
        // for which weak Dirichlet BC are applied
        solution_fun.vector_gradient_list(
          active_fe_face_values->get_quadrature_points(),
          grad_solution_on_weak_bc_full[i_face]);

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
          for (int di = 0; di < dim; ++di)
            for (int dj = 0; dj < dim; ++dj)
              grad_solution_velocity[i_face][q][di][dj] =
                grad_solution_on_weak_bc_full[i_face][q][u_lower + di][dj];
      }
    }

  public:
    template <typename VectorType>
    void reinit_current_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                                     &current_solution,
      const std::vector<LA::MPI::Vector>                   &previous_solutions,
      const Function<dim>                                  &solution_fun,
      const Function<dim>                                  &source_term_fun)
    {
      active_fe_values      = &fe_values;
      active_fe_face_values = &fe_face_values;
      this->reinit(cell,
                   current_solution,
                   previous_solutions,
                   solution_fun,
                   source_term_fun);
    }

    template <typename VectorType>
    void reinit_fixed_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                                     &current_solution,
      const std::vector<LA::MPI::Vector>                   &previous_solutions,
      const Function<dim>                                  &solution_fun,
      const Function<dim>                                  &source_term_fun)
    {
      active_fe_values      = &fe_values_fixed_mapping;
      active_fe_face_values = &fe_face_values_fixed_mapping;
      this->reinit(cell,
                   current_solution,
                   previous_solutions,
                   solution_fun,
                   source_term_fun);
    }

    const FEValues<dim> &get_current_fe_values() const { return fe_values; }
    const FEValues<dim> &get_fixed_fe_values() const
    {
      return fe_values_fixed_mapping;
    }

  public:
    FEValues<dim> *active_fe_values;
    FEValues<dim>  fe_values;
    FEValues<dim>  fe_values_fixed_mapping;

    FEFaceValues<dim> *active_fe_face_values;
    FEFaceValues<dim>  fe_face_values;
    FEFaceValues<dim>  fe_face_values_fixed_mapping;

    const unsigned int n_q_points;
    const unsigned int n_faces;
    const unsigned int n_faces_q_points;
    const unsigned int dofs_per_cell;

    // The tag of the boundary on which weak Dirichlet BC are
    // applied with Lagrange multiplier. Only 1 for now.
    const unsigned int boundary_id;

    const std::vector<double> &bdfCoeffs;

    std::vector<double>              JxW;
    std::vector<std::vector<double>> face_JxW;

    // Jacobian matrix on face
    std::vector<std::vector<Tensor<2, dim>>> face_jacobians;
    // std::vector<std::vector<DerivativeForm<1, dim, dim>>> face_jacobians;

    // If dim = 2, face_dXds is the variation of an edge position w.r.t.
    // the 1-dimensional reference coordinate s. That is,
    // this is the non-unit tangent vector to the edge.
    Tensor<1, dim>                           dxsids;
    std::vector<std::vector<Tensor<1, dim>>> face_dXds;

    // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
    std::array<Tensor<1, dim>, dim - 1>                           dxsids_array;
    std::vector<std::vector<Tensor<2, dim-1>>> face_G;
    // std::vector<std::vector<std::array<Tensor<1, dim>, dim - 1>>> face_preFactor;

    // At face x quad node x phi_position_j
    std::vector<std::vector<std::vector<double>>> delta_dx;

    std::vector<unsigned int> components;

    // Current and previous values and gradients for each quad node
    std::vector<Tensor<1, dim>>              present_velocity_values;
    std::vector<Tensor<2, dim>>              present_velocity_gradients;
    std::vector<double>                      present_pressure_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

    std::vector<Tensor<1, dim>>              present_position_values;
    std::vector<Tensor<2, dim>>              present_position_gradients;
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values;
    std::vector<Tensor<1, dim>>              present_mesh_velocity_values;

    // Current and previous values on faces
    std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> present_face_position_values;
    std::vector<std::vector<Tensor<2, dim>>> present_face_position_gradient;
    std::vector<std::vector<Tensor<1, dim>>> present_face_mesh_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> present_face_lambda_values;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>>
      previous_face_position_values;

    // Source term on cell
    std::vector<Vector<double>>
      source_term_full; // The source term with n_components
    std::vector<Tensor<1, dim>> source_term_velocity;
    std::vector<Tensor<1, dim>> source_term_position;

    // Gradient of source term,
    // at each quad node, for each dof component, result is a Tensor<1, dim>
    std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
    std::vector<Tensor<2, dim>>              grad_source_velocity;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<double>>         phi_p;
    std::vector<std::vector<Tensor<1, dim>>> phi_x;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_x;
    std::vector<std::vector<double>>         div_phi_x;

    // Shape functions on faces for relevant faces, each quad node and each dof
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_x_face;
    std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_x_face;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_l_face;

    // The prescribed full solution and velocity values on the boundary with
    // weak BC One velocity per relevant faces and quad node
    std::vector<std::vector<Vector<double>>> solution_on_weak_bc_full;
    std::vector<std::vector<Tensor<1, dim>>> prescribed_velocity_weak_bc;

    std::vector<std::vector<std::vector<Tensor<1, dim>>>>
                                             grad_solution_on_weak_bc_full;
    std::vector<std::vector<Tensor<2, dim>>> grad_solution_velocity;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);
    present_position_values.resize(n_q_points);
    present_position_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);

    present_face_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_gradient.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    present_face_lambda_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_mesh_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    source_term_full.resize(n_q_points, Vector<double>(n_components));
    source_term_velocity.resize(n_q_points);
    source_term_position.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);

    // BDF
    previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_position_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_face_position_values.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(
        bdfCoeffs.size() - 1, std::vector<Tensor<1, dim>>(n_faces_q_points)));

    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));

    phi_u_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    phi_x_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    grad_phi_x_face.resize(n_faces,
                           std::vector<std::vector<Tensor<2, dim>>>(
                             n_faces_q_points,
                             std::vector<Tensor<2, dim>>(dofs_per_cell)));

    phi_l_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    solution_on_weak_bc_full.resize(
      n_faces,
      std::vector<Vector<double>>(n_faces_q_points,
                                  Vector<double>(n_components)));
    prescribed_velocity_weak_bc.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    grad_solution_on_weak_bc_full.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(
        n_faces_q_points, std::vector<Tensor<1, dim>>(n_components)));
    grad_solution_velocity.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_jacobians.resize(n_faces,
                          std::vector<Tensor<2, dim>>(n_faces_q_points));
    // face_jacobians.resize(n_faces, std::vector<DerivativeForm<1, dim,
    // dim>>(n_faces_q_points));
    face_dXds.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    face_G.resize(n_faces, std::vector<Tensor<2, dim-1>>(n_faces_q_points));
    delta_dx.resize(n_faces, std::vector<std::vector<double>>(n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }

  class SimulationParameters
  {
  public:
    unsigned int velocity_degree;
    unsigned int position_degree;
    unsigned int lambda_degree;

    std::vector<std::string> position_boundary_names;
    std::vector<std::string> strong_velocity_boundary_names;
    std::vector<std::string> weak_velocity_boundary_names;

    double       viscosity;
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
    MMS(const SimulationParameters              &param,
        const FlowManufacturedSolutionBase<dim> &flow_mms);

    void run();

  private:
    void set_bdf_coefficients(const unsigned int order);
    void make_grid(const unsigned int iMesh);
    void setup_system();
    void create_zero_constraints();
    void create_nonzero_constraints();
    void constrain_pressure_point(AffineConstraints<double> &constraints,
                                  bool                       set_to_zero);
    void create_lambda_zero_constraints(const unsigned int boundary_id);
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
                        const unsigned int time_step);
    void compute_errors(const unsigned int time_step);
    void compute_lambda_error_on_boundary(const unsigned int boundary_id,
                                          double            &lambda_l2_error,
                                          double            &lambda_linf_error,
                                          Tensor<1, dim>    &error_on_integral);
    void reset();

    SimulationParameters param;

    MPI_Comm           mpi_communicator;
    const unsigned int mpi_rank;

    FESystem<dim> fe;

    // Ordering of the FE system
    // Each field is in the half-open [lower, upper)
    // Check for matching component by doing e.g.:
    // if(u_lower <= comp && comp < u_upper)
    const unsigned int n_components = 3 * dim + 1;
    const unsigned int u_lower      = 0;
    const unsigned int u_upper      = dim;
    const unsigned int p_lower      = dim;
    const unsigned int p_upper      = dim + 1;
    const unsigned int x_lower      = dim + 1;
    const unsigned int x_upper      = 2 * dim + 1;
    const unsigned int l_lower      = 2 * dim + 1;
    const unsigned int l_upper      = 3 * dim + 1;

  public:
    bool is_velocity(const unsigned int component) const
    {
      return u_lower <= component && component < u_upper;
    }
    bool is_pressure(const unsigned int component) const
    {
      return p_lower <= component && component < p_upper;
    }
    bool is_position(const unsigned int component) const
    {
      return x_lower <= component && component < x_upper;
    }
    bool is_lambda(const unsigned int component) const
    {
      return l_lower <= component && component < l_upper;
    }

  public:
    QSimplex<dim>     quadrature;
    QSimplex<dim - 1> face_quadrature;

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
    AffineConstraints<double> lambda_constraints;

    // The global index of the pressure DoF to constrain to the value
    // of the manufactured solution.
    types::global_dof_index constrained_pressure_dof =
      numbers::invalid_dof_index;
    Point<dim> constrained_pressure_support_point;

    // Dirichlet BC in ALE formulation:
    // Keep track of the BC imposed at previous Newton iteration
    // double previous_pressure_DOF;

    // The id of the boundary where weak Dirichlet BC are prescribed
    // for the velocity
    unsigned int weak_bc_boundary_id;

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

    Solution<dim>                     solution_fun;
    SourceTerm<dim>                   source_term_fun;
    MeshVelocity<dim>                 mesh_velocity_fun;
    SolutionAtFutureMeshPosition<dim> solution_at_future_position_fun;

    // Contiguous maps from global vertex index to its position dofs,
    // and vice versa.
    std::vector<std::vector<unsigned int>> vertex2position_dof;
    std::vector<unsigned int>              position_dof2vertex;

    // L1 in time, L2 in space
    double l2_err_u;
    double l2_err_p;
    double l2_err_x;
    double l2_err_w;
    double l2_err_l;
    // Linf in space and time
    double linf_error_u;
    double linf_error_p;
    double linf_error_x;
    double linf_error_w;
    double linf_error_l;
    double linf_error_Fx;
    double linf_error_Fy;

    ConvergenceTable convergence_table;
  };

  template <int dim>
  MMS<dim>::MMS(const SimulationParameters              &param,
                const FlowManufacturedSolutionBase<dim> &flow_mms)
    : param(param)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , fe(FE_SimplexP<dim>(param.velocity_degree), // Velocity
         dim,
         FE_SimplexP<dim>(param.velocity_degree - 1), // Pressure
         1,
         FE_SimplexP<dim>(param.position_degree), // Position
         dim,
#if defined(DISCONTINUOUS_LAMBDA)
         FE_SimplexDGP<dim>(param.lambda_degree), // Lagrange multiplier
#else
         FE_SimplexP<dim>(param.lambda_degree), // Lagrange multiplier
#endif
         dim)
    , quadrature(QGaussSimplex<dim>(4))
    , face_quadrature(QGaussSimplex<dim - 1>(4))
    , triangulation(mpi_communicator)
    , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
    , dof_handler(triangulation)
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never, // TimerOutput::summary,
                      TimerOutput::wall_times)
    , current_time(param.t0)
    , solution_fun(Solution<dim>(current_time, n_components, flow_mms))
    , source_term_fun(SourceTerm<dim>(current_time, n_components, flow_mms))
    , mesh_velocity_fun(MeshVelocity<dim>(current_time, n_components))
    , solution_at_future_position_fun(
        SolutionAtFutureMeshPosition<dim>(current_time, n_components, flow_mms))
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
    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      // meshFile = "../data/meshes/square" + std::to_string(iMesh) + ".msh";
      meshFile = "../data/meshes/holed" + std::to_string(iMesh) + ".msh";
      // meshFile = "../data/meshes/holed_square" + std::to_string(iMesh) + ".msh";
      // meshFile = "../data/meshes/square_criss_crossed" + std::to_string(iMesh) + ".msh";
    }
    else
    {
      // meshFile = "../data/meshes/cube" + std::to_string(iMesh) + ".msh";
      meshFile = "../data/meshes/holed3D_" + std::to_string(iMesh) + ".msh";
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
      if (VERBOSE)
      {
        std::cout << "Mesh info:" << std::endl
                  << " dimension: " << dim << std::endl
                  << " no. of cells: " << serial_tria.n_active_cells()
                  << std::endl;
      }

      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : serial_tria.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;

      if (VERBOSE)
      {
        std::cout << " boundary indicators: ";
        for (const std::pair<const types::boundary_id, unsigned int> &pair :
             boundary_count)
        {
          std::cout << pair.first << '(' << pair.second << " times) ";
        }
        std::cout << std::endl;
      }

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

      if (VERBOSE)
      {
        for (const auto &[id, name] : mesh_domains_tag2name)
          std::cout << "ID " << id << " -> " << name << "\n";
      }
    }

    for (auto str : param.position_boundary_names)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("Position Dirichlet BC should be prescribed "
                                 "on the boundary named \"" +
                                 str +
                                 "\", but no physical entity with this name "
                                 "was read from the mesh file.");
      }
    }

    for (auto str : param.strong_velocity_boundary_names)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("Strong velocity Dirichlet BC should be "
                                 "prescribed on the boundary named \"" +
                                 str +
                                 "\", but no physical entity with this name "
                                 "was read from the mesh file.");
      }
    }

    if (param.weak_velocity_boundary_names.size() > 1)
      throw std::runtime_error(
        "Only considering a single boundary for weak velocity BC for now.");

    for (auto str : param.weak_velocity_boundary_names)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("Weak velocity Dirichlet BC should be "
                                 "prescribed on the boundary named \"" +
                                 str +
                                 "\", but no physical entity with this name "
                                 "was read from the mesh file.");
      }
      weak_bc_boundary_id = mesh_domains_name2tag.at(str);
    }
  }

  template <int dim>
  void MMS<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    if (VERBOSE)
    {
      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
    }

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

    // Mesh position
    // Initialize directly from the triangulation
    // The parallel vector storing the mesh position is local_evaluation_point,
    // because this is the one to modify when computing finite differences.
    const FEValuesExtractors::Vector position(x_lower);
    VectorTools::get_position_vector(*fixed_mapping,
                                     dof_handler,
                                     local_evaluation_point,
                                     fe.component_mask(position));
    local_evaluation_point.compress(VectorOperation::insert);
    evaluation_point = local_evaluation_point;

    // Set mapping as a solution-dependent mapping
    mapping = std::make_unique<MappingFEField<dim, dim, LA::MPI::Vector>>(
      dof_handler, evaluation_point, fe.component_mask(position));
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
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Update mesh position *BEFORE* evaluating scalar field
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             solution_fun,
                             newton_update,
                             fe.component_mask(position));

    evaluation_point = newton_update;

    // Set velocity with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             newton_update,
                             fe.component_mask(velocity));

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    evaluation_point = newton_update;
    present_solution = newton_update;

    // Dirty copy of the initial condition for BDF2 for now (-:
    for (auto &sol : previous_solutions)
      sol = present_solution;
  }

  template <int dim>
  void
  MMS<dim>::constrain_pressure_point(AffineConstraints<double> &constraints,
                                     bool                       set_to_zero)
  {
    // Determine the pressure dof the first time
    if (constrained_pressure_dof == numbers::invalid_dof_index)
    {
      // Choose a fixed physical reference location
      // Here it's the origin (Point<dim> initialized at 0)
      const Point<dim> reference_point;

      const FEValuesExtractors::Scalar pressure(p_lower);
      IndexSet                         pressure_dofs =
        DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure));

      // Get support points for locally relevant DoFs
      std::map<types::global_dof_index, Point<dim>> support_points;
      DoFTools::map_dofs_to_support_points(*mapping,
                                           dof_handler,
                                           support_points);

      double local_min_dist             = std::numeric_limits<double>::max();
      types::global_dof_index local_dof = numbers::invalid_dof_index;

      for (auto idx : pressure_dofs)
      {
        if (!locally_owned_dofs.is_element(idx))
          continue;

        const double dist = support_points[idx].distance(reference_point);
        if (dist < local_min_dist)
        {
          local_min_dist = dist;
          local_dof      = idx;
        }
      }

      // Prepare for MPI_MINLOC reduction
      struct MinLoc
      {
        double                  dist;
        types::global_dof_index dof;
      } local_pair{local_min_dist, local_dof}, global_pair;

      // MPI reduction to find the global closest DoF
      MPI_Allreduce(&local_pair,
                    &global_pair,
                    1,
                    MPI_DOUBLE_INT,
                    MPI_MINLOC,
                    mpi_communicator);

      constrained_pressure_dof = global_pair.dof;

      // Set support point for MMS evaluation
      if (locally_owned_dofs.is_element(constrained_pressure_dof))
      {
        constrained_pressure_support_point =
          support_points[constrained_pressure_dof];
      }
    }

    // Constrain that DoF globally
    if (locally_owned_dofs.is_element(constrained_pressure_dof))
    {
      // const double pAnalytic =
      // solution_fun.value(constrained_pressure_support_point, p_lower);
      const double pAnalytic = solution_at_future_position_fun.value(
        constrained_pressure_support_point, p_lower);

      constraints.add_line(constrained_pressure_dof);
      constraints.set_inhomogeneity(constrained_pressure_dof,
                                    set_to_zero ? 0. : pAnalytic);
    }

    constraints.make_consistent_in_parallel(locally_owned_dofs,
                                            constraints.get_local_lines(),
                                            mpi_communicator);
  }

  template <int dim>
  void MMS<dim>::create_lambda_zero_constraints(const unsigned int boundary_id)
  {
    lambda_constraints.clear();
    lambda_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector lambda(l_lower);

    // Apply a constraint dof = 0 for all lambda dofs that are not on the
    // prescribed boundary
    std::set<types::global_dof_index> unconstrained_lambda_dofs;

#if defined(DISCONTINUOUS_LAMBDA)

    std::map<types::global_dof_index, Point<dim>> support_points;
    DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                         dof_handler,
                                         support_points,
                                         fe.component_mask(lambda));

    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);
    //
    // Modification for discontinous lambda
    //
    std::vector<types::global_dof_index> local_dof_indices(
      fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int idof = 0; idof < fe.n_dofs_per_cell(); ++idof)
      {
        const unsigned int component = fe.system_to_component_index(idof).first;

        if (is_lambda(component))
        {
          for (const unsigned int f : cell->face_indices())
          {
            // has_support_on_face(idof, f) returns true for all faces, is this
            // normal?

            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == boundary_id)
            {
              // Check if the idof-th basis function is nonzero anywhere on the
              // face
              fe_face_values.reinit(cell, f);
              bool on_face = false;
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
              {
                if (std::abs(fe_face_values.shape_value(idof, q)) > 1e-12)
                {
                  on_face = true;
                  break;
                }
              }

              if (on_face)
              {
                unconstrained_lambda_dofs.insert(local_dof_indices[idof]);
              }
            }
          }
        }
      }
    }
#else
    //
    // Works for continuous lambda
    //
    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto f : cell->face_indices())
      {
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == boundary_id)
        {
          cell->face(f)->get_dof_indices(face_dofs);
          for (unsigned int idof = 0; idof < fe.n_dofs_per_face(); ++idof)
          {
            const unsigned int component =
              fe.face_system_to_component_index(idof).first;

            if (fe.has_support_on_face(idof, f) && is_lambda(component))
            {
              // Lambda DoF on the prescribed boundary: do not constrain
              unconstrained_lambda_dofs.insert(face_dofs[idof]);
            }
          }
        }
      }
    }
#endif

    // If there is no boundary with prescribed weak BC,
    // constrain all lambdas. This allows to keep the problem structure
    // as is, to test with only strong BC.
    if (param.weak_velocity_boundary_names.size() == 0)
      unconstrained_lambda_dofs.clear();

    // Add zero constraints to all lambda DOFs *not* in the boundary set
    IndexSet lambda_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(lambda));
    unsigned int n_constrained_dofs = 0;
    for (const auto dof : lambda_dofs)
    {
      // Only constrain owned DOFs
      if (!locally_owned_dofs.is_element(dof))
        continue;

      if (unconstrained_lambda_dofs.count(dof) == 0)
      {
        // lambda_constraints.add_line(dof); // Set dof to zero (by default)
        lambda_constraints.constrain_dof_to_zero(dof); // More readable (-:
        n_constrained_dofs++;
      }
    }
    lambda_constraints.close();

    pcout << unconstrained_lambda_dofs.size()
          << " lambda DOFs are unconstrained" << std::endl;
    pcout << n_constrained_dofs << " lambda DOFs are constrained" << std::endl;
  }

  template <int dim>
  void MMS<dim>::create_zero_constraints()
  {
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Boundaries where STRONG Dirichlet BC are applied,
    // where the Newton increment should be zero.

    for (auto str : param.position_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               Functions::ZeroFunction<dim>(
                                                 n_components),
                                               zero_constraints,
                                               fe.component_mask(position));
    }

    for (auto str : param.strong_velocity_boundary_names)
    {
      // This prescribes strong velocity BC at CURRENT mesh position,
      // but the mesh will move...
      VectorTools::interpolate_boundary_values(*mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               Functions::ZeroFunction<dim>(
                                                 n_components),
                                               zero_constraints,
                                               fe.component_mask(velocity));
    }

    bool set_to_zero = true;
    this->constrain_pressure_point(zero_constraints, set_to_zero);

    // Lambda constraints have to be enforced at each Newton iteration
    // Add them to both sets of constraints?
    zero_constraints.merge(
      lambda_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);

    zero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    for (auto str : param.position_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               solution_fun,
                                               nonzero_constraints,
                                               fe.component_mask(position));
    }

    for (auto str : param.strong_velocity_boundary_names)
    {
      // This prescribes strong velocity BC at CURRENT mesh position,
      // but the mesh will move...
      VectorTools::interpolate_boundary_values(*mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               solution_fun,
                                               nonzero_constraints,
                                               fe.component_mask(velocity));

      // Instead prescribe Dirichlet BC at future mesh position ?
      // Does not seem to work well yet.
      // VectorTools::interpolate_boundary_values(*fixed_mapping,
      //                                          dof_handler,
      //                                          mesh_domains_name2tag.at(str),
      //                                          solution_at_future_position_fun,
      //                                          nonzero_constraints,
      //                                          fe.component_mask(velocity));
    }

    // Strong Dirichlet BC for pressure for a single DoF
    bool set_to_zero = false;
    this->constrain_pressure_point(nonzero_constraints, set_to_zero);

    // Merge lambda constraints
    nonzero_constraints.merge(
      lambda_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);

    nonzero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::apply_nonzero_constraints()
  {
    nonzero_constraints.distribute(local_evaluation_point);
    nonzero_constraints.distribute(newton_update);
    evaluation_point = local_evaluation_point;
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void MMS<dim>::set_exact_solution()
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Scalar pressure(p_lower);
    const FEValuesExtractors::Vector position(x_lower);
    const FEValuesExtractors::Vector lambda(l_lower);

    // Update mesh position *BEFORE* evaluating fields
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point,
                             fe.component_mask(position));

    // Update MappingFEField *BEFORE* interpolating velocity/pressure
    evaluation_point = local_evaluation_point;

    // Set velocity and pressure with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point,
                             fe.component_mask(velocity));
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point,
                             fe.component_mask(pressure));
    // VectorTools::interpolate(*mapping,
    //                          dof_handler,
    //                          solution_fun,
    //                          local_evaluation_point,
    //                          fe.component_mask(lambda));

    // // Or alternatively evaluate with fixed mapping,
    // // on the future position x(X_0, t^n).
    // VectorTools::interpolate(*fixed_mapping,
    //                          dof_handler,
    //                          solution_at_future_position_fun,
    //                          local_evaluation_point,
    //                          fe.component_mask(velocity));
    // VectorTools::interpolate(*fixed_mapping,
    //                          dof_handler,
    //                          solution_at_future_position_fun,
    //                          local_evaluation_point,
    //                          fe.component_mask(pressure));

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
                                 weak_bc_boundary_id,
                                 bdfCoeffs);

    // Assemble pseudo-solid on initial mesh
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

    ////////////////////
    // unsigned int n_printed = 0;
    // double max_diff = 0.;
    ////////////////////

    // Assemble Navier-Stokes on current mesh
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

      // // std::cout << "Analytical non-elasticity matrix is " << std::endl;
      // // local_matrix.print(std::cout, 12, 3);

      // // Compare with FD matrix
      // FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
      // FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);
      // Vector<double>     ref_local_rhs(dofs_per_cell),
      //   perturbed_local_rhs(dofs_per_cell);
      // this->assemble_local_matrix_fd(first_step,
      //                                cell,
      //                                scratchData,
      //                                evaluation_point,
      //                                previous_solutions,
      //                                local_dof_indices,
      //                                local_matrix_fd,
      //                                ref_local_rhs,
      //                                perturbed_local_rhs,
      //                                cell_dof_values);

      // // std::cout << "FD         non-elasticity matrix is " << std::endl;
      // // local_matrix_fd.print(std::cout, 12, 3);

      // diff_matrix.equ(1.0, local_matrix);
      // diff_matrix.add(-1.0, local_matrix_fd);
      // // std::cout << "Error matrix is " << std::endl;
      // // diff_matrix.print(std::cout, 12, 3);
      // // std::cout << "Max difference is " << diff_matrix.linfty_norm()
      // //           << std::endl;
      // max_diff = std::max(max_diff, diff_matrix.linfty_norm());

      // // if(n_printed++ == 5)
      //   // throw std::runtime_error("Testing FD");
    }

    // pcout << "Max difference over all elements is " << max_diff << std::endl;
    // throw std::runtime_error("Testing FD");

    system_matrix.compress(VectorOperation::add);
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

    // pcout << "Reference non-perturbed RHS is " << ref_local_rhs << std::endl;

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

      if (is_position(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = cell_dof_values[j];
        local_evaluation_point.compress(VectorOperation::insert);
        evaluation_point = local_evaluation_point;
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

      // pcout << "Perturbed               RHS is " << perturbed_local_rhs <<
      // std::endl;

      // Finite differences (with sign change as residual is -NL(u))
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
      }

      // Restore solution
      cell_dof_values[j] = og_value;
      if (is_position(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = og_value;
        local_evaluation_point.compress(VectorOperation::insert);
        evaluation_point = local_evaluation_point;
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
                                       solution_fun,
                                       source_term_fun);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];
      const auto &phi_p      = scratchData.phi_p[q];
      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];

      const auto &present_velocity_values =
        scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients =
        scratchData.present_velocity_gradients[q];
      const double present_velocity_divergence =
        trace(present_velocity_gradients);
      const double present_pressure_values =
        scratchData.present_pressure_values[q];

      const auto &dxdt = scratchData.present_mesh_velocity_values[q];

#if defined(WITH_TRANSIENT_NS)
      // BDF: current dudt
      Tensor<1, dim> dudt = bdfCoeffs[0] * present_velocity_values;
      for (unsigned int i = 1; i < bdfCoeffs.size(); ++i)
        dudt += bdfCoeffs[i] * scratchData.previous_velocity_values[i - 1][q];
#endif

      const auto &source_term_velocity = scratchData.source_term_velocity[q];
      const auto &grad_source_velocity = scratchData.grad_source_velocity[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_u      = is_velocity(component_i);
        const bool         i_is_p      = is_pressure(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_u      = is_velocity(component_j);
          const bool         j_is_p      = is_pressure(component_j);
          const bool         j_is_x      = is_position(component_j);

          double local_matrix_ij = 0.;

          if (i_is_u && j_is_u)
          {
#if defined(WITH_TRANSIENT_NS)
            // Time-dependent
            local_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];
#endif

#if defined(WITH_CONVECTION)
            // Convection (OK)
            local_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                                present_velocity_gradients * phi_u[j]) *
                               phi_u[i];
#endif

            // Diffusion (OK)
            local_matrix_ij +=
              param.viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);

#if defined(WITH_TRANSIENT_NS)
            // ALE acceleration : - w dot grad(delta u)
            local_matrix_ij += -dxdt * grad_phi_u[j] * phi_u[i];
#endif
          }

          if (i_is_u && j_is_p)
          {
            // Pressure gradient (OK)
            local_matrix_ij += -div_phi_u[i] * phi_p[j];
          }

          if (i_is_u && j_is_x)
          {
#if defined(WITH_TRANSIENT_NS)
            // Variation of time-dependent term with mesh position
            local_matrix_ij += dudt * phi_u[i] * trace(grad_phi_x[j]);

            // Variation of ALE term (dxdt cdot grad(u)) with mesh position
            local_matrix_ij += present_velocity_gradients *
                               (-bdfCoeffs[0] * phi_x[j]) * phi_u[i];
            local_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               (-dxdt) * phi_u[i];
            local_matrix_ij += present_velocity_gradients * (-dxdt) * phi_u[i] *
                               trace(grad_phi_x[j]);
#endif

#if defined(WITH_CONVECTION)
            // Convection w.r.t. x (OK)
            local_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               present_velocity_values * phi_u[i];
            local_matrix_ij += present_velocity_gradients *
                               present_velocity_values * phi_u[i] *
                               trace(grad_phi_x[j]);
#endif

            // Diffusion (OK)
            const Tensor<2, dim> d_grad_u =
              -present_velocity_gradients * grad_phi_x[j];
            const Tensor<2, dim> d_grad_phi_u = -grad_phi_u[i] * grad_phi_x[j];
            local_matrix_ij +=
              param.viscosity * scalar_product(d_grad_u, grad_phi_u[i]);
            local_matrix_ij +=
              param.viscosity *
              scalar_product(present_velocity_gradients, d_grad_phi_u);
            local_matrix_ij +=
              param.viscosity *
              scalar_product(present_velocity_gradients, grad_phi_u[i]) *
              trace(grad_phi_x[j]);

            // Pressure gradient (OK)
            local_matrix_ij +=
              -present_pressure_values * trace(-grad_phi_u[i] * grad_phi_x[j]);
            local_matrix_ij +=
              -present_pressure_values * div_phi_u[i] * trace(grad_phi_x[j]);

            // Source term (OK):
            // Variation of the source term integral with mesh position.
            // det J is accounted for at the end when multiplying by JxW(q).
            local_matrix_ij += phi_u[i] * grad_source_velocity * phi_x[j];
            local_matrix_ij +=
              source_term_velocity * phi_u[i] * trace(grad_phi_x[j]);
          }

          if (i_is_p && j_is_u)
          {
            // Continuity : variation w.r.t. u (OK)
            local_matrix_ij += -phi_p[i] * div_phi_u[j];
          }

          if (i_is_p && j_is_x)
          {
            // Continuity : variation w.r.t. x (OK)
            local_matrix_ij +=
              -trace(-present_velocity_gradients * grad_phi_x[j]) * phi_p[i];
            local_matrix_ij +=
              -present_velocity_divergence * phi_p[i] * trace(grad_phi_x[j]);
          }

          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    //
    // Face contributions (Lagrange multiplier)
    //
    // unsigned int n_bdry_faces = 0.;
    // for (const auto &face : cell->face_iterators())
    //   if(face->at_boundary() && face->boundary_id() == weak_bc_boundary_id)
    //     n_bdry_faces++;
    // pcout << "Element has " << n_bdry_faces << " faces on the boundary" <<
    // std::endl;

    if (cell->at_boundary())
    {
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (face->at_boundary() && face->boundary_id() == weak_bc_boundary_id)
        {
          for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
          {
            const double JxW  = scratchData.face_JxW[i_face][q];
            const double W    = face_quadrature.weight(q);
            // const auto  &dXds = scratchData.face_dXds[i_face][q];

            const auto &phi_u      = scratchData.phi_u_face[i_face][q];
            const auto &phi_x      = scratchData.phi_x_face[i_face][q];
            const auto &grad_phi_x = scratchData.grad_phi_x_face[i_face][q];
            const auto &phi_l      = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            // const auto &present_x =
            // scratchData.present_face_position_values[i_face][q];
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];
            const auto &present_grad_x =
              scratchData.present_face_position_gradient[i_face][q];

            // For 2D only for now
            // const Tensor<1, dim> present_dxds = present_grad_x * dXds;
            // Use norm square to allow using JxW once for all contributions
            // Thus, this is not actually the tangent x/norm(x), it is
            // x/norm(x)^2.
            // const Tensor<1, dim> present_tangent =
            //   present_dxds / present_dxds.norm_square();

            const auto &prescribed_velocity_weak_bc =
              scratchData.prescribed_velocity_weak_bc[i_face][q];
            const auto &grad_solution_velocity =
              scratchData.grad_solution_velocity[i_face][q];

            // const Tensor<2, dim> &J = scratchData.face_jacobians[i_face][q];

            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              const unsigned int component_i = scratchData.components[i];
              const bool         i_is_u      = is_velocity(component_i);
              const bool         i_is_l      = is_lambda(component_i);

              for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              {
                const unsigned int component_j = scratchData.components[j];
                const bool         j_is_u      = is_velocity(component_j);
                const bool         j_is_x      = is_position(component_j);
                const bool         j_is_l      = is_lambda(component_j);

                // const double delta_dx_j = grad_phi_x[j] * dXds * present_tangent;
                const double delta_dx_2 = scratchData.delta_dx[i_face][q][j];

                double local_matrix_ij = 0.;

                // local_matrix(i, j) += (phi_u[j] - bdfCoeffs[0] * phi_x[j]) *
                // phi_l[i];

                if (i_is_u && j_is_x)
                {
                  local_matrix_ij += present_l * phi_u[i] * delta_dx_2;
                }

                if (i_is_u && j_is_l)
                {
                  local_matrix_ij += phi_l[j] * phi_u[i];
                }

                if (i_is_l && j_is_u)
                {
                  local_matrix_ij += phi_u[j] * phi_l[i];
                  // ADD MESH VELOCITY AFTER
                }

                if (i_is_l && j_is_x)
                {
                  // Variation of int_K1D (present_u -
                  // prescribed_velocity_weak_bc) * phi_l[i] dx
                  local_matrix_ij +=
                    -grad_solution_velocity * phi_x[j] * phi_l[i];
                  local_matrix_ij += (present_u - prescribed_velocity_weak_bc) *
                                     phi_l[i] * delta_dx_2;
                }

                local_matrix_ij *= JxW;
                local_matrix(i, j) += local_matrix_ij;
              }
            }
          }
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
                                     solution_fun,
                                     source_term_fun);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_x      = is_position(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_x      = is_position(component_j);

          double local_matrix_ij = 0.;

          // Position - Position block (x-x)
          if (i_is_x && j_is_x)
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

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<double>                  cell_dof_values(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Assemble pseudo-solid residual on initial mesh
    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 weak_bc_boundary_id,
                                 bdfCoeffs);

    // Assemble pseudo-solid RHS on initial mesh
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
                                         solution_fun,
                                         source_term_fun);
    }
    else
      scratchData.reinit_current_mapping(cell,
                                         cell_dof_values,
                                         previous_solutions,
                                         solution_fun,
                                         source_term_fun);

    local_rhs = 0;

    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<Tensor<1, dim>> velocity(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values =
        scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients =
        scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values =
        scratchData.present_pressure_values[q];
      const auto &present_mesh_velocity_values =
        scratchData.present_mesh_velocity_values[q];
      const auto  &source_term_velocity = scratchData.source_term_velocity[q];
      const double present_velocity_divergence =
        trace(present_velocity_gradients);

      // BDF
      velocity[0] = present_velocity_values;
      for (unsigned int i = 1; i < nBDF; ++i)
      {
        velocity[i] = scratchData.previous_velocity_values[i - 1][q];
      }

      const auto &phi_p      = scratchData.phi_p[q];
      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        double local_rhs_i =
          -(
            // Start with zero to allows + and - and ifdefs...
            0.

#if defined(WITH_CONVECTION)
            // Convection (OK)
            + (present_velocity_gradients * present_velocity_values) * phi_u[i]
#endif

#if defined(WITH_TRANSIENT_NS)
            // Mesh movement
            - (present_velocity_gradients * present_mesh_velocity_values) *
                phi_u[i]
        // - phi_u[i] * (w * present_velocity_gradients)
#endif

            // Diffusion (OK)
            + param.viscosity *
                scalar_product(present_velocity_gradients, grad_phi_u[i])

            // Pressure gradient (OK)
            - present_pressure_values * div_phi_u[i]

            // Momentum source term (OK)
            + source_term_velocity * phi_u[i]

            // Continuity (OK)
            - present_velocity_divergence * phi_p[i]) *
          JxW;

#if defined(WITH_TRANSIENT_NS)
        // Transient terms:
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
        {
          local_rhs_i -= bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i] * JxW;
        }
#endif

        local_rhs(i) += local_rhs_i;
      }
    }

    //
    // Face contributions (Lagrange multiplier)
    //
    if (cell->at_boundary())
    {
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (face->at_boundary() && face->boundary_id() == weak_bc_boundary_id)
        {
          for (unsigned int q = 0; q < scratchData.n_faces_q_points; ++q)
          {
            const double JxW   = scratchData.face_JxW[i_face][q];
            const auto  &phi_u = scratchData.phi_u_face[i_face][q];
            const auto  &phi_l = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            // const auto &present_w =
            // scratchData.present_face_mesh_velocity_values; const auto
            // &present_x = scratchData.present_face_position_values; const auto
            // &present_grad_x = scratchData.present_face_position_gradient;
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];

            const auto &prescribed_velocity_weak_bc =
              scratchData.prescribed_velocity_weak_bc[i_face][q];

            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              const unsigned int component_i = scratchData.components[i];
              const bool         i_is_u      = is_velocity(component_i);
              const bool         i_is_l      = is_lambda(component_i);

              if (i_is_u)
              {
                local_rhs(i) -= present_l * phi_u[i] * JxW;
              }

              if (i_is_l)
              {
                // local_rhs(i) -= (present_u[q] - present_w[q]) * phi_l[i] *
                // JxW;
                local_rhs(i) -=
                  (present_u - prescribed_velocity_weak_bc) * phi_l[i] * JxW;
              }

              /////////////////////////////////////////////////////
              // if (i_is_l)
              // {
              //   // Test with boundary element length/area:
              //   // local_rhs(i) -= sqrt(determinant(scratchData.face_G[i_face][q])) * W;
              //   // local_rhs(i) -= JxW;
              // }
              /////////////////////////////////////////////////////

            }
          }
        }
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
                                       solution_fun,
                                       source_term_fun);
    }
    else
      scratchData.reinit_fixed_mapping(cell,
                                       cell_dof_values,
                                       previous_solutions,
                                       solution_fun,
                                       source_term_fun);

    local_rhs = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_position_gradients =
        scratchData.present_position_gradients[q];

      const auto &source_term_position = scratchData.source_term_position[q];

      const double present_displacement_divergence =
        trace(present_position_gradients);

      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const auto present_displacement_gradient_sym =
          present_position_gradients + transpose(present_position_gradients);

        double local_rhs_i =
          -(
            // Linear elasticity
            +param.pseudo_solid_lambda * present_displacement_divergence *
              div_phi_x[i] +
            param.pseudo_solid_mu *
              scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

            // Linear elasticity source term
            + phi_x[i] * source_term_position) *
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
    const double       tol        = 1e-13;
    bool               converged  = false;

    while (current_res > tol && iter <= max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_rhs(first_step);

      // // If residual norm is low enough, return
      current_res = system_rhs.linfty_norm();
      // if (current_res <= tol)
      // {
      //   if (VERBOSE)
      //   {
      //     pcout << "Converged in " << iter
      //           << " iteration(s) because next nonlinear residual is below "
      //              "tolerance: "
      //           << current_res << " < " << tol << std::endl;
      //   }
      //   converged = true;
      //   break;
      // }

      this->assemble_matrix(first_step);
      this->solve_direct(first_step);
      first_step = false;
      iter++;

      norm_correction = newton_update.linfty_norm(); // On this proc only!
      if (VERBOSE)
      {
        pcout << std::scientific << std::setprecision(8)
              << "Newton iteration: " << iter
              << " - ||du|| = " << norm_correction
              << " - ||NL(u)|| = " << current_res << std::endl;
      }

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
      // After the first Newton iteration, the mesh has moved
      // and the Dirichlet BC no longer match.
      // The moving mapping has been updated through evaluation_point,
      // now recreate
      // this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->apply_nonzero_constraints();
      //////////////////////////////////////////////

      // this->assemble_rhs(first_step);
      // current_res = system_rhs.linfty_norm();

      if (VERBOSE && current_res <= tol)
      {
        pcout << "Converged in " << iter
              << " iteration(s) because next nonlinear residual is below "
                 "tolerance: "
              << current_res << " < " << tol << std::endl;
        converged = true;
      }

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
                                const unsigned int time_step)
  {
    // Plot FE solution
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("mesh_position");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("lambda");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < 2 * dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Plot exact solution
    std::vector<std::string> exact_solution_names(dim, "exact_velocity");
    exact_solution_names.push_back("exact_pressure");
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_mesh_position");
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_lambda");

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

    std::string root = "./ns_ale_mms" + std::to_string(convergence_index) + "/";
    // + "_" + std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator))
    // + "proc/";

    data_out.write_vtu_with_pvtu_record(
      root, "solution", time_step, mpi_communicator, 2);
  }

  template <int dim>
  void
  MMS<dim>::compute_lambda_error_on_boundary(const unsigned int boundary_id,
                                             double            &lambda_l2_error,
                                             double &lambda_linf_error,
                                             Tensor<1, dim> &error_on_integral)
  {
    double lambda_l2_local   = 0;
    double lambda_linf_local = 0;

    Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local, exact_integral_local;
    lambda_integral_local = 0;
    exact_integral_local = 0;

    const FEValuesExtractors::Vector lambda(l_lower);
    const unsigned int n_components = dof_handler.get_fe().n_components();

    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

    const unsigned int          n_faces_q_points = face_quadrature.size();
    std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);
    Tensor<1, dim>              diff, exact;

    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);

        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values.reinit(cell, i_face);

          // Get FE solution values on the face
          fe_face_values[lambda].get_function_values(present_solution,
                                                     lambda_values);

          // Evaluate exact solution at quadrature points
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
            const auto        normal = fe_face_values.normal_vector(q);

            for (unsigned int d = 0; d < dim; ++d)
              exact[d] = solution_fun.value(qpoint, normal, l_lower + d);

            diff = lambda_values[q] - exact;

            lambda_l2_local += diff * diff * fe_face_values.JxW(q);
            lambda_linf_local =
              std::max(lambda_linf_local, std::abs(diff.norm()));

            // Increment the integral of lambda
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
            exact_integral_local  += exact * fe_face_values.JxW(q);
          }
        }
      }
    }

    lambda_l2_error = Utilities::MPI::sum(lambda_l2_local, mpi_communicator);
    lambda_l2_error = std::sqrt(lambda_l2_error);

    lambda_linf_error =
      Utilities::MPI::max(lambda_linf_local, mpi_communicator);

    for(unsigned int d = 0; d < dim; ++d)
    {
      lambda_integral[d] = Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
      exact_integral[d]  = Utilities::MPI::sum(exact_integral_local[d], mpi_communicator);
      error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
    }
  }

  template <int dim>
  void MMS<dim>::compute_errors(const unsigned int time_step)
  {
    const unsigned int n_active_cells = triangulation.n_active_cells();

    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(u_lower,
                                                                    u_upper),
                                                     n_components);
    const ComponentSelectFunction<dim> pressure_mask(p_lower, n_components);
    const ComponentSelectFunction<dim> position_mask(std::make_pair(x_lower,
                                                                    x_upper),
                                                     n_components);
    const ComponentSelectFunction<dim> lambda_mask(std::make_pair(l_lower,
                                                                  l_upper),
                                                   n_components);

    Vector<double> cellwise_errors(n_active_cells);

    // Choose another quadrature rule for error computation
    const unsigned int n_points_1D = (dim == 2) ? 6 : 5;
    const QWitherdenVincentSimplex<dim> err_quadrature(n_points_1D);

    //
    // Linfty errors
    //

    // u
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::Linfty_norm,
                                      &velocity_mask);
    const double u_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_u = std::max(linf_error_u, u_linf);

    // p
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::Linfty_norm,
                                      &pressure_mask);
    const double p_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_p = std::max(linf_error_p, p_linf);

    // x
    VectorTools::integrate_difference(*fixed_mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::Linfty_norm,
                                      &position_mask);
    const double x_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_x = std::max(linf_error_x, x_linf);

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
    bool compute_mesh_velocity_error =
      !(param.bdf_order == 2 && time_step == 0);

    double w_linf = 0.;
    if (compute_mesh_velocity_error)
    {
      // Error on mesh velocity
      VectorTools::integrate_difference(*fixed_mapping,
                                        dof_handler,
                                        mesh_velocity,
                                        mesh_velocity_fun,
                                        cellwise_errors,
                                        err_quadrature,
                                        VectorTools::Linfty_norm,
                                        &position_mask);
      w_linf       = VectorTools::compute_global_error(triangulation,
                                                 cellwise_errors,
                                                 VectorTools::Linfty_norm);
      linf_error_w = std::max(linf_error_w, w_linf);
    }

    //
    // L2 errors
    //

    // u
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    // p
    VectorTools::integrate_difference(*mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    // x
    VectorTools::integrate_difference(*fixed_mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      err_quadrature,
                                      VectorTools::L2_norm,
                                      &position_mask);
    const double x_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    double w_l2_error = 0.;
    if (compute_mesh_velocity_error)
    {
      VectorTools::integrate_difference(*fixed_mapping,
                                        dof_handler,
                                        mesh_velocity,
                                        mesh_velocity_fun,
                                        cellwise_errors,
                                        err_quadrature,
                                        VectorTools::L2_norm,
                                        &position_mask);
      w_l2_error = VectorTools::compute_global_error(triangulation,
                                                     cellwise_errors,
                                                     VectorTools::L2_norm);
    }

    //
    // Errors for lambda on the relevant boundaries
    //
    // Do not compute at first time step for BDF2
    bool   compute_lambda_error = !(param.bdf_order == 2 && time_step == 0);
    double l_l2_error = 0., l_linf = 0.;
    Tensor<1, dim> error_on_integral;
    if (compute_lambda_error)
    {
      this->compute_lambda_error_on_boundary(weak_bc_boundary_id,
                                             l_l2_error,
                                             l_linf,
                                             error_on_integral);
      linf_error_l  = std::max(linf_error_l, l_linf);
      linf_error_Fx = std::max(linf_error_Fx, error_on_integral[0]);
      linf_error_Fy = std::max(linf_error_Fy, error_on_integral[1]);
    }

    l2_err_u += param.dt * u_l2_error;
    l2_err_p += param.dt * p_l2_error;
    l2_err_x += param.dt * x_l2_error;
    l2_err_w += param.dt * w_l2_error;
    l2_err_l += param.dt * l_l2_error;

    if (VERBOSE)
    {
      pcout << std::setprecision(3);
      pcout << "Current L2 errors: "
            << "||e_u||_L2 = " << u_l2_error << " - "
            << "||e_p||_L2 = " << p_l2_error << " - "
            << "||e_x||_L2 = " << x_l2_error << " - "
            << "||e_w||_L2 = " << w_l2_error << " - "
            << "||e_l||_L2 = " << l_l2_error << std::endl;
      pcout << "Cumul.  L2 errors: "
            << "||e_u||_L2 = " << l2_err_u << " - "
            << "||e_p||_L2 = " << l2_err_p << " - "
            << "||e_x||_L2 = " << l2_err_x << " - "
            << "||e_w||_L2 = " << l2_err_w << " - "
            << "||e_l||_L2 = " << l2_err_l << std::endl;
      pcout << "Current Li errors: "
            << "||e_u||_Li = " << u_linf << " - "
            << "||e_p||_Li = " << p_linf << " - "
            << "||e_x||_Li = " << x_linf << " - "
            << "||e_w||_Li = " << w_linf << " - "
            << "||e_l||_Li = " << l_linf << " - "
            << "      e_Fx = " << error_on_integral[0] << " - "
            << "      e_Fy = " << error_on_integral[1] << std::endl;
      pcout << "Cumul.  Li errors: "
            << "||e_u||_Li = " << linf_error_u << " - "
            << "||e_p||_Li = " << linf_error_p << " - "
            << "||e_x||_Li = " << linf_error_x << " - "
            << "||e_w||_Li = " << linf_error_w << " - "
            << "||e_l||_Li = " << linf_error_l << " - "
            << "||eFx||_Li = " << linf_error_Fx << " - "
            << "||eFy||_Li = " << linf_error_Fy << std::endl;

    }
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
    linf_error_u = 0.;
    linf_error_p = 0.;
    linf_error_x = 0.;
    linf_error_w = 0.;
    linf_error_l = 0.;
    linf_error_Fx = 0.;
    linf_error_Fy = 0.;
    l2_err_u     = 0.;
    l2_err_p     = 0.;
    l2_err_x     = 0.;
    l2_err_w     = 0.;
    l2_err_l     = 0.;

    // Constrained pressure DOF
    constrained_pressure_dof = numbers::invalid_dof_index;
  }

  template <int dim>
  void MMS<dim>::run()
  {
    unsigned int iMesh        = 1;
    unsigned int nConvergence = param.nConvergenceCycles;

    if (param.bdf_order == 0)
    {
      param.nTimeSteps = 1;
    }

    for (unsigned int iT = 1; iT <= nConvergence;
    ++iT, this->param.dt /= 2., this->param.nTimeSteps *= 2.)
    // for (unsigned int iT = 1; iT <= nConvergence; ++iT, ++iMesh)
    // for (unsigned int iT = 1; iT <= nConvergence;
    // ++iT, ++iMesh, this->param.dt /= 2., this->param.nTimeSteps *= 2.)
    {
      // pcout << "Convergence step " << iT << "/" << nConvergence << std::endl;

      this->reset();

      this->param.prev_dt = this->param.dt;
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->solution_fun.set_time(current_time);
      this->source_term_fun.set_time(current_time);
      this->mesh_velocity_fun.set_time(current_time);
      this->solution_at_future_position_fun.set_time(current_time);

      this->make_grid(iMesh);
      this->setup_system();
      this->create_lambda_zero_constraints(weak_bc_boundary_id);
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
        this->solution_at_future_position_fun.set_time(current_time);

        if (VERBOSE)
        {
          pcout << std::endl
                << "Time step " << i + 1
                << " - Advancing to t = " << current_time << '.' << std::endl;
        }

        this->create_nonzero_constraints();
        this->apply_nonzero_constraints();

        if (i == 0 && param.bdf_order == 2)
        {
          // For BDF2: set first step to exact solution
          this->set_exact_solution();

          // TODO: Or compute first step with BDF1 and smaller time step
        }
        else if (i == 1 && param.bdf_order == 3)
        {
          // Also set exact solution at 2nd time step for BDF2
          this->set_exact_solution();
        }
        else
        {
          this->solve_newton();
        }

        // this->set_exact_solution();

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
      convergence_table.add_value("L2_p", l2_err_p);
      convergence_table.add_value("Li_p", linf_error_p);
      convergence_table.add_value("L2_x", l2_err_x);
      convergence_table.add_value("Li_x", linf_error_x);
      convergence_table.add_value("L2_w", l2_err_w);
      convergence_table.add_value("Li_w", linf_error_w);
      convergence_table.add_value("L2_l", l2_err_l);
      convergence_table.add_value("Li_l", linf_error_l);
      convergence_table.add_value("e_Fx", linf_error_Fx);
      convergence_table.add_value("e_Fy", linf_error_Fy);
    }

    convergence_table.evaluate_convergence_rates(
      "L2_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_p", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_x", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_w", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_l", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_p", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_x", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_w", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_l", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "e_Fx", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "e_Fy", ConvergenceTable::reduction_rate_log2);

    // Arrange convergence table
    convergence_table.set_precision("L2_u", 4);
    convergence_table.set_precision("L2_p", 4);
    convergence_table.set_precision("L2_x", 4);
    convergence_table.set_precision("L2_w", 4);
    convergence_table.set_precision("L2_l", 4);
    convergence_table.set_precision("Li_u", 4);
    convergence_table.set_precision("Li_p", 4);
    convergence_table.set_precision("Li_x", 4);
    convergence_table.set_precision("Li_w", 4);
    convergence_table.set_precision("Li_l", 4);
    convergence_table.set_precision("e_Fx", 4);
    convergence_table.set_precision("e_Fy", 4);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.set_scientific("L2_x", true);
    convergence_table.set_scientific("L2_w", true);
    convergence_table.set_scientific("L2_l", true);
    convergence_table.set_scientific("Li_u", true);
    convergence_table.set_scientific("Li_p", true);
    convergence_table.set_scientific("Li_x", true);
    convergence_table.set_scientific("Li_w", true);
    convergence_table.set_scientific("Li_l", true);
    convergence_table.set_scientific("e_Fx", true);
    convergence_table.set_scientific("e_Fy", true);

    pcout << "BDF order: " << param.bdf_order << " - Velocity P"
          << param.velocity_degree << " - Pressure P"
          << param.velocity_degree - 1 << " - Position P"
          << param.position_degree << std::endl;
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
    using namespace ManufacturedSolution;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    if (rank == 0)
      print_ifdefs();

    SimulationParameters param;

    param.velocity_degree = 2;
    param.position_degree = 1;
    param.lambda_degree   = 2;

    const bool with_weak_velocity_bc = true;

    param.position_boundary_names = {"OuterBoundary", "InnerBoundary"};
    if(with_weak_velocity_bc)
    {
      param.strong_velocity_boundary_names = {"OuterBoundary"};
      param.weak_velocity_boundary_names   = {"InnerBoundary"};
    }
    else
    {
      param.strong_velocity_boundary_names = {"OuterBoundary", "InnerBoundary"};
      param.weak_velocity_boundary_names   = {};
    }

    param.viscosity           = VISCOSITY;
    param.pseudo_solid_mu     = MU_PS;
    param.pseudo_solid_lambda = LAMBDA_PS;

    // Time integration
    param.bdf_order  = 2;
    param.t0         = 0.;
    param.dt         = 0.1;
    param.nTimeSteps = 11;
    param.t1         = param.dt * param.nTimeSteps;

    param.nConvergenceCycles = 3;

    VERBOSE = true;

    // Possible time dependences G(t)
    std::vector<TimeDependenceBase *> time_functions;
    // time_functions.push_back(new ConstantTimeDep);
    // time_functions.push_back(new PowerTimeDep(1));
    // time_functions.push_back(new PowerTimeDep(2));
    time_functions.push_back(new PowerTimeDep(3));
    // time_functions.push_back(new SineTimeDep);

    const unsigned int dim = 2;

    for (unsigned int iTest = 0; iTest < time_functions.size(); ++iTest)
    {
      // const FlowA<2> flow_mms(*time_functions[iTest]);
      // const FlowB<2> flow_mms(*time_functions[iTest]);
      // const FlowC<2> flow_mms(*time_functions[iTest]);
      // const FlowD<dim> flow_mms(*time_functions[iTest]);

      const double          dpdx = 1.;
      const Poiseuille<dim> flow_mms(*time_functions[iTest],
                                     dpdx,
                                     param.viscosity);

      // WITH_MESH_DISPLACEMENT = false;
      // {
      //   MMS<dim> problem(param, flow_mms);
      //   problem.run();
      // }
      WITH_MESH_DISPLACEMENT = true;
      {
        MMS<dim> problem(param, flow_mms);
        problem.run();
      }
    }
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
