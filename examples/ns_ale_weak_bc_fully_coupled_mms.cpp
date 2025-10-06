
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

/**
 * This tests the implementation of the no-slip
 * on the obstacle, coupling the fluid velocity
 * with the mesh velocity.
 *
 * The manufactured mesh displacement is divergence free
 * and the fluid velocity is d/dt(displacement), so that
 * we have u = dx/dt on the obstacle without source term.
 *
 * The displacement is imposed (strong Dirichlet), and *not*
 * coupled with the Lagrange multiplier.
 */

bool VERBOSE = false;

// Fluid
#define VISCOSITY 1.

// Pseudo-solid
#define LAMBDA_PS 1.
#define MU_PS     1.

#define NO_SLIP_ON_CYLINDER
// #define COMPARE_ANALYTIC_MATRIX_WITH_FD

namespace NS_MMS
{
  using namespace dealii;
  using namespace ManufacturedSolution;

  enum class ConvergenceStudy
  {
    TIME,
    SPACE,
    TIME_AND_SPACE
  };

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    const FlowManufacturedSolutionBase<dim> &flow_mms;
    const MeshPositionMMSBase<dim>          &mesh_mms;
    const unsigned int                       n_components;
    const unsigned int                       u_lower = 0;
    const unsigned int                       p_lower = dim;
    const unsigned int                       x_lower = dim + 1;
    const unsigned int                       l_lower = 2 * dim + 1;

  public:
    Solution(const double                             time,
             const unsigned int                       n_components,
             const FlowManufacturedSolutionBase<dim> &flow_mms,
             const MeshPositionMMSBase<dim>          &mesh_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
      , mesh_mms(mesh_mms)
      , n_components(n_components)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      const double t = this->get_time();

      Vector<double> values(n_components);

      values[p_lower] = flow_mms.pressure(t, p);
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = flow_mms.velocity(t, p, d);
        values[x_lower + d] = mesh_mms.position(t, p, d);
        values[l_lower + d] = 0.;
      }

      return values[component];
    }

    double value(const Point<dim>     &p,
                 const Tensor<1, dim> &normal_to_solid,
                 const unsigned int    component = 0) const
    {
      // Used only to evaluate Lagrange multiplier on the
      // interior of boundary elements
      if (!(l_lower <= component && component < l_lower + dim))
        DEAL_II_ASSERT_UNREACHABLE();

      const double t = this->get_time();

      Tensor<1, dim> lambda;
      Tensor<2, dim> sigma;
      flow_mms.newtonian_stress(t, p, VISCOSITY, sigma);
      lambda = -sigma * normal_to_solid;

      return lambda[component - l_lower];
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t = this->get_time();

      values[p_lower] = flow_mms.pressure(t, p);
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = flow_mms.velocity(t, p, d);
        values[x_lower + d] = mesh_mms.position(t, p, d);
        values[l_lower + d] = 0.;
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
    const MeshPositionMMSBase<dim>          &mesh_mms;
    const unsigned int                       u_lower = 0;
    const unsigned int                       p_lower = dim;
    const unsigned int                       x_lower = dim + 1;
    const unsigned int                       l_lower = 2 * dim + 1;

  public:
    SolutionAtFutureMeshPosition(
      const double                             time,
      const unsigned int                       n_components,
      const FlowManufacturedSolutionBase<dim> &flow_mms,
      const MeshPositionMMSBase<dim>          &mesh_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
      , mesh_mms(mesh_mms)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      const double t = this->get_time();

      // Get prescribed mesh position
      Point<dim> pFinal;
      for (unsigned int d = 0; d < dim; ++d)
        pFinal[d] = mesh_mms.position(t, p, d);

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
      const double t = this->get_time();

      // Get prescribed mesh position for this point
      Point<dim> pFinal;
      for (unsigned int d = 0; d < dim; ++d)
        pFinal[d] = mesh_mms.position(t, p, d);

      values[p_lower] = flow_mms.pressure(t, pFinal);
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = flow_mms.velocity(t, pFinal, d);
        values[x_lower + d] = mesh_mms.position(t, p, d);
        values[l_lower + d] = 0.;
      }
    }
  };


  template <int dim>
  class MeshVelocity : public Function<dim>
  {
  public:
    const MeshPositionMMSBase<dim> &mesh_mms;
    const unsigned int              u_lower = 0;
    const unsigned int              p_lower = dim;
    const unsigned int              x_lower = dim + 1;
    const unsigned int              l_lower = 2 * dim + 1;

  public:
    MeshVelocity(const double                    time,
                 const unsigned int              n_components,
                 const MeshPositionMMSBase<dim> &mesh_mms)
      : Function<dim>(n_components, time)
      , mesh_mms(mesh_mms)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      if (!(x_lower <= component && component < x_lower + dim))
        DEAL_II_ASSERT_UNREACHABLE();
      const double t = this->get_time();
      return mesh_mms.mesh_velocity(t, p, component - x_lower);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t = this->get_time();
      for (unsigned int d = 0; d < dim; ++d)
        values[x_lower + d] = mesh_mms.mesh_velocity(t, p, d);
    }
  };

  template <int dim>
  class SourceTerm : public Function<dim>
  {
  public:
    const FlowManufacturedSolutionBase<dim> &flow_mms;
    const MeshPositionMMSBase<dim>          &mesh_mms;
    const unsigned int                       u_lower = 0;
    const unsigned int                       p_lower = dim;
    const unsigned int                       x_lower = dim + 1;
    const unsigned int                       l_lower = 2 * dim + 1;

  public:
    SourceTerm(const double                             time,
               const unsigned int                       n_components,
               const FlowManufacturedSolutionBase<dim> &flow_mms,
               const MeshPositionMMSBase<dim>          &mesh_mms)
      : Function<dim>(n_components, time)
      , flow_mms(flow_mms)
      , mesh_mms(mesh_mms)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t  = this->get_time();
      const double mu = VISCOSITY;

      // 
      // Navier-Stokes momentum
      //
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
      f = -(dudt_eulerian + uDotGradu + grad_p - mu * lap_u);

      for (unsigned int d = 0; d < dim; ++d)
        values[u_lower + d] = f[d];

      //
      // Pressure
      //
      values[p_lower] = flow_mms.velocity_divergence(t, p);

      //
      // Pseudo-solid
      //
      // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
      Tensor<1, dim> f_PS;
      mesh_mms.divergence_stress_tensor(t, p, MU_PS, LAMBDA_PS, f_PS);

      for (unsigned int d = 0; d < dim; ++d)
        values[x_lower + d] = f_PS[d];

      //
      // Lagrange multiplier: to have u = dxdt on boundary.
      // dxdt must be evaluated on initial mesh!
      // For now, return only u(x,t) on current mesh, and
      // "assemble" the lambda source term where it is needed.
      //
      Tensor<1, dim> dxdt;
      // mesh_mms.mesh_velocity(t, pInitial, dxdt);
      // Tensor<1, dim> f_lambda = - (u - dxdt);
      for (unsigned int d = 0; d < dim; ++d)
        values[l_lower + d] = u[d];
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
                const Function<dim>                &source_term_fun,
                const Function<dim>                &mesh_velocity_fun)
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
      source_term_fun.vector_value_list(active_fe_values->get_quadrature_points(), source_term_full);
      // source_term_fun.vector_value_list(fe_values_fixed_mapping.get_quadrature_points(), source_term_full);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (int d = 0; d < dim; ++d)
          source_term_velocity[q][d] = source_term_full[q](u_lower + d);

        source_term_pressure[q] = source_term_full[q](p_lower);

        for (int d = 0; d < dim; ++d)
          source_term_position[q][d] = source_term_full[q](x_lower + d);
      }

      // Gradient of source term
      // Only need to fill in for the scalar field
      source_term_fun.vector_gradient_list(active_fe_values->get_quadrature_points(), grad_source_term_full);
      // source_term_fun.vector_gradient_list(fe_values_fixed_mapping.get_quadrature_points(), grad_source_term_full);

      // Layout: grad_source_velocity[q] = df_i/dx_j
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (int di = 0; di < dim; ++di)
        {
          grad_source_pressure[q][di] = grad_source_term_full[q][p_lower][di];
          for (int dj = 0; dj < dim; ++dj)
            grad_source_velocity[q][di][dj] =
              grad_source_term_full[q][u_lower + di][dj];
        }
      }

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
        fe_face_values_fixed_mapping.reinit(cell, face);

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
        //   std::cout << std::setprecision(10) << "  " << cell->vertex(iv) <<
        //   "\n";
        // std::cout << "  Face " << i_face << " vertices:\n";
        // for (unsigned int iv = 0; iv < face->n_vertices(); ++iv)
        //   std::cout << std::setprecision(10) << "    " << face->vertex(iv) <<
        //   "\n";
        // std::cout << std::setprecision(10) << "J = " <<
        // active_fe_face_values->jacobian(0) << std::endl;
        // ////////////////////////////////////////////

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          face_JxW[i_face][q]       = active_fe_face_values->JxW(q);
          face_jacobians[i_face][q] = active_fe_face_values->jacobian(q);
          const Tensor<2, dim> J    = active_fe_face_values->jacobian(q);

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
                dxsids_array[0][1] = 1.;
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
                dxsids_array[0][0] = -1.;
                dxsids_array[1][0] = -1.;
                dxsids_array[0][1] = 1.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 2: // Opposite to v1
                dxsids_array[0][0] = 0.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 1.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 1: // Opposite to v2
                dxsids_array[0][0] = 1.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.;
                dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 1.;
                break;
              case 0: // Opposite to v3
                dxsids_array[0][0] = 1.;
                dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.;
                dxsids_array[1][1] = 1.;
                dxsids_array[0][2] = 0.;
                dxsids_array[1][2] = 0.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          }

          face_dXds[i_face][q] = face_jacobians[i_face][q] * dxsids;

          Tensor<2, dim - 1> G;
          G = 0;
          for (unsigned int di = 0; di < dim - 1; ++di)
            for (unsigned int dj = 0; dj < dim - 1; ++dj)
              for (unsigned int im = 0; im < dim; ++im)
                for (unsigned int in = 0; in < dim; ++in)
                  for (unsigned int ip = 0; ip < dim; ++ip)
                    G[di][dj] += dxsids_array[di][im] * J[in][im] * J[in][ip] *
                                 dxsids_array[dj][ip];
          face_G[i_face][q] = G;
          // const double             sqrt_det_G = sqrt(determinant(G));
          const Tensor<2, dim - 1> G_inverse = invert(G);
          // std::cout << "G is " << G << std::endl;

          // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
          Tensor<2, dim - 1> res;

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
            // std::cout << "pres_grad_x is " <<
            // present_face_position_gradient[i_face][q] << std::endl;


            Tensor<2, dim> A =
              transpose(J) *
              (transpose(present_face_position_gradient[i_face][q]) *
                 grad_phi_x +
               transpose(grad_phi_x) *
                 present_face_position_gradient[i_face][q]) *
              J;

            // std::cout << "A           is " << A << std::endl;

            res = 0;
            for (unsigned int di = 0; di < dim - 1; ++di)
              for (unsigned int dj = 0; dj < dim - 1; ++dj)
                for (unsigned int im = 0; im < dim - 1; ++im)
                  for (unsigned int in = 0; in < dim; ++in)
                    for (unsigned int io = 0; io < dim; ++io)
                      res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] *
                                     A[in][io] * dxsids_array[dj][io];
            // std::cout << "G_inverse   is " << G_inverse << std::endl;
            // std::cout << "res         is " << res << std::endl;
            // delta_dx[i_face][q][k] = 0.5 * sqrt_det_G * trace(res); // Choose
            // this if multiplying by W
            delta_dx[i_face][q][k] =
              0.5 *
              trace(res); // Choose this if multiplying by JxW in the matrix
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

        // Face source term and gradient
        source_term_fun.vector_value_list(active_fe_face_values->get_quadrature_points(), face_source_velocity);
        mesh_velocity_fun.vector_value_list(fe_face_values_fixed_mapping.get_quadrature_points(), face_source_mesh_velocity);
        source_term_fun.vector_gradient_list(active_fe_face_values->get_quadrature_points(), face_grad_source_term_full);
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          for (int di = 0; di < dim; ++di)
          {
            // face_source_term_lambda[i_face][q][di] = face_source_term_full[q](l_lower + di);

            // Lambda source terms requires evaluating mesh velocity on initial mesh.
            // Assemble it here instead after having evaluated u on current mesh,
            // and w on initial mesh.
            const double u = face_source_velocity[q](l_lower + di);
            const double w = face_source_mesh_velocity[q](x_lower + di);
            face_source_term_lambda[i_face][q][di] = -(u - w);

            for (int dj = 0; dj < dim; ++dj)
              face_grad_source_lambda[i_face][q][di][dj] =
                face_grad_source_term_full[q][l_lower + di][dj];
          }

          // Continue testing with an MMS without source term for now.
          AssertThrow(face_source_term_lambda[i_face][q].norm() < 1e-13,
                    ExcMessage("Lambda source term is implemented, but expected to be zero for now"));
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
      const Function<dim>                                  &source_term_fun,
      const Function<dim>                                  &mesh_velocity_fun)
    {
      active_fe_values      = &fe_values;
      active_fe_face_values = &fe_face_values;
      this->reinit(cell,
                   current_solution,
                   previous_solutions,
                   solution_fun,
                   source_term_fun,
                   mesh_velocity_fun);
    }

    template <typename VectorType>
    void reinit_fixed_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                                     &current_solution,
      const std::vector<LA::MPI::Vector>                   &previous_solutions,
      const Function<dim>                                  &solution_fun,
      const Function<dim>                                  &source_term_fun,
      const Function<dim>                                  &mesh_velocity_fun)
    {
      active_fe_values      = &fe_values_fixed_mapping;
      active_fe_face_values = &fe_face_values_fixed_mapping;
      this->reinit(cell,
                   current_solution,
                   previous_solutions,
                   solution_fun,
                   source_term_fun,
                   mesh_velocity_fun);
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
    std::array<Tensor<1, dim>, dim - 1>          dxsids_array;
    std::vector<std::vector<Tensor<2, dim - 1>>> face_G;
    // std::vector<std::vector<std::array<Tensor<1, dim>, dim - 1>>>
    // face_preFactor;

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
    std::vector<double>         source_term_pressure;
    std::vector<Tensor<1, dim>> source_term_position;

    // Gradient of source term,
    // at each quad node, for each dof component, result is a Tensor<1, dim>
    std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
    std::vector<Tensor<2, dim>>              grad_source_velocity;
    std::vector<Tensor<1, dim>>              grad_source_pressure;

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

    // Source term on faces
    // std::vector<Vector<double>>              face_source_term_full;
    std::vector<Vector<double>>              face_source_velocity;
    std::vector<Vector<double>>              face_source_mesh_velocity;
    std::vector<std::vector<Tensor<1, dim>>> face_grad_source_term_full;
    std::vector<std::vector<Tensor<1, dim>>> face_source_term_lambda;
    std::vector<std::vector<Tensor<2, dim>>> face_grad_source_lambda;

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
    source_term_pressure.resize(n_q_points);
    source_term_position.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);
    grad_source_pressure.resize(n_q_points);

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

    // face_source_term_full.resize(n_faces_q_points, Vector<double>(n_components));
    face_source_velocity.resize(n_faces_q_points, Vector<double>(n_components));
    face_source_mesh_velocity.resize(n_faces_q_points, Vector<double>(n_components));
    face_grad_source_term_full.resize(n_faces_q_points, std::vector<Tensor<1, dim>>(n_components));
    face_source_term_lambda.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    face_grad_source_lambda.resize(n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_jacobians.resize(n_faces,
                          std::vector<Tensor<2, dim>>(n_faces_q_points));
    // face_jacobians.resize(n_faces, std::vector<DerivativeForm<1, dim,
    // dim>>(n_faces_q_points));
    face_dXds.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    face_G.resize(n_faces, std::vector<Tensor<2, dim - 1>>(n_faces_q_points));
    delta_dx.resize(n_faces,
                    std::vector<std::vector<double>>(
                      n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }

  template <int dim>
  class SimulationParameters
  {
  public:
    unsigned int velocity_degree;
    unsigned int position_degree;
    unsigned int lambda_degree;

    std::vector<std::string> position_boundary_names;
    std::vector<std::string> strong_velocity_boundary_names;
    std::vector<std::string> weak_velocity_boundary_names;

    // Boundaries on which we want to compute the error ||w - wh||
    std::vector<std::string> mesh_velocity_error_boundary_names;

    bool with_weak_velocity_bc;
    bool with_position_coupling;

    Tensor<1, dim> translation;

    double       viscosity;
    double       pseudo_solid_mu;
    double       pseudo_solid_lambda;
    double       spring_constant;

    unsigned int bdf_order;
    double       t0;
    double       t1;
    double       dt;
    double       prev_dt;
    unsigned int nTimeSteps;

    double newton_tolerance = 1e-13;

    ConvergenceStudy type_of_convergence_study;
    unsigned int     nConvergenceCycles;
    unsigned int     starting_mesh;

  public:
    SimulationParameters<dim>(){};
  };

  template <int dim>
  class MMS
  {
  public:
    MMS(const SimulationParameters<dim>         &param,
        const FlowManufacturedSolutionBase<dim> &flow_mms,
        const MeshPositionMMSBase<dim>          &mesh_mms);

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
    void create_position_lambda_coupling_constraints(
      const unsigned int boundary_id);
    void apply_position_lambda_constraints(const unsigned int boundary_id,
                                           const bool homogeneous);
    void create_sparsity_pattern();
    void set_initial_condition();
    // void apply_nonzero_constraints();
    void update_boundary_conditions();
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
    void solve_newton2(const bool is_initial_step);
    void output_results(const unsigned int convergence_index,
                        const unsigned int time_step,
                        const bool         write_newton_iteration = false,
                        const unsigned int newton_step = 0);
    void compute_errors(const unsigned int time_step);
    void compute_lambda_error_on_boundary(const unsigned int boundary_id,
                                          double            &lambda_l2_error,
                                          double            &lambda_linf_error,
                                          Tensor<1, dim>    &error_on_integral);
    void compute_boundary_errors(const unsigned int boundary_id,
                                 double            &l2_error_dxdt,
                                 double            &linf_error_dxdt,
                                 double            &l2_error_fluid_velocity,
                                 double            &linf_error_fluid_velocity,
                                 double            &l2_x_error,
                                 double            &linf_x_error);
    void check_velocity_boundary(const unsigned int boundary_id);
    void compare_lambda_position_on_boundary(const unsigned int boundary_id);
    void check_manufactured_solution_boundary(const unsigned int boundary_id);
    void reset();

    SimulationParameters<dim> param;

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
    AffineConstraints<double> velocity_constraints;

    // Position-lambda constraints on the cylinder
    AffineConstraints<double> position_constraints;
    // The affine coefficients c_ij: [dim][{lambdaDOF_j : c_ij}]
    std::vector<std::vector<std::pair<unsigned int, double>>> position_lambda_coeffs;
    std::map<types::global_dof_index, Point<dim>> initial_positions;

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
    unsigned int mesh_velocity_error_boundary_id;

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
    unsigned int current_time_step;
    unsigned int current_convergence_cycle;

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
    double l2_err_w_boundary;
    double l2_err_u_boundary;
    double l2_err_x_boundary;
    double l2_err_l;
    double l1_time_error_Fx;
    double l1_time_error_Fy;
    // Linf in space and time
    double linf_error_u;
    double linf_error_p;
    double linf_error_x;
    double linf_error_w;
    double linf_error_w_boundary;
    double linf_error_u_boundary;
    double linf_error_x_boundary;
    double linf_error_l;
    double linf_error_Fx;
    double linf_error_Fy;

    // Use vector of pairs to maintain prescribed order
    std::vector<std::pair<std::string, const double*>> domain_errors;
    std::vector<std::pair<std::string, const double*>> boundary_errors;

    ConvergenceTable convergence_domain;
    ConvergenceTable convergence_boundary;

    const FlowManufacturedSolutionBase<dim> &flow_mms;
    const MeshPositionMMSBase<dim>          &mesh_mms;
  };

  template <int dim>
  MMS<dim>::MMS(const SimulationParameters<dim>         &param,
                const FlowManufacturedSolutionBase<dim> &flow_mms,
                const MeshPositionMMSBase<dim>          &mesh_mms)
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
    , solution_fun(
        Solution<dim>(current_time, n_components, flow_mms, mesh_mms))
    , source_term_fun(
        SourceTerm<dim>(current_time, n_components, flow_mms, mesh_mms))
    , mesh_velocity_fun(MeshVelocity<dim>(current_time, n_components, mesh_mms))
    , solution_at_future_position_fun(
        SolutionAtFutureMeshPosition<dim>(current_time,
                                          n_components,
                                          flow_mms,
                                          mesh_mms))
    , flow_mms(flow_mms)
    , mesh_mms(mesh_mms)
  {
    domain_errors.push_back({"L2_u", &l2_err_u});
    domain_errors.push_back({"Li_u", &linf_error_u});
    domain_errors.push_back({"L2_p", &l2_err_p});
    domain_errors.push_back({"Li_p", &linf_error_p});
    domain_errors.push_back({"L2_x", &l2_err_x});
    domain_errors.push_back({"Li_x", &linf_error_x});
    domain_errors.push_back({"L2_w", &l2_err_w});
    domain_errors.push_back({"Li_w", &linf_error_w});

    // boundary_errors.push_back({"|w-wh|L2", &l2_err_w_boundary});
    // boundary_errors.push_back({"|w-wh|Li", &linf_error_w_boundary});
    boundary_errors.push_back({"|w-uh|L2", &l2_err_u_boundary});
    boundary_errors.push_back({"|w-uh|Li", &linf_error_u_boundary});
    boundary_errors.push_back({"|x-xh|L2", &l2_err_x_boundary});
    boundary_errors.push_back({"|x-xh|Li", &linf_error_x_boundary});
    boundary_errors.push_back({"L2_l",  &l2_err_l});
    boundary_errors.push_back({"Li_l",  &linf_error_l});
    boundary_errors.push_back({"L1_Fx",  &l1_time_error_Fx});
    boundary_errors.push_back({"L1_Fy",  &l1_time_error_Fy});
    boundary_errors.push_back({"Li_Fx",  &linf_error_Fx});
    boundary_errors.push_back({"Li_Fy",  &linf_error_Fy});
  }

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
      // meshFile = "../data/meshes/holed" + std::to_string(iMesh) + ".msh";
      meshFile = "../data/meshes/holed_far" + std::to_string(iMesh) + ".msh";
      // meshFile = "../data/meshes/holed_rectangle" + std::to_string(iMesh) + ".msh";
      // meshFile = "../data/meshes/holed_square" + std::to_string(iMesh) + ".msh";
      // meshFile = "../data/meshes/holed_square_far" + std::to_string(iMesh) + ".msh";
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

    for (auto str : param.mesh_velocity_error_boundary_names)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("Mesh velocity error should be "
                                 "computed on the boundary named \"" +
                                 str +
                                 "\", but no physical entity with this name "
                                 "was read from the mesh file.");
      }
      mesh_velocity_error_boundary_id = mesh_domains_name2tag.at(str);
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
      const Point<dim> reference_point(11., 10.);

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

    //
    // For continuous lambda
    //
    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // if (!cell->is_locally_owned())
      //   continue;

      for (const auto f : cell->face_indices())
      {
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == boundary_id)
        {
          cell->face(f)->get_dof_indices(face_dofs);
          for (unsigned int idof = 0; idof < fe.n_dofs_per_face(); ++idof)
          {
            if (!locally_owned_dofs.is_element(face_dofs[idof]))
              continue;

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

    // If there is no boundary with prescribed weak BC,
    // constrain all lambdas. This allows to keep the problem structure
    // as is, to test with only strong BC.
    if (param.weak_velocity_boundary_names.size() == 0)
      unconstrained_lambda_dofs.clear();

    // Add zero constraints to all lambda DOFs *not* in the boundary set
    IndexSet lambda_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(lambda));
    unsigned int n_constrained_local = 0;
    for (const auto dof : lambda_dofs)
    {
      // Only constrain owned DOFs
      if (!locally_owned_dofs.is_element(dof))
        continue;

      if (unconstrained_lambda_dofs.count(dof) == 0)
      {
        // lambda_constraints.add_line(dof); // Set dof to zero (by default)
        lambda_constraints.constrain_dof_to_zero(dof); // More readable (-:
        n_constrained_local++;
      }
    }
    lambda_constraints.close();

    const unsigned int n_unconstrained = Utilities::MPI::sum(unconstrained_lambda_dofs.size(), mpi_communicator);
    const unsigned int n_constrained = Utilities::MPI::sum(n_constrained_local, mpi_communicator);

    if(VERBOSE)
    {
      pcout << n_unconstrained
            << " lambda DOFs are unconstrained" << std::endl;
      pcout << n_constrained << " lambda DOFs are constrained" << std::endl;
    }
  }

  /**
   * Create the affine constraints between position and lambda on the cylinder.
   * On the cylinder, we have
   * 
   * x = X - int_Gamma lambda dx,
   * 
   * yielding the affine constraints
   * 
   * x_i = X_i + sum_j c_ij * lambda_j, with c_ij = - int_Gamma phi_global_j dx.
   * 
   * Each position DoF is linked to all lambda DoF on the cylinder, which may
   * not be owned of even ghosts of the current process. In a first naive approach,
   * all cylinder lambda DoF are added as relevant DoF to the current process if it has at least
   * one lambda DoF on the cylinder.
   * 
   * boundary_id: the id of the cylinder boundary.
   * 
   * homogeneous: if true, do not add the initial position X_i as inhomogeneity.
   *              Add them if false.
   */
  template <int dim>
  void MMS<dim>::create_position_lambda_coupling_constraints(
    const unsigned int boundary_id)
  {
    const FEValuesExtractors::Vector position(x_lower);
    const FEValuesExtractors::Vector lambda(l_lower);

    //
    // Get and synchronize the lambda DoFs on the cylinder
    //
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(boundary_id);

    IndexSet local_lambda_dofs =
      DoFTools::extract_boundary_dofs(dof_handler,
                                      fe.component_mask(lambda),
                                      boundary_ids);
    IndexSet local_position_dofs =
      DoFTools::extract_boundary_dofs(dof_handler,
                                      fe.component_mask(position),
                                      boundary_ids);

    const unsigned int n_local_lambda_dofs = local_lambda_dofs.n_elements();

    local_lambda_dofs   = local_lambda_dofs & locally_owned_dofs;
    local_position_dofs = local_position_dofs & locally_owned_dofs;

    // Gather all lists to all processes
    std::vector<std::vector<types::global_dof_index>> gathered_dofs =
      Utilities::MPI::all_gather(mpi_communicator, local_lambda_dofs.get_index_vector());

    std::vector<types::global_dof_index> gathered_dofs_flattened;
    for (const auto &vec : gathered_dofs)
      gathered_dofs_flattened.insert(gathered_dofs_flattened.end(),
                                     vec.begin(),
                                     vec.end());

    std::sort(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());

    // Add the lambda DoFs to the list of locally relevant
    // DoFs: Do this only if partition contains a chunk of the cylinder
    if (n_local_lambda_dofs > 0)
    {
      locally_relevant_dofs.add_indices(gathered_dofs_flattened.begin(),
                                        gathered_dofs_flattened.end());
      locally_relevant_dofs.compress();
    }

    //
    // Compute the weights c_ij.
    // Done only once as cylinder is rigid and those weights will not change.
    //
    std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        fe_face_values_fixed.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int q = 0; q < face_quadrature.size(); ++q)
        {
          const double JxW = fe_face_values_fixed.JxW(q);

          for (unsigned int i_dof = 0; i_dof < fe.n_dofs_per_face(); ++i_dof)
          {
            const unsigned int comp =
              fe.face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which contribute to the
            // integral on this element
            if (!locally_relevant_dofs.is_element(face_dofs[i_dof]))
              continue;

            if (is_lambda(comp))
            {
              const types::global_dof_index lambda_dof = face_dofs[i_dof];

              // Very, very, very important:
              // Even though fe_face_values_fixed is a FEFaceValues, the dof index
              // given to shape_value is still a CELL dof index.
              const unsigned int i_cell_dof = fe.face_to_cell_index(i_dof, i_face);

              const unsigned int d     = comp - l_lower;
              const double       phi_i = fe_face_values_fixed.shape_value(i_cell_dof, q);
              coeffs[d][lambda_dof]   += - phi_i * JxW / param.spring_constant;
            }
          }
        }
      }
    }

    //
    // Gather the constraint weights
    //
    position_lambda_coeffs.resize(dim);
    std::vector<std::map<unsigned int, double>> gathered_coeffs_map(dim);

    for (unsigned int d = 0; d < dim; ++d)
    {
      std::vector<std::pair<unsigned int, double>> coeffs_vector(
        coeffs[d].begin(), coeffs[d].end());
      std::vector<std::vector<std::pair<unsigned int, double>>> gathered =
        Utilities::MPI::all_gather(MPI_COMM_WORLD, coeffs_vector);

      // Put back into map and sum contributions to same DoF from different
      // processes
      for (const auto &vec : gathered)
        for (const auto &pair : vec)
          gathered_coeffs_map[d][pair.first] += pair.second;

      position_lambda_coeffs[d].insert(position_lambda_coeffs[d].end(),
                                gathered_coeffs_map[d].begin(),
                                gathered_coeffs_map[d].end());

      //
      // Divide by spring constant k
      //
      // for (auto &vec : position_lambda_coeffs)
      //   for (auto &pair : vec)
      //     pair.second /= param.spring_constant;
    }

    // Get support points for position DoFs (the initial positions)
    // std::map<types::global_dof_index, Point<dim>> initial_positions;
    DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                         dof_handler,
                                         this->initial_positions,
                                         fe.component_mask(position));
  }

  template <int dim>
  void MMS<dim>::apply_position_lambda_constraints(const unsigned int boundary_id,
                                                   const bool homogeneous)
  {
    // Resize the position constraints with the updated locally_relevant_dofs
    position_constraints.clear();
    position_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    // std::map<types::global_dof_index, Point<dim>> current_positions;
    // DoFTools::map_dofs_to_support_points(*mapping,
    //                                      dof_handler,
    //                                      current_positions,
    //                                      fe.component_mask(position));

    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        fe_face_values_fixed.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
        {
          if (!locally_owned_dofs.is_element(face_dofs[i]))
            continue;

          const unsigned int comp =
            fe.face_system_to_component_index(i, i_face).first;

          if (is_position(comp))
          {
            const unsigned int d = comp - x_lower;
            position_constraints.add_line(face_dofs[i]);
            position_constraints.add_entries(face_dofs[i], position_lambda_coeffs[d]);

            if(!homogeneous)
            {
              // Add the initial position X_0 as inhomogeneity
              // if(this->current_time_step == 1)
                position_constraints.set_inhomogeneity(face_dofs[i], this->initial_positions.at(face_dofs[i])[d]);
              // else
              //   position_constraints.set_inhomogeneity(face_dofs[i], current_positions.at(face_dofs[i])[d]);
            }
          }
        }
      }
    }
    position_constraints.make_consistent_in_parallel(locally_owned_dofs,
                                                     locally_relevant_dofs,
                                                     mpi_communicator);
    position_constraints.close();
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

    if(param.with_position_coupling)
    {
      // Create and merge the coupling between lambda and position on cylinder
      const bool homogeneous = true;
      this->apply_position_lambda_constraints(weak_bc_boundary_id, homogeneous);
      zero_constraints.merge(
        position_constraints,
        AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
        true);
    }

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

    if(param.with_position_coupling)
    {
      // Apply the coupling between lambda and position on cylinder
      const bool homogeneous = false;
      this->apply_position_lambda_constraints(weak_bc_boundary_id, homogeneous);
      nonzero_constraints.merge(
        position_constraints,
        AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
        true);
    }

    nonzero_constraints.close();
  }

  // template <int dim>
  // void MMS<dim>::apply_nonzero_constraints()
  // {
  //   nonzero_constraints.distribute(local_evaluation_point);
  //   nonzero_constraints.distribute(newton_update);
  //   evaluation_point = local_evaluation_point;
  //   present_solution = local_evaluation_point;
  // }

  template <int dim>
  void
  MMS<dim>::update_boundary_conditions()
  {
    local_evaluation_point = present_solution;
    this->create_nonzero_constraints();
    // Distribute constraints
    nonzero_constraints.distribute(local_evaluation_point);
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

#if defined(COMPARE_ANALYTIC_MATRIX_WITH_FD)
    double max_diff = 0.;
#endif

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

#if defined(COMPARE_ANALYTIC_MATRIX_WITH_FD)
      // std::cout << "Analytical non-elasticity matrix is " << std::endl;
      // local_matrix.print(std::cout, 12, 3);

      // Compare with FD matrix
      FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
      FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     ref_local_rhs(dofs_per_cell),
        perturbed_local_rhs(dofs_per_cell);
      this->assemble_local_matrix_fd(first_step,
                                     cell,
                                     scratchData,
                                     evaluation_point,
                                     previous_solutions,
                                     local_dof_indices,
                                     local_matrix_fd,
                                     ref_local_rhs,
                                     perturbed_local_rhs,
                                     cell_dof_values);

      // std::cout << "FD         non-elasticity matrix is " << std::endl;
      // local_matrix_fd.print(std::cout, 12, 3);

      diff_matrix.equ(1.0, local_matrix);
      diff_matrix.add(-1.0, local_matrix_fd);
      // std::cout << "Error matrix is " << std::endl;
      // diff_matrix.print(std::cout, 12, 3);
      // std::cout << "Max difference is " << diff_matrix.linfty_norm()
      //           << std::endl;
      max_diff = std::max(max_diff, diff_matrix.linfty_norm());
#endif
    }

#if defined(COMPARE_ANALYTIC_MATRIX_WITH_FD)
    const double global_max_diff =
      Utilities::MPI::max(max_diff, mpi_communicator);

    pcout << "Max difference over all elements is " << global_max_diff
          << std::endl;
    // throw std::runtime_error("Testing FD");
#endif

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
                                       source_term_fun,
                                       mesh_velocity_fun);

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

      // BDF: current dudt
      Tensor<1, dim> dudt = bdfCoeffs[0] * present_velocity_values;
      for (unsigned int i = 1; i < bdfCoeffs.size(); ++i)
        dudt += bdfCoeffs[i] * scratchData.previous_velocity_values[i - 1][q];

      const auto &source_term_velocity = scratchData.source_term_velocity[q];
      const auto &source_term_pressure = scratchData.source_term_pressure[q];
      const auto &grad_source_velocity = scratchData.grad_source_velocity[q];
      const auto &grad_source_pressure = scratchData.grad_source_pressure[q];

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
            // Time-dependent
            local_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];

            // Convection (OK)
            local_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                                present_velocity_gradients * phi_u[j]) *
                               phi_u[i];

            // Diffusion (OK)
            local_matrix_ij +=
              param.viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);

            // ALE acceleration : - w dot grad(delta u)
            local_matrix_ij += grad_phi_u[j] * (-dxdt) * phi_u[i];
          }

          if (i_is_u && j_is_p)
          {
            // Pressure gradient (OK)
            local_matrix_ij += -div_phi_u[i] * phi_p[j];
          }

          if (i_is_u && j_is_x)
          {
            // Variation of time-dependent term with mesh position
            local_matrix_ij += dudt * phi_u[i] * trace(grad_phi_x[j]);

            // Variation of ALE term (dxdt cdot grad(u)) with mesh position
            local_matrix_ij += present_velocity_gradients *
                               (-bdfCoeffs[0] * phi_x[j]) * phi_u[i];
            local_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               (-dxdt) * phi_u[i];
            local_matrix_ij += present_velocity_gradients * (-dxdt) * phi_u[i] *
                               trace(grad_phi_x[j]);

            // Convection w.r.t. x (OK)
            local_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               present_velocity_values * phi_u[i];
            local_matrix_ij += present_velocity_gradients *
                               present_velocity_values * phi_u[i] *
                               trace(grad_phi_x[j]);

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

            // Source term for velocity (OK):
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

            // Source term for pressure:
            local_matrix_ij += phi_p[i] * grad_source_pressure * phi_x[j];
            local_matrix_ij +=
              source_term_pressure * phi_p[i] * trace(grad_phi_x[j]);
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
            const double JxW = scratchData.face_JxW[i_face][q];
            // const double W   = face_quadrature.weight(q);
            // const auto  &dXds = scratchData.face_dXds[i_face][q];

            const auto &phi_u = scratchData.phi_u_face[i_face][q];
            const auto &phi_x = scratchData.phi_x_face[i_face][q];
            // const auto &grad_phi_x = scratchData.grad_phi_x_face[i_face][q];
            const auto &phi_l = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            const auto &present_w =
              scratchData.present_face_mesh_velocity_values[i_face][q];
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];
            
            const auto &face_source_term_lambda = scratchData.face_source_term_lambda[i_face][q];
            // This term is harder to implement if it is split into 2 contributions...
            // Could do finite differences inside the scratch.reinit function...
            // const auto &face_grad_source_lambda = scratchData.face_grad_source_lambda[i_face][q];

#if !defined(NO_SLIP_ON_CYLINDER)
            const auto &prescribed_velocity_weak_bc =
              scratchData.prescribed_velocity_weak_bc[i_face][q];
            const auto &grad_solution_velocity =
              scratchData.grad_solution_velocity[i_face][q];
#endif

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

                const double delta_dx_j = scratchData.delta_dx[i_face][q][j];

                double local_matrix_ij = 0.;

                if (i_is_u && j_is_x)
                {
                  local_matrix_ij += - (present_l * phi_u[i]) * delta_dx_j;
                }

                if (i_is_u && j_is_l)
                {
                  local_matrix_ij += - (phi_l[j] * phi_u[i]);
                }

                if (i_is_l && j_is_u)
                {
                  local_matrix_ij += phi_u[j] * phi_l[i];
                }

                if (i_is_l && j_is_x)
                {
#if defined(NO_SLIP_ON_CYLINDER)
                  local_matrix_ij += -bdfCoeffs[0] * phi_x[j] * phi_l[i];
                  local_matrix_ij +=
                    (present_u - present_w) * phi_l[i] * delta_dx_j;
                  local_matrix_ij += face_source_term_lambda * phi_l[i] * delta_dx_j;
                  // local_matrix_ij += face_grad_source_lambda * phi_x[j] * phi_l[i];
#else
                  local_matrix_ij +=
                    -grad_solution_velocity * phi_x[j] * phi_l[i];
                  local_matrix_ij += (present_u - prescribed_velocity_weak_bc) *
                                     phi_l[i] * delta_dx_j;
#endif
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
                                     source_term_fun,
                                     mesh_velocity_fun);

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
                                         source_term_fun,
                                         mesh_velocity_fun);
    }
    else
      scratchData.reinit_current_mapping(cell,
                                         cell_dof_values,
                                         previous_solutions,
                                         solution_fun,
                                         source_term_fun,
                                         mesh_velocity_fun);

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
      const auto  &source_term_pressure = scratchData.source_term_pressure[q];
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
            // Convection (OK)
            (present_velocity_gradients * present_velocity_values) * phi_u[i]

            // Mesh movement
            - (present_velocity_gradients * present_mesh_velocity_values) *
                phi_u[i]

            // Diffusion (OK)
            + param.viscosity *
                scalar_product(present_velocity_gradients, grad_phi_u[i])

            // Pressure gradient (OK)
            - present_pressure_values * div_phi_u[i]

            // Momentum source term (OK)
            + source_term_velocity * phi_u[i]

            // Continuity (OK)
            - present_velocity_divergence * phi_p[i]

            // Pressure source term
            + source_term_pressure * phi_p[i]) *
          JxW;

        // Transient terms:
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
        {
          local_rhs_i -= bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i] * JxW;
        }

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
            const double face_JxW   = scratchData.face_JxW[i_face][q];
            const auto  &phi_u = scratchData.phi_u_face[i_face][q];
            const auto  &phi_l = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            const auto &present_w =
              scratchData.present_face_mesh_velocity_values[i_face][q];
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];

            const auto &face_source_term_lambda = scratchData.face_source_term_lambda[i_face][q];

#if !defined(NO_SLIP_ON_CYLINDER)
            const auto &prescribed_velocity_weak_bc =
              scratchData.prescribed_velocity_weak_bc[i_face][q];
#endif

            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              const unsigned int component_i = scratchData.components[i];
              const bool         i_is_u      = is_velocity(component_i);
              const bool         i_is_l      = is_lambda(component_i);

              if (i_is_u)
              {
                local_rhs(i) -= - (phi_u[i] * present_l) * face_JxW;
              }

              if (i_is_l)
              {
#if defined(NO_SLIP_ON_CYLINDER)
                local_rhs(i) -= (present_u - present_w) * phi_l[i] * face_JxW;
                local_rhs(i) -= face_source_term_lambda * phi_l[i] * face_JxW;
#else
                // Manufactured velocity on cylinder
                local_rhs(i) -=
                  (present_u - prescribed_velocity_weak_bc) * phi_l[i] * face_JxW;
#endif
              }

              /////////////////////////////////////////////////////
              // if (i_is_l)
              // {
              //   // Test with boundary element length/area:
              //   // local_rhs(i) -=
              //   sqrt(determinant(scratchData.face_G[i_face][q])) * W;
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
                                       source_term_fun,
                                       mesh_velocity_fun);
    }
    else
      scratchData.reinit_fixed_mapping(cell,
                                       cell_dof_values,
                                       previous_solutions,
                                       solution_fun,
                                       source_term_fun,
                                       mesh_velocity_fun);

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
    const double       tol        = param.newton_tolerance;
    bool               converged  = false;

    // Export before iterations
    this->output_results(this->current_convergence_cycle, this->current_time_step, true, iter);

    this->create_nonzero_constraints();

    while (current_res > tol && iter <= max_iter)
    {
      ////////////////////////////////////////
      // this->create_zero_constraints();
      // this->create_nonzero_constraints();
      // this->apply_nonzero_constraints();
      ////////////////////////////////////////

      evaluation_point = present_solution;

      this->assemble_rhs(first_step);

      // If residual norm is low enough, return
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
      // this->compare_lambda_position_on_boundary(weak_bc_boundary_id);
      //////////////////////////////////////////////

      //////////////////////////////////////////////
      // After the first Newton iteration, the mesh has moved
      // and the Dirichlet BC no longer match.
      // The moving mapping has been updated through evaluation_point,
      // now recreate
      // this->create_zero_constraints();
      // this->create_nonzero_constraints();
      // this->apply_nonzero_constraints();
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

      // Export after this iteration
      this->output_results(this->current_convergence_cycle, this->current_time_step, true, iter);
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
  void
  MMS<dim>::solve_newton2(const bool is_initial_step)
  {
    double global_res;
    double current_res;
    double last_res;
    bool   first_step     = is_initial_step;
    unsigned int outer_iteration = 0;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;

    // current_res and global_res are different as one is defined based on the l2
    // norm of the residual vector (current_res) and the other (global_res) is
    // defined by the physical solver and may differ from the l2_norm of the
    // residual vector. Only the global_res is compared to the tolerance in order
    // to evaluate if the nonlinear system is solved. Only current_res is used for
    // the alpha scheme as this scheme only monitors the convergence of the
    // non-linear system of equation (the matrix problem).

    // auto &evaluation_point = solver->get_evaluation_point();
    // auto &present_solution = solver->get_present_solution();

    while ((global_res > this->param.newton_tolerance) &&
           outer_iteration < 50)
      {
        evaluation_point = present_solution;

        this->assemble_matrix(false);

        if (outer_iteration == 0)
          this->assemble_rhs(false);

        if (outer_iteration == 0)
          {
            current_res      = this->system_rhs.l2_norm();
            last_res         = current_res;
          }

        if (VERBOSE)
          {
            pcout << "Newton iteration: " << outer_iteration << "  - Residual:  " << current_res << std::endl;
          }

        this->solve_direct(first_step);
        double last_alpha_res = current_res;

        unsigned int alpha_iter = 0;
        for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
          {
            // auto &local_evaluation_point = solver->get_local_evaluation_point();
            // auto &newton_update          = solver->get_newton_update();
            local_evaluation_point       = present_solution;
            local_evaluation_point.add(alpha, newton_update);
            // solver->apply_constraints();
            nonzero_constraints.distribute(local_evaluation_point);
            evaluation_point = local_evaluation_point;
            this->assemble_rhs(false);

            // auto &system_rhs = solver->get_system_rhs();
            current_res      = system_rhs.l2_norm();

            if (VERBOSE)
              {
                pcout << "\talpha = " << std::setw(6) << alpha
                              << std::setw(0) << " res = "
                              << std::setprecision(6)
                              << std::setw(6) << current_res << std::endl;

                // solver->output_newton_update_norms(
                //   this->params.display_precision);
              }

            // If it's not the first iteration of alpha check if the residual is
            // smaller than the last alpha iteration. If it's not smaller, we fall
            // back to the last alpha iteration.
            if (current_res > last_alpha_res and alpha_iter != 0)
              {
                alpha                  = 2 * alpha;
                local_evaluation_point = present_solution;
                local_evaluation_point.add(alpha, newton_update);
                // solver->apply_constraints();
                nonzero_constraints.distribute(local_evaluation_point);
                evaluation_point = local_evaluation_point;

                if (VERBOSE)
                  {
                    pcout
                      << "\t\talpha value was kept at alpha = " << alpha
                      << " since alpha = " << alpha / 2
                      << " increased the residual" << std::endl;
                  }
                current_res = last_alpha_res;
                break;
              }
            if (current_res < 0.1 * last_res ||
                last_res < param.newton_tolerance)
              {
                break;
              }
            last_alpha_res = current_res;
            alpha_iter++;
          }

        // global_res       = solver->get_current_residual();
        global_res       = current_res;
        present_solution = evaluation_point;
        last_res         = current_res;
        ++outer_iteration;
      }

    // If the non-linear solver has not converged abort simulation if
    // abort_at_convergence_failure=true
    if ((global_res > param.newton_tolerance) &&
        outer_iteration >= 50)
      {
        pcout << "\tCurrent residual = "
                              << std::setprecision(6)
                              << std::setw(6) << current_res << std::endl;
        throw(std::runtime_error(
          "Stopping simulation because the non-linear solver has failed to converge"));
      }
  }

  std::string double_to_pstring(double value, int precision = 15)
  {
    std::ostringstream oss;
    oss.imbue(std::locale::classic());       // ensure '.' decimal separator
    oss << std::fixed << std::setprecision(precision) << value;
    std::string s = oss.str();

    // Remove trailing zeros after decimal point
    auto pos = s.find('.');
    if (pos != std::string::npos)
    {
      // trim trailing zeros
      while (!s.empty() && s.back() == '0')
        s.pop_back();
      // if a trailing dot remains, remove it (so 2.000 -> "2")
      if (!s.empty() && s.back() == '.')
        s.pop_back();
    }

    // Replace dot with 'p' (if any)
    for (char &c : s)
      if (c == '.')
        c = 'p';

    return s;
  }

  template <int dim>
  void MMS<dim>::output_results(const unsigned int convergence_index,
                                const unsigned int time_step,
                                const bool         write_newton_iteration,
                                const unsigned int newton_step)
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

    //////////////////////////////////////////
    // Compute mesh velocity in post-processing
    // This is not ideal, this is done by modifying the displacement and
    // reexporting.
    LA::MPI::Vector mesh_velocity;
    mesh_velocity.reinit(locally_owned_dofs, mpi_communicator);
    const FEValuesExtractors::Vector position(x_lower);
    IndexSet                         disp_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(position));

    for (const auto &i : disp_dofs)
    {
      if (!locally_owned_dofs.is_element(i))
        continue;

      double value = bdfCoeffs[0] * present_solution[i];
      for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        value += bdfCoeffs[iBDF] * previous_solutions[iBDF - 1][i];
      mesh_velocity[i] = value;
    }
    mesh_velocity.compress(VectorOperation::insert);
    std::vector<std::string> mesh_velocity_name(dim, "ph_velocity");
    mesh_velocity_name.emplace_back("ph_pressure");
    for (unsigned int i = 0; i < dim; ++i)
      mesh_velocity_name.push_back("mesh_velocity");
    for (unsigned int i = 0; i < dim; ++i)
      mesh_velocity_name.push_back("ph_lambda");

    data_out.add_data_vector(mesh_velocity,
                             mesh_velocity_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Now plot the source term for the position:
    std::vector<std::string> source_term_names(dim, "ph2_velocity");
    source_term_names.emplace_back("ph2_pressure");
    for (unsigned int i = 0; i < dim; ++i)
      source_term_names.push_back("source_term_position");
    for (unsigned int i = 0; i < dim; ++i)
      source_term_names.push_back("ph2_lambda");

    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             source_term_fun,
                             mesh_velocity,
                             fe.component_mask(position));

    data_out.add_data_vector(mesh_velocity,
                             source_term_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    //////////////////////////////////////////

    // Plot exact solution.
    // The exact mesh position is rarely useful,
    // instead display the exact mesh velocity in the same solution array.
    std::vector<std::string> exact_solution_names(dim, "exact_velocity");
    exact_solution_names.push_back("exact_pressure");
    for (unsigned int d = 0; d < dim; ++d)
      // exact_solution_names.push_back("exact_mesh_position");
      exact_solution_names.push_back("exact_mesh_velocity");
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_lambda");

    // Start by evaluating fields on moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             solution_fun,
                             exact_solution);
    // Evaluate position on fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             mesh_velocity_fun,
                             exact_solution,
                             fe.component_mask(position));

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

    // Export a Newton iteration in dedicated folder
    if(write_newton_iteration)
    {
      std::string root = "../data/ns_ale_mms" + std::to_string(convergence_index) + "_newton_iterations/";
      std::string fileName = "solution_time_step_" + std::to_string(time_step);
      data_out.write_vtu_with_pvtu_record(
        root, fileName, newton_step, mpi_communicator, 2);
    }
    else
    {
      // Export regular time step
      std::string root = "../data/ns_ale_mms_k"
        + double_to_pstring(param.spring_constant)
        + "_"
        + std::to_string(convergence_index) + "/";
      std::string fileName = "solution";
      data_out.write_vtu_with_pvtu_record(
        root, fileName, time_step, mpi_communicator, 2);
    }
  }

  template <int dim>
  void
  MMS<dim>::compute_lambda_error_on_boundary(const unsigned int boundary_id,
                                             double            &lambda_l2_error,
                                             double         &lambda_linf_error,
                                             Tensor<1, dim> &error_on_integral)
  {
    double lambda_l2_local   = 0;
    double lambda_linf_local = 0;

    Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local,
      exact_integral_local;
    lambda_integral_local = 0;
    exact_integral_local  = 0;

    const FEValuesExtractors::Vector lambda(l_lower);

    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

    const unsigned int          n_faces_q_points = face_quadrature.size();
    std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);
    Tensor<1, dim>              diff, exact;

    // std::ofstream out("normals.pos");
    // out << "View \"normals\" {\n";

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
            const Point<dim> &qpoint          = fe_face_values.quadrature_point(q);
            const auto        normal_to_mesh  = fe_face_values.normal_vector(q);
            const auto        normal_to_solid = -normal_to_mesh;

            // Careful: 
            // int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
            //                                                   =
            //                                             normal_to_mesh
            //                                                   =
            //                                            -normal_to_solid
            //
            // Got to take the consistent normal to compare int lambda_h with solution.
            //
            // Solution<dim> computes lambda_exact = - sigma cdot ns, where n is 
            // expected to be the normal to the SOLID.

            // out << "VP(" << qpoint[0] << "," << qpoint[1] << "," << 0. << "){"
            //   << normal[0] << "," << normal[1] << "," << 0. << "};\n";

            for (unsigned int d = 0; d < dim; ++d)
              exact[d] = solution_fun.value(qpoint, normal_to_solid, l_lower + d);

            diff = lambda_values[q] - exact;

            lambda_l2_local += diff * diff * fe_face_values.JxW(q);
            lambda_linf_local =
              std::max(lambda_linf_local, std::abs(diff.norm()));

            // Increment the integral of lambda
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
            exact_integral_local += exact * fe_face_values.JxW(q);
          }
        }
      }
    }

    // out << "};\n";
    // out.close();

    lambda_l2_error = Utilities::MPI::sum(lambda_l2_local, mpi_communicator);
    lambda_l2_error = std::sqrt(lambda_l2_error);

    lambda_linf_error =
      Utilities::MPI::max(lambda_linf_local, mpi_communicator);

    for (unsigned int d = 0; d < dim; ++d)
    {
      lambda_integral[d] =
        Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
      exact_integral[d] =
        Utilities::MPI::sum(exact_integral_local[d], mpi_communicator);
      error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
    }
  }

  template <int dim>
  void
  MMS<dim>::check_manufactured_solution_boundary(const unsigned int boundary_id)
  {
    Tensor<1, dim> lambdaMMS_integral, lambdaMMS_integral_local;
    Tensor<1, dim> lambda_integral, lambda_integral_local;
    Tensor<1, dim> pns_integral, pns_integral_local;
    lambdaMMS_integral_local = 0;
    lambda_integral_local = 0;
    pns_integral_local = 0;

    const FEValuesExtractors::Vector lambda(l_lower);

    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);
    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    const unsigned int n_faces_q_points = face_quadrature.size();
    Tensor<1, dim> lambda_MMS;
    std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

    //
    // First compute integral over cylinder of lambda_MMS
    //
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
            const Point<dim> &qpoint          =   fe_face_values.quadrature_point(q);
            const auto        normal_to_solid = - fe_face_values.normal_vector(q);

            const double p_MMS = solution_fun.value(qpoint, p_lower);
            for (unsigned int d = 0; d < dim; ++d)
              lambda_MMS[d] = solution_fun.value(qpoint, normal_to_solid, l_lower + d);

            // Increment the integrals of lambda:

            // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
            lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

            // This is int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
            //                                                    -normal_to_solid
            lambda_integral_local    += lambda_values[q] * fe_face_values.JxW(q);

            // Increment integral of p * n_solid
            pns_integral_local += p_MMS * normal_to_solid * fe_face_values.JxW(q);
          }
        }
      }
    }

    for (unsigned int d = 0; d < dim; ++d)
    {
      lambdaMMS_integral[d] =
        Utilities::MPI::sum(lambdaMMS_integral_local[d], mpi_communicator);
      lambda_integral[d] =
        Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
    }
    pns_integral = Utilities::MPI::sum(pns_integral_local, mpi_communicator);

    // Reference solution for int_Gamma p*n_solid dx is - k * d * f(t).
    const Tensor<1, dim> ref_pns = - param.spring_constant * param.translation * mesh_mms.time_function.value(this->current_time);
    const double err_pns = (ref_pns - pns_integral).norm();

    //
    // Check x_MMS
    //
    Tensor<1, dim> x_MMS;
    double max_x_error = 0.;
    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values_fixed.reinit(cell, i_face);

          // Evaluate exact solution at quadrature points
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const Point<dim> &qpoint_fixed = fe_face_values_fixed.quadrature_point(q);

            for (unsigned int d = 0; d < dim; ++d)
              x_MMS[d] = solution_fun.value(qpoint_fixed, x_lower + d);

            const Tensor<1, dim> ref = -1./param.spring_constant * lambdaMMS_integral;
            const double err = ((x_MMS - qpoint_fixed) - ref).norm();
            // std::cout << "x_MMS - X0 at quad node is " << x_MMS  - qpoint_fixed << " - diff = " << err << std::endl;
            max_x_error = std::max(max_x_error, err);
          }
        }
      }
    }

    //
    // Check u_MMS
    //
    Tensor<1, dim> u_MMS, w_MMS;
    double max_u_error = 0.;
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
          fe_face_values_fixed.reinit(cell, i_face);

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
            const Point<dim> &qpoint_fixed  = fe_face_values_fixed.quadrature_point(q);

            for (unsigned int d = 0; d < dim; ++d)
            {
              u_MMS[d] = solution_fun.value(qpoint, u_lower + d);
              w_MMS[d] = mesh_velocity_fun.value(qpoint_fixed, x_lower + d);
            }

            const double err = (u_MMS - w_MMS).norm();
            // std::cout << "u_MMS & w_MMS at quad node are " << u_MMS << " , " << w_MMS << " - norm diff = " << err << std::endl;
            max_u_error = std::max(max_u_error, err);
          }
        }
      }
    }

    if(VERBOSE)
    {
      pcout << std::endl;
      pcout << "Checking manufactured solution for k = " << param.spring_constant << " :" << std::endl;
      pcout << "integral lambda         = " << lambda_integral << std::endl;
      pcout << "integral lambdaMMS      = " << lambdaMMS_integral << std::endl;
      pcout << "integral p * n_solid    = " << pns_integral << std::endl; 
      pcout << "reference: -k*d*f(t)    = " << ref_pns << " - err = " << err_pns << std::endl;
      pcout << "max error on (x_MMS -    X0) vs -1/k * integral lambda = " << max_x_error << std::endl;
      pcout << "max error on  u_MMS          vs w_MMS                  = " << max_u_error << std::endl;
      pcout << std::endl;
    }
  }

  /**
   * Compute integral of lambda (fluid force), compare to position dofs
   */
  template <int dim>
  void
  MMS<dim>::compare_lambda_position_on_boundary(const unsigned int boundary_id)
  {
    Tensor<1, dim> lambda_integral, lambda_integral_local;
    lambda_integral_local = 0;

    const FEValuesExtractors::Vector lambda(l_lower);

    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

    const unsigned int          n_faces_q_points = face_quadrature.size();
    std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

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

            // Increment the integral of lambda
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
          }
        }
      }
    }

    for (unsigned int d = 0; d < dim; ++d)
    {
      lambda_integral[d] =
        Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);
    }


    // Get the initial positions:
    // const FEValuesExtractors::Vector position(x_lower);
    // std::map<types::global_dof_index, Point<dim>> initial_positions;
    // DoFTools::map_dofs_to_support_points(*fixed_mapping,
    //                                      dof_handler,
    //                                      initial_positions,
    //                                      fe.component_mask(position));

    //
    // Position BC
    //
    Tensor <1, dim> cylinder_displacement_local, max_diff_local;
    bool first_displacement_x = true;
    bool first_displacement_y = true;
    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        face->get_dof_indices(face_dofs);

        for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
        {
          if (!locally_owned_dofs.is_element(face_dofs[i]))
            continue;

          const unsigned int comp =
            fe.face_system_to_component_index(i, i_face).first;

          //
          // Displacement or position coupling
          //
          if (is_position(comp))
          {
            const unsigned int d = comp - x_lower;
            if(d == 0 && first_displacement_x)
            {first_displacement_x = false;
              cylinder_displacement_local[d] = present_solution[face_dofs[i]] - this->initial_positions.at(face_dofs[i])[d];
            }
            if(d == 1 && first_displacement_y)
            {
              first_displacement_y = false;
              cylinder_displacement_local[d] = present_solution[face_dofs[i]] - this->initial_positions.at(face_dofs[i])[d];
            }
            if(!first_displacement_x && !first_displacement_y)
            {
              // Compare with cylinder_displacement_local
              const double displ = present_solution[face_dofs[i]] - this->initial_positions.at(face_dofs[i])[d];
              max_diff_local[d] = std::max(max_diff_local[d], cylinder_displacement_local[d] - displ);
            }
            // pcout << "Value of pos dof (d = " << d << ")   is " << present_solution[face_dofs[i]]
            //       << " minus initial " << present_solution[face_dofs[i]] - this->initial_positions.at(face_dofs[i])[d] << std::endl;
          }
        }
      }
    }

    // To take the max displacement while preserving sign
    struct MaxAbsOp
    {
      static void apply(void *invec, void *inoutvec, int *len, MPI_Datatype *dtype)
      {
        double *in    = static_cast<double*>(invec);
        double *inout = static_cast<double*>(inoutvec);
        for (int i = 0; i < *len; ++i)
        {
          if (std::fabs(in[i]) > std::fabs(inout[i]))
            inout[i] = in[i];
        }
      }
    };
    MPI_Op mpi_maxabs;
    MPI_Op_create(&MaxAbsOp::apply, /*commutative=*/true, &mpi_maxabs);

    Tensor <1, dim> cylinder_displacement, max_diff, ratio;
    for (unsigned int d = 0; d < dim; ++d)
    {
      // cylinder_displacement[d] =
      //   Utilities::MPI::max(cylinder_displacement_local[d], mpi_communicator);

      // The cylinder displacement is trivially 0 on processes which do not own
      // a part of the boundary, and is nontrivial otherwise.
      // Taking the max to synchronize does not work because displacement
      // can be negative. Instead, we take the max while preserving the sign.
      MPI_Allreduce(&cylinder_displacement_local[d], &cylinder_displacement[d], 1, MPI_DOUBLE, mpi_maxabs, mpi_communicator);

      // Take the max between all max differences disp_i - disp_j
      // for x_i and x_j both on the cylinder.
      // Checks that all displacement are identical.
      max_diff[d] =
        Utilities::MPI::max(max_diff_local[d], mpi_communicator);

      // Check that the ratio of both terms in the position
      // boundary condition is -spring_constant
      if(std::abs(cylinder_displacement[d]) > 1e-10)
        ratio[d] = lambda_integral[d] / cylinder_displacement[d];
    }

    if(VERBOSE)
    {
      pcout << std::endl;
      pcout << std::scientific << std::setprecision(8) << std::showpos;
      pcout << "Checking consistency between lambda integral and position BC:" << std::endl;
      pcout << "Integral of lambda on cylinder is " << lambda_integral << std::endl;
      pcout << "Prescribed displacement        is " << cylinder_displacement << std::endl;
      pcout << "                         Ratio is " << ratio << " (expected: " << -param.spring_constant << ")" << std::endl;
      pcout << "Max diff between displacements is " << max_diff << std::endl;
    }
      AssertThrow(max_diff.norm() <= 1e-10,
        ExcMessage("Displacement values of the cylinder are not all the same."));
      for (unsigned int d = 0; d < dim; ++d)
      {
        if(std::abs(ratio[d]) < 1e-10)
          continue;

        const double absolute_error = std::abs(ratio[d] - (-param.spring_constant));

        if(absolute_error <= 1e-6)
          continue;
        
        const double relative_error = absolute_error / param.spring_constant;
        AssertThrow(relative_error <= 1e-2,
          ExcMessage("Ratio integral vs displacement values is not -k"));
      }
    if(VERBOSE)
    {
      pcout << std::endl;
    }
  }

  template <int dim>
  void MMS<dim>::check_velocity_boundary(const unsigned int boundary_id)
  {
    // Check difference between uh and dxhdt
    double l2_local = 0;
    double li_local = 0;

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);
    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    const unsigned int n_faces_q_points = face_quadrature.size();

    std::vector<std::vector<Tensor<1, dim>>> position_values(
      bdfCoeffs.size(), std::vector<Tensor<1, dim>>(n_faces_q_points));
    std::vector<Tensor<1, dim>> mesh_velocity_values(n_faces_q_points);
    std::vector<Tensor<1, dim>> fluid_velocity_values(n_faces_q_points);
    Tensor<1, dim>              diff;

    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values_fixed.reinit(cell, i_face);
          fe_face_values.reinit(cell, i_face);

          // Get current and previous FE solution values on the face
          fe_face_values[velocity].get_function_values(present_solution,
                                                       fluid_velocity_values);
          fe_face_values_fixed[position].get_function_values(present_solution,
                                                       position_values[0]);
          for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
            fe_face_values_fixed[position].get_function_values(
              previous_solutions[iBDF - 1], position_values[iBDF]);

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            // Compute FE mesh velocity at node
            mesh_velocity_values[q] = 0;
            for (unsigned int iBDF = 0; iBDF < bdfCoeffs.size(); ++iBDF)
              mesh_velocity_values[q] +=
                bdfCoeffs[iBDF] * position_values[iBDF][q];

            diff = mesh_velocity_values[q] - fluid_velocity_values[q];

            // std::cout << "wh = " << mesh_velocity_values[q] << " - uh = " << fluid_velocity_values[q] << " - diff = " << diff << std::endl;

            // u_h - w_h
            l2_local += diff * diff * fe_face_values_fixed.JxW(q);
            li_local = std::max(li_local, std::abs(diff.norm()));
          }
        }
      }
    }

    const double l2_error = std::sqrt(Utilities::MPI::sum(l2_local, mpi_communicator));
    const double li_error = Utilities::MPI::max(li_local, mpi_communicator);

    if(VERBOSE)
    {
      pcout << "||uh - wh||_L2 = " << l2_error << std::endl; 
      pcout << "||uh - wh||_Li = " << li_error << std::endl;
    } 
  }

  template <int dim>
  void MMS<dim>::compute_boundary_errors(
    const unsigned int boundary_id,
    double            &l2_error_dxdt,
    double            &linf_error_dxdt,
    double            &l2_error_fluid_velocity,
    double            &linf_error_fluid_velocity,
    double            &l2_x_error,
    double            &linf_x_error)
  {
    double l2_local_dxdt   = 0;
    double linf_local_dxdt = 0;
    double l2_local_fluid   = 0;
    double linf_local_fluid = 0;
    double l2_local_x = 0;
    double linf_local_x = 0;

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);
    FEFaceValues<dim> fe_face_values(*mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    const unsigned int n_faces_q_points = face_quadrature.size();
    std::vector<std::vector<Tensor<1, dim>>> position_values(
      bdfCoeffs.size(), std::vector<Tensor<1, dim>>(n_faces_q_points));
    std::vector<Tensor<1, dim>> mesh_velocity_values(n_faces_q_points);
    std::vector<Tensor<1, dim>> fluid_velocity_values(n_faces_q_points);
    Tensor<1, dim>              diff, diff_fluid, diff_x, w_exact, x_exact;

    for (auto cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);

        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values_fixed.reinit(cell, i_face);
          fe_face_values.reinit(cell, i_face);

          // Get current and previous FE solution values on the face
          fe_face_values[velocity].get_function_values(present_solution,
                                                       fluid_velocity_values);
          fe_face_values_fixed[position].get_function_values(present_solution,
                                                       position_values[0]);
          for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
            fe_face_values_fixed[position].get_function_values(
              previous_solutions[iBDF - 1], position_values[iBDF]);

          // Evaluate exact solution at quadrature points
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            // Compute FE mesh velocity at node
            mesh_velocity_values[q] = 0;
            for (unsigned int iBDF = 0; iBDF < bdfCoeffs.size(); ++iBDF)
              mesh_velocity_values[q] +=
                bdfCoeffs[iBDF] * position_values[iBDF][q];

            const Point<dim> &qpoint = fe_face_values_fixed.quadrature_point(q);

            for (unsigned int d = 0; d < dim; ++d)
            {
              x_exact[d] = solution_fun.value(qpoint, x_lower + d);
              w_exact[d] = mesh_velocity_fun.value(qpoint, x_lower + d);
            }

            diff = mesh_velocity_values[q] - w_exact;
            diff_fluid = fluid_velocity_values[q] - w_exact;
            diff_x = position_values[0][q] - x_exact;

            // w_exact - dx_h/dt
            l2_local_dxdt += diff * diff * fe_face_values_fixed.JxW(q);
            linf_local_dxdt = std::max(linf_local_dxdt, std::abs(diff.norm()));

            // w_exact - u_h
            l2_local_fluid += diff_fluid * diff_fluid * fe_face_values.JxW(q);
            linf_local_fluid = std::max(linf_local_fluid, std::abs(diff_fluid.norm()));

            // x_exact - x_h
            l2_local_x += diff_x * diff_x * fe_face_values.JxW(q);
            linf_local_x = std::max(linf_local_x, std::abs(diff_x.norm()));
          }
        }
      }
    }

    l2_error_dxdt   = Utilities::MPI::sum(l2_local_dxdt, mpi_communicator);
    l2_error_dxdt   = std::sqrt(l2_error_dxdt);
    linf_error_dxdt = Utilities::MPI::max(linf_local_dxdt, mpi_communicator);

    l2_error_fluid_velocity   = Utilities::MPI::sum(l2_local_fluid, mpi_communicator);
    l2_error_fluid_velocity   = std::sqrt(l2_error_fluid_velocity);
    linf_error_fluid_velocity = Utilities::MPI::max(linf_local_fluid, mpi_communicator);

    l2_x_error   = Utilities::MPI::sum(l2_local_x, mpi_communicator);
    l2_x_error   = std::sqrt(l2_x_error);
    linf_x_error = Utilities::MPI::max(linf_local_x, mpi_communicator);
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
    const unsigned int                  n_points_1D = (dim == 2) ? 6 : 5;
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

    double w_linf = 0., w_l2_boundary = 0., w_linf_boundary = 0.;
    double u_l2_boundary = 0., u_linf_boundary = 0.;
    double x_l2_boundary = 0., x_linf_boundary = 0.;
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

      // Mesh velocity on boundary
      this->compute_boundary_errors(mesh_velocity_error_boundary_id,
                                                    w_l2_boundary,
                                                    w_linf_boundary,
                                                    u_l2_boundary,
                                                    u_linf_boundary,
                                                    x_l2_boundary,
                                                    x_linf_boundary);
      linf_error_w_boundary = std::max(linf_error_w_boundary, w_linf_boundary);
      linf_error_u_boundary = std::max(linf_error_u_boundary, u_linf_boundary);
      linf_error_x_boundary = std::max(linf_error_x_boundary, x_linf_boundary);
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
    l2_err_w_boundary += param.dt * w_l2_boundary;
    l2_err_u_boundary += param.dt * u_l2_boundary;
    l2_err_x_boundary += param.dt * x_l2_boundary;
    l2_err_l += param.dt * l_l2_error;
    l1_time_error_Fx += param.dt * error_on_integral[0];
    l1_time_error_Fy += param.dt * error_on_integral[1];

    if (VERBOSE)
    {
      pcout << std::setprecision(3);

      // Error on fields in the domain
      pcout << "Current L2 errors in domain   : "
            << "||e_u||_L2 = " << u_l2_error << " - "
            << "||e_p||_L2 = " << p_l2_error << " - "
            << "||e_x||_L2 = " << x_l2_error << " - "
            << "||e_w||_L2 = " << w_l2_error << std::endl;
      pcout << "Cumul.  L2 errors in domain   : "
            << "||e_u||_L2 = " << l2_err_u << " - "
            << "||e_p||_L2 = " << l2_err_p << " - "
            << "||e_x||_L2 = " << l2_err_x << " - "
            << "||e_w||_L2 = " << l2_err_w << std::endl;
      pcout << "Current Li errors in domain   : "
            << "||e_u||_Li = " << u_linf << " - "
            << "||e_p||_Li = " << p_linf << " - "
            << "||e_x||_Li = " << x_linf << " - "
            << "||e_w||_Li = " << w_linf << std::endl;
      pcout << "Cumul.  Li errors in domain   : "
            << "||e_u||_Li = " << linf_error_u << " - "
            << "||e_p||_Li = " << linf_error_p << " - "
            << "||e_x||_Li = " << linf_error_x << " - "
            << "||e_w||_Li = " << linf_error_w << std::endl;

      // Error on fields on the inner boundary (cylinder)
      pcout << "Current L2 errors on boundary : "
            << "|w-wh|_L2 = " << w_l2_boundary << " - "
            << "|w-uh|_L2 = " << u_l2_boundary << " - "
            << "|x-xh|_L2 = " << x_l2_boundary << " - "
            << "|l-lh|_L2 = " << l_l2_error << std::endl;
      pcout << "Cumul.  L2 errors on boundary : "
            << "|w-wh|_L2 = " << l2_err_w_boundary << " - "
            << "|w-uh|_L2 = " << l2_err_u_boundary << " - "
            << "|x-xh|_L2 = " << l2_err_x_boundary << " - "
            << "|l-lh|_L2 = " << l2_err_l << std::endl;
      pcout << "Current Li errors on boundary : "
            << "|w-wh|_Li = " << w_linf_boundary << " - "
            << "|w-uh|_Li = " << u_linf_boundary << " - "
            << "|x-xh|_Li = " << x_linf_boundary << " - "
            << "|l-lh|_Li = " << l_linf << " - "
            << "     e_Fx = " << error_on_integral[0] << " - "
            << "     e_Fy = " << error_on_integral[1] << std::endl;
      pcout << "Cumul.  Li errors on boundary : "
            << "|w-wh|_Li = " << linf_error_w_boundary << " - "
            << "|w-uh|_Li = " << linf_error_u_boundary << " - "
            << "|x-xh|_Li = " << linf_error_x_boundary << " - "
            << "|l-lh|_Li = " << linf_error_l << " - "
            << "   eFx_L1 = " << l1_time_error_Fx << " - "
            << "   eFy_L1 = " << l1_time_error_Fy << std::endl;
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
    linf_error_u          = 0.;
    linf_error_p          = 0.;
    linf_error_x          = 0.;
    linf_error_w          = 0.;
    linf_error_w_boundary = 0.;
    linf_error_u_boundary = 0.;
    linf_error_x_boundary = 0.;
    linf_error_l          = 0.;
    linf_error_Fx         = 0.;
    linf_error_Fy         = 0.;
    l2_err_u              = 0.;
    l2_err_p              = 0.;
    l2_err_x              = 0.;
    l2_err_w              = 0.;
    l2_err_w_boundary     = 0.;
    l2_err_u_boundary     = 0.;
    l2_err_x_boundary     = 0.;
    l2_err_l              = 0.;
    l1_time_error_Fx      = 0.;
    l1_time_error_Fy      = 0.;

    // Constrained pressure DOF
    constrained_pressure_dof = numbers::invalid_dof_index;

    // Position - lambda constraints
    for(auto &vec : position_lambda_coeffs)
      vec.clear();
    position_lambda_coeffs.clear();
    initial_positions.clear();
  }

  template <int dim>
  void MMS<dim>::run()
  {
    unsigned int iMesh        = param.starting_mesh;
    unsigned int nConvergence = param.nConvergenceCycles;

    if (param.bdf_order == 0)
    {
      param.nTimeSteps = 1;
    }

    this->current_convergence_cycle = 1;

    for (unsigned int iConv = 1; iConv <= nConvergence; ++iConv, ++(this->current_convergence_cycle))
    {
      this->reset();

      this->param.prev_dt = this->param.dt;
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->current_time_step = 1;
      this->solution_fun.set_time(current_time);
      this->source_term_fun.set_time(current_time);
      this->mesh_velocity_fun.set_time(current_time);
      this->solution_at_future_position_fun.set_time(current_time);

      this->make_grid(iMesh);
      this->setup_system();
      this->create_lambda_zero_constraints(weak_bc_boundary_id);
      this->create_position_lambda_coupling_constraints(weak_bc_boundary_id);
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();

      this->set_initial_condition();
      this->output_results(iConv, 0);

      for (unsigned int i = 0; i < param.nTimeSteps; ++i, ++(this->current_time_step))
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

        ////////////////////////////////////////////////////////////
        this->update_boundary_conditions();
        ////////////////////////////////////////////////////////////
        // Start the Newton with the right boundary conditions
        // this->create_nonzero_constraints();
        // this->apply_nonzero_constraints();
        ////////////////////////////////////////////////////////////

        if (i == 0 && param.bdf_order == 2)
        {
          // For BDF2: set first step to exact solution
          this->set_exact_solution();
        }
        else if (i == 1 && param.bdf_order == 3)
        {
          // Also set exact solution at 2nd time step for BDF2
          this->set_exact_solution();
        }
        else
        {
          // this->solve_newton();
          // this->solve_newton2(i == 0);
          this->solve_newton2(false);
        }

        if(param.with_position_coupling && !(i == 0 && param.bdf_order == 2))
          this->compare_lambda_position_on_boundary(weak_bc_boundary_id);

        this->check_manufactured_solution_boundary(weak_bc_boundary_id);
        this->check_velocity_boundary(weak_bc_boundary_id);

        this->compute_errors(i);
        this->output_results(iConv, i + 1);

        // Rotate solutions
        if (param.bdf_order > 0)
        {
          for (unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
            previous_solutions[i] = previous_solutions[i - 1];
          previous_solutions[0] = present_solution;
        }
      }

      // Add Linf error and L2 error at last time step
      convergence_domain.add_value("nElm", triangulation.n_active_cells());
      convergence_domain.add_value("dt", param.dt);
      for(const auto &[key, val] : domain_errors)
        convergence_domain.add_value(key, *val);

      convergence_boundary.add_value("nElm", triangulation.n_active_cells());
      convergence_boundary.add_value("dt", param.dt);
      for(const auto &[key, val] : boundary_errors)
        convergence_boundary.add_value(key, *val);

      if (param.type_of_convergence_study == ConvergenceStudy::TIME ||
          param.type_of_convergence_study == ConvergenceStudy::TIME_AND_SPACE)
      {
        this->param.dt /= 2.;
        this->param.nTimeSteps *= 2.;
      }
      if (param.type_of_convergence_study == ConvergenceStudy::SPACE ||
          param.type_of_convergence_study == ConvergenceStudy::TIME_AND_SPACE)
      {
        ++iMesh;
      }
    }

    // Arrange convergence tables
    for(const auto &[key, val] : domain_errors)
    {
      convergence_domain.evaluate_convergence_rates(key, ConvergenceTable::reduction_rate_log2);
      convergence_domain.set_precision(key, 4);
      convergence_domain.set_scientific(key, true);
    }
    for(const auto &[key, val] : boundary_errors)
    {
      convergence_boundary.evaluate_convergence_rates(key, ConvergenceTable::reduction_rate_log2);
      convergence_boundary.set_precision(key, 4);
      convergence_boundary.set_scientific(key, true);
    }

    pcout << std::endl;
    pcout << "Spring constant = " << param.spring_constant << " - BDF order: " << param.bdf_order << " - Velocity P"
          << param.velocity_degree << " - Pressure P"
          << param.velocity_degree - 1 << " - Position P"
          << param.position_degree << std::endl;
    pcout << std::endl;
    if (mpi_rank == 0)
    {
      std::cout << "Error on domain:" << std::endl;
      convergence_domain.write_text(std::cout);
      std::cout << "Error on boundary:" << std::endl;
      convergence_boundary.write_text(std::cout);
    }
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

    const unsigned int dim = 2;

    std::vector<double> spring_constants = {100., 10., 1., 0.1, 0.01};

    SimulationParameters<dim> param;

    param.velocity_degree = 2;
    param.position_degree = 1;
    param.lambda_degree   = 2;

    param.with_weak_velocity_bc  = true;
    param.with_position_coupling = true; 

    if(param.with_position_coupling)
      param.position_boundary_names = {"OuterBoundary"};
    else
      param.position_boundary_names = {"OuterBoundary", "InnerBoundary"};

    if (param.with_weak_velocity_bc)
    {
      param.strong_velocity_boundary_names = {"OuterBoundary"};
      param.weak_velocity_boundary_names   = {"InnerBoundary"};
    }
    else
    {
      param.strong_velocity_boundary_names = {"OuterBoundary", "InnerBoundary"};
      param.weak_velocity_boundary_names   = {};
    }

    // Specify that mesh velocity error should be computed on this boundary
    param.mesh_velocity_error_boundary_names = {"InnerBoundary"};

    param.viscosity           = VISCOSITY;
    param.pseudo_solid_mu     = MU_PS;
    param.pseudo_solid_lambda = LAMBDA_PS;
    // param.spring_constant     = 1.;

    // Time integration
    param.bdf_order  = 2;
    param.t0         = 0.;
    param.dt         = 0.1;
    param.nTimeSteps = 10;
    param.t1         = param.dt * param.nTimeSteps;

    param.newton_tolerance = 1e-8 * LAMBDA_PS;

    // param.type_of_convergence_study = ConvergenceStudy::TIME;
    // param.type_of_convergence_study = ConvergenceStudy::SPACE;
    param.type_of_convergence_study = ConvergenceStudy::TIME_AND_SPACE;
    param.nConvergenceCycles  = 3;
    param.starting_mesh       = 1;

    VERBOSE = false;

    for(unsigned int i_spring = 0; i_spring < spring_constants.size(); ++i_spring)
    {
      param.spring_constant = spring_constants[i_spring];

      // // Constant mesh - Zero flow
      // const ConstantTimeDep mesh_time_function(1.);
      // const ConstantTimeDep flow_time_function(0.);
      // mesh_time_function.check_dependency(flow_time_function);

      // // Linear mesh - Constant flow
      // const PowerTimeDep    mesh_time_function(1. / 4., 1);
      // const ConstantTimeDep flow_time_function(1. / 4.);
      // mesh_time_function.check_dependency(flow_time_function);

      // // Cubic mesh - Quadratic flow
      // const PowerTimeDep mesh_time_function(1., 3);
      // const PowerTimeDep flow_time_function(3., 2);
      // mesh_time_function.check_dependency(flow_time_function);

      // // Quartic mesh - Cubic flow
      // const PowerTimeDep mesh_time_function(1., 4);
      // const PowerTimeDep flow_time_function(4., 3);
      // mesh_time_function.check_dependency(flow_time_function);

      // Sin mesh - Cos flow
      const SineTimeDep   mesh_time_function(1./2.);
      const CosineTimeDep flow_time_function(2. * M_PI * 1./2.);
      mesh_time_function.check_dependency(flow_time_function);

      // Mesh origin
      const Point<dim> X0(10., 10.);
      const Point<dim> L(1., 1.);
      const Point<dim> center_relative = 0.5 * L;
      const Point<dim> center = X0 + center_relative;
      const double R0 = 0.15;
      const double R1 = 0.45;
      param.translation[0] = 0.10;
      param.translation[1] = 0.05;

      const RigidMeshPosition<dim> mesh_position_mms(mesh_time_function,
                                                     center,
                                                     R0,
                                                     R1,
                                                     param.translation,
                                                     param.spring_constant);

      // const bool coupled_pressure = true;
      // const RigidFlow<dim> flow_mms(flow_time_function,
      //                               mesh_time_function,
      //                               center,
      //                               R0,
      //                               R1,
      //                               param.translation,
      //                               coupled_pressure,
      //                               param.spring_constant);

      //
      // Uniform velocity MMS u = w on cylinder
      //
      const ConstantFlowCoupledPressure<dim> flow_mms(flow_time_function,
                                                      mesh_time_function,
                                                      center,
                                                      R0,
                                                      R1,
                                                      param.translation,
                                                      param.spring_constant);

      // Tensor<1, dim> u_mms;
      // u_mms[0] = 1.;
      // u_mms[1] = 1.;
      // const ConstantFlow<dim> flow_mms(flow_time_function, u_mms);

      MMS<dim> problem(param, flow_mms, mesh_position_mms);
      problem.run();
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
