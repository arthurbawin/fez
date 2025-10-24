
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

// #define COMPARE_ANALYTIC_MATRIX_WITH_FD

namespace fsi_coupled
{
  using namespace dealii;

  template <int dim>
  class SimulationParameters
  {
  public:
    std::string output_dir;
    std::string mesh_file;

    unsigned int velocity_degree;
    unsigned int position_degree;
    unsigned int lambda_degree;

    // std::vector<std::string> position_boundary_names;
    std::vector<std::string> position_fixed_boundary_names;
    std::vector<std::string> position_moving_boundary_names;
    std::vector<std::string> strong_velocity_boundary_names;
    std::vector<std::string> weak_velocity_boundary_names;
    std::vector<std::string> noflux_velocity_boundary_names;

    // Boundaries on which we want to compute the error ||w - wh||
    std::vector<std::string> mesh_velocity_error_boundary_names;

    bool with_weak_velocity_bc;
    bool with_position_coupling;

    Tensor<1, dim> translation;

    double       kinematic_viscosity;
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
    bool analytic_jacobian_matrix = true;
    bool with_line_search = true;

    // Geometry and flow parameters
    double H;
    double L;
    double D;
    double U;
    double Re;
    double rho;

  public:
    SimulationParameters<dim>(){};
  };

  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;
    const unsigned int l_lower = 2 * dim + 1;
  public:
    InitialCondition(const unsigned int n_components)
      : Function<dim>(n_components)
    {}

    double initial_velocity(const Point<dim> &/*p*/, const unsigned int component) const
    {
      if(component == 0)
        return 1.;
      if(component == 1)
        return 0.;
      if(component == 2)
        return 0.;
      DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      values[p_lower] = 0.;
      for(unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = this->initial_velocity(p, d);
        values[x_lower + d] = 0.;
        values[l_lower + d] = 0.;
      }
    }
  };

  template <int dim>
  class Inlet : public Function<dim>
  {
  public:
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;
    const unsigned int l_lower = 2 * dim + 1;
  public:
    Inlet(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    double inlet_velocity(const Point<dim> &/*p*/,
                          const double       /*t*/,
                          const unsigned int component) const
    {
      if(component == 0)
        return 1.;
      if(component == 1)
        return 0.;
      if(component == 2)
        return 0.;
      DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t = this->get_time();

      values[p_lower] = 0.;
      for(unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = this->inlet_velocity(p, t, d);
        values[x_lower + d] = 0.;
        values[l_lower + d] = 0.;
      }
    }
  };

  template <int dim>
  class FixedMeshPosition : public Function<dim>
  {
  public:
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;
    const unsigned int l_lower = 2 * dim + 1;
  public:
    FixedMeshPosition(const unsigned int x_lower,
                      const unsigned int n_components)
      : Function<dim>(n_components)
      , x_lower(x_lower)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      for (unsigned int d = 0; d < dim; ++d)
        values[x_lower + d] = p[d];
    }
  };

  template <int dim>
  class MeshPositionCircle : public Function<dim>
  {
  public:
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;
    const unsigned int l_lower = 2 * dim + 1;

    const SimulationParameters<dim> &param;
  public:
    MeshPositionCircle(const double                     time,
                       const unsigned int               n_components,
                       const SimulationParameters<dim> &param)
      : Function<dim>(n_components, time)
      , param(param)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t = this->get_time();

      for(unsigned int d = 0; d < dim; ++d)
      {
        values[x_lower + d] = p[d];
        if(d == 1)
          values[x_lower + d] = p[d] + param.H / 50. * sin(0.1 * M_PI * t);
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
      , fe_values_fixed(fixed_mapping,
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
      , fe_face_values_fixed(fixed_mapping,
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

  public:
    template <typename VectorType>
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const VectorType                   &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions)
    {
      fe_values.reinit(cell);
      fe_values_fixed.reinit(cell);

      for (const unsigned int i : fe_values.dof_indices())
        components[i] =
          fe_values.get_fe().system_to_component_index(i).first;

      const FEValuesExtractors::Vector velocity(u_lower);
      const FEValuesExtractors::Scalar pressure(p_lower);
      const FEValuesExtractors::Vector position(x_lower);
      const FEValuesExtractors::Vector lambda(l_lower);

      //
      // Volume-related quantities
      //
      if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
      {
        //
        // Evaluate velocity and pressure on moving mapping
        //
        fe_values[velocity].get_function_values(
          current_solution, present_velocity_values);
        fe_values[velocity].get_function_gradients(
          current_solution, present_velocity_gradients);
        fe_values[pressure].get_function_values(
          current_solution, present_pressure_values);

        //
        // Evaluate position on fixed mapping
        //
        fe_values_fixed[position].get_function_values(
          current_solution, present_position_values);
        fe_values_fixed[position].get_function_gradients(
          current_solution, present_position_gradients);
      }
      else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
      {
        //
        // Evaluate velocity and pressure on moving mapping
        //
        fe_values[velocity].get_function_values_from_local_dof_values(
          current_solution, present_velocity_values);
        fe_values[velocity]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_velocity_gradients);
        fe_values[pressure].get_function_values_from_local_dof_values(
          current_solution, present_pressure_values);

        //
        // Evaluate position on fixed mapping
        //
        fe_values_fixed[position].get_function_values_from_local_dof_values(
          current_solution, present_position_values);
        fe_values_fixed[position]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_position_gradients);
      }
      else
      {
        static_assert(false,
                      "reinit expects LA::MPI::Vector or std::vector<double>");
      }

      // Previous solutions
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_values[velocity].get_function_values(
          previous_solutions[i], previous_velocity_values[i]);
        fe_values_fixed[position].get_function_values(
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
        JxW_moving[q] = fe_values.JxW(q);
        JxW_fixed[q]  = fe_values_fixed.JxW(q);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]      = fe_values[velocity].value(k, q);
          grad_phi_u[q][k] = fe_values[velocity].gradient(k, q);
          div_phi_u[q][k]  = fe_values[velocity].divergence(k, q);
          phi_p[q][k]      = fe_values[pressure].value(k, q);

          phi_x[q][k]      = fe_values_fixed[position].value(k, q);
          grad_phi_x[q][k] = fe_values_fixed[position].gradient(k, q);
          div_phi_x[q][k]  = fe_values_fixed[position].divergence(k, q);
        }
      }

      //
      // Face-related values and shape functions,
      // for the faces touching the prescribed boundary_id
      //
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        // if (!(face->at_boundary() && face->boundary_id() == boundary_id))
        //   continue;

        fe_face_values.reinit(cell, face);
        fe_face_values_fixed.reinit(cell, face);

        if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
        {
          fe_face_values[velocity].get_function_values(
            current_solution, present_face_velocity_values[i_face]);
          fe_face_values[lambda].get_function_values(
            current_solution, present_face_lambda_values[i_face]);

          fe_face_values_fixed[position].get_function_values(
            current_solution, present_face_position_values[i_face]);
          fe_face_values_fixed[position].get_function_gradients(
            current_solution, present_face_position_gradient[i_face]);

        }
        else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
        {
          fe_face_values[velocity].get_function_values_from_local_dof_values(
              current_solution, present_face_velocity_values[i_face]);
          fe_face_values[lambda].get_function_values_from_local_dof_values(
              current_solution, present_face_lambda_values[i_face]);

          fe_face_values_fixed[position].get_function_values_from_local_dof_values(
              current_solution, present_face_position_values[i_face]);
          fe_face_values_fixed[position].get_function_gradients_from_local_dof_values(
              current_solution, present_face_position_gradient[i_face]);
        }
        else
        {
          static_assert(
            false, "reinit expects LA::MPI::Vector or std::vector<double>");
        }

        for (unsigned int i = 0; i < previous_solutions.size(); ++i)
        {
          fe_face_values_fixed[position].get_function_values(
            previous_solutions[i], previous_face_position_values[i_face][i]);
        }

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          face_JxW_moving[i_face][q] = fe_face_values.JxW(q);
          face_JxW_fixed[i_face][q]  = fe_face_values_fixed.JxW(q);

          //
          // Jacobian of geometric transformation is needed to compute
          // tangent metric for the lambda equation, on moving mapping
          //
          const Tensor<2, dim> J    = fe_face_values.jacobian(q);

          if constexpr (dim == 2)
          {
            switch (i_face)
            {
              case 0:
                dxsids_array[0][0] = 1.;
                dxsids_array[0][1] = 0.;
                break;
              case 1:
                dxsids_array[0][0] = -1.;
                dxsids_array[0][1] = 1.;
                break;
              case 2:
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
          const Tensor<2, dim - 1> G_inverse = invert(G);

          // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
          Tensor<2, dim - 1> res;

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            phi_u_face[i_face][q][k] = fe_face_values[velocity].value(k, q);
            phi_l_face[i_face][q][k] = fe_face_values[lambda].value(k, q);

            phi_x_face[i_face][q][k]      = fe_face_values_fixed[position].value(k, q);
            grad_phi_x_face[i_face][q][k] = fe_face_values_fixed[position].gradient(k, q);

            const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

            Tensor<2, dim> A =
              transpose(J) *
              (transpose(present_face_position_gradient[i_face][q]) *
                 grad_phi_x +
               transpose(grad_phi_x) *
                 present_face_position_gradient[i_face][q]) *
              J;

            res = 0;
            for (unsigned int di = 0; di < dim - 1; ++di)
              for (unsigned int dj = 0; dj < dim - 1; ++dj)
                for (unsigned int im = 0; im < dim - 1; ++im)
                  for (unsigned int in = 0; in < dim; ++in)
                    for (unsigned int io = 0; io < dim; ++io)
                      res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] *
                                     A[in][io] * dxsids_array[dj][io];
            delta_dx[i_face][q][k] =
              0.5 *
              trace(res); // Choose this if multiplying by JxW in the matrix
            // delta_dx[i_face][q][k] = 0.5 * sqrt_det_G * trace(res); // Choose
            // this if multiplying by W
          }

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
      }
    }

  public:
    FEValues<dim>  fe_values;
    FEValues<dim>  fe_values_fixed;

    FEFaceValues<dim>  fe_face_values;
    FEFaceValues<dim>  fe_face_values_fixed;

    const unsigned int n_q_points;
    const unsigned int n_faces;
    const unsigned int n_faces_q_points;
    const unsigned int dofs_per_cell;

    // The tag of the boundary on which weak Dirichlet BC are
    // applied with Lagrange multiplier. Only 1 for now.
    const unsigned int boundary_id;

    const std::vector<double> &bdfCoeffs;

    std::vector<double>              JxW_moving;
    std::vector<double>              JxW_fixed;
    std::vector<std::vector<double>> face_JxW_moving;
    std::vector<std::vector<double>> face_JxW_fixed;

    // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
    std::array<Tensor<1, dim>, dim - 1>          dxsids_array;
    std::vector<std::vector<Tensor<2, dim - 1>>> face_G;

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

    // // Gradient of source term,
    // // at each quad node, for each dof component, result is a Tensor<1, dim>
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
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    JxW_moving.resize(n_q_points);
    JxW_fixed.resize(n_q_points);

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

    face_JxW_moving.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_JxW_fixed.resize(n_faces, std::vector<double>(n_faces_q_points));

    face_G.resize(n_faces, std::vector<Tensor<2, dim - 1>>(n_faces_q_points));
    delta_dx.resize(n_faces,
                    std::vector<std::vector<double>>(
                      n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }

  template <int dim>
  class FSI
  {
  public:
    FSI(const SimulationParameters<dim> &param);
    void run();

  private:
    void set_bdf_coefficients(const unsigned int order);
    void make_grid();
    void setup_system();
    void create_zero_constraints();
    void create_nonzero_constraints();
    void create_lambda_zero_constraints(const unsigned int boundary_id);

    void create_position_lambda_coupling_data(
      const unsigned int boundary_id);

    void apply_position_lambda_constraints(const unsigned int boundary_id,
                                           const bool homogeneous);

    void add_no_flux_constraints(AffineConstraints<double> &constraints);
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
      bool use_full_solution);

    void add_algebraic_position_coupling_to_matrix();
    void add_algebraic_position_coupling_to_rhs();

    void solve_direct(bool first_step);
    void solve_newton(const bool is_initial_step);
    void output_results(const unsigned int time_step,
                        const bool         write_newton_iteration = false,
                        const unsigned int newton_step = 0);
    void compare_lambda_position_on_boundary(const unsigned int boundary_id);
    void compute_force_coefficients(const unsigned int boundary_id,
                                    const bool         export_force_table);
    void write_cylinder_position(const unsigned int boundary_id,
                                 const bool         export_position_table);
    void check_velocity_boundary(const unsigned int boundary_id);
    void create_mask_arrays();

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
    // std::set<types::global_dof_index> coupled_position_dofs;
    std::map<types::global_dof_index, unsigned int> coupled_position_dofs;

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

    InitialCondition<dim>   initial_condition_fun;
    Inlet<dim>              inlet_fun;
    FixedMeshPosition<dim>  fixed_mesh_position_fun;
    MeshPositionCircle<dim> mesh_position_circle_fun;

    TableHandler forces_table;
    TableHandler cylinder_position_table;

    std::vector<ComponentMask> masks;
    std::vector<unsigned int> component_of_dof;
  };

  template <int dim>
  FSI<dim>::FSI(const SimulationParameters<dim>         &param)
    : param(param)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , fe(FE_SimplexP<dim>(param.velocity_degree), // Velocity
         dim,
         FE_SimplexP<dim>(param.velocity_degree - 1), // Pressure
         1,
         FE_SimplexP<dim>(param.position_degree), // Position
         dim,
         FE_SimplexP<dim>(param.lambda_degree), // Lagrange multiplier
         dim)
    , quadrature(QGaussSimplex<dim>(4))
    , face_quadrature(QGaussSimplex<dim - 1>(4))
    , triangulation(mpi_communicator)
    , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
    , dof_handler(triangulation)
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , current_time(param.t0)
    , initial_condition_fun(InitialCondition<dim>(n_components))
    , inlet_fun(Inlet<dim>(current_time, n_components))
    , fixed_mesh_position_fun(FixedMeshPosition<dim>(x_lower, n_components))
    , mesh_position_circle_fun(
        MeshPositionCircle<dim>(current_time, n_components, param))
  {
  }

  template <int dim>
  void FSI<dim>::set_bdf_coefficients(const unsigned int order)
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
  void FSI<dim>::make_grid()
  {
    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::ifstream input(param.mesh_file);
    AssertThrow(input, ExcMessage("Could not open mesh file: " + param.mesh_file));
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

    read_gmsh_physical_names(param.mesh_file,
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

    std::vector<std::string> all_boundaries;
    for (auto str : param.position_fixed_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.position_moving_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.strong_velocity_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.weak_velocity_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.noflux_velocity_boundary_names)
      all_boundaries.push_back(str);

    // Check that specified boundaries exist
    for (auto str : all_boundaries)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("A boundary condition should be prescribed "
                                 "on the boundary named \"" +
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
      weak_bc_boundary_id = mesh_domains_name2tag.at(str);
    }
  }

  template <int dim>
  void FSI<dim>::setup_system()
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
    // Also put them in initial_positions:
    DoFTools::map_dofs_to_support_points(*fixed_mapping,
                                         dof_handler,
                                         this->initial_positions,
                                         fe.component_mask(position));
    local_evaluation_point.compress(VectorOperation::insert);
    evaluation_point = local_evaluation_point;

    // Set mapping as a solution-dependent mapping
    mapping = std::make_unique<MappingFEField<dim, dim, LA::MPI::Vector>>(
      dof_handler, evaluation_point, fe.component_mask(position));
  }

  template <int dim>
  void FSI<dim>::create_sparsity_pattern()
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

    // Add the position-lambda couplings
    // Each position dof is coupled to all lambda dofs on cylinder
    // Take the lambda dofs from dim = 0 of the coupling coeffs structure
    for(const auto &[position_dof, d] : coupled_position_dofs)
      for(const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
        dsp.add(position_dof, lambda_dof);

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
  void FSI<dim>::set_initial_condition()
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Update mesh position *BEFORE* evaluating scalar field
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             fixed_mesh_position_fun,
                             newton_update,
                             fe.component_mask(position));

    evaluation_point = newton_update;

    // Set velocity with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             initial_condition_fun,
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
  void FSI<dim>::create_lambda_zero_constraints(const unsigned int boundary_id)
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

    pcout << n_unconstrained
          << " lambda DOFs are unconstrained" << std::endl;
    pcout << n_constrained << " lambda DOFs are constrained" << std::endl;
  }

  template <int dim>
  void FSI<dim>::add_no_flux_constraints(AffineConstraints<double> &constraints)
  {
    std::set<types::boundary_id> no_normal_flux_boundaries;
    for (auto str : param.noflux_velocity_boundary_names)
    {
      no_normal_flux_boundaries.insert(this->mesh_domains_name2tag.at(str));
    }
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_normal_flux_boundaries, constraints, *mapping);
  }

  /**
   * On the cylinder, we have
   * 
   * x = X - int_Gamma lambda dx,
   * 
   * yielding the affine constraints
   * 
   * x_i = X_i + sum_j c_ij * lambda_j, with c_ij = - int_Gamma phi_global_j dx.
   * 
   * Each position DoF is linked to all lambda DoF on the cylinder, which may
   * not be owned of even ghosts of the current process.
   * 
   * This function does the following:
   * 
   * - It computes the coefficients c_ij of the coupling x_i = X_i + c_ij * lambda_j,
   *   which are the integral of the global shape functions associated to lambda_j.
   * 
   * - It creates the DOF pairings (x_i, vector of lambda_j), which specify to which
   *   lambda DOFs a position DOF on the cylinder is constrained (all of them actually).
   * 
   *   THERE IS ONLY ONE VECTOR ACTUALLY
   */
  template <int dim>
  void FSI<dim>::create_position_lambda_coupling_data(
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
    // Compute the weights c_ij and identify the constrained position DOFs.
    // Done only once as cylinder is rigid and those weights will not change.
    //
    std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    double sum_local = 0.;
    const unsigned int n_dofs_per_face = fe.n_dofs_per_face();
    std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);
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
          sum_local += JxW;

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
          {
            const unsigned int comp =
              fe.face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which contribute to the
            // integral on this element
            if (!locally_relevant_dofs.is_element(face_dofs[i_dof]))
              continue;

            //
            // Lambda face dofs contribute to the weights
            //
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

            //
            // Position face dofs are added to the list of coupled dofs
            //
            if(is_position(comp))
            {
              const unsigned int d = comp - x_lower;
              coupled_position_dofs.insert({face_dofs[i_dof], d});
            }
          }
        }
      }
    }

    ///////////////////////////
    // const double length = Utilities::MPI::sum(sum_local, mpi_communicator);
    // pcout << "Length = " << length << std::endl;
    ///////////////////////////

    // // Expected sum is -1/k * |Cylinder|
    // const double expected_weights_sum = -1./param.spring_constant * M_PI;

    // for(unsigned int d = 0; d < dim; ++d)
    // {
    //   double weights_sum = 0.;
    //   std::cout << "Weights for dim = " << d << std::endl;
    //   for(const auto &[lambda_dof, weight] : coeffs[d])
    //   {
    //     std::cout << "Lambda dof: " << lambda_dof << " - weight: " << weight << std::endl;
    //     weights_sum += weight;
    //   }
    //   std::cout << "Sum is = " << weights_sum << " - Expected : " << expected_weights_sum << std::endl;
    // }

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
        Utilities::MPI::all_gather(mpi_communicator, coeffs_vector);

      // Put back into map and sum contributions to same DoF from different
      // processes
      for (const auto &vec : gathered)
        for (const auto &pair : vec)
          gathered_coeffs_map[d][pair.first] += pair.second;

      position_lambda_coeffs[d].insert(position_lambda_coeffs[d].end(),
                                gathered_coeffs_map[d].begin(),
                                gathered_coeffs_map[d].end());
    }
  }

  template <int dim>
  void FSI<dim>::create_zero_constraints()
  {
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    //
    // Mesh position BC
    //
    for (auto str : param.position_fixed_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               Functions::ZeroFunction<dim>(
                                                 n_components),
                                               zero_constraints,
                                               fe.component_mask(position));
    }
    for (auto str : param.position_moving_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               Functions::ZeroFunction<dim>(
                                                 n_components),
                                               zero_constraints,
                                               fe.component_mask(position));
    }

    //
    // Velocity BC
    //
    for (auto str : param.strong_velocity_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               Functions::ZeroFunction<dim>(
                                                 n_components),
                                               zero_constraints,
                                               fe.component_mask(velocity));
    }

    // Add no velocity flux constraints
    this->add_no_flux_constraints(zero_constraints);

    zero_constraints.close();

    // Lambda constraints have to be enforced at each Newton iteration
    // Add them to both sets of constraints?
    zero_constraints.merge(
      lambda_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);
  }

  template <int dim>
  void FSI<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    //
    // Mesh position BC
    //
    for (auto str : param.position_fixed_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               fixed_mesh_position_fun,
                                               nonzero_constraints,
                                               fe.component_mask(position));
    }
    for (auto str : param.position_moving_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               mesh_position_circle_fun,
                                               nonzero_constraints,
                                               fe.component_mask(position));
    }

    //
    // Velocity BC
    //
    for (auto str : param.strong_velocity_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               dof_handler,
                                               mesh_domains_name2tag.at(str),
                                               inlet_fun,
                                               nonzero_constraints,
                                               fe.component_mask(velocity));
    }

    // Add no velocity flux constraints
    this->add_no_flux_constraints(nonzero_constraints);

    nonzero_constraints.close();

    // Merge lambda constraints
    nonzero_constraints.merge(
      lambda_constraints,
      AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed,
      true);
  }

  template <int dim>
  void
  FSI<dim>::update_boundary_conditions()
  {
    local_evaluation_point = present_solution;
    this->create_nonzero_constraints();
    // Distribute constraints
    nonzero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void FSI<dim>::assemble_matrix(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double>                   local_matrix(dofs_per_cell, dofs_per_cell);

    // Data to compute matrix with finite differences
    // The local dofs values, which will be perturbed
    std::vector<double> cell_dof_values(dofs_per_cell);
    Vector<double>      ref_local_rhs(dofs_per_cell);
    Vector<double>      perturbed_local_rhs(dofs_per_cell);

    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 weak_bc_boundary_id,
                                 bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(local_dof_indices);

      if(param.analytic_jacobian_matrix)
      {
        //
        // Analytic jacobian matrix
        //
        local_matrix = 0.;
        const bool distribute = true;
        this->assemble_local_matrix(first_step,
                                    cell,
                                    scratchData,
                                    evaluation_point,
                                    previous_solutions,
                                    local_dof_indices,
                                    local_matrix,
                                    distribute);
      }
      else
      {
        //
        // Finite differences
        //
        const double h      = 1.e-8;
        local_matrix        = 0.;
        ref_local_rhs       = 0.;
        perturbed_local_rhs = 0.;

        // Get the local dofs values
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_dof_values[j] = evaluation_point[local_dof_indices[j]];

        const bool distribute_rhs    = false;
        const bool use_full_solution = false;

        // Compute non-perturbed RHS
        this->assemble_local_rhs(first_step,
                                 cell,
                                 scratchData,
                                 evaluation_point,
                                 previous_solutions,
                                 local_dof_indices,
                                 ref_local_rhs,
                                 cell_dof_values,
                                 distribute_rhs,
                                 use_full_solution);

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

          // Compute perturbed RHS
          // Reinit is called in the local rhs function
          this->assemble_local_rhs(first_step,
                                   cell,
                                   scratchData,
                                   evaluation_point,
                                   previous_solutions,
                                   local_dof_indices,
                                   perturbed_local_rhs,
                                   cell_dof_values,
                                   distribute_rhs,
                                   use_full_solution);

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
      }
    }

    system_matrix.compress(VectorOperation::add);

    if(param.with_position_coupling)
      this->add_algebraic_position_coupling_to_matrix();
  }

  template <int dim>
  void FSI<dim>::add_algebraic_position_coupling_to_matrix()
  {
    //
    // Add algebraic constraints position-lambda
    //
    std::map<types::global_dof_index, std::vector<PETScWrappers::MatrixIterators::const_iterator>> position_row_entries;
    // Get row entries for each pos_dof
    for (const auto &[pos_dof, d] : coupled_position_dofs)
    {
      if(locally_owned_dofs.is_element(pos_dof))
      {
        std::vector<PETScWrappers::MatrixIterators::const_iterator> row_entries;
        for (auto it = system_matrix.begin(pos_dof); it != system_matrix.end(pos_dof); ++it)
          row_entries.push_back(it);
        position_row_entries[pos_dof] = row_entries;
      }
    }

    // Constrain matrix and RHS
    for (const auto &[pos_dof, d] : coupled_position_dofs)
    {
      if(locally_owned_dofs.is_element(pos_dof))
      {
        for (auto it : position_row_entries.at(pos_dof))
        {
          // std::cout << "zeroing " << pos_dof << " - " << it->column() << std::endl;
          system_matrix.set(pos_dof, it->column(), 0.0);
        }

        // Set constraint row: x_i - sum_j c_ij * lambda_j = 0
        system_matrix.set(pos_dof, pos_dof, 1.);
        for(const auto &[lambda_dof, weight] : position_lambda_coeffs[d])
          system_matrix.set(pos_dof, lambda_dof, -weight);
      }
    }

    system_matrix.compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI<dim>::add_algebraic_position_coupling_to_rhs()
  {
    // Set RHS to zero for coupled position dofs
    for (const auto &[pos_dof, d] : coupled_position_dofs)
      if(locally_owned_dofs.is_element(pos_dof))
        system_rhs(pos_dof) = 0.;

    system_rhs.compress(VectorOperation::insert);
  }

  template <int dim>
  void FSI<dim>::assemble_local_matrix(
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

    scratchData.reinit(cell,
                       current_solution,
                       previous_solutions);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW_moving = scratchData.JxW_moving[q];
      const double JxW_fixed  = scratchData.JxW_fixed[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];
      const auto &phi_p      = scratchData.phi_p[q];
      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

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
        const bool         i_is_x      = is_position(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_u      = is_velocity(component_j);
          const bool         j_is_p      = is_pressure(component_j);
          const bool         j_is_x      = is_position(component_j);

          bool assemble = false;
          double local_flow_matrix_ij = 0.;
          double local_ps_matrix_ij   = 0.;

          if (i_is_u && j_is_u)
          {
            assemble = true;

            // Time-dependent
            local_flow_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];

            // Convection
            local_flow_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                                present_velocity_gradients * phi_u[j]) *
                               phi_u[i];

            // Diffusion
            local_flow_matrix_ij +=
              param.kinematic_viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]);

            // ALE acceleration : - w dot grad(delta u)
            local_flow_matrix_ij += grad_phi_u[j] * (-dxdt) * phi_u[i];
          }

          if (i_is_u && j_is_p)
          {
            assemble = true;

            // Pressure gradient
            local_flow_matrix_ij += -div_phi_u[i] * phi_p[j];
          }

          if (i_is_u && j_is_x)
          {
            assemble = true;

            // Variation of time-dependent term with mesh position
            local_flow_matrix_ij += dudt * phi_u[i] * trace(grad_phi_x[j]);

            // Variation of ALE term (dxdt cdot grad(u)) with mesh position
            local_flow_matrix_ij += present_velocity_gradients *
                               (-bdfCoeffs[0] * phi_x[j]) * phi_u[i];
            local_flow_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               (-dxdt) * phi_u[i];
            local_flow_matrix_ij += present_velocity_gradients * (-dxdt) * phi_u[i] *
                               trace(grad_phi_x[j]);

            // Convection w.r.t. x
            local_flow_matrix_ij += (-present_velocity_gradients * grad_phi_x[j]) *
                               present_velocity_values * phi_u[i];
            local_flow_matrix_ij += present_velocity_gradients *
                               present_velocity_values * phi_u[i] *
                               trace(grad_phi_x[j]);

            // Diffusion
            const Tensor<2, dim> d_grad_u =
              -present_velocity_gradients * grad_phi_x[j];
            const Tensor<2, dim> d_grad_phi_u = -grad_phi_u[i] * grad_phi_x[j];
            local_flow_matrix_ij +=
              param.kinematic_viscosity * scalar_product(d_grad_u, grad_phi_u[i]);
            local_flow_matrix_ij +=
              param.kinematic_viscosity *
              scalar_product(present_velocity_gradients, d_grad_phi_u);
            local_flow_matrix_ij +=
              param.kinematic_viscosity *
              scalar_product(present_velocity_gradients, grad_phi_u[i]) *
              trace(grad_phi_x[j]);

            // Pressure gradient
            local_flow_matrix_ij +=
              -present_pressure_values * trace(-grad_phi_u[i] * grad_phi_x[j]);
            local_flow_matrix_ij +=
              -present_pressure_values * div_phi_u[i] * trace(grad_phi_x[j]);

            // Source term for velocity:
            // Variation of the source term integral with mesh position.
            // det J is accounted for at the end when multiplying by JxW(q).
            local_flow_matrix_ij += phi_u[i] * grad_source_velocity * phi_x[j];
            local_flow_matrix_ij +=
              source_term_velocity * phi_u[i] * trace(grad_phi_x[j]);
          }

          if (i_is_p && j_is_u)
          {
            assemble = true;

            // Continuity : variation w.r.t. u
            local_flow_matrix_ij += -phi_p[i] * div_phi_u[j];
          }

          if (i_is_p && j_is_x)
          {
            assemble = true;

            // Continuity : variation w.r.t. x
            local_flow_matrix_ij +=
              -trace(-present_velocity_gradients * grad_phi_x[j]) * phi_p[i];
            local_flow_matrix_ij +=
              -present_velocity_divergence * phi_p[i] * trace(grad_phi_x[j]);

            // Source term for pressure:
            local_flow_matrix_ij += phi_p[i] * grad_source_pressure * phi_x[j];
            local_flow_matrix_ij +=
              source_term_pressure * phi_p[i] * trace(grad_phi_x[j]);
          }

          //
          // Pseudo-solid
          //
          if (i_is_x && j_is_x)
          {
            assemble = true;

            // Linear elasticity
            local_ps_matrix_ij += 
              param.pseudo_solid_lambda * div_phi_x[j] * div_phi_x[i] +
              // param.pseudo_solid_mu * scalar_product((grad_phi_x[j] + transpose(grad_phi_x[j])), grad_phi_x[i]);
              param.pseudo_solid_mu *
                scalar_product((grad_phi_x[i] + transpose(grad_phi_x[i])),
                               grad_phi_x[j]);
          }

          if(assemble)
          {
            local_flow_matrix_ij *= JxW_moving;
            local_ps_matrix_ij   *= JxW_fixed;
            local_matrix(i, j)   += local_flow_matrix_ij + local_ps_matrix_ij;

            // Check that flow and pseudo-solid matrices don't overlap
            AssertThrow(!(std::abs(local_ps_matrix_ij) > 1e-14 && std::abs(local_flow_matrix_ij) > 1e-14), ExcMessage("Mismatch"));
          }
        }
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
            const double face_JxW_moving = scratchData.face_JxW_moving[i_face][q];

            const auto &phi_u = scratchData.phi_u_face[i_face][q];
            const auto &phi_x = scratchData.phi_x_face[i_face][q];
            const auto &phi_l = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            const auto &present_w =
              scratchData.present_face_mesh_velocity_values[i_face][q];
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];
            
            for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
            {
              const unsigned int component_i = scratchData.components[i];
              const bool         i_is_u      = is_velocity(component_i);
              const bool         i_is_l      = is_lambda(component_i);

              for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
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
                  local_matrix_ij += - phi_u[j] * phi_l[i];
                }

                if (i_is_l && j_is_x)
                {
                  local_matrix_ij += -( -bdfCoeffs[0] * phi_x[j] * phi_l[i]             );
                  local_matrix_ij += -( (present_u - present_w) * phi_l[i] * delta_dx_j );
                }

                local_matrix_ij *= face_JxW_moving;
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
        throw std::runtime_error("First step");
        nonzero_constraints.distribute_local_to_global(local_matrix,
                                                       local_dof_indices,
                                                       system_matrix);
      }
      else
      {
        zero_constraints.distribute_local_to_global(local_matrix,
                                                    local_dof_indices,
                                                    system_matrix);
        // for (unsigned int ii = 0; ii < scratchData.dofs_per_cell; ++ii)
        //   for (unsigned int jj = 0; jj < scratchData.dofs_per_cell; ++jj)
        //     system_matrix.add(local_dof_indices[ii],
        //                       local_dof_indices[jj],
        //                       local_matrix(ii, jj));
      }
    }
  }

  template <int dim>
  void FSI<dim>::assemble_rhs(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<double>                  cell_dof_values(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 weak_bc_boundary_id,
                                 bdfCoeffs);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      local_rhs = 0;
      bool distribute        = true;
      bool use_full_solution = true;
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               evaluation_point,
                               previous_solutions,
                               local_dof_indices,
                               local_rhs,
                               cell_dof_values,
                               distribute,
                               use_full_solution);
    }

    system_rhs.compress(VectorOperation::add);

    if(param.with_position_coupling)
      this->add_algebraic_position_coupling_to_rhs();
  }

  template <int dim>
  void FSI<dim>::assemble_local_rhs(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    Vector<double>                                       &local_rhs,
    std::vector<double>                                  &cell_dof_values,
    bool                                                  distribute,
    bool                                                  use_full_solution)
  {
    if (use_full_solution)
    {
      scratchData.reinit(cell,
                         current_solution,
                         previous_solutions);
    }
    else
      scratchData.reinit(cell,
                         cell_dof_values,
                         previous_solutions);

    local_rhs = 0;

    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<Tensor<1, dim>> velocity(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      //
      // Flow related data
      //
      const double JxW_moving = scratchData.JxW_moving[q];

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

      //
      // Pseudo-solid related data
      //
      const double JxW_fixed = scratchData.JxW_fixed[q];

      const auto &present_position_gradients =
        scratchData.present_position_gradients[q];
      const double present_displacement_divergence =
        trace(present_position_gradients);
      const auto present_displacement_gradient_sym =
          present_position_gradients + transpose(present_position_gradients);
      const auto &source_term_position = scratchData.source_term_position[q];

      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        //
        // Flow residual
        //
        double local_rhs_flow_i =
          -(
            // Convection
            (present_velocity_gradients * present_velocity_values) * phi_u[i]

            // Mesh movement
            - (present_velocity_gradients * present_mesh_velocity_values) *
                phi_u[i]

            // Diffusion
            + param.kinematic_viscosity *
                scalar_product(present_velocity_gradients, grad_phi_u[i])

            // Pressure gradient
            - div_phi_u[i] * present_pressure_values

            // Momentum source term
            + source_term_velocity * phi_u[i]

            // Continuity
            - present_velocity_divergence * phi_p[i]

            // Pressure source term
            + source_term_pressure * phi_p[i]);

        // Transient terms:
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
        {
          local_rhs_flow_i -= bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i];
        }

        local_rhs_flow_i *= JxW_moving;

        //
        // Pseudo-solid
        //
        double local_rhs_ps_i = 
            - ( // Linear elasticity
              param.pseudo_solid_lambda * present_displacement_divergence * div_phi_x[i] +
              // param.pseudo_solid_mu * scalar_product(grad_phi_x[i], present_displacement_gradient_sym)
              param.pseudo_solid_mu * scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

              // Linear elasticity source term
            + phi_x[i] * source_term_position);

        local_rhs_ps_i *= JxW_fixed;

        local_rhs(i) += local_rhs_flow_i + local_rhs_ps_i;
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
            //
            // Flow related data (no-slip)
            //
            const double face_JxW_moving = scratchData.face_JxW_moving[i_face][q];
            const auto  &phi_u           = scratchData.phi_u_face[i_face][q];
            const auto  &phi_l           = scratchData.phi_l_face[i_face][q];

            const auto &present_u =
              scratchData.present_face_velocity_values[i_face][q];
            const auto &present_w =
              scratchData.present_face_mesh_velocity_values[i_face][q];
            const auto &present_l =
              scratchData.present_face_lambda_values[i_face][q];

            for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              double local_rhs_i = 0.;

              const unsigned int component_i = scratchData.components[i];
              const bool         i_is_u      = is_velocity(component_i);
              const bool         i_is_l      = is_lambda(component_i);

              if (i_is_u)
              {
                local_rhs_i -= - (phi_u[i] * present_l);
              }

              if (i_is_l)
              {
                local_rhs_i -= - (present_u - present_w) * phi_l[i];
              }

              local_rhs_i *= face_JxW_moving;
              local_rhs(i) += local_rhs_i;
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
        throw std::runtime_error("First step");
        nonzero_constraints.distribute_local_to_global(local_rhs,
                                                       local_dof_indices,
                                                       system_rhs);
      }
      else
        zero_constraints.distribute_local_to_global(local_rhs,
                                                    local_dof_indices,
                                                    system_rhs);
    }
  }

  template <int dim>
  void FSI<dim>::solve_direct(bool first_step)
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
    {
      throw std::runtime_error("First step");
      nonzero_constraints.distribute(newton_update);
    }
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void
  FSI<dim>::create_mask_arrays()
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Scalar pressure(p_lower);
    const FEValuesExtractors::Vector position(x_lower);
    const FEValuesExtractors::Vector lambda(l_lower);

    // std::vector<ComponentMask> masks(4);
    masks.resize(4);
    masks[0] = fe.component_mask(velocity);
    masks[1] = fe.component_mask(pressure);
    masks[2] = fe.component_mask(position);
    masks[3] = fe.component_mask(lambda);

    const unsigned int n_dofs = dof_handler.n_dofs();
    component_of_dof.resize(n_dofs, numbers::invalid_unsigned_int);
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
      {
        const unsigned int comp = fe.system_to_component_index(i).first;
        component_of_dof[local_dof_indices[i]] = comp;
      }
    }

    std::ofstream outfile(param.output_dir + "masks.txt");
    std::string field_name, constrained_pos;
    for (unsigned int i = 0; i < n_dofs; ++i)
    {
      const unsigned int comp = component_of_dof[i];
      for (unsigned int f = 0; f < 4; ++f)
      {
        outfile << masks[f][comp] << (f == 3 ? '\n' : ' ');
      }
      if(masks[0][component_of_dof[i]]) field_name = "u";
      else if(masks[1][component_of_dof[i]]) field_name = "p";
      else if(masks[2][component_of_dof[i]]) field_name = "x";
      else if(masks[3][component_of_dof[i]]) field_name = "l";
      else
        throw std::runtime_error("Unexected field component");

      if(zero_constraints.is_constrained(i))
        constrained_pos = "constrained";
      else
        constrained_pos = "";

      std::cout << "comp " << i << " is " << field_name << " " << constrained_pos << std::endl;
    }
    outfile.close();
  }

  template <int dim>
  void
  FSI<dim>::solve_newton(const bool is_initial_step)
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

        // if (outer_iteration == 0)
          this->assemble_rhs(false);

        current_res      = this->system_rhs.l2_norm();
        if (outer_iteration == 0)
        {
          last_res         = current_res;
        }

        if (VERBOSE)
        {
          pcout << std::scientific << std::setprecision(16) << std::showpos;
          pcout << "Newton iteration: " << outer_iteration << "  - Residual:  " << current_res << std::endl;
        }

        this->solve_direct(first_step);
        double last_alpha_res = current_res;

        if(param.with_line_search)
        {
          unsigned int alpha_iter = 0;
          for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
          {
            local_evaluation_point       = present_solution;
            local_evaluation_point.add(alpha, newton_update);
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
            }

            // If it's not the first iteration of alpha check if the residual is
            // smaller than the last alpha iteration. If it's not smaller, we fall
            // back to the last alpha iteration.
            if (current_res > last_alpha_res and alpha_iter != 0)
              {
                alpha                  = 2 * alpha;
                local_evaluation_point = present_solution;
                local_evaluation_point.add(alpha, newton_update);
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
        }
        else
        {
          local_evaluation_point       = present_solution;
          local_evaluation_point.add(1., newton_update);
          nonzero_constraints.distribute(local_evaluation_point);
          evaluation_point = local_evaluation_point;
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
        throw(std::runtime_error(
          "Stopping simulation because the non-linear solver has failed to converge"));
      }
  }

  template <int dim>
  void FSI<dim>::output_results(const unsigned int time_step,
                                const bool         write_newton_iteration,
                                const unsigned int newton_step)
  {
    //
    // Plot FE solution
    //
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

    //
    // Compute mesh velocity in post-processing
    // This is not ideal, this is done by modifying the displacement and
    // reexporting.
    //
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

    //
    // Partition
    //
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    // Export a Newton iteration in dedicated folder
    if(write_newton_iteration)
    {
      std::string root = "../data/fsi_coupled_newton_iterations/";
      std::string fileName = "solution_time_step_" + std::to_string(time_step);
      data_out.write_vtu_with_pvtu_record(
        root, fileName, newton_step, mpi_communicator, 2);
    }
    else
    {
      // Export regular time step
      std::string fileName = "solution";
      data_out.write_vtu_with_pvtu_record(
        param.output_dir, fileName, time_step, mpi_communicator, 2);
    }
  }

  template <int dim>
  void FSI<dim>::compute_force_coefficients(const unsigned int boundary_id,
                                            const bool export_force_table)
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

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
        }
      }
    }

    for (unsigned int d = 0; d < dim; ++d)
      lambda_integral[d] =
        Utilities::MPI::sum(lambda_integral_local[d], mpi_communicator);

    const double factor = 1. / (0.5 * param.rho * param.U * param.U * param.D);

    //
    // Forces on the cylinder are the NEGATIVE of the integral of lambda
    //
    forces_table.add_value("time", current_time);
    forces_table.add_value("CFx", - lambda_integral[0] * factor);
    forces_table.add_value("CFy", - lambda_integral[1] * factor);
    if constexpr (dim == 3)
    {
      forces_table.add_value("CFz", - lambda_integral[2] * factor);
    }

    if (export_force_table && mpi_rank == 0)
    {
      std::ofstream outfile(param.output_dir + "forces.txt");
      forces_table.write_text(outfile);
    }
  }

  /**
   * Compute the average of the position vector on the cylinder.
   */
  template <int dim>
  void FSI<dim>::write_cylinder_position(const unsigned int boundary_id,
                                         const bool export_position_table)
  {
    Tensor<1, dim> average_position, position_integral_local;
    double boundary_measure_local = 0.;

    const FEValuesExtractors::Vector position(x_lower);

    FEFaceValues<dim> fe_face_values_fixed(*fixed_mapping,
                                     fe,
                                     face_quadrature,
                                     update_values | update_quadrature_points |
                                       update_JxW_values |
                                       update_normal_vectors);

    const unsigned int          n_faces_q_points = face_quadrature.size();
    std::vector<Tensor<1, dim>> position_values(n_faces_q_points);

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

          // Get FE solution values on the face
          fe_face_values_fixed[position].get_function_values(present_solution,
                                                             position_values);

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            boundary_measure_local  += fe_face_values_fixed.JxW(q);
            position_integral_local += position_values[q] * fe_face_values_fixed.JxW(q);
          }
        }
      }
    }

    const double boundary_measure = Utilities::MPI::sum(boundary_measure_local, mpi_communicator);
    for (unsigned int d = 0; d < dim; ++d)
      average_position[d] = 1./boundary_measure *
        Utilities::MPI::sum(position_integral_local[d], mpi_communicator);

    cylinder_position_table.add_value("time", current_time);
    cylinder_position_table.add_value("xc", average_position[0]);
    cylinder_position_table.add_value("yc", average_position[1]);
    if constexpr (dim == 3)
      cylinder_position_table.add_value("zc", average_position[2]);

    if (export_position_table && mpi_rank == 0)
    {
      std::ofstream outfile(param.output_dir + "cylinder_center.txt");
      cylinder_position_table.write_text(outfile);
    }
  }

  /**
   * Compute integral of lambda (fluid force), compare to position dofs
   */
  template <int dim>
  void
  FSI<dim>::compare_lambda_position_on_boundary(const unsigned int boundary_id)
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
            // const Point<dim> &qpoint = fe_face_values.quadrature_point(q);

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
            {
              first_displacement_x = false;
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

    pcout << std::endl;
    pcout << std::scientific << std::setprecision(8) << std::showpos;
    pcout << "Checking consistency between lambda integral and position BC:" << std::endl;
    pcout << "Integral of lambda on cylinder is " << lambda_integral << std::endl;
    pcout << "Prescribed displacement        is " << cylinder_displacement << std::endl;
    pcout << "                         Ratio is " << ratio << " (expected: " << -param.spring_constant << ")" << std::endl;
    pcout << "Max diff between displacements is " << max_diff << std::endl;
    AssertThrow(max_diff.norm() <= 1e-10,
      ExcMessage("Displacement values of the cylinder are not all the same."));

    //
    // Check relative error between lambda/disp ratio vs spring constant
    //
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
    pcout << std::endl;
    
    // Tensor <1, dim> cylinder_displacement, max_diff, ratio;
    // for (unsigned int d = 0; d < dim; ++d)
    // {
    //   cylinder_displacement[d] =
    //     Utilities::MPI::max(cylinder_displacement_local[d], mpi_communicator);
    //   max_diff[d] =
    //     Utilities::MPI::max(max_diff_local[d], mpi_communicator);
    //   ratio[d] = lambda_integral[d] / cylinder_displacement[d];
    // }

    // pcout << "Integral of lambda on cylinder is " << lambda_integral << std::endl;
    // pcout << "Prescribed displacement        is " << cylinder_displacement << std::endl;
    // pcout << "                         Ratio is " << ratio << std::endl;
    // pcout << "Max diff between displacements is " << max_diff << std::endl;
    // AssertThrow(max_diff.norm() <= 1e-10,
    //   ExcMessage("Displacement values of the cylinder are not all the same."));
  }

  template <int dim>
  void FSI<dim>::check_velocity_boundary(const unsigned int boundary_id)
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

            // std::cout << std::scientific << std::setprecision(8) << std::showpos;
            // // std::cout << "wh = " << mesh_velocity_values[q] << std::endl;
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
    pcout << "Checking no-slip enforcement on cylinder:" << std::endl;
    pcout << "||uh - wh||_L2 = " << l2_error << std::endl;
    pcout << "||uh - wh||_Li = " << li_error << std::endl;

    if(this->current_time_step >= param.bdf_order)
    {
      AssertThrow(l2_error < 1e-12, ExcMessage("L2 norm of uh - wh is too large."));
      AssertThrow(li_error < 1e-12, ExcMessage("Linf norm of uh - wh is too large."));
    }
  }

  template <int dim>
  void FSI<dim>::run()
  {
      this->param.prev_dt = this->param.dt;
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->current_time_step = 1;
      this->inlet_fun.set_time(current_time);
      this->fixed_mesh_position_fun.set_time(current_time);
      this->mesh_position_circle_fun.set_time(current_time);

      this->make_grid();
      this->setup_system();
      this->create_lambda_zero_constraints(weak_bc_boundary_id);
      this->create_position_lambda_coupling_data(weak_bc_boundary_id);
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();

      this->set_initial_condition();
      this->output_results(0);
      this->write_cylinder_position(weak_bc_boundary_id, true);

      ///////////////////////////////////////
      // this->create_mask_arrays();
      ///////////////////////////////////////

      for (unsigned int i = 0; i < param.nTimeSteps; ++i, ++(this->current_time_step))
      {
        this->current_time += param.dt;
        this->inlet_fun.set_time(current_time);
        this->fixed_mesh_position_fun.set_time(current_time);
        this->mesh_position_circle_fun.set_time(current_time);

        if (VERBOSE)
        {
          pcout << std::endl
                << "Time step " << i + 1
                << " - Advancing to t = " << current_time << '.' << std::endl;
        }

        ////////////////////////////////////////////////////////////
        this->update_boundary_conditions();
        ////////////////////////////////////////////////////////////

        if (i == 0 && param.bdf_order == 2)
        {
          this->set_initial_condition();
        }
        else if (i == 1 && param.bdf_order == 3)
        {
          this->set_initial_condition();
        }
        else
        {
          // Entering the Newton solver with a solution satisfying the nonzero constraints,
          // which were applied in update_boundary_condition().
          this->solve_newton(false);
        }

        //
        // Check position - lambda coupling if coupled
        //
        if(param.with_position_coupling)
          this->compare_lambda_position_on_boundary(weak_bc_boundary_id);
  
        //
        // Always check that weak no-slip is satisfied
        //
        this->check_velocity_boundary(weak_bc_boundary_id);

        const bool export_force_table = (i % 5) == 0;
        this->compute_force_coefficients(weak_bc_boundary_id, export_force_table);
        const bool export_position_table = (i % 5) == 0;
        this->write_cylinder_position(weak_bc_boundary_id, export_position_table);
        this->output_results(i + 1);

        // Rotate solutions
        if (param.bdf_order > 0)
        {
          for (unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
            previous_solutions[i] = previous_solutions[i - 1];
          previous_solutions[0] = present_solution;
        }
      }

      // Write forces and cylinder position
      if (mpi_rank == 0)
      {
        {
          std::ofstream outfile(param.output_dir + "forces.txt");
          forces_table.write_text(outfile);
        }

        {
          std::ofstream outfile(param.output_dir + "cylinder_center.txt");
          cylinder_position_table.write_text(outfile);
        }
      }
  }
} // namespace fsi_coupled

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace fsi_coupled;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const unsigned int dim = 2;

    SimulationParameters<dim> param;

    param.output_dir = "../data/fsi_coupled/";

    if constexpr (dim == 2)
    {
      // param.mesh_file = "../data/meshes/cyl_not_confined_uber_coarse.msh";
      // param.mesh_file = "../data/meshes/cyl_not_confined_ultra_coarse.msh";
      // param.mesh_file = "../data/meshes/cyl_not_confined_coarser.msh";
      param.mesh_file = "../data/meshes/cyl_not_confined1.msh";
      // param.mesh_file = "../data/meshes/cyl_not_confined2.msh";
    }
    else
    {
      param.mesh_file = "../data/meshes/cylinderCoarse3D.msh";
    }

    param.velocity_degree = 2;
    param.position_degree = 1;
    param.lambda_degree   = 2;

    param.with_position_coupling = true; 

    //
    // Mesh position BC
    //
    param.position_fixed_boundary_names = {"Inlet", "Outlet", "NoFlux"};
    if(!param.with_position_coupling)
      // param.position_fixed_boundary_names.push_back("InnerBoundary");
      param.position_moving_boundary_names = {"InnerBoundary"};

    //
    // Velocity BC
    //
    param.strong_velocity_boundary_names = {"Inlet"};
    param.weak_velocity_boundary_names   = {"InnerBoundary"};
    param.noflux_velocity_boundary_names = {"NoFlux"};

    // Specify that mesh velocity error should be computed on this boundary
    // param.mesh_velocity_error_boundary_names = {"InnerBoundary"};

    // For the unconfined case
    param.Re  = 200.;
    param.H   = 16.;
    param.D   = 1.;
    param.U   = 1.;
    param.rho = 1.;

    // Solve the nondimensional Navier-Stokes and set nu = 1/Re.
    param.kinematic_viscosity = 1. / param.Re;
    param.pseudo_solid_mu     = 1.;
    param.pseudo_solid_lambda = 1.;

    param.spring_constant     = 1.;

    // Time integration
    param.bdf_order  = 2;
    param.t0         = 0.;
    param.dt         = 0.1;
    param.nTimeSteps = 5;
    param.t1         = param.dt * param.nTimeSteps;

    param.newton_tolerance = 1e-12;
    param.with_line_search = true;

    VERBOSE = true;

    FSI<dim> problem(param);
    problem.run();
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
