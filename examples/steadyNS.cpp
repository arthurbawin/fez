/**
 * Adapted from step-55.cc for simplices.
 * Steady Stokes flow around a confined cylinder.
 */

#include <Mesh.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

// The following chunk out code is identical to step-40 and allows
// switching between PETSc and Trilinos:

#include <deal.II/lac/generic_linear_algebra.h>

// #define FORCE_USE_OF_TRILINOS

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
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#ifdef HEX
#  include <deal.II/distributed/tria.h>
#else
#include <deal.II/grid/grid_in.h>
#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/distributed/fully_distributed_tria.h>

#  include <deal.II/fe/fe_pyramid_p.h>
#  include <deal.II/fe/fe_simplex_p.h>
#  include <deal.II/fe/fe_simplex_p_bubbles.h>
#  include <deal.II/fe/fe_wedge_p.h>
#  include <deal.II/fe/mapping_fe.h>
#endif

#include <cmath>
#include <fstream>
#include <iostream>

namespace Step55
{
  using namespace dealii;

  template <int dim>
  class VelocityInlet : public Function<dim>
  {
  public:
    VelocityInlet()
      : Function<dim>(dim+1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double x = p[0];
      const double y = p[1];
      const double H = 0.41;
      const double uMax = 0.3;

      if constexpr (dim == 2)
      {
        values[0] = 4. * uMax * y * (H-y)/(H*H);
        values[1] = 0.;
        values[2] = 0.;
      } else
      {
        values[0] = 0.;
        values[1] = 4. * uMax * x * (H-x)/(H*H);
        values[2] = 0.;
        values[3] = 0.;
      }
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    ScratchData(const unsigned int n_q_points, const unsigned int dofs_per_cell)
     : n_q_points(n_q_points)
     , dofs_per_cell(dofs_per_cell)
    {
      this->allocate();
    }

    void allocate();
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const LA::MPI::Vector &current_solution,
                FEValues<dim> &fe_values);

  public:

    const unsigned int n_q_points;
    const unsigned int dofs_per_cell;

    // The vector component of each dof in the cell
    std::vector<unsigned int> components;

    // Current values and gradients for each quad node
    std::vector<Tensor<1, dim>> present_velocity_values;
    std::vector<Tensor<2, dim>> present_velocity_gradients;
    std::vector<double>         present_pressure_values;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         phi_p;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);

    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
  }

  template <int dim>
  void ScratchData<dim>::reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                const LA::MPI::Vector &current_solution,
                                FEValues<dim> &fe_values)
  {
    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    fe_values[velocities].get_function_values(current_solution, present_velocity_values);
    fe_values[velocities].get_function_gradients( current_solution, present_velocity_gradients);
    fe_values[pressure].get_function_values(current_solution, present_pressure_values);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u[q][k]      = fe_values[velocities].value(k, q);
        grad_phi_u[q][k] = fe_values[velocities].gradient(k, q);
        div_phi_u[q][k]  = fe_values[velocities].divergence(k, q);
        phi_p[q][k]      = fe_values[pressure].value(k, q);
      }
    }
  }

  template <int dim>
  class StokesProblem
  {
  public:
    StokesProblem(unsigned int velocity_degree);

    void
    run();

  private:
    void
    make_grid();
    void
    setup_system();
    void
    assemble_matrix(bool first_step);
    void
    assemble_local_matrix(bool first_step, const typename DoFHandler<dim>::active_cell_iterator &cell,
                          ScratchData<dim> &scratchData,
                          FEValues<dim> &fe_values,
                          LA::MPI::Vector &current_solution,
                          std::vector<types::global_dof_index> &local_dof_indices,
                          FullMatrix<double> &local_matrix);
    void
    assemble_rhs(bool first_step);
    void
    assemble_local_rhs(bool first_step, const typename DoFHandler<dim>::active_cell_iterator &cell,
                       ScratchData<dim> &scratchData,
                       FEValues<dim> &fe_values,
                       LA::MPI::Vector &current_solution,
                       std::vector<types::global_dof_index> &local_dof_indices,
                       Vector<double> &local_rhs);

    void solve(bool first_step);
    void
    solve_newton(
      const double       tolerance,
      const bool         is_initial_step,
      const bool         output_result);
    void
    output_results(const unsigned int cycle) const;

    unsigned int velocity_degree;
    double       viscosity;
    MPI_Comm     mpi_communicator;

    FESystem<dim> fe;

    parallel::fullydistributed::Triangulation<dim> triangulation;
    MappingFE<dim>                                 mapping;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    LA::MPI::SparseMatrix system_matrix;

    // With ghosts (read only)
    LA::MPI::Vector       present_solution;
    LA::MPI::Vector       evaluation_point;

    // Withotu ghosts (owned)
    LA::MPI::Vector       local_evaluation_point;
    LA::MPI::Vector       newton_update;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };

  template <int dim>
  StokesProblem<dim>::StokesProblem(unsigned int velocity_degree)
    : velocity_degree(velocity_degree)
    , viscosity(0.001)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_SimplexP<dim>(velocity_degree),
         dim,
         FE_SimplexP<dim>(velocity_degree - 1),
         1)
    , triangulation(mpi_communicator)
    , mapping(FE_SimplexP<dim>(1))
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}

  template <int dim>
  void
  StokesProblem<dim>::make_grid()
  {
    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    ///////////////////////////////////////////////////////////////////
    // auto mesh = read_mesh<dim>("../data/meshes/cylinderCoarse.msh", mpi_communicator);

    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      // std::ifstream input("../data/meshes/cylinderUltraCoarse.msh");
      // std::ifstream input("../data/meshes/cylinderSuperCoarse.msh");
      meshFile = "../data/meshes/cylinderCoarse.msh";
      // meshFile = "../data/meshes/cylinderFine.msh";
      
    }
    else
    {
      meshFile = "../data/meshes/cylinderCoarse3D.msh";
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
    // copy_triangulation does not seems to work, so maybe give reference to the mesh
    triangulation.create_triangulation(description);

    // {
    //   std::map<types::boundary_id, unsigned int> boundary_count;
    //   for (const auto &face : mesh.active_face_iterators())
    //     if (face->at_boundary())
    //       boundary_count[face->boundary_id()]++;
   
    //   std::cout << " boundary indicators: ";
    //   for (const std::pair<const types::boundary_id, unsigned int> &pair :
    //        boundary_count)
    //     {
    //       std::cout << pair.first << '(' << pair.second << " times) ";
    //     }
    //   std::cout << std::endl;
    // }

    // triangulation.copy_triangulation(mesh);
    ///////////////////////////////////////////////////////////////////

    // auto construction_data = TriangulationDescription::Utilities::
    //   create_description_from_triangulation_in_groups<dim, dim>(
    //     [&](Triangulation<dim> &tria) {
    //       Triangulation<dim> hex_tria;
    //       GridGenerator::hyper_cube(hex_tria, -0.5, 1.5);
    //       GridGenerator::convert_hypercube_to_simplex_mesh(hex_tria, tria);
    //       tria.refine_global(4);
    //     },
    //     [&](Triangulation<dim> &tria_serial,
    //         const MPI_Comm /*mpi_comm*/,
    //         const unsigned int /*group_size*/) {
    //       GridTools::partition_triangulation(mpi_size, tria_serial);
    //     },
    //     mpi_communicator,
    //     1);
    // triangulation.create_triangulation(construction_data);
  }

  // @sect3{System Setup}
  //
  // The construction of the block matrices and vectors is new compared to
  // step-40 and is different compared to serial codes like step-22, because
  // we need to supply the set of rows that belong to our processor.
  template <int dim>
  void
  StokesProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    // system_matrix.clear();

    dof_handler.distribute_dofs(fe);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    // Constraints for Newton solver
    zero_constraints.clear();
    nonzero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocities(0);

    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

    unsigned int inletBoundary = (dim == 2) ? 2 : 5;
    std::vector<unsigned int> noSlipBoundaries;
    if constexpr(dim == 2)
    {
      noSlipBoundaries = {3, 4};
    }
    else
    {
      noSlipBoundaries = {1, 2, 3, 4, 7};
    }

    // Inlet
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             inletBoundary,
                                             VelocityInlet<dim>(),
                                             nonzero_constraints,
                                             fe.component_mask(velocities));

    // No slip
    for(unsigned int id : noSlipBoundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             id,
                                             Functions::ZeroFunction<dim>(dim + 1),
                                             nonzero_constraints,
                                             fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    noSlipBoundaries.push_back(inletBoundary);
    for(unsigned int id : noSlipBoundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             id,
                                             Functions::ZeroFunction<dim>(dim + 1),
                                             zero_constraints,
                                             fe.component_mask(velocities));
    }
    zero_constraints.close();

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    // These don't need ghosts, I suppose ?
    local_evaluation_point.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    //
    // Sparsity pattern and allocate matrix
    //
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints, false);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      locally_owned_dofs,
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  }

  template <int dim>
  void
  StokesProblem<dim>::assemble_matrix(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const QGaussSimplex<dim> quadrature_formula(4);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_matrix(first_step, cell, scratchData, fe_values, evaluation_point, local_dof_indices, local_matrix);
    }

    // Add epsilon on the pressure diagonal entries
    // const double epsilon = 1e-6;
    // FEValuesExtractors::Scalar pressure(dim);
    // auto res = DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure));
    // auto indices = res.get_index_vector();
    // for (unsigned int i = 0; i < indices.size(); ++i)
    // {
    //   system_matrix.add(indices[i], indices[i], epsilon);
    // }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void
  StokesProblem<dim>::assemble_local_matrix(bool first_step,
                                            const typename DoFHandler<dim>::active_cell_iterator &cell,
                                            ScratchData<dim> &scratchData,
                                            FEValues<dim> &fe_values,
                                            LA::MPI::Vector &current_solution,
                                            std::vector<types::global_dof_index> &local_dof_indices,
                                            FullMatrix<double> &local_matrix)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit(cell, current_solution, fe_values);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const auto &present_velocity_values    = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients = scratchData.present_velocity_gradients[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];
      const auto &phi_p      = scratchData.phi_p[q];

      //
      // Not super efficient because looping over all dofs, even trivially 0 ?
      //
      // for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      // {
      //   for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
      //   {
      //     local_matrix(i, j) +=
      //     (
      //       // Convective
      //       (grad_phi_u[j] * present_velocity_values
      //         + present_velocity_gradients * phi_u[j]) * phi_u[i]

      //       // Diffusive
      //       + viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j])

      //       // Pressure gradient
      //       - div_phi_u[i] * phi_p[j]

      //       // Incompressibility
      //       - phi_p[i] * div_phi_u[j]
      //       ) * fe_values.JxW(q);
      //   }
      // }

      // Better? Assemble by local matrix block
      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];

          double local_matrix_ij = 0.;
          
          // Velocity - velocity block
          if(component_i < dim && component_j < dim)
          {
            // Convective
            local_matrix_ij +=
              (grad_phi_u[j] * present_velocity_values
                + present_velocity_gradients * phi_u[j]) * phi_u[i];

            if(component_i == component_j)
            {
              // Diffusive
              local_matrix_ij += viscosity * grad_phi_u[i][component_i] * grad_phi_u[j][component_j];
            }
          }

          // Pressure - velocity block
          if(component_i == dim)
          {
            // Incompressibility
            local_matrix_ij += - phi_p[i] * div_phi_u[j];
          }

          // Velocity - pressure block
          if(component_j == dim)
          {
            // Incompressibility
            local_matrix_ij += - div_phi_u[i] * phi_p[j];
          }

          local_matrix_ij *= fe_values.JxW(q);
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
      nonzero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    else
      zero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
  }

  template <int dim>
  void
  StokesProblem<dim>::assemble_rhs(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    const QGaussSimplex<dim> quadrature_formula(4);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_rhs(first_step, cell, scratchData, fe_values, evaluation_point, local_dof_indices, local_rhs);
    }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void
  StokesProblem<dim>::assemble_local_rhs(bool first_step,
                                         const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData<dim> &scratchData,
                                         FEValues<dim> &fe_values,
                                         LA::MPI::Vector &current_solution,
                                         std::vector<types::global_dof_index> &local_dof_indices,
                                         Vector<double> &local_rhs)
  {
    scratchData.reinit(cell, current_solution, fe_values);

    local_rhs = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];
      const auto &phi_p      = scratchData.phi_p[q];

      const auto &present_velocity_values    = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients = scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values    = scratchData.present_pressure_values[q];

      const Tensor<1, dim> uDotGradU = present_velocity_gradients * present_velocity_values;

      double present_velocity_divergence =
                  trace(present_velocity_gradients);

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        double local_rhs_i = - (
          // Convective
          uDotGradU * phi_u[i]

          // Diffusive
          + viscosity * scalar_product(present_velocity_gradients, grad_phi_u[i])

          // Pressure gradient
          - div_phi_u[i] * present_pressure_values

          // Incompressibility
          - phi_p[i] * present_velocity_divergence
          ) * fe_values.JxW(q);
          
          local_rhs(i) += local_rhs_i;
      }
    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
      nonzero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
    else
      zero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
  }

  template <int dim>
  void StokesProblem<dim>::solve(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);
  
    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    // SolverControl solver_control;
    // PETScWrappers::SolverPreOnly preonly(solver_control);
    // PETScWrappers::PreconditionLU preconditioner(system_matrix);
    // preonly.solve(system_matrix, completely_distributed_solution, system_rhs, preconditioner);

    // // Solve with iterative solver
    // SolverControl solver_control(10000, 1e-10);  // max_iter, tolerance
    // // PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);
    // // PETScWrappers::PreconditionILU ilu_preconditioner;
    // PETScWrappers::SolverBicgstab solver(solver_control, mpi_communicator);
    // PETScWrappers::PreconditionJacobi pc;
    // pc.initialize(system_matrix);
    // solver.solve(system_matrix, completely_distributed_solution, system_rhs, pc);

    // Optional: Output solver info
    pcout << "Solver converged in " << solver_control.last_step()
              << " iterations with final residual " << solver_control.last_value()
              << std::endl;

    newton_update = completely_distributed_solution;

    if(first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void StokesProblem<dim>::solve_newton(
    const double       tolerance,
    const bool         is_initial_step,
    const bool         output_result)
  {
    double global_res;
    double current_res;
    double last_res;
    bool   first_step     = true;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;
    unsigned int iter = 0;
    const unsigned int max_iter = 10;

    nonzero_constraints.distribute(newton_update);
    present_solution = newton_update;

    while (current_res > 1e-13 && iter < max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_matrix(first_step);
      // system_matrix.print(std::cout);
      this->assemble_rhs(first_step);
      current_res      = system_rhs.linfty_norm();
      // std::cout << "RHS" << std::endl;
      // system_rhs.print(std::cout);
      pcout << "Current residual is " << current_res << std::endl;

      // this->solveDirect(first_step);
      // local_evaluation_point       = present_solution;
      // local_evaluation_point.add(1., newton_update);
      // nonzero_constraints.distribute(local_evaluation_point);
      // // local_evaluation_point.print(std::cout);
      // evaluation_point = local_evaluation_point;
      // present_solution = evaluation_point;
      // this->assemble_rhs(first_step);
      // current_res      = system_rhs.linfty_norm();
      // std::cout << "Current residual is " << current_res << std::endl;
      // break;

      if (iter == 0)
      {
        current_res      = system_rhs.linfty_norm();
        last_res         = current_res;
      }

      pcout << "Newton iteration: " << iter << "  - Residual:  " << current_res << std::endl;

      this->solve(first_step);
      first_step = false;

      pcout << "Solved system" << std::endl;

      double last_alpha_res = current_res;

      // for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
      // {
        local_evaluation_point       = present_solution;
        local_evaluation_point.add(1., newton_update);
        nonzero_constraints.distribute(local_evaluation_point);
        evaluation_point = local_evaluation_point;
        this->assemble_rhs(first_step);

        current_res      = system_rhs.linfty_norm();

        pcout << "\talpha = " << std::setw(6) << 1.
                      << std::setw(0) << " res = "
                      << std::setprecision(6)
                      << std::setw(6) << current_res << std::endl;

        // if (current_res < 0.1 * last_res ||
        //     last_res < 0.1)
        // {
        //   break;
        // }
        last_alpha_res = current_res;
      // }

      // global_res       = solver->get_current_residual();
      present_solution = evaluation_point;
      last_res         = current_res;
      ++iter;
    }
  }

  template <int dim>
  void
  StokesProblem<dim>::output_results(const unsigned int cycle) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    ////////////////////////////////////////////////////////
    // Shenanigans
    // FEValuesExtractors::Vector vel(0);
    // auto res = DoFTools::extract_dofs(dof_handler, fe.component_mask(vel));
    // for(const auto index : res)
    // {
    //   std::cout << "present_solution for vel : " << present_solution[index] << std::endl;
    // }

    // // for(const auto index : res.get_index_vector())
    // // {
    // //   local_evaluation_point(index) = 1.23;
    // // }

    // // local_evaluation_point.compress(VectorOperation::insert);

    // // present_solution = local_evaluation_point;

    // // for(const auto index : res)
    // // {
    // //   std::cout << "new present_solution for vel : " << present_solution[index] << std::endl;
    // // }

    // FEValuesExtractors::Scalar pres(dim);
    // auto resp = DoFTools::extract_dofs(dof_handler, fe.component_mask(pres));
    // for(const auto index : resp)
    // {
    //   std::cout << "present_solution for p : " << present_solution[index] << std::endl;
    // }
    ////////////////////////////////////////////////////////

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, 2);

    std::ofstream output("foo.vtk");
    data_out.write_vtk(output);

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);

  }

  template <int dim>
  void
  StokesProblem<dim>::run()
  {
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif

    this->make_grid();
    this->setup_system();
    this->solve_newton(1., true, true);
    this->output_results(1);
  }
} // namespace Step55



int
main(int argc, char *argv[])
{
  
  try
    {
      using namespace dealii;
      using namespace Step55;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      StokesProblem<2> problem(2);
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
