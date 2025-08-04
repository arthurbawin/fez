
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


#include <deal.II/grid/grid_in.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace MovingMeshTest
{
  using namespace dealii;

  template <int dim>
  class CylinderDisplacement : public Function<dim>
  {
  public:
    CylinderDisplacement(const double time = 0.)
      : Function<dim>(dim, time)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double x0 = 0.2;
      const double t = this->get_time();

      if constexpr (dim == 2)
      {
        values[0] = 0.1 * sin(M_PI * t);
        std::cout << "Imposed displacement is " << values[0] << std::endl;
        values[1] = 0.;
      } else
      {
        values[0] = 0.1;
        values[1] = 0.;
        values[2] = 0.;
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

    std::vector<unsigned int> components;

    std::vector<Tensor<2, dim>> present_displacement_gradients;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<Tensor<1, dim>>> phi_disp;
    std::vector<std::vector<double>>         div_phi_disp;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_disp;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_displacement_gradients.resize(n_q_points);

    div_phi_disp.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_disp.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_disp.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  }

  template <int dim>
  void ScratchData<dim>::reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                const LA::MPI::Vector &current_solution,
                                FEValues<dim> &fe_values)
  {
    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    const FEValuesExtractors::Vector displacement(0);

    fe_values[displacement].get_function_gradients(current_solution, present_displacement_gradients);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_disp[q][k]      = fe_values[displacement].value(k, q);
        grad_phi_disp[q][k] = fe_values[displacement].gradient(k, q);
        div_phi_disp[q][k]  = fe_values[displacement].divergence(k, q);
      }
    }
  }

  template <int dim>
  class MovingMesh
  {
  public:
    MovingMesh(unsigned int displacement_degree,
               double pseudo_solid_mu,
               double pseudo_solid_lambda);

    void
    run();

  private:
    void
    make_grid();
    void
    setup_system();
    void
    create_zero_constraints();
    void
    create_nonzero_constraints();
    void
    apply_zero_constraints();
    void
    apply_nonzero_constraints();
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
    move_mesh();
    void
    output_results(const unsigned int cycle) const;

    unsigned int displacement_degree;
    const double pseudo_solid_mu;
    const double pseudo_solid_lambda;
    MPI_Comm     mpi_communicator;

    double current_time;

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

    CylinderDisplacement<dim> cyl_bc;
    std::vector<Point<dim>> initial_mesh_position;
  };

  template <int dim>
  MovingMesh<dim>::MovingMesh(unsigned int displacement_degree,
                              double pseudo_solid_mu,
                              double pseudo_solid_lambda)
    : displacement_degree(displacement_degree)
    , pseudo_solid_mu(pseudo_solid_mu)
    , pseudo_solid_lambda(pseudo_solid_lambda)
    , mpi_communicator(MPI_COMM_WORLD)
    , current_time(0.)
    , fe(FE_SimplexP<dim>(displacement_degree) ^ dim)
    , triangulation(mpi_communicator)
    , mapping(FE_SimplexP<dim>(1))
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    cyl_bc = CylinderDisplacement<dim>(current_time);
  }

  template <int dim>
  void
  MovingMesh<dim>::make_grid()
  {
    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      // std::ifstream input("../data/meshes/cylinderUltraCoarse.msh");
      // std::ifstream input("../data/meshes/cylinderSuperCoarse.msh");
      meshFile = "../data/meshes/cylinderCoarse.msh";
      // std::ifstream input("../data/meshes/cylinderFine.msh");
      
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

    // Save initial position of the mesh vertices
    initial_mesh_position.resize(triangulation.n_vertices());
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index = cell->vertex_index(v);
        initial_mesh_position[global_vertex_index] = cell->vertex(v);
      }
    }
  }

  template <int dim>
  void
  MovingMesh<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    // // Constraints for Newton solver
    // zero_constraints.clear();
    // nonzero_constraints.clear();
    // zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    // nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    // const FEValuesExtractors::Vector displacement(0);

    // DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
    // DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

    // unsigned int cylinder_boundary_id = (dim == 2) ? 4 : 5;
    // std::vector<unsigned int> other_boundaries;
    // if constexpr(dim == 2)
    // {
    //   other_boundaries = {1, 2, 3};
    // }
    // else
    // {
    //   other_boundaries = {1, 2, 3, 4, 7};
    // }

    // // Cylinder
    // VectorTools::interpolate_boundary_values(mapping,
    //                                          dof_handler,
    //                                          cylinder_boundary_id,
    //                                          cyl_bc,
    //                                          nonzero_constraints,
    //                                          fe.component_mask(displacement));

    // // No displacement
    // for(unsigned int id : other_boundaries)
    // {
    //   VectorTools::interpolate_boundary_values(mapping,
    //                                          dof_handler,
    //                                          id,
    //                                          Functions::ZeroFunction<dim>(dim),
    //                                          nonzero_constraints,
    //                                          fe.component_mask(displacement));
    // }
    // nonzero_constraints.close();

    // other_boundaries.push_back(cylinder_boundary_id);
    // for(unsigned int id : other_boundaries)
    // {
    //   VectorTools::interpolate_boundary_values(mapping,
    //                                            dof_handler,
    //                                            id,
    //                                            Functions::ZeroFunction<dim>(dim),
    //                                            zero_constraints,
    //                                            fe.component_mask(displacement));
    // }
    // zero_constraints.close();

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

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
  MovingMesh<dim>::create_zero_constraints()
  {
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector displacement(0);

    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

    unsigned int cylinder_boundary_id = (dim == 2) ? 4 : 5;
    std::vector<unsigned int> other_boundaries;
    if constexpr(dim == 2)
    {
      other_boundaries = {1, 2, 3};
    }
    else
    {
      other_boundaries = {1, 2, 3, 4, 7};
    }

    other_boundaries.push_back(cylinder_boundary_id);
    for(unsigned int id : other_boundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(dim),
                                               zero_constraints,
                                               fe.component_mask(displacement));
    }
    zero_constraints.close();
  }

  template <int dim>
  void MovingMesh<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector displacement(0);

    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

    unsigned int cylinder_boundary_id = (dim == 2) ? 4 : 5;
    std::vector<unsigned int> other_boundaries;
    if constexpr(dim == 2)
    {
      other_boundaries = {1, 2, 3};
    }
    else
    {
      other_boundaries = {1, 2, 3, 4, 7};
    }

    // Cylinder
    this->cyl_bc.set_time(current_time);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             cylinder_boundary_id,
                                             cyl_bc,
                                             nonzero_constraints,
                                             fe.component_mask(displacement));

    // No displacement
    for(unsigned int id : other_boundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             id,
                                             Functions::ZeroFunction<dim>(dim),
                                             nonzero_constraints,
                                             fe.component_mask(displacement));
    }
    nonzero_constraints.close();
  }

  template <int dim>
  void
  MovingMesh<dim>::apply_zero_constraints()
  {
    zero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void
  MovingMesh<dim>::apply_nonzero_constraints()
  {
    nonzero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_matrix(bool first_step)
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

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_local_matrix(bool first_step,
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
      const auto &phi_u      = scratchData.phi_disp[q];
      const auto &grad_phi_u = scratchData.grad_phi_disp[q];
      const auto &div_phi_u  = scratchData.div_phi_disp[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];

          // double local_matrix_ij = 0.;

          // local_matrix_ij += pseudo_solid_lambda * grad_phi_u[i][component_i] * grad_phi_u[j][component_j];                  
          // local_matrix_ij += pseudo_solid_mu     * grad_phi_u[i][component_j] * grad_phi_u[j][component_i];                           
          // local_matrix_ij += (component_i == component_j) ?  (pseudo_solid_mu * grad_phi_u[i] *  grad_phi_u[j]) : 0.;                                
          // local_matrix_ij *= fe_values.JxW(q);
          // local_matrix(i, j) += local_matrix_ij;

          local_matrix(i, j) +=
          (
            // 
            pseudo_solid_lambda * div_phi_u[j] * div_phi_u[i]

            //
            + pseudo_solid_mu * scalar_product((grad_phi_u[i] + transpose(grad_phi_u[i])), grad_phi_u[j])

            ) * fe_values.JxW(q);
        }
      }

      // for (const unsigned int i : fe_values.dof_indices())
      // {
      //   const unsigned int component_i =
      //     fe.system_to_component_index(i).first;

      //   for (const unsigned int j : fe_values.dof_indices())
      //     {
      //       const unsigned int component_j =
      //         fe.system_to_component_index(j).first;

      //       for (const unsigned int q_point :
      //            fe_values.quadrature_point_indices())
      //         {
      //           local_matrix(i, j) +=
      //             (                                                  
      //               (fe_values.shape_grad(i, q_point)[component_i] * 
      //                fe_values.shape_grad(j, q_point)[component_j] * 
      //                pseudo_solid_lambda)                         
      //               +                                                
      //               (fe_values.shape_grad(i, q_point)[component_j] * 
      //                fe_values.shape_grad(j, q_point)[component_i] * 
      //                pseudo_solid_mu)                             
      //               +                                                
      //               ((component_i == component_j) ?        
      //                  (fe_values.shape_grad(i, q_point) * 
      //                   fe_values.shape_grad(j, q_point) * 
      //                   pseudo_solid_mu) :              
      //                  0)                                  
      //               ) *                                    
      //             fe_values.JxW(q_point);                  
      //         }
      //     }
      // }

    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
      nonzero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    else
      zero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_rhs(bool first_step)
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
  MovingMesh<dim>::assemble_local_rhs(bool first_step,
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
      const auto &grad_phi_u = scratchData.grad_phi_disp[q];
      const auto &div_phi_u  = scratchData.div_phi_disp[q];

      const auto &present_displacement_gradients = scratchData.present_displacement_gradients[q];

      double present_displacement_divergence =
                  trace(present_displacement_gradients);

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const auto present_displacement_gradient_sym = present_displacement_gradients + transpose(present_displacement_gradients);

        double local_rhs_i = - (

          pseudo_solid_lambda * present_displacement_divergence * div_phi_u[i]
          
          + pseudo_solid_mu * scalar_product(present_displacement_gradient_sym, grad_phi_u[i])

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
  void MovingMesh<dim>::solve(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);
  
    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    newton_update = completely_distributed_solution;

    if(first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void MovingMesh<dim>::solve_newton(
    const double       tolerance,
    const bool         is_initial_step,
    const bool         output_result)
  {
    double global_res;
    double current_res;
    double last_res;
    double norm_correction;
    bool   first_step     = true;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;
    unsigned int iter = 1;
    const unsigned int max_iter = 10;
    const double tol = 1e-13;

    // nonzero_constraints.distribute(newton_update);
    // present_solution = newton_update;

    while (current_res > tol && iter <= max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_matrix(first_step);
      this->assemble_rhs(first_step);
      current_res      = system_rhs.linfty_norm();

      if (iter == 1)
      {
        current_res      = system_rhs.linfty_norm();
        last_res         = current_res;
      }

      this->solve(first_step);
      first_step = false;

      norm_correction = newton_update.linfty_norm();
      pcout << "Newton iteration: " << iter << " - ||du|| = " << norm_correction << " - ||NL(u)|| = " << current_res << std::endl;

      if(norm_correction > 1e4 || current_res > 1e4)
      {
        pcout << "Diverged after " << iter << " iteration(s)" << std::endl;
        throw std::runtime_error("Nonlinear solver diverged");
      }

      double last_alpha_res = current_res;

      local_evaluation_point       = present_solution;
      local_evaluation_point.add(1., newton_update);
      nonzero_constraints.distribute(local_evaluation_point);
      evaluation_point = local_evaluation_point;
      this->assemble_rhs(first_step);

      current_res      = system_rhs.linfty_norm();

      last_alpha_res = current_res;

      if(current_res <= tol)
      {
        pcout << "Converged in " << iter << " iteration(s) because next nonlinear residual is below tolerance: " << current_res << " < " << tol << std::endl;
      }

      // global_res       = solver->get_current_residual();
      present_solution = evaluation_point;
      last_res         = current_res;
      ++iter;
    }

    if(iter == max_iter)
    {
      pcout << "Did not converge after " << iter << " iteration(s)" << std::endl;
      throw std::runtime_error("Nonlinear solver did not convege");
    }
  }

  template <int dim>
  void MovingMesh<dim>::move_mesh()
  {
    pcout << "    Moving mesh..." << std::endl;

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (auto &cell : dof_handler.active_cell_iterators())
      for (const auto v : cell->vertex_indices())
        if (vertex_touched[cell->vertex_index(v)] == false)
          {
            vertex_touched[cell->vertex_index(v)] = true;

            //
            // Modify this for when there are multiple fields in the FESystem
            //
            Point<dim> vertex_displacement;
            for (unsigned int d = 0; d < dim; ++d)
              vertex_displacement[d] =
                // incremental_displacement(cell->vertex_dof_index(v, d));
                present_solution(cell->vertex_dof_index(v, d));

            // cell->vertex(v) += vertex_displacement;
            const unsigned int global_vertex_index = cell->vertex_index(v);
            cell->vertex(v) = initial_mesh_position[global_vertex_index] + vertex_displacement;
          }
  }

  template <int dim>
  void
  MovingMesh<dim>::output_results(const unsigned int cycle) const
  {
    std::vector<std::string> solution_names(dim, "displacement");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);

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
  MovingMesh<dim>::run()
  {
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif

    const double dt  = 0.1;
    const double final_time = 2.;
    unsigned int time_step  = 1;

    this->make_grid();
    this->setup_system();
    this->create_zero_constraints();
    this->apply_zero_constraints();
    this->output_results(0);

    for (current_time += dt; current_time <= final_time; current_time += dt, ++time_step)
    {
      this->create_nonzero_constraints();
      this->apply_nonzero_constraints();

      pcout << std::endl
                << "Time step " << time_step << " - Advancing to t = " << current_time << '.' << std::endl;

      this->solve_newton(1., true, true);
      this->move_mesh();
      this->output_results(time_step);

    }
  }
}



int
main(int argc, char *argv[])
{
  
  try
    {
      using namespace dealii;
      using namespace MovingMeshTest;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const double pseudo_solid_mu = 1.0;
      const double pseudo_solid_lambda = 1.0;

      MovingMesh<2> problem(1, pseudo_solid_mu, pseudo_solid_lambda);
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
