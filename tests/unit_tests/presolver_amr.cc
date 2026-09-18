#include <deal.II/base/function_parser.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <elasticity_solver.h>

#include "../tests.h"

// Include topology and all cell labels, not only the number of cells.
std::string snapshot(const parallel::distributed::Triangulation<2> &mesh)
{
  std::ostringstream out;
  out << std::setprecision(17);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      out << cell->id() << ' ' << cell->material_id();
      for (const auto v : cell->vertex_indices())
        out << ' ' << cell->vertex(v);
      for (const auto f : cell->face_indices())
        out << ' ' << cell->face(f)->boundary_id();
      out << '\n';
    }
  return out.str();
}

class Presolver : public ElasticitySolver<2>
{
public:
  using ElasticitySolver<2>::ElasticitySolver;

  void check_hanging_nodes()
  {
    setup_dofs();
    create_zero_constraints();
    create_nonzero_constraints();
    AffineConstraints<double> hanging(locally_owned_dofs,
                                      locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, hanging);
    hanging.close();
    unsigned int position_lines = 0, psi_lines = 0;
    const auto   psi_dofs = with_enlarged_psi ?
                              DoFTools::extract_dofs(dof_handler, psi_mask) :
                              IndexSet();
    for (const auto &line : hanging.get_lines())
    {
      for (const auto *constraints : {&zero_constraints, &nonzero_constraints})
      {
        AssertThrow(constraints->is_constrained(line.index),
                    ExcMessage("Missing hanging-node constraint"));
      }
      if (with_enlarged_psi && psi_dofs.is_element(line.index))
        ++psi_lines;
      else
        ++position_lines;
    }
    // Closing the full constraints can substitute prescribed boundary masters.
    // Check their action, rather than comparing the stored interpolation lines.
    for (const auto *constraints : {&zero_constraints, &nonzero_constraints})
    {
      LA::ParVectorType values(locally_owned_dofs, mpi_communicator);
      for (const auto index : locally_owned_dofs)
        values[index] = std::sin(0.37 * (index + 1));
      values.compress(VectorOperation::insert);
      constraints->distribute(values);
      LA::ParVectorType interpolated(locally_owned_dofs, mpi_communicator);
      interpolated = values;
      hanging.distribute(interpolated);
      interpolated -= values;
      AssertThrow(interpolated.linfty_norm() < 1e-12,
                  ExcMessage("Distributed values violate hanging nodes"));
    }
    AssertThrow(Utilities::MPI::sum(position_lines, mpi_communicator) > 0,
                ExcMessage("The fixture needs hanging position DoFs"));
    if (with_enlarged_psi)
      AssertThrow(Utilities::MPI::sum(psi_lines, mpi_communicator) > 0,
                  ExcMessage("The fixture needs hanging psi DoFs"));
  }

  void load_cache_in_reuse_mode()
  {
    using Mode = Parameters::Elasticity::PresolvedMeshPositionMode;
    param.elasticity.presolved_mesh_position_mode = Mode::reuse;
    try
    {
      try_load_presolved_mesh_cache();
    }
    catch (...)
    {
      param.elasticity.presolved_mesh_position_mode = Mode::off;
      throw;
    }
    param.elasticity.presolved_mesh_position_mode = Mode::off;
  }

  void check_solution()
  {
    FunctionParser<2> expected(fe->n_components());
    expected.initialize("x,y",
                        with_enlarged_psi ? "1.1*x+0.03*y+0.2;0.9*y;0.2" :
                                            "1.1*x+0.03*y+0.2;0.9*y",
                        {});
    LA::ParVectorType error(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(*mapping, dof_handler, expected, error);
    error -= present_solution;
    AssertThrow(error.linfty_norm() < 1e-9,
                ExcMessage("Presolver lost affine positions or initial time"));
  }
};

template <typename Action>
void must_reject(Action action)
{
  bool rejected = false;
  try
  {
    action();
  }
  catch (const ExceptionBase &)
  {
    rejected = true;
  }
  AssertThrow(rejected, ExcMessage("Shared-mesh operation must be rejected"));
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  Parameters::BoundaryConditionsData bc;
  bc.n_pseudosolid_bc = 1;
  ParameterHandler   prm;
  ParameterReader<2> param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Timer
  set enable timer = false
end
subsection Time integration
  set scheme = stationary
  set t_initial = 0.2
  set t_end = 1
  set verbosity = quiet
end
subsection Output
  set write vtu results = true
end
subsection FiniteElements
  set use quads = true
  set Mesh position degree = 1
end
subsection Nonlinear solver
  set verbosity = quiet
  set tolerance = 1e-11
  set max_iterations = 20
  set analytic_jacobian = true
end
subsection Linear solver
  subsection elasticity
    set verbosity = quiet
    set method = direct_mumps
  end
end
subsection Physical properties
  set number of pseudosolids = 1
  subsection Pseudosolid 0
    subsection lame lambda
      set Function expression = 1
    end
    subsection lame mu
      set Function expression = 1
    end
  end
end
subsection Initial conditions
  subsection cahn hilliard tracer
    set Function expression = t
  end
end
subsection Cahn Hilliard
  set mff source term = chns form
  set mff physics compression factor = 0
  set mff transport factor = 0
end
subsection Elasticity
  subsection presolver
    set continuation steps = 1
  end
end
subsection Pseudosolid boundary conditions
  set number = 1
  subsection boundary 0
    set id = 0
    set type = input_function
    subsection x
      set Function expression = 1.1*x+0.03*y+t
    end
    subsection y
      set Function expression = 0.9*y
    end
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.output.output_dir = "./";
  parallel::distributed::Triangulation<2> mesh(MPI_COMM_WORLD);
  GridGenerator::subdivided_hyper_cube(mesh, 2);
  for (const auto &cell : mesh.active_cell_iterators())
    if (cell->is_locally_owned() && cell->center()[0] < 0.5 &&
        cell->center()[1] < 0.5)
      cell->set_refine_flag();
  mesh.execute_coarsening_and_refinement();
  const auto    reference = snapshot(mesh);
  DoFHandler<2> external_dofs(mesh);
  FE_Q<2>       external_fe(1);
  external_dofs.distribute_dofs(external_fe);
  const auto external_count = external_dofs.n_dofs();
  for (const bool enlarged : {false, true})
  {
    param.output.output_prefix = enlarged ? "enlarged_" : "standard_";
    {
      Presolver solver(param, enlarged, &mesh);
      solver.check_hanging_nodes();
      solver.run();
      solver.check_solution();
      AssertThrow(snapshot(mesh) == reference, ExcMessage("Reference moved"));
      must_reject([&]() { solver.load_cache_in_reuse_mode(); });
      must_reject([&]() { solver.write_presolved_mesh_cache(); });
      must_reject([&]() { solver.move_mesh(); });
    }
    AssertThrow(snapshot(mesh) == reference,
                ExcMessage("Presolver destruction changed the mesh"));
    AssertThrow(external_dofs.n_dofs() == external_count,
                ExcMessage("External DoFHandler was invalidated"));
    LA::ParVectorType field(external_dofs.locally_owned_dofs(), MPI_COMM_WORLD);
    VectorTools::interpolate(external_dofs,
                             Functions::ConstantFunction<2>(1.),
                             field);
    AssertThrow(std::abs(field.linfty_norm() - 1.) < 1e-14,
                ExcMessage("External DoFHandler is no longer usable"));
  }
  using Mode = Parameters::Elasticity::PresolvedMeshPositionMode;
  for (const auto mode : {Mode::reuse, Mode::force_recompute})
  {
    param.elasticity.presolved_mesh_position_mode = mode;
    must_reject([&]() { Presolver solver(param, false, &mesh); });
  }
  param.elasticity.presolved_mesh_position_mode = Mode::off;
  param.elasticity.write_final_msh              = true;
  must_reject([&]() { Presolver solver(param, false, &mesh); });
  AssertThrow(snapshot(mesh) == reference,
              ExcMessage("Rejected operation changed the reference"));
  deallog
    << "Shared presolver preserves reference, hanging nodes and initial time"
    << std::endl;
}
