#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

/**
 *
 */
namespace Parameters
{
  // enum class Verbosity
  // {

  // };

  // The problem dimension is read in a first pass to instantiate the right
  // data. Because dimension in deal.II is not a simulation parameters per se,
  // this is not done here, but in utilities.h. However, the "Dimension" block
  // read by the function in utilities.h should still be read in the real run to
  // avoid an exception. This is done here, through a dummy parameters
  // structure.
  struct DummyDimension
  {
    // This value is unused
    unsigned int dummy_dimension;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct BoundaryConditionsCount
  {
    unsigned int n_fluid_bc;
    unsigned int n_pseudosolid_bc;
  };

  struct Mesh
  {
    std::string filename;

    // Name of each mesh physical entities
    std::map<types::boundary_id, std::string> id2name;
    std::map<std::string, types::boundary_id> name2id;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Output
  {
    std::string output_dir;
    std::string output_prefix;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct FiniteElements
  {
    // Degree of the velocity interpolation
    unsigned int velocity_degree;

    // Degree of the pressure interpolation
    unsigned int pressure_degree;

    // Degree of the mesh position interpolation
    unsigned int mesh_position_degree;

    // Degree of the Lagrange multipliers interpolation
    // when enforcing weak no-slip constraints
    unsigned int no_slip_lagrange_mult_degree;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Fluid
  {
    double density;
    double kinematic_viscosity;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct PseudoSolid
  {
    double lame_lambda;
    double lame_mu;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  class PhysicalProperties
  {
  public:
    const unsigned int max_fluids = 2;
    unsigned int       n_fluids;
    std::vector<Fluid> fluids;

    const unsigned int       max_pseudosolids = 1;
    unsigned int             n_pseudosolids;
    std::vector<PseudoSolid> pseudosolids;

  public:
    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct NonLinearSolver
  {
    double       tolerance;
    double       divergence_tolerance;
    unsigned int max_iterations;
    bool         enable_line_search;
    bool         analytic_jacobian;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct LinearSolver
  {};

  struct TimeIntegration
  {
    double       dt;
    double       t_initial;
    double       t_end;
    unsigned int n_constant_timesteps;

    enum class Scheme
    {
      stationary,
      BDF1,
      BDF2
    } scheme;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Fluid-structure interaction
   */
  struct FSI
  {
    bool   enable_coupling;
    double spring_constant;
    double damping;
    double mass;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };
} // namespace Parameters

#endif