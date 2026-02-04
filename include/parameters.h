#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <parsed_function_symengine.h>

/**
 * This namespace contains the parameters used to control the various
 * parts of the solvers : mesh, (non-)linear solver, time integration, etc.
 */
namespace Parameters
{
  using namespace dealii;

  /**
   * Verbosity is set to "verbose" by default for all structures.
   */
  enum class Verbosity
  {
    quiet,
    verbose
  };

  // The problem dimension is read in a first pass to instantiate the right
  // pre-compiled solver. Because dimension in deal.II is not a simulation
  // parameters per se, this is not done here, but in utilities.h. However, the
  // "Dimension" block read by the function in utilities.h should still be read
  // in the real run to avoid an exception. This is done here, and the dimension
  // is not parsed.
  struct DummyDimension
  {
    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Timer
  {
    bool enable_timer;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct BoundaryConditionsData
  {
    // These are parsed in utilities.h
    unsigned int n_fluid_bc         = 0;
    unsigned int n_pseudosolid_bc   = 0;
    unsigned int n_cahn_hilliard_bc = 0;
    unsigned int n_heat_bc          = 0;

    bool fix_pressure_constant;
    bool enforce_zero_mean_pressure;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Mesh
  {
    // Gmsh mesh file
    std::string filename;

    bool         use_deal_ii_cube_mesh;
    std::string  deal_ii_preset_mesh;
    std::string  deal_ii_mesh_param;
    unsigned int refinement_level;

    // Name of each mesh physical entities
    std::map<types::boundary_id, std::string> id2name;
    std::map<std::string, types::boundary_id> name2id;

    Verbosity verbosity;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Output
  {
    bool         write_results        ;
    std::string  output_dir           ;
    std::string  output_prefix        ;
    unsigned int vtu_output_frequency ;

    // Skin output
    bool               write_skin_results ;
    types::boundary_id skin_boundary_id     ;
    unsigned int       skin_vtu_output_frequency ;

    static void declare_parameters(ParameterHandler &prm);
    void        read_parameters(ParameterHandler &prm);
  };

  struct PostProcessing
  {
    // Total force + position
    bool         write_force                  ;
    bool         write_body_position                ;
    unsigned int force_and_position_output_frequency ;

    // Slicing
    bool               enable_slicing      ;
    types::boundary_id slicing_boundary_id ;

    std::string  slicing_direction ;
    unsigned int number_of_slices  ;

    bool         write_force_per_slice          ;
    unsigned int force_per_slice_output_frequency;

    bool write_slice_vtu = false;

    static void declare_parameters(ParameterHandler &prm);
    void        read_parameters(ParameterHandler &prm);
  };


  template <int dim>
  struct FiniteElements
  {
    // If true, use hypercubes, otherwise use simplices (default).
    bool use_quads;

    // Degree of the velocity interpolation
    unsigned int velocity_degree;

    // Degree of the pressure interpolation
    unsigned int pressure_degree;

    // Degree of the mesh position interpolation
    unsigned int mesh_position_degree;

    // Degree of the Lagrange multipliers interpolation
    // when enforcing weak no-slip constraints
    unsigned int no_slip_lagrange_mult_degree;

    // Degree of the tracer and potential interpolation for two-phase
    // flows with a Cahn-Hilliard Navier-Stokes model
    unsigned int tracer_degree;
    unsigned int potential_degree;

    // Degree of the temperature for the heat equation
    unsigned int temperature_degree;

    struct QuadratureRule
    {
      enum class Type
      {
        GaussSimplex,
        WitherdenVincent
      } type;

      unsigned int n_pts_1D_simplex_cell_quad;
      unsigned int n_pts_1D_simplex_face_quad;
    };

    QuadratureRule rule;
    QuadratureRule rule_for_error;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct Fluid
  {
    double density;
    double kinematic_viscosity;

    void declare_parameters(ParameterHandler &prm, unsigned int index);
    void read_parameters(ParameterHandler &prm, unsigned int index);
  };

  template <int dim>
  class PseudoSolid
  {
  public:
    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      lame_lambda_fun;
    std::shared_ptr<ManufacturedSolutions::ParsedFunctionSDBase<dim>>
      lame_mu_fun;

  public:
    void set_time(const double newtime)
    {
      lame_lambda_fun->set_time(newtime);
      lame_mu_fun->set_time(newtime);
    }
    void declare_parameters(ParameterHandler &prm, unsigned int index);
    void read_parameters(ParameterHandler &prm, unsigned int index);
  };

  template <int dim>
  class PhysicalProperties
  {
  public:
    const unsigned int max_fluids = 2;
    unsigned int       n_fluids;
    std::vector<Fluid> fluids;

    const unsigned int            max_pseudosolids = 1;
    unsigned int                  n_pseudosolids;
    std::vector<PseudoSolid<dim>> pseudosolids;

  public:
    void set_time(const double newtime)
    {
      for (auto &ps : pseudosolids)
        ps.set_time(newtime);
    }
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
    Verbosity    verbosity;

    double reassembly_decrease_tol;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct LinearSolver
  {
    Verbosity verbosity;

    enum class Method
    {
      direct_mumps,
      gmres
    } method;

    double       tolerance;
    unsigned int max_iterations;
    unsigned int ilu_fill_level;

    bool renumber;
    bool reuse;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct TimeIntegration
  {
    double dt;
    double t_initial;
    double t_end;
    // unsigned int n_constant_timesteps; // To remove
    Verbosity verbosity;

    enum class Scheme
    {
      stationary,
      BDF1,
      BDF2
    } scheme;

    enum class BDFStart
    {
      BDF1,
      initial_condition
    } bdfstart;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
    bool is_steady() const { return scheme == Scheme::stationary; }
  };

  template <int dim>
  class CahnHilliard
  {
  public:
    enum class MobilityModel
    {
      constant
    } mobility_model;

    double mobility;
    double surface_tension;
    double epsilon_interface;

    /**
     * We differentiate between the body force which is multiplied by the
     * mixture density (typically gravity), and the generic source term (e.g.,
     * for manufactured solutions) which is not.
     */
    Tensor<1, dim> body_force;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct CheckpointRestart
  {
    bool enable_checkpoint;
    // If true, restart simulation from the given checkpoint file
    bool restart;
    // Name of the file to write to/read when checkpointing/restarting resp.
    std::string filename;
    // Write checkpoint every N time steps
    unsigned int checkpoint_frequency;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  struct MMS
  {
    bool enable;

    enum class Type
    {
      space,
      time,
      spacetime
    } type;

    enum class TimeLpNorm
    {
      L1,
      L2,
      Linfty
    } time_norm;

    bool subtract_mean_pressure;

    bool force_source_term;

    unsigned int n_convergence;
    unsigned int current_step = 0;
    int          run_only_step;

    bool         use_deal_ii_cube_mesh;
    bool         use_deal_ii_holed_plate_mesh;
    std::string  mesh_prefix;
    unsigned int first_mesh_index;
    unsigned int mesh_suffix = 0;

    std::vector<VectorTools::NormType> norms_to_compute;

    bool         use_space_convergence_mesh;
    unsigned int spatial_mesh_index;
    double       time_step_reduction_factor;

    void override_mesh_filename(Mesh &mesh_param, const unsigned int index)
    {
      mesh_param.filename = mesh_prefix + std::to_string(index) + ".msh";
    }

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Fluid-structure interaction
   */
  struct FSI
  {
    Verbosity verbosity;

    bool   enable_coupling;
    double spring_constant;
    double damping;
    double mass;

    double cylinder_radius;
    double cylinder_length;

    bool fix_z_component;

    bool compute_error_on_forces;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };

  /**
   * Options for debugging
   */
  struct Debug
  {
    Verbosity    verbosity;
    bool         write_partition_pos_gmsh;
    bool         apply_exact_solution;
    bool         compare_analytical_jacobian_with_fd;
    double       analytical_jacobian_absolute_tolerance;
    double       analytical_jacobian_relative_tolerance;
    bool         fsi_apply_erroneous_coupling;
    bool         fsi_check_mms_on_boundary;
    unsigned int fsi_coupling_option;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };
} // namespace Parameters

#endif